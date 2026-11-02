/** Batch-only connection. The existing single-request adapter is unchanged. */
import childProcess from 'node:child_process';
import {readFileSync} from 'node:fs';
import {createInterface} from 'node:readline';
import {pathToFileURL} from 'node:url';
import {codexPath, serverArgs, originalInputs} from './codex_app_server.mjs';
import {infer, sdkOptions} from './codex_bridge.mjs';

const fault = code => Object.assign(new Error(code), {code});

function abortable(promise, signal) {
  if (signal.aborted) return Promise.reject(fault('timeout'));
  return new Promise((resolve, reject) => {
    const abort = () => reject(fault('timeout'));
    signal.addEventListener('abort', abort, {once: true});
    promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', abort));
  });
}

class Inbox {
  constructor() { this.queue = []; this.waiter = null; this.error = null; }
  push(message) {
    if (this.error) return;
    if (this.waiter) { this.waiter.resolve(message); this.waiter = null; }
    else if (this.queue.length < 4096) this.queue.push(message);
    else this.fail(fault('invalid'));
  }
  fail(error) { this.error = error; this.waiter?.reject(error); this.waiter = null; this.queue = []; }
  next() {
    if (this.error) return Promise.reject(this.error);
    if (this.queue.length) return Promise.resolve(this.queue.shift());
    return new Promise((resolve, reject) => { this.waiter = {resolve, reject}; });
  }
}

export class SharedServer {
  constructor({spawn = childProcess.spawn, executable, cwd = process.cwd(), rpcTimeout = 30000} = {}) {
    this.pending = new Map(); this.inboxes = new Map(); this.id = 0;
    this.rpcTimeout = rpcTimeout; this.failure = null; this.closed = false;
    const options = sdkOptions({});
    delete options.config.model_instructions_file;
    const config = {...options.config, web_search: 'disabled', sandbox_mode: 'read-only',
      approval_policy: 'never', features: {...options.config.features, apps: false, hooks: false}};
    this.child = spawn(executable || codexPath(options.codexPathOverride), serverArgs(config), {
      cwd, stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true, env: process.env,
    });
    let stderrTail = '';
    this.child.stderr?.on('data', chunk => { stderrTail = (stderrTail + chunk).slice(-16384); });
    this.lines = createInterface({input: this.child.stdout});
    this.lines.on('line', line => {
      try {
        if (Buffer.byteLength(line) > 24 * 1024 * 1024) throw fault('invalid');
        this.receive(JSON.parse(line));
      } catch (error) { this.fail(error); }
    });
    this.child.on('error', error => this.fail(error));
    this.child.stdin.on('error', error => this.fail(error));
    this.child.on('close', () => this.fail(/Could not find home directory/i.test(stderrTail)
      ? fault('home_unavailable') : fault('failed')));
    this.ready = this.initialize();
    this.ready.catch(() => {});
  }
  async initialize() {
    await this.call('initialize', {clientInfo: {name: 'collage_analysis_batch', version: '1.0.0'}});
    this.write({method: 'initialized'});
  }
  write(message) { this.child.stdin.write(JSON.stringify(message) + '\n'); }
  call(method, params) {
    if (this.failure) return Promise.reject(this.failure);
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        // An unacknowledged RPC may still create a turn. Stop the shared connection
        // rather than pretending that its resource state is known and retrying it.
        this.fail(fault('failed'));
      }, this.rpcTimeout);
      this.pending.set(id, {resolve, reject, timer});
      this.write({id, method, params});
    });
  }
  receive(message) {
    if (message.id !== undefined && !message.method) {
      const entry = this.pending.get(message.id);
      if (!entry) return;
      clearTimeout(entry.timer); this.pending.delete(message.id);
      if (message.error) entry.reject(new Error(message.error.message || 'Codex request failed'));
      else entry.resolve(message.result);
      return;
    }
    const inbox = this.inboxes.get(message.params?.threadId);
    if (message.id !== undefined) {
      this.write({id: message.id, error: {code: -32601, message: 'Tools disabled'}});
      if (inbox) inbox.fail(fault('tools'));
      return;
    }
    // Deltas can be large and are unnecessary: completed items contain full text.
    if (['item/started', 'item/completed', 'turn/completed', 'error',
      'model/rerouted', 'thread/tokenUsage/updated'].includes(message.method)) inbox?.push(message);
  }
  fail(error) {
    if (this.failure) return;
    this.failure = error;
    for (const entry of this.pending.values()) { clearTimeout(entry.timer); entry.reject(error); }
    this.pending.clear();
    for (const inbox of this.inboxes.values()) inbox.fail(error);
  }
  detach(threadId) {
    this.inboxes.delete(threadId);
    return this.call('thread/unsubscribe', {threadId}).catch(() => {});
  }
  interrupt(threadId, turnId) {
    return this.call('turn/interrupt', {threadId, turnId}).catch(() => {});
  }
  adapter(request) {
    const server = this;
    return class {
      startThread(options) {
        return {id: null, async runStreamed(input, turn) {
          // Read files before awaiting: a timed-out caller may remove its staging folder.
          const inputs = originalInputs(input);
          const instructions = readFileSync(request.instructions_file, 'utf8');
          const signal = turn.signal;
          let threadId, turnId, completed = false, cancelled = false;
          const inbox = new Inbox();
          const abort = () => { cancelled = true; inbox.fail(fault('timeout')); };
          signal.addEventListener('abort', abort, {once: true});
          const cleanup = async () => {
            signal.removeEventListener('abort', abort);
            if (threadId) {
              if (turnId && !completed) await server.interrupt(threadId, turnId);
              await server.detach(threadId);
            }
          };
          try {
            await abortable(server.ready, signal);
            const effective = await abortable(server.call('config/read', {
              includeLayers: false, cwd: options.workingDirectory}), signal);
            const overrides = {};
            for (const [name, value] of Object.entries(effective.config?.mcp_servers || {})) {
              (overrides.mcp_servers ||= {})[name] = {...value, enabled: false};
            }
            const starting = server.call('thread/start', {model: options.model, modelProvider: 'openai',
              cwd: options.workingDirectory, ephemeral: true, sandbox: 'read-only', approvalPolicy: 'never',
              baseInstructions: instructions, developerInstructions: '', config: overrides});
            starting.then(started => {
              if (cancelled) server.detach(started.thread.id);
            }, () => {});
            const started = await abortable(starting, signal);
            this.id = threadId = started.thread.id;
            server.inboxes.set(threadId, inbox);
            if (started.model && started.model !== options.model) throw new Error('model not supported: rerouted');
            const turning = server.call('turn/start', {threadId, model: options.model,
              effort: options.modelReasoningEffort, input: inputs, approvalPolicy: 'never',
              sandboxPolicy: {type: 'readOnly', networkAccess: false},
              ...(turn.outputSchema ? {outputSchema: turn.outputSchema} : {})});
            turning.then(result => {
              if (cancelled && result.turn?.id) server.interrupt(threadId, result.turn.id);
            }, () => {});
            const result = await abortable(turning, signal);
            turnId = result.turn?.id;
            if (!turnId) throw fault('invalid');
          } catch (error) { await cleanup(); throw error; }
          return {events: (async function* () {
            let usage = null;
            try {
              while (true) {
                const message = await inbox.next(), p = message.params || {};
                if (p.turnId && p.turnId !== turnId) continue;
                if (message.method === 'thread/tokenUsage/updated') {
                  const u = p.tokenUsage?.total;
                  if (u) usage = {input_tokens: u.inputTokens, output_tokens: u.outputTokens,
                    cached_input_tokens: u.cachedInputTokens};
                } else if (message.method === 'model/rerouted' && p.toModel !== options.model) {
                  throw new Error('model not supported: rerouted');
                } else if (['item/started', 'item/completed'].includes(message.method)) {
                  const item = p.item || {};
                  if (item.type === 'userMessage') continue;
                  const type = {agentMessage: 'agent_message', reasoning: 'reasoning'}[item.type];
                  if (!type) throw fault('tools');
                  yield {type: message.method.replace('/', '.'), item: {type, text: item.text}};
                } else if (message.method === 'error' && !p.willRetry) {
                  throw new Error(p.error?.message || 'Codex turn failed');
                } else if (message.method === 'turn/completed' && p.turn?.id === turnId) {
                  completed = true;
                  if (p.turn.status !== 'completed') throw new Error(p.turn.error?.message || 'Codex turn failed');
                  yield {type: 'turn.completed', usage};
                  return;
                }
              }
            } finally { await cleanup(); }
          })()};
        }};
      }
    };
  }
  infer(request) { return infer(request, this.adapter(request)); }
  close() {
    if (this.closed) return;
    this.closed = true; this.fail(fault('failed')); this.lines.close();
    if (this.child.exitCode === null) {
      if (process.platform === 'win32' && this.child.pid) {
        childProcess.spawnSync('taskkill', ['/PID', String(this.child.pid), '/T', '/F'],
          {windowsHide: true, stdio: 'ignore', timeout: 5000});
      }
      this.child.kill();
    }
    this.child.stdin.destroy(); this.child.stdout.destroy(); this.child.stderr?.destroy(); this.child.unref();
  }
}

async function main() {
  let server;
  const active = new Set();
  const lines = createInterface({input: process.stdin});
  try {
    for await (const line of lines) {
      const message = JSON.parse(line);
      if (message.method === 'shutdown') break;
      if (Buffer.byteLength(line) > 16 * 1024 * 1024) throw fault('invalid');
      server ||= new SharedServer();
      const task = server.infer(message.request).then(result => {
        process.stdout.write(JSON.stringify({id: message.id, result}) + '\n');
      });
      active.add(task); task.finally(() => active.delete(task));
    }
    // EOF means the owner has finished or died; terminate its outstanding work.
  } finally {
    server?.close();
    await Promise.allSettled(active);
    lines.close();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main().catch(() => { process.exitCode = 1; });
}
