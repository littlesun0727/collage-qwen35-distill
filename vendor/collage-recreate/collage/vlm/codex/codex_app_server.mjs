/** Codex App Server adapter: preserve original image bytes and the SDK event contract. */
import childProcess from 'node:child_process';
import {createRequire} from 'node:module';
import {readFileSync, existsSync} from 'node:fs';
import path from 'node:path';
import {createInterface} from 'node:readline';

const require = createRequire(import.meta.url);

export function codexPath(override) {
  if (override) return override;
  const platform = process.platform === 'android' ? 'linux' : process.platform;
  const arch = {x64: 'x86_64', arm64: 'aarch64'}[process.arch];
  const suffix = {win32: 'pc-windows-msvc', linux: 'unknown-linux-musl', darwin: 'apple-darwin'}[platform];
  if (!arch || !suffix) throw new Error('Cannot find supported Codex binary');
  const cliRequire = createRequire(require.resolve('@openai/codex/package.json'));
  const pkg = cliRequire.resolve(`@openai/codex-${platform}-${process.arch}/package.json`);
  const root = path.join(path.dirname(pkg), 'vendor', `${arch}-${suffix}`);
  const name = platform === 'win32' ? 'codex.exe' : 'codex';
  for (const folder of ['bin', 'codex']) {
    const candidate = path.join(root, folder, name);
    if (existsSync(candidate)) return candidate;
  }
  throw new Error('Cannot find Codex binary');
}

function flatten(config, prefix = '', result = {}) {
  for (const [key, value] of Object.entries(config)) {
    const name = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length) {
      flatten(value, name, result);
    } else result[name] = value;
  }
  return result;
}

export function serverArgs(config) {
  return ['app-server', '--listen', 'stdio://', ...Object.entries(flatten(config)).flatMap(
    ([key, value]) => ['-c', `${key}=${JSON.stringify(value)}`])];
}

export function originalInputs(input) {
  return input.map(item => {
    if (item.type === 'text') return {type: 'text', text: item.text};
    if (item.type === 'local_image') {
      const mime = {'.png': 'png', '.jpeg': 'jpeg', '.jpg': 'jpeg', '.webp': 'webp'}[
        path.extname(item.path).toLowerCase()];
      if (!mime) throw Object.assign(new Error('Unsupported image format'), {code: 'invalid'});
      // Keep staging paths and Codex's localImage attachment labels out of the prompt.
      return {type: 'image', url: `data:image/${mime};base64,` + readFileSync(item.path).toString('base64'),
        detail: 'original'};
    }
    throw Object.assign(new Error('Unsupported image input'), {code: 'invalid'});
  });
}

class Rpc {
  constructor(child, signal) {
    this.child = child; this.pending = new Map(); this.queue = []; this.waiters = []; this.id = 0;
    this.closed = false; this.failure = null;
    // Drain stderr, but expose only a recognized diagnosis, never raw logs or credentials.
    let stderrTail = '';
    child.stderr?.on('data', chunk => { stderrTail = (stderrTail + chunk).slice(-16384); });
    this.lines = createInterface({input: child.stdout});
    this.lines.on('line', line => {
      if (Buffer.byteLength(line) > 24 * 1024 * 1024) return this.fail(new Error('Oversized Codex event'));
      let message;
      try { message = JSON.parse(line); } catch { return this.fail(new Error('Invalid Codex event')); }
      if (message.id !== undefined && !message.method) {
        const entry = this.pending.get(message.id);
        if (entry) {
          this.pending.delete(message.id);
          if (message.error) entry.reject(new Error(message.error.message || 'Codex request failed'));
          else entry.resolve(message.result);
        }
      } else if (message.id !== undefined) {
        // This inference component never approves tools, elicitation, or external actions.
        child.stdin.write(JSON.stringify({id: message.id, error: {code: -32601, message: 'Tools disabled'}})+'\n');
        this.fail(Object.assign(new Error('Tools disabled'), {code: 'tools'}));
      } else {
        const waiter = this.waiters.shift();
        if (waiter) waiter.resolve(message);
        else this.queue.push(message);
      }
    });
    child.on('error', error => this.fail(error));
    child.stdin.on('error', error => this.fail(error));
    child.on('close', () => {
      if (this.closed) return;
      const error = new Error('Codex process exited before completion');
      if (/Could not find home directory/i.test(stderrTail)) error.code = 'home_unavailable';
      this.fail(error);
    });
    this.signal = signal;
    this.abort = () => this.fail(new Error('Codex request aborted'));
    signal?.addEventListener('abort', this.abort, {once: true});
    if (signal?.aborted) this.abort();
  }
  fail(error) {
    if (this.failure) return;
    this.failure = error;
    for (const entry of this.pending.values()) entry.reject(error);
    this.pending.clear();
    for (const waiter of this.waiters.splice(0)) waiter.reject(error);
  }
  call(method, params) {
    if (this.failure) return Promise.reject(this.failure);
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {resolve, reject});
      this.child.stdin.write(JSON.stringify({id, method, params})+'\n');
    });
  }
  notify(method) { this.child.stdin.write(JSON.stringify({method})+'\n'); }
  next() {
    if (this.failure) return Promise.reject(this.failure);
    if (this.queue.length) return Promise.resolve(this.queue.shift());
    return new Promise((resolve, reject) => this.waiters.push({resolve, reject}));
  }
  close() {
    this.closed = true;
    this.signal?.removeEventListener('abort', this.abort);
    this.lines.close();
    this.fail(new Error('Codex connection closed'));
    if (this.child.exitCode === null) {
      if (process.platform === 'win32' && this.child.pid) {
        childProcess.spawnSync('taskkill', ['/PID', String(this.child.pid), '/T', '/F'],
          {windowsHide: true, stdio: 'ignore', timeout: 5000});
      }
      this.child.kill();
    }
    this.child.stdin.destroy();
    this.child.stdout.destroy();
    this.child.stderr?.destroy();
    this.child.unref();
  }
}

export class OriginalImageCodex {
  constructor(options = {}, spawn = childProcess.spawn) { this.options = options; this.spawn = spawn; }
  startThread(options) {
    const owner = this;
    return {id: null, async runStreamed(input, turn) {
      const provider = owner.options.config?.model_provider || 'openai';
      const config = {...owner.options.config, model_provider: provider, web_search: 'disabled',
        sandbox_mode: 'read-only', approval_policy: 'never',
        features: {...owner.options.config?.features, apps: false, hooks: false}};
      const child = owner.spawn(codexPath(owner.options.codexPathOverride), serverArgs(config), {
        cwd: options.workingDirectory, stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true,
        env: owner.options.env || process.env,
      });
      const rpc = new Rpc(child, turn.signal);
      try {
        await rpc.call('initialize', {clientInfo: {name: 'collage_visual_json', version: '2.0.0'}});
        rpc.notify('initialized');
        // App Server uses native authentication/config. Disable each inherited MCP server
        // on this thread without editing user configuration or copying authentication.
        const effective = await rpc.call('config/read', {includeLayers: false, cwd: options.workingDirectory});
        const overrides = {};
        for (const [name, value] of Object.entries(effective.config?.mcp_servers || {})) {
          (overrides.mcp_servers ||= {})[name] = {...value, enabled: false};
        }
        const started = await rpc.call('thread/start', {model: options.model, modelProvider: provider,
          cwd: options.workingDirectory, ephemeral: true, sandbox: 'read-only', approvalPolicy: 'never',
          baseInstructions: readFileSync(owner.options.config.model_instructions_file, 'utf8'),
          developerInstructions: '', config: overrides});
        this.id = started.thread.id;
        if (started.model && started.model !== options.model) throw new Error('model not supported: rerouted');
        await rpc.call('turn/start', {threadId: this.id, model: options.model, effort: options.modelReasoningEffort,
          input: originalInputs(input), approvalPolicy: 'never', sandboxPolicy: {type: 'readOnly', networkAccess: false},
          ...(turn.outputSchema ? {outputSchema: turn.outputSchema} : {})});
      } catch (error) { rpc.close(); throw error; }
      return {events: (async function* () {
        let usage = null;
        try {
          while (true) {
            const message = await rpc.next();
            const p = message.params || {};
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
              if (!type) throw Object.assign(new Error('Tools disabled'), {code: 'tools'});
              yield {type: message.method.replace('/', '.'), item: {type, text: item.text}};
            } else if (message.method === 'error' && !p.willRetry) {
              throw new Error(p.error?.message || 'Codex turn failed');
            } else if (message.method === 'turn/completed') {
              if (p.turn?.status !== 'completed') throw new Error(p.turn?.error?.message || 'Codex turn failed');
              yield {type: 'turn.completed', usage};
              return;
            }
          }
        } finally { rpc.close(); }
      })()};
    }};
  }
}
