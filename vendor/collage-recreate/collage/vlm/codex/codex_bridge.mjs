/** One fresh SDK thread per request; stdout contains only a sanitized result. */
import childProcess from 'node:child_process';
import {syncBuiltinESMExports} from 'node:module';
import {pathToFileURL} from 'node:url';

// SDK 0.163.0 does not expose these exec flags or Windows window visibility.
// Scope this compatibility shim to this dedicated worker, before importing SDK.
// --ignore-user-config preserves authentication and managed requirements.
export function invocationOptions(args, options = {}) {
  return {args: args[0] === 'exec'
    ? [args[0], '--ephemeral', '--ignore-user-config', ...args.slice(1)] : args,
  options: {...options, windowsHide: true}};
}

function prepareSpawn() {
  const spawn = childProcess.spawn;
  childProcess.spawn = (file, args, options) => {
    const request = invocationOptions(args, options);
    return spawn(file, request.args, request.options);
  };
  syncBuiltinESMExports();
}

export function sdkOptions(request) {
  return {
    ...(process.env.COLLAGE_CODEX_PATH ? {codexPathOverride: process.env.COLLAGE_CODEX_PATH} : {}),
    config: {
      model_provider: 'openai',
      model_instructions_file: request.instructions_file,
      project_doc_max_bytes: 0,
      developer_instructions: '',
      hide_agent_reasoning: true,
      history: {persistence: 'none'},
      features: {shell_tool: false, multi_agent: false, remote_plugin: false},
      mcp_servers: {},
    },
  };
}

export function errorCode(error, aborted = false) {
  if (aborted) return 'timeout';
  if (['tools', 'invalid', 'home_unavailable'].includes(error?.code)) return error.code;
  const message = String(error?.message || '');
  if (/unauthorized|not logged in|sign in|401|refresh.token|authentication/i.test(message)) return 'auth';
  if (/429|quota|usage.limit|rate.limit/i.test(message)) return 'quota';
  if (/model.*(not found|not supported|unavailable|access)|403/i.test(message)) return 'model';
  if (/ENOENT|locate Codex|Cannot find|CODEX_HOME|home directory/i.test(message)) return 'runtime';
  return 'failed';
}

function failure(code) {
  return Object.assign(new Error(code), {code});
}

export async function infer(request, CodexClass) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), request.timeout_ms);
  try {
    const codex = new CodexClass(sdkOptions(request));
    const thread = codex.startThread({
      model: request.model,
      modelReasoningEffort: request.effort,
      sandboxMode: 'read-only',
      approvalPolicy: 'never',
      webSearchMode: 'disabled',
      networkAccessEnabled: false,
      skipGitRepoCheck: true,
      workingDirectory: request.working_directory,
    });
    const {events} = await thread.runStreamed(request.input, {
      ...(request.schema ? {outputSchema: request.schema} : {}), signal: controller.signal,
    });
    let content = '', usage = null, completed = false;
    for await (const event of events) {
      if (event.type.startsWith('item.')) {
        if (!['agent_message', 'reasoning'].includes(event.item?.type)) throw failure('tools');
        if (event.type === 'item.completed' && event.item.type === 'agent_message') {
          content = event.item.text;
          if (typeof content !== 'string' || Buffer.byteLength(content) > 7 * 1024 * 1024) throw failure('invalid');
        }
      }
      if (event.type === 'error' || event.type === 'turn.failed') {
        throw new Error(event.error?.message || event.message || 'Codex turn failed');
      }
      if (event.type === 'turn.completed') {
        completed = true;
        usage = event.usage;
      }
    }
    if (!completed || !content) throw failure('invalid');
    const counts = {};
    for (const [source, target] of [['input_tokens', 'prompt_tokens'], ['output_tokens', 'completion_tokens']]) {
      if (Number.isSafeInteger(usage?.[source]) && usage[source] >= 0) counts[target] = usage[source];
    }
    if (Number.isSafeInteger(usage?.cached_input_tokens) && usage.cached_input_tokens >= 0) {
      counts.prompt_tokens_details = {cached_tokens: usage.cached_input_tokens};
    }
    return {id: thread.id, model: request.model, provider: 'codex', usage: counts,
      choices: [{finish_reason: 'stop', message: {role: 'assistant', content}}]};
  } catch (error) {
    return {error: errorCode(error, controller.signal.aborted)};
  } finally {
    clearTimeout(timer);
    controller.abort();
  }
}

async function main() {
  try {
    prepareSpawn();
    // SDK 0.163 local_image drops detail. App Server exposes original image inputs.
    const {OriginalImageCodex: Codex} = await import('./codex_app_server.mjs');
    const chunks = [];
    let size = 0;
    for await (const chunk of process.stdin) {
      size += chunk.length;
      if (size > 16 * 1024 * 1024) throw failure('invalid');
      chunks.push(chunk);
    }
    const request = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    const result = await infer(request, Codex);
    process.stdout.write(JSON.stringify(result));
    process.exitCode = result.error ? 1 : 0;
  } catch {
    process.stdout.write(JSON.stringify({error: 'runtime'}));
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
