"""Provider-specific chat parameters. Credentials never enter policies or receipts."""

import json
import re
from copy import deepcopy

ENDPOINTS = {
    'openrouter': 'https://openrouter.ai/api/v1/chat/completions',
    'yibu': 'http://127.0.0.1:17860/v1/chat/completions',
    'codex': 'codex-sdk://local',
    'vllm': 'http://127.0.0.1:8000/v1/chat/completions',
}


def family(model):
    name = model.lower().split('/')[-1].removeprefix('h:')
    if name.startswith('qwen'):
        return 'qwen'
    if name.startswith('gemini'):
        return 'gemini'
    if name.startswith('claude'):
        return 'claude'
    if name.startswith('deepseek-v4'):
        return 'deepseek'
    if name.startswith(('gpt-5', 'gpt-6', 'o1', 'o3', 'o4')):
        return 'gpt'
    return 'other'


def capabilities(provider, model):
    if provider == 'vllm':
        return {'modes': ['default', 'off', 'on'], 'efforts': [],
                'note': '自托管 vLLM；Qwen thinking 通过 chat_template_kwargs 设置。'}
    if provider == 'codex':
        return {'modes': ['default', 'effort'], 'efforts': ['low', 'medium', 'high', 'xhigh'],
                'note': 'Codex SDK 使用本机登录；默认 gpt-6.1-sol / medium。支持图片，可关闭远端 JSON Schema；关闭后靠提示词约束 JSON 并在本地校验。SDK 不提供硬性输出 token 上限；模型权限以实际调用为准。'}
    kind = family(model)
    name = model.lower().split('/')[-1].removeprefix('h:')
    if name.startswith('gpt-6.1-sol'):
        # Sol requires reasoning; expose the efforts supported by our policy UI.
        return {'modes': ['default', 'effort'],
                'efforts': ['low', 'medium', 'high', 'xhigh'],
                'note': 'GPT-6.1 Sol 不支持关闭推理或 minimal；默认推理档位为 medium。'}
    if provider == 'openrouter':
        return {'modes': ['default', 'off', 'on', 'budget', 'effort'],
                'efforts': ['minimal', 'low', 'medium', 'high', 'xhigh'],
                'note': '参数由 OpenRouter 路由；具体模型支持情况以接口响应为准。'}
    if kind == 'deepseek':
        return {'modes': ['default', 'off', 'on', 'effort'],
                'efforts': ['low', 'high'],
                'note': 'DeepSeek V4 支持 low / high 推理档位；JSON 输出可选择 JSON 对象。'}
    modes = {
        'qwen': ['default', 'off', 'on', 'budget', 'effort'],
        'gemini': ['default', 'off', 'budget', 'effort'],
        'gpt': ['default', 'off', 'effort'],
        'claude': ['default', 'off', 'budget'],
        'other': ['default'],
    }[kind]
    # The current Max route is a thinking model; do not offer a false off switch.
    if 'qwen3.8-max' in model:
        modes = [m for m in modes if m != 'off']
    return {'modes': modes,
            'efforts': ['minimal', 'low', 'medium', 'high', 'xhigh'] if kind == 'gpt' else ['low', 'medium', 'high'],
            'note': '兼容接口支持情况需实测；模型列表不保证识图、JSON 或推理参数可用。'}


def reasoning_mode(config):
    if config.reasoning_max_tokens is not None:
        return 'budget'
    if config.reasoning_effort:
        return 'effort'
    return 'default' if config.reasoning_enabled is None else 'on' if config.reasoning_enabled else 'off'


def request_options(config):
    """Wire fields only, including the actual output cap and thinking controls."""
    mode = reasoning_mode(config)
    caps = capabilities(config.provider, config.model)
    if mode not in caps['modes']:
        raise ValueError(f'{config.provider} / {config.model} 不支持此推理配置，请选择模型默认或可用选项')
    if config.reasoning_effort and config.reasoning_effort not in caps['efforts']:
        raise ValueError('当前模型适配不支持此推理档位')
    if config.provider == 'codex':
        if family(config.model) != 'gpt' or '/' in config.model or config.model.startswith('h:'):
            raise ValueError('Codex SDK 请使用原始 GPT 模型 ID，例如 gpt-6.1-sol')
        return {'modelReasoningEffort': config.reasoning_effort or 'medium',
                'output_token_limit_supported': False, 'bridge_version': 3,
                'transport': 'codex-app-server', 'image_detail': 'original'}
    result = {'max_tokens': config.max_tokens, 'temperature': 0}
    if config.provider == 'vllm':
        if config.reasoning_enabled is not None:
            result['chat_template_kwargs'] = {'enable_thinking': config.reasoning_enabled}
        return result
    if config.provider == 'openrouter':
        result['provider'] = {'require_parameters': True}
        if family(config.model) in {'gpt', 'claude'}:
            result.pop('temperature')
        if mode != 'default':
            reasoning = {'enabled': config.reasoning_enabled}
            if mode == 'budget':
                reasoning['max_tokens'] = config.reasoning_max_tokens
            if mode == 'effort':
                reasoning['effort'] = config.reasoning_effort
            result['reasoning'] = reasoning
        return result
    kind = family(config.model)
    if kind == 'gpt':
        result.pop('temperature')
        result['max_completion_tokens'] = result.pop('max_tokens')
        # Keep the upstream connection active during long strict-JSON generations.
        # The client still waits for a complete response before validating it.
        result['stream'] = True
        result['stream_options'] = {'include_usage': True}
        if mode in {'off', 'effort'}:
            result['reasoning_effort'] = 'none' if mode == 'off' else config.reasoning_effort
    elif kind == 'qwen' and mode != 'default':
        result['enable_thinking'] = mode != 'off'
        if mode == 'budget':
            result['thinking_budget'] = config.reasoning_max_tokens
        elif mode == 'effort':
            result['reasoning_effort'] = config.reasoning_effort
    elif kind == 'gemini':
        if mode in {'off', 'effort'}:
            result['reasoning_effort'] = 'none' if mode == 'off' else config.reasoning_effort
        elif mode == 'budget':
            result['google'] = {'thinking_config': {'thinking_budget': config.reasoning_max_tokens}}
    elif kind == 'claude':
        result.pop('temperature')
        if mode == 'off':
            result['thinking'] = {'type': 'disabled'}
        elif mode == 'budget':
            if config.reasoning_max_tokens < 1024:
                raise ValueError('Claude thinking token 上限至少为 1024')
            result['thinking'] = {'type': 'enabled', 'budget_tokens': config.reasoning_max_tokens}
    elif kind == 'deepseek' and mode != 'default':
        result['thinking'] = {'type': 'disabled' if mode == 'off' else 'enabled'}
        if mode == 'effort':
            result['reasoning_effort'] = config.reasoning_effort
    return result


def wire_schema(config, schema):
    """Adapt remote schema subsets; the original schema stays authoritative locally."""
    kind = family(config.model)
    if config.provider in {'yibu', 'codex'} and kind == 'gpt':
        from collage.vlm.gpt_schema import project
        return project(schema)
    if config.provider != 'yibu' or kind != 'gemini':
        return schema
    def convert(node):
        if not isinstance(node, dict):
            return node
        result = deepcopy(node)
        if 'const' in result:
            constant = result.pop('const')
            # Gemini's legacy Schema enum accepts strings, not arrays or numbers.
            # The full original schema still appears in the prompt and local validation.
            if isinstance(constant, str):
                result['enum'] = [constant]
        for group in ('properties', '$defs', 'definitions'):
            if isinstance(result.get(group), dict):
                result[group] = {key: convert(value) for key, value in result[group].items()}
        for key in ('items', 'additionalProperties', 'not'):
            if isinstance(result.get(key), dict):
                result[key] = convert(result[key])
        for key in ('anyOf', 'oneOf', 'allOf', 'prefixItems'):
            if isinstance(result.get(key), list):
                result[key] = [convert(value) for value in result[key]]
        return result
    return convert(schema)


def restore_response(config, answer, schema):
    if config.provider in {'yibu', 'codex'} and family(config.model) == 'gpt' and config.response_format == 'json_schema':
        from collage.vlm.gpt_schema import restore
        return restore(answer, schema)
    return answer


def payload_for(config, stage, messages, schema, correction=False):
    payload = {'model': config.model, 'messages': messages, 'stream': False, **request_options(config)}
    output = config.response_format
    if correction and output == 'json_schema' and not (config.provider in {'yibu', 'codex'} and family(config.model) == 'gpt'):
        output = 'json_object'
    if output == 'json_schema':
        payload['response_format'] = {'type': output, 'json_schema': {'name': stage, 'strict': True, 'schema': wire_schema(config, schema)}}
    elif output == 'json_object':
        payload['response_format'] = {'type': output}
    return payload


def decode_response(body):
    """Accept JSON or a complete Chat SSE response, retaining usage, never thoughts."""
    text = body.decode('utf-8-sig')
    if text.lstrip().startswith('{'):
        result = json.loads(text)
    else:
        result = {'choices': [{'message': {'content': ''}, 'finish_reason': None}]}
        choice = result['choices'][0]
        for block in re.split(r'\r?\n\r?\n', text):
            data = '\n'.join(line[5:].lstrip() for line in block.splitlines() if line.startswith('data:'))
            if not data or data == '[DONE]':
                continue
            row = json.loads(data)
            if row.get('error'):
                raise ValueError('模型流式响应返回错误')
            for name in ('id', 'model', 'provider', 'usage'):
                if row.get(name) is not None:
                    result[name] = row[name]
            for part in row.get('choices', []):
                if part.get('index', 0) != 0:
                    continue
                delta = part.get('delta') or part.get('message') or {}
                if isinstance(delta.get('content'), str):
                    choice['message']['content'] += delta['content']
                if delta.get('tool_calls'):
                    choice['message']['tool_calls'] = True
                if part.get('finish_reason') is not None:
                    choice['finish_reason'] = part['finish_reason']
    if not isinstance(result, dict) or result.get('error') or not result.get('choices'):
        raise ValueError('模型响应缺少完整 choices')
    return result
