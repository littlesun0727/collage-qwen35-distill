"""Multi-provider image/JSON transport with durable request budgets and evidence."""

from http.client import IncompleteRead
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler
from jsonschema import Draft202012Validator
from collage.config import Config
from collage.vlm.codex.transport import CodexError
from collage import prompts
from collage.storage import digest, exclusive, now, read, save
from collage.vlm.providers import ENDPOINTS, decode_response, payload_for, request_options, restore_response


ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'


class ModelError(RuntimeError):
    pass


class ModelTransportError(ModelError):
    """No usable model response: an output-repair prompt cannot fix transport."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_error_detail(error, config):
    """Expose bounded provider diagnostics without retaining echoed requests or secrets."""
    try:
        document = json.loads(error.read(16384))
        detail = document.get('error', {})
        message = detail.get('message', '') if isinstance(detail, dict) else ''
        if not isinstance(message, str):
            return ''
        # Only known API diagnostics are suitable for the UI; never retain a raw body.
        if not re.match(r"(?i)^(invalid schema|invalid parameter|unsupported (parameter|value)|"
                        r"unknown (parameter|model)|model .{0,100}(not found|not supported)|"
                        r"rate limit|insufficient quota|当前分组|额度|余额)", message.strip()):
            return ''
        try:
            secret = config.api_key()
        except ValueError:
            secret = ''
        if secret:
            message = message.replace(secret, '[凭证已隐藏]')
        message = re.sub(r'(?i)data:[^\s,]+,[^\s]+', '[图片数据已隐藏]', message)
        message = re.sub(r'(?i)https?://\S+', '[URL已隐藏]', message)
        message = re.sub(r'(?i)\bsk-[A-Za-z0-9_-]+', '[凭证已隐藏]', message)
        message = re.sub(r'(?i)(bearer\s+|(?:api[_-]?key|authorization)\s*[=:]\s*)\S+',
                         '[凭证已隐藏]', message)
        return ' '.join(message.split())[:700]
    except (ValueError, OSError, AttributeError):
        return ''


def parse_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Non-finite JSON value: ' + value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


class Client:
    def __init__(self, root, config=None, transport=None, sleep=time.sleep):
        self.root = Path(root)
        self.config = config or Config.load()
        self.transport = transport
        self.sleep = sleep
        self._last_folders = {}
        self._last_coordinates = {}
        self._last_schemas = {}

    def candidates(self, stage):
        """Only complete replies from this exact request, newest first; no extra POST."""
        folder = self._last_folders.get(stage)
        if folder is None:
            return []
        values = []
        for path in sorted(folder.glob('response-*.json'), reverse=True):
            response = read(path)
            if response.get('tool_calls') or response.get('finish_reason') != 'stop':
                continue
            try:
                value = parse_json(response.get('content'))
                value = restore_response(self.config.for_stage(stage), value, self._last_schemas.get(stage, {}))
                coordinates = self._last_coordinates.get(stage)
                if coordinates:
                    value = coordinates.decode(value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                values.append(value)
        return values

    def _post(self, payload, config=None):
        config = config or self.config
        if config.provider == 'codex':
            from collage.vlm.codex.transport import complete
            return complete(payload, config.timeout)
        key = config.api_key()
        headers = {'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'}
        if config.provider == 'openrouter':
            headers['X-Title'] = 'Collage Recreate v6'
        request = Request(ENDPOINTS[config.provider], data=json.dumps(payload, allow_nan=False).encode(),
                          headers=headers)
        # Never forward an Authorization header to a redirected host.
        deadline = time.monotonic() + config.timeout
        handlers = [NoRedirect()]
        if config.provider == 'yibu':
            handlers.append(ProxyHandler({}))  # Loopback gateway handles its own upstream route.
        with build_opener(*handlers).open(request, timeout=config.timeout) as response:
            chunks = []
            total = 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Model request deadline exceeded')
                try:
                    response.fp.raw._sock.settimeout(remaining)
                except AttributeError:
                    pass
                block = response.read1(65536)
                if not block:
                    break
                chunks.append(block); total += len(block)
                if total > 8*1024*1024:
                    raise ValueError('Model response exceeds 8 MiB')
            return decode_response(b''.join(chunks))

    def _reserve(self, stage, fingerprint):
        with exclusive(self.root / 'budget.lock'):
            ledger = read(self.root / 'budget.json', {'requests': []})
            if len(ledger['requests']) >= self.config.max_requests:
                raise ModelError(f'VLM 请求预算已用尽（{self.config.max_requests} 次），停止自动执行')
            number = len(ledger['requests']) + 1
            ledger['requests'].append({'number': number, 'stage': stage, 'input_key': fingerprint,
                                       'reserved_at': now()})
            save(self.root / 'budget.json', ledger)
            return number

    def ask(self, stage, prompt, context, images, schema, validate=None, *,
            single_request=False, reference_input_only=False, normalize_answer=None, reference_mode=None):
        # Inventory/repair already encode their bbox schema; declaring the mode
        # controls dimension disclosure without applying a second coordinate conversion.
        if reference_mode not in {None, 'pixel', 'normalized_1000'}:
            raise ValueError('Unknown reference coordinate mode')
        if not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', stage):
            raise ValueError('Invalid stage name')
        config = self.config.for_stage(stage)
        reasoning = {'enabled': config.reasoning_enabled}
        if config.reasoning_max_tokens is not None:
            reasoning['max_tokens'] = config.reasoning_max_tokens
        if config.reasoning_effort is not None:
            reasoning['effort'] = config.reasoning_effort
        wire_options = request_options(config)
        canonical_schema, canonical_context = schema, context
        request = prompts.prepare(prompt, context, images, schema, config,
                                  reference_input_only=reference_input_only, reference_mode=reference_mode)
        prompt, context, schema = request.prompt, request.context, request.schema
        coordinates, hide_dimensions = request.coordinates, request.hide_dimensions
        remote_schema, image_meta = request.remote_schema, request.image_meta
        self._last_coordinates[stage] = coordinates
        self._last_schemas[stage] = schema
        identity = {'model': config.model, 'prompt': prompt, 'context': context,
                      'images': image_meta, 'schema': schema, 'max_tokens': config.max_tokens,
                      'reasoning': reasoning, 'contract_version': 3,
                      'coordinates': coordinates.metadata if coordinates else None,
                      'canonical_context': canonical_context}
        if hide_dimensions:
            # The caption policy must also invalidate caches when context/schema are unchanged.
            identity['image_dimensions_in_prompt'] = False
        if single_request:
            identity['single_request'] = True
        if normalize_answer is not None:
            identity['normalization_contract'] = normalize_answer.__module__ + '.' + normalize_answer.__name__ + ':v2'
        identity.update(provider=config.provider, endpoint=ENDPOINTS[config.provider],
                        wire_options=wire_options, response_format=config.response_format)
        if config.response_format == 'json_schema' and remote_schema != schema:
            identity['wire_schema'] = remote_schema
        key = digest(identity)
        folder = self.root / stage / key
        self._last_folders[stage] = folder
        with exclusive(folder / 'call.lock'):
            cached = read(folder / 'accepted.json')
            if cached:
                answer = cached['answer']
                Draft202012Validator(canonical_schema).validate(answer)
                if validate:
                    validate(answer)
                return answer
            save(folder / 'input.json', {'provider': config.provider, 'model': config.model, 'prompt': prompt, 'context': context,
                                         'images': image_meta, 'schema': schema, 'input_key': key})
            save(folder / 'parameters.json', {'model': config.model, 'reasoning': reasoning, 'temperature': wire_options.get('temperature'),
                                              'max_tokens': config.max_tokens, 'single_request': single_request,
                                              'provider': config.provider, 'endpoint': ENDPOINTS[config.provider],
                                              'image_dimensions_in_prompt': not hide_dimensions,
                                              'request_options': wire_options, 'response_format': config.response_format,
                                              'coordinates': coordinates.metadata if coordinates else None})
            if coordinates:
                save(folder / 'coordinates.json', coordinates.metadata)
            if config.response_format == 'json_schema' and remote_schema != schema:
                save(folder / 'wire-schema.json', remote_schema)
            errors = []
            # Corrections and HTTP retries are both bounded; every POST reserves a durable slot.
            attempts = 1 if single_request else 2
            for correction in range(attempts):
                previous = (read(folder/'response-0.json').get('content') or '') if errors else ''
                messages = prompts.messages(request, error=errors[-1] if errors else None, previous=previous)
                payload = payload_for(config, stage, messages, schema, correction=bool(correction))
                prompts.record(folder, request, payload, correction)
                for network_attempt in range(attempts):
                    # Reuse a response persisted before a local crash; do not pay again for it.
                    response_path = folder / f'response-{correction}.json'
                    response = read(response_path)
                    if response:
                        break
                    if config.provider == 'codex' and self.transport is None:
                        config.check_ready()
                    number = self._reserve(stage, key)
                    receipt = self.root / 'requests' / f'{number:03d}.json'
                    started = time.monotonic()
                    route = {'provider': config.provider, 'requested_model': config.model,
                             'endpoint': ENDPOINTS[config.provider]}
                    save(receipt, {**route, 'stage': stage, 'input_key': key, 'status': 'submitted', 'at': now()})
                    try:
                        raw = self.transport(payload) if self.transport else self._post(payload, config)
                        # Intentionally omit reasoning and any provider debug/header payloads.
                        choice = (raw.get('choices') or [{}])[0]
                        response = {'id': raw.get('id'), 'model': raw.get('model'), 'provider': raw.get('provider'),
                                    'usage': raw.get('usage', {}), 'finish_reason': choice.get('finish_reason'),
                                    'content': choice.get('message', {}).get('content'),
                                    'tool_calls': bool(choice.get('message', {}).get('tool_calls'))}
                        save(response_path, response)
                        save(receipt, {**route, 'stage': stage, 'input_key': key, 'status': 'received',
                                       'seconds': round(time.monotonic()-started, 3), 'at': now(),
                                       'id': response['id'], 'model': response['model'], 'usage': response['usage']})
                        break
                    except CodexError as exc:
                        save(receipt, {**route, 'stage': stage, 'input_key': key,
                                       'status': 'sdk_error', 'error': str(exc),
                                       'seconds': round(time.monotonic()-started, 3), 'at': now()})
                        raise ModelTransportError(str(exc)) from None
                    except HTTPError as exc:
                        detail = http_error_detail(exc, config)
                        save(receipt, {**route, 'stage': stage, 'input_key': key, 'status': 'http_error',
                                       'http_status': exc.code, 'error': detail,
                                       'seconds': round(time.monotonic()-started, 3), 'at': now()})
                        if exc.code in {429, 502, 503, 504} and network_attempt + 1 < attempts:
                            self.sleep(config.http_retry_delay)
                            continue
                        reason = detail or {
                            401: '接口凭证未通过认证', 403: '接口拒绝访问，请检查模型权限',
                            429: '接口限流，请稍后重试', 524: '上游请求超时，未获得完整结果',
                            502: '上游网关错误', 503: '上游服务暂不可用', 504: '上游网关超时',
                        }.get(exc.code, '请检查模型及参数支持情况')
                        raise ModelTransportError(f'{config.provider} / {config.model} HTTP {exc.code}：{reason}') from None
                    except (URLError, TimeoutError, ConnectionError, OSError, IncompleteRead) as exc:
                        save(receipt, {**route, 'stage': stage, 'input_key': key,
                                       'status': 'transport_unknown', 'error_type': type(exc).__name__,
                                       'seconds': round(time.monotonic()-started, 3), 'at': now()})
                        raise ModelTransportError(f'{config.provider} 网络中断，远端结果未知；未自动重发，重试仍计入预算') from None
                    except ValueError:
                        save(receipt, {**route, 'stage': stage, 'status': 'invalid_response', 'at': now()})
                        raise ModelError(f'{config.provider} 返回无法解析的响应；未自动重发，请检查接口兼容性') from None
                try:
                    if response['tool_calls'] or response['finish_reason'] != 'stop':
                        raise ValueError('Expected complete JSON-only response; finish_reason=' + str(response['finish_reason']))
                    answer = parse_json(response['content'])
                    # A stage normalizer may repair raw schema defects locally.
                    # Its restored result still passes the full local schema below.
                    if payload.get('response_format', {}).get('type') == 'json_schema' and normalize_answer is None:
                        Draft202012Validator(remote_schema).validate(answer)
                    answer = restore_response(config, answer, schema)
                    if normalize_answer is not None:
                        original = answer
                        answer = normalize_answer(answer)
                        save(folder / f'normalization-{correction}.json', {
                            'contract': identity['normalization_contract'], 'changed': original != answer,
                            'raw_schema_valid': Draft202012Validator(schema).is_valid(original)})
                    Draft202012Validator(schema).validate(answer)
                    if coordinates:
                        answer = coordinates.decode(answer)
                        Draft202012Validator(canonical_schema).validate(answer)
                    if validate:
                        validate(answer)
                    save(folder / 'accepted.json', {'answer': answer, 'input_key': key, 'at': now()})
                    return answer
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append(str(exc)[:3000])
                except Exception as exc:
                    # Schema and engine domain errors are returned as bounded correction data.
                    if exc.__class__.__module__.startswith(('jsonschema', 'referencing')):
                        location = '.'.join(map(str, getattr(exc, 'absolute_path', []))) or '$'
                        if getattr(exc, 'validator', None) == 'additionalProperties':
                            allowed = list(getattr(exc, 'schema', {}).get('properties', {}))
                            errors.append(location + ': 包含未允许的字段；只允许：' + ', '.join(allowed))
                        else:
                            errors.append(location + ': ' + getattr(exc, 'message', str(exc))[:1000])
                    else:
                        raise
                save(folder / f'validation-{correction}.json', {'error': errors[-1]})
            policy = '首次输出未通过校验，未自动重试或修正' if single_request else '两次结构化输出未通过校验'
            raise ModelError(f'{stage} {policy}：{errors[-1][:700]}')

    def usage(self):
        receipts = [read(path) for path in sorted((self.root / 'requests').glob('*.json'))]
        received = [r for r in receipts if r.get('status') == 'received']
        cost_complete = len(received) == len(receipts) and all(
            isinstance((r.get('usage') or {}).get('cost'), (int, float)) for r in received)
        return {'requests': len(read(self.root / 'budget.json').get('requests', [])),
                'prompt_tokens': sum((r.get('usage') or {}).get('prompt_tokens', 0) for r in receipts),
                'completion_tokens': sum((r.get('usage') or {}).get('completion_tokens', 0) for r in receipts),
                'reported_cost_usd': sum((r.get('usage') or {}).get('cost') or 0 for r in receipts),
                'cost_complete': cost_complete}
