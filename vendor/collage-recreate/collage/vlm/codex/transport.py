"""Adapt the existing image/JSON request contract to the official Codex SDK."""

import base64
import binascii
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
BRIDGE = Path(__file__).with_name('codex_bridge.mjs')
MAX_RESPONSE = 8 * 1024 * 1024


class CodexError(RuntimeError):
    """Only fixed, credential-free diagnostics cross the worker boundary."""


ERRORS = {
    'home_unavailable': 'Codex 启动失败：服务进程无法访问当前用户目录。请从已登录的 Windows 用户环境启动工作台；本次未发起模型推理。',
    'auth': 'Codex 登录无效或已过期，请先在终端运行 codex login。',
    'model': '当前 Codex 账号不可使用此模型，请检查模型 ID 和账号权限。',
    'quota': 'Codex 额度不足或受到限流，请稍后重试。',
    'timeout': 'Codex SDK 请求超时，已终止本次调用；远端结果未知，未自动重发。',
    'tools': 'Codex 返回了工具调用，本接口只接受图片分析与 JSON 结果。',
    'invalid': 'Codex SDK 未返回完整 JSON 结果，请检查模型与 Schema 兼容性。',
    'runtime': 'Codex SDK 启动失败，请在项目目录运行 npm ci，并确认本机 codex login 可用。',
    'failed': 'Codex SDK 调用失败；请检查本机 Codex 登录、模型权限和网络。未自动重发。',
}


def check_runtime():
    node = shutil.which('node')
    if not node or not (ROOT/'node_modules/@openai/codex-sdk/dist/index.js').is_file():
        raise ValueError('Codex SDK 尚未安装：需要 Node.js 18+，然后在项目目录运行 npm ci，再运行 codex login。')
    override = os.getenv('COLLAGE_CODEX_PATH')
    if override and not Path(override).is_file():
        raise ValueError('COLLAGE_CODEX_PATH 必须指向存在的 Codex 可执行文件。')
    return node


def worker_request(payload, folder, timeout):
    """Stage supplied bytes for the worker; retain the original text/image order."""
    instructions, inputs = [], []
    for message in payload['messages']:
        content = message['content']
        if message['role'] == 'system':
            if not isinstance(content, str):
                raise ValueError('Codex system instructions must be text')
            instructions.append(content)
            continue
        if message['role'] != 'user':
            raise ValueError('Codex bridge accepts system/user messages only')
        for part in ([{'type': 'text', 'text': content}] if isinstance(content, str) else content):
            if part['type'] == 'text':
                inputs.append({'type': 'text', 'text': part['text']})
            elif part['type'] == 'image_url':
                match = re.fullmatch(r'data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=\r\n]+)',
                                     part['image_url']['url'])
                if not match:
                    raise ValueError('Codex bridge requires inline PNG/JPEG/WebP images')
                try:
                    data = base64.b64decode(match[2], validate=True)
                except binascii.Error:
                    raise ValueError('Invalid inline image') from None
                index = sum(item['type'] == 'local_image' for item in inputs) + 1
                path = folder/f'image-{index}.{match[1]}'
                path.write_bytes(data)
                # Files keep large images out of the worker's stdin envelope.
                # The App Server adapter sends their bytes as native image inputs.
                inputs.append({'type': 'local_image', 'path': str(path), 'detail': 'original'})
            else:
                raise ValueError('Unsupported Codex input part')
    instruction_file = folder/'instructions.txt'
    instruction_file.write_text('\n\n'.join(instructions), encoding='utf-8')
    request = {'model': payload['model'], 'effort': payload['modelReasoningEffort'],
               'input': inputs, 'instructions_file': str(instruction_file),
               'working_directory': str(folder), 'timeout_ms': max(1, int(timeout * 1000))}
    output = payload.get('response_format', {})
    if output.get('type') == 'json_schema':
        request['schema'] = output['json_schema']['schema']
    # The SDK has no separate JSON-object mode. Both non-schema settings rely
    # on the existing JSON instructions and Client's local validation.
    return request


def stop_worker(process):
    if os.name == 'nt':
        # Terminate the SDK's Codex child as well as the Node bridge.
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=10, check=False)
    else:
        import signal
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.communicate()


def complete(payload, timeout):
    node = check_runtime()
    with tempfile.TemporaryDirectory(prefix='collage-codex-') as temp:
        request = worker_request(payload, Path(temp), timeout)
        kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}
        try:
            process = subprocess.Popen([node, str(BRIDGE)], cwd=temp, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **kwargs)
        except OSError:
            raise CodexError(ERRORS['runtime']) from None
        try:
            output, _ = process.communicate(json.dumps(request, ensure_ascii=False).encode('utf-8'),
                                            timeout=timeout + 10)
        except subprocess.TimeoutExpired:
            stop_worker(process)
            raise CodexError(ERRORS['timeout']) from None
        except BaseException:
            stop_worker(process)
            raise
        if len(output) > MAX_RESPONSE:
            raise CodexError(ERRORS['invalid'])
        try:
            result = json.loads(output)
        except (ValueError, UnicodeError):
            raise CodexError(ERRORS['runtime'] if process.returncode else ERRORS['invalid']) from None
        if not isinstance(result, dict):
            raise CodexError(ERRORS['invalid'])
        if result.get('error') or process.returncode:
            code = result.get('error')
            raise CodexError(ERRORS.get(code if isinstance(code, str) else '', ERRORS['failed']))
        if not result.get('choices'):
            raise CodexError(ERRORS['invalid'])
        return result
