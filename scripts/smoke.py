"""Check service health and optionally send one image, using only stdlib."""
import argparse
import base64
import json
import mimetypes
from pathlib import Path
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8000')
parser.add_argument('--image', type=Path)
args = parser.parse_args()
with urlopen(args.url + '/v1/models', timeout=30) as response:
    models = json.load(response)
print(json.dumps(models, ensure_ascii=False))
if args.image:
    mime = mimetypes.guess_type(args.image.name)[0] or 'image/png'
    encoded = base64.b64encode(args.image.read_bytes()).decode()
    body = {'model': models['data'][0]['id'], 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': '请简短描述这张图片。'},
        {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{encoded}'}}
    ]}], 'max_tokens': 256, 'chat_template_kwargs': {'enable_thinking': False}}
    request = Request(args.url + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=300) as response:
        print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
