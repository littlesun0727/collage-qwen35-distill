"""Remote downloader requiring only Python 3.8+ and curl; verifies LFS SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {'Qwen/Qwen3.5-9B': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
             'Qwen/Qwen3.5-4B': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-4B'], default='Qwen/Qwen3.5-9B')
    parser.add_argument('--endpoint', default='https://huggingface.co')
    parser.add_argument('--revision', help='Default: the revision used in the recorded baseline')
    args = parser.parse_args()
    if os.name == 'nt':
        parser.error('Run on the remote Linux GPU host.')
    endpoint = args.endpoint.rstrip('/')
    revision = args.revision or REVISIONS[args.model]
    data = subprocess.check_output(['curl', '-fsSL', '--connect-timeout', '20', '--max-time', '120', '--retry', '3', endpoint + '/api/models/' + args.model + '/revision/' + quote(revision, safe='') + '?blobs=true'])
    info = json.loads(data)
    revision = info['sha']
    destination = ROOT / 'models' / args.model.split('/')[-1]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'source-manifest.json').write_bytes(data)

    def verify(path, item):
        if not path.is_file() or (item.get('size') is not None and path.stat().st_size != item['size']):
            return False
        expected = item.get('lfs', {}).get('sha256')
        digest = hashlib.sha256() if expected else hashlib.sha1()
        if not expected:
            digest.update(('blob %d\0' % path.stat().st_size).encode())
            expected = item.get('blobId')
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        return bool(expected) and digest.hexdigest() == expected

    def download(item):
        name = item['rfilename']
        if '..' in Path(name).parts or Path(name).is_absolute():
            raise ValueError('Unsafe model filename')
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if verify(target, item):
            print('Verified existing: ' + name, flush=True)
            return name
        partial = target.with_name(target.name + '.partial')
        url = endpoint + '/' + args.model + '/resolve/' + revision + '/' + quote(name, safe='/')
        print('Downloading: ' + name, flush=True)
        subprocess.run(['curl', '-fL', '--silent', '--show-error', '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '120', '--retry', '8', '--retry-delay', '5', '-C', '-', '-o', str(partial), url], check=True)
        if not verify(partial, item):
            raise RuntimeError('File verification failed: ' + name)
        partial.replace(target)
        print('Verified: ' + name, flush=True)
        return name

    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(download, info['siblings']))
    manifest = {'model': args.model, 'revision': revision, 'endpoint': endpoint, 'verified_files': verified}
    (destination / 'download-manifest.json').write_text(json.dumps(manifest, indent=2))
    print('DOWNLOAD_COMPLETE ' + str(destination), flush=True)

if __name__ == '__main__':
    main()
