"""Run on the Linux GPU host only; weights and cache remain on that host."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.cache/download-tools'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-4B'], default='Qwen/Qwen3.5-9B')
    parser.add_argument('--revision')
    args = parser.parse_args()
    if os.name == 'nt':
        parser.error('Download on the remote Linux GPU host, not Windows.')
    os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
    from huggingface_hub import HfApi, snapshot_download
    from download_curl import REVISIONS
    info = HfApi().model_info(args.model, revision=args.revision or REVISIONS[args.model], files_metadata=True)
    revision = info.sha
    destination = ROOT / 'models' / args.model.split('/')[-1]
    snapshot_download(args.model, revision=revision, local_dir=str(destination))
    import hashlib
    verified = []
    for item in info.siblings:
        path = destination / item.rfilename
        if not path.is_file() or (item.size is not None and path.stat().st_size != item.size):
            raise RuntimeError('Missing or incomplete file: ' + item.rfilename)
        if item.lfs:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != item.lfs.sha256:
                raise RuntimeError('SHA256 mismatch: ' + item.rfilename)
        verified.append(item.rfilename)
    (destination / 'download-manifest.json').write_text(json.dumps({'model': args.model, 'revision': revision, 'endpoint': os.environ.get('HF_ENDPOINT', 'https://huggingface.co'), 'verified_files': verified}, indent=2))
    print(destination)

if __name__ == '__main__':
    main()
