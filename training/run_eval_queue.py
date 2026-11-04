"""Run base/LoRA evaluations serially on one GPU; all paths relative to the working directory."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys


def make_plan(config, gpu, output_root, batch_size=8):
    jobs = config.get('jobs')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('jobs must be a nonempty list')
    data = Path(config['data']).resolve()
    if not (data / 'test.jsonl').is_file():
        raise ValueError(f'Missing test.jsonl: {data}')
    if batch_size < 1 or ',' in gpu or not gpu.strip():
        raise ValueError('Use a positive batch size and exactly one GPU')
    plan, seen = [], {}
    for job in jobs:
        name = job['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) or name in seen:
            raise ValueError(f'Invalid or duplicate job name: {name}')
        model = Path(job.get('model', config.get('model', ''))).resolve()
        if not (model / 'config.json').is_file():
            raise ValueError(f'Missing model config.json for {name}: {model}')
        out = output_root / name
        cmd = [sys.executable, '-u', str(Path(__file__).with_name('run_lora.py').resolve()),
               '--model', str(model), '--data', str(data), '--gpu', gpu,
               '--infer-backend', 'vllm', '--preview-split', 'test', '--limit', '137',
               '--infer-batch-size', str(batch_size), '--image-tokens', '1024',
               '--max-length', '12288', '--max-new-tokens', '8192',
               '--vllm-gpu-memory-utilization', '0.7', '--temperature', '0.7',
               '--top-p', '0.8', '--top-k', '20', '--min-p', '0',
               '--presence-penalty', '1.5', '--repetition-penalty', '1.0',
               '--output', str(out)]
        merge_lora = job.get('merge_lora', False)
        if not isinstance(merge_lora, bool):
            raise ValueError(f'{name}: merge_lora must be a JSON boolean')
        if merge_lora and not job.get('adapter'):
            raise ValueError(f'{name}: merge_lora requires an adapter')
        if job.get('adapter'):
            adapter = Path(job['adapter']).resolve()
            if not (adapter / 'adapter_config.json').is_file():
                raise ValueError(f'Missing adapter_config.json for {name}: {adapter}')
            cmd += ['--adapter', str(adapter)]
            if merge_lora:
                cmd += ['--merge-lora']
        else:
            cmd += ['--preview']
        baseline = job.get('compare_to')
        if baseline:
            if baseline not in seen:
                raise ValueError(f'{name}: compare_to must name an earlier job')
            cmd += ['--compare-to', str(seen[baseline])]
        plan.append({'name': name, 'compare_to': baseline, 'output': str(out), 'argv': cmd})
        seen[name] = out
    return plan


def execute_plan(plan, output_root, continue_on_error=False):
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    results, succeeded = [], set()
    def save():
        temp = output_root / 'queue-status.json.tmp'
        temp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(output_root / 'queue-status.json')
    for index, job in enumerate(plan, 1):
        result = {'name': job['name'], 'output': job['output'], 'status': 'running'}
        results.append(result)
        if job['compare_to'] and job['compare_to'] not in succeeded:
            result['status'] = 'skipped: baseline failed'
            save()
            print(f"[QUEUE] 跳过 {job['name']}：基线未成功", flush=True)
            continue
        save()
        print(f"[QUEUE] 开始 {index}/{len(plan)}：{job['name']}", flush=True)
        try:
            # Blocking: next job starts only after this full inference process exits.
            code = subprocess.run(job['argv'], check=False).returncode
        except KeyboardInterrupt:
            result['status'] = 'interrupted'
            save()
            raise
        except OSError as exc:
            code = -1
            result['error'] = str(exc)
        result.update(status='success' if code == 0 else 'failed', returncode=code)
        save()
        print(f"[QUEUE] 结束 {job['name']}：{result['status']}，输出 {job['output']}", flush=True)
        if code == 0:
            succeeded.add(job['name'])
        elif not continue_on_error:
            print('[QUEUE] 已停止；剩余任务未启动。排查对应 run.log 后用新目录重跑。', flush=True)
            return 1
    return int(any(r['status'] != 'success' for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--gpu', default='4')
    parser.add_argument('--infer-batch-size', type=int, default=8)
    parser.add_argument('--output-root', type=Path,
                        default=Path('runs') / ('eval-queue-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate paths and print commands; no inference')
    args = parser.parse_args()
    try:
        root = args.output_root.resolve()
        if root.exists():
            raise ValueError('Use a new output-root directory')
        config = json.loads(args.config.read_text(encoding='utf-8-sig'))
        plan = make_plan(config, args.gpu, root, args.infer_batch_size)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    raise SystemExit(execute_plan(plan, root, args.continue_on_error))


if __name__ == '__main__':
    main()
