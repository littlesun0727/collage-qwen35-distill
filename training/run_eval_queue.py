"""Run base/LoRA evaluations serially on one GPU; all paths relative to the working directory."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import json
import math
from pathlib import Path
import re
import subprocess
import sys


SAMPLING_DEFAULTS = {
    'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0,
    'presence_penalty': 1.5, 'repetition_penalty': 1.0,
}


def sampling_overrides(value, location):
    if not isinstance(value, dict):
        raise ValueError(f'{location} must be a JSON object')
    unknown = set(value) - SAMPLING_DEFAULTS.keys()
    if unknown:
        raise ValueError(f'{location}: unknown sampling keys: {sorted(unknown)}')
    for key, number in value.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f'{location}.{key} must be a finite JSON number')
        valid = {
            'temperature': number >= 0,
            'top_p': 0 < number <= 1,
            'top_k': isinstance(number, int) and number >= 1,
            'min_p': 0 <= number <= 1,
            'presence_penalty': -2 <= number <= 2,
            'repetition_penalty': number > 0,
        }[key]
        if not valid:
            raise ValueError(f'{location}.{key} is outside its allowed range/type')
    return value


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]


def option(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def inspect_result(job):
    """Recognize complete old standalone runs too; a directory alone is not success."""
    out = Path(job['output'])
    if not out.exists():
        return 'run', '没有已有结果'
    try:
        saved = json.loads((out / 'command.json').read_text(encoding='utf-8'))
        inputs = read_jsonl(out / 'input.jsonl')
        predictions = read_jsonl(out / 'predictions.jsonl')
        argv = job['argv']
        # Compare the actual dataset inputs, including prompts and image paths.
        data = Path(option(argv, '--data'))
        expected = read_jsonl(data / 'test.jsonl')[:int(option(argv, '--limit'))]
        for row in expected:
            row['messages'] = row['messages'][:-1]
            row['images'] = [str((data / image).resolve()) for image in row['images']]
        expected = [{'messages': r['messages'], 'images': r['images']} for r in expected]
        if inputs != expected:
            return 'rerun', '输入图片或 prompt 已变化'
        for key in ('image_tokens', 'max_new_tokens', 'temperature', 'top_p', 'top_k',
                    'min_p', 'presence_penalty', 'repetition_penalty'):
            value = option(argv, '--' + key.replace('_', '-'))
            if saved.get(key) is None or float(saved[key]) != float(value):
                return 'rerun', f'配置不同：{key}'
        for requested, recorded in (('--model', '--model'), ('--adapter', '--adapters'),
                                    ('--max-length', '--max_length')):
            if option(argv, requested) != option(saved['argv'], recorded):
                return 'rerun', f'配置不同：{requested}'
        if saved.get('infer_backend') != 'vllm' or saved.get('split') != 'test':
            return 'rerun', '后端或数据 split 不同'
        if ('--merge-lora' in argv) != (option(saved['argv'], '--merge_lora') == 'true'):
            return 'rerun', 'LoRA 合并方式不同'
        if option(saved['argv'], '--enable_thinking') != 'false':
            return 'rerun', 'thinking 配置不同'
        def image_key(row):
            paths = row['images']
            return tuple(str(Path(p['path'] if isinstance(p, dict) else p).resolve())
                         for p in paths)
        if (not inputs or len(predictions) != len(inputs)
                or Counter(map(image_key, inputs)) != Counter(map(image_key, predictions))
                or any(not isinstance(r.get('response'), str) for r in predictions)):
            return 'rerun', '预测记录不完整'
        # Bad model JSON is still a completed evaluation, not a reason to resample.
        return 'reuse', f'完整预测 {len(predictions)} 条，配置一致'
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        return 'rerun', f'结果缺失或无法核验：{exc}'


@contextmanager
def queue_lock(root):
    """OS releases the lock on crashes; stale lock files do not block resume."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.queue.lock').open('a+b') as handle:
        handle.write(b'0')
        handle.flush()
        handle.seek(0)
        try:
            if sys.platform == 'win32':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError(f'输出目录已有队列正在运行：{root}') from exc
        try:
            yield
        finally:
            if sys.platform == 'win32':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def make_plan(config, gpu, output_root, batch_size=8):
    jobs = config.get('jobs')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('jobs must be a nonempty list')
    defaults = {**SAMPLING_DEFAULTS,
                **sampling_overrides(config.get('sampling', {}), 'sampling')}
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
        sampling = {**defaults,
                    **sampling_overrides(job.get('sampling', {}), f'jobs[{name}].sampling')}
        cmd = [sys.executable, '-u', str(Path(__file__).with_name('run_lora.py').resolve()),
               '--model', str(model), '--data', str(data), '--gpu', gpu,
               '--infer-backend', 'vllm', '--preview-split', 'test', '--limit', '137',
               '--infer-batch-size', str(batch_size), '--image-tokens', '1024',
               '--max-length', '12288', '--max-new-tokens', '8192',
               '--vllm-gpu-memory-utilization', '0.7',
               '--output', str(out)]
        for key, value in sampling.items():
            cmd += ['--' + key.replace('_', '-'), str(value)]
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
        plan.append({'name': name, 'compare_to': baseline, 'output': str(out),
                     'sampling': sampling, 'argv': cmd})
        seen[name] = out
    return plan


def execute_plan(plan, output_root, continue_on_error=False):
    with queue_lock(output_root):
        return _execute_plan(plan, output_root, continue_on_error)


def _execute_plan(plan, output_root, continue_on_error=False):
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
        action, reason = inspect_result(job)
        result.update(action=action, reason=reason)
        print(f"[QUEUE] {job['name']}：{action}，{reason}", flush=True)
        try:
            if action == 'reuse':
                code = 0
            else:
                original_out = Path(job['output'])
                out = original_out.resolve()
                if out.exists():
                    # Keep incomplete or superseded runs under the same root, never delete them.
                    if out.parent != output_root.resolve() or original_out.is_symlink():
                        raise ValueError(f'Unsafe result directory: {out}')
                    archive = output_root / '_history' / job['name']
                    archive.mkdir(parents=True, exist_ok=True)
                    destination = archive / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
                    out.rename(destination)
                    result['archived_to'] = str(destination)
                    save()
                print(f"[QUEUE] 开始 {index}/{len(plan)}：{job['name']}", flush=True)
                code = subprocess.run(job['argv'], check=False).returncode
                if code == 0:
                    verified, detail = inspect_result(job)
                    if verified != 'reuse':
                        code = 1
                        result['error'] = f'进程退出成功，但结果校验失败：{detail}'
            if code == 0:
                # Live preview is best-effort. Require a final scored report for BOTH
                # fresh inference and reused predictions before declaring success.
                result['stage'] = 'report'
                save()
                report = [sys.executable, '-u', str(Path(__file__).with_name('preview_report.py')),
                          '--run', job['output'], '--gt',
                          str(Path(option(job['argv'], '--data')) / 'test.jsonl')]
                if job['compare_to']:
                    baseline = next(j['output'] for j in plan if j['name'] == job['compare_to'])
                    report += ['--baseline', baseline]
                print(f"[QUEUE] Generating GT metrics and final preview: {job['name']}", flush=True)
                code = subprocess.run(report, check=False).returncode
                if code != 0:
                    result['error'] = ('Final metrics report failed; predictions retained. '
                                       'Rerun with the same output-root to retry the report.')
        except KeyboardInterrupt:
            result['status'] = 'interrupted'
            save()
            raise
        except (OSError, ValueError) as exc:
            code = -1
            result['error'] = str(exc)
        result.update(status='success' if code == 0 else 'failed', returncode=code)
        save()
        print(f"[QUEUE] 结束 {job['name']}：{result['status']}，输出 {job['output']}", flush=True)
        if code == 0:
            succeeded.add(job['name'])
        elif not continue_on_error:
            print('[QUEUE] 已停止；修复问题后用相同 output-root 重跑即可。', flush=True)
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
        config = json.loads(args.config.read_text(encoding='utf-8-sig'))
        plan = make_plan(config, args.gpu, root, args.infer_batch_size)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        for job in plan:
            job['action'], job['reason'] = inspect_result(job)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    try:
        code = execute_plan(plan, root, args.continue_on_error)
    except ValueError as exc:
        parser.error(str(exc))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
