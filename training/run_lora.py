"""Single-GPU ms-swift 4.0.4 smoke training / adapter inference launcher."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True, help='Complete local Qwen3.5-4B directory')
    p.add_argument('--gpu', required=True, help='Physical GPU index/UUID allocated to you')
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parent / 'smoke32')
    p.add_argument('--output', type=Path)
    p.add_argument('--max-steps', type=int, default=10)
    p.add_argument('--save-steps', type=int, default=10)
    p.add_argument('--resume', type=Path, help='Resume full trainer state from checkpoint')
    p.add_argument('--max-length', type=int, default=12288)
    p.add_argument('--image-tokens', type=int, default=1024)
    p.add_argument('--adapter', type=Path, help='If supplied, run inference instead of training')
    p.add_argument('--preview', action='store_true', help='Preview original model without adapter')
    p.add_argument('--images', type=Path, help='Directory of fixed preview images, not training data')
    p.add_argument('--limit', type=int, default=4)
    p.add_argument('--compare-to', type=Path, help='Previous inference run directory for side-by-side report')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    infer = bool(args.adapter or args.preview)
    model = args.model.resolve()
    if not (model / 'config.json').is_file():
        p.error('--model must be a complete local model directory with config.json')
    if min(args.max_steps, args.save_steps, args.max_length, args.image_tokens, args.limit) < 1:
        p.error('Lengths and step count must be positive')
    if ',' in args.gpu:
        p.error('This launcher is single GPU only')
    if args.adapter and not (args.adapter / 'adapter_config.json').is_file():
        p.error('--adapter must point to a saved LoRA checkpoint')
    if args.resume and (infer or not (args.resume / 'trainer_state.json').is_file()):
        p.error('--resume requires training mode and a checkpoint with trainer_state.json')
    if args.images and not infer:
        p.error('--images is for preview; use --preview or --adapter')
    records = []
    lines = [] if args.images else (args.data / 'train.jsonl').read_text(encoding='utf-8').splitlines()
    for line in lines:
        row = json.loads(line)
        row['images'] = [str((args.data / im).resolve()) for im in row['images']]
        if not all(Path(im).is_file() for im in row['images']):
            p.error('Missing image in data package')
        target = json.loads(row['messages'][-1]['content'])
        if set(target) != {'background', 'objects', 'layer_order'}:
            p.error('Unexpected assistant target fields')
        if infer:
            row['messages'] = row['messages'][:-1]
        records.append(row)
    if args.images:
        prompt = (args.data / 'prompt.txt').read_text(encoding='utf-8')
        for image in sorted(args.images.iterdir()):
            if image.is_file() and image.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                records.append({'messages': [
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': '<image>请解析参考图，只返回完整 JSON 对象，不输出代码围栏或解释。'},
                ], 'images': [str(image.resolve())]})
    if not records:
        p.error('Empty dataset')
    # Inference reuses up to four training examples solely to check loading/output.
    if infer:
        records = records[:args.limit]
    out = (args.output or Path('runs/lora') / datetime.now().strftime('%Y%m%d-%H%M%S-%f')).resolve()
    if out.exists():
        p.error('Use a new output directory')
    executable = shutil.which('swift')
    if not executable and not args.dry_run:
        p.error('swift is missing; activate collage-qwen35-train')
    out.mkdir(parents=True)
    dataset = out / 'input.jsonl'
    dataset.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    cmd = [executable or 'swift', 'infer' if infer else 'sft',
           '--model', str(model), '--use_hf', 'true', '--torch_dtype', 'bfloat16',
           '--attn_impl', 'sdpa', '--max_length', str(args.max_length),
           '--dataset_num_proc', '1']
    if infer:
        if args.adapter:
            cmd += ['--adapters', str(args.adapter.resolve())]
        cmd += ['--infer_backend', 'transformers',
                '--val_dataset', str(dataset), '--result_path', str(out / 'predictions.jsonl'),
                '--enable_thinking', 'false', '--stream', 'false', '--temperature', '0',
                '--max_new_tokens', '8192', '--max_batch_size', '1']
    else:
        cmd += ['--dataset', str(dataset), '--split_dataset_ratio', '0',
                '--tuner_type', 'lora', '--target_modules', 'all-linear',
                '--freeze_vit', 'true', '--freeze_aligner', 'true', '--freeze_llm', 'false',
                '--lora_rank', '16', '--lora_alpha', '32', '--learning_rate', '5e-5',
                '--per_device_train_batch_size', '1', '--gradient_accumulation_steps', '4',
                '--gradient_checkpointing', 'true', '--max_steps', str(args.max_steps),
                '--add_non_thinking_prefix', 'true', '--loss_scale', 'default+ignore_empty_think',
                '--packing', 'false', '--padding_free', 'false', '--truncation_strategy', 'delete',
                '--dataloader_num_workers', '0', '--logging_steps', '1',
                '--save_strategy', 'steps', '--save_steps', str(args.save_steps),
                '--save_only_model', 'false',
                '--eval_strategy', 'no', '--report_to', 'tensorboard',
                '--warmup_ratio', '0.05', '--seed', '42', '--data_seed', '42',
                '--output_dir', str(out / 'checkpoints')]
        # Keep all checkpoints for concurrent readers; cleanup is manual after preview.
        if args.resume:
            cmd += ['--resume_from_checkpoint', str(args.resume.resolve())]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, IMAGE_MAX_TOKEN_NUM=str(args.image_tokens),
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', NPROC_PER_NODE='1')
    (out / 'command.json').write_text(json.dumps({'argv': cmd, 'gpu': args.gpu,
        'image_tokens': args.image_tokens, 'purpose': 'smoke_only'}, indent=2), encoding='utf-8')
    print('Samples:', len(records), 'Output:', out, flush=True)
    print('Overlength training samples are DELETED, never truncated. Check encoded dataset size in log.', flush=True)
    if args.dry_run:
        print(json.dumps(cmd, indent=2))
        return
    with (out / 'run.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise
    if code:
        raise SystemExit(f'swift failed ({code}); see {out / "run.log"}')
    if infer:
        from preview_report import build_report
        build_report(out, args.compare_to)
    print('SWIFT_COMPLETED. Inspect loss, encoded sample count and checkpoint; this is not an accuracy evaluation.')


if __name__ == '__main__':
    main()
