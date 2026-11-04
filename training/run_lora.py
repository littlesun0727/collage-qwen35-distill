"""Single-node DDP SFT / single-GPU preview for local or HF structure datasets."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import signal
import sys

from dataset_io import load_split, inputs_only, assert_split_isolation, download_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True, help='Complete local Qwen3.5-4B directory')
    p.add_argument('--gpu', '--gpus', dest='gpu', required=True, help='Allocated GPU indices/UUIDs, e.g. 4,5')
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parent / 'smoke32')
    p.add_argument('--hf-dataset', help='Download a complete HF dataset snapshot before launching workers')
    p.add_argument('--hf-revision', help='Defaults to the inspected revision for collage-structure-sft-v1')
    p.add_argument('--output', type=Path)
    duration = p.add_mutually_exclusive_group()
    duration.add_argument('--max-steps', type=int)
    duration.add_argument('--epochs', type=float)
    p.add_argument('--save-steps', type=int, default=10)
    p.add_argument('--eval-steps', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=1, help='Per-device micro batch')
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--attn-impl', choices=['sdpa', 'flash_attention_2', 'eager'],
                   help='Default: flash_attention_2 for SFT, sdpa for Transformers preview; not used by vLLM')
    p.add_argument('--master-port', type=int, default=29500)
    p.add_argument('--resume', type=Path, help='Resume full trainer state from checkpoint')
    p.add_argument('--max-length', type=int, default=12288)
    p.add_argument('--image-tokens', type=int, default=1024)
    p.add_argument('--adapter', type=Path, help='If supplied, run inference instead of training')
    p.add_argument('--preview', action='store_true', help='Preview original model without adapter')
    p.add_argument('--infer-backend', choices=['transformers', 'vllm'], help='Preview only; default transformers')
    p.add_argument('--infer-batch-size', type=int, help='Preview submission/write batch; default 1 (Transformers), 8 (vLLM)')
    p.add_argument('--max-new-tokens', type=int, help='Preview output limit; default 8192')
    p.add_argument('--temperature', type=float, help='Inference only; default 0 (greedy)')
    p.add_argument('--repetition-penalty', type=float,
                   help='Inference only; 1 disables penalty; omitted preserves SWIFT/model default')
    p.add_argument('--merge-lora', action='store_true', help='Merge saved adapter on CPU before inference (SWIFT export)')
    p.add_argument('--vllm-gpu-memory-utilization', type=float, help='Fraction of whole GPU memory; default 0.7')
    p.add_argument('--vllm-max-model-len', type=int, help='Total context; default max-length + max-new-tokens')
    p.add_argument('--vllm-enforce-eager', action='store_true', help='Skip vLLM CUDA graphs to reduce memory use')
    p.add_argument('--images', type=Path, help='Directory of fixed preview images, not training data')
    p.add_argument('--limit', type=int, default=4)
    p.add_argument('--preview-split', choices=['train', 'validation', 'test'],
                   help='Defaults to validation when present, otherwise train; never used for SFT')
    p.add_argument('--compare-to', type=Path, help='Previous inference run directory for side-by-side report')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    infer = bool(args.adapter or args.preview)
    vllm_options = (args.vllm_gpu_memory_utilization is not None or
                    args.vllm_max_model_len is not None or args.vllm_enforce_eager)
    if not infer and (args.infer_backend is not None or args.infer_batch_size is not None or
                      args.max_new_tokens is not None or args.merge_lora or vllm_options or
                      args.temperature is not None or args.repetition_penalty is not None):
        p.error('Inference options require --preview or --adapter; they cannot affect SFT')
    backend = (args.infer_backend or 'transformers') if infer else None
    if vllm_options and backend != 'vllm':
        p.error('vLLM options require --infer-backend vllm')
    if backend == 'vllm' and args.attn_impl:
        p.error('--attn-impl controls Transformers, not vLLM; omit it for vLLM inference')
    if args.merge_lora and not args.adapter:
        p.error('--merge-lora requires --adapter pointing to a completed checkpoint')
    attn_impl = args.attn_impl or ('sdpa' if infer else 'flash_attention_2')
    infer_batch = args.infer_batch_size if args.infer_batch_size is not None else (8 if backend == 'vllm' else 1)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 8192
    temperature = args.temperature if args.temperature is not None else 0.0
    if not math.isfinite(temperature) or temperature < 0:
        p.error('--temperature must be finite and >= 0')
    if args.repetition_penalty is not None and (
            not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0):
        p.error('--repetition-penalty must be finite and > 0')
    vllm_memory = args.vllm_gpu_memory_utilization if args.vllm_gpu_memory_utilization is not None else 0.7
    vllm_length = args.vllm_max_model_len if args.vllm_max_model_len is not None else args.max_length + max_new_tokens
    if infer_batch < 1 or max_new_tokens < 1:
        p.error('Inference batch size and max-new-tokens must be positive')
    if not 0 < vllm_memory <= 1:
        p.error('vLLM GPU memory utilization must be in (0, 1]')
    if backend == 'vllm' and vllm_length < args.max_length + max_new_tokens:
        p.error('vLLM context must cover max-length (prompt cap) + max-new-tokens; lower those caps or increase context')
    if args.epochs is None and args.max_steps is None:
        args.max_steps = 10
    gpu_ids = [part.strip() for part in args.gpu.split(',')]
    if not all(gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        p.error('GPU list must contain distinct nonempty indices/UUIDs')
    world_size = len(gpu_ids)
    model = args.model.resolve()
    if not (model / 'config.json').is_file():
        p.error('--model must be a complete local model directory with config.json')
    if min(args.save_steps, args.eval_steps, args.batch_size, args.grad_accum,
           args.max_length, args.image_tokens, args.limit) < 1:
        p.error('Lengths and step count must be positive')
    if (args.max_steps is not None and args.max_steps < 1) or (args.epochs is not None and args.epochs <= 0):
        p.error('Training duration must be positive')
    if args.learning_rate <= 0 or not 1 <= args.master_port <= 65535:
        p.error('Invalid learning rate or master port')
    if infer and world_size != 1:
        p.error('Preview uses one GPU; multi-GPU is supported for training only')
    if args.preview_split and not infer:
        p.error('--preview-split is only for inference; test must never enter training')
    if args.adapter and not (args.adapter / 'adapter_config.json').is_file():
        p.error('--adapter must point to a saved LoRA checkpoint')
    if args.resume and (infer or not (args.resume / 'trainer_state.json').is_file()):
        p.error('--resume requires training mode and a checkpoint with trainer_state.json')
    if args.images and not infer:
        p.error('--images is for preview; use --preview or --adapter')
    if args.hf_revision and not args.hf_dataset:
        p.error('--hf-revision requires --hf-dataset')
    provenance = {'local_dir': str(args.data.resolve())}
    if args.hf_dataset:
        args.data, provenance = download_snapshot(args.hf_dataset, args.hf_revision)
    elif (args.data / 'download-source.json').is_file():
        provenance.update(json.loads((args.data / 'download-source.json').read_text(encoding='utf-8')))
    validation = []
    split = (args.preview_split or ('validation' if (args.data / 'validation.jsonl').is_file() else 'train')) if infer else 'train'
    records = load_split(args.data, split)
    if not infer and (args.data / 'validation.jsonl').is_file():
        validation = load_split(args.data, 'validation')
        assert_split_isolation(records, validation)
    if args.images:
        # Preserve the selected dataset's exact message roles and prompt.
        template = inputs_only(records[0])
        records = []
        for image in sorted(args.images.iterdir()):
            if image.is_file() and image.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                records.append({'messages': template['messages'], 'images': [str(image.resolve())]})
    elif infer:
        records = [inputs_only(row) for row in records]
    if not records:
        p.error('Empty dataset')
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
    val_path = out / 'validation.jsonl'
    if validation:
        val_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in validation), encoding='utf-8')
    cmd = [executable or 'swift', 'infer' if infer else 'sft',
           '--model', str(model), '--use_hf', 'true', '--torch_dtype', 'bfloat16',
           '--max_length', str(args.max_length),
           '--dataset_num_proc', '1']
    if backend != 'vllm':
        cmd += ['--attn_impl', attn_impl]
    if infer:
        if args.adapter:
            cmd += ['--adapters', str(args.adapter.resolve())]
        if args.merge_lora:
            cmd += ['--merge_lora', 'true']
        cmd += ['--infer_backend', backend,
                '--val_dataset', str(dataset), '--result_path', str(out / 'predictions.jsonl'),
                '--enable_thinking', 'false', '--stream', 'false', '--temperature', str(temperature),
                '--max_new_tokens', str(max_new_tokens), '--write_batch_size', str(infer_batch)]
        if args.repetition_penalty is not None:
            cmd += ['--repetition_penalty', str(args.repetition_penalty)]
        if backend == 'vllm':
            # Do not inherit the training attention backend during CPU merge.
            # Model, data, precision and generation settings are explicit here.
            cmd += ['--load_args', 'false',
                    '--vllm_gpu_memory_utilization', str(vllm_memory),
                    '--vllm_max_model_len', str(vllm_length),
                    '--vllm_max_num_seqs', str(infer_batch),
                    '--vllm_tensor_parallel_size', '1',
                    '--vllm_limit_mm_per_prompt', '{"image": 1, "video": 0}']
            if args.vllm_enforce_eager:
                cmd += ['--vllm_enforce_eager', 'true']
        else:
            cmd += ['--max_batch_size', str(infer_batch)]
    else:
        cmd += ['--dataset', str(dataset), '--split_dataset_ratio', '0',
                '--tuner_type', 'lora', '--target_modules', 'all-linear',
                '--freeze_vit', 'true', '--freeze_aligner', 'true', '--freeze_llm', 'false',
                '--lora_rank', '16', '--lora_alpha', '32', '--learning_rate', str(args.learning_rate),
                '--per_device_train_batch_size', str(args.batch_size), '--gradient_accumulation_steps', str(args.grad_accum),
                '--gradient_checkpointing', 'true',
                '--add_non_thinking_prefix', 'true', '--loss_scale', 'default+ignore_empty_think',
                '--packing', 'false', '--padding_free', 'false', '--truncation_strategy', 'delete',
                '--dataloader_num_workers', '0', '--logging_steps', '1',
                '--save_strategy', 'steps', '--save_steps', str(args.save_steps),
                '--save_only_model', 'false',
                '--report_to', 'tensorboard',
                '--warmup_ratio', '0.05', '--seed', '42', '--data_seed', '42',
                '--output_dir', str(out / 'checkpoints')]
        if args.epochs is not None:
            cmd += ['--max_steps', '-1', '--num_train_epochs', str(args.epochs)]
        else:
            cmd += ['--max_steps', str(args.max_steps)]
        if validation:
            cmd += ['--val_dataset', str(val_path), '--eval_strategy', 'steps',
                    '--eval_steps', str(args.eval_steps), '--per_device_eval_batch_size', '1']
        else:
            cmd += ['--eval_strategy', 'no']
        if world_size > 1:
            cmd += ['--ddp_find_unused_parameters', 'false']
        # Keep all checkpoints for concurrent readers; cleanup is manual after preview.
        if args.resume:
            cmd += ['--resume_from_checkpoint', str(args.resume.resolve())]
    if backend == 'vllm':
        # Keep the compatibility shim confined to the inference subprocess.
        cmd = [sys.executable, str(Path(__file__).with_name('vllm_infer.py'))] + cmd[2:]
    env = os.environ.copy()
    # This is a single-node parent launcher, not a script to run inside torchrun.
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                'ROLE_RANK', 'ROLE_WORLD_SIZE', 'NNODES', 'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT', 'NPROC_PER_NODE'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=','.join(gpu_ids), IMAGE_MAX_TOKEN_NUM=str(args.image_tokens),
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               PYTHONUNBUFFERED='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if not infer:
        env.update(NPROC_PER_NODE=str(world_size), NNODES='1', NODE_RANK='0',
                   MASTER_ADDR='127.0.0.1', MASTER_PORT=str(args.master_port))
    (out / 'command.json').write_text(json.dumps({'argv': cmd, 'gpu': args.gpu,
        'world_size': world_size, 'effective_batch_size': None if infer else world_size * args.batch_size * args.grad_accum,
        'image_tokens': args.image_tokens, 'source': provenance, 'split': split,
        'infer_backend': backend, 'infer_batch_size': infer_batch if infer else None,
        'max_new_tokens': max_new_tokens if infer else None,
        'temperature': temperature if infer else None,
        'repetition_penalty': args.repetition_penalty if infer else None,
        'train_count': len(records) if not infer else 0, 'validation_count': len(validation),
        'launch_env': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'NPROC_PER_NODE', 'MASTER_PORT') if k in env},
        'purpose': 'preview' if infer else 'sft'}, indent=2), encoding='utf-8')
    print('Samples:', len(records), 'Validation:', len(validation), 'Output:', out, flush=True)
    if not infer:
        print('GPUs:', world_size, 'Effective batch:', world_size * args.batch_size * args.grad_accum, flush=True)
    if infer:
        print('Inference backend:', backend, 'Submission/write batch:', infer_batch,
              'Max new tokens:', max_new_tokens, flush=True)
        print('Temperature:', temperature, 'Repetition penalty:',
              args.repetition_penalty if args.repetition_penalty is not None else 'SWIFT/model default', flush=True)
        print('Preview updates after each completed batch; use --infer-batch-size 1 for per-image updates.', flush=True)
    else:
        print('Overlength training samples are DELETED, never truncated. Check encoded dataset size in log.', flush=True)
    if args.dry_run:
        print(json.dumps(cmd, indent=2))
        return
    watcher = None
    if infer:
        from live_preview import LivePreview
        watcher = LivePreview(out, len(records), args.compare_to)
    with (out / 'run.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace', bufsize=1,
                                start_new_session=(os.name == 'posix'))
        if watcher:
            watcher.start()
        code = None
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = proc.wait()
        except KeyboardInterrupt:
            if os.name == 'posix':
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait()
            raise
        finally:
            if watcher:
                try:
                    watcher.finish(success=(code == 0))
                except Exception as exc:
                    print(f'Preview generation failed; raw predictions retained: {exc}', flush=True)
    if code:
        raise SystemExit(f'swift failed ({code}); see {out / "run.log"}')
    print('SWIFT_COMPLETED. Inspect loss, encoded sample count and checkpoint; this is not an accuracy evaluation.')


if __name__ == '__main__':
    main()
