"""CPU tests for HF contract preservation, split isolation and DDP launch arguments."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
from dataset_io import load_split, assert_split_isolation, validate_target


class DatasetDDPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'dataset'
        (self.data / 'images').mkdir(parents=True)
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        self.target = {'version': 'structure-analysis-v9-12-frame-photo-list',
                       'coordinate_mode': 'normalized_1000', 'background': {'type': 'transparent'},
                       'objects': [{'id': 'photo_1', 'kind': 'photo', 'bbox': [0, 0, 1000, 1000],
                                    'description': '测试照片'}], 'layer_order': ['photo_1']}
        self.prompt = '<image>\n冻结的用户消息任务指令'
        self.answer = json.dumps(self.target, ensure_ascii=False)
        for split in ['train', 'validation', 'test']:
            Image.new('RGB', (10, 20), 'white').save(self.data / 'images' / f'{split}.png')
            row = {'messages': [{'role': 'user', 'content': self.prompt},
                                {'role': 'assistant', 'content': self.answer}],
                   'images': [f'images/{split}.png']}
            (self.data / f'{split}.jsonl').write_text(json.dumps(row, ensure_ascii=False)+'\n', encoding='utf-8')
        self.launcher = Path(__file__).with_name('run_lora.py')

    def tearDown(self):
        self.temp.cleanup()

    def launch(self, extra, name='run', success=True):
        out = self.root / name
        result = subprocess.run([sys.executable, str(self.launcher), '--model', str(self.model),
            '--data', str(self.data), '--output', str(out), '--dry-run'] + extra,
            capture_output=True, text=True, encoding='utf-8', errors='replace')
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return out, json.loads((out / 'command.json').read_text(encoding='utf-8'))
        self.assertNotEqual(result.returncode, 0)
        return result

    def test_two_gpu_epochs_and_split_preservation(self):
        out, info = self.launch(['--gpus', '4,5', '--epochs', '1', '--grad-accum', '8'])
        self.assertEqual(info['effective_batch_size'], 16)
        self.assertEqual(info['launch_env']['NPROC_PER_NODE'], '2')
        self.assertEqual(info['launch_env']['CUDA_VISIBLE_DEVICES'], '4,5')
        cmd = info['argv']
        self.assertEqual(cmd[cmd.index('--max_steps')+1], '-1')
        self.assertEqual(cmd[cmd.index('--num_train_epochs')+1], '1.0')
        self.assertEqual(cmd[cmd.index('--split_dataset_ratio')+1], '0')
        self.assertEqual(cmd[cmd.index('--attn_impl')+1], 'flash_attention_2')
        self.assertFalse(any(x.startswith('--vllm') for x in cmd))
        self.assertNotIn('--infer_backend', cmd)
        self.assertEqual(cmd[1], 'sft')
        self.assertIn('--val_dataset', cmd)
        self.assertNotIn('test.jsonl', ' '.join(cmd))
        for split, file in [('train', 'input.jsonl'), ('validation', 'validation.jsonl')]:
            row = json.loads((out / file).read_text(encoding='utf-8'))
            self.assertEqual(row['messages'][0], {'role': 'user', 'content': self.prompt})
            self.assertEqual(row['messages'][1]['content'], self.answer)
            self.assertEqual(row['images'], [str((self.data / 'images' / f'{split}.png').resolve())])

    def test_preview_defaults_to_validation_and_strips_answer(self):
        out, info = self.launch(['--gpu', '6', '--preview'])
        row = json.loads((out / 'input.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(row['messages'], [{'role': 'user', 'content': self.prompt}])
        self.assertTrue(row['images'][0].endswith('validation.png'))
        self.assertNotIn('NPROC_PER_NODE', info['launch_env'])
        self.assertEqual(info['split'], 'validation')
        self.assertEqual(info['argv'][info['argv'].index('--write_batch_size') + 1], '1')
        self.assertEqual(info['argv'][info['argv'].index('--attn_impl') + 1], 'sdpa')

    def test_vllm_batch_context_and_adapter(self):
        adapter = self.root / 'checkpoint-10'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text('{}')
        out, info = self.launch(['--gpu', '6', '--adapter', str(adapter), '--merge-lora',
            '--infer-backend', 'vllm', '--preview-split', 'test', '--max-new-tokens', '4096'])
        cmd = info['argv']
        self.assertEqual(cmd[cmd.index('--infer_backend')+1], 'vllm')
        self.assertEqual(cmd[cmd.index('--write_batch_size')+1], '8')
        self.assertEqual(cmd[cmd.index('--vllm_max_num_seqs')+1], '8')
        self.assertEqual(cmd[cmd.index('--vllm_max_model_len')+1], '16384')
        self.assertEqual(cmd[cmd.index('--adapters')+1], str(adapter.resolve()))
        self.assertIn('--merge_lora', cmd)
        self.assertEqual(cmd[cmd.index('--load_args')+1], 'false')
        self.assertNotIn('--attn_impl', cmd)
        self.assertNotIn('--max_batch_size', cmd)
        self.assertEqual(Path(cmd[1]).name, 'vllm_infer.py')
        self.assertEqual(cmd[0], sys.executable)
        self.assertNotIn('NPROC_PER_NODE', info['launch_env'])
        row = json.loads((out/'input.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(row['messages'], [{'role': 'user', 'content': self.prompt}])
        self.assertEqual(info['split'], 'test')

    def test_single_gpu_training_override_and_inference_guards(self):
        _, info = self.launch(['--gpu', '4', '--attn-impl', 'sdpa'])
        cmd = info['argv']
        self.assertEqual(info['launch_env']['NPROC_PER_NODE'], '1')
        self.assertEqual(cmd[cmd.index('--attn_impl')+1], 'sdpa')
        self.assertNotIn('--ddp_find_unused_parameters', cmd)
        for extra in (['--infer-backend', 'vllm'], ['--max-new-tokens', '512'], ['--merge-lora']):
            self.launch(['--gpu', '4'] + extra, success=False)
        for extra in (['--infer-batch-size', '0'], ['--vllm-gpu-memory-utilization', '1.1'],
                      ['--vllm-max-model-len', '8192'], ['--attn-impl', 'flash_attention_2']):
            self.launch(['--gpu', '4', '--preview', '--infer-backend', 'vllm'] + extra, success=False)
        self.launch(['--gpu','4','--preview','--merge-lora'], success=False)

    def test_generation_controls_for_both_backends(self):
        for backend in ('transformers', 'vllm'):
            _, info = self.launch(['--gpu', '6', '--preview', '--infer-backend', backend,
                '--temperature', '0.2', '--repetition-penalty', '1.05'], name=backend)
            cmd = info['argv']
            self.assertEqual(cmd[cmd.index('--temperature')+1], '0.2')
            self.assertEqual(cmd[cmd.index('--repetition_penalty')+1], '1.05')
            self.assertEqual(info['temperature'], 0.2)
            self.assertEqual(info['repetition_penalty'], 1.05)
        _, info = self.launch(['--gpu', '6', '--preview'], name='defaults')
        self.assertEqual(float(info['argv'][info['argv'].index('--temperature')+1]), 0)
        self.assertNotIn('--repetition_penalty', info['argv'])
        self.assertIsNone(info['repetition_penalty'])
        _, info = self.launch(['--gpu', '6'], name='train-defaults')
        self.assertNotIn('--temperature', info['argv'])
        self.assertNotIn('--repetition_penalty', info['argv'])
        for flag, values in (('--temperature', ['-1', 'nan', 'inf']),
                             ('--repetition-penalty', ['0', '-1', 'nan', 'inf'])):
            for value in values:
                self.launch(['--gpu', '6', '--preview', flag, value], success=False)
        for extra in (['--temperature', '0'], ['--repetition-penalty', '1']):
            self.launch(['--gpu', '6'] + extra, success=False)

    def test_guards(self):
        self.launch(['--gpus', '4,4'], success=False)
        self.launch(['--gpus', '4,5', '--preview'], success=False)
        self.launch(['--gpu', '4', '--preview-split', 'test'], success=False)
        row = json.loads((self.data / 'train.jsonl').read_text(encoding='utf-8'))
        (self.data / 'validation.jsonl').write_text(json.dumps(row), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            assert_split_isolation(load_split(self.data, 'train'), load_split(self.data, 'validation'))

    def test_checksum_and_coordinates(self):
        self.assertEqual(validate_target(self.answer), self.target)
        target = dict(self.target, coordinate_mode='pixel')
        with self.assertRaisesRegex(ValueError, 'normalized_1000'):
            validate_target(json.dumps(target))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_target('{"background":{},"background":{}}')
        checksum = hashlib.sha256((self.data / 'train.jsonl').read_bytes()).hexdigest()
        (self.data / 'checksums.json').write_text(json.dumps({'train.jsonl': checksum}))
        self.assertEqual(len(load_split(self.data, 'train')), 1)
        with (self.data / 'train.jsonl').open('a', encoding='utf-8') as f:
            f.write('\n')
        with self.assertRaisesRegex(ValueError, 'Checksum'):
            load_split(self.data, 'train')


if __name__ == '__main__':
    unittest.main()
