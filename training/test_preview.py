"""CPU-only regression tests: python -m unittest discover -s training -p test_preview.py"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
from preview_report import build_report, unwrap_response
from dataset_io import validate_target


class PreviewTests(unittest.TestCase):
    def test_response_wrappers(self):
        raw = json.dumps({'background': {'type': 'transparent'}, 'objects': [], 'layer_order': []})
        for response in (raw, '<think>\n\n</think>\n' + raw,
                         '```json\n' + raw + '\n```',
                         '<think>\r\n</think>\r\n```json\r\n' + raw + '\r\n```'):
            text, wrappers = unwrap_response(response)
            self.assertEqual(validate_target(text), validate_target(raw))
        for response in ('<think>actual reasoning</think>' + raw,
                         '```json\n' + raw, '```json\n{broken}\n```',
                         '```json\n' + raw + '\n```\nextra text'):
            with self.assertRaises(ValueError):
                validate_target(unwrap_response(response)[0])

    def test_preview_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            t = Path(temp)
            model = t / 'model'
            model.mkdir()
            (model / 'config.json').write_text('{}')
            images = t / 'images'
            images.mkdir()
            Image.new('RGB', (100, 200), 'white').save(images / 'one.png')
            adapter = t / 'checkpoint-10'
            adapter.mkdir()
            (adapter / 'adapter_config.json').write_text('{}')
            (adapter / 'trainer_state.json').write_text('{}')
            common = [sys.executable, str(Path(__file__).with_name('run_lora.py')),
                      '--model', str(model), '--gpu', '5', '--dry-run']
            for name, extra in [('base', ['--preview']), ('tuned', ['--adapter', str(adapter)])]:
                subprocess.run(common + extra + ['--images', str(images), '--output', str(t / name)],
                               check=True, stdout=subprocess.DEVNULL)
                row = json.loads((t / name / 'input.jsonl').read_text(encoding='utf-8'))
                self.assertEqual(len(row['messages']), 2)
                row['response'] = json.dumps({'background': {'type': 'transparent'},
                    'objects': [{'id': 'p', 'kind': 'photo', 'description': 'test',
                                 'bbox': [100, 100, 900, 900]}], 'layer_order': ['p']})
                (t / name / 'predictions.jsonl').write_text(json.dumps(row), encoding='utf-8')
            build_report(t / 'base')
            build_report(t / 'tuned', t / 'base')
            page = (t / 'tuned/preview.html').read_text(encoding='utf-8')
            self.assertIn('preview-assets/0-baseline.png', page)
            structure = t / 'tuned/preview-assets/0-structure.json'
            self.assertTrue(structure.is_file())
            # Real SWIFT 4.0.4 output can use datasets.Image's path/bytes record.
            row['images'] = [{'path': row['images'][0], 'bytes': None}]
            (t / 'tuned/predictions.jsonl').write_text(json.dumps(row), encoding='utf-8')
            build_report(t / 'tuned')
            self.assertTrue(structure.is_file())
            row['response'] = '<think>\n\n</think>\n\n```json\n' + row['response'] + '\n```'
            (t / 'tuned/predictions.jsonl').write_text(json.dumps(row), encoding='utf-8')
            build_report(t / 'tuned')
            self.assertTrue(structure.is_file())
            page = (t / 'tuned/preview.html').read_text(encoding='utf-8')
            self.assertIn('解析前已移除空 think 标签、JSON 代码围栏', page)
            self.assertIn('&lt;think&gt;', page)
            row['response'] = '<script>alert(1)</script>'
            (t / 'tuned/predictions.jsonl').write_text(json.dumps(row), encoding='utf-8')
            build_report(t / 'tuned')
            page = (t / 'tuned/preview.html').read_text(encoding='utf-8')
            self.assertIn('&lt;script&gt;', page)
            self.assertNotIn('<script>', page)
            self.assertFalse(structure.exists())
            subprocess.run(common + ['--resume', str(adapter), '--max-steps', '100',
                           '--save-steps', '10', '--output', str(t / 'resumed')],
                           check=True, stdout=subprocess.DEVNULL)
            cmd = json.loads((t / 'resumed/command.json').read_text(encoding='utf-8'))['argv']
            self.assertIn('--resume_from_checkpoint', cmd)
            self.assertNotIn('--save_total_limit', cmd)
            self.assertEqual(cmd[cmd.index('--save_steps') + 1], '10')
            # Reject a comparison made under a different visual token budget.
            prior = t / 'base/command.json'
            value = json.loads(prior.read_text(encoding='utf-8'))
            value['image_tokens'] = 2048
            prior.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'image-token budget'):
                build_report(t / 'tuned', t / 'base')


if __name__ == '__main__':
    unittest.main()
