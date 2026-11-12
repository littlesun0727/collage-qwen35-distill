import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from rebuild_reports import discover, rebuild
from preview_metrics import write_jsonl


class RebuildTests(unittest.TestCase):
    def test_batch_rebuild_comparisons_failure_isolation_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'one.png'
            Image.new('RGB', (20, 20), 'white').save(image)
            target = {'background': {'type': 'transparent'}, 'objects': [
                {'id': 'p', 'kind': 'photo', 'bbox': [0, 0, 1000, 1000]}], 'layer_order': ['p']}
            row = {'images': [str(image)], 'messages': [
                {'role': 'user', 'content': '<image>test'},
                {'role': 'assistant', 'content': json.dumps(target)}]}
            gt = root / 'test.jsonl'
            write_jsonl(gt, [row])
            for name in ('base', 'lora', 'broken', '_history/old'):
                run = root / name
                run.mkdir(parents=True)
                write_jsonl(run / 'input.jsonl', [dict(row, messages=row['messages'][:-1])])
                (run / 'command.json').write_text(json.dumps({'image_tokens': 1024}))
                write_jsonl(run / 'predictions.jsonl', [dict(row, response='```json\n' + json.dumps(target) + '\n```')])
            (root / 'broken/predictions.jsonl').write_text('{broken')
            (root / 'plan.json').write_text(json.dumps([
                {'name': 'base'}, {'name': 'lora', 'compare_to': 'base'}]))
            self.assertEqual(len(discover(root)), 3)
            self.assertEqual(len(discover(root, True)), 4)
            original = (root / 'lora/predictions.jsonl').read_bytes()
            self.assertEqual(rebuild(root, gt=gt, dry_run=True), 0)
            self.assertFalse((root / 'base/preview.html').exists())
            self.assertEqual(rebuild(root, gt=gt), 1)
            statuses = json.loads((root / 'report-rebuild-status.json').read_text())
            self.assertEqual([r['status'] for r in statuses], ['success', 'failed', 'success'])
            report = json.loads((root / 'lora/metrics/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(len(report['models']), 2)
            self.assertTrue(all(m['photo']['recall']['mean'] == 1 for m in report['models'].values()))
            self.assertEqual((root / 'lora/predictions.jsonl').read_bytes(), original)
            self.assertFalse((root / '_history/old/preview.html').exists())


if __name__ == '__main__':
    unittest.main()
