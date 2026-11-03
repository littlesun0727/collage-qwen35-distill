"""Incremental JSONL writes must be visible before the inference process exits."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from live_preview import LivePreview


class LivePreviewTests(unittest.TestCase):
    def test_complete_lines_partial_utf8_and_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'image.png'
            Image.new('RGB', (20, 30)).save(image)
            watcher = LivePreview(root, 2)
            watcher.update()
            self.assertIn('已完成 0 / 2', (root / 'preview.html').read_text(encoding='utf-8'))
            row = {'images': [str(image)], 'response': json.dumps({
                'background': {'type': 'transparent'}, 'objects': [], 'layer_order': []}),
                'note': '中文'}
            encoded = (json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8')
            path = root / 'predictions.jsonl'
            path.write_bytes(encoded)
            watcher.update()
            self.assertEqual(watcher.count, 1)
            self.assertTrue((root / 'preview-assets/0-structure.json').exists())
            # The second row has not finished writing, including a split UTF-8 character.
            cut = encoded.index('中文'.encode('utf-8')) + 1
            with path.open('ab') as f:
                f.write(encoded[:cut])
            watcher.update()
            self.assertEqual(watcher.count, 1)
            with path.open('ab') as f:
                f.write(encoded[cut:])
            watcher.update()
            self.assertEqual(watcher.count, 2)
            watcher.update(final=True, stopped=True)
            page = (root / 'preview.html').read_text(encoding='utf-8')
            self.assertIn('已完成 2 / 2', page)
            self.assertNotIn('http-equiv="refresh"', page)
            # Interrupted run retains only complete rows and is not labelled as running.
            path.write_bytes(encoded + encoded[:cut])
            watcher.update(stopped=True)
            page = (root / 'preview.html').read_text(encoding='utf-8')
            self.assertIn('已完成 1 / 2', page)
            self.assertNotIn('http-equiv="refresh"', page)


if __name__ == '__main__':
    unittest.main()
