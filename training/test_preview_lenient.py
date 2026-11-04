import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from preview_report import build_report, drawable_target, load_results


class LenientPreviewTests(unittest.TestCase):
    def test_bad_structure_still_draws_good_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'image.png'
            Image.new('RGB', (100, 100), 'white').save(image)
            response = json.dumps({'objects': [{'bbox': [100, 100, 800, 800]},
                                              {'bbox': ['bad', 0, 1, 2]}]})
            prediction = json.dumps({'images': [str(image)], 'response': response})+'\n'
            (root / 'predictions.jsonl').write_text(prediction)
            row = load_results(root)[0]
            self.assertIsNone(row['target'])
            self.assertEqual(len(row['drawing']['objects']), 1)
            build_report(root)
            with Image.open(root / 'preview-assets/0-current.png') as rendered:
                self.assertNotEqual(rendered.getpixel((10, 40)), (255, 255, 255))
            self.assertFalse((root / 'preview-assets/0-structure.json').exists())
            self.assertEqual((root / 'predictions.jsonl').read_text(), prediction)

    def test_display_adjustments_and_unknown_coordinates(self):
        drawing, notes = drawable_target(json.dumps({'objects': [
            {'bbox': [900, 800, -10, 200]}, {'bbox': [1, 1, 1, 2]},
            {'bbox': [float('nan'), 1, 2, 3]}]}))
        self.assertEqual(drawing['objects'][0]['bbox'], [0, 200, 900, 800])
        self.assertEqual(len(drawing['objects']), 1)
        self.assertEqual(len(notes), 3)
        for text in ('{"objects":[', '{"coordinate_mode":"pixel","objects":[{"bbox":[0,0,10,10]}]}'):
            drawing, notes = drawable_target(text)
            self.assertEqual(drawing['objects'], [])
            self.assertTrue(notes)
