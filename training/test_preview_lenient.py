import json
from pathlib import Path
import tempfile
import unittest
import io
from contextlib import redirect_stdout

from PIL import Image
from preview_report import build_report, draw_boxes, drawable_target, load_results


class LenientPreviewTests(unittest.TestCase):
    def test_duplicate_ids_keep_both_boxes_and_report_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'image.png'
            Image.new('RGB', (2000, 1000), 'white').save(image)
            response = json.dumps({'background': {'type': 'transparent'}, 'objects': [
                {'id': 'photo_1', 'kind': 'photo', 'bbox': [100, 100, 400, 400]},
                {'id': 'photo_1', 'kind': 'photo', 'bbox': [600, 600, 900, 900]}],
                'layer_order': ['photo_1', 'photo_1']})
            (root / 'predictions.jsonl').write_text(json.dumps({
                'images': [str(image)], 'response': response})+'\n')
            row = load_results(root)[0]
            self.assertIsNone(row['target'])
            self.assertEqual(len(row['drawing']['objects']), 2)
            self.assertEqual(row['image'].size, (1600, 800))
            output = io.StringIO()
            with redirect_stdout(output):
                build_report(root)
            self.assertIn('读取图片 1/1', output.getvalue())
            self.assertIn('绘制 1/1', output.getvalue())
            page = (root / 'preview.html').read_text(encoding='utf-8')
            self.assertIn('已绘制 2 个框', page)
            self.assertIn('对象 ID 重复', page)
            with Image.open(root / 'preview-assets/0-current.png') as rendered:
                self.assertNotEqual(rendered.getpixel((160, 200)), (255, 255, 255))
                self.assertNotEqual(rendered.getpixel((960, 600)), (255, 255, 255))

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

    def test_window_geometry_survives_prediction_preview(self):
        target = {'objects': [
            {'kind': 'photo', 'id': 'tilted', 'bbox': [100, 100, 500, 800],
             'corners': {'top_left': [200, 100], 'top_right': [500, 200],
                         'bottom_right': [400, 800], 'bottom_left': [100, 700]},
             'popout': [50, 300, 250, 900]},
            {'kind': 'photo', 'id': 'triangle', 'shape': 'other', 'bbox': [600, 100, 900, 700],
             'points': [[600, 700], [750, 100], [900, 700]]},
        ], 'layer_order': ['triangle', 'tilted']}
        drawing, notes = drawable_target(json.dumps(target))
        self.assertFalse(notes)
        original = Image.new('RGB', (400, 200), 'white')
        rendered = draw_boxes(original, drawing)
        self.assertEqual(rendered.tobytes(), draw_boxes(original, target).tobytes())
        self.assertEqual(rendered.getpixel((80, 20)), (255, 209, 102))  # TL
        self.assertEqual(rendered.getpixel((300, 20)), (255, 209, 102))  # triangle tip
        self.assertEqual(rendered.getpixel((270, 80)), (0, 217, 220))   # diagonal
        self.assertEqual(rendered.getpixel((20, 60)), (255, 66, 136))   # popout
