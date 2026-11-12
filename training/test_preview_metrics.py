import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from preview_report import build_report
from preview_metrics import save_ground_truth, write_jsonl
from eval.focused import evaluate_focused, overlap_pairs, summarize_focused
from eval.schema import parse_structure


def structure():
    return {'background': {'type': 'transparent'}, 'objects': [
        {'id': 'p', 'kind': 'photo', 'bbox': [100, 100, 500, 500], 'description': 'photo'},
        {'id': 'q', 'kind': 'photo', 'bbox': [400, 300, 700, 700], 'linked_to': 'p', 'description': 'pop-out'},
        {'id': 'f', 'kind': 'frame', 'bbox': [90, 90, 510, 510], 'photo_id': ['p'], 'description': 'frame'},
        {'id': 'o', 'kind': 'overlay', 'bbox': [200, 200, 300, 300], 'description': 'sticker'},
    ], 'layer_order': ['p', 'q', 'f', 'o']}


class SixMetricTests(unittest.TestCase):
    def score(self, prediction, gt=None):
        target = parse_structure(gt or structure())
        return evaluate_focused('one', target, parse_structure(prediction, present=prediction is not None), overlap_pairs(target))

    def test_perfect_and_missing_fixed_cohort(self):
        good = self.score(structure())
        for group, key in [('photo', 'recall'), ('photo', 'matched_mean_iou'), ('fixed', 'recall'),
                           ('layer_order', 'rate'), ('frame_links', 'connected_rate'),
                           ('frame_links', 'correct_rate'), ('linked_to', 'rate')]:
            self.assertEqual(good[group][key], 1, (group, key))
        missing = self.score(None)
        self.assertEqual(missing['photo']['recall'], 0)
        self.assertIsNone(missing['photo']['matched_mean_iou'])
        self.assertEqual(missing['frame_links']['correct_rate'], 0)
        summary = summarize_focused([good, missing])
        self.assertEqual(summary['photo']['recall']['mean'], .5)
        self.assertEqual(summary['photo']['matched_mean_iou']['eligible_images'], 1)
        self.assertEqual(summary['output_usability']['first_pass_core_valid_rate'], .5)
        self.assertIsNone(summary['output_usability']['final_core_valid_rate'])

    def test_fixed_group_and_relations_are_different_questions(self):
        pred = structure()
        pred['objects'][2] = {'id': 'f', 'kind': 'text', 'bbox': [90, 90, 510, 510], 'text': 'frame'}
        result = self.score(pred)
        self.assertEqual(result['fixed']['recall'], 1)
        self.assertEqual(result['frame_links']['connected_rate'], 0)
        pred = structure()
        pred['objects'][2]['photo_id'] = ['q']
        pred['objects'][1].pop('linked_to')
        result = self.score(pred)
        self.assertEqual(result['frame_links']['connected_rate'], 1)
        self.assertEqual(result['frame_links']['correct_rate'], 0)
        self.assertEqual(result['linked_to']['rate'], 0)

    def test_duplicates_false_positives_and_order(self):
        pred = structure()
        pred['layer_order'].reverse()
        result = self.score(pred)
        self.assertEqual(result['layer_order']['rate'], 0)
        self.assertGreater(result['diagnostics']['order_outcomes']['reversed'], 0)
        pred = structure()
        pred['objects'].append(copy.deepcopy(pred['objects'][0]))
        result = self.score(pred)
        self.assertFalse(result['structure_valid'])
        self.assertEqual(result['photo']['recall'], 1)
        self.assertEqual(result['diagnostics']['extra_photo'], 1)
        self.assertEqual(result['frame_links']['correct_rate'], 0)

    def test_nonoverlap_and_empty_groups_are_na(self):
        gt = {'background': {'type': 'transparent'}, 'objects': [
            {'id': 'a', 'kind': 'photo', 'bbox': [0, 0, 500, 500]},
            {'id': 'b', 'kind': 'photo', 'bbox': [500, 0, 1000, 500]},
        ], 'layer_order': ['a', 'b']}
        self.assertEqual(overlap_pairs(parse_structure(gt)), [])
        result = self.score(gt, gt)
        self.assertIsNone(result['layer_order']['rate'])
        self.assertIsNone(result['fixed']['recall'])
        self.assertIsNone(result['linked_to']['rate'])

    def test_macro_not_pooled_and_raw_wrappers_not_repaired(self):
        many = self.score(structure())
        one = {'background': {'type': 'transparent'}, 'objects': [structure()['objects'][0]], 'layer_order': ['p']}
        summary = summarize_focused([many, self.score(None, one)])
        self.assertEqual(summary['photo']['recall']['mean'], .5)
        wrapped = self.score('```json\n' + json.dumps(structure()) + '\n```')
        self.assertFalse(wrapped['structure_valid'])
        self.assertEqual(wrapped['photo']['recall'], 0)


class ReportMetricTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = []
        for index, color in enumerate(('white', 'ivory')):
            path = self.root / f'{index}.png'
            Image.new('RGB', (120, 180), color).save(path)
            self.records.append({'images': [str(path)], 'messages': [
                {'role': 'user', 'content': '<image>test'},
                {'role': 'assistant', 'content': json.dumps(structure())}]})

    def run_dir(self, name, records=None, snapshot=True):
        path = self.root / name
        path.mkdir()
        records = self.records if records is None else records
        write_jsonl(path / 'input.jsonl', [dict(r, messages=r['messages'][:-1]) for r in records])
        (path / 'command.json').write_text(json.dumps({'image_tokens': 1024}), encoding='utf-8')
        if snapshot:
            save_ground_truth(records, path)
        return path

    def predictions(self, path, indices):
        write_jsonl(path / 'predictions.jsonl', [dict(self.records[i], response=json.dumps(structure())) for i in indices])

    def test_gt_left_baseline_scores_missing_and_csv(self):
        run, base = self.run_dir('current'), self.run_dir('base')
        self.predictions(run, [0])
        self.predictions(base, [1, 0])
        original = (run / 'predictions.jsonl').read_bytes()
        self.assertEqual(build_report(run, base, quiet=True), 1)
        page = (run / 'preview.html').read_text(encoding='utf-8')
        self.assertLess(page.index('GT 标注框线图'), page.index('<h3>基线</h3>'))
        self.assertLess(page.index('<h3>基线</h3>'), page.index('<h3>当前模型</h3>'))
        self.assertTrue((run / 'preview-assets/1-gt.png').exists())
        report = json.loads((run / 'metrics/summary.json').read_text(encoding='utf-8'))
        self.assertEqual(report['models']['当前模型 · current']['photo']['recall']['mean'], .5)
        self.assertEqual(report['models']['基线 · base']['photo']['recall']['mean'], 1)
        self.assertIn('已完成 1 / 2', page)
        with (run / 'metrics/summary.csv').open(encoding='utf-8-sig', newline='') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 20)
        self.assertEqual((run / 'predictions.jsonl').read_bytes(), original)
        with Image.open(run / 'preview-assets/0-gt.png') as image:
            self.assertNotEqual(image.getpixel((12, 30)), (255, 255, 255))

    def test_legacy_gt_recovery_and_explicit_override(self):
        dataset = self.root / 'dataset'
        dataset.mkdir()
        write_jsonl(dataset / 'test.jsonl', self.records)
        run = self.run_dir('legacy', snapshot=False)
        (run / 'command.json').write_text(json.dumps({'source': {'local_dir': str(dataset)}, 'split': 'test'}))
        self.predictions(run, [0, 1])
        build_report(run, quiet=True)
        self.assertTrue((run / 'ground_truth.jsonl').exists())
        self.assertTrue((run / 'metrics/summary.csv').exists())
        explicit = self.run_dir('explicit', snapshot=False)
        self.predictions(explicit, [0])
        build_report(explicit, gt=dataset / 'test.jsonl', quiet=True)
        bad = self.root / 'bad.jsonl'
        write_jsonl(bad, [self.records[0]])
        with self.assertRaisesRegex(ValueError, '未匹配到 GT'):
            build_report(explicit, gt=bad, quiet=True)

    def test_incremental_cache_and_no_predictions(self):
        run = self.run_dir('live')
        cache = {}
        self.assertEqual(build_report(run, partial=True, cache=cache, quiet=True), 0)
        self.predictions(run, [0])
        self.assertEqual(build_report(run, partial=True, cache=cache, quiet=True), 1)
        self.predictions(run, [0, 1])
        self.assertEqual(build_report(run, partial=True, cache=cache, quiet=True, running=False), 2)
        result = json.loads((run / 'metrics/summary.json').read_text(encoding='utf-8'))
        self.assertFalse(result['protocol']['running'])
        self.assertEqual(result['models']['当前模型 · live']['output_usability']['first_pass_core_valid_rate'], 1)

    def test_snapshot_pixel_mismatch_fails(self):
        run = self.run_dir('changed')
        self.predictions(run, [0, 1])
        Image.new('RGB', (120, 180), 'black').save(self.records[0]['images'][0])
        with self.assertRaisesRegex(ValueError, 'pixels changed'):
            build_report(run, quiet=True)

    def test_wrapped_predictions_score_in_html_csv_and_json_without_rewriting(self):
        run, base = self.run_dir('wrapped'), self.run_dir('plain')
        self.predictions(base, [0, 1])
        wrapped = '<think>\n\n</think>\n\n```json\n' + json.dumps(structure()) + '\n```'
        write_jsonl(run / 'predictions.jsonl', [dict(r, response=wrapped) for r in self.records])
        original = (run / 'predictions.jsonl').read_bytes()
        build_report(run, base, quiet=True)
        report = json.loads((run / 'metrics/summary.json').read_text(encoding='utf-8'))
        self.assertEqual(report['protocol']['response_parsing'], 'empty_think_and_outer_json_fence_v1')
        for metrics in report['models'].values():
            self.assertEqual(metrics['photo']['recall']['mean'], 1)
            self.assertEqual(metrics['photo']['matched_mean_iou']['mean'], 1)
            self.assertEqual(metrics['output_usability']['first_pass_core_valid_rate'], 1)
        current = report['models']['当前模型 · wrapped']
        self.assertEqual(current['diagnostics']['wrapper_images'], 2)
        self.assertEqual(current['diagnostics']['raw_structure_valid_images'], 0)
        page = (run / 'preview.html').read_text(encoding='utf-8')
        self.assertIn('召回 100.00%', page)
        self.assertIn('已找到框 IoU 1.000', page)
        with (run / 'metrics/summary.csv').open(encoding='utf-8-sig', newline='') as f:
            scores = [r for r in csv.DictReader(f) if r['metric'] == 'photo 召回率']
        self.assertTrue(all(float(r['value_0_to_1']) == 1 for r in scores))
        self.assertEqual((run / 'predictions.jsonl').read_bytes(), original)

    def test_wrapper_parsing_does_not_repair_bad_json_or_nonempty_thinking(self):
        for index, response in enumerate((
                '```json\n{"objects": [\n```',
                '<think>reasoning</think>\n' + json.dumps(structure()),
                'Here is the answer: ' + json.dumps(structure()),
                '```json\n' + json.dumps(structure()))):
            with self.subTest(response=response):
                run = self.run_dir(f'invalid-{index}')
                write_jsonl(run / 'predictions.jsonl', [dict(self.records[0], response=response)])
                build_report(run, quiet=True)
                report = json.loads((run / 'metrics/summary.json').read_text(encoding='utf-8'))
                metrics = next(iter(report['models'].values()))
                self.assertEqual(metrics['photo']['recall']['mean'], 0)
                self.assertEqual(metrics['photo']['recall']['total_images'], 2)
                self.assertIsNone(metrics['photo']['matched_mean_iou']['mean'])
                page = (run / 'preview.html').read_text(encoding='utf-8')
                self.assertIn('无匹配框', page)
                self.assertIn('invalid_json', page)

    def test_run_lora_gt_never_enters_inference_input(self):
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        dataset = self.root / 'data'
        dataset.mkdir()
        write_jsonl(dataset / 'test.jsonl', self.records)
        run = self.root / 'dry-run'
        subprocess.run([sys.executable, str(Path(__file__).with_name('run_lora.py')),
                        '--model', str(model), '--data', str(dataset), '--gpu', '0',
                        '--preview', '--preview-split', 'test', '--limit', '1',
                        '--output', str(run), '--dry-run'], check=True, stdout=subprocess.DEVNULL)
        inputs = json.loads((run / 'input.jsonl').read_text(encoding='utf-8'))
        self.assertFalse(any(m['role'] == 'assistant' for m in inputs['messages']))
        gt = json.loads((run / 'ground_truth.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(gt['structure'], structure())
        self.assertGreater(len(gt['occlusion_pairs']), 0)

    def test_full137_cli_keeps_invalid_and_missing_results(self):
        records = []
        for i in range(137):
            path = self.root / f'case-{i:03}.png'
            Image.new('RGB', (20, 30), (i, 100, 200)).save(path)
            records.append(dict(self.records[0], images=[str(path)]))
        run = self.run_dir('test137', records)
        predictions = [dict(r, response=json.dumps(structure())) for r in records[:135]]
        predictions.append(dict(records[135], response='invalid JSON'))
        write_jsonl(run / 'predictions.jsonl', predictions)
        subprocess.run([sys.executable, str(Path(__file__).with_name('preview_report.py')),
                        '--run', str(run)], check=True, stdout=subprocess.DEVNULL)
        data = json.loads((run / 'metrics/summary.json').read_text(encoding='utf-8'))
        metrics = data['models']['当前模型 · test137']
        self.assertEqual(metrics['output_usability']['total_images'], 137)
        self.assertEqual(metrics['output_usability']['predictions_present'], 136)
        self.assertAlmostEqual(metrics['photo']['recall']['mean'], 135 / 137)
        page = (run / 'preview.html').read_text(encoding='utf-8')
        self.assertEqual(page.count('<article '), 137)
        self.assertIn('已完成 136 / 137', page)


if __name__ == '__main__':
    unittest.main()
