import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from run_eval_queue import make_plan, execute_plan, inspect_result, queue_lock, option


class QueueTests(unittest.TestCase):
    def test_full_test_split_and_explicit_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.json').write_text('{}')
            dataset = root / 'test.jsonl'
            dataset.write_text('{}\n' * 202)
            config = {'model': str(root), 'data': str(root), 'jobs': [{'name': 'base'}]}
            def plan(**kwargs):
                return make_plan(config, '0', root / 'out', **kwargs)[0]
            self.assertEqual(option(plan()['argv'], '--limit'), '202')
            self.assertEqual(plan()['sample_count'], 202)
            config['limit'] = 137
            self.assertEqual(option(plan()['argv'], '--limit'), '137')
            self.assertEqual(option(plan(limit=20)['argv'], '--limit'), '20')
            self.assertEqual(option(plan(limit=999)['argv'], '--limit'), '202')
            for invalid in (0, -1, True, 1.5, '202'):
                config['limit'] = invalid
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    plan()
            config.pop('limit')
            dataset.write_text('{}\n' * 250)
            self.assertEqual(option(plan()['argv'], '--limit'), '250')
            dataset.write_text('')
            with self.assertRaisesRegex(ValueError, 'Empty test'):
                plan()

    def test_base_and_adapter_plan_and_failure_handling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for filename in ('config.json', 'test.jsonl', 'adapter_config.json'):
                (root / filename).write_text('{}')
            config = {'model': str(root), 'data': str(root), 'jobs': [
                {'name': 'base'}, {'name': 'lora', 'adapter': str(root), 'compare_to': 'base'},
                {'name': 'other_base', 'model': str(root)}]}
            out = root / 'output'
            plan = make_plan(config, '4', out)
            self.assertIn('--preview', plan[0]['argv'])
            self.assertNotIn('--adapter', plan[0]['argv'])
            self.assertIn('--adapter', plan[1]['argv'])
            self.assertNotIn('--merge-lora', plan[1]['argv'])
            config['jobs'][1]['merge_lora'] = True
            self.assertIn('--merge-lora', make_plan(config, '4', out)[1]['argv'])
            config['jobs'][1]['merge_lora'] = 'false'
            with self.assertRaisesRegex(ValueError, 'JSON boolean'):
                make_plan(config, '4', out)
            config['jobs'][1].pop('merge_lora')
            config['jobs'][0]['merge_lora'] = True
            with self.assertRaisesRegex(ValueError, 'requires an adapter'):
                make_plan(config, '4', out)
            config['jobs'][0].pop('merge_lora')
            self.assertNotIn('--preview', plan[1]['argv'])
            self.assertIn(str(out / 'base'), plan[1]['argv'])
            cmd = plan[1]['argv']
            self.assertEqual(cmd[cmd.index('--infer-batch-size')+1], '8')
            calls = []
            def run(argv, check):
                calls.append(argv)
                return SimpleNamespace(returncode=0)
            with patch('run_eval_queue.subprocess.run', side_effect=run), patch(
                    'run_eval_queue.inspect_result', side_effect=[
                        ('run', ''), ('reuse', ''), ('run', ''), ('reuse', ''),
                        ('run', ''), ('reuse', '')]):
                self.assertEqual(execute_plan(plan, out), 0)
            self.assertEqual(calls[::2], [j['argv'] for j in plan])
            self.assertTrue(all('--gt' in argv for argv in calls[1::2]))
            self.assertEqual(option(calls[3], '--baseline'), plan[0]['output'])
            self.assertEqual(len(json.loads((out/'queue-status.json').read_text())), 3)
            with patch('run_eval_queue.subprocess.run', return_value=SimpleNamespace(returncode=1)) as run:
                self.assertEqual(execute_plan(plan, root/'failed'), 1)
                self.assertEqual(run.call_count, 1)
            with patch('run_eval_queue.subprocess.run', return_value=SimpleNamespace(returncode=1)) as run:
                self.assertEqual(execute_plan(plan, root/'continued', True), 1)
                self.assertEqual(run.call_count, 2)  # dependent LoRA is skipped
            config['jobs'][1]['compare_to'] = 'missing'
            with self.assertRaisesRegex(ValueError, 'earlier job'):
                make_plan(config, '4', root/'bad')


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'results'
        (self.root / 'config.json').write_text('{}')
        self.row = {'messages': [{'role': 'user', 'content': '<image>test'},
                                 {'role': 'assistant', 'content': '{}'}],
                    'images': ['one.png']}
        (self.root / 'test.jsonl').write_text(json.dumps(self.row))
        self.config = {'model': str(self.root), 'data': str(self.root),
                       'jobs': [{'name': 'base'}, {'name': 'next', 'compare_to': 'base'}]}
        self.plan = make_plan(self.config, '4', self.out)

    def complete(self, job):
        out = Path(job['output'])
        out.mkdir(parents=True)
        row = {'messages': self.row['messages'][:-1],
               'images': [str((self.root / 'one.png').resolve())]}
        (out / 'input.jsonl').write_text(json.dumps(row))
        # A schema-invalid model answer must still be reused, not sampled again.
        (out / 'predictions.jsonl').write_text(json.dumps(dict(row, response='bad JSON')))
        saved = {'image_tokens': 1024, 'max_new_tokens': 8192, 'temperature': .7,
                 'top_p': .8, 'top_k': 20, 'min_p': 0, 'presence_penalty': 1.5,
                 'repetition_penalty': 1., 'infer_backend': 'vllm', 'split': 'test',
                 'argv': ['--model', str(self.root.resolve()), '--max_length', '12288',
                          '--enable_thinking', 'false']}
        (out / 'command.json').write_text(json.dumps(saved))

    def run_fake(self, argv, check):
        if '--output' in argv:
            self.complete(next(j for j in self.plan if j['argv'] == argv))
        return SimpleNamespace(returncode=0)

    def test_sampling_inheritance_override_and_reuse(self):
        self.complete(self.plan[0])
        self.config['sampling'] = {'temperature': .2, 'presence_penalty': 0,
                                   'top_p': .95, 'top_k': 40, 'min_p': .05,
                                   'repetition_penalty': 1.05}
        self.config['jobs'][0]['sampling'] = {
            'temperature': .7, 'presence_penalty': 1.5, 'top_p': .8,
            'top_k': 20, 'min_p': 0, 'repetition_penalty': 1.0}
        plan = make_plan(self.config, '4', self.out)
        self.assertEqual(inspect_result(plan[0])[0], 'reuse')
        for key, value in self.config['sampling'].items():
            flag = '--' + key.replace('_', '-')
            self.assertEqual(float(option(plan[1]['argv'], flag)), value)
            self.assertEqual(plan[1]['argv'].count(flag), 1)
        self.config['jobs'][0]['sampling'] = {'temperature': 0}
        plan = make_plan(self.config, '4', self.out)
        self.assertEqual(option(plan[0]['argv'], '--temperature'), '0')
        self.assertEqual(inspect_result(plan[0])[0], 'rerun')
        self.assertEqual(self.config['sampling']['temperature'], .2)

    def test_invalid_sampling_rejected_before_launch(self):
        for invalid in (None, [], {'top-k': 20}, {'temperature': '0.2'},
                        {'temperature': True}, {'temperature': float('nan')},
                        {'temperature': -1}, {'top_p': 0}, {'top_p': 1.1},
                        {'top_k': 0}, {'top_k': 2.5}, {'min_p': -1},
                        {'presence_penalty': 3}, {'repetition_penalty': 0}):
            for target in (self.config, self.config['jobs'][0]):
                with self.subTest(invalid=invalid, job=target is not self.config):
                    target['sampling'] = invalid
                    with self.assertRaises(ValueError):
                        make_plan(self.config, '4', self.out)
                    del target['sampling']

    def test_legacy_base_reused_new_job_runs_then_all_reused(self):
        self.complete(self.plan[0])
        with patch('run_eval_queue.subprocess.run', side_effect=self.run_fake) as run:
            self.assertEqual(execute_plan(self.plan, self.out), 0)
            self.assertNotIn('--output', run.call_args_list[0].args[0])
            self.assertIn('--output', run.call_args_list[1].args[0])
        with patch('run_eval_queue.subprocess.run', side_effect=self.run_fake) as run:
            self.assertEqual(execute_plan(self.plan, self.out), 0)
            self.assertTrue(all('--output' not in c.args[0] for c in run.call_args_list))
            self.assertIn('--baseline', run.call_args_list[1].args[0])

    def test_partial_result_archived_and_retried(self):
        self.complete(self.plan[0])
        predictions = self.out / 'base/predictions.jsonl'
        predictions.write_text('{broken')
        with patch('run_eval_queue.subprocess.run', side_effect=self.run_fake):
            self.assertEqual(execute_plan(self.plan, self.out), 0)
        archives = list((self.out / '_history/base').glob('*/predictions.jsonl'))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_text(), '{broken')
        self.assertEqual(inspect_result(self.plan[0])[0], 'reuse')

    def test_configuration_and_input_mismatch(self):
        self.complete(self.plan[0])
        path = self.out / 'base/command.json'
        saved = json.loads(path.read_text())
        saved['temperature'] = .2
        path.write_text(json.dumps(saved))
        self.assertEqual(inspect_result(self.plan[0])[0], 'rerun')
        saved['temperature'] = .7
        path.write_text(json.dumps(saved))
        self.row['messages'][0]['content'] = 'changed prompt'
        (self.root / 'test.jsonl').write_text(json.dumps(self.row))
        self.assertEqual(inspect_result(self.plan[0])[0], 'rerun')

    def test_expanded_test_split_does_not_reuse_partial_cohort(self):
        self.complete(self.plan[0])
        extra = dict(self.row, images=['two.png'])
        (self.root / 'test.jsonl').write_text(json.dumps(self.row) + '\n' + json.dumps(extra))
        full = make_plan(self.config, '4', self.out)
        self.assertEqual(option(full[0]['argv'], '--limit'), '2')
        self.assertEqual(inspect_result(full[0])[0], 'rerun')
        limited = make_plan(self.config, '4', self.out, limit=1)
        self.assertEqual(inspect_result(limited[0])[0], 'reuse')

    def test_missing_predictions_not_success_and_lock_released(self):
        with patch('run_eval_queue.subprocess.run', return_value=SimpleNamespace(returncode=0)):
            self.assertEqual(execute_plan(self.plan, self.out), 1)
        with queue_lock(self.out):
            with self.assertRaises(ValueError):
                with queue_lock(self.out):
                    self.fail('second queue acquired the same output directory')
        with queue_lock(self.out):
            pass

    def test_wrong_image_and_report_failure_do_not_trigger_silent_reuse(self):
        self.complete(self.plan[0])
        path = self.out / 'base/predictions.jsonl'
        prediction = json.loads(path.read_text())
        original = path.read_text()
        prediction['images'] = ['wrong.png']
        path.write_text(json.dumps(prediction))
        self.assertEqual(inspect_result(self.plan[0])[0], 'rerun')
        path.write_text(original)
        with patch('run_eval_queue.subprocess.run', return_value=SimpleNamespace(returncode=1)) as run:
            self.assertEqual(execute_plan(self.plan, self.out), 1)
            self.assertNotIn('--output', run.call_args.args[0])
        self.assertEqual(path.read_text(), original)
        self.assertEqual(inspect_result(self.plan[0])[0], 'reuse')

    def test_first_run_generates_real_metrics_and_base_comparison(self):
        from PIL import Image
        Image.new('RGB', (32, 32), 'white').save(self.root / 'one.png')
        gt = {'background': {'type': 'transparent'}, 'objects': [
            {'id': 'p', 'kind': 'photo', 'bbox': [0, 0, 1000, 1000]}],
            'layer_order': ['p']}
        self.row['messages'][-1]['content'] = json.dumps(gt)
        (self.root / 'test.jsonl').write_text(json.dumps(self.row))
        real_run = subprocess.run

        def run(argv, check):
            if '--output' in argv:
                self.run_fake(argv, check)
                # Simulate successful inference with NO live preview or GT snapshot.
                # Base is correct; current output stays invalid to test actual scoring.
                if argv == self.plan[0]['argv']:
                    path = self.out / 'base/predictions.jsonl'
                    row = json.loads(path.read_text())
                    row['response'] = json.dumps(gt)
                    path.write_text(json.dumps(row))
                return SimpleNamespace(returncode=0)
            return real_run(argv, check=check)

        with patch('run_eval_queue.subprocess.run', side_effect=run):
            self.assertEqual(execute_plan(self.plan, self.out), 0)
        for name, count in [('base', 1), ('next', 2)]:
            report = self.out / name
            html = (report / 'preview.html').read_text(encoding='utf-8')
            self.assertIn('metrics/summary.csv', html)
            data = json.loads((report / 'metrics/summary.json').read_text(encoding='utf-8'))
            self.assertFalse(data['protocol']['running'])
            self.assertEqual(len(data['models']), count)
        scores = {label.split(' · ')[-1]: value['photo']['recall']['mean']
                  for label, value in data['models'].items()}
        self.assertEqual(scores, {'base': 1, 'next': 0})

    def test_fresh_report_failure_retains_predictions_and_retry_skips_inference(self):
        def fail_report(argv, check):
            if '--output' in argv:
                return self.run_fake(argv, check)
            return SimpleNamespace(returncode=1)

        with patch('run_eval_queue.subprocess.run', side_effect=fail_report):
            self.assertEqual(execute_plan(self.plan, self.out, True), 1)
        status = json.loads((self.out / 'queue-status.json').read_text())
        self.assertEqual(status[0]['status'], 'failed')
        self.assertEqual(status[0]['stage'], 'report')
        self.assertIn('predictions retained', status[0]['error'])
        self.assertEqual(status[1]['status'], 'skipped: baseline failed')
        self.assertEqual(inspect_result(self.plan[0])[0], 'reuse')
        with patch('run_eval_queue.subprocess.run', side_effect=self.run_fake) as run:
            self.assertEqual(execute_plan(self.plan, self.out), 0)
        self.assertNotIn('--output', run.call_args_list[0].args[0])
