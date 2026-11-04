import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from run_eval_queue import make_plan, execute_plan


class QueueTests(unittest.TestCase):
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
            self.assertIn('--merge-lora', plan[1]['argv'])
            self.assertNotIn('--preview', plan[1]['argv'])
            self.assertIn(str(out / 'base'), plan[1]['argv'])
            cmd = plan[1]['argv']
            self.assertEqual(cmd[cmd.index('--infer-batch-size')+1], '8')
            calls = []
            def run(argv, check):
                calls.append(argv)
                return SimpleNamespace(returncode=0)
            with patch('run_eval_queue.subprocess.run', side_effect=run):
                self.assertEqual(execute_plan(plan, out), 0)
            self.assertEqual(calls, [j['argv'] for j in plan])
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
