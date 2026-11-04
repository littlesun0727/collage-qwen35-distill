"""CPU regression tests for the inference-only config bridge."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest

from vllm_infer import install_config_bridge, install_sampling_overrides, parse_extra_args


class ConfigBridgeTests(unittest.TestCase):
    def test_sampling_override_reaches_request_without_changing_defaults(self):
        class Engine:
            def _prepare_generation_config(self, request):
                return SimpleNamespace(temperature=request.temperature, top_p=0.8, top_k=20,
                                       presence_penalty=0, min_p=0.1, repetition_penalty=1.0)
        extra, remaining = parse_extra_args(['--model', '/model', '--presence-penalty', '1.5',
                                            '--min-p', '0', '--temperature', '0.7'])
        self.assertEqual(remaining, ['--model', '/model', '--temperature', '0.7'])
        install_sampling_overrides(Engine, extra.presence_penalty, extra.min_p)
        engine = Engine()
        request = SimpleNamespace(temperature=0.7)
        for _ in range(2):
            params = engine._prepare_generation_config(request)
            self.assertEqual(params.presence_penalty, 1.5)
            self.assertEqual(params.min_p, 0)
            self.assertEqual(params.temperature, 0.7)
            self.assertEqual(params.top_k, 20)
        self.assertEqual(vars(request), {'temperature': 0.7})

    def test_conversion_is_scoped_and_preserves_original(self):
        payload = {'text_config': {'hidden_size': 2560},
                   'vision_config': {'spatial_merge_size': 2}}
        source = SimpleNamespace(model_type='qwen3_5', text_config=object(),
                                 to_dict=lambda: payload)

        class Target:
            @classmethod
            def from_dict(cls, data):
                obj = cls()
                obj.data = data
                return obj

        active = []

        @contextmanager
        def patch(config):
            active.append(config)
            try:
                yield
            finally:
                active.pop()

        module = SimpleNamespace(patch_auto_config=patch)
        install_config_bridge(module, Target)
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with module.patch_auto_config(source):
                self.assertIsInstance(active[0], Target)
                self.assertEqual(active[0].data, payload)
                active[0].data['text_config']['hidden_size'] = 1
                raise RuntimeError('test')
        self.assertEqual(active, [])
        self.assertEqual(payload['text_config']['hidden_size'], 2560)
        unrelated = SimpleNamespace(model_type='other')
        with module.patch_auto_config(unrelated):
            self.assertIs(active[0], unrelated)


if __name__ == '__main__':
    unittest.main()
