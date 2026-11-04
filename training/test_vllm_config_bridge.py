"""CPU regression tests for the inference-only config bridge."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest

from vllm_infer import install_config_bridge


class ConfigBridgeTests(unittest.TestCase):
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
