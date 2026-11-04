import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file
from repair_qwen35_merged import BAD, GOOD, read_header, repair


class RepairTests(unittest.TestCase):
    def test_sharded_copy_preserves_payload_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / 'source', Path(tmp) / 'fixed'
            src.mkdir()
            (src / 'config.json').write_text('{"model_type":"qwen3_5"}')
            mapping, originals, payloads = {}, {}, {}
            tensors = {BAD + 'patch_embed.proj.weight': torch.arange(24).reshape(2, 3, 4),
                       'model.language_model.layers.0.weight': torch.randn(4, 4).bfloat16()}
            for i, (key, tensor) in enumerate(tensors.items()):
                name = f'model-0000{i+1}-of-00002.safetensors'
                save_file({key: tensor}, str(src / name), metadata={'format': 'pt'})
                mapping[key] = name
                _, offset, _ = read_header(src / name)
                payloads[name] = (src / name).read_bytes()[offset:]
            (src / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))
            (src / 'tokenizer_config.json').write_text('{"test":true}')
            originals = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir()}
            self.assertTrue(repair(src))
            self.assertFalse(out.exists())
            self.assertTrue(repair(src, out))
            for name, payload in payloads.items():
                _, offset, _ = read_header(out / name)
                self.assertEqual(payload, (out / name).read_bytes()[offset:])
            for key, tensor in tensors.items():
                fixed_key = key.replace(BAD, GOOD)
                self.assertTrue(torch.equal(load_file(str(out / mapping[key]))[fixed_key], tensor))
            self.assertEqual(originals, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir()})
            self.assertFalse(repair(out))
            with self.assertRaises(ValueError):
                repair(src, out)

    def test_collision_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / 'source', Path(tmp) / 'fixed'
            src.mkdir()
            (src / 'config.json').write_text('{"model_type":"qwen3_5"}')
            save_file({BAD + 'weight': torch.zeros(1), GOOD + 'weight': torch.ones(1)}, str(src / 'model.safetensors'))
            with self.assertRaises(ValueError):
                repair(src, out)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
