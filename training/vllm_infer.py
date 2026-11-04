"""SWIFT inference entry point with Qwen3.5 config compatibility for vLLM."""
from contextlib import contextmanager
from copy import deepcopy
import argparse
import sys
import math


def convert_config(config, config_cls):
    """Rebuild nested configs; changing __class__ would leave incompatible children."""
    if isinstance(config, config_cls):
        return config
    return config_cls.from_dict(deepcopy(config.to_dict()))


def install_config_bridge(engine_module, config_cls):
    original = engine_module.patch_auto_config

    @contextmanager
    def compatible_auto_config(config):
        if getattr(config, 'model_type', None) == 'qwen3_5' and hasattr(config, 'text_config'):
            converted = convert_config(config, config_cls)
            if converted is not config:
                print('[compat] Qwen3.5 config converted for vLLM (text + vision); '
                      'original SWIFT config retained.', flush=True)
            config = converted
        with original(config):
            yield

    engine_module.patch_auto_config = compatible_auto_config


def parse_extra_args(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--presence-penalty', type=float)
    parser.add_argument('--min-p', type=float)
    extra, remaining = parser.parse_known_args(argv)
    for name, value, low, high in (
            ('presence_penalty', extra.presence_penalty, -2, 2), ('min_p', extra.min_p, 0, 1)):
        if value is not None and (not math.isfinite(value) or not low <= value <= high):
            parser.error(f'{name} must be finite and in [{low}, {high}]')
    return extra, remaining


def install_sampling_overrides(engine_cls, presence_penalty=None, min_p=None):
    original = engine_cls._prepare_generation_config
    logged = False

    def prepare(self, request_config):
        nonlocal logged
        params = original(self, request_config)
        # SWIFT returns a fresh SamplingParams for each request; never edit model defaults.
        for key, value in (('presence_penalty', presence_penalty), ('min_p', min_p)):
            if value is not None:
                if not hasattr(params, key):
                    raise RuntimeError(f'Installed vLLM does not support {key}')
                setattr(params, key, value)
        if not logged:
            keys = ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty')
            print('[sampling] ' + ' '.join(f'{k}={getattr(params, k, None)}' for k in keys), flush=True)
            logged = True
        return params

    engine_cls._prepare_generation_config = prepare


def main():
    extra, remaining = parse_extra_args(sys.argv[1:])
    sys.argv[1:] = remaining
    import swift.infer_engine.vllm_engine as engine_module
    from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config
    from swift.pipelines import infer_main

    install_config_bridge(engine_module, Qwen3_5Config)
    install_sampling_overrides(engine_module.VllmEngine, extra.presence_penalty, extra.min_p)
    infer_main()


if __name__ == '__main__':
    main()
