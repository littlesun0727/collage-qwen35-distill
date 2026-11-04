"""SWIFT inference entry point with Qwen3.5 config compatibility for vLLM."""
from contextlib import contextmanager
from copy import deepcopy


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


def main():
    import swift.infer_engine.vllm_engine as engine_module
    from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config
    from swift.pipelines import infer_main

    install_config_bridge(engine_module, Qwen3_5Config)
    infer_main()


if __name__ == '__main__':
    main()
