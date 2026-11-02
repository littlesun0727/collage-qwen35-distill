from collage.config import Config
from collage.vlm.providers import payload_for

def test_vllm_uses_qwen_template_switch():
    config = Config(provider='vllm', model='Qwen3.5-9B', reasoning_enabled=False)
    payload = payload_for(config, 'analyze', [], {})
    assert payload['chat_template_kwargs'] == {'enable_thinking': False}
    assert payload['response_format'] == {'type': 'json_object'}
    assert 'reasoning' not in payload
    config.check_ready()

def test_vllm_policy_round_trip():
    config = Config(provider='vllm', model='Qwen3.5-9B', coordinate_mode='normalized_1000')
    assert config.with_policy(config.model_policy()).model_policy() == config.model_policy()
