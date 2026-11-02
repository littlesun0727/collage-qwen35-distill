
from dataclasses import dataclass, replace
from copy import deepcopy
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'engine' / 'scripts'


def env_value(path, name):
    """Read only the requested value; never import unrelated dotenv settings."""
    path = Path(path)
    if not path.is_file():
        return ''
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        match = re.match(r'^\s*(?:export\s+)?' + re.escape(name) + r'\s*=\s*(.*?)\s*$', line)
        if match:
            value = match[1]
            if value[:1] in ('"', "'"):
                quote = value[0]
                end = value.find(quote, 1)
                if end < 1:
                    raise ValueError(f'Unclosed quote for {name}')
                return value[1:end]
            return value.split(' #', 1)[0].strip()
    return ''


@dataclass(frozen=True)
class Config:
    model: str = 'qwen/qwen3.8-flash'
    env_file: str = str(ROOT / '.env')
    reveal_key_file: str = str(ROOT / '.env')
    timeout: int = 180
    max_requests: int = 16
    max_tokens: int = 16000
    reasoning_enabled: bool | None = False
    coordinate_mode: str = 'auto'
    reasoning_max_tokens: int | None = None
    # Explicit Config() instances remain suitable for single-model experiments.
    # Production Config.load() selects Max for reference analysis only.
    analysis_model: str | None = None
    http_retry_delay: float = 30
    provider: str = 'openrouter'
    yibu_credentials: str = str(ROOT / '.env')
    reasoning_effort: str | None = None
    response_format: str = 'json_object'
    stage_overrides: dict | None = None

    def __post_init__(self):
        if self.provider not in {'openrouter', 'yibu', 'codex', 'vllm'}:
            raise ValueError('供应商必须是 openrouter、yibu、codex 或 vllm')
        if not isinstance(self.model, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./:+-]{0,199}', self.model):
            raise ValueError('模型 ID 格式无效')
        if type(self.max_tokens) is not int or not 128 <= self.max_tokens <= 131072:
            raise ValueError('输出 token 上限须为 128～131072 的整数')
        if self.reasoning_enabled is not None and type(self.reasoning_enabled) is not bool:
            raise ValueError('thinking enabled 必须为 true、false 或 null（模型默认）')
        if self.response_format not in {'json_schema', 'json_object', 'prompt'}:
            raise ValueError('JSON 输出方式无效')
        if self.reasoning_effort is not None:
            if self.reasoning_effort not in {'minimal', 'low', 'medium', 'high', 'xhigh'}:
                raise ValueError('推理档位无效')
            if self.reasoning_enabled is not True or self.reasoning_max_tokens is not None:
                raise ValueError('推理档位须开启 thinking，且不能同时设置 thinking token 上限')
        if self.coordinate_mode not in {'auto', 'pixel', 'normalized_1000'}:
            raise ValueError('coordinate_mode must be auto, pixel or normalized_1000')
        if self.http_retry_delay < 0:
            raise ValueError('http_retry_delay cannot be negative')
        if self.reasoning_max_tokens is not None:
            if type(self.reasoning_max_tokens) is not int or not 0 < self.reasoning_max_tokens < self.max_tokens:
                raise ValueError('reasoning_max_tokens must be a positive integer below max_tokens')
            if not self.reasoning_enabled:
                raise ValueError('reasoning_max_tokens requires reasoning_enabled=True')

    @classmethod
    def load(cls, policy_path=None):
        provider = os.getenv('COLLAGE_VLM_PROVIDER', 'openrouter')
        prefix = '' if provider == 'yibu' else 'qwen/'
        codex = provider == 'codex'
        config = cls(model=os.getenv('COLLAGE_VLM_MODEL', 'gpt-6.1-sol' if codex else prefix+'qwen3.8-flash'),
                   analysis_model=os.getenv('COLLAGE_ANALYSIS_MODEL', '' if codex else prefix+'qwen3.8-max-0902') or None,
                   reasoning_enabled=True if codex else False,
                   reasoning_effort='medium' if codex else None,
                   coordinate_mode=os.getenv('COLLAGE_VLM_COORDINATES', cls.coordinate_mode),
                   env_file=os.getenv('COLLAGE_VLM_ENV', cls.env_file),
                   reveal_key_file=os.getenv('COLLAGE_REVEAL_ENV', cls.reveal_key_file),
                   timeout=int(os.getenv('COLLAGE_VLM_TIMEOUT', '600')),
                   max_requests=int(os.getenv('COLLAGE_VLM_MAX_REQUESTS', cls.max_requests)),
                   provider=provider,
                   yibu_credentials=os.getenv('COLLAGE_YIBU_CREDENTIALS', cls.yibu_credentials))
        path = policy_path or os.getenv('COLLAGE_MODEL_CONFIG')
        return config.with_policy(json.loads(Path(path).read_text(encoding='utf-8-sig'))) if path else config

    def for_stage(self, stage):
        if self.stage_overrides:
            return self._stage(self.stage_overrides['analyze' if stage in {'analyze', 'analyze_repair'} else 'subsequent'])
        if stage in {'analyze', 'analyze_repair'} and self.analysis_model:
            return replace(self, model=self.analysis_model, reasoning_enabled=True,
                           reasoning_max_tokens=None, reasoning_effort=None, analysis_model=None)
        return self

    def model_policy(self):
        def entry(config):
            reasoning = {'enabled': config.reasoning_enabled}
            if config.reasoning_max_tokens is not None:
                reasoning['max_tokens'] = config.reasoning_max_tokens
            if config.reasoning_effort is not None:
                reasoning['effort'] = config.reasoning_effort
            return {'model': config.model, 'reasoning': reasoning,
                    'max_tokens': config.max_tokens, 'coordinate_mode': config.coordinate_mode,
                    'provider': config.provider, 'response_format': config.response_format}
        return {'analyze': entry(self.for_stage('analyze')), 'subsequent': entry(self.for_stage('subsequent'))}

    def _stage(self, entry):
        if not isinstance(entry, dict) or set(entry) - {
            'provider', 'model', 'reasoning', 'max_tokens', 'coordinate_mode', 'response_format'
        } or not {'model', 'reasoning', 'max_tokens', 'coordinate_mode'} <= set(entry):
            raise ValueError('阶段模型配置字段无效')
        reasoning = entry['reasoning']
        if not isinstance(reasoning, dict) or set(reasoning) - {'enabled', 'effort', 'max_tokens'}:
            raise ValueError('推理配置字段无效')
        return replace(self, provider=entry.get('provider', 'openrouter'), model=entry['model'],
                       reasoning_enabled=reasoning.get('enabled'), reasoning_effort=reasoning.get('effort'),
                       reasoning_max_tokens=reasoning.get('max_tokens'), max_tokens=entry['max_tokens'],
                       coordinate_mode=entry['coordinate_mode'], response_format=entry.get('response_format', Config.response_format),
                       stage_overrides=None, analysis_model=None)

    def with_policy(self, policy):
        if not isinstance(policy, dict) or set(policy) != {'analyze', 'subsequent'}:
            raise ValueError('模型配置必须包含 analyze 和 subsequent')
        from collage.vlm.providers import request_options
        stages = {name: self._stage(entry) for name, entry in policy.items()}
        for stage in stages.values():
            request_options(stage)  # Fail unsupported combinations before enqueueing or paying.
        base = stages['subsequent']
        return replace(base, stage_overrides=deepcopy(policy))

    def policy_matches(self, policy):
        return policy is None or self.with_policy(policy).model_policy() == self.model_policy()

    def for_run(self, run):
        """Task snapshots survive default changes and apply to later VLM edits too."""
        run = Path(run)
        path = run / 'workflow-models.json'
        if path.is_file():
            return self.with_policy(json.loads(path.read_text(encoding='utf-8-sig')))
        timing = run / 'workflow-timing.json'
        if timing.is_file():
            policy = json.loads(timing.read_text(encoding='utf-8-sig')).get('model_policy')
            if policy:
                return self.with_policy(policy)
        return self

    def check_keys(self):
        for stage in ('analyze', 'subsequent'):
            self.for_stage(stage).check_ready()

    def check_ready(self):
        if self.provider == 'vllm':
            return
        if self.provider == 'codex':
            from collage.vlm.codex.transport import check_runtime
            check_runtime()
        else:
            self.api_key()

    def api_key(self):
        if self.provider == 'vllm':
            return 'EMPTY'
        if self.provider == 'codex':
            raise ValueError('Codex SDK 使用本机 Codex 登录，请运行 codex login；不使用 yibu/OpenRouter API Key')
        if self.provider == 'yibu':
            key = os.getenv('YIBU_API_KEY')
            if key:
                return key
            path = Path(self.yibu_credentials)
            if path.suffix.lower() != '.json':
                key = env_value(path, 'YIBU_API_KEY')
                if not key:
                    raise ValueError(f'YIBU_API_KEY 缺失，请填写 {path}，或设置 YIBU_API_KEY 环境变量')
                return key
            # Explicit legacy JSON paths remain supported for existing callers.
            try:
                data = json.loads(path.read_text(encoding='utf-8-sig'))
                keys = data.get('api_keys', [])
                if not isinstance(keys, list):
                    raise ValueError('api_keys must be an array')
                key = next((k.strip() for k in keys if isinstance(k, str) and k.strip()), '')
            except (OSError, ValueError, AttributeError):
                raise ValueError(f'yibu JSON 凭证文件无法读取：{path}；请检查 COLLAGE_YIBU_CREDENTIALS，默认凭证为工程目录 .env 中的 YIBU_API_KEY') from None
            if not key:
                raise ValueError('yibu 凭证文件缺少非空 api_keys')
            return key
        key = os.getenv('OPENROUTER_API_KEY') or env_value(self.env_file, 'OPENROUTER_API_KEY')
        if not key:
            raise ValueError(f'OPENROUTER_API_KEY 缺失，请填写 {self.env_file}，或设置 OPENROUTER_API_KEY 环境变量')
        return key
