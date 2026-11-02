"""Per-task workflow switches, independent of model and coordinate settings."""

from pathlib import Path
from collage.storage import read


DEFAULTS = {'bbox_refinement': False, 'screen': False}


def normalize_options(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - DEFAULTS.keys():
        raise ValueError('工作流开关字段无效')
    options = {**DEFAULTS, **value}
    for name, enabled in options.items():
        if type(enabled) is not bool:
            raise ValueError(name + ' 必须为 true 或 false')
    return options


def load_options(run):
    return normalize_options(read(Path(run)/'workflow-options.json'))
