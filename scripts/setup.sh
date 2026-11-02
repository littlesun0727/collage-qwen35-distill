#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs .cache
exec > >(tee -a logs/install-commands.log) 2>&1
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=120
export PIP_CACHE_DIR="$PWD/.cache/pip"
index="${PIP_INDEX_URL:-https://pypi.org/simple}"
if [ ! -x .bootstrap/bin/uv ]; then
  python3 -m venv .bootstrap
  .bootstrap/bin/python -m pip install --index-url "$index" uv
fi
if [ ! -x .venv/bin/python ]; then
  .bootstrap/bin/uv venv --python 3.12 --managed-python .venv
fi
.bootstrap/bin/uv pip install --python .venv/bin/python --index-url "$index" -r requirements-server.txt -r requirements.txt
.venv/bin/python -m pip --version >/dev/null 2>&1 || .bootstrap/bin/uv pip install --python .venv/bin/python pip
.venv/bin/python -m pip freeze > logs/environment.txt
.venv/bin/python -m pip list --format=json > logs/pip-packages.json
.venv/bin/python --version > logs/python-version.txt
nvidia-smi > logs/nvidia-smi.txt
printf 'Setup complete. Use .venv/bin/python for all commands.\n'
