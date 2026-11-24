#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL_DIR="${1:-/mnt/d/s_dis/Downloads/ai-infra-wheels/flashinfer-0.7.2.post1-cu130}"
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate ai-infra
cd "$PROJECT_DIR"
python -m pip install --no-index --find-links "$WHEEL_DIR" \
    'flashinfer-cubin==0.7.2.post1' 'flashinfer-jit-cache==0.7.2.post1+cu130'
python -m pip freeze > results/environment/requirements.lock.txt
python -m pip check | tee results/environment/pip-check.txt
flashinfer show-config > results/environment/flashinfer-config.txt 2>&1
python scripts/flashinfer_smoke.py | tee results/environment/flashinfer-smoke.json
