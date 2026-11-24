#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate ai-infra
cd "$PROJECT_DIR"
mkdir -p results/environment results/profiling

python -m pip check | tee results/environment/pip-check.txt
python scripts/env_smoke.py | tee results/environment/gpu-triton.json
python -m pip freeze > results/environment/requirements.lock.txt
conda env export --from-history 2> results/environment/conda-export-warnings.txt \
    | sed '/^prefix:/d' > results/environment/environment.yml
python -m torch.utils.collect_env > results/environment/torch-environment.txt \
    2> results/environment/torch-environment-warnings.txt

{
    nvcc --version
    nsys --version
    ncu --version
    compute-sanitizer --version
} > results/environment/nvidia-tools.txt

printf 'Environment validation finished: %s/results/environment\n' "$PROJECT_DIR"
