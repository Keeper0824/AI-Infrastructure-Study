#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate ai-infra
cd "$PROJECT_DIR"
mkdir -p results/profiling

# Nsight uses POSIX FIFOs internally; DrvFS/NTFS cannot host them.
# Only profiler IPC uses /tmp. User scripts, caches and reports stay on /mnt/.
TMPDIR=/tmp NSYS_TMPDIR=/tmp nsys profile \
    --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true \
    -o results/profiling/env-smoke python scripts/env_smoke.py \
    > results/profiling/nsys.log 2>&1
test -s results/profiling/env-smoke.nsys-rep

TMPDIR=/tmp compute-sanitizer --tool memcheck --error-exitcode 1 \
    python scripts/env_smoke.py > results/profiling/compute-sanitizer.log 2>&1

TMPDIR=/tmp ncu --set basic --kernel-name regex:add_one --launch-count 1 \
    --clock-control none --force-overwrite -o results/profiling/env-smoke python scripts/env_smoke.py \
    > results/profiling/ncu.log 2>&1
if grep -q 'ERR_NVGPUCTRPERM' results/profiling/ncu.log; then
    printf 'NCU requires Windows NVIDIA GPU performance counter permission. See results/profiling/ncu.log\n' >&2
    exit 2
fi
if [[ ! -s results/profiling/env-smoke.ncu-rep && ! -s results/profiling/env-smoke.ncu-repz ]]; then
    printf 'NCU did not generate a readable report. See results/profiling/ncu.log\n' >&2
    exit 3
fi
printf 'All profiler reports generated in %s/results/profiling\n' "$PROJECT_DIR"
