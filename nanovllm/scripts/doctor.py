#!/usr/bin/env python3
"""Environment doctor for the nanovllm course.

Run from anywhere:

    python /home/dictator/ai_infra/nanovllm/scripts/doctor.py

It checks the things M00 assumes:
  * the repo layout is intact (nanovllm/, tests/, docs/);
  * the current Python is the one you think it is;
  * pytest is importable;
  * pytest can collect the M00 tests.

Exit code 0 = ready.  Exit code 1 = something must be fixed first.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def line(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> int:
    ok = True

    line("repo")
    print(f"root: {ROOT}")
    for rel in ("nanovllm", "tests", "docs", "pyproject.toml"):
        p = ROOT / rel
        exists = p.exists()
        print(f"  {'OK ' if exists else 'MISS'} {rel}")
        ok = ok and exists

    line("python")
    env_name = os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
    print(f"conda env:  {env_name}")
    print(f"executable: {sys.executable}")
    print(f"version:    {sys.version.split()[0]}")
    if env_name != "vllm_learn":
        print("  NOTE: the course env is `vllm_learn`; activate it if this is the base env")
    if sys.version_info < (3, 10):
        print("  ERROR: Python >= 3.10 is required")
        ok = False

    line("pytest")
    try:
        import pytest  # noqa: PLC0415

        print(f"  OK  pytest {pytest.__version__}")
        print(f"      module: {pytest.__file__}")
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"  MISS pytest: {exc}")
        print("  fix (in your own terminal):")
        print("       conda activate vllm_learn")
        print("       python -m pip install pytest")
        print('       # or, from the repo root: python -m pip install -e ".[dev]"')
        ok = False
        pytest = None

    if pytest is not None:
        line("pytest collection")
        cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=60,
            )
        except Exception as exc:  # pragma: no cover
            print(f"  ERROR: {exc}")
            ok = False
        else:
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            for ln in tail[-8:]:
                print(f"  {ln}")
            if proc.returncode != 0:
                print("  ERROR: pytest could not collect tests")
                ok = False

    line("optional: gpu stack")
    try:
        import torch  # noqa: PLC0415

        cuda = torch.cuda.is_available()
        print(f"  torch {torch.__version__}  cuda={cuda}")
        if cuda:
            print(f"  gpu: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"  (not installed / not importable: {exc})")
        print("  this is fine for M00; M01+ needs torch")

    line("next")
    if ok:
        print("ready. run:")
        print(f"  cd {ROOT} && python -m pytest tests -q")
        print("expected for the unfinished M00 skeleton: 31 failed, 3 passed")
    else:
        print("fix the errors above, then re-run this doctor")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
