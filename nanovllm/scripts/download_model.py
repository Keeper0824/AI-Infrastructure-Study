#!/usr/bin/env python3
"""Download Qwen3-0.6B into ./models/Qwen3-0.6B.

Prefer ModelScope on mainland-China networks; fall back to the HF mirror.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --source hf
    python scripts/download_model.py --model Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B",
    )
    parser.add_argument("--source", choices=["modelscope", "hf"], default="modelscope")
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "modelscope":
        if shutil.which("modelscope"):
            cmd = ["modelscope"]
        else:
            cmd = [sys.executable, "-m", "modelscope"]
        run(
            cmd
            + [
                "download",
                "--model",
                args.model,
                "--local_dir",
                str(args.local_dir),
            ]
        )
    else:
        env = os.environ.copy()
        env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        cli = "hf" if shutil.which("hf") else "huggingface-cli"
        subprocess.run(
            [
                cli,
                "download",
                args.model,
                "--local-dir",
                str(args.local_dir),
            ],
            check=True,
            env=env,
        )

    config = args.local_dir / "config.json"
    if not config.exists():
        print(f"download finished but {config} is missing", file=sys.stderr)
        return 1
    print(f"ok: {args.local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
