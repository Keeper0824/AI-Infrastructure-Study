#!/usr/bin/env python3
"""Check that every `vllm/...py:LINE` citation in the course docs is resolvable.

Usage:
    python scripts/verify_refs.py
    python scripts/verify_refs.py --vllm /path/to/vllm-0.32.0

It checks:
  1. the file exists under the vLLM tree;
  2. the cited line is within the file;
  3. (best effort) if the citation is followed by `· symbol`, the symbol name
     appears somewhere in the file.

It does NOT prove that the line is the *right* line — only that it exists.
Use it as a cheap smoke test before committing docs.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CITATION_RE = re.compile(
    r"(?P<path>vllm/[A-Za-z0-9_./-]+\.py):(?P<line>\d+)"
    r"(?:\s*·\s*`?(?P<symbol>[A-Za-z_][A-Za-z0-9_.]*))?"
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VLLM = ROOT.parent / "target" / "vllm-0.32.0"


def iter_docs(root: Path):
    yield from sorted((root / "docs").glob("*.md"))
    for extra in ("README.md", "ROADMAP.md"):
        p = root / extra
        if p.exists():
            yield p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", type=Path, default=DEFAULT_VLLM)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    vllm = args.vllm.resolve()
    if not vllm.exists():
        print(f"vLLM tree not found: {vllm}", file=sys.stderr)
        return 2

    total = 0
    broken: list[tuple[Path, int, str]] = []
    missing_symbol: list[tuple[Path, int, str]] = []

    for doc in iter_docs(args.root):
        text = doc.read_text(encoding="utf-8")
        for m in CITATION_RE.finditer(text):
            total += 1
            rel = m.group("path")
            line_no = int(m.group("line"))
            symbol = m.group("symbol")
            target = vllm / rel
            if not target.exists():
                broken.append((doc, line_no, f"missing file: {rel}"))
                continue
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            if line_no < 1 or line_no > len(lines):
                broken.append(
                    (doc, line_no, f"line {line_no} out of range ({len(lines)}): {rel}")
                )
                continue
            if symbol:
                short = symbol.split(".")[-1]
                # Search the whole file; a symbol may be defined on a nearby line.
                if short not in target.read_text(
                    encoding="utf-8", errors="replace"
                ):
                    missing_symbol.append((doc, line_no, f"{rel}:{line_no} · {short}"))

    print(f"checked {total} citations in {len(list(iter_docs(args.root)))} docs")
    if broken:
        print(f"\nBROKEN ({len(broken)}):")
        for doc, line, msg in broken:
            print(f"  {doc.relative_to(args.root)}:{line}: {msg}")
    if missing_symbol:
        print(f"\nSYMBOL NOT FOUND ({len(missing_symbol)}):")
        for doc, line, msg in missing_symbol:
            print(f"  {doc.relative_to(args.root)}:{line}: {msg}")

    if broken or missing_symbol:
        return 1
    print("all citations resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
