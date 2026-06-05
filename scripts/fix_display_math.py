#!/usr/bin/env python3
"""Wrap multi-line $$...$$ blocks in align* so MathJax respects \\\\ line breaks."""

import re
import sys
from pathlib import Path

BLOCK_PATTERN = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
LINE_BREAK_PATTERN = re.compile(r"\\\\")
BEGIN_ENV_PATTERN = re.compile(r"\\begin\{[a-zA-Z*]+\}")


def needs_align_wrapper(inner: str) -> bool:
    if not LINE_BREAK_PATTERN.search(inner):
        return False
    if BEGIN_ENV_PATTERN.search(inner):
        return False
    return True


def fix_block(match: re.Match[str]) -> str:
    inner = match.group(1).strip("\n")
    if not needs_align_wrapper(inner):
        return match.group(0)
    return f"$$\n\\begin{{align*}}\n{inner}\n\\end{{align*}}\n$$"


def fix_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    updated = BLOCK_PATTERN.sub(fix_block, text)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "src")
    changed = [p for p in root.rglob("*.md") if fix_file(p)]
    for path in changed:
        print(f"fixed display math: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
