#!/usr/bin/env python3
"""
Guardrail for model-specific artifact pathing.

This check intentionally audits only files where outputs should now be model-scoped.
Shared, model-independent assets (for example static prompt fixtures) are out of scope.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files that should not write model artifacts into flat `results/`.
AUDITED_FILES = [
  "7.py",
  "12.py",
  "23.py",
  "28.py",
  "51.py",
  "52.py",
  "53.py",
  "54.py",
  "55.py",
  "56.py",
  "57.py",
]

BANNED_PATTERNS = [
  re.compile(r'os\.path\.join\(\s*["\']results["\']\s*,'),
  re.compile(r'Path\(\s*["\']results["\']\s*\)'),
  re.compile(r'["\']results/'),
]


def main() -> int:
  violations: list[tuple[str, int, str]] = []
  for rel_path in AUDITED_FILES:
    file_path = REPO_ROOT / rel_path
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for line_no, line in enumerate(lines, start=1):
      for pattern in BANNED_PATTERNS:
        if pattern.search(line):
          violations.append((rel_path, line_no, line.strip()))
          break

  if violations:
    print("Flat results path audit failed. Model-specific files still reference flat `results/`:")
    for rel_path, line_no, line in violations:
      print(f" - {rel_path}:{line_no}: {line}")
    return 1

  print("Path audit passed.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
