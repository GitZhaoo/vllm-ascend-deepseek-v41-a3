#!/usr/bin/env python3
"""Helpers to reason about the state of a unified-diff patch series.

`git apply --reverse --check` cannot be used alone once the gate defaults have
been pinned to 1: those post-apply edits change lines introduced by the patch,
so the reverse check fails even though the patch is fully applied. This module
adds a content-based fallback that looks for each patch's distinctive added
code lines in the target files.

Gate definitions are normalised on both sides before comparison, so
``_strict_binary_env("X")`` and ``_strict_binary_env("X", "1")`` are treated as
the same evidence.

CLI:
    python3 scripts/patch_utils.py content-state --src <tree> --patch <file>
      -> prints: applied | not_applied | partial

    python3 scripts/patch_utils.py files --patch <file>
      -> prints the target paths, one per line (for revert)
"""

from __future__ import annotations

import argparse
import os
import re
import sys

DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$")
GATE_CALL_RE = re.compile(
    r'(_strict_binary_env\(\s*"[A-Za-z0-9_]+")\s*,\s*"[01]"\s*\)'
)
ENV_GET_RE = re.compile(
    r'(os\.environ\.get\(\s*"[A-Za-z0-9_]+"\s*,\s*)"[01]"(\s*\))'
)


def normalize_gates(text: str) -> str:
    """Remove the 0/1 default difference introduced after applying patches."""
    text = GATE_CALL_RE.sub(r"\1)", text)
    text = ENV_GET_RE.sub(r'\1"0"\2', text)
    return text


def parse_patch(patch_path: str) -> dict[str, list[str]]:
    """Return {target_path: [added lines]} for a unified diff."""
    files: dict[str, list[str]] = {}
    current: str | None = None
    with open(patch_path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            match = DIFF_FILE_RE.match(line)
            if match:
                current = match.group(1)
                files.setdefault(current, [])
                continue
            if current is None:
                continue
            if line.startswith("+") and not line.startswith("+++"):
                added = line[1:]
                if added.strip():
                    files[current].append(added)
    return files


def _distinctive(lines: list[str]) -> list[str]:
    """Pick long, likely unique added code lines (comments are ignored)."""
    code = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Skip pure structural noise that appears everywhere.
        if stripped in {")", "}", "],", "),", "else:", "try:", "finally:"}:
            continue
        code.append(normalize_gates(stripped))
    candidates = [line for line in code if len(line) >= 12]
    if not candidates:
        candidates = code
    candidates.sort(key=len, reverse=True)
    return candidates[:3]


def content_state(src: str, patch_path: str) -> str:
    files = parse_patch(patch_path)
    if not files:
        return "not_applied"

    matched_files = 0
    checked_files = 0
    for rel, added_lines in files.items():
        evidence = _distinctive(added_lines)
        target = os.path.join(src, rel)
        checked_files += 1
        if not evidence or not os.path.exists(target):
            continue
        try:
            with open(target, encoding="utf-8", errors="replace") as fh:
                raw_text = fh.read()
        except OSError:
            continue
        target_lines = {
            normalize_gates(line.strip()) for line in raw_text.splitlines() if line.strip()
        }
        present = sum(1 for line in evidence if line in target_lines)
        if present >= (len(evidence) + 1) // 2:
            matched_files += 1

    if checked_files == 0 or matched_files == 0:
        return "not_applied"
    if matched_files == checked_files:
        return "applied"
    return "partial"


def list_files(patch_path: str) -> list[str]:
    return sorted(parse_patch(patch_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    state = sub.add_parser("content-state")
    state.add_argument("--src", required=True)
    state.add_argument("--patch", required=True)

    files = sub.add_parser("files")
    files.add_argument("--patch", required=True)

    args = parser.parse_args()
    if args.command == "content-state":
        print(content_state(args.src, args.patch))
        return 0
    if args.command == "files":
        for path in list_files(args.patch):
            print(path)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
