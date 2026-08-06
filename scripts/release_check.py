#!/usr/bin/env python3
"""Check that a Train and Eval Monitor checkout is portable and sanitized."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REQUIRED_FILES = (
    "SKILL.md",
    "README.md",
    "README.zh-CN.md",
    "assets/eval-monitor/dashboard.html",
    "assets/eval-monitor/monitor_server.py",
    "assets/eval-monitor/training_dashboard.html",
    "assets/eval-monitor/training_monitor_server.py",
    "scripts/eval_monitor.py",
    "scripts/experiment_manifest.py",
    "scripts/release_check.py",
    "scripts/streamlake_experiments.py",
)
RUNTIME_NAMES = {"cookie", "project_id", "config", "data", "raw", "exports", "monitor_server.pid", "monitor_server.log"}
SENSITIVE_PATTERNS = (
    re.compile(r"(?i)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # Match a concrete long token value, not source references such as
    # ``self._cookie`` or documentation placeholders such as ``...``.
    re.compile(r"(?i)\b(?:authorization|cookie|access[_-]?token|api[_-]?key)\s*[:=]\s*(?:Bearer\s+)?[A-Za-z0-9+/=_-]{16,}"),
    re.compile(r"(?<!<)(?<![A-Za-z0-9_])/root/(?!\.\.\.)"),
    re.compile(r"(?<!<)(?<![A-Za-z0-9_])/home/(?!\.\.\.)"),
    re.compile(r"https?://github\.com/(?!<OWNER>/)[A-Za-z0-9_.-]+/"),
)


def files_under(root: Path) -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"],
            text=True,
        )
        return [root / line for line in output.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        return [path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts]


def check(root: Path) -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            errors.append(f"missing required file: {relative}")

    config_path = root / "assets/eval-monitor/training_monitor_config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            errors.append(f"invalid training config template: {error}")
        else:
            if config.get("outputs_roots") != [] or config.get("targets") != []:
                errors.append("training config template contains machine-specific roots or targets")

    for path in files_under(root):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root)
        if path.name in RUNTIME_NAMES or path.suffix in {".sqlite", ".pyc"}:
            errors.append(f"runtime/generated file is present: {relative}")
            continue
        if path.suffix.lower() not in {".md", ".py", ".html", ".json", ".yaml", ".yml", ".ps1"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            errors.append(f"unreadable text file: {relative}")
            continue
        # Unit tests intentionally use obvious fake values to exercise credential
        # handling.  They are not runtime credentials and should not make a
        # sanitized source checkout fail this check.
        if relative.parts and relative.parts[0] == "tests":
            continue
        if path.name == "release_check.py":
            continue
        for pattern in SENSITIVE_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible personal path, account, or credential in {relative}: {pattern.pattern}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    errors = check(args.root.expanduser().resolve())
    if errors:
        print("release check failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("release check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
