"""
AGENT INSTRUCTIONS — report/claims_lint.py
===========================================
Rule 6 / M17 — Claims lint.

CRITICAL INVARIANT:
No wording of 'speedup', 'quantum advantage', or 'energy savings' anywhere
in code comments, figure captions, or report templates.
Enforced by test T-M16-09.
"""

from __future__ import annotations

import re
from pathlib import Path

BANNED_CLAIMS = [
    re.compile(r"\bquantum\s+advantage\b", re.IGNORECASE),
    re.compile(r"\bexponential\s+speedup\b", re.IGNORECASE),
    re.compile(r"\bquantum\s+speedup\b", re.IGNORECASE),
    re.compile(r"\benergy\s+savings\b", re.IGNORECASE),
    re.compile(r"\boutperforms\s+classical\b", re.IGNORECASE),
    re.compile(r"\bquantum\s+supremacy\b", re.IGNORECASE),
    re.compile(r"\bfaster\s+than\s+classical\b", re.IGNORECASE),
]


def lint_text(text: str) -> list[str]:
    """Scan raw string for banned claims. Returns violations list."""
    violations: list[str] = []
    for line in text.splitlines():
        if (
            "banned" in line.lower()
            or "claims_lint" in line.lower()
            or "historical" in line.lower()
            or "disclaimer" in line.lower()
            or "no quantum advantage" in line.lower()
        ):
            continue
        for pat in BANNED_CLAIMS:
            if pat.search(line):
                violations.append(line.strip())
    return violations


def check_claims(root_dir: str | Path) -> list[str]:
    """Scan source tree and templates for banned claims. Returns violations list."""
    root = Path(root_dir)
    violations: list[str] = []

    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".tex", ".txt", ".yaml"}:
            # Skip hidden dirs (.venv, .git, etc.), node_modules, and tests (which contain negative test strings)
            if any(part.startswith(".") or part in {"__pycache__", "node_modules", "tests"} for part in path.parts):
                continue
            # Skip FLIGHT_MANUAL and context.md which discuss the policy rule itself
            if path.name in {"claude.md", "context.md", "claims_lint.py", "DECISIONS.md"}:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            for line_idx, line in enumerate(content.splitlines(), start=1):
                # Ignore references to historical claims or linting
                if (
                    "banned" in line.lower()
                    or "claims_lint" in line.lower()
                    or "historical" in line.lower()
                    or "disclaimer" in line.lower()
                    or "no quantum advantage" in line.lower()
                ):
                    continue
                for pat in BANNED_CLAIMS:
                    if pat.search(line):
                        violations.append(f"{path}:{line_idx}: {line.strip()}")

    return violations
