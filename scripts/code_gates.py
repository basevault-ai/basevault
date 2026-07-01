#!/usr/bin/env python3
"""Code-correctness gates: source-level tripwires for recurring bug shapes.

Run in the lint + test flow (lint.yml + the test_code_gates.py pytest); fails
the build (nonzero exit) printing the offending ``file:line``. These are
correctness tripwires, NOT the trust-surface boundary — that lives in
``trust_gates.py`` (which additionally gates the release flow).

  (a) A Tauri ``listen()`` subscription inside a React ``useEffect`` that
      lacks the ``cancelled`` guard. ``listen()`` is async; under StrictMode's
      mount->cleanup->remount the cleanup runs before the listen promise
      resolves, leaving the first subscription live alongside the remount's
      second one -> every event handled twice.

Escape hatch (precise, not a blanket ban): ``// ci-allow:listen-guard -
<reason>`` anywhere in the effect. Run with no args to scan from repo root.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Gate (a): listen()-in-useEffect cancelled guard ------------------------

SRC_DIR = REPO_ROOT / "src"
LISTEN_CALL = re.compile(r"\blisten\s*\(")


def _effect_body_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) char spans of every useEffect arrow-fn body, by
    brace-matching while skipping strings and comments so braces inside
    them don't throw the depth count off."""
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"\buseEffect\s*\(", text):
        j = text.find("{", m.end())
        if j == -1:
            continue
        depth = 0
        i = j
        n = len(text)
        while i < n:
            c = text[i]
            if c in "'\"`":
                quote = c
                i += 1
                while i < n and text[i] != quote:
                    if text[i] == "\\":
                        i += 1
                    i += 1
            elif c == "/" and i + 1 < n and text[i + 1] == "/":
                i = text.find("\n", i)
                if i == -1:
                    i = n
            elif c == "/" and i + 1 < n and text[i + 1] == "*":
                end = text.find("*/", i + 2)
                i = n if end == -1 else end + 1
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    spans.append((j, i))
                    break
            i += 1
    return spans


def _gate_a_violations() -> list[str]:
    out: list[str] = []
    for path in sorted(SRC_DIR.rglob("*.js")) + sorted(SRC_DIR.rglob("*.jsx")):
        name = path.name
        if name.endswith((".test.js", ".test.jsx")):
            continue
        text = path.read_text(encoding="utf-8")
        for start, end in _effect_body_spans(text):
            body = text[start:end]
            lm = LISTEN_CALL.search(body)
            if not lm:
                continue
            if "ci-allow:listen-guard" in body:
                continue
            # The canonical f30faca fix introduces a `cancelled` flag the
            # late-resolving .then() checks to tear its own sub down.
            if re.search(r"\bcancelled\b", body):
                continue
            line = text.count("\n", 0, start + lm.start()) + 1
            out.append(
                f"{path.relative_to(REPO_ROOT)}:{line}: listen() in a "
                f"useEffect with no `cancelled` guard (StrictMode "
                f"double-subscribe -> doubled events)"
            )
    return out


def main() -> int:
    violations = _gate_a_violations()
    if not violations:
        print("code_gates: clean (gate a)")
        return 0
    print("code_gates: FAIL\n")
    for v in violations:
        print(f"  {v}")
    print(
        "\nFix the site, or if it is a legitimate exception add an inline "
        "`// ci-allow:listen-guard - <reason>` anywhere in the effect."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
