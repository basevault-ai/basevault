"""Name-leak guard (maintainer-side pytest).

Fails if a maintainer-held personal name appears in tracked source — the leak
class the static overfit-token tests kept re-introducing (a name hardcoded into
a forbidden-token list, shipping the very name it forbids).

The list is never tracked: it lives in `data/leak_guard_names.txt` (the
local-only `data/` repo, symlinked into each worktree — see WORKER.md), one
name per line, with the `LEAK_GUARD_NAMES` env var (comma-separated) as a
fallback. No list -> the test skips, so CI / forks / fresh clones (which have
none) don't break; the maintainer, who introduces the names, catches a leak
locally at the source.

To run this automatically on every push (closing the forgot-to-run gap while
keeping the names local), install the pre-push hook:

    git config core.hooksPath scripts/hooks

See `scripts/hooks/pre-push`.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SELF = Path(__file__).resolve()
_NAMES_FILE = _REPO_ROOT / "data" / "leak_guard_names.txt"


def _forbidden_names() -> list[str]:
    """Configured names: the data file (one per line, `#` comments ok), else
    the LEAK_GUARD_NAMES env var (comma-separated). Trimmed, empties dropped."""
    try:
        text = _NAMES_FILE.read_text(encoding="utf-8")
        raw = [ln.split("#", 1)[0] for ln in text.splitlines()]
    except OSError:
        raw = os.environ.get("LEAK_GUARD_NAMES", "").split(",")
    return [n.strip() for n in raw if n.strip()]


def _word_bounded(name: str) -> re.Pattern:
    # (?:^|[^a-z])<name>(?:[^a-z]|$), case-insensitive so the [^a-z] boundary
    # rejects A-Z too -> a short name never fires inside a longer word
    # ("examplename" won't hit "examplenamex"). re.escape keeps a literal
    # value literal.
    return re.compile(r"(?:^|[^a-z])" + re.escape(name) + r"(?:[^a-z]|$)", re.I)


def find_violations(names: list[str], root: Path = _REPO_ROOT) -> list[str]:
    """`file:line` for every tracked line matching any name. Skips binaries
    (NUL heuristic) and this file."""
    if not names:
        return []
    pats = [(n, _word_bounded(n)) for n in names]
    try:
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []
    out: list[str] = []
    for rel in tracked:
        path = root / rel
        if path.resolve() == _SELF:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8192]:
            continue
        for i, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            for name, pat in pats:
                if pat.search(line):
                    out.append(f"{rel}:{i}: forbidden personal name '{name}'")
                    break  # one hit per line is enough
    return out


def test_no_forbidden_name_in_tracked_source():
    names = _forbidden_names()
    if not names:
        pytest.skip("no name list (data/leak_guard_names.txt or LEAK_GUARD_NAMES)")
    bad = find_violations(names)
    assert not bad, (
        "personal-name leak in tracked source — remove it (the list is "
        "maintainer-held, never add a name to a tracked file):\n  "
        + "\n  ".join(bad)
    )


def test_matcher_is_word_bounded_and_case_insensitive():
    # The one load-bearing property: bounded, case-insensitive, no super-string
    # hit, regex-safe. "examplename" is synthetic — never a real name.
    p = _word_bounded("examplename")
    assert p.search("hi examplename!") and p.search("EXAMPLENAME")
    assert not p.search("examplenamex") and not p.search("xexamplename")
    assert _word_bounded("a.b").search("a.b") and not _word_bounded("a.b").search("axb")
