"""`.env` discovery for BaseVault.

The PRODUCTION pipeline reads its `.env` from exactly one location:
``~/Library/Application Support/BaseVault/.env``. That's what the
first-run wizard writes, what the Settings UI edits, and what every
subprocess launch picks up.

Consumers import and call :func:`load` once at module/script start:

    from engine._dotenv import load
    load()  # fire-and-forget, or capture the returned Path for logs

EVALS deliberately do NOT use that prod dotenv — they keep their own
keys (separate budgets / a friends-and-family key) in a `.env` at the
REPO ROOT. An eval entrypoint calls :func:`load_eval` FIRST (before any
``engine`` import that would auto-``load()``); that loads the repo-root
`.env` and marks the prod app-support dotenv off-limits for the rest of
the process, so an eval can never silently fall back to — or inherit
operational flags (e.g. ``BASEVAULT_INJECT_FAILURES``) from — the
developer's production config.
"""
from __future__ import annotations

import os
from pathlib import Path

# Process-wide guard: the first `.env` claimed wins. An eval entrypoint
# calls :func:`load_eval` before any engine import would auto-``load()``
# the prod dotenv; that claims loading for the process so the prod
# app-support `.env` is never read during an eval. Plain module state —
# NOT an env var, not a user-facing knob.
_claimed = False


def candidates() -> list[Path]:
    """Return the single canonical PROD `.env` path (as a list for
    backward compatibility with call sites that iterate)."""
    return [Path.home() / "Library" / "Application Support" / "BaseVault" / ".env"]


def load() -> Path | None:
    """Load the user's PROD `.env` into the process env.

    Returns the path that was loaded, or ``None`` if it doesn't exist,
    python-dotenv isn't installed, or an eval already claimed loading
    via :func:`load_eval` (then the prod dotenv is off-limits).
    """
    if _claimed:
        return None
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None
    for p in candidates():
        if p.exists():
            load_dotenv(p)
            return p
    return None


def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the repository root — the first
    ancestor carrying a ``.git`` entry (a dir in a normal clone, a file
    in a linked worktree). Falls back to ``start`` itself when none is
    found, so callers always get a usable directory."""
    start = start.resolve()
    for d in (start, *start.parents):
        if (d / ".git").exists():
            return d
    return start


def _main_worktree_root(root: Path) -> Path:
    """The MAIN working tree for ``root`` — the same directory for every
    linked worktree, so a single repo-root `.env` serves them all.

    For a normal clone ``root`` already IS the main tree. For a linked
    worktree ``root/.git`` is a file ``gitdir: <common>/worktrees/<name>``;
    the main tree is the parent of that ``.git`` common dir. Resolved by
    parsing the pointer file (no subprocess). Falls back to ``root`` on
    any surprise."""
    gitfile = root / ".git"
    if gitfile.is_file():
        try:
            text = gitfile.read_text().strip()
        except OSError:
            return root
        if text.startswith("gitdir:"):
            gitdir = Path(text.split(":", 1)[1].strip())
            for anc in gitdir.parents:
                if anc.name == ".git":
                    return anc.parent
    return root


def load_eval(start: Path | None = None) -> Path | None:
    """Load the EVAL `.env` from the repo root and claim dotenv loading
    for the rest of the process.

    Evals keep their own keys at ``<repo-root>/.env`` so they run on a
    separate budget / friends-and-family key, never the developer's
    production config. Call this FIRST in an eval entrypoint — before
    any ``engine`` import that would auto-``load()`` — so the claim is
    in place before that fires and the prod dotenv is never read.

    ``start`` seeds the repo-root walk (default: this file's location).
    Returns the loaded `.env` path, or ``None`` if there isn't one at
    the repo root (the eval then runs with whatever keys are already in
    the environment — and a missing key surfaces loudly at first use,
    not as a silent fall-through to prod).
    """
    global _claimed
    # Claim unconditionally — even with no repo-root `.env`, an eval must
    # never fall back to the prod app dotenv.
    _claimed = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None
    root = _find_repo_root(start or Path(__file__).parent)
    # Prefer a worktree-local `.env` (per-worktree override), else the
    # main working tree's `.env` so every linked worktree shares one
    # eval env file. De-duped so a normal clone checks its single `.env`
    # once.
    seen: set[Path] = set()
    for p in (root / ".env", _main_worktree_root(root) / ".env"):
        if p in seen:
            continue
        seen.add(p)
        if p.exists():
            # override=True so eval keys win even if a stray earlier
            # import already pulled something into the environment.
            load_dotenv(p, override=True)
            return p
    return None


def is_dev() -> bool:
    """True when ``IS_DEV=1`` is set in the user's dotenv (or in the
    process env). This is the convention developers use to mark a
    workstation that runs `tauri dev` / direct-from-source — the
    scheduler reads it to switch the throttler interval from 20s to
    1s and double the pool. Falsey on production / app-bundle runs
    that never see the dev `.env`.
    """
    raw = os.environ.get("IS_DEV", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}
