"""Tests for engine._dotenv — prod vs eval `.env` loading.

The load-bearing contract: evals load their OWN keys from a repo-root
`.env` and must NEVER fall back to (or inherit operational flags from)
the prod app-support dotenv. `load_eval` enforces that by claiming
dotenv loading for the process (`_claimed`), which turns `load` (the
prod loader) into a no-op. `_claimed` is plain module state, not an
env var — each test resets it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from engine import _dotenv


@pytest.fixture(autouse=True)
def _reset_claim(monkeypatch):
    """Keep the process-wide claim out of cross-test bleed."""
    monkeypatch.setattr(_dotenv, "_claimed", False)


def test_load_skips_prod_once_claimed(tmp_path, monkeypatch):
    """`load` is a no-op once an eval has claimed loading — the prod
    dotenv is off-limits, even if the canonical path exists."""
    prod = tmp_path / "prod.env"
    prod.write_text("MARKER_SKIP=prod_value\n")
    monkeypatch.setattr(_dotenv, "candidates", lambda: [prod])
    monkeypatch.delenv("MARKER_SKIP", raising=False)

    monkeypatch.setattr(_dotenv, "_claimed", True)
    assert _dotenv.load() is None
    # The fake prod dotenv was never read.
    assert "MARKER_SKIP" not in os.environ


def test_load_reads_prod_when_unclaimed(tmp_path, monkeypatch):
    """In the normal (production) path — nothing claimed — `load` reads
    the canonical dotenv as before."""
    prod = tmp_path / "prod.env"
    prod.write_text("MARKER_PROD=yes\n")
    monkeypatch.setattr(_dotenv, "candidates", lambda: [prod])
    monkeypatch.delenv("MARKER_PROD", raising=False)

    assert _dotenv.load() == prod
    assert os.environ["MARKER_PROD"] == "yes"


def test_load_eval_loads_repo_root_and_claims(tmp_path, monkeypatch):
    """`load_eval` loads `<repo-root>/.env` AND claims loading so a
    subsequent prod `load()` does nothing — the two-part guarantee that
    an eval uses separate keys and never touches prod."""
    root = tmp_path / "repo"
    (root / "sub" / "deep").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / ".env").write_text("TINFOIL_API_KEY=eval_key\nMARKER_EVAL=1\n")
    prod = tmp_path / "prod.env"
    prod.write_text("TINFOIL_API_KEY=prod_key\n")
    monkeypatch.setattr(_dotenv, "candidates", lambda: [prod])
    for k in ("TINFOIL_API_KEY", "MARKER_EVAL"):
        monkeypatch.delenv(k, raising=False)

    # Seed the walk from deep inside the tree — it must climb to `.git`.
    loaded = _dotenv.load_eval(root / "sub" / "deep")
    assert loaded == root / ".env"
    assert os.environ["TINFOIL_API_KEY"] == "eval_key"
    assert os.environ["MARKER_EVAL"] == "1"

    # Prod is now off-limits — load() is a no-op, eval key survives.
    assert _dotenv.load() is None
    assert os.environ["TINFOIL_API_KEY"] == "eval_key"


def test_load_eval_overrides_a_stale_env_value(tmp_path, monkeypatch):
    """If a stray earlier import already set a key, the repo-root eval
    value still wins (override=True)."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").write_text("gitdir: /elsewhere\n")  # worktree-style file
    (root / ".env").write_text("TINFOIL_API_KEY=eval_key\n")
    monkeypatch.setenv("TINFOIL_API_KEY", "stale_prod_key")

    _dotenv.load_eval(root)
    assert os.environ["TINFOIL_API_KEY"] == "eval_key"


def test_load_eval_claims_even_without_repo_env(tmp_path, monkeypatch):
    """No repo-root `.env` → still claim loading (return None, prod
    off-limits), so the eval surfaces a missing key loudly instead of
    silently falling back to the developer's prod key."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()

    assert _dotenv.load_eval(root) is None
    assert _dotenv._claimed is True


def test_find_repo_root_falls_back_to_start_when_no_git(tmp_path):
    """No `.git` anywhere up the tree → fall back to the start dir, so
    callers always get a usable directory."""
    start = tmp_path / "a" / "b"
    start.mkdir(parents=True)
    assert _dotenv._find_repo_root(start) == start.resolve()


def _make_linked_worktree(tmp_path) -> tuple[Path, Path]:
    """Build a fake main clone + a linked worktree pointing into its
    `.git/worktrees/<name>`. Returns (main_root, worktree_root)."""
    main = tmp_path / "oss"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    wt = tmp_path / "worktrees" / "wt"
    wt.mkdir(parents=True)
    (wt / ".git").write_text(
        f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n"
    )
    return main, wt


def test_main_worktree_root_resolves_from_linked_worktree(tmp_path):
    """A linked worktree resolves to the MAIN clone's root (where the
    shared `.env` lives), via the `.git` pointer file."""
    main, wt = _make_linked_worktree(tmp_path)
    assert _dotenv._main_worktree_root(wt) == main


def test_load_eval_uses_main_worktree_env_when_no_local(tmp_path, monkeypatch):
    """From a linked worktree with NO local `.env`, load_eval falls to
    the MAIN worktree's `.env` — so every worktree shares one eval env
    file (keys live in one place)."""
    main, wt = _make_linked_worktree(tmp_path)
    (main / ".env").write_text("TINFOIL_API_KEY=shared_key\n")
    monkeypatch.delenv("TINFOIL_API_KEY", raising=False)

    loaded = _dotenv.load_eval(wt)
    assert loaded == main / ".env"
    assert os.environ["TINFOIL_API_KEY"] == "shared_key"


def test_load_eval_prefers_local_worktree_env_over_main(tmp_path, monkeypatch):
    """A worktree-local `.env` wins over the main tree's — per-worktree
    override."""
    main, wt = _make_linked_worktree(tmp_path)
    (main / ".env").write_text("TINFOIL_API_KEY=shared_key\n")
    (wt / ".env").write_text("TINFOIL_API_KEY=local_key\n")
    monkeypatch.delenv("TINFOIL_API_KEY", raising=False)

    loaded = _dotenv.load_eval(wt)
    assert loaded == wt / ".env"
    assert os.environ["TINFOIL_API_KEY"] == "local_key"
