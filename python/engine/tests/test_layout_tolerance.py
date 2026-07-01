"""Tests for `_find_run_dir_by_id`'s tolerance of multiple on-disk
layouts.

The pipeline historically wrote runs to `<logs>/<session>/<eval>/<run>/`
(legacy nested). The flatten work introduces:
  - `<logs>/<run>/` for app + ad-hoc CLI runs (post-flatten)
  - `<logs>/sweeps/<sweep_id>/<run>/` for the sweep harness

`_find_run_dir_by_id` resolves a run by name and is used by the
`--resume-run-id` path. It must accept all three shapes so resume
keeps working across the migration window.
"""
from __future__ import annotations

from pathlib import Path

import pytest


from engine import runner


def _seed_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")


@pytest.fixture
def logs_root(tmp_path, monkeypatch):
    root = tmp_path / "logs"
    root.mkdir()
    monkeypatch.setattr(runner, "_LOGS_ROOT", root)
    monkeypatch.setattr(runner, "_RUN_ROOT", root)
    return root


def test_find_run_dir_by_id_locates_flat_run(logs_root):
    run_dir = logs_root / "2026-05-08T10-00-00Z-flat"
    _seed_run(run_dir)
    found = runner._find_run_dir_by_id("2026-05-08T10-00-00Z-flat")
    assert found == run_dir


def test_find_run_dir_by_id_locates_legacy_nested_run(logs_root):
    run_dir = (
        logs_root
        / "2026-04-29T12-00-00Z-app"
        / "eval-2026-04-29T12-00-00Z"
        / "2026-04-29T12-00-00Z-abcd"
    )
    _seed_run(run_dir)
    found = runner._find_run_dir_by_id("2026-04-29T12-00-00Z-abcd")
    assert found == run_dir


def test_find_run_dir_by_id_locates_sweep_run(logs_root):
    run_dir = (
        logs_root
        / "sweeps"
        / "2026-05-08T11-00-00Z-experiment-test"
        / "01_case_a-tee"
    )
    _seed_run(run_dir)
    found = runner._find_run_dir_by_id("01_case_a-tee")
    assert found == run_dir


def test_find_run_dir_by_id_returns_none_when_absent(logs_root):
    assert runner._find_run_dir_by_id("does-not-exist") is None


def test_find_run_dir_by_id_picks_most_recent_on_collision(
    logs_root, monkeypatch,
):
    """In the rare case the same run_id exists in both the legacy and
    the flat trees (e.g. mid-migration when something resumes via
    BASEVAULT_RUN_NAME), pick the most-recently-touched one — that's
    the one the user is actively working with."""
    flat = logs_root / "shared-id"
    nested = (
        logs_root
        / "2026-04-29T12-00-00Z-app"
        / "eval-2026-04-29T12-00-00Z"
        / "shared-id"
    )
    _seed_run(nested)
    _seed_run(flat)
    # `flat` was just written second → newer mtime → expected match.
    import os
    import time
    older = time.time() - 3600
    os.utime(nested, (older, older))
    found = runner._find_run_dir_by_id("shared-id")
    assert found == flat
