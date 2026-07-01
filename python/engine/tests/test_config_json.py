"""Tests for the start-of-run config.json snapshot.

config.json holds every field that's FIXED at run-start time (routing,
inputs, identity, version stamps). Anything dynamic — status, progress,
duration, error — derives from llm-calls.jsonl and never lands in
config.json. Issue #165.

The split protects the start-of-run record from torn writes during the
run: config.json is opened ONCE and never touched again.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


from engine import runner


@pytest.fixture(autouse=True)
def _reset_run_state(tmp_path):
    runner._run_state = None
    runner._run_dir = tmp_path
    yield
    runner._run_state = None
    runner._run_dir = None


def _seed_run_state(extra: dict | None = None) -> dict:
    """Populate `runner._run_state` with a representative shape so the
    helpers under test have something to project. Mirrors the structure
    `_init_run_json` produces post-init."""
    base = {
        # Static — should land in config.json.
        "run_id": "2026-05-07T12-00-00Z-abcd",
        "short_id": "abcd",
        "agent": "app",
        "created_at": "2026-05-07T12:00:00Z",
        "inputs": ["/tmp/a.txt"],
        "input_count": 1,
        "mode": "test",
        "vault_dir": "/tmp/vault/run",
        "run_dir": "/tmp/logs/run",
        "run_config": {"mode": "test", "stage_models": {}},
        "provider": "test",
        "model": "test-model",
        # Dynamic — must NOT land in config.json.
        "updated_at": "2026-05-07T12:00:01Z",
        "status": "running",
        "progress": {"stage": "init", "completed": 0, "total": 0},
        "duration_ms": None,
        "error": None,
        "pid": 12345,
        "warnings": {"timeouts": 1},
        "llm_cache": {"hits": 5},
        "resumed_at": "2026-05-07T12:00:30Z",
    }
    if extra:
        base.update(extra)
    runner._run_state = base
    return base


class TestConfigJsonSchema:
    """The static field set is the on-disk contract. Adding a field is a
    schema bump for old readers; these tests pin the contract."""

    def test_config_contains_only_whitelisted_static_fields(self, tmp_path):
        runner._run_dir = tmp_path
        _seed_run_state()
        runner._write_run_config_once()
        cfg = json.loads((tmp_path / "config.json").read_text())
        # Every key landed must be in the static set.
        for k in cfg.keys():
            assert k in runner._CONFIG_JSON_FIELDS, (
                f"unexpected dynamic field {k!r} in config.json"
            )

    def test_config_excludes_known_dynamic_fields(self, tmp_path):
        runner._run_dir = tmp_path
        _seed_run_state()
        runner._write_run_config_once()
        cfg = json.loads((tmp_path / "config.json").read_text())
        for k in ("status", "progress", "duration_ms", "error", "pid",
                  "warnings", "llm_cache", "resumed_at", "updated_at"):
            assert k not in cfg, (
                f"dynamic field {k!r} leaked into config.json"
            )

    def test_config_includes_full_run_config_snapshot(self, tmp_path):
        runner._run_dir = tmp_path
        _seed_run_state(extra={
            "run_config": {
                "mode": "tee",
                "primary_model": "kimi-k2-6",
                "stage_models": {"extract": "kimi-k2-6"},
                "stage_reasoning": {"extract": False},
                "temperature": 0.0,
                "sentiment": "neutral",
                "inputs": [{"path": "/tmp/a.txt", "size_bytes": 11,
                            "sha256": "abc"}],
                "pipeline_git_sha": "deadbeef",
                "app_version": "0.1.0",
            },
        })
        runner._write_run_config_once()
        cfg = json.loads((tmp_path / "config.json").read_text())
        assert cfg["run_config"]["primary_model"] == "kimi-k2-6"
        assert cfg["run_config"]["stage_models"] == {"extract": "kimi-k2-6"}
        assert cfg["run_config"]["pipeline_git_sha"] == "deadbeef"


class TestConfigJsonAtomicity:
    """Atomic write: tmp+rename so a SIGKILL mid-write can't leave a
    half-serialized JSON on disk that breaks readers."""

    def test_write_uses_tmp_then_rename(self, tmp_path, monkeypatch):
        # Monkeypatch tmp.replace to capture the staging path so we
        # can prove tmp+rename is the path taken.
        observed: dict = {}
        runner._run_dir = tmp_path
        _seed_run_state()
        orig_replace = Path.replace

        def spy_replace(self, target):
            if self.name.endswith(".tmp"):
                observed["tmp_existed"] = self.exists()
                observed["target_name"] = Path(target).name
            return orig_replace(self, target)

        monkeypatch.setattr(Path, "replace", spy_replace)
        runner._write_run_config_once()
        assert observed.get("tmp_existed") is True
        assert observed.get("target_name") == "config.json"
        # Tmp must not linger after the rename.
        assert not (tmp_path / "config.json.tmp").exists()
        assert (tmp_path / "config.json").exists()


class TestConfigJsonIdempotent:
    """Write-once: a second call must be a no-op so resume cycles don't
    overwrite the original time-zero capture."""

    def test_second_write_does_not_overwrite(self, tmp_path):
        runner._run_dir = tmp_path
        _seed_run_state()
        runner._write_run_config_once()
        original_mtime = (tmp_path / "config.json").stat().st_mtime_ns
        original_text = (tmp_path / "config.json").read_text()
        # Mutate the in-memory state in a way a real second cycle would
        # (e.g. resumed_at, new pid). Then call write again — config.json
        # should be untouched.
        runner._run_state["resumed_at"] = "2026-05-07T13:00:00Z"
        runner._run_state["pid"] = 99999
        # Even if a future reviewer adds provider mid-run, idempotency
        # protects the original.
        runner._run_state["model"] = "different-model-DO-NOT-PERSIST"
        runner._write_run_config_once()
        assert (tmp_path / "config.json").read_text() == original_text
        assert (tmp_path / "config.json").stat().st_mtime_ns == original_mtime

    def test_no_op_when_run_state_unset(self, tmp_path):
        runner._run_dir = tmp_path
        runner._run_state = None
        runner._write_run_config_once()
        assert not (tmp_path / "config.json").exists()

    def test_no_op_when_run_dir_unset(self, tmp_path):
        runner._run_dir = None
        _seed_run_state()
        runner._write_run_config_once()
        # Nothing was written anywhere.
        assert not list(tmp_path.iterdir())
