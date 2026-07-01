"""Tests for the on-demand rollup function (issue #189).

`materialize_run_stats(jsonl_path, config_path)` is the single source
of truth for the llm-stats/v1 payload — same logic running and
finished, no on-disk cache, derived from the event log + config
snapshot. These tests pin:

  - The function returns the documented schema even with no jsonl /
    no config (defensive defaults for ad-hoc readers).
  - cycles_count is read from the on-disk jsonl, not module globals,
    so eval/judge.py and ad-hoc scripts get the right value.
  - The mode field comes from config.json when no override is passed.
  - Legacy llm-stats.json on disk is NOT auto-read by this function —
    that fallback lives in the Tauri reader path. The Python rollup
    is always derived (or empty when the jsonl is missing).

Run with:
    cd engine && pytest tests/test_materialize_run_stats.py -v
"""
from __future__ import annotations

import json
from pathlib import Path


from engine.runner import materialize_run_stats


def _write_jsonl(p: Path, events: list[dict]) -> None:
    p.write_text("".join(json.dumps(e) + "\n" for e in events))


def _write_config(p: Path, **overrides) -> None:
    cfg = {
        "run_id": "test-run",
        "short_id": "abcd",
        "agent": "app",
        "mode": "test",
        "model": "fixture",
        "created_at": "2026-05-08T00:00:00.000Z",
    }
    cfg.update(overrides)
    p.write_text(json.dumps(cfg))


def test_empty_jsonl_returns_well_formed_payload(tmp_path):
    """A run dir with no llm-calls.jsonl still produces a usable
    payload: empty calls, all counters zeroed, schema stamped. The
    UI's modal renders 'no calls' against this rather than crashing."""
    jsonl = tmp_path / "llm-calls.jsonl"  # absent
    payload = materialize_run_stats(jsonl)

    assert payload["schema"] == "llm-stats/v1"
    assert payload["calls"] == []
    assert payload["totals"]["calls"] == 0
    assert payload["totals"]["successful"] == 0
    assert payload["per_stage"] == {}
    assert payload["by_stage"] == {}
    assert payload["cycles_count"] == 1


def test_metadata_threaded_from_config_json(tmp_path):
    """run_id / short_id / agent / mode / primary_model /
    started_at_iso are pulled from config.json when the caller doesn't
    pass overrides. (session_id / eval_id were dropped in the flat
    run-dir layout — #194.)"""
    jsonl = tmp_path / "llm-calls.jsonl"
    _write_jsonl(jsonl, [])
    config = tmp_path / "config.json"
    _write_config(config, run_id="my-run-id", short_id="zZz9",
                  agent="experiment",
                  mode="local", model="kimi-k2-6",
                  created_at="2026-05-08T01:23:45.000Z")
    payload = materialize_run_stats(jsonl, config)
    assert payload["run_id"] == "my-run-id"
    assert payload["short_id"] == "zZz9"
    assert payload["agent"] == "experiment"
    assert payload["mode"] == "local"
    assert payload["primary_model"] == "kimi-k2-6"
    assert payload["started_at_iso"] == "2026-05-08T01:23:45.000Z"


def test_metadata_kwarg_overrides_win_over_config(tmp_path):
    """End-of-run callsite passes primary_model + started_at_iso that
    may not be in config.json. Kwargs win when both are present."""
    jsonl = tmp_path / "llm-calls.jsonl"
    _write_jsonl(jsonl, [])
    config = tmp_path / "config.json"
    _write_config(config, model="config-model", created_at="2026-05-08T00:00Z")
    payload = materialize_run_stats(
        jsonl, config,
        primary_model="kwarg-model",
        started_at_iso="2026-05-08T12:34:56.000Z",
    )
    assert payload["primary_model"] == "kwarg-model"
    assert payload["started_at_iso"] == "2026-05-08T12:34:56.000Z"


def test_cycles_count_reads_from_jsonl_not_globals(tmp_path):
    """`cycles_count` is derived by counting `cycle_start` events on
    the passed jsonl path. Ad-hoc readers (eval/judge.py running
    against a foreign run dir) get the right value without setting
    module globals."""
    jsonl = tmp_path / "llm-calls.jsonl"
    _write_jsonl(jsonl, [
        {"event": "cycle_start", "ts": "2026-05-08T00:00:00Z"},
        {"event": "cycle_start", "ts": "2026-05-08T01:00:00Z"},
        {"event": "cycle_start", "ts": "2026-05-08T02:00:00Z"},
    ])
    payload = materialize_run_stats(jsonl)
    assert payload["cycles_count"] == 3


def test_warnings_block_is_leaf_aware(tmp_path):
    """Warnings counts come from leaf outcomes only — chains that
    recovered via halving don't inflate counts. Mirrors the issue's
    acceptance for chain-leaf classification inside the rollup."""
    jsonl = tmp_path / "llm-calls.jsonl"
    # Chain: timeout (parent) → success (leaf)
    _write_jsonl(jsonl, [
        {"event": "begin", "call_id": "0001", "stage": "extract",
         "category": "split_00", "model": "fixture",
         "started_at_iso": "2026-05-08T00:00:00.000Z"},
        {"event": "end", "call_id": "0001",
         "duration_ms": 1000, "success": False,
         "error": {"class": "openai.APITimeoutError", "message": "t"}},
        {"event": "begin", "call_id": "0002", "stage": "extract",
         "category": "split_00", "model": "fixture",
         "retry_of_call_id": "0001",
         "started_at_iso": "2026-05-08T00:00:01.000Z"},
        {"event": "end", "call_id": "0002",
         "duration_ms": 500, "success": True, "error": None,
         "prompt_tokens": 100, "completion_tokens": 50},
    ])
    payload = materialize_run_stats(jsonl)
    # Leaf is 0002 (success) → zero leaf timeouts.
    assert payload["warnings"]["timeouts"] == 0


def test_input_overflows_zero_without_call_warnings(tmp_path):
    """Without an in-process `call_warnings` list, the rollup carries
    `input_overflows = 0` — the warning is in-memory only and isn't
    persisted to the jsonl. Tauri / eval/judge readers see this
    behavior; only the end-of-run writer sees a non-zero value."""
    jsonl = tmp_path / "llm-calls.jsonl"
    _write_jsonl(jsonl, [])
    payload = materialize_run_stats(jsonl)
    assert payload["warnings"]["input_overflows"] == 0
    assert payload["warnings"]["empty_responses"] == 0


def test_call_warnings_kwarg_filters_by_leaf_call_id(tmp_path):
    """When the end-of-run writer threads `call_warnings`, only those
    stamped with a leaf call_id contribute to `input_overflows`. A
    parent attempt that overflowed and was halved produces a non-leaf
    warning that should NOT count."""
    jsonl = tmp_path / "llm-calls.jsonl"
    # Chain: parent overflow → halved leaf success.
    _write_jsonl(jsonl, [
        {"event": "begin", "call_id": "0001", "stage": "extract",
         "category": "split_00", "model": "fixture",
         "started_at_iso": "2026-05-08T00:00:00.000Z"},
        {"event": "end", "call_id": "0001",
         "duration_ms": 1000, "success": False,
         "error": {"class": "InputOverflow", "message": "too big"}},
        {"event": "begin", "call_id": "0002", "stage": "extract",
         "category": "split_00/half-1", "model": "fixture",
         "retry_of_call_id": "0001",
         "started_at_iso": "2026-05-08T00:00:01.000Z"},
        {"event": "end", "call_id": "0002",
         "duration_ms": 500, "success": True, "error": None,
         "prompt_tokens": 100, "completion_tokens": 50},
    ])
    # Two warnings emitted at run time: one for the parent overflow
    # (call_id=0001), one for the leaf overflow (call_id=0002). Only
    # the leaf one should land in the badge.
    payload = materialize_run_stats(
        jsonl,
        call_warnings=[
            {"kind": "input_overflow", "call_id": "0001"},  # parent
            {"kind": "input_overflow", "call_id": "0002"},  # leaf
        ],
    )
    assert payload["warnings"]["input_overflows"] == 1


def test_legacy_llm_stats_json_is_NOT_read_here(tmp_path):
    """The Python rollup function does NOT auto-read a legacy
    llm-stats.json file. That fallback lives in the Tauri reader path
    (lib.rs `overlay_rollup_warnings_and_cache` / `read_run_llm_stats`)
    so on-disk files are only used when the event log is genuinely
    missing. Keeps the Python function pure + deterministic."""
    legacy = tmp_path / "llm-stats.json"
    legacy.write_text(json.dumps({"schema": "llm-stats/v1",
                                   "totals": {"calls": 99}}))
    jsonl = tmp_path / "llm-calls.jsonl"  # absent
    payload = materialize_run_stats(jsonl)
    # Fresh derivation, NOT 99 → the legacy file was correctly ignored.
    assert payload["totals"]["calls"] == 0


def test_rollup_calls_carry_outcome_field(tmp_path):
    """Every materialized call carries an `outcome` (issue #104).
    Same code path running and finished — the live banner sees the
    same outcomes the post-rollup view does."""
    jsonl = tmp_path / "llm-calls.jsonl"
    _write_jsonl(jsonl, [
        {"event": "begin", "call_id": "0001", "stage": "extract",
         "category": "doc-1", "model": "fixture",
         "started_at_iso": "2026-05-08T00:00:00.000Z"},
        {"event": "end", "call_id": "0001",
         "duration_ms": 100, "success": True, "error": None,
         "prompt_tokens": 50, "completion_tokens": 20,
         "finish_reason": "stop"},
    ])
    payload = materialize_run_stats(jsonl)
    assert payload["calls"][0]["outcome"] == "success"
