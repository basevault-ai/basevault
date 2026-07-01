"""Tests for per-call materialization + per-stage rollup shape.

`_materialize_calls_from_jsonl` threads `output`, `cache_key`, and
`parse_error` from the on-disk events onto the materialized record;
`materialize_run_stats` (issue #189) produces the rollup dict the UI
consumes. These tests pin the shape directly against synthetic jsonl,
so the assertions don't need a full pipeline run.

Run with:
    cd engine && pytest tests/test_per_call_rollup.py -v
"""
from __future__ import annotations

import json
from pathlib import Path


from engine.runner import _materialize_calls_from_jsonl, materialize_run_stats
def _write_events(p: Path, events: list[dict]) -> None:
    """Write a synthetic llm-calls.jsonl by line. Each event is a dict
    with at minimum `event` + `call_id`."""
    with p.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_per_stage_rollup_shape(tmp_path):
    """End-to-end: materializer + rollup function exposes per-stage
    outcomes correctly. The rollup carries no `dropped_total` field
    anywhere — per_stage, totals, warnings."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    jsonl = out_dir / "llm-calls.jsonl"
    events: list[dict] = []
    cases = [
        ("0001", {"patterns": 2}),
        ("0002", {"patterns": 0}),
        ("0003", {"patterns": 0}),
    ]
    for cid, out in cases:
        events.append({"event": "begin", "call_id": cid, "stage": "patterns",
                       "category": f"topic-{cid}", "model": "fixture",
                       "started_at_iso": "2026-05-01T00:00:00.000Z",
                       "attempt": 1, "retry_of_call_id": None,
                       "budget": None, "template_hash": "abc123abcdef"})
        events.append({"event": "end", "call_id": cid, "duration_ms": 1000,
                       "success": True, "error": None,
                       "prompt_tokens": 1000, "completion_tokens": 200,
                       "model": "fixture", "mode": "test", "cached": False})
        events.append({"event": "counts", "call_id": cid,
                       "input": {"facts": 10}, "output": out})
    _write_events(jsonl, events)

    data = materialize_run_stats(jsonl)

    # No dropped_total surface anywhere in the rollup.
    pat = data["per_stage"]["patterns"]
    assert "dropped_total" not in pat
    assert "dropped_total" not in data["totals"]
    assert "dropped_total" not in (data.get("warnings") or {})

    # Outcome buckets: 1 success (patterns=2), 2 success_empty (patterns=0).
    assert pat["outcomes"]["success"] == 1
    assert pat["outcomes"]["success_empty"] == 2

    # Each call carries the outcome field.
    for rec in data["calls"]:
        assert "outcome" in rec
        assert rec["outcome"] in (
            "success", "success_empty", "blanked",
            "parse_error", "timeout", "other_failure",
        )
        # Template hash threaded through.
        assert rec["template_hash"] == "abc123abcdef"


def test_cache_key_threaded_through_end_event_to_record(tmp_path):
    """The materializer must pull `cache_key` from the end event onto
    the materialized record. Without this thread, llm-stats.json's
    calls[] entries would all carry cache_key=null even when
    llm.complete stamped the live record at lookup time — the on-disk
    UI's per-call cache-bust button would never show. Regression test
    for the bug observed when the end-event handler in
    `_materialize_calls_from_jsonl` was missing the `cache_key`
    fan-in (read of `cached` was there, read of `cache_key` was not)."""
    p = tmp_path / "llm-calls.jsonl"
    _write_events(p, [
        {"event": "begin", "call_id": "0001", "stage": "patterns",
         "category": "topic", "model": "fixture",
         "started_at_iso": "2026-05-01T00:00:00.000Z", "attempt": 1,
         "retry_of_call_id": None, "budget": None, "template_hash": None},
        {"event": "end", "call_id": "0001", "duration_ms": 1000,
         "success": True, "error": None,
         "prompt_tokens": 1000, "completion_tokens": 200,
         "model": "fixture", "mode": "test", "cached": True,
         "cache_key": "abc123def456789"},
    ])
    recs = _materialize_calls_from_jsonl(p, "2026-05-01T00:01:00Z")
    assert len(recs) == 1
    assert recs[0]["cache_key"] == "abc123def456789"
    assert recs[0]["cached"] is True


def test_parse_error_threaded_through_materializer(tmp_path):
    """Counts event with parse_error=True flips the call's parse_error
    field; the classifier later buckets it as parse_error outcome."""
    p = tmp_path / "llm-calls.jsonl"
    _write_events(p, [
        {"event": "begin", "call_id": "0001", "stage": "patterns",
         "category": "topic", "model": "fixture",
         "started_at_iso": "2026-05-01T00:00:00.000Z", "attempt": 1,
         "retry_of_call_id": None, "budget": None, "template_hash": None},
        {"event": "end", "call_id": "0001", "duration_ms": 1000,
         "success": True, "error": None,
         "prompt_tokens": 1000, "completion_tokens": 200,
         "model": "fixture", "mode": "test", "cached": False},
        {"event": "counts", "call_id": "0001",
         "input": None, "output": None, "parse_error": True},
    ])
    recs = _materialize_calls_from_jsonl(p, "2026-05-01T00:01:00Z")
    assert recs[0]["parse_error"] is True
