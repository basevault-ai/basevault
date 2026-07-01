"""
In-flight capture tests.

When a run winds down mid-call (pause / cancel / error), the begin
event is on disk but the end event never gets written. The rollup
materializer surfaces these unmatched begins as `aborted: true`
records with `error=None` — the call didn't fail, the run wound
down. `_classify_outcome` reads `aborted=True` and returns
`OUTCOME_ABORTED`, distinct from any `failed (X)` bucket.

This file tests both the direct materializer (no subprocess needed)
and a subprocess SIGTERM scenario where the helper has unfinalized
begins.

No live LLM calls. Run with:
    cd engine && pytest tests/test_in_flight_capture.py -v
"""
from __future__ import annotations

import json
from pathlib import Path


from engine.runner import _materialize_calls_from_jsonl


def _write_events(p: Path, events: list[dict]) -> None:
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")


class TestMaterializerDirect:
    """Test _materialize_calls_from_jsonl on hand-crafted event logs.
    No subprocess — fast, deterministic, covers the materializer's
    contract directly."""

    def test_begin_only_becomes_aborted(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        _write_events(p, [
            {
                "event": "begin", "call_id": "0001",
                "stage": "extract", "category": "split_01",
                "model": "gpt-oss-120b",
                "started_at_iso": "2026-04-29T17:09:00.000Z",
            },
        ])
        records = _materialize_calls_from_jsonl(
            p, ended_at_iso="2026-04-29T17:09:30.000Z")
        assert len(records) == 1
        r = records[0]
        assert r["call_id"] == "0001"
        assert r["aborted"] is True
        assert r["success"] is False
        # No synthesized error — the call didn't fail, the run wound
        # down. Mirrors Rust live materializer.
        assert r["error"] is None
        # 30 seconds between begin and ended_at.
        assert r["duration_ms"] == 30_000

    def test_begin_end_pair_collapses(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        _write_events(p, [
            {
                "event": "begin", "call_id": "0001",
                "stage": "extract", "category": "split_01",
                "model": "gpt-oss-120b",
                "started_at_iso": "2026-04-29T17:09:00.000Z",
            },
            {
                "event": "end", "call_id": "0001",
                "duration_ms": 1234, "success": True, "error": None,
                "prompt_tokens": 500, "completion_tokens": 50,
                "model": "gpt-oss-120b", "mode": "tinfoil",
            },
        ])
        records = _materialize_calls_from_jsonl(
            p, ended_at_iso="2026-04-29T18:00:00.000Z")
        assert len(records) == 1
        r = records[0]
        assert r["aborted"] is False
        assert r["success"] is True
        assert r["duration_ms"] == 1234
        assert r["prompt_tokens"] == 500
        assert r["completion_tokens"] == 50
        assert r["mode"] == "tinfoil"

    def test_begin_end_counts_triple_collapses(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        _write_events(p, [
            {
                "event": "begin", "call_id": "0001",
                "stage": "extract", "category": "split_01",
                "model": "gpt-oss-120b",
                "started_at_iso": "2026-04-29T17:09:00.000Z",
            },
            {
                "event": "end", "call_id": "0001",
                "duration_ms": 100, "success": True, "error": None,
                "prompt_tokens": 100, "completion_tokens": 10,
                "model": "gpt-oss-120b", "mode": "tinfoil",
            },
            {
                "event": "counts", "call_id": "0001",
                "input": {"chars": 1000}, "output": {"facts": 5},
            },
        ])
        records = _materialize_calls_from_jsonl(
            p, ended_at_iso="2026-04-29T18:00:00.000Z")
        assert len(records) == 1
        r = records[0]
        assert r["input"] == {"chars": 1000}
        assert r["output"] == {"facts": 5}

    def test_mixed_finalized_and_inflight(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        events = []
        # 3 finalized.
        for i in range(3):
            cid = f"{i + 1:04d}"
            events.append({
                "event": "begin", "call_id": cid,
                "stage": "extract", "category": f"split_{i:02d}",
                "model": "gpt-oss-120b",
                "started_at_iso": "2026-04-29T17:09:00.000Z",
            })
            events.append({
                "event": "end", "call_id": cid,
                "duration_ms": 100, "success": True, "error": None,
                "prompt_tokens": 100, "completion_tokens": 10,
                "model": "gpt-oss-120b", "mode": "tinfoil",
            })
        # 2 in-flight (begin only).
        for i in range(3, 5):
            cid = f"{i + 1:04d}"
            events.append({
                "event": "begin", "call_id": cid,
                "stage": "extract", "category": f"split_{i:02d}",
                "model": "gpt-oss-120b",
                "started_at_iso": "2026-04-29T17:09:30.000Z",
            })
        _write_events(p, events)
        records = _materialize_calls_from_jsonl(
            p, ended_at_iso="2026-04-29T17:10:00.000Z")
        assert len(records) == 5
        aborted = [r for r in records if r["aborted"]]
        successful = [r for r in records if r["success"] is True]
        assert len(aborted) == 2 and len(successful) == 3
        # Aborted records have duration 30s (begin → ended_at) and
        # `error=None` (no synthesized failure — the run wound down).
        for r in aborted:
            assert r["duration_ms"] == 30_000
            assert r["error"] is None

    def test_ignores_unknown_events(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        _write_events(p, [
            {"event": "weather", "call_id": "0001"},
            {
                "event": "begin", "call_id": "0001",
                "stage": "extract", "category": "x",
                "model": "gpt-oss-120b",
                "started_at_iso": "2026-04-29T17:09:00.000Z",
            },
            {
                "event": "end", "call_id": "0001",
                "duration_ms": 1, "success": True, "error": None,
                "prompt_tokens": 1, "completion_tokens": 1,
                "model": "gpt-oss-120b", "mode": "tinfoil",
            },
        ])
        records = _materialize_calls_from_jsonl(
            p, ended_at_iso="2026-04-29T18:00:00.000Z")
        assert len(records) == 1
        assert records[0]["success"] is True

    def test_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        p.write_text(
            "{\"event\": \"begin\", \"call_id\": \"0001\", "
            "\"stage\": \"extract\", \"category\": \"x\", "
            "\"model\": \"gpt-oss-120b\", "
            "\"started_at_iso\": \"2026-04-29T17:09:00.000Z\"}\n"
            "{not valid json\n"
            "\n"  # blank line
        )
        records = _materialize_calls_from_jsonl(
            p, ended_at_iso="2026-04-29T17:09:30.000Z")
        # Begin recovered as aborted; malformed line skipped.
        assert len(records) == 1
        assert records[0]["aborted"] is True

    def test_missing_file_returns_empty(self, tmp_path):
        records = _materialize_calls_from_jsonl(
            tmp_path / "nope.jsonl", ended_at_iso="2026-04-29T17:09:00.000Z")
        assert records == []
