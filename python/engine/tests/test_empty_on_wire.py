"""
Empty-on-wire detection + prompt_tokens estimate stamping (issue #225).

Two related fixes:

  Bug 1 — empty-on-wire: a stream that completed with zero chunks
    (no content delta, no usage chunk, no finish_reason — typically a
    clean mid-stream connection drop on the provider side) used to
    fall through complete() as a successful empty response. For
    extract specifically, the wrapper has parser=None and parsing
    happens in stage code that treats empty raw as "0 facts in this
    chunk." Other stages caught the case via their parser raising
    `_EmptyResponse` on whitespace-only raw, but extract did not.
    The fix raises `_EmptyResponse` inside complete()
    — one detection point covering every stage uniformly.

  Bug 2 — prompt_tokens stamping: pre-fix, begin_stat_record set
    prompt_tokens=None and _record_usage unconditionally overwrote
    with the response's value. On a wire-cut response the value was
    0, so the run-details UI showed `0 in / 0 out` for a call that
    genuinely sent thousands of input tokens. The fix stamps a
    pre-flight estimate at begin time and protects it from being
    clobbered by 0/None on the way down (in _record_usage, in the
    end event payload, and in the offline rollup materializer).

No live LLM calls. Run with:
    cd engine && pytest tests/test_empty_on_wire.py -v
"""
from __future__ import annotations

import json
from pathlib import Path



from engine import llm


# ── Bug 1: _raise_if_empty_on_wire ────────────────────────────────────────────


# ── Empty-stream stub (Tinfoil-style mid-stream connection drop) ──────────────


# ── prompt_tokens estimate stamping ──────────────────────────────────────────


class TestPromptTokensEstimateStamping:
    """begin_stat_record carries the caller's pre-flight prompt-token
    estimate; _record_usage and the end-event serializer never let a
    falsy provider value clobber it."""

    def setup_method(self):
        llm.reset_stat_records()

    def test_begin_with_estimate_stamps_rec(self):
        llm.begin_stat_record(
            "extract", "cat-x", "kimi-k2-6",
            prompt_tokens_est=14000,
        )
        rec = llm.get_stat_records()[-1]
        assert rec["prompt_tokens"] == 14000

    def test_begin_without_estimate_stamps_none(self):
        # Backward-compatible default — pre-fix callers (tests, ad-hoc
        # scripts, the vision path which has its own helper) keep
        # working without the new kwarg.
        llm.begin_stat_record("extract", "cat-x", "kimi-k2-6")
        rec = llm.get_stat_records()[-1]
        assert rec["prompt_tokens"] is None

    def test_record_usage_zero_does_not_overwrite_estimate(self):
        # The wire-cut shape: usage chunk never arrived, _record_usage
        # is called with prompt_tokens=0. Estimate must survive.
        cid = llm.begin_stat_record(
            "extract", "cat-x", "kimi-k2-6",
            prompt_tokens_est=14000,
        )
        llm._record_usage(
            "tinfoil", "kimi-k2-6", prompt_tokens=0, completion_tokens=0,
            finish_reason=None, ttft_ms=None, last_token_ms=None, call_id=cid,
        )
        rec = llm.get_stat_records()[-1]
        assert rec["prompt_tokens"] == 14000  # estimate intact

    def test_record_usage_truthy_overwrites_estimate(self):
        # When the provider's usage chunk arrives, its number is
        # ground truth. Estimate is just a fallback.
        cid = llm.begin_stat_record(
            "extract", "cat-x", "kimi-k2-6",
            prompt_tokens_est=14000,
        )
        llm._record_usage(
            "tinfoil", "kimi-k2-6", prompt_tokens=13957, completion_tokens=420,
            finish_reason="stop", ttft_ms=100, last_token_ms=8000, call_id=cid,
        )
        rec = llm.get_stat_records()[-1]
        assert rec["prompt_tokens"] == 13957  # provider value wins

    def test_end_event_carries_estimate_on_wire_cut(self, tmp_path):
        # Full event-log integration: a wire-cut call's end event
        # carries the begin-time estimate, not 0.
        p = tmp_path / "llm-calls.jsonl"
        llm.set_calls_jsonl_path(p)
        try:
            cid = llm.begin_stat_record(
                "extract", "cat-x", "kimi-k2-6",
                prompt_tokens_est=14000,
            )
            llm._record_usage(
                "tinfoil", "kimi-k2-6", prompt_tokens=0, completion_tokens=0,
                finish_reason=None, ttft_ms=None, last_token_ms=None, call_id=cid,
            )
            llm.finalize_stat_record(cid, success=False, duration_ms=1412000)
        finally:
            llm.set_calls_jsonl_path(None)
        events = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        begin = next(e for e in events if e["event"] == "begin")
        end = next(e for e in events if e["event"] == "end")
        assert begin["prompt_tokens_est"] == 14000
        assert end["prompt_tokens"] == 14000  # estimate, not 0

    def test_begin_event_omits_estimate_field_when_unset(self, tmp_path):
        # Callers without an estimate (vision, tests, ad-hoc scripts)
        # don't pollute the event with a null field — keeps older
        # readers backward-compatible.
        p = tmp_path / "llm-calls.jsonl"
        llm.set_calls_jsonl_path(p)
        try:
            llm.begin_stat_record("extract", "cat-x", "kimi-k2-6")
        finally:
            llm.set_calls_jsonl_path(None)
        events = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        begin = events[0]
        assert "prompt_tokens_est" not in begin


# ── Bug 2: rollup materializer respects begin's estimate ──────────────────────


class TestMaterializerPreservesEstimate:
    """The offline rollup materializer reads the .jsonl event log and
    rebuilds per-call records. End event's prompt_tokens overwrites
    begin's estimate ONLY when truthy — a 0 must not clobber it."""

    def _materialize(self, jsonl_path: Path) -> list[dict]:
        # Local import: runner.py is the materializer's home and it
        # imports a pile of pipeline modules; only pull it in when
        # actually needed.
        from engine.runner import _materialize_calls_from_jsonl
        return _materialize_calls_from_jsonl(
            jsonl_path, ended_at_iso="2026-05-10T05:00:00Z",
        )

    def _write_events(self, p: Path, events: list[dict]) -> None:
        p.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    def test_begin_estimate_seeds_rec_when_end_is_zero(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        self._write_events(p, [
            {
                "event": "begin", "schema": "llm-calls/v1",
                "call_id": "0001", "stage": "extract",
                "category": "cat-x", "model": "kimi-k2-6",
                "started_at_iso": "2026-05-10T04:00:29.849Z",
                "attempt": 1, "retry_of_call_id": None,
                "prompt_tokens_est": 14000,
            },
            {
                "event": "end", "schema": "llm-calls/v1",
                "call_id": "0001",
                "duration_ms": 1412000,
                "success": True,
                "error": None,
                "prompt_tokens": 0,        # wire-cut: provider reported nothing
                "completion_tokens": 0,
                "finish_reason": None,
            },
        ])
        records = self._materialize(p)
        assert len(records) == 1
        assert records[0]["prompt_tokens"] == 14000  # estimate preserved

    def test_truthy_end_overwrites_begin_estimate(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        self._write_events(p, [
            {
                "event": "begin", "schema": "llm-calls/v1",
                "call_id": "0001", "stage": "extract",
                "category": "cat-x", "model": "kimi-k2-6",
                "started_at_iso": "2026-05-10T04:00:29.849Z",
                "attempt": 1, "retry_of_call_id": None,
                "prompt_tokens_est": 14000,
            },
            {
                "event": "end", "schema": "llm-calls/v1",
                "call_id": "0001",
                "duration_ms": 8000,
                "success": True,
                "error": None,
                "prompt_tokens": 13957,    # provider's usage chunk arrived
                "completion_tokens": 420,
                "finish_reason": "stop",
            },
        ])
        records = self._materialize(p)
        assert records[0]["prompt_tokens"] == 13957  # provider value wins

    def test_old_jsonl_without_estimate_falls_back_to_end_value(self, tmp_path):
        # Pre-this-fix log files have no `prompt_tokens_est` field on
        # the begin event. Materializer must still produce a sensible
        # record — falls back to the end event's value (None or 0).
        p = tmp_path / "llm-calls.jsonl"
        self._write_events(p, [
            {
                "event": "begin", "schema": "llm-calls/v1",
                "call_id": "0001", "stage": "extract",
                "category": "cat-x", "model": "kimi-k2-6",
                "started_at_iso": "2026-05-10T04:00:29.849Z",
                "attempt": 1, "retry_of_call_id": None,
            },
            {
                "event": "end", "schema": "llm-calls/v1",
                "call_id": "0001",
                "duration_ms": 8000,
                "success": True,
                "error": None,
                "prompt_tokens": 13957,
                "completion_tokens": 420,
                "finish_reason": "stop",
            },
        ])
        records = self._materialize(p)
        assert records[0]["prompt_tokens"] == 13957
