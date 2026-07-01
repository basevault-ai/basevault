"""
Interrupted-on-wire detection.

Sibling of `test_empty_on_wire.py`. Covers the partial-stream-cut
case: bytes flowed but the stream closed cleanly without a
terminating `finish_reason` chunk and below `max_tokens_reserved`.

Trigger conjunction (in `_raise_if_interrupted_on_wire`):
  - `finish_reason is None`
  - `ttft_ms set` OR `last_token_ms set` (bytes flowed)
  - `completion_tokens < max_tokens_reserved` (when cap is known)

Distinct from:
  - empty-on-wire (zero chunks): handled by `_raise_if_empty_on_wire`.
    Strictly disjoint trigger.
  - cap-hit-with-marker (`finish_reason="length"`): routes through
    `_CapHitResponse` to sizing (halve cascade).
  - mid-stream cut that raises (httpx.RemoteProtocolError, etc.):
    handled by `_classify_failure` exception path → load via the
    stream-cut branch.

Routes to load (not sizing) because the common cause of a partial
cut without a finish_reason marker is a transport blip, not size.
Halving on a transport blip multiplies spend without addressing the
root cause.

No live LLM calls. Run with:
    cd engine && pytest tests/test_interrupted_on_wire.py -v
"""
from __future__ import annotations

import json
from pathlib import Path



from engine import llm


# ── Unit tests for _raise_if_interrupted_on_wire ──────────────────────────────


# ── Partial-stream completion-token estimate ──────────────────────────────────


class TestPartialStreamCountsCompletionTokens:
    """When the usage chunk never arrives (interrupted stream OR
    provider omitted include-usage), `_consume_chat_stream` falls
    back to estimating completion_tokens from the streamed content +
    reasoning text. The per-call detail UI shows real numbers for
    partial streams instead of `0 out`."""

    def setup_method(self):
        llm.reset_stat_records()
        self._prev_stage = llm._current_stage

    def teardown_method(self):
        llm.set_stage(self._prev_stage)


# ── record_stage_counts threading ─────────────────────────────────────────────


class TestRecordStageCountsInterruptedFlag:
    """`record_stage_counts(failure_kind="interrupted")` stamps the
    rec field and emits the counts event with the flag set. Wrapper
    failure-finalize routes `_InterruptedResponse` through this
    path."""

    def setup_method(self):
        llm.reset_stat_records()

    def test_stamps_interrupted_on_rec(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        llm.set_calls_jsonl_path(p)
        try:
            cid = llm.begin_stat_record("extract", "cat-x", "kimi-k2-6")
            llm.finalize_stat_record(cid, success=False, duration_ms=100)
            llm.record_stage_counts(cid, failure_kind="interrupted")
        finally:
            llm.set_calls_jsonl_path(None)
        rec = llm.get_stat_records()[-1]
        assert rec.get("interrupted") is True

    def test_emits_counts_event_with_interrupted_flag(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        llm.set_calls_jsonl_path(p)
        try:
            cid = llm.begin_stat_record("extract", "cat-x", "kimi-k2-6")
            llm.finalize_stat_record(cid, success=False, duration_ms=100)
            llm.record_stage_counts(cid, failure_kind="interrupted")
        finally:
            llm.set_calls_jsonl_path(None)
        events = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        counts = next(e for e in events if e["event"] == "counts")
        assert counts["interrupted"] is True


# ── Materializer ──────────────────────────────────────────────────────────────


class TestMaterializerInterruptedFlag:
    """Materializer initializes `interrupted: False` on every record
    and reads the flag from counts events. Same shape as
    `parse_error` / `empty_response`."""

    def _materialize(self, jsonl_path: Path) -> list[dict]:
        from engine.runner import _materialize_calls_from_jsonl
        return _materialize_calls_from_jsonl(
            jsonl_path, ended_at_iso="2026-05-10T05:00:00Z",
        )

    def _write_events(self, p: Path, events: list[dict]) -> None:
        p.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    def test_initializes_interrupted_false(self, tmp_path):
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
                "duration_ms": 100, "success": True, "error": None,
                "prompt_tokens": 100, "completion_tokens": 50,
                "finish_reason": "stop",
            },
        ])
        records = self._materialize(p)
        assert records[0]["interrupted"] is False

    def test_picks_up_interrupted_from_counts_event(self, tmp_path):
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
                "duration_ms": 100, "success": False,
                "error": {"class": "retry._InterruptedResponse",
                          "message": "interrupted (stage=extract)"},
                "prompt_tokens": 14000, "completion_tokens": 0,
                "finish_reason": None,
            },
            {
                "event": "counts", "schema": "llm-calls/v1",
                "call_id": "0001",
                "input": None, "output": None,
                "parse_error": None, "empty_response": None,
                "interrupted": True,
            },
        ])
        records = self._materialize(p)
        assert records[0]["interrupted"] is True

    def test_outcome_classifier_buckets_interrupted(self, tmp_path):
        # End-to-end: an interrupted-flagged failed record routes
        # through `_classify_outcome` to an OUTCOME_INTERRUPTED_*
        # bucket. With no ttft_ms on the rec, the TTFT-fraction
        # discriminator returns "load" (silent interrupt).
        from engine.runner import _classify_outcome, OUTCOME_INTERRUPTED_LOAD
        rec = {
            "call_id": "0001",
            "stage": "extract",
            "success": False,
            "error": {"class": "retry._InterruptedResponse",
                      "message": "interrupted (stage=extract)"},
            "interrupted": True,
            "aborted": False,
            "finish_reason": None,
        }
        assert _classify_outcome(rec) == OUTCOME_INTERRUPTED_LOAD
