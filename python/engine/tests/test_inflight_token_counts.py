"""
Per-chunk live-mirror onto the live stat record + one-shot
`stream_progress` JSONL event at first-token-received.

`_consume_chat_stream` updates the live stat record's
`completion_tokens` / `reasoning_tokens` in memory on every content /
reasoning delta (chars/3 estimate, mirrors the post-stream fallback's
math). This rescues partial-byte counts on mid-stream exceptions
(httpx.RemoteProtocolError, ReadError, ReadTimeout, APITimeoutError,
etc.) — the wrapper's failure-finalize captures whatever the rec
carries at the moment of the cut. Without it the rec would carry
completion_tokens=None even though real bytes flowed before the cut.

Separately, a single `stream_progress` JSONL event fires at the
first-token-received moment, carrying TTFT (and TTFR for reasoning
models). The Rust live materializer reads it as a durable "stream
started" signal. No further per-chunk JSONL emission — in-flight
token counts in the per-call UI freeze at the first-token snapshot
until the end event lands.

No live LLM calls. Run with:
    cd engine && pytest tests/test_inflight_token_counts.py -v
"""
from __future__ import annotations

import json
from pathlib import Path





# ── Per-chunk live mirror onto the rec ────────────────────────────────────────


# ── stream_progress JSONL event emission ──────────────────────────────────────


# ── live_tokens stdout heartbeat ──────────────────────────────────────────────


# ── Materializer reads stream_progress ────────────────────────────────────────


class TestMaterializerStreamProgress:
    """`_materialize_calls_from_jsonl` updates the rec's
    completion_tokens / reasoning_tokens / token-timing fields when
    a stream_progress event is in the log. End event takes priority."""

    def _materialize(self, jsonl_path: Path) -> list[dict]:
        from engine.runner import _materialize_calls_from_jsonl
        return _materialize_calls_from_jsonl(
            jsonl_path, ended_at_iso="2026-05-10T12:00:00Z",
        )

    def _write_events(self, p: Path, events: list[dict]) -> None:
        p.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    def test_inflight_rec_carries_progress_count(self, tmp_path):
        # Begin + stream_progress, no end yet. Materializer surfaces
        # the running count. (Rust version is the prod path; the
        # Python materializer mirror is exercised here for parity.)
        p = tmp_path / "llm-calls.jsonl"
        self._write_events(p, [
            {
                "event": "begin", "schema": "llm-calls/v1",
                "call_id": "0001", "stage": "extract",
                "category": "doc-1", "model": "kimi-k2-6",
                "started_at_iso": "2026-05-10T11:30:00Z",
                "attempt": 1, "retry_of_call_id": None,
                "prompt_tokens_est": 14000,
            },
            {
                "event": "stream_progress", "schema": "llm-calls/v1",
                "call_id": "0001",
                "completion_tokens": 1200,
                "reasoning_tokens": 800,
                "ttft_ms": 5000, "ttfr_ms": 1500,
                "last_token_ms": 28000,
            },
        ])
        records = self._materialize(p)
        assert len(records) == 1
        rec = records[0]
        assert rec["completion_tokens"] == 1200
        assert rec["reasoning_tokens"] == 800
        assert rec["last_token_ms"] == 28000

    def test_end_event_overwrites_progress(self, tmp_path):
        # Sequence: begin → stream_progress → end. End's
        # completion_tokens (provider-reported usage, ground truth)
        # overwrites the chars/3 progress estimate.
        p = tmp_path / "llm-calls.jsonl"
        self._write_events(p, [
            {
                "event": "begin", "schema": "llm-calls/v1",
                "call_id": "0001", "stage": "extract",
                "category": "doc-1", "model": "kimi-k2-6",
                "started_at_iso": "2026-05-10T11:30:00Z",
                "attempt": 1, "retry_of_call_id": None,
            },
            {
                "event": "stream_progress", "schema": "llm-calls/v1",
                "call_id": "0001",
                "completion_tokens": 1200,
                "reasoning_tokens": 800,
            },
            {
                "event": "end", "schema": "llm-calls/v1",
                "call_id": "0001",
                "duration_ms": 30000, "success": True, "error": None,
                "prompt_tokens": 14000, "completion_tokens": 1180,
                "reasoning_tokens": 790,
                "finish_reason": "stop",
            },
        ])
        records = self._materialize(p)
        rec = records[0]
        assert rec["completion_tokens"] == 1180  # end wins
        assert rec["reasoning_tokens"] == 790

    def test_content_tokens_surfaces_from_progress_event(self, tmp_path):
        # Reader-side compat: when a historical jsonl carries
        # content_tokens on a stream_progress event, the materializer
        # passes it onto the rec for the UI's "out" column. Current
        # emitter no longer writes the field, but pre-existing logs
        # still parse correctly.
        p = tmp_path / "llm-calls.jsonl"
        self._write_events(p, [
            {
                "event": "begin", "schema": "llm-calls/v1",
                "call_id": "0001", "stage": "extract",
                "category": "doc-1", "model": "kimi-k2-6",
                "started_at_iso": "2026-05-10T11:30:00Z",
                "attempt": 1, "retry_of_call_id": None,
                "prompt_tokens_est": 14000,
            },
            {
                "event": "stream_progress", "schema": "llm-calls/v1",
                "call_id": "0001",
                "completion_tokens": 1200,
                "reasoning_tokens": 800,
                "content_tokens": 400,
                "ttft_ms": 5000, "last_token_ms": 28000,
            },
        ])
        records = self._materialize(p)
        rec = records[0]
        assert rec["content_tokens"] == 400
        assert rec["completion_tokens"] == 1200
        assert rec["reasoning_tokens"] == 800

    def test_old_jsonl_without_progress_works(self, tmp_path):
        # Pre-#237 logs: begin + end only. Materializer behaves as
        # before (backward compat).
        p = tmp_path / "llm-calls.jsonl"
        self._write_events(p, [
            {
                "event": "begin", "schema": "llm-calls/v1",
                "call_id": "0001", "stage": "extract",
                "category": "doc-1", "model": "kimi-k2-6",
                "started_at_iso": "2026-05-10T11:30:00Z",
                "attempt": 1, "retry_of_call_id": None,
            },
            {
                "event": "end", "schema": "llm-calls/v1",
                "call_id": "0001",
                "duration_ms": 1000, "success": True, "error": None,
                "prompt_tokens": 100, "completion_tokens": 50,
                "finish_reason": "stop",
            },
        ])
        records = self._materialize(p)
        assert records[0]["completion_tokens"] == 50
