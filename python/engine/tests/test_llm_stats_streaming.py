"""
Streaming llm-calls.jsonl tests.

llm-calls.jsonl is the append-only NDJSON event log that survives
SIGKILL / segfaults / crashes the rollup file (.json) cant. It's
written by `begin_stat_record` (begin event), `finalize_stat_record`
(end event), and `record_stage_counts` (counts event). This file
verifies:

  - Every begin appends one valid JSON line with event="begin".
  - Every finalize appends one valid JSON line with event="end" that
    references the begin via call_id.
  - record_stage_counts appends a counts event (only when input or
    output is non-None).
  - The .jsonl exists on disk after each event (no buffering).
  - Concurrent finalizes from a threadpool produce N×2 lines (begin
    + end), no torn records (the bytes-interleave failure mode the
    lock prevents).
  - Each line carries `"schema": "llm-calls/v1"` so a single line is
    self-describing.
  - When `set_calls_jsonl_path(None)` (the default), no file is
    created (tests / ad-hoc scripts shouldn't write a sidecar).

No live LLM calls. Run with:
    cd engine && pytest tests/test_llm_stats_streaming.py -v
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


from engine import llm
@pytest.fixture
def jsonl_path(tmp_path):
    """Set up a temp jsonl path, reset stats, and tear down the
    streaming pointer so cross-test pollution can't happen. The lock
    + path are module-globals; tests in the same process share them
    so the cleanup matters."""
    p = tmp_path / "llm-calls.jsonl"
    llm.reset_stat_records()
    llm.set_calls_jsonl_path(p)
    yield p
    llm.set_calls_jsonl_path(None)
    llm.reset_stat_records()


def _read_events(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ── Single-thread streaming ───────────────────────────────────────────────────

class TestSingleThreadStreaming:
    def test_no_path_no_file(self, tmp_path):
        # Default state: jsonl path None → no file written.
        llm.set_calls_jsonl_path(None)
        cid = llm.begin_stat_record("extract", "cat-x", "gpt-oss-120b")
        llm.finalize_stat_record(cid, success=True, duration_ms=100)
        # No file should exist anywhere — the test passes by absence.
        assert not (tmp_path / "llm-calls.jsonl").exists()

    def test_begin_writes_begin_event(self, jsonl_path):
        # begin emits a begin event immediately, before any finalize.
        cid = llm.begin_stat_record("extract", "cat-x", "gpt-oss-120b")
        assert jsonl_path.exists(), "begin should have written a line"
        events = _read_events(jsonl_path)
        assert len(events) == 1
        assert events[0]["event"] == "begin"
        assert events[0]["call_id"] == cid
        assert events[0]["stage"] == "extract"
        assert events[0]["category"] == "cat-x"
        assert events[0]["model"] == "gpt-oss-120b"
        assert events[0]["started_at_iso"]
        assert events[0]["schema"] == "llm-calls/v1"

    def test_finalize_appends_end_event(self, jsonl_path):
        cid = llm.begin_stat_record("extract", "cat-x", "gpt-oss-120b")
        llm.finalize_stat_record(cid, success=True, duration_ms=100)
        events = _read_events(jsonl_path)
        assert len(events) == 2
        assert events[0]["event"] == "begin"
        assert events[1]["event"] == "end"
        assert events[1]["call_id"] == cid
        assert events[1]["success"] is True
        assert events[1]["duration_ms"] == 100
        assert events[1]["error"] is None

    def test_ten_calls_write_twenty_lines(self, jsonl_path):
        for i in range(10):
            cid = llm.begin_stat_record("extract", f"cat-{i}", "gpt-oss-120b")
            llm.finalize_stat_record(cid, success=True, duration_ms=10 + i)
        events = _read_events(jsonl_path)
        assert len(events) == 20
        begins = [e for e in events if e["event"] == "begin"]
        ends = [e for e in events if e["event"] == "end"]
        assert len(begins) == 10 == len(ends)
        assert {e["call_id"] for e in begins} == {e["call_id"] for e in ends}

    def test_failure_record_streams_error_field(self, jsonl_path):
        cid = llm.begin_stat_record("patterns", "topic-x", "kimi-k2-6")
        err = {"class": "RuntimeError", "message": "boom", "traceback": ""}
        llm.finalize_stat_record(
            cid, success=False, duration_ms=50, error=err)
        end = [e for e in _read_events(jsonl_path) if e["event"] == "end"][-1]
        assert end["success"] is False
        assert end["error"]["class"] == "RuntimeError"
        assert end["error"]["message"] == "boom"

    def test_record_stage_counts_emits_counts_event(self, jsonl_path):
        cid = llm.begin_stat_record("extract", "cat-x", "gpt-oss-120b")
        llm.finalize_stat_record(cid, success=True, duration_ms=10)
        llm.record_stage_counts(cid, input={"chars": 1000}, output={"facts": 5})
        events = _read_events(jsonl_path)
        assert [e["event"] for e in events] == ["begin", "end", "counts"]
        counts = events[-1]
        assert counts["call_id"] == cid
        assert counts["input"] == {"chars": 1000}
        assert counts["output"] == {"facts": 5}


# ── Concurrent streaming ──────────────────────────────────────────────────────

class TestConcurrentStreaming:
    def test_sixteen_concurrent_calls(self, jsonl_path):
        """Fire 16 begin+finalize pairs from a threadpool; assert
        exactly 32 valid JSON lines appear in the .jsonl. The lock
        around the append protects against torn records (interleaved
        bytes from two threads' partial json.dumps writes)."""
        N = 16
        cids = []
        for i in range(N):
            cid = llm.begin_stat_record(
                "extract", f"cat-{i}", "gpt-oss-120b")
            cids.append(cid)

        # Concurrent finalize — release a barrier so all threads
        # start writing as close together as possible.
        barrier = threading.Barrier(N)

        def _finalize(call_id: str) -> None:
            barrier.wait()
            llm.finalize_stat_record(call_id, success=True, duration_ms=15)

        with ThreadPoolExecutor(max_workers=N) as ex:
            futs = [ex.submit(_finalize, c) for c in cids]
            for f in futs:
                f.result()

        events = _read_events(jsonl_path)
        # N begins (sequential, before pool) + N ends (concurrent).
        assert len(events) == N * 2
        begins = [e for e in events if e["event"] == "begin"]
        ends = [e for e in events if e["event"] == "end"]
        assert len(begins) == N == len(ends)
        # Every line parses as JSON with no truncation / interleave.
        # Each call_id appears exactly once on each side.
        assert sorted(e["call_id"] for e in begins) == sorted(cids)
        assert sorted(e["call_id"] for e in ends) == sorted(cids)


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestStreamingEdgeCases:
    def test_finalize_without_begin_no_op(self, jsonl_path):
        # finalize_stat_record on a non-existent call_id silently
        # returns. Streaming should not write anything.
        llm.finalize_stat_record("9999", success=True, duration_ms=10)
        # File may not exist (no events ever) or it exists but is empty.
        if jsonl_path.exists():
            assert jsonl_path.read_text() == ""

    def test_double_finalize_writes_two_end_events(self, jsonl_path):
        # Defensive: if a caller calls finalize twice with the same
        # call_id (bug, not expected), two end events appear. Not
        # something to "guard" against — the behavior is documented
        # so consumers can de-dupe by (call_id, event).
        cid = llm.begin_stat_record("extract", "cat-x", "gpt-oss-120b")
        llm.finalize_stat_record(cid, success=True, duration_ms=10)
        llm.finalize_stat_record(cid, success=False, duration_ms=20)
        events = _read_events(jsonl_path)
        # 1 begin + 2 ends.
        assert [e["event"] for e in events] == ["begin", "end", "end"]

    def test_setting_path_none_disables_streaming(self, tmp_path):
        # set_calls_jsonl_path(None) after a previous finalize works
        # correctly — subsequent events don't touch any file.
        p = tmp_path / "llm-calls.jsonl"
        llm.set_calls_jsonl_path(p)
        cid = llm.begin_stat_record("extract", "x", "gpt-oss-120b")
        llm.finalize_stat_record(cid, success=True, duration_ms=10)
        assert p.exists()
        line_count_before = len(p.read_text().splitlines())

        llm.set_calls_jsonl_path(None)
        cid2 = llm.begin_stat_record("extract", "y", "gpt-oss-120b")
        llm.finalize_stat_record(cid2, success=True, duration_ms=10)
        # No new lines written.
        assert len(p.read_text().splitlines()) == line_count_before
