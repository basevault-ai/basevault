"""Issue #52: cycle markers in llm-calls.jsonl + cycles_count in the
materialized rollup.

Multi-cycle runs (cancel+resume, replay-validation) make the on-disk
event log span multiple resume cycles. jsonl gets `cycle_start` /
`cycle_end` boundary markers so consumers can split per-cycle, and
the rollup carries `cycles_count` so single-number consumers can
detect that they should split.

These tests cover the helpers in isolation (no full pipeline run).

Run with:
    cd engine && pytest tests/test_cycle_markers.py -v
"""
from __future__ import annotations

import json
import re
from pathlib import Path


from engine import llm  # noqa: E402
from engine import runner  # noqa: E402


def _read_events(jsonl_path: Path) -> list[dict]:
    """Parse the jsonl into a list of event dicts. Empty lines skipped."""
    events: list[dict] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


# ── _iso_z emit format pin (#261) ───────────────────────────────────────


class TestIsoZEmitFormat:
    """Pin the on-disk shape of cycle event `ts` fields.

    The Rust runs-list materializer derives a finished run's total
    duration from `cycle_start.ts` → terminator-ts via `iso_delta_ms`.
    `iso_delta_ms` accepts both colon and dashes time portions, but
    the parser path was already silently broken once for the dashes
    form (#200, #261) — the colon-form fixture masked it. Pinning the
    runner's emit format here means a stealth flip from `_iso_z()`
    (dashes) to `_iso_z_full()` (colons), or any change to the format
    string itself, fails this test instead of going user-visible.
    """

    def test_iso_z_uses_dashes_in_time_portion(self):
        s = runner._iso_z()
        # YYYY-MM-DDTHH-MM-SSZ — 20 chars, dashes only, no colons.
        assert len(s) == 20, f"unexpected len for {s!r}"
        assert s.endswith("Z")
        assert ":" not in s, "colon in _iso_z() output would break dir-name use"
        date_part, time_part = s[:-1].split("T")
        # Date keeps dashes; time-portion separators must also be dashes.
        assert date_part.count("-") == 2
        assert time_part.count("-") == 2

    def test_iso_z_full_keeps_colons_in_time_portion(self):
        s = runner._iso_z_full()
        assert s.endswith("Z")
        _, time_part = s[:-1].split("T")
        assert time_part.count(":") == 2, (
            f"_iso_z_full must keep colons for ISO-8601 fields, got {s!r}"
        )


# ── emit_cycle_event + count_cycle_starts_in_jsonl ──────────────────────


class TestEmitCycleEvent:
    def test_emit_writes_cycle_start_with_v1_schema(self, tmp_path):
        path = tmp_path / "llm-calls.jsonl"
        llm.set_calls_jsonl_path(path)
        try:
            llm.emit_cycle_event("cycle_start", {
                "ts": "2026-04-30T00:00:00Z",
                "run_id": "test-run-1",
                "is_resume": False,
                "cycle_seq": 1,
            })
            events = _read_events(path)
            assert len(events) == 1
            assert events[0]["event"] == "cycle_start"
            assert events[0]["run_id"] == "test-run-1"
            assert events[0]["is_resume"] is False
            assert events[0]["cycle_seq"] == 1
            assert events[0]["schema"] == "llm-calls/v1"
        finally:
            llm.set_calls_jsonl_path(None)

    def test_emit_cycle_end_carries_reason(self, tmp_path):
        path = tmp_path / "llm-calls.jsonl"
        llm.set_calls_jsonl_path(path)
        try:
            llm.emit_cycle_event("cycle_end", {
                "ts": "2026-04-30T00:01:00Z",
                "reason": "atexit",
            })
            events = _read_events(path)
            assert events[-1]["event"] == "cycle_end"
            assert events[-1]["reason"] == "atexit"
        finally:
            llm.set_calls_jsonl_path(None)

    def test_emit_rejects_unknown_event(self, tmp_path):
        path = tmp_path / "llm-calls.jsonl"
        llm.set_calls_jsonl_path(path)
        try:
            try:
                llm.emit_cycle_event("frobnicate", {})
            except ValueError as e:
                assert "frobnicate" in str(e)
            else:
                raise AssertionError(
                    "emit_cycle_event should reject unknown event names"
                )
        finally:
            llm.set_calls_jsonl_path(None)

    def test_emit_no_op_when_path_unset(self):
        # No jsonl path configured → no-op (mirrors _append_event_jsonl
        # behavior for tests / ad-hoc scripts).
        llm.set_calls_jsonl_path(None)
        # Should not raise.
        llm.emit_cycle_event("cycle_start", {"ts": "x", "run_id": "y"})


class TestCountCycleStartsInJsonl:
    def test_zero_when_no_file(self, tmp_path):
        # Path set but file doesn't exist yet → returns 0.
        llm.set_calls_jsonl_path(tmp_path / "missing.jsonl")
        try:
            assert llm.count_cycle_starts_in_jsonl() == 0
        finally:
            llm.set_calls_jsonl_path(None)

    def test_zero_when_path_unset(self):
        llm.set_calls_jsonl_path(None)
        assert llm.count_cycle_starts_in_jsonl() == 0

    def test_counts_only_cycle_start_events(self, tmp_path):
        path = tmp_path / "llm-calls.jsonl"
        llm.set_calls_jsonl_path(path)
        try:
            # Mix of every event type the jsonl carries — only
            # cycle_start events should count, not begin/end/counts/
            # cycle_end.
            llm.emit_cycle_event("cycle_start", {"ts": "t1", "run_id": "r"})
            llm._append_event_jsonl("begin", {"call_id": "0001"})
            llm._append_event_jsonl("end", {"call_id": "0001"})
            llm._append_event_jsonl("counts", {"call_id": "0001"})
            llm.emit_cycle_event("cycle_end", {"ts": "t2", "reason": "atexit"})
            llm.emit_cycle_event("cycle_start", {"ts": "t3", "run_id": "r"})
            llm._append_event_jsonl("begin", {"call_id": "0002"})
            llm.emit_cycle_event("cycle_end", {"ts": "t4", "reason": "atexit"})
            assert llm.count_cycle_starts_in_jsonl() == 2
        finally:
            llm.set_calls_jsonl_path(None)

    def test_skips_malformed_lines(self, tmp_path):
        # Real jsonl files in the wild can carry the occasional
        # truncated tail line (SIGKILL mid-write) — counter should
        # tolerate, not crash.
        path = tmp_path / "llm-calls.jsonl"
        llm.set_calls_jsonl_path(path)
        try:
            llm.emit_cycle_event("cycle_start", {"ts": "t1", "run_id": "r"})
            # Append a half-written line by hand.
            with open(path, "a", encoding="utf-8") as f:
                f.write('{"event": "cycle_start", "ts":\n')
            llm.emit_cycle_event("cycle_start", {"ts": "t2", "run_id": "r"})
            assert llm.count_cycle_starts_in_jsonl() == 2
        finally:
            llm.set_calls_jsonl_path(None)


class TestCycleSeqMatchesOnDiskCount:
    """Repro of the canonical multi-cycle sequence: a fresh cycle's
    `cycle_seq` should equal `count_cycle_starts_in_jsonl() + 1`
    BEFORE the new cycle_start is emitted (matches the runner's
    pre-emit computation in `runner.run`)."""

    def test_two_cycle_sequence(self, tmp_path):
        path = tmp_path / "llm-calls.jsonl"
        llm.set_calls_jsonl_path(path)
        try:
            # Cycle 1.
            assert llm.count_cycle_starts_in_jsonl() == 0
            llm.emit_cycle_event("cycle_start", {
                "ts": "t1", "run_id": "r", "is_resume": False, "cycle_seq": 1,
            })
            llm.emit_cycle_event("cycle_end", {"ts": "t1e", "reason": "atexit"})
            # Cycle 2 (resume). The next cycle_seq is 1-indexed so it
            # equals on-disk count + 1.
            assert llm.count_cycle_starts_in_jsonl() == 1
            llm.emit_cycle_event("cycle_start", {
                "ts": "t2", "run_id": "r", "is_resume": True,
                "cycle_seq": llm.count_cycle_starts_in_jsonl() + 1,
            })
            events = _read_events(path)
            cycle_starts = [e for e in events if e["event"] == "cycle_start"]
            assert len(cycle_starts) == 2
            assert cycle_starts[0]["cycle_seq"] == 1
            assert cycle_starts[0]["is_resume"] is False
            assert cycle_starts[1]["cycle_seq"] == 2
            assert cycle_starts[1]["is_resume"] is True
        finally:
            llm.set_calls_jsonl_path(None)


# ── Issue #353: every line carries an ISO-Z `ts` ─────────────────────────


_ISO_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class TestEveryLineHasIsoZTimestamp:
    """Every event line in llm-calls.jsonl carries a `ts` field in
    ISO-Z form (colons in the time portion, trailing Z). The central
    emitter stamps it at write time so adding a new event kind never
    requires the author to remember the timestamp."""

    def test_every_kind_gets_emit_time_ts(self, tmp_path):
        path = tmp_path / "llm-calls.jsonl"
        llm.set_calls_jsonl_path(path)
        try:
            # The five event kinds that the Python emitter writes today
            # (begin / end / counts / stream_progress via
            # `_append_event_jsonl` directly, plus cycle markers via
            # `emit_cycle_event`). New kinds added later inherit the
            # stamp for free.
            llm._append_event_jsonl("begin", {"call_id": "0001"})
            llm._append_event_jsonl("end", {"call_id": "0001"})
            llm._append_event_jsonl("counts", {"call_id": "0001"})
            llm._append_event_jsonl("stream_progress", {"call_id": "0001"})
            llm.emit_cycle_event("cycle_start", {"run_id": "r", "cycle_seq": 1})
            llm.emit_cycle_event("cycle_end", {"reason": "atexit"})
            llm.emit_cycle_event("progress_tick", {"stage": "preflight"})
            for ev in _read_events(path):
                assert "ts" in ev, f"missing ts on {ev['event']!r}: {ev}"
                assert _ISO_Z_RE.match(ev["ts"]), (
                    f"ts on {ev['event']!r} not ISO-Z form: {ev['ts']!r}"
                )
        finally:
            llm.set_calls_jsonl_path(None)

    def test_caller_supplied_ts_wins(self, tmp_path):
        # Test fixtures pin literal `ts` values across emits to assert
        # ordering ("t1" before "t2"). The central stamp respects them
        # when present so those tests don't regress.
        path = tmp_path / "llm-calls.jsonl"
        llm.set_calls_jsonl_path(path)
        try:
            llm.emit_cycle_event("cycle_start", {"ts": "fixed-1", "run_id": "r"})
            llm._append_event_jsonl("begin", {"ts": "fixed-2", "call_id": "0001"})
            events = _read_events(path)
            assert [e["ts"] for e in events] == ["fixed-1", "fixed-2"]
        finally:
            llm.set_calls_jsonl_path(None)
