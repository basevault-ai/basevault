"""Tests for the auto-log-on-failure path (issue #197).

When a call's terminal classification is `parse_error: true` or its
output dict reads as zero-size (primary count == 0), `record_stage_counts`
streams the call's full prompt + response to `llm-payloads.jsonl`
regardless of the dev-tab toggle. Diagnosis from the run dir alone, no
re-run needed.

Tests exercise `_stamp_full_io` + `record_stage_counts` directly with a
synthetic stat record; no provider client / network involved.
"""
from __future__ import annotations

import json



def _wire_active_rec(monkeypatch, llm, rec):
    """Drop `rec` into `_stats_records` so `_get_rec(rec["call_id"])`
    lookups inside `_stamp_full_io` and `record_stage_counts` hit. Pre-#264
    parked the rec on a thread-scoped slot; post-#264 the rec lives in the
    process-wide store keyed by call_id."""
    llm.reset_stat_records()
    llm._stats_records.append(rec)


def _setup(monkeypatch, tmp_path, *, dev_toggle=None, stage="patterns"):
    """Common setup: clean pending buffer, route the payloads stream
    to a tmp file, set the active stage, install the dev-tab config.
    Returns (llm module, payloads path, stat record)."""
    from engine import llm
    monkeypatch.setattr(llm, "_calls_jsonl_path", None)
    payloads_path = tmp_path / "llm-payloads.jsonl"
    monkeypatch.setattr(llm, "_payloads_jsonl_path", payloads_path)
    monkeypatch.setattr(llm, "_current_stage", stage)
    cfg = {}
    if dev_toggle is not None:
        cfg = {"dev_full_prompt_logging": {stage: dev_toggle}}
    monkeypatch.setattr(llm, "_read_app_config", lambda: cfg)
    llm._pending_payloads.clear()
    rec = {"call_id": "0001"}
    _wire_active_rec(monkeypatch, llm, rec)
    return llm, payloads_path, rec


def _payload_records(path):
    """Read the payloads JSONL and return the parsed records (list)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_parse_error_triggers_payload_write_with_toggle_off(monkeypatch, tmp_path):
    """parse_error: True with dev-toggle OFF still streams the full
    prompt + response to llm-payloads.jsonl. This is the core diagnostic
    win — failures are debuggable from the run dir alone."""
    llm, payloads, rec = _setup(monkeypatch, tmp_path, dev_toggle=None)
    msgs = [{"role": "user", "content": "the prompt"}]
    llm._stamp_full_io("0001", msgs, "the broken response")
    # Toggle off ⇒ no immediate write.
    assert not payloads.exists()
    llm.record_stage_counts("0001", 
        input={"facts": 100},
        output={"patterns": 0},
        failure_kind="parse_error",
    )
    records = _payload_records(payloads)
    assert len(records) == 1
    p = records[0]
    assert p["call_id"] == "0001"
    assert p["full_prompt"] == [{"role": "user", "content": "the prompt"}]
    assert p["full_response"] == "the broken response"
    assert p["schema"] == "llm-payloads/v1"


def test_zero_size_output_triggers_payload_write_with_toggle_off(monkeypatch, tmp_path):
    """patterns returning `[]` for substantial input → output
    `{"patterns": 0}` → auto-log writes a payload. No parse_error
    flag needed; the zero primary count alone trips the trigger."""
    llm, payloads, rec = _setup(monkeypatch, tmp_path, dev_toggle=None)
    llm._stamp_full_io("0001", 
        [{"role": "user", "content": "5000 facts ..."}],
        "[]",
    )
    llm.record_stage_counts("0001", 
        input={"facts": 5000},
        output={"patterns": 0},
    )
    records = _payload_records(payloads)
    assert len(records) == 1
    assert records[0]["full_response"] == "[]"


def test_clean_success_with_toggle_off_writes_nothing(monkeypatch, tmp_path):
    """Toggle off + clean success → no payload write. The auto-log
    must not bloat the file on the happy path."""
    llm, payloads, rec = _setup(monkeypatch, tmp_path, dev_toggle=None)
    llm._stamp_full_io("0001", 
        [{"role": "user", "content": "hi"}],
        "the response",
    )
    llm.record_stage_counts("0001", 
        input={"facts": 10},
        output={"patterns": 7},
    )
    assert not payloads.exists()


def test_clean_success_with_toggle_on_still_writes(monkeypatch, tmp_path):
    """Existing dev-tab toggle behavior: input+output both ON → one
    payload record on success. record_stage_counts (with no failure
    signal) must not write a second record."""
    llm, payloads, rec = _setup(
        monkeypatch, tmp_path,
        dev_toggle={"input": True, "output": True},
    )
    llm._stamp_full_io("0001", 
        [{"role": "user", "content": "hi"}],
        "the response",
    )
    llm.record_stage_counts("0001", 
        input={"facts": 10},
        output={"patterns": 5},
    )
    records = _payload_records(payloads)
    assert len(records) == 1
    assert records[0]["full_prompt"] == [{"role": "user", "content": "hi"}]
    assert records[0]["full_response"] == "the response"


def test_dedup_when_toggle_on_and_failure_hits(monkeypatch, tmp_path):
    """Toggle ON for both directions AND parse_error → exactly ONE
    payload record. The dev-tab path wrote it; the auto-log path sees
    `written: True` in the buffer and skips."""
    llm, payloads, rec = _setup(
        monkeypatch, tmp_path,
        dev_toggle={"input": True, "output": True},
    )
    llm._stamp_full_io("0001", 
        [{"role": "user", "content": "p"}],
        "broken",
    )
    llm.record_stage_counts("0001", 
        input={"facts": 100},
        output={"patterns": 0},
        failure_kind="parse_error",
    )
    records = _payload_records(payloads)
    assert len(records) == 1


def test_zero_size_with_aggregate_subdicts_still_triggers(monkeypatch, tmp_path):
    """Stages like `insights` emit aggregate sub-dicts (`kinds: {}`)
    and cap fields alongside the primary count. The primary count
    leads the dict; when it's 0, auto-log fires regardless of the
    cap fields."""
    llm, payloads, rec = _setup(
        monkeypatch, tmp_path,
        dev_toggle=None, stage="insights",
    )
    llm._stamp_full_io("0001", 
        [{"role": "user", "content": "..."}],
        "{}",
    )
    llm.record_stage_counts("0001", 
        input={"patterns": 50, "topics_with_patterns": 5},
        output={
            "insights": 0,
            "cross_domain": 0,
            "critical": 0,
            "kinds": {},
            "total_cap": 50,
            "cross_cap": 30,
            "critical_cap": 20,
        },
    )
    records = _payload_records(payloads)
    assert len(records) == 1


def test_non_zero_output_with_caps_does_not_trigger(monkeypatch, tmp_path):
    """Counter-test for the heuristic: a successful insights call has
    a non-zero `insights` count even though its dict still includes
    cap fields. No auto-log fires."""
    llm, payloads, rec = _setup(
        monkeypatch, tmp_path,
        dev_toggle=None, stage="insights",
    )
    llm._stamp_full_io("0001", 
        [{"role": "user", "content": "..."}],
        "...",
    )
    llm.record_stage_counts("0001", 
        input={"patterns": 50, "topics_with_patterns": 5},
        output={
            "insights": 7,
            "cross_domain": 4,
            "critical": 3,
            "kinds": {"opportunity": 4, "risk": 3},
            "total_cap": 50,
        },
    )
    assert not payloads.exists()


def test_pending_buffer_popped_on_clean_success(monkeypatch, tmp_path):
    """Memory hygiene: the call_id entry is popped from
    `_pending_payloads` even on clean success, so the buffer doesn't
    grow unboundedly across the run."""
    llm, payloads, rec = _setup(monkeypatch, tmp_path, dev_toggle=None)
    llm._stamp_full_io("0001", [{"role": "user", "content": "x"}], "y")
    assert "0001" in llm._pending_payloads
    llm.record_stage_counts("0001", 
        input={"facts": 10},
        output={"patterns": 5},
    )
    assert "0001" not in llm._pending_payloads


def test_pending_buffer_popped_on_parse_error(monkeypatch, tmp_path):
    """Same hygiene check for the failure path: the call's snapshot
    is consumed by the auto-log write and removed from the buffer."""
    llm, payloads, rec = _setup(monkeypatch, tmp_path, dev_toggle=None)
    llm._stamp_full_io("0001", [{"role": "user", "content": "x"}], "y")
    llm.record_stage_counts("0001", 
        input={"facts": 10},
        output={"patterns": 0},
        failure_kind="parse_error",
    )
    assert "0001" not in llm._pending_payloads


def test_output_is_zero_size_helper():
    """Direct exercise of the zero-size heuristic."""
    from engine import llm
    assert llm._output_is_zero_size({"facts": 0}) is True
    assert llm._output_is_zero_size({"patterns": 0}) is True
    assert llm._output_is_zero_size({"merges": 0}) is True
    assert llm._output_is_zero_size(
        {"insights": 0, "cross_domain": 0, "critical": 0, "kinds": {}}
    ) is True
    assert llm._output_is_zero_size({"facts": 5}) is False
    assert llm._output_is_zero_size({"actions": 3, "kinds": {}}) is False
    # bool leaves are skipped (bool is an int subclass)
    assert llm._output_is_zero_size({"flag": True, "facts": 0}) is True
    assert llm._output_is_zero_size({"flag": True, "facts": 1}) is False
    # No int leaf at all → False (we only fire when we have evidence)
    assert llm._output_is_zero_size({"kinds": {}}) is False
    assert llm._output_is_zero_size(None) is False
    assert llm._output_is_zero_size({}) is False


def test_pending_buffer_evicts_oldest_at_cap(monkeypatch, tmp_path):
    """When a stage skips record_stage_counts, the buffer would grow
    unbounded. The cap evicts the oldest entry on overflow."""
    from engine import llm
    monkeypatch.setattr(llm, "_calls_jsonl_path", None)
    monkeypatch.setattr(llm, "_payloads_jsonl_path", None)
    monkeypatch.setattr(llm, "_current_stage", "extract")
    monkeypatch.setattr(llm, "_read_app_config", lambda: {})
    llm._pending_payloads.clear()
    cap = llm._PENDING_PAYLOADS_MAX
    llm.reset_stat_records()
    for i in range(cap + 5):
        cid = f"{i:04d}"
        llm._stats_records.append({"call_id": cid})
        llm._stamp_full_io(cid, [{"role": "user", "content": str(i)}], "r")
    assert len(llm._pending_payloads) == cap
    # Oldest 5 should be gone.
    for i in range(5):
        assert f"{i:04d}" not in llm._pending_payloads
    # Newest entry is present.
    assert f"{cap + 4:04d}" in llm._pending_payloads


def test_reset_stat_records_clears_pending_buffer(monkeypatch, tmp_path):
    """`reset_stat_records` between runs / tests clears the auto-log
    buffer so a stale entry doesn't bleed into the next run."""
    from engine import llm
    monkeypatch.setattr(llm, "_calls_jsonl_path", None)
    monkeypatch.setattr(llm, "_payloads_jsonl_path", None)
    monkeypatch.setattr(llm, "_current_stage", "extract")
    monkeypatch.setattr(llm, "_read_app_config", lambda: {})
    llm._pending_payloads.clear()
    rec = {"call_id": "0001"}
    _wire_active_rec(monkeypatch, llm, rec)
    llm._stamp_full_io("0001", [{"role": "user", "content": "x"}], "y")
    assert len(llm._pending_payloads) == 1
    llm.reset_stat_records()
    assert len(llm._pending_payloads) == 0


def test_no_payload_write_when_path_unset(monkeypatch, tmp_path):
    """Auto-log is a no-op disk-wise when `_payloads_jsonl_path` is
    None (tests, ad-hoc scripts) — the buffer still gets cleaned up
    so memory doesn't leak."""
    from engine import llm
    monkeypatch.setattr(llm, "_calls_jsonl_path", None)
    monkeypatch.setattr(llm, "_payloads_jsonl_path", None)
    monkeypatch.setattr(llm, "_current_stage", "patterns")
    monkeypatch.setattr(llm, "_read_app_config", lambda: {})
    llm._pending_payloads.clear()
    rec = {"call_id": "0001"}
    _wire_active_rec(monkeypatch, llm, rec)
    llm._stamp_full_io("0001", [{"role": "user", "content": "x"}], "y")
    llm.record_stage_counts("0001", 
        input={"facts": 10},
        output={"patterns": 0},
        failure_kind="parse_error",
    )
    assert "0001" not in llm._pending_payloads
