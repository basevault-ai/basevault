"""Tests for the dev-tab full prompt + response logging path.

When the user toggles `dev_full_prompt_logging.<stage>.input` and/or
`.output` on, the active stat record gets `full_prompt` and/or
`full_response` stamped on it. Default-OFF — without the toggle, no
prompt content lands on the stat record (preserving the share-safe
default).

Tests directly exercise `_stamp_full_io` + `_dev_logging_for_stage`
against a synthetic stat record + monkeypatched config reader, so no
provider client / network is involved.
"""
from __future__ import annotations




def _put_rec(llm, call_id: str = "0001") -> dict:
    """Drop a stat record into `_stats_records` so `_stamp_full_io`'s
    `_get_rec(call_id)` lookup hits. Returns the rec dict so the test
    can assert on the post-stamp shape."""
    llm.reset_stat_records()
    rec = {"call_id": call_id}
    llm._stats_records.append(rec)
    return rec


def test_no_logging_when_toggle_off(monkeypatch):
    """Default state: no toggles → neither field lands on the record."""
    from engine import llm
    monkeypatch.setattr(llm, "_read_app_config", lambda: {})
    monkeypatch.setattr(llm, "_current_stage", "extract")
    rec = _put_rec(llm)
    llm._stamp_full_io(
        "0001", [{"role": "user", "content": "hi"}], "hello back")
    assert "full_prompt" not in rec
    assert "full_response" not in rec


def test_input_only_stamps_prompt_not_response(monkeypatch):
    """input toggle ON, output OFF → only full_prompt lands."""
    from engine import llm
    monkeypatch.setattr(llm, "_read_app_config", lambda: {
        "dev_full_prompt_logging": {"extract": {"input": True, "output": False}},
    })
    monkeypatch.setattr(llm, "_current_stage", "extract")
    rec = _put_rec(llm)
    msgs = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "what is 1+1"},
    ]
    llm._stamp_full_io("0001", msgs, "2")
    assert rec["full_prompt"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "what is 1+1"},
    ]
    assert "full_response" not in rec


def test_output_only_stamps_response_not_prompt(monkeypatch):
    """Mirror: output toggle ON, input OFF → only full_response lands."""
    from engine import llm
    monkeypatch.setattr(llm, "_read_app_config", lambda: {
        "dev_full_prompt_logging": {"patterns": {"input": False, "output": True}},
    })
    monkeypatch.setattr(llm, "_current_stage", "patterns")
    rec = _put_rec(llm)
    llm._stamp_full_io(
        "0001", [{"role": "user", "content": "x"}], "the response")
    assert "full_prompt" not in rec
    assert rec["full_response"] == "the response"


def test_per_stage_isolation(monkeypatch):
    """Toggle is keyed on the active stage. A toggle for `extract`
    doesn't fire when the live stage is `patterns`."""
    from engine import llm
    monkeypatch.setattr(llm, "_read_app_config", lambda: {
        "dev_full_prompt_logging": {"extract": {"input": True, "output": True}},
    })
    monkeypatch.setattr(llm, "_current_stage", "patterns")
    rec = _put_rec(llm)
    llm._stamp_full_io("0001", [{"role": "user", "content": "x"}], "y")
    assert "full_prompt" not in rec
    assert "full_response" not in rec


def test_dev_logging_for_stage_handles_malformed_config(monkeypatch):
    """Defensive: empty / malformed config returns (False, False)."""
    from engine import llm
    monkeypatch.setattr(llm, "_current_stage", "extract")
    monkeypatch.setattr(llm, "_read_app_config", lambda: {})
    assert llm._dev_logging_for_stage("extract") == (False, False)
    monkeypatch.setattr(llm, "_read_app_config", lambda: {
        "dev_full_prompt_logging": "not-a-dict",
    })
    assert llm._dev_logging_for_stage("extract") == (False, False)
    monkeypatch.setattr(llm, "_read_app_config", lambda: {
        "dev_full_prompt_logging": {"extract": "not-a-dict"},
    })
    assert llm._dev_logging_for_stage("extract") == (False, False)
    monkeypatch.setattr(llm, "_read_app_config", lambda: {
        "dev_full_prompt_logging": {"extract": {}},
    })
    assert llm._dev_logging_for_stage("extract") == (False, False)


def test_no_active_record_is_noop(monkeypatch):
    """If there's no active stat record (begin_stat_record never fired
    or ad-hoc test path), the stamp is a clean no-op. With #264 the
    record is looked up by call_id; passing None bypasses the lookup
    entirely. Same defense covers the `call_id` not in `_stats_records`
    case."""
    from engine import llm
    monkeypatch.setattr(llm, "_read_app_config", lambda: {
        "dev_full_prompt_logging": {"extract": {"input": True, "output": True}},
    })
    monkeypatch.setattr(llm, "_current_stage", "extract")
    llm.reset_stat_records()
    # Should not raise.
    llm._stamp_full_io(None, [{"role": "user", "content": "x"}], "y")
    llm._stamp_full_io(
        "no-such-id", [{"role": "user", "content": "x"}], "y")


def test_materializer_threads_payload_record_onto_call(tmp_path):
    """Issue #195 split path: full_io payloads live in the sibling
    `llm-payloads.jsonl`. Materializer reads the metadata events from
    `llm-calls.jsonl`, then walks the payloads file and threads
    `full_prompt` / `full_response` onto matching records."""
    import json
    from engine.runner import _materialize_calls_from_jsonl
    calls = tmp_path / "llm-calls.jsonl"
    payloads = tmp_path / "llm-payloads.jsonl"
    events = [
        {"event": "begin", "call_id": "0001", "stage": "extract",
         "category": "doc", "model": "fixture",
         "started_at_iso": "2026-05-07T00:00:00.000Z",
         "attempt": 1, "retry_of_call_id": None,
         "budget": None, "template_hash": None},
        {"event": "end", "call_id": "0001", "duration_ms": 1234,
         "success": True, "error": None,
         "prompt_tokens": 100, "completion_tokens": 50,
         "model": "fixture", "mode": "test", "cached": False},
    ]
    with calls.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    payload_record = {
        "call_id": "0001",
        "full_prompt": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "extract facts from..."},
        ],
        "full_response": "fact 1\nfact 2",
        "schema": "llm-payloads/v1",
    }
    with payloads.open("w") as f:
        f.write(json.dumps(payload_record) + "\n")
    recs = _materialize_calls_from_jsonl(calls, "2026-05-07T00:01:00Z")
    assert len(recs) == 1
    r = recs[0]
    assert r["full_prompt"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "extract facts from..."},
    ]
    assert r["full_response"] == "fact 1\nfact 2"


def test_materializer_legacy_full_io_in_calls_jsonl(tmp_path):
    """Backward compat (issue #195): pre-split runs embedded `full_io`
    events directly in llm-calls.jsonl. The materializer must still
    thread those onto records so old runs render correctly in the
    Run Details modal — no llm-payloads.jsonl on disk in this case."""
    import json
    from engine.runner import _materialize_calls_from_jsonl
    p = tmp_path / "llm-calls.jsonl"
    events = [
        {"event": "begin", "call_id": "0001", "stage": "extract",
         "category": "doc", "model": "fixture",
         "started_at_iso": "2026-05-07T00:00:00.000Z",
         "attempt": 1, "retry_of_call_id": None,
         "budget": None, "template_hash": None},
        {"event": "end", "call_id": "0001", "duration_ms": 1234,
         "success": True, "error": None,
         "prompt_tokens": 100, "completion_tokens": 50,
         "model": "fixture", "mode": "test", "cached": False},
        {"event": "full_io", "call_id": "0001",
         "full_prompt": [
             {"role": "system", "content": "be helpful"},
             {"role": "user", "content": "extract facts from..."},
         ],
         "full_response": "fact 1\nfact 2"},
    ]
    with p.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    assert not (tmp_path / "llm-payloads.jsonl").exists()
    recs = _materialize_calls_from_jsonl(p, "2026-05-07T00:01:00Z")
    assert len(recs) == 1
    r = recs[0]
    assert r["full_prompt"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "extract facts from..."},
    ]
    assert r["full_response"] == "fact 1\nfact 2"


def test_materializer_omits_full_io_fields_when_no_event(tmp_path):
    """Default path: no `full_io` event AND no payloads file → record
    has neither key. Important so a normal run's materialized rollup
    doesn't carry empty placeholder fields."""
    import json
    from engine.runner import _materialize_calls_from_jsonl
    p = tmp_path / "llm-calls.jsonl"
    events = [
        {"event": "begin", "call_id": "0001", "stage": "extract",
         "category": "doc", "model": "fixture",
         "started_at_iso": "2026-05-07T00:00:00.000Z",
         "attempt": 1, "retry_of_call_id": None,
         "budget": None, "template_hash": None},
        {"event": "end", "call_id": "0001", "duration_ms": 1234,
         "success": True, "error": None,
         "prompt_tokens": 100, "completion_tokens": 50,
         "model": "fixture", "mode": "test", "cached": False},
    ]
    with p.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    recs = _materialize_calls_from_jsonl(p, "2026-05-07T00:01:00Z")
    assert "full_prompt" not in recs[0]
    assert "full_response" not in recs[0]


def test_stamp_full_io_writes_to_payloads_not_calls(monkeypatch, tmp_path):
    """Issue #195 writer split: when the dev-tab toggle is on,
    `_stamp_full_io` streams the payload to llm-payloads.jsonl ONLY.
    llm-calls.jsonl stays untouched by full_io records — the metadata
    stream is no longer drowned by 50KB+/call payloads."""
    import json as _json
    from engine import llm
    monkeypatch.setattr(llm, "_read_app_config", lambda: {
        "dev_full_prompt_logging": {"extract": {"input": True, "output": True}},
    })
    monkeypatch.setattr(llm, "_current_stage", "extract")
    # Register a rec into the call-id-keyed store so `_stamp_full_io`'s
    # `_get_rec("0007")` lookup hits. Return value isn't read here — the
    # rec mutation is observed via the on-disk payload, not via the in-
    # memory dict.
    _put_rec(llm, call_id="0007")
    calls_path = tmp_path / "llm-calls.jsonl"
    payloads_path = tmp_path / "llm-payloads.jsonl"
    monkeypatch.setattr(llm, "_calls_jsonl_path", calls_path)
    monkeypatch.setattr(llm, "_payloads_jsonl_path", payloads_path)
    llm._stamp_full_io(
        "0007",
        [{"role": "user", "content": "the prompt"}], "the response",
    )
    assert not calls_path.exists()
    assert payloads_path.exists()
    lines = payloads_path.read_text().splitlines()
    assert len(lines) == 1
    parsed = _json.loads(lines[0])
    assert parsed["call_id"] == "0007"
    assert parsed["full_prompt"] == [
        {"role": "user", "content": "the prompt"},
    ]
    assert parsed["full_response"] == "the response"
    assert parsed["schema"] == "llm-payloads/v1"
    # No `event` field — single-purpose file makes it redundant.
    assert "event" not in parsed


def test_stamp_full_io_noop_when_payloads_path_unset(monkeypatch):
    """Toggle on but `_payloads_jsonl_path` unset (tests, ad-hoc
    scripts): stamp still updates the live record (in-memory), but
    the disk write is a clean no-op — no crash, no path error."""
    from engine import llm
    monkeypatch.setattr(llm, "_read_app_config", lambda: {
        "dev_full_prompt_logging": {"extract": {"input": True, "output": True}},
    })
    monkeypatch.setattr(llm, "_current_stage", "extract")
    monkeypatch.setattr(llm, "_payloads_jsonl_path", None)
    rec = _put_rec(llm)
    llm._stamp_full_io(
        "0001", [{"role": "user", "content": "x"}], "y")
    assert rec["full_prompt"] == [{"role": "user", "content": "x"}]
    assert rec["full_response"] == "y"
