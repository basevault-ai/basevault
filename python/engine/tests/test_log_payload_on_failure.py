"""Tests for the failure-payload logger (issue #266).

`llm._log_call_failure_payload` writes one `llm-payloads.jsonl` record for a
failed call so timed-out / transport-failed / `from_status` (LOAD / timeout /
parse_error) calls leave their full prompt (and any partial streamed
response) on disk — the diagnostic data the success-only `_stamp_full_io`
path never saved. Pre-fix, run x4va had 52/53 timed-out calls with no
captured prompt.

On the kernel path the helper is driven by `KernelTelemetryHook`'s
`failure_payload_sink` (wired in `build_stage_env` / `entities_job`); the
end-to-end crediting is covered by
`test_telemetry_hook_offline.py::test_failed_call_prompt_reaches_failure_payload_sink`.
This file pins the helper's own edge case.

No live LLM calls. Run with:
    cd engine && pytest tests/test_log_payload_on_failure.py -v
"""
from __future__ import annotations

from engine import llm


def test_failure_helper_noop_when_payloads_path_unset():
    """`set_payloads_jsonl_path(None)` (tests / ad-hoc scripts)
    silently skips the disk write. No exception bubbles."""
    llm.reset_stat_records()
    llm.set_payloads_jsonl_path(None)
    # Pretend a call started so begin_stat_record initializes the
    # threadlocal partial buffer.
    llm._active_partial_response.parts = []
    # Should not raise.
    llm._log_call_failure_payload(
        "9999", [{"role": "user", "content": "x"}])
    llm.reset_stat_records()
