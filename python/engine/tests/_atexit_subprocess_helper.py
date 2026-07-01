"""
Subprocess helper for test_llm_stats_atexit.

Imports llm + runner, exercises the same wiring run() uses (set_calls_jsonl_path
+ atexit.register(_write_llm_stats, ...) + SIGTERM → sys.exit handler), pumps
N finalize calls into the stats system, prints "READY" on stdout, then sleeps
until killed. The parent test sends SIGTERM (atexit fires → both files exist)
or SIGKILL (atexit can't fire → only .jsonl exists).

The helper deliberately doesn't import anything beyond llm + runner so the
subprocess startup is fast (<1s typical). NOT a pytest test file — exec'd
via subprocess.Popen by `test_llm_stats_atexit.py`.
"""
from __future__ import annotations

import atexit
import os
import signal
import sys
import time
from pathlib import Path


def _main() -> None:

    from engine import llm
    from engine.runner import _write_llm_stats, _reset_llm_stats_dump_state

    out_dir = Path(os.environ["TEST_OUT_DIR"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "llm-calls.jsonl"
    txt_path = out_dir / "llm-stats.txt"
    for p in (jsonl_path, txt_path):
        if p.exists():
            p.unlink()

    # Same wiring run() does: streaming sidecar + idempotent rollup
    # registered as an atexit handler + SIGTERM → sys.exit.
    llm.reset_stat_records()
    llm.set_calls_jsonl_path(jsonl_path)
    _reset_llm_stats_dump_state()
    atexit.register(
        _write_llm_stats, out_dir, llm.Mode.TEE,
        primary_model="gpt-oss-120b",
        run_started_at_iso="2026-04-27T00:00:00.000Z",
    )

    def _sigterm_to_sysexit(_signum, _frame):
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm_to_sysexit)

    n_calls = int(os.environ.get("TEST_N_CALLS", "3"))
    n_inflight = int(os.environ.get("TEST_N_INFLIGHT", "0"))
    for i in range(n_calls):
        cid = llm.begin_stat_record(
            "extract", f"cat-{i}", "gpt-oss-120b", attempt=1)
        llm.finalize_stat_record(cid, success=True, duration_ms=10 + i)

    # Optionally simulate calls in flight at cancel time: open a
    # begin_stat_record without ever finalizing. The .jsonl gets a
    # begin event with no matching end, which the materializer
    # surfaces as a cancelled record.
    for i in range(n_inflight):
        llm.begin_stat_record(
            "extract", f"inflight-{i}", "gpt-oss-120b", attempt=1)

    # Tell the parent test we're past the finalize calls and ready
    # for the kill signal. Flushed so the parent doesn't block on
    # stdio buffering.
    print("READY", flush=True)

    # Wait to be killed. Loop sleeps in small slices so SIGTERM
    # delivery → handler invocation is prompt.
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    _main()
