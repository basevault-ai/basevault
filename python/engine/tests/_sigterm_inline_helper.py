"""
Subprocess helper for test_sigterm_inline_write.

Mimics the production SIGTERM handler (Round 4): rollup written
INLINE in the signal handler, then os._exit(0). Unlike the
sys.exit + atexit pattern in _atexit_subprocess_helper.py, this
pattern is what runs in real `runner.run()` — it dodges the
ThreadPoolExecutor.__exit__'s shutdown(wait=True) which can block
the main thread for tens of seconds while in-flight LLM calls
time out.

The helper additionally simulates that very blocker: it spawns a
worker thread that "holds" a fake LLM call for 30 seconds. If the
sys.exit + atexit pattern were in use, the executor's shutdown
would wait for that thread, atexit would not fire within the
3-5s SIGTERM grace window, and SIGKILL would arrive first. With
the inline-write pattern, the rollup is on disk before any
shutdown begins.

NOT a pytest test file — exec'd via subprocess.Popen by
`test_sigterm_inline_write.py`.
"""
from __future__ import annotations

import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _main() -> None:

    from engine import llm
    from engine.runner import _write_llm_stats, _reset_llm_stats_dump_state

    out_dir = Path(os.environ["TEST_OUT_DIR"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "llm-calls.jsonl"
    for p in (jsonl_path, out_dir / "llm-stats.txt"):
        if p.exists():
            p.unlink()

    llm.reset_stat_records()
    llm.set_calls_jsonl_path(jsonl_path)
    _reset_llm_stats_dump_state()

    _atexit_kwargs = dict(
        primary_model="gpt-oss-120b",
        run_started_at_iso="2026-04-29T00:00:00.000Z",
    )

    # Production-shape SIGTERM handler: write rollup inline, then
    # os._exit(0). This bypasses the executor.shutdown(wait=True)
    # path that would otherwise block atexit past the SIGKILL grace.
    def _sigterm_flush_and_exit(_signum, _frame):
        try:
            _write_llm_stats(out_dir, llm.Mode.TEE, **_atexit_kwargs)
        except BaseException:
            pass
        os._exit(0)
    signal.signal(signal.SIGTERM, _sigterm_flush_and_exit)

    # Three completed calls + a long-running "in flight" pseudo-call
    # in a worker thread that the executor would wait for on a
    # graceful shutdown. The wrapper opens a begin_stat_record but
    # never finalizes (the call_id stays unmatched in the .jsonl).
    for i in range(3):
        cid = llm.begin_stat_record(
            "extract", f"cat-{i}", "gpt-oss-120b", attempt=1)
        llm.finalize_stat_record(cid, success=True, duration_ms=10 + i)

    def _slow_worker(slot: int):
        # Pretend to start an extract call — emit a begin event but
        # don't finalize. This becomes a cancelled record in the
        # rollup.
        llm.begin_stat_record(
            "extract", f"inflight-{slot}", "gpt-oss-120b", attempt=1)
        # Block for 30 seconds — well past the 5s SIGTERM grace.
        # If the production handler weren't using inline-write +
        # os._exit, the executor.shutdown(wait=True) on SystemExit
        # would block here and atexit would fire too late.
        time.sleep(30)

    pool = ThreadPoolExecutor(max_workers=2)
    pool.submit(_slow_worker, 0)
    pool.submit(_slow_worker, 1)
    # Give the workers a moment to start + emit begins.
    time.sleep(0.2)

    print("READY", flush=True)

    # Park forever; SIGTERM will fire the inline handler.
    while True:
        time.sleep(0.5)


if __name__ == "__main__":
    _main()
