"""
Round 4 — SIGTERM handler must complete the rollup write INLINE
before os._exit(0). The prior sys.exit + atexit pattern blocked on
ThreadPoolExecutor.__exit__'s shutdown(wait=True) for in-flight
LLM calls (60s+), causing the Tauri SIGKILL fallback to fire before
atexit could run. p63u was the production failure that surfaced this.

This test spawns a subprocess that:
  - emits 3 finalized calls,
  - spawns a thread pool with 2 workers each parked in a 30s sleep
    (simulating in-flight LLM calls),
  - installs the production-shape SIGTERM handler (inline write +
    os._exit).

We send SIGTERM, wait for exit, and assert:
  - The process exited within ~2s (not blocked by the executor).
  - llm-stats.txt exists.
  - The materialized rollup (post-#189 — derived from
    llm-calls.jsonl) carries 5 calls (3 ok + 2 cancelled in flight).

Run with:
    cd engine && pytest tests/test_sigterm_inline_write.py -v
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


HELPER = Path(__file__).parent / "_sigterm_inline_helper.py"
# python/ package root so the helper (launched as a script) resolves its
# fully-qualified `from engine import …` imports.
_PYTHON_ROOT = Path(__file__).resolve().parents[2]


def _spawn(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(_PYTHON_ROOT),
           "TEST_OUT_DIR": str(tmp_path)}
    proc = subprocess.Popen(
        [sys.executable, str(HELPER)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            err = proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"helper exited before READY: {err!r}")
        if b"READY" in line:
            return proc
    proc.kill()
    raise RuntimeError("helper didn't print READY in time")


class TestSigtermInlineWrite:
    def test_inline_write_beats_executor_shutdown(self, tmp_path):
        proc = _spawn(tmp_path)
        try:
            # Send SIGTERM. The inline handler should write the
            # rollup + llm-stats.txt and os._exit(0) within
            # milliseconds — not block on the 30s thread sleeps.
            t0 = time.monotonic()
            proc.send_signal(signal.SIGTERM)
            # Generous timeout — we expect <2s. If executor shutdown
            # were blocking, it would be 30s.
            ret = proc.wait(timeout=8)
            elapsed = time.monotonic() - t0
        except Exception:
            proc.kill()
            raise

        # Process exited cleanly via os._exit(0).
        assert ret == 0, f"exit code {ret}"
        # Should have happened fast — no executor shutdown block.
        assert elapsed < 5, (
            f"SIGTERM took {elapsed:.1f}s — executor blocking? "
            f"(threshold 5s; actual was 30s in the regression case)"
        )

        # llm-stats.json is no longer written end-of-run (issue #189).
        assert not (tmp_path / "llm-stats.json").exists()
        txt_path = tmp_path / "llm-stats.txt"
        assert txt_path.exists(), "llm-stats.txt missing — atexit didn't fire"

        from engine.runner import materialize_run_stats
        rollup = materialize_run_stats(tmp_path / "llm-calls.jsonl")
        # 3 finalized + 2 in-flight = 5 total.
        assert rollup["totals"]["calls"] == 5
        assert rollup["totals"]["successful"] == 3
        assert rollup["totals"]["aborted"] == 2
        # Rollup carries top-level aborted_calls list with both
        # in-flight workers.
        assert len(rollup["aborted_calls"]) == 2
