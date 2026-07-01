"""
Atexit + SIGTERM rollup tests for llm-calls.jsonl + llm-stats.txt.

Spawns a Python subprocess that exercises the same atexit registration
runner.run() does (`runner._write_llm_stats` registered against `out_dir`,
mode, etc., plus a SIGTERM → sys.exit handler) and exercises both the
SIGTERM and SIGKILL kill paths.

Acceptance:
  - SIGTERM mid-run: `llm-calls.jsonl` and `llm-stats.txt` exist on
    disk after the subprocess exits. The streaming sidecar wrote the
    .jsonl as each begin/end ran; the SIGTERM handler converted the
    signal into sys.exit, atexit fired the rollup writer, the .txt
    summary got produced. The rollup dict itself is derived
    on-demand via `materialize_run_stats(jsonl, config)` and
    asserted on the materialized payload — issue #189.
  - SIGKILL: only `llm-calls.jsonl` exists. atexit cannot run on
    SIGKILL — the streaming sidecar is the only crash-survivor by
    design. The test asserts the .txt is absent so the contract
    "atexit doesn't fire on SIGKILL" stays explicit.
  - Idempotency: registering the atexit handler AND calling
    `_write_llm_stats` explicitly inside the subprocess produces
    exactly one rollup write (the second is a no-op).
  - In-flight begin without matching end is surfaced as an aborted
    record in the materialized rollup.

Run with:
    cd engine && pytest tests/test_llm_stats_atexit.py -v

These are NOT marked `integration` — no live LLM calls. They are
slower than pure unit tests (~1-2s per case for subprocess startup
+ signal delivery) but well below the integration threshold.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


HELPER = Path(__file__).parent / "_atexit_subprocess_helper.py"
# python/ package root so the helper (launched as a script) resolves
# its fully-qualified `from engine import …` imports.
_PYTHON_ROOT = Path(__file__).resolve().parents[2]


def _spawn_helper(tmp_path: Path, n_calls: int = 3, n_inflight: int = 0):
    """Launch the helper subprocess and wait for the READY marker on
    stdout. Returns (proc, jsonl_path, txt_path)."""
    env = {
        **os.environ,
        "PYTHONPATH": str(_PYTHON_ROOT),
        "TEST_OUT_DIR": str(tmp_path),
        "TEST_N_CALLS": str(n_calls),
        "TEST_N_INFLIGHT": str(n_inflight),
    }
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
            raise RuntimeError(
                f"helper subprocess exited before READY. stderr={err!r}"
            )
        if b"READY" in line:
            return (
                proc,
                tmp_path / "llm-calls.jsonl",
                tmp_path / "llm-stats.txt",  # human-readable summary sibling
            )
    proc.kill()
    raise RuntimeError("helper subprocess didn't print READY within 10s")


def _read_events(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _materialize(jsonl_path: Path) -> dict:
    """Re-derive the rollup from the on-disk event log — same code
    path the runtime uses for every llm-stats.json read post-#189."""
    from engine.runner import materialize_run_stats
    return materialize_run_stats(jsonl_path)


# ── SIGTERM: all three files exist ────────────────────────────────────────────


class TestSigterm:
    def test_sigterm_writes_all_files(self, tmp_path):
        proc, jsonl_path, txt_path = _spawn_helper(tmp_path, n_calls=3)
        try:
            assert jsonl_path.exists()
            # 3 begin + 3 end events at READY time.
            assert len(_read_events(jsonl_path)) == 6

            proc.send_signal(signal.SIGTERM)
            ret = proc.wait(timeout=15)
        except Exception:
            proc.kill()
            raise
        assert ret == 0, f"exit code {ret}"
        assert jsonl_path.exists(), "streaming sidecar missing"
        # llm-stats.json is no longer written end-of-run (issue #189).
        assert not (tmp_path / "llm-stats.json").exists(), (
            "llm-stats.json should no longer be written")
        assert txt_path.exists(), "llm-stats.txt missing — atexit didn't fire"
        rollup = _materialize(jsonl_path)
        assert rollup["schema"] == "llm-stats/v1"
        assert rollup["totals"]["calls"] == 3
        assert rollup["totals"]["successful"] == 3
        assert rollup["totals"]["aborted"] == 0
        assert len(rollup["calls"]) == 3
        # Rollup call_ids should match the begin events.
        begins = [e for e in _read_events(jsonl_path) if e["event"] == "begin"]
        assert {c["call_id"] for c in rollup["calls"]} == {
            e["call_id"] for e in begins
        }


# ── SIGTERM with in-flight begins → aborted records ──────────────────────────


class TestSigtermInFlight:
    def test_unmatched_begin_becomes_aborted(self, tmp_path):
        # 3 calls finalize cleanly; 2 calls have begin but no end
        # (simulating "in flight at SIGTERM"). Rollup should show
        # 5 total: 3 successful + 2 aborted.
        proc, jsonl_path, _txt_path = _spawn_helper(
            tmp_path, n_calls=3, n_inflight=2)
        try:
            events = _read_events(jsonl_path)
            # 3 begin+end (6) plus 2 unmatched begins (2) = 8 lines.
            assert len(events) == 8
            n_begin = sum(1 for e in events if e["event"] == "begin")
            n_end = sum(1 for e in events if e["event"] == "end")
            assert n_begin == 5 and n_end == 3

            proc.send_signal(signal.SIGTERM)
            ret = proc.wait(timeout=15)
        except Exception:
            proc.kill()
            raise
        assert ret == 0
        rollup = _materialize(jsonl_path)
        assert rollup["totals"]["calls"] == 5
        assert rollup["totals"]["successful"] == 3
        assert rollup["totals"]["aborted"] == 2
        aborted = [c for c in rollup["calls"] if c.get("aborted")]
        assert len(aborted) == 2
        for c in aborted:
            assert c["success"] is False
            # No synthesized error — the call didn't fail, the run
            # wound down. Mirrors Rust live materializer.
            assert c["error"] is None
            assert c["duration_ms"] is not None
        # Top-level aborted_calls list.
        assert len(rollup["aborted_calls"]) == 2
        assert {c["call_id"] for c in rollup["aborted_calls"]} == {
            c["call_id"] for c in aborted
        }


# ── SIGKILL: only jsonl ───────────────────────────────────────────────────────


class TestSigkill:
    def test_sigkill_only_streams_jsonl(self, tmp_path):
        # SIGKILL cannot be intercepted by Python — atexit doesn't
        # fire. The streaming sidecar is the ONLY observability
        # surviving this kill mode, by design.
        proc, jsonl_path, txt_path = _spawn_helper(tmp_path, n_calls=4)
        try:
            assert jsonl_path.exists()
            assert len(_read_events(jsonl_path)) == 8  # 4 begin + 4 end
            proc.send_signal(signal.SIGKILL)
            ret = proc.wait(timeout=10)
        except Exception:
            proc.kill()
            raise
        assert ret < 0, f"expected SIGKILL exit, got {ret}"
        assert jsonl_path.exists(), "streaming sidecar missing"
        assert not txt_path.exists(), (
            f"llm-stats.txt unexpectedly written under SIGKILL: {txt_path}"
        )
        events = _read_events(jsonl_path)
        for e in events:
            assert e["event"] in ("begin", "end")
            assert e["schema"] == "llm-calls/v1"


# ── Idempotency under atexit ──────────────────────────────────────────────────


class TestIdempotency:
    def test_atexit_after_explicit_write_no_double(self, tmp_path):
        # Helper writes 3 finalizes. We send SIGTERM. The runner-side
        # contract is that even if `_write_llm_stats` is called
        # multiple times, the rollup is materialized once. We can't
        # observe "called twice" directly from outside, but we CAN
        # re-derive the rollup post-mortem and assert the shape is
        # well-formed (no torn / lost records, which would be the
        # failure mode if the writers raced).
        proc, jsonl_path, _txt = _spawn_helper(tmp_path, n_calls=3)
        try:
            proc.send_signal(signal.SIGTERM)
            ret = proc.wait(timeout=15)
        except Exception:
            proc.kill()
            raise
        assert ret == 0
        rollup = _materialize(jsonl_path)
        assert len(rollup["calls"]) == 3
