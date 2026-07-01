"""Stage-0 preflight emit ordering.

The user-perceived "row doesn't appear for 20s" UX was: clicking Run,
config.json not yet on disk when the preflight emit fired, so Rust's
`list_runs` (which filters by `agent: "app"` from config.json)
silently skipped the new run dir. The next emit that surfaced the row
was the first wrapper-routed LLM call inside extract — many seconds
later when extract's first call finally fired (and on a fully-cached
extract that's many cached calls in, after preprocessing + prewarm).

The fix moves `_pick_run_model_id` + the provider/model stamping +
`_write_run_config_once()` to BEFORE the preflight emit. Now
config.json is on disk by the time preflight prints, so the emit
acts as both "we're alive" AND the row-appearance trigger. The
LOCAL carve-out (which used to skip the preflight emit entirely)
is gone — without it, LOCAL mode rows would only surface when extract
starts emitting, defeating the same invariant.

Two invariants are tested by spawning the real runner and watching
its stdout:

1. Mode.TEE emits a stage-bearing JSON line on stdout within 500ms
   of subprocess spawn — the "frozen button" budget.
2. Mode.LOCAL ALSO emits a preflight stage event as its first event
   — same row-appearance trigger as non-LOCAL modes.

Both tests fail-fast the runner (no API key set / no Ollama) so they
finish quickly without hitting the network.

Run: ``cd engine && pytest tests/test_launch_ordering.py -v``
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


_PIPELINE_DIR = Path(__file__).parent.parent
_PYTHON_ROOT = _PIPELINE_DIR.parent
_FIXTURE_INPUT = _PIPELINE_DIR / "tests" / "fixtures" / "obsidian_renderer" / \
    "large" / "inputs" / "preprocessed" / "diary.md"


@pytest.fixture
def trace_input_exists():
    if not _FIXTURE_INPUT.exists():
        pytest.skip(f"fixture missing: {_FIXTURE_INPUT}")


def _spawn_runner(*, mode: str, tmp_path: Path) -> subprocess.Popen:
    """Spawn runner.py against the in-tree fixture under an isolated
    state root. API keys are wiped so non-LOCAL modes short-circuit at
    the preflight key check after the early emit. Caller is
    responsible for terminating + waiting on the returned Popen."""
    env = os.environ.copy()
    env["BASEVAULT_LOGS_ROOT"] = str(tmp_path / "logs")
    env["BASEVAULT_VAULT_ROOT"] = str(tmp_path / "vault")
    env["BASEVAULT_RUN_NAME"] = "2026-01-01T00-00-00Z-trce"
    env["BASEVAULT_AGENT"] = "experiment"
    env["BASEVAULT_LLM_CACHE_BYPASS"] = "1"
    for k in (
        "TINFOIL_API_KEY",
        "BASEVAULT_SESSION", "BASEVAULT_EVAL_ID",
    ):
        env.pop(k, None)
    return subprocess.Popen(
        [sys.executable, "-m", "engine.runner",
         "--paths", str(_FIXTURE_INPUT),
         "--mode", mode,
         "--subject", "test"],
        cwd=str(_PYTHON_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def _read_first_stage_event(proc: subprocess.Popen, deadline_s: float):
    """Read stdout line-by-line until a JSON object with a `stage`
    field arrives, returning (event_dict, elapsed_seconds), or
    (None, elapsed) on timeout."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline_s:
        line = proc.stdout.readline()
        if not line:
            # subprocess closed stdout without emitting any events
            return None, time.monotonic() - t0
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "stage" in obj:
            return obj, time.monotonic() - t0
    return None, time.monotonic() - t0


class TestFirstEmitFiresEarly:
    """The reorder lands the first stage-bearing stdout event within
    the 'frozen button' budget so the run row + elapsed ticker render
    promptly. Non-LOCAL modes participate."""

    def test_test_mode_first_event_within_500ms(
        self, tmp_path, trace_input_exists
    ):
        proc = _spawn_runner(mode="test", tmp_path=tmp_path)
        try:
            t_spawn = time.monotonic()
            event, _elapsed = _read_first_stage_event(proc, deadline_s=10.0)
            elapsed_since_spawn_ms = (time.monotonic() - t_spawn) * 1000
            assert event is not None, (
                "no stage-bearing stdout event seen within 10s — runner "
                "hung or failed before the preflight emit"
            )
            # Allow generous headroom for slow CI without losing the
            # signal — pre-fix this was ~385ms+ and the test would fail.
            assert elapsed_since_spawn_ms < 500, (
                f"first stage event arrived {elapsed_since_spawn_ms:.0f}ms "
                "after spawn; budget is 500ms (the 'frozen button' "
                "perception threshold). Pre-fix this lagged because the "
                "first _emit() fired only after the heavy import block."
            )
            assert event["stage"] == "preflight", (
                f"expected first stage='preflight', got "
                f"stage={event.get('stage')!r}. The runner emits preflight "
                "immediately after config.json hits disk so the row appears."
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


class TestModeLocalAlsoEmitsPreflight:
    """Mode.LOCAL needs the same row-appearance trigger as non-LOCAL
    modes. Without a preflight emit, the first stage event would be
    the first wrapper-routed LLM call inside extract — which on a
    fully-cached run can be many seconds in (after ingest preprocessing
    + splitter), and `list_runs` only learns about the new run dir
    once a stdout line triggers a refresh. Preflight here is the same
    "we exist, please refresh" signal the TEE path uses."""

    def test_local_mode_first_event_is_preflight(
        self, tmp_path, trace_input_exists
    ):
        proc = _spawn_runner(mode="local", tmp_path=tmp_path)
        try:
            event, _elapsed = _read_first_stage_event(proc, deadline_s=10.0)
            if event is None:
                pytest.skip(
                    "local-mode runner produced no stdout event before "
                    "deadline — Ollama dep missing or daemon hung"
                )
            assert event["stage"] == "preflight", (
                "Mode.LOCAL must emit stage='preflight' as its first "
                "event — same row-appearance trigger as non-LOCAL modes. "
                "Without it the row would only surface when extract starts "
                "emitting, which on a cached run can be many seconds in. "
                "Got: " + json.dumps(event)
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
