"""
Hard per-call wall-clock timeout (issue #692).

A single `entities_dedupe` call once ran 152 min (gemma4 reasoning-on
on a ~75k-token input) and sat `pending` for 2.5h without ever timing
out. Root cause: the per-stage `timeout=` kwarg is, on a STREAMING
response, an httpx PER-READ deadline (max wait between byte chunks) —
a model trickling `reasoning_content` for hours keeps resetting it
every chunk, so it never fires.

The guard under test bounds elapsed time per call regardless of byte
flow, in TWO tiers: 60 min pre-first-token (the dead-before-TTFT hang,
caught faster) and 120 min total once a token has arrived (the slow-
but-producing reasoning grind). Each tier fires via a per-chunk elapsed
check (when chunks ARE flowing) AND a per-call watchdog timer that arms
#684's socket-shutdown abort (the parked-recv case, no chunk to check
between). The abort surfaces as `retry._WallClockTimeout`, which the
classifier routes to LOAD (retried as-is — same input, same model).

The real ceilings are impractical to test live, so each tier is
injectable via its env var (`BASEVAULT_LLM_WALL_CLOCK_NO_TOKEN_S` /
`BASEVAULT_LLM_WALL_CLOCK_TOTAL_S`) and `llm._resolve_wall_clock_timeouts`,
and these tests drive tiny bounds against synthetic over-bound streams
(red→green on the guard).

No live LLM calls. Run with:
    cd engine && pytest tests/test_wall_clock_timeout.py -v
"""
from __future__ import annotations




from engine import llm  # noqa: E402
from engine import runner  # noqa: E402

from engine.tests._streaming_stubs import (  # noqa: E402
    _StubChunk,
    _StubChunkChoice,
    _StubDelta,
    _StubUsage,
)


def _trickle_chunks(n: int = 6):
    """A stream that 'trickles' n content-delta chunks then closes
    cleanly — the shape of a slow reasoning grind. Returned as a fresh
    iterator each call."""
    chunks = [
        _StubChunk(choices=[_StubChunkChoice(delta=_StubDelta(content="."))])
        for _ in range(n)
    ]
    chunks.append(_StubChunk(choices=[
        _StubChunkChoice(delta=_StubDelta(), finish_reason="stop")]))
    chunks.append(_StubChunk(usage=_StubUsage(
        prompt_tokens=10, completion_tokens=n)))
    return iter(chunks)


def _no_token_chunks(n: int = 6):
    """A stream that yields n EMPTY-delta chunks (no content, no
    reasoning) then closes — the dead-before-first-token shape: bytes
    arrive but the model never emits a token, so the NO-TOKEN tier
    governs. Fresh iterator each call."""
    chunks = [
        _StubChunk(choices=[_StubChunkChoice(delta=_StubDelta())])
        for _ in range(n)
    ]
    chunks.append(_StubChunk(choices=[
        _StubChunkChoice(delta=_StubDelta(), finish_reason="stop")]))
    return iter(chunks)


class _RampMonotonic:
    """`time.monotonic` stand-in that advances by `step` every call, so
    the elapsed between ANY two consecutive reads is `step`. Robust
    regardless of how many monotonic reads `complete()` makes before the
    stream loop: the per-chunk wall-clock check is the next read after
    `t0`, so it always sees `step` elapsed — set `step` >> bound to trip
    on the first chunk deterministically (no real sleeps, no flakiness)."""

    def __init__(self, step: float = 1_000_000.0):
        self.n = 0
        self.step = step

    def __call__(self) -> float:
        self.n += 1
        return self.n * self.step


# ── Bound resolution / configurability (the test injectability hook) ──────────


# ── _consume_chat_stream two-tier guard (red→green on the bounds) ──────────────


# ── Watchdog → socket-shutdown → translation ──────────────────────────────────


# ── Classifier: wall-clock timeout → LOAD (live + replay) ─────────────────────


# ── Outcome label: surfaces as "timeout (load)", never silent ─────────────────


class TestOutcomeLabel:
    def test_classify_outcome_timeout_load(self):
        rec = {
            "skipped": False, "aborted": False, "success": False,
            "finish_reason": None, "duration_ms": 7_300_000,
            "ttft_ms": 10,
            "error": {
                "class": "retry._WallClockTimeout",
                "message": "wall-clock timeout after 7300.0s (limit 7200s)",
            },
        }
        assert runner._classify_outcome(rec) == runner.OUTCOME_TIMEOUT_LOAD


# ── complete() wiring on the TEE provider ───────────────────────────────────


class TestCompleteWiringModeAgnostic:
    """Proves the bound is resolved + threaded into the stream consumer
    on the OpenAI-shape provider (Tinfoil) so a real `complete()` aborts
    an over-bound stream."""

    def setup_method(self):
        llm.reset_stat_records()
        llm.clear_skipped()
        self._prev_stage = llm._current_stage
        llm.set_stage("extract")

    def teardown_method(self):
        llm.set_stage(self._prev_stage)
        llm.reset_stat_records()
        llm.clear_skipped()


# ── complete() wiring, local per-chunk path (Ollama) ──────────────────────────


class TestCompleteWiringLocal:
    """The local providers (Ollama / MLX) have no httpx socket to shut
    down, so the per-chunk elapsed check is their only interrupt — a
    SEPARATE copy of the check from `_consume_chat_stream`. This guards
    that copy against drift (Ollama; MLX shares the identical pattern)."""

    def setup_method(self):
        llm.reset_stat_records()
        self._prev_stage = llm._current_stage
        llm.set_stage("extract")

    def teardown_method(self):
        llm.set_stage(self._prev_stage)
        llm.reset_stat_records()

