"""
Per-call streaming + observability tests (issue #104 part 1).

Pins the stream-consumption helper (`_consume_chat_stream`) and the
shape of the per-call observability fields (`reasoning_tokens`,
`content_tokens`, `finish_reason`, `ttft_ms`, `ttfr_ms`,
`last_token_ms`) on three layers:

  - Helper level: feeding `_consume_chat_stream` a synthetic chunk
    iterator yields the right reassembled text + timing + token counts.

  - `complete()` level: a streaming Tinfoil stub flows through to the
    active stat record with all fields populated.

  - Cache short-circuit: a cache hit produces the synthetic
    "single-chunk" observability fingerprint (ttft=0, ttfr=None,
    reasoning_tokens=0, finish_reason="stop") so downstream rollup
    code doesn't need to branch on cache vs not.

No live LLM calls. Run with:
    cd engine && pytest tests/test_per_call_streaming.py -v
"""
from __future__ import annotations



from engine import llm  # noqa: E402
from engine import llm_cache  # noqa: E402



# ── _consume_chat_stream — the reassembly helper ──────────────────────────────


# ── complete() end-to-end through the Tinfoil branch ──────────────────────────


class TestCompleteStreamingFlowsToStatRecord:
    """`complete()` → streaming Tinfoil stub → active stat record. Pins
    the contract that every per-call observability field reaches the
    record (not just the streaming helper)."""

    def setup_method(self):
        # Reset the per-call stat threadlocal — these tests inspect the
        # active record after complete() returns.
        llm.reset_stat_records()
        self._prev_stage = llm._current_stage
        llm.set_stage("extract")

    def teardown_method(self):
        llm.set_stage(self._prev_stage)
        llm.set_calls_jsonl_path(None)


# ── Cache short-circuit — synthetic single-chunk observability ────────────────


class TestCacheHitObservability:
    """A cache hit short-circuits before the provider call. Per the
    issue brief, the active stat record still receives synthetic
    streaming-observability fields so the rollup doesn't fork on cache
    vs not — same shape, ttft=0 (cache lookup is instant), reasoning
    fields zeroed (we don't store reasoning alongside the cached text)."""

    def setup_method(self):
        llm.reset_stat_records()
        self._prev_stage = llm._current_stage
        llm.set_stage("extract")
        llm_cache.reset_cache_stats()

    def teardown_method(self):
        llm.set_stage(self._prev_stage)


