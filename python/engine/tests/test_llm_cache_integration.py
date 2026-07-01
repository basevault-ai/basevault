"""
Integration tests for the LLM prompt-hash cache wired into
`llm.complete()`.

Exercises the three brief-required validation scenarios end-to-end at
the `complete()` boundary, with the Tinfoil client stubbed to count
calls so we can directly assert "zero LLM calls on identical re-run":

1. Re-run on identical inputs + settings → zero provider calls (all
   responses cached).
2. Bump one stage's setting (different model) → only that stage's
   call busts; upstream stages still cache-hit.
3. Add a new file to inputs (simulated by adding a new
   different-content prompt) → only that file's stage call is uncached;
   existing stages keep cache hits.

These tests turn the cache ON locally (overriding the conftest.py
session bypass) and pin the cache root to a per-test tmp dir so the
counters and disk artifacts are isolated per test.
"""
from __future__ import annotations


import pytest

from engine import llm
from engine import llm_cache


@pytest.fixture
def cache_enabled(tmp_path, monkeypatch):
    """Override the conftest session bypass so the cache is ACTIVE for
    this test, pointed at a fresh tmp dir. Also clear the per-stage
    model routing map so the `model=...` arg passed to `complete()`
    flows through unchanged — otherwise `_resolve_stage_override`
    would coerce all stages back to whatever the active config maps
    them to, defeating the per-stage cache-key isolation we're
    testing."""
    monkeypatch.setenv("BASEVAULT_LLM_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("BASEVAULT_LLM_CACHE_BYPASS", raising=False)
    monkeypatch.setattr(llm, "_STAGE_MODEL_MAP", {})
    llm_cache.reset_cache_stats()


def _msgs(text: str) -> list[dict]:
    return [
        {"role": "system", "content": "you are an assistant"},
        {"role": "user", "content": text},
    ]










# ── Issue #246: cache-hit token stamping ───────────────────────────────────

def _find_rec(call_id: str) -> dict:
    return next(r for r in llm.get_stat_records() if r["call_id"] == call_id)






def test_store_persists_token_counts_for_post_fix_entries(tmp_path, monkeypatch):
    """Unit-level: `store()` writes prompt_tokens / completion_tokens
    into the on-disk payload when both are passed truthy. `lookup()`
    returns them in the 4th tuple slot. Pin the wire shape so a future
    refactor of the cache file format doesn't silently regress #246."""
    monkeypatch.setenv("BASEVAULT_LLM_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("BASEVAULT_LLM_CACHE_BYPASS", raising=False)
    llm_cache.reset_cache_stats()

    msgs = [{"role": "user", "content": "x"}]
    _, key, _, _ = llm_cache.lookup("extract", "tinfoil", msgs, "m", 0.0, {})
    llm_cache.store(
        "extract", key, "resp", "tinfoil", "m", msgs, {},
        prompt_tokens=123, completion_tokens=45,
    )
    cached, _, _, usage = llm_cache.lookup("extract", "tinfoil", msgs, "m", 0.0, {})
    assert cached == "resp"
    assert usage == {"prompt_tokens": 123, "completion_tokens": 45}

    # Zero / None for either count: drop both — never persist a
    # misleading "0 out" against a non-empty response.
    llm_cache.store(
        "extract", key, "resp", "tinfoil", "m", msgs, {},
        prompt_tokens=99, completion_tokens=0,
    )
    _, _, _, usage_partial = llm_cache.lookup("extract", "tinfoil", msgs, "m", 0.0, {})
    assert usage_partial is None
