"""
Tests for the LLM prompt-hash cache (`llm_cache`).

Covers:
- Determinism: identical inputs → identical key, byte-for-byte.
- Wrapper-only kwargs (_stat_*, payload_tokens) don't influence the key.
- lookup/store round-trip; second call returns cached response.
- Bypass env var routes around the disk completely.
- Hit/miss/store counters by stage.
"""
from __future__ import annotations


import pytest

from engine import llm_cache


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Pin the cache root to tmp_path and clear bypass each test."""
    monkeypatch.setenv("BASEVAULT_LLM_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("BASEVAULT_LLM_CACHE_BYPASS", raising=False)
    llm_cache.reset_cache_stats()
    yield


def _msgs(text: str) -> list[dict]:
    return [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": text},
    ]


def test_compute_cache_key_is_byte_identical_across_calls():
    """Same inputs produce the same key on every invocation."""
    msgs = _msgs("hello world")
    kwargs = {"max_tokens": 100, "top_p": 0.9}
    k1 = llm_cache.compute_cache_key("tinfoil", msgs, "model-a", 0.0, kwargs)
    k2 = llm_cache.compute_cache_key("tinfoil", msgs, "model-a", 0.0, kwargs)
    assert k1 == k2
    # Different content → different key
    k3 = llm_cache.compute_cache_key("tinfoil", _msgs("hello other"), "model-a", 0.0, kwargs)
    assert k1 != k3


def test_compute_cache_key_ignores_kwarg_dict_order():
    """Dict iteration order should not leak into the hash."""
    msgs = _msgs("x")
    k_a = llm_cache.compute_cache_key("tinfoil", msgs, "m", 0.0, {"a": 1, "b": 2, "c": 3})
    k_b = llm_cache.compute_cache_key("tinfoil", msgs, "m", 0.0, {"c": 3, "a": 1, "b": 2})
    assert k_a == k_b


def test_compute_cache_key_strips_wrapper_only_kwargs():
    """`_stat_category`, `_stat_stage`, `payload_tokens` are wrapper-only
    metadata — they never reach the provider client and must not
    affect the cache key, otherwise two structurally identical calls
    from different stages would miss each other's cache."""
    msgs = _msgs("x")
    base = llm_cache.compute_cache_key("tinfoil", msgs, "m", 0.0, {"max_tokens": 50})
    with_stat = llm_cache.compute_cache_key("tinfoil", msgs, "m", 0.0, {
        "max_tokens": 50,
        "_stat_category": "single_call",
        "_stat_stage": "actions",
        "payload_tokens": 1234,
    })
    assert base == with_stat


def test_compute_cache_key_changes_with_model_or_temperature():
    """Provider-meaningful fields belong in the key."""
    msgs = _msgs("x")
    base = llm_cache.compute_cache_key("tinfoil", msgs, "m", 0.0, {})
    diff_model = llm_cache.compute_cache_key("tinfoil", msgs, "m2", 0.0, {})
    diff_temp = llm_cache.compute_cache_key("tinfoil", msgs, "m", 0.5, {})
    assert base != diff_model
    assert base != diff_temp


def test_lookup_miss_then_hit_round_trip():
    msgs = _msgs("round trip")
    cached, key, _, _ = llm_cache.lookup("extract", "tinfoil", msgs, "m", 0.0, {})
    assert cached is None
    llm_cache.store("extract", key, "response-text", "tinfoil", "m", msgs, {})
    cached2, key2, _, _ = llm_cache.lookup("extract", "tinfoil", msgs, "m", 0.0, {})
    assert cached2 == "response-text"
    assert key2 == key
    stats = llm_cache.get_cache_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 1
    assert stats["stores"] == 1
    assert stats["by_stage"]["extract"] == {"hits": 1, "misses": 1, "stores": 1}


def test_lookup_per_stage_isolation(tmp_path):
    """Two stages with same key bytes still bucket separately on disk —
    the file path includes the stage. This makes per-stage invalidation
    (rm -rf cache/<stage>/) viable without leaking into other stages."""
    msgs = _msgs("isolated")
    _, key, _, _ = llm_cache.lookup("extract", "tinfoil", msgs, "m", 0.0, {})
    llm_cache.store("extract", key, "extract-resp", "tinfoil", "m", msgs, {})
    cached_other, _, _, _ = llm_cache.lookup("patterns", "tinfoil", msgs, "m", 0.0, {})
    assert cached_other is None  # different stage bucket → miss


def test_bypass_env_var_routes_around_disk(monkeypatch):
    msgs = _msgs("bypassed")
    _, key, _, _ = llm_cache.lookup("extract", "tinfoil", msgs, "m", 0.0, {})
    llm_cache.store("extract", key, "stored-response", "tinfoil", "m", msgs, {})
    # Before bypass: hit.
    cached, _, _, _ = llm_cache.lookup("extract", "tinfoil", msgs, "m", 0.0, {})
    assert cached == "stored-response"

    monkeypatch.setenv("BASEVAULT_LLM_CACHE_BYPASS", "1")
    assert llm_cache.cache_bypass_enabled() is True
    cached_bypass, _, _, _ = llm_cache.lookup("extract", "tinfoil", msgs, "m", 0.0, {})
    assert cached_bypass is None
    # Store under bypass is a no-op; nothing to assert other than no crash.
    llm_cache.store("extract", key, "should-not-write", "tinfoil", "m", msgs, {})

    monkeypatch.delenv("BASEVAULT_LLM_CACHE_BYPASS")
    cached_after, _, _, _ = llm_cache.lookup("extract", "tinfoil", msgs, "m", 0.0, {})
    # Original entry still on disk; bypass-mode store didn't overwrite.
    assert cached_after == "stored-response"


def test_bypass_truthiness_variants(monkeypatch):
    for val in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("BASEVAULT_LLM_CACHE_BYPASS", val)
        assert llm_cache.cache_bypass_enabled() is True, val
    for val in ("0", "false", "off", "no", ""):
        monkeypatch.setenv("BASEVAULT_LLM_CACHE_BYPASS", val)
        assert llm_cache.cache_bypass_enabled() is False, val


def test_corrupt_cache_file_is_treated_as_miss(tmp_path):
    msgs = _msgs("corrupt")
    _, key, _, _ = llm_cache.lookup("extract", "tinfoil", msgs, "m", 0.0, {})
    llm_cache.store("extract", key, "ok-response", "tinfoil", "m", msgs, {})
    # Truncate the on-disk file to invalid JSON.
    cache_file = tmp_path / "extract" / f"{key}.json"
    assert cache_file.exists()
    cache_file.write_text("not valid json", encoding="utf-8")
    cached, _, _, _ = llm_cache.lookup("extract", "tinfoil", msgs, "m", 0.0, {})
    assert cached is None
    stats = llm_cache.get_cache_stats()
    assert stats["misses"] >= 1


def test_chat_cache_key_partitions_by_provider():
    """Same (messages, model, temperature, kwargs) on different
    providers must produce DIFFERENT cache keys. Chat already
    partitions naturally via the model string in production (Tinfoil's
    ``kimi-k2-6`` vs Ollama's ``qwen3.5:9b`` are distinct), but having
    the partition implicit invites a silent collision the day two
    providers ever serve the same wire id. The provider field in the
    payload makes the partition structural — same shape as the
    embedding cache key, where the model id IS shared today."""
    msgs = _msgs("hello")
    k_tinfoil = llm_cache.compute_cache_key("tinfoil", msgs, "m", 0.0, {})
    k_ollama = llm_cache.compute_cache_key("ollama", msgs, "m", 0.0, {})
    k_mlx = llm_cache.compute_cache_key("mlx", msgs, "m", 0.0, {})
    assert k_tinfoil != k_ollama
    assert k_tinfoil != k_mlx
    assert k_ollama != k_mlx
    # Deterministic within a single provider.
    assert k_tinfoil == llm_cache.compute_cache_key(
        "tinfoil", msgs, "m", 0.0, {})


def test_embedding_cache_key_partitions_by_provider():
    """Same (model, text) on two different providers must produce
    DIFFERENT cache keys. Regression: pre-fix the key was
    ``sha256({"model": model, "text": text})`` which collided across
    Tinfoil and Ollama (both register the same wire id
    ``nomic-embed-text``), so a LOCAL run could silently serve a
    vector cached from a prior Tinfoil run — provenance ambiguous + a
    soft trust-posture violation. Partition fixes that."""
    k_tinfoil = llm_cache.compute_embedding_cache_key(
        "tinfoil", "nomic-embed-text", "hello world")
    k_ollama = llm_cache.compute_embedding_cache_key(
        "ollama", "nomic-embed-text", "hello world")
    assert k_tinfoil != k_ollama
    # Same provider + same (model, text) is deterministic.
    k_tinfoil2 = llm_cache.compute_embedding_cache_key(
        "tinfoil", "nomic-embed-text", "hello world")
    assert k_tinfoil == k_tinfoil2


def test_embedding_lookup_store_partition_round_trip(tmp_path, monkeypatch):
    """Storing under one provider doesn't satisfy a lookup under
    another provider for the same (model, text). End-to-end check that
    the partition flows through the disk layer too."""
    monkeypatch.setenv("BASEVAULT_LLM_CACHE_DIR", str(tmp_path))
    _, key_tinfoil = llm_cache.embedding_lookup(
        "tinfoil", "nomic-embed-text", "doc one")
    llm_cache.embedding_store(
        key_tinfoil, "tinfoil", "nomic-embed-text", "doc one",
        [1.0, 0.0, 0.0])
    # Same text + same model, different provider — must miss.
    vec_ollama, key_ollama = llm_cache.embedding_lookup(
        "ollama", "nomic-embed-text", "doc one")
    assert vec_ollama is None
    assert key_ollama != key_tinfoil
    # Tinfoil's own re-lookup hits.
    vec_tinfoil, _ = llm_cache.embedding_lookup(
        "tinfoil", "nomic-embed-text", "doc one")
    assert vec_tinfoil == [1.0, 0.0, 0.0]


def test_reset_cache_stats_clears_counters():
    msgs = _msgs("stats")
    _, key, _, _ = llm_cache.lookup("extract", "tinfoil", msgs, "m", 0.0, {})
    llm_cache.store("extract", key, "x", "tinfoil", "m", msgs, {})
    llm_cache.lookup("extract", "tinfoil", msgs, "m", 0.0, {})
    assert llm_cache.get_cache_stats()["misses"] >= 1
    llm_cache.reset_cache_stats()
    stats = llm_cache.get_cache_stats()
    assert stats == {"hits": 0, "misses": 0, "stores": 0,
                     "by_stage": {}, "bypass": False}
