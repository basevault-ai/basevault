"""Offline round-trip for KernelDiskCache (#912).

The disk cache backs both shapes the kernel produces: string content (chat /
synthesis) and ``list[list[float]]`` (a batched embedding's N vectors). Assert
save -> has -> load round-trips the RESPONSE for both, that the call/messages
are NOT stored (load returns a ``None`` call — legacy parity), that an empty
payload is not cached, and that bypass disables the hook.
"""
from __future__ import annotations

import pytest

from kernel.abstractions import LlmCall, LlmResponse
from kernel.enums import LlmStatus

from engine.phases.kernel_cache import KernelDiskCache


def _call(messages):
    return LlmCall("c", messages, 0, "", None, None)


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    # Redirect the cache root + disable bypass so the test owns the dir.
    monkeypatch.setenv("BASEVAULT_LLM_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("BASEVAULT_LLM_CACHE_BYPASS", raising=False)


def test_vector_payload_round_trips():
    cache = KernelDiskCache(stage="embeddings")
    key = "embed-key-1"
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # a 2-record batch's vectors
    call = _call([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    resp = LlmResponse(LlmStatus.OK, vectors, None, 7, 0, 0, None, 0.0)

    assert cache.has(key) is False
    cache.save(key, call, resp)
    assert cache.has(key) is True

    loaded = cache.load(key)
    assert loaded is not None
    cached_call, cached_resp = loaded
    assert cached_call is None  # the call/messages are NOT stored (legacy parity)
    assert cached_resp.payload == vectors
    assert cached_resp.status == LlmStatus.OK


def test_string_payload_still_round_trips():
    cache = KernelDiskCache(stage="extract")
    key = "chat-key-1"
    call = _call([{"role": "user", "content": "hi"}])
    resp = LlmResponse(LlmStatus.OK, "the answer", None, 3, 2, 0, None, 0.0)

    cache.save(key, call, resp)
    loaded = cache.load(key)
    assert loaded is not None
    assert loaded[0] is None  # call not stored
    assert loaded[1].payload == "the answer"


def test_empty_payload_not_cached():
    cache = KernelDiskCache(stage="embeddings")
    call = _call([{"role": "user", "content": "a"}])
    cache.save("k-empty-list", call, LlmResponse(LlmStatus.OK, [], None, 0, 0, 0, None, 0.0))
    cache.save("k-empty-str", call, LlmResponse(LlmStatus.OK, "", None, 0, 0, 0, None, 0.0))
    assert cache.has("k-empty-list") is False
    assert cache.has("k-empty-str") is False


def test_bypass_disables_hook(monkeypatch):
    cache = KernelDiskCache(stage="embeddings")
    key = "embed-key-2"
    call = _call([{"role": "user", "content": "a"}])
    cache.save(key, call, LlmResponse(LlmStatus.OK, [[1.0, 0.0]], None, 0, 0, 0, None, 0.0))
    monkeypatch.setenv("BASEVAULT_LLM_CACHE_BYPASS", "1")
    assert cache.has(key) is False
    assert cache.load(key) is None
