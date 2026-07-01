"""Disk-backed ``CachingHook`` for the kernel path (issue #912).

The legacy ``complete()`` path cached every successful call under
``~/.basevault/cache/`` (``llm_cache``) so a re-run of the same corpus is
near-free. The kernel path bypasses ``complete()``, so without a caching hook
every kernel run pays full freight — and the run-details UI shows zero cache
hits (the symptom that surfaced on the first GUI run: ``cached=true`` on
0/676 kernel calls vs 50/51 on a legacy re-run).

``KernelDiskCache`` restores that. It implements ``kernel.CachingHook``:

* ``save(key, call, response)`` — the kernel calls this only for cacheable
  successes (``RetryPolicy.cache_policy``); we persist ONLY the response
  (status / payload / token counts) under ``key``. The call's messages are
  NOT stored — mirroring legacy ``llm_cache`` (the prompt is captured by dev
  prompt logging / payload capture) and the key already encodes them.
* ``load(key) -> (None, LlmResponse) | None`` — on a hit the kernel
  short-circuits the scheduler and replays the cached response against the
  live call (the ``None`` call skips the messages-equality assertion, which
  ``BoundExecutionEnv`` guards with ``if cached_call``).

Keyspace: the kernel computes its OWN key (``BoundExecutionEnv._cache_key``:
model + thinking + max_tokens + sha256(messages)). ``load`` only receives that
opaque key (no messages), so we cannot bridge into the legacy ``llm_cache``
keyspace (which hashes messages at lookup) — instead this is a self-contained
kernel cache, bucketed under ``<cache_root>/kernel/<stage>/``. It reuses the
legacy cache root, bypass switch, and 30-day TTL so the daily cache wipe and
``BASEVAULT_LLM_CACHE_BYPASS`` govern both keyspaces identically.

Two payload shapes are cached: strings (chat / synthesis content) and
``list[list[float]]`` (a batched embedding's N vectors). Both round-trip
through JSON. For embeddings the key covers the WHOLE batch (the kernel hashes
``call.messages``), so caching is per-batch, not per-record — one changed
record misses its whole batch, coarser than the legacy per-record embed cache.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import override

from kernel.abstractions import CachingHook, LlmCall, LlmResponse
from kernel.enums import LlmStatus

from engine.llm_cache import CACHE_TTL_S, _cache_root, cache_bypass_enabled


def cache_entry_id(kernel_key: str) -> str:
    """sha256-hex of the kernel cache key — the on-disk filename AND the
    per-call record's ``cache_key`` column. Single source so the cache file
    and the run-details Copy/Delete-cache buttons agree."""
    return hashlib.sha256(kernel_key.encode("utf-8")).hexdigest()


class KernelDiskCache(CachingHook):
    def __init__(self, stage: str | None = None):
        # Bucket per-stage for human navigability + per-stage invalidation,
        # mirroring llm_cache._cache_path's layout.
        bucket = stage or "_unknown"
        self._safe_bucket = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in bucket
        )

    def _path(self, key: str) -> Path:
        # Store under <cache_root>/<stage>/<entry_id>.json — the SAME layout
        # llm_cache._cache_path uses (no "kernel/" subdir) so the run-details
        # Copy/Delete-cache buttons work: they call read_llm_cache_entry /
        # bust_llm_cache_entry(stage, cache_key) which compute exactly this
        # path, and the per-call record's cache_key == entry_id (set by the
        # telemetry hook). Coexists with legacy entries — different key
        # derivations never collide as sha256 filenames.
        return _cache_root() / self._safe_bucket / f"{cache_entry_id(key)}.json"

    @override
    def has(self, key: str) -> bool:
        # Cheap existence check (no deserialize) — CombinedSpec dispatch uses
        # it to prefer a constituent model that already has a cached entry.
        if cache_bypass_enabled():
            return False
        return self._path(key).exists()

    @override
    def load(self, key: str) -> tuple[LlmCall, LlmResponse] | None:
        if cache_bypass_enabled():
            return None
        p = self._path(key)
        if not p.exists():
            return None
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError, OSError):
            return None
        computed_at = rec.get("computed_at")
        if isinstance(computed_at, (int, float)):
            if (time.time() - computed_at) > CACHE_TTL_S:
                try:
                    p.unlink()
                except OSError:
                    pass
                return None
        payload = rec.get("payload")
        status_name = rec.get("status")
        # payload is a str (chat / synthesis content) or a list[list[float]]
        # (a batched embedding's N vectors) — both round-trip through JSON.
        if not isinstance(payload, (str, list)):
            return None
        try:
            status = LlmStatus[status_name]
        except (KeyError, TypeError):
            return None
        response = LlmResponse(
            status,
            payload,
            None,
            int(rec.get("prompt_tokens") or 0),
            int(rec.get("completion_tokens") or 0),
            int(rec.get("reasoning_tokens") or 0),
            None,
            0.0,
        )
        # The call (messages) is intentionally NOT stored, mirroring legacy
        # llm_cache: the prompt is captured by other mechanisms (dev prompt
        # logging / payload capture) and the cache KEY already encodes the
        # messages (the kernel hashes call.messages into it), so a stored copy
        # is redundant. Return a None call — the kernel replays the response
        # against the LIVE call and skips the messages-equality assertion when
        # cached_call is None (BoundExecutionEnv: ``if cached_call``).
        return None, response

    @override
    def save(self, key: str, call: LlmCall, response: LlmResponse) -> None:
        # Cache successful payloads: a str (chat / synthesis content) or a
        # list[list[float]] (a batched embedding's N vectors). The kernel
        # already gates this call by cache_policy; guard against empty payloads.
        if not isinstance(response.payload, (str, list)) or not response.payload:
            return
        if response.status is None:
            return
        p = self._path(key)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(
                    {
                        # The call (messages / max_tokens) is intentionally NOT
                        # stored — mirroring legacy llm_cache, the prompt is
                        # captured elsewhere (dev prompt logging / payload
                        # capture) and the cache key already encodes it. Persist
                        # only the response; load returns a None call.
                        "kernel_key": key,
                        "status": response.status.name,
                        "payload": response.payload,
                        # `response` mirrors the legacy llm_cache shape so the
                        # run-details "Copy cache payload" button (which reads
                        # this field) works on kernel entries too. Only string
                        # content is copyable; a vector payload has no text form.
                        "response": (
                            response.payload
                            if isinstance(response.payload, str)
                            else None
                        ),
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "reasoning_tokens": response.reasoning_tokens,
                        "computed_at": time.time(),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass  # cache write is best-effort; never break the call
