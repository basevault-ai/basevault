"""
Disk cache for LLM completion calls — keyed on the exact bytes that
would be sent to the provider.

Wrapped into `llm.complete()` (the lowest-level API call) so every
stage benefits uniformly without per-stage glue. Bypass via env var
`BASEVAULT_LLM_CACHE_BYPASS=1`. Storage root: `~/.basevault/cache/`
(override via `BASEVAULT_LLM_CACHE_DIR`).

Cache key construction (`compute_cache_key`):
    sha256(canonical_json({
        "messages": messages,
        "model":    model,
        "temperature": temperature,
        "request_params": stripped_kwargs,
    }))

`stripped_kwargs` excludes wrapper-only fields (_stat_*, payload_tokens)
that affect retry/observability but not the bytes sent to the API. The
hash is computed in `complete()` AFTER kwargs resolution
(max_tokens, reasoning kwargs, override model) so the key reflects
what actually crosses the wire.

Storage layout:
    ~/.basevault/cache/<stage>/<hash>.json
    {
      "response": <stored response text>,
      "model":    <resolved model id>,
      "request_params": <stripped kwargs dict>,
      "computed_at":    <ISO-Z timestamp>,
      "prompt_first_200_chars": <debug-only prefix of last user message>,
    }

The on-disk record is debug-friendly (you can grep `cache/<stage>/`
for the prefix to find a hit) but only `response` and the cache key
itself are load-bearing.

Hit/miss accounting is module-global, thread-safe, and reset by
`reset_cache_stats()` at run start (mirrored in llm.reset_usage_log()).
The runner reads `get_cache_stats()` at run end and writes the totals
into the llm-stats.json rollup (since #165 — pre-#165 they were
mirrored to run.json).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Issue #190: every cache entry has a 30-day TTL. Lookup checks the
# entry's `computed_at` timestamp; if `now > computed_at + CACHE_TTL_S`,
# the lookup treats the entry as a miss AND deletes the file so the
# eviction is permanent (the next miss → store cycle replaces it with
# a fresh entry). Pre-#190 the cache had no TTL and entries persisted
# indefinitely — fine for short-lived dev runs, problematic for
# long-running setups where a model's output drifts over months.
CACHE_TTL_S: int = 30 * 24 * 3600  # 30 days


# ── Configuration ────────────────────────────────────────────────────────────

def _cache_root() -> Path:
    override = os.environ.get("BASEVAULT_LLM_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".basevault" / "cache"


def cache_bypass_enabled() -> bool:
    raw = os.environ.get("BASEVAULT_LLM_CACHE_BYPASS", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# Wrapper-only kwargs that the runner threads through `complete()` for
# stat tagging / token budgeting. They never reach the provider client
# (`complete()` pops them before dispatch) so they MUST NOT influence
# the cache key — otherwise two structurally identical calls from
# different stages would miss each other's cache.
_NON_PAYLOAD_KWARG_KEYS = frozenset({
    "_stat_category",
    "_stat_stage",
    "payload_tokens",
})


def _stripped_kwargs(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k not in _NON_PAYLOAD_KWARG_KEYS}


# ── Hash key ─────────────────────────────────────────────────────────────────

def _canonical_json(obj: Any) -> str:
    """Stable serialization for hashing.

    `sort_keys=True` ensures dict iteration order doesn't leak into the
    key. Every value must be JSON-serializable; provider clients
    sometimes inject objects (e.g. typed enums) — those callers are
    responsible for converting before passing through to `complete()`.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def compute_cache_key(
    provider: str,
    messages: list[dict],
    model: str,
    temperature: float,
    kwargs: dict,
) -> str:
    """Per-(provider, messages, model, temperature, kwargs) cache key.

    Provider is in the payload as defensive structural partitioning so
    a future scenario where two backends serve the same model_id
    string can't silently collide. Today the chat cache is already
    naturally partitioned by ``model`` (Tinfoil's ``kimi-k2-6`` vs
    Ollama's ``qwen3.5:9b`` vs MLX's ``mlx-community/Qwen3.5-9B-4bit``
    are distinct strings), but the partition was implicit; this makes
    it structural — same shape as the embedding key, which carries
    provider for the same reason (its model id IS shared across
    providers, so structural partitioning wasn't optional there).
    """
    payload = {
        "provider": provider,
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "request_params": _stripped_kwargs(kwargs),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return digest


def _cache_path(stage: str | None, key: str) -> Path:
    # Bucket by stage for human navigability and to make per-stage
    # invalidation a `rm -rf <stage>/` away. `<unknown>` covers ad-hoc
    # / test calls that didn't go through llm.set_stage(...).
    bucket = stage if stage else "_unknown"
    # Defensive: stage names are internal, but keep the directory
    # component shell-safe.
    safe_bucket = "".join(c if c.isalnum() or c in ("-", "_") else "_"
                          for c in bucket)
    return _cache_root() / safe_bucket / f"{key}.json"


# ── Hit/miss counters ────────────────────────────────────────────────────────

_stats_lock = threading.Lock()
_hits: dict[str, int] = {}
_misses: dict[str, int] = {}
_stores: dict[str, int] = {}


def reset_cache_stats() -> None:
    with _stats_lock:
        _hits.clear()
        _misses.clear()
        _stores.clear()


def get_cache_stats() -> dict:
    with _stats_lock:
        hits = dict(_hits)
        misses = dict(_misses)
        stores = dict(_stores)
    by_stage = {}
    all_stages = set(hits) | set(misses) | set(stores)
    for stage in all_stages:
        by_stage[stage] = {
            "hits": hits.get(stage, 0),
            "misses": misses.get(stage, 0),
            "stores": stores.get(stage, 0),
        }
    return {
        "hits": sum(hits.values()),
        "misses": sum(misses.values()),
        "stores": sum(stores.values()),
        "by_stage": by_stage,
        "bypass": cache_bypass_enabled(),
    }


def _record(counter: dict[str, int], stage: str | None) -> None:
    bucket = stage if stage else "_unknown"
    with _stats_lock:
        counter[bucket] = counter.get(bucket, 0) + 1


# ── Lookup / store ───────────────────────────────────────────────────────────

def _entry_age_s(rec: dict, fallback_path: Path) -> float | None:
    """Issue #190 TTL check helper. Parses the entry's computed_at
    field (ISO-8601 Z) and returns the age in seconds. Falls back to
    the file's mtime when the field is missing — pre-#190 entries
    don't carry the timestamp, so we treat the file's last-modified
    time as the cached-at moment. Returns None on unparseable input
    so the caller can choose whether to fail-open or fail-closed."""
    raw = rec.get("computed_at") if isinstance(rec, dict) else None
    if isinstance(raw, str):
        try:
            ts = datetime.strptime(
                raw.rstrip("Z"), "%Y-%m-%dT%H:%M:%S",
            ).replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - ts).total_seconds()
        except (ValueError, TypeError):
            pass
    try:
        mtime = fallback_path.stat().st_mtime
    except OSError:
        return None
    return max(0.0, _time.time() - mtime)


def lookup(
    stage: str | None,
    provider: str,
    messages: list[dict],
    model: str,
    temperature: float,
    kwargs: dict,
) -> tuple[str | None, str, dict | None, dict | None]:
    """Return (cached_response_or_None, cache_key, wr_failure_or_None,
    usage_or_None).

    The key is always returned so the caller can pass it to `store()`
    on a miss without recomputing. On bypass, returns (None, key, None,
    None) and increments the miss counter.

    `wr_failure` is the issue #190 non-degrading WR failure memo when
    set (`{"kind": "cap_hit"|"timeout"}`). On a real cache hit of a
    success response, it's None. On a hit of a failure-memo entry, it
    carries the failure shape so `complete()` can synthesize the
    original failure (raise APITimeoutError for "timeout", stamp
    finish_reason="length" for "cap_hit") and the wrapper / stage
    helper drives the halving cascade as it would on a fresh failure
    — without paying for the doomed parent call again.

    `usage` (issue #246) carries the original call's
    `{prompt_tokens, completion_tokens}` when both were known at write
    time, so `complete()` can stamp the cache-hit's per-call rec with
    real token counts (otherwise the per-call detail UI shows 0/0 for
    cache hits even though work was real, just retrieved). Pre-#246
    entries don't carry the field — caller falls back to estimating
    completion_tokens from the cached response text.

    Entries past `CACHE_TTL_S` (30 days) are treated as misses AND
    deleted from disk. Pre-#190 entries without a `computed_at` field
    fall back to the file's mtime so their TTL is still bounded.
    """
    key = compute_cache_key(provider, messages, model, temperature, kwargs)
    if cache_bypass_enabled():
        _record(_misses, stage)
        return None, key, None, None
    p = _cache_path(stage, key)
    if not p.exists():
        _record(_misses, stage)
        return None, key, None, None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, OSError):
        # Corrupt entry: treat as a miss.
        _record(_misses, stage)
        return None, key, None, None
    age = _entry_age_s(rec, p)
    if age is not None and age > CACHE_TTL_S:
        try:
            p.unlink()
        except OSError:
            pass
        _record(_misses, stage)
        return None, key, None, None
    response = rec.get("response")
    if not isinstance(response, str):
        _record(_misses, stage)
        return None, key, None, None
    wr_failure = rec.get("wr_failure")
    if not isinstance(wr_failure, dict):
        wr_failure = None
    pt = rec.get("prompt_tokens")
    ct = rec.get("completion_tokens")
    if isinstance(pt, int) and isinstance(ct, int) and pt > 0 and ct > 0:
        usage: dict | None = {"prompt_tokens": pt, "completion_tokens": ct}
    else:
        usage = None
    _record(_hits, stage)
    return response, key, wr_failure, usage


def _last_user_prefix(messages: list[dict], n: int = 200) -> str:
    """Debug-only: prompt prefix from the last `user`-role message,
    falling back to the last message of any role. Stored in the cache
    file solely so a human grepping ~/.basevault/cache/ can identify
    what a hash represents without re-parsing the JSON payload."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content[:n]
    if messages:
        last = messages[-1]
        if isinstance(last, dict):
            content = last.get("content", "")
            if isinstance(content, str):
                return content[:n]
    return ""


def store(
    stage: str | None,
    cache_key: str,
    response: str,
    provider: str,
    model: str,
    messages: list[dict],
    kwargs: dict,
    wr_failure: dict | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> None:
    """Write a response (or failure memo) to the cache. Bypass mode
    skips storage — `BASEVAULT_LLM_CACHE_BYPASS=1` is "act as if cache
    doesn't exist," not "still seed it for next time."

    `wr_failure` marks the entry as a non-degrading work-reducing
    failure memo. Shape: `{"kind": "cap_hit"|"timeout"}`. On lookup,
    complete() detects the marker and synthesizes the original
    failure shape so the stage's halving cascade fires — without
    re-paying for the doomed parent call. Per the retry spec, only
    non-degrading WR stages (extract / entities-summarize) write
    these memos.

    `prompt_tokens` / `completion_tokens` (issue #246) persist the
    provider's reported usage alongside the response so a future cache
    hit can stamp the per-call rec with real token counts instead of
    falling back to a chars/3 estimate of the cached text. Both must
    be truthy (>0) to be written; partial / zero values are dropped so
    we never persist a misleading "0 out" against a non-empty response.
    """
    if cache_bypass_enabled():
        return
    p = _cache_path(stage, cache_key)
    payload = {
        "response": response,
        "provider": provider,
        "model": model,
        "request_params": _stripped_kwargs(kwargs),
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prompt_first_200_chars": _last_user_prefix(messages),
    }
    if wr_failure is not None:
        payload["wr_failure"] = wr_failure
    if prompt_tokens and completion_tokens:
        payload["prompt_tokens"] = int(prompt_tokens)
        payload["completion_tokens"] = int(completion_tokens)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass
    _record(_stores, stage)


def bust(stage: str | None, cache_key: str) -> bool:
    """Remove a cache entry. Used by stage parsers when a previously
    stored response turns out to be unusable (parse_error). Returns
    True if a file was removed, False otherwise (already absent, missing
    key, IO error). Best-effort; never raises.

    Issue #190 cache hygiene: combined with the success-only gates in
    `complete()`, `bust` lets the parse_error code path in stage code
    keep the cache free of model outputs that look fine on the wire but
    don't survive parsing. Without this the next run of an identical
    prompt would short-circuit on the bad cache entry."""
    if not cache_key:
        return False
    p = _cache_path(stage, cache_key)
    try:
        if p.exists():
            p.unlink()
            return True
    except OSError:
        pass
    return False


# ── Embedding cache (per-input) ──────────────────────────────────────────────
#
# Chat-side cache keys on the full (messages, model, temperature, kwargs)
# payload — one call's worth of bytes. Embeddings are different: one wire
# call carries N independent inputs, each independently embeddable. Caching
# per-batch would miss every time the batch composition shifts; caching
# per-input means an identical record's vector is reused across runs
# (extraction is deterministic, so the same fact text → same embed input
# → same vector).
#
# Same storage primitives, same TTL, same bypass env var; only the key
# shape and bucket differ.

_EMBEDDING_CACHE_BUCKET = "embeddings"


def compute_embedding_cache_key(
    provider: str, model: str, text: str,
) -> str:
    """Per-(provider, model, text) cache key. Embedding inputs are
    independent; each text gets its own entry so partial-batch reuse
    works (a batch of 32 records with 10 already cached fires one wire
    call for the remaining 22, not 32).

    Partitioned by ``provider`` so a vector embedded via one backend
    never silently feeds a run on a different backend. nomic-v1.5
    served by Tinfoil and by Ollama is cosine-compatible (same model,
    same default quant — the settled ruling) but NOT bit-identical,
    and reusing cloud-origin vectors on a LOCAL run is a soft
    violation of the user's mode pick (the wire never leaked, but the
    cached corpus is partially cloud-derived). Earlier slice-A sanction
    of a shared key was reversed after a run showed ``cached: true`` on
    a LOCAL embedding call that had originally been computed via
    Tinfoil — provenance ambiguity outweighed the marginal save of a
    cross-mode hit (rare in practice; daily cache wipe + fast on-device
    embed make a forced miss-then-store cheap).
    """
    payload = {"provider": provider, "model": model, "text": text}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def embedding_lookup(
    provider: str, model: str, text: str,
) -> tuple[list[float] | None, str]:
    """Look up the cached vector for `(provider, model, text)`. Returns
    `(vector_or_None, cache_key)`. Same TTL / bypass / counter
    semantics as the completion lookup; misses still return the
    key so the caller can `embedding_store(...)` on a fresh fetch
    without recomputing it."""
    key = compute_embedding_cache_key(provider, model, text)
    if cache_bypass_enabled():
        _record(_misses, _EMBEDDING_CACHE_BUCKET)
        return None, key
    p = _cache_path(_EMBEDDING_CACHE_BUCKET, key)
    if not p.exists():
        _record(_misses, _EMBEDDING_CACHE_BUCKET)
        return None, key
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _record(_misses, _EMBEDDING_CACHE_BUCKET)
        return None, key
    age_s = _entry_age_s(rec, p)
    if age_s is not None and age_s > CACHE_TTL_S:
        try:
            p.unlink()
        except OSError:
            pass
        _record(_misses, _EMBEDDING_CACHE_BUCKET)
        return None, key
    vec = rec.get("vector")
    if not isinstance(vec, list):
        _record(_misses, _EMBEDDING_CACHE_BUCKET)
        return None, key
    _record(_hits, _EMBEDDING_CACHE_BUCKET)
    return [float(x) for x in vec], key


def embedding_store(
    cache_key: str, provider: str, model: str, text: str, vector: list[float],
) -> None:
    """Persist a freshly-computed vector. Bypass mode skips storage.
    Files land at `<cache_root>/embeddings/<key>.json`. `text`
    prefix is stored in `text_first_200_chars` for debug navigation
    only (the response surface is `vector`). ``provider`` is recorded
    on the entry so a debugger can confirm which backend produced the
    vector — the key partitions by provider too, so cross-provider
    collisions don't happen, but the field is the human-readable
    breadcrumb."""
    if cache_bypass_enabled():
        return
    p = _cache_path(_EMBEDDING_CACHE_BUCKET, cache_key)
    payload = {
        "vector": [float(x) for x in vector],
        "provider": provider,
        "model": model,
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "text_first_200_chars": text[:200],
    }
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass
    _record(_stores, _EMBEDDING_CACHE_BUCKET)
