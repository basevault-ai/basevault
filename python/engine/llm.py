"""
LLM wrapper — switch between Tinfoil (TEE) and local Ollama/MLX.

Production user-data routing is TEE-only or LOCAL. Non-attested cloud
providers are not reachable from this module; the dispatch arms that
once existed are deleted so a code-path bug can't reintroduce them.

Usage:
    from engine.llm import complete, Mode

    response = complete(messages, model="kimi-k2-6", mode=Mode.TEE)
    response = complete(messages, model="qwen3.5:9b", mode=Mode.LOCAL)
"""
from __future__ import annotations

import os
import re
import socket
import time
from contextlib import contextmanager as _contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import threading
from threading import Lock

from engine._dotenv import load as _load_dotenv
_load_dotenv()


# ── Token usage log ────────────────────────────────────────────────────────
# Each `complete()` call appends one record. Caller (runner.py) resets at
# start of a pipeline run and dumps the log at the end so we can attribute
# tokens + derived cost per case. Thread-safe since sweeps fan out.
_usage_log: list[dict] = []
_usage_lock = Lock()
_current_stage: str | None = None  # set by caller (runner.py) before each stage


def reset_usage_log() -> None:
    with _usage_lock:
        _usage_log.clear()


def get_usage_log() -> list[dict]:
    with _usage_lock:
        return list(_usage_log)


# ── Response parsing utilities ────────────────────────────────────────────
def strip_fences(raw: str) -> str:
    """Strip Markdown code fences from an LLM JSON response.

    Three observed response shapes are handled:
    * Bare JSON, no fence: returned whitespace-stripped, unchanged.
    * Single fenced block (``` ```json\\n{...}\\n``` ```): contents
      returned. Leading whitespace before the opening fence is fine.
    * Multiple fenced blocks (model emitted a first draft, then
      second-guessed itself with text like "Wait — I need to redo
      this properly", then emitted a corrected block): return the
      LAST block. Observed on kimi-k2-6 reasoning-off as the cap-hit
      Step-3 fallback (entities_dedupe call 0020, run mexn). The
      previous all-strip-and-join behavior collapsed every fence
      marker, then `json.loads` saw two JSON objects concatenated
      via interstitial prose and raised "Extra data". Picking the
      last block preserves the model's intent.

    Truncated responses (opening fence with no closing fence — cap-hit
    cut the call mid-stream) fall back to stray-marker stripping so
    `json.loads` still fails downstream and the cap-hit cascade fires
    as expected.
    """
    if not raw:
        return ""
    blocks = re.findall(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return re.sub(r"```(?:json)?|```", "", raw).strip()


# ── CompletionResult ──────────────────────────────────────────────────────
# Structured return value of `complete()`. Replaces the previous
# `str`-only return + the `_active_stat_record` threadlocal back-channel
# that wrapper / stage helpers used to read finish_reason, call_id,
# cache_key, etc. (Issue #264.)
@dataclass(frozen=True)
class CompletionResult:
    content: str
    call_id: str | None
    cache_key: str | None
    cached: bool
    finish_reason: str | None
    model: str
    mode: str
    prompt_tokens: int | None
    completion_tokens: int
    reasoning_tokens: int
    reasoning_tokens_source: str | None
    content_tokens: int
    ttft_ms: int | None
    ttfr_ms: int | None
    last_token_ms: int | None
    max_tokens_reserved: int | None


def set_stage(name: str | None) -> None:
    """Tag subsequent `_record_usage` calls with this stage name.

    Thread-safety caveat: `_current_stage` is module-global. Within one
    pipeline process, stages are sequential; within-stage concurrency
    shares the same stage name, so this is safe. Do NOT use across
    concurrent pipelines in the same process.
    """
    global _current_stage
    _current_stage = name


@_contextmanager
def stage_scope(name: str | None):
    """Set `_current_stage` for the duration of the block, then restore
    the prior value.

    Interactive callers (retrieval rerank, chatbot answer composition) run
    `complete()` outside the runner's per-stage `set_stage` bracketing.
    Without a stage tag, `complete()`'s budget computation hits
    `_ratio_for_stage(None)` → `KeyError` (the silent unknown-stage
    fallback was deliberately removed in #357 so a routing bug surfaces
    loudly). These callers wrap their `complete()` in
    `stage_scope("rerank")` / `stage_scope("chatbot")`; both names are
    registered as single-LLM-call stages in `_MAX_RATIO_BY_STAGE`.

    Save/restore (vs a bare `set_stage`) keeps a nested or
    pipeline-adjacent caller from leaking the interactive tag into a
    subsequent pipeline stage. The sidecar is single-shot so the
    restore is belt-and-suspenders there, but `retrieve()` is library
    code and shouldn't mutate global stage state as a side effect.
    """
    global _current_stage
    prior = _current_stage
    _current_stage = name
    try:
        yield
    finally:
        _current_stage = prior


def _record_usage(
    mode: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    call_id: str | None = None,
    reasoning_tokens: int = 0,
    reasoning_tokens_source: str | None = None,
    content_tokens: int | None = None,
    finish_reason: str | None = None,
    ttft_ms: int | None = None,
    ttfr_ms: int | None = None,
    last_token_ms: int | None = None,
    max_tokens_reserved: int | None = None,
) -> None:
    # Update the live stat record (looked up by call_id) BEFORE the
    # legacy log. _record_usage is called from inside complete() once
    # the provider response is parsed, so the record's `model` field
    # gets the per-stage-routed model id (e.g. "kimi-k2-6" for a
    # synthesis stage), not the budget-anchor placeholder.
    #
    # Per-call observability fields (issue #104, part 1):
    #   reasoning_tokens   from usage.completion_tokens_details when the
    #                      provider reports it; falls back to a token
    #                      count of the streamed reasoning_content.
    #   content_tokens     completion_tokens - reasoning_tokens; computed
    #                      here when caller doesn't pass it explicitly.
    #   finish_reason      stop / length / content_filter / etc — load-
    #                      bearing for retry classification (length
    #                      → _CapHitResponse → sizing cascade).
    #   ttft_ms / ttfr_ms / last_token_ms
    #                      monotonic ms from request fire to first
    #                      content delta / first reasoning delta / last
    #                      delta. ttfr separates "server didn't admit"
    #                      from "model degenerated mid-stream".
    if content_tokens is None:
        content_tokens = max(0, int(completion_tokens) - int(reasoning_tokens or 0))
    rec = _get_rec(call_id) if call_id is not None else None
    if rec is not None:
        # Issue #225: protect the begin-time pre-flight estimate from
        # being clobbered by a 0 from a wire-cut response. The provider
        # only reports prompt_tokens when the stream completed cleanly
        # (final usage chunk arrived); on any mid-stream cut we'd
        # otherwise overwrite a real estimate with 0. Truthy values
        # always win — if the provider reports a real count, that's
        # ground truth; the estimate was just a fallback.
        if prompt_tokens:
            rec["prompt_tokens"] = prompt_tokens
        rec["completion_tokens"] = completion_tokens
        rec["model"] = model
        rec["mode"] = mode
        rec["reasoning_tokens"] = int(reasoning_tokens or 0)
        # Source label distinguishes API-reported / streamed-counted /
        # delta-estimated reasoning attribution. Stays None when no
        # reasoning was detected. retry-policy v2 + the per-call UI
        # both branch on this so an "estimated" zero doesn't read the
        # same as an API-reported zero.
        rec["reasoning_tokens_source"] = reasoning_tokens_source
        rec["content_tokens"] = int(content_tokens)
        rec["finish_reason"] = finish_reason
        rec["ttft_ms"] = ttft_ms
        rec["ttfr_ms"] = ttfr_ms
        rec["last_token_ms"] = last_token_ms
        # Actual max_tokens reservation passed to the API for THIS call
        # (post `dynamic_max_tokens()`, post per-spec clamp). The static
        # `budget` snapshot on the stat record carries the per-stage
        # ceiling; this field carries what got reserved on the wire.
        # Together they let a debug bundle reader verify post-hoc that
        # a per-call reservation was sized correctly without re-deriving
        # the formula from the run config.
        rec["max_tokens_reserved"] = max_tokens_reserved
    if not prompt_tokens and not completion_tokens:
        return  # providers sometimes return 0/0; don't log empty records
    with _usage_lock:
        _usage_log.append({
            "stage": _current_stage,
            "mode": mode,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ts": time.time(),
        })


# ── Dev tab: full prompt + response logging ──────────────────────────────
# Off by default. Toggled per-stage via Settings → Development. When ON,
# the stat record gets `full_prompt` (input messages) and/or
# `full_response` (output text) stamped on it for every call to that
# stage. Rendered in the Run Details modal as expandable sub-sections.
#
# WARNING: this deliberately breaks the "stat record is a SHAREABLE
# debug artifact, no message content ever lands here" invariant called
# out at the top of this file. The dev tab disclaimer covers the trade-
# off ("you're on your own"), but anything that bundles llm-stats.json
# / llm-calls.jsonl for sharing MUST strip these fields if the toggle
# was on. Disk impact is real — extract on long runs can produce 100MB+
# of logged prompts per run.
def _dev_logging_for_stage(stage: str | None) -> tuple[bool, bool]:
    """Read config.json and return (log_input, log_output) for `stage`.
    Both False when the toggle is absent / off / config unreadable.
    Reads on every call — config.json is small and this only fires for
    explicitly-enabled stages."""
    if not stage:
        return (False, False)
    cfg = _read_app_config()
    raw = cfg.get("dev_full_prompt_logging") if isinstance(cfg, dict) else None
    if not isinstance(raw, dict):
        return (False, False)
    entry = raw.get(stage)
    if not isinstance(entry, dict):
        return (False, False)
    return (bool(entry.get("input")), bool(entry.get("output")))


def _stamp_full_io(
    call_id: str | None,
    messages: list[dict] | None,
    content: str | None,
) -> None:
    """Stamp full_prompt + full_response on the live stat record when
    the active stage's toggles are ON. No-op otherwise. Called from
    complete() at both cache-hit and cache-miss return points.

    Also streams a `full_io` JSONL event so the offline rollup
    materializer can rebuild the fields from disk alone (e.g. when
    re-rolling-up a SIGKILL'd run). The on-disk event mirrors the
    in-memory shape; the materializer attaches it to the matching call
    record by call_id.

    Side-effect: ALWAYS stashes a (messages, content) snapshot under the
    call_id in `_pending_payloads` so `record_stage_counts` can write a
    full payload retroactively if the call turns out to be a parse_error
    or zero-size output (issue #197). The dev-tab toggle path is
    independent of that auto-log path; we just track whether THIS call
    already wrote a payload via the toggle so we don't double-write."""
    rec = _get_rec(call_id) if call_id is not None else None
    if rec is None:
        return
    log_in, log_out = _dev_logging_for_stage(_current_stage)
    snap_messages: list[dict] | None = None
    if messages is not None:
        # Take the snapshot once: shared between the toggle path and the
        # auto-log buffer. Storing a list-of-dict copy guards against a
        # later caller mutation (kwargs threading, etc.) corrupting the
        # captured prompt.
        snap_messages = [
            {"role": m.get("role"), "content": m.get("content")}
            for m in messages if isinstance(m, dict)
        ]
    written = False
    if log_in or log_out:
        payload: dict = {"call_id": call_id}
        if log_in and snap_messages is not None:
            rec["full_prompt"] = snap_messages
            payload["full_prompt"] = snap_messages
        if log_out and content is not None:
            rec["full_response"] = content
            payload["full_response"] = content
        if "full_prompt" in payload or "full_response" in payload:
            _append_payload_jsonl(payload)
            written = True
    if call_id is not None:
        _stash_pending_payload(call_id, snap_messages, content, written)


# ── Per-call stat records (llm-stats.json + llm-calls.jsonl) ──────────────
# Richer than _usage_log: one record per LLM call attempt, including
# success/failure, duration, structured input/output counts, retry
# linkage. Designed to be a SHAREABLE debug artifact — no message
# content (prompts/responses/fact summaries) ever lands here; only
# stage names, batch indices, topic tags, and hashed-or-public file
# identifiers via `category`.
#
# Three output artifacts (see runner._write_llm_stats schema docstring
# for the full contract):
#   llm-calls.jsonl   append-only NDJSON, EVENT-PER-LINE log:
#                       {event:"begin",  call_id, stage, ...} at begin
#                       {event:"end",    call_id, success, ...}  at finalize
#                       {event:"counts", call_id, input, output}  at stage_counts
#                     A begin without a matching end means the call
#                     was in flight when the run was killed — the
#                     rollup materializer surfaces these as
#                     cancelled records. Survives SIGKILL.
#   llm-stats.json    rollup snapshot at run end (totals, by_stage,
#                     by_model, per-stage statistics, full calls[]).
#                     Idempotent rewrite. Materialized from the
#                     llm-calls.jsonl event log.
#   llm-stats.txt     human-readable monospace summary, sibling of
#                     llm-stats.json. Format owned by runner.
_stats_records: list[dict] = []
_stats_lock = Lock()
# Per-thread mirror of the in-flight stream's content+reasoning parts.
# `_consume_chat_stream` appends each delta as it arrives so the failure-
# payload helper (issue #266) can read whatever was buffered when a stream
# raised mid-iteration. Reset by `begin_stat_record` so each attempt has
# its own buffer; cleared again on the success-path return so the buffer
# can't be misread by a later non-streaming code path. Cache-hit returns
# never populate it (cache hits don't run `_consume_chat_stream`); the
# helper falls back to the `_pending_payloads` snapshot in that case.
_active_partial_response = threading.local()
_call_id_counter = [0]

# Issue #333: per-call user-skip signal. The runner's marker-dir
# poller writes call_ids into this set whenever the Tauri side
# touches `<run_dir>/skipped_calls/<call_id>`. `_consume_chat_stream`
# checks the set per chunk and raises `_SkippedByUser` for the
# matching call_id, which the classifier routes to "skipped" so
# every stage thunk short-circuits — no record_outcome charge, no
# retry. Marker-dir presence is the durable record; this set is the
# in-process fast path so the stream consumer doesn't stat the
# filesystem on every chunk.
_skipped_call_ids: set[str] = set()
_skipped_lock = Lock()

# Per-call live-stream registry. `_consume_chat_stream` registers the
# in-flight OpenAI-SDK Stream object on entry and unregisters in
# finally; the skip path reaches that stream's underlying socket and
# `shutdown()`s it from another thread so a read blocked on the
# pre-first-token wait (tens of seconds on Tinfoil's kimi tier) aborts
# immediately instead of waiting for the next chunk.
#
# Why shutdown and not close: closing the httpx response from another
# thread does NOT interrupt a recv() already blocked on the first byte
# — it only takes effect on the next read attempt, which never comes
# during the pre-TTFT wait. `socket.shutdown(SHUT_RDWR)` is the POSIX
# mechanism that wakes a blocked recv (verified ~1ms on both plain TCP
# and TLS-wrapped sockets). The blocked iterator then raises an
# httpx error which the consumer translates to `_SkippedByUser`.
#
# httpx evicts the shutdown connection from its pool on the resulting
# read error, so the next call gets a fresh connection — no pool
# poisoning.
_live_streams: dict[str, object] = {}
_live_streams_lock = Lock()

# Per-call hard wall-clock timeout. A per-call watchdog
# timer (armed in `_consume_chat_stream`) reuses the SAME socket-shutdown
# abort path as the user-skip above, but fired by a timer instead of a
# ✕ click. When it fires it registers the call_id here and shuts the
# stream's socket down; the consumer's exception wrapper observes the
# registration and translates the resulting iteration error to
# `_WallClockTimeout` (vs. `_SkippedByUser` for the skip path). The
# per-chunk elapsed check in the loop is the partner trigger for the
# case where chunks ARE flowing (the trickled-reasoning grind that
# motivated this) — it raises `_WallClockTimeout` directly without
# needing the socket teardown.


def _register_live_stream(call_id: str, stream: object) -> None:
    """Make `stream` reachable from the skip path so a marker arriving
    mid-iteration can abort it. Caller guarantees a matching
    `_unregister_live_stream` in a finally."""
    with _live_streams_lock:
        _live_streams[call_id] = stream


def _unregister_live_stream(call_id: str) -> None:
    """Drop the registry entry. Idempotent."""
    with _live_streams_lock:
        _live_streams.pop(call_id, None)


def _stream_socket(stream: object):
    """Reach the underlying socket of an in-flight OpenAI-SDK stream so
    the skip path can `shutdown()` it. Returns None when the transport
    doesn't expose a socket (the caller then relies on the per-chunk
    skip check — no worse than the pre-shutdown behavior).

    Path: the SDK Stream holds the `httpx.Response`; httpx exposes the
    live network stream via `response.extensions["network_stream"]`,
    and httpcore's stream exposes the raw socket via
    `get_extra_info("socket")`. Tinfoil wraps a standard `httpx.Client`
    with a custom pinned-TLS context; TLS connections return the
    `SSLSocket`, which is shutdown-able."""
    try:
        response = getattr(stream, "response", None)
        if response is None:
            return None
        network_stream = response.extensions.get("network_stream")
        if network_stream is None:
            return None
        return network_stream.get_extra_info("socket")
    except Exception:
        return None


def _abort_stream_socket(call_id: str) -> None:
    """Shut down the socket of the in-flight stream registered for
    `call_id`, interrupting a recv blocked on the pre-first-token wait
    (or a mid-stream stall) so the call aborts immediately instead of
    hanging until the provider responds. Shared by the user-skip path
    (`register_skip`) and the wall-clock watchdog
    (`_trigger_wall_clock_timeout`) — both reach a blocked read the same
    way; only the marker set they consult (and the translated exception)
    differs.

    No-op when no stream is registered or the transport doesn't expose a
    socket — in the latter case the partner per-chunk check still fires
    once a chunk arrives, no regression vs. the pre-shutdown behavior.

    Snapshots the stream under the registry lock, then does the socket
    teardown outside it (no reason to serialize a network shutdown
    behind unrelated registry traffic)."""
    with _live_streams_lock:
        stream = _live_streams.get(call_id)
    if stream is None:
        return
    sock = _stream_socket(stream)
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        # Already torn down / mid-teardown. Idempotent — the consumer's
        # exception wrapper observes the registered marker and converts
        # any resulting iteration error to the right class.
        pass


def register_skip(call_id: str) -> None:
    """Mark `call_id` as user-skipped and abort any in-flight stream
    for that call. Two effects:

      1. The next chunk read by `_consume_chat_stream` for this call
         raises `_SkippedByUser` via the per-chunk check (covers the
         post-first-token case where chunks are already flowing).
      2. If a stream is registered for this call, its socket is
         `shutdown()`-ed from this thread (`_abort_stream_socket`) —
         interrupting a recv blocked on the pre-first-token wait so the
         call aborts within the marker-poll cadence instead of hanging
         until the provider responds or times out. The blocked iterator
         raises an httpx error which the consumer's exception wrapper
         translates to `_SkippedByUser`.

    The runner's marker-dir poller calls this; tests can call it
    directly to drive the skip path without writing markers."""
    with _skipped_lock:
        _skipped_call_ids.add(call_id)
    _abort_stream_socket(call_id)


def clear_skipped() -> None:
    """Wipe the skip set and the live-stream registry. Called at
    runner-startup so a leaked id from a prior run / test doesn't haunt
    the next one."""
    with _skipped_lock:
        _skipped_call_ids.clear()
    with _live_streams_lock:
        _live_streams.clear()


def _get_rec(call_id: str | None) -> dict | None:
    """Look up an in-memory stat record by call_id. Returns None if no
    matching record exists (caller can decide to noop). Threading: the
    lookup runs under `_stats_lock` so it doesn't race with appends in
    `begin_stat_record`; the returned dict is mutated in-place by the
    streaming consumer / `_record_usage` / `finalize_stat_record`,
    which is safe because each call_id is unique to one logical call."""
    if call_id is None:
        return None
    with _stats_lock:
        for r in _stats_records:
            if r["call_id"] == call_id:
                return r
    return None

# Path to the streaming .jsonl. Set by the runner once `_run_dir` is
# known (`set_calls_jsonl_path`); None disables streaming (tests, ad-
# hoc scripts, anything not running inside `runner.run()`).
_calls_jsonl_path: "Path | None" = None
_calls_jsonl_lock = Lock()

# Sibling stream for full_prompt + full_response payloads (issue #195).
# Split out so the metadata file (`llm-calls.jsonl`) stays small +
# analyzable; payloads can balloon to 50KB+ per call when the per-stage
# dev-tab toggle is on. None disables the payload stream (tests, ad-hoc
# scripts).
_payloads_jsonl_path: "Path | None" = None
_payloads_jsonl_lock = Lock()

# Opt-in human-readable YAML companion to the jsonl payload stream. The
# chat sidecar turns this on (low-volume, legibility matters); pipeline /
# perf runners leave it off (high-volume, the jsonl is enough). The YAML
# emits ONE document per chat turn (= one user-message round-trip),
# regardless of how many LLM calls a turn fired — buffered between an
# explicit `begin_payloads_yaml_turn()` from the sidecar and either the
# next `begin` (auto-flush) or an explicit `flush_payloads_yaml()`.
_payloads_yaml_path: "Path | None" = None
_payloads_yaml_lock = Lock()
# In-flight turn state per YAML path: {"turn", "session_id", "calls",
# "started_at", "started_mono"}. `calls` is a list of dicts (call_id,
# system_label, request, response, started_at, duration_ms) buffered
# until the turn is flushed. ``started_at`` is the wall-clock ISO-8601
# timestamp at turn begin (so a reader can pin it to a real moment);
# ``started_mono`` is the monotonic counterpart used to compute the
# turn duration at flush time without clock-skew risk.
_payloads_yaml_turn: "dict[Path, dict]" = {}

# Per-call timing side channel, populated by the sidecar's
# ``_tracked_complete`` wrapper around ``complete()`` and read by
# ``_append_payload_yaml`` so each call entry in the YAML can carry a
# wall-clock start timestamp and elapsed duration. Keyed by call_id;
# entries are popped on read so the dict stays bounded. The same data
# also lands in ``llm-calls.jsonl`` via the stat-record path — this is
# the chat-readable mirror, not a second source of truth.
_payloads_yaml_call_timings: "dict[str, dict]" = {}

# Set of call_ids for which a payload has already been streamed to
# llm-payloads.jsonl (any source — `_stamp_full_io`'s toggle-on write,
# `record_stage_counts`'s auto-log on parse_error / empty / interrupted
# / zero-output, or `_log_call_failure_payload`'s wire-failure write).
# `_log_call_failure_payload` (issue #266) consults this to avoid a
# duplicate record when one of the prior paths already wrote for the
# same call_id (most commonly: parser inside the wrapper raises a
# `_PostStreamFailure` → record_stage_counts writes first, then the
# wrapper's failure path runs unconditionally; the set check prevents
# a second record). Reset by `reset_stat_records`.
_payload_call_ids_written: set = set()
_payload_call_ids_written_lock = Lock()

# In-memory buffer for the auto-log-on-failure path (issue #197). Maps
# call_id -> {"messages": [...], "content": str, "written": bool}.
# Populated by _stamp_full_io for every successful response; consumed
# (and popped) by record_stage_counts when parse_error or zero-size
# output is detected. Kept OFF the stat record so it never leaks into
# llm-stats.json — the toggle-gated `full_prompt` / `full_response` rec
# fields are still the only intentional disk leaks of message bytes.
# Bounded by `_PENDING_PAYLOADS_MAX` to keep peak memory bounded if a
# stage skips record_stage_counts (the buffer entry would otherwise sit
# until process exit). FIFO eviction tolerates the rare case where a
# call's auto-log signal arrives after the cap fills. Bypass cases:
# stage helpers that don't call record_stage_counts at all leave their
# entry sitting until cap eviction or reset_stat_records (between
# runs/tests) — the failure-trigger window for that call is lost.
from collections import OrderedDict as _OrderedDict  # noqa: E402
_pending_payloads: "_OrderedDict[str, dict]" = _OrderedDict()
_pending_payloads_lock = Lock()
_PENDING_PAYLOADS_MAX = 64


def _stash_pending_payload(
    call_id: str,
    messages: list[dict] | None,
    content: str | None,
    written: bool,
) -> None:
    """Buffer (messages, content) for a call so record_stage_counts can
    later auto-log a payload on parse_error / zero-size. `written=True`
    means the dev-tab toggle already streamed a payload for this call;
    record_stage_counts then skips its auto-log to avoid double-writing
    (issue #197)."""
    with _pending_payloads_lock:
        _pending_payloads[call_id] = {
            "messages": messages,
            "content": content,
            "written": written,
        }
        while len(_pending_payloads) > _PENDING_PAYLOADS_MAX:
            _pending_payloads.popitem(last=False)


def _output_is_zero_size(output: dict | None) -> bool:
    """Detect zero-size output: the producing stage emitted no entries.

    Each stage's `output` dict starts with the primary count (facts,
    patterns, entities, merges, actions, insights). We treat the call
    as zero-size when that first integer leaf is 0. Bool values and
    non-int leaves (sub-dicts, floats, strings) are skipped — bool
    subclasses int so we'd otherwise misread `parse_error: True` as a
    primary count, and capacity fields (`total_cap`, `cross_cap`,
    `critical_cap`, `max_actions_cap`) are typically non-leading
    numerics that shouldn't gate the trigger.

    Single source of truth for the "is this an empty success?" rule.
    Used both by the auto-payload-log trigger here in llm.py AND by
    `_classify_outcome` in runner.py to route `OUTCOME_SUCCESS_EMPTY`
    — keeping the two consumers on one helper prevents the kept-count
    convention from drifting between them."""
    if not isinstance(output, dict):
        return False
    for v in output.values():
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            return v == 0
    return False


def set_calls_jsonl_path(p) -> None:
    """Runner sets this once `_run_dir` is known so begin/finalize/counts
    can stream events to a sibling NDJSON file. Pass None to disable
    streaming (tests reset between cases). Idempotent."""
    global _calls_jsonl_path
    _calls_jsonl_path = p


def set_payloads_jsonl_path(p) -> None:
    """Runner sets this once `_run_dir` is known so the dev-tab full
    prompt + response logger can stream to `llm-payloads.jsonl`. Pass
    None to disable the payload stream (tests reset between cases).
    Idempotent."""
    global _payloads_jsonl_path
    _payloads_jsonl_path = p


def payloads_yaml_turn_floor() -> int:
    """Highest ``turn: N`` value already on disk in the YAML, or 0 if
    the file is absent / unset / unreadable. Sidecar reads this at
    session boot to seed its turn counter past any prior sessions of
    the same chat (perma-id) — so turn numbering stays monotonic
    across sidecar restarts instead of resetting to 1 on every process
    spawn. Same role as ``bootstrap_call_id_counter_from_jsonl`` but
    for the per-turn YAML document boundary.
    """
    p = _payloads_yaml_path
    if p is None:
        return 0
    try:
        if not p.exists():
            return 0
        import re as _re
        max_turn = 0
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                m = _re.match(r"^turn:\s*(\d+)\s*$", line)
                if m:
                    n = int(m.group(1))
                    if n > max_turn:
                        max_turn = n
        return max_turn
    except OSError:
        return 0


def set_payloads_yaml_path(p) -> None:
    """Opt-in human-readable YAML companion to the jsonl payload stream.
    The YAML emits ONE document per chat turn (= one user-message
    round-trip), with every LLM call of that turn listed under a
    ``calls:`` list — so both the decision call and the grounded call
    are visible per turn, and slice-2's multi-hop turns get all their
    hops in one doc. Pass ``None`` to disable.

    Sidecar contract: call ``begin_payloads_yaml_turn(turn, session_id)``
    once per user message before any ``complete()`` fires, then call
    ``flush_payloads_yaml()`` on each return path out of the turn body
    (early return + normal completion both flush). Two safety nets cover
    the cases an explicit flush misses: the next ``begin_payloads_yaml_turn``
    auto-flushes any in-flight buffer left over from a crash, and
    ``atexit.register(flush_payloads_yaml)`` writes whatever's buffered
    if the process exits mid-turn.
    """
    global _payloads_yaml_path
    _payloads_yaml_path = None if p is None else Path(p)
    if _payloads_yaml_path is not None:
        _payloads_yaml_turn.pop(_payloads_yaml_path, None)


def begin_payloads_yaml_turn(turn: int, session_id: str) -> None:
    """Mark the start of a chat turn for the YAML writer. Flushes any
    in-flight turn on this path first (defensive — a crash mid-turn
    would otherwise leave its buffer dangling and the next turn would
    inherit those calls). No-op when no YAML path is set.

    Captures both wall-clock and monotonic start times so the flushed
    turn doc carries an ISO-8601 ``started_at`` (real moment a reader
    can pin to) and a clock-skew-free ``duration_ms`` (computed via
    ``time.monotonic`` at flush)."""
    p = _payloads_yaml_path
    if p is None:
        return
    import time as _time
    with _payloads_yaml_lock:
        _flush_yaml_turn_locked(p)
        _payloads_yaml_turn[p] = {
            "turn": int(turn),
            "session_id": str(session_id or ""),
            "calls": [],
            "started_at": _iso_now(),
            "started_mono": _time.monotonic(),
        }


def _iso_now() -> str:
    """UTC ISO-8601 timestamp with seconds resolution, ``Z`` suffix —
    matches the format used in ``llm-calls.jsonl`` so a reader can
    cross-reference the two streams by start time."""
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def flush_payloads_yaml() -> None:
    """Explicit end-of-turn flush. Idempotent; safe to call from
    ``atexit`` for a final flush on shutdown."""
    p = _payloads_yaml_path
    if p is None:
        return
    with _payloads_yaml_lock:
        _flush_yaml_turn_locked(p)


def emit_cycle_event(event: str, payload: dict) -> None:
    """Emit a cycle-boundary event into the llm-calls.jsonl log so
    downstream consumers can split a multi-cycle file (cancel+resume,
    replay-validation) into per-cycle slices, and so the post-#165
    derivation can read run state without consulting run.json.

    Allowed event names:
      cycle_start      runner started a fresh / resumed cycle
      cycle_end        runner finished naturally
      cycle_cancelled  user-initiated cancel landed (Rust-side or runner-side)
      cycle_error      runner / Rust observed a fatal error mid-cycle
      entities_decision   stage stamped subject_resolution / bundle_mode

    No-op when `_calls_jsonl_path` is unset (matches `_append_event_jsonl`'s
    contract). Tests patch `_calls_jsonl_path` directly.

    See issue #52 (multi-cycle jsonl scope) + #165 (run.json retirement)."""
    allowed = (
        "cycle_start", "cycle_end",
        "cycle_cancelled", "cycle_error",
        "entities_decision",
        # progress_tick — runner emits one per `_emit()` call (LLM-call
        # boundaries + stage transitions). Carries bar_position /
        # eta_seconds / total / stage so the Tauri runs-list can
        # surface progress on stages that don't fan out begin/end
        # events fast enough (insights / actions are single-call —
        # once begin lands no more events fire until the call ends).
        "progress_tick",
    )
    if event not in allowed:
        raise ValueError(
            f"emit_cycle_event: unknown event {event!r}; expected one "
            f"of {allowed}"
        )
    _append_event_jsonl(event, payload)


def count_cycle_starts_in_jsonl(path: "Path | None" = None) -> int:
    """Count `cycle_start` events in a jsonl file. Used by the rollup
    to surface `cycles_count` so consumers see the multi-cycle scope
    at a glance.

    `path` defaults to the active `_calls_jsonl_path` (the in-process
    runner's stream); on-demand readers (eval/judge.py, ad-hoc
    scripts) pass an explicit path so derivation works against a
    foreign run dir without setting module globals.

    Pre-marker historical jsonl files have zero cycle_start records;
    callers default to 1 in that case (single-cycle assumption — that's
    the regime the file format was originally designed for)."""
    p = path if path is not None else _calls_jsonl_path
    if p is None:
        return 0
    try:
        if not p.exists():
            return 0
        text = p.read_text(encoding="utf-8")
    except OSError:
        return 0
    count = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("event") == "cycle_start":
            count += 1
    return count


def bootstrap_call_id_counter_from_jsonl() -> int:
    """If `_calls_jsonl_path` is set and the file already has events
    on disk, advance `_call_id_counter` past the highest call_id seen
    so a resumed session emits non-colliding ids.

    Resume-by-run-id re-uses the prior session's run dir, including
    its `llm-calls.jsonl`. Without this bootstrap, session 2 would
    restart at 0001 and collide with session 1's ids; the rollup
    materializer keys on call_id alone, so collisions yield mis-paired
    begin/end events (wrong stage, wrong durations).

    Always safe to call: missing file or fresh run → returns 0 and
    leaves the counter alone. Only `begin` events are scanned (each
    pair has exactly one begin). Returns the new floor for callers
    that want to log it."""
    p = _calls_jsonl_path
    if p is None:
        return 0
    try:
        if not p.exists():
            return 0
        text = p.read_text(encoding="utf-8")
    except OSError:
        return 0
    max_id = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("event") != "begin":
            continue
        cid = ev.get("call_id")
        if not isinstance(cid, str):
            continue
        try:
            n = int(cid)
        except ValueError:
            continue
        if n > max_id:
            max_id = n
    if max_id <= 0:
        return 0
    with _stats_lock:
        if _call_id_counter[0] < max_id:
            _call_id_counter[0] = max_id
    return max_id


def _append_event_jsonl(event: str, payload: dict) -> None:
    """Append one event line to llm-calls.jsonl. Append-only is
    naturally crash-safe: SIGKILL between calls leaves a complete
    prefix; no atomic-replace dance, no partial-write window across
    a record boundary.

    Thread-safe via `_calls_jsonl_lock` — concurrent emissions from
    the runner threadpool would otherwise interleave bytes from
    partial json.dumps writes.

    Open mode "a" each call. Per-event open/close is negligible at
    peak ~50 calls/sec × 3 events; simpler FD lifecycle than holding
    a handle (no leaked handles on early exit, no flush worries).

    Three event types:
      begin   emitted from begin_stat_record  (call started)
      end     emitted from finalize_stat_record (call ended, with
              success/duration/error/tokens but NOT input/output —
              record_stage_counts runs after finalize)
      counts  emitted from record_stage_counts (input/output counts
              attached, OPTIONAL per stage)

    A begin without an end means the call was in flight when the
    process was killed — the rollup materializer surfaces these as
    cancelled records. SIGKILL between begin and end leaves the
    begin on disk; the rollup picks it up if it later runs (e.g.
    via atexit on a SIGTERM-cancelled run).

    Every line carries an ISO-Z `ts` stamped here at write time so
    downstream consumers can correlate the jsonl with app.log + any
    other clock-stamped artifact without relying on order-in-file as
    a proxy for order-in-time. If a caller supplies its own `ts`
    (test fixtures pin literal values), that wins."""
    p = _calls_jsonl_path
    if p is None:
        return
    ts = payload.get("ts") or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    body = {k: v for k, v in payload.items() if k != "ts"}
    line = json.dumps(
        {"event": event, "ts": ts, **body, "schema": "llm-calls/v1"},
        default=str,
    )
    with _calls_jsonl_lock:
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _append_payload_jsonl(payload: dict) -> None:
    """Append one full_io record to llm-payloads.jsonl. Sibling of
    `_append_event_jsonl` — same crash-safety + lock discipline, but
    routes to the dedicated payload stream so `llm-calls.jsonl` stays
    small (issue #195: full_io records were 82% of file size at 2% of
    record count and drowned the structured metadata stream).

    No `event` field: the file is single-purpose, every record is a
    full_io payload. `schema` stays so future format changes can
    discriminate.

    The ``.jsonl`` write is gated by ``_payloads_jsonl_path``; when
    unset the file is skipped (tests, ad-hoc scripts, and the chat
    sidecar — chat reads the per-turn YAML and the structured calls
    stream, never the jsonl bytes). The YAML companion + call-id
    bookkeeping still fire regardless, so a YAML-only consumer keeps
    its surface even when the jsonl path is off.
    """
    p = _payloads_jsonl_path
    if p is not None:
        line = json.dumps(
            {**payload, "schema": "llm-payloads/v1"}, default=str,
        )
        with _payloads_jsonl_lock:
            with open(p, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    # Opt-in human-readable companion (chat sidecar enables; pipeline /
    # perf runners leave off).
    _append_payload_yaml(payload)
    # Track that a payload landed for this call_id so duplicate writers
    # (issue #266: failure-path helper running after record_stage_counts
    # auto-log) can short-circuit. Bare set; `reset_stat_records` clears
    # between runs / tests.
    cid = payload.get("call_id")
    if cid is not None:
        with _payload_call_ids_written_lock:
            _payload_call_ids_written.add(cid)


def _yaml_block(key: str, text: str, indent: int = 0) -> str:
    """Render ``key: |2- <text>`` as a YAML literal block scalar.
    ``indent`` is the column where ``key`` sits; body lines are indented
    two more spaces (the standard nested-block convention). Empty text
    degrades to ``key: ''`` so the file stays valid.

    The ``2`` after ``|`` is the explicit indentation indicator (relative
    to the parent node): without it, YAML auto-detects the indent from
    the first content line, so a body whose first line happens to begin
    with whitespace would be mis-parsed. Setting it explicitly pins the
    content indent regardless of the body's leading characters.
    """
    key_pad = " " * indent
    body_pad = " " * (indent + 2)
    if not text:
        return f"{key_pad}{key}: ''\n"
    body = "\n".join(body_pad + line for line in text.split("\n"))
    return f"{key_pad}{key}: |2-\n{body}\n"


def _append_payload_yaml(payload: dict) -> None:
    """Buffer this call into the in-flight turn's ``calls:`` list.
    Actual file write happens when the turn flushes (next
    ``begin_payloads_yaml_turn`` or explicit ``flush_payloads_yaml``).
    No-op when no YAML path is set or no turn has been begun."""
    p = _payloads_yaml_path
    if p is None:
        return
    msgs = payload.get("full_prompt") or []
    if not isinstance(msgs, list):
        return
    last_user = next(
        (m.get("content", "") for m in reversed(msgs)
         if m.get("role") == "user"),
        "",
    )
    sys_msg = next(
        (m.get("content", "") for m in msgs if m.get("role") == "system"),
        "",
    )
    resp = payload.get("full_response") or ""
    # System-persona label, derived from prompt-builder contracts the
    # personas honor: (1) only grounded turns inject the
    # ``CONTEXT (numbered):`` block into the last user message;
    # (2) only personas authorized to issue tool calls (``decision`` and
    # ``grounded_decision``) include the tool-protocol's
    # ``Available tools`` directory in their system prompt —
    # ``grounded_final`` omits it. The three-way split is exact under
    # those contracts and avoids plumbing an explicit persona kwarg
    # through ``complete()`` for what is fundamentally a telemetry label.
    has_context = "CONTEXT (numbered):" in last_user
    has_tool_protocol = "Available tools" in sys_msg
    if not has_context:
        system_label = "decision"
    elif has_tool_protocol:
        system_label = "grounded_decision"
    else:
        system_label = "grounded_final"

    call_id = str(payload.get("call_id") or "")
    # Pull the side-channel timing the sidecar recorded after this
    # complete() returned. Absent for code paths that don't go through
    # ``_tracked_complete`` (tests, ad-hoc scripts) — the YAML entry
    # then just omits the timing fields.
    timing = _payloads_yaml_call_timings.pop(call_id, None) if call_id else None

    with _payloads_yaml_lock:
        state = _payloads_yaml_turn.get(p)
        if state is None:
            # Caller forgot to begin a turn. Be defensive: open an
            # anonymous turn so the call isn't silently dropped.
            state = {"turn": -1, "session_id": "", "calls": []}
            _payloads_yaml_turn[p] = state
        call_entry = {
            "call_id": call_id,
            "system_label": system_label,
            "request": last_user,
            "response": resp,
        }
        if timing is not None:
            call_entry["started_at"] = timing["started_at"]
            call_entry["duration_ms"] = timing["duration_ms"]
        state["calls"].append(call_entry)


def _flush_yaml_turn_locked(p: "Path") -> None:
    """Write the buffered turn (if any) to ``p`` as one YAML document
    and clear the buffer. Caller holds ``_payloads_yaml_lock``.

    The turn doc carries timestamps + durations at two scopes: a
    turn-level ``started_at`` + ``duration_ms`` (set at
    ``begin_payloads_yaml_turn`` / computed at flush via the monotonic
    clock) and a per-call ``started_at`` + ``duration_ms`` for each
    entry under ``calls:`` (pulled from
    ``_payloads_yaml_call_timings`` when ``_append_payload_yaml`` was
    invoked).
    """
    state = _payloads_yaml_turn.pop(p, None)
    if state is None or not state["calls"]:
        return
    import time as _time
    turn_started = state.get("started_at") or ""
    turn_started_mono = state.get("started_mono")
    turn_duration_ms: int | None = None
    if isinstance(turn_started_mono, float):
        turn_duration_ms = int((_time.monotonic() - turn_started_mono) * 1000)
    parts: list[str] = ["---\n"]
    parts.append(f"turn: {state['turn']}\n")
    parts.append(f"session_id: {state['session_id']}\n")
    if turn_started:
        parts.append(f"started_at: {turn_started}\n")
    if turn_duration_ms is not None:
        parts.append(f"duration_ms: {turn_duration_ms}\n")
    parts.append("calls:\n")
    for c in state["calls"]:
        parts.append(f"  - call_id: {c['call_id']}\n")
        parts.append(f"    system: {c['system_label']}\n")
        if c.get("started_at"):
            parts.append(f"    started_at: {c['started_at']}\n")
        if c.get("duration_ms") is not None:
            parts.append(f"    duration_ms: {c['duration_ms']}\n")
        parts.append(_yaml_block("request", c["request"], indent=4))
        parts.append(_yaml_block("response", c["response"], indent=4))
    with open(p, "a", encoding="utf-8") as f:
        f.write("".join(parts))


# json import deferred to avoid pulling json at module top — already
# imported via `_read_app_config`'s `import json as _json`.
import json  # noqa: E402


def _log_call_failure_payload(
    call_id: str | None,
    messages: list[dict] | None,
) -> None:
    """Issue #266: write one llm-payloads.jsonl record for a call that
    raised. Called from the runner wrapper's exception path so every
    failed attempt — timeout, transport error, attestation failure,
    `_raise_if_empty_on_wire` / `_raise_if_interrupted_on_wire`,
    `wr_failure` cache-hit replay — leaves its full prompt on disk for
    debugging. Independent of the dev-tab toggle: failures are rare
    enough that always-on is the right default for the case where you
    most need the prompt.

    `full_response` is included only when partial bytes were buffered
    before the raise (mid-stream cut on `_consume_chat_stream`); omitted
    when no bytes flowed (per-spec — don't fabricate empty).

    Cache-hit replay coordination: `_stamp_full_io` runs at the cache-
    hit return point (3301) BEFORE the synthetic `wr_failure` raise, so
    when the dev-tab toggle is ON the success-path write already landed.
    A pending-payloads entry with `written=True` short-circuits this
    helper to avoid a duplicate record. Toggle-OFF cache-hit replays
    still get a record from this helper (new behavior).

    No-op when `call_id` is None (defensive — wrapper always has one)."""
    if call_id is None:
        return
    # Deconfliction check #1: `record_stage_counts`'s parse_error /
    # empty / interrupted / zero-output auto-log path (issue #197)
    # runs BEFORE this helper when the parser inside the wrapper
    # raised a `_PostStreamFailure`. That path already streamed a
    # record to llm-payloads.jsonl and announced the call_id via the
    # shared written-set. Bail so we don't double-write the call_id.
    with _payload_call_ids_written_lock:
        if call_id in _payload_call_ids_written:
            return
    with _pending_payloads_lock:
        pending = _pending_payloads.pop(call_id, None)
    # Deconfliction check #2: `_stamp_full_io` with the dev-tab
    # toggle ON already wrote a payload AND stashed pending with
    # `written=True`. The written-set check above usually catches
    # this first (`_append_payload_jsonl` adds to the set), but the
    # flag is the structural signal coming from `_stamp_full_io`
    # itself — defending both paths keeps the contract obvious even
    # if a future caller writes a payload outside `_append_payload_jsonl`.
    if pending is not None and pending.get("written"):
        return
    snap_messages: list[dict] | None = None
    if pending is not None and pending.get("messages") is not None:
        snap_messages = pending["messages"]
    elif messages is not None:
        snap_messages = [
            {"role": m.get("role"), "content": m.get("content")}
            for m in messages if isinstance(m, dict)
        ]
    parts = getattr(_active_partial_response, "parts", None)
    streamed_partial = "".join(parts) if parts else ""
    full_response: str | None = None
    if pending is not None and pending.get("content"):
        full_response = pending["content"]
    elif streamed_partial:
        full_response = streamed_partial
    if snap_messages is None and full_response is None:
        return
    payload: dict = {"call_id": call_id}
    # Mirror the chat-session id (set at begin) so a failed call's
    # payload groups by session like its success-path sibling. Reads the
    # in-memory rec; absent / None for pipeline-runner calls (no key
    # added, payload shape unchanged for non-chatbot consumers).
    _fail_rec = _get_rec(call_id)
    if _fail_rec is not None and _fail_rec.get("session_id"):
        payload["session_id"] = _fail_rec["session_id"]
    if snap_messages is not None:
        payload["full_prompt"] = snap_messages
    if full_response is not None:
        payload["full_response"] = full_response
    _append_payload_jsonl(payload)


def reset_stat_records() -> None:
    with _stats_lock:
        _stats_records.clear()
        _call_id_counter[0] = 0
    with _pending_payloads_lock:
        _pending_payloads.clear()
    with _payload_call_ids_written_lock:
        _payload_call_ids_written.clear()


def get_stat_records() -> list[dict]:
    with _stats_lock:
        return list(_stats_records)


def begin_stat_record(
    stage: str | None,
    category: str | None,
    model_hint: str,
    attempt: int = 1,
    retry_of_call_id: str | None = None,
    retry_delay_ms: int = 0,
    budget: dict | None = None,
    request_extras: dict | None = None,
    template_hash: str | None = None,
    prompt_tokens_est: int | None = None,
    session_id: str | None = None,
    max_tokens_reserved: int | None = None,
    provider: str | None = None,
) -> str:
    """Open a new stat record for an LLM call. The runner wrapper calls
    this before invoking complete(). Returns the call_id; caller should
    pass it back to finalize_stat_record() AND thread it to complete()
    via the `_call_id` kwarg so the streaming consumer / `_record_usage`
    can mutate the rec by id (issue #264 dropped the thread-scoped slot
    that previously routed those writes).

    `request_extras` carries the materially-relevant call kwargs the
    wrapper resolved (reasoning state, temperature if non-default,
    explicit max_tokens, etc.) so a debug bundle reader can answer
    "why was this call slow / expensive" without correlating against
    config.json. Empty / None elides the field from the begin event so
    older readers don't see a new key on every line.

    `prompt_tokens_est` (issue #225) is the caller's pre-flight estimate
    of input tokens (chars/3 via tokens.estimate_prompt_tokens). Stamped
    onto the live rec as `prompt_tokens` immediately so the field carries
    a useful number even when the response stream cuts before usage
    lands. `_record_usage` overwrites with the provider's reported value
    only when truthy (> 0); a 0 from a wire-cut response never clobbers
    the estimate. Same protection in `finalize_stat_record` and the
    rollup materializer. Pre-this-fix, wire-cut responses surfaced as
    `0 in / 0 out` in the run-details UI even though the prompt was
    fully built and sent.

    `session_id` (issue #503) demarcates a chatbot conversation: the
    sidecar passes a per-conversation id so every begin/end (and, via
    the rec, failure-payload) record groups by chat session, making one
    session's calls selectable without guessing the cutoff in the
    append-only chatbot log. Stamped on the rec and the begin event only
    when supplied — pipeline-runner calls never pass it, so their event
    lines are byte-unchanged (no new key for older readers).
    """
    with _stats_lock:
        _call_id_counter[0] += 1
        call_id = f"{_call_id_counter[0]:04d}"
        started_at_iso = _now_iso_utc()
        rec = {
            "call_id": call_id,
            "stage": stage,
            "category": category,
            "model": model_hint,
            # `mode` is the legacy field (filled by `_record_usage` on
            # wire-success only — null on cache-hit and failure paths).
            # Kept for backward compat with older readers; new code
            # consumes `provider` below, which is the finer-grained
            # backend tag (mode collapses MLX+Ollama into "local"
            # whereas provider distinguishes them).
            "mode": None,
            # Stamped at begin so cache-hit and failure paths (which
            # skip `_record_usage`) carry the actual backend the call
            # ran against — otherwise short-circuit paths would
            # surface as "unknown provider" in run-details and the UI
            # couldn't tell post-hoc whether a particular call ran
            # local-Ollama, local-MLX, attested-Tinfoil, etc. Caller
            # passes a string (e.g. `_provider_str(spec.provider)`); None for
            # tests/ad-hoc calls that don't resolve a spec.
            "provider": provider,
            "started_at_iso": started_at_iso,
            "started_at_human": _now_human_local(),
            "duration_ms": None,
            # success starts None — finalize_stat_record sets True/False.
            # A None on disk means the call never finalized (process
            # killed mid-call). Consumers that need a strict boolean
            # should treat None as failure-like.
            "success": None,
            "error": None,
            # `prompt_tokens` carries the caller's pre-flight estimate
            # at begin time (issue #225). `_record_usage` overwrites
            # ONLY with truthy provider values — a 0 from a wire-cut
            # response leaves the estimate intact so the run-details UI
            # never shows `0 in` for a call that genuinely sent a
            # multi-thousand-token prompt. The estimate uses
            # tokens.estimate_prompt_tokens (chars/3); over-counts
            # slightly relative to most real tokenizers but is
            # self-consistent with the splitter / cap formulas.
            "prompt_tokens": (
                int(prompt_tokens_est)
                if prompt_tokens_est is not None
                else None
            ),
            "completion_tokens": None,
            # Per-call streaming observability (issue #104 part 1).
            # All None at begin so a record cancelled mid-stream surfaces
            # the absence cleanly (vs. zero, which would say "ran to
            # completion with no tokens").
            "reasoning_tokens": None,
            "reasoning_tokens_source": None,
            "content_tokens": None,
            "finish_reason": None,
            "ttft_ms": None,
            "ttfr_ms": None,
            "last_token_ms": None,
            # Actual `max_tokens` reservation the call will pass to the
            # API (`dynamic_max_tokens()` output, post-clamp). The runner
            # wrapper computes it pre-begin and threads it in so the live
            # rec + the begin event carry the reservation while the call
            # is still in flight (UI surfaces it on pending rows, not
            # just on completion). `_record_usage` later overwrites with
            # the same value at end-of-call. None when the runner didn't
            # pre-compute (Ollama path uses num_predict, set by complete();
            # caller-set `max_tokens` short-circuits this too). Pre-this-
            # PR jsonl files don't carry the field — materializer reads
            # it as None which the rollup naturally skips.
            "max_tokens_reserved": (
                int(max_tokens_reserved)
                if max_tokens_reserved is not None
                else None
            ),
            "budget": dict(budget) if budget else {
                "stage_cap": None, "max_output": None, "scaffolding": None,
                "has_llm_calls": None,
            },
            "input": None,
            "output": None,
            "attempt": attempt,
            "retry_of_call_id": retry_of_call_id,
            "retry_delay_ms": int(retry_delay_ms),
            "request_extras": dict(request_extras) if request_extras else None,
            "template_hash": template_hash,
            "parse_error": False,
            # None for pipeline-runner calls; set by the chatbot sidecar
            # so finalize + the failure-payload helper can mirror it onto
            # the end / payload records by reading the rec.
            "session_id": session_id,
        }
        _stats_records.append(rec)
    # Reset the per-attempt partial-response buffer (issue #266). Each
    # attempt streams into a fresh list so the failure-payload helper
    # only ever sees this attempt's bytes — the previous attempt's
    # content (if any) was already consumed by its own end-of-call
    # handler.
    _active_partial_response.parts = []
    # Stream the begin event so an in-flight call leaves a smoking-gun
    # record on disk if the process is killed before finalize_stat_record
    # runs (rbw8 had 6+ extract calls in flight at SIGKILL with no jsonl
    # trace; the rollup materializer pairs begin/end and surfaces
    # unmatched begins as cancelled).
    begin_payload = {
        "call_id": call_id,
        "stage": stage,
        "category": category,
        "model": model_hint,
        "started_at_iso": started_at_iso,
        "attempt": attempt,
        "retry_of_call_id": retry_of_call_id,
        "retry_delay_ms": int(retry_delay_ms),
        "budget": dict(budget) if budget else None,
        "template_hash": template_hash,
    }
    if request_extras:
        begin_payload["request_extras"] = dict(request_extras)
    # Issue #225: streamed pre-flight prompt-token estimate so the
    # offline rollup materializer can repopulate `prompt_tokens` even
    # when the end event reports 0 (wire-cut). Only emitted when caller
    # passed a value — older jsonl files without the field stay
    # backward-compatible (materializer treats absent as None).
    if prompt_tokens_est is not None:
        begin_payload["prompt_tokens_est"] = int(prompt_tokens_est)
    # Streamed pre-call max_tokens reservation so the UI shows it on
    # in-flight rows. Mirrors `prompt_tokens_est`: only emitted when
    # the wrapper supplied a value, keeping older begin lines unchanged
    # for backward-compat readers.
    if max_tokens_reserved is not None:
        begin_payload["max_tokens_reserved"] = int(max_tokens_reserved)
    # Elided when absent so pipeline-runner begin lines stay unchanged
    # for older readers; present only on chatbot-sidecar calls.
    if session_id:
        begin_payload["session_id"] = session_id
    # Streamed so in-flight calls in the run-details UI surface the
    # actual backend the call ran against (cache-hit + failure paths
    # would otherwise read provider:null until end). Only emitted when
    # caller passed a value — older readers without the field stay
    # backward-compatible.
    if provider is not None:
        begin_payload["provider"] = provider
    _append_event_jsonl("begin", begin_payload)
    return call_id


def finalize_stat_record(
    call_id: str,
    success: bool,
    duration_ms: int,
    error: dict | None = None,
) -> None:
    """Close out a stat record. Sets success / duration / (on failure)
    error fields. Tokens were already populated by _record_usage on
    success. Caller should not mutate the record after this returns.

    `success` is derived from the caller's signal AND `error is None`
    so the per-call boolean is internally consistent: aggregators
    that bucket by `record["success"]` agree with aggregators that
    bucket by `record["error"] is None`. The pre-fix serializer left
    success absent on every record, forcing consumers to derive it
    from `error` (which works for the totals aggregator but is
    invisible at the per-call level)."""
    with _stats_lock:
        rec = next((r for r in _stats_records if r["call_id"] == call_id), None)
    if rec is None:
        return
    rec["duration_ms"] = int(duration_ms)
    if error is not None:
        rec["error"] = error
    rec["success"] = bool(success and error is None)
    # On failure tokens were never recorded by the provider; leave as
    # None so consumers can distinguish "0 tokens used" from "call
    # never produced a usage record".
    # Stream the end event. Survives SIGKILL / sigsegv / a stuck
    # atexit — each event fsyncs before the next call's begin. No-op
    # if the runner hasn't called set_calls_jsonl_path (tests, ad-hoc
    # scripts). NOTE: input/output are NOT here — record_stage_counts
    # fires AFTER finalize and emits a separate "counts" event.
    end_payload = {
        "call_id": rec["call_id"],
        "duration_ms": rec["duration_ms"],
        "success": rec["success"],
        "error": rec["error"],
        "prompt_tokens": rec["prompt_tokens"],
        "completion_tokens": rec["completion_tokens"],
        "model": rec["model"],
        "mode": rec["mode"],
        # Backend tag stamped at begin so cache-hit and failure paths
        # carry it; `mode` (above) stays for backward compat and is
        # wire-success-only via _record_usage.
        "provider": rec.get("provider"),
        # Streaming observability (issue #104 part 1). Captured in
        # `_record_usage` from the streamed response. None on failure
        # paths that never reached the response (attestation, early
        # network errors). retry-policy v2 reads `finish_reason` +
        # `ttfr_ms` to discriminate work-reducing retries.
        "reasoning_tokens": rec.get("reasoning_tokens"),
        "reasoning_tokens_source": rec.get("reasoning_tokens_source"),
        "content_tokens": rec.get("content_tokens"),
        "finish_reason": rec.get("finish_reason"),
        "ttft_ms": rec.get("ttft_ms"),
        "ttfr_ms": rec.get("ttfr_ms"),
        "last_token_ms": rec.get("last_token_ms"),
        "max_tokens_reserved": rec.get("max_tokens_reserved"),
        # `cached` is set on the live record by `complete()` when the
        # call short-circuited on the prompt-hash cache. Surface it in
        # the event log so the materializer can preserve it on the
        # llm-stats.json calls[] entry — without this, cache hits look
        # identical to provider calls in the per-call view (only their
        # 0/0 token + ~0ms duration give them away). Fall back to
        # False so non-cache callers don't grow a key.
        "cached": bool(rec.get("cached", False)),
        # `cache_key` lets the per-call detail UI invalidate just this
        # entry via the bust_llm_cache_entry Tauri command. None for
        # calls that never reached the cache layer (early-failure,
        # tests that monkeypatch complete()).
        "cache_key": rec.get("cache_key"),
        # Time we slept BEFORE launching this attempt (set by the
        # wrapper after a backoff). None on the first attempt of any
        # wrapper invocation — surfaces in the per-call detail
        # panel as "waited Xs before this call."
        "retry_delay_ms": rec.get("retry_delay_ms"),
    }
    # Mirror the chat-session id stamped at begin onto the end record so
    # a reader groups paired begin/end by session. Elided when absent so
    # pipeline-runner end lines stay byte-unchanged for older readers.
    if rec.get("session_id"):
        end_payload["session_id"] = rec["session_id"]
    # The kernel's categorized outcome (`LlmStatus.name`, stamped on the
    # rec by KernelTelemetryHook before this finalize). Carry it onto the
    # end event so the run-details rollup can bucket the failure label
    # (load / sizing / other) off the kernel's own classification instead
    # of collapsing every retry to `other` — pre-fix it lived only on the
    # in-memory rec and never reached llm-calls.jsonl. Elided when absent
    # (non-kernel callers, tests) so those end lines stay unchanged.
    if rec.get("llm_status"):
        end_payload["llm_status"] = rec["llm_status"]
    _append_event_jsonl("end", end_payload)


# The stat-record fields a post-stream failure can stamp. Mirrors
# the `stat_field` class attributes on `retry._PostStreamFailure`'s
# subclasses; an unrecognized `failure_kind` is a silent no-op.
_FAILURE_KIND_FIELDS = frozenset(
    {"parse_error", "empty_response", "interrupted"})


def record_stage_counts(
    call_id: str | None,
    input: dict | None = None,
    output: dict | None = None,
    failure_kind: str | None = None,
) -> None:
    """Stage-level helper: attach structured input/output counts to the
    LLM call identified by `call_id`. Called by stage code AFTER the
    LLM response has been parsed, so the parser-side counts (e.g.
    output={"facts": N}) are known. Caller passes the call_id from
    the `CompletionResult` returned by `complete()`. Silently no-ops
    when `call_id` is None or doesn't match a known record (tests that
    exercise pure parsing without a call context).

    `failure_kind` flags the call body as a post-stream failure and
    is one of:

      "parse_error"    — parser-rejected (json.JSONDecodeError,
                          schema mismatch, etc). Materializer buckets
                          the call as `parse_error`.
      "empty_response" — model returned a literal empty /
                          whitespace-only response on the wire.
                          Distinct from `success_empty` (parseable
                          JSON with zero entries): the former renders
                          red, the latter gray.
      "interrupted"    — partial-stream cut: bytes flowed but the
                          stream closed cleanly without a terminating
                          finish_reason chunk and below
                          max_tokens_reserved. Distinct from
                          `empty_response` (no bytes at all) and
                          `cap_hit` (finish_reason="length").

    An unrecognized value is a no-op on the failure path (counts
    still record)."""
    rec = _get_rec(call_id) if call_id is not None else None
    if rec is None:
        return
    if input is not None:
        rec["input"] = input
    if output is not None:
        rec["output"] = output
    is_failure = failure_kind in _FAILURE_KIND_FIELDS
    if is_failure:
        rec[failure_kind] = True
        # Cache hygiene per the retry spec: don't cache load
        # failures, degrading sizing failures, or post-stream
        # failures. Bust the just-stored entry so a re-run can't
        # short-circuit on cached bad content. cache_key was
        # stamped by complete() / vision on both hit and miss paths.
        ck = rec.get("cache_key")
        if ck:
            try:
                from engine.llm_cache import bust as _cache_bust
                _cache_bust(rec.get("stage"), ck)
            except Exception:
                pass
    # Stream a counts event so the .jsonl-only recovery path (e.g.
    # SIGKILL after finalize but before atexit) still has the
    # post-finalize input/output counts available to the materializer.
    if input is not None or output is not None or is_failure:
        _append_event_jsonl("counts", {
            "call_id": rec["call_id"],
            "input": input,
            "output": output,
            "parse_error": True if failure_kind == "parse_error" else None,
            "empty_response": True if failure_kind == "empty_response" else None,
            "interrupted": True if failure_kind == "interrupted" else None,
        })
    # Auto-log full prompt + response on a post-stream failure.
    # Pops the (messages, content) snapshot stashed by `_stamp_full_io`
    # regardless of trigger so the buffer doesn't accumulate across the
    # run; only writes a payload record when this call hit a failure
    # signal AND the dev-tab toggle didn't already write one.
    call_id = rec["call_id"]
    with _pending_payloads_lock:
        pending = _pending_payloads.pop(call_id, None)
    if pending is None:
        return
    if pending["written"]:
        return
    if not (is_failure or _output_is_zero_size(output)):
        return
    payload: dict = {"call_id": call_id}
    if pending["messages"] is not None:
        payload["full_prompt"] = pending["messages"]
    if pending["content"] is not None:
        payload["full_response"] = pending["content"]
    if "full_prompt" in payload or "full_response" in payload:
        _append_payload_jsonl(payload)


from engine.common.dates import (  # noqa: E402
    now_human_local as _now_human_local,
    now_iso_z_ms as _now_iso_utc,
)


class Mode(str, Enum):
    """User-facing operational category — 2 values.
    Maps 1-to-1 with the UI's mode picker.

    TEE routes through an attested Tinfoil enclave; LOCAL runs on the
    user's own machine via the bundled MLX backend (default) or Ollama
    (opt-in). No non-attested cloud route exists in the production
    binary."""
    LOCAL  = "local"
    TEE    = "tee"


class Provider(str, Enum):
    """The backend service an individual call goes to. Orthogonal to
    Mode but pinned 1-to-1 in production: Mode.TEE → TINFOIL
    (attested), Mode.LOCAL → MLX (primary, bundled) or OLLAMA (opt-in).
    Direct connections only — no aggregator gateway, which would add a
    trust party and reintroduce a single point of failure across
    backends.

    ``ModelSpec.provider`` is typed ``Provider | str`` so the eval tree
    can register extension specs against bare-string provider tags
    defined on its own side (see ``register_modelspec``); the
    production enum stays minimal and never names a non-attested
    backend."""
    TINFOIL = "tinfoil"
    OLLAMA  = "ollama"
    MLX     = "mlx"


def _provider_str(p: "Provider | str") -> str:
    """Return the string label of a provider regardless of whether
    it's a ``Provider`` enum member or a bare-string extension tag
    registered by the eval tree."""
    return p.value if isinstance(p, Provider) else str(p)


@dataclass(frozen=True)
class ModelSpec:
    """All properties of a (provider, model) combo the rest of the
    code needs to size calls + handle quirks. One ModelSpec is
    pinned per Mode — see MODE_SPEC below.

    ``provider`` accepts either a ``Provider`` enum value (production
    backends) or a bare string (extension specs registered by the eval
    tree). The dispatch path matches on equality, so both forms work
    transparently; the type widening is what lets the eval add specs
    for backends the production enum doesn't name.

    `max_output` defaults to `context_window` — the right answer for
    providers that don't enforce a separate output cap (Tinfoil, Local).
    Set it explicitly only when the provider documents a tighter
    engine-arg cap."""
    provider: "Provider | str"
    model_id: str
    context_window: int          # total ctx we plan against (input + output)
    # Sentinel: 0 → resolved to `context_window` in __post_init__.
    # Real-world max_output is always > 0, so 0 unambiguously means
    # "use the default."
    max_output: int = 0          # hard cap on max_tokens in our call shape
    require_streaming: bool = False

    def __post_init__(self):
        if self.max_output == 0:
            object.__setattr__(self, "max_output", self.context_window)


# ── Reasoning whitelist ───────────────────────────────────────────────────────
#
# Default: reasoning OFF EVERYWHERE. Every operational run with reasoning on
# has been slow / expensive / broken (Kimi at 768s wall-clock, gpt-oss
# blowing 28k tokens of reasoning into empty JSON, etc).
#
# Two-gate model now:
#   1. _REASONING_WHITELIST: (provider, model_id, stage) tuples where reasoning
#      ON has been verified safe. Acts as a hard allow-list — combos NOT in
#      this set NEVER turn reasoning on regardless of user config.
#   2. Per-stage user toggle in config.json's `stage_models.<stage>.reasoning`.
#      Defaults to False per stage (see _STAGE_DEFAULTS); user opts in via
#      Settings.
#
# Final answer: reasoning_on = whitelisted(provider, model, stage) AND user_toggle.
#
# Whitelist a combo only after personally testing:
#   1. The model emits clean output (no <think> blocks leaking into
#      delta.content) at that provider with reasoning on.
#   2. The reasoning budget fits within dynamic_max_tokens without
#      truncating visible output.
#
# Stage names: vision, extract, entities, entities_dedupe, patterns,
# insights, actions. All Tinfoil models with a meaningful reasoning
# control surface (gpt-oss reasoning_effort, kimi/gemma/glm
# chat_template_kwargs) are pre-registered for every stage. The whitelist
# exists to keep reasoning OFF for models without verified controls (e.g.
# minimax — always on, no kwarg). The user's per-stage toggle is what
# actually flips it on at call time.
#
# vision is a first-class reasoning-toggle stage like every other: its
# Settings row exposes the same model + reasoning controls, and the only
# model the vision dropdown offers is kimi-k2-6 (the dedicated Tinfoil
# vision model was dropped — no attested vision backend today). Omitting
# "vision" here left the user's vision reasoning toggle dead — clickable +
# persisted but forced off at `_reasoning_enabled_for`'s whitelist gate —
# the same trap entities_dedupe hit.
_REASONING_STAGES = ("vision", "extract", "entities", "entities_dedupe",
                     "patterns", "insights", "actions")

# Stages whose work-reducing strategy is non-degrading (input is
# halved + fanned out, no information loss). Issue #190 caches
# WR failures on these stages so a re-run short-circuits straight
# into the halving cascade. Other stages use degrading WR (sample
# 50% least-frequent items) — sampling failures aren't cached
# because the next run would prefer to try the original input
# again.
_WR_NON_DEGRADING_STAGES: frozenset = frozenset({"extract", "entities"})
_REASONING_WHITELIST: set[tuple[str, str, str]] = {
    (Provider.TINFOIL.value, m, s)
    for m in ("gpt-oss-120b", "kimi-k2-6", "gemma4-31b", "glm-5-2")
    for s in _REASONING_STAGES
}


def _reasoning_enabled_for(spec: ModelSpec, stage: str | None) -> bool:
    """True iff reasoning should be ON for this (spec, stage) call.

    Pipeline stages are two-AND-gated:
      1. (provider, model_id, stage) is in `_REASONING_WHITELIST` — the
         hard allow-list of verified-safe combos.
      2. The user's per-stage `reasoning` toggle in config.json's
         `stage_models` is True for this stage.

    The interactive ``chatbot`` surface is NOT a pipeline stage: its
    reasoning is config-driven straight from the top-level ``chatbot``
    field (``resolve_chatbot_from_config``), with no whitelist gate and
    no per-call force in either direction — the user's Settings toggle
    flows verbatim to the model, and the ship-default for that field is
    reasoning-ON (a directed, interactive-surface-only divergence from
    the pipeline's reasoning-OFF-everywhere default). The pipeline
    whitelist deliberately does not list "chatbot"; routing it through
    that gate is what made the toggle a no-op.
    """
    if stage is None:
        return False
    if stage == "chatbot":
        # Lazy import: llm → chatbot → retrieval → llm is a load-time
        # cycle; at call time all three modules are resolved. Reusing
        # the resolver keeps the ship-default single-sourced in
        # chatbot.DEFAULT_CHATBOT_REASONING.
        from engine.chatbot import resolve_chatbot_from_config
        return bool(
            resolve_chatbot_from_config(_read_app_config()).get("reasoning")
        )
    # #628 escape hatch for the regression suite: when the gpt-medium
    # profile is active, the harness sets BASEVAULT_FORCE_REASONING=1 to
    # pin reasoning ON without the user-config toggle gate. Bypasses the
    # whitelist too — extract_items()/etc. don't expose the per-call
    # _force_reasoning_on kwarg, so an env-var gate is the minimal path.
    # Off in production (unset); only the regression stage runners set it.
    if os.environ.get("BASEVAULT_FORCE_REASONING") == "1":
        return True
    if (_provider_str(spec.provider), spec.model_id, stage) not in _REASONING_WHITELIST:
        return False
    stage_cfg = _STAGE_MODEL_MAP.get(stage)
    if not stage_cfg:
        return False
    return bool(stage_cfg.get("reasoning", False))


def _reasoning_kwargs(spec: ModelSpec, enabled: bool) -> dict:
    """Translate "reasoning on/off" into the per-model parameter the
    inference engine actually understands. Each branch documents one
    model family's control mechanism. Models not listed below have no
    verified control mechanism — they run with provider defaults.

    Returns a kwargs delta to merge into the chat completion call.
    """
    model = spec.model_id.lower()
    if "gpt-oss" in model:
        # OpenAI's open-weights uses reasoning_effort. "low" is the
        # documented floor (no fully-off). The per-stage on/off toggle
        # exposes a single binary switch; we map ON=medium (NOT high),
        # because at "high" gpt-oss-120b burned 28,407 completion tokens
        # of pure reasoning into an empty JSON response. Medium is the
        # "spend extra latency for synthesis depth" middle ground; high
        # is reserved for explicit per-(provider,model,stage) whitelist
        # entries that have already verified dynamic_max_tokens accounts
        # for the reasoning budget.
        #
        # Measurement-scoped override (gated, off by default): the
        # eval_perf benchmark must characterize the off→medium→high
        # speed/cost curve, but "high" is otherwise unreachable. When
        # EVAL_PERF_GPT_OSS_EFFORT is set to one of low/medium/high, the
        # ON level maps to that value instead of "medium". Unset (the
        # production default) leaves behavior exactly as before. This is
        # a measurement knob only — it does not change routing or the
        # shipped per-stage default.
        _ov = os.environ.get("EVAL_PERF_GPT_OSS_EFFORT", "").strip().lower()
        if _ov in ("low", "medium", "high"):
            return {"reasoning_effort": _ov if enabled else "low"}
        return {"reasoning_effort": "medium" if enabled else "low"}
    if "glm" in model:
        # GLM-5.2 (Zhipu/zai) takes chat_template_kwargs.enable_thinking;
        # thinking is ON by default on the vLLM/SGLang serving Tinfoil
        # uses, disabled via enable_thinking=False. Same control shape as
        # Gemma. Refs: vLLM GLM-5.2 recipe, Z.AI GLM-5 docs.
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": enabled}}}
    if "kimi" in model:
        # `thinking: False` verified to disable Kimi's CoT (~5× shorter
        # completion) on Tinfoil. The `True` direction has not been tested
        # for clean JSON output — wrapped <think> blocks may leak into
        # delta.content and break our parser. Whitelist Kimi only after
        # testing.
        return {"extra_body": {"chat_template_kwargs": {"thinking": enabled}}}
    if "gemma" in model:
        # Gemma takes chat_template_kwargs.enable_thinking. Default-off;
        # whitelist a tuple to flip enabled=True per stage.
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": enabled}}}
    if "minimax" in model:
        # MiniMax M2/M2.5 reasoning is ALWAYS ON — there is no on/off
        # kwarg per the MiniMax docs. What we control is routing:
        #   reasoning_split=True  → reasoning lands in `reasoning_content`
        #                           (separate from `content` field)
        #   reasoning_split=False → reasoning embedded in <think>...</think>
        #                           inside `delta.content`
        # Send True to keep our JSON parser away from <think> blocks. The
        # `enabled` flag is ignored — there's no "more reasoning" to turn on.
        return {"extra_body": {"reasoning_split": True}}
    return {}


# ── Mode → pinned (provider, model) spec ──────────────────────────────────────
#
# Each of the 3 Modes has ONE pinned ModelSpec at runtime, resolved at
# module load from ~/.basevault/config.json (`tee_provider`, `tee_model`,
# `test_model`, `local_model`). Pinning at module load means every stage
# in a run uses the same model — if we later want per-stage overrides,
# this is the single choke point to adjust.

# Pinned baseline ModelSpecs per (provider, model) combo. The
# _build_*_spec() functions below pick from these by reading
# config.json.
_TINFOIL_GEMMA4_31B = ModelSpec(
    provider=Provider.TINFOIL,
    model_id="gemma4-31b",
    context_window=256_000,
    require_streaming=False,
)

_TINFOIL_KIMI_K2_6 = ModelSpec(
    provider=Provider.TINFOIL,
    model_id="kimi-k2-6",
    context_window=256_000,
    require_streaming=False,
)

# OpenAI's open-weight 120B (released Aug 2025, Apache-2.0). MoE,
# 5.1B active params per token, native 131k context. Reasoning control
# via top-level `reasoning_effort` (low / medium / high); see
# `_reasoning_kwargs` for the per-stage default + whitelist mapping.

_TINFOIL_GPT_OSS_120B = ModelSpec(
    provider=Provider.TINFOIL,
    model_id="gpt-oss-120b",
    context_window=128_000,        # native ~131k; round-down for splitter math
    require_streaming=False,
)

# GLM-5.2 on Tinfoil. 384K context per Tinfoil's chat-models catalog
# (https://docs.tinfoil.sh/models/chat); 754B params / 40B active.
# Reasoning on/off via chat_template_kwargs.enable_thinking (see
# _reasoning_kwargs) — verified against live Tinfoil: OFF→~0 reasoning
# tokens, ON→reasoning tokens emitted. Whitelisted in _REASONING_WHITELIST.
_TINFOIL_GLM_5_2 = ModelSpec(
    provider=Provider.TINFOIL,
    model_id="glm-5-2",
    context_window=384_000,
    require_streaming=False,
)


def _read_app_config() -> dict:
    """Read ~/.basevault/config.json. Returns {} on any error.

    config.json is the single source of truth for app preferences:
    subject, tee_provider, tee_model, test_model, local_model,
    obsidian_*, local_setup_mode. Written by the wizard / Settings;
    read directly by the pipeline (not bridged through env vars).
    """
    p = Path.home() / ".basevault" / "config.json"
    if not p.exists():
        return {}
    try:
        import json as _json
        data = _json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ── Per-(provider, model_id) registry ────────────────────────────────────────
#
# Single source of truth for every ModelSpec we know about. _build_*_spec()
# selects from here for the mode-pinned default; per-stage routing (see
# _STAGE_MODEL_MAP below) also resolves through here. Every (provider,
# model_id) used anywhere in the pipeline MUST appear here so quirks
# (streaming, max_output caps, context_window) are respected uniformly.
_MODEL_SPECS: dict[tuple[Provider, str], ModelSpec] = {
    (Provider.TINFOIL, "gemma4-31b"):   _TINFOIL_GEMMA4_31B,
    (Provider.TINFOIL, "kimi-k2-6"):    _TINFOIL_KIMI_K2_6,
    (Provider.TINFOIL, "gpt-oss-120b"): _TINFOIL_GPT_OSS_120B,
    (Provider.TINFOIL, "glm-5-2"):      _TINFOIL_GLM_5_2,
}


def _spec_for_model_id(provider: Provider, model_id: str) -> ModelSpec:
    """Look up the registered ModelSpec for (provider, model_id).

    Used for resolving per-stage routing. Raises KeyError if missing —
    every (provider, model_id) must be registered so context_window /
    max_output / streaming quirks match the actual model behavior.
    """
    return _MODEL_SPECS[(provider, model_id)]


def register_modelspec(spec: ModelSpec) -> None:
    """Public API for the eval tree to register a ModelSpec into the
    pipeline's registry at eval startup.

    Trust-surface contract: the production binary's core registry
    holds Tinfoil specs only (Mode.TEE) plus the bare MLX/Ollama
    dispatch used by Mode.LOCAL. The eval tree under
    ``testing/eval/`` may call this to add specs for providers it
    dispatches separately at measurement time; production code paths
    never reach this function — it's a one-way extension hook readable
    from ``testing/`` only.

    Idempotent: re-registering an identical (provider, model_id) pair
    replaces the prior entry; tests that mutate the registry can rely
    on a fresh ``register_modelspec(spec)`` call to reset their slot.
    """
    _MODEL_SPECS[(spec.provider, spec.model_id)] = spec


# ── Multi-model sentinels (within-stage, parallel dispatch) ─────────────────
#
# A multi-model sentinel names a SINGLE-STAGE parallel-dispatch config:
# when a stage's `stage_models[<stage>]["model"]` is the sentinel, the
# stage hands it verbatim to `run_stage_multi`, which stands up one
# `Scheduler` per sub-model pulling from a shared thread-safe work
# iterator. This is uniform for every stage — no stage owns dispatch
# logic. Wire-level dispatch never sees the sentinel: each sub-model
# scheduler runs `force_model`, so its thunks bind a concrete model id.
_MULTI_MODEL_SENTINELS: dict[str, tuple[str, ...]] = {
    "kimi+glm": ("kimi-k2-6", "glm-5-2"),
}


def expand_multi_model_sentinel(model_id: str) -> list[str]:
    """Expand a multi-model sentinel to its real-backend constituents.
    Non-sentinel ids return `[model_id]`. `run_stage_multi` stands up
    one `force_model` `Scheduler` per constituent (in TEE, for any
    stage); a non-sentinel id takes the single-scheduler path."""
    return list(_MULTI_MODEL_SENTINELS.get(model_id, (model_id,)))


# ── Retired whole-pipeline model presets ──────────────────────────────────────
#
# Config ids that once selected a whole-pipeline model preset (one
# `tee_model` value expanding to a per-stage map). Per-stage `stage_models`
# (the Settings rows) is the only routing source now, so a config still
# carrying one of these in `tee_model` is treated as unset: the resolvers
# below fall back to the ship per-stage defaults instead of broadcasting the
# id as a backend model. Without this set, such a config would route every
# stage to a model id that isn't registered in `_MODEL_SPECS`.
_RETIRED_TEE_MODEL_IDS: frozenset[str] = frozenset({"mixed-gpt-oss-kimi-k2-6"})


def tinfoil_attest_model_ids() -> list[str]:
    """Every Tinfoil model_id registered in ``_MODEL_SPECS`` (chat +
    vision). The verify_attestation flow iterates this list so the
    panel covers every backend the pipeline could call, regardless of
    the user's current ``stage_models`` config.
    """
    return sorted(
        m for (p, m) in _MODEL_SPECS.keys() if p == Provider.TINFOIL
    )


def _build_tee_spec() -> ModelSpec:
    """Pin for Mode.TEE — the "budget anchor" spec. Tinfoil is the only
    attested route.

    Resolution order (config.json):
      1. `stage_models.extract.model` if present → look that up. extract
         is the budget anchor because it has the highest call volume
         and dominates run wall-clock; sizing the run against its model
         keeps the splitter math honest.
      2. `tee_model` is a registered Tinfoil model id → that spec.
      3. None of the above → Tinfoil default (gpt-oss-120b, matching
         the per-stage default).
    """
    cfg = _read_app_config()
    raw_model = (cfg.get("tee_model") or "").strip() or None
    if raw_model in _RETIRED_TEE_MODEL_IDS:
        raw_model = None

    # Per-stage map (new source of truth): extract is the budget anchor.
    raw_stage_models = cfg.get("stage_models")
    if isinstance(raw_stage_models, dict):
        extract_raw = raw_stage_models.get("extract")
        if isinstance(extract_raw, dict):
            extract_model = str(extract_raw.get("model") or "").strip()
            if extract_model and (Provider.TINFOIL, extract_model) in _MODEL_SPECS:
                return _MODEL_SPECS[(Provider.TINFOIL, extract_model)]

    if raw_model and (Provider.TINFOIL, raw_model) in _MODEL_SPECS:
        return _MODEL_SPECS[(Provider.TINFOIL, raw_model)]

    spec = _TINFOIL_GPT_OSS_120B
    if raw_model:
        # Best-effort: keep Tinfoil's spec quirks and swap model_id only.
        # Likely wrong if the model needs different params — register
        # it in _MODEL_SPECS instead.
        from dataclasses import replace
        spec = replace(spec, model_id=raw_model)
    return spec


# Bundled MLX local-inference default: a 4-bit Qwen3.5 9B quant that
# runs on stable released mlx-lm. The Rust layer mirrors this id in
# DEFAULT_MLX_MODEL; keep the two in sync.
DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-9B-4bit"


def mlx_model_dir(model_id: str) -> Path:
    """On-disk location of a downloaded MLX model snapshot. The model
    is fetched here by the in-app downloader (no inference-time auto-
    download). The Hugging Face repo id is kept as a nested path so the
    directory is self-documenting; the Rust delete command mirrors this
    join — keep the two in sync if the convention changes."""
    return Path.home() / ".basevault" / "models" / model_id


def _build_local_spec() -> ModelSpec:
    """Pin for Mode.LOCAL. `local_backend` selects the provider:
    "mlx" (primary, bundled — default) or "ollama" (opt-in, only if the
    user already has it). MLX reads `local_mlx_model` (default
    DEFAULT_MLX_MODEL); Ollama reads `local_model` (default qwen3.5:9b).
    MLX pins a 32k context window. The picker stays a single "Local"
    mode."""
    cfg = _read_app_config()
    backend = (cfg.get("local_backend") or "mlx").strip().lower()
    if backend == "ollama":
        model_id = (cfg.get("local_model") or "").strip() or "qwen3.5:9b"
        return ModelSpec(
            provider=Provider.OLLAMA,
            model_id=model_id,
            context_window=64_000,
            max_output=64_000,
            require_streaming=True,   # Ollama always streams in our client
        )
    model_id = (cfg.get("local_mlx_model") or "").strip() or DEFAULT_MLX_MODEL
    return ModelSpec(
        provider=Provider.MLX,
        model_id=model_id,
        context_window=32_768,
        max_output=32_768,
        require_streaming=True,   # we stream tokens out of mlx_lm
    )


MODE_SPEC: dict["Mode | str", ModelSpec] = {
    Mode.LOCAL: _build_local_spec(),
    Mode.TEE:   _build_tee_spec(),
}


def get_mode_spec(mode: "Mode | str") -> ModelSpec:
    """Return the pinned ModelSpec for this Mode.

    Accepts either a ``Mode`` enum value (production callers) or a
    bare string label (eval-side callers using extension modes
    registered via ``register_mode``). Both forms key into the same
    ``MODE_SPEC`` dict.
    """
    return MODE_SPEC[mode]


def register_mode(label: str, spec: ModelSpec) -> None:
    """Register an extension mode under a string label. Same
    trust-surface contract as ``register_modelspec`` /
    ``register_dispatcher``: production binary ships ``MODE_SPEC``
    pre-populated with Mode.LOCAL + Mode.TEE only; the eval tree adds
    string-keyed extension modes at startup so eval code can call
    ``complete(mode="test", ...)`` and route through the eval's
    registered dispatcher for that mode's spec provider.
    """
    MODE_SPEC[label] = spec


def _mode_str(m: "Mode | str") -> str:
    """Stringify a mode value regardless of whether it's a ``Mode``
    enum member or a bare-string extension mode."""
    return m.value if isinstance(m, Mode) else str(m)


# ── Per-stage model + reasoning map ───────────────────────────────────────────
#
# Source of truth for "what model runs each stage on Mode.TEE":
# config.json's `stage_models` field — a dict keyed by stage name with
# {"model": "<id>", "reasoning": <bool>} entries. The UI exposes 5 rows
# (extract, entities, patterns, insights, actions).
#
# Migration ladder when `stage_models` is absent in config:
#   - tee_model is a single model id → broadcast to all stages, reasoning OFF.
#   - tee_model is empty / unset / a retired preset id → ship default. The ship map routes
#     the heavy/dominant stages off the single-enclave Kimi pinch
#     point: extract + entities → gpt-oss-120b reasoning ON (3
#     Tinfoil enclaves, dissolves the contention wall for ~88% of
#     pipeline call volume); entities_dedupe → gemma4-31b reasoning
#     OFF and patterns → kimi-k2-6 reasoning OFF. A large-prompt
#     concurrency bench showed reasoning-ON at scale cap-hits or grinds
#     on every model for both synthesis stages (gemma's multi-hour
#     hang on the big cross-entity merge was reasoning-ON only);
#     reasoning-OFF is clean — OFF gemma has the best dedupe merge
#     correctness, OFF kimi the best per-topic patterns content (OFF
#     gemma the reliable patterns fallback). insights + actions stay
#     kimi-k2-6 reasoning OFF. Bench refs: data/eval_perf/runs
#     (extract + entities), testing/eval/query/results-synthesis-tail-cut.md
#     (earlier synthesis-tail bench).
#
# This resolution runs once at module load. Settings → save writes the
# resolved map back into config.json so users see the structured field
# instead of the legacy tee_model after first save.

_STAGE_MODEL_MAP: dict[str, dict] = {}

# Ship-default per-stage map. Mirrored in src/teeProviders.js's
# defaultStageModels() — fresh installs (Python + UI) must agree on
# what gets written into config.json on first init. Vision is
# included so `_STAGE_MODEL_MAP["vision"]` is always present after
# `resolve_stage_models_from_config`, which lets `describe_image`
# read its model from the same map every other stage uses (single
# source of truth + Settings UI surfaces it).
_DEFAULT_STAGE_MODELS: dict[str, dict] = {
    # Ship-default per #703 routing (extract/entities on gpt-oss
    # reasoning ON; synthesis tail on kimi reasoning OFF; dedupe on
    # gemma reasoning OFF). The `kimi+glm` parallel-dispatch sentinel
    # (#666) is a SELECTABLE per-stage option, NOT the ship-default —
    # see `_MULTI_MODEL_SENTINELS`.
    # Vision routes through vision._VISION_MODEL[mode] today; the entry
    # here exists so a stage_models map without an explicit "vision"
    # key still resolves to a usable default. The model id matches
    # `_VISION_MODEL[Mode.TEE]` so plain TEE runs see the same model
    # whether or not config.json carries stage_models.
    "vision":          {"model": "kimi-k2-6",    "reasoning": False},
    "extract":         {"model": "gpt-oss-120b", "reasoning": True},
    "entities":        {"model": "gpt-oss-120b", "reasoning": True},
    "entities_dedupe": {"model": "gemma4-31b",   "reasoning": False},
    "patterns":        {"model": "kimi-k2-6",    "reasoning": False},
    "insights":        {"model": "kimi-k2-6",    "reasoning": False},
    "actions":         {"model": "kimi-k2-6",    "reasoning": False},
}

# Stage ids to include in stage_models config flow. Kept as a single
# tuple constant so resolve_stage_models_from_config + _broadcast_or_default
# don't drift apart when a new stage is added.
_STAGE_MODEL_KEYS: tuple[str, ...] = (
    "vision", "extract", "entities", "entities_dedupe",
    "patterns", "insights", "actions",
)

# Stages that participate in CHAT preset routing. Vision is excluded —
# chat presets describe text routing only, and broadcasting a chat
# model id to vision (e.g. mapping vision → "kimi-k2-6") would route
# describe_image to a chat model that can't process images.
_CHAT_STAGE_KEYS: tuple[str, ...] = (
    "extract", "entities", "entities_dedupe",
    "patterns", "insights", "actions",
)


def _coerce_stage_models_entry(raw) -> dict | None:
    """Validate one stage_models entry. Returns sanitized
    {"model": str, "reasoning": bool} or None if malformed.
    Accepts a plain model-id string as shorthand for
    {"model": id, "reasoning": false}."""
    if isinstance(raw, str):
        model = raw.strip()
        return {"model": model, "reasoning": False} if model else None
    if not isinstance(raw, dict):
        return None
    model = str(raw.get("model") or "").strip()
    if not model:
        return None
    reasoning = bool(raw.get("reasoning", False))
    return {"model": model, "reasoning": reasoning}


def resolve_stage_models_from_config(cfg: dict) -> dict[str, dict]:
    """Materialize the full stage routing map from config.

    Returns a dict keyed by every stage in {extract, entities,
    entities_dedupe, patterns, insights, actions}.

    Resolution order:
      1. cfg.stage_models present → use it. Missing stages within that
         field fall back to the next layer (do not silently default;
         partial maps are filled from the broadcast/default below).
      2. cfg.tee_model is a single registered model id → broadcast it
         to every stage with reasoning OFF.
      3. otherwise (unset / a retired preset id) → _DEFAULT_STAGE_MODELS
         (gpt-oss + gemma + kimi).

    Public so the Tauri attestation flow + tests can call it without
    going through module-load global state.
    """
    raw_stage_models = cfg.get("stage_models")
    if isinstance(raw_stage_models, dict):
        sanitized: dict[str, dict] = {}
        for stage in _STAGE_MODEL_KEYS:
            entry = _coerce_stage_models_entry(raw_stage_models.get(stage))
            if entry is not None:
                sanitized[stage] = entry
        if sanitized:
            # Fill any missing visible stage from the broadcast layer
            # below so stage_models can be partial without a half-empty
            # routing table.
            fallback = _broadcast_or_default(cfg)
            for stage in _STAGE_MODEL_KEYS:
                sanitized.setdefault(stage, fallback[stage])
            return sanitized

    # No structured stage_models — derive from legacy tee_model.
    return _broadcast_or_default(cfg)


def _broadcast_or_default(cfg: dict) -> dict[str, dict]:
    """Build a stage map by broadcasting legacy `tee_model` to every chat
    stage, or return the ship default when it's unset (or a retired
    preset id — see `_RETIRED_TEE_MODEL_IDS`).

    Vision never picks up the chat anchor — `tee_model` describes chat
    routing, and broadcasting it to vision would point describe_image
    at a model that can't process images. Vision is always filled from
    `_DEFAULT_STAGE_MODELS["vision"]` regardless of the chat anchor;
    the user can override it via Settings (writes to
    `stage_models.vision`)."""
    raw_model = (cfg.get("tee_model") or "").strip() or None
    if raw_model in _RETIRED_TEE_MODEL_IDS:
        raw_model = None
    out: dict[str, dict]
    if raw_model:
        # Broadcast a single model id to chat stages, reasoning OFF.
        # We intentionally do NOT validate it against _MODEL_SPECS here —
        # the per-call resolver does the spec lookup, and an unregistered
        # id surfaces there with the rest of the error context.
        out = {
            stage: {"model": raw_model, "reasoning": False}
            for stage in _CHAT_STAGE_KEYS
        }
    else:
        # Fresh install / unconfigured: ship default.
        out = {stage: dict(entry) for stage, entry in _DEFAULT_STAGE_MODELS.items()}
    # Always backfill vision from the ship default — chat tee_model
    # never owns vision routing.
    out.setdefault("vision", dict(_DEFAULT_STAGE_MODELS["vision"]))
    return out


def _resolve_stage_model_map() -> dict[str, dict]:
    """Module-load wrapper: read config.json and produce the active
    per-stage routing map. Called once on import; per-call code reads
    `_STAGE_MODEL_MAP`.

    `BASEVAULT_STAGE_MODELS_OVERRIDE` env var (JSON object, same shape
    as config.json `stage_models`) overrides the user's config when set.
    Used by the eval framework to enforce per-suite model specs without
    requiring the worker to reconfigure their config.json."""
    cfg = _read_app_config()
    _smo = os.environ.get("BASEVAULT_STAGE_MODELS_OVERRIDE")
    if _smo:
        try:
            cfg = {**cfg, "stage_models": json.loads(_smo)}
        except (json.JSONDecodeError, TypeError):
            pass
    return resolve_stage_models_from_config(cfg)


_STAGE_MODEL_MAP = _resolve_stage_model_map()


def unique_models_in_stage_map(stage_map: dict[str, dict] | None = None) -> list[str]:
    """Return the unique model ids in `stage_map` (defaults to the
    module-loaded `_STAGE_MODEL_MAP`), ordered by first appearance
    across the canonical pipeline stage order so consumers (the UI
    model rows) render in run order.

    Returns an empty list when the map is not populated — callers
    should fall back to the mode-pinned single-model resolution.
    """
    sm = stage_map if stage_map is not None else _STAGE_MODEL_MAP
    if not sm:
        return []
    seen: list[str] = []
    # Vision is intentionally excluded from this list — it feeds the
    # UI model rows and routes orthogonally to chat (vision models can't
    # run chat prompts and vice versa), so it doesn't belong in the
    # chat-stage model rollup.
    for stage in ("extract", "entities",
                  "patterns", "insights", "actions"):
        entry = sm.get(stage)
        if not entry:
            continue
        m = entry.get("model")
        if not m:
            continue
        # Multi-model sentinels expand to their real-backend
        # constituents — attestation rows + model-pin diagnostics
        # need every concrete model the run will actually call.
        for real in expand_multi_model_sentinel(m):
            if real not in seen:
                seen.append(real)
    return seen


def stage_model_id(stage: str) -> str | None:
    """Return the raw configured model id for `stage` — may be a
    multi-model sentinel. Stage code reads this before going through
    `_resolve_stage_override` so it can branch on the sentinel without
    a spec lookup. Returns None if the stage is not in the map."""
    entry = _STAGE_MODEL_MAP.get(stage)
    if not entry:
        return None
    return entry.get("model")


def _resolve_stage_override(mode: Mode, stage: str | None) -> tuple[ModelSpec, str | None]:
    """Return (effective_spec, override_model_id_or_None) for this call.

    Routing rules (in order):
      1. If `mode` is TEE and a per-stage `_STAGE_MODEL_MAP` entry maps
         this stage to a model different from the mode-pinned anchor,
         swap to that stage's spec and return its real model id.
      2. Otherwise return the mode-pinned spec unchanged.

    `_STAGE_MODEL_MAP` is populated at module load by
    `_resolve_stage_model_map()` from config.json's `stage_models`.

    The "different from the mode-pinned model" check matters: when a
    stage's mapped model equals the budget anchor, we return None for
    `override_model` so the caller's stat record reads identically to
    a non-routed run on that model.

    Call sites:
      - The runner wrapper calls this BEFORE `begin_stat_record` so the
        begin event in `llm-calls.jsonl` records the routed model (not
        the budget anchor placeholder a caller threaded through as
        `model=`).
      - `complete()` calls it again post-entry to perform the actual
        provider dispatch.
      - Stage modules call it directly for warning records emitted
        off-wrapper-path (cap-overflow, call-failed) — those run before
        the wrapper's begin/end pair and otherwise wouldn't know the
        routed model.
    """
    base = get_mode_spec(mode)
    if stage is None or mode != Mode.TEE:
        return base, None
    stage_cfg = _STAGE_MODEL_MAP.get(stage)
    if not stage_cfg:
        return base, None
    target_model = stage_cfg.get("model")
    if not target_model or target_model == base.model_id:
        return base, None
    # Today every stage_models entry resolves against the active TEE
    # provider (Tinfoil); cross-provider per-stage routing isn't
    # exposed in the UI. If we ever surface that, lift `provider` into
    # the per-stage entry and read it here instead.
    provider = _tee_provider_from_config()
    # Multi-model sentinel reaching this resolver: resolve to the FIRST
    # constituent — a real backend model id — for BOTH the spec and the
    # override_model. The wire never lands here with a sentinel: every
    # stage hands its (sentinel-preserving) configured model to
    # `run_stage_multi`, which stands up one `force_model` scheduler per
    # constituent, so each dispatch is pinned to a concrete constituent
    # via `_force_model_id` and bypasses this resolver. The only callers
    # that reach here with a sentinel are off-wire spec/budget queries
    # (compute_budget, reasoning, ctx) and warning/observability
    # emitters; they need one concrete spec, and the first constituent
    # is the representative anchor. Uniform across stages — no stage is
    # special-cased.
    if target_model in _MULTI_MODEL_SENTINELS:
        first = _MULTI_MODEL_SENTINELS[target_model][0]
        return _spec_for_model_id(provider, first), first
    try:
        return _spec_for_model_id(provider, target_model), target_model
    except KeyError:
        # Cross-provider fallback for eval-registered extension specs:
        # the stage_models map points at a model registered under a
        # provider tag other than the TEE base. Falls through to the
        # dispatcher registry; production callers' stage_models point
        # at core models so this branch never fires for them.
        #
        # Strict-contract: exactly one cross-provider match is allowed.
        # A second registration under the same model_id is a
        # registration bug, not a routing one — surfacing it here as
        # ``RuntimeError`` is louder + easier to track than silently
        # picking the first match out of dict-iteration order.
        cross = [
            s for (p, m), s in _MODEL_SPECS.items()
            if m == target_model and p != provider
        ]
        if not cross:
            raise
        if len(cross) > 1:
            providers = sorted({_provider_str(s.provider) for s in cross})
            raise RuntimeError(
                f"Ambiguous cross-provider spec for model_id "
                f"{target_model!r}: registered under {providers}. "
                f"Each (provider, model_id) must be unique in the "
                f"registry; deregister the duplicate via "
                f"register_modelspec()."
            )
        return cross[0], target_model


def _tee_provider_from_config() -> Provider:
    """Cached active TEE provider. Reads config.json at module load."""
    return MODE_SPEC[Mode.TEE].provider


# ── Per-mode context windows & input/output budgeting ─────────────────────────
#
# Each LLM call gets a context window CTX. We split it into:
#   chunk_in + (OUT_IN_RATIO * chunk_in) + PROMPT_PADDING ≤ CTX
# i.e. reserve OUT_IN_RATIO times the input for the model's output, with a
# fixed padding for system/user scaffolding overhead. Solving:
#   chunk_in_cap = (CTX - PROMPT_PADDING) // (1 + OUT_IN_RATIO)
#   max_output   = OUT_IN_RATIO * chunk_in_cap
#
# 3:1 (output:input) is a safety factor. Observed healthy extract calls ran
# at ~0.45 mean / 0.75 max output-to-input. 3:1 gives ~4× headroom over that,
# room for denser corpora (chat logs, meeting notes) without cap risk.
#
# Per-stage scaffolding cost (system prompt + task header + small
# per-call wrappers). Used in two places:
#   1. chunk_cap_for_stage subtracts scaffolding from ctx before
#      dividing, so each stage's input budget reflects variable-
#      content room only.
#   2. dynamic_max_tokens adds scaffolding to payload on the input
#      side — the model processes the WHOLE prompt (scaffolding +
#      payload), so output budget should scale with total input.
# Measured fixed scaffolding (system + task header) ranges 49-985
# tokens per stage; values below are ~1.5× that for slop + per-call
# wrappers (date hints, entities_context blocks, etc).
# Calibrated against chars/3 measurement (NOT tiktoken). Comments
# show the empirically measured system+template token count for an
# empty payload; constants are rounded up for safety margin.
#
# Re-measure with:
#   from tokens import count_tokens
#   from <stage_module> import _SYSTEM, _TASK
#   count_tokens(_SYSTEM + _TASK_with_empty_fields)
#
# The pre-call check in complete() prefers `payload_tokens` from
# kwargs when available (exact actual scaffolding via subtraction),
# falling back to these constants. Splitter (chunk_cap_for_stage)
# always uses these constants — bumping them shrinks chunk size,
# avoiding cargo-cult overflow warnings.
_SCAFFOLDING_TOKENS_BY_STAGE: dict[str, int] = {
    "extract":         2500,   # ~2416 tok system + taxonomy + date hint
    "entities":        1500,   # ~1168 tok system; dynamic per-batch context adds
    # Single LLM call over compact (id, name, type, aliases, desc) rows.
    # System prompt is small, task header small, no per-batch overhead.
    "entities_dedupe":  600,
    "patterns":        1700,   # ~1561 tok system + per-topic header
    "insights":        2500,   # ~2386 tok system + entities context block
    "actions":         1700,   # ~1585 tok system + insights context block
    # Vision: ~150 tok prompt (transcribe-or-describe). Image bytes
    # ride a separate channel and don't count against text scaffolding.
    "vision":           200,
}
_DEFAULT_SCAFFOLDING_TOKENS = 2048


def _scaffolding_tokens_for_stage(stage: str | None) -> int:
    return _SCAFFOLDING_TOKENS_BY_STAGE.get(stage or "", _DEFAULT_SCAFFOLDING_TOKENS)

# Per-stage output:input ratio. Calibrated from observed
# completion_tokens/prompt_tokens across recent runs (n=58 extract,
# 10 entities, 43 patterns, 10 insights, 7 actions). Each stage's
# value is ~3-4× the observed max so dense corpora and edge-case
# inflation don't hit the cap. Stages map cleanly into two groups:
#   - Expand (extract, entities): input → many enumerated outputs
#   - Distill (patterns, insights, actions): input → compact summary
# The default (3) covers extract-shaped expanders; entities needs
# more because it produces a longer entity+relation list per group;
# distillation stages need much less because they shrink their input.
# ── Per-stage data flow (drives the ratios below) ────────────────────────────
#
# What each stage's LLM-call INPUT actually is, traced from the
# real prompt-building functions. If a prompt-building function
# changes shape (e.g. patterns starts batching across topics, or
# entities stops dedupe'ing), the ratios MUST be re-calibrated.
#
#   extract: 1 call per chunk (parent docs split by splitter).
#     INPUT  = chunk content + small header (date hint, taxonomy)
#     OUTPUT = JSON list of fact items per chunk
#
#   entities: 1 call per BATCH of deduplicated entity groups.
#     INPUT  = pre-deduplicated groups (each: gid, canonical_name,
#              type, aliases, topics, mention_count, ≤3 sample facts).
#              NOT the raw facts. Group count drives input size, not
#              fact count — entities-batching reads chunk_cap to slice.
#     OUTPUT = one entity record per group: id + role + 1-line desc,
#              plus relation edges
#
#   patterns: 1 call per TOPIC (taxonomy entry). Topics fan out in parallel.
#     INPUT  = facts of THAT one topic only (after dedup), formatted
#              one fact per line, + entities_context block prepended
#              (~500-1500 tokens of top entities). NOT all patterns,
#              NOT all facts — just one topic's facts.
#     OUTPUT = pattern summaries for that topic
#
#   insights: 1 call.
#     INPUT  = ALL strong patterns from all topics, formatted one
#              line each, + entities_context block prepended
#     OUTPUT = cross_domain + critical insight summaries
#
#   actions:  1 call.
#     INPUT  = the InsightOutput (cross_domain + critical insights,
#              each with name/desc/mechanism/implication/domains/
#              proposed_actions), + entities_context block prepended.
#              NOT raw insights' source data.
#     OUTPUT = prioritized action list
#
# Distillation cascade: extract output >> entities input >> patterns
# input >> insights input >> actions input. Each downstream stage
# sees a strict SUMMARY of upstream output, not the full upstream
# data — that's why the chain-through estimator over-counts if you
# use upstream's TOTAL output as downstream's input.

_MAX_RATIO_BY_STAGE: dict[str, float | None] = {
    # MAX ratio: used for max_tokens RESERVATIONS. Conservative
    # ceiling that should never be exceeded by a healthy call.
    # Calibrated against gemma4-31b BIG-PAYLOAD calls (≥5k input
    # tokens) — small-payload samples inflate ratios because the
    # per-item fixed cost (~50-150 tokens of JSON framing) isn't
    # amortized. Observed (last 3d): extract big-payload mean 0.29,
    # max 0.65; others have no big-payload data so stay conservative.
    #
    # metadata is special: output is SCHEMA-BOUNDED (date + topics +
    # N people + M events). Typical 50-500 tokens out on 4k truncated
    # input → ratio ~0.01-0.13. Even very dense docs rarely exceed
    # 0.5. With 2k headroom, 0.1 max gives ~2.4k token reservation,
    # enough for ~10-20 people+events. Extreme docs (50+ entities)
    # will cap — acceptable trade for not over-reserving on every
    # doc in a corpus.
    "extract":  4,    # kimi-k2-6 produces wider output per input than
                      # gemma4-31b: full-corpus runs at ratio=2 cap-hit
                      # repeatedly on big chunks; back-solved effective
                      # desired ratio ~3.7× of payload. Bumped to 4 to
                      # absorb dense-passage outliers (e.g. journal
                      # batches that emit 80k+ output on 15k input).
                      # Halving cascade stays as the safety net.
    "entities": 1,
    # Single-LLM-call SPOF stages — no batching, no fan-out, one
    # shot per pipeline run. A linear ratio × payload cap is
    # actively harmful here: small payloads collapse the cap to a
    # few thousand tokens where routine outputs cap-hit, even
    # though the spec describes outputs as "simple" (dedupe) or
    # "bounded"/"strictly bounded" (patterns / insights / actions).
    # Setting `None` opts out of the linear formula: per-call mtr
    # becomes `min(spec.max_output, ctx − payload − scaffolding)` —
    # the entire remaining context. Input cap stays whatever
    # `_PROMPT_HARD_CAP_BY_STAGE` says (or unconstrained beyond
    # ctx).
    "entities_dedupe": None,
    "patterns": None,
    "insights": None,
    "actions":  None,
    # Vision: image transcripts approximate the visible text in the
    # image; ratio is the safety margin around that estimate, not a
    # quality cap. 1.0 leaves headroom for dense screenshots without
    # the per-call mtr collapsing on small images.
    "vision":   1.0,
    # Interactive retrieval-time stages (not pipeline stages). One
    # `complete()` per user query, no batching / fan-out — same shape
    # as the single-call SPOF stages above, so `None` (full remaining
    # context, output bounded by spec.max_output). Registered so the
    # interactive callers (retrieval rerank, chatbot answer composition)
    # don't trip `_ratio_for_stage`'s unknown-stage KeyError when they
    # run `complete()` outside the runner's per-stage bracketing.
    "rerank":   None,
    "chatbot":  None,
}

# Measured: avg entities-per-fact ≈ 1.07 across 49,444 facts in the
# last 5 days (mostly 1, distribution: 17% have 0, 65% have 1, 15%
# have 2). Rounded to 1 for the planner formula. If this drifts
# (corpus changes, prompt changes), re-measure with:
#   python3 -c "..."  # see comment in scripts/measure_ratios.py
_AVG_ENTITIES_PER_FACT = 1


_TYPICAL_RATIO_BY_STAGE: dict[str, float | None] = {
    # TYPICAL ratio: used for ESTIMATES (progress bar, expected output
    # size, batch counts). Coarse on purpose — back-of-envelope.
    #
    # Each ratio is paired with its EXPECTED INPUT (what feeds it from
    # the previous stage). The estimator chains: stage_N input is
    # derived from stage_{N-1} output via the formula in the comment.
    #
    # `None` entries: single-LLM-call SPOF stages (entities_dedupe /
    # insights / actions). Always 1 call, no batching, no fan-out —
    # a typical-ratio doesn't buy the call-count estimator anything.
    # Entries are present (vs absent) for every known stage so a
    # genuinely-unknown stage name is distinguishable from a SPOF stage.
    "extract":         0.5,                            # input ≈ file_size (per chunk); output: facts JSON
    "entities":        0.5 / _AVG_ENTITIES_PER_FACT,   # input ≈ facts_size × _AVG_ENTITIES_PER_FACT; output: id+role+desc per group
    "entities_dedupe": None,
    "patterns":        0.1,                            # input per call ≈ (facts_size + entities_size) / n_topics; output: pattern summaries
    "insights":        None,
    "actions":         None,
    "vision":          0.3,                            # input = image (off-band); output: transcript or description text
    # Interactive retrieval-time stages — single `complete()` per user
    # query, no call-count estimator (the progress bar doesn't track
    # interactive chatbot flows). `None` mirrors the SPOF-stage treatment;
    # entries present so a genuine typo is distinguishable from a SPOF stage.
    "rerank":          None,
    "chatbot":         None,
}


def _ratio_for_stage(stage: str | None) -> float | None:
    """Return the MAX output:input ratio for a stage (used for
    reservations).

    Returns `None` for stages whose `_MAX_RATIO_BY_STAGE` entry is
    explicitly `None` — single-LLM-call stages where the linear
    formula doesn't apply and the per-call mtr is the full remaining
    context window (capped only at `spec.max_output`).

    Raises `KeyError` on unknown stage — every pipeline stage must
    have a calibrated entry; a silent fallback would mask a routing
    bug or a new stage that forgot to register its ratio."""
    if stage not in _MAX_RATIO_BY_STAGE:
        raise KeyError(
            f"unknown stage {stage!r}: no entry in _MAX_RATIO_BY_STAGE. "
            f"Register the stage with a calibrated max output:input ratio "
            f"(or None for single-LLM-call stages)."
        )
    return _MAX_RATIO_BY_STAGE[stage]


# Quality-driven hard ceiling per stage, independent of the budget
# formula. The formula gives a *budget-safe* cap from the model's
# context window; this would be a *quality-safe* cap from observed
# model behavior. Effective per-call input cap is min(budget, quality).
#
# All values are None today — we have no measurements showing a
# specific size at which any stage's quality degrades. The
# infrastructure is in place so that when we DO measure a quality
# cliff for a stage (e.g. "entities relation extraction starts
# hallucinating above N input tokens"), wiring it in is one number.
# Until then, leave at None and trust the budget cap. Don't add
# guess-numbers here — that's exactly the overclaiming we caught
# ourselves doing on the first pass.
_PROMPT_HARD_CAP_BY_STAGE: dict[str, int | None] = {
    "extract":  None,
    "entities": None,
    "entities_dedupe": None,
    "patterns": None,
    "insights": None,
    "actions":  None,
    "vision":   None,
}


# Stages that exist in the pipeline but don't directly issue LLM
# calls. compute_budget returns a sentinel StageBudget with
# has_llm_calls=False for these so callers can skip per-call
# observability without None-checking numeric fields.
#
# `ingest` is a parent stage: it walks the input tree, materializes
# Documents, and may transitively invoke `vision` per image. The
# vision sub-stage IS an LLM stage and gets a normal budget; the
# ingest level itself does not.
_NO_LLM_STAGES: frozenset[str] = frozenset({"ingest"})


@dataclass(frozen=True)
class StageBudget:
    """Per-call budget for a (mode, stage) pair.

    `max_input` is the input ceiling the chunker should target. It
    solves two simultaneous constraints from the model spec:
      input + ratio×input + scaffolding ≤ context_window  (ctx_cap)
      ratio × input ≤ provider_cap                        (output_cap)
    `max_input = min(ctx_cap, output_cap, quality_cap)`.

    `max_output` is the per-call max_tokens reservation. When
    `payload_tokens` is None it's the static stage ceiling
    (`ratio × max_input`, clamped at provider cap). When a payload
    size is supplied it's sized to that specific payload — formula
    `(ratio + reasoning_bump) × payload + headroom`, floored at
    _MIN_CONTEXT_TOKENS total context, capped at
    `min(ctx - payload - scaffolding, provider_cap)`.

    `has_llm_calls=False` flags parent stages (ingest) that don't
    issue LLM calls at this level — `max_input`/`max_output`/
    `scaffolding` are 0 for these and observability emitters should
    skip them.
    """
    max_input: int
    max_output: int
    scaffolding: int
    has_llm_calls: bool


def compute_budget(mode: Mode, stage: str | None = None,
                   payload_tokens: int | None = None,
                   spec_override: "ModelSpec | None" = None) -> StageBudget:
    """Return the StageBudget for a (mode, stage) call.

    Provider-agnostic at the formula level: differentiation between
    Tinfoil and Local lives entirely in the ModelSpec data
    (`context_window`, `max_output`). No branching on `Provider` here.

    `payload_tokens=None` returns the static stage ceiling — what the
    chunker uses to size inputs and what the begin.budget event reports
    pre-call. `payload_tokens=N` returns the per-call reservation
    sized to that exact payload (used by the dispatch wrapper at
    `complete()` to set `max_tokens=`).

    `spec_override` overrides the stage-routed ModelSpec when set —
    issue #626: the degrading-retry cascade's Step-3 model-fallback
    swaps the dispatch to an alternate model (e.g. kimi-k2-6 →
    gemma4-31b on cap-hit), and the per-call
    `max_tokens` must be sized against that alternate's window, not
    the stage anchor's. Same formula, different spec, no separate
    retry-specific code path.

    Round-trip property (verified by tests): for any (mode, stage),
    feeding `compute_budget(...).max_input` back as
    `payload_tokens=max_input` produces a `max_output` ≤ provider_cap
    and ≤ remaining context — i.e. the dual formulas don't drift.
    """
    if stage in _NO_LLM_STAGES:
        return StageBudget(
            max_input=0, max_output=0, scaffolding=0, has_llm_calls=False,
        )

    if spec_override is not None:
        spec = spec_override
    else:
        try:
            spec, _ = _resolve_stage_override(mode, stage)
        except KeyError:
            # Stage-models map points to a model id with no registered
            # ModelSpec — happens when the user picks a custom vision
            # model id from Settings that isn't pre-registered. Vision
            # dispatch (`describe_image`) tolerates this via a synthesized
            # spec; budget reporting falls back to the mode anchor so the
            # snapshot still produces sensible numbers.
            spec = get_mode_spec(mode)
    scaffolding = _scaffolding_tokens_for_stage(stage)
    base_ratio = _ratio_for_stage(stage)
    quality_cap = _PROMPT_HARD_CAP_BY_STAGE.get(stage or "")

    # base_ratio=None → single-LLM-call stage (patterns / insights /
    # actions today). Linear formula doesn't apply: max_input stays
    # whatever the explicit hard cap says (or the full ctx minus
    # scaffolding when none is set), and the per-call output is
    # capped only by the provider's max_output and the remaining
    # context after the actual payload lands. See
    # `_MAX_RATIO_BY_STAGE` comment for rationale.
    if base_ratio is None:
        if quality_cap is not None:
            max_input = quality_cap
        else:
            max_input = max(0, spec.context_window - scaffolding)
        if payload_tokens is None:
            max_output = spec.max_output
        else:
            real_cap = max(
                0, spec.context_window - payload_tokens - scaffolding)
            max_output = min(real_cap, spec.max_output)
        return StageBudget(
            max_input=max_input,
            max_output=max_output,
            scaffolding=scaffolding,
            has_llm_calls=True,
        )

    # max_input: solve both constraints.
    # ctx_cap from input + ratio×input + scaffolding ≤ context_window.
    # output_cap from ratio × input ≤ provider_cap (so reservations
    # never exceed what the model actually emits — Tinfoil specs are
    # wide-cap; MLX/Ollama specs declare their own max_output).
    ctx_cap = int((spec.context_window - scaffolding) / (1 + base_ratio))
    output_cap = int(spec.max_output / base_ratio) if base_ratio > 0 else ctx_cap
    max_input = min(ctx_cap, output_cap)
    if quality_cap is not None:
        max_input = min(max_input, quality_cap)

    if payload_tokens is None:
        # Static stage ceiling for the chunker / begin.budget event.
        max_output = min(int(base_ratio * max_input), spec.max_output)
    else:
        # Per-call reservation sized to the actual payload.
        # Reasoning bump (+1) reserves room for <thinking> tokens
        # alongside content output — they share max_tokens. Universal
        # +1 (not per-stage) until empirical data justifies tuning.
        reasoning_bump = 1.0 if _is_reasoning_on_for_stage(stage, mode) else 0.0
        ratio = base_ratio + reasoning_bump
        # Ratio applies to PAYLOAD only — scaffolding is fixed framing
        # (system prompt, task header), not amplified per-output-token.
        # Headroom is per-output-item fixed cost slack.
        raw = int(ratio * payload_tokens) + _HEADROOM_TOKENS
        # Pad up to _MIN_CONTEXT_TOKENS total (payload + max_tokens) so
        # small payloads still get room for dense outputs.
        min_for_context = max(0, _MIN_CONTEXT_TOKENS - payload_tokens)
        # Real headroom: total ctx minus what payload + scaffolding
        # already consume. Provider cap is the OTHER ceiling — silent
        # server-side clamp at spec.max_output is wasted slot budget.
        real_cap = max(0, spec.context_window - payload_tokens - scaffolding)
        provider_cap = spec.max_output
        max_output = min(max(raw, min_for_context), real_cap, provider_cap)

    return StageBudget(
        max_input=max_input,
        max_output=max_output,
        scaffolding=scaffolding,
        has_llm_calls=True,
    )


def chunk_cap_for_stage(mode: Mode, stage: str | None = None) -> int:
    """Max input tokens per call for `stage` on `mode`. Thin wrapper
    over `compute_budget(..., payload_tokens=None).max_input`."""
    return compute_budget(mode, stage).max_input


# Token-counting helpers (count_tokens / tokens_from_chars /
# estimate_prompt_tokens / CHARS_PER_TOKEN) live in tokens.py — single
# home. llm.py imports them at the use sites; no re-exports here.
#
# Why dynamic max_tokens matters at all: vLLM/SGLang-style schedulers
# (Tinfoil) reserve KV slots up to max_tokens at admission time. A flat
# ceiling on every call starved Tinfoil's batcher and produced 30-minute
# timeouts on large chunks.

from engine.tokens import estimate_prompt_tokens  # noqa: E402,F401 (re-export for phases.chat / telemetry_hook)

_HEADROOM_TOKENS = 2048  # per-output-item fixed cost slack

# Minimum total context (payload + max_tokens) we reserve on every
# call. Ensures small payloads still have room for dense outputs
# even when the linear ratio would give a tiny max_tokens. On big
# payloads where payload + ratio×payload already exceeds this, no
# effect — the formula stands.
_MIN_CONTEXT_TOKENS = 32_000


def _is_reasoning_on_for_stage(stage: str | None, mode: Mode) -> bool:
    """True iff reasoning will be enabled for this (mode, stage) call.

    Resolves the same (spec, stage) pair `complete()` resolves at
    dispatch time — so dynamic_max_tokens reserves the bigger budget
    only when the runtime path will actually emit reasoning tokens.

    Note on the dead-toggle case (#107): `_reasoning_enabled_for`
    AND-gates whitelist membership with the user toggle, so a stage
    absent from `_REASONING_STAGES` returns False here too — the
    runtime won't emit reasoning, and we won't over-reserve.
    `entities_dedupe` is now in the whitelist, so the bump applies
    when its toggle is on."""
    spec, _ = _resolve_stage_override(mode, stage)
    return _reasoning_enabled_for(spec, stage)


def dynamic_max_tokens(payload_tokens: int, mode: Mode,
                       stage: str | None = None,
                       spec_override: "ModelSpec | None" = None) -> int:
    """Return a per-call max_tokens budget sized to the ACTUAL payload.

    Thin wrapper over `compute_budget(..., payload_tokens=payload_tokens)
    .max_output`. Kept as a named entry point for the dispatch site in
    `complete()` and the LOCAL/Ollama `num_predict` site, plus for
    tests that pin the per-call formula directly.

    Caller passes payload_tokens — the chunk/group/facts content that
    will be inserted into the prompt template, converted to tokens
    via tokens_from_chars(len(payload)). The caller knows this exactly;
    we don't estimate by subtracting scaffolding from a total-message-
    blob guess.

    `spec_override` recomputes the budget against an alternate
    ModelSpec (issue #626 — the model-fallback retry must size
    `max_tokens` against the alternate's window, not the stage
    anchor's). Forwarded to `compute_budget` unchanged.

    Formula (sourced from compute_budget): (ratio[stage] +
    reasoning_bump) × payload_tokens + headroom, floored at
    _MIN_CONTEXT_TOKENS total context, capped at min(real headroom
    `ctx - payload - scaffolding`, provider cap `spec.max_output`).
    Scaffolding is NOT added on the output side — it's fixed input
    framing, not output we need to budget."""
    effective_stage = stage or _current_stage
    return compute_budget(mode, effective_stage,
                          payload_tokens=payload_tokens,
                          spec_override=spec_override).max_output


def _mode_ctx(mode: Mode) -> int:
    """Effective context window for the mode's pinned (provider, model).
    Sourced from MODE_SPEC — the per-mode ModelSpec."""
    return get_mode_spec(mode).context_window


# ── Hard per-call wall-clock timeout ─────────────────────────────────────────
#
# The per-stage `timeout=` above is, on a streaming response, an httpx
# PER-READ deadline — the max wait BETWEEN byte chunks, not total
# wall-clock. A reasoning-on model that trickles `reasoning_content` for
# hours keeps resetting that per-read clock on every chunk, so it never
# fires (observed: a single entities_dedupe call ran 152 min, gemma4
# reasoning-on on a ~75k-token input, and sat `pending` for 2.5h). The
# guard below is the independent TOTAL-wall-clock ceiling that bounds a
# call regardless of byte flow.
#
# Two tiers, both PER CALL (armed fresh per call; a stage that fans out
# dozens of splits can run hours in aggregate — only any single call
# that crosses its tier is aborted):
#   * NO-TOKEN tier (60 min): zero tokens emitted for 60 min → abort.
#     Catches the dead-before-first-token hang faster (the same parked-recv
#     class the user-skip socket-shutdown handles).
#   * TOTAL tier (120 min): once the first token has arrived, the total
#     wall-clock is bounded at 120 min. Accommodates a slow-but-producing
#     call — the reasoning-on grind WAS trickling tokens the whole time,
#     so it lives in this tier and aborts at 120 min.
# The abort routes through the retry taxonomy as `_WallClockTimeout` →
# load (retried as-is; the load reasoning-off-after-first-retry policy
# bounds a persistent grind), so the run moves on instead of wedging.


# ── Call-level warnings ───────────────────────────────────────────────────────
# Two failure modes both produce silent-empty downstream output and both
# need to surface to the user:
#   kind="max_tokens"  finish_reason / done_reason == "length"; model hit
#                      the cap and was cut off mid-output. Garbage JSON
#                      or repetition loops.
#   kind="empty"       call returned no content at all (provider-side
#                      timeout, vLLM scheduler kill, etc). We saw this
#                      on Tinfoil at 30-min hard timeout when max_tokens
#                      was over-reserved (Apr 24 data).
# Runner reads both at end of run via get_call_warnings().
_call_warnings: list[dict] = []


def reset_call_warnings() -> None:
    with _usage_lock:
        _call_warnings.clear()


def get_call_warnings() -> list[dict]:
    with _usage_lock:
        return list(_call_warnings)


def _record_call_warning(
    kind: str,
    mode: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    max_tokens: int,
    note: str = "",
    call_id: str | None = None,
) -> None:
    # Caller passes the live call's call_id when one is active. Lets
    # the rollup filter warnings to chain LEAVES — a parent attempt
    # that input-overflowed and got halved is an intermediate retry,
    # not a user-visible failure; the badge should only count overflows
    # that survived to the leaf. None when no call context (e.g. wrapper-
    # level retry-policy warnings fired before begin_stat_record).
    with _usage_lock:
        _call_warnings.append({
            "kind": kind,
            "stage": _current_stage,
            "mode": mode,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "max_tokens": max_tokens,
            "note": note,
            "call_id": call_id,
            "ts": time.time(),
        })


def max_workers(mode: Mode) -> int:
    """Concurrency cap for LLM parallelism per mode.

    LOCAL=1: one GPU, concurrent requests just contend for the same
    VRAM and serialize anyway.

    Cloud=16: matches the Tinfoil router's per-model
    `MaxRequestsWaiting` (clusters in 8-16 across models). Pushing
    more than that just stuffs the wait queue — model throughput is
    enclave-count × tokens/sec, not threadpool size. Also bounds blast
    radius on enclave-side disconnects: when one kills the in-flight
    batch, at most 16 calls flip to zero-token-success instead of 50+.

    Keep this in lockstep with progress.PARALLELISM_PER_STAGE — the
    ETA estimator's wall-clock divider must mirror the actual cap."""
    if mode == Mode.LOCAL:
        return 1
    return 16


TINFOIL_API_KEY = os.environ.get("TINFOIL_API_KEY")

# (model, tokenizer, model_id) — MLX weights are multi-GB; load once
# per process. Unlike the Ollama path (an HTTP daemon that queues
# requests server-side), MLX runs in-process against a single GPU
# model, so concurrent scheduler calls would each race a ~5 GB load
# and then contend on one Metal context. `_mlx_lock` serializes both
# the load and generation: the scheduler may pool >1 LOCAL call, but
# MLX executes them one at a time (the only sane mode for one local
# model, and what the matched-model bench measured).
_mlx_bundle = None
_mlx_lock = Lock()


def _get_mlx(model_id: str):
    """Load (or return the cached) MLX model + tokenizer from the local
    snapshot. Never auto-downloads — a missing snapshot raises with the
    in-app remedy so setup diagnostics can surface it verbatim."""
    global _mlx_bundle
    if _mlx_bundle is not None and _mlx_bundle[2] == model_id:
        return _mlx_bundle[0], _mlx_bundle[1]
    path = mlx_model_dir(model_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Local model {model_id!r} is not downloaded. Open "
            f"Settings → Local model → Download to fetch it "
            f"(expected at {path})."
        )
    from mlx_lm import load
    model, tokenizer = load(str(path))
    _mlx_bundle = (model, tokenizer, model_id)
    return model, tokenizer


# ── Streaming response consumer ───────────────────────────────────────────────
#
# Every provider branch in `complete()` issues `chat.completions.create(
# stream=True, stream_options={"include_usage": True})` and feeds the
# resulting iterator through `_consume_chat_stream`. The helper:
#   - reassembles `delta.content` into the visible response text
#   - reassembles `delta.reasoning_content` (chain-of-thought from
#     reasoning models like gpt-oss, kimi w/ thinking,
#     minimax with reasoning_split)
#   - captures TTFT (first content delta), TTFR (first reasoning delta),
#     and last-token timing from `time.monotonic()` against `t0`
#   - extracts `usage` from the final include-usage chunk: prompt /
#     completion / reasoning_tokens (when the provider populates
#     `completion_tokens_details.reasoning_tokens`)
#   - records the choice's `finish_reason` (stop / length / etc)
#
# Cache hits build a synthetic result via `_synthetic_cache_hit_stream`
# so the downstream code path is uniform: no cache-vs-not branching in
# `complete()`'s post-stream block.

@dataclass
class _StreamCollected:
    """Structured result of consuming one streaming chat completion.

    All token counts default to 0; all timing fields default to None
    so a stream that never emitted a content token can be distinguished
    from a stream that emitted at t=0ms.

    `reasoning_tokens_source` is one of:
      - "api"        usage.completion_tokens_details.reasoning_tokens
                     was populated (most authoritative).
      - "streamed"   counted via tokens.count_tokens over the streamed
                     reasoning_content text (provider streamed CoT but
                     didn't populate the usage field).
      - "estimated"  fell back to `completion_tokens - content_tokens`
                     after stripping <think> blocks + code fences from
                     the visible content. This is the Tinfoil case as
                     of 2026-05-07: Tinfoil hides reasoning server-side,
                     reports `reasoning_tokens=0` and never streams
                     reasoning_content, so the only signal is the
                     inflated `completion_tokens` vs. parsed content
                     delta. Conservative — `tokens.count_tokens` over-
                     counts (chars/3), so the estimate is a lower
                     bound on the real reasoning overhead.
      - None         no reasoning detected (delta ≈ 0 or completion=0).
    """
    content: str = ""
    reasoning_content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_tokens_source: str | None = None
    finish_reason: str | None = None
    ttft_ms: int | None = None
    ttfr_ms: int | None = None
    last_token_ms: int | None = None


def _extract_reasoning_tokens(usage) -> int:
    """Pull `reasoning_tokens` from `usage.completion_tokens_details` if
    the provider populated it. OpenAI-compatible reasoning models put
    the count there alongside `completion_tokens`. Returns 0 when the
    field is absent — caller falls back to counting reasoning_content."""
    if usage is None:
        return 0
    details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        return 0
    rt = getattr(details, "reasoning_tokens", None)
    if rt is None and isinstance(details, dict):
        rt = details.get("reasoning_tokens")
    try:
        return int(rt or 0)
    except (TypeError, ValueError):
        return 0


