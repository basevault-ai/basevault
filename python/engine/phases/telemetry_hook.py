"""Kernel telemetry hook (issue #912).

The legacy per-call observability — the ``llm-calls.jsonl`` stat records the
run-details UI + debug bundles read — was produced inside
``llm.complete()`` (via ``begin_stat_record`` / ``_record_usage`` /
``finalize_stat_record``) and, for chat, the sidecar's ``_tracked_complete``
bracket. The kernel path bypasses ``complete()``, so the phase entries would otherwise lose that telemetry.

``KernelTelemetryHook`` is the ENABLER: a ``kernel.LlmHook`` that reproduces
the same stat records from the kernel's call lifecycle. Register it on the
``ExecutionEnv`` (``env.register_llm_hook(hook)``) and every leaf call —
including retries — opens / records / closes a stat record exactly as the
legacy wrapper did. The runner / sidecar driver swaps then keep their
observability with no other change.

This module only depends on the (surviving) ``llm`` telemetry functions —
not on ``complete()`` / scheduler / retry — so it outlives the cutover.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, override

from kernel.abstractions import LlmCall, LlmHook, LlmResponse
from kernel.enums import LlmStatus, PhaseName, RetryType

from engine.llm import (
    begin_stat_record,
    estimate_prompt_tokens,
    finalize_stat_record,
    record_stage_counts,
    _append_event_jsonl,
    _get_rec,
    _record_usage,
)


class _LiveStreamEmitter:
    """Per-call stream callback that reproduces the legacy live telemetry the
    run-details UI needs to come alive mid-call:

    * a one-shot ``stream_progress`` JSONL event at the first token (the
      durable "stream started" signal the Rust live materializer reads), and
    * a ~1Hz ``live_tokens`` STDOUT heartbeat carrying the running chars/3
      token estimate (the Tauri shell overlays it on the per-call detail).

    The kernel provider calls ``call.stream_handler(piece)`` per chunk; the
    telemetry hook installs one of these as that handler. Also tracks
    ``last_token_ms`` so the finalize record carries it."""

    _INTERVAL_S = 1.0

    def __init__(self, call_id: str):
        self._cid = call_id
        self._t0 = time.monotonic()
        self._chars = 0
        self._started = False
        self._last_hb = self._t0
        self._quiet = bool(os.environ.get("BASEVAULT_LIVE_TOKENS_QUIET"))
        self.last_token_ms: int | None = None

    def __call__(self, piece: str) -> None:
        if not isinstance(piece, str) or not piece:
            return
        self._chars += len(piece)
        now = time.monotonic()
        self.last_token_ms = int((now - self._t0) * 1000)
        if not self._started:
            self._started = True
            try:
                _append_event_jsonl("stream_progress", {
                    "call_id": self._cid,
                    "ttft_ms": self.last_token_ms,
                    "ttfr_ms": None,
                })
            except Exception:
                pass
        if (now - self._last_hb) >= self._INTERVAL_S and not self._quiet:
            self._last_hb = now
            ct = self._chars // 3
            try:
                print(json.dumps({
                    "event": "live_tokens",
                    "call_id": self._cid,
                    "completion_tokens": ct,
                    "reasoning_tokens": 0,
                    "content_tokens": ct,
                    "last_token_ms": self.last_token_ms,
                }), flush=True)
            except (BrokenPipeError, OSError):
                pass


def _budget_for(mode, stage: str | None) -> dict | None:
    """The static budget snapshot (stage_cap / max_output / scaffolding) the
    per-call detail shows. Computed straight from ``llm.compute_budget(mode,
    stage)`` — NOT via the runner's mode global, which is unreadable here: the
    process runs runner.py as ``__main__`` so ``import runner`` would bind a
    second module copy whose ``_active_mode`` is None. ``mode`` is threaded in
    from build_stage_env instead. None when unresolvable (eval/off-path)."""
    if mode is None or stage is None:
        return None
    try:
        from engine.llm import compute_budget
        b = compute_budget(mode, stage)
        return {
            "stage_cap": b.max_input,
            "max_output": b.max_output,
            "scaffolding": b.scaffolding,
            "has_llm_calls": b.has_llm_calls,
        }
    except Exception:
        return None

# Statuses that mean the leaf did not return usable content.
_FAILURE_STATUSES = {
    LlmStatus.LOAD,
    LlmStatus.OTHER,
    LlmStatus.CAP_HIT,
    LlmStatus.PARSE_ERROR,
    LlmStatus.TIMEOUT_WITH_TOKENS,
    LlmStatus.ABORTED,
    LlmStatus.SKIPPED,
}


# The kernel's StageName.EXTRACTION.value is "extraction_splitter", but the
# legacy runner + run-details UI key the extraction stage on "extract". Map it
# so kernel extraction calls land in the same UI section / stat bucket as
# legacy (otherwise the Extraction panel reads empty and a stray
# "extraction_splitter" group appears).
_STAGE_ALIAS = {"extraction_splitter": "extract"}


# The per-call category label for a retry attempt, ported VERBATIM from the
# legacy `app/pipeline/retry.py::category_for_retry` that the kernel cutover
# deleted (#912 removed its callers in scheduler.py; #943 deleted the function).
# It is a pure string transform on the PARENT's emitted category:
#   1. strip the parent's trailing ` - retry/<class>`,
#   2. ACCUMULATE the structural step (`/half-1`, `/half-2`, `/sample-N`,
#      `/reasoning-off`, `/model-fallback`) onto the prefix,
#   3. append a fresh ` - retry/<class>` for THIS retry's failure class.
# Because it transforms the parent (which already carries the accumulated
# prefix), the structural path stacks — nested halves get unique names like
# `topic/half-1/half-2 - retry/sizing`. EVERY retry is labelled, including a
# plain transient load retry (`topic - retry/load`). The run-details UI renders
# these AND the chain-aware reclassifiers (`runner._apply_chain_aware_outcomes`
# + the Rust materializer) substring-match `/sample-` and `/reasoning-off` to
# derive success_sampled / success_reasoning_off.
_RETRY_SUFFIX_SEP = " - retry/"


def category_for_retry(
    parent_category: str,
    classification: str,
    structural_transform: str | None = None,
) -> str:
    """Build the per-call category string for a retry attempt:

      <structural-chain> - retry/<class>

    Structural transforms (halving, sampling, reasoning-off) ACCUMULATE
    in the prefix as `/half-1`, `/half-2`, `/sample-N`, `/reasoning-off`
    (caller passes the new step via `structural_transform`). The trailing
    ` - retry/<class>` suffix is REPLACED each call so it reflects only the
    MOST RECENT retry's classification.

    Examples:
      category_for_retry("topic", "load")
        → "topic - retry/load"
      category_for_retry("topic - retry/load", "sizing", "/half-1")
        → "topic/half-1 - retry/sizing"
      category_for_retry("topic/half-1 - retry/sizing", "load")
        → "topic/half-1 - retry/load"
      category_for_retry("topic/half-1 - retry/sizing", "sizing", "/half-2")
        → "topic/half-1/half-2 - retry/sizing"
    """
    base = parent_category
    idx = base.find(_RETRY_SUFFIX_SEP)
    if idx >= 0:
        base = base[:idx]
    if structural_transform:
        base = base + structural_transform
    return f"{base}{_RETRY_SUFFIX_SEP}{classification}"


# Structural-transform token for the work-reducing RetryTypes that carry no
# index (the indexed ones — HALVES, SAMPLE — are built in `_structural_transform`).
_RETRY_TRANSFORM_TOKEN = {
    RetryType.REASONING_OFF: "/reasoning-off",
    RetryType.MODEL_FALLBACK: "/model-fallback",
}

# The triggering-failure class for the ` - retry/<class>` suffix. Same
# load/sizing/other mapping the failure-label classifier uses; an unknown /
# None status leaves the child on its bare inherited category (no retry label).
_STATUS_RETRY_BUCKET = {
    LlmStatus.LOAD: "load",
    LlmStatus.CAP_HIT: "sizing",
    LlmStatus.PARSE_ERROR: "sizing",
    LlmStatus.TIMEOUT_WITH_TOKENS: "sizing",
    LlmStatus.OTHER: "other",
}


def stage_label(phase: PhaseName) -> str:
    v = phase.stage_name().value
    return _STAGE_ALIAS.get(v, v)


def _provider_mode_tag(provider_name: str) -> str:
    """Coarse mode tag for a provider's name() in stat records. The production
    binary only ever runs the attested TEE provider or a local one
    (ollama / mlx). Any other provider is non-attested and eval-only (never
    shipped), so it is tagged with its own name rather than naming a specific
    cloud provider here — keeping the production trust surface free of a
    non-attested provider reference."""
    n = provider_name.lower()
    if "tinfoil" in n:
        return "tinfoil"
    if "ollama" in n or "mlx" in n:
        return "local"
    return n.split()[0] if n.split() else "cloud"


# Live-skip bridge: the runner's skip-marker poller knows only stat call_ids;
# the telemetry hooks hold the stat→kernel-id map + the env cancellation
# manager. Hooks register here (subprocess-scoped, one per stage); the poller
# fans a skipped stat call_id out to all of them (no-op on finished stages).
_ACTIVE_HOOKS: list = []
_ACTIVE_HOOKS_LOCK = threading.Lock()


def _register_active_hook(hook) -> None:
    with _ACTIVE_HOOKS_LOCK:
        _ACTIVE_HOOKS.append(hook)


def skip_kernel_call(stat_cid: str) -> None:
    """Route a skipped stat call_id to the kernel (abort the matching in-flight
    call). Called by the runner's skip-marker poller alongside llm.register_skip
    so skip works on both the legacy and kernel paths."""
    with _ACTIVE_HOOKS_LOCK:
        hooks = list(_ACTIVE_HOOKS)
    for h in hooks:
        try:
            h.skip(stat_cid)
        except Exception:
            pass


class KernelTelemetryHook(LlmHook):
    """Reproduces ``llm-calls.jsonl`` stat records on the kernel path.

    ``category_for`` optionally maps a ``PhaseName`` to the per-call
    category column (chat distinguishes converse / grounded; pipeline
    stages can pass ``None`` to use just the stage)."""

    def __init__(
        self,
        session_id: str | None = None,
        category_for: Callable[[PhaseName], str | None] | None = None,
        payload_sink: Callable[[str, list, str], None] | None = None,
        failure_payload_sink: Callable[[str, list], None] | None = None,
        mode=None,
    ):
        self._session_id = session_id
        self._category_for = category_for
        # The BaseVault Mode (TEE / LOCAL) — used to resolve the per-call
        # budget (stage_cap / max_output / scaffolding). Threaded in from
        # build_stage_env since the runner's mode global isn't reachable here.
        self._mode = mode
        # Optional full-IO capture (stat_call_id, messages, content) — the
        # sidecar passes its _capture_payload so the dev-tab prompt/response
        # capture is reproduced on the kernel chat path.
        self._payload_sink = payload_sink
        # Optional always-on FAILURE-prompt capture (stat_call_id, messages) —
        # the pipeline passes llm._log_call_failure_payload so every failed
        # call leaves its prompt in llm-payloads.jsonl, exactly as the legacy
        # runner wrapper's exception path did (#266). The success sink only
        # fires for str payloads, so a from_status failure (payload=None — an
        # injected or real LOAD/timeout/parse_error) would otherwise log
        # nothing and the dev-tab "full prompt + response" view shows it blank.
        self._failure_payload_sink = failure_payload_sink
        self._lock = threading.Lock()
        self._call_ids: dict[str, str] = {}   # kernel call.id → stat call_id
        self._t0: dict[str, float] = {}
        # kernel call.id → live-stream emitter (stream_progress + live_tokens),
        # so finalize can read its last_token_ms.
        self._emitters: dict[str, _LiveStreamEmitter] = {}
        # kernel call.id → stat call_id, NOT popped at completion, so a phase
        # can emit a per-call `counts` event after it parses the output (the
        # cid is gone from _call_ids by then). Subprocess-scoped, bounded by
        # the run's call count.
        self._cid_by_call: dict[str, str] = {}
        # kernel call.id → attempt depth (1 = original, 2+ = a retry/halve/
        # sample child). Same not-popped lifetime as _cid_by_call so a child
        # started after its parent completed can still read the parent's
        # depth. Lets the diagnostics adapter surface is_retry / attempt for
        # the kernel path (the legacy retry-suffix category is gone).
        self._attempt_by_call: dict[str, int] = {}
        # Live-skip bridge: stat call_id → kernel call.id (the inverse of
        # _call_ids), + the env's cancellation manager + stat_cids the user
        # skipped before their call mapped. The runner's skip-marker poller
        # carries stat call_ids; the kernel aborts by kernel call.id.
        self._kernel_call_by_cid: dict[str, str] = {}
        self._cancellation_manager = None
        self._pending_skips: set[str] = set()
        # Per-call category column (the legacy "batch-3-of-5" / per-topic /
        # "insights" labels). Phases register a label on each root call via
        # set_category(); retries / halves inherit it down the
        # previous_call_id chain.
        self._categories: dict[str, str] = {}
        # Retry-strategy labels: the kernel decides the strategy for each
        # retry (RetryType) and hands it to hook_llm_completed. We stash the
        # decision keyed by the FAILING call's id, then — when its child
        # starts — build the child's category via `category_for_retry` from the
        # PARENT's emitted label, accumulating the `/<strategy>` step.
        self._retry_strategy_by_call: dict[str, RetryType] = {}
        # Parent call.id → the failure bucket (load/sizing/other) that
        # triggered its retry, for the ` - retry/<bucket>` label half.
        self._retry_bucket_by_call: dict[str, str] = {}
        # call.id → the FULL emitted category (with accumulated retry prefix +
        # suffix). `category_for_retry` transforms this to build each child's
        # label, so the structural path stacks down a halve/sample tree.
        self._emitted_category: dict[str, str] = {}
        # Parent call.id → count of HALVES children seen, so the two halves a
        # HALVES decision spawns get `/half-1` and `/half-2`.
        self._retry_child_idx: dict[str, int] = {}
        # call.id → SAMPLE ordinal down the chain (a sample ladder spawns one
        # child per parent; the ordinal increments so successive samples read
        # `/sample-1`, `/sample-2`, …).
        self._sample_ordinal: dict[str, int] = {}
        _register_active_hook(self)

    def skip(self, stat_cid: str) -> None:
        """Abort the in-flight kernel call behind a stat call_id (the id the
        run-details ✕ writes to skipped_calls/). Remembers the skip if the
        call hasn't mapped yet so hook_llm_started can apply it on start."""
        with self._lock:
            self._pending_skips.add(stat_cid)
            kernel_id = self._kernel_call_by_cid.get(stat_cid)
            cm = self._cancellation_manager
        if kernel_id is not None and cm is not None:
            try:
                cm.skip_call(kernel_id)
            except Exception:
                pass

    def set_category(self, call_id: str, category: str) -> None:
        """Tag a call's stat-record category column. Called by phases (which
        reach this hook through ``execution_env.llm_hooks``) at call-build
        time — see ``set_call_category``."""
        if category:
            with self._lock:
                self._categories[call_id] = category

    def _resolve_category(self, call: LlmCall, phase: PhaseName) -> str | None:
        # Prefer the phase-assigned per-call label; inherit it down the
        # retry / halve chain (children get fresh ids); fall back to the
        # optional per-phase category_for (chat's converse / grounded).
        with self._lock:
            cat = self._categories.get(call.id)
            if cat is None and call.previous_call_id:
                cat = self._categories.get(call.previous_call_id)
                if cat is not None:
                    self._categories[call.id] = cat  # propagate down the chain
        if cat is not None:
            return cat
        return self._category_for(phase) if self._category_for else None

    def _retry_category(self, call: LlmCall, base: str) -> str:
        """The full category label for `call`, via the legacy `category_for_retry`
        transform. For a root / first attempt (parent didn't trigger a retry) the
        bare `base` is returned unchanged. For a retry child it transforms the
        PARENT's emitted category — stripping the old ` - retry/<class>`,
        accumulating this step's structural token (`/half-N`, `/sample-N`,
        `/reasoning-off`, `/model-fallback`; none for a plain FULL_RETRY) onto the
        prefix, and appending ` - retry/<class>` for the failure that triggered
        it. So nested halves stack (`…/half-1/half-2 - retry/sizing`) and every
        retry — load/other included — shows why it fired."""
        parent = call.previous_call_id
        if not parent:
            return base
        with self._lock:
            strat = self._retry_strategy_by_call.get(parent)
            if strat is None:
                # Parent wasn't a retry-triggering failure (root / NO_RETRY /
                # successful terminal) — keep the bare inherited category.
                return base
            classification = self._retry_bucket_by_call.get(parent)
            if not classification:
                # Triggering status didn't map to a class — leave it bare.
                return base
            transform = self._structural_transform(strat, parent, call.id)
            parent_category = self._emitted_category.get(parent, base)
        return category_for_retry(parent_category, classification, transform)

    def _structural_transform(
        self, strat: RetryType, parent: str, child_id: str
    ) -> str | None:
        """The structural step token a retry child accumulates, per RetryType.
        Caller holds `self._lock`. HALVES spawns two children → a per-parent
        `/half-1` / `/half-2`; SAMPLE is a one-child-per-parent ladder → a
        `/sample-N` ordinal that increments down the chain. REASONING_OFF /
        MODEL_FALLBACK are fixed tokens; FULL_RETRY has none."""
        if strat == RetryType.HALVES:
            idx = self._retry_child_idx.get(parent, 0) + 1
            self._retry_child_idx[parent] = idx
            return f"/half-{idx}"
        if strat == RetryType.SAMPLE:
            n = self._sample_ordinal.get(parent, 0) + 1
            self._sample_ordinal[child_id] = n
            return f"/sample-{n}"
        return _RETRY_TRANSFORM_TOKEN.get(strat)

    @staticmethod
    def _request_extras(execution_env) -> dict:
        # The two always-present per-call kwargs the run-details modal shows:
        # the pipeline runs at temperature 0, and reasoning is the phase's
        # thinking flag.
        return {
            "temperature": 0.0,
            "reasoning": bool(getattr(execution_env, "thinking", False)),
        }

    def _stamp_cache_key(self, cid: str, call: LlmCall, execution_env) -> None:
        # Stamp the per-call record's cache_key so the run-details Copy/Delete
        # cache buttons appear (canBust = !!cache_key) and resolve to the
        # KernelDiskCache file. cache_key == cache_entry_id(<kernel key>), the
        # same id the cache uses for its filename (single source of truth).
        try:
            from engine.phases.kernel_cache import cache_entry_id
            kkey = execution_env._cache_key(call)
            rec = _get_rec(cid)
            if rec is not None:
                rec["cache_key"] = cache_entry_id(kkey)
        except Exception:
            pass  # telemetry must never break the call

    @override
    def hook_llm_queued(self, call: LlmCall, execution_env) -> None:
        # No record yet: the run-details row should appear when the call
        # STARTS (hook_llm_started), not when it's queued. A separate queued
        # state is tracked in a follow-up ticket.
        pass

    @override
    def hook_llm_started(self, call: LlmCall, execution_env) -> None:
        phase: PhaseName = execution_env.phase.name()
        stage = stage_label(phase)
        # Base (inherited) category, then build this attempt's full retry label
        # via `category_for_retry` from the PARENT's emitted category (so the
        # structural prefix accumulates down a halve/sample tree). `_categories`
        # keeps the BARE inherited label propagating; the full label is recorded
        # in `_emitted_category` so THIS call's own children transform from it.
        category = self._resolve_category(call, phase)
        if category is not None:
            category = self._retry_category(call, category)
            with self._lock:
                self._emitted_category[call.id] = category
        model = execution_env.model_spec.model()
        provider = execution_env.model_spec.inference_provider().name()
        try:
            pt_est = int(estimate_prompt_tokens(call.messages))
        except Exception:
            pt_est = None
        # Resolve retry linkage BEFORE opening the stat record so the
        # attempt / retry_of_call_id land in the begin event written to
        # llm-calls.jsonl — which is what the run-details rollup reads.
        # A retry / halve / sample child is a fresh call whose
        # ``previous_call_id`` points at the call it descends from;
        # record the parent's cid + this call's attempt depth so the
        # rollup derives is_retry / attempt instead of reading every
        # kernel call as an independent first attempt (which hid, e.g.,
        # 5 LOAD retries of one chat hop as 5 fresh calls — the
        # motivating chat-503/502 case). Stamping these on the
        # in-memory rec AFTER begin_stat_record (as the pre-fix code
        # did) never reached the jsonl, so retries showed null linkage.
        with self._lock:
            _parent_kid = call.previous_call_id
            _attempt = (
                self._attempt_by_call.get(_parent_kid, 0) + 1
                if _parent_kid else 1
            )
            _retry_of_cid = (
                self._cid_by_call.get(_parent_kid) if _parent_kid else None
            )
        cid = begin_stat_record(
            stage,
            category,
            model_hint=model,
            attempt=_attempt,
            retry_of_call_id=_retry_of_cid,
            prompt_tokens_est=pt_est,
            session_id=self._session_id,
            provider=_provider_mode_tag(provider),
            max_tokens_reserved=call.max_tokens or None,
            request_extras=self._request_extras(execution_env),
            budget=_budget_for(self._mode, stage),
        )
        self._stamp_cache_key(cid, call, execution_env)
        # Install the live-stream emitter as the call's stream_handler so the
        # provider's per-chunk callback fires stream_progress + live_tokens
        # (don't clobber a handler a phase already set, e.g. chat streaming).
        emitter = _LiveStreamEmitter(cid)
        if call.stream_handler is None:
            call.stream_handler = emitter
        with self._lock:
            self._call_ids[call.id] = cid
            self._t0[call.id] = time.monotonic()
            self._emitters[call.id] = emitter
            self._cid_by_call[call.id] = cid
            self._kernel_call_by_cid[cid] = call.id
            # Persist this call's attempt depth so a later retry child
            # (whose ``previous_call_id`` is this call) resolves its own
            # depth + parent cid above, before its begin_stat_record.
            self._attempt_by_call[call.id] = _attempt
            # Capture the env's cancellation manager so skip() can abort calls.
            if self._cancellation_manager is None:
                self._cancellation_manager = getattr(
                    execution_env, "cancellation_manager", None
                )
            already_skipped = cid in self._pending_skips
            cm = self._cancellation_manager
        # If the user already skipped this call_id (marker landed before the
        # call started), abort it now.
        if already_skipped and cm is not None:
            try:
                cm.skip_call(call.id)
            except Exception:
                pass

    def emit_counts(self, call_id: str, input: dict | None, output: dict | None) -> None:
        """Emit the per-call `counts` event (input chunks / output items) the
        legacy stages write via record_stage_counts. Called by a phase after it
        parses a call's output — see ``record_call_counts``."""
        with self._lock:
            cid = self._cid_by_call.get(call_id)
        if cid is None:
            return
        try:
            record_stage_counts(cid, input=input, output=output)
        except Exception:
            pass

    @override
    def hook_llm_completed(
        self,
        call: LlmCall,
        execution_env,
        response: LlmResponse,
        retry: RetryType,
        from_cache: bool,
        should_cache: bool,
    ) -> None:
        # The kernel's retry decision for THIS failure picks the strategy of
        # the child it spawns (a HALVES/SAMPLE/REASONING_OFF/MODEL_FALLBACK
        # cascade step). Stash it keyed by this call's id; the child reads it
        # in `_retry_category` when it starts (hook ordering guarantees
        # this completion fires before the child's hook_llm_started). Only a
        # real retry is recorded — NO_RETRY / a successful terminal call
        # spawns nothing.
        if retry is not None and retry != RetryType.NO_RETRY:
            bucket = _STATUS_RETRY_BUCKET.get(response.status)
            with self._lock:
                self._retry_strategy_by_call[call.id] = retry
                if bucket:
                    self._retry_bucket_by_call[call.id] = bucket
        with self._lock:
            cid = self._call_ids.pop(call.id, None)
            t0 = self._t0.pop(call.id, None)
            emitter = self._emitters.pop(call.id, None)
        last_token_ms = emitter.last_token_ms if emitter is not None else None
        model = execution_env.model_spec.model()
        provider = execution_env.model_spec.inference_provider().name()
        # Cache hits skip the scheduler, so hook_llm_started never fired and
        # there's no open record. Synthesize one now (duration 0) so the hit
        # is VISIBLE in run-details with cached=true — matching how the legacy
        # complete() path always wrote a record then stamped it cached.
        if cid is None:
            if not from_cache:
                return  # genuinely no record (non-cache path shouldn't hit this)
            phase = execution_env.phase.name()
            try:
                pt_est = int(estimate_prompt_tokens(call.messages))
            except Exception:
                pt_est = None
            cid = begin_stat_record(
                stage_label(phase),
                self._resolve_category(call, phase),
                model_hint=model,
                prompt_tokens_est=pt_est,
                session_id=self._session_id,
                provider=_provider_mode_tag(provider),
                max_tokens_reserved=call.max_tokens or None,
                request_extras=self._request_extras(execution_env),
                budget=_budget_for(self._mode, stage_label(phase)),
            )
            self._stamp_cache_key(cid, call, execution_env)
            t0 = None

        duration_ms = (
            int((time.monotonic() - t0) * 1000)
            if t0 is not None
            else int((response.duration or 0) * 1000)
        )
        finish_reason = "length" if response.status == LlmStatus.CAP_HIT else "stop"
        ttft_ms = int(response.ttft * 1000) if response.ttft else None
        # Split completion tokens into visible content vs reasoning, the same
        # "estimated" way legacy does (reasoning = completion - content, where
        # content is chars/3 of the returned text). The kernel response payload
        # is the visible content; the API's completion_tokens covers both.
        payload_str = response.payload if isinstance(response.payload, str) else ""
        content_tokens = (len(payload_str) // 3) if payload_str else None
        reasoning_tokens = 0
        reasoning_tokens_source = None
        if response.completion_tokens and content_tokens is not None:
            reasoning_tokens = max(0, response.completion_tokens - content_tokens)
            reasoning_tokens_source = "estimated"
        # Embedding responses carry a vector payload; chat/synthesis carry a
        # string. Either way the token counts come off the response.
        _record_usage(
            _provider_mode_tag(provider),
            model,
            response.prompt_tokens,
            response.completion_tokens,
            call_id=cid,
            finish_reason=finish_reason,
            ttft_ms=ttft_ms,
            last_token_ms=last_token_ms,
            content_tokens=content_tokens,
            reasoning_tokens=reasoning_tokens,
            reasoning_tokens_source=reasoning_tokens_source,
            max_tokens_reserved=call.max_tokens or None,
        )
        success = response.exception is None and response.status not in _FAILURE_STATUSES
        error: dict[str, Any] | None = None
        if response.exception is not None:
            etype = type(response.exception)
            error = {
                "type": f"{etype.__module__}.{etype.__name__}",
                "message": str(response.exception),
            }
        rec = _get_rec(cid)
        if rec is not None:
            if from_cache:
                # Stamp the live record so run-details shows cached=true,
                # exactly as legacy complete() did on a cache hit.
                rec["cached"] = True
            # Record the kernel's categorized status so the failure-label
            # adapter (runner._failure_class_for_label) reads the bucket
            # straight off it — the kernel ALREADY classified the outcome
            # (LlmStatus encodes load / sizing / other), so there is no need
            # to re-derive it from the exception class via the legacy
            # retry.classify_bucket.
            if response.status is not None:
                rec["llm_status"] = response.status.name
        finalize_stat_record(
            cid, success=success, duration_ms=duration_ms, error=error
        )
        if self._payload_sink is not None and isinstance(response.payload, str):
            try:
                self._payload_sink(cid, call.messages, response.payload)
            except Exception:
                pass  # telemetry must never break the call
        # Failed call: log its prompt unconditionally (the success sink above
        # only fires for str payloads, so a from_status failure logs nothing).
        # _log_call_failure_payload's own written-set dedup makes this a no-op
        # when the success path already captured a partial mid-stream payload.
        if not success and self._failure_payload_sink is not None:
            try:
                self._failure_payload_sink(cid, call.messages)
            except Exception:
                pass  # telemetry must never break the call


def set_call_category(execution_env, call: LlmCall, category: str | None) -> None:
    """Tag ``call``'s stat-record category column (the legacy per-call label:
    per-topic for patterns, ``batch-N-of-M`` for entities, etc.).

    Phases call this at call-build time. The telemetry hook isn't handed to
    phases directly, but the bound env exposes its hooks — so we find the
    ``KernelTelemetryHook`` among ``execution_env.llm_hooks`` and register the
    label there. No-op when telemetry isn't attached (e.g. eval/offline envs).
    """
    if not category:
        return
    for hook in getattr(execution_env, "llm_hooks", []) or []:
        if isinstance(hook, KernelTelemetryHook):
            hook.set_category(call.id, category)
            return


def record_call_counts(execution_env, call: LlmCall, input: dict, output: dict) -> None:
    """Emit a per-call ``counts`` event (legacy record_stage_counts parity).

    Phases call this after parsing a call's output (e.g. extract: input
    ``{"chunks": 1}``, output ``{"facts": n}``). Routed through the telemetry
    hook on ``execution_env.llm_hooks`` so it resolves the call's stat id."""
    for hook in getattr(execution_env, "llm_hooks", []) or []:
        if isinstance(hook, KernelTelemetryHook):
            hook.emit_counts(call.id, input, output)
            return
