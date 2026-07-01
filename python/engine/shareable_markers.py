"""Builders that reduce live retrieval / telemetry state to the typed,
content-free markers in ``shareable.py``.

These helpers do the reading of vault-adjacent state (the llm-calls
telemetry, the per-run vector store, the run dir). They are content-free
by construction at the boundary: they may only ever return the frozen
``shareable`` dataclasses, every field of which is a number / bool /
ISO-Z timestamp / closed-enum. ``shareable.emit()`` re-validates every
leaf regardless, so a builder bug crashes rather than leaking. No raw
``record_id``, query, candidate, prompt, or response text is ever read
into a marker — only counts, lengths, norms, distances, flags, and
closed enums.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

from engine.common.dates import (
    derive_ended_at_iso as _derive_ended_at_iso,
    iso_delta_ms as _iso_delta_ms,
    iso_min_max as _iso_min_max,
    iso_or_none as _iso_or_none,
    now_iso_z as _now_iso_z,
)
from engine.common.stats import mean as _mean, percentile as _percentile
from engine.common.status import (
    is_success_outcome as _is_success_outcome,
    outcome_from_label as _outcome_from_label,
)
from engine.common.utils import (
    call_id_str as _call_id_str,
    safe_cid_int as _safe_cid_int,
    stage_order_rank as _stage_order_rank,
)
from engine.shareable import (
    SCHEMA_VERSION,
    _PERMA_ID_RE,
    _SESSION_ID_RE,
    ChatMarker,
    EmbeddingStats,
    FinishReason,
    HopMarker,
    HopOutcome,
    KindCount,
    LlmCall,
    LlmCallsBlock,
    LookupShape,
    Outcome,
    OutcomeCount,
    RecordKind,
    RetrieveSkippedReason,
    RetryClass,
    RetryClassCount,
    RetryTransform,
    RunMarker,
    ScoreSamples,
    SizeBucket,
    SizeHistBucket,
    StageStat,
    StageToken,
    StoreStats,
    category_token,
    model_token,
    provider_token,
    stage_token,
)

# Char-length bucket edges for the size histograms. Right-open ranges;
# the last bucket is unbounded. Order matches ``SizeBucket``.
_SIZE_EDGES: tuple[tuple[SizeBucket, int, int | None], ...] = (
    (SizeBucket.lt_200, 0, 200),
    (SizeBucket.b_200_500, 200, 500),
    (SizeBucket.b_500_1k, 500, 1000),
    (SizeBucket.b_1k_2k, 1000, 2000),
    (SizeBucket.gte_2k, 2000, None),
)


_FINISH = {m.value for m in FinishReason} - {"other", "none"}


def _finish_reason(v: object) -> FinishReason:
    if not v:
        return FinishReason.none
    s = str(v).lower()
    return FinishReason(s) if s in _FINISH else FinishReason.other


def _retry_suffix(category: object) -> str:
    """Everything AFTER the first ``/`` in the category, lowercased.

    The structural retry chain (``/half-N/sample-N/reasoning-off -
    retry/<class>``) only ever appears after the first ``/``; the part
    BEFORE the first ``/`` is the free topic/filename/entry prefix
    (vault content) and is dropped entirely here — it is never read or
    matched. As a second, independent guarantee, the functions below
    emit ONLY closed enum tokens parsed from this suffix, never any
    substring of it, so nothing free-form can reach the marker even if
    the suffix still held content."""
    if not isinstance(category, str) or "/" not in category:
        return ""
    return category.split("/", 1)[1].lower()


_RETRY_CLASS_RE = re.compile(r"retry/([a-z]+)")
_RETRY_XF = (
    (re.compile(r"half-\d"), RetryTransform.half),
    (re.compile(r"sample-\d"), RetryTransform.sample),
    (re.compile(r"reasoning-off"), RetryTransform.reasoning_off),
)
_RETRY_CLASSES = {m.value for m in RetryClass} - {"other", "none"}


def _retry_class(category: object) -> RetryClass:
    suffix = _retry_suffix(category)
    if not suffix:
        return RetryClass.none
    m = _RETRY_CLASS_RE.search(suffix)
    if not m:
        return RetryClass.none
    return RetryClass(m.group(1)) if m.group(1) in _RETRY_CLASSES \
        else RetryClass.other


def _retry_transforms(category: object) -> tuple[RetryTransform, ...]:
    suffix = _retry_suffix(category)
    if not suffix:
        return ()
    return tuple(tok for rx, tok in _RETRY_XF if rx.search(suffix))


def _max_call_id(calls_jsonl: Path) -> int:
    """Highest numeric ``call_id`` already on disk — the per-turn slice
    baseline. 0 when the file is absent/empty (so the whole turn's
    calls are included)."""
    hi = 0
    try:
        with calls_jsonl.open() as fh:
            for raw in fh:
                try:
                    cid = json.loads(raw).get("call_id")
                    if cid is not None:
                        hi = max(hi, int(cid))
                except (ValueError, TypeError):
                    continue
    except OSError:
        return 0
    return hi


def llm_calls_baseline(calls_jsonl: Path) -> int:
    """Public: snapshot the call-id high-water mark before a turn so
    ``build_llm_calls_block`` can slice exactly this turn's calls."""
    return _max_call_id(calls_jsonl)


def build_llm_calls_block(
    calls_jsonl: Path, since_call_id: int,
    *,
    run_dir: Path | None = None,
) -> LlmCallsBlock:
    """Project the ``llm-calls.jsonl`` records with ``call_id`` strictly
    greater than ``since_call_id`` to a content-free block.

    Delegates classification to the runner's existing pipeline —
    ``_materialize_calls_from_jsonl`` (rebuilds the begin/end/counts
    record + the ``aborted``/``skipped`` flags from on-disk markers),
    ``_classify_outcome`` (the SINGLE source-of-truth label producer),
    and ``_apply_chain_aware_outcomes`` (re-labels leaves to
    ``success_sampled`` / ``success_reasoning_off``). This is the
    same pipeline the run-details UI consumes, so the diagnostic
    YAML carries the SAME outcome label the user sees on screen,
    no parallel classifier to keep in sync.

    The runner imports lazily — ``runner.py`` already imports this
    module (via ``_emit_shareable_run_marker``), so a top-level
    import would cycle. Lazy import inside the function works.

    ``run_dir`` is passed through to the runner so its skip-marker
    walk (``stages/skipped_calls/<call_id>``) can flag user-skipped
    calls. ``None`` when only a slice (chat per-turn) is needed —
    the runner gracefully degrades to no-skip-marker detection.
    """
    if not calls_jsonl.exists():
        return LlmCallsBlock(0, 0, 0, 0.0, ())
    # Lazy import: runner.py imports shareable_markers via
    # ``_emit_shareable_run_marker``, so a top-level import here
    # would cycle. The function-level import resolves only when
    # build_llm_calls_block actually runs.
    from engine.runner import (
        _apply_chain_aware_outcomes,
        _classify_outcome,
        _materialize_calls_from_jsonl,
    )

    ended_at = _now_iso_z()
    records = _materialize_calls_from_jsonl(calls_jsonl, ended_at)
    # Slice to the requested call-id range. Used by the chat sidecar
    # so a per-turn marker shows only that turn's calls.
    if since_call_id > 0:
        records = [
            r for r in records
            if _safe_cid_int(r.get("call_id"), default=0) > since_call_id
        ]
    if run_dir is not None:
        _stamp_skipped_from_markers(records, run_dir)
    for rec in records:
        _normalize_error_to_dict(rec)
        rec["outcome"] = _classify_outcome(rec)
    _apply_chain_aware_outcomes(records)

    calls: list[LlmCall] = []
    tot_p = tot_c = 0
    wall = 0.0
    for rec in records:
        pt = rec.get("prompt_tokens")
        ct = rec.get("completion_tokens")
        rt = rec.get("reasoning_tokens")
        cont = rec.get("content_tokens")
        dur = rec.get("duration_ms")
        tot_p += int(pt) if isinstance(pt, (int, float)) else 0
        tot_c += int(ct) if isinstance(ct, (int, float)) else 0
        wall += float(dur) if isinstance(dur, (int, float)) else 0.0
        total = None
        if isinstance(pt, (int, float)) or isinstance(ct, (int, float)):
            total = (int(pt) if isinstance(pt, (int, float)) else 0) + (
                int(ct) if isinstance(ct, (int, float)) else 0
            )
        calls.append(LlmCall(
            stage=stage_token(rec.get("stage")),
            category=category_token(rec.get("category")),
            model=model_token(rec.get("model")),
            provider=provider_token(rec.get("mode")),
            reasoning=bool(rec.get("reasoning")),
            outcome=_outcome_from_label(rec.get("outcome")),
            prompt_tokens=int(pt) if isinstance(pt, (int, float)) else None,
            completion_tokens=int(ct) if isinstance(ct, (int, float)) else None,
            reasoning_tokens=int(rt) if isinstance(rt, (int, float)) else None,
            content_tokens=int(cont) if isinstance(cont, (int, float)) else None,
            total_tokens=total,
            duration_ms=float(dur) if isinstance(dur, (int, float)) else None,
            ttft_ms=(
                float(rec["ttft_ms"])
                if isinstance(rec.get("ttft_ms"), (int, float))
                else None
            ),
            max_tokens_reserved=(
                int(rec["max_tokens_reserved"])
                if isinstance(rec.get("max_tokens_reserved"), (int, float))
                else None
            ),
            attempt=int(rec.get("attempt") or 1),
            is_retry=bool(rec.get("retry_of_call_id")),
            parse_error=bool(rec.get("parse_error")),
            started_at=_iso_or_none(rec.get("started_at_iso")),
            # ``_materialize_calls_from_jsonl`` doesn't propagate the
            # end event's ``ts`` onto the rec, so derive ended_at from
            # started_at + duration_ms instead of re-scanning the jsonl.
            ended_at=_derive_ended_at_iso(
                rec.get("started_at_iso"), rec.get("duration_ms"),
            ),
            call_id=_call_id_str(rec.get("call_id")),
            cached=(
                bool(rec["cached"])
                if isinstance(rec.get("cached"), bool)
                else None
            ),
            finish_reason=_finish_reason(rec.get("finish_reason")),
            retry_of_call_id=_call_id_str(rec.get("retry_of_call_id")),
            retry_class=_retry_class(rec.get("category")),
            retry_transforms=_retry_transforms(rec.get("category")),
        ))
    return LlmCallsBlock(
        call_count=len(calls),
        total_prompt_tokens=tot_p,
        total_completion_tokens=tot_c,
        wall_ms_total=wall,
        calls=tuple(calls),
    )


def _normalize_error_to_dict(rec: dict) -> None:
    """``runner._classify_outcome`` expects ``rec[\"error\"]`` to be a
    dict with ``\"class\"``/``\"message\"`` keys (the live runtime
    builds it that way). Older jsonl streams and synthetic test
    fixtures sometimes write plain strings — normalize them in place
    so the classifier doesn't ``AttributeError`` on ``.get(\"class\")``.

    This is a SHAPE adapter, not a classification step — the
    classifier remains the single source of truth for the label."""
    err = rec.get("error")
    if err is None:
        return
    if isinstance(err, dict):
        return
    if isinstance(err, str):
        rec["error"] = {"class": "", "message": err}


def _stamp_skipped_from_markers(
    records: list[dict], run_dir: Path,
) -> None:
    """Stamp ``rec[\"skipped\"] = True`` on records whose call_id has a
    matching ``skipped_calls/<call_id>`` marker file in the run dir.
    Mirrors the same step the runner's `_materialize_calls_from_jsonl`
    follow-up performs (the marker is stage-side state, separate from
    the jsonl event stream)."""
    marker_dir = run_dir / "skipped_calls"
    if not marker_dir.is_dir():
        return
    try:
        skipped = {p.name for p in marker_dir.iterdir() if p.is_file()}
    except OSError:
        return
    for rec in records:
        cid = rec.get("call_id")
        if cid in skipped:
            rec["skipped"] = True


def _kind(k: str) -> RecordKind | None:
    try:
        return RecordKind(k)
    except ValueError:
        return None


def build_store_stats(store) -> StoreStats:
    """Per-turn store snapshot — total records + closed-enum-keyed kind
    map. Reads only counts (a cheap SELECT count(*) inside SQLite); no
    record text, file id, topic, or record_id is materialized.

    Raises through on read failure rather than masking. A swallowed
    exception that returned ``total_records=0`` would have been
    indistinguishable from a real empty store, and
    ``_resolve_skipped_reason`` keys ``empty_store`` off exactly that
    number — Codex P2 (#818). The caller wraps the call when it wants
    best-effort behavior; the resolver gates ``empty_store`` on
    ``diag`` absence so a transient zero here can't override a healthy
    retrieve diagnostic. Tail-end ``count_by_kind`` failures still
    degrade to an empty kind map (the total is the load-bearing field
    for skip-reason; per-kind counts are cosmetic)."""
    total = int(store.count() or 0)
    by_kind: list[KindCount] = []
    try:
        for k, n in (store.count_by_kind() or {}).items():
            rk = _kind(str(k))
            if rk is not None:
                by_kind.append(KindCount(rk, int(n)))
    except Exception:
        pass
    return StoreStats(total_records=total, records_by_kind=tuple(by_kind))


def _perma_id_or_none(v: object) -> str | None:
    """Accept only a string that matches the 4-letter perma-id alphabet;
    anything else → ``None``. The runtime guard rejects free strings on
    every path except ``.bound_run``, so this gating at the builder
    boundary keeps a malformed value from ever reaching the marker."""
    if isinstance(v, str) and _PERMA_ID_RE.match(v):
        return v
    return None


def _session_id_or_empty(v: object) -> str:
    """Accept only a string that matches the 16-hex session-id shape;
    anything else → ``""`` (which the guard rejects unless on a
    permitted path — but ``ChatMarker.session_id`` is the only path
    that accepts the 16-hex shape, so an empty string would crash
    loudly rather than smuggle a free string). The chatbot sidecar
    today mints session_id as ``uuid.uuid4().hex[:16]``; this gate
    ensures a malformed inbound value (test fixture, future minting
    bug) can't bypass the runtime guard."""
    if isinstance(v, str) and _SESSION_ID_RE.match(v):
        return v
    return ""


# Closed mapping from ``chatbot_turn._persona_for_call`` outputs (and
# the dispatch-time ``hop_outcome`` decision in chatbot_turn.run) to
# the closed-enum values the marker carries. Keeps the mapping in one
# place — change it here when the persona vocab evolves.
_HOP_OUTCOME_FROM_STR: dict[str, HopOutcome] = {
    "tool_call": HopOutcome.tool_call,
    "prose_answer": HopOutcome.prose_answer,
    "invalid_tool_call": HopOutcome.invalid_tool_call,
    "malformed_tool_call": HopOutcome.malformed_tool_call,
}


def _kind_counts_from_dict(by_kind: object) -> tuple[KindCount, ...]:
    """The per-lookup ``kind_counts: dict[str, int]`` that
    ``chatbot_dispatch._dispatch_search`` accumulates → a tuple of
    ``KindCount`` rows (closed-enum kind, int count). Unknown / non-
    RecordKind strings are dropped, mirroring the run-marker builder
    pattern. Stable order: sorted by kind name so the YAML diff stays
    deterministic across runs of the same input."""
    out: list[KindCount] = []
    if isinstance(by_kind, dict):
        for k, n in sorted(by_kind.items()):
            rk = _kind(str(k))
            if rk is not None and isinstance(n, (int, float)):
                out.append(KindCount(rk, int(n)))
    return tuple(out)


def _score_samples_from_tuple(v: object) -> ScoreSamples | None:
    """The dispatcher emits ``score_samples`` as either ``None`` (zero
    rows kept, filter-only lookup) or a 3-tuple
    ``(closest, middle, farthest)`` of floats. Marker shape converts
    that to a frozen ``ScoreSamples`` dataclass; anything malformed
    degrades to ``None`` so the field is omitted in the YAML."""
    if v is None:
        return None
    try:
        c, m, f = v
        return ScoreSamples(
            closest=float(c), middle=float(m), farthest=float(f),
        )
    except (TypeError, ValueError):
        return None


def _record_kinds_from_list(v: object) -> tuple[RecordKind, ...]:
    """Per-lookup tuple of ``RecordKind`` closed-enum values, derived
    from the dispatcher's ``entry_types: list[str]`` (the model's
    request) or ``has_neighbor_kinds: list[str]`` (the kind-half of
    each anchor pair). Unknown / non-RecordKind strings drop silently —
    the validator upstream constrains the kind vocab, so a value here
    that doesn't map cleanly is structurally impossible from
    production code but defensive against test fixtures."""
    if not isinstance(v, (list, tuple)):
        return ()
    out: list[RecordKind] = []
    for k in v:
        rk = _kind(str(k))
        if rk is not None:
            out.append(rk)
    return tuple(out)


def _build_lookup_shape(d: dict) -> LookupShape:
    """One per-lookup diag dict (from ``_dispatch_search``'s
    ``per_lookup`` list) → one ``LookupShape``. Reads only content-free
    leaves: closed-enum kinds, ints, floats, bools. Free-text request
    fields (query text, exact-match substrings, neighbor record_ids)
    were never emitted by the dispatcher — only their counts / kinds /
    presence-bools cross this boundary."""
    query_present = bool(d.get("query_present"))
    # Embed signals are populated only when query_present. Filter-only
    # lookups null these so the YAML omits the block entirely.
    return LookupShape(
        entry_types=_record_kinds_from_list(d.get("entry_types", [])),
        k_requested=int(d.get("k_requested", 0)),
        has_neighbor_count=int(d.get("has_neighbor_count", 0)),
        has_neighbor_kinds=_record_kinds_from_list(
            d.get("has_neighbor_kinds", [])
        ),
        exact_match_count=int(d.get("exact_match_count", 0)),
        query_present=query_present,
        query_char_len=(
            int(d["query_char_len"])
            if query_present and "query_char_len" in d
            else None
        ),
        embed_dim=(
            int(d["embed_dim"])
            if query_present and "embed_dim" in d
            else None
        ),
        embed_norm=(
            float(d["embed_norm"])
            if query_present and "embed_norm" in d
            else None
        ),
        embed_all_zero=(
            bool(d["embed_all_zero"])
            if query_present and "embed_all_zero" in d
            else None
        ),
        embed_non_finite=(
            bool(d["embed_non_finite"])
            if query_present and "embed_non_finite" in d
            else None
        ),
        degenerate=bool(d.get("degenerate", False)),
        tied=bool(d.get("tied", False)),
        k_returned=int(d.get("k_returned", 0)),
        junk_dropped=int(d.get("junk_dropped", 0)),
        kind_counts=_kind_counts_from_dict(d.get("kind_counts", {})),
        score_samples=_score_samples_from_tuple(d.get("score_samples")),
    )


def _build_hop_marker(hop: dict, call_id: str | None) -> HopMarker:
    """One per-hop diag dict (from ``chatbot_turn.run``'s
    ``ctx.hops_diag``) → one ``HopMarker``. ``call_id`` is joined in by
    the chat-marker builder (the dispatcher doesn't know which LLM call
    fired this hop; the marker builder zips the hop list against
    ``llm_calls.calls`` in turn order). ``call_id=None`` falls back to
    the 4-digit zero-string sentinel so the field stays well-formed for
    the rare case where the join can't be made (test fixtures, a
    hop_diag entry without a matching llm_call)."""
    outcome_str = str(hop.get("hop_outcome", ""))
    outcome = _HOP_OUTCOME_FROM_STR.get(
        outcome_str, HopOutcome.prose_answer
    )
    lookups_raw = hop.get("per_lookup")
    lookups: tuple[LookupShape, ...] = ()
    if outcome is HopOutcome.tool_call and isinstance(lookups_raw, list):
        lookups = tuple(_build_lookup_shape(l) for l in lookups_raw)
    return HopMarker(
        call_id=call_id or "0000",
        hop_outcome=outcome,
        streamed_to_user=bool(hop.get("streamed_to_user", False)),
        lookups_remaining_in_budget=(
            int(hop["lookups_remaining_in_budget"])
            if hop.get("lookups_remaining_in_budget") is not None
            else None
        ),
        previous_attempts_count=int(hop.get("previous_attempts_count", 0)),
        store_open_latency_ms=(
            float(hop["store_open_latency_ms"])
            if hop.get("store_open_latency_ms") is not None
            else None
        ),
        dispatch_latency_ms=(
            float(hop["dispatch_latency_ms"])
            if hop.get("dispatch_latency_ms") is not None
            else None
        ),
        union_size_after=(
            int(hop["union_size_after"])
            if hop.get("union_size_after") is not None
            else None
        ),
        lookups=lookups,
    )


def _resolve_skipped_reason_from_hops(
    *,
    lookup_fired: bool,
    store_bound: bool,
    store_stats: StoreStats | None,
    hops_diag: list[dict] | None,
) -> RetrieveSkippedReason:
    """The hops-aware version of the v1 single-diag resolver. The
    resolution order is preserved (real retrieval signal beats stale
    telemetry; chat-side guards beat retrieve-internal short-circuits)
    but now reads from the per-hop list:

    - ``not lookup_fired`` → ``none`` (pure conversational turn)
    - ``not store_bound`` → ``no_bound_run`` (chat-side guard)
    - any hop's dispatch reported ``empty=True`` (empty store, retrieval
      path entered but store.count()==0) → ``empty_store``
    - any hop's per-lookup tracking shows a degenerate lookup that left
      the union empty for the turn → ``degenerate_query``
    - any hop's dispatch fired AT ALL (per_lookup is set) → ``none``
      (retrieval actually ran, regardless of how empty)
    - bound non-empty store but no hop dispatched → ``sink_never_called``
      (defensive — every grounded_decision hop should have either
      dispatched or finalized voluntarily)
    - store_stats reports zero with no hop dispatch → ``empty_store``"""
    if not lookup_fired:
        return RetrieveSkippedReason.none
    if not store_bound:
        return RetrieveSkippedReason.no_bound_run
    hops = hops_diag or []
    any_dispatched = False
    for hop in hops:
        # ``per_lookup`` is the dispatcher's signature — present iff
        # _dispatch_search ran (regardless of empty / degenerate /
        # tied outcomes). Its absence on a hop means the hop didn't
        # dispatch at all (decision turn, prose_answer hop, invalid
        # tool call, or no store_path).
        if hop.get("per_lookup") is not None:
            any_dispatched = True
            if hop.get("empty"):  # _dispatch_search short-circuit
                return RetrieveSkippedReason.empty_store
            if (
                hop.get("degenerate_dropped", 0) > 0
                and hop.get("union_size") == 0
            ):
                return RetrieveSkippedReason.degenerate_query
    if any_dispatched:
        return RetrieveSkippedReason.none
    if store_stats is not None and store_stats.total_records == 0:
        return RetrieveSkippedReason.empty_store
    return RetrieveSkippedReason.sink_never_called


def build_chat_marker(
    *,
    turn_index: int,
    session_id: str,
    lookup_fired: bool,
    hops_diag: list[dict],
    llm_calls: LlmCallsBlock,
    store_bound: bool = False,
    bound_run: str | None = None,
    store_stats: StoreStats | None = None,
    history_turn_count: int = 0,
    resources_emitted_count: int = 0,
) -> ChatMarker:
    """Per-turn content-free marker. The per-hop ReAct trace lives in
    ``hops_diag`` (one dict per LLM call, populated by
    ``chatbot_turn.run``). Each hop's dict carries:

      * hop-level fields stamped by chatbot_turn —
        ``hop_outcome`` / ``streamed_to_user`` /
        ``lookups_remaining_in_budget`` / ``previous_attempts_count`` /
        ``store_open_latency_ms`` / ``dispatch_latency_ms`` /
        ``union_size_after``
      * dispatch-level fields stamped by ``chatbot_dispatch._dispatch_search``
        via the per-hop sink — ``per_lookup: list[dict]`` plus the
        global aggregates (used only by the skip-reason resolver)

    Order is the firing order, which is also the order
    ``llm_calls.calls`` carries — so each ``HopMarker.call_id`` is
    joined in by index from the corresponding ``LlmCall``. The marker
    never reads any free string from the dicts: every leaf is a
    content-free number / bool / closed-enum value enforced by
    ``shareable._assert_content_free`` before write.
    """
    # Map each hop entry to its LlmCall by ORDER. The chatbot's turn
    # loop fires exactly one LLM call per hop iteration and the
    # dispatcher itself doesn't fire LLM calls (chatbot uses
    # ``llm_rerank=False``-equivalent dense-KNN — slice-2 has no
    # rerank call to fold in), so the two lists are 1:1 in order.
    # Falling back to ``None`` when llm_calls has fewer entries lets
    # tests synthesize hop markers without a parallel llm-calls
    # fixture; the HopMarker ctor stamps the sentinel "0000".
    hops: list[HopMarker] = []
    for i, hop in enumerate(hops_diag or []):
        call_id = (
            llm_calls.calls[i].call_id
            if i < len(llm_calls.calls)
            else None
        )
        hops.append(_build_hop_marker(hop, call_id))

    return ChatMarker(
        schema_version=SCHEMA_VERSION,
        ts=_now_iso_z(),
        turn_index=int(turn_index),
        session_id=_session_id_or_empty(session_id),
        lookup_fired=bool(lookup_fired),
        llm_calls=llm_calls,
        hops=tuple(hops),
        retrieve_skipped_reason=_resolve_skipped_reason_from_hops(
            lookup_fired=lookup_fired,
            store_bound=store_bound,
            store_stats=store_stats,
            hops_diag=hops_diag,
        ),
        bound_run=_perma_id_or_none(bound_run),
        store_stats=store_stats,
        history_turn_count=int(history_turn_count),
        resources_emitted_count=int(resources_emitted_count),
    )


def _size_hist(lengths: list[int]) -> tuple[SizeHistBucket, ...]:
    counts = {b: 0 for b, _, _ in _SIZE_EDGES}
    for n in lengths:
        for bucket, lo, hi in _SIZE_EDGES:
            if n >= lo and (hi is None or n < hi):
                counts[bucket] += 1
                break
    return tuple(SizeHistBucket(b, counts[b]) for b, _, _ in _SIZE_EDGES)


def build_run_marker(*, store, run_dir: Path | None) -> RunMarker:
    """Run/corpus structural marker — run stuff only, no chat. Reads
    only counts, the embed dim, char-length buckets via SQL aggregates,
    and stage presence/completion — never any record text, file id,
    topic, or record_id into Python. Written once per run perma-id."""
    by_kind: list[KindCount] = []
    embed_dim = None
    chunk_hist: tuple[SizeHistBucket, ...] = ()
    file_hist: tuple[SizeHistBucket, ...] = ()
    try:
        embed_dim = int(getattr(store, "dim", 0)) or None
        for k, n in (store.count_by_kind() or {}).items():
            rk = _kind(str(k))
            if rk is not None:
                by_kind.append(KindCount(rk, int(n)))
        conn = getattr(store, "conn", None)
        if conn is not None:
            # Bucket char-lengths inside SQLite: only integer counts
            # cross back into Python — no text is ever materialized.
            chunk_hist = _hist_from_sql(
                conn, "SELECT length(text) FROM records WHERE kind='chunk'"
            )
            file_hist = _hist_from_sql(
                conn,
                "SELECT length(text) FROM records "
                "GROUP BY file_id" if _has_col(conn, "file_id")
                else "SELECT length(text) FROM records",
            )
    except Exception:
        pass

    embed_calls = None
    batch_size = None
    if run_dir is not None:
        em = _load_marker(
            run_dir / "stages" / "06-embeddings" / "phase_1_marker.json"
        )
        if isinstance(em, dict):
            embed_calls = _as_int(em.get("calls"))
            batch_size = _as_int(em.get("batch_size"))

    embedding = EmbeddingStats(
        records_embedded=(
            int(store.count()) if hasattr(store, "count") else None
        ),
        embed_dim=embed_dim,
        embed_outcome=Outcome.success if by_kind else None,
        embed_call_count=embed_calls,
        batch_size=batch_size,
    )

    # The run's OWN pipeline-stage LLM calls (extraction / entities /
    # patterns / insights / actions / embeddings), itemized + content-
    # free, from this run's llm-calls.jsonl — NOT the chatbot's. The
    # per-call list lives under each StageStat.calls instead of as a
    # flat top-level tuple; the run-level block keeps only the four-int
    # aggregate (totals across all stages).
    llm_calls = None
    calls_by_stage: dict[StageToken, list[LlmCall]] = {}
    if run_dir is not None:
        calls_path = run_dir / "llm-calls.jsonl"
        if calls_path.is_file():
            full = build_llm_calls_block(
                calls_path, since_call_id=0, run_dir=run_dir,
            )
            llm_calls = LlmCallsBlock(
                call_count=full.call_count,
                total_prompt_tokens=full.total_prompt_tokens,
                total_completion_tokens=full.total_completion_tokens,
                wall_ms_total=full.wall_ms_total,
                calls=(),  # run scope: per-call detail lives under stages
            )
            for c in full.calls:
                calls_by_stage.setdefault(c.stage, []).append(c)

    stages = (
        _stage_stats(run_dir, calls_by_stage) if run_dir is not None else ()
    )

    return RunMarker(
        schema_version=SCHEMA_VERSION,
        created_at=_now_iso_z(),
        embedding=embedding,
        record_counts_by_kind=tuple(by_kind),
        chunk_size_hist=chunk_hist,
        file_size_hist=file_hist,
        stages=tuple(stages),
        llm_calls=llm_calls,
    )


def _as_int(v: object) -> int | None:
    return int(v) if isinstance(v, (int, float)) and not isinstance(
        v, bool
    ) else None


def _load_marker(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _marker_item_count(path: Path) -> int | None:
    """A stage's produced-item count, extracted CONTENT-FREE from its
    phase_1_marker.json: only integers ever cross back — list lengths,
    a summed ``counts`` map, or a top-level int field. Never a name,
    path, or any text."""
    m = _load_marker(path)
    if isinstance(m, list):
        return len(m)
    if isinstance(m, dict):
        counts = m.get("counts")
        if isinstance(counts, dict):
            vals = [_as_int(x) for x in counts.values()]
            nums = [x for x in vals if x is not None]
            if nums:
                return sum(nums)
        if isinstance(counts, (int, float)) and not isinstance(counts, bool):
            return int(counts)
        list_lens = [len(v) for v in m.values() if isinstance(v, list)]
        if list_lens:
            return max(list_lens)
        for key in ("count", "calls", "items", "n"):
            iv = _as_int(m.get(key))
            if iv is not None:
                return iv
    return None


def _has_col(conn, col: str) -> bool:
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(records)")}
        return col in cols
    except Exception:
        return False


def _hist_from_sql(conn, sql: str) -> tuple[SizeHistBucket, ...]:
    try:
        lengths = [int(r[0]) for r in conn.execute(sql) if r[0] is not None]
    except Exception:
        return ()
    return _size_hist(lengths)


# High-volume stages get sample+failures compression on their per-stage
# ``calls`` list; everything else carries every call. Closed-enum set,
# pinned here so the gate and the sampler agree.
_HIGH_VOLUME_STAGES: frozenset[StageToken] = frozenset({
    StageToken.extraction,
    StageToken.entities,
    StageToken.embeddings,
})

# Cap on the sampled-success count per high-volume stage. The sampler
# picks deterministic first / median / last by ``started_at`` ISO-Z
# (lexicographic order is correct for ISO-Z; the sampler also filters to
# successes that carry a timestamp). All failures are always present,
# never sampled — observability standard for rare high-signal events.
_SUCCESS_SAMPLE_CAP = 3


def _sample_calls(
    stage: StageToken, calls: list[LlmCall]
) -> tuple[tuple[LlmCall, ...], bool]:
    """Return ``(calls_to_emit, successful_calls_sampled)`` for one stage.

    Low-volume stages return every call verbatim
    (``successful_calls_sampled=False``).

    High-volume stages return ``all_failures + sampled_successes`` with
    ``successful_calls_sampled=True``: failures pass through wholesale;
    successes are reduced to the deterministic first / median / last
    by ``started_at``. The order of the returned tuple sorts by
    ``started_at`` so a reader sees the calls chronologically.

    Sub-types of success the runner labels distinctly
    (``success_empty`` / ``success_sampled`` /
    ``success_reasoning_off``) are PRESERVED, not sampled — each
    carries a signal a reader should not lose to the cap-3 picker.
    The classification comes from ``runner._classify_outcome`` (and
    ``_apply_chain_aware_outcomes``); the sampler just consumes it.

    A successful call with no ``started_at`` (telemetry drift) is
    treated as un-orderable and falls through with the failures —
    never silently dropped, since a non-timestamped success is still
    a real call worth surfacing.
    """
    if stage not in _HIGH_VOLUME_STAGES:
        return tuple(calls), False
    sampleable_successes: list[LlmCall] = []
    preserved: list[LlmCall] = []
    for c in calls:
        if _preserved_from_sampling(c):
            preserved.append(c)
        else:
            sampleable_successes.append(c)
    sampleable_successes.sort(key=lambda c: c.started_at or "")
    sampled = _pick_first_median_last(
        sampleable_successes, _SUCCESS_SAMPLE_CAP
    )
    combined = list(sampled) + preserved
    combined.sort(key=lambda c: c.started_at or "")
    return tuple(combined), True


def _preserved_from_sampling(c: LlmCall) -> bool:
    """A call that is NEVER sampled out — always present in the emit.

    Anything that isn't a plain ``Outcome.success`` qualifies:
    failures, aborted/skipped, the empty subtypes, AND the success
    subtypes (``success_empty``, ``success_sampled``,
    ``success_reasoning_off``) — each carries a signal a reader
    should not lose to sampling (the empty result, the
    work-reducing recovery, the reasoning-off recovery). Plus
    non-orderable successes (no ``started_at``) — un-orderable
    calls can't be sampled by first/median/last so they pass
    through too. The outcome-based predicate is the single
    chokepoint."""
    if c.outcome is not Outcome.success:
        return True
    if c.started_at is None:
        return True
    return False


def _pick_first_median_last(
    items: list[LlmCall], cap: int
) -> list[LlmCall]:
    """Deterministic sub-sample by position: first, median, last (in
    list-order). Returns the whole list when ``len(items) <= cap``.
    With ``cap == 3``: at 0/1/2 items the list is returned verbatim; at
    >=3 items the picks are ``[0]``, ``[len // 2]``, ``[-1]``,
    de-duplicated to preserve order without repetition. Generalizes
    cleanly to other caps but only cap=3 is used today."""
    n = len(items)
    if n <= cap:
        return list(items)
    if cap <= 0:
        return []
    if cap == 3:
        seen: set[int] = set()
        out: list[LlmCall] = []
        for idx in (0, n // 2, n - 1):
            if idx not in seen:
                seen.add(idx)
                out.append(items[idx])
        return out
    # Evenly-spaced for cap != 3 — kept defensive but not exercised.
    step = (n - 1) / (cap - 1)
    picks = sorted({round(step * i) for i in range(cap)})
    return [items[i] for i in picks if 0 <= i < n]


def _outcome_dist(calls: list[LlmCall]) -> tuple[OutcomeCount, ...]:
    counts: dict[Outcome, int] = {}
    for c in calls:
        counts[c.outcome] = counts.get(c.outcome, 0) + 1
    return tuple(
        OutcomeCount(o, n) for o, n in counts.items() if n > 0
    )


def _retry_class_dist(
    calls: list[LlmCall],
) -> tuple[RetryClassCount, ...]:
    """Per-stage retry-class counts. ``RetryClass.none`` rows are
    omitted — the absence of a row means count 0 — so a clean stage's
    distribution serializes empty (no noise)."""
    counts: dict[RetryClass, int] = {}
    for c in calls:
        if c.retry_class is RetryClass.none:
            continue
        counts[c.retry_class] = counts.get(c.retry_class, 0) + 1
    return tuple(
        RetryClassCount(rc, n) for rc, n in counts.items() if n > 0
    )


def _build_stage_stat(
    *,
    stage: StageToken,
    stage_index: int,
    completed: bool,
    item_count: int | None,
    calls: list[LlmCall],
) -> StageStat:
    """Roll one stage's call slice up to a content-free ``StageStat``.

    ``present`` is derived here: a stage is ``present`` iff it
    actually RAN — at least one LLM call fired OR the
    ``phase_1_marker.json`` completion sentinel exists. A
    scaffolding dir with no calls and no marker (e.g. an actions
    stage skipped because no inputs were produced) is NOT present.

    Every leaf is a closed-type value (number / bool / closed-enum /
    ISO-Z) so the runtime guard never has to widen."""
    success_count = sum(1 for c in calls if _is_success_outcome(c.outcome))
    failure_count = len(calls) - success_count
    cache_hit_count = sum(1 for c in calls if c.cached is True)
    retry_count = sum(1 for c in calls if c.is_retry)
    prompt_sum = sum(int(c.prompt_tokens or 0) for c in calls)
    completion_sum = sum(int(c.completion_tokens or 0) for c in calls)

    durations = [
        float(c.duration_ms) for c in calls if c.duration_ms is not None
    ]
    ttfts = [float(c.ttft_ms) for c in calls if c.ttft_ms is not None]

    started_at, _ = _iso_min_max(
        [c.started_at for c in calls if c.started_at is not None]
    )
    # Latest ``ended_at`` across all calls — a parallel fan-out batch
    # may finish out-of-order vs. start, so the started-at paired end
    # isn't the stage's true end.
    _, ended_at = _iso_min_max(
        [c.ended_at for c in calls if c.ended_at is not None]
    )
    wall_ms = _iso_delta_ms(started_at, ended_at)

    calls_out, successful_calls_sampled = _sample_calls(stage, calls)
    present = completed or len(calls) > 0

    return StageStat(
        stage=stage,
        stage_index=stage_index,
        present=present,
        completed=completed,
        success=completed and failure_count == 0,
        started_at=started_at,
        ended_at=ended_at,
        wall_ms=wall_ms,
        call_count=len(calls),
        success_count=success_count,
        failure_count=failure_count,
        cache_hit_count=cache_hit_count,
        retry_count=retry_count,
        prompt_tokens_sum=prompt_sum,
        completion_tokens_sum=completion_sum,
        item_count=item_count,
        duration_ms_p50=_percentile(durations, 0.50),
        duration_ms_p95=_percentile(durations, 0.95),
        duration_ms_mean=_mean(durations),
        ttft_ms_p50=_percentile(ttfts, 0.50),
        ttft_ms_p95=_percentile(ttfts, 0.95),
        outcome_dist=_outcome_dist(calls),
        retry_class_dist=_retry_class_dist(calls),
        calls=calls_out,
        successful_calls_sampled=successful_calls_sampled,
    )


def _stage_stats(
    run_dir: Path,
    calls_by_stage: dict[StageToken, list[LlmCall]],
) -> list[StageStat]:
    """Per-stage rollup. Identity + completion come from the run dir's
    ``stages/`` tree (presence + ``phase_1_marker.json`` sentinel);
    timing, work, perf, distributions, and the per-stage calls list
    come from the run's ``llm-calls.jsonl`` slice for the stage.

    Every leaf is a closed-type content-free value (number / bool /
    closed-enum / ISO-Z); the runtime guard rejects anything else.
    Timing is derived from per-call ISO-Z timestamps in the jsonl,
    sidestepping ``ProgressTracker``'s monotonic clock so a one-shot
    cancel-settle CLI (which runs without a live tracker) gets the
    same timing fidelity as a live finalize.

    Orphan-call carve-out: some stage tokens fire LLM calls without a
    matching ``stages/<NN-name>/`` dir leaf — ``vision`` dispatches
    inside the ingest stage, ``entities_dedupe`` inside the entities
    stage, anything mapped to ``other`` by definition. Without a
    fallback, those calls would silently vanish from the diagnostic
    now that ``RunMarker.llm_calls.calls`` is empty by construction.
    The carve-out: after the dir walk, every stage token with calls
    that no real-dir entry consumed gets its own ``StageStat`` with
    ``present=False`` and ``completed=False`` — full rollup +
    sampling semantics, just flagged as not-a-real-stage-dir."""
    stages_dir = run_dir / "stages"
    out: list[StageStat] = []
    try:
        children = sorted(p for p in stages_dir.iterdir() if p.is_dir())
    except OSError:
        children = []
    seen: set[StageToken] = set()
    for sd in children:
        marker = sd / "phase_1_marker.json"
        stage = stage_token(sd.name)
        seen.add(stage)
        out.append(_build_stage_stat(
            stage=stage,
            stage_index=0,  # reassigned post-sort
            completed=marker.is_file(),
            item_count=_marker_item_count(marker),
            calls=list(calls_by_stage.get(stage, ())),
        ))
    # Orphan-call carve-out: every stage token with calls but no dir
    # entry above gets a ``StageStat`` so the diagnostic never silently
    # drops a call (vision dispatches inside ingest, entities_dedupe
    # inside entities, anything mapping to ``other``). The combined
    # list (real-dir + orphan) is then sorted by canonical pipeline
    # order so e.g. ``entities_dedupe`` sits BETWEEN ``entities`` and
    # ``patterns``.
    for stage, calls in calls_by_stage.items():
        if stage in seen or not calls:
            continue
        out.append(_build_stage_stat(
            stage=stage,
            stage_index=0,  # reassigned post-sort
            completed=False,
            item_count=None,
            calls=list(calls),
        ))
    # Drop scaffolding-only entries — a stage dir with neither calls
    # nor a completion marker did NOT run (it's the runner's mkdir
    # bookkeeping, not work). ``present`` is set by
    # ``_build_stage_stat`` from the same condition; entries where
    # both call_count == 0 AND completed == False are filtered here
    # so a reader doesn't see ghost stages.
    out = [s for s in out if s.present]
    out.sort(key=lambda s: _stage_order_rank(s.stage))
    # Reassign ``stage_index`` so the printed ordinal matches the emit
    # position (canonical pipeline order, top-to-bottom).
    return [
        dataclasses.replace(s, stage_index=i)
        for i, s in enumerate(out)
    ]
