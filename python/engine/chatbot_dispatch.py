"""
Dispatch validated chatbot tool calls to read-only retrieval.

The decision turn yields a ``ToolCall`` — a known tool plus typed,
capped args (see ``chatbot_tools``). This module is the single place
that turns that typed call into actual reads against the run's vector
store, and normalises the result to the ``RetrievedRecord`` list the
grounded answer + citation surface already consume.

A ``search`` call carries a list of lookups. Each lookup contributes
its own filtered SELECT against the store (vector KNN when ``query`` is
given, plain records SELECT when not). Across the list the results
**union** by ``(kind, record_id)``; the merged set is ranked, trimmed to
``MAX_TOTAL_RESULTS``, and returned in one ordered list to the grounded
turn — the model never sees the per-lookup boundary, just the merged
context block. Ranking:

  - If at least one lookup carried a ``query``, union members ranked by
    the best (smallest) cosine distance any lookup observed for them;
    filter-only entries (no distance) slot in after, by the salience
    rule below.
  - Otherwise (every lookup is filter-only), the salience rule alone:
    stage rank ``action > insight > pattern > entity > fact > chunk``,
    within stage by positional index (actions / insights), mention
    count (patterns / entities), or recency (facts / chunks).

Every read here is a fixed, parameterized, SELECT-only query: there is
no path from model output to an interpolated query string. That is the
safety property the structured tool surface exists to hold — a string in
the user's own vault content cannot steer a destructive or malformed
query, because the model only ever produced a tool name and validated
parameters.
"""
from __future__ import annotations

import math
from typing import Callable

from engine.chatbot_tools import (
    Lookup,
    MAX_TOTAL_RESULTS,
    ToolCall,
    ToolCallError,
)
from engine.phases.embeddings import embed_texts
from engine.llm import Mode
from engine.rag_vector_store import COSINE_JUNK_DISTANCE, StoredRecord, VectorStore
from engine.retrieval import RetrievedRecord


# Stage rank for the filter-only / salience ordering. Lower index =
# higher priority. Matches the addendum's "action > insight > pattern >
# entity > fact > chunk".
_STAGE_RANK: dict[str, int] = {
    "action": 0,
    "insight": 1,
    "pattern": 2,
    "entity": 3,
    "fact": 4,
    "document": 5,
    "chunk": 6,
}


# Spread below which a top-k's distances are treated as a single tie:
# KNN ``ORDER BY distance`` is then a no-op and rows come back in
# vec0/insertion order — the "first record regardless of query"
# signature. Healthy corpora separate by ~0.5+; a degenerate
# (zero / constant) query vector ties every distance to a constant,
# so the real gap is between ~0 and ~0.5 and ``1e-6`` never
# false-positives. Same threshold ``retrieval.py`` uses.
_TIE_SPREAD_EPS = 1e-6


def _query_vector_is_degenerate(vec: list[float]) -> bool:
    """A query embedding that carries no usable direction: empty,
    non-finite (NaN/inf), zero-norm, or constant (every component
    equal). KNN against any of these reduces to a total distance tie
    → insertion-order results. Caught so the dispatcher can fail
    closed instead of returning confident garbage."""
    if not vec:
        return True
    if any(not math.isfinite(x) for x in vec):
        return True
    if not any(vec):  # all components exactly zero
        return True
    lo = min(vec)
    hi = max(vec)
    if hi - lo <= _TIE_SPREAD_EPS:  # constant vector — no direction
        return True
    norm = math.sqrt(math.fsum(x * x for x in vec))
    return norm <= _TIE_SPREAD_EPS


def _all_distances_tied(distances: list[float]) -> bool:
    """True when every distance in a ≥2-row pool is the same (within
    ``_TIE_SPREAD_EPS``): the rank carries no signal, so serving the
    pool would be insertion-order dressed as relevance. Belt-and-
    suspenders behind the query-vector check — also catches a
    degenerate *corpus* (all-equal stored vectors)."""
    if len(distances) < 2:
        return False
    return (max(distances) - min(distances)) <= _TIE_SPREAD_EPS


def dispatch(
    call: ToolCall,
    *,
    store: VectorStore,
    mode: Mode = Mode.TEE,
    diag_sink: Callable[[dict], None] | None = None,
) -> list[RetrievedRecord]:
    """Run a validated tool call; return its results in display order.

    ``search`` runs one filter-aware SELECT per lookup, unions the
    results by ``(kind, record_id)``, trims to ``MAX_TOTAL_RESULTS``,
    and returns them in the ordering rule described in the module
    docstring. A tool with no branch here raises ``ToolCallError`` — a
    registry/dispatch mismatch, surfaced loudly rather than silently
    returning nothing.
    """
    if call.tool == "search":
        return _dispatch_search(call, store=store, mode=mode, diag_sink=diag_sink)
    raise ToolCallError(f"no dispatch for tool {call.tool!r}")


# A union entry keyed by ``(kind, record_id)``. Carries the record
# itself, the best (smallest) cosine distance observed across the
# array's lookups (``None`` if every lookup that returned it was
# filter-only), and whether ANY lookup that returned it carried a
# query — used to slot the entry into the cosine-ranked tier vs the
# salience-ranked tier when the array mixes both modes.
class _UnionEntry:
    __slots__ = ("record", "best_distance", "any_query_lookup")

    def __init__(self, record: StoredRecord) -> None:
        self.record = record
        self.best_distance: float | None = None
        self.any_query_lookup: bool = False

    def update_distance(self, dist: float) -> None:
        if self.best_distance is None or dist < self.best_distance:
            self.best_distance = dist


def _dispatch_search(
    call: ToolCall,
    *,
    store: VectorStore,
    mode: Mode,
    diag_sink: Callable[[dict], None] | None,
) -> list[RetrievedRecord]:
    lookups: tuple[Lookup, ...] = call.args.get("lookups") or ()
    diag: dict = {
        "lookups": len(lookups),
        "any_query": any(lk.query for lk in lookups),
    }
    # Per-lookup detail accumulator. The dispatcher historically emitted
    # only global tallies (junk_dropped / degenerate_dropped /
    # tied_dropped / union_size), but the multi-hop chat-diagnostic
    # marker (#781) needs to attribute each lookup's contribution. Each
    # ``per_lookup_diag`` entry mirrors the model's request (entry_type
    # / count / has_neighbor count + kinds / exact_match count /
    # query_present) and the post-dispatch outcome (degenerate / tied /
    # k_returned / junk_dropped / kind_counts / score_samples / embed
    # signals). Captured inside the iteration loop and emitted as
    # ``diag["per_lookup"]`` beside the existing globals — back-compat
    # for any caller that still reads only the global tallies.
    per_lookup_diag: list[dict] = []

    if not lookups or store.count() == 0:
        diag["empty"] = True
        # Per-lookup detail still emitted (one stub entry per lookup
        # with the request knobs and ``k_returned=0``) so the marker
        # can render a hop that asked for things against an empty
        # store rather than dropping the lookup entries entirely.
        for lk in lookups:
            per_lookup_diag.append(_lookup_stub(lk, empty_store=True))
        diag["per_lookup"] = per_lookup_diag
        _fire(diag_sink, diag)
        return []

    # Embed every distinct ``query`` once. A lookup that reuses the same
    # query (legal: pair "show me X" + "show me X but only facts") gets
    # the same vector without a second embed call. The embed runs on the
    # kernel (``embed_texts``) — same nomic-embed-text model + per-mode
    # provider (Tinfoil/Ollama) the pipeline embeddings stage uses, so
    # retrieval shares the one inference path instead of a parallel one.
    distinct_queries: list[str] = []
    seen_queries: set[str] = set()
    for lk in lookups:
        if lk.query and lk.query not in seen_queries:
            distinct_queries.append(lk.query)
            seen_queries.add(lk.query)
    query_vectors: dict[str, list[float]] = {}
    if distinct_queries:
        vectors = embed_texts(distinct_queries, mode)
        for q, v in zip(distinct_queries, vectors):
            query_vectors[q] = v
    diag["distinct_queries"] = len(distinct_queries)

    # Run every lookup, accumulating into a union keyed by (kind,
    # record_id). The dispatcher does NOT trim per-lookup beyond the
    # validator-clamped ``count`` — the global cap fires once after the
    # union, so an array that asks for 5 + 5 + 5 with heavy overlap
    # still yields a tight final list.
    union: dict[tuple[str, str], _UnionEntry] = {}
    junk_dropped = 0
    degenerate_dropped = 0
    tied_dropped = 0
    for lk in lookups:
        # Start a per-lookup detail entry. Populate request-side knobs
        # straight from the validated ``Lookup``; dispatch-outcome
        # fields fill in below as the lookup runs.
        lk_diag = _lookup_request_diag(lk)
        # Resolve has_neighbor anchors once per lookup. Empty anchor list
        # → no neighbor restriction. Non-empty anchor list with no
        # matching edges → empty neighbor_ids (treated as "no neighbors
        # known" by both store methods, which short-circuit).
        neighbor_ids = (
            store.neighbors_of(list(lk.has_neighbor))
            if lk.has_neighbor
            else None
        )
        if lk.query is not None:
            vec = query_vectors[lk.query]
            # Stamp the per-lookup embed signal regardless of degenerate
            # outcome — the embed_norm / all_zero / non_finite leaves
            # tell a reader WHY a degenerate flag fired (zero-norm vs
            # non-finite vs constant), and are content-free numerics.
            lk_diag.update(_embed_signal(lk.query, vec))
            # Fail closed on a degenerate query embedding. A zero /
            # constant / non-finite vector ties every KNN distance, so
            # ``ORDER BY distance`` falls back to insertion order and
            # the same records come back regardless of the query —
            # indistinguishable to the user from confident, grounded
            # retrieval. Drop this lookup's contribution rather than
            # letting it pollute the union with insertion-order rows.
            if _query_vector_is_degenerate(vec):
                degenerate_dropped += 1
                lk_diag["degenerate"] = True
                per_lookup_diag.append(lk_diag)
                continue
            rows = store.query_filtered(
                vec,
                k=lk.count,
                kinds=lk.entry_type,
                neighbor_ids=neighbor_ids,
                exact_match=lk.exact_match,
                file_ids=lk.source,
            )
            # Same failure, detected at the result: a non-discriminating
            # top-k (all distances tied) is insertion-order, not
            # relevance. Catches a degenerate *corpus* (all-equal
            # stored vectors) the query-vector check above wouldn't.
            if _all_distances_tied([d for _r, d in rows]):
                tied_dropped += 1
                lk_diag["tied"] = True
                # Record what the store returned so a reader can see
                # the corpus-degenerate signature even though the rows
                # are being dropped.
                lk_diag["k_returned"] = len(rows)
                per_lookup_diag.append(lk_diag)
                continue
            # Per-lookup tallies: rows kept (post-junk-filter),
            # rows-by-kind, and the closest/middle/farthest distance
            # triple. ``junk_dropped`` here is the per-lookup count, not
            # the global cumulative; the global stays accurate via the
            # outer counter.
            kept_dists: list[float] = []
            kind_counts: dict[str, int] = {}
            lk_junk = 0
            for rec, dist in rows:
                if dist > COSINE_JUNK_DISTANCE:
                    junk_dropped += 1
                    lk_junk += 1
                    continue
                key = (rec.kind, rec.record_id)
                entry = union.get(key)
                if entry is None:
                    entry = _UnionEntry(rec)
                    union[key] = entry
                entry.update_distance(dist)
                entry.any_query_lookup = True
                kept_dists.append(dist)
                kind_counts[rec.kind] = kind_counts.get(rec.kind, 0) + 1
            lk_diag["k_returned"] = len(kept_dists)
            lk_diag["junk_dropped"] = lk_junk
            lk_diag["kind_counts"] = kind_counts
            lk_diag["score_samples"] = _score_samples(kept_dists)
        else:
            # Pull a generous slice — the salience sort below the union
            # picks the per-stage-best entries. Trimming to `lk.count`
            # here (before the dispatcher's salience pass) would lock
            # in records-table insertion order and leave higher-salience
            # rows past the first `count` matches unseen. Cap at
            # MAX_TOTAL_RESULTS — that's the per-turn ceiling, so a
            # single lookup can never contribute more than the global
            # cap and the records scan stays bounded.
            recs = store.filter_select(
                limit=MAX_TOTAL_RESULTS,
                kinds=lk.entry_type,
                neighbor_ids=neighbor_ids,
                exact_match=lk.exact_match,
                file_ids=lk.source,
            )
            # Salience-sort this lookup's matches and keep `lk.count` —
            # the per-lookup cap survives as a "this lookup contributes
            # at most N salience-best rows to the union" semantics.
            recs_sorted = sorted(
                recs,
                key=lambda r: _salience_sort_key(_UnionEntry(r)),
            )
            kept = recs_sorted[: lk.count]
            kind_counts = {}
            for rec in kept:
                key = (rec.kind, rec.record_id)
                if key not in union:
                    union[key] = _UnionEntry(rec)
                kind_counts[rec.kind] = kind_counts.get(rec.kind, 0) + 1
            lk_diag["k_returned"] = len(kept)
            lk_diag["junk_dropped"] = 0
            lk_diag["kind_counts"] = kind_counts
            # Filter-only lookups have no per-record distances, so no
            # score_samples — null in the marker, omitted in the yaml.
            lk_diag["score_samples"] = None
        per_lookup_diag.append(lk_diag)

    diag["junk_dropped"] = junk_dropped
    diag["degenerate_dropped"] = degenerate_dropped
    diag["tied_dropped"] = tied_dropped
    diag["union_size"] = len(union)
    diag["per_lookup"] = per_lookup_diag

    # Two tiers: query-bearing union members ranked by best cosine
    # distance ascending; filter-only members ranked by stage +
    # salience. Query tier comes first. Tier boundary matters only when
    # the array mixes a query lookup with a filter-only one — in a
    # pure-query array everything lands in the query tier; in a
    # pure-filter array everything lands in the salience tier.
    query_tier = [e for e in union.values() if e.any_query_lookup]
    filter_tier = [e for e in union.values() if not e.any_query_lookup]
    query_tier.sort(key=lambda e: e.best_distance or 0.0)
    filter_tier.sort(key=_salience_sort_key)

    ranked = query_tier + filter_tier
    if len(ranked) > MAX_TOTAL_RESULTS:
        ranked = ranked[:MAX_TOTAL_RESULTS]

    diag["returned"] = len(ranked)
    _fire(diag_sink, diag)

    out: list[RetrievedRecord] = []
    for entry in ranked:
        out.append(RetrievedRecord(
            record=entry.record,
            distance=(
                entry.best_distance if entry.best_distance is not None else 0.0
            ),
            rerank_score=None,
        ))
    return out


def _salience_sort_key(entry: _UnionEntry) -> tuple:
    """Salience-tier sort key. Smaller tuple sorts first.

    - Primary: stage rank (action > insight > pattern > entity > fact >
      chunk).
    - Secondary: per-stage salience (positional index for actions /
      insights, ``mention_count`` desc for patterns / entities,
      ``(file_date, char_offset)`` desc for facts / chunks).
    - Final tiebreaker: ``record_id`` ascending, so the order is
      deterministic across runs even when all upstream signals tie.
    """
    rec = entry.record
    stage = _STAGE_RANK.get(rec.kind, 99)
    return (stage, _within_stage_key(rec), rec.record_id)


def _within_stage_key(rec: StoredRecord) -> tuple:
    """Per-stage salience key. Lower sorts first; we negate desc-sorted
    counts and dates so the same comparator handles asc and desc."""
    extra = rec.extra or {}
    if rec.kind in ("action", "insight"):
        # Positional index — derived from the integer suffix of the
        # record_id (``str(i)`` for actions, ``{scope}:{i}`` for
        # insights). An unparseable suffix falls back to infinity so it
        # sorts last but doesn't crash the comparator on a synthetic /
        # malformed record.
        idx = _index_suffix(rec.record_id)
        return (idx,)
    if rec.kind in ("pattern", "entity"):
        count = int(extra.get("mention_count") or 0)
        return (-count,)
    if rec.kind in ("fact", "chunk", "document"):
        # Recency desc on file_date, then char_offset desc. file_date
        # is the per-record ISO string the embed stage stashes; missing
        # values sort last via the empty-string tail. char_offset is
        # the within-file position; negated so later text sorts first
        # (documents carry char_offset 0, so they order by date alone).
        date = str(extra.get("file_date") or "")
        return (_neg_iso_date(date), -int(rec.char_offset or 0))
    return ()


def _index_suffix(record_id: str) -> int:
    """Trailing integer in a positional record_id, or a large sentinel
    when the id doesn't follow the expected shape. Used as the
    actions / insights salience key."""
    tail = record_id.rsplit(":", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 10_000


def _neg_iso_date(date: str) -> str:
    """Return a string that sorts ASCENDING in reverse-chronological
    order, so empty / unknown dates land LAST while newer dates land
    first. ISO YYYY-MM-DD compares lexicographically, so the trick is
    a fixed-length complement: replace each digit ``d`` with ``9-d``.
    Non-digit chars (``-``) pass through. Empty input → a tail string
    that sorts after every valid date."""
    if not date:
        return "~"  # any tilde > '9' so empties land last
    out = []
    for ch in date:
        if ch.isdigit():
            out.append(str(9 - int(ch)))
        else:
            out.append(ch)
    return "".join(out)


def _fire(sink: Callable[[dict], None] | None, diag: dict) -> None:
    if sink is None:
        return
    try:
        sink(diag)
    except Exception:
        # Diagnostics are best-effort; never break dispatch.
        pass


# ── Per-lookup diag helpers (chat-diagnostics marker #781) ──────────────────


def _lookup_request_diag(lk: Lookup) -> dict:
    """The model's request side of a per-lookup diag entry — content-
    free reads from the validated ``Lookup`` dataclass. Entry types and
    neighbor anchor KINDS are closed-vocab strings; the count, neighbor
    count, and exact-match count are integers. Neither the query text
    nor the anchor ``record_id``s nor the exact-match substrings ever
    leave this function: only the structural shape (count, kinds,
    presence-bool) crosses into the diag."""
    return {
        "entry_types": list(lk.entry_type),
        "k_requested": int(lk.count),
        "has_neighbor_count": len(lk.has_neighbor),
        "has_neighbor_kinds": [k for (k, _r) in lk.has_neighbor],
        "exact_match_count": len(lk.exact_match),
        "source_count": len(lk.source),
        "query_present": lk.query is not None,
        # Dispatch-outcome fields default to "did not happen" so the
        # marker builder can rely on every entry having these keys
        # whether the lookup ran, degenerated, tied, or never reached
        # the store.
        "degenerate": False,
        "tied": False,
        "k_returned": 0,
        "junk_dropped": 0,
        "kind_counts": {},
        "score_samples": None,
    }


def _lookup_stub(lk: Lookup, *, empty_store: bool) -> dict:
    """A per-lookup entry for the no-dispatch-fired paths: empty store
    or empty lookups array. Records the request knobs so a reader can
    still see what the model ASKED for, even though no rows came back.
    ``empty_store=True`` lookups stamp ``k_returned=0`` straight; that
    distinguishes "store was empty" (turn-level diag has ``empty=True``)
    from "lookup actually ran and got zero hits"."""
    d = _lookup_request_diag(lk)
    if empty_store and lk.query is not None:
        # Still surface the query length so the reader can confirm the
        # model emitted a non-trivial query even though the store was
        # empty. embed_signal isn't computed (we short-circuit before
        # embed) — those fields stay None.
        d["query_char_len"] = len(lk.query)
    return d


def _embed_signal(query: str, vec: list[float]) -> dict:
    """Content-free embed-vector signal for one lookup. ``query_char_len``
    is a length, never the string. The four embed leaves
    (dim / norm / all_zero / non_finite) mirror what ``retrieval.py``'s
    diag has stamped per-turn; here they're per-lookup so a multi-
    lookup turn can attribute a degenerate flag to a specific lookup."""
    return {
        "query_char_len": len(query),
        "embed_dim": len(vec),
        "embed_norm": (
            math.sqrt(math.fsum(x * x for x in vec)) if vec else 0.0
        ),
        "embed_all_zero": bool(vec) and not any(vec),
        "embed_non_finite": any(not math.isfinite(x) for x in vec),
    }


def _score_samples(dists: list[float]) -> tuple[float, float, float] | None:
    """Closest / middle / farthest distance from this lookup's kept
    rows. The marker builder turns this triple into a ``ScoreSamples``
    dataclass; ``None`` here means the marker omits the block (no
    distances to summarize: zero rows kept, or filter-only lookup).
    Middle is the median by index (not value) so it stays deterministic
    when several distances coincide — k=1 collapses all three to the
    same value, k=2 to (closest, closest, farthest)."""
    if not dists:
        return None
    sorted_dists = sorted(dists)
    closest = sorted_dists[0]
    farthest = sorted_dists[-1]
    middle = sorted_dists[len(sorted_dists) // 2]
    return (closest, middle, farthest)


__all__ = ["dispatch"]
