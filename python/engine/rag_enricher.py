"""
Graph-enriched embed text builder for the Embeddings stage.

Each record kind embedded into the vector store carries a prefix of its
upstream + downstream graph context (per spec § RAG enhancements) before
its bare text. Dense retrieval can then match on the surrounding graph
rather than only the record's own fields — a chunk-level query about
"sleep + running" finds chunks whose enclosing facts mention those
topics even when the bare text doesn't repeat the lexical surface.

This module is a one-pass graph view of the in-memory upstream artifacts
the embeddings stage already receives (facts / entities / patterns /
insights / actions) plus the RAG chunker's output, and per-kind text
helpers that consume that view to render a deterministic enriched
string. Selection rules from spec live here (top-N by confidence,
top-K most recent, histograms across fact-set intersections); ordering
within each list is stable across runs so identical upstream produces
identical embed input (cache-stable).

Parallel to `rag_chunker.py` (token-window slicing) and
`rag_vector_store.py` (sqlite-vec persistence). `embeddings.py` calls
`build_graph_view` once per run, then invokes a per-kind builder when
materializing each StoredRecord.text.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from engine.rag_chunker import RagChunk, _section_spans, chunk_documents
from engine.rag_vector_store import (
    EDGE_KIND_CONTAINMENT,
    EDGE_KIND_DERIVATION,
    EDGE_KIND_MENTION,
    EDGE_KIND_RELATION,
    EDGE_KIND_SIBLING,
)

if TYPE_CHECKING:
    from engine.actions import Action
    from engine.content_extractor import ExtractedItem
    from engine.entities import EntitiesOutput, EntityRecord, RelationEdge
    from engine.ingestor import Document
    from engine.insights import Insight, InsightOutput
    from engine.patterns import Pattern


# Per-spec selection caps. Locked here (not buried at call sites) so a
# spec change is one edit and the golden-string tests immediately
# surface drift if these move.
TOP_FACTS_BY_CONFIDENCE_ENTITY = 3   # spec: top-3 most-confident facts
TOP_FACTS_BY_RECENCY_ENTITY = 3      # spec: top-3 most-recent facts
TOP_PATTERNS_ENTITY = 2              # spec: top-2 patterns mentioning entity
TOP_INSIGHTS_ENTITY = 2              # spec: top-2 insights mentioning entity
TOP_ACTIONS_ENTITY = 2               # spec: top-2 actions mentioning entity
TOP_FACTS_PATTERN = 5                # spec: top-5 fact titles for patterns

# Prose truncation for embedded prefix text. Bare records carry their
# own full text (no truncation) — the limits below only apply to fields
# included as upstream / downstream context on a DIFFERENT record's
# prefix. Tuned so a heavily-enriched record stays well under the
# nomic-embed 8192-token cap; the cascade halves if a record still
# exceeds the wire cap.
MAX_PROSE_CHARS = 500
MAX_QUOTE_CHARS = 400


# Human-facing type label baked into each record's embedded text (as a
# leading `Type: <label>` line) so the embedding itself carries the type
# signal, not just the prompt-layer annotation. `chunk` reads as
# "raw input" — it's the unprocessed source text, not a processed
# artifact. Single source of truth: the grounded-context prefix
# (`chatbot._context_line`) routes the same record's kind through
# `kind_label` so the embed-text label and the model-visible label
# can't drift.
KIND_EMBED_LABEL = {
    "document": "source file",
    "chunk": "raw input",
    "fact": "fact",
    "entity": "entity",
    "pattern": "pattern",
    "insight": "insight",
    "action": "action",
}


def kind_label(kind: str) -> str:
    """Map a record kind to its human-facing type label. Unknown kinds
    fall through unchanged so a new kind degrades to its raw name rather
    than vanishing."""
    return KIND_EMBED_LABEL.get(kind, kind)


def _type_header(kind: str) -> str:
    """Leading `Type: <label>` line prepended uniformly by every
    per-kind builder."""
    return f"Type: {kind_label(kind)}"


def _truncate(s: str, limit: int) -> str:
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    return s[: max(1, limit - 1)].rstrip() + "…"


def _fact_title(fact: "ExtractedItem") -> str:
    """One-line title for a fact. Used in upstream / downstream
    contexts on other records; bare facts use their full summary
    elsewhere."""
    return _truncate(fact.summary, MAX_PROSE_CHARS)


def _fact_type(fact: "ExtractedItem") -> str:
    return getattr(fact, "item_type", "") or "fact"


def _fact_provenance_key(
    fact: "ExtractedItem",
) -> tuple[str, int, int, str, str, str] | None:
    """Dedup key for the cross-category fact collapse: content-hash by
    enumeration over the persisted fields — first evidence span's
    source-file pointer (path + offset + length) + item type +
    occurred_at + summary text. Same key ⇒ same underlying fact.

    Within-span tie-breaker is `summary`: two distinct facts extracted
    from the same quoted sentence necessarily carry different atomic
    summaries (one atomic statement per item is an extraction rule), so
    they stay distinct under this key. Category-copies of one fact have
    identical summaries (the runner appends the SAME ExtractedItem to
    every topic bucket the fact carries, so the field is byte-equal
    across copies) and collapse to one canonical.

    `summary` is the load-bearing discriminator HERE — not the entity-
    side display-dedup key per [[project_entity_fact_refs_canonical_vs_display]]
    (that surface explicitly rejects summary-based collapse because
    same-text facts from different extractions are a deliberate anomaly
    signal). The embed-layer dedup is solving a different problem: the
    runner's multi-topic fanout produces facts that are byte-identical
    on every field, and the key needs to identify "same persisted
    fact" stably ACROSS the resume round-trip (`_load_facts_by_topic`
    reconstructs ExtractedItems as fresh Python objects, so process-
    local handles like `id(fact)` don't survive — fresh-run dedup
    works but resume re-emits the duplicates).

    Returns None when the fact's first evidence span lacks a usable
    discriminator — no evidence at all, or evidence with missing
    file_offset / file_length (extraction's span-attribution failure
    paths). The caller treats each such fact as its own singleton
    canonical, so two facts whose attribution failed don't silently
    merge just because their trailing fields (item_type, date, summary)
    happen to match. When location is unknown the safe default is to
    NOT collapse.
    """
    evidence = getattr(fact, "evidence", None) or []
    if not evidence:
        return None
    ev = evidence[0]
    fo = getattr(ev, "file_offset", None)
    fl = getattr(ev, "file_length", None)
    if fo is None or fl is None:
        return None
    return (
        getattr(ev, "file_path", None) or "",
        int(fo),
        int(fl),
        getattr(fact, "item_type", "") or "",
        getattr(fact, "occurred_at", None) or "",
        getattr(fact, "summary", "") or "",
    )


# ── Graph view ────────────────────────────────────────────────────────────────


@dataclass
class GraphView:
    """One-pass graph view of upstream artifacts. Built once per run by
    `build_graph_view`; consumed by every per-kind text helper below.

    The dataclass holds source artifacts AND derived indices side-by-
    side. The indices are deliberately not lazy: building them once up
    front keeps the per-record builders to dict lookups, which is the
    hot path when the plan has thousands of records.
    """

    # Sources (held by reference; not mutated).
    docs: list["Document"]
    facts_by_topic: dict[str, list["ExtractedItem"]]
    entities: list["EntityRecord"]
    relations: list["RelationEdge"]
    patterns_by_topic: dict[str, list["Pattern"]]
    insights_cross: list["Insight"]
    insights_critical: list["Insight"]
    actions: list["Action"]

    # Chunks (RAG-chunked).
    chunks: list[RagChunk] = field(default_factory=list)

    # Splitter-chunk_id → list of split summaries (one per inner entry
    # in a batched doc; usually just one for a non-batched split).
    summaries_by_splitter_chunk: dict[str, list[str]] = field(default_factory=dict)

    # Source path → list of (RAG chunk index) covering that file. Used
    # to fan a fact's owning source file out to candidate RAG chunks
    # via offset overlap.
    rag_chunks_by_file_id: dict[str, list[int]] = field(default_factory=dict)

    # Splitter chunk id (== fact evidence source_ref) → ordered list of
    # (topic, fact_idx) facts that came from this splitter chunk, in
    # extraction order. Used for "next / previous" fact sibling lookup.
    facts_by_splitter_chunk: dict[str, list[tuple[str, int]]] = field(
        default_factory=dict
    )

    # Entity canonical_id → set of (topic, fact_idx) the entity is mentioned in.
    entity_facts: dict[str, set[tuple[str, int]]] = field(default_factory=dict)

    # (topic, fact_idx) → list of (canonical_id) of every entity that mentions it.
    fact_entities: dict[tuple[str, int], list[str]] = field(default_factory=dict)

    # Pattern (topic, pattern_idx) → set of facts it cites as source.
    pattern_facts: dict[tuple[str, int], set[tuple[str, int]]] = field(
        default_factory=dict
    )

    # Insight (scope, idx) → set of facts (via source_patterns expansion).
    insight_facts: dict[tuple[str, int], set[tuple[str, int]]] = field(
        default_factory=dict
    )

    # Action idx → set of facts (via source_insights → source_patterns).
    action_facts: dict[int, set[tuple[str, int]]] = field(default_factory=dict)

    # Insight (scope, idx) → list of source-pattern keys (topic, pat_idx).
    insight_patterns: dict[tuple[str, int], list[tuple[str, int]]] = field(
        default_factory=dict
    )

    # Action idx → list of source-insight keys (scope, idx).
    action_insights: dict[int, list[tuple[str, int]]] = field(default_factory=dict)

    # Action idx → list of pattern keys reached via source_insights.
    action_patterns: dict[int, list[tuple[str, int]]] = field(default_factory=dict)

    # Source path → Document. Used for section-path lookup from a
    # fact's evidence offset on docs short enough that the RAG chunker
    # took the whole-doc fast-path (section_path empty on the chunk
    # itself).
    docs_by_file_id: dict[str, "Document"] = field(default_factory=dict)

    # Quick lookups for entity rendering.
    entities_by_id: dict[str, "EntityRecord"] = field(default_factory=dict)
    entities_by_name_normalized: dict[str, "EntityRecord"] = field(
        default_factory=dict
    )

    # Relations indexed by either endpoint.
    relations_by_entity: dict[str, list["RelationEdge"]] = field(
        default_factory=dict
    )

    # Insight lookup helpers.
    insights_all: list[tuple[str, int, "Insight"]] = field(default_factory=list)

    # Cross-category fact dedup. The runner appends the same
    # ExtractedItem to one bucket per topic it carries, so a fact in N
    # topics surfaces N (topic, idx) keys for the SAME underlying fact.
    # The maps below collapse that fanout for the embedding layer (one
    # record per unique fact) while leaving the per-topic facts_by_topic
    # + (topic, idx) keying that upstream stages (patterns, entities)
    # already consume unchanged.
    #
    # `fact_canonical` maps every (topic, idx) alias to the canonical
    # (topic, idx) chosen for its underlying fact; the canonical itself
    # maps to itself. Canonical pick is deterministic: the smallest
    # (topic, idx) under tuple ordering — topic alphabetic, then idx.
    #
    # `fact_aliases` maps a canonical (topic, idx) to the full sorted
    # list of its aliases (>=1, includes the canonical). Used by the
    # alias-aware pattern lookup and by the invariant-1 verification.
    fact_canonical: dict[tuple[str, int], tuple[str, int]] = field(
        default_factory=dict
    )
    fact_aliases: dict[tuple[str, int], list[tuple[str, int]]] = field(
        default_factory=dict
    )

    # Per-canonical record of entity-mention-set drift across aliases:
    # director invariant 1 says the set of entities mentioning each
    # underlying fact must be identical across its category-copies
    # (entity mention is a property of the fact, not the category). When
    # it isn't, the embeddings stage surfaces a per-stage warning record
    # carrying the canonical id + the diff so a regression in the
    # entities stage that drops mentions on some copies but not others
    # is visible at the phase marker. Empty on a clean run.
    entity_alias_drift: list[dict] = field(default_factory=list)

    def fact_at(self, topic: str, idx: int) -> "ExtractedItem | None":
        items = self.facts_by_topic.get(topic) or []
        if 0 <= idx < len(items):
            return items[idx]
        return None

    def canonical_fact_key(
        self, topic: str, idx: int,
    ) -> tuple[str, int]:
        """Resolve a (topic, idx) to the canonical (topic, idx) under
        the cross-category dedup. Aliases collapse to one canonical;
        an unmapped key (no dedup pass ran, or no facts at all) returns
        itself unchanged."""
        return self.fact_canonical.get((topic, idx), (topic, idx))

    def canonical_fact_id(self, topic: str, idx: int) -> str:
        """Canonical record-id string `{topic}:{idx}` — the embedding-
        layer-stable handle on the underlying fact regardless of which
        category-copy the caller starts from."""
        ctopic, cidx = self.canonical_fact_key(topic, idx)
        return f"{ctopic}:{cidx}"

    def aliases_of(
        self, topic: str, idx: int,
    ) -> list[tuple[str, int]]:
        """All (topic, idx) keys that map to the same underlying fact
        as (topic, idx). Includes the input key. A fact in N topics has
        N aliases; a singleton has one. Stable sort (topic, idx)."""
        canonical = self.canonical_fact_key(topic, idx)
        aliases = self.fact_aliases.get(canonical)
        if aliases is not None:
            return aliases
        return [(topic, idx)]

    def canonical_siblings(
        self, source_ref: str,
    ) -> list[tuple[str, int]]:
        """Per-splitter-chunk fact list collapsed to canonical
        (topic, idx) keys with first-occurrence semantics — the order
        each underlying fact first appears in extraction order. After
        the collapse, no two adjacent entries are aliases of the same
        underlying fact, so sibling prev/next derived from this list
        cannot resolve to a self-edge after the embedding-layer dedup.
        """
        seen: set[tuple[str, int]] = set()
        out: list[tuple[str, int]] = []
        for key in self.facts_by_splitter_chunk.get(source_ref, []):
            canonical = self.canonical_fact_key(*key)
            if canonical in seen:
                continue
            seen.add(canonical)
            out.append(canonical)
        return out

    def pattern_at(self, topic: str, idx: int) -> "Pattern | None":
        items = self.patterns_by_topic.get(topic) or []
        if 0 <= idx < len(items):
            return items[idx]
        return None

    def insight_at(self, scope: str, idx: int) -> "Insight | None":
        items = (
            self.insights_cross if scope == "cross_domain" else self.insights_critical
        )
        if 0 <= idx < len(items):
            return items[idx]
        return None

    def action_at(self, idx: int) -> "Action | None":
        if 0 <= idx < len(self.actions):
            return self.actions[idx]
        return None


def build_graph_view(
    *,
    docs: "list[Document] | None",
    facts_by_topic: "dict[str, list[ExtractedItem]] | None",
    entities_output: "EntitiesOutput | None",
    patterns_by_topic: "dict[str, list[Pattern]] | None",
    insight_output: "InsightOutput | None",
    action_list: "list[Action] | None",
    extract_calls: "list[dict] | None" = None,
) -> GraphView:
    """Build a one-pass GraphView from in-memory upstream artifacts.

    `extract_calls` are the per-LLM-call records the runner accumulates
    during extraction (`runner.py:_extract_calls`); each carries the
    splitter chunk_id + the LLM-produced split_summaries. Optional
    (tests + minimal callers pass None).
    """
    view = GraphView(
        docs=list(docs or []),
        facts_by_topic=dict(facts_by_topic or {}),
        entities=list(entities_output.entities) if entities_output else [],
        relations=list(entities_output.relations) if entities_output else [],
        patterns_by_topic=dict(patterns_by_topic or {}),
        insights_cross=list(insight_output.cross_domain) if insight_output else [],
        insights_critical=list(insight_output.critical) if insight_output else [],
        actions=list(action_list or []),
    )

    # RAG chunks once, indexed by ``file_id`` (the canonical basename /
    # doc-identifier) for fact→chunk overlap. Evidence carries
    # ``ev.file_path = doc.file_id`` (content_extractor sets it that way
    # at extraction time); chunks and docs each carry both ``file_id``
    # and the full-path ``source_path``. The lookup key MUST be
    # ``file_id`` to match what evidence carries — keying on
    # ``source_path`` silently breaks every chunk/doc lookup downstream
    # (fact_facing_chunk_index, section_path_for_fact, build_edges'
    # chunk↔fact containment, build_chunk_text's contained-facts list).
    view.chunks = chunk_documents(view.docs)
    for i, ch in enumerate(view.chunks):
        view.rag_chunks_by_file_id.setdefault(ch.file_id, []).append(i)
    for d in view.docs:
        view.docs_by_file_id[d.file_id] = d

    # Split summaries by splitter chunk id (from runner's _extract_calls).
    for call in extract_calls or []:
        cid = str(call.get("chunk_id") or "").strip()
        if not cid:
            continue
        summaries = []
        for s in call.get("split_summaries") or []:
            text = (s.get("summary") if isinstance(s, dict) else "") or ""
            text = text.strip()
            if text:
                summaries.append(text)
        if summaries:
            view.summaries_by_splitter_chunk[cid] = summaries

    # Facts indexed by their splitter chunk source_ref (for siblings).
    for topic in sorted(view.facts_by_topic.keys()):
        for idx, fact in enumerate(view.facts_by_topic[topic]):
            evidence = getattr(fact, "evidence", None) or []
            for ev in evidence:
                src = getattr(ev, "source_ref", None)
                if src:
                    view.facts_by_splitter_chunk.setdefault(str(src), []).append(
                        (topic, idx)
                    )
                    break  # first evidence's source_ref defines the owning split

    # Cross-category fact dedup. Group (topic, idx) by provenance tuple;
    # the smallest key in each group becomes the canonical, every member
    # of the group (including the canonical) maps to it. Facts with no
    # evidence span (degenerate) are passed through as singleton
    # canonicals; the rest follow the provenance-key collapse.
    _prov_groups: dict[tuple, list[tuple[str, int]]] = {}
    for topic in sorted(view.facts_by_topic.keys()):
        for idx, fact in enumerate(view.facts_by_topic[topic]):
            prov = _fact_provenance_key(fact)
            if prov is None:
                view.fact_canonical[(topic, idx)] = (topic, idx)
                view.fact_aliases[(topic, idx)] = [(topic, idx)]
                continue
            _prov_groups.setdefault(prov, []).append((topic, idx))
    for _prov, aliases in _prov_groups.items():
        aliases.sort()
        canonical = aliases[0]
        view.fact_aliases[canonical] = list(aliases)
        for alias in aliases:
            view.fact_canonical[alias] = canonical

    # Entities indexed by id + normalized name (for fact-time entity-ref
    # resolution). evidence_fact_refs IS the authoritative mention map
    # (the entities stage already resolved + grouped names by then).
    for ent in view.entities:
        view.entities_by_id[ent.canonical_id] = ent
        view.entity_facts[ent.canonical_id] = set(ent.evidence_fact_refs)
        for (topic, idx) in ent.evidence_fact_refs:
            view.fact_entities.setdefault((topic, idx), []).append(ent.canonical_id)
        # Normalized-name + alias lookup for the fact-side EntityRef
        # resolution path. Keys are lower/stripped to match the entities
        # stage's normalization without re-importing it.
        for name in [ent.canonical_name] + list(ent.aliases or []):
            key = (name or "").strip().lower()
            if key and key not in view.entities_by_name_normalized:
                view.entities_by_name_normalized[key] = ent

    for rel in view.relations:
        view.relations_by_entity.setdefault(rel.from_id, []).append(rel)
        view.relations_by_entity.setdefault(rel.to_id, []).append(rel)

    # Director invariant 1 — entity-mention drift across category-copies.
    # For each canonical fact with multiple aliases, the set of entities
    # referencing each alias must be identical (a property of the
    # underlying fact, not of the category it was binned under). Compute
    # per-alias mention sets and surface a drift record when any pair
    # disagrees; an upstream entities-stage regression that drops
    # mentions on some category-copies but not others becomes observable
    # at the embeddings phase marker rather than silently producing
    # inconsistent `[neighbor]` enrichment after the alias collapse.
    for canonical, aliases in view.fact_aliases.items():
        if len(aliases) <= 1:
            continue
        per_alias = [frozenset(view.fact_entities.get(a, [])) for a in aliases]
        if all(s == per_alias[0] for s in per_alias[1:]):
            continue
        view.entity_alias_drift.append({
            "canonical_id": f"{canonical[0]}:{canonical[1]}",
            "aliases": [f"{t}:{i}" for (t, i) in aliases],
            "per_alias_entities": [sorted(s) for s in per_alias],
        })

    # Pattern → facts (source_facts: list[(fact_idx, confidence)]).
    for topic in sorted(view.patterns_by_topic.keys()):
        for pidx, pat in enumerate(view.patterns_by_topic[topic]):
            fact_set = set()
            for (fact_idx, _conf) in getattr(pat, "source_facts", []) or []:
                fact_set.add((topic, int(fact_idx)))
            view.pattern_facts[(topic, pidx)] = fact_set

    # Insights: list cross_domain + critical with their scope.
    for scope, items in (
        ("cross_domain", view.insights_cross),
        ("critical", view.insights_critical),
    ):
        for iidx, ins in enumerate(items):
            view.insights_all.append((scope, iidx, ins))
            pkeys: list[tuple[str, int]] = []
            fset: set[tuple[str, int]] = set()
            for (topic, pat_idx, _conf) in getattr(ins, "source_patterns", []) or []:
                key = (str(topic), int(pat_idx))
                pkeys.append(key)
                fset.update(view.pattern_facts.get(key, set()))
            view.insight_patterns[(scope, iidx)] = pkeys
            view.insight_facts[(scope, iidx)] = fset

    # Actions: traverse through source_insights → source_patterns.
    for aidx, act in enumerate(view.actions):
        ikeys: list[tuple[str, int]] = []
        pkeys: list[tuple[str, int]] = []
        fset: set[tuple[str, int]] = set()
        for (scope, ins_idx, _conf) in getattr(act, "source_insights", []) or []:
            key = (str(scope), int(ins_idx))
            ikeys.append(key)
            pkeys.extend(view.insight_patterns.get(key, []))
            fset.update(view.insight_facts.get(key, set()))
        view.action_insights[aidx] = ikeys
        view.action_patterns[aidx] = pkeys
        view.action_facts[aidx] = fset

    return view


# ── Helpers shared across per-kind builders ──────────────────────────────────


def _date_span(
    fact_keys: "set[tuple[str, int]] | list[tuple[str, int]]",
    view: GraphView,
) -> str:
    """Min..max occurred_at across the given facts (ISO YYYY-MM-DD).
    Empty when no fact in the set has an occurred_at."""
    dates: list[str] = []
    for (topic, idx) in fact_keys:
        f = view.fact_at(topic, idx)
        if f is None:
            continue
        d = (getattr(f, "occurred_at", None) or "").strip()
        if d:
            dates.append(d)
    if not dates:
        return ""
    lo = min(dates)
    hi = max(dates)
    return lo if lo == hi else f"{lo} … {hi}"


def _topic_histogram(
    fact_keys: "set[tuple[str, int]] | list[tuple[str, int]]",
    view: GraphView,
) -> list[tuple[str, int]]:
    """(topic, count) descending by count, then alpha by topic for ties."""
    c: Counter[str] = Counter()
    for (topic, idx) in fact_keys:
        f = view.fact_at(topic, idx)
        if f is None:
            continue
        for t in (getattr(f, "topics", None) or []):
            c[t] += 1
    return sorted(c.items(), key=lambda x: (-x[1], x[0]))


def _tag_histogram(
    fact_keys: "set[tuple[str, int]] | list[tuple[str, int]]",
    view: GraphView,
) -> list[tuple[str, int]]:
    c: Counter[str] = Counter()
    for (topic, idx) in fact_keys:
        f = view.fact_at(topic, idx)
        if f is None:
            continue
        for t in (getattr(f, "tags", None) or []):
            c[t] += 1
    return sorted(c.items(), key=lambda x: (-x[1], x[0]))


def _entity_histogram(
    fact_keys: "set[tuple[str, int]] | list[tuple[str, int]]",
    view: GraphView,
) -> list[tuple[str, int]]:
    """(canonical_id, count) over facts in the set, descending by count
    then alpha by canonical_id."""
    c: Counter[str] = Counter()
    for key in fact_keys:
        for cid in view.fact_entities.get(key, []):
            c[cid] += 1
    return sorted(c.items(), key=lambda x: (-x[1], x[0]))


def _format_entity_with_type_role(ent: "EntityRecord") -> str:
    parts: list[str] = [ent.canonical_name.strip() or ent.canonical_id]
    type_role = " · ".join(p for p in (ent.entity_type, ent.role) if p)
    if type_role:
        parts.append(f"({type_role})")
    return " ".join(parts)


def _section_path_for_fact(
    fact: "ExtractedItem", view: GraphView,
) -> tuple[str, ...]:
    """Section path of the doc location a fact's first evidence quote
    points at. Falls back to the RAG chunk's section_path when the doc
    text isn't available; falls back to () when neither resolves.
    Independent of chunker fast-path behavior (a small whole-doc chunk
    has section_path=() but the doc body may still carry headers)."""
    for ev in getattr(fact, "evidence", None) or []:
        fp = getattr(ev, "file_path", None)
        fo = getattr(ev, "file_offset", None)
        if not fp or fo is None:
            continue
        doc = view.docs_by_file_id.get(fp)
        if doc is not None:
            # The chunker's section_spans operate on doc.content; fact
            # evidence file_offset is absolute within the source file,
            # so subtract the doc's origin_char to translate into the
            # doc-local content offset.
            local = int(fo) - int(doc.origin_char or 0)
            if 0 <= local <= len(doc.content):
                for (start, end, path) in _section_spans(doc.content):
                    if start <= local < end:
                        return path
    # Fallback to the enclosing chunk's section_path (still useful for
    # large docs where the chunker computed it).
    ci = _fact_facing_chunk_index(fact, view)
    if ci is not None:
        return view.chunks[ci].section_path
    return ()


def _fact_facing_chunk_index(
    fact: "ExtractedItem", view: GraphView
) -> int | None:
    """Find the RAG chunk that physically contains this fact (by source
    file + character-offset overlap). Returns the chunk's index in
    `view.chunks`, or None if the fact has no usable evidence offsets.
    """
    if not view.chunks:
        return None
    for ev in getattr(fact, "evidence", None) or []:
        fp = getattr(ev, "file_path", None)
        fo = getattr(ev, "file_offset", None)
        if not fp or fo is None:
            continue
        for ci in view.rag_chunks_by_file_id.get(fp, []):
            ch = view.chunks[ci]
            if ch.char_offset <= fo < ch.char_offset + len(ch.text):
                return ci
    return None


def _rank_by_fact_overlap(
    target_facts: "set[tuple[str, int]]",
    candidate_fact_sets: "dict",
    limit: int,
) -> list:
    """Generic top-N picker by |target ∩ candidate_facts| descending.
    Tiebreaker is the candidate key (which is a tuple of strings/ints),
    ensuring a stable order. Drops candidates with zero overlap.
    """
    scored: list[tuple[int, object]] = []
    for key, fs in candidate_fact_sets.items():
        overlap = len(target_facts & fs)
        if overlap > 0:
            scored.append((overlap, key))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [key for _, key in scored[:limit]]


# ── Per-kind enriched text builders ───────────────────────────────────────────


def _bare_fact_text(item: "ExtractedItem") -> str:
    parts: list[str] = [item.summary.strip()]
    if item.occurred_at:
        parts.append(f"date: {item.occurred_at}")
    if item.topics:
        parts.append("topics: " + ", ".join(item.topics))
    if item.tags:
        parts.append("tags: " + ", ".join(item.tags))
    return "\n".join(p for p in parts if p)


def _bare_entity_text(ent: "EntityRecord") -> str:
    # Bare-layer = OG content only: canonical_name, type, role,
    # aliases, description. canonical_id is a key field, not entity
    # content — its retrieval-side surfacing (so a dense query that
    # mentions the entity by id can match the entity's own embed
    # text via similarity) is an enrichment concern and lives in
    # `build_entity_text` on top of this bare body, not in here.
    parts: list[str] = []
    name = ent.canonical_name.strip()
    if name:
        parts.append(name)
    type_role = " · ".join(p for p in (ent.entity_type, ent.role) if p)
    if type_role:
        parts.append(type_role)
    if ent.aliases:
        parts.append("aliases: " + ", ".join(ent.aliases))
    if ent.description:
        parts.append(ent.description.strip())
    return "\n".join(parts)


def _bare_pattern_text(pat: "Pattern") -> str:
    parts: list[str] = [pat.name.strip()]
    kind_dom = " · ".join(p for p in (pat.kind or "", pat.domain) if p)
    if kind_dom:
        parts.append(kind_dom)
    if pat.description:
        parts.append(pat.description.strip())
    return "\n".join(parts)


def _bare_insight_text(ins: "Insight") -> str:
    parts: list[str] = [ins.name.strip()]
    if ins.kind:
        parts.append(ins.kind)
    if ins.description:
        parts.append(ins.description.strip())
    if ins.mechanism:
        parts.append("mechanism: " + ins.mechanism.strip())
    if ins.implication:
        parts.append("implication: " + ins.implication.strip())
    return "\n".join(parts)


# Positional insight references the actions LLM emits in prose, e.g.
# `Insight [5] shows ...`. The negative lookbehind keeps array-index
# syntax like `arr[5]` out of the pool — same discriminator the chatbot
# citation scanner uses for its `[N]` markers.
_INSIGHT_REF_RE = re.compile(r"(?<![A-Za-z0-9])\[(\d+)\]")


def _resolve_insight_refs(text: str, view: "GraphView") -> str:
    """Replace each positional `[N]` insight reference in `text` with the
    referenced insight's title in double quotes, e.g.
    `why: Insight [5] shows ...` → `why: Insight "Toxic self-medication
    loop" shows ...`.

    `[N]` is the actions-prompt enumeration index (1-based over
    cross-domain insights first, then critical) — the same order
    `actions._build_prompt` numbers by, which the phase-boundary
    normalizer preserves end-to-end. Resolution walks that full
    enumeration, NOT the action's cited `source_insights` subset: the
    model often names an insight in prose without formally citing it, so
    the cited subset would miss refs.

    Applied only when building the embedded / chatbot-CONTEXT text, where
    a bare `[N]` is both meaningless (it resolves to nothing once stored)
    and collides with the chatbot's own `[N]` citation convention. The
    raw ``Action.why`` keeps its literal `[N]`; the UI surfaces resolve
    the bracket to a clickable insight link instead. An `[N]` outside the
    enumeration is left untouched — we never fabricate a title."""
    cross = view.insights_cross
    crit = view.insights_critical
    n_cross = len(cross)
    n_total = n_cross + len(crit)

    def _sub(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        if not (1 <= n <= n_total):
            return m.group(0)
        ins = cross[n - 1] if n <= n_cross else crit[n - 1 - n_cross]
        title = (ins.name or "").strip()
        return f'"{title}"' if title else m.group(0)

    return _INSIGHT_REF_RE.sub(_sub, text)


def _bare_action_text(act: "Action") -> str:
    parts: list[str] = [act.recommendation.strip()]
    if act.kind:
        parts.append(act.kind)
    if act.objective:
        parts.append("objective: " + act.objective.strip())
    if act.why:
        parts.append("why: " + act.why.strip())
    if act.immediate_action:
        parts.append("immediate action: " + act.immediate_action.strip())
    if act.habit:
        parts.append("habit: " + act.habit.strip())
    if act.success_metric:
        parts.append("success metric: " + act.success_metric.strip())
    return "\n".join(parts)


# ── Per-kind display-text builders ───────────────────────────────────────────
#
# Companions to the `build_*_text` enriched-prefix builders. The
# enriched form is what the embedder consumes (graph context boosts
# retrieval relevance — well-validated "contextual retrieval"
# pattern); the bare-display form is what the chatbot loop renders
# into CONTEXT for the answering model, with the graph-context
# prefix dropped so the model never sees the canonical-id-shape
# surface (kind brackets, name-lists referencing other records) it
# would otherwise infer as an in-band citation format.
#
# Section: line is preserved on chunk + fact records (useful
# semantic context, not a citation leak channel); everything else
# from the enriched prefix (Type:, Quote:, Date span, histograms,
# Top-N lists, downstream entities/patterns/insights/actions) is
# dropped. Substantive body fields stay verbatim (chunk.text,
# `_bare_*_text` per kind).


def _section_prefix(section: tuple[str, ...]) -> str:
    """`Section: a › b › c\n\n` when ``section`` is non-empty; empty
    string otherwise. Joined into the bare body by callers so the
    blank-line separator only appears when the section line does."""
    if not section:
        return ""
    return "Section: " + " › ".join(section) + "\n\n"


def build_chunk_display(chunk: RagChunk) -> str:
    """Bare-display form of a chunk: a Source line naming the file the
    chunk came from, an optional Section line, then the raw chunk text.

    Filename is core provenance the user expects to be able to reference;
    surfacing it here lets the answering model identify which file a cited
    chunk belongs to. ``file_id`` is the corpus-relative identifier — a
    bare basename for single-file ingests, or a folder-relative subpath
    (e.g. ``journals/2024/notes.md``) for directory / archive ingests,
    which usefully disambiguates same-named files across folders. The full
    absolute ``source_path`` stays out of display."""
    body = _section_prefix(chunk.section_path) + chunk.text
    if not chunk.file_id:
        return body
    # Pad to a blank line before the body only when there's no Section line
    # already providing the separator.
    sep = "\n" if chunk.section_path else "\n\n"
    return f"Source: {chunk.file_id}{sep}" + body


def build_fact_display(
    fact: "ExtractedItem", view: GraphView,
) -> str:
    """Bare-display form of a fact: optional Section line (resolved
    against the doc the fact's evidence points at) then the bare
    fact body."""
    return _section_prefix(_section_path_for_fact(fact, view)) + _bare_fact_text(fact)


def build_entity_display(ent: "EntityRecord") -> str:
    return _bare_entity_text(ent)


def build_pattern_display(pat: "Pattern") -> str:
    return _bare_pattern_text(pat)


def build_insight_display(ins: "Insight") -> str:
    return _bare_insight_text(ins)


def build_action_display(act: "Action", view: "GraphView") -> str:
    # display_text is the bare body the answering model reads in the
    # chatbot CONTEXT block (and the user's source-preview panel). Resolve
    # positional `[N]` insight refs to the quoted title here too — this is
    # the surface where a dangling `[N]` both reads as nothing and collides
    # with the chatbot's own `[N]` citation brackets. The structured
    # Action.why the actions UI renders keeps the literal bracket (surfaced
    # there as a clickable insight link).
    return _resolve_insight_refs(_bare_action_text(act), view)


def build_chunk_text(chunk: RagChunk, view: GraphView) -> str:
    """Enriched text for an input file chunk. Per spec:
      - raw chunk text + file path + date + section path
      - split summaries from extraction
      - topics/tags histogram from facts in this chunk
      - downstream: fact titles+types, entities (id, canonical, aliases,
        type, role), relations where both endpoints are mentioned via
        facts in this chunk
    """
    lines: list[str] = [_type_header("chunk")]

    # File + date
    doc_date = ""
    for d in view.docs:
        if d.source_path == chunk.source_path:
            doc_date = d.date or ""
            break
    file_line = f"File: {chunk.source_path}"
    if doc_date:
        file_line += f" | {doc_date}"
    lines.append(file_line)
    if chunk.section_path:
        lines.append("Section: " + " › ".join(chunk.section_path))

    # Facts contained in this chunk, located by file_id + offset overlap.
    # ``ev.file_path`` is set to ``doc.file_id`` at extraction time
    # (content_extractor.py:_file_path_for — basename / doc-identifier),
    # while ``chunk.source_path`` is the FULL absolute path. Compare on
    # ``chunk.file_id`` so the basename forms match — same fix as
    # ``build_edges``'s chunk↔fact containment loop.
    contained: list[tuple[str, int]] = []
    for topic in sorted(view.facts_by_topic.keys()):
        for idx, fact in enumerate(view.facts_by_topic[topic]):
            for ev in getattr(fact, "evidence", None) or []:
                fp = getattr(ev, "file_path", None)
                fo = getattr(ev, "file_offset", None)
                if fp != chunk.file_id or fo is None:
                    continue
                if chunk.char_offset <= fo < chunk.char_offset + len(chunk.text):
                    contained.append((topic, idx))
                    break

    # Split summaries — via the splitter chunks that produced these facts.
    splitter_ids: list[str] = []
    seen_split: set[str] = set()
    for (topic, idx) in contained:
        f = view.fact_at(topic, idx)
        if f is None:
            continue
        for ev in getattr(f, "evidence", None) or []:
            src = getattr(ev, "source_ref", None)
            if src and src not in seen_split:
                seen_split.add(str(src))
                splitter_ids.append(str(src))
            break
    summaries: list[str] = []
    for sid in splitter_ids:
        for s in view.summaries_by_splitter_chunk.get(sid, []):
            if s not in summaries:
                summaries.append(s)
    if summaries:
        lines.append("Split summaries:")
        for s in summaries:
            lines.append(f"  - {_truncate(s, MAX_PROSE_CHARS)}")

    # Topics + tags histograms across facts in this chunk.
    th = _topic_histogram(contained, view)
    if th:
        lines.append(
            "Topics: " + ", ".join(f"{t}×{n}" for (t, n) in th)
        )
    gh = _tag_histogram(contained, view)
    if gh:
        lines.append(
            "Tags: " + ", ".join(f"{t}×{n}" for (t, n) in gh)
        )

    # Downstream: facts in this chunk (titles + types).
    if contained:
        lines.append("Facts in chunk:")
        for (topic, idx) in contained:
            f = view.fact_at(topic, idx)
            if f is None:
                continue
            lines.append(f"  - [{_fact_type(f)}] {_fact_title(f)}")

    # Downstream: entities mentioned via facts in this chunk.
    entity_ids: list[str] = []
    seen_eid: set[str] = set()
    for key in contained:
        for cid in view.fact_entities.get(key, []):
            if cid not in seen_eid:
                seen_eid.add(cid)
                entity_ids.append(cid)
    if entity_ids:
        lines.append("Entities:")
        for cid in entity_ids:
            ent = view.entities_by_id.get(cid)
            if ent is None:
                continue
            alias_part = (
                " aka " + ", ".join(ent.aliases) if ent.aliases else ""
            )
            lines.append(
                f"  - {_format_entity_with_type_role(ent)}"
                f" [id: {ent.canonical_id}]{alias_part}"
            )

    # Downstream: relations where BOTH endpoints are entities mentioned
    # in this chunk's facts.
    chunk_eids = set(entity_ids)
    rel_lines: list[str] = []
    seen_rel: set[tuple[str, str, str]] = set()
    for cid in entity_ids:
        for rel in view.relations_by_entity.get(cid, []):
            if rel.from_id in chunk_eids and rel.to_id in chunk_eids:
                key = (rel.from_id, rel.to_id, rel.relation)
                if key in seen_rel:
                    continue
                seen_rel.add(key)
                rel_lines.append(
                    f"  - {rel.from_id} --{rel.relation}--> {rel.to_id}"
                )
    if rel_lines:
        lines.append("Relations:")
        lines.extend(rel_lines)

    lines.append("")  # blank separator before bare text
    lines.append(chunk.text)
    return "\n".join(lines)


# ── Document (file-level) record builders ─────────────────────────────────────
#
# One ``document`` record per ingested source file. It makes a file a
# first-class retrieval target: embedded so a dense query about a file
# ("do I have my meditations notes?") can hit it, displayed so the model
# can answer file-inventory questions, and edge-linked to its chunks so
# ``has_neighbor`` walks both ways (chunk → its file, file → its chunks).
# The summary reuses existing extraction signal (split summaries) rather
# than spending a new LLM pass.


def _document_file_id(doc: "Document") -> str:
    """The file identifier a document record keys on — the same value
    its chunks carry in ``RagChunk.file_id`` (basename for single-file
    ingests, corpus-relative subpath for directory / archive ingests),
    so chunk↔document edges and the ``source`` filter line up."""
    return doc.file_id or doc.id


def _document_sections(view: GraphView, file_id: str) -> list[tuple[str, ...]]:
    """The file's section paths in first-seen order, deduped — the
    header structure the chunker recorded across the file's chunks.
    Empty when the file has no headers (the whole-doc fast path)."""
    seen: set[tuple[str, ...]] = set()
    out: list[tuple[str, ...]] = []
    for ci in view.rag_chunks_by_file_id.get(file_id, []):
        sp = view.chunks[ci].section_path
        if sp and sp not in seen:
            seen.add(sp)
            out.append(sp)
    return out


def _document_summaries(view: GraphView, file_id: str) -> list[str]:
    """The split summaries belonging to this file, deduped in
    first-seen order. A splitter chunk belongs to the file when a fact
    drawn from it cites the file in its evidence ``file_path`` (set to
    ``doc.file_id`` at extraction time). Reuses the summary signal the
    chunk builder already consults — no new LLM pass."""
    out: list[str] = []
    seen: set[str] = set()
    for splitter_id, fact_keys in view.facts_by_splitter_chunk.items():
        belongs = False
        for (topic, idx) in fact_keys:
            f = view.fact_at(topic, idx)
            for ev in getattr(f, "evidence", None) or []:
                if getattr(ev, "file_path", None) == file_id:
                    belongs = True
                    break
            if belongs:
                break
        if not belongs:
            continue
        for s in view.summaries_by_splitter_chunk.get(splitter_id, []):
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def build_document_text(doc: "Document", view: GraphView) -> str:
    """Enriched embed text for a source-file record: filename + date +
    title + section structure + chunk count + the file's split
    summaries. Embedding this lets a dense query about a file by name or
    topic land on the document record directly."""
    file_id = _document_file_id(doc)
    lines: list[str] = [_type_header("document")]
    head = f"File: {file_id}"
    if doc.date:
        head += f" | {doc.date}"
    lines.append(head)
    if doc.title:
        lines.append(f"Title: {doc.title}")
    n_chunks = len(view.rag_chunks_by_file_id.get(file_id, []))
    lines.append(f"Chunks: {n_chunks}")
    sections = _document_sections(view, file_id)
    if sections:
        lines.append("Sections:")
        for sp in sections:
            lines.append("  - " + " › ".join(sp))
    summaries = _document_summaries(view, file_id)
    if summaries:
        lines.append("Summary:")
        for s in summaries:
            lines.append(f"  - {_truncate(s, MAX_PROSE_CHARS)}")
    return "\n".join(lines)


def build_document_display(doc: "Document", view: GraphView) -> str:
    """Bare-display form of a source-file record — what the answering
    model reads in CONTEXT. Names the file and describes its shape (date,
    title, chunk count, section list, summary) so the model can answer
    "do I have file X?" / "what's in file X?" / "how is file X
    structured?" without inventing details. ``file_id`` is the basename /
    corpus-relative identifier; the absolute path stays out of display."""
    file_id = _document_file_id(doc)
    lines: list[str] = [f"File: {file_id}"]
    if doc.date:
        lines.append(f"Date: {doc.date}")
    if doc.title:
        lines.append(f"Title: {doc.title}")
    n_chunks = len(view.rag_chunks_by_file_id.get(file_id, []))
    lines.append(f"{n_chunks} chunk{'' if n_chunks == 1 else 's'}")
    sections = _document_sections(view, file_id)
    if sections:
        lines.append(
            "Sections: " + "; ".join(" › ".join(sp) for sp in sections)
        )
    summaries = _document_summaries(view, file_id)
    if summaries:
        lines.append("")
        lines.append("Summary:")
        for s in summaries:
            lines.append(f"- {_truncate(s, MAX_PROSE_CHARS)}")
    return "\n".join(lines)


def build_fact_text(
    topic: str, idx: int, fact: "ExtractedItem", view: GraphView,
) -> str:
    """Enriched text for a fact. Per spec:
      - fact record (title, date, occurred_at, topics, tags, confidence)
      - upstream: section of input quoted as source
      - downstream: entities (id, canonical, aliases, type, role),
        relation candidate, patterns mentioning this fact
        (title+kind+topic)
      - sibling: next + previous facts from same splitter chunk
    """
    lines: list[str] = [_type_header("fact")]

    # Upstream: section of input the fact quotes. Computed from the
    # doc text + fact offset (works for small whole-doc chunks where
    # the RAG chunker's section_path is empty); falls back to the
    # enclosing chunk's section_path on docs too large to parse.
    section = _section_path_for_fact(fact, view)
    if section:
        lines.append("Section: " + " › ".join(section))
    quote = ""
    for ev in getattr(fact, "evidence", None) or []:
        if (ev.text or "").strip():
            quote = ev.text.strip()
            break
    if quote:
        lines.append("Quote: " + _truncate(quote, MAX_QUOTE_CHARS))
    if getattr(fact, "confidence", None) is not None:
        lines.append(f"Confidence: {fact.confidence:.2f}")

    # Sibling facts from same splitter chunk. Position is resolved
    # against the canonical-deduped sibling list so an underlying fact
    # carried under N category-copies (same provenance, multiple topics)
    # contributes ONE entry — prev/next always resolve to a genuinely
    # different underlying fact, never to an alias of the current one.
    source_ref = ""
    for ev in getattr(fact, "evidence", None) or []:
        if getattr(ev, "source_ref", None):
            source_ref = str(ev.source_ref)
            break
    if source_ref:
        siblings = view.canonical_siblings(source_ref)
        canonical_self = view.canonical_fact_key(topic, idx)
        try:
            position = siblings.index(canonical_self)
        except ValueError:
            position = -1
        if position > 0:
            prev = view.fact_at(*siblings[position - 1])
            if prev is not None:
                lines.append(
                    f"Previous fact: [{_fact_type(prev)}] {_fact_title(prev)}"
                )
        if 0 <= position < len(siblings) - 1:
            nxt = view.fact_at(*siblings[position + 1])
            if nxt is not None:
                lines.append(
                    f"Next fact: [{_fact_type(nxt)}] {_fact_title(nxt)}"
                )

    # Downstream: entities mentioning this fact. Union across all
    # aliases of the canonical so the record carries the maximally
    # informative entity list when the upstream entities stage gave
    # each category-copy the same set (director invariant 1); a
    # disagreement is surfaced separately via `entity_alias_drift`.
    seen_eid: set[str] = set()
    eids: list[str] = []
    for alias in view.aliases_of(topic, idx):
        for cid in view.fact_entities.get(alias, []):
            if cid in seen_eid:
                continue
            seen_eid.add(cid)
            eids.append(cid)
    if eids:
        lines.append("Entities:")
        for cid in eids:
            ent = view.entities_by_id.get(cid)
            if ent is None:
                continue
            alias_part = (
                " aka " + ", ".join(ent.aliases) if ent.aliases else ""
            )
            lines.append(
                f"  - {_format_entity_with_type_role(ent)}"
                f" [id: {ent.canonical_id}]{alias_part}"
            )

    # Downstream: relation candidate (single per fact, per extractor).
    rc = getattr(fact, "relation_candidate", None) or None
    if isinstance(rc, dict) and rc.get("from") and rc.get("to"):
        verb = rc.get("verb") or rc.get("relation") or "related"
        lines.append(
            f"Relation candidate: {rc['from']} --{verb}--> {rc['to']}"
        )

    # Downstream: patterns mentioning this fact — UNION across category-
    # copies. A pattern in topic A cites facts by their fact-idx within
    # facts_by_topic[A], so the same underlying fact reached via topic
    # B's bucket is invisible to A's pattern under a strict-membership
    # check. Intersecting the pattern's fact set with the alias set of
    # the current fact recovers every pattern across every category the
    # fact lived under.
    my_aliases = set(view.aliases_of(topic, idx))
    pat_lines: list[str] = []
    for pkey, fs in view.pattern_facts.items():
        if not (my_aliases & fs):
            continue
        pat = view.pattern_at(*pkey)
        if pat is None:
            continue
        kind = pat.kind or ""
        pat_lines.append(
            f"  - {pat.name.strip()} | kind: {kind} | topic: {pat.domain}"
        )
    if pat_lines:
        lines.append("Patterns mentioning this fact:")
        lines.extend(pat_lines)

    lines.append("")
    lines.append(_bare_fact_text(fact))
    return "\n".join(lines)


def build_entity_text(ent: "EntityRecord", view: GraphView) -> str:
    """Enriched text for an entity. Per spec:
      - entity record + date span + relation type+id of other entities
      - upstream: TOP-3 most-confident + TOP-3 most-recent fact
        titles + types
      - downstream: TOP-2 patterns / TOP-2 insights / TOP-2 actions
        mentioning facts that mention this entity
    """
    lines: list[str] = [_type_header("entity")]

    # Surface the canonical_id on the enriched record so a dense query
    # that mentions the entity by id can match the entity's own embed
    # text via similarity (cross-record references in chunk + fact
    # prefixes already carry the id as `[id: ...]`). Lives in the
    # additive enriched layer, never in the bare body — the bare body
    # carries OG entity content only.
    lines.append(f"ID: {ent.canonical_id}")

    fact_keys = view.entity_facts.get(ent.canonical_id, set())
    lines.append(f"Mention count: {ent.mention_count}")
    span = _date_span(fact_keys, view)
    if span:
        lines.append(f"Date span: {span}")

    # Relations.
    rels = view.relations_by_entity.get(ent.canonical_id, [])
    if rels:
        lines.append("Relations:")
        for rel in rels:
            other = (
                rel.to_id if rel.from_id == ent.canonical_id else rel.from_id
            )
            direction = "→" if rel.from_id == ent.canonical_id else "←"
            lines.append(
                f"  - {direction} {rel.relation} → {other}"
            )

    # Upstream: top-3 most-confident.
    facts = []
    for key in fact_keys:
        f = view.fact_at(*key)
        if f is not None:
            facts.append((key, f))
    by_conf = sorted(
        facts,
        key=lambda kf: (-(getattr(kf[1], "confidence", 0.0) or 0.0), kf[0]),
    )[:TOP_FACTS_BY_CONFIDENCE_ENTITY]
    if by_conf:
        lines.append(
            f"Top {TOP_FACTS_BY_CONFIDENCE_ENTITY} most-confident facts:"
        )
        for (_k, f) in by_conf:
            lines.append(f"  - [{_fact_type(f)}] {_fact_title(f)}")

    # Upstream: top-3 most-recent.
    def _recency_key(kf):
        _key, f = kf
        d = (getattr(f, "occurred_at", None) or "").strip()
        return (d == "", -ord_date(d), kf[0])

    by_recent = sorted(facts, key=_recency_key)[:TOP_FACTS_BY_RECENCY_ENTITY]
    if by_recent:
        lines.append(
            f"Top {TOP_FACTS_BY_RECENCY_ENTITY} most-recent facts:"
        )
        for (_k, f) in by_recent:
            d = (getattr(f, "occurred_at", None) or "").strip()
            tag = f" ({d})" if d else ""
            lines.append(f"  - [{_fact_type(f)}] {_fact_title(f)}{tag}")

    # Downstream: top-2 patterns / insights / actions by overlap.
    top_patterns = _rank_by_fact_overlap(
        fact_keys, view.pattern_facts, TOP_PATTERNS_ENTITY,
    )
    if top_patterns:
        lines.append(f"Top {TOP_PATTERNS_ENTITY} patterns:")
        for pkey in top_patterns:
            p = view.pattern_at(*pkey)
            if p is None:
                continue
            lines.append(
                f"  - {p.name.strip()} | kind: {p.kind or ''} "
                f"| topic: {p.domain}"
            )

    top_insights = _rank_by_fact_overlap(
        fact_keys, view.insight_facts, TOP_INSIGHTS_ENTITY,
    )
    if top_insights:
        lines.append(f"Top {TOP_INSIGHTS_ENTITY} insights:")
        for ikey in top_insights:
            ins = view.insight_at(*ikey)
            if ins is None:
                continue
            lines.append(f"  - [{ins.kind or ''}] {ins.name.strip()}")

    top_actions = _rank_by_fact_overlap(
        fact_keys, view.action_facts, TOP_ACTIONS_ENTITY,
    )
    if top_actions:
        lines.append(f"Top {TOP_ACTIONS_ENTITY} actions:")
        for aidx in top_actions:
            act = view.action_at(aidx)
            if act is None:
                continue
            lines.append(
                f"  - [{act.kind or ''}] {act.recommendation.strip()}"
            )

    lines.append("")
    lines.append(_bare_entity_text(ent))
    return "\n".join(lines)


def ord_date(s: str) -> int:
    """Crude ISO-date → ordinal for descending recency sort. Empty
    strings return -1 so they sort to the end via the caller's
    (empty, ord, key) tuple. Days-since-epoch is fine — we only need
    a strict order, not arithmetic."""
    if not s or len(s) < 10:
        return 0
    try:
        y = int(s[0:4])
        m = int(s[5:7])
        d = int(s[8:10])
        return y * 10000 + m * 100 + d
    except (ValueError, IndexError):
        return 0


def build_pattern_text(
    topic: str, idx: int, pat: "Pattern", view: GraphView,
) -> str:
    """Enriched text for a pattern. Per spec:
      - pattern record (title, summary, kind, topic, fact count)
      - date span
      - upstream: TOP-5 fact titles+types (most confident);
        entity histogram with counts
      - downstream: insights mentioning this pattern, actions
        mentioning this pattern
    """
    lines: list[str] = [_type_header("pattern")]
    fact_keys = view.pattern_facts.get((topic, idx), set())
    lines.append(f"Fact count: {len(fact_keys)}")
    span = _date_span(fact_keys, view)
    if span:
        lines.append(f"Date span: {span}")

    # Upstream: top-5 most-confident facts from pattern.source_facts
    # (already sorted by confidence desc; per spec we take top-5).
    by_conf = []
    for (fact_idx, _conf) in (
        getattr(pat, "source_facts", []) or []
    )[:TOP_FACTS_PATTERN]:
        f = view.fact_at(topic, int(fact_idx))
        if f is not None:
            by_conf.append(f)
    if by_conf:
        lines.append(
            f"Top {TOP_FACTS_PATTERN} most-confident facts:"
        )
        for f in by_conf:
            lines.append(f"  - [{_fact_type(f)}] {_fact_title(f)}")

    # Upstream: entity histogram across pattern facts.
    eh = _entity_histogram(fact_keys, view)
    if eh:
        lines.append("Entities (histogram):")
        for cid, n in eh:
            ent = view.entities_by_id.get(cid)
            label = (
                _format_entity_with_type_role(ent) if ent is not None else cid
            )
            lines.append(f"  - {label} ×{n}")

    # Downstream: insights mentioning this pattern (via source_patterns).
    ins_lines: list[str] = []
    for (scope, iidx, ins) in view.insights_all:
        for pkey in view.insight_patterns.get((scope, iidx), []):
            if pkey == (topic, idx):
                ins_lines.append(f"  - [{ins.kind or ''}] {ins.name.strip()}")
                break
    if ins_lines:
        lines.append("Insights mentioning this pattern:")
        lines.extend(ins_lines)

    # Downstream: actions mentioning this pattern (via source_insights
    # → source_patterns).
    act_lines: list[str] = []
    for aidx, act in enumerate(view.actions):
        if (topic, idx) in view.action_patterns.get(aidx, []):
            act_lines.append(
                f"  - [{act.kind or ''}] {act.recommendation.strip()}"
            )
    if act_lines:
        lines.append("Actions mentioning this pattern:")
        lines.extend(act_lines)

    lines.append("")
    lines.append(_bare_pattern_text(pat))
    return "\n".join(lines)


def build_insight_text(
    scope: str, idx: int, ins: "Insight", view: GraphView,
) -> str:
    """Enriched text for an insight. Per spec:
      - insight record (title, type, prose blob)
      - date span + pattern count + fact count
      - upstream: topics & entities histograms from facts mentioned
        in this insight; patterns mentioned (title+kind+topic)
      - downstream: actions mentioning this insight
    """
    lines: list[str] = [_type_header("insight")]
    fact_keys = view.insight_facts.get((scope, idx), set())
    pkeys = view.insight_patterns.get((scope, idx), [])
    lines.append(f"Scope: {scope}")
    lines.append(f"Pattern count: {len(pkeys)}")
    lines.append(f"Fact count: {len(fact_keys)}")
    span = _date_span(fact_keys, view)
    if span:
        lines.append(f"Date span: {span}")

    th = _topic_histogram(fact_keys, view)
    if th:
        lines.append("Topics (histogram):")
        for t, n in th:
            lines.append(f"  - {t} ×{n}")

    eh = _entity_histogram(fact_keys, view)
    if eh:
        lines.append("Entities (histogram):")
        for cid, n in eh:
            ent = view.entities_by_id.get(cid)
            label = (
                _format_entity_with_type_role(ent) if ent is not None else cid
            )
            lines.append(f"  - {label} ×{n}")

    if pkeys:
        lines.append("Patterns mentioned:")
        for pkey in pkeys:
            p = view.pattern_at(*pkey)
            if p is None:
                continue
            lines.append(
                f"  - {p.name.strip()} | kind: {p.kind or ''} "
                f"| topic: {p.domain}"
            )

    # Downstream: actions mentioning this insight.
    act_lines: list[str] = []
    for aidx, act in enumerate(view.actions):
        if (scope, idx) in view.action_insights.get(aidx, []):
            act_lines.append(
                f"  - [{act.kind or ''}] {act.recommendation.strip()}"
            )
    if act_lines:
        lines.append("Actions mentioning this insight:")
        lines.extend(act_lines)

    lines.append("")
    lines.append(_bare_insight_text(ins))
    return "\n".join(lines)


def build_action_text(idx: int, act: "Action", view: GraphView) -> str:
    """Enriched text for an action. Per spec — upstream only:
      - action record (title, type, prose blob)
      - date span + insight count + pattern count + fact count
      - upstream: topics histogram, entities histogram, patterns
        mentioned (title+kind+topic), insights mentioned (titles+types)
    """
    lines: list[str] = [_type_header("action")]
    fact_keys = view.action_facts.get(idx, set())
    pkeys = view.action_patterns.get(idx, [])
    ikeys = view.action_insights.get(idx, [])

    lines.append(f"Horizon: {act.horizon}")
    if act.review_date:
        lines.append(f"Review date: {act.review_date}")
    lines.append(f"Insight count: {len(ikeys)}")
    lines.append(f"Pattern count: {len(pkeys)}")
    lines.append(f"Fact count: {len(fact_keys)}")
    span = _date_span(fact_keys, view)
    if span:
        lines.append(f"Date span: {span}")

    th = _topic_histogram(fact_keys, view)
    if th:
        lines.append("Topics (histogram):")
        for t, n in th:
            lines.append(f"  - {t} ×{n}")

    eh = _entity_histogram(fact_keys, view)
    if eh:
        lines.append("Entities (histogram):")
        for cid, n in eh:
            ent = view.entities_by_id.get(cid)
            label = (
                _format_entity_with_type_role(ent) if ent is not None else cid
            )
            lines.append(f"  - {label} ×{n}")

    if pkeys:
        lines.append("Patterns mentioned:")
        for pkey in pkeys:
            p = view.pattern_at(*pkey)
            if p is None:
                continue
            lines.append(
                f"  - {p.name.strip()} | kind: {p.kind or ''} "
                f"| topic: {p.domain}"
            )

    if ikeys:
        lines.append("Insights mentioned:")
        for ikey in ikeys:
            ins = view.insight_at(*ikey)
            if ins is None:
                continue
            lines.append(f"  - [{ins.kind or ''}] {ins.name.strip()}")

    lines.append("")
    # Resolve positional `[N]` insight refs to the insight title here, at
    # the embedding/CONTEXT seam — the bare text feeds the vectorized
    # record + the chatbot CONTEXT block, where a dangling `[N]` resolves
    # to nothing and collides with the chatbot's own citation brackets.
    lines.append(_resolve_insight_refs(_bare_action_text(act), view))
    return "\n".join(lines)


# ── Edge emission ─────────────────────────────────────────────────────────────


# Edge tuple shape: (src_kind, src_id, dst_kind, dst_id, edge_kind).
# Mirrors the `edges` table column order in rag_vector_store so the
# walker can hand its output straight to `VectorStore.add_edges`.
Edge = tuple[str, str, str, str, str]


def _chunk_record_id(chunk: RagChunk) -> str:
    """Record id minted by the embed stage for chunk records. Kept in
    one place so the edge walker and `build_embeddings_plan` can't
    drift on the format (`{file_id}@{char_offset}`)."""
    return f"{chunk.file_id}@{chunk.char_offset}"


def build_edges(view: GraphView) -> list[Edge]:
    """Walk the GraphView and emit directed edges between records keyed
    on the same `(kind, record_id)` the records table uses. Edges follow
    the renderer-intent rule: an anchor record points at the records
    its embed prefix references, persisted at the COMPLETE adjacency
    (not the TOP_* render caps that bound the embed-text token budget).

    Directionality is deliberate per slice-2 spec:
      - chunk ↔ fact bidirectional (containment); chunk → entity only
        (entity records don't list their enclosing chunks).
      - fact ↔ fact siblings (prev/next within same splitter chunk).
      - fact ↔ entity bidirectional (mention).
      - fact ↔ pattern bidirectional (derivation).
      - entity → {pattern, insight, action} via complete fact overlap
        (not top-2). Action records do NOT emit action → entity edges:
        anti-fan-out, the renderer's entity histogram is for embed
        semantics only.
      - entity ↔ entity (relation), one directed row per anchor side of
        a RelationEdge so `has_neighbor` from either endpoint hits the
        same neighbor.
      - pattern ↔ insight, pattern ↔ action, insight ↔ action: all
        bidirectional via derivation chain.

    The action → chunk hop is reached transitively (action → pattern →
    fact → chunk), so no direct edge is emitted there.
    """
    # All fact endpoints emitted into the edges table use the canonical
    # (topic, idx) for the underlying fact under cross-category dedup
    # (director invariant 4: no neighbor edge may resolve to a category-
    # copy of the source fact). `_fact_id` projects any (topic, idx) to
    # the canonical record-id string the embeddings plan writes for the
    # surviving record.
    def _fact_id(topic: str, idx: int) -> str:
        return view.canonical_fact_id(topic, idx)

    edges: list[Edge] = []

    # chunk ↔ fact (containment), chunk → entity (mention via facts).
    # Category-copies of the same underlying fact in the same chunk
    # collapse to one containment pair via the canonical id; PK on the
    # edges table makes the re-emission idempotent.
    for chunk in view.chunks:
        c_id = _chunk_record_id(chunk)
        contained: list[tuple[str, int]] = []
        for topic in sorted(view.facts_by_topic.keys()):
            for idx, fact in enumerate(view.facts_by_topic[topic]):
                for ev in getattr(fact, "evidence", None) or []:
                    fp = getattr(ev, "file_path", None)
                    fo = getattr(ev, "file_offset", None)
                    # ``ev.file_path`` is set to ``doc.file_id`` at
                    # extraction time (content_extractor.py:_file_path_for
                    # — basename or doc-identifier), while
                    # ``chunk.source_path`` is the FULL absolute path.
                    # Compare on ``chunk.file_id`` instead so the basename
                    # forms match — otherwise every fact's evidence is
                    # skipped and zero chunk↔fact edges land in the table,
                    # breaking the bottom of the traceability chain
                    # (action → … → chunk via has_neighbor).
                    if fp != chunk.file_id or fo is None:
                        continue
                    if chunk.char_offset <= fo < chunk.char_offset + len(chunk.text):
                        contained.append((topic, idx))
                        break
        for (topic, idx) in contained:
            f_id = _fact_id(topic, idx)
            edges.append(("chunk", c_id, "fact", f_id, EDGE_KIND_CONTAINMENT))
            edges.append(("fact", f_id, "chunk", c_id, EDGE_KIND_CONTAINMENT))
        # chunk ↔ document (containment). The document record keys on the
        # file_id, the same value the chunk carries, so `has_neighbor`
        # walks both ways: from a chunk to the file it belongs to, and
        # from a file record to its chunks. Dead chunk↔document edges
        # (a file_id with no emitted document record) are dropped by the
        # live-endpoint filter in run_embeddings_stage.
        if chunk.file_id:
            edges.append(
                ("chunk", c_id, "document", chunk.file_id, EDGE_KIND_CONTAINMENT)
            )
            edges.append(
                ("document", chunk.file_id, "chunk", c_id, EDGE_KIND_CONTAINMENT)
            )
        seen_entity_in_chunk: set[str] = set()
        for key in contained:
            for cid in view.fact_entities.get(key, []):
                if cid in seen_entity_in_chunk:
                    continue
                seen_entity_in_chunk.add(cid)
                edges.append(("chunk", c_id, "entity", cid, EDGE_KIND_MENTION))

    # fact ↔ fact siblings within a splitter chunk. The per-splitter
    # fact list is collapsed to canonicals (first-occurrence semantics)
    # BEFORE deriving prev/next, so adjacent positions are always
    # genuinely different underlying facts; sibling edges never form a
    # self-loop after the embedding-layer dedup.
    for source_ref in view.facts_by_splitter_chunk.keys():
        siblings = view.canonical_siblings(source_ref)
        for i in range(len(siblings) - 1):
            a_id = f"{siblings[i][0]}:{siblings[i][1]}"
            b_id = f"{siblings[i + 1][0]}:{siblings[i + 1][1]}"
            edges.append(("fact", a_id, "fact", b_id, EDGE_KIND_SIBLING))
            edges.append(("fact", b_id, "fact", a_id, EDGE_KIND_SIBLING))

    # fact ↔ entity (mention). Multiple aliases of the same underlying
    # fact carried in `evidence_fact_refs` collapse to one canonical
    # edge via the canonical projection; the edges-table PK keeps the
    # re-emission a single row.
    for (topic, idx), entity_ids in view.fact_entities.items():
        f_id = _fact_id(topic, idx)
        for ent_id in entity_ids:
            edges.append(("fact", f_id, "entity", ent_id, EDGE_KIND_MENTION))
            edges.append(("entity", ent_id, "fact", f_id, EDGE_KIND_MENTION))

    # fact ↔ pattern (derivation). A pattern in topic A cites facts via
    # fact-idx in topic A; canonical projection re-keys each citation to
    # the underlying fact's canonical id, so patterns across category-
    # copies share one fact endpoint per unique fact (union semantics
    # for the patterns mentioning each canonical record).
    for (p_topic, p_idx), fact_keys in view.pattern_facts.items():
        p_id = f"{p_topic}:{p_idx}"
        for (f_topic, f_idx) in fact_keys:
            f_id = _fact_id(f_topic, f_idx)
            edges.append(("pattern", p_id, "fact", f_id, EDGE_KIND_DERIVATION))
            edges.append(("fact", f_id, "pattern", p_id, EDGE_KIND_DERIVATION))

    # entity ↔ entity (relation). RelationEdge is directional but the
    # renderer lists relations on BOTH endpoints' records — emit one
    # row per anchor side so `has_neighbor` from either endpoint resolves.
    for rel in view.relations:
        edges.append(("entity", rel.from_id, "entity", rel.to_id, EDGE_KIND_RELATION))
        edges.append(("entity", rel.to_id, "entity", rel.from_id, EDGE_KIND_RELATION))

    # entity → {pattern, insight, action} via complete fact overlap.
    # Iterate every entity once; for each candidate, check overlap with
    # the entity's fact set. Complete adjacency (no TOP_* cap).
    for ent_id, ent_fact_keys in view.entity_facts.items():
        if not ent_fact_keys:
            continue
        for pkey, fs in view.pattern_facts.items():
            if ent_fact_keys & fs:
                p_id = f"{pkey[0]}:{pkey[1]}"
                edges.append(("entity", ent_id, "pattern", p_id, EDGE_KIND_MENTION))
                edges.append(("pattern", p_id, "entity", ent_id, EDGE_KIND_MENTION))
        for ikey, fs in view.insight_facts.items():
            if ent_fact_keys & fs:
                i_id = f"{ikey[0]}:{ikey[1]}"
                edges.append(("entity", ent_id, "insight", i_id, EDGE_KIND_MENTION))
                edges.append(("insight", i_id, "entity", ent_id, EDGE_KIND_MENTION))
        for aidx, fs in view.action_facts.items():
            if ent_fact_keys & fs:
                a_id = str(aidx)
                # entity → action only. action → entity is suppressed
                # (anti-fan-out per slice-2 spec): an action's fact
                # provenance hops out through patterns + insights, not
                # back to every entity touching any source fact.
                edges.append(("entity", ent_id, "action", a_id, EDGE_KIND_MENTION))

    # pattern ↔ insight (derivation).
    for (scope, iidx), pkeys in view.insight_patterns.items():
        i_id = f"{scope}:{iidx}"
        for (p_topic, p_idx) in pkeys:
            p_id = f"{p_topic}:{p_idx}"
            edges.append(("insight", i_id, "pattern", p_id, EDGE_KIND_DERIVATION))
            edges.append(("pattern", p_id, "insight", i_id, EDGE_KIND_DERIVATION))

    # action ↔ insight + action → pattern (transitive). NO action ↔
    # entity here (suppressed above).
    for aidx, ikeys in view.action_insights.items():
        a_id = str(aidx)
        for (scope, ins_idx) in ikeys:
            i_id = f"{scope}:{ins_idx}"
            edges.append(("action", a_id, "insight", i_id, EDGE_KIND_DERIVATION))
            edges.append(("insight", i_id, "action", a_id, EDGE_KIND_DERIVATION))
    for aidx, pkeys in view.action_patterns.items():
        a_id = str(aidx)
        for (p_topic, p_idx) in pkeys:
            p_id = f"{p_topic}:{p_idx}"
            edges.append(("action", a_id, "pattern", p_id, EDGE_KIND_DERIVATION))
            edges.append(("pattern", p_id, "action", a_id, EDGE_KIND_DERIVATION))

    return edges


def count_dead_end_anchors(edges: list[Edge], view: GraphView) -> dict[str, int]:
    """Per-anchor-kind count of records with zero outgoing edges. Surfaces
    #387-class corruption (e.g. an entity whose `evidence_fact_refs` was
    dropped by the upstream slug-key-drift) loudly at embed time, so a
    silent `has_neighbor` lossiness on stage-3 traversal is observable
    via the phase marker.

    Returns a dict keyed by anchor kind with the count of anchors having
    zero outgoing edges. Action anchors are intentionally excluded from
    the "dead-end" category for the `entity` neighbor type (anti-fan-out
    is by design, not corruption).
    """
    src_set: dict[str, set[str]] = {}
    for (src_kind, src_id, _dk, _di, _ek) in edges:
        src_set.setdefault(src_kind, set()).add(src_id)

    dead: dict[str, int] = {}

    chunks_total = {_chunk_record_id(c) for c in view.chunks}
    dead["chunk"] = len(chunks_total - src_set.get("chunk", set()))

    # Fact totals count canonical (topic, idx) only — the embeddings
    # plan emits one record per unique fact under cross-category dedup,
    # so non-canonical aliases are not part of the records table and
    # would falsely inflate the dead-end count if included here.
    facts_total: set[str] = set()
    for topic, facts in view.facts_by_topic.items():
        for i, _f in enumerate(facts):
            canonical = view.canonical_fact_key(topic, i)
            facts_total.add(f"{canonical[0]}:{canonical[1]}")
    dead["fact"] = len(facts_total - src_set.get("fact", set()))

    entities_total = {e.canonical_id for e in view.entities}
    dead["entity"] = len(entities_total - src_set.get("entity", set()))

    patterns_total: set[str] = set()
    for topic, pats in view.patterns_by_topic.items():
        for i, _p in enumerate(pats):
            patterns_total.add(f"{topic}:{i}")
    dead["pattern"] = len(patterns_total - src_set.get("pattern", set()))

    insights_total: set[str] = set()
    for scope, items in (
        ("cross_domain", view.insights_cross),
        ("critical", view.insights_critical),
    ):
        for i, _ins in enumerate(items):
            insights_total.add(f"{scope}:{i}")
    dead["insight"] = len(insights_total - src_set.get("insight", set()))

    actions_total = {str(i) for i in range(len(view.actions))}
    dead["action"] = len(actions_total - src_set.get("action", set()))

    return dead
