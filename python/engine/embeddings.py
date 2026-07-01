"""
Embeddings call surface + Embeddings pipeline stage.

The low-level callable (`embed`) is the stateless wire surface — one
call into Tinfoil's OpenAI-compatible `.embeddings` resource over the
same enclave-pinned transport the completion stages use. Modelled on
`vision.py`: the call surface lives in this module, `llm.py` owns the
shared client lifecycle.

The stage-level driver (`run_embeddings_stage`) takes the in-memory
upstream artifacts (input documents + per-stage record collections for
facts / entities / patterns / insights / actions), turns them into
embed-able rows, partitions them into fixed-size batches, and fires one
`embed()` call per batch through `scheduler.build_cascade_thunk`'s
halve cascade. Vectors land in a sqlite-vec-backed local vector store
under the run's `stages/06-embeddings/` tree. Per-record texts carry
GRAPH-ENRICHED context (upstream + downstream excerpts per spec § RAG
enhancements) prepended to the bare record text — assembly lives in
`rag_enricher.py` so the call surface here stays focused on batching,
cache, retry, and persistence.

Retry shape (inherits the chat-stage cascade core): Load failures
retry the same batch up to 5×; Sizing failures (context-window-exceeded
provider 400s, synthesised into cap-hit shape) halve the records
batch and dispatch two parallel children per level, up to the spec's
5 halve depths. Other failures retry once. The work-item type carried
through the cascade is `list[StoredRecord]` so halve siblings preserve
the records ↔ vectors alignment without losing identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from engine.llm import (
    Mode,
    Provider,
    _read_app_config,
)
from engine.rag_enricher import (
    Edge,
    build_action_display,
    build_action_text,
    build_chunk_display,
    build_chunk_text,
    build_document_display,
    build_document_text,
    build_edges,
    build_entity_display,
    build_entity_text,
    build_fact_display,
    build_fact_text,
    build_graph_view,
    build_insight_display,
    build_insight_text,
    build_pattern_display,
    build_pattern_text,
    count_dead_end_anchors,
)
from engine.rag_vector_store import (
    RECORD_KINDS,
    StoredRecord,
)

if TYPE_CHECKING:
    from engine.actions import Action
    from engine.content_extractor import ExtractedItem
    from engine.entities import EntitiesOutput
    from engine.ingestor import Document
    from engine.insights import InsightOutput
    from engine.patterns import Pattern


@dataclass(frozen=True)
class EmbeddingSpec:
    """Per-(provider, model_id) record for an embedding model. Parallel
    to llm.py's chat/vision ModelSpec — embedding models have no output-
    token cap, no streaming knob, and no reasoning analogue, so the
    chat-shaped fields don't apply and a parallel record keeps both
    sides honest.

    `context_window` is the per-input token cap the model advertises;
    `embedding_dim` is the vector dimensionality returned per input.
    """
    provider: Provider
    model_id: str
    context_window: int
    embedding_dim: int


# Registered model id for the Tinfoil-hosted nomic embedding. Used at
# every site that looks up the spec or stamps an llm-calls.jsonl event;
# centralising the literal here means a wire-protocol-id correction
# (e.g. the served wire name vs the HF id nomic-ai/nomic-embed-text-v1.5)
# lands in one place rather than threading through stage code + tests.
TINFOIL_NOMIC_MODEL_ID = "nomic-embed-text"


# Nomic Embed Text v1.5 (nomic-ai/nomic-embed-text-v1.5): 8192-token
# input, 768-dim output. Apache-2.0, Matryoshka representation learning
# so the 768-dim output can be truncated to smaller dims downstream if
# needed. The id above is the Tinfoil router's wire-protocol name
# (``nomic-embed-text``, the served-model-name; not the upstream HF id
# ``nomic-ai/nomic-embed-text-v1.5``) so the registry key,
# the auth probe, the proxy.models lookup, and the embeddings.create
# call all resolve to the same enclave.
_TINFOIL_NOMIC_EMBED_TEXT = EmbeddingSpec(
    provider=Provider.TINFOIL,
    model_id=TINFOIL_NOMIC_MODEL_ID,
    context_window=8192,
    embedding_dim=768,
)


# Ollama registry tag for the same nomic-embed-text-v1.5. On Apple
# Silicon Ollama runs the embed model on the Metal GPU by default — the
# GPU-accelerated local runtime the LOCAL stack mandates. The tag string
# is deliberately identical to the Tinfoil wire id: it is the SAME model
# at the same default/standard quant, so (per the settled director
# ruling) the two providers' vectors are cosine-compatible and a shared
# embedding-cache key across providers is correct, not a collision. A
# cache hit performs zero network I/O regardless of provider, so reusing
# a prior run's vector never weakens LOCAL's "the query never leaves the
# machine" posture.
OLLAMA_NOMIC_MODEL_ID = "nomic-embed-text"


_OLLAMA_NOMIC_EMBED_TEXT = EmbeddingSpec(
    provider=Provider.OLLAMA,
    model_id=OLLAMA_NOMIC_MODEL_ID,
    context_window=8192,
    embedding_dim=768,
)


# Per-(provider, model_id) registry for embedding models. Lookup +
# registration discipline mirrors llm._MODEL_SPECS: every embedding
# model the pipeline can call MUST appear here so its quirks
# (input cap, output dim) are respected uniformly.
_EMBEDDING_SPECS: dict[tuple[Provider, str], EmbeddingSpec] = {
    (Provider.TINFOIL, TINFOIL_NOMIC_MODEL_ID): _TINFOIL_NOMIC_EMBED_TEXT,
    (Provider.OLLAMA, OLLAMA_NOMIC_MODEL_ID): _OLLAMA_NOMIC_EMBED_TEXT,
}


# Config `mode` strings → the embedding provider. Only LOCAL forks off
# the attested default; tee (and an unset / unreadable config) keep
# the Tinfoil path. Mirrors runner.py's mode-string map so the two
# never drift.
def _active_embedding_spec() -> EmbeddingSpec:
    """The EmbeddingSpec for the user's current mode.

    The pipeline embeddings stage reads the active provider here (the
    chatbot's query embed now routes by mode through the kernel, picking
    its provider from `embedding_spec_for_mode`). The user's mode is read
    from `config.json` (the single source of truth React persists and the
    sidecar already reads).

    LOCAL → Ollama `nomic-embed-text` (Metal GPU). Anything else, or an
    unreadable config, → the attested Tinfoil path unchanged.
    """
    mode = str(_read_app_config().get("mode") or "").strip().lower()
    if mode == Mode.LOCAL.value:
        return _OLLAMA_NOMIC_EMBED_TEXT
    return _TINFOIL_NOMIC_EMBED_TEXT


def embedding_spec_for(provider: Provider, model_id: str) -> EmbeddingSpec:
    """Look up the EmbeddingSpec for (provider, model_id). Raises
    KeyError on miss — every model used by the pipeline must be
    registered."""
    return _EMBEDDING_SPECS[(provider, model_id)]


def tinfoil_embedding_attest_model_ids() -> list[str]:
    """Tinfoil embedding model ids covered by the attestation chain.
    Sibling of `llm.tinfoil_attest_model_ids()` (chat / vision); the
    two accessors are kept separate because their spec records differ
    (`EmbeddingSpec` has no streaming / output-token / reasoning fields)
    but the union is what both the runner pre-warm and the UI panel
    iterate over.
    """
    return sorted(
        m for (p, m) in _EMBEDDING_SPECS.keys() if p == Provider.TINFOIL
    )


# Tolerance for the unit-norm guard below. The cosine vec0 store ranks
# by `1 - dot(a,b)` and is only well-defined on unit-normalized vectors;
# the embedder ships unit-norm already, but the guard catches a future
# model swap or wire-shape regression before non-unit vectors silently
# degrade the cosine ranking. Tolerance picked wide enough to absorb
# float32-roundtrip noise without admitting genuinely non-unit responses.
_UNIT_NORM_TOLERANCE = 1e-3


def _assert_unit_norm(vectors: list[list[float]]) -> None:
    """Verify every embedding is unit-normalized (L2 norm ≈ 1.0). The
    cosine distance the vec0 store ranks on is meaningful only on
    unit-norm vectors; a non-unit vector silently demotes itself in
    ranking by a factor of its norm. Raise on a first miss so the
    deviation surfaces loud at the wire instead of compounding into
    retrieval quality."""
    for i, v in enumerate(vectors):
        norm_sq = sum(x * x for x in v)
        if abs(norm_sq - 1.0) > _UNIT_NORM_TOLERANCE:
            raise ValueError(
                f"embedding[{i}] is not unit-normalized "
                f"(|v|^2 = {norm_sq:.6f}); cosine ranking requires unit vectors"
            )


# Stage name used both for the runner's `_log_stage` transition and for
# the begin/end events streamed into `llm-calls.jsonl`. Kept as a module
# constant so the runner and the stage driver can never drift.
STAGE_NAME = "embeddings"


# Default top-level batch size — how many records the planner packs
# into one `embed()` call's `input=[...]`. Sized to Tinfoil's nomic-
# router cap (32 inputs per call); larger batches trip the router's
# 413 "batch size N > maximum allowed batch size 32" error. The
# cascade's halve floor is size 1, so a 32-record batch can survive
# up to 5 halve depths before flooring — exactly the spec's Sizing
# cap. If the router cap shifts later, `_maybe_synthesize_context_exceeded`
# maps the 413 onto the synthetic cap-hit shape so the cascade
# auto-halves to whatever the new cap accepts.
DEFAULT_BATCH_SIZE = 32


# Substring matches used to recognise an embedding-side sizing failure
# and route it to the existing synthetic cap-hit shape so the
# classifier maps it to Sizing (halve cascade) — the same path the
# completion side uses for cap-hit. Two failure shapes:
#
#   - **400 "context window exceeded"** — single input is too long
#     for the model's per-input token cap (8192 for nomic). Provider
#     wording varies; the keyword set covers OpenAI canonical
#     phrasings plus the variants Tinfoil's router surfaces.
#   - **413 "batch size N > maximum allowed batch size M"** — the
#     batch carries too many inputs for the router's aggregate cap
#     (32 inputs/call for nomic on Tinfoil today). Halving cuts the
#     records list in two until both halves fit under the cap.
#
# Case-insensitive substring match against `str(exc).lower()`.
_CONTEXT_EXCEEDED_KEYWORDS = (
    "context length",
    "context window",
    "context_length",
    "maximum context",
    "input length",
    "input too long",
    "input is too long",
    "exceeds the maximum",
    "exceed the maximum",
    "token limit",
    "too many tokens",
)
_BATCH_TOO_LARGE_KEYWORDS = (
    "batch size",
    "maximum allowed batch",
    "payload too large",
    "request entity too large",
)




@dataclass(frozen=True)
class EmbeddingsBatchPlan:
    """Result of `build_embeddings_plan`. Records are partitioned into
    fixed-size batches; `num_calls` is the embed() call count the
    progress tracker should register on. `edges` carries the
    derivation/mention adjacency between records as 5-tuples to be
    written to the vector-store edges table after vectors are persisted;
    `dead_end_anchors` is the per-anchor-kind count of records with zero
    outgoing edges, surfaced into the phase marker for #387-class
    corruption observability. `entity_alias_drift` captures the director-
    invariant-1 verification: per-canonical records of category-copies
    whose entity-mention sets disagree, surfaced as warnings at stage
    run.
    """
    records: list[StoredRecord]
    batch_size: int
    edges: list[Edge] = field(default_factory=list)
    dead_end_anchors: dict[str, int] = field(default_factory=dict)
    entity_alias_drift: list[dict] = field(default_factory=list)

    @property
    def num_calls(self) -> int:
        if not self.records:
            return 0
        return (len(self.records) + self.batch_size - 1) // self.batch_size

    def counts_by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {k: 0 for k in RECORD_KINDS}
        for r in self.records:
            out[r.kind] = out.get(r.kind, 0) + 1
        return out


@dataclass(frozen=True)
class EmbeddingsStageResult:
    """Returned by `run_embeddings_stage`. The shape matches the phase
    marker payload one-to-one so the runner can pass it straight
    through to `_dump`.
    """
    calls: int
    counts: dict[str, int]
    model: str
    dim: int
    store_path: str
    batch_size: int = DEFAULT_BATCH_SIZE
    edges_count: int = 0
    dead_end_anchors: dict[str, int] = field(default_factory=dict)
    entity_alias_drift_count: int = 0


def build_embeddings_plan(
    *,
    docs: "list[Document]",
    facts_by_topic: "dict[str, list[ExtractedItem]] | None",
    entities_output: "EntitiesOutput | None",
    patterns_by_topic: "dict[str, list[Pattern]] | None",
    insight_output: "InsightOutput | None",
    action_list: "list[Action] | None",
    extract_calls: "list[dict] | None" = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbeddingsBatchPlan:
    """Walk upstream artifacts and materialize one StoredRecord per
    embed target. Chunks come from the RAG chunker over the ingested
    documents; every other kind is one-record-one-chunk per spec § RAG
    enhancements (no sliding-window split on stage outputs).

    Each record's `text` is the GRAPH-ENRICHED embed input — bare
    record fields plus the upstream + downstream context the prefix-
    builder assembles from a one-pass GraphView of all upstream
    artifacts (per spec § RAG enhancements). `extract_calls` are the
    per-LLM-call records the runner accumulates during extraction;
    they carry split-summaries by splitter chunk id and flow into the
    chunk-kind prefix builder.

    The graph view and the per-kind builders are deterministic given
    identical upstream artifacts, so identical input produces
    identical embed input (cache-stable).
    """
    view = build_graph_view(
        docs=docs,
        facts_by_topic=facts_by_topic,
        entities_output=entities_output,
        patterns_by_topic=patterns_by_topic,
        insight_output=insight_output,
        action_list=action_list,
        extract_calls=extract_calls,
    )

    records: list[StoredRecord] = []

    # Doc-date lookup the per-fact stamp consults. Each doc is keyed
    # by BOTH `file_id` and `source_path` — fact evidence carries
    # `file_path = doc.file_id or doc.source_path` (see
    # `content_extractor.py` ~1067), so a lookup keyed only on
    # `source_path` would miss every fact in the common case where
    # `file_id != source_path` (directory ingests, segmented
    # Day-One bundles).
    doc_date_lookup: dict[str, str] = {}
    for d in view.docs:
        date = d.date or ""
        if not date:
            continue
        if d.file_id:
            doc_date_lookup[d.file_id] = date
        if d.source_path:
            doc_date_lookup.setdefault(d.source_path, date)

    # One document record per ingested source file. ``docs_by_file_id``
    # is keyed by ``file_id`` (segmented sub-docs of one file collapse to
    # a single entry), so this is one record per file the user can name —
    # the record kind that answers "what files do I have?" /
    # "tell me about file X". Emitted before chunks to match RECORD_KINDS
    # order; chunk↔document edges (built in build_edges) link the two.
    for file_id in sorted(view.docs_by_file_id.keys()):
        doc = view.docs_by_file_id[file_id]
        txt = build_document_text(doc, view)
        if not txt.strip():
            continue
        records.append(StoredRecord(
            kind="document",
            record_id=file_id,
            text=txt,
            display_text=build_document_display(doc, view),
            file_id=file_id,
            source_ref=doc.source_path,
            extra={
                "file_date": doc.date or "",
                "chunk_count": len(view.rag_chunks_by_file_id.get(file_id, [])),
            },
        ))

    for ch in view.chunks:
        # `extra` carries the per-record metadata the embeddings stage
        # records but the dispatcher reads later — chunk length for the
        # UI's source-span highlight, file_date for the chatbot's
        # recency ranking when no semantic query is given. Old
        # embeddings without these keys fall back to the dispatcher's
        # secondary ordering (record_id) and the UI's paragraph-length
        # approximation.
        records.append(StoredRecord(
            kind="chunk",
            record_id=f"{ch.file_id}@{ch.char_offset}",
            text=build_chunk_text(ch, view),
            display_text=build_chunk_display(ch),
            file_id=ch.file_id,
            source_ref=ch.source_path,
            section_path=" › ".join(ch.section_path),
            char_offset=ch.char_offset,
            extra={"chunk_len": len(ch.text), "file_date": ch.date},
        ))

    if facts_by_topic:
        # Cross-category fact dedup: a fact in N topics today produces
        # N (topic, idx) keys (same ExtractedItem fanned across topic
        # buckets in runner._facts_by_topic). Emit ONE record per unique
        # fact at the canonical (topic, idx) the graph view picked; the
        # non-canonical aliases are skipped here, while their references
        # in upstream stages (patterns, entities) still resolve through
        # the canonical map. Director invariant 4: no neighbor edge may
        # point at a category-copy of the source fact — structurally
        # upheld by the canonical projection in `build_edges`.
        for topic in sorted(facts_by_topic.keys()):
            for i, fact in enumerate(facts_by_topic[topic]):
                if view.canonical_fact_key(topic, i) != (topic, i):
                    continue  # alias of an earlier canonical; skip.
                txt = build_fact_text(topic, i, fact, view)
                if not txt.strip():
                    continue
                # Pin the fact's recency to the most recent source
                # doc's date. Facts split out of a multi-year journal
                # otherwise have no per-record date for the dispatcher
                # to sort on. Multiple evidence spans → keep the latest
                # ISO string (lexicographic compare works for ISO dates).
                # A fact is file-scoped — its evidence quotes one source
                # file. Stamp that file_id so the `source` filter reaches
                # facts directly (facts from file X), and pin recency to
                # the most recent source doc's date (facts split out of a
                # multi-year journal otherwise have no per-record date for
                # the dispatcher to sort on; latest ISO string wins).
                fact_date = ""
                fact_file_id = ""
                for ev in getattr(fact, "evidence", None) or []:
                    fp = getattr(ev, "file_path", "") or ""
                    if fp and not fact_file_id:
                        fact_file_id = fp
                    d = doc_date_lookup.get(fp, "")
                    if d and d > fact_date:
                        fact_date = d
                records.append(StoredRecord(
                    kind="fact",
                    record_id=f"{topic}:{i}",
                    text=txt,
                    display_text=build_fact_display(fact, view),
                    file_id=fact_file_id,
                    topic=topic,
                    extra={
                        "item_type": getattr(fact, "item_type", ""),
                        "file_date": fact_date,
                    },
                ))

    if entities_output is not None:
        for ent in entities_output.entities:
            txt = build_entity_text(ent, view)
            if not txt.strip():
                continue
            # `mention_count` is the salience signal the dispatcher
            # uses when ordering filter-only entity lookups. Falls back
            # to the evidence-fact-refs length when an upstream entity
            # didn't carry the count explicitly (older fixtures /
            # synthetic test entities).
            mention_count = (
                int(getattr(ent, "mention_count", 0) or 0)
                or len(getattr(ent, "evidence_fact_refs", []) or [])
            )
            records.append(StoredRecord(
                kind="entity",
                record_id=ent.canonical_id,
                text=txt,
                display_text=build_entity_display(ent),
                extra={
                    "entity_type": ent.entity_type,
                    "role": ent.role,
                    "mention_count": mention_count,
                },
            ))

    if patterns_by_topic:
        for topic in sorted(patterns_by_topic.keys()):
            for i, pat in enumerate(patterns_by_topic[topic]):
                txt = build_pattern_text(topic, i, pat, view)
                if not txt.strip():
                    continue
                records.append(StoredRecord(
                    kind="pattern",
                    record_id=f"{topic}:{i}",
                    text=txt,
                    display_text=build_pattern_display(pat),
                    topic=topic,
                    extra={
                        "kind": pat.kind or "",
                        "mention_count": len(
                            getattr(pat, "source_facts", []) or []
                        ),
                    },
                ))

    if insight_output is not None:
        for scope, items in (
            ("cross_domain", insight_output.cross_domain),
            ("critical", insight_output.critical),
        ):
            for i, ins in enumerate(items):
                txt = build_insight_text(scope, i, ins, view)
                if not txt.strip():
                    continue
                records.append(StoredRecord(
                    kind="insight",
                    record_id=f"{scope}:{i}",
                    text=txt,
                    display_text=build_insight_display(ins),
                    extra={"scope": scope, "kind": ins.kind or ""},
                ))

    if action_list:
        for i, act in enumerate(action_list):
            txt = build_action_text(i, act, view)
            if not txt.strip():
                continue
            records.append(StoredRecord(
                kind="action",
                record_id=str(i),
                text=txt,
                display_text=build_action_display(act, view),
                extra={"kind": act.kind or "", "horizon": act.horizon},
            ))

    edges = build_edges(view)
    dead_ends = count_dead_end_anchors(edges, view)

    return EmbeddingsBatchPlan(
        records=records,
        batch_size=batch_size,
        edges=edges,
        dead_end_anchors=dead_ends,
        entity_alias_drift=list(view.entity_alias_drift),
    )

