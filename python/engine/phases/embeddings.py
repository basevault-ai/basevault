"""Embeddings phase on the kernel (issue #912).

EMBEDDINGS is BATCHED: the kernel batch data model is one call carrying N
texts and one ``LlmResponse`` carrying N vectors
(``LlmResponse.payload: list[list[float]]``), so the phase groups records
into ``DEFAULT_BATCH_SIZE`` chunks, fires one embed call per batch, and maps
each batch's vectors back to its records POSITIONALLY (``results[i]`` ↔
``calls[i]``; the embed API keeps ``vectors[j]`` ↔ ``input[j]``). This is the
~20-32x wire-call reduction over one-call-per-record — the legacy stage's
batching, now on the kernel. The plan build (``build_embeddings_plan`` →
``StoredRecord``s) and the vector-store write stay in the caller; the phase
owns only the batched embed calls + the unit-norm invariant
(``_assert_unit_norm``, reused verbatim).

Provider dispatch is by model: the embedding spec's ``model()`` is
``"nomic-embed-text"``, which both ``TinfoilProvider`` (TEE) and
``OllamaProvider`` (LOCAL) special-case to their embeddings branch (a list of
inputs → a vector per input). MLX has no embeddings path — LOCAL embeds via
Ollama, matching the legacy ``_active_embedding_spec``.

Sizing: each chunk's text is sized to the 8192-token window by the RAG
chunker, so a batch rarely exceeds the endpoint. The kernel does NOT halve
embedding batches (it collects halve-tree leaves in completion order, which
would scramble the positional vector↔record mapping); instead ``run_main``
re-embeds a wrong-sized batch one record at a time — the parity backoff
legacy gets from the batch-too-large path.

Concurrency: batches are independent, so the phase fans them all out via
``execution_env.run_all(calls)`` — the kernel scheduler embeds up to
``max_parallelism`` batches concurrently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import override

from kernel.abstractions import Job, LlmCall, Phase, PhaseResult
from kernel.enums import JobName, LlmStatus, PhaseName
from kernel.execution_env import BoundExecutionEnv, ExecutionEnv

from engine.embeddings import DEFAULT_BATCH_SIZE, _assert_unit_norm


class EmbeddingsPhase(Phase):
    def __init__(self, job: Job):
        super().__init__(job)
        self._records: list = []

    @override
    def name(self) -> PhaseName:
        return PhaseName.EMBEDDINGS

    @override
    def init_from_memory(self, input: PhaseResult) -> bool:
        self._state = dict(input.data)
        records = input.data.get("records")
        if records is None:
            # Pipeline path: no pre-built records — derive the embeddings
            # plan from the accumulated upstream artifacts (the runner's
            # build_embeddings_plan), reused verbatim.
            from engine.embeddings import build_embeddings_plan

            plan = build_embeddings_plan(
                docs=input.data.get("all_docs", []),
                facts_by_topic=input.data.get("facts_by_topic"),
                entities_output=input.data.get("entities_output"),
                patterns_by_topic=input.data.get("patterns_by_topic"),
                insight_output=input.data.get("insight_output"),
                action_list=input.data.get("action_list"),
                extract_calls=input.data.get("extract_calls"),
            )
            records = list(plan.records)
        self._records = list(records)
        return True

    @override
    def init_from_disk(self) -> None:
        raise NotImplementedError("EmbeddingsPhase always inits from memory")

    @override
    def validate(self, payload) -> LlmStatus | None:
        # A well-formed batch embedding is a non-empty list of non-empty
        # vectors (payload: list[list[float]] — one vector per input text).
        # Empty → LOAD is handled upstream (``compute_status`` returns LOAD when
        # payload is falsy); a malformed shape is a wire error.
        if (
            not isinstance(payload, list)
            or not payload
            or not all(isinstance(v, list) and v for v in payload)
        ):
            return LlmStatus.OTHER
        return None

    @override
    def halve_llm_call(self, call: LlmCall) -> list[LlmCall]:
        # Embeddings batches are NOT halved by the kernel: the kernel collects
        # halve-tree leaf responses in COMPLETION order (execution_env's
        # maybe_combine_futures extends new_responses as each child resolves),
        # which would scramble the positional vector[i] <-> record[i] mapping a
        # batch relies on. run_main owns sizing instead — a wrong-sized batch is
        # re-embedded one record at a time (_embed_in_batches), order preserved.
        return []

    @override
    def sample_llm_call(self, call: LlmCall) -> LlmCall | None:
        raise NotImplementedError("embeddings is non-degrading; it never samples")

    @override
    def checkpoint(self) -> None:
        pass

    @override
    def run_main(self, execution_env: BoundExecutionEnv | None) -> PhaseResult:
        assert execution_env is not None
        records = [r for r in self._records if (getattr(r, "text", "") or "")]
        pairs = self._embed_in_batches(
            records, DEFAULT_BATCH_SIZE, execution_env
        )
        return PhaseResult({**self._state, "pairs": pairs, "embedding_pairs": pairs})

    def _batch_call(self, batch: list) -> LlmCall:
        # One call per batch: N user messages, one per record's text. The
        # provider embeds them in a single wire call and returns N vectors
        # (payload: list[list[float]]); context carries the batch for backoff.
        return self.new_call(
            [{"role": "user", "content": r.text} for r in batch],
            0, None, context=batch,
        )

    def _embed_in_batches(
        self, records: list, batch_size: int,
        execution_env: BoundExecutionEnv,
    ) -> list[tuple]:
        # Batch the records (DEFAULT_BATCH_SIZE per call, ~20-32x fewer wire
        # calls than one-per-record), fire one call per batch, and map each
        # batch's N vectors back to its N records POSITIONALLY: run_all keeps
        # results[i] <-> calls[i], and the embed API keeps vectors[j] <->
        # input[j]. A batch that returns wrong-sized / empty is re-embedded one
        # record at a time so a single oversize batch never drops the whole
        # group (the parity backoff legacy gets from the batch-too-large path).
        if not records:
            return []
        batches = [
            records[i:i + batch_size]
            for i in range(0, len(records), batch_size)
        ]
        results = execution_env.run_all([self._batch_call(b) for b in batches])

        pairs: list[tuple] = []
        for batch, responses in zip(batches, results):
            vectors = self._vectors_from(responses)
            if vectors is not None and len(vectors) == len(batch):
                _assert_unit_norm(vectors)  # fail-loud unit-norm invariant (#784)
                pairs.extend(zip(batch, vectors))
            elif len(batch) == 1:
                continue  # a single text already failed; nothing to reduce
            else:
                pairs.extend(self._embed_in_batches(batch, 1, execution_env))
        return pairs

    @staticmethod
    def _vectors_from(responses: list) -> list | None:
        # A batch's vectors = the last response carrying a well-formed
        # list-of-vectors payload (a later retry supersedes an earlier failure).
        vectors = None
        for response in responses:
            p = response.payload
            if isinstance(p, list) and p and all(
                isinstance(v, list) and v for v in p
            ):
                vectors = p
        return vectors


class EmbeddingsJob(Job):
    def __init__(self, records: list):
        super().__init__()
        self._records = records

    @override
    def name(self) -> JobName:
        return JobName.EMBEDDING_ONLY

    @override
    def generate_phases(self) -> list[Phase]:
        return [EmbeddingsPhase(self)]

    def initial_input(self) -> PhaseResult:
        return PhaseResult({"records": self._records})


def run_embeddings_stage(
    plan,
    store_path,
    mode,
    execution_env: ExecutionEnv | None = None,
):
    """Kernel analogue of ``embeddings.run_embeddings_stage``: embed the
    plan's records on the kernel, then run the SAME store-write + edge
    filtering + result shaping VERBATIM. Returns an ``EmbeddingsStageResult``.

    The plan build (``build_embeddings_plan``) stays in the caller. Note: the
    kernel embed path does not use the per-record embedding cache yet (every
    record is embedded fresh) — correctness-fine, a perf follow-up.
    """
    from engine.embeddings import (
        EmbeddingsStageResult,
        _active_embedding_spec,
    )
    from engine.rag_vector_store import VectorStore

    spec = _active_embedding_spec()
    counts = plan.counts_by_kind()

    if not plan.records:
        with VectorStore(store_path, dim=spec.embedding_dim):
            pass
        return EmbeddingsStageResult(
            calls=0, counts=counts, model=spec.model_id,
            dim=spec.embedding_dim, store_path=str(store_path),
            batch_size=plan.batch_size, edges_count=0,
            dead_end_anchors=dict(plan.dead_end_anchors),
            entity_alias_drift_count=len(plan.entity_alias_drift),
        )

    pairs = embed_records(
        list(plan.records), mode, execution_env=execution_env
    )

    edges_inserted = 0
    persisted: set[tuple[str, str]] = set()
    with VectorStore(store_path, dim=spec.embedding_dim) as store:
        if pairs:
            records_in = [r for r, _ in pairs]
            vectors_in = [v for _, v in pairs]
            store.add(records_in, vectors_in)
            for r in records_in:
                persisted.add((r.kind, r.record_id))
        live_edges = [
            e for e in plan.edges
            if (e[0], e[1]) in persisted and (e[2], e[3]) in persisted
        ]
        edges_inserted = store.add_edges(live_edges)

    return EmbeddingsStageResult(
        # `calls` is the number of batched WIRE calls (ceil(records/batch_size)),
        # NOT len(pairs) (the per-record vector count) — batching means one call
        # covers up to batch_size records, matching the legacy stage's metric.
        calls=plan.num_calls, counts=counts, edges_count=edges_inserted,
        dead_end_anchors=dict(plan.dead_end_anchors),
        entity_alias_drift_count=len(plan.entity_alias_drift),
        model=spec.model_id, dim=spec.embedding_dim,
        store_path=str(store_path), batch_size=plan.batch_size,
    )


def embed_records(
    records: list,
    mode,
    execution_env: ExecutionEnv | None = None,
) -> list[tuple]:
    """Embed each ``StoredRecord`` on the kernel; return ``(record, vector)``
    pairs. The caller builds the plan (``build_embeddings_plan``) and writes
    the vectors to the store."""
    from engine.phases.model_specs import embedding_spec_for_mode

    job = EmbeddingsJob(records)
    if execution_env is None:
        spec = embedding_spec_for_mode(mode)
        env = ExecutionEnv()
        env.register_spec(PhaseName.EMBEDDINGS, spec, spec, False)
    else:
        env = execution_env
    result = job.run(job.initial_input(), env)
    return result.data["pairs"]


@dataclass(frozen=True)
class _QueryText:
    """Minimal record-shape wrapper so a raw query string can ride the
    batched ``EmbeddingsPhase`` path, which reads ``.text`` off each record."""
    text: str


def embed_texts(
    texts: list[str],
    mode,
    execution_env: ExecutionEnv | None = None,
) -> list[list[float]]:
    """Embed raw strings (the chatbot's dense-KNN queries) on the kernel and
    return their vectors in input order.

    Same model (``nomic-embed-text``) and provider routing as the pipeline
    embeddings stage — TEE → Tinfoil, LOCAL → Ollama — via the shared
    ``EmbeddingsPhase``, so retrieval rides the exact kernel path inference
    runs on instead of a parallel engine-side direct-provider embed call.
    Empty input short-circuits to ``[]``. Fails loud (``KeyError``) if the
    kernel drops a vector rather than silently misaligning the caller's
    positional query↔vector mapping."""
    if not texts:
        return []
    pairs = embed_records([_QueryText(t) for t in texts], mode,
                          execution_env=execution_env)
    by_text = {r.text: v for r, v in pairs}
    return [by_text[t] for t in texts]
