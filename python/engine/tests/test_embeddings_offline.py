"""Offline validation for the migrated EMBEDDINGS phase (#912).

EMBEDDINGS is BATCHED — records are grouped into DEFAULT_BATCH_SIZE chunks,
one kernel call per batch carrying N texts, one LlmResponse carrying N
vectors. This drives the migrated phase with a scripted provider that returns
a distinct unit vector per input text and asserts: batching (one call per
batch, not per record), POSITIONAL mapping (vector[i] <-> record[i], no
swaps), empty-text records skipped, the unit-norm invariant enforced, and the
wrong-sized-batch backoff (re-embed one record at a time).
"""
from __future__ import annotations

import math

import pytest

from kernel.abstractions import InferenceProvider, LlmResponse
from kernel.enums import PhaseName
from kernel.execution_env import ExecutionEnv

from engine.embeddings import DEFAULT_BATCH_SIZE
from engine.llm import ModelSpec as LegacyModelSpec
from engine.phases.embeddings import EmbeddingsJob
from engine.phases.model_specs import PipelineModelSpec
from engine.rag_vector_store import StoredRecord


def _unit_for(text: str) -> list[float]:
    """A distinct deterministic UNIT vector per text — lets the test detect a
    record<->vector swap (identical vectors would hide ordering bugs)."""
    angle = (sum(ord(c) for c in text) % 360) * math.pi / 180.0
    return [math.cos(angle), math.sin(angle)]  # L2 norm == 1.0


class _Scripted(InferenceProvider):
    """Returns one vector per input message (the batch contract): payload is a
    list[list[float]] of len(call.messages). ``fail_multi`` makes a >1-message
    batch come back wrong-sized so the phase's backoff path fires."""

    def __init__(self, vecfn=_unit_for, fail_multi: bool = False):
        self._vecfn = vecfn
        self._fail_multi = fail_multi
        self.call_count = 0
        self.batch_sizes: list[int] = []
        self.seen_models: list[str] = []
        self._inj = {p: [] for p in PhaseName}

    def name(self):
        return "scripted-embed"

    def run(self, call, execution_env) -> LlmResponse:
        self.call_count += 1
        self.batch_sizes.append(len(call.messages))
        self.seen_models.append(execution_env.model_spec.model())
        if self._fail_multi and len(call.messages) > 1:
            return LlmResponse(None, [], None, 0, 0, 0, 0.0, 0.0)  # wrong-sized
        payload = [self._vecfn(m["content"]) for m in call.messages]
        return LlmResponse(None, payload, None, 0, 0, 0, 0.0, 0.0)

    def inject_errors(self, phase, errors):
        self._inj[phase] += errors


def _run(records, **kw):
    provider = _Scripted(**kw)
    legacy = LegacyModelSpec(
        provider="scripted", model_id="nomic-embed-text", context_window=8192
    )
    spec = PipelineModelSpec(
        provider, legacy.model_id, legacy.context_window, max_parallelism=4
    )
    env = ExecutionEnv()
    env.register_spec(PhaseName.EMBEDDINGS, spec, spec, thinking=False)
    job = EmbeddingsJob(records)
    out = job.run(job.initial_input(), env)
    return out.data["pairs"], provider


def _rec(rid, text):
    return StoredRecord(kind="chunk", record_id=rid, text=text)


def test_embeddings_batches_into_one_call():
    # Fewer than DEFAULT_BATCH_SIZE records => a single batched wire call,
    # not one-per-record. Pairs come back in record order with each record's
    # own vector (positional mapping).
    records = [_rec("a", "alpha text"), _rec("b", "beta text"), _rec("c", "gamma")]
    pairs, provider = _run(records)
    assert provider.call_count == 1
    assert provider.batch_sizes == [3]
    assert [r.record_id for r, _v in pairs] == ["a", "b", "c"]
    assert [v for _r, v in pairs] == [_unit_for(r.text) for r in records]
    assert provider.seen_models == ["nomic-embed-text"]  # one call, embed model


def test_embeddings_positional_mapping_no_swap():
    # Distinct per-text vectors => a record must carry ITS text's vector.
    records = [_rec(str(i), f"text-{i}") for i in range(5)]
    pairs, _ = _run(records)
    for r, v in pairs:
        assert v == _unit_for(r.text)


def test_embeddings_multiple_batches():
    # More than DEFAULT_BATCH_SIZE records => ceil(n / batch_size) wire calls.
    n = DEFAULT_BATCH_SIZE * 2 + 3
    records = [_rec(str(i), f"text-{i}") for i in range(n)]
    pairs, provider = _run(records)
    expected_calls = (n + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE
    assert provider.call_count == expected_calls
    assert len(pairs) == n
    assert [r.record_id for r, _v in pairs] == [str(i) for i in range(n)]


def test_embeddings_skips_empty_text():
    records = [_rec("a", "alpha"), _rec("empty", ""), _rec("c", "gamma")]
    pairs, provider = _run(records)
    assert provider.batch_sizes == [2]  # empty record never reaches a call
    assert [r.record_id for r, _v in pairs] == ["a", "c"]


def test_embeddings_wrong_sized_batch_backs_off_per_record():
    # A batch that returns the wrong vector count is re-embedded one record at
    # a time (parity with the legacy batch-too-large backoff). No record drops.
    records = [_rec("a", "alpha"), _rec("b", "beta"), _rec("c", "gamma")]
    pairs, provider = _run(records, fail_multi=True)
    assert [r.record_id for r, _v in pairs] == ["a", "b", "c"]
    assert [v for _r, v in pairs] == [_unit_for(r.text) for r in records]
    # The doomed multi-record batch (after the kernel's own transient-failure
    # retries of it) falls back to one call per record — every record still
    # embedded, in order. Assert the outcome, not the policy-dependent count of
    # batch retries that precede the backoff.
    assert provider.batch_sizes.count(1) == 3  # one single-record call each


def test_embeddings_unit_norm_enforced():
    # A non-unit vector must fail loud (#784 invariant).
    records = [_rec("a", "alpha")]
    with pytest.raises(ValueError, match="unit-norm"):
        _run(records, vecfn=lambda _t: [0.5, 0.5, 0.5])


# ── embed_texts: the chatbot query-embed seam, on the kernel ────────────────


def _scripted_embed_env(**kw):
    """A kernel ExecutionEnv backed by the scripted embed provider, for
    driving ``embed_texts`` without a live endpoint."""
    provider = _Scripted(**kw)
    spec = PipelineModelSpec(provider, "nomic-embed-text", 8192, max_parallelism=4)
    env = ExecutionEnv()
    env.register_spec(PhaseName.EMBEDDINGS, spec, spec, thinking=False)
    return env, provider


def test_embed_texts_returns_vectors_in_input_order():
    # The chatbot dispatch contract: vectors[i] is the embedding of texts[i].
    from engine.phases.embeddings import embed_texts

    env, provider = _scripted_embed_env()
    texts = ["who is alice", "show me beta", "gamma notes"]
    vectors = embed_texts(texts, mode=None, execution_env=env)
    assert vectors == [_unit_for(t) for t in texts]
    assert provider.call_count == 1  # one batched wire call for the queries


def test_embed_texts_empty_input_short_circuits():
    from engine.phases.embeddings import embed_texts

    env, provider = _scripted_embed_env()
    assert embed_texts([], mode=None, execution_env=env) == []
    assert provider.call_count == 0  # no wire call for an empty query set
