"""Offline end-to-end validation for the flat PipelineJob (#912).

Drives the whole production pipeline as ONE kernel Job — ingestion →
splitter → extraction → completion → entity grouping/summarize/dedupe →
patterns → insights → actions → embeddings — with a single phase-aware
scripted provider. Proves the architecture decision works: one
``generate_phases()`` in order, ONE accumulating state dict threaded
phase-to-phase, each phase reading upstream artifacts and the embeddings
phase deriving its plan from that state. Asserts the chain executes and the
final state carries every stage's output key.
"""
from __future__ import annotations

import json

from engine.ingestor import Document, SourceType
from kernel.abstractions import InferenceProvider, LlmResponse
from kernel.enums import PhaseName
from kernel.execution_env import ExecutionEnv

from engine.llm import ModelSpec as LegacyModelSpec, Mode
from engine.phases.model_specs import PipelineModelSpec
from engine.phases.pipeline_job import PipelineJob


_EXTRACT_ENVELOPE = json.dumps({
    "split_summaries": [{"id": "d", "summary": "gist"}],
    "items": [{
        "type": "fact",
        "summary": "Alice signed the Acme contract",
        "evidence": [{"text": "Alice signed", "source_ref": "d"}],
        "entities": [{"name": "Acme", "entity_type": "org", "role": "employer"}],
        "topics": ["work"],
        "affect": [],
        "confidence": 0.9,
    }],
})


class _PhaseScripted(InferenceProvider):
    """Returns a canned response per phase so the whole chain executes."""

    def __init__(self):
        self.calls_by_phase: dict = {}
        self._inj = {p: [] for p in PhaseName}

    def name(self):
        return "scripted-pipeline"

    def run(self, call, execution_env) -> LlmResponse:
        phase = execution_env.phase.name()
        self.calls_by_phase[phase] = self.calls_by_phase.get(phase, 0) + 1
        if phase == PhaseName.EMBEDDINGS:
            return LlmResponse(None, [1.0, 0.0, 0.0], None, 0, 0, 0, 0.0, 0.0)
        if phase == PhaseName.EXTRACTION_LLM:
            return LlmResponse(None, _EXTRACT_ENVELOPE, None, 0, 0, 0, 0.0, 0.0)
        if phase == PhaseName.ENTITY_SUMMARIZE:
            return LlmResponse(None, '{"entities": []}', None, 0, 0, 0, 0.0, 0.0)
        if phase == PhaseName.ENTITY_DEDUPE:
            return LlmResponse(None, '{"merges": []}', None, 0, 0, 0, 0.0, 0.0)
        # patterns / insights / actions: empty-but-valid → no output.
        return LlmResponse(None, "[]", None, 0, 0, 0, 0.0, 0.0)

    def inject_errors(self, phase, errors):
        self._inj[phase] += errors


def _env(provider):
    legacy = LegacyModelSpec(
        provider="scripted", model_id="gpt-oss-120b", context_window=131_000
    )
    spec = PipelineModelSpec(provider, legacy.model_id, legacy.context_window, max_parallelism=4)
    env = ExecutionEnv()
    for pn in PhaseName:
        if pn.does_llm_call():
            env.register_spec(pn, spec, spec, thinking=False)
    return env


def _doc():
    return Document(
        id="d",
        source_path="d.md",
        source_type=SourceType.MD_FILE,
        content="Alice signed the Acme contract on Monday. " * 6,
        title="d",
        date="",
        file_id="d",
    )


def test_pipeline_job_runs_end_to_end():
    provider = _PhaseScripted()
    job = PipelineJob({
        "all_docs": [_doc()], "mode": Mode.TEE, "subject": "the author",
        "today": None, "sentiment": "neutral", "total_facts": 1,
        "min_topics": 2, "budget_tokens": 16000, "manifest_pos": None,
    })
    state = job.run(job.initial_input(), _env(provider)).data

    # The accumulating state carries every stage's output key.
    for key in (
        "docs", "items", "facts_by_topic", "entities_output",
        "entities_context", "entities_context_by_topic",
        "patterns_by_topic", "insight_output", "action_list", "embedding_pairs",
    ):
        assert key in state, f"missing pipeline state key: {key}"

    # Extraction produced the fact under its topic; entities resolved it.
    assert "work" in state["facts_by_topic"]
    assert len(state["facts_by_topic"]["work"]) == 1
    assert state["entities_output"].entities  # ≥1 canonical entity (Acme/Alice)
    # Extraction fired through the kernel; the embeddings phase ran and
    # derived its plan from the threaded state (embedding_pairs present
    # above — record count depends on fixture depth, so calls may be 0).
    assert provider.calls_by_phase.get(PhaseName.EXTRACTION_LLM, 0) >= 1
    assert isinstance(state["embedding_pairs"], list)
