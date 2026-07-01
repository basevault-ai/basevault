"""Offline end-to-end for detect_entities (#912).

The runner's BASEVAULT_KERNEL_ENTITIES path calls
``detect_entities`` — the full three-phase entities job (grouping →
summarize → dedupe) through ONE env. This drives it with a phase-aware
scripted provider and asserts the entry runs end-to-end and returns an
EntitiesOutput with the canonical entities resolved from the facts (records
materialize from the deterministic groups even when the summarize
annotations are empty).
"""
from __future__ import annotations

from engine.content_extractor import Entity, EntityRef, EvidenceSpan, ExtractedItem
from kernel.abstractions import InferenceProvider, LlmResponse
from kernel.enums import PhaseName
from kernel.execution_env import ExecutionEnv

from engine.llm import ModelSpec as LegacyModelSpec, Mode
from engine.phases.entities_job import detect_entities
from engine.phases.model_specs import PipelineModelSpec


def _item(summary, entities, evidence):
    return ExtractedItem(
        item_type="fact",
        summary=summary,
        evidence=[EvidenceSpan(text=evidence, source_ref="t")],
        entities=[
            EntityRef(entity=Entity(name=n, entity_type=t), role=r)
            for n, t, r in entities
        ],
        topics=["work"],
        tags=[],
        confidence=1.0,
        relation_candidate=None,
        occurred_at=None,
    )


class _PhaseScripted(InferenceProvider):
    def __init__(self):
        self.calls = {}
        self._inj = {p: [] for p in PhaseName}

    def name(self):
        return "scripted-entities"

    def run(self, call, execution_env) -> LlmResponse:
        phase = execution_env.phase.name()
        self.calls[phase] = self.calls.get(phase, 0) + 1
        if phase == PhaseName.ENTITY_DEDUPE:
            return LlmResponse(None, '{"merges": []}', None, 0, 0, 0, 0.0, 0.0)
        # ENTITY_SUMMARIZE: valid empty annotations → records still come from
        # the deterministic groups.
        return LlmResponse(None, '{"entities": []}', None, 0, 0, 0, 0.0, 0.0)

    def inject_errors(self, phase, errors):
        self._inj[phase] += errors


def test_detect_entities_end_to_end():
    facts = {
        "work": [
            _item("Alice signed the Acme contract.",
                  [("Alice", "person", "subject"), ("Acme", "org", "employer")],
                  "Alice signed the Acme contract."),
            _item("Bob joined Acme.",
                  [("Bob", "person", "colleague"), ("Acme", "org", "employer")],
                  "Bob joined Acme."),
        ],
    }
    provider = _PhaseScripted()
    legacy = LegacyModelSpec(
        provider="scripted", model_id="gpt-oss-120b", context_window=131_000
    )
    spec = PipelineModelSpec(provider, legacy.model_id, legacy.context_window, max_parallelism=4)
    env = ExecutionEnv()
    env.register_spec(PhaseName.ENTITY_SUMMARIZE, spec, spec, thinking=False)
    env.register_spec(PhaseName.ENTITY_DEDUPE, spec, spec, thinking=False)

    out = detect_entities(facts, Mode.TEE, execution_env=env)

    # Full entry ran: summarize + dedupe fired; canonical entities resolved.
    assert provider.calls.get(PhaseName.ENTITY_SUMMARIZE, 0) >= 1
    assert provider.calls.get(PhaseName.ENTITY_DEDUPE, 0) >= 1
    names = {e.canonical_name for e in out.entities}
    assert "Acme" in names  # the org grouped across both facts
    assert any(n in names for n in ("Alice", "Bob"))
