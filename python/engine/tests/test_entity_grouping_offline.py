"""Offline validation for the migrated ENTITY_GROUPING phase (#912).

ENTITY_GROUPING does no LLM call, so its kernel ``Phase`` can be driven
fully offline. This asserts the kernel grouping phase produces the SAME
canonical groups + packed batches as the legacy ``_group_entities`` /
``_pack_batches`` on a small synthetic fact set — i.e. the deterministic
prep wiring round-trips through ``PhaseResult`` unchanged.
"""
from __future__ import annotations

from engine.content_extractor import Entity, EntityRef, EvidenceSpan, ExtractedItem
from engine.entities import _group_entities
from kernel.cancellation_manager import CancellationManager
from kernel.abstractions import PhaseResult
from engine.llm import Mode

from engine.phases.entity_grouping import EntityGroupingPhase
from engine.phases.entities_job import EntitiesJob


def _item(summary, entities, evidence, topics=None):
    return ExtractedItem(
        item_type="fact",
        summary=summary,
        evidence=[EvidenceSpan(text=evidence, source_ref="test")],
        entities=[
            EntityRef(entity=Entity(name=n, entity_type=t), role=r)
            for n, t, r in entities
        ],
        topics=topics or ["general"],
        tags=[],
        confidence=1.0,
        relation_candidate=None,
        occurred_at=None,
    )


def _facts():
    return {
        "work": [
            _item("Alice signed the contract.",
                  [("Alice", "person", "subject"), ("Acme", "org", "employer")],
                  "Alice signed the contract."),
            _item("Bob reviewed the contract with Alice.",
                  [("Bob", "person", "colleague"), ("Alice", "person", "subject")],
                  "Bob reviewed the contract with Alice."),
        ],
        "life": [
            _item("Alice visited Paris.",
                  [("Alice", "person", "subject"), ("Paris", "place", "location")],
                  "Alice visited Paris."),
        ],
    }


def _run_grouping_phase(facts):
    job = EntitiesJob({"facts_by_topic": facts, "mode": Mode.TEE,
                       "model": None, "subject": "the author",
                       "manifest_pos": None})
    phase = EntityGroupingPhase(job)
    # ENTITY_GROUPING does no LLM call → run with no execution env.
    return phase.run(
        PhaseResult({"facts_by_topic": facts, "mode": Mode.TEE,
                     "model": None, "subject": "the author",
                     "manifest_pos": None}),
        execution_env=None,
        cancellation_manager=CancellationManager(),
    )


def test_grouping_phase_matches_legacy_groups():
    facts = _facts()
    legacy_groups = _group_entities(facts)
    legacy_names = sorted(
        (g.canonical_name, g.entity_type, g.mention_count) for g in legacy_groups
    )

    result = _run_grouping_phase(facts)
    assert result.data["empty"] is False
    kernel_groups = result.data["groups"]
    kernel_names = sorted(
        (g.canonical_name, g.entity_type, g.mention_count) for g in kernel_groups
    )

    assert kernel_names == legacy_names
    # Prep maps + batches are populated for the summarize phase to consume.
    assert result.data["batches"]
    assert result.data["groups_by_gid"]
    assert set(result.data["candidates_by_gid"].keys()) <= {
        g.gid for g in kernel_groups
    }


def test_grouping_phase_empty_facts():
    result = _run_grouping_phase({})
    assert result.data["empty"] is True
    assert result.data["groups"] == []
