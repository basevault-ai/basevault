"""Unit tests for the Embeddings call surface + pipeline stage driver."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from engine import embeddings as _emb_mod
from engine.actions import Action
from engine.content_extractor import Entity, EntityRef, EvidenceSpan, ExtractedItem
from engine.embeddings import DEFAULT_BATCH_SIZE, STAGE_NAME, build_embeddings_plan, embedding_spec_for, tinfoil_embedding_attest_model_ids
from engine.entities import EntitiesOutput, EntityRecord
from engine.ingestor import Document, SourceType
from engine.insights import Insight, InsightOutput
from engine.llm import Provider
from engine.patterns import Pattern


@pytest.fixture(autouse=True)
def _pin_embed_dispatch_to_tinfoil(monkeypatch):
    """Default `_read_app_config` to `{}` for every embeddings test, so
    `_active_embedding_spec` resolves to the Tinfoil branch deterministically
    instead of inheriting whichever `mode` the developer's real config
    happens to carry. Tests that exercise the LOCAL fork explicitly
    re-patch `_read_app_config` inside their own `with` block, overriding
    this default."""
    monkeypatch.setattr(_emb_mod, "_read_app_config", lambda: {})


# ── Embedding registry + active-spec resolution ──────────────────────────


def test_registry_contains_tinfoil_nomic():
    spec = embedding_spec_for(Provider.TINFOIL, "nomic-embed-text")
    assert spec.provider == Provider.TINFOIL
    assert spec.model_id == "nomic-embed-text"
    assert spec.context_window > 0
    assert spec.embedding_dim > 0


def test_unregistered_lookup_raises_keyerror():
    with pytest.raises(KeyError):
        embedding_spec_for(Provider.TINFOIL, "no-such-model")


def test_tinfoil_attest_ids_lists_nomic_and_is_sorted():
    ids = tinfoil_embedding_attest_model_ids()
    assert "nomic-embed-text" in ids
    assert ids == sorted(ids)


def test_registry_contains_ollama_nomic():
    spec = embedding_spec_for(Provider.OLLAMA, "nomic-embed-text")
    assert spec.provider == Provider.OLLAMA
    assert spec.model_id == "nomic-embed-text"
    assert spec.embedding_dim == 768
    assert spec.context_window == 8192


def test_active_spec_local_picks_ollama():
    with patch.object(_emb_mod, "_read_app_config", return_value={"mode": "local"}):
        spec = _emb_mod._active_embedding_spec()
    assert spec.provider == Provider.OLLAMA


@pytest.mark.parametrize("cfg", [{}, {"mode": "tee"}, {"mode": "tee"},
                                 {"mode": "TEST"}, {"mode": None}])
def test_active_spec_non_local_keeps_tinfoil(cfg):
    with patch.object(_emb_mod, "_read_app_config", return_value=cfg):
        spec = _emb_mod._active_embedding_spec()
    assert spec.provider == Provider.TINFOIL


def test_attest_ids_exclude_ollama():
    """The new Ollama entry must NOT leak into the Tinfoil attestation
    list — Ollama has no attestation surface."""
    ids = tinfoil_embedding_attest_model_ids()
    assert ids == ["nomic-embed-text"]  # only the Tinfoil one


# ── Stage name pin ───────────────────────────────────────────────────────


def test_stage_name_matches_marker_path():
    """The stage name string is the single source of truth for the
    `stage="..."` tag on llm-calls.jsonl events AND for the runner's
    `_log_stage` transition; pin it explicitly so a rename surfaces
    here before the runner-side wiring drifts."""
    assert STAGE_NAME == "embeddings"


# ── Plan helpers ─────────────────────────────────────────────────────────


def _doc(content: str, file_id: str = "f1") -> Document:
    return Document(
        id=file_id, source_path=f"/tmp/{file_id}.md",
        source_type=SourceType.MD_FILE, content=content, file_id=file_id,
    )


def _fact(summary: str, topic_hint: str | None = None) -> ExtractedItem:
    return ExtractedItem(
        item_type="event",
        summary=summary,
        evidence=[],
        occurred_at="2025-01-01",
        topics=[topic_hint] if topic_hint else [],
        tags=["t1"],
        confidence=0.9,
    )


def _entity(rec_id: str, name: str) -> EntityRecord:
    return EntityRecord(
        canonical_id=rec_id, canonical_name=name,
        entity_type="person", aliases=["nick"], role="friend",
        description="brief desc", mention_count=3,
    )


def _pattern(name: str, domain: str = "health") -> Pattern:
    return Pattern(
        name=name, description="pat desc", domain=domain,
        kind="behavior", count=4,
    )


def _insight(name: str) -> Insight:
    return Insight(
        name=name, description="ins desc",
        mechanism="why it happens", implication="what to do",
        domains=["health"], kind="cross_domain",
    )


def _action(rec: str) -> Action:
    return Action(
        recommendation=rec, objective="obj", why="bcs",
        immediate_action="do x", habit="weekly",
        success_metric="metric", horizon="short",
        review_date="2025-02-01",
    )


def _plan_facts_only(n: int, batch_size: int = DEFAULT_BATCH_SIZE):
    """Build a plan with `n` fact records and nothing else."""
    return build_embeddings_plan(
        docs=[],
        facts_by_topic={"x": [_fact(f"s{i}", "x") for i in range(n)]},
        entities_output=None, patterns_by_topic=None,
        insight_output=None, action_list=None,
        batch_size=batch_size,
    )


# ── build_embeddings_plan ────────────────────────────────────────────────


def test_plan_collects_one_record_per_kind():
    plan = build_embeddings_plan(
        docs=[_doc("Short body.")],
        facts_by_topic={"health": [_fact("ate well", "health")]},
        entities_output=EntitiesOutput(
            entities=[_entity("e1", "Alice")], relations=[],
        ),
        patterns_by_topic={"health": [_pattern("p1")]},
        insight_output=InsightOutput(
            cross_domain=[_insight("i1")],
            critical=[_insight("i2")],
        ),
        action_list=[_action("act now")],
    )
    counts = plan.counts_by_kind()
    assert counts["chunk"] == 1
    assert counts["fact"] == 1
    assert counts["entity"] == 1
    assert counts["pattern"] == 1
    assert counts["insight"] == 2
    assert counts["action"] == 1


def test_plan_handles_missing_upstream_artifacts():
    plan = build_embeddings_plan(
        docs=[],
        facts_by_topic=None,
        entities_output=None,
        patterns_by_topic=None,
        insight_output=None,
        action_list=None,
    )
    assert plan.records == []
    assert plan.num_calls == 0


def test_plan_num_calls_ceils_records_over_batch_size():
    plan = build_embeddings_plan(
        docs=[_doc("body")],
        facts_by_topic={"x": [_fact(f"s{i}", "x") for i in range(140)]},
        entities_output=None,
        patterns_by_topic=None,
        insight_output=None,
        action_list=None,
        batch_size=64,
    )
    # 1 document + 1 chunk + 140 facts = 142 records → ceil(142/64) = 3 calls.
    assert len(plan.records) == 142
    assert plan.num_calls == 3


def test_plan_fact_text_includes_topics_and_date():
    plan = build_embeddings_plan(
        docs=[], facts_by_topic={"work": [_fact("shipped MVP", "work")]},
        entities_output=None, patterns_by_topic=None,
        insight_output=None, action_list=None,
    )
    fact_rec = plan.records[0]
    assert "shipped MVP" in fact_rec.text
    assert "work" in fact_rec.text
    assert "2025-01-01" in fact_rec.text
    assert fact_rec.topic == "work"


def test_plan_fact_text_is_deterministic_across_calls():
    """Embed input must be a pure function of the record fields so
    identical upstream produces identical embed input — load-bearing
    for cache stability + retrieval reproducibility. Two independent
    calls with the same input must produce byte-identical record
    texts."""
    facts = {"work": [_fact("shipped MVP", "work")]}
    plan_a = build_embeddings_plan(
        docs=[_doc("Short body.")], facts_by_topic=facts,
        entities_output=EntitiesOutput(
            entities=[_entity("e1", "Alice")], relations=[]),
        patterns_by_topic=None,
        insight_output=None,
        action_list=[_action("act now")],
    )
    plan_b = build_embeddings_plan(
        docs=[_doc("Short body.")], facts_by_topic=facts,
        entities_output=EntitiesOutput(
            entities=[_entity("e1", "Alice")], relations=[]),
        patterns_by_topic=None,
        insight_output=None,
        action_list=[_action("act now")],
    )
    a_texts = [r.text for r in plan_a.records]
    b_texts = [r.text for r in plan_b.records]
    assert a_texts == b_texts


# ── run_embeddings_stage — real driver ──────────────────────────────────


















def test_default_batch_size_fits_under_tinfoil_nomic_cap():
    """The router's batch cap (32 inputs/call) is enforced by the
    provider; running over it trips a 413 on every full batch.
    DEFAULT_BATCH_SIZE must stay at or below 32 so the no-failure
    path doesn't have to lean on the Sizing halve for routine work."""
    # Tinfoil's nomic-embed-text router cap, observed from live run
    # #gatm's 413 message body. If Tinfoil raises this cap and we
    # want larger default batches, bump the assertion's RHS in lock
    # step with the constant.
    NOMIC_TINFOIL_BATCH_CAP = 32
    assert DEFAULT_BATCH_SIZE <= NOMIC_TINFOIL_BATCH_CAP




def test_default_batch_size_is_publicly_exposed():
    assert DEFAULT_BATCH_SIZE > 0








# ── Embedding cache integration ─────────────────────────────────────────────








# ── Edge persistence (slice-2 stage 2 — #785) ────────────────────────────


def _wired_fact(summary: str, topic: str, source_path: str,
                file_offset: int) -> ExtractedItem:
    """Fact with the minimum evidence wiring `build_graph_view` needs
    to derive an entity↔fact mention edge AND a chunk↔fact containment
    edge from a real document."""
    return ExtractedItem(
        item_type="event", summary=summary,
        evidence=[EvidenceSpan(
            text=summary, source_ref="splitX",
            file_path=source_path, file_offset=file_offset,
            file_length=len(summary),
        )],
        occurred_at="2025-01-01", topics=[topic], tags=[],
        confidence=0.9,
        entities=[EntityRef(
            entity=Entity(name="Alice", entity_type="person"),
            role="subject",
        )],
    )


def _wired_entity(ent_id: str, fact_refs):
    return EntityRecord(
        canonical_id=ent_id, canonical_name="Alice",
        entity_type="person", aliases=[], role="subject",
        description="", mention_count=len(fact_refs),
        evidence_fact_refs=list(fact_refs),
    )


def _wired_fixture():
    doc = _doc("Alice ran 5k and slept 8 hours.")
    facts = [
        _wired_fact("ran 5k", "health", doc.source_path, file_offset=6),
        _wired_fact("slept 8 hours", "health", doc.source_path, file_offset=18),
    ]
    entities = EntitiesOutput(
        entities=[_wired_entity("alice", [("health", 0), ("health", 1)])],
        relations=[],
    )
    return doc, facts, entities








# ── Cross-category fact dedup (#801) ──────────────────────────────────────────


def _xcat_fact(summary: str, topics: list[str], source_path: str,
               file_offset: int) -> ExtractedItem:
    """Fact carrying N topics — the runner appends the SAME object to
    N topic buckets, so the dedup pass sees identical provenance across
    aliases. Used by the cross-category tests below."""
    return ExtractedItem(
        item_type="event", summary=summary,
        evidence=[EvidenceSpan(
            text=summary, source_ref="splitX",
            file_path=source_path, file_offset=file_offset,
            file_length=len(summary),
        )],
        occurred_at="2025-01-01", topics=list(topics), tags=[],
        confidence=0.9,
    )


def test_plan_dedup_emits_one_record_per_unique_fact_across_topics():
    """A fact carrying topics ["A", "B"] is appended to both topic
    buckets by the runner. The embedding plan must emit ONE fact record
    (at the canonical topic, which is "A" under tuple ordering), NOT
    one per category — otherwise retrieval pollutes with duplicate
    vectors for the same underlying fact."""
    doc = _doc("Cross-cat statement here.")
    shared = _xcat_fact(
        "shared fact", ["health", "work"],
        source_path=doc.source_path, file_offset=0,
    )
    plan = build_embeddings_plan(
        docs=[doc],
        facts_by_topic={"health": [shared], "work": [shared]},
        entities_output=None, patterns_by_topic=None,
        insight_output=None, action_list=None,
    )
    fact_records = [r for r in plan.records if r.kind == "fact"]
    assert len(fact_records) == 1
    # Canonical topic is the lexicographically smallest — "health".
    assert fact_records[0].record_id == "health:0"
    assert fact_records[0].topic == "health"




def test_plan_dedup_is_a_no_op_for_singleton_topic_facts():
    """The dedup pass is structurally a no-op when no fact carries more
    than one topic — the canonical map collapses to identity. Existing
    callers (and existing tests counting one record per fact) keep
    working without change."""
    doc = _doc("Single-topic.")
    only = _xcat_fact(
        "single", ["health"],
        source_path=doc.source_path, file_offset=0,
    )
    plan = build_embeddings_plan(
        docs=[doc], facts_by_topic={"health": [only]},
        entities_output=None, patterns_by_topic=None,
        insight_output=None, action_list=None,
    )
    fact_records = [r for r in plan.records if r.kind == "fact"]
    assert len(fact_records) == 1
    assert fact_records[0].record_id == "health:0"
    assert plan.entity_alias_drift == []
