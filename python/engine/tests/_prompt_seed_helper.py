"""
Subprocess helper for test_prompt_seed_independence.

NOT a pytest file. Exec'd via subprocess by the parent test under a
fixed PYTHONHASHSEED. Builds one stage's LLM prompt from a fixture and
writes the assembled bytes to stdout; the parent runs this helper under
several seeds and asserts the bytes are identical for every builder.

Covers a prompt-assembly entry point for EVERY stage that produces an
LLM call — extract, entities (per-batch), entities_dedupe, patterns,
insights, actions — assembled the way each stage's run() does (same
system + user message shape). The entities-family fixtures additionally
seed their set-typed inputs (`_Group.aliases`, `_Group.topics`, the
alias sets unioned inside `_apply_merges_to_records`) with hash-
collision pairs — surface forms that tie on the `(-len, lowercased)`
sort key, e.g. ``W.N.P. BARBELLION`` vs ``W.N.P. Barbellion`` — so a
non-total sort key surfaces immediately rather than only on the
~1-in-N corpus that happens to contain a collision. The other stages'
prompt paths iterate lists / `sorted()` over distinct keys today; the
fixtures still feed them multi-element collections in scrambled order
so any future set/dict-iteration leak is caught the same way.

Builders (stage → real assembly entry point):
  - extract                content_extractor._build_prompt + _SYSTEM
  - entities_block         entities._render_entity_block + _build_prompt
  - dedupe_materialize     entities._materialize_record → dedupe rows
  - dedupe_materialize_s1  same, sample_step=1 (minimal-row branch)
  - dedupe_merge           entities._apply_merges_to_records → rows
  - context_block          entities.build_context_block
  - patterns               patterns._build_messages
  - insights               insights._build_prompt + system assembly
  - actions                actions._build_prompt + system assembly
"""
from __future__ import annotations

import json
import sys

# Pinned dates — a builder that interpolates a date must use a fixed
# value, else two subprocesses run on different wall-clock seconds
# would diverge for a reason unrelated to hash seeding.
_FACT_DATE = "2025-01-15"
_FACT_DATE_2 = "2025-01-17"

# Hash-collision aliases: each pair ties on (-len, a.lower()), so their
# order is decided by `set` iteration unless the sort key is total.
_COLLISION_ALIASES = [
    "W.N.P. BARBELLION", "W.N.P. Barbellion",   # len 17, same .lower()
    "Wilhelm", "wilhelm",                       # len 7,  same .lower()
    "BARBELLION", "Barbellion",                 # len 10, same .lower()
    "B.", "Cynthia", "the Author", "Author",
]
_TOPICS = {"zoology", "books", "health", "art", "work", "science"}


# ── entities-family fixtures (the stage with the real leak) ──────────────────

def _group(entities_mod, *, gid: str, name: str):
    g = entities_mod._Group(gid=gid, canonical_name=name, entity_type="person")
    g.aliases = set(_COLLISION_ALIASES) | {name}
    g.topics = set(_TOPICS)
    g.mention_count = 42
    g.facts = [
        ("zoology", 0, "Catalogued specimens at the museum.", _FACT_DATE),
        ("health", 1, "Recorded a relapse of the illness.", _FACT_DATE_2),
    ]
    g.evidence_fact_refs = [("zoology", 0), ("health", 1)]
    return g


def _materialized_record(entities_mod):
    g = _group(entities_mod, gid="g1", name="W.N.P. Barbellion")
    records: list = []
    entities_mod._materialize_record(g, {}, records, {}, [], {"g1": g})
    return records


def build_entities_block(entities_mod) -> str:
    g = _group(entities_mod, gid="g1", name="W.N.P. Barbellion")
    other_a = _group(entities_mod, gid="g2", name="Cynthia")
    other_b = _group(entities_mod, gid="g3", name="Marie Bashkirtseff")
    g.other_catalog = {"g3": other_b.canonical_name, "g2": other_a.canonical_name}
    groups_by_gid = {"g1": g, "g2": other_a, "g3": other_b}
    block = entities_mod._render_entity_block(g, [], groups_by_gid)
    return entities_mod._build_prompt(
        [block], subject="the author", n_groups=1, is_bundle=False,
    )


def _dedupe_prompt(entities_mod, records, sample_step: int) -> str:
    rows = entities_mod._render_dedupe_rows(records, sample_step=sample_step)
    return entities_mod._DEDUPE_TASK.format(n=len(records), rows=rows)


def build_dedupe_materialize(entities_mod) -> str:
    return _dedupe_prompt(entities_mod, _materialized_record(entities_mod), 0)


def build_dedupe_materialize_s1(entities_mod) -> str:
    # Two records so sample_step=1 splits full/minimal rows.
    g1 = _group(entities_mod, gid="g1", name="W.N.P. Barbellion")
    g2 = _group(entities_mod, gid="g2", name="Marie Bashkirtseff")
    g2.mention_count = 9
    records: list = []
    entities_mod._materialize_record(g1, {}, records, {}, [], {"g1": g1})
    entities_mod._materialize_record(g2, {}, records, {}, [], {"g2": g2})
    return _dedupe_prompt(entities_mod, records, 1)


def build_dedupe_merge(entities_mod) -> str:
    from engine.llm import Mode
    # Two records that merge; their alias LISTS each carry half the
    # collision pairs so the union built inside _apply_merges_to_records
    # (a set) holds both surface forms and must sort them totally.
    rec_a = entities_mod.EntityRecord(
        canonical_id="w-n-p-barbellion",
        canonical_name="W.N.P. Barbellion",
        entity_type="person",
        aliases=["W.N.P. BARBELLION", "Wilhelm", "BARBELLION", "the Author"],
        role="subject",
        description="The journal's author.",
        mention_count=40,
        topics=["zoology", "health"],
        evidence_fact_refs=[("zoology", 0)],
    )
    rec_b = entities_mod.EntityRecord(
        canonical_id="w-n-p-barbellion-2",
        canonical_name="W.N.P. Barbellion",
        entity_type="person",
        aliases=["W.N.P. Barbellion", "wilhelm", "Barbellion", "Author"],
        role="subject",
        description="Same author, alias spelling.",
        mention_count=8,
        topics=["art", "books"],
        evidence_fact_refs=[("art", 1)],
    )
    merges = [{
        "a_id": "w-n-p-barbellion",
        "b_id": "w-n-p-barbellion-2",
        "confidence": 1.0,
        "synthesized_description": "",
    }]
    out, _rel, _rm = entities_mod._apply_merges_to_records(
        [rec_a, rec_b], merges, [], Mode.TEE,
    )
    return _dedupe_prompt(entities_mod, out, 0)


def build_context_block(entities_mod) -> str:
    records = _materialized_record(entities_mod)
    out = entities_mod.EntitiesOutput(
        subject=entities_mod.SubjectRef(
            canonical_id=records[0].canonical_id,
            display=records[0].canonical_name,
            source="argmax",
        ),
        entities=records,
        relations=[entities_mod.RelationEdge(
            from_id=records[0].canonical_id, to_id="cynthia",
            relation="married_to", confidence=0.9,
        )],
    )
    return entities_mod.build_context_block(out)


# ── other stages — mirror each stage's real messages assembly ────────────────

def build_extract(_unused) -> list:
    """content_extractor per-doc prompt. Exercises the batched-entries
    branch and the `_topics_for_run()` join."""
    from engine.content_extractor import _SYSTEM, _build_prompt
    from engine.ingestor import Document, SourceType
    doc = Document(
        id="fixture.md",
        source_path="fixture.md",
        source_type=SourceType.MD_FILE,
        content=(
            "--- ENTRY: 2025-01-15 | Day [a.md] ---\n"
            "Catalogued specimens; resolved to run every Monday.\n"
            "--- ENTRY: 2025-01-17 | Day [b.md] ---\n"
            "A relapse, but kept the routine."
        ),
        title="Fixture Journal",
        date=_FACT_DATE,
        file_id="fixture.md",
    )
    doc.metadata = {"combined_entries": [
        {"content_start": 0, "content_end": 80, "id": "a.md"},
        {"content_start": 80, "content_end": 160, "id": "b.md"},
    ]}
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _build_prompt(doc)},
    ]


def build_patterns(_unused) -> list:
    """Per-topic patterns messages. `entities_context` is produced by
    the entities stage's own context block over collision-laden records
    so an entities-side leak would also surface through this builder."""
    from engine import entities as entities_mod
    from engine.patterns import _build_messages
    from engine.content_extractor import (
        ExtractedItem, EvidenceSpan, EntityRef, Entity,
    )
    ctx = build_context_block(entities_mod)
    facts = [
        ExtractedItem(
            item_type="fact",
            summary="Ran 5km on Monday.",
            evidence=[EvidenceSpan(text="ran 5km", source_ref="fixture.md")],
            occurred_at=_FACT_DATE,
            entities=[
                EntityRef(entity=Entity(name="W.N.P. Barbellion",
                                        entity_type="person"), role="subject"),
                EntityRef(entity=Entity(name="Cynthia",
                                        entity_type="person"), role="other"),
            ],
            topics=["health"],
        ),
        ExtractedItem(
            item_type="fact",
            summary="Ran 6km on Wednesday.",
            evidence=[EvidenceSpan(text="ran 6km", source_ref="fixture.md")],
            occurred_at=_FACT_DATE_2,
            entities=[EntityRef(entity=Entity(name="W.N.P. Barbellion",
                                              entity_type="person"),
                                role="subject")],
            topics=["health"],
        ),
    ]
    return _build_messages(
        facts, topic="health", hard_cap=3,
        subject="the author", entities_context=ctx,
    )


def build_insights(_unused) -> list:
    """Insights single call. `patterns_by_topic` keys are inserted in
    scrambled order to prove the `sorted(keys)` render holds."""
    from engine.insights import (
        _SYSTEM, _SUBJECT_DISCIPLINE, _EVENT_VS_INSIGHT,
        _SENTIMENT_FRAMING, _SENTIMENT_BIAS_CLAUSES, _build_prompt,
    )
    from engine.patterns import Pattern
    patterns_by_topic = {
        "work": [Pattern(name="Diligent reporter",
                         description="Files reports on time.", domain="work",
                         kind="behavior", count=2, source_facts=[(0, 1.0)])],
        "art": [Pattern(name="Sketches daily",
                        description="Keeps a sketchbook.", domain="art",
                        kind="behavior", count=4, source_facts=[(1, 0.7)])],
        "health": [Pattern(name="Consistent runner",
                           description="Runs multiple times per week.",
                           domain="health", kind="behavior", count=3,
                           source_facts=[(0, 1.0), (1, 0.8)])],
        "zoology": [Pattern(name="Field cataloguer",
                            description="Catalogues specimens.",
                            domain="zoology", kind="behavior", count=5,
                            source_facts=[(2, 0.9)])],
    }
    prompt, _ = _build_prompt(patterns_by_topic, cross_cap=2, critical_cap=1)
    system_content = (
        _SYSTEM
        + "\n\n" + _SUBJECT_DISCIPLINE.format(subject="the author")
        + "\n\n" + _EVENT_VS_INSIGHT
        + "\n\n" + _SENTIMENT_FRAMING.format(
            sentiment_clause=_SENTIMENT_BIAS_CLAUSES["neutral"],
        )
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]


def build_actions(_unused) -> list:
    """Actions single call. Multiple cross-domain + critical insights,
    each with multiple domains / proposed actions (all list-ordered)."""
    from engine.actions import (
        _SYSTEM, _SUBJECT_DISCIPLINE, _EVENT_VS_HORIZON,
        _SENTIMENT_FRAMING, _SENTIMENT_BIAS_CLAUSES, _HARM_CLAUSE,
        _build_prompt,
    )
    from engine.insights import Insight, InsightOutput
    insight_output = InsightOutput(
        cross_domain=[
            Insight(
                name="Consistency across health + work",
                description="Same trait in two domains.",
                mechanism="Routines transfer.",
                implication="Will sustain new commitments.",
                domains=["health", "work"],
                kind="amplifier",
                proposed_actions=["Stack new habit on existing routine.",
                                  "Review weekly."],
                source_patterns=[("health", 0, 1.0)],
            ),
            Insight(
                name="Observation discipline",
                description="Detailed field notes.",
                mechanism="Habit of recording.",
                implication="Reliable longitudinal record.",
                domains=["zoology", "art"],
                kind="amplifier",
                proposed_actions=["Digitize the notebooks."],
                source_patterns=[("zoology", 0, 0.9)],
            ),
        ],
        critical=[
            Insight(
                name="Illness vs routine tension",
                description="Relapses interrupt routines.",
                mechanism="Energy depletion.",
                implication="Plans need slack.",
                domains=["health"],
                kind="blocker",
                proposed_actions=["Build rest buffers into commitments."],
                source_patterns=[("health", 0, 1.0)],
            ),
        ],
    )
    prompt, _ = _build_prompt(insight_output, max_actions=3)
    system_content = "\n\n".join([
        _SYSTEM,
        _SUBJECT_DISCIPLINE.format(subject="the author"),
        _EVENT_VS_HORIZON,
        _SENTIMENT_FRAMING.format(
            sentiment_clause=_SENTIMENT_BIAS_CLAUSES["neutral"],
        ),
        _HARM_CLAUSE,
    ])
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]


_BUILDERS = {
    "extract": build_extract,
    "entities_block": build_entities_block,
    "dedupe_materialize": build_dedupe_materialize,
    "dedupe_materialize_s1": build_dedupe_materialize_s1,
    "dedupe_merge": build_dedupe_merge,
    "context_block": build_context_block,
    "patterns": build_patterns,
    "insights": build_insights,
    "actions": build_actions,
}


def main() -> None:
    from engine import entities as entities_mod

    key = sys.argv[1]
    result = _BUILDERS[key](entities_mod)
    # Builders return either the assembled user-prompt string (entities
    # render helpers) or the full [system, user] messages list (stages
    # whose run() sends both). Serialize messages with json so message
    # order + per-message content are both byte-compared; dict key order
    # is construction-fixed.
    if isinstance(result, str):
        payload = result.encode("utf-8")
    else:
        payload = json.dumps(result, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
