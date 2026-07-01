"""Per-kind golden-string tests for the prefix-builder.

The spec § RAG enhancements names exact counts: top-3 most-confident +
top-3 most-recent facts for entities; top-5 fact titles for patterns;
top-2 patterns / insights / actions for entities. Tests below lock
those numbers explicitly — a future drift from the spec breaks the
test, surfacing the deviation deliberately.

Per-kind layouts are pinned via exact-substring assertions on the
assembled prefix shape (NOT a single full-string match — the bare-text
tail is tested elsewhere). The intent is: the spec-cited fields each
appear, in the documented order, with the spec-cited count.
"""
from __future__ import annotations

from engine.actions import Action
from engine.content_extractor import (
    Entity,
    EntityRef,
    EvidenceSpan,
    ExtractedItem,
)
from engine.entities import (
    EntitiesOutput,
    EntityRecord,
    RelationEdge,
)
from engine.ingestor import Document, SourceType
from engine.insights import Insight, InsightOutput
from engine.patterns import Pattern
from engine.rag_enricher import (
    TOP_ACTIONS_ENTITY,
    TOP_FACTS_BY_CONFIDENCE_ENTITY,
    TOP_FACTS_BY_RECENCY_ENTITY,
    TOP_FACTS_PATTERN,
    TOP_INSIGHTS_ENTITY,
    TOP_PATTERNS_ENTITY,
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
    build_pattern_text,
    count_dead_end_anchors,
)


# ── Spec constants pinned ─────────────────────────────────────────────────────


def test_spec_constants_match_spec_text():
    """Lock the four counts spec § RAG enhancements names verbatim.
    If the spec ever revises these, the test name + comment here must
    move too — this is the deliberate-friction surface."""
    # Entities: "Titles + types of top-3 most confident facts and
    # top-3 most recent facts"
    assert TOP_FACTS_BY_CONFIDENCE_ENTITY == 3
    assert TOP_FACTS_BY_RECENCY_ENTITY == 3
    # Entities: "Top 2 patterns mentioning facts that mention this
    # entity" / "Titles + types of Top 2 insights..." / "Titles +
    # types of Top 2 actions..."
    assert TOP_PATTERNS_ENTITY == 2
    assert TOP_INSIGHTS_ENTITY == 2
    assert TOP_ACTIONS_ENTITY == 2
    # Patterns: "Top-5 fact titles + types (most confident)"
    assert TOP_FACTS_PATTERN == 5


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _doc(content: str, file_id: str = "f1", date: str = "2025-01-15") -> Document:
    return Document(
        id=file_id,
        source_path=f"/tmp/{file_id}.md",
        source_type=SourceType.MD_FILE,
        content=content,
        title=file_id,
        date=date,
        file_id=file_id,
    )


def _ev(text: str, file_path: str = "f1",
        file_offset: int = 0, source_ref: str = "f1") -> EvidenceSpan:
    return EvidenceSpan(
        text=text, source_ref=source_ref,
        file_path=file_path, file_offset=file_offset,
        file_length=len(text),
    )


def _fact(
    summary: str,
    *,
    item_type: str = "fact",
    confidence: float = 0.9,
    occurred_at: str | None = "2025-01-01",
    topics: list[str] | None = None,
    tags: list[str] | None = None,
    evidence: list[EvidenceSpan] | None = None,
    entity_names: list[tuple[str, str, str]] | None = None,
) -> ExtractedItem:
    return ExtractedItem(
        item_type=item_type,
        summary=summary,
        evidence=evidence or [_ev(summary)],
        occurred_at=occurred_at,
        topics=topics or ["health"],
        tags=tags or [],
        confidence=confidence,
        entities=[
            EntityRef(
                entity=Entity(name=n, entity_type=t), role=r,
            )
            for (n, t, r) in (entity_names or [])
        ],
    )


def _entity(canonical_id: str, name: str,
            evidence_fact_refs: list[tuple[str, int]] | None = None,
            mention_count: int = 1,
            entity_type: str = "person",
            role: str = "subject",
            aliases: list[str] | None = None,
            description: str = "") -> EntityRecord:
    return EntityRecord(
        canonical_id=canonical_id, canonical_name=name,
        entity_type=entity_type, aliases=aliases or [],
        role=role, description=description,
        mention_count=mention_count,
        evidence_fact_refs=evidence_fact_refs or [],
    )


def _pattern(name: str, domain: str = "health",
             kind: str = "behavior",
             source_facts: list[tuple[int, float]] | None = None) -> Pattern:
    return Pattern(
        name=name, description="pat desc", domain=domain, kind=kind,
        count=len(source_facts or []),
        source_facts=source_facts or [],
    )


def _insight(name: str,
             source_patterns: list[tuple[str, int, float]] | None = None,
             kind: str = "defensive-loop") -> Insight:
    return Insight(
        name=name, description="ins desc",
        mechanism="mech", implication="impl",
        domains=["health"], kind=kind,
        source_patterns=source_patterns or [],
    )


def _action(name: str,
            source_insights: list[tuple[str, int, float]] | None = None,
            kind: str = "protocol") -> Action:
    return Action(
        recommendation=name, objective="obj", why="bcs",
        immediate_action="do x", habit="weekly",
        success_metric="metric", horizon="short",
        review_date="2025-02-01", kind=kind,
        source_insights=source_insights or [],
    )


# ── Chunk-kind enrichment ─────────────────────────────────────────────────────


def test_chunk_prefix_carries_file_section_summary_topics_facts_entities_relations():
    """Spec § Input File Chunks — chunk prefix names file path + date,
    section, split summary, topics/tags histograms, downstream facts,
    entities (id + canonical + aliases + type + role) and relations
    where both endpoints are mentioned in the chunk."""
    doc_content = "# Health\n\nI ran 5k this morning and slept 8 hours."
    doc = _doc(doc_content, file_id="f1", date="2025-01-15")
    # Two facts that come from this chunk's source file.
    f1 = _fact(
        "ran 5k",
        confidence=0.95,
        topics=["health"],
        tags=["fitness"],
        item_type="event",
        evidence=[_ev("ran 5k this morning", file_path=doc.file_id,
                      file_offset=12, source_ref="splitX")],
        entity_names=[("Alice", "person", "subject")],
    )
    f2 = _fact(
        "slept 8h",
        confidence=0.9,
        topics=["health"],
        tags=["sleep"],
        item_type="fact",
        evidence=[_ev("slept 8 hours", file_path=doc.file_id,
                      file_offset=28, source_ref="splitX")],
        entity_names=[
            ("Alice", "person", "subject"),
            ("Bob", "person", "object"),
        ],
    )
    entities = EntitiesOutput(
        entities=[
            _entity(
                "alice", "Alice",
                evidence_fact_refs=[("health", 0), ("health", 1)],
                aliases=["Al"], description="protagonist",
            ),
            _entity(
                "bob", "Bob",
                evidence_fact_refs=[("health", 1)],
                entity_type="person", role="object",
            ),
        ],
        relations=[
            RelationEdge(
                from_id="alice", to_id="bob", relation="trains-with",
                confidence=0.8,
            ),
        ],
    )
    extract_calls = [
        {
            "chunk_id": "splitX",
            "split_summaries": [
                {"id": "splitX", "summary": "Morning health log."},
            ],
        },
    ]
    view = build_graph_view(
        docs=[doc],
        facts_by_topic={"health": [f1, f2]},
        entities_output=entities,
        patterns_by_topic=None,
        insight_output=None,
        action_list=None,
        extract_calls=extract_calls,
    )
    assert view.chunks, "RAG chunker produced no chunks"
    text = build_chunk_text(view.chunks[0], view)

    # Type header is the first line; chunk reads as "raw input".
    assert text.splitlines()[0] == "Type: raw input"
    assert f"File: {doc.source_path} | 2025-01-15" in text
    assert "Split summaries:" in text
    assert "Morning health log." in text
    assert "Topics: health×2" in text  # histogram with count
    assert "Tags: fitness×1, sleep×1" in text  # alpha-tiebreak
    assert "Facts in chunk:" in text
    assert "[event] ran 5k" in text
    assert "[fact] slept 8h" in text
    assert "Entities:" in text
    assert "Alice (person · subject)" in text
    assert "[id: alice]" in text
    assert "aka Al" in text
    assert "Relations:" in text
    assert "alice --trains-with--> bob" in text
    # Bare chunk text follows the prefix.
    assert "ran 5k this morning and slept 8 hours" in text


# ── Fact-kind enrichment ──────────────────────────────────────────────────────


def test_fact_prefix_includes_section_quote_siblings_entities_relation_patterns():
    """Spec § Facts — fact prefix names the upstream section + the
    quoted source, the prev/next sibling facts from the same splitter
    chunk, the downstream entities + relation candidate, and the
    patterns mentioning this fact (title + kind + topic)."""
    doc = _doc("# Health\n\nA. B. C.", file_id="f1")
    f_prev = _fact(
        "prev event", item_type="event",
        evidence=[_ev("A.", file_path=doc.file_id,
                      file_offset=10, source_ref="splitX")],
    )
    f_target = _fact(
        "target event",
        item_type="event",
        confidence=0.85,
        evidence=[_ev("B.", file_path=doc.file_id,
                      file_offset=13, source_ref="splitX")],
        entity_names=[("Alice", "person", "subject")],
    )
    f_target.relation_candidate = {
        "from": "Alice", "to": "Bob", "verb": "trains-with",
    }
    f_next = _fact(
        "next event", item_type="signal",
        evidence=[_ev("C.", file_path=doc.file_id,
                      file_offset=16, source_ref="splitX")],
    )
    entities = EntitiesOutput(
        entities=[
            _entity("alice", "Alice",
                    evidence_fact_refs=[("health", 1)]),
        ],
        relations=[],
    )
    pattern = _pattern(
        "morning routine", domain="health", kind="behavior",
        source_facts=[(1, 0.9)],
    )
    view = build_graph_view(
        docs=[doc],
        facts_by_topic={"health": [f_prev, f_target, f_next]},
        entities_output=entities,
        patterns_by_topic={"health": [pattern]},
        insight_output=None,
        action_list=None,
    )
    text = build_fact_text("health", 1, f_target, view)
    assert text.splitlines()[0] == "Type: fact"
    assert "Section: Health" in text
    assert "Quote: B." in text
    assert "Confidence: 0.85" in text
    assert "Previous fact: [event] prev event" in text
    assert "Next fact: [signal] next event" in text
    assert "Entities:" in text
    assert "Alice (person · subject)" in text
    assert "Relation candidate: Alice --trains-with--> Bob" in text
    assert "Patterns mentioning this fact:" in text
    assert "morning routine | kind: behavior | topic: health" in text
    assert "target event" in text  # bare summary


# ── Entity-kind enrichment ────────────────────────────────────────────────────


def test_entity_prefix_locks_spec_top_n_counts_for_facts_patterns_insights_actions():
    """Spec § Entities — entity prefix locks exact-N selection per
    spec:
      - top-3 most-confident facts
      - top-3 most-recent facts
      - top-2 patterns / top-2 insights / top-2 actions

    Build an entity that mentions 6 facts so the top-3 caps both bite;
    build 4 patterns / 4 insights / 4 actions so the top-2 caps bite.
    Assertions check the cap (correct count of dash-prefixed lines
    after each header) AND the identity of the picked items (correct
    ranking) — both must hold.
    """
    # 6 facts, varying confidence + date.
    facts: list[ExtractedItem] = []
    confs = [0.99, 0.95, 0.91, 0.81, 0.71, 0.51]
    dates = [
        "2025-01-01", "2025-03-15", "2025-06-30",
        "2025-10-10", "2024-05-05", "2024-12-25",
    ]
    for i, (c, d) in enumerate(zip(confs, dates)):
        facts.append(_fact(
            f"fact-{i}", confidence=c, occurred_at=d,
            item_type=("event" if i % 2 == 0 else "fact"),
        ))
    entity = _entity(
        "alice", "Alice",
        evidence_fact_refs=[("health", i) for i in range(len(facts))],
        mention_count=len(facts),
        aliases=["Al"], description="protagonist",
    )
    # 4 patterns, all touching the entity's facts (so all overlap is
    # non-zero). Increasing overlap counts disambiguate the top-2.
    patterns = [
        _pattern(f"pat-{j}", source_facts=[
            (k, 0.9) for k in range(j + 1)
        ])
        for j in range(4)
    ]
    # 4 insights, each tied to one pattern.
    insights = [
        _insight(f"ins-{j}", source_patterns=[("health", j, 0.8)])
        for j in range(4)
    ]
    # 4 actions, each tied to one insight (cross_domain scope).
    actions = [
        _action(f"act-{j}", source_insights=[("cross_domain", j, 0.7)])
        for j in range(4)
    ]

    view = build_graph_view(
        docs=[],
        facts_by_topic={"health": facts},
        entities_output=EntitiesOutput(
            entities=[entity],
            relations=[
                RelationEdge(
                    from_id="alice", to_id="bob", relation="trains-with",
                ),
            ],
        ),
        patterns_by_topic={"health": patterns},
        insight_output=InsightOutput(cross_domain=insights, critical=[]),
        action_list=actions,
    )
    text = build_entity_text(entity, view)

    assert text.splitlines()[0] == "Type: entity"
    # Date span.
    assert "Date span: 2024-05-05 … 2025-10-10" in text

    # Relations one-liner.
    assert "Relations:" in text
    assert "trains-with → bob" in text

    # Top-3 most-confident facts: f0, f1, f2 (confidences 0.99, 0.95, 0.91).
    conf_section = text.split("Top 3 most-confident facts:")[1].split(
        "Top 3 most-recent facts:")[0]
    conf_lines = [
        ln for ln in conf_section.splitlines()
        if ln.strip().startswith("- ")
    ]
    assert len(conf_lines) == TOP_FACTS_BY_CONFIDENCE_ENTITY == 3
    assert "fact-0" in conf_lines[0]
    assert "fact-1" in conf_lines[1]
    assert "fact-2" in conf_lines[2]

    # Top-3 most-recent: f3 (2025-10-10), f2 (2025-06-30), f1 (2025-03-15).
    recent_section = text.split("Top 3 most-recent facts:")[1].split(
        "Top 2 patterns:")[0]
    recent_lines = [
        ln for ln in recent_section.splitlines()
        if ln.strip().startswith("- ")
    ]
    assert len(recent_lines) == TOP_FACTS_BY_RECENCY_ENTITY == 3
    assert "fact-3" in recent_lines[0]
    assert "fact-2" in recent_lines[1]
    assert "fact-1" in recent_lines[2]

    # Top-2 patterns by overlap. pat-3 overlaps 4, pat-2 overlaps 3 → top 2.
    patterns_section = text.split("Top 2 patterns:")[1].split(
        "Top 2 insights:")[0]
    pat_lines = [
        ln for ln in patterns_section.splitlines()
        if ln.strip().startswith("- ")
    ]
    assert len(pat_lines) == TOP_PATTERNS_ENTITY == 2
    assert "pat-3" in pat_lines[0]
    assert "pat-2" in pat_lines[1]

    # Top-2 insights: same ranking via pattern→fact set.
    insights_section = text.split("Top 2 insights:")[1].split(
        "Top 2 actions:")[0]
    ins_lines = [
        ln for ln in insights_section.splitlines()
        if ln.strip().startswith("- ")
    ]
    assert len(ins_lines) == TOP_INSIGHTS_ENTITY == 2
    assert "ins-3" in ins_lines[0]
    assert "ins-2" in ins_lines[1]

    # Top-2 actions: same chain through insight→pattern→fact.
    actions_section = text.split("Top 2 actions:")[1]
    act_lines = [
        ln for ln in actions_section.splitlines()
        if ln.strip().startswith("- ")
    ]
    # Only the first TOP_ACTIONS_ENTITY entries are inside the actions
    # bucket; whatever comes after may be bare-text appendix.
    act_lines = act_lines[:TOP_ACTIONS_ENTITY]
    assert len(act_lines) == TOP_ACTIONS_ENTITY == 2
    assert "act-3" in act_lines[0]
    assert "act-2" in act_lines[1]


# ── Pattern-kind enrichment ───────────────────────────────────────────────────


def test_pattern_prefix_locks_top_5_facts_and_entity_histogram_and_downstream():
    """Spec § Patterns — pattern prefix:
      - top-5 fact titles + types (most confident)
      - entity histogram with counts
      - downstream: insights + actions mentioning this pattern
    """
    # 7 facts (more than the top-5 cap) so the cap bites.
    confs = [0.97, 0.93, 0.89, 0.85, 0.80, 0.70, 0.60]
    facts = [
        _fact(
            f"fact-{i}", confidence=c, item_type="event",
            entity_names=[("Alice", "person", "subject")] if i < 4 else [
                ("Bob", "person", "object"),
            ],
        )
        for i, c in enumerate(confs)
    ]
    pattern = _pattern(
        "morning routine", domain="health", kind="behavior",
        # source_facts pre-sorted by confidence desc, as the patterns
        # stage produces.
        source_facts=[(i, c) for i, c in enumerate(confs)],
    )
    entities = EntitiesOutput(
        entities=[
            _entity(
                "alice", "Alice",
                evidence_fact_refs=[("health", i) for i in range(4)],
            ),
            _entity(
                "bob", "Bob", entity_type="person", role="object",
                evidence_fact_refs=[("health", i) for i in range(4, 7)],
            ),
        ],
        relations=[],
    )
    insight = _insight(
        "ins-1", source_patterns=[("health", 0, 0.9)],
    )
    action = _action(
        "act-1", source_insights=[("cross_domain", 0, 0.9)],
    )
    view = build_graph_view(
        docs=[],
        facts_by_topic={"health": facts},
        entities_output=entities,
        patterns_by_topic={"health": [pattern]},
        insight_output=InsightOutput(cross_domain=[insight], critical=[]),
        action_list=[action],
    )
    text = build_pattern_text("health", 0, pattern, view)

    assert text.splitlines()[0] == "Type: pattern"
    # Fact count + date span.
    assert "Fact count: 7" in text

    # Top-5 most-confident facts: the first 5 of source_facts.
    conf_section = text.split("Top 5 most-confident facts:")[1].split(
        "Entities (histogram):")[0]
    conf_lines = [
        ln for ln in conf_section.splitlines()
        if ln.strip().startswith("- ")
    ]
    assert len(conf_lines) == TOP_FACTS_PATTERN == 5
    for i in range(5):
        assert f"fact-{i}" in conf_lines[i]

    # Entity histogram: Alice×4, Bob×3.
    hist_section = text.split("Entities (histogram):")[1].split(
        "Insights mentioning this pattern:")[0]
    hist_lines = [
        ln for ln in hist_section.splitlines()
        if ln.strip().startswith("- ")
    ]
    assert "Alice" in hist_lines[0] and "×4" in hist_lines[0]
    assert "Bob" in hist_lines[1] and "×3" in hist_lines[1]

    # Downstream insights + actions.
    assert "Insights mentioning this pattern:" in text
    assert "ins-1" in text
    assert "Actions mentioning this pattern:" in text
    assert "act-1" in text


# ── Insight-kind enrichment ───────────────────────────────────────────────────


def test_insight_prefix_carries_pattern_count_fact_count_histograms_and_actions():
    """Spec § Insights — insight prefix carries date span + pattern
    count + fact count, topics & entities histograms (unbounded per
    spec — no top-N), patterns mentioned, and downstream actions."""
    facts = [
        _fact(
            f"fact-{i}",
            occurred_at=f"2025-0{(i % 9) + 1}-01",
            confidence=0.9,
            topics=["health" if i < 3 else "work"],
            entity_names=[("Alice", "person", "subject")],
        )
        for i in range(5)
    ]
    # health-bucket facts are 0/1/2 (3 total); work-bucket facts are
    # 0/1 (2 total). Patterns reference local-to-topic indices.
    patterns = [
        _pattern(
            "pat-0", domain="health", kind="behavior",
            source_facts=[(k, 0.9) for k in range(3)],
        ),
        _pattern(
            "pat-1", domain="work", kind="behavior",
            source_facts=[(k, 0.9) for k in range(2)],
        ),
    ]
    insight = _insight(
        "morning consistency",
        source_patterns=[
            ("health", 0, 0.9), ("work", 0, 0.85),
        ],
    )
    action = _action(
        "act-1", source_insights=[("cross_domain", 0, 0.9)],
    )
    entities = EntitiesOutput(
        entities=[
            _entity(
                "alice", "Alice",
                evidence_fact_refs=[("health", i) for i in range(3)] + [
                    ("work", i) for i in range(3, 5)
                ],
                mention_count=5,
            ),
        ],
        relations=[],
    )
    view = build_graph_view(
        docs=[],
        facts_by_topic={
            "health": facts[:3], "work": facts[3:],
        },
        entities_output=entities,
        patterns_by_topic={
            "health": [patterns[0]], "work": [patterns[1]],
        },
        insight_output=InsightOutput(cross_domain=[insight], critical=[]),
        action_list=[action],
    )
    # `facts_by_topic` for the entity record uses bucketing per topic,
    # so re-bucket evidence_fact_refs the same way: alice has facts
    # 0/1/2 under "health" and (indices into the WORK bucket) 0/1
    # under "work". Fix the entity's refs to match.
    view.entities[0].evidence_fact_refs = [
        ("health", 0), ("health", 1), ("health", 2),
        ("work", 0), ("work", 1),
    ]
    view.entity_facts["alice"] = set(view.entities[0].evidence_fact_refs)
    # Refresh fact_entities to reflect the corrected refs.
    view.fact_entities.clear()
    for ent in view.entities:
        for key in ent.evidence_fact_refs:
            view.fact_entities.setdefault(key, []).append(ent.canonical_id)

    text = build_insight_text("cross_domain", 0, insight, view)

    assert text.splitlines()[0] == "Type: insight"
    assert "Scope: cross_domain" in text
    assert "Pattern count: 2" in text
    assert "Fact count: 5" in text
    assert "Topics (histogram):" in text
    assert "health ×3" in text
    assert "work ×2" in text
    assert "Entities (histogram):" in text
    assert "Alice" in text
    assert "Patterns mentioned:" in text
    assert "pat-0" in text
    assert "pat-1" in text
    assert "Actions mentioning this insight:" in text
    assert "act-1" in text


# ── Action-kind enrichment ────────────────────────────────────────────────────


def test_action_prefix_carries_counts_topics_entities_patterns_insights():
    """Spec § Actions — action prefix (upstream-only):
      - title, type, prose blob
      - date span + insight/pattern/fact counts
      - topics histogram, entities histogram
      - patterns mentioned (title+kind+topic)
      - insights mentioned (titles+types)
    """
    facts = [
        _fact(f"fact-{i}", topics=["health"], confidence=0.9)
        for i in range(3)
    ]
    pattern = _pattern(
        "pat-1", domain="health", kind="behavior",
        source_facts=[(0, 0.9), (1, 0.8), (2, 0.7)],
    )
    insight = _insight(
        "ins-1", source_patterns=[("health", 0, 0.9)],
    )
    action = _action(
        "ship MVP",
        source_insights=[("cross_domain", 0, 0.95)],
        kind="build",
    )
    entities = EntitiesOutput(
        entities=[
            _entity(
                "alice", "Alice",
                evidence_fact_refs=[("health", i) for i in range(3)],
                mention_count=3,
            ),
        ],
        relations=[],
    )
    view = build_graph_view(
        docs=[],
        facts_by_topic={"health": facts},
        entities_output=entities,
        patterns_by_topic={"health": [pattern]},
        insight_output=InsightOutput(cross_domain=[insight], critical=[]),
        action_list=[action],
    )

    text = build_action_text(0, action, view)

    assert text.splitlines()[0] == "Type: action"
    assert "Horizon: short" in text
    assert "Review date: 2025-02-01" in text
    assert "Insight count: 1" in text
    assert "Pattern count: 1" in text
    assert "Fact count: 3" in text
    assert "Topics (histogram):" in text
    assert "health ×3" in text
    assert "Entities (histogram):" in text
    assert "Alice" in text
    assert "Patterns mentioned:" in text
    assert "pat-1" in text
    assert "Insights mentioned:" in text
    assert "ins-1" in text
    assert "ship MVP" in text  # bare


def test_build_action_text_resolves_positional_insight_refs():
    """`Insight [N]` in action prose is resolved to the referenced
    insight's quoted title in BOTH the embedded ``text`` and the bare
    ``display_text`` (the latter is what the answering model reads in the
    chatbot CONTEXT block). `[N]` maps over the full cross-domain++critical
    enumeration (continuous index). The raw ``Action.why`` is left
    untouched (the UI keeps the clickable bracket); array-index syntax and
    out-of-range refs are not touched."""
    cross = [_insight("Toxic loop"), _insight("Energy source")]
    crit = [_insight("Hidden constraint")]
    why = ("Insight [1] shows the loop; insight [3] names it; "
           "compare arr[2] and [9].")
    action = Action(
        recommendation="ship MVP", objective="obj", why=why,
        immediate_action="do x", habit="weekly", success_metric="metric",
        horizon="short", review_date="2025-02-01", kind="build",
        source_insights=[("cross_domain", 0, 0.95)],
    )
    view = build_graph_view(
        docs=[], facts_by_topic={},
        entities_output=EntitiesOutput(entities=[], relations=[]),
        patterns_by_topic={},
        insight_output=InsightOutput(cross_domain=cross, critical=crit),
        action_list=[action],
    )

    text = build_action_text(0, action, view)
    display = build_action_display(action, view)

    for body in (text, display):
        # [1] → cross[0], [3] → critical[0] (continuous index past the two
        # cross-domain insights).
        assert 'Insight "Toxic loop" shows the loop' in body
        assert 'insight "Hidden constraint" names it' in body
        # `arr[2]` is array syntax (letter before `[`), not an insight ref.
        assert "arr[2]" in body
        # [9] is past the enumeration (3 insights) — left as-is, never faked.
        assert "[9]" in body
    # Raw dataclass field is NOT mutated — the UI surfaces the bracket
    # as a clickable link, not a resolved title.
    assert action.why == why


# ── Bare-display split ────────────────────────────────────────────────────────
#
# The enriched `text` field carries the graph-context prefix the
# embedder consumes; the bare `display_text` field is what the
# answering model sees in CONTEXT, with the prefix stripped so the
# model can't infer a canonical-id-shape citation surface from in-band
# names. Section: line is preserved on chunk + fact records (useful
# semantic context, not a citation leak); everything else from the
# enriched prefix is dropped.


def test_chunk_display_drops_enriched_prefix_keeps_section_and_body():
    """Chunk display carries a Source line (file_id — basename or
    corpus-relative subpath), the Section line + the raw chunk text and
    nothing else from the enriched prefix (Type, full File path, Topics,
    Facts, Entities, Relations all absent)."""
    doc_content = "# Health\n\nI ran 5k this morning."
    doc = _doc(doc_content, file_id="f1", date="2025-01-15")
    f1 = _fact(
        "ran 5k",
        evidence=[_ev("ran 5k this morning", file_path=doc.file_id,
                      file_offset=12, source_ref="splitX")],
        entity_names=[("Alice", "person", "subject")],
    )
    entities = EntitiesOutput(
        entities=[_entity(
            "alice", "Alice", evidence_fact_refs=[("health", 0)],
        )],
        relations=[],
    )
    view = build_graph_view(
        docs=[doc], facts_by_topic={"health": [f1]},
        entities_output=entities,
        patterns_by_topic=None, insight_output=None, action_list=None,
    )
    chunk = view.chunks[0]
    display = build_chunk_display(chunk)

    assert chunk.text in display
    # Source line names the file (file_id — basename or corpus-relative
    # subpath) the chunk came from.
    assert display.startswith("Source: f1")
    assert ("Section: " in display) or not chunk.section_path
    # Enriched-prefix surfaces absent from the bare display. The full
    # `File: <absolute path>` line in particular stays out (Source carries
    # the basename instead).
    assert "Type:" not in display
    assert "File:" not in display
    assert "Topics:" not in display
    assert "Facts in chunk:" not in display
    assert "Entities:" not in display
    assert "Relations:" not in display
    assert "[id:" not in display


def _doc_graph(doc, facts=None):
    """A GraphView over one doc (+ optional facts) — enough to drive the
    document/chunk builders + edges."""
    return build_graph_view(
        docs=[doc],
        facts_by_topic=facts or {},
        entities_output=None,
        patterns_by_topic=None, insight_output=None, action_list=None,
    )


def test_document_text_carries_filename_date_and_chunk_count():
    """The document embed text names the file, its date, and its chunk
    count so a dense query about the file by name can land on it."""
    doc = _doc(
        "# Trip\n\nAlice and I went to Lisbon.", file_id="alice-trip.txt",
        date="2026-05-01",
    )
    view = _doc_graph(doc)
    txt = build_document_text(doc, view)
    assert "Type: source file" in txt
    assert "File: alice-trip.txt" in txt
    assert "2026-05-01" in txt
    assert "Chunks:" in txt


def test_document_display_names_file_for_inventory_queries():
    """The document display form leads with the filename + a chunk count
    so the model can answer "do I have file X?" — and keeps the full
    absolute source path out (only the file_id basename shows)."""
    doc = _doc(
        "# Trip\n\nAlice and I went to Lisbon.", file_id="alice-trip.txt",
        date="2026-05-01",
    )
    view = _doc_graph(doc)
    display = build_document_display(doc, view)
    assert display.startswith("File: alice-trip.txt")
    assert "2026-05-01" in display
    assert "chunk" in display
    # Absolute source path (/tmp/...) never leaks into display.
    assert "/tmp/" not in display


def test_embeddings_plan_emits_one_document_record_per_file():
    """build_embeddings_plan mints a `document` record keyed on file_id,
    and build_edges wires chunk↔document both ways so has_neighbor walks
    between a file and its chunks."""
    from engine.embeddings import build_embeddings_plan

    doc = _doc("# Trip\n\nAlice went to Lisbon.", file_id="alice-trip.txt")
    plan = build_embeddings_plan(
        docs=[doc], facts_by_topic=None, entities_output=None,
        patterns_by_topic=None, insight_output=None, action_list=None,
    )
    docs = [r for r in plan.records if r.kind == "document"]
    assert len(docs) == 1
    assert docs[0].record_id == "alice-trip.txt"
    assert docs[0].file_id == "alice-trip.txt"

    chunk_ids = {r.record_id for r in plan.records if r.kind == "chunk"}
    fwd = {(s, d) for (sk, s, dk, d, _e) in plan.edges
           if sk == "chunk" and dk == "document"}
    back = {(s, d) for (sk, s, dk, d, _e) in plan.edges
            if sk == "document" and dk == "chunk"}
    # Every chunk points at its document and vice versa.
    for cid in chunk_ids:
        assert (cid, "alice-trip.txt") in fwd
        assert ("alice-trip.txt", cid) in back


def test_fact_records_carry_source_file_id():
    """A fact is file-scoped — its record carries the file_id its
    evidence quotes, so the `source` filter reaches facts directly
    (entry_type ["fact"], source [name]) instead of forcing a long
    document→chunk→fact walk."""
    from engine.embeddings import build_embeddings_plan

    doc = _doc("# Health\n\nI ran 5k.", file_id="health-journal.txt")
    fact = _fact(
        "ran 5k",
        evidence=[_ev("ran 5k", file_path="health-journal.txt", file_offset=11)],
    )
    plan = build_embeddings_plan(
        docs=[doc], facts_by_topic={"health": [fact]},
        entities_output=None, patterns_by_topic=None,
        insight_output=None, action_list=None,
    )
    facts = [r for r in plan.records if r.kind == "fact"]
    assert facts and all(r.file_id == "health-journal.txt" for r in facts)


def test_fact_display_strips_kind_brackets_and_patterns_name_list():
    """A fact whose enriched form lists a `[fact]` kind-bracket on
    sibling lines and a `Patterns mentioning this fact:` block has
    neither in display; the bare summary + topics/tags remain."""
    doc = _doc(
        "# Logs\n\nA. B.", file_id="f1", date="2025-01-15",
    )
    sib_a = _fact(
        "earlier event",
        item_type="event",
        evidence=[_ev("A.", file_path=doc.file_id,
                      file_offset=9, source_ref="splitX")],
    )
    target = _fact(
        "core fact summary",
        item_type="habit",
        topics=["health"],
        tags=["sleep"],
        evidence=[_ev("B.", file_path=doc.file_id,
                      file_offset=12, source_ref="splitX")],
        entity_names=[("Alice", "person", "subject")],
    )
    entities = EntitiesOutput(
        entities=[_entity(
            "alice", "Alice",
            evidence_fact_refs=[("health", 0), ("health", 1)],
        )],
        relations=[],
    )
    pattern = _pattern(
        "Sleep regularity",
        domain="health", kind="behavior",
        source_facts=[(1, 0.9)],
    )
    view = build_graph_view(
        docs=[doc],
        facts_by_topic={"health": [sib_a, target]},
        entities_output=entities,
        patterns_by_topic={"health": [pattern]},
        insight_output=None, action_list=None,
    )
    enriched = build_fact_text("health", 1, target, view)
    display = build_fact_display(target, view)

    assert "Patterns mentioning this fact:" in enriched
    assert "Sleep regularity" in enriched

    # Bare display omits the prefix's bracketed-kind sibling lines, the
    # entities name-list, and the patterns-mentioning block.
    assert "Patterns mentioning this fact:" not in display
    assert "Sleep regularity" not in display
    assert "Previous fact:" not in display
    assert "[event]" not in display
    assert "[habit]" not in display
    assert "Entities:" not in display
    assert "[id:" not in display
    assert "Type:" not in display

    # Substantive body survives.
    assert "core fact summary" in display
    assert "topics: health" in display
    assert "tags: sleep" in display


def test_entity_display_drops_canonical_id_keeps_name_type_description():
    """An entity's bare display carries OG content only — canonical
    name, type/role, aliases, description — and never the canonical_id
    surface that lives on the enriched record. A real-corpus regression
    was an entity display starting with a slugified id like
    `41-watchung-plaza-... · 41 Watchung Plaza ...`; the head must be
    the name alone."""
    ent = _entity(
        "41-watchung-plaza-516-montclair-nj-07042-usa",
        "41 Watchung Plaza #516, Montclair NJ 07042, USA",
        entity_type="place", role="place",
        description="The Foundation's business office is at this address.",
    )
    display = build_entity_display(ent)

    assert "41-watchung-plaza-516-montclair-nj-07042-usa" not in display
    assert "ID:" not in display
    # Substantive body survives.
    assert display.startswith("41 Watchung Plaza #516, Montclair NJ 07042, USA")
    assert "place · place" in display
    assert "The Foundation's business office is at this address." in display


def test_insight_display_drops_patterns_name_list_keeps_description():
    """An insight whose enriched form has a `Patterns mentioned:` name
    list and a `[kind]`-bracketed `Actions mentioning this insight:`
    block has neither in display; the name + description prose
    remain."""
    facts = [
        _fact(f"fact-{i}", topics=["health"],
              evidence=[_ev(f"q{i}", file_path="f1",
                            file_offset=i, source_ref="splitX")])
        for i in range(2)
    ]
    pattern = _pattern(
        "Sleep regularity", domain="health", kind="behavior",
        source_facts=[(0, 0.9), (1, 0.9)],
    )
    insight = _insight(
        "morning consistency",
        source_patterns=[("health", 0, 0.9)],
        kind="defensive-loop",
    )
    insight.description = "Mornings drive the rest of the day."
    action = _action(
        "ship MVP", source_insights=[("cross_domain", 0, 0.9)],
    )
    view = build_graph_view(
        docs=[],
        facts_by_topic={"health": facts},
        entities_output=None,
        patterns_by_topic={"health": [pattern]},
        insight_output=InsightOutput(cross_domain=[insight], critical=[]),
        action_list=[action],
    )
    enriched = build_insight_text("cross_domain", 0, insight, view)
    display = build_insight_display(insight)

    assert "Patterns mentioned:" in enriched
    assert "Sleep regularity" in enriched
    assert "Actions mentioning this insight:" in enriched
    assert "ship MVP" in enriched

    # Bare display strips the pattern + action name-lists, the
    # bracketed kind markers and the Type: header.
    assert "Patterns mentioned:" not in display
    assert "Sleep regularity" not in display
    assert "Actions mentioning this insight:" not in display
    assert "ship MVP" not in display
    assert "Type:" not in display
    assert "[" not in display  # no bracketed kind / scope markers
    # Substantive body survives.
    assert "morning consistency" in display
    assert "Mornings drive the rest of the day." in display


# ── Determinism ───────────────────────────────────────────────────────────────


def test_graph_view_build_is_deterministic_across_calls():
    """Two independent calls with the same input must produce
    byte-identical enriched texts. Load-bearing for cache stability +
    retrieval reproducibility."""
    doc = _doc("A. B.", file_id="f1")
    f1 = _fact(
        "fact-a",
        evidence=[_ev("A.", file_path=doc.file_id,
                      file_offset=0, source_ref="splitX")],
        entity_names=[("Alice", "person", "subject")],
    )
    f2 = _fact(
        "fact-b",
        evidence=[_ev("B.", file_path=doc.file_id,
                      file_offset=3, source_ref="splitX")],
        entity_names=[("Alice", "person", "subject")],
    )
    entities = EntitiesOutput(
        entities=[_entity(
            "alice", "Alice",
            evidence_fact_refs=[("health", 0), ("health", 1)],
        )],
        relations=[],
    )

    def _build():
        view = build_graph_view(
            docs=[doc],
            facts_by_topic={"health": [f1, f2]},
            entities_output=entities,
            patterns_by_topic=None,
            insight_output=None,
            action_list=None,
            extract_calls=[
                {"chunk_id": "splitX", "split_summaries": [
                    {"id": "splitX", "summary": "Two short statements."}]},
            ],
        )
        return (
            build_chunk_text(view.chunks[0], view),
            build_fact_text("health", 0, f1, view),
            build_fact_text("health", 1, f2, view),
            build_entity_text(view.entities[0], view),
        )

    a = _build()
    b = _build()
    assert a == b


# ── Sparse-input degradation ──────────────────────────────────────────────────


def test_chunk_prefix_no_facts_skips_downstream_sections():
    """A chunk whose source file has no extracted facts still gets a
    valid prefix (file + section) but skips the empty downstream
    sections (facts / entities / relations) rather than emitting
    bare-headed empty buckets."""
    doc = _doc("# Header\n\nBody text.", file_id="f1")
    view = build_graph_view(
        docs=[doc], facts_by_topic=None, entities_output=None,
        patterns_by_topic=None, insight_output=None, action_list=None,
    )
    text = build_chunk_text(view.chunks[0], view)
    assert f"File: {doc.source_path} | 2025-01-15" in text
    assert "Facts in chunk:" not in text
    assert "Entities:" not in text
    assert "Relations:" not in text
    assert "Body text" in text


def test_entity_with_no_downstream_signal_still_renders():
    """An entity with no facts (degenerate) still gets a valid prefix
    + bare text."""
    ent = _entity("orphan", "Orphan", evidence_fact_refs=[])
    view = build_graph_view(
        docs=[], facts_by_topic=None,
        entities_output=EntitiesOutput(entities=[ent], relations=[]),
        patterns_by_topic=None, insight_output=None, action_list=None,
    )
    text = build_entity_text(ent, view)
    assert "Orphan" in text
    assert "Top 3 most-confident facts:" not in text
    assert "Top 2 patterns:" not in text


def test_entity_enriched_text_carries_canonical_id_in_additive_layer():
    """Spec § Entities — the entity record's "names" group is (id,
    canonical, aliases). The canonical_id must appear in the entity's
    OWN embed text so a dense query mentioning the id by string
    matches the entity record itself (not just the vector-store key
    column). It lives on a dedicated `ID:` line in the enriched
    prefix (additive layer) — never in the bare body, which carries
    OG entity content only."""
    ent = _entity("alice", "Alice", evidence_fact_refs=[])
    view = build_graph_view(
        docs=[], facts_by_topic=None,
        entities_output=EntitiesOutput(entities=[ent], relations=[]),
        patterns_by_topic=None, insight_output=None, action_list=None,
    )
    text = build_entity_text(ent, view)
    assert "ID: alice" in text
    assert "Alice" in text
    # The id is carried as a dedicated additive line, not collapsed
    # into the bare head as a `{id} · {name}` rendering.
    assert "alice · Alice" not in text


# ── Edge emission (slice-2 stage 2 — #785) ───────────────────────────────────


def _slice2_view():
    """One-of-every-kind GraphView for edge-shape tests. Built so each
    record has the minimum upstream/downstream wiring needed to exercise
    every edge type build_edges emits."""
    doc_content = (
        "# Health\n\nI ran 5k and slept 8 hours."
    )
    doc = _doc(doc_content, file_id="f1", date="2025-01-15")
    f0 = _fact(
        "ran 5k", item_type="event", confidence=0.95,
        evidence=[_ev(
            "ran 5k", file_path=doc.file_id,
            file_offset=12, source_ref="splitX",
        )],
        entity_names=[("Alice", "person", "subject")],
    )
    f1 = _fact(
        "slept 8h", item_type="fact", confidence=0.9,
        evidence=[_ev(
            "slept 8 hours", file_path=doc.file_id,
            file_offset=19, source_ref="splitX",
        )],
        entity_names=[("Bob", "person", "object")],
    )
    entities = EntitiesOutput(
        entities=[
            _entity(
                "alice", "Alice",
                evidence_fact_refs=[("health", 0)],
                mention_count=1,
            ),
            _entity(
                "bob", "Bob", entity_type="person", role="object",
                evidence_fact_refs=[("health", 1)],
                mention_count=1,
            ),
        ],
        relations=[
            RelationEdge(
                from_id="alice", to_id="bob",
                relation="trains-with", confidence=0.8,
            ),
        ],
    )
    pattern = _pattern(
        "morning routine", domain="health", kind="behavior",
        source_facts=[(0, 0.95), (1, 0.9)],
    )
    insight = _insight(
        "consistent-mornings",
        source_patterns=[("health", 0, 0.9)],
    )
    action = _action(
        "keep mornings sacred",
        source_insights=[("cross_domain", 0, 0.9)],
    )
    return build_graph_view(
        docs=[doc],
        facts_by_topic={"health": [f0, f1]},
        entities_output=entities,
        patterns_by_topic={"health": [pattern]},
        insight_output=InsightOutput(cross_domain=[insight], critical=[]),
        action_list=[action],
    )


def _edges_set(view):
    """build_edges output as a set keyed on the 4-tuple of endpoints,
    so containment checks are order-agnostic and free of duplicate
    bookkeeping."""
    return {(src_kind, src_id, dst_kind, dst_id)
            for (src_kind, src_id, dst_kind, dst_id, _label)
            in build_edges(view)}


def test_build_edges_emits_anti_fan_out_entity_action_asymmetry():
    """Director rule: `entity → action` exists; `action → entity` does
    NOT. The renderer's action-side entity histogram is for embed
    semantics; the edge index must not let `has_neighbor:[action_id],
    entry_type:[entity]` fan out from one action to every entity that
    touches any source fact."""
    view = _slice2_view()
    s = _edges_set(view)

    # entity → action present (alice's fact backs the action's pattern
    # chain).
    assert ("entity", "alice", "action", "0") in s
    assert ("entity", "bob", "action", "0") in s

    # action → entity DELIBERATELY ABSENT — exact asymmetry the spec
    # names. A failure here means we re-symmetrized by accident.
    assert not any(
        (src_kind == "action" and dst_kind == "entity")
        for (src_kind, _, dst_kind, _) in s
    )


def test_build_edges_fact_to_chunk_hop_is_exact():
    """Slice-3 marquee acceptance: action → pattern → fact → CHUNK
    resolves to the exact source chunk hop-by-hop. This test pins the
    fact → chunk leg: every fact with an in-bounds file_offset must
    emit one fact → chunk edge to the containing chunk's record_id."""
    view = _slice2_view()
    s = _edges_set(view)
    assert view.chunks, "fixture must produce at least one chunk"
    chunk_id = f"{view.chunks[0].file_id}@{view.chunks[0].char_offset}"
    assert ("fact", "health:0", "chunk", chunk_id) in s
    assert ("fact", "health:1", "chunk", chunk_id) in s
    # And the reverse hop — chunk-anchored neighbor walk reaches both
    # facts (containment is bidirectional so a chunk citation can list
    # its source facts via the same `has_neighbor` query shape).
    assert ("chunk", chunk_id, "fact", "health:0") in s
    assert ("chunk", chunk_id, "fact", "health:1") in s


def test_build_edges_fact_to_chunk_handles_evidence_file_id_basename():
    """Regression pin for the real-extraction shape: evidence carries
    ``file_path = doc.file_id`` (a basename / doc-identifier), while
    chunks carry both ``file_id`` AND a full-path ``source_path``. The
    containment lookup MUST key on ``chunk.file_id`` — keying on
    ``chunk.source_path`` silently mismatches every basename, drops
    every fact's evidence, and emits zero ``chunk ↔ fact`` edges,
    breaking the bottom of the action → … → chunk traceability chain.

    Reproduced in 2026-05-28T03-07-38Z-2n6e (full pipeline barbellion
    run): 2,254 edges across all other kinds, **zero** chunk↔fact.
    """
    doc = _doc("# Health\n\nA. B. C. D. E. F. G. H. I.\n", file_id="diary")
    f0 = _fact(
        "health:0",
        evidence=[_ev(
            "A.", file_path=doc.file_id, file_offset=10,
            source_ref="splitX",
        )],
    )
    # Mimics the production shape: the doc's full path includes /tmp/
    # AND the file_id is a different short name. The fact's evidence
    # carries ONLY the file_id basename.
    assert doc.source_path != doc.file_id, (
        "fixture must distinguish full-path source_path from file_id "
        "basename so the regression branch is actually exercised")
    view = build_graph_view(
        docs=[doc],
        facts_by_topic={"health": [f0]},
        entities_output=None,
        patterns_by_topic=None,
        insight_output=None,
        action_list=None,
    )
    s = _edges_set(view)
    chunk_id = f"{view.chunks[0].file_id}@{view.chunks[0].char_offset}"
    assert ("fact", "health:0", "chunk", chunk_id) in s, (
        "chunk↔fact edge missing — the containment lookup likely "
        "compares evidence.file_path (basename) against "
        "chunk.source_path (full path) instead of chunk.file_id.")
    assert ("chunk", chunk_id, "fact", "health:0") in s


def test_build_edges_persists_complete_adjacency_not_top_caps():
    """The TOP_FACTS_BY_CONFIDENCE_ENTITY = 3 cap and friends bound the
    embed-text token budget. The edge index must NOT inherit those
    caps — `has_neighbor` traversal silently loses the source if we do.
    Build an entity that mentions 6 facts and assert all 6 entity →
    fact edges land in the index."""
    confs = [0.99, 0.95, 0.91, 0.81, 0.71, 0.51]
    facts = [
        _fact(f"f-{i}", confidence=c, occurred_at=f"2025-0{i+1}-01")
        for i, c in enumerate(confs)
    ]
    ent = _entity(
        "alice", "Alice",
        evidence_fact_refs=[("health", i) for i in range(6)],
        mention_count=6,
    )
    view = build_graph_view(
        docs=[], facts_by_topic={"health": facts},
        entities_output=EntitiesOutput(entities=[ent], relations=[]),
        patterns_by_topic=None, insight_output=None, action_list=None,
    )
    s = _edges_set(view)
    for i in range(6):
        assert ("entity", "alice", "fact", f"health:{i}") in s, (
            f"missing entity→fact edge for health:{i} — render-cap leaked into edge index"
        )


def test_build_edges_entity_relation_is_directionally_emitted_from_both_endpoints():
    """A RelationEdge(from=A, to=B) lists on BOTH entity records'
    renderer prefixes. The edge index emits one row per anchor side so
    `has_neighbor:[A], entry_type:[entity]` and `has_neighbor:[B],
    entry_type:[entity]` both resolve to the other endpoint without
    needing a reverse-direction query."""
    view = _slice2_view()
    s = _edges_set(view)
    assert ("entity", "alice", "entity", "bob") in s
    assert ("entity", "bob", "entity", "alice") in s


def test_build_edges_sibling_facts_are_bidirectional_within_splitter_chunk():
    """Spec § Facts — prev + next fact from the same splitter chunk are
    listed on each fact's renderer. The edge index emits both
    directions so a sibling walk works from either side."""
    view = _slice2_view()
    s = _edges_set(view)
    assert ("fact", "health:0", "fact", "health:1") in s
    assert ("fact", "health:1", "fact", "health:0") in s


def test_build_edges_action_reaches_pattern_and_insight_transitively():
    """Spec § slice-2 §4 — the action's neighbor walk hops:
    `has_neighbor:[action_id], entry_type:[pattern]` → action's
    patterns; `has_neighbor:[action_id], entry_type:[insight]` →
    action's insights. Both edges land directly on the action anchor;
    no entity edge (anti-fan-out, see asymmetry test above)."""
    view = _slice2_view()
    s = _edges_set(view)
    assert ("action", "0", "pattern", "health:0") in s
    assert ("action", "0", "insight", "cross_domain:0") in s
    # Direct action → fact is NOT emitted — hop goes action → pattern → fact.
    assert not any(
        (src_kind == "action" and dst_kind == "fact")
        for (src_kind, _, dst_kind, _) in s
    )


def test_build_edges_is_idempotent_on_repeat_view():
    """build_edges is a pure function of the GraphView; calling it
    twice on the same view returns the same multiset (no hidden state,
    no order-dependent extras). Pin this so a future refactor that
    accidentally appends to a module-level list surfaces here."""
    view = _slice2_view()
    a = sorted(build_edges(view))
    b = sorted(build_edges(view))
    assert a == b


def test_count_dead_end_anchors_zero_on_clean_view():
    """The slice-2 #387-validation surface: with `evidence_fact_refs`
    properly populated by the entities stage, every anchor has at
    least one outgoing edge. A non-zero count here would be the
    #387-class corruption signal at embed time."""
    view = _slice2_view()
    edges = build_edges(view)
    dead = count_dead_end_anchors(edges, view)
    assert dead["chunk"] == 0
    assert dead["fact"] == 0
    assert dead["entity"] == 0
    assert dead["pattern"] == 0
    assert dead["insight"] == 0
    assert dead["action"] == 0


def test_count_dead_end_anchors_surfaces_orphan_entity():
    """An entity with empty `evidence_fact_refs` (the #387 symptom)
    produces zero outgoing edges from that entity — and the dead-end
    counter must report it so the phase marker can flag the
    corruption rather than silently lose the source on `has_neighbor`."""
    view = build_graph_view(
        docs=[],
        facts_by_topic={"health": [_fact("noise")]},
        entities_output=EntitiesOutput(
            entities=[
                # Orphaned entity — #387 shape.
                _entity("orphan", "Orphan", evidence_fact_refs=[]),
            ],
            relations=[],
        ),
        patterns_by_topic=None, insight_output=None, action_list=None,
    )
    edges = build_edges(view)
    dead = count_dead_end_anchors(edges, view)
    assert dead["entity"] == 1


# ── Cross-category fact dedup (#801) ──────────────────────────────────────────

# A fact carrying N topics is appended to N topic buckets by the runner
# (`runner._facts_by_topic[topic].append(it)` — same ExtractedItem
# object, distinct (topic, idx) keys). The dedup pass collapses that
# fanout to one canonical fact record at the embedding layer; the tests
# below pin the four director-invariants: canonical record-id space,
# pattern union across aliases, no self-neighbor on sibling edges, and
# the entity-drift warning that surfaces upstream inconsistency.


def test_dedup_provenance_collapse_one_record_per_unique_fact():
    """A fact carrying two topics surfaces as ONE canonical (topic, idx)
    in the graph view's `fact_canonical` map, with the alias set carrying
    both (topic, idx) keys. The canonical is the smallest under tuple
    ordering so the pick is deterministic across runs."""
    doc = _doc("Shared body.", file_id="f1")
    shared = _fact(
        "shared statement",
        topics=["health", "work"],
        evidence=[_ev("Shared body.", file_path=doc.file_id,
                      file_offset=0, source_ref="splitX")],
    )
    view = build_graph_view(
        docs=[doc],
        facts_by_topic={"health": [shared], "work": [shared]},
        entities_output=None, patterns_by_topic=None,
        insight_output=None, action_list=None,
    )
    # Canonical is ("health", 0) — smallest tuple. Both aliases collapse.
    assert view.canonical_fact_key("health", 0) == ("health", 0)
    assert view.canonical_fact_key("work", 0) == ("health", 0)
    # Alias set lists every (topic, idx) for the underlying fact.
    assert set(view.aliases_of("health", 0)) == {("health", 0), ("work", 0)}
    assert set(view.aliases_of("work", 0)) == {("health", 0), ("work", 0)}
    # canonical_fact_id is a deterministic record-id string.
    assert view.canonical_fact_id("work", 0) == "health:0"


def test_dedup_survives_resume_roundtrip_through_jsonl():
    """Codex P1 #2 regression — `id(fact)` collapses category-copies on
    a fresh run (runner appends the SAME ExtractedItem to N topic
    buckets, same id) but breaks on resume from extraction: the runner's
    `_load_facts_by_topic` reconstructs each JSONL line as a NEW
    ExtractedItem instance, so the same logical fact carried in two
    topic JSONL files lands as two distinct Python objects with two
    distinct ids and the dedup misses. The dedup key uses content-hash
    fields (path + offset + length + type + date + summary) that survive
    the JSON round-trip, so fresh and resume paths produce the same
    canonical map.
    """
    doc = _doc("Shared statement.", file_id="f1")
    # The "fresh-run" view: single ExtractedItem in two topic buckets
    # (same Python object, what runner.py:5717-5720 produces).
    shared = _fact(
        "shared fact summary",
        topics=["health", "work"],
        evidence=[_ev(
            "Shared statement.", file_path=doc.file_id,
            file_offset=0, source_ref="splitX",
        )],
    )
    fresh_view = build_graph_view(
        docs=[doc],
        facts_by_topic={"health": [shared], "work": [shared]},
        entities_output=None, patterns_by_topic=None,
        insight_output=None, action_list=None,
    )

    # The "resume-from-disk" view: each topic bucket holds an
    # independently-constructed ExtractedItem with byte-identical
    # fields (no shared object), what `_load_facts_by_topic` produces
    # after reading the per-topic JSONL files. Different Python ids,
    # identical content.
    def _loaded_copy() -> ExtractedItem:
        return ExtractedItem(
            item_type="fact", summary="shared fact summary",
            evidence=[EvidenceSpan(
                text="Shared statement.", source_ref="splitX",
                file_path=doc.file_id, file_offset=0,
                file_length=len("Shared statement."),
            )],
            occurred_at="2025-01-01", topics=["health", "work"],
            tags=[], confidence=0.9,
        )
    resume_view = build_graph_view(
        docs=[doc],
        facts_by_topic={"health": [_loaded_copy()], "work": [_loaded_copy()]},
        entities_output=None, patterns_by_topic=None,
        insight_output=None, action_list=None,
    )

    # The two views must agree on the canonical map. Same canonical id,
    # same alias set, same sibling sequence — the resume path can't be
    # allowed to drift from the fresh path.
    assert fresh_view.canonical_fact_key("health", 0) == ("health", 0)
    assert resume_view.canonical_fact_key("health", 0) == ("health", 0)
    assert fresh_view.canonical_fact_key("work", 0) == ("health", 0)
    assert resume_view.canonical_fact_key("work", 0) == ("health", 0)
    assert set(fresh_view.aliases_of("health", 0)) == {("health", 0), ("work", 0)}
    assert set(resume_view.aliases_of("health", 0)) == {("health", 0), ("work", 0)}


def test_dedup_missing_offsets_force_singleton_no_collapse():
    """Codex P1 #3 regression — when extraction's span attribution
    fails, evidence may land with `file_offset=None` / `file_length=None`.
    Falling back to a sentinel (-1) in the dedup key would let two
    distinct facts with missing offsets in the same file silently
    merge if their item_type/occurred_at/summary happened to match.
    The dedup pass treats missing offsets as "location unknown ⇒ no
    safe collapse" and emits each such fact as its own singleton."""
    doc = _doc("Body.", file_id="f1")
    # Two facts in the same file with no offset/length attribution.
    # Everything else (type, date, summary) intentionally identical to
    # exercise the worst case — the safe default is still NOT collapse.
    span = EvidenceSpan(
        text="Body.", source_ref="splitX",
        file_path=doc.file_id,
        file_offset=None, file_length=None,
    )
    fact_a = ExtractedItem(
        item_type="fact", summary="same summary",
        evidence=[span], occurred_at="2025-01-01",
        topics=["health"], tags=[], confidence=0.9,
    )
    fact_b = ExtractedItem(
        item_type="fact", summary="same summary",
        evidence=[span], occurred_at="2025-01-01",
        topics=["health"], tags=[], confidence=0.9,
    )
    view = build_graph_view(
        docs=[doc],
        facts_by_topic={"health": [fact_a, fact_b]},
        entities_output=None, patterns_by_topic=None,
        insight_output=None, action_list=None,
    )
    # Each fact is its own canonical. No alias collapse.
    assert view.canonical_fact_key("health", 0) == ("health", 0)
    assert view.canonical_fact_key("health", 1) == ("health", 1)
    assert view.aliases_of("health", 0) == [("health", 0)]
    assert view.aliases_of("health", 1) == [("health", 1)]


def test_dedup_two_distinct_facts_at_same_span_stay_distinct():
    """Codex P1 regression — extraction can emit two different facts
    from the same quoted span (same file_path/offset/length, possibly
    same item_type and occurred_at). Provenance alone would collapse
    them into one canonical, silently dropping a record and rewriting
    downstream pattern/entity edges onto the survivor. The dedup key
    includes Python object identity so distinct ExtractedItem objects
    at the same span keep distinct canonicals — only the runner's
    multi-topic fanout (where the SAME object is appended to N topic
    buckets) collapses."""
    doc = _doc("Two facts in one sentence.", file_id="f1")
    span = _ev(
        "Two facts in one sentence.",
        file_path=doc.file_id, file_offset=0, source_ref="splitX",
    )
    fact_a = _fact(
        "first claim from the sentence",
        topics=["health"],
        evidence=[span],
    )
    fact_b = _fact(
        "second claim from the same sentence",
        topics=["health"],
        evidence=[span],
    )
    view = build_graph_view(
        docs=[doc],
        facts_by_topic={"health": [fact_a, fact_b]},
        entities_output=None, patterns_by_topic=None,
        insight_output=None, action_list=None,
    )
    # Both facts keep their own canonical — no collapse, no alias.
    assert view.canonical_fact_key("health", 0) == ("health", 0)
    assert view.canonical_fact_key("health", 1) == ("health", 1)
    assert view.aliases_of("health", 0) == [("health", 0)]
    assert view.aliases_of("health", 1) == [("health", 1)]
    # Sibling structure preserved — two underlying facts in the chunk.
    assert view.canonical_siblings("splitX") == [("health", 0), ("health", 1)]


def test_dedup_singletons_have_self_as_canonical():
    """A fact in just one topic is its own canonical with a singleton
    alias list — the dedup pass is a no-op for the common case."""
    doc = _doc("Body.", file_id="f1")
    only = _fact(
        "single-topic fact", topics=["health"],
        evidence=[_ev("Body.", file_path=doc.file_id,
                      file_offset=0, source_ref="splitX")],
    )
    view = build_graph_view(
        docs=[doc], facts_by_topic={"health": [only]},
        entities_output=None, patterns_by_topic=None,
        insight_output=None, action_list=None,
    )
    assert view.canonical_fact_key("health", 0) == ("health", 0)
    assert view.aliases_of("health", 0) == [("health", 0)]


def test_dedup_pattern_union_across_aliases():
    """A fact in topics A + B with a pattern in EACH topic citing the
    fact (by its topic-local fact_idx). The canonical fact record must
    list BOTH patterns (union across category-copies) — strict
    `(topic, idx) in fs` would only catch the alias-local pattern and
    silently drop the other."""
    doc = _doc("Shared statement here.", file_id="f1")
    shared = _fact(
        "cross-category fact",
        topics=["health", "work"],
        evidence=[_ev("Shared statement here.", file_path=doc.file_id,
                      file_offset=0, source_ref="splitX")],
    )
    pat_h = _pattern(
        "morning routine", domain="health", kind="behavior",
        source_facts=[(0, 0.95)],
    )
    pat_w = _pattern(
        "focus block", domain="work", kind="behavior",
        source_facts=[(0, 0.9)],
    )
    view = build_graph_view(
        docs=[doc],
        facts_by_topic={"health": [shared], "work": [shared]},
        entities_output=None,
        patterns_by_topic={"health": [pat_h], "work": [pat_w]},
        insight_output=None, action_list=None,
    )
    # Canonical fact's text lists BOTH patterns under "Patterns
    # mentioning this fact" — union semantics.
    text = build_fact_text("health", 0, shared, view)
    assert "Patterns mentioning this fact:" in text
    assert "morning routine" in text
    assert "focus block" in text

    # Edges side: both patterns reach the canonical fact id, and the
    # canonical fact reaches both patterns. No edge endpoints sit on
    # the non-canonical alias (`work:0`).
    edges = build_edges(view)
    s = {(sk, si, dk, di) for (sk, si, dk, di, _ek) in edges}
    assert ("pattern", "health:0", "fact", "health:0") in s
    assert ("pattern", "work:0", "fact", "health:0") in s
    assert ("fact", "health:0", "pattern", "health:0") in s
    assert ("fact", "health:0", "pattern", "work:0") in s
    assert not any(
        di == "work:0" and dk == "fact"
        for (_sk, _si, dk, di) in s
    )


def test_dedup_no_self_neighbor_on_sibling_edges():
    """Two cross-category facts adjacent within the SAME splitter chunk.
    The naive sibling walk over the raw `facts_by_splitter_chunk` list
    would emit sibling edges between alias-copies of the SAME underlying
    fact (a self-edge after the canonical projection). The canonical
    sibling list collapses aliases so prev/next always resolve to a
    genuinely different underlying fact (director invariant 4)."""
    doc = _doc("F1 here. F2 here.", file_id="f1")
    f_one = _fact(
        "first fact", topics=["health", "work"],
        evidence=[_ev("F1 here.", file_path=doc.file_id,
                      file_offset=0, source_ref="splitX")],
    )
    f_two = _fact(
        "second fact", topics=["health", "work"],
        evidence=[_ev("F2 here.", file_path=doc.file_id,
                      file_offset=9, source_ref="splitX")],
    )
    view = build_graph_view(
        docs=[doc],
        facts_by_topic={"health": [f_one, f_two], "work": [f_one, f_two]},
        entities_output=None, patterns_by_topic=None,
        insight_output=None, action_list=None,
    )
    # canonical_siblings collapses to two underlying facts in order.
    canonical_sibs = view.canonical_siblings("splitX")
    assert canonical_sibs == [("health", 0), ("health", 1)]

    # Sibling edges only between the two canonicals; no edge targets a
    # work:* alias.
    edges = build_edges(view)
    sibling_pairs = {
        (si, di) for (sk, si, dk, di, ek) in edges
        if sk == "fact" and dk == "fact" and ek == "sibling"
    }
    assert ("health:0", "health:1") in sibling_pairs
    assert ("health:1", "health:0") in sibling_pairs
    # NEVER a self-edge from canonical onto its own alias.
    for (si, di) in sibling_pairs:
        assert si != di, f"self-edge sibling: {si} -> {di}"
        assert not (si == "health:0" and di == "work:0")
        assert not (si == "health:1" and di == "work:1")

    # Fact text on the canonical surfaces prev/next from the underlying
    # facts, not from an alias of self.
    text_zero = build_fact_text("health", 0, f_one, view)
    assert "Previous fact:" not in text_zero  # first in chunk
    assert "Next fact: [fact] second fact" in text_zero
    text_one = build_fact_text("health", 1, f_two, view)
    assert "Previous fact: [fact] first fact" in text_one
    assert "Next fact:" not in text_one  # last in chunk


def test_dedup_entity_alias_drift_flagged_when_mention_sets_differ():
    """Director invariant 1: an entity's mention set must be identical
    across all aliases of a canonical fact (mention is a property of
    the fact, not the category). When the entities stage's
    `evidence_fact_refs` lists only some aliases — a regression that
    drops a category-copy from the mention map — `entity_alias_drift`
    captures the per-alias diff so the embeddings stage can surface a
    warning record."""
    doc = _doc("Cross-cat fact.", file_id="f1")
    shared = _fact(
        "cross-cat fact", topics=["health", "work"],
        evidence=[_ev("Cross-cat fact.", file_path=doc.file_id,
                      file_offset=0, source_ref="splitX")],
    )
    # Alice references the health alias only; Bob references both.
    # Director claim is that the entity stage should produce the same
    # mention set across aliases — Alice's missing (work, 0) is drift.
    ent_alice = _entity(
        "alice", "Alice",
        evidence_fact_refs=[("health", 0)],
    )
    ent_bob = _entity(
        "bob", "Bob",
        evidence_fact_refs=[("health", 0), ("work", 0)],
    )
    view = build_graph_view(
        docs=[doc],
        facts_by_topic={"health": [shared], "work": [shared]},
        entities_output=EntitiesOutput(
            entities=[ent_alice, ent_bob], relations=[],
        ),
        patterns_by_topic=None, insight_output=None, action_list=None,
    )
    assert len(view.entity_alias_drift) == 1
    drift = view.entity_alias_drift[0]
    assert drift["canonical_id"] == "health:0"
    assert set(drift["aliases"]) == {"health:0", "work:0"}
    # Per-alias entity sets: one alias has both, the other only Bob.
    sets = [set(s) for s in drift["per_alias_entities"]]
    assert {"alice", "bob"} in sets
    assert {"bob"} in sets


def test_dedup_entity_alias_drift_quiet_on_clean_view():
    """When the entities stage faithfully replicates the mention set
    across every alias (the expected steady state), the drift list is
    empty and no warning fires."""
    doc = _doc("Shared body.", file_id="f1")
    shared = _fact(
        "cross-cat", topics=["health", "work"],
        evidence=[_ev("Shared body.", file_path=doc.file_id,
                      file_offset=0, source_ref="splitX")],
    )
    ent = _entity(
        "alice", "Alice",
        evidence_fact_refs=[("health", 0), ("work", 0)],
    )
    view = build_graph_view(
        docs=[doc],
        facts_by_topic={"health": [shared], "work": [shared]},
        entities_output=EntitiesOutput(entities=[ent], relations=[]),
        patterns_by_topic=None, insight_output=None, action_list=None,
    )
    assert view.entity_alias_drift == []


def test_dedup_chunk_fact_edge_uses_canonical_id():
    """Director invariant 4 on the chunk-containment hop: a chunk
    containing a cross-category fact emits ONE chunk → fact edge to
    the canonical id, not one per alias."""
    doc = _doc("Body of f1 inside.", file_id="f1")
    shared = _fact(
        "cross-cat fact", topics=["health", "work"],
        evidence=[_ev("Body of f1 inside.", file_path=doc.file_id,
                      file_offset=0, source_ref="splitX")],
    )
    view = build_graph_view(
        docs=[doc],
        facts_by_topic={"health": [shared], "work": [shared]},
        entities_output=None, patterns_by_topic=None,
        insight_output=None, action_list=None,
    )
    edges = build_edges(view)
    chunk_id = f"{view.chunks[0].file_id}@{view.chunks[0].char_offset}"
    fact_endpoints = {
        di for (sk, si, dk, di, _ek) in edges
        if sk == "chunk" and si == chunk_id and dk == "fact"
    }
    # Exactly the canonical fact id; no alias.
    assert fact_endpoints == {"health:0"}
