"""
Phase-boundary normalization tests (issue #339).

Every cross-phase data shape (`facts_by_topic`, `entities_output`,
`patterns_by_topic`, `insight_output`) is consumed by the next stage's
prompt construction VERBATIM — list-order ends up as line-order in the
LLM prompt, dict-iteration-order ends up as section-order. The fresh
in-memory build path and the resume-from-disk load path produce
structurally equivalent payloads with slightly different default
orderings (insertion order vs alphabetical-glob order vs marker-write
order). Without one canonical sort at each boundary, prompts hash to
different cache keys across pause-and-resume even though facts on disk
are unchanged — the bug observed in run yjs6.

The normalizers (`runner._normalize_*`) are the single source of truth
for cross-phase ordering. These tests pin them.
"""
from __future__ import annotations

import json
from pathlib import Path


from engine.content_extractor import (
    EvidenceSpan,
    ExtractedItem,
)
from engine.entities import EntitiesOutput, EntityRecord, RelationEdge, SubjectRef
from engine.actions import Action
from engine.insights import Insight, InsightOutput
from engine.patterns import Pattern
from engine.runner import (
    _load_actions,
    _normalize_action_list,
    _normalize_entities_output,
    _normalize_facts_by_topic,
    _normalize_insight_output,
    _normalize_patterns_by_topic,
    _sort_patterns_within_topic,
)


# ── facts_by_topic ────────────────────────────────────────────────────────


def _item(summary: str, file_path: str, file_offset: int, topics: list[str]):
    return ExtractedItem(
        item_type="fact",
        summary=summary,
        evidence=[EvidenceSpan(
            text=summary,
            source_ref="x",
            file_path=file_path,
            file_offset=file_offset,
        )],
        topics=topics,
    )


class TestNormalizeFactsByTopic:
    def test_topic_keys_alphabetized(self):
        """Fresh-path build inserts topic keys in extract-emit order;
        resume-path load walks `sorted(glob)` alphabetical. Normalizer
        forces alphabetical on both."""
        fresh = {
            "work": [_item("a", "f.txt", 0, ["work"])],
            "health": [_item("b", "f.txt", 100, ["health"])],
            "admin": [_item("c", "f.txt", 200, ["admin"])],
        }
        resume = {
            "admin": [_item("c", "f.txt", 200, ["admin"])],
            "health": [_item("b", "f.txt", 100, ["health"])],
            "work": [_item("a", "f.txt", 0, ["work"])],
        }
        assert list(_normalize_facts_by_topic(fresh).keys()) == [
            "admin", "health", "work",
        ]
        assert list(_normalize_facts_by_topic(resume).keys()) == [
            "admin", "health", "work",
        ]

    def test_items_within_topic_sorted_by_file_offset(self):
        """Items within each topic are sorted by (file_path, file_offset,
        summary) — same key the canonical extract write uses on disk."""
        unsorted = {
            "health": [
                _item("z", "f.txt", 500, ["health"]),
                _item("a", "f.txt", 100, ["health"]),
                _item("m", "f.txt", 300, ["health"]),
            ],
        }
        out = _normalize_facts_by_topic(unsorted)
        assert [it.summary for it in out["health"]] == ["a", "m", "z"]

    def test_idempotent(self):
        """Re-normalizing an already-normalized dict yields equal
        topic order and equal per-topic item order."""
        d = {
            "admin": [_item("a", "f.txt", 0, ["admin"])],
            "health": [_item("b", "f.txt", 50, ["health"])],
        }
        once = _normalize_facts_by_topic(d)
        twice = _normalize_facts_by_topic(once)
        assert list(once.keys()) == list(twice.keys())
        for topic in once:
            assert [i.summary for i in once[topic]] == [i.summary for i in twice[topic]]

    def test_fresh_and_resume_paths_converge(self):
        """The regression contract. Two semantically identical
        payloads with different insertion / glob orderings must
        normalize to the same shape."""
        fresh = {
            "work": [
                _item("w1", "f.txt", 200, ["work"]),
                _item("w0", "f.txt", 50, ["work"]),
            ],
            "admin": [_item("a0", "f.txt", 10, ["admin"])],
        }
        resume = {
            "admin": [_item("a0", "f.txt", 10, ["admin"])],
            "work": [
                _item("w0", "f.txt", 50, ["work"]),
                _item("w1", "f.txt", 200, ["work"]),
            ],
        }
        nf = _normalize_facts_by_topic(fresh)
        nr = _normalize_facts_by_topic(resume)
        assert list(nf.keys()) == list(nr.keys())
        for topic in nf:
            assert [i.summary for i in nf[topic]] == [i.summary for i in nr[topic]]


# ── entities_output ───────────────────────────────────────────────────────


def _erec(canonical_id: str, aliases=(), topics=()):
    return EntityRecord(
        canonical_id=canonical_id,
        canonical_name=canonical_id,
        entity_type="person",
        aliases=list(aliases),
        role="",
        description="",
        mention_count=1,
        topics=list(topics),
        evidence_fact_refs=[],
    )


def _rel(from_id: str, to_id: str, relation: str = "knows"):
    return RelationEdge(
        from_id=from_id,
        to_id=to_id,
        relation=relation,
        confidence=1.0,
        evidence_fact_refs=[],
    )


class TestNormalizeEntitiesOutput:
    def test_entities_sorted_by_canonical_id(self):
        eo = EntitiesOutput(
            subject=None,
            entities=[_erec("zeta"), _erec("alpha"), _erec("mu")],
            relations=[],
        )
        out = _normalize_entities_output(eo)
        assert [e.canonical_id for e in out.entities] == ["alpha", "mu", "zeta"]

    def test_relations_sorted_by_tuple_key(self):
        eo = EntitiesOutput(
            subject=None,
            entities=[],
            relations=[
                _rel("b", "a", "knows"),
                _rel("a", "c", "knows"),
                _rel("a", "b", "trusts"),
                _rel("a", "b", "knows"),
            ],
        )
        out = _normalize_entities_output(eo)
        assert [(r.from_id, r.to_id, r.relation) for r in out.relations] == [
            ("a", "b", "knows"),
            ("a", "b", "trusts"),
            ("a", "c", "knows"),
            ("b", "a", "knows"),
        ]

    def test_per_entity_alias_and_topic_lists_sorted(self):
        """Aliases and topics on each entity are sorted — same content
        regardless of which order they entered the record from."""
        eo = EntitiesOutput(
            subject=None,
            entities=[_erec("x", aliases=["zebra", "apple", "mango"],
                            topics=["work", "admin"])],
            relations=[],
        )
        out = _normalize_entities_output(eo)
        assert out.entities[0].aliases == ["apple", "mango", "zebra"]
        assert out.entities[0].topics == ["admin", "work"]

    def test_subject_passthrough(self):
        """Subject is a single ref — normalizer doesn't touch it."""
        subj = SubjectRef(canonical_id="x", display="X", source="cli")
        eo = EntitiesOutput(subject=subj, entities=[], relations=[])
        out = _normalize_entities_output(eo)
        assert out.subject is subj

    def test_none_passthrough(self):
        """`_load_entities` returns None when the marker doesn't exist
        (e.g. resume before stage 2 finished). Normalizer must handle."""
        assert _normalize_entities_output(None) is None


# ── patterns_by_topic ─────────────────────────────────────────────────────


def _pat(name: str, topic: str, description: str = "", *, n_facts: int = 0, count: int = 1):
    return Pattern(
        name=name,
        description=description,
        domain=topic,
        kind=None,
        count=count,
        source_facts=[(i, 1.0) for i in range(n_facts)],
        hallucinated_ref_count=0,
    )


class TestSortPatternsWithinTopic:
    """Per-topic sort key drives both the on-disk per-topic JSON (via
    `_persist_topic_patterns`) and the in-memory dict consumed by
    insights / actions (via `_normalize_patterns_by_topic`)."""

    def test_fact_count_descending(self):
        pats = [
            _pat("a", "t", n_facts=2),
            _pat("b", "t", n_facts=5),
            _pat("c", "t", n_facts=3),
        ]
        assert [p.name for p in _sort_patterns_within_topic(pats)] == ["b", "c", "a"]

    def test_repros_observed_topic_order(self):
        """nwc4 finance.json was emitted in order [5, 2, 2, 3, 3, 3] —
        approximately-but-not-strictly count-descending. Pinning the
        sort here keeps the bug from regressing."""
        pats = [
            _pat("Frequent loans", "finance", n_facts=5),
            _pat("Asset sales", "finance", n_facts=2),
            _pat("Declining offers", "finance", n_facts=2),
            _pat("Trusts for creditors", "finance", n_facts=3),
            _pat("Regular allowance", "finance", n_facts=3),
            _pat("Negotiating contracts", "finance", n_facts=3),
        ]
        out_counts = [len(p.source_facts) for p in _sort_patterns_within_topic(pats)]
        assert out_counts == sorted(out_counts, reverse=True)
        assert out_counts == [5, 3, 3, 3, 2, 2]

    def test_keys_off_source_facts_not_llm_count(self):
        pats = [
            _pat("a", "t", n_facts=2, count=99),
            _pat("b", "t", n_facts=5, count=1),
        ]
        assert [p.name for p in _sort_patterns_within_topic(pats)] == ["b", "a"]


class TestNormalizePatternsByTopic:
    def test_topic_keys_alphabetized(self):
        d = {
            "work": [_pat("a", "work")],
            "admin": [_pat("b", "admin")],
        }
        assert list(_normalize_patterns_by_topic(d).keys()) == ["admin", "work"]

    def test_patterns_within_topic_sorted_by_fact_count_desc(self):
        d = {
            "health": [
                _pat("a", "health", n_facts=2),
                _pat("b", "health", n_facts=5),
                _pat("c", "health", n_facts=3),
            ],
        }
        out = _normalize_patterns_by_topic(d)
        assert [(p.name, len(p.source_facts)) for p in out["health"]] == [
            ("b", 5), ("c", 3), ("a", 2),
        ]

    def test_ties_broken_by_name(self):
        d = {
            "health": [
                _pat("zebra", "health", n_facts=3),
                _pat("apple", "health", n_facts=3),
                _pat("mango", "health", n_facts=3),
            ],
        }
        out = _normalize_patterns_by_topic(d)
        assert [p.name for p in out["health"]] == ["apple", "mango", "zebra"]

    def test_sort_keys_off_source_facts_not_llm_count(self):
        """`count` is the LLM's self-estimate; `len(source_facts)` is
        the authoritative weight after citation resolution. When they
        diverge (hallucinated/deduped refs), `source_facts` wins."""
        d = {
            "health": [
                _pat("a", "health", n_facts=2, count=99),
                _pat("b", "health", n_facts=5, count=1),
            ],
        }
        out = _normalize_patterns_by_topic(d)
        assert [p.name for p in out["health"]] == ["b", "a"]

    def test_idempotent(self):
        d = {
            "admin": [_pat("a", "admin", n_facts=3), _pat("b", "admin", n_facts=1)],
            "health": [_pat("c", "health", n_facts=2)],
        }
        once = _normalize_patterns_by_topic(d)
        twice = _normalize_patterns_by_topic(once)
        assert list(once.keys()) == list(twice.keys())
        for t in once:
            assert [p.name for p in once[t]] == [p.name for p in twice[t]]


# ── insight_output ────────────────────────────────────────────────────────


def _insight(name: str):
    return Insight(
        name=name,
        description="",
        mechanism="",
        implication="",
        domains=[],
        proposed_actions=[],
        source_patterns=[],
        hallucinated_ref_count=0,
    )


class TestNormalizeInsightOutput:
    def test_emission_order_preserved(self):
        """Insights are NOT reordered: we keep the LLM's emission order
        because we treat it as already ranked by importance. (Contrast
        the other phase-boundary normalizers, which sort by a
        content-derived key.) The old by-name canonicalization is gone
        — alphabetical order carries no importance signal."""
        io = InsightOutput(
            cross_domain=[_insight("z"), _insight("a"), _insight("m")],
            critical=[_insight("y"), _insight("b")],
        )
        out = _normalize_insight_output(io)
        assert [i.name for i in out.cross_domain] == ["z", "a", "m"]
        assert [i.name for i in out.critical] == ["y", "b"]

    def test_none_passthrough(self):
        assert _normalize_insight_output(None) is None


# ── action_list ───────────────────────────────────────────────────────────
#
# Issue #510: the actions stage gained a resume-load branch (load
# action_list from stages/05-actions/phase_1_marker.json when resuming
# past actions, since 06-embeddings made resume_from=="embeddings"
# reachable). The embeddings stage consumes action_list verbatim, so —
# like the sibling stages — the resume-loaded list must be type- and
# order-identical to the fresh `generate_actions()` list or the
# embeddings prompt-cache keys drift across pause-and-resume.


def _action(recommendation: str, *, regret: float, leverage: float):
    return Action(
        recommendation=recommendation,
        objective="obj",
        why="why",
        immediate_action="step",
        habit="habit",
        success_metric="metric",
        horizon="medium",
        review_date="2026-08-01",
        kind="build",
        regret_reduction=regret,
        leverage=leverage,
        consequence=0.1,
        generativity=0.1,
        decisiveness=0.1,
        time_to_feedback=0.1,
        constraint_fit=0.1,
        confidence=0.8,
        source_insights=[("cross_domain", 0, 0.9), ("critical", 1, 0.5)],
        hallucinated_ref_count=2,
    )


def _dump_actions_marker(out_dir: Path, actions: list[Action]) -> None:
    """Serialize exactly as runner's actions phase_1_marker dump does,
    so the loader test pins the on-disk contract, not a paraphrase."""
    d = out_dir / "stages" / "05-actions"
    d.mkdir(parents=True, exist_ok=True)
    (d / "phase_1_marker.json").write_text(json.dumps({
        "actions": [
            {
                "recommendation": a.recommendation, "kind": a.kind,
                "objective": a.objective, "why": a.why,
                "immediate_action": a.immediate_action, "habit": a.habit,
                "success_metric": a.success_metric, "horizon": a.horizon,
                "review_date": a.review_date,
                "regret_reduction": a.regret_reduction, "leverage": a.leverage,
                "consequence": a.consequence, "generativity": a.generativity,
                "decisiveness": a.decisiveness,
                "time_to_feedback": a.time_to_feedback,
                "constraint_fit": a.constraint_fit, "confidence": a.confidence,
                "score": a.score,
                "source_insights": [[k, i, c] for k, i, c in a.source_insights],
                "hallucinated_ref_count": a.hallucinated_ref_count,
            }
            for a in actions
        ],
    }))


class TestNormalizeActionList:
    def test_score_descending(self):
        lo = _action("lo", regret=0.1, leverage=0.1)
        hi = _action("hi", regret=0.9, leverage=0.9)
        mid = _action("mid", regret=0.5, leverage=0.5)
        out = _normalize_action_list([lo, hi, mid])
        assert [a.recommendation for a in out] == ["hi", "mid", "lo"]

    def test_idempotent(self):
        lst = [_action("a", regret=0.2, leverage=0.2),
               _action("b", regret=0.8, leverage=0.8)]
        once = _normalize_action_list(lst)
        twice = _normalize_action_list(once)
        assert [a.recommendation for a in once] == [a.recommendation for a in twice]

    def test_none_passthrough(self):
        assert _normalize_action_list(None) is None


class TestLoadActionsRoundTrip:
    def test_marker_round_trips_type_identical(self, tmp_path):
        """`_load_actions` reconstructs the dataclass list
        `generate_actions` returns: every field equal, `source_insights`
        re-tupled (not left as JSON lists), `score` derived back from
        the scoring dims (not a constructor arg)."""
        fresh = [_action("x", regret=0.7, leverage=0.3)]
        _dump_actions_marker(tmp_path, fresh)
        loaded = _load_actions(tmp_path)
        assert loaded == fresh                       # dataclass field equality
        assert all(isinstance(t, tuple) for t in loaded[0].source_insights)
        assert loaded[0].score == fresh[0].score     # property round-trips

    def test_fresh_and_resume_paths_converge(self, tmp_path):
        """The bug repro at the data layer: a fresh score-descending
        list and the same list reloaded from its marker, both routed
        through `_normalize_action_list`, are identical in shape AND
        order — so the downstream embeddings input + cache keys match
        regardless of fresh-vs-resume."""
        # Fresh path: generate_actions returns score-descending.
        fresh = _normalize_action_list([
            _action("hi", regret=0.9, leverage=0.9),
            _action("mid", regret=0.5, leverage=0.5),
            _action("lo", regret=0.1, leverage=0.1),
        ])
        _dump_actions_marker(tmp_path, fresh)
        resumed = _normalize_action_list(_load_actions(tmp_path))
        assert resumed == fresh
        assert [a.recommendation for a in resumed] == [
            a.recommendation for a in fresh]
