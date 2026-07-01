"""
Unit tests for entities.py — no LLM calls.

Covers the per-entity rewrite:
  - _has_first_person regex precision
  - _normalize_name / _slugify determinism
  - _group_entities deterministic bucketing
  - _resolve_candidates — name resolution + drop-on-unresolved + type
    disambiguation
  - _build_other_catalogs — catalog scoping (id+name+type only, no facts)
  - _pack_batches — sqrt(N) batching, greedy-pack under T_avg×sqrt(N)
  - _split_heavy_entity — split-by-name into K clones
  - _parse_per_entity_response — schema validation, intra-call A→B
    consolidation
  - _deterministic_collapse — load-bearing same-name+type collapse
  - _apply_merges_to_records — LLM fuzzy merge with synthesized_description,
    mention_count summed, descriptions consolidated, bilateral relations kept
  - _resolve_subject — argmax + tie-break + fallback
  - build_context_block — output formatting

Run with:
    cd engine && pytest tests/test_entities.py -v
"""
import json
from unittest.mock import patch




from engine.content_extractor import Entity, EntityRef, EvidenceSpan, ExtractedItem
from engine.entities import EntitiesOutput, EntityRecord, RelationEdge, SubjectRef, _apply_merges_to_records, _build_name_key_map, _build_other_catalogs, _collapse_cross_category_fact_refs, _DEDUPE_CONFIDENCE_FLOOR, _DEDUPE_TRUST_HIGH_CONF, _deterministic_collapse, _fact_provenance_key, _Group, _group_entities, _has_first_person, _name_substantive_tokens, _normalize_name, _pack_batches, _parse_per_entity_response, _passes_name_overlap_gate, _resolve_candidates, _resolve_subject, _slugify, _split_heavy_entity, build_context_block
from engine.llm import Mode


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mk_item(
    summary: str,
    entities: list[tuple[str, str, str]],
    evidence_text: str,
    topics: list[str] | None = None,
    item_type: str = "fact",
    relation_candidate: dict | None = None,
    occurred_at: str | None = None,
) -> ExtractedItem:
    """Build an ExtractedItem. entities: [(name, entity_type, role), ...]"""
    return ExtractedItem(
        item_type=item_type,
        summary=summary,
        evidence=[EvidenceSpan(text=evidence_text, source_ref="test")],
        entities=[
            EntityRef(entity=Entity(name=n, entity_type=t), role=r)
            for n, t, r in entities
        ],
        topics=topics or ["general"],
        tags=[],
        confidence=1.0,
        relation_candidate=relation_candidate,
        occurred_at=occurred_at,
    )


def _mk_provenance_item(
    summary: str,
    topics: list[str],
    *,
    file_path: str = "input_journal.md",
    file_offset: int = 100,
    file_length: int = 40,
    item_type: str = "emotion",
    entity: tuple[str, str, str] = ("the author", "person", "subject"),
) -> ExtractedItem:
    """A fact carrying real source provenance. A fact extracted under
    several categories is fanned out as an identical record per category,
    so every copy shares the same provenance — pass the full `topics` list
    and place the same item under each topic to model the fan-out."""
    n, t, r = entity
    return ExtractedItem(
        item_type=item_type,
        summary=summary,
        evidence=[EvidenceSpan(
            text=summary, source_ref="test",
            file_path=file_path, file_offset=file_offset, file_length=file_length,
        )],
        entities=[EntityRef(entity=Entity(name=n, entity_type=t), role=r)],
        topics=list(topics),
        tags=[],
        confidence=1.0,
        occurred_at=None,
    )


class TestCollapseCrossCategoryFactRefs:
    def test_fanout_collapses_to_first_alpha_category(self):
        # One fact fanned out across two categories -> identical provenance
        # in both files. The entity cites both; collapse keeps one ref,
        # anchored to the alphabetically-first category ("spirituality").
        fact = _mk_provenance_item("repulsion at the Vatican paintings",
                                   ["travel", "spirituality"])
        facts_by_topic = {"spirituality": [fact], "travel": [fact]}
        refs = [("travel", 0), ("spirituality", 0)]
        out = _collapse_cross_category_fact_refs(refs, facts_by_topic)
        assert out == [("spirituality", 0)]

    def test_same_text_different_extraction_stays_separate(self):
        # Same summary text but DIFFERENT provenance (different byte span)
        # = a duplicate-extraction anomaly. Must NOT collapse — both rows
        # stay visible so the anomaly is never hidden.
        a = _mk_provenance_item("x", ["work"], file_offset=10)
        b = _mk_provenance_item("x", ["health"], file_offset=999)
        facts_by_topic = {"work": [a], "health": [b]}
        refs = [("work", 0), ("health", 0)]
        out = _collapse_cross_category_fact_refs(refs, facts_by_topic)
        assert out == [("work", 0), ("health", 0)]

    def test_facts_without_provenance_never_collapse(self):
        # No file_path -> no provenance key -> never merged even if text
        # matches, so an unkeyable pair is never silently collapsed.
        a = _mk_item("x", [("Alice", "person", "subject")], "x", topics=["work"])
        b = _mk_item("x", [("Alice", "person", "subject")], "x", topics=["health"])
        facts_by_topic = {"work": [a], "health": [b]}
        refs = [("work", 0), ("health", 0)]
        out = _collapse_cross_category_fact_refs(refs, facts_by_topic)
        assert out == [("work", 0), ("health", 0)]

    def test_distinct_facts_in_one_category_preserved(self):
        a = _mk_provenance_item("a", ["work"], file_offset=1)
        b = _mk_provenance_item("b", ["work"], file_offset=2)
        facts_by_topic = {"work": [a, b]}
        refs = [("work", 0), ("work", 1)]
        out = _collapse_cross_category_fact_refs(refs, facts_by_topic)
        assert out == [("work", 0), ("work", 1)]

    def test_out_of_range_ref_kept(self):
        facts_by_topic = {"work": []}
        out = _collapse_cross_category_fact_refs([("work", 5)], facts_by_topic)
        assert out == [("work", 5)]


class TestFactProvenanceKey:
    def test_none_without_evidence_file(self):
        it = _mk_item("x", [("Alice", "person", "subject")], "x")
        assert _fact_provenance_key(it) is None

    def test_equal_keys_for_fanout_copies(self):
        a = _mk_provenance_item("x", ["work"])
        b = _mk_provenance_item("x", ["travel"])
        assert _fact_provenance_key(a) is not None
        assert _fact_provenance_key(a) == _fact_provenance_key(b)


# ── _has_first_person ────────────────────────────────────────────────────────


class TestHasFirstPerson:
    def test_standalone_capital_I(self):
        assert _has_first_person("I went home")

    def test_contractions(self):
        assert _has_first_person("I'm tired")
        assert _has_first_person("I've been here")
        assert _has_first_person("I'll go")
        assert _has_first_person("I'd rather not")

    def test_possessives_and_objects(self):
        assert _has_first_person("It's my fault")
        assert _has_first_person("Tom met me there")
        assert _has_first_person("The car is mine")
        assert _has_first_person("I hurt myself")

    def test_rejects_lowercase_i_in_typos(self):
        assert not _has_first_person("i went home")

    def test_rejects_word_containing_I(self):
        assert not _has_first_person("Ian went home")
        assert not _has_first_person("Mumbai is hot")

    def test_rejects_empty(self):
        assert not _has_first_person("")
        assert not _has_first_person("nothing personal here")


# ── _normalize_name / _slugify ───────────────────────────────────────────────


class TestNormalizeAndSlug:
    def test_case_fold(self):
        assert _normalize_name("John") == "john"
        assert _normalize_name("JOHN DOE") == "john doe"

    def test_strip_accents(self):
        assert _normalize_name("Mihăila") == _normalize_name("Mihaila")
        assert _normalize_name("Café") == _normalize_name("Cafe")

    def test_collapse_whitespace_punct(self):
        assert _normalize_name("  Alice  Smith  ") == "alice smith"
        assert _normalize_name("Alice-Smith") == "alice smith"

    def test_slugify(self):
        assert _slugify("John Doe") == "john-doe"
        assert _slugify("Alice-Smith") == "alice-smith"
        assert _slugify("  ") == "entity"


# ── _le1_distance (Damerau-Levenshtein-1) ──────────────────────────────────


class TestLe1Distance:
    """Edit-distance-1 helper used by the entities dedupe overlap gate.
    Damerau-Levenshtein semantics: insert / delete / substitute / transpose
    each count as one edit."""

    def test_equal_strings(self):
        from engine.entities import _le1_distance
        assert _le1_distance("smith", "smith") is True

    def test_substitution_one_char(self):
        from engine.entities import _le1_distance
        assert _le1_distance("smith", "smyth") is True

    def test_insertion_one_char(self):
        from engine.entities import _le1_distance
        assert _le1_distance("brunker", "bruncker") is True

    def test_deletion_one_char(self):
        from engine.entities import _le1_distance
        assert _le1_distance("bruncker", "brunker") is True

    def test_adjacent_transposition(self):
        from engine.entities import _le1_distance
        # Damerau extension: a single adjacent-char swap counts as ONE
        # edit. Plain Levenshtein would call these distance 2 and reject.
        assert _le1_distance("smith", "smtih") is True
        assert _le1_distance("brian", "brain") is True
        assert _le1_distance("carol", "carlo") is True

    def test_two_substitutions_rejected(self):
        from engine.entities import _le1_distance
        # Two non-adjacent edits → not within distance 1.
        assert _le1_distance("smith", "snyto") is False

    def test_length_diff_two_rejected(self):
        from engine.entities import _le1_distance
        assert _le1_distance("a", "abc") is False


# ── _group_entities ──────────────────────────────────────────────────────────


class TestGroupEntities:
    def test_groups_by_normalized_name_plus_type(self):
        facts = {
            "work": [
                _mk_item("Alice signed a contract",
                         [("Alice", "person", "subject")],
                         "Alice signed the contract."),
                _mk_item("alice reviewed the draft",
                         [("alice", "person", "subject")],
                         "alice reviewed the draft."),
            ],
        }
        groups = _group_entities(facts)
        assert len(groups) == 1
        g = groups[0]
        assert g.canonical_name == "Alice"
        assert g.mention_count == 2
        assert "alice" in g.aliases
        assert "Alice" in g.aliases

    def test_different_types_stay_separate(self):
        facts = {"t": [
            _mk_item("a", [("Google", "org", "mentioned")], "At Google"),
            _mk_item("a", [("Google", "place", "mentioned")], "Near Google HQ"),
        ]}
        groups = _group_entities(facts)
        assert len(groups) == 2

    def test_evidence_refs_preserved(self):
        facts = {"t": [
            _mk_item("a", [("Alice", "person", "subject")], "Alice 1"),
            _mk_item("b", [("Alice", "person", "subject")], "Alice 2"),
        ]}
        groups = _group_entities(facts)
        refs = groups[0].evidence_fact_refs
        assert ("t", 0) in refs
        assert ("t", 1) in refs

    def test_keeps_all_facts_no_sampling(self):
        """Per-entity rewrite drops the _SAMPLE_FACTS_PER_GROUP cap.
        Each group must retain ALL its facts."""
        facts = {"t": [
            _mk_item(f"Alice did thing {i}", [("Alice", "person", "subject")],
                     f"Alice did thing {i}.")
            for i in range(20)
        ]}
        groups = _group_entities(facts)
        assert len(groups) == 1
        # mention_count counts EntityRefs, facts list counts how many
        # times this group appears in source facts.
        assert len(groups[0].facts) == 20


# ── _resolve_candidates ──────────────────────────────────────────────────────


class TestResolveCandidates:
    def test_resolves_unambiguous(self):
        facts = {"t": [
            _mk_item("Alice met Bob",
                     [("Alice", "person", "subject"),
                      ("Bob", "person", "object")],
                     "Alice met Bob.",
                     relation_candidate={
                         "from": "Alice", "to": "Bob",
                         "verb": "met_with", "confidence": 0.9,
                     }),
        ]}
        groups = _group_entities(facts)
        cbg, n_resolved, n_dropped = _resolve_candidates(facts, groups)
        assert n_resolved == 1
        assert n_dropped == 0
        # The single candidate is indexed under both endpoints.
        assert sum(len(v) for v in cbg.values()) == 2

    def test_drops_unresolvable(self):
        facts = {"t": [
            _mk_item("Alice met someone",
                     [("Alice", "person", "subject")],
                     "Alice met X.",
                     relation_candidate={
                         "from": "Alice", "to": "MysteryPerson",
                         "verb": "met_with", "confidence": 0.9,
                     }),
        ]}
        groups = _group_entities(facts)
        cbg, n_resolved, n_dropped = _resolve_candidates(facts, groups)
        assert n_resolved == 0
        assert n_dropped == 1

    def test_disambiguates_by_type(self):
        # Two entities normalize to the same name but have different types.
        facts = {"t": [
            _mk_item("a", [("Apple", "org", "mentioned")], "Apple Inc."),
            _mk_item("b", [("Apple", "place", "mentioned")], "Apple Park."),
            _mk_item("c",
                     [("Alice", "person", "subject"),
                      ("Apple", "org", "mentioned")],
                     "Alice works at Apple.",
                     relation_candidate={
                         "from": "Alice", "to": "Apple",
                         "verb": "employed_by", "confidence": 1.0,
                     }),
        ]}
        groups = _group_entities(facts)
        cbg, n_resolved, n_dropped = _resolve_candidates(facts, groups)
        assert n_resolved == 1
        # The resolved Apple should be the org, not the place.
        # Both groups received the candidate; check that the org's
        # canonical_name appears in candidates_by_gid keys.
        apple_org_gid = next(
            (g.gid for g in groups
             if g.canonical_name == "Apple" and g.entity_type == "org"),
            None,
        )
        assert apple_org_gid is not None
        assert apple_org_gid in cbg


# ── _build_other_catalogs ────────────────────────────────────────────────────


class TestBuildOtherCatalogs:
    def test_catalog_id_name_type_only(self):
        facts = {"t": [
            _mk_item("Alice met Bob and Carol",
                     [("Alice", "person", "subject"),
                      ("Bob", "person", "object"),
                      ("Carol", "person", "mentioned")],
                     "Alice met Bob and Carol."),
        ]}
        groups = _group_entities(facts)
        _build_other_catalogs(facts, groups)
        alice = next(g for g in groups if g.canonical_name == "Alice")
        # Catalog must contain Bob and Carol (gids), not Alice herself.
        assert len(alice.other_catalog) == 2
        # Catalog values are names; type lookup goes through groups_by_gid.
        assert "Bob" in alice.other_catalog.values()
        assert "Carol" in alice.other_catalog.values()


# ── _pack_batches sqrt(N) shape ──────────────────────────────────────────────


class TestPackBatches:
    def _make_facts(self, n: int, facts_per: int = 1) -> dict:
        return {"general": [
            _mk_item(
                f"Person{i:03d} did thing {k}",
                [(f"Person{i:03d}", "person", "mentioned")],
                f"Person{i:03d} did something on day {k}.",
            )
            for i in range(n) for k in range(facts_per)
        ]}

    def test_small_corpus_one_batch(self):
        """A 5-entity corpus packs into ~1 batch (sqrt(5) ≈ 2.2 but the
        floor of 4000 tokens / budget keeps it at 1 batch easily)."""
        facts = self._make_facts(5)
        groups = _group_entities(facts)
        cbg, _, _ = _resolve_candidates(facts, groups)
        _build_other_catalogs(facts, groups)
        groups_by_gid = {g.gid: g for g in groups}
        batches = _pack_batches(groups, cbg, groups_by_gid, Mode.TEE)
        assert len(batches) >= 1
        assert sum(len(b) for b in batches) >= len(groups)

    def test_narrator_dominates_does_not_explode_batch_count(self):
        """Regression for the n_clones=435 bug observed on Pepys.

        Setup: one entity (`Narrator`) appearing in 200 facts alongside
        many co-mentioned entities (so the narrator's render has a
        large header AND a large facts list). Pre-fix: split-by-name
        produced one clone per fact and packed them solo, exploding
        the batch count to ~200. Post-fix: heavy entities ride the
        chunk_cap as a SOLO batch, so Pepys-shaped narrators add a
        small constant of batches (1 per heavy entity, not N).
        """
        facts = {"general": []}
        # 50 light co-entities, each mentioned in 1 fact alongside the
        # narrator. This bloats Narrator's other_catalog so its render
        # has a heavy header.
        for i in range(50):
            facts["general"].append(_mk_item(
                f"Narrator did X with Person{i:03d}",
                [("Narrator", "person", "subject"),
                 (f"Person{i:03d}", "person", "object")],
                f"Narrator did thing with Person{i:03d}.",
            ))
        # 150 more facts that mention only the narrator + a small
        # rotating cast → narrator's facts list becomes large.
        for i in range(150):
            facts["general"].append(_mk_item(
                f"Narrator did internal-thing-{i}",
                [("Narrator", "person", "subject"),
                 (f"Person{i % 50:03d}", "person", "mentioned")],
                f"Narrator reflected on internal thing {i}.",
            ))
        groups = _group_entities(facts)
        cbg, _, _ = _resolve_candidates(facts, groups)
        _build_other_catalogs(facts, groups)
        groups_by_gid = {g.gid: g for g in groups}
        batches = _pack_batches(groups, cbg, groups_by_gid, Mode.TEE)
        # Tight bound: heavy narrators get 1 solo batch, the 50 light
        # co-entities pack into ≈ sqrt(50) ≈ 7 batches → total ≤ 15.
        # Pre-fix this returned ~200.
        assert len(batches) <= 20, (
            f"narrator-heavy corpus produced {len(batches)} batches; "
            f"the heavy-entity solo path should keep it near sqrt(N)"
        )
        # Narrator must appear exactly once (no clones explosion).
        narrator = next(g for g in groups if g.canonical_name == "Narrator")
        narrator_appearances = sum(
            1 for b in batches for g in b if g.gid == narrator.gid
        )
        narrator_clone_count = sum(
            1 for b in batches for g in b
            if g.gid.startswith(narrator.gid + "c")
        )
        assert narrator_appearances == 1, (
            f"narrator should be in exactly 1 solo batch, got "
            f"{narrator_appearances} appearances"
        )
        assert narrator_clone_count == 0, (
            f"narrator was split into {narrator_clone_count} clones; "
            "heavy-entity-but-not-ultra-heavy should NOT clone"
        )

    def test_heavy_entity_ships_solo_under_total_cap(self):
        """Single-group exception: a heavy entity whose block exceeds
        `_BATCH_TOTAL_TOKEN_CAP` (the per-call cap that bounds annotation
        invalidation footprint) but fits in the raw `chunk_cap_for_stage`
        ships SOLO at its natural size, never cloned. The cap clamps the
        LIGHT-packing budget for multi-group batches only; it doesn't
        fragment an entity that's the only group in its call.

        Setup: a narrator-shaped corpus large enough that the corpus
        total exceeds chunk_cap_for_stage("entities") on TEE — so the
        cap engages — combined with light co-entities packed under the
        cap. The narrator must stay 1 group across 1 batch.
        """
        from engine.splitter import _BATCH_TOTAL_TOKEN_CAP  # type: ignore  # noqa: F401
        facts = {"general": []}
        # 60 light entities, each in 6 facts alongside Narrator. The
        # narrator's own facts list and other_catalog accumulate to a
        # heavy block size. With 60 × 6 = 360 narrator-facts each touching
        # a co-entity, the narrator's block exceeds the new cap easily.
        for i in range(60):
            for k in range(6):
                facts["general"].append(_mk_item(
                    f"Narrator did thing {k} with Person{i:03d}",
                    [("Narrator", "person", "subject"),
                     (f"Person{i:03d}", "person", "object")],
                    f"Narrator interaction {k} with Person{i:03d}.",
                ))
        groups = _group_entities(facts)
        cbg, _, _ = _resolve_candidates(facts, groups)
        _build_other_catalogs(facts, groups)
        groups_by_gid = {g.gid: g for g in groups}
        batches = _pack_batches(groups, cbg, groups_by_gid, Mode.TEE)

        narrator = next(g for g in groups if g.canonical_name == "Narrator")
        # Narrator appears exactly once — no clone explosion from the cap
        # being misapplied to a solo entity.
        narrator_appearances = sum(
            1 for b in batches for g in b if g.gid == narrator.gid
        )
        narrator_clones = sum(
            1 for b in batches for g in b
            if g.gid.startswith(narrator.gid + "c")
        )
        assert narrator_appearances == 1, (
            f"Narrator should ship in exactly 1 solo batch under the cap; "
            f"got {narrator_appearances} appearances."
        )
        assert narrator_clones == 0, (
            f"Narrator got cloned into {narrator_clones} pieces; the cap "
            f"should not fragment a solo entity (single-group exception)."
        )
        # Every group routes somewhere.
        all_gids = {g.gid for b in batches for g in b}
        for g in groups:
            assert g.gid in all_gids, f"missing {g.gid}"

    def test_180_entities_batches_sqrt_ish(self):
        """The Pepys-shaped corpus (~180 entities) packs into roughly
        sqrt(N) batches. We allow a generous range because exact count
        depends on average block size; the point is it's NOT 1 mega-
        batch and NOT N tiny batches."""
        facts = self._make_facts(180, facts_per=3)
        groups = _group_entities(facts)
        cbg, _, _ = _resolve_candidates(facts, groups)
        _build_other_catalogs(facts, groups)
        groups_by_gid = {g.gid: g for g in groups}
        batches = _pack_batches(groups, cbg, groups_by_gid, Mode.TEE)
        # sqrt(180) ≈ 13.4; allow [3, 30] as a loose bound.
        assert 3 <= len(batches) <= 30, f"got {len(batches)} batches"
        # Every group routed somewhere.
        all_gids = {g.gid for b in batches for g in b}
        for g in groups:
            assert g.gid in all_gids, f"missing {g.gid}"

    def test_ultra_heavy_does_not_overfragment_total_batches(self):
        """End-to-end symptom: an ultra-heavy entity (own render >
        chunk_cap) on a large catalog must NOT blow up `total_batches`.

        Protects against: the m7pp regression where a halved entities
        chunk_cap (an entities-stage model flip to a smaller-window model)
        pushed narrators over the ultra-heavy line, and unscoped
        clone-splitting bisected each to one fact per clone — 218 → 21,831
        computed batches. With the catalog scoped per clone, the same
        corpus stays in the hundreds.

        How to evaluate: force a small entities chunk_cap so the narrator
        is ultra-heavy on a modest corpus, then assert the total batch
        count is bounded (near sqrt(N) + a small clone constant) and the
        narrator splits into a handful of clones, not one-per-fact.
        """
        n = 200
        facts = {"general": [
            _mk_item(
                f"Narrator did thing {i} with Person{i:04d}",
                [("Narrator", "person", "subject"),
                 (f"Person{i:04d}", "person", "object")],
                f"Narrator interaction {i}.",
            )
            for i in range(n)
        ]}
        groups = _group_entities(facts)
        cbg, _, _ = _resolve_candidates(facts, groups)
        _build_other_catalogs(facts, groups)
        groups_by_gid = {g.gid: g for g in groups}
        narrator = next(g for g in groups if g.canonical_name == "Narrator")

        # Force a small entities chunk_cap so the narrator (full render
        # ~6k tokens) is ultra-heavy and takes the clone-split path.
        with patch("engine.entities._llm_chunk_cap_for_stage", return_value=2500):
            batches = _pack_batches(
                groups, cbg, groups_by_gid, Mode.TEE, facts_by_topic=facts,
            )

        narrator_clones = sum(
            1 for b in batches for g in b
            if g.gid.startswith(narrator.gid + "c")
        )
        # The narrator IS ultra-heavy here, so it DOES clone — but into a
        # handful, not one-per-fact.
        assert 2 <= narrator_clones <= 20, (
            f"narrator split into {narrator_clones} clones; scoped split "
            f"should keep it small, pre-fix this was ~{n}"
        )
        # Total stays bounded — pre-fix this corpus computed ~{n}+ batches.
        assert len(batches) <= 40, (
            f"ultra-heavy entity over-fragmented to {len(batches)} batches"
        )


# ── _split_heavy_entity ──────────────────────────────────────────────────────


class TestSplitHeavyEntity:
    def test_splits_into_clones_under_budget(self):
        # Synthesize a single group with many large facts.
        facts = {"t": [
            _mk_item(
                f"Alice did thing {i}: " + ("very long fact text " * 60),
                [("Alice", "person", "subject")],
                "Alice did long thing.",
                occurred_at=f"2025-01-{(i % 28) + 1:02d}",
            )
            for i in range(40)
        ]}
        groups = _group_entities(facts)
        _build_other_catalogs(facts, groups)
        assert len(groups) == 1
        g = groups[0]
        groups_by_gid = {g.gid: g}
        cbg = {}
        # Pick a tiny budget to force a real split.
        budget = 800
        clones = _split_heavy_entity(g, cbg, groups_by_gid, budget)
        assert len(clones) >= 2
        # Every clone shares canonical_name + type with parent
        # (load-bearing for deterministic collapse later).
        for c in clones:
            assert c.canonical_name == g.canonical_name
            assert c.entity_type == g.entity_type
        # Each clone's facts subset is non-empty.
        for c in clones:
            assert len(c.facts) >= 1

    def test_clones_share_canonical_name_for_collapse(self):
        """The deterministic floor in _deterministic_collapse depends on
        clones having identical (canonical_name, entity_type)."""
        facts = {"t": [
            _mk_item(f"Alice {i}", [("Alice", "person", "subject")],
                     f"Alice did {i} times.")
            for i in range(10)
        ]}
        groups = _group_entities(facts)
        _build_other_catalogs(facts, groups)
        g = groups[0]
        groups_by_gid = {g.gid: g}
        clones = _split_heavy_entity(g, {}, groups_by_gid, budget_tokens=100)
        names = {(c.canonical_name, c.entity_type) for c in clones}
        # All clones reduce to the SAME (name, type) — that's the
        # promise the deterministic collapse relies on.
        assert len(names) == 1

    def test_clones_carry_explicit_parent_gid(self):
        """Phase 2's runner-side enrichment resolves clone gids back to
        their parent's canonical_id via this field. Originals leave it
        empty (so `parent_gid or gid` self-resolves them); clones carry
        the parent's gid verbatim. Pinning this invariant so a future
        rename of the gid scheme doesn't silently misroute clones."""
        facts = {"t": [
            _mk_item(f"Alice {i}", [("Alice", "person", "subject")],
                     f"Alice did {i} times.")
            for i in range(10)
        ]}
        groups = _group_entities(facts)
        _build_other_catalogs(facts, groups)
        g = groups[0]
        # Originals: parent_gid is empty.
        assert g.parent_gid == ""
        groups_by_gid = {g.gid: g}
        clones = _split_heavy_entity(g, {}, groups_by_gid, budget_tokens=100)
        assert len(clones) >= 2
        for c in clones:
            assert c.parent_gid == g.gid, (
                f"clone {c.gid!r} must carry parent_gid={g.gid!r}; "
                f"got {c.parent_gid!r}"
            )

    def test_split_scopes_catalog_to_each_clones_facts(self):
        """Regression for the ultra-heavy clone over-fragmentation cliff.

        Protects against: an ultra-heavy narrator whose OTHER-entities
        catalog is large (co-mentioned with many distinct entities across
        its facts) bisecting toward one-clone-per-fact. The catalog does
        NOT shrink with facts and GROWS with corpus size, so on a large
        corpus an unscoped clone overflows even at one fact — and the
        split runs to the floor (6 narrators × ~3.6k clones = ~21.7k
        batches on the m7pp corpus).

        How to evaluate: the SAME split, run unscoped (no by_key/facts →
        full catalog on every clone, the pre-fix behavior) vs scoped (the
        fix → each clone's catalog rebuilt from its own facts), at a
        budget the full catalog alone overflows. The scoped split must
        produce dramatically fewer clones AND each scoped clone must
        actually fit the budget (proof the split is now effective).
        """
        from engine.entities import _render_entity_block
        from engine.tokens import count_tokens as _ct
        # Narrator co-mentioned with 200 DISTINCT others, one per fact:
        # full catalog = 200 rows (constant per clone), but each fact
        # touches only one other.
        n = 200
        facts = {"general": [
            _mk_item(
                f"Narrator did thing {i} with Person{i:04d}",
                [("Narrator", "person", "subject"),
                 (f"Person{i:04d}", "person", "object")],
                f"Narrator interaction {i}.",
            )
            for i in range(n)
        ]}
        groups = _group_entities(facts)
        cbg, _, _ = _resolve_candidates(facts, groups)
        _build_other_catalogs(facts, groups)
        groups_by_gid = {g.gid: g for g in groups}
        by_key = _build_name_key_map(groups)
        narrator = next(g for g in groups if g.canonical_name == "Narrator")
        assert len(narrator.other_catalog) == n  # full catalog is large

        # Budget the full 200-row catalog overflows even at one fact.
        budget = 2000
        unscoped = _split_heavy_entity(
            narrator, dict(cbg), dict(groups_by_gid), budget,
        )
        scoped = _split_heavy_entity(
            narrator, dict(cbg), dict(groups_by_gid), budget,
            by_key=by_key, facts_by_topic=facts,
        )
        # Pre-fix path bisects to the one-fact-per-clone floor.
        assert len(unscoped) >= n // 2, (
            f"unscoped split should explode toward the floor; "
            f"got {len(unscoped)} clones"
        )
        # The fix keeps it small AND > 10x fewer than the pre-fix path.
        assert len(scoped) <= 20, (
            f"scoped split should stay bounded; got {len(scoped)} clones"
        )
        assert len(unscoped) > 10 * len(scoped), (
            f"scoping must dramatically reduce clones: "
            f"unscoped={len(unscoped)} scoped={len(scoped)}"
        )
        # Every scoped clone actually fits the budget — fact-splitting is
        # effective again (it carries only its own facts' catalog).
        for c in scoped:
            gbg = {**groups_by_gid, c.gid: c}
            block = _ct(_render_entity_block(c, [], gbg))
            assert block <= budget, (
                f"scoped clone {c.gid} renders {block}t > budget {budget}t"
            )


# ── _parse_per_entity_response ────────────────────────────────────────────────


class TestParsePerEntityResponse:
    def _batch(self) -> list[_Group]:
        return [
            _Group(gid="g1", canonical_name="Alice", entity_type="person",
                   mention_count=5, aliases={"Alice"}),
            _Group(gid="g2", canonical_name="Bob", entity_type="person",
                   mention_count=3, aliases={"Bob"}),
        ]

    def test_intra_call_consolidation_one_relation_per_pair(self):
        """If the LLM emits 5 different verbs for the same (X→Y) pair
        in one call, the parser MUST keep only ONE."""
        batch = self._batch()
        groups_by_gid = {g.gid: g for g in batch}
        raw = json.dumps({
            "entities": [
                {
                    "group_id": "g1",
                    "canonical_name": "Alice",
                    "role": "subject",
                    "description": "Author.",
                    "is_subject_likelihood": 0.95,
                    "relations": [
                        {"to_id": "g2", "verb": "met_with", "confidence": 0.9},
                        {"to_id": "g2", "verb": "called", "confidence": 0.7},
                        {"to_id": "g2", "verb": "wrote_to", "confidence": 0.5},
                    ],
                },
            ],
        })
        out, parse_error = _parse_per_entity_response(raw, batch, groups_by_gid)
        assert parse_error is False  # valid JSON parsed cleanly
        rels = out["g1"]["relations"]
        # First emitted wins; the 4 duplicates collapse to 1.
        assert len(rels) == 1
        assert rels[0]["to_id"] == "g2"
        assert rels[0]["verb"] == "met_with"

    def test_drops_invalid_to_ids(self):
        batch = self._batch()
        groups_by_gid = {g.gid: g for g in batch}
        raw = json.dumps({
            "entities": [
                {
                    "group_id": "g1",
                    "canonical_name": "Alice",
                    "role": "subject",
                    "description": "Author.",
                    "is_subject_likelihood": 0.95,
                    "relations": [
                        {"to_id": "ghost", "verb": "met_with", "confidence": 0.9},
                        {"to_id": "g1", "verb": "self_loop", "confidence": 0.9},
                        {"to_id": "g2", "verb": "met_with", "confidence": 0.9},
                    ],
                },
            ],
        })
        out, parse_error = _parse_per_entity_response(raw, batch, groups_by_gid)
        assert parse_error is False  # valid JSON parsed cleanly
        rels = out["g1"]["relations"]
        assert len(rels) == 1  # only the valid g2 relation survives
        assert rels[0]["to_id"] == "g2"

    def test_unparseable_returns_empty(self):
        batch = self._batch()
        groups_by_gid = {g.gid: g for g in batch}
        out, parse_error = _parse_per_entity_response(
            "not json", batch, groups_by_gid)
        assert out == {}
        # Non-empty content that didn't survive JSON parsing → parse_error
        # True, so the sizing cascade routes to halve (vs parseable-empty).
        assert parse_error is True

    def test_clamps_likelihood(self):
        batch = self._batch()
        groups_by_gid = {g.gid: g for g in batch}
        raw = json.dumps({
            "entities": [
                {"group_id": "g1", "is_subject_likelihood": 99.0,
                 "canonical_name": "A", "role": "x", "description": "y",
                 "relations": []},
                {"group_id": "g2", "is_subject_likelihood": -1.0,
                 "canonical_name": "B", "role": "y", "description": "z",
                 "relations": []},
            ],
        })
        out, parse_error = _parse_per_entity_response(raw, batch, groups_by_gid)
        assert parse_error is False  # valid JSON parsed cleanly
        assert 0.0 <= out["g1"]["is_subject_likelihood"] <= 1.0
        assert 0.0 <= out["g2"]["is_subject_likelihood"] <= 1.0

    def test_parseable_empty_is_not_parse_error(self):
        """The reason this parser returns a (out, parse_error) tuple:
        parseable JSON that yields zero entities must NOT set parse_error
        — only malformed JSON does. The sizing cascade halves on
        parse_error; a model that legitimately returned no entities must
        not be mistaken for a malformed response and force a halve."""
        batch = self._batch()
        groups_by_gid = {g.gid: g for g in batch}
        out, parse_error = _parse_per_entity_response(
            json.dumps({"entities": []}), batch, groups_by_gid)
        assert out == {}
        assert parse_error is False


# ── _deterministic_collapse ──────────────────────────────────────────────────


class TestDeterministicCollapse:
    def test_same_name_same_type_collapsed(self):
        """Load-bearing: heavy-entity clones produce records that share
        (canonical_name, entity_type). Collapse must merge them
        unconditionally."""
        records = [
            EntityRecord(canonical_id="alice", canonical_name="Alice",
                         entity_type="person", description="a desc",
                         mention_count=5, evidence_fact_refs=[("t", 0)]),
            EntityRecord(canonical_id="alice-2", canonical_name="Alice",
                         entity_type="person", description="b desc",
                         mention_count=3, evidence_fact_refs=[("t", 1)]),
        ]
        merged, _ = _deterministic_collapse(records, [], Mode.TEE)
        assert len(merged) == 1
        assert merged[0].mention_count == 8  # summed
        # Both descriptions consolidated, not just first non-empty.
        assert "a desc" in merged[0].description
        assert "b desc" in merged[0].description
        # Evidence union.
        assert ("t", 0) in merged[0].evidence_fact_refs
        assert ("t", 1) in merged[0].evidence_fact_refs

    def test_different_types_stay_separate(self):
        records = [
            EntityRecord(canonical_id="apple-1", canonical_name="Apple",
                         entity_type="org", mention_count=5),
            EntityRecord(canonical_id="apple-2", canonical_name="Apple",
                         entity_type="place", mention_count=3),
        ]
        merged, _ = _deterministic_collapse(records, [], Mode.TEE)
        assert len(merged) == 2

    def test_no_op_on_unique_records(self):
        records = [
            EntityRecord(canonical_id="alice", canonical_name="Alice",
                         entity_type="person", mention_count=5),
            EntityRecord(canonical_id="bob", canonical_name="Bob",
                         entity_type="person", mention_count=3),
        ]
        merged, _ = _deterministic_collapse(records, [], Mode.TEE)
        assert len(merged) == 2


# ── Dedupe gates: confidence floor + name-overlap ────────────────────────────


class TestNameOverlapGate:
    """Regression for the post-15e5cb7 over-merge cases. Bad merges
    in that run: `Sir W. Pen` ↔ `Sir William Batten`, `King` ↔
    `Queen`, `Captain Cocke` ↔ `Sir W. Warren`, etc. The
    deterministic name-overlap gate must reject all of these while
    still allowing legitimate fuzzy merges."""

    def test_substantive_tokens_strips_honorifics(self):
        assert _name_substantive_tokens("Sir W. Pen") == {"w", "pen"}
        assert _name_substantive_tokens("Sir William Penn") == {"william", "penn"}
        assert _name_substantive_tokens("Lord Sandwich") == {"sandwich"}
        assert _name_substantive_tokens("Duke of York") == {"york"}
        assert _name_substantive_tokens("the King") == set()  # all honorifics

    def test_rejects_distinct_surnames_under_same_title(self):
        # The headline pre-fix bad merges:
        assert not _passes_name_overlap_gate("Sir W. Pen", "Sir William Batten")
        assert not _passes_name_overlap_gate("Sir Philip Warwicke", "Sir W. Pen")
        assert not _passes_name_overlap_gate("Lord Sandwich", "Lord Bruncker")

    def test_rejects_role_word_pairs(self):
        assert not _passes_name_overlap_gate("King", "Queen")
        assert not _passes_name_overlap_gate("Father", "Mother")

    def test_rejects_unrelated_specific_names(self):
        assert not _passes_name_overlap_gate("Captain Cocke", "Sir W. Warren")
        assert not _passes_name_overlap_gate("John Pepys", "Balthasar St. Michel")
        assert not _passes_name_overlap_gate("Captain Cocke", "Dutch fleet")

    def test_accepts_initial_to_full_name(self):
        # The intended merge case: "W." matches "William".
        assert _passes_name_overlap_gate("Sir W. Pen", "Sir William Pen")
        assert _passes_name_overlap_gate("Sir W. Pen", "Sir William Penn")  # surname overlap on "pen"
        assert _passes_name_overlap_gate("Dr. Smith", "Smith")
        # Initial that DOESN'T match a token shouldn't cross-pollinate:
        assert not _passes_name_overlap_gate("Sir W. Pen", "Sir Batten")

    def test_accepts_substring_token_overlap(self):
        assert _passes_name_overlap_gate("Smith", "Dr. Smith")
        assert _passes_name_overlap_gate("Jane Doe", "Jane")
        assert _passes_name_overlap_gate("St. John", "St John")  # accent/punct variants

    def test_short_form_relationship_word_lenient(self):
        # Relationship words in the allowlist (mom/dad/wife/etc.) ARE
        # allowed to merge with specific names. Mirrors the brief's
        # explicitly-endorsed "Mom" / "Jane Doe" case.
        assert _passes_name_overlap_gate("Mom", "Jane Doe")
        assert _passes_name_overlap_gate("Dad", "John Smith")
        assert _passes_name_overlap_gate("Wife", "Elizabeth")
        assert _passes_name_overlap_gate("Brother", "John")
        # Common short surnames NOT in the allowlist: Snow, Cook, Ford
        # do not qualify for short-form leniency.
        assert not _passes_name_overlap_gate("Snow", "Smith")  # both surnames, not relationship words
        assert not _passes_name_overlap_gate("Cook", "Smith")
        # NOT two role words (caught by honorific stripping above).
        assert not _passes_name_overlap_gate("Mother", "Father")

    def test_rejects_no_substantive_tokens(self):
        # Two honorific-only names → no anchors → reject.
        assert not _passes_name_overlap_gate("the King", "Lord")

    def test_honorific_only_fallback_accepts(self):
        """When one side has only honorific tokens (e.g. "King"
        alone), fall back to unstripped-token overlap. "King" +
        "King Charles II" share "king" unstripped — clearly the
        category-name vs specific-name pattern."""
        assert _passes_name_overlap_gate("King", "King Charles II")
        assert _passes_name_overlap_gate("Duke", "Duke of York")
        assert _passes_name_overlap_gate("Lord", "Lord Sandwich")
        # But two honorific-only names with NO unstripped overlap
        # still reject.
        assert not _passes_name_overlap_gate("King", "Queen")
        assert not _passes_name_overlap_gate("Duke", "Lord")

    def test_both_honorific_only_with_unstripped_overlap_accepts(self):
        """Regression: "King" + "the King" both strip to ∅ but
        share "king" unstripped — same role term repeated. Earlier
        the gate hard-rejected when both sides had no substantive
        tokens; now it falls back to unstripped overlap regardless."""
        assert _passes_name_overlap_gate("King", "the King")
        assert _passes_name_overlap_gate("Duke", "the Duke")
        # Still reject when there's no unstripped overlap either:
        assert not _passes_name_overlap_gate("King", "Lord")
        assert not _passes_name_overlap_gate("the King", "the Queen")


class TestDedupeTrustHighConfThreshold:
    """At conf ≥ _DEDUPE_TRUST_HIGH_CONF, the LLM's judgment
    overrides the deterministic name-overlap gate for SAME-TYPE
    merges. Cross-type merges do NOT get the high-conf bypass —
    name overlap is the safeguard against the bkwp/spu2 person↔org
    false-merge class (those had near-zero token overlap)."""

    def test_threshold_value(self):
        # Document the active threshold so future tunes surface here.
        assert 0.85 <= _DEDUPE_TRUST_HIGH_CONF <= 1.0
        assert _DEDUPE_TRUST_HIGH_CONF >= _DEDUPE_CONFIDENCE_FLOOR

    def test_typo_variants_accepted(self):
        """Spelling typos with 1-char difference on long tokens
        should match: 'Bruncker' + 'Brunker' is the same name with
        a missing 'c'."""
        assert _passes_name_overlap_gate("Lord Bruncker", "Lord Brunker")
        # Short tokens (< 5 chars) don't trigger the typo-tolerance
        # path because false-positive risk is too high.
        assert not _passes_name_overlap_gate("Anna", "Anne")  # both 4 chars, no leniency for plain names
        # Different surnames with same first 3 chars and same length
        # ARE flagged as typo variants by my rule (acceptable cost —
        # downstream confidence floor catches false-positive merges):
        assert _passes_name_overlap_gate("Smith", "Smyth")  # typo variant
        # But same first 3 chars + length differs by >1 → not similar
        assert not _passes_name_overlap_gate("Bruncker", "Brun")


class TestApplyMergesToRecordsDedupeGates:
    """The new dedupe pipeline drops merges below the confidence floor
    or failing the name-overlap gate BEFORE invoking
    `_apply_merges_to_records`. Test by simulating both layers' outputs
    inline since the gates live in `_llm_dedupe`."""

    def test_confidence_floor_default(self):
        # Document the active threshold so a future tune surfaces here.
        assert 0.5 < _DEDUPE_CONFIDENCE_FLOOR < 1.0








class TestAliasNoiseFilter:
    """After merging, aliases that are just `<determiner> <canonical>`
    add no information vs the canonical_name. Drop them. Concrete
    cases observed on bkwp's Pepys run: 'King' canonical with aliases
    ['King Charles II', 'the King'] → 'the King' is just 'King' +
    determiner; should drop. 'King Charles II' adds the regnal
    number; should stay."""

    def test_drops_determiner_prefixed_alias(self):
        records = [
            EntityRecord(canonical_id="king", canonical_name="King",
                         entity_type="person", mention_count=20,
                         aliases=["King Charles II", "the King"]),
            EntityRecord(canonical_id="king-charles-ii",
                         canonical_name="King Charles II",
                         entity_type="person", mention_count=5),
        ]
        merges = [{
            "a_id": "king", "b_id": "king-charles-ii",
            "confidence": 0.95,
            "synthesized_description": "King Charles II of England.",
        }]
        merged, _, _ = _apply_merges_to_records(records, merges, [], Mode.TEE)
        assert len(merged) == 1
        # canonical_name picks the longer alias-rich one (longer name as
        # tiebreak, mention_count primary). Either way, "the King"
        # should be filtered as determiner-noise relative to whichever
        # canonical wins.
        primary = merged[0]
        if primary.canonical_name == "King":
            assert "King Charles II" in primary.aliases
            assert "the King" not in primary.aliases
        else:
            assert primary.canonical_name == "King Charles II"
            # "the King" → norm-no-determiner = "king" ≠ "king charles ii"
            # → KEEP. "King" alone → norm-no-determiner = "king" ≠
            # "king charles ii" → KEEP. So aliases retain both.
            assert "King" in primary.aliases
            assert "the King" in primary.aliases

    def test_keeps_alias_that_adds_information(self):
        # Sir William Coventry vs Sir W. Coventry — different but
        # related; both should stay (one is the initial form of the
        # other; neither is just the canonical + determiner).
        records = [
            EntityRecord(canonical_id="sir-w-coventry",
                         canonical_name="Sir W. Coventry",
                         entity_type="person", mention_count=20,
                         aliases=["Sir William Coventry", "Mr. Coventry"]),
            EntityRecord(canonical_id="sir-william-coventry",
                         canonical_name="Sir William Coventry",
                         entity_type="person", mention_count=5),
        ]
        merges = [{
            "a_id": "sir-w-coventry", "b_id": "sir-william-coventry",
            "confidence": 0.95, "synthesized_description": "...",
        }]
        merged, _, _ = _apply_merges_to_records(records, merges, [], Mode.TEE)
        assert len(merged) == 1
        primary = merged[0]
        # Neither "Sir William Coventry" nor "Mr. Coventry" is just
        # the canonical + determiner — both add information (full
        # first name vs initial; honorific variant). Both stay.
        assert "Sir William Coventry" in primary.aliases or \
               "Sir W. Coventry" in primary.aliases
        assert "Mr. Coventry" in primary.aliases

    def test_keeps_relationship_word_alias(self):
        # Elizabeth's "wife" alias is not a determiner-prefixed form
        # of "Elizabeth St. Michel" — should stay.
        records = [
            EntityRecord(canonical_id="elizabeth",
                         canonical_name="Elizabeth St. Michel",
                         entity_type="person", mention_count=58,
                         aliases=["wife"]),
            EntityRecord(canonical_id="pepys-wife",
                         canonical_name="Pepys's wife",
                         entity_type="person", mention_count=40,
                         aliases=["wife"]),
        ]
        merges = [{
            "a_id": "elizabeth", "b_id": "pepys-wife",
            "confidence": 0.95,
            "synthesized_description": "Pepys's wife Elizabeth.",
        }]
        merged, _, _ = _apply_merges_to_records(records, merges, [], Mode.TEE)
        assert len(merged) == 1
        primary = merged[0]
        # Whichever canonical wins, "wife" adds information vs the
        # other and should stay.
        assert "wife" in primary.aliases

    def test_drops_a_an_determiner_too(self):
        # "a Smith" / "an Adams" — same noise pattern as "the King".
        records = [
            EntityRecord(canonical_id="smith", canonical_name="Smith",
                         entity_type="person", mention_count=10,
                         aliases=["a Smith"]),
            EntityRecord(canonical_id="adams", canonical_name="Adams",
                         entity_type="person", mention_count=8,
                         aliases=["an Adams"]),
        ]
        # Two non-merged records — alias filter applies AFTER the
        # collapse step in this code path, so we trigger via a self-
        # mergeable record path. Use a distinct collision instead:
        # the _deterministic_collapse path also runs `_apply_merges`.
        # Easier: feed a single LLM merge that fuses the two; the
        # alias filter runs on the merged result.
        merges = [{
            "a_id": "smith", "b_id": "adams",
            "confidence": 1.0, "synthesized_description": "Same.",
        }]
        merged, _, _ = _apply_merges_to_records(records, merges, [], Mode.TEE)
        assert len(merged) == 1
        primary = merged[0]
        # Whichever canonical wins, the determiner-prefixed alias
        # for that canonical should drop; the OTHER record's name
        # (and its determiner-prefix) stay because it's a different
        # name.
        if primary.canonical_name == "Smith":
            assert "a Smith" not in primary.aliases
            assert "Adams" in primary.aliases or "an Adams" in primary.aliases
        else:
            assert primary.canonical_name == "Adams"
            assert "an Adams" not in primary.aliases
            assert "Smith" in primary.aliases or "a Smith" in primary.aliases


# ── _apply_merges_to_records — LLM fuzzy merge layer ─────────────────────────


class TestApplyMergesToRecords:
    def test_self_merge_rewrites_singleton_description(self):
        # Self-merge (a_id == b_id) on a row with no other merges:
        # the row stays a singleton, but its description is replaced
        # with the LLM-supplied rewrite. Mechanism that lets the LLM
        # dedupe clean up bloated descriptions left behind by
        # _deterministic_collapse's empty-synth + " · " concat fallback
        # (the m7pp Pepys 11k-char wall).
        bloated = ("Alice is a senior engineer in 2024. · Alice is a "
                   "senior engineer at Acme in 2024. · Alice is a "
                   "senior engineer at Acme on payments in 2024.")
        records = [
            EntityRecord(canonical_id="alice", canonical_name="Alice",
                         entity_type="person", description=bloated,
                         mention_count=42),
        ]
        merges = [{
            "a_id": "alice", "b_id": "alice",
            "confidence": 1.0,
            "synthesized_description":
                "Alice is a senior engineer on the Acme payments team.",
        }]
        merged, _, _ = _apply_merges_to_records(
            records, merges, [], Mode.TEE,
        )
        assert len(merged) == 1
        assert merged[0].description == (
            "Alice is a senior engineer on the Acme payments team."
        )
        assert merged[0].canonical_id == "alice"

    def test_self_merge_rewrite_falls_back_when_no_pair_syn_on_rep(self):
        # Multi-member rep where the LLM emitted both a real pair-merge
        # (without synth) AND a self-merge rewrite on one member. The
        # self-merge rewrite should win over the " · " concat fallback.
        records = [
            EntityRecord(canonical_id="a", canonical_name="Alice",
                         entity_type="person",
                         description="orig a desc.",
                         mention_count=10),
            EntityRecord(canonical_id="b", canonical_name="Alice",
                         entity_type="person",
                         description="orig b desc 1. · orig b desc 2.",
                         mention_count=8),
        ]
        merges = [
            # Pair merge with no synth (e.g. LLM omitted it).
            {"a_id": "a", "b_id": "b", "confidence": 0.95,
             "synthesized_description": ""},
            # Self-merge rewrite for the bloated b.
            {"a_id": "b", "b_id": "b", "confidence": 1.0,
             "synthesized_description": "Alice (compressed b)."},
        ]
        merged, _, _ = _apply_merges_to_records(
            records, merges, [], Mode.TEE,
        )
        assert len(merged) == 1
        # Self-merge rewrite picked up as the fallback over " · " concat.
        assert merged[0].description == "Alice (compressed b)."

    def test_pair_syn_still_wins_over_self_merge(self):
        # When BOTH a pair-merge synth AND a self-merge rewrite are
        # present for the same rep, the pair-syn (which operates on
        # the post-merge entity) wins.
        records = [
            EntityRecord(canonical_id="a", canonical_name="Alice",
                         entity_type="person", description="a.",
                         mention_count=10),
            EntityRecord(canonical_id="b", canonical_name="Alice",
                         entity_type="person", description="b1. · b2.",
                         mention_count=8),
        ]
        merges = [
            {"a_id": "a", "b_id": "b", "confidence": 0.95,
             "synthesized_description": "Alice (pair-syn winner)."},
            {"a_id": "b", "b_id": "b", "confidence": 1.0,
             "synthesized_description": "Alice (self-merge loser)."},
        ]
        merged, _, _ = _apply_merges_to_records(
            records, merges, [], Mode.TEE,
        )
        assert len(merged) == 1
        assert merged[0].description == "Alice (pair-syn winner)."

    def test_merge_uses_synthesized_description(self):
        records = [
            EntityRecord(canonical_id="mom", canonical_name="Mom",
                         entity_type="person", description="The author's mother.",
                         mention_count=10),
            EntityRecord(canonical_id="jane-doe",
                         canonical_name="Jane Doe",
                         entity_type="person",
                         description="Family elder.",
                         mention_count=4),
        ]
        merges = [{
            "a_id": "mom",
            "b_id": "jane-doe",
            "confidence": 0.85,
            "synthesized_description": "The author's mother (Jane Doe).",
        }]
        merged, relations, id_remap = _apply_merges_to_records(
            records, merges, [], Mode.TEE,
        )
        assert len(merged) == 1
        # synthesized_description wins over concat.
        assert merged[0].description == "The author's mother (Jane Doe)."
        assert merged[0].mention_count == 14  # summed across endpoints
        # Both ids point at the merged primary.
        primary_id = merged[0].canonical_id
        assert id_remap["mom"] == primary_id
        assert id_remap["jane-doe"] == primary_id

    def test_relations_remapped_through_merge(self):
        records = [
            EntityRecord(canonical_id="a", canonical_name="A",
                         entity_type="person", mention_count=5),
            EntityRecord(canonical_id="b", canonical_name="B",
                         entity_type="person", mention_count=2),
            EntityRecord(canonical_id="c", canonical_name="C",
                         entity_type="person", mention_count=1),
        ]
        relations = [RelationEdge(from_id="b", to_id="c", relation="met_with")]
        merges = [{"a_id": "a", "b_id": "b", "confidence": 1.0,
                   "synthesized_description": "A=B"}]
        merged, remapped_rels, id_remap = _apply_merges_to_records(
            records, merges, relations, Mode.TEE,
        )
        # b→c becomes a→c after merge (a is primary, larger mention_count).
        assert len(remapped_rels) == 1
        assert remapped_rels[0].from_id == id_remap["b"]
        assert remapped_rels[0].to_id == "c"

    def test_bilateral_relations_kept(self):
        """The brief: when both A→B and B→A exist with potentially
        different verbs, keep both."""
        records = [
            EntityRecord(canonical_id="alice", canonical_name="Alice",
                         entity_type="person", mention_count=5),
            EntityRecord(canonical_id="bob", canonical_name="Bob",
                         entity_type="person", mention_count=4),
        ]
        relations = [
            RelationEdge(from_id="alice", to_id="bob", relation="employer_of"),
            RelationEdge(from_id="bob", to_id="alice", relation="employed_by"),
        ]
        merged, remapped_rels, _ = _apply_merges_to_records(
            records, [], relations, Mode.TEE,
        )
        assert len(remapped_rels) == 2  # both directions preserved

    def test_relation_conflict_picks_winner_by_evidence_count(self):
        """Spec 2026-05-02: when canonical and alias both carry an A→B
        edge after merging, keep ONE edge per directed pair. Winner =
        the (verb, confidence) backed by the most fact citations.
        Loser's evidence_fact_refs union under the winner — citations
        are evidence-preserving so no fact's attribution is dropped."""
        records = [
            EntityRecord(canonical_id="alice", canonical_name="Alice",
                         entity_type="person", mention_count=10,
                         evidence_fact_refs=[("t", 0), ("t", 1)]),
            EntityRecord(canonical_id="alice-2", canonical_name="Alice",
                         entity_type="person", mention_count=4,
                         evidence_fact_refs=[("t", 2)]),
            EntityRecord(canonical_id="bob", canonical_name="Bob",
                         entity_type="person", mention_count=6),
        ]
        # Pre-merge edges: canonical alice → bob with verb "spouse"
        # (2 fact citations); alias alice-2 → bob with verb
        # "married_to" (1 fact citation). After dedupe consolidates
        # alice-2 into alice, both edges land in the same (alice, bob)
        # bucket and the high-evidence winner takes the slot.
        relations = [
            RelationEdge(from_id="alice", to_id="bob", relation="spouse",
                         confidence=0.9,
                         evidence_fact_refs=[("t", 0), ("t", 1)]),
            RelationEdge(from_id="alice-2", to_id="bob",
                         relation="married_to", confidence=0.95,
                         evidence_fact_refs=[("t", 2)]),
        ]
        merges = [{
            "a_id": "alice", "b_id": "alice-2",
            "confidence": 0.95,
            "synthesized_description": "Alice (canonical).",
        }]
        merged, remapped_rels, id_remap = _apply_merges_to_records(
            records, merges, relations, Mode.TEE,
        )
        assert len(merged) == 2  # alice (merged) + bob
        assert len(remapped_rels) == 1, (
            "two edges into same (from, to) pair must collapse to one"
        )
        rel = remapped_rels[0]
        primary_alice = id_remap["alice"]
        assert rel.from_id == primary_alice
        assert rel.to_id == "bob"
        # Winner picked by evidence count: spouse (2) > married_to (1).
        assert rel.relation == "spouse"
        assert rel.confidence == 0.9
        # Evidence union is sorted + deduped; loser's ref preserved.
        assert rel.evidence_fact_refs == [("t", 0), ("t", 1), ("t", 2)]

    def test_relation_conflict_evidence_union_no_duplicates(self):
        """Two edges into the same (from, to) bucket with overlapping
        evidence — the union must dedupe shared fact_refs and the
        winner is the higher-evidence-count edge after the dedup."""
        records = [
            EntityRecord(canonical_id="alice", canonical_name="Alice",
                         entity_type="person", mention_count=10),
            EntityRecord(canonical_id="alice-2", canonical_name="Alice",
                         entity_type="person", mention_count=4),
            EntityRecord(canonical_id="bob", canonical_name="Bob",
                         entity_type="person", mention_count=6),
        ]
        relations = [
            RelationEdge(from_id="alice", to_id="bob", relation="spouse",
                         confidence=0.9,
                         evidence_fact_refs=[("t", 0), ("t", 1), ("t", 2)]),
            RelationEdge(from_id="alice-2", to_id="bob",
                         relation="lives_with", confidence=0.7,
                         evidence_fact_refs=[("t", 1), ("t", 3)]),
        ]
        merges = [{"a_id": "alice", "b_id": "alice-2", "confidence": 0.95}]
        merged, remapped_rels, _ = _apply_merges_to_records(
            records, merges, relations, Mode.TEE,
        )
        assert len(remapped_rels) == 1
        rel = remapped_rels[0]
        # spouse (3 refs) beats lives_with (2 refs).
        assert rel.relation == "spouse"
        # Union is {("t",0), ("t",1), ("t",2), ("t",3)} sorted.
        assert rel.evidence_fact_refs == [
            ("t", 0), ("t", 1), ("t", 2), ("t", 3),
        ]


# ── _resolve_subject (argmax) ────────────────────────────────────────────────


class TestResolveSubject:
    def test_argmax_picks_highest_likelihood(self):
        records = [
            EntityRecord(canonical_id="alice", canonical_name="Alice",
                         entity_type="person", mention_count=5),
            EntityRecord(canonical_id="bob", canonical_name="Bob",
                         entity_type="person", mention_count=10),
        ]
        likelihoods = {"alice": 0.95, "bob": 0.10}
        s = _resolve_subject(records, likelihoods, "the author")
        assert s is not None
        assert s.canonical_id == "alice"
        assert s.source == "argmax"

    def test_falls_back_to_mention_count_when_all_zero(self):
        records = [
            EntityRecord(canonical_id="alice", canonical_name="Alice",
                         entity_type="person", mention_count=2),
            EntityRecord(canonical_id="bob", canonical_name="Bob",
                         entity_type="person", mention_count=10),
        ]
        likelihoods = {"alice": 0.0, "bob": 0.0}
        s = _resolve_subject(records, likelihoods, "the author")
        assert s is not None
        assert s.canonical_id == "bob"
        assert s.source == "mention_count_fallback"

    def test_skips_non_persons(self):
        records = [
            EntityRecord(canonical_id="acme", canonical_name="Acme",
                         entity_type="org", mention_count=20),
            EntityRecord(canonical_id="alice", canonical_name="Alice",
                         entity_type="person", mention_count=2),
        ]
        likelihoods = {"acme": 0.9, "alice": 0.1}
        s = _resolve_subject(records, likelihoods, "the author")
        # Non-persons can't be subject; alice wins.
        assert s is not None
        assert s.canonical_id == "alice"

    def test_returns_none_when_no_persons(self):
        records = [
            EntityRecord(canonical_id="acme", canonical_name="Acme",
                         entity_type="org", mention_count=20),
        ]
        s = _resolve_subject(records, {"acme": 0.9}, "the author")
        assert s is None


# ── detect_entities (empty path) ─────────────────────────────────────────────




# ── build_context_block ──────────────────────────────────────────────────────


class TestBuildContextBlock:
    def test_empty_output_returns_empty_string(self):
        assert build_context_block(EntitiesOutput()) == ""

    def test_includes_subject_entities_relations(self):
        out = EntitiesOutput(
            subject=SubjectRef(canonical_id="alice", display="Alice", source="argmax"),
            entities=[
                EntityRecord(canonical_id="alice", canonical_name="Alice",
                             entity_type="person", role="subject",
                             description="Author.", mention_count=5),
                EntityRecord(canonical_id="bob", canonical_name="Bob",
                             entity_type="person", role="friend",
                             description="Close friend.", mention_count=3),
            ],
            relations=[
                RelationEdge(from_id="alice", to_id="bob",
                             relation="close_friend", confidence=0.9),
            ],
        )
        block = build_context_block(out)
        assert "subject: Alice (alice)" in block
        assert "alice [subject]" in block
        assert "bob [friend]" in block
        assert "close_friend" in block

    def test_caps_entities_at_max(self):
        entities = [
            EntityRecord(canonical_id=f"e{i}", canonical_name=f"E{i}",
                         entity_type="person", mention_count=i)
            for i in range(25)
        ]
        out = EntitiesOutput(entities=entities)
        block = build_context_block(out, max_entities=5)
        assert "(20 more entities omitted)" in block


# ── End-to-end with mocked LLM ───────────────────────────────────────────────


class TestDetectEntitiesEndToEnd:
    """Patch llm.complete so detect_entities exercises the full Phase
    A → B → C → D pipeline without hitting a real model. The mocked
    LLM returns deterministic JSON that the parser accepts."""

        # No relations expected from the mocked per-entity call (relations
        # list was empty), but relation_candidate still got resolved.
        # Dedupe step ran once; no merges proposed.




