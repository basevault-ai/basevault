"""Unit tests for bundle-narrator separation in entities.py.

Composes with PR #11's non-person-guard: per-file narrator groups are
`entity_type="person"` so they survive tier-1 scrutiny, but the bundle
post-filter (`_scrub_bundle_subject`) wipes any single subject pick
regardless of whether tier-1 accepted it.

Order of precedence (documented in the rebuild task):
  person-check > file-scope > LLM tier > alias-match > mention-count-fallback.
"""


from engine.content_extractor import Entity, EntityRef, EvidenceSpan, ExtractedItem
from engine.entities import (
    EntitiesOutput,
    EntityRecord,
    SubjectRef,
    _fact_file_id,
    _group_entities,
    _has_first_person,
    _is_bundle,
    _scrub_bundle_subject,
)


def _mk(
    summary: str,
    entities: list[tuple[str, str, str]],
    evidence_text: str,
    file_path: str | None,
    topics: list[str] | None = None,
) -> ExtractedItem:
    return ExtractedItem(
        item_type="fact",
        summary=summary,
        evidence=[EvidenceSpan(text=evidence_text, source_ref="t",
                               file_path=file_path)],
        entities=[
            EntityRef(entity=Entity(name=n, entity_type=t), role=r)
            for n, t, r in entities
        ],
        topics=topics or ["general"],
    )


# ── _fact_file_id ────────────────────────────────────────────────────────────


class TestFactFileId:
    def test_returns_first_file_path(self):
        it = _mk("x", [], "quote", "foo.md")
        assert _fact_file_id(it) == "foo.md"

    def test_none_when_no_file_path(self):
        it = _mk("x", [], "quote", None)
        assert _fact_file_id(it) is None


# ── _is_bundle ───────────────────────────────────────────────────────────────


class TestIsBundle:
    def test_single_file_is_not_bundle(self):
        facts = {"t": [
            _mk("a", [], "I did X", "file_a.md"),
            _mk("b", [], "I did Y", "file_a.md"),
        ]}
        assert _is_bundle(facts) is False

    def test_multi_file_is_bundle(self):
        facts = {"t": [
            _mk("a", [], "I did X", "file_a.md"),
            _mk("b", [], "I did Y", "file_b.md"),
        ]}
        assert _is_bundle(facts) is True

    def test_empty_is_not_bundle(self):
        assert _is_bundle({}) is False


# ── _has_first_person regex widening (regression) ───────────────────────────


class TestHasFirstPersonWidened:
    """PR #11's regex was case-sensitive on `my`/`me`/`mine`/`myself` —
    this branch widens to also match sentence-initial "My mom called me"
    which is a natural narrator signal."""

    def test_matches_sentence_initial_My(self):
        assert _has_first_person("My knuckles were raw.")

    def test_matches_sentence_initial_Me(self):
        assert _has_first_person("Me neither.")

    def test_still_rejects_lowercase_i_typo(self):
        """Capital-I still required for the standalone pronoun."""
        assert not _has_first_person("i went home")

    def test_still_matches_lowercase_my(self):
        assert _has_first_person("It's my fault")


# ── per-file narrator synthesis ──────────────────────────────────────────────


class TestNarratorSynthesis:
    def test_single_file_no_synthesis(self):
        """Single-file run must NOT add synthetic narrator groups."""
        facts = {"t": [
            _mk("x", [("Alice", "person", "subject")], "I went home.",
                "solo.md"),
            _mk("y", [("Alice", "person", "subject")], "I slept well.",
                "solo.md"),
        ]}
        groups = _group_entities(facts)
        names = [g.canonical_name for g in groups]
        assert "Alice" in names
        assert not any(n.startswith("Narrator of") for n in names)

    def test_bundle_with_unnamed_narrators_synthesizes_per_file(self):
        """Multi-file bundle where each file has first-person facts
        without named subjects → one synthetic narrator per file."""
        facts = {"t": [
            _mk("f1", [], "I went home on Tuesday.", "file_01.md"),
            _mk("f1", [], "My mom called me.",        "file_01.md"),
            _mk("f2", [], "I broke my hand fighting.", "file_02.md"),
            _mk("f2", [], "My knuckles were raw.",     "file_02.md"),
        ]}
        groups = _group_entities(facts)
        names = [g.canonical_name for g in groups]
        assert "Narrator of file_01" in names
        assert "Narrator of file_02" in names

    def test_bundle_skips_file_with_named_subject(self):
        """If a file's first-person facts have a named subject entity,
        don't also synthesize a 'Narrator of …' — would duplicate."""
        facts = {"t": [
            _mk("f1", [("Bex", "person", "subject")],
                "I used fentanyl.", "file_05.md"),
            _mk("f1", [("Bex", "person", "subject")],
                "My hands shook.",  "file_05.md"),
            _mk("f2", [],
                "I broke my hand.", "file_02.md"),
            _mk("f2", [],
                "My knuckles were raw.", "file_02.md"),
        ]}
        groups = _group_entities(facts)
        names = [g.canonical_name for g in groups]
        assert "Bex" in names
        # file_05 has named subject → no Narrator of file_05
        assert "Narrator of file_05" not in names
        # file_02 has no named subject → Narrator of file_02 synthesized
        assert "Narrator of file_02" in names

    def test_bundle_requires_min_fp_evidence(self):
        """A file with only 1 first-person fact + no named subject
        doesn't trigger synthesis (too noisy)."""
        facts = {"t": [
            _mk("f1", [], "I did X once.",    "file_03.md"),  # only 1 FP
            _mk("f2", [], "I did Y.",         "file_04.md"),
            _mk("f2", [], "My Z.",            "file_04.md"),  # 2 FP
        ]}
        groups = _group_entities(facts)
        names = [g.canonical_name for g in groups]
        assert "Narrator of file_03" not in names  # below threshold
        assert "Narrator of file_04" in names

    def test_synthetic_narrator_is_person(self):
        """Synthesized narrator groups must be `entity_type=person` so
        they survive person-only filters downstream."""
        facts = {"t": [
            _mk("f1", [], "I went X", "file_01.md"),
            _mk("f1", [], "my Y",     "file_01.md"),
            _mk("f2", [], "I did Z",  "file_02.md"),
            _mk("f2", [], "my W",     "file_02.md"),
        ]}
        groups = _group_entities(facts)
        synth = [g for g in groups if g.canonical_name.startswith("Narrator of")]
        assert len(synth) == 2
        for g in synth:
            assert g.entity_type == "person"
            # mention_count tracks how many first-person facts in this
            # file went into the synth group (replaces the removed
            # first_person_evidence_count signal).
            assert g.mention_count >= 2

    def test_label_strips_input_prefix_and_extensions(self):
        """`input_04_intense_illegal.md` → `Narrator of 04_intense_illegal`."""
        facts = {"t": [
            _mk("a", [], "I did X", "input_04_intense_illegal.md"),
            _mk("b", [], "my Y",    "input_04_intense_illegal.md"),
            _mk("c", [], "I did Z", "input_05_intense_substance.md"),
            _mk("d", [], "my W",    "input_05_intense_substance.md"),
        ]}
        groups = _group_entities(facts)
        names = [g.canonical_name for g in groups]
        assert "Narrator of 04_intense_illegal" in names
        assert "Narrator of 05_intense_substance" in names


# ── _scrub_bundle_subject — file-scope precedence over LLM tier ─────────────


class TestScrubBundleSubject:
    def test_single_file_preserves_subject(self):
        """Non-bundle output passes through untouched."""
        out = EntitiesOutput(
            subject=SubjectRef(canonical_id="alice", display="Alice",
                               source="llm"),
            entities=[
                EntityRecord(canonical_id="alice", canonical_name="Alice",
                             entity_type="person", role="subject"),
            ],
        )
        result = _scrub_bundle_subject(out, is_bundle=False)
        assert result.subject is not None
        assert result.subject.canonical_id == "alice"
        assert result.entities[0].role == "subject"

    def test_bundle_clears_subject_and_role_even_on_person_pick(self):
        """Bundle wipes the subject even when tier-1 accepted a valid
        person (file-scope beats LLM tier)."""
        out = EntitiesOutput(
            subject=SubjectRef(canonical_id="author", display="Author",
                               source="llm"),
            entities=[
                EntityRecord(canonical_id="author", canonical_name="Author",
                             entity_type="person", role="subject"),
                EntityRecord(canonical_id="jess", canonical_name="Jess",
                             entity_type="person", role="friend"),
            ],
        )
        result = _scrub_bundle_subject(out, is_bundle=True)
        assert result.subject is None
        # role=subject stamp stripped; other roles preserved
        author = next(r for r in result.entities if r.canonical_id == "author")
        jess = next(r for r in result.entities if r.canonical_id == "jess")
        assert author.role == ""
        assert jess.role == "friend"

    def test_bundle_scrub_is_noop_when_subject_already_none(self):
        """Bundle + already-null subject → no change."""
        out = EntitiesOutput(
            subject=None,
            entities=[
                EntityRecord(canonical_id="jess", canonical_name="Jess",
                             entity_type="person", role="friend"),
            ],
        )
        result = _scrub_bundle_subject(out, is_bundle=True)
        assert result.subject is None
        assert result.entities[0].role == "friend"
