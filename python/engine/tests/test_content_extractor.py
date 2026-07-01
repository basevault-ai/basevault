"""
Unit tests for content_extractor.py — no LLM calls.

Tests cover:
  - _build_prompt: title/date/content injection, source_ref
  - _resolve_span: 4 paths — exact, whitespace-normalized, prefix, LCS fallback
  - _parse_items: schema parsing, invalid item types, file_offset math via
                  doc.origin_char, topic taxonomy filtering, confidence clamping
  - extract_items: parallel dispatch, empty input

Run with:
    cd engine && pytest tests/test_content_extractor.py -v
"""
import json



from engine.ingestor import Document, SourceType
from engine.content_extractor import _resolve_span, _parse_items, _build_prompt, _topics_for_run, _DEFAULT_TOPICS, ITEM_TYPES, AFFECT_REGISTERS, _MAX_AFFECT
from engine import llm


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _doc(content: str = "Alice signed the contract.",
         doc_id: str = "doc1::split_00",
         origin_char: int = 0,
         file_id: str = "doc1.txt",
         date: str = "2026-04-15",
         title: str = "Contract Notes") -> Document:
    return Document(
        id=doc_id, source_path=f"/x/{file_id}", source_type=SourceType.TXT,
        content=content, title=title, date=date, file_id=file_id,
        origin_char=origin_char,
    )


def _item_json(**kwargs) -> dict:
    base = {
        "type": "fact",
        "summary": "Alice signed a contract",
        "evidence": [{"text": "Alice signed the contract.",
                      "source_ref": "doc1::split_00"}],
        "occurred_at": "2026-04-15",
        "occurred_at_text": None,
        "entities": [{"name": "Alice", "entity_type": "person", "role": "subject"}],
        "topics": ["work"],
        "tags": [],
        "confidence": 1.0,
    }
    base.update(kwargs)
    return base


# ── _resolve_span ─────────────────────────────────────────────────────────────

class TestResolveSpan:
    def test_exact_match(self):
        text = "the quick brown fox jumps"
        s, e, approx = _resolve_span("brown fox", text)
        assert text[s:e] == "brown fox" and approx is False

    def test_full_content_match(self):
        s, e, approx = _resolve_span("exact", "exact")
        assert (s, e, approx) == (0, 5, False)

    def test_first_occurrence_returned(self):
        s, e, _ = _resolve_span("cat", "cat and cat")
        assert (s, e) == (0, 3)

    def test_span_at_end(self):
        s, e, _ = _resolve_span("suffix", "prefix and suffix")
        assert (s, e) == (11, 17)

    def test_whitespace_normalized(self):
        # Source has multiple internal spaces / newlines; quote uses single spaces
        text = "the quick   brown\nfox jumps over"
        s, e, approx = _resolve_span("brown fox", text)
        assert s is not None and approx is False
        assert "brown" in text[s:e] and "fox" in text[s:e]

    def test_smart_quotes_normalized(self):
        text = "She said \u201chello world\u201d quietly"
        s, e, approx = _resolve_span('"hello world"', text)
        assert s is not None and approx is False

    def test_prefix_when_quote_has_extra_tail(self):
        # Prefix path triggers when the LLM emitted a quote whose first 40
        # chars ARE a literal substring of the source — typically when it
        # appended an ellipsis or paraphrased the tail.
        text = "The quick brown fox jumps over the lazy dog and runs around the yard"
        quote = "The quick brown fox jumps over the lazy dog REPLACED TAIL HERE"
        s, e, approx = _resolve_span(quote, text)
        assert s == 0 and approx is False

    def test_lcs_fallback_marks_approximate(self):
        text = "We closed a $10M Section 1202 exit last month."
        quote = "target $10M Section 1202 exit via the tax exemption strategy"
        s, e, approx = _resolve_span(quote, text)
        assert s is not None and approx is True
        assert "$10M Section 1202 exit" in text[s:e]

    def test_no_match_returns_none(self):
        s, e, approx = _resolve_span("totally unrelated stuff", "hello world")
        assert (s, e, approx) == (None, None, False)

    def test_empty_text_returns_none(self):
        assert _resolve_span("", "anything") == (None, None, False)


# ── _build_prompt ─────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_doc_id_injected(self):
        p = _build_prompt(_doc(doc_id="my::doc::id"))
        assert "my::doc::id" in p

    def test_title_injected(self):
        p = _build_prompt(_doc(title="My Important Doc"))
        assert "My Important Doc" in p

    def test_date_from_doc(self):
        p = _build_prompt(_doc(date="2026-03-15"))
        assert "2026-03-15" in p

    def test_date_unknown_when_doc_lacks_one(self):
        p = _build_prompt(_doc(date=""))
        assert "Document date: unknown" in p

    def test_content_injected(self):
        p = _build_prompt(_doc(content="UNIQUE_CONTENT_MARKER"))
        assert "UNIQUE_CONTENT_MARKER" in p

    def test_all_item_types_listed(self):
        p = _build_prompt(_doc())
        for t in ITEM_TYPES:
            assert t in p

    def test_affect_registers_in_prompt(self):
        p = _build_prompt(_doc())
        for r in AFFECT_REGISTERS:
            assert r in p
        # The affect dimension must be framed as distinct from the item type.
        assert "Affect dimension" in p

    def test_default_topics_in_prompt(self, monkeypatch):
        monkeypatch.setattr(llm, "_read_app_config", lambda: {})
        p = _build_prompt(_doc())
        for topic in _DEFAULT_TOPICS:
            assert topic in p

    def test_custom_categories_replace_defaults_in_prompt(self, monkeypatch):
        monkeypatch.setattr(
            llm, "_read_app_config",
            lambda: {"categories": ["fundraising", "research", "other"]},
        )
        p = _build_prompt(_doc())
        assert "fundraising" in p
        assert "research" in p
        # A removed default no longer appears in the topics line. We
        # check the line specifically — "family" can appear in
        # unrelated rubric prose elsewhere in the prompt.
        topics_line = next(
            line for line in p.splitlines() if line.startswith("health,")
            or line.startswith("fundraising,")
        )
        assert "family" not in topics_line


# ── _topics_for_run ───────────────────────────────────────────────────────────

class TestTopicsForRun:
    def test_returns_defaults_when_config_absent(self, monkeypatch):
        monkeypatch.setattr(llm, "_read_app_config", lambda: {})
        assert _topics_for_run() == list(_DEFAULT_TOPICS)

    def test_returns_defaults_when_categories_missing(self, monkeypatch):
        monkeypatch.setattr(
            llm, "_read_app_config",
            lambda: {"subject": "User", "tee_provider": "tinfoil"},
        )
        assert _topics_for_run() == list(_DEFAULT_TOPICS)

    def test_returns_defaults_when_categories_empty(self, monkeypatch):
        monkeypatch.setattr(llm, "_read_app_config", lambda: {"categories": []})
        assert _topics_for_run() == list(_DEFAULT_TOPICS)

    def test_returns_defaults_when_categories_wrong_shape(self, monkeypatch):
        monkeypatch.setattr(
            llm, "_read_app_config", lambda: {"categories": "not-a-list"},
        )
        assert _topics_for_run() == list(_DEFAULT_TOPICS)

    def test_returns_defaults_when_categories_has_no_strings(self, monkeypatch):
        monkeypatch.setattr(
            llm, "_read_app_config", lambda: {"categories": [1, 2, None, ""]},
        )
        assert _topics_for_run() == list(_DEFAULT_TOPICS)

    def test_returns_config_list_when_present(self, monkeypatch):
        monkeypatch.setattr(
            llm, "_read_app_config",
            lambda: {"categories": ["fundraising", "research", "other"]},
        )
        assert _topics_for_run() == ["fundraising", "research", "other"]

    def test_strips_whitespace_and_filters_empties(self, monkeypatch):
        monkeypatch.setattr(
            llm, "_read_app_config",
            lambda: {"categories": ["  work  ", "", "health"]},
        )
        assert _topics_for_run() == ["work", "health"]


# ── _parse_items ──────────────────────────────────────────────────────────────

class TestParseItems:
    """Cover the legacy bare-array shape that the parser accepts for
    robustness. The envelope shape (split_summaries + items) is
    exercised separately under TestParseEnvelope below."""

    def _parse(self, items: list[dict], doc: Document | None = None):
        parsed_items, _summaries, _parse_err = _parse_items(
            json.dumps(items), doc or _doc(),
        )
        return parsed_items

    def test_all_six_item_types_parsed(self):
        items = [
            _item_json(type=t, summary=f"summary {t}",
                       evidence=[{"text": "Alice signed the contract.",
                                  "source_ref": "doc1::split_00"}])
            for t in ITEM_TYPES
        ]
        out = self._parse(items)
        assert {it.item_type for it in out} == set(ITEM_TYPES)

    def test_invalid_type_dropped(self):
        out = self._parse([
            _item_json(type="not_a_type"),
            _item_json(type="fact", summary="kept"),
        ])
        assert [it.summary for it in out] == ["kept"]

    def test_empty_summary_dropped(self):
        assert self._parse([_item_json(summary="")]) == []

    def test_invalid_json_returns_empty(self):
        items, _, parse_err = _parse_items("not json", _doc())
        assert items == [] and parse_err is True

    def test_non_envelope_non_array_returns_empty(self):
        items, _, parse_err = _parse_items('"hello"', _doc())
        assert items == [] and parse_err is True

    def test_file_offset_includes_doc_origin(self):
        text = "Alice signed the contract."
        doc = _doc(content=text, origin_char=500)
        out = self._parse([_item_json(
            evidence=[{"text": text, "source_ref": doc.id}]
        )], doc=doc)
        ev = out[0].evidence[0]
        assert ev.file_offset == 500   # doc.origin_char + start (0)
        assert ev.file_length == len(text)
        assert ev.file_path == "doc1.txt"
        assert ev.approximate is False

    def test_file_offset_for_inner_substring(self):
        text = "lead in Alice signed the contract. trailing"
        doc = _doc(content=text, origin_char=10)
        out = self._parse([_item_json(
            evidence=[{"text": "Alice signed the contract.",
                       "source_ref": doc.id}]
        )], doc=doc)
        ev = out[0].evidence[0]
        # local start = 8 ("lead in "), file_offset = 10 + 8 = 18
        assert ev.file_offset == 18
        assert ev.file_length == len("Alice signed the contract.")

    def test_topics_filtered_to_taxonomy(self):
        out = self._parse([_item_json(topics=["work", "made_up", "health"])])
        assert sorted(out[0].topics) == ["health", "work"]

    def test_topics_filtered_to_custom_taxonomy(self, monkeypatch):
        # Custom list keeps "fundraising" + "research", drops "work"
        # (now removed) and "made_up" (never listed).
        monkeypatch.setattr(
            llm, "_read_app_config",
            lambda: {"categories": ["fundraising", "research", "other"]},
        )
        out = self._parse([_item_json(
            topics=["fundraising", "research", "work", "made_up"],
        )])
        assert sorted(out[0].topics) == ["fundraising", "research"]

    def test_confidence_clamped(self):
        out = self._parse([
            _item_json(summary="hi", confidence=1.5),
            _item_json(summary="lo", confidence=-0.3),
            _item_json(summary="ok", confidence=0.7),
        ])
        confs = {it.summary: it.confidence for it in out}
        assert confs == {"hi": 1.0, "lo": 0.0, "ok": 0.7}

    def test_invalid_entity_type_normalized(self):
        out = self._parse([_item_json(entities=[
            {"name": "Acme Corp", "entity_type": "company", "role": "subject"},
        ])])
        # "company" not in ENTITY_TYPES → coerced to "other"
        assert out[0].entities[0].entity.entity_type == "other"

    def test_invalid_role_normalized(self):
        out = self._parse([_item_json(entities=[
            {"name": "Bob", "entity_type": "person", "role": "owner"},
        ])])
        assert out[0].entities[0].role == "mentioned"

    def test_nameless_entity_dropped(self):
        out = self._parse([_item_json(entities=[
            {"name": "", "entity_type": "person", "role": "subject"},
            {"name": "Alice", "entity_type": "person", "role": "subject"},
        ])])
        names = [r.entity.name for r in out[0].entities]
        assert names == ["Alice"]

    def test_affect_filtered_to_taxonomy(self):
        out = self._parse([_item_json(affect=["anxiety", "made_up", "grief"])])
        assert out[0].affect == ["anxiety", "grief"]

    def test_affect_empty_when_absent(self):
        # _item_json carries no affect key — the neutral / abstain default.
        assert self._parse([_item_json()])[0].affect == []

    def test_affect_lowercased_and_deduped(self):
        out = self._parse([_item_json(affect=["Anger", "anger", "JOY"])])
        assert out[0].affect == ["anger", "joy"]

    def test_affect_capped_at_max(self):
        # More registers than the ceiling → first _MAX_AFFECT kept in order.
        many = ["joy", "grief", "anger", "fear"]
        out = self._parse([_item_json(affect=many)])
        assert out[0].affect == many[:_MAX_AFFECT]
        assert len(out[0].affect) == _MAX_AFFECT

    def test_affect_non_list_collapses_to_empty(self):
        # A bare string must NOT be iterated character-by-character.
        assert self._parse([_item_json(affect="anxiety")])[0].affect == []
        assert self._parse([_item_json(affect=None)])[0].affect == []


# ── extract_items ─────────────────────────────────────────────────────────────



# ── Combined / batched documents ──────────────────────────────────────────────

class TestCombinedDocAttribution:
    """When the splitter packs multiple small entries into one batched
    Document, content_extractor must route each fact's evidence_ref back to
    the inner entry that contains the quoted span — using the
    `combined_entries` metadata, not the synthetic batch-level file_id."""

    def _batch_doc(self):
        # Manually construct a combined Document with two inner entries.
        a_text = "Alice signed the contract on Monday."
        sep = "\n\n--- ENTRY: 2026-04-16 | Day 2 [j::entry_001] ---\n\n"
        b_text = "Bob shipped the release on Tuesday."
        content = a_text + sep + b_text
        a_start = 0
        a_end = len(a_text)
        b_start = a_end + len(sep)
        b_end = b_start + len(b_text)
        return Document(
            id="j::batch_000",
            source_path="/x/j.json",
            source_type=SourceType.DAYONE_JSON,
            content=content,
            title="Day 1 → Day 2 (2 entries)",
            date="2026-04-15..2026-04-16",
            file_id="j::batch_000",
            origin_char=0,
            metadata={
                "combined_count": 2,
                "combined_entries": [
                    {
                        "id": "j::entry_000",
                        "file_id": "j.json",
                        "source_path": "/x/j.json",
                        "origin_char": 1000,
                        "title": "Day 1",
                        "date": "2026-04-15",
                        "content_start": a_start,
                        "content_end": a_end,
                        "tags": [], "uuid": "u-0", "location": None,
                    },
                    {
                        "id": "j::entry_001",
                        "file_id": "j.json",
                        "source_path": "/x/j.json",
                        "origin_char": 5000,
                        "title": "Day 2",
                        "date": "2026-04-16",
                        "content_start": b_start,
                        "content_end": b_end,
                        "tags": [], "uuid": "u-1", "location": None,
                    },
                ],
            },
        )

    def test_evidence_routes_to_first_entry(self):
        doc = self._batch_doc()
        items, _summaries, _err = _parse_items(json.dumps([
            _item_json(
                summary="Alice signed contract",
                evidence=[{"text": "Alice signed the contract.",
                           "source_ref": doc.id}],
            )
        ]), doc)
        ev = items[0].evidence[0]
        # Span starts at content_start=0 in entry_000, whose origin_char=1000.
        assert ev.file_path == "j.json"
        assert ev.file_offset == 1000
        assert ev.source_ref == "j::entry_000"

    def test_evidence_routes_to_second_entry(self):
        doc = self._batch_doc()
        items, _summaries, _err = _parse_items(json.dumps([
            _item_json(
                summary="Bob shipped release",
                evidence=[{"text": "Bob shipped the release on Tuesday.",
                           "source_ref": doc.id}],
            )
        ]), doc)
        ev = items[0].evidence[0]
        # Span starts at content_start of entry_001 (origin_char=5000); local
        # offset within that entry is 0 → file_offset == 5000.
        assert ev.file_path == "j.json"
        assert ev.file_offset == 5000
        assert ev.source_ref == "j::entry_001"

    def test_inner_substring_offset_within_entry(self):
        doc = self._batch_doc()
        items, _summaries, _err = _parse_items(json.dumps([
            _item_json(
                summary="signing on Monday",
                evidence=[{"text": "signed the contract", "source_ref": doc.id}],
            )
        ]), doc)
        ev = items[0].evidence[0]
        # Substring "signed the contract" starts at index 6 within
        # "Alice signed the contract on Monday." (entry_000). With entry
        # origin_char=1000, the file_offset should be 1006.
        assert ev.file_path == "j.json"
        assert ev.file_offset == 1006
        assert ev.source_ref == "j::entry_000"

    def test_prompt_announces_batch_to_llm(self):
        doc = self._batch_doc()
        prompt = _build_prompt(doc)
        # The note must mention the batch (case-insensitive) so the LLM
        # knows to attribute occurred_at per entry and not dedupe across.
        lower = prompt.lower()
        assert "batch of 2 independent entries" in lower
        assert "--- ENTRY:" in prompt
        # Per-entry occurred_at hint must be present.
        assert "per-entry date" in lower
        # Cross-entry dedupe must be explicitly forbidden.
        assert "within an entry, not across entries" in lower
        # The batch note must require ONE summary per inner entry (not one
        # per batch) so per-chunk RAG enrichment downstream gets per-entry
        # summaries.
        assert "one element per inner entry" in lower


# ── Envelope shape: split_summaries + items ───────────────────────────────────

class TestParseEnvelope:
    """The new wire shape is
    `{"split_summaries": [...], "items": [...]}`. The parser must
    extract summaries alongside items, tolerate missing/empty
    summaries, and gracefully fall back to a bare-array shape from a
    prompt-non-compliant LLM response."""

    def _envelope(self, summaries: list[dict] | None, items: list[dict]) -> str:
        payload: dict = {"items": items}
        if summaries is not None:
            payload["split_summaries"] = summaries
        return json.dumps(payload)

    def test_envelope_parses_summary_and_items(self):
        doc = _doc()
        raw = self._envelope(
            summaries=[{"id": doc.id, "summary": "Two-line gist."}],
            items=[_item_json(summary="Alice signed.")],
        )
        items, summaries, parse_err = _parse_items(raw, doc)
        assert parse_err is False
        assert len(items) == 1 and items[0].summary == "Alice signed."
        assert summaries == {doc.id: "Two-line gist."}

    def test_envelope_with_no_summaries_returns_empty_dict(self):
        doc = _doc()
        raw = self._envelope(summaries=[], items=[_item_json()])
        items, summaries, parse_err = _parse_items(raw, doc)
        assert parse_err is False
        assert len(items) == 1
        assert summaries == {}

    def test_envelope_with_missing_summary_field_returns_empty_dict(self):
        doc = _doc()
        raw = json.dumps({"items": [_item_json()]})
        items, summaries, parse_err = _parse_items(raw, doc)
        assert parse_err is False
        assert len(items) == 1
        assert summaries == {}

    def test_bare_array_legacy_shape_accepted_no_summaries(self):
        """Some LLM responses may ignore the envelope contract and emit
        a bare array. The parser accepts it gracefully to avoid burning
        a parse-error retry — items survive, summaries are empty."""
        doc = _doc()
        raw = json.dumps([_item_json(summary="Bob signed.")])
        items, summaries, parse_err = _parse_items(raw, doc)
        assert parse_err is False
        assert items[0].summary == "Bob signed."
        assert summaries == {}

    def test_envelope_with_non_list_summaries_treats_as_empty(self):
        doc = _doc()
        raw = json.dumps({
            "split_summaries": "not-a-list",
            "items": [_item_json()],
        })
        items, summaries, parse_err = _parse_items(raw, doc)
        assert parse_err is False
        assert len(items) == 1
        assert summaries == {}

    def test_envelope_summary_id_preserved_verbatim(self):
        """The id field is the join key for downstream consumers (PR 4.2
        chunk enrichment). It must survive parsing unchanged so
        per-inner-entry routing works for batched docs."""
        doc = _doc()
        raw = self._envelope(
            summaries=[
                {"id": "j::entry_001", "summary": "Entry one."},
                {"id": "j::entry_002", "summary": "Entry two."},
            ],
            items=[_item_json()],
        )
        _items, summaries, _ = _parse_items(raw, doc)
        assert summaries == {
            "j::entry_001": "Entry one.",
            "j::entry_002": "Entry two.",
        }

    def test_envelope_summary_text_blank_dropped(self):
        doc = _doc()
        raw = self._envelope(
            summaries=[
                {"id": doc.id, "summary": "   "},
                {"id": "other", "summary": "real text"},
            ],
            items=[_item_json()],
        )
        _items, summaries, _ = _parse_items(raw, doc)
        assert summaries == {"other": "real text"}

    def test_envelope_truncation_keeps_summary_and_partial_items(self):
        """Truncated envelope salvage: split_summaries lands fully (it's
        first in the response), items array is trimmed to the last
        complete element. Most-salient-first ordering means the items
        we keep are the high-value ones."""
        doc = _doc()
        good_a = _item_json(summary="aaa")
        good_b = _item_json(summary="bbb")
        intact = json.dumps({
            "split_summaries": [{"id": doc.id, "summary": "Gist."}],
            "items": [good_a, good_b],
        })
        # Snip mid-third-item — the LLM's most-salient items survive.
        truncated = intact[:-2] + ',{"type":"fact","summa'
        items, summaries, parse_err = _parse_items(truncated, doc)
        assert parse_err is False
        assert summaries == {doc.id: "Gist."}
        assert [it.summary for it in items] == ["aaa", "bbb"]
