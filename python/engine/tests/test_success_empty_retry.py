"""
Issue #616: success_empty detector + Other-retry reasoning-off.

A model call that returns valid JSON which parses to an empty
structure (insights' `{"cross_domain":[],"critical":[]}` — the m7pp
class) is a failure mode worth retrying. Pre-fix it was labelled
plain `success` and silently produced nothing. The minimal fix wires
parsed-empty into the existing Other retry bucket, and (per the spec
update that dropped line 213's no-retry carve-out for valid empties)
ANY Other retry flips reasoning off when the original call had it on.

Three wire-ups, exercised here:

  1. Per-stage parser raises `_SuccessEmpty` on parsed-empty (insights
     parser is the m7pp surface).
  2. `_SuccessEmpty` is a `_PostStreamFailure` with `bucket="other"`,
     so the existing classifier dispatches it through the Other arm
     with no fork.
  3. The cascade's Other dispatch sites flip `reasoning_off_used` and
     tag the chain `/reasoning-off` when the parent had reasoning on,
     reusing the existing `_force_reasoning_off` plumbing already
     wired for the Sizing reasoning-off step.

No live LLM calls. Run with:
    cd engine && pytest tests/test_success_empty_retry.py -v
"""
from __future__ import annotations

import json

import pytest


from engine import llm
from engine import insights as insights_mod
from engine import patterns as patterns_mod
from engine import runner
from engine.content_extractor import (
    Entity, EntityRef, EvidenceSpan, ExtractedItem,
)
from engine.ingestor import Document, SourceType
from engine.llm import CompletionResult
from engine.retry import (
    _PostStreamFailure,
    _SuccessEmpty,
)


_call_id_counter = [0]


def _mk_result(content: str, *, finish_reason: str = "stop",
               reasoning_tokens: int = 0,
               completion_tokens: int = 200,
               call_id: str | None = None) -> CompletionResult:
    _call_id_counter[0] += 1
    cid = call_id or f"test-{_call_id_counter[0]:04d}"
    return CompletionResult(
        content=content, call_id=cid, cache_key=f"key-{cid}",
        cached=False, finish_reason=finish_reason,
        model="m", mode="test",
        prompt_tokens=10, completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens, reasoning_tokens_source=None,
        content_tokens=completion_tokens, ttft_ms=5, ttfr_ms=None,
        last_token_ms=5, max_tokens_reserved=4096,
    )


def _mk_fact(summary: str, confidence: float = 1.0) -> ExtractedItem:
    return ExtractedItem(
        item_type="fact",
        summary=summary,
        evidence=[EvidenceSpan(text=summary, source_ref="src")],
        entities=[EntityRef(
            entity=Entity(name="Alice", entity_type="person"),
            role="subject",
        )],
        topics=["work"],
        tags=[],
        confidence=confidence,
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Reset call-warning + stat-record state and quiet logging."""
    llm.reset_stat_records()
    llm.reset_call_warnings()
    monkeypatch.setattr(insights_mod, "_log_exception",
                        lambda *a, **kw: None)
    monkeypatch.setattr(patterns_mod, "_log_exception",
                        lambda *a, **kw: None)
    monkeypatch.setattr(patterns_mod, "_log_info",
                        lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_log_write", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_emit", lambda *a, **kw: None)
    yield
    llm.reset_stat_records()
    llm.reset_call_warnings()


# ── _SuccessEmpty exception is in the post-stream-failure family ────────────


class TestSuccessEmptyExceptionShape:
    """`_SuccessEmpty` lives in retry.py next to `_EmptyResponse` /
    `_ParseError` / `_InterruptedResponse`. Carries `bucket="other"`
    so the classifier dispatches it without a special case."""

    def test_is_post_stream_failure_subclass(self):
        assert issubclass(_SuccessEmpty, _PostStreamFailure)

    def test_bucket_is_other(self):
        assert _SuccessEmpty.bucket == "other"

    def test_stat_field_is_success_empty(self):
        assert _SuccessEmpty.stat_field == "success_empty"

    def test_carries_stage_on_instance(self):
        exc = _SuccessEmpty(stage="insights")
        assert exc.stage == "insights"
        assert "success_empty" in str(exc)


# ── Classifier routes _SuccessEmpty to "other" ──────────────────────────────




# ── _classify_outcome surfaces success_empty for the _SuccessEmpty rec ──────


class TestSuccessEmptyOutcomeLabel:
    """A rec with `error.class = "retry._SuccessEmpty"` must label as
    `OUTCOME_SUCCESS_EMPTY`, NOT `OUTCOME_FAILED_OTHER`. Pre-fix the
    label fell through to the generic-failure bucket because there was
    no specific branch for the class; the fix adds one so the
    produce-nothing signal is honest in run records."""

    def test_rec_with_success_empty_class_labels_as_success_empty(self):
        rec = {
            "success": False,
            "error": {
                "class": "retry._SuccessEmpty",
                "message": "success_empty (stage=insights)",
            },
            "aborted": False,
            "parse_error": False,
            "empty_response": False,
            "output": None,
            "input": None,
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "duration_ms": 5000,
            "ttft_ms": 100,
            "success_empty": True,
        }
        assert runner._classify_outcome(rec) == runner.OUTCOME_SUCCESS_EMPTY


# ── Insights parser raises _SuccessEmpty on parsed-empty (m7pp class) ───────


class TestInsightsParserRaisesOnParsedEmpty:
    """The m7pp class: model returned a valid JSON object with both
    `cross_domain` and `critical` arrays empty. Pre-fix the parser
    returned the raw unchanged (the call completed as plain success).
    Post-fix the parser raises `_SuccessEmpty(stage="insights")` so
    the cascade routes it through Other for one retry."""

    def test_both_arrays_empty_raises_success_empty(self):
        raw = json.dumps({"cross_domain": [], "critical": []})
        with pytest.raises(_SuccessEmpty) as ei:
            insights_mod._validate_insights_or_raise(raw)
        assert ei.value.stage == "insights"

    def test_missing_arrays_raises_success_empty(self):
        # Defensive: a model that omits both keys entirely is the same
        # produce-nothing signal as `[] + []`.
        raw = json.dumps({})
        with pytest.raises(_SuccessEmpty):
            insights_mod._validate_insights_or_raise(raw)

    def test_only_cross_domain_populated_returns_normally(self):
        # Mixed: cross_domain has content, critical empty → success.
        raw = json.dumps({
            "cross_domain": [{
                "title": "x",
                "narrative": "...",
                "source_pattern_ids": [1, 2],
            }],
            "critical": [],
        })
        out = insights_mod._validate_insights_or_raise(raw)
        assert out == raw

    def test_only_critical_populated_returns_normally(self):
        raw = json.dumps({
            "cross_domain": [],
            "critical": [{
                "title": "y",
                "narrative": "...",
                "source_pattern_ids": [3],
            }],
        })
        out = insights_mod._validate_insights_or_raise(raw)
        assert out == raw

    def test_whitespace_still_routes_to_empty_response(self):
        # `_EmptyResponse` (wire-empty / zero-content-token) keeps its
        # existing path — whitespace-only raw is NOT the m7pp class.
        from engine.retry import _EmptyResponse
        with pytest.raises(_EmptyResponse):
            insights_mod._validate_insights_or_raise("   \n  ")

    def test_unparseable_still_routes_to_parse_error(self):
        # `_ParseError` (sizing) keeps its existing path — broken JSON
        # is NOT the m7pp class.
        from engine.retry import _ParseError
        with pytest.raises(_ParseError):
            insights_mod._validate_insights_or_raise("not json {")

    def test_non_dict_shape_passes_through(self):
        # A list-shaped response (wrong outer shape for insights) is
        # NOT the m7pp class — only the canonical dict shape triggers
        # success_empty here. Non-dict falls through so the downstream
        # `_parse_output` flags parse_error (sizing-halve cascade)
        # rather than being mislabeled as parsed-empty.
        raw = json.dumps([{"title": "x"}])
        assert insights_mod._validate_insights_or_raise(raw) == raw


# ── Other synthesis stages' parsers also raise on parsed-empty ──────────────


class TestPatternsParserRaisesOnEmptyList:
    """Patterns returns a JSON array of pattern dicts. Empty array →
    `_SuccessEmpty`. Non-empty → success."""

    def test_empty_list_raises(self):
        from engine import patterns as patterns_mod
        with pytest.raises(_SuccessEmpty) as ei:
            patterns_mod._validate_patterns_or_raise("[]")
        assert ei.value.stage == "patterns"

    def test_non_empty_list_returns_normally(self):
        from engine import patterns as patterns_mod
        raw = json.dumps([{"name": "x", "description": "d"}])
        assert patterns_mod._validate_patterns_or_raise(raw) == raw


class TestActionsParserRaisesOnEmptyList:
    """Actions returns a JSON array of action dicts. Same shape as
    patterns — empty array → `_SuccessEmpty`."""

    def test_empty_list_raises(self):
        from engine import actions as actions_mod
        with pytest.raises(_SuccessEmpty) as ei:
            actions_mod._validate_actions_or_raise("[]")
        assert ei.value.stage == "actions"

    def test_non_empty_list_returns_normally(self):
        from engine import actions as actions_mod
        raw = json.dumps([{"title": "y", "narrative": "..."}])
        assert actions_mod._validate_actions_or_raise(raw) == raw


def _extract_parser():
    """The parser callable actually wired into the extract cascade —
    `_make_doc_parser(doc)`, exercised against a throwaway doc. The doc
    content is irrelevant to the empty/parse verdicts (the item loop
    that uses it never runs on an empty parse)."""
    from engine.content_extractor import _make_doc_parser
    doc = Document(
        id="d1",
        source_path="fixture",
        source_type=SourceType.TXT,
        content="some document content with a quote in it",
        title="fixture",
        date="2026-04-15",
        file_id="fixture",
        origin_char=0,
    )
    return _make_doc_parser(doc)


class TestExtractParserRaisesOnEmpty:
    """Extract's wired parser must route a clean parse with zero items
    through `_SuccessEmpty` (one Other retry, reasoning forced off)
    rather than recording it as plain success.

    The empty-check used to live on a standalone
    `_validate_extract_or_raise` that was never wired in AND keyed on a
    phantom `facts` field the schema never produces (it emits `items`).
    The real parser `_make_doc_parser` had no empty-check at all, so an
    `items: []` response was recorded plain-success during the run while
    the display classifier — reading `output={"facts": len(items)}` = 0
    — labelled it `success_empty`. The two sides disagreed; the earlier
    success_empty rollout left extract's empty path display-only, never
    actually retried. These tests pin the fix against the parser that
    runs in production."""

    def test_whitespace_raises(self):
        with pytest.raises(_SuccessEmpty) as ei:
            _extract_parser()("   \n  ")
        assert ei.value.stage == "extract"

    def test_empty_list_raises(self):
        with pytest.raises(_SuccessEmpty):
            _extract_parser()("[]")

    def test_empty_object_raises(self):
        with pytest.raises(_SuccessEmpty):
            _extract_parser()("{}")

    def test_envelope_empty_items_raises(self):
        """The reported prod symptom (gpt-oss-120b, v0.1.43): ~40 good
        split_summaries but `items: []`. Must raise `_SuccessEmpty` so
        it eats an Other retry — NOT pass through as plain success."""
        raw = json.dumps({
            "split_summaries": [
                {"id": "d1", "summary": "about the day's events"},
                {"id": "d1", "summary": "more durable context"},
            ],
            "items": [],
        })
        with pytest.raises(_SuccessEmpty) as ei:
            _extract_parser()(raw)
        assert ei.value.stage == "extract"

    def test_envelope_full_items_returns(self):
        """Acceptance contrast: a real extraction (non-empty `items`)
        must pass through unchanged — no false success_empty."""
        raw = json.dumps({
            "split_summaries": [{"id": "d1", "summary": "about x"}],
            "items": [{"type": "fact", "summary": "a durable fact"}],
        })
        assert _extract_parser()(raw) == raw

    def test_phantom_facts_key_raises(self):
        """A response that puts the facts under the phantom `facts` key
        (the schema asks for `items`) yields zero parsed items → empty →
        `_SuccessEmpty`. Confirms the empty-check keys on the field the
        schema/parser actually produce, not on `facts`."""
        raw = json.dumps({"facts": [{"type": "fact", "summary": "x"}]})
        with pytest.raises(_SuccessEmpty):
            _extract_parser()(raw)

    def test_unparseable_still_routes_to_parse_error(self):
        from engine.retry import _ParseError
        with pytest.raises(_ParseError):
            _extract_parser()("not json {")


# ── Other-retry flips reasoning off when parent had reasoning on ────────────






# ── End-to-end: patterns cascade with m7pp-style empty result ───────────────




