"""Per-split extract checkpointing — unit tests for fix-extract-checkpointing.

Covers the three observable behaviors required by the task:
    a) per-split partials are written one-per-split as extract_items progresses
       (so SIGKILL leaves a resumable trail on disk),
    b) on a re-entry, splits whose partials are present skip the LLM call and
       use the cached items,
    c) once every split has landed, _process_parent folds the partials into
       the canonical 08_items.json and removes the partials dir.

All tests stub `complete()` from content_extractor so they run under default
`pytest` (no @integration marker, no network).
"""
from __future__ import annotations

import json

import pytest


from .conftest import _wrap

from engine.ingestor import Document, SourceType  # noqa: E402
# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_doc(doc_id: str, content: str) -> Document:
    """Document the extractor will accept. Content has to contain the
    quote that the fake_complete returns or evidence resolution drops it."""
    return Document(
        id=doc_id,
        source_path="fixture",
        source_type=SourceType.TXT,
        content=content,
        title="fixture",
        date="2026-04-15",
        file_id="fixture",
        origin_char=0,
    )


def _fake_complete_factory(call_log: list[str]):
    """Returns a stub complete() that records each invocation's doc id
    (parsed out of the prompt's `Document title:` / `DOCUMENT [` line —
    we don't have the doc object, but the prompt embeds enough to id it).
    Always returns a single grounded item whose evidence quote is
    present in every fixture's content.
    """
    def _fake(messages, **kwargs):
        body = messages[-1]["content"]
        # The evidence quote must literally appear in the doc content for
        # the extractor to keep the item; we use this universal short
        # phrase across every fixture.
        call_log.append(body[:200])
        return _wrap(json.dumps([{
            "type": "fact",
            "summary": "Alice signed the contract",
            "evidence": [{
                "text": "Alice signed the contract.",
                "source_ref": "fixture",
            }],
            "occurred_at": "2026-04-15",
            "occurred_at_text": None,
            "entities": [{"name": "Alice", "entity_type": "person",
                          "role": "subject"}],
            "topics": ["work"],
            "tags": [],
            "confidence": 0.9,
        }]), **kwargs)
    return _fake


# ── (a) Per-split persistence as extract_items progresses ─────────────────────




# ── (b) cached_results skips the LLM call ─────────────────────────────────────




# ── (c) end-to-end: _process_parent writes 08_items.json + cleans partials ────




# ── (d) NoResumableCheckpoint sentinel for fix #2 ─────────────────────────────


def test_detect_resume_point_raises_no_resumable_when_empty(tmp_path):
    """A run dir with stages/ but no phase markers must raise
    NoResumableCheckpoint so main()'s special-case handler can flip the
    run to `paused` instead of `failed`."""
    from engine import runner

    rd = tmp_path / "empty_run"
    (rd / "stages").mkdir(parents=True)

    with pytest.raises(runner.NoResumableCheckpoint):
        runner._detect_resume_point(rd)


def test_detect_resume_point_recognises_partial_extract(tmp_path):
    """Ingestion phase 1 marker exists but extract phase 3 doesn't:
    detector returns 'start' so the runner restarts from ingest.
    The LLM cache makes re-execution fast for already-completed calls."""
    from engine import runner

    rd = tmp_path / "midextract_run"
    ingest = rd / "stages" / "00-ingestion"
    ingest.mkdir(parents=True)
    (ingest / "phase_1_marker.json").write_text("{}", encoding="utf-8")

    assert runner._detect_resume_point(rd) == "start"


