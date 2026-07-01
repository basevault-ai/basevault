"""
Golden-hash tests for every stage that produces an LLM call.

The tests pin a small fixture per stage, assemble the full
`messages` list the way each stage's `run()` does, and compute the
prompt-cache key via `llm_cache.compute_cache_key`. They assert:

1. **Determinism** — assembling the same fixture twice yields the
   exact same key, byte-for-byte.
2. **Golden** — the key matches a stored value in
   `tests/golden_prompt_hashes.json`. Any future change that
   introduces non-determinism (e.g. someone slips `datetime.now()` or
   a random UUID into a prompt) flips the second assertion immediately
   in CI.

To regenerate the goldens after an INTENTIONAL prompt change:

    BASEVAULT_UPDATE_GOLDENS=1 pytest tests/test_golden_prompt_hashes.py

This rewrites `golden_prompt_hashes.json`. Review the diff in the PR
to confirm the prompt change is the one you meant to make.

Stage coverage:
- extract           (`content_extractor`)
- entities          (`entities` per-batch)
- entities_dedupe   (`entities` _llm_dedupe)
- patterns          (`patterns` per-topic)
- insights          (`insights` single call)
- actions           (`actions` single call)
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest

from engine import llm_cache


GOLDEN_PATH = Path(__file__).parent / "golden_prompt_hashes.json"

# Every fixture below MUST pin any date that flows into a prompt to a
# fixed value. `date.today()` / `datetime.now()` would make the
# golden hash drift across calendar days even though no prompt source
# changed — the entire point of the golden test is to catch silent
# non-determinism. Stages that touch dates today: actions (today is
# rendered into the prompt), extract (doc.date is interpolated),
# patterns (occurred_at is
# rendered next to each fact). Pinning convention: pipeline
# date-of-occurrence inputs use 2025-01-15 / 2025-01-17 and the
# actions "today" snapshot uses 2026-01-01 — same sentinel as
# test_harm_gate / test_exception_logging / test_sentiment_plumbing.
_GOLDEN_TODAY = date(2026, 1, 1)
_GOLDEN_FACT_DATE = "2025-01-15"
_GOLDEN_FACT_DATE_2 = "2025-01-17"


def _load_goldens() -> dict[str, str]:
    if not GOLDEN_PATH.exists():
        return {}
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _save_goldens(goldens: dict[str, str]) -> None:
    GOLDEN_PATH.write_text(
        json.dumps(goldens, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _key_for(messages: list[dict]) -> str:
    """All goldens use the same nominal model + temperature so the
    hash is purely a function of the prompt text. The cache key
    function is what production uses, so the test surfaces ANY
    behavior change in either prompt assembly OR key computation."""
    return llm_cache.compute_cache_key(
        "golden-test-provider", messages, "golden-test-model", 0.0, {},
    )


def _assert_golden(
    name: str, messages: list[dict], updated: dict[str, str]
) -> None:
    """Determinism check + golden comparison. Updates the in-memory
    `updated` dict for an end-of-run rewrite when running in
    BASEVAULT_UPDATE_GOLDENS=1 mode."""
    k1 = _key_for(messages)
    k2 = _key_for(messages)
    assert k1 == k2, f"{name}: key not deterministic across two calls"

    if os.environ.get("BASEVAULT_UPDATE_GOLDENS") == "1":
        updated[name] = k1
        return

    goldens = _load_goldens()
    expected = goldens.get(name)
    if expected is None:
        pytest.skip(
            f"No golden recorded for {name!r}. "
            f"Run `BASEVAULT_UPDATE_GOLDENS=1 pytest "
            f"tests/test_golden_prompt_hashes.py` to seed."
        )
    assert k1 == expected, (
        f"{name}: prompt-hash drifted.\n"
        f"  expected (golden): {expected}\n"
        f"  got              : {k1}\n"
        f"This means a prompt-assembly change happened. If intentional, "
        f"regenerate with BASEVAULT_UPDATE_GOLDENS=1 pytest. If not, "
        f"check for non-determinism (datetime.now, UUIDs, set ordering) "
        f"or accidental prompt edits."
    )


# ── Per-stage fixtures + assembly ────────────────────────────────────────────

def _build_extract_messages() -> list[dict]:
    """content_extractor prompt for one Document. Mirrors content_extractor.extract_items()'s
    per-doc loop."""
    from engine.content_extractor import _SYSTEM, _build_prompt
    from engine.ingestor import Document, SourceType
    doc = Document(
        id="fixture.md",
        source_path="fixture.md",
        source_type=SourceType.MD_FILE,
        content="On 2025-01-15 Alice noted she'd start running every Monday.",
        title="Fixture",
        date=_GOLDEN_FACT_DATE,
        file_id="fixture.md",
    )
    user = _build_prompt(doc)
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def _build_entities_messages() -> list[dict]:
    """One per-entity batch as the entities stage emits it. Bypasses
    the heavier `_pack_batches` / `_render_entity_block` machinery by
    using the same `_build_prompt(blocks, subject, n_groups, is_bundle)`
    entry, with hand-crafted blocks. The blocks themselves are
    pinned strings — they'd normally be assembled from real groups,
    but the prompt-determinism property we're testing is at the
    `_TASK.format(...)` boundary."""
    from engine.entities import _SYSTEM, _build_prompt
    blocks = [
        "    entity g1 (person): Alice\n"
        "    aliases: Alice, A.\n"
        "    facts:\n"
        "      [2025-01-15] (health/0) Alice started running\n",
        "    entity g2 (person): Bob\n"
        "    aliases: Bob\n"
        "    facts:\n"
        "      [2025-01-16] (work/0) Bob filed a report\n",
    ]
    user = _build_prompt(blocks, subject="the author",
                         n_groups=len(blocks), is_bundle=False)
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def _build_entities_dedupe_messages() -> list[dict]:
    """The dedupe stage's single-call prompt. Mirrors entities._llm_dedupe()."""
    from engine.entities import _DEDUPE_SYSTEM, _DEDUPE_TASK, _format_dedupe_row, EntityRecord
    records = [
        EntityRecord(
            canonical_id="alice-1",
            canonical_name="Alice",
            entity_type="person",
            aliases=["Alice", "A."],
            role="subject",
            description="The author of the journal.",
            mention_count=5,
            topics=["health"],
            evidence_fact_refs=[],
        ),
        EntityRecord(
            canonical_id="alice-2",
            canonical_name="Alice Smith",
            entity_type="person",
            aliases=["Alice Smith"],
            role="subject",
            description="Same person as Alice.",
            mention_count=2,
            topics=["health"],
            evidence_fact_refs=[],
        ),
    ]
    rows = "\n".join(_format_dedupe_row(r) for r in records)
    user = _DEDUPE_TASK.format(n=len(records), rows=rows)
    return [
        {"role": "system", "content": _DEDUPE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _build_patterns_messages() -> list[dict]:
    """Per-topic patterns prompt. Mirrors patterns._build_messages()."""
    from engine.patterns import _build_messages
    from engine.content_extractor import ExtractedItem, EvidenceSpan, EntityRef, Entity
    facts = [
        ExtractedItem(
            item_type="fact",
            summary="Alice ran 5km on Monday.",
            evidence=[EvidenceSpan(text="ran 5km", source_ref="fixture.md")],
            occurred_at=_GOLDEN_FACT_DATE,
            entities=[EntityRef(entity=Entity(name="Alice", entity_type="person"),
                                role="subject")],
            topics=["health"],
        ),
        ExtractedItem(
            item_type="fact",
            summary="Alice ran 6km on Wednesday.",
            evidence=[EvidenceSpan(text="ran 6km", source_ref="fixture.md")],
            occurred_at=_GOLDEN_FACT_DATE_2,
            entities=[EntityRef(entity=Entity(name="Alice", entity_type="person"),
                                role="subject")],
            topics=["health"],
        ),
    ]
    return _build_messages(
        facts, topic="health", hard_cap=3,
        subject="the author", entities_context=None,
    )


def _build_insights_messages() -> list[dict]:
    """Insights single-call prompt. Mirrors insights.detect_insights()."""
    from engine.insights import (
        _SYSTEM, _SUBJECT_DISCIPLINE, _EVENT_VS_INSIGHT,
        _SENTIMENT_FRAMING, _SENTIMENT_BIAS_CLAUSES, _build_prompt,
    )
    from engine.patterns import Pattern
    patterns_by_topic = {
        "health": [
            Pattern(name="Consistent runner", description="Runs multiple times per week.",
                    domain="health", kind="behavior", count=3,
                    source_facts=[(0, 1.0), (1, 0.8)]),
        ],
        "work": [
            Pattern(name="Diligent reporter", description="Files reports on time.",
                    domain="work", kind="behavior", count=2,
                    source_facts=[(0, 1.0), (1, 0.9)]),
        ],
    }
    prompt, _ = _build_prompt(patterns_by_topic, cross_cap=2, critical_cap=1)
    sentiment_block = _SENTIMENT_FRAMING.format(
        sentiment_clause=_SENTIMENT_BIAS_CLAUSES["neutral"],
    )
    system_content = (
        _SYSTEM
        + "\n\n" + _SUBJECT_DISCIPLINE.format(subject="the author")
        + "\n\n" + _EVENT_VS_INSIGHT
        + "\n\n" + sentiment_block
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]


def _build_actions_messages() -> list[dict]:
    """Actions single-call prompt. Mirrors actions.generate_actions().

    Note: `_GOLDEN_TODAY` is no longer threaded into the actions
    prompt — the LLM emits a horizon enum and the runner computes
    review_date from (horizon, today) AFTER the call. Same hash on
    any calendar day; the date-pinning convention at the top of this
    file is a holdover for stages that still interpolate dates
    (extract / patterns)."""
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
                proposed_actions=["Stack new habit on existing routine."],
                source_patterns=[("health", 0, 1.0)],
            ),
        ],
        critical=[],
    )
    prompt, _ = _build_prompt(insight_output, max_actions=3)
    sentiment_block = _SENTIMENT_FRAMING.format(
        sentiment_clause=_SENTIMENT_BIAS_CLAUSES["neutral"],
    )
    system_content = "\n\n".join([
        _SYSTEM,
        _SUBJECT_DISCIPLINE.format(subject="the author"),
        _EVENT_VS_HORIZON,
        sentiment_block,
        _HARM_CLAUSE,
    ])
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]


# ── Tests ────────────────────────────────────────────────────────────────────

# The seven stages that produce LLM calls in the pipeline. Each entry
# is (stage_name, builder); the test runs them all under one runner so
# the goldens-update mode can rewrite the JSON in one pass.
_STAGES = [
    ("extract",         _build_extract_messages),
    ("entities",        _build_entities_messages),
    ("entities_dedupe", _build_entities_dedupe_messages),
    ("patterns",        _build_patterns_messages),
    ("insights",        _build_insights_messages),
    ("actions",         _build_actions_messages),
]


@pytest.fixture(scope="module")
def _updated_goldens():
    """Collect updates across the parameterized run so we write the
    JSON once at module teardown, not once per stage."""
    updated: dict[str, str] = {}
    yield updated
    if os.environ.get("BASEVAULT_UPDATE_GOLDENS") == "1" and updated:
        merged = _load_goldens()
        merged.update(updated)
        _save_goldens(merged)


@pytest.mark.parametrize("name,builder", _STAGES,
                         ids=[name for name, _ in _STAGES])
def test_stage_prompt_hash(name, builder, _updated_goldens):
    messages = builder()
    _assert_golden(name, messages, _updated_goldens)


def test_actions_prompt_is_calendar_day_stable():
    """The actions stage's prompt MUST be date-free post-this-PR so the
    cache survives across calendar days. Build the prompt with no date
    in the loop (the builder doesn't take one) and assert that the
    rendered text contains no YYYY-MM-DD pattern. Catches a regression
    where someone re-introduces today.isoformat() into the prompt or
    references a fact's occurred_at directly in the action-side
    template.

    Date-free at the prompt boundary is what makes the hash stable
    across days — combined with the runner-side review_date
    computation, the actions stage is fully cache-replayable on day N
    AND day N+30 with no provider call."""
    import re as _re
    msgs = _build_actions_messages()
    full = "\n".join(m["content"] for m in msgs)
    iso_dates = _re.findall(r"\b\d{4}-\d{2}-\d{2}\b", full)
    assert iso_dates == [], (
        f"actions prompt contains literal date(s) {iso_dates}; "
        f"this breaks calendar-day cache stability. Move any "
        f"date-derived text out of the prompt; let the runner "
        f"compute it post-LLM-call."
    )
