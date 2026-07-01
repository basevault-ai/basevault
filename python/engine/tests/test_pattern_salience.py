"""Pattern salience: parser keeps singletons, prompts surface fact_count.

Pins:
  - Singleton patterns (one cited fact) survive `_parse_patterns`.
  - Patterns whose every cited ID is hallucinated drop silently —
    no observable pattern to surface.
  - `_parse_patterns` returns `(patterns, parse_error)`.
  - Insights and actions prompts annotate each pattern with its
    `len(source_facts)` so the synthesizer can weight by recurrence.

Run with:
    cd engine && pytest tests/test_pattern_salience.py -v
"""
from __future__ import annotations

import json


from engine.patterns import Pattern, _parse_patterns, build_context_block


# ── _parse_patterns: singletons survive, fully-hallucinated drop ────────


def test_singleton_pattern_survives_parser():
    """One cited source is enough — the parser keeps the pattern."""
    raw = json.dumps([
        {
            "name": "Solo observation",
            "description": "Cited by exactly one fact.",
            "kind": "blind-spot",
            "count": 1,
            "sources": [{"id": 1, "confidence": 0.9}],
        }
    ])
    # dedup_to_orig: deduped position 0 maps back to original facts [4].
    parsed, parse_err = _parse_patterns(raw, "work", [[4]])
    assert parse_err is False
    assert len(parsed) == 1
    p = parsed[0]
    assert p.name == "Solo observation"
    assert p.source_facts == [(4, 0.9)]


def test_fully_hallucinated_pattern_still_drops():
    """A pattern whose every cited ID is out-of-range produces an
    empty source_facts list and is dropped silently — there's no
    observable pattern there to surface."""
    raw = json.dumps([
        {
            "name": "All hallucinated",
            "description": "Cites IDs that don't exist.",
            "kind": "defensive-loop",
            "count": 2,
            "sources": [
                {"id": 99, "confidence": 1.0},
                {"id": 100, "confidence": 1.0},
            ],
        }
    ])
    # Only 2 deduped facts (positions 0, 1 → IDs 1, 2). 99 and 100
    # are out of range.
    parsed, parse_err = _parse_patterns(raw, "work", [[0], [1]])
    assert parse_err is False
    assert parsed == []


def test_mixed_singleton_and_multi_source_kept_together():
    """Two patterns: one singleton, one multi-source. Both survive."""
    raw = json.dumps([
        {
            "name": "Singleton",
            "description": "d",
            "kind": None,
            "count": 1,
            "sources": [{"id": 1, "confidence": 1.0}],
        },
        {
            "name": "Multi",
            "description": "d",
            "kind": None,
            "count": 3,
            "sources": [
                {"id": 1, "confidence": 1.0},
                {"id": 2, "confidence": 0.8},
            ],
        },
    ])
    parsed, parse_err = _parse_patterns(raw, "work", [[0], [1]])
    assert parse_err is False
    assert [p.name for p in parsed] == ["Singleton", "Multi"]
    assert len(parsed[0].source_facts) == 1
    assert len(parsed[1].source_facts) == 2


# ── Insights prompt surfaces fact_count ────────────────────────────────


def test_insights_prompt_renders_fact_count_per_pattern():
    """Each rendered pattern carries its `len(source_facts)` in a
    `facts: N` annotation, and the prompt instructs the model to
    weight by recurrence (singletons inform but don't dominate)."""
    from engine.insights import _build_prompt as build_insights_prompt

    patterns_by_topic = {
        "work": [
            Pattern(name="Recurrent",  description="d", domain="work",
                    count=8, kind="defensive-loop",
                    source_facts=[(i, 1.0) for i in range(8)]),
            Pattern(name="Singleton",  description="d", domain="work",
                    count=1, kind="blind-spot",
                    source_facts=[(0, 0.9)]),
        ],
    }
    prompt, _ = build_insights_prompt(
        patterns_by_topic, cross_cap=2, critical_cap=2,
    )
    assert "facts: 8" in prompt
    assert "facts: 1" in prompt
    # The recurrence-weighting instruction must be present so the
    # model knows what the count is FOR.
    assert "Weight by recurrence" in prompt
    assert "Singletons" in prompt


# ── Actions: build_context_block surfaces fact_count ───────────────────


def test_actions_patterns_reference_renders_fact_count_per_pattern():
    """`build_context_block` renders each pattern with its
    `facts=N` annotation alongside `count`. The recurrence-weighting
    instruction is appended so the actions prompt sees both signal
    and guidance."""
    patterns_by_topic = {
        "work": [
            Pattern(name="Recurrent",  description="d", domain="work",
                    count=8, kind="defensive-loop",
                    source_facts=[(i, 1.0) for i in range(8)]),
            Pattern(name="Singleton",  description="d", domain="work",
                    count=1, kind="blind-spot",
                    source_facts=[(0, 0.9)]),
        ],
    }
    block = build_context_block(patterns_by_topic)
    # Every pattern shows facts=N alongside count=N.
    assert "facts=8" in block
    assert "facts=1" in block
    assert "count=8" in block
    assert "count=1" in block
    # The instruction footer pinning what `facts` is for.
    assert "Weight by recurrence" in block
    assert "Singletons" in block


def test_actions_patterns_reference_empty_when_no_patterns():
    """No patterns → empty string (caller skips the prepend). The
    weighting footer must not appear on its own."""
    assert build_context_block({}) == ""
    assert build_context_block({"work": []}) == ""


def test_actions_patterns_reference_no_per_topic_cap_or_truncation():
    """Issue #257 piece 1: actions consumes the full pattern body —
    every pattern is rendered with full description, no per-topic
    cap, no character clamp. The WR-sample cascade is the escape
    valve, not pre-emptive truncation."""
    long_desc = "x" * 1000  # well over the previous 220-char cap
    patterns_by_topic = {
        "work": [
            Pattern(name=f"P{i}", description=long_desc, domain="work",
                    count=20 - i, kind="defensive-loop",
                    source_facts=[(0, 1.0)])
            for i in range(20)  # over the previous 10-per-topic cap
        ],
    }
    block = build_context_block(patterns_by_topic)
    # Every pattern present (no cap).
    for i in range(20):
        assert f"P{i}" in block
    # No "more omitted" footer — caps removed.
    assert "more omitted" not in block
    assert "more work patterns" not in block
    # Description rendered in full (no 220-char clamp). Count
    # occurrences of the long desc pattern; with 20 patterns each
    # carrying it, we should see all 20.
    assert block.count(long_desc) == 20
