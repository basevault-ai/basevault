"""Tests for the synthesis-stage cap formulas.

Three formulas, all sub-linear in corpus size and clamped at the top:
  - patterns: min(75, round(n_facts_in_topic ** 0.6))
  - insights: min(10, round(ln(max(1, total_facts)))) split 60/40 cross/critical
  - actions:  min(10, round(ln(max(1, total_facts))))

Run with:
    cd engine && pytest tests/test_cap_formulas.py -v
"""
from __future__ import annotations



from engine.patterns import _cap as patterns_cap, _PATTERN_CAP_CEILING
from engine.insights import insight_caps, _INSIGHTS_TOTAL_CEILING
from engine.actions import action_cap, _ACTION_CAP_CEILING


# ── Patterns ─────────────────────────────────────────────────────────────────


class TestPatternsCap:
    def test_zero_input_returns_zero(self):
        # Degenerate. No floor; the prompt + parser already require ≥2
        # supporting facts per pattern, so a 0-fact topic produces 0
        # patterns regardless of cap.
        assert patterns_cap(0) == 0

    def test_small_topic_uses_formula(self):
        # n=20: 20**0.6 ≈ 6.03 → 6. No floor.
        assert patterns_cap(20) == round(20 ** 0.6)
        # Below the previous fixed-cap of 8 — that's expected; small
        # topics legitimately don't earn 8 patterns.
        assert patterns_cap(20) < 8

    def test_medium_topic(self):
        # n=100: 100**0.6 ≈ 15.85 → 16. Roughly the old fixed cap of
        # 16 for 20≤n<100, so this transition point is smooth.
        assert patterns_cap(100) == 16

    def test_large_topic_scales_above_old_fixed_cap(self):
        # n=500: 500**0.6 ≈ 41.6 → 42. Old code returned 24.
        assert patterns_cap(500) == 42
        assert patterns_cap(500) > 24

    def test_very_large_topic_hits_ceiling(self):
        # Ceiling is 75. n=1500 ** 0.6 ≈ 75.4, just above; n=5000 well
        # above. Both clamp.
        assert patterns_cap(1500) == _PATTERN_CAP_CEILING
        assert patterns_cap(5000) == _PATTERN_CAP_CEILING
        # Sanity: ceiling is 75 (constant).
        assert _PATTERN_CAP_CEILING == 75

    def test_monotonic(self):
        prev = -1
        for n in (1, 5, 20, 50, 100, 200, 500, 1000, 2000, 10000):
            cur = patterns_cap(n)
            assert cur >= prev, f"non-monotonic at n={n}: {prev} → {cur}"
            prev = cur


# ── Insights ─────────────────────────────────────────────────────────────────


class TestInsightCaps:
    def test_zero_corpus_yields_zero_caps(self):
        assert insight_caps(0) == (0, 0, 0)

    def test_tiny_corpus_below_e_yields_zero(self):
        # round(ln(2)) = 1. round(ln(1)) = 0.
        # ln(2.71828) ≈ 1.0 → round = 1; ln(1.6) → 0.47 → round = 0.
        # We only get an integer corpus size though, so n=1 → 0, n=2 → 1.
        total, cross, critical = insight_caps(1)
        assert (total, cross, critical) == (0, 0, 0)

    def test_small_corpus_yields_one_or_two(self):
        # n=5: ln(5) ≈ 1.609 → 2. cross=round(0.6*2)=1, critical=1.
        assert insight_caps(5) == (2, 1, 1)

    def test_medium_corpus_yields_round_log_total(self):
        # n=100: ln(100) ≈ 4.605 → 5. cross=3, critical=2.
        # Matches the old hardcoded "max 3 + max 2" by coincidence at
        # ~100 facts. That's intentional — the formula was tuned so
        # the typical case lands near the previous behavior.
        assert insight_caps(100) == (5, 3, 2)

    def test_large_corpus_scales_above_old_cap(self):
        # n=1000: ln ≈ 6.908 → 7. cross=4, critical=3.
        assert insight_caps(1000) == (7, 4, 3)
        # Old hardcoded cap was 5 (3 + 2). New formula exceeds it.
        total, _, _ = insight_caps(1000)
        assert total > 5

    def test_below_ceiling_formula_untouched(self):
        # n=10000: ln ≈ 9.21 → 9, below the ceiling. The clamp must be
        # a no-op for everything under ~22K facts.
        assert insight_caps(10_000) == (9, 5, 4)

    def test_at_threshold_yields_exactly_ten(self):
        # n=22000: ln ≈ 9.999 → 10 — the formula reaches the ceiling
        # on its own, unclamped. cross=round(0.6*10)=6, critical=4.
        assert insight_caps(22_000) == (10, 6, 4)

    def test_large_corpus_clamps_to_ceiling(self):
        # n=100K: ln ≈ 11.51 → 12 unclamped; n=1M: ln ≈ 13.8 → 14.
        # Both clamp to 10, and the sub-split still sums to total.
        for n in (100_000, 1_000_000):
            total, cross, critical = insight_caps(n)
            assert total == _INSIGHTS_TOTAL_CEILING, n
            assert (cross, critical) == (6, 4), n
        assert _INSIGHTS_TOTAL_CEILING == 10

    def test_split_sums_to_total(self):
        for n in (3, 10, 50, 200, 1000, 5000, 100_000):
            total, cross, critical = insight_caps(n)
            assert cross + critical == total, n

    def test_monotonic_total(self):
        prev = -1
        for n in (0, 1, 2, 5, 20, 100, 500, 2000, 10000, 100_000, 1_000_000):
            total, _, _ = insight_caps(n)
            assert total >= prev, f"non-monotonic at n={n}"
            prev = total


# ── Actions ──────────────────────────────────────────────────────────────────


class TestActionCap:
    def test_zero_corpus_yields_zero(self):
        assert action_cap(0) == 0

    def test_tiny_corpus_yields_zero(self):
        assert action_cap(1) == 0

    def test_small_corpus_yields_one_or_two(self):
        # n=5: round(ln(5)) = 2.
        assert action_cap(5) == 2

    def test_medium_corpus_matches_old_max(self):
        # n=100: round(ln(100)) ≈ 5. Coincides with the old _MAX_ACTIONS.
        assert action_cap(100) == 5

    def test_large_corpus_scales_above_old_max(self):
        # n=1000: round(ln(1000)) ≈ 7.
        assert action_cap(1000) == 7
        assert action_cap(1000) > 5  # old _MAX_ACTIONS

    def test_below_ceiling_formula_untouched(self):
        # n=10000: ln ≈ 9.21 → 9, below the ceiling — clamp is a no-op.
        assert action_cap(10_000) == 9

    def test_at_threshold_yields_exactly_ten(self):
        # n=22000: ln ≈ 9.999 → 10 — formula reaches the ceiling
        # on its own, unclamped.
        assert action_cap(22_000) == 10

    def test_large_corpus_clamps_to_ceiling(self):
        # n=100K: ln ≈ 11.51 → 12 unclamped; n=1M: ln ≈ 13.8 → 14.
        for n in (100_000, 1_000_000):
            assert action_cap(n) == _ACTION_CAP_CEILING, n
        assert _ACTION_CAP_CEILING == 10

    def test_monotonic(self):
        prev = -1
        for n in (0, 1, 2, 5, 20, 100, 500, 2000, 100_000, 1_000_000):
            cur = action_cap(n)
            assert cur >= prev, n
            prev = cur


# ── Caps appear in prompt ────────────────────────────────────────────────────
#
# The cap is what controls output count by being in the prompt body.
# These tests assert the cap value lands in the prompt as an integer
# (not a stale "max 3 / max 2" literal or the old "default target"
# range).


class TestCapsInPrompts:
    def test_patterns_prompt_carries_only_the_cap_no_target_range(self):
        from engine.patterns import _build_prompt
        from engine.content_extractor import (
            Entity, EntityRef, EvidenceSpan, ExtractedItem,
        )

        def _mk(s):
            return ExtractedItem(
                item_type="fact", summary=s,
                evidence=[EvidenceSpan(text=s, source_ref="t")],
                entities=[
                    EntityRef(
                        entity=Entity(name="Subject", entity_type="person"),
                        role="subject",
                    )
                ],
                topics=["work"], tags=[], confidence=1.0, occurred_at=None,
            )

        facts = [_mk(f"fact {i}") for i in range(100)]
        prompt = _build_prompt(facts, "work", hard_cap=16)

        assert "Hard cap: 16" in prompt
        # The old "Default target: lo-hi patterns" framing is gone —
        # it was the thing causing the LLM to cluster output around
        # the lower-bound regardless of corpus richness.
        assert "Default target" not in prompt

    def test_insights_prompt_carries_caps_no_max_3_or_max_2_literals(self):
        from engine.insights import _build_prompt as build_insights_prompt
        from engine.patterns import Pattern

        patterns_by_topic = {
            "work": [
                Pattern(name=f"P{i}", description="d", domain="work",
                        count=2, source_facts=[(0, 1.0), (1, 1.0)])
                for i in range(3)
            ],
            "health": [
                Pattern(name="Q1", description="d", domain="health",
                        count=2, source_facts=[(0, 1.0), (1, 1.0)])
            ],
        }
        prompt, _ = build_insights_prompt(
            patterns_by_topic, cross_cap=4, critical_cap=3,
        )
        assert "hard cap: 4" in prompt
        assert "hard cap: 3" in prompt
        # The old literal "max 3" / "max 2" framing must be gone — it
        # was forcing every corpus to ≤5 insights regardless of size.
        assert "max 3" not in prompt
        assert "max 2" not in prompt

    def test_actions_prompt_carries_cap_no_fixed_5(self):
        from engine.actions import _build_prompt as build_actions_prompt
        from engine.insights import InsightOutput, Insight

        insight_out = InsightOutput(
            cross_domain=[
                Insight(
                    name="ix", description="d", mechanism="m",
                    implication="i", domains=["work", "health"],
                    kind="defensive-loop",
                    proposed_actions=["a"],
                    source_patterns=[("work", 0, 1.0), ("health", 0, 1.0)],
                ),
            ],
            critical=[],
        )
        # Cap=7 (large-corpus equivalent). No `today` arg post-PR —
        # the actions prompt is now date-free; review_date is computed
        # runner-side from horizon + today AFTER the LLM call.
        prompt, _ = build_actions_prompt(insight_out, 7)
        assert "{max_actions}" not in prompt  # placeholder substituted
        assert "hard cap is 7" in prompt
