"""Tests for the sentiment_bias plumbing.

Verifies that:
  - all five sentiment values produce distinct system-prompt blocks for
    insights AND actions
  - patterns.py has NO sentiment plumbing (descriptive layer; off-spec
    for sentiment to touch it)
  - unknown sentiment values fall back to neutral instead of erroring
  - the chosen sentiment's clause lands in the rendered system prompt

We don't test the actual generation behavior — that's a runtime/eval
concern. Here we just verify the prompt-shape plumbing.

The "lands in the system prompt" checks drive the LIVE kernel phases
(`phases/insights.py` + `phases/actions.py`), which own the
sentiment → `_system_content` wiring after the cutover — the legacy
`detect_insights()`/`generate_actions()` + `complete()` harness is gone.

Run with:
    cd engine && pytest tests/test_sentiment_plumbing.py -v
"""
from __future__ import annotations



from engine.insights import _SENTIMENT_BIAS_CLAUSES as INS_CLAUSES
from engine.actions import _SENTIMENT_BIAS_CLAUSES as ACT_CLAUSES
from engine.insights import InsightOutput, Insight  # noqa: F401  (Insight re-exported for callers)
from engine.patterns import Pattern


_ALL_SENTIMENTS = (
    "brutally-honest", "critical", "neutral", "uplifting", "bubbly",
)


# ── The clauses are real, distinct, and cover all five values ─────────────


class TestSentimentClauses:
    def test_insights_has_all_five_sentiments(self):
        for s in _ALL_SENTIMENTS:
            assert s in INS_CLAUSES, s
            assert INS_CLAUSES[s].strip(), f"{s} clause is empty"

    def test_actions_has_all_five_sentiments(self):
        for s in _ALL_SENTIMENTS:
            assert s in ACT_CLAUSES, s
            assert ACT_CLAUSES[s].strip(), f"{s} clause is empty"

    def test_insights_clauses_are_distinct(self):
        # Five sentiments → five distinct clause bodies. If two are
        # identical the knob is decorative.
        bodies = {INS_CLAUSES[s] for s in _ALL_SENTIMENTS}
        assert len(bodies) == 5, "two or more insight clauses are identical"

    def test_actions_clauses_are_distinct(self):
        bodies = {ACT_CLAUSES[s] for s in _ALL_SENTIMENTS}
        assert len(bodies) == 5, "two or more action clauses are identical"

    def test_uplifting_and_bubbly_explicitly_forbid_fabrication(self):
        # The user's brief specifically calls out that uplifting must
        # not fabricate positives and bubbly must not erase real
        # defensive loops. Lock that in.
        for s in ("uplifting", "bubbly"):
            for clauses in (INS_CLAUSES, ACT_CLAUSES):
                body = clauses[s].lower()
                assert "fabricate" in body or "invent" in body or "manufacture" in body, (
                    f"{s} clause must explicitly forbid fabrication"
                )

    def test_brutally_honest_is_not_a_verb_ban(self):
        # The brief says verb bans ("don't say X") are the wrong
        # design. The brutally-honest clause uses framing language
        # ("drop hedging qualifiers"), not lexical bans.
        body = INS_CLAUSES["brutally-honest"].lower()
        # Sanity: doesn't contain "do not say" / "never say" / "verbs to avoid"
        assert "do not say" not in body
        assert "never say" not in body
        assert "verbs to avoid" not in body


# ── Plumbing into prompts (live kernel phases own _system_content) ────────


class TestSentimentInPrompts:
    @staticmethod
    def _insights_system_content(**overrides) -> str:
        from kernel.abstractions import PhaseResult
        from engine.phases.insights import InsightsJob, InsightsPhase
        from engine.llm import Mode

        initial = {
            "patterns_by_topic": {
                "work": [Pattern(name="P1", description="d", domain="work",
                                 count=2, source_facts=[(0, 1.0), (1, 1.0)])],
                "health": [Pattern(name="Q1", description="d", domain="health",
                                   count=2, source_facts=[(0, 1.0), (1, 1.0)])],
            },
            "mode": Mode.TEE, "subject": "Alice", "total_facts": 1000,
        }
        initial.update(overrides)
        phase = InsightsPhase(InsightsJob(initial))
        phase.init_from_memory(PhaseResult(initial))
        return phase._system_content

    @staticmethod
    def _actions_system_content(**overrides) -> str:
        from datetime import date
        from kernel.abstractions import PhaseResult
        from engine.phases.actions import ActionsJob, ActionsPhase
        from engine.llm import Mode

        initial = {
            "mode": Mode.TEE,
            "insight_output": InsightOutput(cross_domain=[], critical=[]),
            "today": date(2026, 1, 1), "subject": "Alice", "total_facts": 1000,
        }
        initial.update(overrides)
        phase = ActionsPhase(ActionsJob(initial))
        phase.init_from_memory(PhaseResult(initial))
        return phase._system_content

    def test_each_sentiment_lands_in_the_insights_system_prompt(self):
        for s in _ALL_SENTIMENTS:
            sc = self._insights_system_content(sentiment=s)
            assert INS_CLAUSES[s] in sc, (
                f"sentiment={s!r}: clause not present in insights system prompt"
            )

    def test_each_sentiment_lands_in_the_actions_system_prompt(self):
        for s in _ALL_SENTIMENTS:
            sc = self._actions_system_content(sentiment=s)
            assert ACT_CLAUSES[s] in sc, (
                f"sentiment={s!r}: clause not present in actions system prompt"
            )

    def test_unknown_sentiment_falls_back_to_neutral(self):
        # Unknown values fall back to the neutral clause, not "" / an error.
        sc = self._insights_system_content(sentiment="ultra-passive-aggressive")
        assert INS_CLAUSES["neutral"] in sc

    def test_default_sentiment_is_neutral(self):
        # No sentiment key → default is "neutral".
        sc = self._insights_system_content()
        assert INS_CLAUSES["neutral"] in sc


# ── Patterns is NOT sentiment-tunable ────────────────────────────────────


class TestPatternsIsToneNeutral:
    def test_patterns_module_has_no_sentiment_clauses(self):
        # The brief is explicit: patterns is the descriptive layer and
        # the sentiment knob does not touch it. Catch any future
        # accidental wiring by asserting the module exposes no
        # sentiment-clause map.
        from engine import patterns as patterns_mod
        assert not hasattr(patterns_mod, "_SENTIMENT_BIAS_CLAUSES"), (
            "patterns.py should not have _SENTIMENT_BIAS_CLAUSES — "
            "patterns is the descriptive layer and stays tone-neutral"
        )
