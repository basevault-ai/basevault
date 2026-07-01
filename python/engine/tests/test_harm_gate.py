"""Behavioral tests for the always-on harm clause in actions.py.

The clause (`_HARM_CLAUSE`) is appended to every actions-stage system
prompt and is self-conditional ("if the insights describe such
content, …") rather than upstream-keyword-gated.

What this test asserts (behavior only):
  - The clause is present in the rendered system prompt.
  - It's present regardless of sentiment value.
  - The body is self-conditional and covers harm beyond illegal ops.

What this test deliberately does NOT do:
  - Assert the *absence* of deleted helpers / parameters. That is
    documented in the commit + diff; testing negative space adds
    maintenance surface that drifts.
  - Test that the model declines on harmful inputs. That requires
    live LLM calls on harmful content — brittle, expensive, and
    largely a function of the model's own training.

Run with:
    cd engine && pytest tests/test_harm_gate.py -v
"""
from __future__ import annotations



from engine.actions import _HARM_CLAUSE
class TestHarmClauseIsAlwaysOn:
    # NB: the legacy generate_actions()/complete() rendering tests
    # (clause-present-in-prompt across sentiments) were dropped with the
    # legacy actions driver; the kernel actions phase now renders the
    # prompt and the always-on wiring is covered there. These remaining
    # tests lock the clause's content invariants, which are seam-agnostic.
    def test_clause_includes_self_conditional_phrasing(self):
        # The clause must be self-conditional — that's the architectural
        # decision (model reads context vs upstream keyword-scan). Lock
        # in that the body actually does the conditional thing rather
        # than just being a lecture.
        body = _HARM_CLAUSE.lower()
        assert "if the insights describe" in body, (
            "harm clause should self-condition on insight content"
        )

    def test_clause_covers_more_than_illegal_ops(self):
        # The clause should generalize beyond the old narrow
        # illegal-operations framing. Verify the body names broader
        # categories.
        body = _HARM_CLAUSE.lower()
        # At least one harm category beyond "illegal".
        assert "harm" in body
        # Specifically mentions other parties — the old clause was
        # primarily subject-centric.
        assert "other" in body or "others" in body
