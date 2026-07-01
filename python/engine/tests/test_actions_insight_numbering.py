"""Actions prompt: insight numbering is monotonic across scopes.

Guards the invariant that `_build_prompt` numbers insights continuously
— cross-domain 1..M, then critical M+1..M+K — rather than restarting the
bracket index per scope. The actions LLM writes `Insight [N]` against
this enumeration, and both the parser (`source_index_map`) and the
embedding-time ref resolver (`rag_enricher._resolve_insight_refs`) map
`[N]` back through it. A per-scope reset would make a given `[N]`
ambiguous between the two scopes and mislink the resolved insight.
"""
from __future__ import annotations

from engine.actions import _build_prompt
from engine.insights import Insight, InsightOutput


def _ins(name: str) -> Insight:
    return Insight(
        name=name, description="d", mechanism="m",
        implication="i", domains=["health"], kind="defensive-loop",
    )


def test_prompt_ids_are_monotonic_across_scopes():
    io = InsightOutput(
        cross_domain=[_ins("c0"), _ins("c1"), _ins("c2")],
        critical=[_ins("k0"), _ins("k1")],
    )
    prompt, source_index_map = _build_prompt(io, max_actions=5)

    # source_index_map index i ↔ prompt id (i+1): cross-domain first, then
    # critical, with NO reset of the running id at the scope boundary.
    assert source_index_map == [
        ("cross_domain", 0), ("cross_domain", 1), ("cross_domain", 2),
        ("critical", 0), ("critical", 1),
    ]

    # The rendered prompt enumerates [1]..[5] once, monotonically.
    for n in range(1, 6):
        assert f"[{n}]" in prompt

    # Cross-domain occupies [1]..[3]; critical continues at [4]..[5] and
    # does NOT restart at [1].
    assert "[1] (cross-domain" in prompt
    assert "[4] (critical" in prompt
    assert "[5] (critical" in prompt
    assert "[1] (critical" not in prompt


def test_critical_only_still_starts_at_one():
    """With no cross-domain insights the critical scope simply numbers
    from [1] — the monotonic rule has no boundary to cross."""
    io = InsightOutput(cross_domain=[], critical=[_ins("k0"), _ins("k1")])
    prompt, source_index_map = _build_prompt(io, max_actions=5)
    assert source_index_map == [("critical", 0), ("critical", 1)]
    assert "[1] (critical" in prompt
    assert "[2] (critical" in prompt
