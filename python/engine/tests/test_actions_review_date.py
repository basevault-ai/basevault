"""
Horizon → review_date computation, runner-side.

Locks the post-LLM date derivation contract for the actions stage:
the LLM emits only a `horizon` enum (short / medium / long); the
runner computes `review_date = today + _HORIZON_TO_DAYS[horizon]`
deterministically. This was previously done by the LLM emitting a
date that we then clamped via `_clamp_review_date` (now removed),
which forced the `today` literal into the actions prompt and broke
the cache key across calendar days.

These tests cover the helper directly and the parser path that
materializes Actions, so any future drift (renaming the table,
collapsing horizons, dropping the today thread) fails CI.
"""
from __future__ import annotations

from datetime import date

import pytest

from engine.actions import (
    _HORIZON_TO_DAYS,
    _parse_action,
    _review_date_for_horizon,
)


_TODAY = date(2026, 1, 1)


# ── Helper directly ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("horizon,expected_offset_days", [
    ("short", 14),
    ("medium", 90),
    ("long", 90),
])
def test_review_date_for_horizon_pins_to_table(horizon, expected_offset_days):
    """`today + N days` from the table — deterministic for known horizons."""
    out = _review_date_for_horizon(horizon, _TODAY)
    expected = (_TODAY.replace().toordinal() + expected_offset_days)
    from datetime import date as _date
    assert out == _date.fromordinal(expected).isoformat()


def test_review_date_unknown_horizon_falls_back_to_medium():
    """Unknown horizon string MUST behave like medium (matches the
    parser's `if horizon not in (...): horizon = 'medium'` normalize
    step). Without this guarantee, a typo or future-renamed horizon
    would silently emit today.isoformat() (0-day offset)."""
    fallback = _review_date_for_horizon("unknown", _TODAY)
    medium = _review_date_for_horizon("medium", _TODAY)
    assert fallback == medium


def test_review_date_table_covers_every_horizon_the_parser_accepts():
    """Defensive coverage: if someone adds a horizon to the parser's
    accepted set (`short`, `medium`, `long`), the table must grow too.
    Catches the half-update bug where horizons drift apart."""
    accepted_by_parser = {"short", "medium", "long"}
    table_keys = set(_HORIZON_TO_DAYS.keys())
    assert accepted_by_parser <= table_keys, (
        f"parser accepts {accepted_by_parser - table_keys} but the "
        f"table doesn't list it — those horizons would silently fall "
        f"back to medium"
    )


# ── End-to-end via the parser ────────────────────────────────────────────────

def _mock_source_index_map() -> list[tuple[str, int]]:
    """A 2-entry source map so a parsed action with `sources: [{id: 1}]`
    resolves cleanly. The parser drops actions with no resolved
    sources, so we need at least one valid index."""
    return [("cross_domain", 0), ("critical", 0)]


def _mock_entry(horizon: str, **overrides) -> dict:
    """Minimal valid action JSON the parser will accept. Override
    `horizon` to test each value; override anything else for edge-case
    coverage."""
    base = {
        "recommendation": "do the thing",
        "objective": "thing is done",
        "immediate_action": "start now",
        "horizon": horizon,
        "sources": [{"id": 1, "confidence": 1.0}],
    }
    base.update(overrides)
    return base


def test_parse_action_computes_review_date_from_horizon_and_today():
    """End-to-end through the parser: feed a horizon, get back an
    Action whose review_date is the table's offset from `today`. This
    is the contract the runner relies on — if it breaks, every action
    in the run gets the wrong review date."""
    action = _parse_action(
        _mock_entry("short"), _mock_source_index_map(), _TODAY,
    )
    assert action is not None
    assert action.horizon == "short"
    assert action.review_date == "2026-01-15"  # 2026-01-01 + 14 days

    action_med = _parse_action(
        _mock_entry("medium"), _mock_source_index_map(), _TODAY,
    )
    assert action_med.review_date == "2026-04-01"  # +90 days

    action_long = _parse_action(
        _mock_entry("long"), _mock_source_index_map(), _TODAY,
    )
    assert action_long.review_date == "2026-04-01"  # +90 days (same as medium per table)


def test_parse_action_ignores_any_review_date_in_llm_output():
    """The LLM is no longer asked for a review_date, but a stale prompt
    or a misbehaving model might still emit one. The parser must IGNORE
    it and recompute from horizon+today instead — otherwise the cache
    invariant (date-free prompt → stable hash) leaks back through the
    parsed output. Cache-hit replay would otherwise carry a stale
    LLM-emitted date forward indefinitely."""
    entry = _mock_entry("short", review_date="2099-12-31")
    action = _parse_action(entry, _mock_source_index_map(), _TODAY)
    assert action is not None
    # The LLM-supplied 2099 date is dropped; runner-side computation wins.
    assert action.review_date == "2026-01-15"


def test_parse_action_unknown_horizon_normalizes_to_medium_review_date():
    entry = _mock_entry("yearly")  # not in {short, medium, long}
    action = _parse_action(entry, _mock_source_index_map(), _TODAY)
    assert action is not None
    assert action.horizon == "medium"
    assert action.review_date == "2026-04-01"
