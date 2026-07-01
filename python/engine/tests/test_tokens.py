"""Unit tests for tokens.count_tokens.

The estimator is intentionally `len(text) // CHARS_PER_TOKEN` with one
global ratio — see tokens.py module docstring for the why.
"""

from engine.tokens import count_tokens, CHARS_PER_TOKEN


def test_constant_is_three():
    """If we ever loosen this, splitter sizing widens and the
    pre-call input_overflow check starts firing false positives.
    Bumping past 3 should be a deliberate decision, not a drift."""
    assert CHARS_PER_TOKEN == 3


def test_empty_returns_zero():
    assert count_tokens("") == 0


def test_short_text_floors_at_one():
    """A non-empty string shorter than CHARS_PER_TOKEN still counts
    as 1 token — never 0. Floor matters for 'is this prompt non-empty?'
    sentinel checks."""
    assert count_tokens("x") == 1
    assert count_tokens("ab") == 1


def test_long_text_scales_linearly():
    assert count_tokens("x" * 300) == 100
    assert count_tokens("x" * 3000) == 1000


def test_matches_chars_div_three():
    """Spot-checks the documented formula for unsurprising inputs."""
    for n in [3, 30, 300, 3000, 30_000, 300_000]:
        assert count_tokens("x" * n) == n // CHARS_PER_TOKEN
