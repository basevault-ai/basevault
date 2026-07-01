"""Unit tests for patterns.py overflow + sampling.

Covers:
  - _evenly_spaced_indices: stride math, most-recent inclusion
  - _sample_to_fit: returns full set under cap, samples down when over
  - detect_patterns: dense-topic flow records the
    `patterns_facts_sampled` warning and translates sample IDs back to
    original fact indices

No real LLM calls — `complete` is monkey-patched.

Run with:
    cd engine && pytest tests/test_patterns.py -v
"""



from engine.content_extractor import Entity, EntityRef, EvidenceSpan, ExtractedItem
from engine.patterns import _build_messages, _evenly_spaced_indices, _sample_to_fit


def _mk(summary: str, occurred: str | None = None) -> ExtractedItem:
    return ExtractedItem(
        item_type="fact",
        summary=summary,
        evidence=[EvidenceSpan(text=summary, source_ref="test")],
        entities=[EntityRef(entity=Entity(name="Alice", entity_type="person"),
                            role="subject")],
        topics=["work"],
        tags=[],
        confidence=1.0,
        occurred_at=occurred,
    )


# ── _evenly_spaced_indices ───────────────────────────────────────────────────


class TestEvenlySpacedIndices:
    def test_returns_full_when_k_ge_n(self):
        assert _evenly_spaced_indices(5, 5) == [0, 1, 2, 3, 4]
        assert _evenly_spaced_indices(5, 10) == [0, 1, 2, 3, 4]

    def test_empty_when_k_zero(self):
        assert _evenly_spaced_indices(5, 0) == []

    def test_k1_picks_most_recent(self):
        """k=1 prefers the latest fact (input is chronologically ordered)."""
        assert _evenly_spaced_indices(10, 1) == [9]

    def test_strides_evenly_and_pins_last(self):
        idx = _evenly_spaced_indices(100, 10)
        assert len(idx) == 10
        assert idx == sorted(idx)
        assert idx[-1] == 99  # most-recent always included
        # Stride roughly len/k=10; first index near 5.
        assert 0 <= idx[0] <= 10

    def test_k_equal_n_minus_one(self):
        idx = _evenly_spaced_indices(5, 4)
        assert len(idx) <= 4
        assert idx[-1] == 4


# ── _sample_to_fit ───────────────────────────────────────────────────────────


class TestSampleToFit:
    def test_full_set_when_under_cap(self):
        deduped = [_mk(f"f{i}") for i in range(5)]
        idx = _sample_to_fit(
            deduped, "work", hard_cap=8, subject="Alice",
            entities_context=None, cap=10_000)
        assert idx == [0, 1, 2, 3, 4]

    def test_samples_down_under_tight_cap(self):
        # 200 long facts and a deliberately tight cap to force sampling.
        deduped = [_mk(f"long fact {i} " + "x" * 200) for i in range(200)]
        # Tight cap → sampling. Choose a cap roughly 1/4 the full size.
        full_msgs = _build_messages(
            deduped, "work", 16, "Alice", entities_context=None)
        from engine.tokens import estimate_prompt_tokens
        full_size = estimate_prompt_tokens(full_msgs)
        tight_cap = full_size // 4
        idx = _sample_to_fit(
            deduped, "work", hard_cap=16, subject="Alice",
            entities_context=None, cap=tight_cap)
        assert 2 <= len(idx) < 200
        # Sampled indices include the most-recent fact.
        assert idx[-1] == 199
        # Resulting prompt fits.
        sample = [deduped[i] for i in idx]
        msgs = _build_messages(
            sample, "work", 16, "Alice", entities_context=None)
        assert estimate_prompt_tokens(msgs) <= tight_cap


# ── detect_patterns: end-to-end with sampling ────────────────────────────────







# ── detect_patterns_all: per-topic on_topic_done callback ────────────────────





