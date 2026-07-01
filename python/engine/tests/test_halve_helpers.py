"""
Unit tests for the `_halve_doc_content` byte-split helper used by the
extraction phase's sizing cascade.

On a sizing failure (timeout-with-TTFT, cap-hit, parse_error) the phase
splits the current doc with `_halve_doc_content` and re-dispatches each
half as a fresh child. The helper is a pure input transform — no LLM, no
scheduler dependency. This file pins its split math.

No live LLM calls. Run with:
    cd engine && pytest tests/test_halve_helpers.py -v
"""
from __future__ import annotations


from engine.content_extractor import _halve_doc_content


# ── _halve_doc_content ──────────────────────────────────────────────────────


class TestHalveDocContent:
    def test_short_returns_none(self):
        assert _halve_doc_content("short") is None

    def test_round_trip(self):
        content = "Alpha sentence. " * 30 + "\n\n" + "Beta sentence. " * 30
        result = _halve_doc_content(content)
        assert result is not None
        first, second, off = result
        assert first + second == content
        assert 0 < off < len(content)

    def test_prefers_boundary_not_mid_token(self):
        a = "Alpha sentence. " * 30
        b = "Beta sentence. " * 30
        content = a + "\n\n" + b
        first, second, off = _halve_doc_content(content)
        assert content[off - 1] in {"\n", " ", "."}
