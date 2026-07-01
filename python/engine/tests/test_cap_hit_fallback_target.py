"""
Parser-surface contracts around malformed-JSON failures. Pins:

1. The parser symptom — structurally-malformed JSON (OOV/multilingual
   tokens spliced mid-value) is rejected at the entities_dedupe parser
   surface with `_ParseError`. Locks the contract so a future
   "tolerant parser" attempt can't silently accept this garbage.

2. Kimi self-correction — when a model emits multiple fenced blocks
   (a retry-in-one-response), `strip_fences` picks the last block so
   the corrected output wins over the abandoned first attempt.

No live LLM calls. Run with:
    cd engine && pytest tests/test_cap_hit_fallback_target.py -v
"""
from __future__ import annotations

import json

import pytest


from engine.llm import strip_fences


# ── Parser symptom: malformed garbage-token JSON is rejected ────────────────

# Real fixture extracted from `~/.basevault/logs/2026-05-21T06-09-14Z-mhed/`
# call 0021 (entities_dedupe, finish_reason=stop,
# completion_tokens=1211). Structural JSON skeleton is well-formed; OOV
# tokens are spliced mid-value at numeric and string positions. Sources
# of garbage: English fragments (`iat`, `OD`), Chinese (`略`), Greek
# (`ετυμολογία`). All four 0021–0024 retries on the same prompt at
# temp=0.0 produced different garbage — provider-side non-determinism.
_CALL_0021_GARBAGE_EXCERPT = """{
  "merges": [
    {
      "a_id": "author",
      "b_id": "the-author",
      "confidence": iat0.99,
      "synthesized_description": "The author, who experienced nervous breakdowns impression, received advice about happiness, and plans to marry on September 15, is the central figure of the diary."
    },
    {
      "a_id": "doctor",
      "b_id": "the-doctor",
      "confidence": 0.略95,
      "synthesized_description": "The Doctor told the author that enjoying twelve months of happiness is worthwhile."
    }
  ]
}"""


class TestMalformedJSONIsRejected:
    """Pins the parser contract: garbage-token bleed inside leaf values
    is a parse error, not silently salvaged. If this test ever passes
    with json.loads accepting the input, the parser has grown
    tolerance it shouldn't have (the bug is upstream; salvage masks
    real failure mode and risks fusing distinct entities)."""

    def test_call_0021_raw_payload_is_unparseable(self):
        with pytest.raises(json.JSONDecodeError):
            json.loads(strip_fences(_CALL_0021_GARBAGE_EXCERPT))

    @pytest.mark.parametrize("synthetic", [
        # Each variant pins one observed corruption shape.
        '{"merges": [{"confidence": iat0.99}]}',              # English frag prefix on numeric
        '{"merges": [{"confidence": 0.略 95}]}',           # CJK mid-decimal
        '{"merges": [{"confidence": 去打 0.95}]}',     # CJK numeric prefix
        '{"merges": [{"confidence": 0. score9}]}',            # English frag mid-decimal
        '{"merges": [{"confidence":  zon0.95}]}',             # whitespace + frag prefix
    ])
    def test_synthetic_garbage_shapes_unparseable(self, synthetic):
        with pytest.raises(json.JSONDecodeError):
            json.loads(strip_fences(synthetic))


# ── Parser fix: kimi multi-block self-correction picks the last block ─────

class TestKimiSelfCorrectionPicksLastBlock:
    """Kimi-k2-6 reasoning-off (a degrading-stage primary post-#666)
    sometimes emits a first draft fenced block, then second-guesses
    itself with
    interstitial prose ("Wait — I need to redo this properly"), then
    emits a corrected block. Observed in run mexn call 0020. The
    previous all-strip-and-join implementation concatenated both
    blocks and json.loads raised "Extra data". `strip_fences` now
    picks the LAST fenced block — the model's final answer."""

    KIMI_SELF_CORRECTION = """ ```json
{
  "merges": [
    {"a_id": "first_attempt", "b_id": "x", "confidence": 0.5}
  ]
}
```

Wait - I need to remove the non-merges and fix this. Let me redo properly with only actual merges.

```json
{
  "merges": [
    {"a_id": "corrected", "b_id": "y", "confidence": 0.95}
  ]
}
```"""

    def test_picks_corrected_block(self):
        out = strip_fences(self.KIMI_SELF_CORRECTION)
        data = json.loads(out)
        assert data["merges"] == [
            {"a_id": "corrected", "b_id": "y", "confidence": 0.95}
        ], "Last block (corrected) must win; first block is the discarded draft."

    def test_single_block_unchanged(self):
        single = "```json\n{\"merges\": [{\"a\": 1}]}\n```"
        assert json.loads(strip_fences(single)) == {"merges": [{"a": 1}]}

    def test_no_fence_passthrough(self):
        bare = '{"merges": [{"a": 1}]}'
        assert json.loads(strip_fences(bare)) == {"merges": [{"a": 1}]}

    def test_leading_whitespace_before_fence(self):
        # Real kimi output: leading space before the opening fence.
        leading = " ```json\n{\"ok\": true}\n```"
        assert json.loads(strip_fences(leading)) == {"ok": True}

    def test_truncated_opening_fence_no_close(self):
        # Cap-hit cuts the stream mid-output: opening fence, no close.
        # Falls back to stray-marker stripping so json.loads still
        # fails downstream and parse_error / cap-hit classification
        # fires as expected.
        truncated = "```json\n{\"merges\": [{\"a\": "
        out = strip_fences(truncated)
        with pytest.raises(json.JSONDecodeError):
            json.loads(out)

    def test_empty_input(self):
        assert strip_fences("") == ""
        assert strip_fences(None) == ""
