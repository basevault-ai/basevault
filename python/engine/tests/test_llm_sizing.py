"""
Unit tests for the sizing math in llm.py:
  - compute_budget (the unified entry point; .max_output per stage)
  - chunk_cap_for_stage (thin wrapper for max_input)
  - dynamic_max_tokens (thin wrapper for per-call max_output)
  - _TYPICAL_RATIO_BY_STAGE (calibrated per-stage output:input ratios)
  - tokens_from_chars / estimate_prompt_tokens (re-exported from tokens.py)

No LLM calls. Run with:
    cd engine && pytest tests/test_llm_sizing.py -v
"""


from engine.llm import (
    Mode,
    StageBudget,
    compute_budget,
    chunk_cap_for_stage,
    dynamic_max_tokens,
    max_workers,
    _TYPICAL_RATIO_BY_STAGE,
    _HEADROOM_TOKENS,
    _MIN_CONTEXT_TOKENS,
    _SCAFFOLDING_TOKENS_BY_STAGE,
    _scaffolding_tokens_for_stage,
)
from engine.tokens import (
    CHARS_PER_TOKEN as _CHARS_PER_TOKEN,
    tokens_from_chars,
    estimate_prompt_tokens,
)


# ── tokens_from_chars ──────────────────────────────────────────────────────────

class TestTokensFromChars:
    def test_exact_multiple(self):
        # CPT is 3.0, so 3000 chars / 3 = 1000 tokens
        assert tokens_from_chars(3000) == 1000

    def test_truncates_to_int(self):
        # 10 / 3 = 3.33, int() truncates to 3
        assert tokens_from_chars(10) == 3

    def test_zero(self):
        assert tokens_from_chars(0) == 0

    def test_uses_cpt_constant(self):
        # If the constant changes, this test's expected value changes too
        # — it's documenting that tokens_from_chars honors _CHARS_PER_TOKEN
        assert tokens_from_chars(12000) == int(12000 / _CHARS_PER_TOKEN)


# ── estimate_prompt_tokens ─────────────────────────────────────────────────────

class TestEstimatePromptTokens:
    def test_sums_all_message_contents(self):
        msgs = [
            {"role": "system", "content": "a" * 300},
            {"role": "user", "content": "b" * 3000},
        ]
        # Total chars 3300 / 3 = 1100
        assert estimate_prompt_tokens(msgs) == 1100

    def test_empty_messages(self):
        assert estimate_prompt_tokens([]) == 0

    def test_handles_non_string_content_gracefully(self):
        # Message content that isn't a string falls through to str() — "None"
        # has length 4, / 3 = 1 token estimate. Test just covers the path
        # that a non-string doesn't crash.
        msgs = [{"role": "system", "content": None}]
        assert estimate_prompt_tokens(msgs) >= 0


# ── chunk_cap_for_stage ────────────────────────────────────────────────────────

class TestChunkCapForStage:
    """Formula: (ctx - scaffolding) / (1 + ratio), clamped by quality_cap."""

    def test_extract_tee(self):
        # extract ratio = 4. Cap is min of two bounds: ctx-based
        # (input + ratio*input fits the context window) and
        # output-based (ratio*input ≤ provider's max_output ceiling).
        # Tinfoil specs default max_output to context_window (wide-cap)
        # so ctx_cap typically dominates; the explicit min() below
        # stays correct if a future spec pins a tighter max_output.
        from engine.llm import _resolve_stage_override, _scaffolding_tokens_for_stage
        spec, _ = _resolve_stage_override(Mode.TEE, "extract")
        scaffolding = _scaffolding_tokens_for_stage("extract")
        ctx_cap = int((spec.context_window - scaffolding) / (1 + 4))
        output_cap = int(spec.max_output / 4)
        expected = min(ctx_cap, output_cap)
        assert chunk_cap_for_stage(Mode.TEE, "extract") == expected

    def test_entities_tee(self):
        # entities ratio = 1. See test_extract_tee for the per-stage
        # resolution rationale.
        from engine.llm import _resolve_stage_override, _scaffolding_tokens_for_stage
        spec, _ = _resolve_stage_override(Mode.TEE, "entities")
        scaffolding = _scaffolding_tokens_for_stage("entities")
        expected = int((spec.context_window - scaffolding) / 2)
        assert chunk_cap_for_stage(Mode.TEE, "entities") == expected

    def test_unknown_stage_raises(self):
        # Every pipeline stage must register a calibrated ratio entry.
        # An unknown stage signals a routing bug or a missing
        # registration — _ratio_for_stage raises rather than silently
        # falling back to a default.
        import pytest
        with pytest.raises(KeyError):
            chunk_cap_for_stage(Mode.TEE, "nonexistent")

    def test_each_stage_non_zero(self):
        for stage in ("extract", "entities", "patterns", "insights", "actions"):
            assert chunk_cap_for_stage(Mode.TEE, stage) > 0


class TestOutputCapClamp:
    """Regression guard: chunk_cap_for_stage clamps at
    `max_output / ratio` in addition to the ctx bound, sizing input DOWN
    on tight-output models so reservations never exceed what the model
    can emit. Synthetic narrow-output spec covers the previously
    cap-hit-prone case without needing a non-Tinfoil provider in the
    registry."""

    def test_narrow_output_clamps_chunk_cap(self, monkeypatch):
        # Synthetic spec: 256k ctx, narrow 65,535 max_output, extract
        # ratio=4 → output_cap = 65535 / 4 = 16,383. ctx_cap = (256k -
        # scaffolding)/5 ≈ 51k. min binds at output_cap.
        from engine import llm
        from engine.llm import ModelSpec, Provider, _resolve_stage_override
        narrow = ModelSpec(
            provider=Provider.TINFOIL,
            model_id="synthetic-narrow-out",
            context_window=256_000,
            max_output=65_535,
        )
        monkeypatch.setitem(llm.MODE_SPEC, Mode.TEE, narrow)
        monkeypatch.setattr(llm, "_STAGE_MODEL_MAP", {})
        spec, _ = _resolve_stage_override(Mode.TEE, "extract")
        ratio = 4
        output_cap = int(spec.max_output / ratio)
        cap = chunk_cap_for_stage(Mode.TEE, "extract")
        assert cap == output_cap, (
            f"expected output_cap={output_cap} to bind on a narrow "
            f"max_output spec, got chunk_cap={cap}"
        )

    def test_dynamic_max_tokens_bounded_by_provider_cap(self, monkeypatch):
        # Per-call max_tokens never exceeds spec.max_output. Synthetic
        # narrow-output spec — payload picked so 3×payload + headroom
        # exceeds max_output, exercising the provider-cap clamp.
        from engine import llm
        from engine.llm import ModelSpec, Provider
        narrow = ModelSpec(
            provider=Provider.TINFOIL,
            model_id="synthetic-narrow-out",
            context_window=256_000,
            max_output=65_535,
        )
        monkeypatch.setitem(llm.MODE_SPEC, Mode.TEE, narrow)
        monkeypatch.setattr(llm, "_STAGE_MODEL_MAP", {})
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage",
                            lambda stage, mode: False)
        # 4 × 25k + 2048 = 102,048 > 65,535. ctx headroom (256k-25k-s)
        # is ~230k so real_cap doesn't bind; provider_cap binds.
        payload = 25_000
        mt = dynamic_max_tokens(payload, Mode.TEE, "extract")
        assert mt == narrow.max_output


# ── compute_budget.max_output per stage ─────────────────────────────────────────

class TestMaxOutputForStage:
    def test_extract_tee(self):
        # max_output = ratio × chunk_cap
        cap = chunk_cap_for_stage(Mode.TEE, "extract")
        assert compute_budget(Mode.TEE, "extract").max_output == int(4 * cap)

    def test_entities_tee(self):
        # ratio=1 → max_output = chunk_cap
        cap = chunk_cap_for_stage(Mode.TEE, "entities")
        assert compute_budget(Mode.TEE, "entities").max_output == int(1 * cap)

    def test_patterns_tee_ratio_none(self):
        # patterns / insights / actions have ratio=None — single-call
        # SPOF stages where the linear formula doesn't apply. Static
        # max_output is the provider cap (full window), independent
        # of max_input.
        from engine.llm import _resolve_stage_override
        spec, _ = _resolve_stage_override(Mode.TEE, "patterns")
        assert compute_budget(Mode.TEE, "patterns").max_output == spec.max_output


# ── dynamic_max_tokens ─────────────────────────────────────────────────────────

class TestDynamicMaxTokens:
    """Formula: min(ratio × payload_tokens + headroom, max_output_for_stage).
    Ratio applies to PAYLOAD only — scaffolding is fixed framing."""

    def test_extract_big_chunk(self, monkeypatch):
        # Payload 14,119 tokens (42357-char chunk), extract ratio=4, headroom=2048
        # → 4 × 14119 + 2048 = 58,524.
        # Pin reasoning OFF: these assert the base ratio. The shipped default
        # enables reasoning for extract, which adds the +1 ratio bump — that
        # bump is covered separately by TestReasoningBump.
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage", lambda stage, mode: False)
        mt = dynamic_max_tokens(14_119, Mode.TEE, "extract")
        assert mt == 4 * 14_119 + _HEADROOM_TOKENS == 58_524

    def test_extract_medium_chunk(self, monkeypatch):
        # Payload 12,500 (37500-char chunk) → 4 × 12500 + 2048 = 52,048
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage", lambda stage, mode: False)
        mt = dynamic_max_tokens(12_500, Mode.TEE, "extract")
        assert mt == 52_048

    def test_extract_small_chunk_floored_by_min_context(self):
        # Payload 1,798 → formula gives 5,644 but total is only
        # 1,798 + 5,644 = 7,442 < _MIN_CONTEXT_TOKENS (32,000).
        # Floor kicks in: max_tokens = 32,000 - 1,798 = 30,202
        # Total context = 32,000 exactly.
        mt = dynamic_max_tokens(1_798, Mode.TEE, "extract")
        assert mt == _MIN_CONTEXT_TOKENS - 1_798 == 30_202
        assert 1_798 + mt == _MIN_CONTEXT_TOKENS

    def test_entities_ratio_1(self, monkeypatch):
        # Payload 10,000, entities ratio=1 → 1 × 10000 + 2048 = 12,048
        # Total = 10,000 + 12,048 = 22,048 < 32,000 → floor kicks in
        # Floor: 32,000 - 10,000 = 22,000 → max_tokens = 22,000
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage", lambda stage, mode: False)
        mt = dynamic_max_tokens(10_000, Mode.TEE, "entities")
        assert mt == 22_000
        assert 10_000 + mt == _MIN_CONTEXT_TOKENS

    def test_patterns_ratio_none_full_window(self):
        # patterns has ratio=None — per-call mtr is the full remaining
        # context (ctx - payload - scaffolding), not a ratio × payload
        # fraction. Pre-#256 a ratio=1 formula at payload=5k floored
        # mtr at 27k via _MIN_CONTEXT pad; now the SPOF gets the
        # entire remaining window.
        from engine.llm import _resolve_stage_override
        spec, _ = _resolve_stage_override(Mode.TEE, "patterns")
        scaffolding = _scaffolding_tokens_for_stage("patterns")
        payload = 5_000
        mt = dynamic_max_tokens(payload, Mode.TEE, "patterns")
        expected = min(spec.context_window - payload - scaffolding,
                       spec.max_output)
        assert mt == expected
        # Sanity: mt dwarfs the old ratio=1 + floor result (~27k).
        # The SPOF carve-out hands the call the rest of the model's
        # window, exactly what #256 asked for.
        assert mt >= spec.context_window - payload - scaffolding
        assert mt > 100_000  # well above the old min-context floor

    def test_capped_at_real_payload_headroom(self):
        # Large-but-realistic payload where the formula exceeds the
        # real headroom (ctx - payload - scaffolding). Cap must bind
        # at remaining context, not at the static max_output_for_stage
        # halving.
        from engine.llm import _resolve_stage_override, _scaffolding_tokens_for_stage
        spec, _ = _resolve_stage_override(Mode.TEE, "extract")
        scaffolding = _scaffolding_tokens_for_stage("extract")
        # Pick payload that makes raw=3p+2048 exceed ctx-p-scaffolding
        # (need p > (ctx-scaffolding-2048)/4). For 128k ctx that's ~31k.
        payload = 60_000
        cap = spec.context_window - payload - scaffolding
        mt = dynamic_max_tokens(payload, Mode.TEE, "extract")
        assert mt == cap, (
            f"expected cap {cap} to bind (real-payload headroom), got {mt}"
        )

    def test_zero_payload_gets_full_min_context(self):
        # No payload → max_tokens = _MIN_CONTEXT_TOKENS (all budget
        # goes to output side since input is 0).
        mt = dynamic_max_tokens(0, Mode.TEE, "extract")
        assert mt == _MIN_CONTEXT_TOKENS

    def test_scaffolding_does_NOT_multiply_output(self, monkeypatch):
        """Regression guard: scaffolding is fixed framing, not payload.
        If this test breaks, someone reintroduced the scaffolding-in-ratio bug."""
        # Only way to be sure: formula must equal exactly
        # ratio × payload + headroom (NOT ratio × (payload + scaffolding))
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage", lambda stage, mode: False)
        payload = 10_000
        mt = dynamic_max_tokens(payload, Mode.TEE, "extract")
        # If the buggy formula were in place, result would be
        # 4 × (10000 + scaffolding) + 2048 — 4×scaffolding more than correct.
        assert mt == 4 * payload + _HEADROOM_TOKENS
        assert mt != 4 * (payload + _scaffolding_tokens_for_stage("extract")) + _HEADROOM_TOKENS

    def test_per_stage_ratio_is_respected(self):
        """Extract (ratio=4) reserves more per payload than entities (ratio=1)
        on payloads where formula > floor AND formula < max_output cap."""
        # Use a mid-size payload that beats the 32k floor but stays
        # under extract's max_output cap.
        # extract formula: 4 × 15000 + 2048 = 62,048 > floor; < cap
        # entities formula: 1 × 15000 + 2048 = 17,048 > floor (17,000); < cap
        payload = 15_000
        extract_mt = dynamic_max_tokens(payload, Mode.TEE, "extract")
        entities_mt = dynamic_max_tokens(payload, Mode.TEE, "entities")
        assert extract_mt - entities_mt == (4 - 1) * payload


class TestMinContextFloor:
    """Every call reserves at least _MIN_CONTEXT_TOKENS total
    context (payload + max_tokens), even tiny payloads."""

    def test_tiny_payload_gets_boosted_max_tokens(self):
        mt = dynamic_max_tokens(500, Mode.TEE, "extract")
        assert 500 + mt >= _MIN_CONTEXT_TOKENS

    def test_exactly_at_floor_payload(self):
        # payload = _MIN_CONTEXT_TOKENS exactly → min_for_context = 0.
        # mt = min(formula, real_cap, provider_cap) where:
        #   formula      = ratio × payload + headroom
        #   real_cap     = ctx - payload - scaffolding
        #   provider_cap = spec.max_output (model emission ceiling)
        from engine.llm import _resolve_stage_override, _scaffolding_tokens_for_stage
        spec, _ = _resolve_stage_override(Mode.TEE, "extract")
        scaffolding = _scaffolding_tokens_for_stage("extract")
        payload = _MIN_CONTEXT_TOKENS
        formula = 4 * payload + _HEADROOM_TOKENS
        real_cap = spec.context_window - payload - scaffolding
        provider_cap = spec.max_output
        mt = dynamic_max_tokens(payload, Mode.TEE, "extract")
        assert mt == min(formula, real_cap, provider_cap)

    def test_big_payload_floor_does_not_apply(self, monkeypatch):
        # Formula beats floor for big payloads
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage", lambda stage, mode: False)
        mt = dynamic_max_tokens(14_119, Mode.TEE, "extract")
        assert mt == 4 * 14_119 + _HEADROOM_TOKENS  # formula, not floor

    def test_floor_never_exceeds_cap(self):
        # Floor must still respect stage's max_output cap
        # Tiny payload on a stage with small cap should still cap out
        huge_cap = compute_budget(Mode.TEE, "extract").max_output
        mt = dynamic_max_tokens(1, Mode.TEE, "extract")
        assert mt <= huge_cap


# ── _TYPICAL_RATIO_BY_STAGE ─────────────────────────────────────────────────────

class TestTypicalRatios:
    def test_extract_typical(self):
        # Typical is 0.5 for extract
        assert _TYPICAL_RATIO_BY_STAGE["extract"] == 0.5

    def test_entities_typical(self):
        # entities typical = 0.5 / _AVG_ENTITIES_PER_FACT (=1) = 0.5
        assert _TYPICAL_RATIO_BY_STAGE["entities"] == 0.5

    def test_patterns_typical(self):
        assert _TYPICAL_RATIO_BY_STAGE["patterns"] == 0.1

    def test_single_call_stages_return_none(self):
        # entities_dedupe / insights / actions are single-LLM-call
        # SPOFs — always 1 call, no batching, no fan-out. Typical-
        # ratio isn't informative for call counts; entries are
        # explicitly None to distinguish them from unknown stages.
        assert _TYPICAL_RATIO_BY_STAGE["entities_dedupe"] is None
        assert _TYPICAL_RATIO_BY_STAGE["insights"] is None
        assert _TYPICAL_RATIO_BY_STAGE["actions"] is None


# ── Scaffolding coverage per stage ─────────────────────────────────────────────

class TestScaffoldingTokens:
    def test_every_stage_has_scaffolding(self):
        """Every LLM stage must have a scaffolding entry — drift-guard."""
        for stage in ("metadata", "extract", "entities", "patterns", "insights", "actions"):
            assert _scaffolding_tokens_for_stage(stage) > 0

    def test_values_in_reasonable_range(self):
        """Scaffolding shouldn't be absurd relative to the 64k ctx.
        If someone sets it too high or too low, tests would catch."""
        for stage, scaffolding in _SCAFFOLDING_TOKENS_BY_STAGE.items():
            # All measured stage fixed scaffolding is 49-1100 tokens; our
            # values should cover that with slack but not exceed ~3k
            # (which would indicate over-padding).
            assert 100 <= scaffolding <= 3_000, (
                f"{stage}: scaffolding {scaffolding}t out of expected range"
            )


# ── Reasoning-aware ratio bump ─────────────────────────────────────────────────

class TestReasoningBump:
    """When reasoning is ON for the (mode, stage) call, dynamic_max_tokens
    bumps the effective ratio by +1 so reasoning tokens have headroom
    alongside content tokens (they share the same max_tokens budget)."""

    def test_bump_applied_when_reasoning_on(self, monkeypatch):
        # Force reasoning ON for entities by monkeypatching the helper
        # — keeps the formula assertion independent of whitelist /
        # config plumbing (covered in test_helper_integration).
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage",
                            lambda stage, mode: True)
        # Payload chosen so the floor doesn't bind: 32k floor − 20k = 12k,
        # raw with bump = 2 × 20k + 2048 = 42,048 > 12k.
        payload = 20_000
        mt = dynamic_max_tokens(payload, Mode.TEE, "entities")
        # entities ratio=1 + reasoning bump = 2 → 2 × 20000 + 2048 = 42,048
        assert mt == 2 * payload + _HEADROOM_TOKENS == 42_048

    def test_no_bump_when_reasoning_off(self, monkeypatch):
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage",
                            lambda stage, mode: False)
        payload = 20_000
        mt = dynamic_max_tokens(payload, Mode.TEE, "entities")
        # entities ratio=1, no bump → 1 × 20000 + 2048 = 22,048
        assert mt == 1 * payload + _HEADROOM_TOKENS == 22_048

    def test_bump_diff_equals_payload(self, monkeypatch):
        """The bump is exactly +1 × payload (universal, not per-stage).
        Run two calls with identical inputs differing only in reasoning
        state and assert the diff."""
        from engine import llm
        payload = 20_000
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage",
                            lambda stage, mode: False)
        off = dynamic_max_tokens(payload, Mode.TEE, "entities")
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage",
                            lambda stage, mode: True)
        on = dynamic_max_tokens(payload, Mode.TEE, "entities")
        assert on - off == payload

    def test_helper_resolves_via_stage_override(self, monkeypatch):
        """End-to-end: when the user toggle is OFF, the helper must
        return False so no bump is applied. Hermetic — patches the
        stage map to a known false-everywhere state so a user's local
        config (which may toggle some stages ON) doesn't sway the
        assertion."""
        from engine import llm
        forced_off = {
            stage: {**(llm._STAGE_MODEL_MAP.get(stage, {})), "reasoning": False}
            for stage in ("extract", "entities", "entities_dedupe",
                          "patterns", "insights", "actions")
        }
        monkeypatch.setattr(llm, "_STAGE_MODEL_MAP", forced_off)
        for stage in forced_off:
            assert llm._is_reasoning_on_for_stage(stage, Mode.TEE) is False, (
                f"reasoning unexpectedly ON for {stage} when toggle is False"
            )

    def test_helper_honors_user_toggle(self, monkeypatch):
        """When the user toggles reasoning ON for a whitelisted
        (provider, model, stage), the helper returns True and the
        bump applies. Covers the post-#107 path for entities_dedupe."""
        from engine import llm
        # entities_dedupe (provider=tinfoil, model=gpt-oss-120b,
        # stage=entities_dedupe) IS in _REASONING_WHITELIST per the
        # post-#107 fix; flip the user toggle and verify.
        original = llm._STAGE_MODEL_MAP.get("entities_dedupe", {})
        monkeypatch.setitem(llm._STAGE_MODEL_MAP, "entities_dedupe",
                            {**original, "reasoning": True})
        assert llm._is_reasoning_on_for_stage("entities_dedupe", Mode.TEE) is True


# ── Real-payload cap inside dynamic_max_tokens ─────────────────────────────────

class TestRealPayloadCap:
    """dynamic_max_tokens caps at `ctx - payload - scaffolding`
    (real remaining headroom), NOT at max_output_for_stage's static
    halving. The static function still exists for budget snapshots
    in stat records, but isn't used as the per-call cap."""

    def test_cap_is_real_payload_headroom_at_large_payload(self, monkeypatch):
        # At p=60k extract, raw=122k exceeds both old (83666) and new
        # (~65.5k) caps. Verify the new cap binds. Force reasoning
        # OFF so the assertion is independent of user config (which
        # may toggle some stages ON).
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage",
                            lambda stage, mode: False)
        from engine.llm import _resolve_stage_override
        spec, _ = _resolve_stage_override(Mode.TEE, "extract")
        scaffolding = _scaffolding_tokens_for_stage("extract")
        payload = 60_000
        expected_cap = spec.context_window - payload - scaffolding
        mt = dynamic_max_tokens(payload, Mode.TEE, "extract")
        assert mt == expected_cap
        # Sanity: this is *less* than the old static cap, because
        # remaining ctx after a 60k input is tighter than the halving
        # estimate. The old code would have requested mt + payload >
        # ctx, risking a server-side overflow.
        assert mt < compute_budget(Mode.TEE, "extract").max_output

    def test_cap_does_not_bind_for_typical_entities_payload(self, monkeypatch):
        # p=14k entities reasoning OFF: raw=16048, floor=18000 →
        # mt=18000 (floor wins). The cap (~112.5k) must not bind.
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage",
                            lambda stage, mode: False)
        payload = 14_000
        mt = dynamic_max_tokens(payload, Mode.TEE, "entities")
        from engine.llm import _resolve_stage_override
        spec, _ = _resolve_stage_override(Mode.TEE, "entities")
        scaffolding = _scaffolding_tokens_for_stage("entities")
        cap = spec.context_window - payload - scaffolding
        assert mt < cap, (
            f"cap {cap} should not bind at typical payload — got mt={mt}"
        )
        # Floor + headroom dominate, not the cap
        assert mt == _MIN_CONTEXT_TOKENS - payload == 18_000

    def test_cap_widens_for_smaller_payloads(self, monkeypatch):
        # Same stage, different payloads → smaller payload → larger
        # cap. Verifies the cap is computed against actual payload
        # (regression guard against re-introducing the static cap).
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage",
                            lambda stage, mode: False)
        from engine.llm import _resolve_stage_override
        spec, _ = _resolve_stage_override(Mode.TEE, "extract")
        scaffolding = _scaffolding_tokens_for_stage("extract")
        # Smaller pair so the formula at ratio=4 still fits under cap
        # for the small case but the big case overruns it.
        small_p = 20_000
        big_p = 60_000
        small_cap = spec.context_window - small_p - scaffolding
        big_cap = spec.context_window - big_p - scaffolding
        assert small_cap > big_cap
        # At p=60k the cap binds (raw=242048 > cap≈67k). At p=20k the
        # formula wins (raw=82048 < cap≈107k). Verify formula
        # dominates in the small case.
        mt_small = dynamic_max_tokens(small_p, Mode.TEE, "extract")
        mt_big = dynamic_max_tokens(big_p, Mode.TEE, "extract")
        assert mt_small == 4 * small_p + _HEADROOM_TOKENS  # formula
        assert mt_big == big_cap                           # cap binds


# ── Regression: cap-hit scenario from the journal payload ──────────────────────

class TestCapHitRegression:
    """Specific values the pipeline cap-hit on 2026-04-24. These chunks now
    get max_tokens > their observed cap, so the failure should not recur."""

    def test_42k_char_chunk(self, monkeypatch):
        # Real chunk that cap-hit at 17,648 output tokens. With ratio=4
        # the formula reserves 58,524 — well above what the model used
        # before failing. Pin reasoning OFF so the explicit 58,524 holds
        # (the +1 reasoning bump is covered by TestReasoningBump).
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage", lambda stage, mode: False)
        payload = tokens_from_chars(42_357)
        mt = dynamic_max_tokens(payload, Mode.TEE, "extract")
        assert mt > 17_648, (
            f"Chunk previously cap-hit at 17,648; reservation now {mt}t — "
            f"should exceed historical cap-hit value"
        )
        assert mt == 58_524  # explicit: 4 × 14,119 + 2,048

    def test_37k_char_chunk(self, monkeypatch):
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage", lambda stage, mode: False)
        payload = tokens_from_chars(37_500)
        mt = dynamic_max_tokens(payload, Mode.TEE, "extract")
        assert mt > 16_029
        assert mt == 52_048  # explicit: 4 × 12,500 + 2,048


# ── compute_budget: unified entry point ────────────────────────────────────────

class TestComputeBudgetWrapperParity:
    """compute_budget is the single source of truth; the named functions
    (chunk_cap_for_stage, dynamic_max_tokens) are thin wrappers. These
    tests pin parity so regressions in either side are caught."""

    def test_chunk_cap_matches_compute_budget_max_input(self):
        for stage in ("extract", "entities", "patterns", "insights",
                      "actions", "entities_dedupe"):
            assert (
                chunk_cap_for_stage(Mode.TEE, stage)
                == compute_budget(Mode.TEE, stage).max_input
            )

    def test_dynamic_max_tokens_matches_compute_budget_per_call(self, monkeypatch):
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage",
                            lambda stage, mode: False)
        for stage, payload in [
            ("extract", 14_119), ("entities", 10_000),
            ("patterns", 5_000),  ("insights", 3_000),
        ]:
            assert (
                dynamic_max_tokens(payload, Mode.TEE, stage)
                == compute_budget(Mode.TEE, stage,
                                  payload_tokens=payload).max_output
            )


class TestComputeBudgetIngestSentinel:
    """ingest is a parent stage that does NOT issue LLM calls at this
    level. compute_budget reports has_llm_calls=False with zero numeric
    fields — callers detect this via the explicit flag, not via
    None-checking."""

    def test_ingest_has_llm_calls_false(self):
        b = compute_budget(Mode.TEE, "ingest")
        assert b.has_llm_calls is False

    def test_ingest_numeric_fields_are_zero_not_none(self):
        b = compute_budget(Mode.TEE, "ingest")
        assert b.max_input == 0
        assert b.max_output == 0
        assert b.scaffolding == 0
        # Type check: zero, not None — callers can do `if b.max_input`
        # without hitting NoneType comparisons.
        assert isinstance(b.max_input, int)
        assert isinstance(b.max_output, int)
        assert isinstance(b.scaffolding, int)

    def test_ingest_payload_tokens_still_returns_sentinel(self):
        # Even when called per-call (which shouldn't happen for ingest),
        # the sentinel shape is preserved.
        b = compute_budget(Mode.TEE, "ingest", payload_tokens=10_000)
        assert b.has_llm_calls is False
        assert b.max_output == 0


class TestComputeBudgetVisionFirstClass:
    """Vision is a real LLM stage (per-image transcription via the
    configured vision model, kimi-k2-6 on Tinfoil). It must produce a
    real budget, not the None-fallback that pre-#205 callers got."""

    def test_vision_has_llm_calls_true(self):
        b = compute_budget(Mode.TEE, "vision")
        assert b.has_llm_calls is True

    def test_vision_returns_real_budget(self):
        b = compute_budget(Mode.TEE, "vision")
        assert b.max_input > 0
        assert b.max_output > 0
        assert b.scaffolding > 0

    def test_vision_per_call_budget_sized_to_payload(self, monkeypatch):
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage",
                            lambda stage, mode: False)
        # ratio=1.0 for vision. payload=20k → raw = 1.0*20k + 2k = 22k.
        # _MIN_CONTEXT floor: 32k - 20k = 12k. raw beats floor → mt=22k.
        payload = 20_000
        b = compute_budget(Mode.TEE, "vision", payload_tokens=payload)
        assert b.max_output == max(int(1.0 * payload) + _HEADROOM_TOKENS,
                                   _MIN_CONTEXT_TOKENS - payload)


class TestComputeBudgetRoundTrip:
    """Round-trip property: feeding `compute_budget(...).max_input` back
    as `payload_tokens=max_input` produces a per-call budget that's
    coherent — output cap doesn't exceed the provider cap, and the
    payload + max_output never breach the context window. Catches
    regressions where the dual formulas (input-side vs payload-side)
    drift apart."""

    def test_round_trip_extract_tee(self, monkeypatch):
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage",
                            lambda stage, mode: False)
        from engine.llm import _resolve_stage_override
        b_static = compute_budget(Mode.TEE, "extract")
        b_per_call = compute_budget(Mode.TEE, "extract",
                                    payload_tokens=b_static.max_input)
        spec, _ = _resolve_stage_override(Mode.TEE, "extract")
        # Per-call output never exceeds provider cap.
        assert b_per_call.max_output <= spec.max_output
        # Payload + scaffolding + max_output fits in context window.
        # (Real-cap clamp inside compute_budget enforces this.)
        assert (b_static.max_input + b_per_call.scaffolding +
                b_per_call.max_output) <= spec.context_window

    def test_round_trip_all_chat_stages(self, monkeypatch):
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage",
                            lambda stage, mode: False)
        from engine.llm import _resolve_stage_override
        for stage in ("extract", "entities", "entities_dedupe",
                      "patterns", "insights", "actions", "vision"):
            b_static = compute_budget(Mode.TEE, stage)
            b_per_call = compute_budget(Mode.TEE, stage,
                                        payload_tokens=b_static.max_input)
            spec, _ = _resolve_stage_override(Mode.TEE, stage)
            assert b_per_call.max_output <= spec.max_output, (
                f"{stage}: per-call max_output {b_per_call.max_output} "
                f"exceeds provider cap {spec.max_output}"
            )
            assert (b_static.max_input + b_per_call.scaffolding +
                    b_per_call.max_output) <= spec.context_window, (
                f"{stage}: payload + scaffolding + max_output exceeds ctx"
            )


class TestComputeBudgetEdgeCases:
    def test_payload_exceeds_context_window_does_not_go_negative(self, monkeypatch):
        # When the caller passes a payload bigger than the context
        # window, compute_budget must clamp gracefully — real_cap
        # would otherwise go negative.
        from engine import llm
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage",
                            lambda stage, mode: False)
        from engine.llm import _resolve_stage_override
        spec, _ = _resolve_stage_override(Mode.TEE, "extract")
        oversized = spec.context_window + 50_000
        b = compute_budget(Mode.TEE, "extract", payload_tokens=oversized)
        assert b.max_output >= 0, (
            f"max_output went negative on oversized payload: {b.max_output}"
        )

    def test_ratio_zero_falls_back_to_default(self, monkeypatch):
        # Stage with ratio=0 must not divide-by-zero. The fallback path
        # uses ctx-only sizing (no output_cap halving, since ratio=0
        # means no output is reserved).
        from engine import llm
        # Inject a synthetic ratio=0 entry.
        monkeypatch.setitem(llm._MAX_RATIO_BY_STAGE, "_zero_ratio_test", 0)
        monkeypatch.setitem(llm._SCAFFOLDING_TOKENS_BY_STAGE,
                            "_zero_ratio_test", 100)
        b = compute_budget(Mode.TEE, "_zero_ratio_test")
        assert b.has_llm_calls is True
        assert b.max_input > 0
        # ratio=0 means ratio×max_input=0, so static max_output is 0.
        assert b.max_output == 0


class TestStageBudgetDataclass:
    """Sanity: StageBudget is a frozen dataclass exporting the four
    documented fields. Drift-guard so a refactor that adds/renames
    fields trips the test."""

    def test_field_set(self):
        b = compute_budget(Mode.TEE, "extract")
        assert isinstance(b, StageBudget)
        assert hasattr(b, "max_input")
        assert hasattr(b, "max_output")
        assert hasattr(b, "scaffolding")
        assert hasattr(b, "has_llm_calls")

    def test_frozen(self):
        b = compute_budget(Mode.TEE, "extract")
        # Frozen dataclasses raise on attribute assignment.
        import dataclasses

        import pytest
        with pytest.raises(dataclasses.FrozenInstanceError):
            b.max_input = 0  # type: ignore[misc]


# ── Model-boundary recompute on retry fallback ────────────────────────────────


class TestRetryModelSwapRecomputesMaxOutput:
    """Issue #626: when the degrading-retry cascade falls back to an
    alternate model (e.g. gpt-oss-120b → gemma4-31b on entities_dedupe
    cap-hit), `dynamic_max_tokens` / `compute_budget` must size against
    the alternate's window, not the stage anchor's.

    Pre-fix symptom: try 1 (gpt-oss, 128k ctx) and try 2 (gemma4-31b,
    256k ctx) both got the same max_tokens — the fallback inherited the
    anchor's old cap instead of getting the wider headroom gemma's 256k
    window affords. Both retries cap-hit at the same number, defeating
    the point of the model swap.

    The fix threads a `spec_override` through `compute_budget` /
    `dynamic_max_tokens` so the same sizing formula runs against the
    forced-spec at the model boundary, no separate retry-specific
    code path."""

    def test_compute_budget_uses_spec_override(self):
        # entities_dedupe is one of the cap-hit-fallback stages; gpt-oss
        # (128k) is the production anchor, gemma4-31b (256k) is the
        # cap-hit fallback target. Run compute_budget with each as the
        # override and assert the wider window produces a wider per-call cap.
        from engine.llm import _TINFOIL_GPT_OSS_120B, _TINFOIL_GEMMA4_31B
        payload = 54_000
        gptoss_b = compute_budget(
            Mode.TEE, "entities_dedupe",
            payload_tokens=payload, spec_override=_TINFOIL_GPT_OSS_120B,
        )
        gemma_b = compute_budget(
            Mode.TEE, "entities_dedupe",
            payload_tokens=payload, spec_override=_TINFOIL_GEMMA4_31B,
        )
        scaffolding = _scaffolding_tokens_for_stage("entities_dedupe")
        # entities_dedupe has ratio=None, so max_output = min(real_cap,
        # spec.max_output). Both models default max_output = ctx.
        gptoss_expected = min(
            _TINFOIL_GPT_OSS_120B.context_window - payload - scaffolding,
            _TINFOIL_GPT_OSS_120B.max_output,
        )
        gemma_expected = min(
            _TINFOIL_GEMMA4_31B.context_window - payload - scaffolding,
            _TINFOIL_GEMMA4_31B.max_output,
        )
        assert gptoss_b.max_output == gptoss_expected
        assert gemma_b.max_output == gemma_expected
        # The whole point: the wider-window override produces a
        # meaningfully larger cap (gemma's 256k beats gpt-oss's 128k).
        assert gemma_b.max_output > gptoss_b.max_output + 100_000

    def test_dynamic_max_tokens_uses_spec_override(self, monkeypatch):
        # Same payload, same (mode, stage); only the override spec
        # differs. The override is what the runtime passes from
        # complete() when `_force_model_id=True` swaps the dispatch
        # to the cap-hit fallback model.
        from engine import llm
        # Reasoning OFF: the model-fallback step always forces
        # reasoning off (patterns.py / scheduler.py). Plus
        # entities_dedupe has ratio=None so the bump branch doesn't
        # apply anyway; pin the helper to make this independent of
        # whitelist state.
        monkeypatch.setattr(llm, "_is_reasoning_on_for_stage",
                            lambda stage, mode: False)
        from engine.llm import _TINFOIL_GPT_OSS_120B, _TINFOIL_GEMMA4_31B
        payload = 54_000
        gptoss_mt = dynamic_max_tokens(
            payload, Mode.TEE, "entities_dedupe",
            spec_override=_TINFOIL_GPT_OSS_120B,
        )
        gemma_mt = dynamic_max_tokens(
            payload, Mode.TEE, "entities_dedupe",
            spec_override=_TINFOIL_GEMMA4_31B,
        )
        # The pre-fix bug: gemma_mt would equal gptoss_mt (both sized
        # against the stage anchor, gpt-oss's 128k). Post-fix the alt
        # spec's 256k window dominates.
        assert gemma_mt > gptoss_mt
        # Concretely: gemma's per-call cap must be at least 2×
        # gpt-oss's on this payload (256k vs 128k ctx).
        assert gemma_mt >= 2 * gptoss_mt



# ── max_workers ────────────────────────────────────────────────────────────────

class TestMaxWorkers:
    """Drift-guard on the global LLM-fan-out cap. Rationale lives in
    `llm.max_workers`."""

    def test_cloud_capped_at_16(self):
        assert max_workers(Mode.TEE) == 16

    def test_local_serial(self):
        assert max_workers(Mode.LOCAL) == 1
