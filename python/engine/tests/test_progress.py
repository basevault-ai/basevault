"""Unit tests for `progress.ProgressTracker` + historical loader.

Covers the core behaviors that drive the bar in App.jsx:
  - cumulative pipeline-total est_calls (Part 1)
  - time-based ETA grounded in historical (stage, model) durations
    via TWO coefficients: sec/call (fixed overhead) + sec/token
    (variable generation cost). eta_per_call = sec/call_fixed +
    sec/token × est_tokens.
  - independent sanity bands per coefficient
  - bar position is monotonic non-decreasing under ETA growth
"""
from __future__ import annotations

import json

import pytest


from engine.progress import (  # noqa: E402
    FALLBACK_SECONDS_PER_CALL,
    ProgressTracker,
    _decompose_coefficients,
    load_historical_durations,
    per_call_seconds_at_preflight,
)


# Test helper: convert a list of plain durations to (dur, tokens=0)
# tuples — the historical-format the new loader returns. Token=0 means
# no token data, sec/token coefficient stays at 0, eta falls back to
# the pure sec/call median (matching the round-1 behavior).
def _no_tokens(durations: list[float]) -> list[tuple[float, int]]:
    return [(d, 0) for d in durations]


# Default fan-out for these tests forces each stage's parallelism to
# 1 so per_call × remaining is the wall-clock without batching. The
# coverage of parallelism behavior lives in the dedicated test below.
SERIAL = {"extract": 1, "metadata": 1, "entities": 1,
          "entities_dedupe": 1, "patterns": 1,
          "insights": 1, "actions": 1}


def test_pipeline_total_sums_across_stages():
    t = ProgressTracker()
    t.register_stage("metadata", "m", 1)
    t.register_stage("extract", "m", 5)
    t.register_stage("entities", "m", 3)
    t.register_stage("entities_dedupe", "m", 1)
    t.register_stage("patterns", "m", 4)
    t.register_stage("insights", "m", 1)
    t.register_stage("actions", "m", 1)
    assert t.compute_pipeline_total_calls() == 1 + 5 + 3 + 1 + 4 + 1 + 1


def test_register_stage_does_not_shrink_below_completed():
    """Refining `est_calls` downward must not drop below already-
    completed; otherwise remaining=negative."""
    t = ProgressTracker()
    t.register_stage("extract", "m", 10)
    t.mark_stage_started("extract")
    for _ in range(7):
        t.record_call("extract", 60.0, completion_tokens=100)
    t.register_stage("extract", "m", 3)  # tries to shrink
    assert t._stages["extract"].est_calls == 7


def test_embeddings_one_unit_while_running_real_count_on_finish():
    """Issue #581 — the 'estimated while running, real on completion'
    contract for embeddings, end to end on the tracker.

    Embeddings registers as ONE collective unit so it does not balloon
    the pipeline denominator while the long stages run. As its many
    sub-second batches complete, `record_call` bumps completed_calls
    but does NOT re-widen est_calls (only register_stage /
    bump_est_calls / mark_stage_finished move it) — so the denominator
    stays put and the bar can't stall/lurch at embeddings. On stage
    finish, `mark_stage_finished` snaps est_calls down to the REAL
    completed count, restoring the true number for the done state."""
    t = ProgressTracker()
    t.register_stage("extract", "m", 5)
    # Embeddings comes in as a single collective unit, NOT its real
    # batch count (would be e.g. 7 for a ~200-record run).
    t.register_stage("embeddings", "e", 1)
    assert t.compute_pipeline_total_calls() == 5 + 1

    t.mark_stage_started("embeddings")
    REAL_BATCHES = 7
    for _ in range(REAL_BATCHES):
        t.record_call("embeddings", 0.3, completion_tokens=0)

    # While running: completed climbed, but the denominator did NOT
    # balloon — embeddings still contributes its one unit.
    assert t._stages["embeddings"].completed_calls == REAL_BATCHES
    assert t._stages["embeddings"].est_calls == 1
    assert t.compute_pipeline_total_calls() == 5 + 1

    # On completion: snap down to the REAL count.
    t.mark_stage_finished("embeddings")
    assert t._stages["embeddings"].est_calls == REAL_BATCHES
    assert t.compute_pipeline_total_calls() == 5 + REAL_BATCHES


def test_resume_init_total_must_dominate_prior_cycles_completed():
    """Pin the post-resume bar-clamping contract.

    Symptom (observed on a paused-mid-entities resume): after resume,
    the run-row's progress chip reads 'entities N/N' for the rest of
    the run even though the jsonl is well past entities and into
    insights / actions / embeddings.

    Mechanism: Rust's `derive_run_state` counts `completed` as
    cumulative leaf successes across ALL cycles. Python's `_emit()`
    reports `total` from the in-process ProgressTracker, which is a
    FRESH instance per cycle. On a resume, the runner seeds the
    tracker's PAST stages from llm-calls.jsonl (extract / entities)
    but does NOT register upcoming stages (patterns / insights /
    actions / embeddings) until each one enters its body. Net: the
    emitted `total` matches the seed sum until each upcoming stage
    enters; meanwhile the cumulative leaf count Rust reports keeps
    growing through cycle 2. Once `completed` reaches the seed-
    derived `total`, JSX clamps `displayDone = min(completed, total)
    = total`, the chip reads 'total/total' and the bar appears
    frozen at the seed snapshot's stage label.

    Fix shape: at resume init, after past-stage seeding, register
    every upcoming stage with its initial estimate (same as the
    fresh-run path does at resume_from=="start"). The tracker total
    then reflects the FULL pipeline, dominating the cumulative leaf
    count and unclamping the bar throughout cycle 2.

    This pins the contract on the tracker side. The runner-side
    change is verified end-to-end via real pause/resume build
    verification described in the PR.
    """
    t = ProgressTracker()
    # Step 1: seed past stages from prior cycle's jsonl (mirrors the
    # _bootstrap_per_stage_from_jsonl path the runner already runs).
    SEEDED = {"extract": 14, "entities": 5}
    for stage, count in SEEDED.items():
        t.register_stage(stage, "m", count)
        t._stages[stage].completed_calls = count
        t._stages[stage].mark_started(0.0)
        t._stages[stage].mark_finished(0.001)
    bootstrap_completed = sum(SEEDED.values())

    # Step 2 (the post-fix sequence): register ALL upcoming stages
    # with full per-stage estimates — same shape as the fresh-run
    # `_register_initial_estimates(_estimate_per_stage(docs))`.
    FULL_PIPELINE = {
        "extract": 14, "entities": 10, "entities_dedupe": 1,
        "patterns": 12, "insights": 1, "actions": 1, "embeddings": 21,
    }
    for stage, est in FULL_PIPELINE.items():
        t.register_stage(stage, "m", est)

    total = t.compute_pipeline_total_calls()
    assert total == sum(FULL_PIPELINE.values()), (
        f"total should reflect full pipeline est ({sum(FULL_PIPELINE.values())}); "
        f"got {total} — register_initial_estimates must widen est_calls for "
        f"all upcoming stages, even those whose initial est is below their seed"
    )
    # Contract: tracker total must dominate the cumulative leaf
    # count from prior cycles, otherwise the bar clamps post-resume.
    assert total > bootstrap_completed, (
        f"total ({total}) must exceed prior-cycle leaves "
        f"({bootstrap_completed}) — otherwise displayDone clamps at total "
        f"and the chip reads 'baseline/baseline' for the rest of the run"
    )


# ── sec/call-only path (token=0) ───────────────────────────────────


def test_estimate_uses_historical_median_at_cold_start_no_tokens():
    """When no token data exists in the rolling window, fall back to
    the pure sec/call median × N — matches round-1 behavior."""
    hist = {("extract", "m"): _no_tokens([50.0, 60.0, 70.0, 80.0, 90.0])}
    t = ProgressTracker(historical_durations=hist, parallelism_per_stage=SERIAL)
    t.register_stage("extract", "m", 4)
    # No live data, no tokens — eta = median × 4 = 70 × 4 = 280.
    assert t.estimate_stage_seconds("extract") == pytest.approx(280.0)


def test_estimate_falls_back_when_history_missing():
    from engine.progress import FALLBACK_SECONDS_PER_CALL
    t = ProgressTracker(parallelism_per_stage=SERIAL)
    t.register_stage("extract", "m", 2)
    eta = t.estimate_stage_seconds("extract")
    # Falls back to FALLBACK_SECONDS_PER_CALL["extract"] × 2.
    assert eta == pytest.approx(2 * FALLBACK_SECONDS_PER_CALL["extract"])


# ── sec/token path (decomposition) ─────────────────────────────────


def test_decompose_coefficients_pure_linear():
    """When duration = α + β × tokens for known α, β, the
    decomposition recovers them within rounding."""
    alpha = 5.0
    beta = 0.01
    samples = [(alpha + beta * t, t) for t in (100, 500, 1000, 5000, 10000)]
    sec_call, sec_token, median_tokens = _decompose_coefficients(samples, 30.0)
    # sec_token = median(d/t). For our linear model, d/t = α/t + β. As
    # t grows, this approaches β. Median over the sample is between
    # α/min_t + β and α/max_t + β — not equal to β exactly, but small.
    assert 0.005 < sec_token < 0.06
    # sec_call = median(d - sec_token × t). For samples where d/t
    # diverges most from sec_token (small t), the residual differs;
    # for large t it's tiny. Median of residuals should be near α
    # but skewed downward.
    assert 0.0 <= sec_call < alpha + 1.0
    # median_tokens = median of token counts.
    assert median_tokens == 1000


def test_decompose_coefficients_no_token_data_falls_back_to_call():
    """When all samples have tokens=0 (legacy logs), sec/token=0 and
    sec/call_fixed = median(duration)."""
    samples = [(60.0, 0), (70.0, 0), (80.0, 0)]
    sec_call, sec_token, median_tokens = _decompose_coefficients(samples, 30.0)
    assert sec_call == pytest.approx(70.0)
    assert sec_token == 0.0
    assert median_tokens == 0


def test_decompose_coefficients_fallback_when_empty():
    """Empty sample list → fallback constant for sec/call, 0 token rate."""
    sec_call, sec_token, _ = _decompose_coefficients([], 99.0)
    assert sec_call == 99.0
    assert sec_token == 0.0


def test_estimate_two_coefficient_path_uses_tokens():
    """When historical samples carry tokens, the estimator uses
    sec/call_fixed + sec/token × tokens. For the typical (median)
    call this approximates the median total duration."""
    # Synthetic: every call has duration ≈ 5s overhead + 0.01 × tokens.
    samples = [(5.0 + 0.01 * t, t) for t in (1000, 2000, 3000, 4000, 5000)]
    hist = {("extract", "m"): samples}
    t = ProgressTracker(historical_durations=hist, parallelism_per_stage=SERIAL)
    # Use the median tokens (3000) as the per-call estimate.
    t.register_stage("extract", "m", 3, est_tokens_per_call=3000)
    eta = t.estimate_stage_seconds("extract")
    # Per-call ≈ 5 + 0.01 × 3000 = 35s. 3 calls × 35 = 105s.
    assert 90 < eta < 120


def test_estimate_scales_with_tokens_per_call():
    """ETA should grow with est_tokens_per_call when the stage has a
    non-zero sec/token coefficient. This is the variable-payload
    behavior the brief required."""
    samples = [(5.0 + 0.01 * t, t) for t in (1000, 2000, 3000)]
    hist = {("extract", "m"): samples}
    t = ProgressTracker(historical_durations=hist, parallelism_per_stage=SERIAL)
    t.register_stage("extract", "m", 1, est_tokens_per_call=1000)
    eta_small = t.estimate_stage_seconds("extract")
    t.register_stage("extract", "m", 1, est_tokens_per_call=10000)
    eta_large = t.estimate_stage_seconds("extract")
    # 10× tokens → roughly 3-5× wall-clock (sec/token × tokens
    # dominates over sec/call). Without sec/token, eta_large would
    # equal eta_small.
    assert eta_large > 3 * eta_small


def test_live_takeover_per_coefficient_independent():
    """sec/call coefficient and sec/token coefficient have independent
    LIVE_TRUST_MIN_SAMPLES gates. Recording 3 calls with token data
    flips BOTH coefficients to live-median. Recording 3 calls with
    tokens=0 flips only sec/call.

    Round-4 follow-up rewrite: the LIVE_TRUST_MIN_SAMPLES=3 step has
    been replaced by a completion-ratio-weighted blend. After 3 of
    10 calls done, weight_live = 0.3 — so the blend is 30% live + 70%
    historical (NOT 100% live as the prior step would give). ETA grows
    proportionally with the live shift but more gradually.
    """
    # Historical: linear model, tokens-bearing samples.
    samples = [(5.0 + 0.01 * t, t) for t in (1000, 2000, 3000, 4000, 5000)]
    hist = {("extract", "m"): samples}
    t = ProgressTracker(historical_durations=hist, parallelism_per_stage=SERIAL)
    t.register_stage("extract", "m", 10, est_tokens_per_call=3000)
    t.mark_stage_started("extract")
    # Live: 3 calls each at 100s with 3000 tokens (much slower than
    # historical's ~35s for 3000 tokens).
    for _ in range(3):
        t.record_call("extract", 100.0, completion_tokens=3000)
    eta_after = t.estimate_stage_seconds("extract")
    # weight_live = 3/10 = 0.3. 7 remaining × per_call.
    # historical per_call (3000 tokens) ≈ 5 + 0.01×3000 = 35s
    # live per_call ≈ ~100s (clamped components)
    # blended ≈ 0.7×35 + 0.3×100 ≈ 54.5s. eta ≈ 7×54.5 = 381s.
    # Accept a window — exact value depends on clamping internals.
    assert 300 < eta_after < 500


def test_coefficient_blend_is_smooth_across_completion_ratio():
    """The completion-ratio blend has NO step function. As completed
    grows from 0 → est_calls, the per-call coefficient transitions
    smoothly from historical-only to live-only. No sample-count
    threshold causes a discontinuous jump in ETA.

    Regression for the user-reported bar 5%→20% + ETA 5m10s→7m
    simultaneous jump caused by the prior LIVE_TRUST_MIN_SAMPLES=3
    step.
    """
    # Historical median per-call ≈ 30s. Live calls landing at 90s.
    hist = {("extract", "m"): [(30.0, 1000)] * 5}
    t = ProgressTracker(
        historical_durations=hist, parallelism_per_stage=SERIAL)
    t.register_stage("extract", "m", 10, est_tokens_per_call=1000)
    t.mark_stage_started("extract")
    # Record 1 slow live call. weight_live = 1/10 = 0.1. Blend
    # should be 90% historical + 10% live.
    t.record_call("extract", 90.0, completion_tokens=1000)
    eta1 = t.estimate_stage_seconds("extract")
    # Record 2 more, completed=3. weight_live = 0.3.
    t.record_call("extract", 90.0, completion_tokens=1000)
    t.record_call("extract", 90.0, completion_tokens=1000)
    eta2 = t.estimate_stage_seconds("extract")
    # Record 4 more, completed=7. weight_live = 0.7.
    for _ in range(4):
        t.record_call("extract", 90.0, completion_tokens=1000)
    eta3 = t.estimate_stage_seconds("extract")
    # ETAs should grow gradually as live takes over (live is slower
    # than historical). No single update should cause a 2x jump.
    # Specifically, the per-call coefficient at 30% (eta2) should be
    # at most ~2x the per-call at 10% (eta1) — smooth ramp.
    # Compute per-call by dividing by remaining count.
    per_call_1 = eta1 / 9   # 9 remaining
    per_call_2 = eta2 / 7   # 7 remaining
    per_call_3 = eta3 / 3   # 3 remaining
    # Each step's per-call grows but not by step-function magnitudes.
    assert per_call_2 < 1.6 * per_call_1, (
        f"per-call jumped too fast 0.1→0.3: {per_call_1:.1f}→{per_call_2:.1f}"
    )
    assert per_call_3 < 1.6 * per_call_2, (
        f"per-call jumped too fast 0.3→0.7: {per_call_2:.1f}→{per_call_3:.1f}"
    )
    # And per_call_3 (mostly-live) > per_call_1 (mostly-hist).
    assert per_call_3 > per_call_1


def test_per_token_clamp_pulls_outlier_down():
    """A single live call with 100x the typical sec/token rate must be
    clamped to PER_CALL_CEIL × historical_sec_per_token before going
    into the live blend."""
    # Historical sec/token rate ≈ 0.01s/token.
    samples = [(5.0 + 0.01 * t, t) for t in (1000, 2000, 3000, 4000, 5000)]
    hist = {("extract", "m"): samples}
    t = ProgressTracker(
        historical_durations=hist, parallelism_per_stage=SERIAL,
    )
    t.register_stage("extract", "m", 5, est_tokens_per_call=3000)
    t.mark_stage_started("extract")
    # 1 outlier: 1000s for 1000 tokens → live rate 1.0 s/token (100×
    # historical 0.01). Clamp ceiling 3.0 × 0.01 = 0.03.
    t.record_call("extract", 1000.0, completion_tokens=1000)
    eta = t.estimate_stage_seconds("extract")
    # If unclamped, live rate would be ~1.0 → per_call = 1.0 × 3000
    # = 3000s, total > 12000s. With clamp at 0.03 the contribution
    # is bounded.
    assert eta < 1500


def test_per_call_fixed_clamp_pulls_overhead_outlier_down():
    """A live call whose residual fixed overhead exceeds historical
    by 10× must be clamped before it dominates the live median."""
    samples = [(5.0 + 0.01 * t, t) for t in (1000, 2000, 3000, 4000, 5000)]
    hist = {("extract", "m"): samples}
    t = ProgressTracker(
        historical_durations=hist, parallelism_per_stage=SERIAL,
    )
    t.register_stage("extract", "m", 5, est_tokens_per_call=3000)
    t.mark_stage_started("extract")
    # 1 call: very high fixed overhead — duration 80s but only 1 token.
    # Residual fixed = 80 - 0.03 × 1 = ~80s, way more than historical
    # ~5s; clamped to PER_CALL_CEIL × 5 = 15s.
    t.record_call("extract", 80.0, completion_tokens=1)
    eta = t.estimate_stage_seconds("extract")
    # Per-call estimate would otherwise blow up; the clamp keeps it
    # bounded.
    assert eta < 500


def test_unknown_model_aggregates_across_models_in_decomposition():
    """When (stage, model_X) is missing, _hist_lookup aggregates
    across models for the stage. Decomposition still works on the
    aggregated population."""
    hist = {
        ("patterns", "model-a"): [(20.0, 100), (22.0, 100)],
        ("patterns", "model-b"): [(20.0, 100), (22.0, 100)],
    }
    t = ProgressTracker(
        historical_durations=hist,
        parallelism_per_stage={"patterns": 1},
    )
    t.register_stage("patterns", "model-c", 1)  # unseen
    eta = t.estimate_stage_seconds("patterns")
    # Aggregated samples = 4 × (~21s, 100 tokens). sec/token ≈ 0.21,
    # sec/call_fixed ≈ 0. With 100 default tokens: per_call ≈ 21s.
    assert 15 < eta < 30


# ── Parallelism / batching ─────────────────────────────────────────


def test_parallelism_uses_real_division_not_ceil():
    """Round 4 dropped `ceil(remaining/parallelism)` because it froze
    the ETA inside the first batch. Real division gives smooth
    proportional countdown as completed grows. With parallelism=16
    and per_call=70s, tail=2.0:
       5 calls remaining → (5/16) × 70 × 2 = 43.75s
       130 calls remaining → (130/16) × 70 × 2 = 1137.5s
    """
    from engine.progress import BATCH_TAIL_FACTOR

    hist = {("extract", "m"): _no_tokens([70.0] * 5)}
    t = ProgressTracker(
        historical_durations=hist,
        parallelism_per_stage={"extract": 16},
    )
    t.register_stage("extract", "m", 5)
    assert t.estimate_stage_seconds("extract") == pytest.approx(
        (5 / 16) * 70.0 * BATCH_TAIL_FACTOR
    )
    t.register_stage("extract", "m", 130)
    assert t.estimate_stage_seconds("extract") == pytest.approx(
        (130 / 16) * 70.0 * BATCH_TAIL_FACTOR
    )


def test_eta_decreases_proportionally_as_calls_complete_within_batch():
    """The freeze-bug regression: with 50 calls, parallelism=16,
    per_call=20s, tail=2: ETA drops smoothly as completed grows.
    The orchestrator's specific table:
       0 done  → ETA ≈ (50/16) × 20 × 2 = 125.0s
      10 done  → ETA ≈ (40/16) × 20 × 2 = 100.0s
      30 done  → ETA ≈ (20/16) × 20 × 2 = 50.0s
      50 done  → ETA = 0
    The pre-round-4 ceil() formula stuck at ceil(50/16)=4 batches
    (160s) for any completed < 50, which is what the user observed
    as the freeze.
    """
    hist = {("extract", "m"): _no_tokens([20.0] * 5)}
    t = ProgressTracker(
        historical_durations=hist,
        parallelism_per_stage={"extract": 16},
    )
    t.register_stage("extract", "m", 50)
    expected_at = {0: 125.0, 10: 100.0, 30: 50.0, 50: 0.0}
    for done, want in expected_at.items():
        # Reset and replay `done` completions.
        t._stages["extract"].completed_calls = done
        eta = t.estimate_stage_seconds("extract")
        assert eta == pytest.approx(want, abs=0.5), (
            f"freeze-bug regression: completed={done} expected {want}s "
            f"got {eta:.1f}s"
        )


def test_parallelism_per_stage_capped_at_16():
    """The global LLM-fan-out cap is 16 (matches Tinfoil router queue
    depth + bounds blast radius on disconnect events). PARALLELISM_PER_STAGE
    mirrors llm.max_workers and must not drift back to 64 without an
    intentional change to both files in lockstep."""
    from engine.progress import PARALLELISM_PER_STAGE
    assert PARALLELISM_PER_STAGE["vision"] == 16
    assert PARALLELISM_PER_STAGE["extract"] == 16
    assert PARALLELISM_PER_STAGE["entities"] == 16
    assert PARALLELISM_PER_STAGE["patterns"] == 16
    assert PARALLELISM_PER_STAGE["entities_dedupe"] == 1
    assert PARALLELISM_PER_STAGE["insights"] == 1
    assert PARALLELISM_PER_STAGE["actions"] == 1


# ── Bar behavior ───────────────────────────────────────────────────


def test_bar_position_equals_completed_over_total_no_inflight():
    """Bar is strict `completed / total` with no in-flight terms.
    1 of 10 done with no in-flight calls → bar = 0.1.
    """
    t = ProgressTracker()
    t.register_stage("extract", "m", 10)
    t.mark_stage_started("extract")
    t.record_call("extract", 30.0)  # 1 completed
    assert t.compute_bar_position() == pytest.approx(0.1, abs=1e-3)


def test_bar_position_ignores_in_flight_partial_credit():
    """In-flight calls do NOT contribute to the bar. The bar must
    exactly equal `completed_calls / total_estimated_calls` so what
    the user reads in the parens matches the percentage to the left.

    Issue #68: PR #55's round-4 partial-credit term made the bar
    diverge from the call ratio (e.g. `99% completed (228/236 calls)`
    where 228/236 = 96.6%). This test pins the new strict behavior:
    1 completed of 4, with 1 in-flight halfway through → bar = 0.25,
    NOT 0.375 as the partial-credit formula would give.
    """
    hist = {("extract", "m"): _no_tokens([30.0] * 5)}
    t = ProgressTracker(historical_durations=hist)
    t.register_stage("extract", "m", 4)
    t._stages["extract"].completed_calls = 1
    t.mark_stage_started("extract")
    import time as _time
    started = _time.monotonic() - 15.0  # 15s ago, halfway through 30s
    t._in_flight_per_stage.setdefault("extract", []).append((started, 30.0))
    bar = t.compute_bar_position()
    # Strict: 1 / 4 = 0.25. The in-flight 0.5 partial credit is gone.
    assert bar == pytest.approx(0.25, abs=1e-3)


def test_bar_position_in_flight_does_not_inflate_past_completed():
    """An in-flight call running past its expected duration must not
    contribute anything to the bar. Strict completed/total ignores
    the in-flight set entirely."""
    t = ProgressTracker()
    t.register_stage("extract", "m", 2)
    t.mark_stage_started("extract")
    import time as _time
    started = _time.monotonic() - 60.0  # way over expected 30s
    t._in_flight_per_stage.setdefault("extract", []).append((started, 30.0))
    bar = t.compute_bar_position()
    # Strict: 0 completed / 2 total = 0.0. The in-flight call no
    # longer contributes ~1.0 to the numerator.
    assert bar == pytest.approx(0.0, abs=1e-3)


def test_bar_strictly_equals_completed_over_total_with_many_in_flight():
    """Issue #68 regression test. Mirror of the screenshot scenario:
    100 completed + 5 in-flight calls each at 99% expected duration,
    total = 110. Bar must be 100/110 ≈ 0.909, NOT inflated by the
    in-flight partial credit.

    Pre-fix: bar = (100 + 5×0.99) / 110 ≈ 0.95 (then capped at 0.99).
    Post-fix: bar = 100 / 110 ≈ 0.909, exactly matching what the
    user reads as `(100 / 110 calls)` in the UI.
    """
    t = ProgressTracker()
    t.register_stage("extract", "m", 110)
    t.mark_stage_started("extract")
    t._stages["extract"].completed_calls = 100
    import time as _time
    now = _time.monotonic()
    # 5 in-flight calls, each at 99% of their expected 30s (29.7s
    # elapsed of 30s expected). Pre-fix this would add ~4.95 to the
    # numerator.
    for _ in range(5):
        t._in_flight_per_stage.setdefault("extract", []).append(
            (now - 29.7, 30.0))
    bar = t.compute_bar_position()
    # Strict: 100 / 110 = 0.9090...
    assert bar == pytest.approx(100 / 110, abs=1e-6)
    # And explicitly NOT inflated past the call ratio.
    assert bar < 0.92, (
        f"in-flight partial credit leaked into bar: got {bar:.4f}, "
        f"expected ~0.909"
    )


def test_bar_capped_at_99_with_999_of_1000_calls():
    """Even at 999/1000 the bar must stay ≤ 0.99 — only the runner's
    explicit done-event flips to 1.0. Catches the case where one
    stuck call makes the bar peg at 1.0 and look done."""
    t = ProgressTracker()
    t.register_stage("extract", "m", 1000)
    t._stages["extract"].completed_calls = 999
    bar = t.compute_bar_position()
    assert bar == pytest.approx(0.99, abs=1e-3)
    assert bar <= 0.99


def test_bar_no_jump_on_stage_transition():
    """Round 4 (4): regression for the stage-boundary jump. With the
    new call-completion bar, transitioning from stage 1 (all done) to
    stage 2 (about to start) should NOT change the bar value other
    than what the new stage's call-completion progress adds.

    Concretely: before transition, stage 1 done, stage 2 not started
    → bar = (s1.completed + 0) / (s1.est + s2.est).
    After transition, stage 1 finished, stage 2 marked started but no
    calls done yet → bar = same numerator / same denominator.
    No jump.
    """

    t = ProgressTracker()
    t.register_stage("extract", "m", 5)
    t.register_stage("entities", "m", 5)
    # All 5 extract done.
    t.mark_stage_started("extract")
    for _ in range(5):
        t.record_call("extract", 10.0)
    bar_before = t.compute_bar_position()
    # Transition: extract finished, entities started.
    t.mark_stage_finished("extract")
    t.mark_stage_started("entities")
    bar_after = t.compute_bar_position()
    # Bar should be identical (no completed calls in entities yet).
    assert bar_after == pytest.approx(bar_before, abs=1e-6), (
        f"stage transition caused bar to jump: before={bar_before:.3f} "
        f"after={bar_after:.3f}"
    )
    # Sanity: bar = 5/10 = 0.5.
    assert bar_before == pytest.approx(0.5, abs=1e-3)


def test_finished_stage_drops_to_actual_completed_count():
    """When a stage finishes with completed < est_calls (estimate
    was over), est_calls snaps down to actual. Past stages must
    contribute 0 to remaining; without this, the cumulative total
    stays inflated forever and bar % is dragged down.

    Regression for zpyv-class run: entities estimate = 4 calls but
    actual = 2. Without the snap, `Prioritizing actions · 17/20`
    appears at end of pipeline (18 actual, 2 phantom). With the
    snap, est tightens to 2 at entities → patterns transition,
    cumulative total drops to 18, bar reads 17/18 truthfully.
    """
    t = ProgressTracker()
    t.register_stage("entities", "m", 4)   # estimate
    t.register_stage("patterns", "m", 8)
    t.mark_stage_started("entities")
    # Actual: only 2 entities calls fire.
    t.record_call("entities", 10.0)
    t.record_call("entities", 10.0)
    # Total before stage finishes: still 12 (4 + 8).
    assert t.compute_pipeline_total_calls() == 12
    # Stage transition tightens entities to 2.
    t.mark_stage_finished("entities")
    t.mark_stage_started("patterns")
    assert t._stages["entities"].est_calls == 2
    assert t.compute_pipeline_total_calls() == 10  # 2 + 8


def test_remaining_excludes_past_stages():
    """The "remaining" calculation (compute_pipeline_total -
    compute_pipeline_completed) must equal current_stage_remaining +
    sum(future_stages.est_calls). Past finished stages should
    contribute exactly 0 to remaining — user instruction: "always
    compute remaining calls based on current and future stages, old
    ones should no longer be relevant".

    The snap fix (mark_stage_finished sets est_calls = completed)
    achieves this: for any finished stage, est == completed, so
    `est - completed = 0` and the stage drops out of the remaining
    sum entirely.
    """
    t = ProgressTracker()
    t.register_stage("a", "m", 5)
    t.register_stage("b", "m", 3)
    t.register_stage("c", "m", 2)
    t.mark_stage_started("a")
    # Stage a: estimate was 5, actual 4 (over-estimate by 1).
    for _ in range(4):
        t.record_call("a", 1.0)
    t.mark_stage_finished("a")
    # Past stage 'a' contributes 0 to remaining.
    assert t._stages["a"].est_calls - t._stages["a"].completed_calls == 0
    # Mid-pipeline: stage b in progress with 1 of 3 done.
    t.mark_stage_started("b")
    t.record_call("b", 1.0)
    total = t.compute_pipeline_total_calls()
    completed = sum(s.completed_calls for s in t._stages.values())
    remaining = total - completed
    # Remaining = b_remaining (3 - 1 = 2) + c_total (2) = 4.
    # Past stage 'a' irrelevant.
    assert remaining == 4
    # Verify by direct computation: only current + future scope.
    current_future_remaining = (
        (t._stages["b"].est_calls - t._stages["b"].completed_calls)
        + t._stages["c"].est_calls
    )
    assert remaining == current_future_remaining


def test_kernel_live_hook_credits_completed_calls(monkeypatch):
    """Regression: the snap (est_calls = completed_calls on finish) is
    correct AND relies on completed_calls being fed in real time. The legacy
    complete() wrapper bumped it on every success; when stage execution moved
    onto the kernel, the live-progress hook stayed cosmetic and the credit was
    dropped. completed_calls then stuck at 0, so mark_stage_finished snapped
    every finished stage to 0, it fell out of the denominator, and `total`
    decayed to bare remaining work — the chip read "99% (N / N)" and reset each
    stage (22/22 -> 11/11). The fix re-credits completed_calls in the hook.

    This drives the hook the way the kernel does and asserts (a) successes
    (incl. cache hits) credit, failures don't, and (b) the denominator stays a
    TRUE cumulative total (finished actuals + future estimates), not a
    remaining countdown. The older tests above feed completed_calls directly,
    so they never caught the missing hook credit — this one does.
    """
    from engine import runner
    from kernel.abstractions import LlmResponse
    from kernel.enums import LlmStatus, PhaseName

    # _emit() prints + reads runner globals; isolate the test to the credit.
    monkeypatch.setattr(runner, "_emit", lambda: None)

    class _Phase:
        def __init__(self, pn):
            self._pn = pn

        def name(self):
            return self._pn

    class _Env:
        def __init__(self, pn):
            self.phase = _Phase(pn)

    def ok():
        # status None + no exception == success (matches the scripted provider).
        return LlmResponse(None, "x", None, 3, 5, 0, 0.02, 0.1)

    def load_fail():
        return LlmResponse.from_status(LlmStatus.LOAD, 0.1)

    hook = runner._KernelLiveProgressHook()
    t = ProgressTracker()
    # Cold pipeline estimate: all stages registered up front, over-estimated
    # (entities 14 / patterns 13 vs actuals 5 / — the real ffqj shape).
    t.register_stage("extract", "m", 10)
    t.register_stage("entities", "m", 14)
    t.register_stage("patterns", "m", 13)
    t.register_stage("insights", "m", 1)
    monkeypatch.setattr(runner, "_progress_tracker", t)

    extract_env = _Env(PhaseName.EXTRACTION_LLM)
    # 6 productive extract leaves + 3 LOAD failures: only successes credit.
    for _ in range(6):
        hook.hook_llm_completed(None, extract_env, ok(), None, False, False)
    for _ in range(3):
        hook.hook_llm_completed(None, extract_env, load_fail(), None, False, False)
    # A cache hit is completed work too (served leaf), and must not decrement
    # in-flight below zero — exercise that path.
    hook.hook_llm_completed(None, extract_env, ok(), None, True, False)
    assert t._stages["extract"].completed_calls == 7   # 6 + 1 cache, 0 failures

    t.mark_stage_finished("extract")                   # snaps est 10 -> 7
    ent_env = _Env(PhaseName.ENTITY_GROUPING)           # ENTITY_* -> "entities"
    for _ in range(5):
        hook.hook_llm_completed(None, ent_env, ok(), None, False, False)
    t.mark_stage_finished("entities")                  # snaps est 14 -> 5

    # THE REGRESSION: pre-fix both finished stages snapped to 0 and the total
    # collapsed to patterns + insights = 13 + 1 = 14 (a remaining countdown).
    # With the credit it is the true cumulative total: 7 + 5 + 13 + 1 = 26.
    assert t.compute_pipeline_total_calls() == 7 + 5 + 13 + 1


def test_full_lifecycle_overestimating_then_underestimating_stages():
    """End-to-end lifecycle: stages with mixed over/under estimates
    all settle to actual on finish. The pipeline total at any point
    reflects only what's actually still ahead.
    """
    t = ProgressTracker()
    # Three stages, initial estimates: 5, 3, 2 (total 10).
    t.register_stage("a", "m", 5)
    t.register_stage("b", "m", 3)
    t.register_stage("c", "m", 2)

    # Stage a: over-estimate. Actual 3 calls.
    t.mark_stage_started("a")
    for _ in range(3):
        t.record_call("a", 1.0)
    t.mark_stage_finished("a")
    # After finish: total drops by 2 (5 → 3). Pipeline total: 3+3+2 = 8.
    assert t.compute_pipeline_total_calls() == 8

    # Stage b: under-estimate. Actual 5 calls. (The runner's
    # _reemit_total can grow est mid-stage; here we simulate by
    # re-registering with a higher count.)
    t.mark_stage_started("b")
    for _ in range(3):
        t.record_call("b", 1.0)
    # Mid-stage: completed=3, est=3, but two more calls land
    # before stage finishes — register_stage(b, m, 5) widens.
    t.register_stage("b", "m", 5)
    for _ in range(2):
        t.record_call("b", 1.0)
    t.mark_stage_finished("b")
    # After b finishes: completed=5, est snaps to 5. Total: 3+5+2 = 10.
    assert t._stages["b"].est_calls == 5
    assert t.compute_pipeline_total_calls() == 10

    # Stage c: estimate matches actual.
    t.mark_stage_started("c")
    for _ in range(2):
        t.record_call("c", 1.0)
    t.mark_stage_finished("c")
    # All stages done. Total = completed = 10. Bar caps at 0.99.
    assert t.compute_pipeline_total_calls() == 10
    assert sum(s.completed_calls for s in t._stages.values()) == 10
    bar = t.compute_bar_position()
    # 10/10 = 1.0 raw → capped to 0.99 (only the runner's done emit
    # flips to 1.0).
    assert bar == pytest.approx(0.99, abs=1e-3)


def test_finished_stage_with_zero_completions_drops_to_zero():
    """A stage that registered with est_calls=N but got skipped
    entirely (completed=0) snaps to 0 on finish, contributing
    nothing to the cumulative total. Edge case: an optional stage
    that gets bypassed (e.g. dedupe with no entities to dedupe).
    """
    t = ProgressTracker()
    t.register_stage("dedupe", "m", 1)  # registered with est=1
    t.register_stage("patterns", "m", 5)
    # Dedupe never started — but mark_finished can still fire on
    # cleanup paths. With completed=0, est snaps to 0.
    t.mark_stage_finished("dedupe")
    assert t._stages["dedupe"].est_calls == 0
    # Total only counts patterns now.
    assert t.compute_pipeline_total_calls() == 5


def test_bar_can_decrease_when_total_grows():
    """If a later stage's est_calls is refined upward mid-run, the
    denominator grows and the bar drops. Round 4 PIVOT: no monotonic
    ratchet — the bar reflects the current best estimate honestly.
    """
    t = ProgressTracker()
    t.register_stage("extract", "m", 5)
    t.register_stage("entities", "m", 5)
    t.mark_stage_started("extract")
    for _ in range(3):
        t.record_call("extract", 10.0)
    bar1 = t.compute_bar_position()
    # 3/10 = 0.30
    assert bar1 == pytest.approx(0.30, abs=1e-3)
    # entities estimate revised upward to 15.
    t.register_stage("entities", "m", 15)
    bar2 = t.compute_bar_position()
    # 3 / (5 + 15) = 0.15
    assert bar2 == pytest.approx(0.15, abs=1e-3)
    assert bar2 < bar1


def test_bar_position_capped_at_99_until_explicit_finish():
    """Single-call stage, all done — bar caps at 0.99 not 1.0. The
    runner's explicit done-event flips to 1.0; until then the UI stays
    just under the rail."""
    t = ProgressTracker()
    t.register_stage("extract", "m", 1)
    t.mark_stage_started("extract")
    t.record_call("extract", 60.0)
    t.mark_stage_finished("extract")
    pos = t.compute_bar_position()
    assert pos == pytest.approx(0.99, abs=1e-3)
    assert pos <= 0.99


# ── Historical loader ──────────────────────────────────────────────


def test_load_historical_durations_returns_dur_token_tuples(tmp_path):
    """Loader returns (duration_s, completion_tokens) tuples per
    (stage, model). Tokens default to 0 when the field is absent."""
    jsonl = tmp_path / "llm-calls.jsonl"
    events = [
        {"event": "begin", "call_id": "0001",
         "stage": "extract", "model": "m"},
        {"event": "end", "call_id": "0001", "success": True,
         "duration_ms": 60_000, "completion_tokens": 1500},
        {"event": "begin", "call_id": "0002",
         "stage": "extract", "model": "m"},
        {"event": "end", "call_id": "0002", "success": True,
         "duration_ms": 80_000, "completion_tokens": 2500},
    ]
    jsonl.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    out = load_historical_durations(logs_root=tmp_path)
    assert out == {("extract", "m"): [(60.0, 1500), (80.0, 2500)]}


def test_load_historical_durations_skips_failed_calls(tmp_path):
    """Failed end events (success=false) must be dropped."""
    jsonl = tmp_path / "llm-calls.jsonl"
    events = [
        {"event": "begin", "call_id": "0001",
         "stage": "extract", "model": "m"},
        {"event": "end", "call_id": "0001", "success": True,
         "duration_ms": 60_000, "completion_tokens": 1000},
        {"event": "begin", "call_id": "0002",
         "stage": "extract", "model": "m"},
        {"event": "end", "call_id": "0002", "success": False,
         "duration_ms": 50},
        {"event": "begin", "call_id": "0003",
         "stage": "extract", "model": "m"},
        {"event": "end", "call_id": "0003", "success": True,
         "duration_ms": 70_000, "completion_tokens": 1200},
    ]
    jsonl.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    out = load_historical_durations(logs_root=tmp_path)
    assert out == {("extract", "m"): [(60.0, 1000), (70.0, 1200)]}


def test_load_historical_durations_skips_in_flight_begin(tmp_path):
    jsonl = tmp_path / "llm-calls.jsonl"
    events = [
        {"event": "begin", "call_id": "0001",
         "stage": "extract", "model": "m"},
        {"event": "end", "call_id": "0001", "success": True,
         "duration_ms": 60_000, "completion_tokens": 1000},
        {"event": "begin", "call_id": "0002",
         "stage": "extract", "model": "m"},
    ]
    jsonl.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    out = load_historical_durations(logs_root=tmp_path)
    assert out == {("extract", "m"): [(60.0, 1000)]}


def test_load_historical_durations_respects_window(tmp_path):
    import os
    for i, ts in enumerate([100, 200, 300]):
        d = tmp_path / f"run_{i}"
        d.mkdir()
        jp = d / "llm-calls.jsonl"
        ev = [
            {"event": "begin", "call_id": "0001",
             "stage": "extract", "model": "m"},
            {"event": "end", "call_id": "0001", "success": True,
             "duration_ms": (i + 1) * 1000,
             "completion_tokens": (i + 1) * 100},
        ]
        jp.write_text("\n".join(json.dumps(e) for e in ev) + "\n")
        os.utime(jp, (ts, ts))

    out = load_historical_durations(logs_root=tmp_path, window_runs=2)
    samples = sorted(out[("extract", "m")])
    assert samples == [(2.0, 200), (3.0, 300)]


def test_load_historical_durations_handles_missing_completion_tokens(tmp_path):
    """Older logs (pre PR #36) may omit completion_tokens. Loader
    treats them as 0 — still loaded, just contributes nothing to
    the sec/token coefficient."""
    jsonl = tmp_path / "llm-calls.jsonl"
    events = [
        {"event": "begin", "call_id": "0001",
         "stage": "extract", "model": "m"},
        {"event": "end", "call_id": "0001", "success": True,
         "duration_ms": 60_000},
        # No completion_tokens.
    ]
    jsonl.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    out = load_historical_durations(logs_root=tmp_path)
    assert out == {("extract", "m"): [(60.0, 0)]}


def test_snapshot_picks_user_visible_stage_via_runner():
    t = ProgressTracker()
    t.register_stage("metadata", "m", 1)
    t.register_stage("extract", "m", 5)
    t.mark_stage_started("extract")
    t.record_call("metadata", 12.5)
    snap = t.snapshot()
    assert snap["stage"] == "metadata"


def test_total_eta_drops_as_calls_complete():
    hist = {("extract", "m"): _no_tokens([60.0] * 5)}
    t = ProgressTracker(historical_durations=hist, parallelism_per_stage=SERIAL)
    t.register_stage("extract", "m", 4)
    eta_initial = t.compute_total_eta_seconds()
    t.mark_stage_started("extract")
    t.record_call("extract", 60.0)
    eta_after_one = t.compute_total_eta_seconds()
    assert eta_after_one < eta_initial
    assert eta_after_one == pytest.approx(eta_initial * 0.75, rel=0.2)


# ── Runtime invariant: end-of-run total == actual completed (issue #244)
#
# The progress chip's denominator is `sum(s.est_calls for s in stages)`.
# At end of run, this MUST equal the sum of completed calls across
# stages — otherwise the chip stalls at e.g. `19/25` despite all 19 of
# the 19-actual calls having landed. Pre-fix, `mark_stage_finished`
# alone tightened est down to actual. The runner's close-loop in
# `_record_stage_started` skips stages whose `started_at is None`,
# leaving over-estimated stages that were registered but never
# entered (no calls fired) at their estimate forever.


def test_end_of_run_total_matches_actuals_full_pipeline():
    """Real-run shape from issue #244 (run 2026-05-10T09-45-18Z-4k2e).
    Post-ingest est = 37; actual = 20. After every stage transition +
    end-of-run close, total MUST equal 20."""
    t = ProgressTracker(parallelism_per_stage=SERIAL)
    # Post-ingest registration with the over-estimates from #244.
    plan = {
        "vision": (5, 5),
        "extract": (7, 7),
        "entities": (7, 1),       # est 7 vs actual 1 — primary culprit
        "entities_dedupe": (1, 1),
        "patterns": (15, 4),      # est 15 vs actual 4
        "insights": (1, 1),
        "actions": (1, 1),
    }
    for stage, (est, _) in plan.items():
        t.register_stage(stage, "m", est)
    assert t.compute_pipeline_total_calls() == sum(e for e, _ in plan.values())

    # Walk the stages in order: start, fire `actual` calls, transition.
    # The runner closes prior stages at the start of the next one
    # (defensive close-loop in _record_stage_started). On the LAST stage
    # the runner's end-of-run loop closes whatever's still open.
    order = list(plan)
    for i, stage in enumerate(order):
        t.mark_stage_started(stage)
        actual = plan[stage][1]
        for _ in range(actual):
            t.record_call(stage, 1.0)
        # Simulate _record_stage_started's close-loop on the next stage.
        if i + 1 < len(order):
            for s, st in list(t._stages.items()):
                if (s != order[i + 1]
                        and st.started_at is not None
                        and st.finished_at is None):
                    t.mark_stage_finished(s)
        else:
            # End-of-run close (runner.py:6024).
            for s, st in list(t._stages.items()):
                if st.started_at is not None and st.finished_at is None:
                    t.mark_stage_finished(s)

    expected = sum(a for _, a in plan.values())
    actual_total = t.compute_pipeline_total_calls()
    assert actual_total == expected, (
        f"end-of-run total {actual_total} != actual completed {expected}; "
        f"per-stage est: "
        f"{[(s, t._stages[s].est_calls, t._stages[s].completed_calls) for s in order]}"
    )


def test_unstarted_overestimated_stage_tightens_to_zero():
    """A stage that's registered but never started must collapse to
    zero at end of run. Issue #244: the prior end-of-run loop in
    runner.py was gated on `started_at is not None`, so an unentered
    stage's pre-estimate leaked into the chip's denominator forever
    and the bar could never reach 100%. Post-fix the close-loop
    iterates ALL registered stages."""
    t = ProgressTracker(parallelism_per_stage=SERIAL)
    t.register_stage("entities", "m", 5)
    # Stage was estimated but the run never actually entered it
    # (e.g. branch skipped; no calls). End-of-run close iterates every
    # registered stage with `finished_at is None` — INCLUDING
    # never-started ones — and tightens to completed_calls (0 here).
    for s, st in list(t._stages.items()):
        if st.finished_at is None:
            t.mark_stage_finished(s)
    assert t.compute_pipeline_total_calls() == 0, (
        f"unstarted stage's est={t._stages['entities'].est_calls} "
        f"leaked into pipeline total")


def test_per_call_seconds_at_preflight_falls_back_when_no_history():
    """No historicals → returns the FALLBACK constant. Same behavior
    as the legacy preflight computation for fresh installs."""
    assert per_call_seconds_at_preflight(
        "extract", "any-model", None,
    ) == FALLBACK_SECONDS_PER_CALL["extract"]
    assert per_call_seconds_at_preflight(
        "extract", "any-model", {},
    ) == FALLBACK_SECONDS_PER_CALL["extract"]


def test_per_call_seconds_at_preflight_uses_historical_decomposition():
    """With historicals, the preflight per-call value matches the live
    tracker's first-emit computation: hist_call_fixed +
    hist_per_token × hist_tokens. Issue #394: pre-fix preflight read
    a token-blind FALLBACK and the tracker read this token-aware
    formula — the two diverged by 7× on TEE-mix kimi-k2-6 runs."""
    # Synthetic historical: every call takes ~20s and produces 1000
    # tokens. Pure-rate decomposition: rate ≈ 0.02 s/token, fixed ≈ 0.
    # Predicted per_call = 0 + 0.02 × 1000 = 20s — matches median dur.
    hist = {("extract", "m"): [(20.0, 1000)] * 5}
    per_call = per_call_seconds_at_preflight("extract", "m", hist)
    assert 19.0 < per_call < 21.0, per_call

    # And the same number the tracker would compute at first emit.
    t = ProgressTracker(historical_durations=hist)
    t.register_stage("extract", "m", 1)
    tracker_per_call = t._per_call_seconds_locked(
        "extract", "m", [], None)
    assert abs(per_call - tracker_per_call) < 0.01


def test_per_call_seconds_at_preflight_diverges_from_fallback_for_token_heavy_models():
    """The bug shape from #394: a model with historical that produces
    much more (or less) per-call work than FALLBACK encodes — the
    preflight value via this helper reflects reality, while the
    legacy FALLBACK-only value misses by a wide margin."""
    # Heavy-output historical (kimi-k2-6 style: ~5000 tokens per dense
    # extract call). Rate ≈ 0.02 s/tok → per_call ≈ 100s. FALLBACK is
    # 68s. Helper picks 100s; legacy preflight would have said 68s.
    heavy = {("extract", "kimi"): [(100.0, 5000)] * 5}
    helper_value = per_call_seconds_at_preflight("extract", "kimi", heavy)
    fallback_value = FALLBACK_SECONDS_PER_CALL["extract"]
    assert helper_value > fallback_value, (
        f"heavy-output model should produce a HIGHER preflight per-call "
        f"than FALLBACK; got helper={helper_value}, fallback={fallback_value}")


def test_vision_recording_path_does_not_double_count():
    """Issue #244 root cause: pre-#216 vision bypassed the chat
    wrapper, so the runner had a post-ingest sweep that retroactively
    fed each successful vision stat record into the tracker via
    `record_call("vision", ...)`. Post-#216 vision routes through
    `complete()` and the wrapper already calls `record_call_begin/end`
    in real time. Re-applying the sweep on top of the wrapper's
    real-time bookkeeping double-counted vision: 5 actual calls
    landed `completed_calls=10`, then `mark_stage_finished` froze
    `est_calls=10`, leaking 5 calls into the chip's denominator
    (run 2026-05-10T09-45-18Z-4k2e showed `total=25` for 20 actual
    calls). Test pins the invariant: a single record_call per vision
    success → completed_calls == actual successes."""
    t = ProgressTracker(parallelism_per_stage=SERIAL)
    t.register_stage("vision", "m", 5)
    t.mark_stage_started("vision")
    # Real-time bookkeeping: wrapper calls record_call_begin/end on
    # each vision success.
    for _ in range(5):
        tk = t.record_call_begin("vision")
        t.record_call_end("vision", tk, 1.0, 100, success=True)
    assert t._stages["vision"].completed_calls == 5
    # Mark finished — est_calls should tighten to the real-time
    # completed count, NOT the doubled count from a redundant sweep.
    t.mark_stage_finished("vision")
    assert t._stages["vision"].est_calls == 5, (
        f"vision est={t._stages['vision'].est_calls} — likely "
        f"double-counted (real-time + post-hoc sweep)")


def test_mid_run_estimate_revision_then_close_tightens_to_completed():
    """`_reemit_total(remaining)` mid-stage sets est = completed +
    remaining. If `remaining` overshoots actual (e.g. estimator says
    2 more, but only 1 fires), `mark_stage_finished` must close the
    gap — est should equal completed after close, not the over-
    estimate."""
    t = ProgressTracker(parallelism_per_stage=SERIAL)
    t.register_stage("entities", "m", 7)
    t.mark_stage_started("entities")
    # Mid-stage revision: estimator now thinks 2 more calls remain.
    t.register_stage("entities", "m", 0 + 2)
    assert t._stages["entities"].est_calls == 2
    # Only 1 call actually lands.
    t.record_call("entities", 1.0)
    assert t._stages["entities"].completed_calls == 1
    # Stage transition fires close.
    t.mark_stage_finished("entities")
    assert t._stages["entities"].est_calls == 1, (
        f"mark_stage_finished left est={t._stages['entities'].est_calls} "
        f"instead of tightening to completed=1")
