"""Per-call outcome classifier tests.

`_classify_outcome` is the single source of truth for the user-visible
label rendered in the per-call table's `outcome` column. Failure
labels carry a parenthetical retry-strategy suffix per the taxonomy
(load / sizing / other); when a more specific name is available —
`cap_hit`, `timeout`, `parse_error`, `empty_response`, `interrupted` —
that name replaces 'failed' but keeps the suffix.

Sizing-vs-load on tokens-flowed failures is decided by the
TTFT-fraction rule: TTFT < 50% of duration → sizing (model warmed up
but choked partway), else → load.

Run with:
    cd engine && pytest tests/test_per_call_outcomes.py -v
"""
from __future__ import annotations



from engine.runner import (
    OUTCOME_CAP_HIT,
    OUTCOME_EMPTY_RESPONSE,
    OUTCOME_FAILED_LOAD,
    OUTCOME_ABORTED,
    OUTCOME_FAILED_OTHER,
    OUTCOME_FAILED_SIZING,
    OUTCOME_INTERRUPTED_LOAD,
    OUTCOME_INTERRUPTED_SIZING,
    OUTCOME_PARSE_ERROR,
    OUTCOME_SUCCESS,
    OUTCOME_SUCCESS_EMPTY,
    OUTCOME_SUCCESS_SAMPLED,
    OUTCOME_TIMEOUT_LOAD,
    OUTCOME_TIMEOUT_SIZING,
    _classify_outcome,
    _is_timeout_error,
    _leaf_aware_warnings,
)


def _rec(**overrides):
    """Minimal stat record dict with sane defaults; tests override
    only the fields they care about. Defaults model a tokens-flowed
    call where TTFT (100ms) is ~2% of duration (5000ms) — well below
    the 50% threshold, so the discriminator routes it to sizing."""
    base = {
        "success": True,
        "error": None,
        "aborted": False,
        "parse_error": False,
        "empty_response": False,
        "output": None,
        "input": None,
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "duration_ms": 5000,
        "ttft_ms": 100,
    }
    base.update(overrides)
    return base


# ── Success-band outcomes ──────────────────────────────────────────────────

class TestSuccessOutcomes:
    def test_success_with_kept_items(self):
        assert _classify_outcome(_rec(output={"facts": 5})) == OUTCOME_SUCCESS

    def test_success_when_no_output_field(self):
        # Stages that don't emit per-call counts default to success
        # (no signal to label otherwise).
        assert _classify_outcome(_rec(output=None)) == OUTCOME_SUCCESS

    def test_success_empty_when_output_is_zero_items(self):
        # Output structure exists but everything is zero. No reasoning
        # tokens spent — model thought briefly and decided nothing
        # applied. Yellow caveat band, data-complete.
        assert (
            _classify_outcome(_rec(output={"facts": 0}))
            == OUTCOME_SUCCESS_EMPTY
        )

    def test_kept_zero_with_reasoning_tokens_stays_success_empty(self):
        # Zero-content-token failure shapes (empty_response /
        # interrupted / parse_error) are labeled separately. A clean
        # valid `[]` after the model spent reasoning tokens is NOT
        # one of those — model thought, then decided nothing applied,
        # which is a legitimate answer for near-empty inputs (e.g.
        # OCR'd images with 5 chars of text).
        assert (
            _classify_outcome(_rec(
                output={"facts": 0},
                reasoning_tokens=420,
            ))
            == OUTCOME_SUCCESS_EMPTY
        )

    def test_kept_zero_no_reasoning_stays_success_empty(self):
        # Same outcome as the reasoning-tokens variant above —
        # `success_empty` covers any clean valid empty payload
        # regardless of upstream compute spent.
        assert (
            _classify_outcome(_rec(
                output={"facts": 0},
                reasoning_tokens=0,
            ))
            == OUTCOME_SUCCESS_EMPTY
        )

    def test_success_empty_for_insights_shape_with_caps(self):
        # The insights stage records the primary entry counts FIRST
        # in its output dict, then auxiliary fields including the
        # capacity limits the prompt was constructed under. A clean
        # empty result (0 entries across the three buckets) must
        # classify as `success_empty` even though the cap fields are
        # non-zero ints. Anchored on the live m7pp insights call
        # shape: 0 entries returned, caps 10/6/4.
        assert (
            _classify_outcome(_rec(output={
                "insights": 0,
                "cross_domain": 0,
                "critical": 0,
                "kinds": {},
                "total_cap": 10,
                "cross_cap": 6,
                "critical_cap": 4,
            }))
            == OUTCOME_SUCCESS_EMPTY
        )

    def test_success_empty_for_actions_shape_with_cap(self):
        # Same convention for actions: primary count first, cap field
        # afterward. Clean empty result must classify as `success_empty`.
        assert (
            _classify_outcome(_rec(output={
                "actions": 0,
                "kinds": {},
                "max_actions_cap": 8,
                "recommendation_lengths": {},
            }))
            == OUTCOME_SUCCESS_EMPTY
        )

    def test_success_when_insights_returned_with_caps_present(self):
        # Counter-check: a non-zero primary count still classifies as
        # clean success even though cap fields sit beside it. Guards
        # against an over-eager "ignore caps" fix that would also
        # ignore the primary count.
        assert (
            _classify_outcome(_rec(output={
                "insights": 7,
                "cross_domain": 4,
                "critical": 3,
                "kinds": {"opportunity": 4, "risk": 3},
                "total_cap": 50,
                "cross_cap": 30,
                "critical_cap": 20,
            }))
            == OUTCOME_SUCCESS
        )


# ── Failure label format: <name> (<class>) ────────────────────────────────

class TestFailureLabels:
    def test_cap_hit_sizing_from_synthetic_exception(self):
        # `_CapHitResponse` is the wrapper-internal exception raised
        # on success+finish_reason=length. Always classifies as
        # sizing (cap-hits are output-cap signals).
        rec = _rec(
            success=False,
            error={"class": "retry._CapHitResponse",
                   "message": "cap-hit response"},
        )
        assert _classify_outcome(rec) == OUTCOME_CAP_HIT
        assert OUTCOME_CAP_HIT == "cap_hit (sizing)"

    def test_cap_hit_sizing_from_finish_reason(self):
        # Defensive: a stage path that didn't go through the wrapper
        # still gets cap_hit labeling on finish_reason=length.
        rec = _rec(success=False, error={"class": "X", "message": ""},
                   finish_reason="length")
        assert _classify_outcome(rec) == OUTCOME_CAP_HIT


    def test_timeout_load_when_ttft_above_half_duration(self):
        # Tokens flowed late / cut came near the end — overload, not
        # size. TTFT 4000ms of 5000ms duration = 80% → load.
        rec = _rec(
            success=False,
            ttft_ms=4000,
            duration_ms=5000,
            error={"class": "openai.APITimeoutError",
                   "message": "Request timed out."},
        )
        assert _classify_outcome(rec) == OUTCOME_TIMEOUT_LOAD
        assert OUTCOME_TIMEOUT_LOAD == "timeout (load)"

    def test_timeout_load_when_ttft_null(self):
        # Provider never emitted a token → silent vLLM interrupt →
        # load (no work to reduce).
        rec = _rec(
            success=False,
            ttft_ms=None,
            error={"class": "openai.APITimeoutError",
                   "message": "Request timed out."},
        )
        assert _classify_outcome(rec) == OUTCOME_TIMEOUT_LOAD

    def test_parse_error_sizing(self):
        rec = _rec(
            success=False,
            parse_error=True,
            error={"class": "retry._ParseError",
                   "message": "parse_error (stage=extract)"},
        )
        assert _classify_outcome(rec) == OUTCOME_PARSE_ERROR
        assert OUTCOME_PARSE_ERROR == "parse_error (sizing)"

    def test_empty_response_load(self):
        # Zero content tokens — the model didn't produce anything.
        # Load (no size reduction would help).
        rec = _rec(
            success=False,
            empty_response=True,
            error={"class": "retry._EmptyResponse",
                   "message": "empty_response (stage=extract)"},
        )
        assert _classify_outcome(rec) == OUTCOME_EMPTY_RESPONSE
        assert OUTCOME_EMPTY_RESPONSE == "empty_response (load)"

    def test_interrupted_low_ttft_is_load(self):
        # Interrupted is application-level "stream cut with tokens"
        # — same spec category as transport-level
        # RemoteProtocolError. Both route to Load unconditionally;
        # the TTFT discriminator does not apply.
        rec = _rec(
            success=False,
            interrupted=True,
            ttft_ms=200,
            duration_ms=5000,
            error={"class": "retry._InterruptedResponse",
                   "message": "interrupted (stage=extract)"},
        )
        assert _classify_outcome(rec) == OUTCOME_INTERRUPTED_LOAD

    def test_interrupted_high_ttft_is_load(self):
        # Pinned for symmetry with the low-TTFT case above.
        rec = _rec(
            success=False,
            interrupted=True,
            ttft_ms=4000,
            duration_ms=5000,
            error={"class": "retry._InterruptedResponse",
                   "message": "interrupted (stage=extract)"},
        )
        assert _classify_outcome(rec) == OUTCOME_INTERRUPTED_LOAD



    def test_failed_other_for_unknown_class(self):
        # Custom class not in the retriable set → other.
        rec = _rec(
            success=False,
            error={"class": "custom.RandomError",
                   "message": "weird thing"},
        )
        assert _classify_outcome(rec) == OUTCOME_FAILED_OTHER



    def test_aborted_classifies_as_aborted_bucket(self):
        # Aborted (begin without end on a wound-down run) gets its own
        # neutral bucket — the call didn't fail, the run wound down.
        # Pre-rename this collapsed to OUTCOME_FAILED_OTHER with a
        # synthesized "Cancelled" error class; promoting it to a
        # distinct outcome matches the Rust live materializer and the
        # JSX OUTCOME_LABELS surface.
        rec = _rec(success=False, aborted=True, error=None)
        assert _classify_outcome(rec) == OUTCOME_ABORTED


# ── Failure-class mapping (helper) ─────────────────────────────────────────



# ── _leaf_aware_warnings prefix bucketing ──────────────────────────────────

class TestLeafAwareWarnings:
    def test_buckets_by_prefix_across_classifications(self):
        # `cap_hit (sizing)` counts toward cap_hits. Both
        # `timeout (sizing)` and `timeout (load)` count toward
        # timeouts (single bucket regardless of strategy suffix).
        # The `failed (X)` flavors aggregate into one `failed` bucket.
        from engine.runner import (
            OUTCOME_FAILED_OTHER,
            OUTCOME_SUCCESS_REASONING_OFF,
        )
        leaf_outcomes = {
            OUTCOME_CAP_HIT: 3,
            OUTCOME_TIMEOUT_SIZING: 4,
            OUTCOME_TIMEOUT_LOAD: 2,
            OUTCOME_PARSE_ERROR: 5,
            OUTCOME_EMPTY_RESPONSE: 1,
            OUTCOME_INTERRUPTED_SIZING: 2,
            OUTCOME_INTERRUPTED_LOAD: 3,
            OUTCOME_SUCCESS_EMPTY: 7,
            OUTCOME_SUCCESS_SAMPLED: 6,
            OUTCOME_SUCCESS_REASONING_OFF: 8,
            OUTCOME_FAILED_LOAD: 9,
            OUTCOME_FAILED_SIZING: 10,
            OUTCOME_FAILED_OTHER: 12,
        }
        warnings = _leaf_aware_warnings(leaf_outcomes)
        assert warnings["cap_hits"] == 3
        assert warnings["timeouts"] == 6  # 4 + 2
        assert warnings["parse_errors"] == 5
        assert warnings["empty_responses"] == 1
        assert warnings["interrupted"] == 5  # 2 + 3
        assert warnings["success_empty"] == 7
        assert warnings["sampled"] == 6
        assert warnings["reasoning_off"] == 8
        assert warnings["failed"] == 9 + 10 + 12  # all three classes

    def test_input_overflows_pass_through(self):
        # input_overflows isn't a per-call outcome — passed in by
        # the rollup as `leaf_input_overflows`.
        warnings = _leaf_aware_warnings({}, leaf_input_overflows=4)
        assert warnings["input_overflows"] == 4
        assert warnings["cap_hits"] == 0


# ── _is_timeout_error (legacy helper kept for the dict-based path) ─────────

class TestIsTimeoutError:
    def test_class_name_match(self):
        assert _is_timeout_error({"class": "openai.APITimeoutError",
                                  "message": ""})

    def test_message_contains_timed_out(self):
        assert _is_timeout_error({"class": "X", "message": "Timed out."})

    def test_no_match_for_non_timeout(self):
        assert not _is_timeout_error({"class": "openai.RateLimitError",
                                      "message": "rate"})

    def test_none_returns_false(self):
        assert not _is_timeout_error(None)
        assert not _is_timeout_error({})


# ── classifier (retry._classify_failure) ──────────────────────────────────
#
# Direct unit tests of `retry._classify_failure` cover the live-exception
# entry point that the scheduler thunks call. `TestFailureClassForLabel`
# above pins the on-disk-record replay side; this section pins the
# in-process exception side, which can have richer state (`exc.__class__`,
# `__module__`, `ttft_ms` + `duration_s` kwargs) than the serialized
# `error.class` string the label mapper sees.




def _make_exc(module: str, name: str, message: str = "synthetic") -> Exception:
    """Synthesize an exception whose `__module__.__qualname__` matches
    `module.name` so retry.py's string-based dispatch matches without
    needing the real library installed."""
    cls = type(name, (Exception,), {})
    cls.__module__ = module
    return cls(message)


def _make_openai_exc(name: str, status_code: int = None,
                     message: str = "synthetic"):
    """openai-shaped exception with optional status_code so the
    classifier's APIStatusError branch can read it."""
    cls = type(name, (Exception,), {})
    cls.__module__ = "openai"
    exc = cls(message)
    if status_code is not None:
        exc.status_code = status_code
    return exc




# ── _exception_dict traceback capture (issue #360) ─────────────────────────

class TestExceptionDictTraceback:
    """Pre-#360 `_exception_dict` called `traceback.format_exc()` from
    outside the originating `except` block, so `sys.exc_info()` was
    `(None, None, None)` and every failure record on disk shipped
    `"NoneType: None"` for the traceback field. The fix walks
    `exc.__traceback__` directly, which the exception carries even
    after the `except` block has exited."""





# ── Classifier parity: live vs replay share the same bucket core ───────────


