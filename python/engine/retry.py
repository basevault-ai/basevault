"""Synthetic retry-class exceptions raised by the LLM wrapper.

The surviving half of the old retry module after the cutover deleted the
legacy scheduler and its degrading-retry classifier. What remains here:

* The synthetic exceptions the LLM wrapper raises so failures classify
  correctly — ``_CapHitResponse`` (truncated ``finish_reason="length"``),
  ``_WallClockTimeout`` (per-call wall-clock ceiling), ``_SkippedByUser``.
* A re-export of the parse-signal exceptions (``_ParseError``,
  ``_EmptyResponse``, ``_SuccessEmpty``, ``_InterruptedResponse``,
  ``_PostStreamFailure``) whose canonical home is ``parse_signals``, kept so
  existing ``from retry import …`` sites keep resolving.

Retry policy itself (load / sizing / sample ladders) now lives in the kernel;
the classifier + decision functions that used to live here were deleted with
the legacy scheduler. Error serialization (``_exception_dict``) moved to
``parse_signals`` so it could outlive this module's deletion.
"""
from __future__ import annotations

# ── Synthetic retry-class exceptions ─────────────────────────────────────────


class _CapHitResponse(Exception):
    """Synthetic exception raised by the wrapper after `complete()`
    returns with `finish_reason == "length"`. The output is truncated
    and unusable, so the wrapper raises this; the classifier routes
    `_CapHitResponse` unconditionally to "sizing" and the stage
    thunk's `_decide_retry` dispatches the cascade (halve for
    extract / entities-summarize; sample-N for synthesis stages).

    Carries the raw response so a give_up can return it to the
    caller if the surrounding stage prefers to consume the truncated
    output rather than treat the call as failed.

    Caught by stage thunks via the classifier — never special-cased
    by stage code directly."""

    def __init__(self, finish_reason: str = "length", raw: str | None = None):
        super().__init__(f"cap-hit response (finish_reason={finish_reason})")
        self.finish_reason = finish_reason
        self.raw = raw


class _WallClockTimeout(Exception):
    """Synthetic exception raised by `llm._consume_chat_stream` (and the
    local-provider loops) when a single call's elapsed time exceeds its
    per-call wall-clock ceiling. Reuses the user-skip socket-shutdown
    abort, armed by a per-call watchdog timer + a per-chunk elapsed check
    instead of a user-skip trigger.

    Distinct from the SDK's per-stage `timeout=` kwarg: on a streaming
    response that value is an httpx PER-READ deadline (max wait BETWEEN
    byte chunks), so a reasoning-on model trickling `reasoning_content`
    for hours resets it every chunk and it never fires (observed: a
    single entities_dedupe call ran 152 min and sat `pending` for 2.5h).
    This bounds elapsed time regardless of byte flow.

    Routes UNCONDITIONALLY to "load" (bypasses the TTFT-fraction
    discriminator): a wall-clock timeout is treated as a transient
    slow-enclave condition and retried as-is — same input, same model.

    The message carries the "timeout" token so `runner._is_timeout_error`
    surfaces it as the "timeout (load)" outcome label, and so a record
    that lost its class string still classifies via the substring
    fallback. Caught by stage thunks via the classifier — never special-
    cased by stage code directly."""

    def __init__(
        self,
        *,
        elapsed_s: float | None = None,
        limit_s: float | None = None,
        call_id: str | None = None,
        tier: str | None = None,
    ):
        elapsed = f"{elapsed_s:.1f}s" if elapsed_s is not None else "?"
        limit = f"{limit_s:.0f}s" if limit_s is not None else "?"
        # `tier` is "no_token" (pre-first-token ceiling) or "total"
        # (post-first-token ceiling) — surfaced for debug bundles; both
        # route to load identically.
        tier_str = f", tier={tier}" if tier else ""
        super().__init__(
            f"wall-clock timeout after {elapsed} (limit {limit}{tier_str}, "
            f"call_id={call_id})"
        )
        self.elapsed_s = elapsed_s
        self.limit_s = limit_s
        self.call_id = call_id
        self.tier = tier


class _SkippedByUser(Exception):
    """Raised by `llm._consume_chat_stream` when a per-call skip
    marker is observed for the in-flight call_id. Flows through
    the wrapper like any other failure; `_classify_failure` returns
    the dedicated `"skipped"` label so every stage thunk short-
    circuits before charging the scheduler's sliding window or
    submitting a retry. The materializer later reads
    `<run_dir>/skipped_calls/<call_id>` to stamp
    `rec["skipped"] = True` on the post-run rollup, distinct from
    any failure bucket — a user skip is a human signal, not a
    provider problem."""

    def __init__(self, call_id: str | None = None):
        super().__init__(f"skipped by user (call_id={call_id})")
        self.call_id = call_id


# Parse-signal exceptions live in ``parse_signals`` (the surviving half of
# this module — see #912 cutover). Re-exported here so the legacy
# classifier + every existing ``from retry import _ParseError, …`` keep
# working unchanged.
from engine.parse_signals import (  # noqa: E402,F401
    _EmptyResponse,
    _InterruptedResponse,
    _ParseError,
    _PostStreamFailure,
    _SuccessEmpty,
)

# The concrete parse-signal shapes (_ParseError / _EmptyResponse /
# _InterruptedResponse / _SuccessEmpty) and their base now live in
# ``parse_signals`` so they survive the cutover (this execution layer
# is deleted; the validators + kernel phases keep raising/catching them).
# Re-exported above for legacy ``from retry import …`` importers.


