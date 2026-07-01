"""Closed-vocabulary per-call outcome.

SINGLE source of truth for the label a per-call rollup carries. Two
surfaces consume it:

- ``runner._classify_outcome`` produces ``Outcome`` values from each
  finalized rec; ``_apply_chain_aware_outcomes`` may re-label leaves
  to ``success_sampled`` / ``success_reasoning_off``.
- The shareable diagnostic carries the same ``Outcome`` value
  verbatim, so the YAML's per-call ``outcome`` field matches the
  run-details UI label byte-for-byte.

Backward-compatible ``OUTCOME_*`` string aliases mirror the historical
runner-module names — older call sites and serialized records can
keep doing direct string compares, and ``rec[\"outcome\"] == OUTCOME_SUCCESS``
still reads naturally. Because the enum subclasses ``str``, an
``Outcome`` member IS the string it spells; no conversion at the
boundary.

Failure outcomes carry a parenthetical retry-strategy suffix
(``(load)`` / ``(sizing)`` / ``(other)``) in the value — the runner's
display contract — so a reader sees the same bucket name in the
diagnostic that the UI shows on screen. Don't strip the suffix; it's
part of the label."""
from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    """Per-call terminal outcome label. ``str``-subclass so instances
    are interchangeable with the literal string in comparisons +
    serialization."""

    # Successes
    success = "success"
    success_empty = "success_empty"        # parseable [] / 0-entry result
    success_sampled = "success_sampled"    # chain leaf with /sample-N
    success_reasoning_off = "success_reasoning_off"  # leaf via /reasoning-off

    # Failure subtypes — parenthetical encodes the retry strategy.
    cap_hit_sizing = "cap_hit (sizing)"
    timeout_sizing = "timeout (sizing)"
    timeout_load = "timeout (load)"
    parse_error_sizing = "parse_error (sizing)"
    empty_response_load = "empty_response (load)"
    interrupted_sizing = "interrupted (sizing)"
    interrupted_load = "interrupted (load)"
    failed_load = "failed (load)"
    failed_sizing = "failed (sizing)"
    failed_other = "failed (other)"

    # Non-failure terminal states.
    aborted = "aborted"   # begin-without-end (run wound down mid-call)
    skipped = "skipped"   # user-skip marker on disk


# Backward-compatible string aliases for the historical runner-module
# names. Older call sites + serialized records keep working.
OUTCOME_SUCCESS = Outcome.success.value
OUTCOME_SUCCESS_EMPTY = Outcome.success_empty.value
OUTCOME_SUCCESS_SAMPLED = Outcome.success_sampled.value
OUTCOME_SUCCESS_REASONING_OFF = Outcome.success_reasoning_off.value
OUTCOME_CAP_HIT = Outcome.cap_hit_sizing.value
OUTCOME_TIMEOUT_SIZING = Outcome.timeout_sizing.value
OUTCOME_TIMEOUT_LOAD = Outcome.timeout_load.value
OUTCOME_PARSE_ERROR = Outcome.parse_error_sizing.value
OUTCOME_EMPTY_RESPONSE = Outcome.empty_response_load.value
OUTCOME_INTERRUPTED_SIZING = Outcome.interrupted_sizing.value
OUTCOME_INTERRUPTED_LOAD = Outcome.interrupted_load.value
OUTCOME_FAILED_LOAD = Outcome.failed_load.value
OUTCOME_FAILED_SIZING = Outcome.failed_sizing.value
OUTCOME_FAILED_OTHER = Outcome.failed_other.value
OUTCOME_ABORTED = Outcome.aborted.value
OUTCOME_SKIPPED = Outcome.skipped.value


# The full set of success subtypes. Consumers that need to ask "is
# this call a success?" check membership here instead of enumerating
# the four ``success_*`` values inline.
SUCCESS_OUTCOMES: frozenset[Outcome] = frozenset({
    Outcome.success,
    Outcome.success_empty,
    Outcome.success_sampled,
    Outcome.success_reasoning_off,
})


def is_success_outcome(o: Outcome) -> bool:
    """``True`` when ``o`` is any ``success*`` subtype the runner
    produces. Plain success + the three caveat'd successes
    (``success_empty`` / ``success_sampled`` /
    ``success_reasoning_off``) all count for ``success_count``
    purposes; the breakdown lives in ``outcome_dist``."""
    return o in SUCCESS_OUTCOMES


def outcome_from_label(label: object) -> Outcome:
    """Map a runner ``OUTCOME_*`` label string to the closed-enum
    ``Outcome``. Every label the runner produces has a 1:1 match in
    ``Outcome``'s vocabulary — both surfaces share the same closed
    set.

    An unrecognized label (defensive against runner-vocab drift)
    raises rather than silently widening: the trust contract holds.
    The error message lists the unrecognized token so an operator
    can teach ``Outcome`` the new value. ``None`` / non-string input
    surfaces as ``failed_other`` (the safest defensive bucket — never
    masquerades as success)."""
    if isinstance(label, str):
        try:
            return Outcome(label)
        except ValueError as exc:
            raise ValueError(
                f"unknown runner outcome label {label!r}; "
                f"add it to common.status.Outcome"
            ) from exc
    return Outcome.failed_other


# ── Stage vocabulary ────────────────────────────────────────────────


class StageToken(str, Enum):
    """Closed pipeline/interactive stage vocabulary. The raw stage
    string is mapped through ``stage_token()``; unknown -> ``other``."""
    chatbot = "chatbot"
    rerank = "rerank"
    embeddings = "embeddings"
    ingestion = "ingestion"
    extraction = "extraction"
    entities = "entities"
    entities_dedupe = "entities_dedupe"
    patterns = "patterns"
    insights = "insights"
    actions = "actions"
    vision = "vision"
    other = "other"
    none = "none"


# Runner stage labels that differ from the canonical token spelling.
_STAGE_ALIASES: dict[str, StageToken] = {
    "extract": StageToken.extraction,
    "entity": StageToken.entities,
    "pattern": StageToken.patterns,
    "insight": StageToken.insights,
    "action": StageToken.actions,
}


_STAGE_NAMES: frozenset[str] = frozenset(
    m.value for m in StageToken
    if m not in (StageToken.other, StageToken.none)
)


def stage_token(raw: str | None) -> StageToken:
    """Map a raw stage label (``\"06-embeddings\"`` / ``\"chatbot\"`` / …)
    to a closed ``StageToken``. Strips a leading ``NN-`` ordinal prefix
    and resolves the runner's short-form aliases (``extract`` →
    ``extraction`` etc.). Unknown labels return ``StageToken.other``;
    falsy input returns ``StageToken.none``."""
    if not raw:
        return StageToken.none
    s = str(raw).lower()
    if len(s) > 3 and s[:2].isdigit() and s[2] == "-":
        s = s[3:]
    if s in _STAGE_NAMES:
        return StageToken(s)
    return _STAGE_ALIASES.get(s, StageToken.other)
