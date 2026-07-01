"""Small generic helpers used across the pipeline.

This module is the home for utilities that are too small to deserve
their own file but generic enough to be shared. Keep additions
defensive + side-effect-free; anything stateful goes elsewhere."""
from __future__ import annotations

import datetime
import secrets

from engine.common.status import StageToken


# Char set matches the Rust short_id generator (lib.rs): lowercase
# letters + digits, minus visually ambiguous 0/1/i/l/o. Any tooling
# that filters by short_id sees the same alphabet on both sides.
_SHORT_ID_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"


def new_id() -> tuple[str, str]:
    """Mint a fresh ``<iso-z>-<short_id>`` perma-id.

    Returns ``(short_id, full_id)`` where ``short_id`` is the 4-char
    suffix and ``full_id`` is ``<YYYY-mm-ddTHH-MM-SSZ>-<short_id>``,
    matching the shape used for run dirs, conversation dirs, and
    session dirs. ~9.4M combinations (31⁴) per second; collision
    risk negligible at user volumes.
    """
    short_id = "".join(
        secrets.choice(_SHORT_ID_ALPHABET) for _ in range(4)
    )
    ts = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H-%M-%SZ"
    )
    return short_id, f"{ts}-{short_id}"


def is_short_id(s: object) -> bool:
    """True iff ``s`` is a 4-char string drawn entirely from the
    perma-id alphabet. Single seam for validating an id read from
    config, a dir suffix, or a sidecar JSON — keeps the
    "what is a perma-id" rule in one place."""
    return (
        isinstance(s, str)
        and len(s) == 4
        and all(c in _SHORT_ID_ALPHABET for c in s)
    )


def short_id_from_name(name: object) -> str | None:
    """Extract the 4-char perma-id from a ``<iso-z>-<short_id>``-shaped
    name (run dir, conversation dir, session dir). Returns ``None`` if
    the suffix isn't a valid perma-id."""
    if not isinstance(name, str) or not name:
        return None
    suffix = name.rsplit("-", 1)[-1]
    return suffix if is_short_id(suffix) else None


def safe_cid_int(v: object, *, default: int) -> int:
    """Parse a call_id (string or int) to ``int``; return ``default``
    on any failure (None / non-numeric / overflow). Used wherever a
    caller needs to compare call ids numerically without crashing on
    legacy / synthetic shapes."""
    if v is None:
        return default
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return default


def call_id_str(v: object) -> str | None:
    """A structural call id rendered in the runner's standard
    zero-padded 4-digit form (``\"0041\"``). Accepts the canonical
    jsonl string verbatim when it already matches, otherwise parses
    to ``int`` and re-formats. Returns ``None`` on non-numeric or
    out-of-range (n < 0 or n > 9999) inputs so callers don't have
    to guard.

    The shareable diagnostic's content-free guard whitelists the
    ``^\\d{4}$`` shape alongside ISO-Z + closed-enum strings — so
    this leaf passes the trust check directly."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        n = int(s)
    except ValueError:
        return None
    if n < 0 or n > 9999:
        # Pad/truncate is a no-op for the runner's bounded counter;
        # an out-of-range id is a sign of caller/telemetry drift —
        # fall through to a defensive ``None`` rather than widen
        # silently.
        return None
    return f"{n:04d}"


# ── Canonical pipeline-stage order ──────────────────────────────────
#
# The execution order the runner traverses. Used by the shareable
# diagnostic (and any other surface that aggregates per-stage data)
# to sort entries top-to-bottom in their natural pipeline sequence —
# so ``entities_dedupe`` (no ``stages/<NN-name>/`` dir leaf — fires
# inside the entities stage code) sits BETWEEN ``entities`` and
# ``patterns`` instead of at the end by enum-value sort. Stages
# absent from this tuple (defensive against drift) sort last.

STAGE_ORDER: tuple[StageToken, ...] = (
    StageToken.ingestion,
    StageToken.vision,
    StageToken.extraction,
    StageToken.entities,
    StageToken.entities_dedupe,
    StageToken.patterns,
    StageToken.insights,
    StageToken.actions,
    StageToken.embeddings,
    StageToken.chatbot,
    StageToken.rerank,
    StageToken.other,
    StageToken.none,
)


def stage_order_rank(stage: StageToken) -> int:
    """Position of ``stage`` in ``STAGE_ORDER``; out-of-band stages
    sort last (defensive against runner-vocab drift). Use as a
    ``sort`` key: ``out.sort(key=lambda s: stage_order_rank(s.stage))``."""
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        return len(STAGE_ORDER)
