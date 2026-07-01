"""Generic ISO-Z timestamp + duration helpers shared across the
pipeline.

The marker's canonical timestamp shape is ``YYYY-MM-DDTHH:MM:SS[.fff]Z``
— fixed-width fields, trailing ``Z``. Two callers use this:

- The shareable diagnostic (``shareable.py`` / ``shareable_markers.py``)
  whose content-free guard accepts ISO-Z as one of three allowed
  string-leaf shapes.
- The runner's per-call telemetry (``llm.py`` / ``runner.py``) whose
  ``llm-calls.jsonl`` records use the same shape so a reader can
  cross-correlate.

Putting the helpers in one module removes the prior duplication —
``_now_iso_z`` lived in ``shareable.py`` AND ``files_manifest.py``;
``_iso_delta_ms`` lived in ``shareable_markers.py`` AND ``runner.py``
with subtly different return types (float vs int). Single source.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone

# The marker's canonical ISO-Z shape. Fixed-width fields, optional
# fractional seconds, trailing ``Z``. The shareable content-free
# guard checks string leaves against this regex.
ISO_Z_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)


def now_iso_z() -> str:
    """``YYYY-MM-DDTHH:MM:SSZ`` — seconds precision, UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_iso_z_ms() -> str:
    """``YYYY-MM-DDTHH:MM:SS.fffZ`` — millisecond precision, UTC.

    The runner's per-call begin/end records use millisecond precision
    so back-to-back calls don't share a timestamp; the seconds-
    precision form (``now_iso_z``) is fine for whole-marker created_at
    and similar coarse-grained stamps."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{int((time.time() % 1) * 1000):03d}Z"


def now_human_local() -> str:
    """Local-time human-readable stamp, e.g. ``2026-04-26 15:57:55 PDT``.
    For run logs / UI surfaces; never for the content-free marker
    (which only accepts ISO-Z)."""
    local = datetime.now().astimezone()
    tz = local.tzname() or ""
    return local.strftime("%Y-%m-%d %H:%M:%S") + (f" {tz}" if tz else "")


def iso_or_none(v: object) -> str | None:
    """Return ``v`` only when it's exactly the marker's ISO-Z shape;
    else ``None``. Defensive against telemetry timestamp-format drift —
    a non-conforming stamp degrades to ``None`` rather than crashing
    a downstream emit / parse."""
    if isinstance(v, str) and ISO_Z_RE.match(v):
        return v
    return None


def parse_iso_z(s: str) -> float:
    """ISO-Z → POSIX seconds. Tolerates optional fractional seconds
    (the marker's form is either ``...:00Z`` or ``...:00.fffZ``).
    Raises ``ValueError`` on anything that doesn't match the marker
    shape — callers that prefer best-effort use ``iso_delta_ms``
    which catches and returns ``None``."""
    # datetime.fromisoformat handles ``+00:00`` but not the trailing
    # ``Z`` alone on older Pythons — swap explicitly so we cover both.
    iso = s.replace("Z", "+00:00")
    return datetime.fromisoformat(iso).replace(
        tzinfo=timezone.utc,
    ).timestamp()


def iso_min_max(values: list[str]) -> tuple[str | None, str | None]:
    """Min / max of a list of ISO-Z timestamps. Lexicographic
    comparison is well-defined for the marker's fixed-width form,
    so ``min`` / ``max`` give the right answer without parsing.
    Returns ``(None, None)`` for an empty input."""
    if not values:
        return (None, None)
    return (min(values), max(values))


def iso_delta_ms(start_iso: str | None, end_iso: str | None) -> float | None:
    """Best-effort millisecond delta between two ISO-Z timestamps.
    Returns ``None`` if either is missing or unparseable (callers
    that need ``int`` cast at the call site). Tolerates the common
    ``Z``-suffix shape."""
    if not start_iso or not end_iso:
        return None
    try:
        t0 = parse_iso_z(start_iso)
        t1 = parse_iso_z(end_iso)
    except (ValueError, TypeError):
        return None
    return max(0.0, (t1 - t0) * 1000.0)


def derive_ended_at_iso(
    started_at: object, duration_ms: object,
) -> str | None:
    """``started_at + duration_ms`` rendered back to the marker's
    ISO-Z (ms-precision) form. ``None`` when either input is
    missing or malformed.

    Used by the shareable diagnostic to recover a call's end-time
    when the upstream materialize doesn't carry the end event's
    ``ts`` onto the rec (it carries ``duration_ms`` instead)."""
    if not isinstance(started_at, str) or not ISO_Z_RE.match(started_at):
        return None
    if not isinstance(duration_ms, (int, float)) or duration_ms < 0:
        return None
    try:
        t0 = parse_iso_z(started_at)
    except ValueError:
        return None
    end_ts = t0 + duration_ms / 1000.0
    # Round to whole ms FIRST, then split into seconds + ms components.
    # Doing it after the split (``int(end_ts)`` for seconds +
    # ``round(fractional * 1000)`` for ms) lets the rounded ms hit 1000
    # when the fractional component sits at .9995+, emitting an
    # invalid ``...:04.1000Z`` (millisecond field must be 000-999) AND
    # losing the second-carry. Rounding the timestamp itself in ms-
    # space avoids both bugs in one step.
    end_ts_ms_total = round(end_ts * 1000)
    end_secs, end_ms = divmod(end_ts_ms_total, 1000)
    end_dt = datetime.fromtimestamp(end_secs, tz=timezone.utc)
    return end_dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{end_ms:03d}Z"
