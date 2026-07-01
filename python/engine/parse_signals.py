"""Parse-signal exceptions — the SURVIVING half of the legacy ``retry``.

These are the failure shapes the prompt/parse VALIDATORS raise
(``content_extractor._make_doc_parser``, ``entities``/``patterns``/
``insights``/``actions`` ``_validate_*``) AND that the kernel phases' kernel
``validate()`` catch to map to ``LlmStatus``. They are a contract between the
surviving parse logic and the kernel — independent of the legacy execution
layer (scheduler / retry classifier / ``complete()``), which is deleted at
the #912 cutover.

Extracted out of ``retry.py`` so they outlive that deletion. ``retry.py``
re-exports them (``from parse_signals import …``) so the legacy classifier
and every existing ``from retry import _ParseError, …`` keep working during
the transition; the kernel phases import them directly from here.

This module has NO project imports (leaf) — safe to import from anywhere.
"""
from __future__ import annotations


class _PostStreamFailure(Exception):
    """Base for the concrete failure shapes detected once a stream has
    finished but the body is unusable. Never raised directly — a concrete
    subclass is raised by stage code or, for the wire-shape kinds, by
    ``complete()`` itself. Each subclass carries its own retry bucket and
    stat-record field as class attributes, so the classifier and the wrapper
    read the routing off the type instead of branching on a discriminator
    string.

    Concrete shapes:

      _ParseError          — model emitted non-empty unparseable JSON;
                             routes to sizing.
      _EmptyResponse       — wire-empty (zero chunks, no usage, no
                             finish_reason, no token timings) OR a
                             synthesis-stage parser saw whitespace-only raw;
                             routes to load.
      _InterruptedResponse — bytes flowed but the stream closed cleanly
                             without a terminating finish_reason chunk and
                             below max_tokens_reserved; routes to load.
      _SuccessEmpty        — call streamed successfully and the JSON parsed,
                             but the structured output is empty (e.g.
                             insights' ``{"cross_domain":[],"critical":[]}``);
                             routes to other.

    The message string carries the ``stat_field`` token so a record that
    lost its boolean flag (older on-disk runs) still classifies via the
    message-substring fallback in ``classify_bucket``."""

    bucket: str
    stat_field: str

    def __init__(self, stage: str, note: str | None = None):
        super().__init__(f"{self.stat_field} (stage={stage})")
        self.stage = stage
        self.note = note


class _ParseError(_PostStreamFailure):
    bucket = "sizing"
    stat_field = "parse_error"


class _EmptyResponse(_PostStreamFailure):
    bucket = "load"
    stat_field = "empty_response"


class _InterruptedResponse(_PostStreamFailure):
    bucket = "load"
    stat_field = "interrupted"


class _SuccessEmpty(_PostStreamFailure):
    # Parsed-empty: valid JSON but empty structured output. Routes through
    # Other for one retry (reasoning flipped off when the parent had it on).
    bucket = "other"
    stat_field = "success_empty"


# ── Serializable error record (moved here from the deleted retry.py) ─────────
# Used by the surviving non-stage failure producers (attestation, the chat
# sidecar's kernel telemetry) to write a content-free error record.
import re as _re

_USER_HOME_PATH_RE = _re.compile(r"/(?:Users|home)/[^/]+/")


def _exception_dict(exc: BaseException) -> dict:
    """Compact, serializable error record for llm-stats.json. No prompt or
    response content — only the error class, message (truncated), and a
    home-path-sanitized traceback. Safe to share in debug bundles. Walks
    ``exc.__traceback__`` directly so capture works outside an ``except``
    block (where ``format_exc()`` would ship ``"NoneType: None"``)."""
    import traceback as _tb
    cls = f"{type(exc).__module__}.{type(exc).__name__}"
    msg = str(exc)
    if len(msg) > 1000:
        msg = msg[:997] + "..."
    tb_str = "".join(
        _tb.format_exception(type(exc), exc, exc.__traceback__)
    ).rstrip()
    tb_str = _USER_HOME_PATH_RE.sub("~/", tb_str)
    if len(tb_str) > 4000:
        tb_str = tb_str[:3997] + "..."
    if tb_str.strip() in ("", "NoneType: None"):
        return {"class": cls, "message": msg, "traceback": None}
    return {"class": cls, "message": msg, "traceback": tb_str}
