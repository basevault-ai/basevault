"""Per-process session directory.

A *session* is one Python process boot — the pipeline runner, the
chatbot sidecar, or ad-hoc CLI scripts. Each gets its own dir under
``~/.basevault/sessions/<iso-z>-<short_id>/``.

This is the place for diagnostics that don't belong to a specific
pipeline run or chatbot conversation: wire-capture fallback (SDK
bootstrap, attestation, ad-hoc UI calls), and future session-scoped
artifacts like ``app.log`` segments. Run-scoped logs
(``llm-calls.jsonl``, ``run.log``, …) stay under
``~/.basevault/logs/<run-id>/``. Conversation-scoped logs stay in
the convo dir.

Created lazily on first ``get_session_dir()`` call so processes that
never need a session-scoped artifact don't litter the filesystem
with empty dirs.
"""
from __future__ import annotations

import datetime
import os
import threading
from pathlib import Path

from engine.common.utils import new_id

_session_dir: Path | None = None
_session_lock = threading.Lock()


def get_session_dir() -> Path:
    """Return this process's session dir, materializing it on first call.

    Resolution order:

      1. ``BASEVAULT_SESSION_DIR`` env var, set by the Tauri shell at
         app launch and inherited by every Python subprocess. This is
         the common case under the app: the chatbot sidecar, the
         pipeline runner, and any other subprocess all share one dir
         per app-launch so ``app.log`` and wire-capture fallbacks stay
         co-located.
      2. Otherwise mint a fresh one — ad-hoc CLI scripts, tests, and
         smoke runs spawned outside the Tauri shell don't have the
         env var set, so they get their own session.

    Path shape: ``~/.basevault/sessions/<YYYY-mm-ddTHH-MM-SSZ>-<short_id>/``,
    matching run dirs and conversation dirs.
    """
    global _session_dir
    with _session_lock:
        if _session_dir is None:
            env_dir = os.environ.get("BASEVAULT_SESSION_DIR", "").strip()
            if env_dir:
                _session_dir = Path(env_dir)
            else:
                _, full_id = new_id()
                _session_dir = (
                    Path.home() / ".basevault" / "sessions" / full_id
                )
            _session_dir.mkdir(parents=True, exist_ok=True)
        return _session_dir


def session_log(message: str) -> None:
    """Append a one-line event to ``<session-dir>/app.log``.

    Lifecycle pointer log — connects session-level events to the
    run-scoped / convo-scoped detail logs that live elsewhere. One
    line per event: UTC timestamp + tab + message. Caller owns
    message text. IO failures are swallowed so a logging hiccup
    never breaks the caller.
    """
    ts = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    line = f"{ts}\t{message}\n"
    try:
        with (get_session_dir() / "app.log").open("a") as f:
            f.write(line)
    except Exception:
        pass
