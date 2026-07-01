"""
Python sidecar driving chatbot conversational turns end-to-end.

Spawned **once per session** by the Rust ``chatbot`` Tauri command and
kept alive across turns: it runs a request loop reading one
newline-delimited JSON frame per iteration from stdin. The frame
shapes (dispatched by ``kind``, default ``"turn"`` when absent so
older shells stay backward-compatible):

  * ``{"kind": "turn", "query": "<new message>",
       "history": [{role, content}, …], "turn_id": <int>}`` — one
    conversational turn. The kept-open pipe means the attested
    TinfoilAI client built lazily on the first turn is reused for
    every later turn (its ~2 s construction is paid once per process,
    not once per message). ``turn_id`` is echoed on every event so a
    late event from a finished/cancelled turn cannot bleed into the
    next turn's UI.
  * ``{"kind": "run_available", "run_id": "...", "store_path": "...",
       "selection": "default"}`` — out-of-band notification that an
    ingest run has finished with a non-empty vectors.db (#780). When
    the session is currently unbound (no run existed at process
    start), this binds it to the pushed run and re-emits
    ``chatbot_bound``; when already bound, it's a no-op (no silent
    swap to a newer corpus mid-conversation — switching corpus stays
    the dropdown's explicit job). Carries no ``turn_id`` (not a turn).

EOF on stdin (Rust drops the pipe on app exit or before a re-spawn)
ends the process cleanly. Falls back to a single argv query (no
history, no framing, one-shot turn) for ad-hoc / test invocation.

The turn runs the converse → (maybe) retrieve → answer loop:

  1. Build the conversational prompt (persona + prior turns + the new
     message) and run the model's reply. This decision turn is NOT
     live-streamed: its deltas are withheld and the full returned
     content is classified once it completes. A reply containing a JSON
     tool call (anywhere — the model sometimes prepends a prose preamble
     or a fence before it) is a corpus-retrieval request; the whole
     turn, preamble included, is suppressed and never reaches the user.
     Any other reply is ordinary conversation and is emitted whole. The
     leak this prevents: streaming the decision turn live committed to
     "this is conversation" on the first chunk, so a late tool call (and
     its preamble) streamed straight to the user.
  2. On a tool call, validate and dispatch it against the bound corpus
     (search runs dense retrieval), then stream a second call with the
     matching records as numbered context — the grounded ANSWER turn
     still streams live. Corpus claims in that answer carry ``[N]``
     references.
  3. References are emitted only for entries the finished answer
     actually cited, so a non-grounded or "not in your data" reply
     never renders with references.

Stdout event schema (one JSON object per line, newline-terminated).
Every event also carries ``"turn_id": <int>`` (the turn fence) when
running framed under the persistent loop; it is omitted only on the
ad-hoc argv path.

    {"event": "chatbot_bound", "run": "<run-name>"|null,
     "store_path": ..., "selection": "user"|"default"}
        The corpus run this session is answering from (the resolved
        vectors.db's run dir), or null when no processed run exists.
        Emitted once per session at process start so the chat panel's
        run selector can mark the bound run as current. ``selection``
        is ``"user"`` when the user explicitly picked this run in the
        selector, ``"default"`` when it's the most-recent-non-empty
        fallback (no explicit pick yet, or the pick went stale).
        Re-emitted exactly once when a session that started unbound
        (no run had finished ingesting at process start) gets a
        ``run_available`` push from Rust and binds to that run
        (``_handle_run_available``). After that the binding is steady
        for the rest of the process — switching corpus stays the
        selector's explicit job and re-spawns the sidecar with a
        fresh ``chatbot_bound``.

    {"event": "chatbot_thinking"}
        Emitted at the very start of EVERY turn. The decision turn is
        always buffered (its deltas are withheld until the full reply
        is classified), so every response has an in-flight gap with no
        visible output while the model works; the panel shows a
        "Thinking…" state from this event until the first chunk /
        retrieving / done / error. Fired for ALL responses regardless
        of reasoning (reasoning-on just lengthens the gap with its
        chain-of-thought). Purely sidecar-driven and per-turn — no
        once-per-session or UI-inferred state that could go stale.
        Turn-fenced like every other turn event.

    {"event": "chatbot_chunk", "delta": "<content delta>"}
        One streamed content delta. Append-only on the UI side.

    {"event": "chatbot_replace", "text": "<replacement text>"}
        Overwrite the in-flight turn's bubble content with ``text``
        verbatim — NOT an append. The mixed-shape recovery: the model
        emitted a prose preamble around a JSON tool call and the
        stream-gate already leaked some of that prose into the bubble
        before suppression kicked in. The loop emits this event after
        ``parse_tool_call`` extracts the call so the UI wipes the
        leaked content and the persisted transcript stays clean. The
        wipe is always empty text in the dispatch path; the next-hop
        ``chatbot_retrieving`` event then drives the visible state
        from there.

    {"event": "chatbot_retrieving", "query": "<what's being looked up>"}
        The assistant called a retrieval tool and it is firing; ``query``
        is a short description of the lookup (the search query). The UI
        shows a visible "searching your data…" state on this turn until
        ``chatbot_done``.

    {"event": "chatbot_done", "resources": null | [ {...}, ... ],
     "run": "<run-name>"|null, "refused": true (only when set)}
        Turn finished cleanly. ``resources`` is ``null`` when no
        resources block should render — either pure conversation (no
        lookup directive) or a lookup turn refused before retrieval
        because the session has no corpus to search (``run: null``).
        A list of ``{index, kind, record_id, preview}`` (the chunks the
        answer actually grounded on) when the tool ran and the answer
        cited some. An empty list means the tool ran but the answer
        grounded on nothing — the UI renders the explicit "no matching
        resources" state, never a silent omit and never a list of
        irrelevant chunks beside a refusal.
        ``run`` is the corpus run this turn was answered against —
        the session binding, stamped on every completion so the UI can
        PIN it onto the persisted turn. A citation then resolves
        against the run that produced it regardless of the currently-
        selected run and across restarts; it does NOT depend on a
        later session re-emitting ``chatbot_bound``. Same value as the
        session's ``chatbot_bound.run`` (the binding is resolved once
        and reused for every turn), repeated here so it travels with
        the message instead of living only in transient UI state.
        ``refused`` is present and ``true`` only for the no-corpus
        carve-out, where the assistant text was the deterministic
        refusal chunk (not a real model reply). The UI uses this to
        mark the persisted turn so its assistant content does NOT ride
        into the next turn's history — the deterministic text, fed
        back as conversation history, taught the model to imitate it
        as prose even after a corpus was bound. Absent (or false) on
        every other completion path; pure-conversation turns and
        empty-result turns are not refusals.

    {"event": "chatbot_error", "message": "<short error>"}
        Anything unexpected (config load, store open, generator raise).
        UI renders a one-line error.

The sidecar exits 0 on stdin-EOF (clean shutdown). A per-turn failure
is signalled in-band via ``chatbot_error`` and the process stays alive
for the next turn — a recoverable error must not discard the warm
attested client. A hard process death (segfault / user-Stop SIGKILL)
is detected Rust-side via the pipe closing, not a non-zero exit code.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import sys
import traceback

# Disable the disk LLM cache for live chat. Hit rate is near zero (every
# turn grows the prompt, so every cache key is unique), and the key is
# (prompt, model, temp, params) — NOT vault state — so a re-asked
# question after the vault changes could replay a stale response.
# `setdefault` keeps a dev override possible: export the env var to "0"
# (or anything non-truthy) to re-enable. Process-local: the eval is a
# separate Python process and keeps its cache-on default.
os.environ.setdefault("BASEVAULT_LLM_CACHE_BYPASS", "1")
import uuid
from pathlib import Path

from engine.chatbot import (
    carryover_refs,
    resolve_chatbot_from_config,
    resources_for_emit,
    seed_records_for,
)
from engine.chatbot_tools import (
    audit_record,
)
from engine.llm import (
    Mode,
    _append_event_jsonl,
    _append_payload_jsonl,
    _payload_call_ids_written,
    _payload_call_ids_written_lock,
    _read_app_config,
    begin_payloads_yaml_turn,
    bootstrap_call_id_counter_from_jsonl,
    flush_payloads_yaml,
    get_mode_spec,
    payloads_yaml_turn_floor,
    set_calls_jsonl_path,
    set_payloads_jsonl_path,
    set_payloads_yaml_path,
)
from engine.chatbot_turn import TurnContext
from engine.rag_vector_store import open_store

from engine import shareable
from engine import shareable_markers


# Turn fence. The persistent sidecar serves many turns over its
# lifetime; Rust stamps each request with a monotonic ``turn_id`` and
# every event this turn echoes it. A late event from a finished or
# cancelled turn can therefore never be misattributed to the next
# turn's UI — defense-in-depth alongside the monotonic counter the
# React panel already filters on. ``None`` on the ad-hoc / argv path
# (no framing) → the key is simply omitted, exactly the pre-persistent
# event shape.
_TURN_ID: int | None = None


def _set_turn(turn_id: int | None) -> None:
    global _TURN_ID
    _TURN_ID = turn_id


def _emit(event: str, **payload) -> None:
    """Write one JSON-line event to stdout and flush. Flushing per
    event keeps the Rust BufReader's line-iterator hot — without it a
    Python-side stdio buffer would hold chunks until it fills or the
    process exits, defeating the streaming half of the surface.
    """
    payload["event"] = event
    if _TURN_ID is not None:
        payload["turn_id"] = _TURN_ID
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _read_one_request() -> tuple[str, object] | None:
    """Read ONE framed line from stdin and dispatch it by ``kind``.

    Returns a ``(kind, payload)`` tuple, or ``None`` when input is
    exhausted (caller stops the loop). The persistent stdin pipe
    carries two frame shapes today:

    - ``{"kind": "turn", "query": str, "history": [...], "turn_id": int}``
      — a conversational turn (the original frame the loop has always
      served). The ``kind`` field defaults to ``"turn"`` when absent,
      so an older Rust shell that wrote ``{"query":..., "history":...,
      "turn_id":...}`` without the discriminator is still dispatched
      as a turn — backward-compat through one shipped pair.
      Payload: ``(query, history, turn_id)``.
    - ``{"kind": "run_available", "run_id": str, "store_path": str,
      "selection": "default"}`` — a Rust-side notification that an
      ingest run has finished and its vectors.db is now bindable.
      Closes #780: a session that started before any run existed
      binds to this run on receipt (when currently unbound), so the
      user doesn't have to restart the app or pick from the dropdown.
      No ``turn_id`` (not a turn). Payload: the parsed dict.

    Other shapes (blank lines, malformed JSON, unknown ``kind``) come
    back as ``("noop", None)`` so the loop survives garbage input —
    the only condition that ends the process is genuine stdin EOF.

    For ad-hoc / test invocation the query may instead arrive as a
    single argv arg (no stdin, no framing) — a one-shot turn with no
    turn_id; returned as ``("turn", (query, [], None))``.
    """
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if line == "":
            return None  # EOF — stdin closed, end the process
        if not line.strip():
            return ("noop", None)  # blank keepalive line
        try:
            obj = json.loads(line)
        except ValueError:
            return ("noop", None)
        if not isinstance(obj, dict):
            return ("noop", None)
        kind = obj.get("kind") or "turn"
        if kind == "run_available":
            run_id = obj.get("run_id")
            store_path = obj.get("store_path")
            if not isinstance(run_id, str) or not isinstance(store_path, str):
                return ("noop", None)
            return ("run_available", {
                "run_id": run_id,
                "store_path": store_path,
                "selection": obj.get("selection") or "default",
            })
        if kind == "turn":
            query = str(obj.get("query") or "")
            history = obj.get("history")
            if not isinstance(history, list):
                history = []
            else:
                # Defensive normalize: drop history entries that aren't
                # role+content dicts. Cited refs (if present on an
                # assistant turn) ride through as the third allowed
                # field so the carryover-seed walk can re-hydrate the
                # most recent grounding into the next turn's brackets.
                history = [h for h in history if isinstance(h, dict)]
            turn_id = obj.get("turn_id")
            if not isinstance(turn_id, int):
                turn_id = None
            return ("turn", (query, history, turn_id))
        return ("noop", None)  # unknown kind — forward-compat ignore
    parser = argparse.ArgumentParser(description="BaseVault chatbot sidecar.")
    parser.add_argument("query", nargs="?", default="")
    args = parser.parse_args()
    return ("turn", (args.query, [], None))


def _logs_root() -> Path:
    """Same resolution rule as ``runner.py``'s ``_LOGS_ROOT``. The Rust
    shell sets ``BASEVAULT_LOGS_ROOT`` and ``BASEVAULT_AGENT=app`` when
    it spawns sidecars, so the deployed app always resolves to
    ``~/.basevault/logs/``.
    """
    override = os.environ.get("BASEVAULT_LOGS_ROOT")
    if override:
        return Path(override)
    agent = os.environ.get("BASEVAULT_AGENT", "").strip()
    sub = "logs" if agent == "app" else "logs-dev"
    return Path.home() / ".basevault" / sub


def _chats_root() -> Path:
    """Chat conversation data root — mirrors ``_logs_root()`` exactly.
    Chats are USER conversation data, not run logs, so they live
    OUTSIDE the logs tree (director call on #568). The Rust shell sets
    ``BASEVAULT_CHATS_ROOT`` (parallel to ``BASEVAULT_LOGS_ROOT``);
    absent that, the same agent/dev split as ``_logs_root()`` — app →
    ``~/.basevault/chats``, non-app/dev → ``~/.basevault/chats-dev``.
    """
    override = os.environ.get("BASEVAULT_CHATS_ROOT")
    if override:
        return Path(override)
    agent = os.environ.get("BASEVAULT_AGENT", "").strip()
    sub = "chats" if agent == "app" else "chats-dev"
    return Path.home() / ".basevault" / sub


def _telemetry_dir() -> Path:
    """Where this turn's ``llm-{calls,payloads}.jsonl`` go. Per
    conversation (#565/#568): the Rust shell sets
    ``BASEVAULT_CHATBOT_CONVO_DIR`` to the ACTIVE conversation's own dir
    (``<chats-root>/<ISO-Z>-<label>/``) so each thread's telemetry
    stays scoped to it. Unset (ad-hoc / test invocation, or an older
    shell) → ``_chats_root()`` (was ``logs/chatbot/`` pre-#568).
    """
    convo_dir = os.environ.get("BASEVAULT_CHATBOT_CONVO_DIR", "").strip()
    return Path(convo_dir) if convo_dir else _chats_root()


def _append_tool_audit(record: dict) -> None:
    """Append one tool-call record to the conversation's
    ``tool-calls.jsonl`` for replay/audit — the per-call trail the
    structured tool surface requires. Carries the tool, its validated
    args (the query, a kind, a record id), and the result count: a
    faithful, replayable trace of what was asked. It lives in the same
    per-conversation
    telemetry dir as the full prompt/response capture, so it is scoped to
    this thread, not the content-free shareable surface. Best-effort: a
    logging-dir problem must never break the chat turn.
    """
    try:
        path = _telemetry_dir() / "tool-calls.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass


_ISO_RUN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z")


def _run_creation_key(run_dir: Path) -> str | None:
    """A chronologically-sortable digits-only key (``YYYYMMDDHHMMSS``)
    for *when the run was created*, NOT when its db file was last
    touched. Source order:

      1. the run-dir name's ISO-8601-UTC prefix
         (``2026-05-16T03-14-54Z-…``) — authoritative for run order;
      2. ``config.json``'s ``created_at`` (``2026-05-16T03:14:54Z``);
      3. ``None`` — neither parseable, so the run is excluded from
         ordering entirely (a run with no determinable creation time
         must never silently win the default binding).

    File-mtime is deliberately not used: it is the direct cause of the
    characterised defect where an older run's db, touched last, shadowed
    a newer run (#507).
    """
    m = _ISO_RUN_RE.match(run_dir.name)
    if m:
        return re.sub(r"\D", "", m.group(0))
    try:
        cfg = json.loads((run_dir / "config.json").read_text())
        created = str(cfg.get("created_at") or "")
    except (OSError, ValueError):
        created = ""
    digits = re.sub(r"\D", "", created)
    return digits[:14] if len(digits) >= 14 else None


def _is_nonempty_store(store_path: Path) -> bool:
    """A bound-able vectors.db: a regular file with content. A 0-byte
    file is an in-flight / aborted store (embeddings never finished
    writing) and must be excluded from both the selector list and the
    default binding — the guarantee this function's callers' docstrings
    long claimed but never enforced (#507 defect #2).
    """
    try:
        return store_path.is_file() and store_path.stat().st_size > 0
    except OSError:
        return False


def _latest_store_path(logs_root: Path) -> Path | None:
    """Most-recent run dir under ``logs_root`` with a **non-empty**
    vectors.db, ordered by **run creation time** (the ISO-8601-UTC
    run-dir prefix / ``config.created_at``), not the db file's mtime.
    ``None`` when none qualify.

    This is the no-explicit-selection default. The deployed app resolves
    the binding Rust-side (explicit user pick, else this same rule over
    the same non-empty predicate) and passes it in via env; this stays
    the canonical resolver for the ad-hoc / argv / test path and as the
    safety fallback when no binding env is set.
    """
    if not logs_root.exists():
        return None
    candidates: list[tuple[str, Path]] = []
    for run_dir in logs_root.iterdir():
        if not run_dir.is_dir():
            continue
        store_path = run_dir / "stages" / "06-embeddings" / "vectors.db"
        if not _is_nonempty_store(store_path):
            continue
        key = _run_creation_key(run_dir)
        if key is None:
            continue
        candidates.append((key, store_path))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


def _bound_run_name(store_path: Path | None) -> str | None:
    """The corpus run a chat turn answered from, derived READ-ONLY from
    whatever ``_latest_store_path`` already resolved. The store layout is
    ``<logs_root>/<run-name>/stages/06-embeddings/vectors.db`` so the
    run name is the path's third parent. ``None`` when no store bound.

    Pure name derivation: it reports the run a resolved path belongs to
    for telemetry + the panel selector; it does not itself pick a store.
    Which store binds is decided by the user's selector pick (else the
    most-recent-non-empty default) — see ``_resolve_session_binding``.
    """
    if store_path is None:
        return None
    try:
        return store_path.parent.parent.parent.name
    except (AttributeError, IndexError):
        return None


# ── Chat-session demarcation ────────────────────────────────────────────────
#
# The dedicated chatbot llm-calls.jsonl / llm-payloads.jsonl are append-
# only across every chat session ever; without a boundary a reader can't
# tell where one conversation's calls end and the next begin. #503
# established the contract; the #454-P1 persistent sidecar performs the
# source swap #503's gate-1 explicitly designed for:
#
#   * A "session" is now the **process lifecycle**. The sidecar is
#     spawned once per session and serves every turn of it; a user-Stop
#     (or a crash) ends the session and the next message starts a fresh
#     process = a fresh session. So:
#       - session_id  = a process-minted uuid (was: content-hash of the
#                        conversation's first message). Stable across all
#                        turns of the process, distinct across processes.
#       - session start = process start (was: the empty-history turn).
#                        Exactly one `session_start` marker per session,
#                        emitted once before the request loop.
#       - bound run    = resolved ONCE at process start (was: per turn) —
#                        one fs scan per session, not per message.
#
# Selected-run binding (#507). The deployed app lets the user explicitly
# pick which run's vectors.db a session answers from; the Rust shell
# resolves the binding (the explicit pick, else the most-recent-non-empty
# default over the SAME predicate) and passes it in via env at spawn —
# changing the selection re-spawns the sidecar, so a fresh pick is just a
# fresh session, no new lifecycle. `_start_session` consumes that env
# (falling back to `_latest_store_path` for the ad-hoc/argv/test path)
# and records whether the binding was user-selected or the default in
# both the `session_start` marker and the `chatbot_bound` UI event.
#
# The contract #503 pinned is unchanged: every begin/end + payload record
# (incl. the nested rerank fired deep inside retrieve()) carries
# `session_id`, and there is exactly one `session_start` marker per
# session. `_start_session` sets these once; `_tracked_complete` reads
# `_SESSION_ID` so records carry it without plumbing through retrieval.
# `_SESSION_STORE_PATH` is the once-resolved binding the turn loop reuses.
_SESSION_ID: str | None = None
_SESSION_STORE_PATH: Path | None = None
_SESSION_BOUND_RUN: str | None = None
# 4-letter perma-id of the bound run dir, resolved once at session
# start (via ``shareable.resolve_perma_id``) and refreshed on any
# mid-session rebind. ``None`` when no run is bound OR a bindable run
# has no resolvable short_id (legacy dirs predating the perma-id model).
# Surfaced informationally in the chat marker; skip-reason resolution
# branches on ``_SESSION_STORE_PATH is not None``, not on this — a
# legacy bindable run still produces clean retrieval. Codex P2 (#818).
_SESSION_BOUND_RUN_PID: str | None = None
_SESSION_BOUND_SELECTION: str = "default"


def _resolve_session_binding() -> tuple[Path | None, str | None, str]:
    """Resolve ``(store_path, run_name, selection)`` for this session.

    The deployed app's Rust shell resolves the binding (the user's
    explicit selector pick, else the most-recent-non-empty default over
    the same predicate) and passes it in via env at spawn:

      * ``BASEVAULT_CHATBOT_STORE_PATH`` — the chosen run's vectors.db;
      * ``BASEVAULT_CHATBOT_RUN_ID``     — its run-dir name (label src);
      * ``BASEVAULT_CHATBOT_BIND_SOURCE``— ``"user"`` | ``"default"``.

    The env path is re-validated here (non-empty) so a store that went
    away / emptied between Rust's scan and this spawn degrades to the
    default rather than binding a 0-byte file. When no env is set (the
    ad-hoc / argv / test path), fall back to ``_latest_store_path`` —
    same rule, same non-empty predicate — and report ``"default"``.
    """
    env_path = os.environ.get("BASEVAULT_CHATBOT_STORE_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if _is_nonempty_store(p):
            run_id = os.environ.get("BASEVAULT_CHATBOT_RUN_ID", "").strip()
            source = os.environ.get(
                "BASEVAULT_CHATBOT_BIND_SOURCE", "").strip() or "default"
            return p, (run_id or _bound_run_name(p)), source
    fallback = _latest_store_path(_logs_root())
    return fallback, _bound_run_name(fallback), "default"


def _start_session() -> None:
    """Open a telemetry session = this process's lifetime. Mints the
    process session_id, resolves the bound corpus run ONCE, and emits
    the single `session_start` marker plus the `chatbot_bound` UI event.
    Called once at process start (before the request loop) — the source
    swap #503's gate-1 designed for; the boundary contract is unchanged.
    """
    global _SESSION_ID, _SESSION_STORE_PATH, _SESSION_BOUND_RUN
    global _SESSION_BOUND_RUN_PID, _SESSION_BOUND_SELECTION
    _SESSION_ID = uuid.uuid4().hex[:16]
    (
        _SESSION_STORE_PATH,
        _SESSION_BOUND_RUN,
        _SESSION_BOUND_SELECTION,
    ) = _resolve_session_binding()
    # Cache the perma-id for the bound run dir once per session. The
    # shareable chat marker keys on this every turn to make rebinds /
    # empty-binding cases legible from the yaml alone.
    _SESSION_BOUND_RUN_PID = shareable.resolve_perma_id(
        _run_dir_of(_SESSION_STORE_PATH)
    )
    store_path_s = (
        str(_SESSION_STORE_PATH) if _SESSION_STORE_PATH is not None else None
    )
    cfg = _read_app_config()
    chatbot = resolve_chatbot_from_config(cfg)
    # The session's provider mode + the model that will actually be
    # called, so telemetry shows whether a chat ran LOCAL or cloud (and
    # on which model) without inferring it from the per-call records.
    session_mode = _resolve_chatbot_mode(cfg)
    session_model = _chat_call_kwargs(session_mode, chatbot)["model"]
    # Surface the bound run + whether the user explicitly picked it so
    # the panel's run selector can mark the current entry. React's
    # boundRun state persists, so one emission covers the whole session.
    _emit(
        "chatbot_bound",
        run=_SESSION_BOUND_RUN,
        store_path=store_path_s,
        selection=_SESSION_BOUND_SELECTION,
    )
    # One marker per session. ts/schema/event are stamped by
    # _append_event_jsonl; no call_id (not a call) so the call-id
    # bootstrap (which scans only "begin" events) is unaffected.
    _append_event_jsonl("session_start", {
        "session_id": _SESSION_ID,
        "mode": session_mode.value,
        "model": session_model,
        "chatbot_config": chatbot,
        "bound_run": _SESSION_BOUND_RUN,
        "bound_store_path": store_path_s,
        "bound_selection": _SESSION_BOUND_SELECTION,
    })


def _handle_run_available(payload: dict) -> None:
    """Bind the session to a just-finished ingest run pushed by Rust.

    The Rust shell writes one ``{"kind":"run_available", ...}`` frame
    on the sidecar's stdin when a pipeline run finishes with a
    non-empty vectors.db. This closes #780: a session that started
    before any run existed (the cached `None` was sticky for the
    process's life — every lookup short-circuiting silently) now
    picks up its first run without restart or dropdown interaction.

    Two cases:

    - Currently unbound (``_SESSION_STORE_PATH is None``): bind to
      the pushed run, then emit a fresh ``chatbot_bound`` so the
      panel's selector tracks the real binding. React's reducer on
      ``chatbot_bound`` already calls ``refreshRuns``, so the new
      run appears in the dropdown AND is marked bound in one event.
    - Already bound: NO-OP on binding. The push is dropped (logged
      to stderr for operational visibility). Director's call: an
      in-conversation user must NOT be silently swapped to a newer
      corpus — the new run becomes visible in the dropdown via
      ``chatbot_list_runs`` on the next open (fs-driven), and the
      user picks it manually via ``chatbot_select_run`` (the
      existing respawn path, which is a fresh session by design).

    The ``session_start`` telemetry marker stays once-per-session by
    contract (it captures STARTING state, not the current state). The
    per-turn shareable + llm-calls.jsonl record the actual answering
    run for each turn, so a rebind is observable in telemetry without
    re-stamping the session marker.
    """
    global _SESSION_STORE_PATH, _SESSION_BOUND_RUN
    global _SESSION_BOUND_RUN_PID, _SESSION_BOUND_SELECTION
    if _SESSION_STORE_PATH is not None:
        # Already bound; no silent swap. Trace to stderr (Rust's
        # stderr drainer logs at warn!) so an operator can correlate
        # an in-flight settled session with a dropped push.
        sys.stderr.write(
            "chatbot_sidecar: run_available push dropped — "
            f"already bound to {_SESSION_BOUND_RUN!r}; "
            f"pushed run_id={payload.get('run_id')!r}\n"
        )
        sys.stderr.flush()
        return
    store_path_s = payload["store_path"]
    _SESSION_STORE_PATH = Path(store_path_s)
    pushed_run_id = payload.get("run_id")
    _SESSION_BOUND_RUN = (
        pushed_run_id if isinstance(pushed_run_id, str) and pushed_run_id
        else _bound_run_name(_SESSION_STORE_PATH)
    )
    _SESSION_BOUND_SELECTION = payload.get("selection") or "default"
    # Refresh the cached perma-id for the newly-bound run dir. The
    # chat marker reads this every turn; without the refresh, turns
    # after the rebind would still report the stale (None) perma-id.
    _SESSION_BOUND_RUN_PID = shareable.resolve_perma_id(
        _run_dir_of(_SESSION_STORE_PATH)
    )
    _emit(
        "chatbot_bound",
        run=_SESSION_BOUND_RUN,
        store_path=store_path_s,
        selection=_SESSION_BOUND_SELECTION,
    )


# ── Per-call chatbot telemetry (dedicated llm-calls.jsonl + llm-payloads.jsonl) ──
#
# Pipeline-stage LLM calls are visible in Run Details because the runner
# *brackets* each one (begin_stat_record → call → finalize_stat_record)
# into a run's llm-calls.jsonl, and captures each call's full prompt +
# response into a sibling llm-payloads.jsonl. The chatbot sidecar is a
# separate process that never runs the runner, so its converse / rerank
# / answer calls emitted nothing — neither timing (the director can't
# see where a chat turn's wall-clock goes) nor the prompts/responses
# (no way to inspect what the chatbot actually sent or got back). We reuse the
# runner's exact primitives at one shared bracket: begin/finalize for
# the call-stats (the same standalone bracket the embeddings stage
# uses) AND the existing payload-capture primitive for full_io, both
# pointed at a DEDICATED chatbot subdir so interactive telemetry stays
# cleanly separated from pipeline-run telemetry (no concurrent-writer
# race against an in-flight run's log; trivially greppable on its own).
# Payloads carry the same per-call discriminator as the calls log, so a
# reader correlates each prompt/response 1:1 to its begin/end record by
# call_id.


def _wire_call_stats() -> None:
    """Point the call-stats stream at the dedicated chatbot log and bracket
    the nested rerank call.

    Best-effort: a logging-dir problem must degrade to "no chatbot
    telemetry", never break the chat turn. The two direct sidecar calls
    are bracketed at their call sites via ``_tracked_complete``; the
    rerank call lives *inside* ``retrieval.retrieve()``, so — exactly as
    the runner rebinds ``complete`` in each stage module — we rebind
    ``retrieval.complete`` to the same shared bracket. One helper, two
    bind points; no fork of the runner's heavy wrapper. Idempotent so a
    persistent (warm) sidecar can call this once per process safely.
    """
    try:
        # Per-conversation telemetry dir (#565) — the ACTIVE
        # conversation's own dir when the shell scopes it, else the
        # legacy shared firehose. Same schema/version as a run's logs
        # (the primitives write "llm-calls/v1" / "llm-payloads/v1");
        # just separate files — calls + payloads side by side. Pre-#561:
        # the dir is shaped so #561's content-free marker can later sit
        # alongside these.
        chatbot_dir = _telemetry_dir()
        chatbot_dir.mkdir(parents=True, exist_ok=True)
        set_calls_jsonl_path(chatbot_dir / "llm-calls.jsonl")
        # ``llm-payloads.yaml`` (set below) carries the same content
        # turn-organized and human-readable; the chat-side
        # ``llm-payloads.jsonl`` had no programmatic readers and just
        # duplicated bytes on disk. Pipeline-side ``llm-payloads.jsonl``
        # stays untouched — ``build_fixtures.py`` reads it.
        set_payloads_jsonl_path(None)
        # Human-readable YAML companion: one document per chat TURN,
        # every LLM call of that turn listed under a `calls:` list, so
        # decision + grounded (and slice-2's multi-hop) are visible
        # together. Chat-only — the pipeline runner's high-volume
        # payload stream doesn't enable it. Atexit-flush in case the
        # process exits mid-turn so the buffered calls still land.
        set_payloads_yaml_path(chatbot_dir / "llm-payloads.yaml")
        atexit.register(flush_payloads_yaml)
        # Continue per-chat turn numbering past any existing YAML turns:
        # session_id is process-local, but the YAML is per-chat (perma-id)
        # and survives process restarts. Without this seed, turn 1 of a
        # fresh sidecar would write a second ``turn: 1`` doc into a YAML
        # that already has prior turns — readable but confusing. Read
        # the highest turn-N on disk and seed the counter past it so
        # numbering stays monotonic per chat, not per process.
        _TURN_COUNTER[0] = payloads_yaml_turn_floor()
        # Successive turns / sidecar invocations append to the same file;
        # advance the call-id counter past ids already on disk so begin/
        # end pairs never collide (the rollup keys on call_id alone).
        bootstrap_call_id_counter_from_jsonl()
    except OSError:
        # Disk / permission failure: skip telemetry, keep chatting.
        return


def _capture_payload(call_id: str, messages, content: str | None) -> None:
    """Stream the call's full prompt + response into the dedicated chatbot
    llm-payloads.jsonl via the existing payload primitive.

    Unconditional for the chatbot (it's the deliverable, not the pipeline's
    per-stage dev-tab toggle). The snapshot shape mirrors the pipeline's
    full_io records so the same readers parse it. Guarded by the
    primitive's own written-set: if the dev-tab toggle happens to be on
    for the active stage, ``complete()``'s ``_stamp_full_io`` already
    wrote (and registered the call_id) before returning — skip to avoid
    a duplicate, the same dedup contract ``_log_call_failure_payload``
    honours.
    """
    with _payload_call_ids_written_lock:
        if call_id in _payload_call_ids_written:
            return
    payload: dict = {
        "call_id": call_id,
        "full_prompt": [
            {"role": m.get("role"), "content": m.get("content")}
            for m in messages if isinstance(m, dict)
        ],
    }
    if _SESSION_ID:
        payload["session_id"] = _SESSION_ID
    if content is not None:
        payload["full_response"] = content
    _append_payload_jsonl(payload)


def _tracked_complete(*_args, **_kwargs):
    """Placeholder that seeds ``TurnContext.tracked_complete`` at construction.

    Chat runs on the kernel: ``run_chat_turn`` injects a
    kernel-backed ``tracked_complete`` onto the TurnContext (phases/chat.py),
    which opens the stat record / captures the payload via the
    KernelTelemetryHook. This stub is replaced before any hop dispatches, so
    it is never actually called."""
    raise NotImplementedError(
        "ctx.tracked_complete is injected by run_chat_turn"
    )


def _iso_now() -> str:
    """UTC ISO-8601 timestamp with seconds resolution, ``Z`` suffix —
    matches the format used in ``llm-calls.jsonl`` so a reader can
    cross-reference the two streams by start time."""
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_TURN_COUNTER = [0]
# Run perma-ids whose static corpus file we've already emitted this
# process — the run/corpus file is written once, never per turn.
_RUN_EMITTED: set[str] = set()


def _run_dir_of(store_path: Path | None) -> Path | None:
    """``<run-dir>/stages/06-embeddings/vectors.db`` -> ``<run-dir>``."""
    if store_path is None:
        return None
    try:
        return store_path.parent.parent.parent
    except (AttributeError, IndexError):
        return None


def _emit_shareable_markers(
    *,
    turn_index: int,
    lookup_fired: bool,
    hops_diag: list[dict],
    calls_baseline: int,
    store_bound: bool = False,
    store_stats: "shareable.StoreStats | None" = None,
    history_turn_count: int = 0,
    resources_emitted_count: int = 0,
) -> None:
    """Emit the two DISJOINT content-free shareable markers:

    - chat-side: this conversation turn — turn-level fields plus the
      per-hop ReAct trace (one entry per LLM call this turn fired,
      from ``ctx.hops_diag``), keyed by the CONVERSATION perma-id;
    - run-side: the bound corpus run's static structure (record counts,
      embed dim, size histograms, stages), written ONCE per RUN
      perma-id (idempotent — never re-appended per turn).

    No field appears in both; the chat file references the corpus only
    by the run perma-id, so nothing is duplicated. Strictly
    best-effort: any failure here must never break the chat turn — but
    a content-free violation inside ``shareable.emit`` is a loud crash
    *inside this guarded scope*, not a silent leak. The perma-id is
    read through the ``resolve_perma_id`` seam (consumed verbatim).
    """
    try:
        calls_path = _telemetry_dir() / "llm-calls.jsonl"
        llm_block = shareable_markers.build_llm_calls_block(
            calls_path, calls_baseline
        )

        convo_dir = os.environ.get("BASEVAULT_CHATBOT_CONVO_DIR", "").strip()
        chat_pid = shareable.resolve_perma_id(convo_dir or None)
        if chat_pid is not None:
            chat_marker = shareable_markers.build_chat_marker(
                turn_index=turn_index,
                # 16-hex session id minted once per sidecar process at
                # ``_start_session``. Surfaced on every chat marker so
                # a reader can tell "same conversation, new sidecar
                # process" apart from missing data — the persistent
                # sidecar respawns on Stop / app-close / re-warm and
                # ``turn_index`` resets per process.
                session_id=_SESSION_ID or "",
                lookup_fired=lookup_fired,
                hops_diag=hops_diag,
                llm_calls=llm_block,
                # ``store_bound`` is the authoritative skip-reason
                # signal: a legacy bindable run with no resolvable
                # 4-letter perma-id still has ``_SESSION_STORE_PATH``
                # set and retrieval runs cleanly — but its perma-id
                # ``bound_run`` resolves to None. Branching the skip-
                # reason on the binding (not the perma-id) avoids a
                # falsely-labelled ``no_bound_run`` for legacy runs.
                store_bound=store_bound,
                bound_run=_SESSION_BOUND_RUN_PID,
                store_stats=store_stats,
                history_turn_count=history_turn_count,
                resources_emitted_count=resources_emitted_count,
            )
            shareable.emit(shareable.Stream.CHAT, chat_pid, chat_marker)

        # The run/corpus file for a bound (completed) run is stable, so
        # the `_RUN_EMITTED` process-set caps this to one emit per run
        # perma-id per process — skip the per-turn store-open + rebuild
        # entirely once emitted. (emit() for the RUN stream is now
        # latest-wins/overwrite, driven by run wind-down events; this
        # guard is what keeps the chat path from re-dumping per turn.)
        run_dir = _run_dir_of(_SESSION_STORE_PATH)
        run_pid = shareable.resolve_perma_id(run_dir)
        if run_pid is not None and run_pid not in _RUN_EMITTED:
            _RUN_EMITTED.add(run_pid)
            store_cm = (
                open_store(_SESSION_STORE_PATH)
                if _SESSION_STORE_PATH is not None
                else None
            )
            try:
                store = store_cm.__enter__() if store_cm is not None else None
                run_marker = shareable_markers.build_run_marker(
                    store=store,
                    run_dir=run_dir,
                )
            finally:
                if store_cm is not None:
                    store_cm.__exit__(None, None, None)
            shareable.emit(shareable.Stream.RUN, run_pid, run_marker)
    except shareable.ContentFreeViolation:
        # By construction unreachable from the typed builders. If it
        # ever fires it is a real leak attempt — re-raise so it is
        # caught loudly in the per-turn handler, never written.
        raise
    except Exception:
        # Any other failure (disk, missing dir, telemetry race) is
        # non-fatal: shareable diagnostics are an aid, not the turn.
        pass


def _resolve_chatbot_mode(cfg: dict) -> Mode:
    """The provider Mode this chat session runs in.

    LOCAL when the user picked the on-device Ollama/MLX mode; otherwise
    the attested cloud path (Mode.TEE). Read from config.json ``mode``
    — the single source the pipeline runner and this sidecar both agree
    on, and the SAME source slice A's embedding dispatch reads, so a
    turn's query embed and its chat calls never split across providers.

    Binary by design: only ``local`` forks; tee (and an unset /
    unreadable config) keep the attested cloud chatbot. TEE is the
    fail-safe default so a missing/garbled config never silently
    downgrades a cloud user onto a local model that may not be
    installed.
    """
    mode = str(cfg.get("mode") or "").strip().lower()
    return Mode.LOCAL if mode == Mode.LOCAL.value else Mode.TEE


def _chat_call_kwargs(mode: Mode, chatbot: dict) -> dict:
    """Model + provider kwargs for the per-turn ``complete()`` calls
    inside the ReAct loop.

    Cloud: the configured chatbot model (glm-5-2 by default) with ``_force_model_id``
    so the call defeats the per-stage ``tee_model`` override and the
    reasoning kwarg is computed for the model actually called.

    LOCAL: the local chat model (``qwen3.5:9b`` or the configured
    ``local_mlx_model`` via ``get_mode_spec(LOCAL)``) on ``Mode.LOCAL``,
    running on the existing GPU-accelerated local path. No
    ``_force_model_id`` — LOCAL has no stage-override to defeat, and
    forcing would KeyError because the local Ollama/MLX spec isn't in
    the static per-model registry; complete()'s own LOCAL resolver
    returns ``get_mode_spec(LOCAL)`` directly. Reasoning stays
    config-driven for both.

    The returned dict is splatted into every ``tracked_complete`` call
    by ``chatbot_turn.run`` via ``TurnContext.complete_kwargs``, so the
    decision call, every grounded hop, and the forced-final call all
    route through the same provider/model — never split modes mid-turn.
    """
    if mode == Mode.LOCAL:
        return {"model": get_mode_spec(Mode.LOCAL).model_id, "mode": Mode.LOCAL}
    return {"model": chatbot["model"], "mode": Mode.TEE, "_force_model_id": True}


def _run(query: str, history: list[dict]) -> int:
    """Drive one conversational turn. Thin I/O wrapper around
    ``chatbot_turn.run`` — builds the per-turn context from session
    state, delegates the LLM loop body, then emits the turn's terminal
    events (shareable markers + ``chatbot_done``). Always returns 0.
    """
    _TURN_COUNTER[0] += 1
    _turn_index = _TURN_COUNTER[0]
    # Open a YAML telemetry turn so every LLM call this turn fires lands
    # in one ``calls:`` block in llm-payloads.yaml. Flushed in the
    # finally below so a mid-turn crash still writes what it had.
    begin_payloads_yaml_turn(_turn_index, _SESSION_ID)
    _calls_baseline = shareable_markers.llm_calls_baseline(
        _telemetry_dir() / "llm-calls.jsonl"
    )

    cfg = _read_app_config()
    chatbot = resolve_chatbot_from_config(cfg)
    # Dispatch this whole turn through the user's actual mode (LOCAL
    # vs the attested cloud path) instead of the historical hardcoded
    # Mode.TEE inside chatbot_turn.run. One resolution per turn is
    # applied uniformly to every ReAct hop's complete() call AND to
    # the dispatch() retrieval call so they never split modes mid-turn.
    _mode = _resolve_chatbot_mode(cfg)
    _call_kwargs = _chat_call_kwargs(_mode, chatbot)

    # Carryover-seed: the most recent assistant turn that grounded
    # something seeds this turn's accumulator at brackets ``[1..K]``,
    # so the user can follow up with "tell me more about [3]" and the
    # bracket resolves to the same record they just clicked. Empty
    # when nothing in the recent window cited anything; the loop then
    # starts from a fresh acc and the model must look up to ground.
    _seed_records = seed_records_for(
        _SESSION_STORE_PATH, carryover_refs(history),
    )

    ctx = TurnContext(
        query=query,
        history=history,
        turn_index=_turn_index,
        session_id=_SESSION_ID,
        store_path=_SESSION_STORE_PATH,
        bound_run=_SESSION_BOUND_RUN,
        chatbot_config=chatbot,
        # Per-mode model + provider kwargs the ReAct loop splats into
        # every ``tracked_complete`` call — see _chat_call_kwargs.
        complete_kwargs=_call_kwargs,
        # Plain provider mode the loop threads into the dispatch() call
        # for retrieval (the query embed inside dispatch already routes
        # by mode via slice A's embed chokepoint).
        mode=_mode,
        tracked_complete=_tracked_complete,
        emit=_emit,
        seed_records=_seed_records,
        # Audit trail: one content-light record per dispatched tool
        # call. The loop body invokes this after each dispatch so the
        # sidecar writes ``tool-calls.jsonl`` without the loop knowing
        # about that I/O concern.
        on_tool_call=lambda call, n: _append_tool_audit(
            audit_record(call, result_count=n),
        ),
    )
    # The kernel chat path runs the SAME ReAct loop (chatbot_turn.run) but
    # routes each hop's LLM call through the kernel (ChatPhase), with the
    # KernelTelemetryHook reproducing the per-call llm-calls.jsonl records AND
    # the _capture_payload dev-tab prompt/response capture (the hook's
    # payload_sink).
    try:
        from kernel.enums import PhaseName as _PhaseName
        from engine.phases.chat import run_chat_turn
        from engine.phases.model_specs import build_stage_env
        result = run_chat_turn(
            ctx, _mode,
            execution_env=build_stage_env(
                _PhaseName.CHAT, _mode, session_id=_SESSION_ID,
                payload_sink=_capture_payload),
        )
    finally:
        flush_payloads_yaml()

    # Terminal markers + done event. ``lookup_fired`` is True iff any
    # hop dispatched OR the #780 no-corpus refuse path fired (the
    # shareable marker shows the lookup intent for legibility even
    # though no records came back). ``refused`` short-circuits the
    # resources branch to pure-conversation shape (``resources=null,
    # run=null``) so the UI doesn't render the "no matching resources"
    # empty state alongside the deterministic refusal text — an
    # empty-query turn or pure-conversation turn still pass through
    # with ``lookup_fired=False``.
    # ``resources`` gates on whether there are records to render (carryover
    # seed OR fresh retrieval), not on whether a fresh lookup fired this
    # turn — see ``resources_for_emit``. Shared with the chat eval so the
    # bottom-panel decision can't drift between the live surface and the
    # eval's contract. Built before the markers emit so its count can be
    # stamped (resources_emitted_count) — the cited subset is what the UI
    # shows, not the dense top-k pool.
    resources = resources_for_emit(
        result.answer, result.retrieved,
        refused=result.refused, lookup_fired=result.lookup_fired,
    )
    _emit_shareable_markers(
        turn_index=_turn_index,
        lookup_fired=result.lookup_fired,
        hops_diag=ctx.hops_diag,
        calls_baseline=_calls_baseline,
        store_bound=_SESSION_STORE_PATH is not None,
        store_stats=result.store_stats,
        history_turn_count=(
            len(history) if isinstance(history, list) else 0
        ),
        resources_emitted_count=len(resources) if resources else 0,
    )
    if result.refused:
        # #780 no-corpus carve-out: don't pin a run (there is none to
        # pin against). The shareable marker already encodes the lookup
        # intent via the hops trace + ``retrieve_skipped_reason=no_bound_run``.
        # ``refused=true`` lets the UI mark the persisted turn so its
        # assistant content is excluded from the history fed into the
        # next turn — without that, the deterministic refusal chunk
        # round-trips back through the prompt and teaches the model to
        # parrot it as prose on subsequent turns even after a corpus
        # is bound.
        _emit("chatbot_done", resources=None, run=None, refused=True)
        return 0
    _emit("chatbot_done", resources=resources, run=_SESSION_BOUND_RUN)
    return 0


def _serve_one(query: str, history: list[dict], turn_id: int | None) -> None:
    """Run exactly one turn, fenced by ``turn_id``. A handled exception
    is surfaced in-band as ``chatbot_error`` and the process stays alive
    for the next turn — a recoverable per-turn failure must not throw
    away the warm attested client (that defeats P1). Only stdin-EOF or
    a user-Stop SIGKILL (Rust side) ends the process."""
    _set_turn(turn_id)
    # Cross-reference into the session's app.log so a reader scanning
    # session events can jump to this turn's per-convo logs. Convo
    # dir's basename is the convo id (ISO-Z + 4-letter sid).
    try:
        from engine.common.session import session_log
        _convo_dir = os.environ.get("BASEVAULT_CHATBOT_CONVO_DIR", "").strip()
        _convo_id = Path(_convo_dir).name if _convo_dir else "<unscoped>"
        session_log(f"chat turn {turn_id} sent: {_convo_id}")
    except Exception:
        pass
    try:
        _run(query, history)
    except Exception as e:  # noqa: BLE001 — surface any failure to the UI
        _emit("chatbot_error", message=f"{type(e).__name__}: {e}")
        # Trace to stderr so the Rust side can capture it for app.log;
        # the UI only sees the short message.
        traceback.print_exc(file=sys.stderr)


def _warm_local_model() -> None:
    """Best-effort warm of the LOCAL chat model before the first turn.

    For the MLX backend, load the multi-GB weights into the process
    cache now so the first turn doesn't pay that load. For the Ollama
    backend the daemon owns model residency (it loads on first call and
    keep-alive holds it), so there is nothing in-process to warm. No
    Tinfoil attestation runs on this path — LOCAL has no attestation
    surface, so the cloud client construction is skipped entirely
    (consistent with the attestation-discipline: no attest off the
    three sanctioned app-side sites). Best-effort: any failure is
    swallowed and the first turn pays the cost.
    """
    try:
        from engine.llm import Provider, _get_mlx
        spec = get_mode_spec(Mode.LOCAL)
        if spec.provider == Provider.MLX:
            _get_mlx(spec.model_id)
    except Exception:  # noqa: BLE001 — warmup is advisory, never fatal
        traceback.print_exc(file=sys.stderr)


def _warm_client() -> None:
    """Warm the per-process inference path before the first request so
    its cost overlaps the user's compose/think time. Mode-aware: LOCAL
    warms the local model (``_warm_local_model``); the attested cloud
    path constructs the TinfoilAI client as before.

    Construct the attested TinfoilAI singleton now, before the first
    request, so its cost overlaps the user's compose/think time instead
    of blocking their first message.

    Constructing the client *is* the per-request attestation: the
    `TinfoilAI()` ctor (in the kernel `TinfoilProvider`) runs the full
    cryptographic enclave verification + sigstore TUF refresh and pins
    the transport — there is no separate per-model attest step to
    pre-run, because every subsequent `complete()` rides the already-
    attested, pinned client intrinsically. This call therefore just
    moves the one unavoidable client construction off the user's first
    turn; it does NOT itself perform a supplementary attest pass (that
    lives only at the three sanctioned app-side call sites).

    This is what makes the eager-re-warm work: Rust re-spawns a fresh
    persistent sidecar the instant the user hits Stop, and by the time
    they finish typing the next message this process already holds a
    constructed, attested client. Best-effort: any failure here
    (offline / no key / transient verification blip) is swallowed —
    the first real turn's ``complete()`` re-builds the client and
    surfaces the same error in-band via ``chatbot_error``, exactly as
    before. The persistent loop must never crash on a startup blip; on
    failure the singleton stays unbuilt and the first turn pays the
    construction cost as it would have anyway.

    ATTESTATION DISCIPLINE: this only warms the SDK client the sidecar
    must build regardless — it is not an attest call site. Real per-
    request attestation is intrinsic to that client. The supplementary
    cross-check + audit log + UI exposure run from exactly three
    sanctioned points and nowhere else: app startup verify, the
    Settings re-check control, and the hourly background timer. Do NOT
    add an attest call here or in per-turn / per-call / inference code.
    """
    if _resolve_chatbot_mode(_read_app_config()) == Mode.LOCAL:
        _warm_local_model()
        return
    try:
        from kernel.tinfoil_provider import TinfoilProvider

        # Constructing the kernel provider kicks off its background client
        # warm (the TinfoilAI ctor: enclave verify + sigstore TUF refresh +
        # TLS pin), so the first turn's inference rides an already-attested,
        # pinned client. The provider holds the client as a process
        # singleton, so the throwaway instance here warms the same client
        # the real chat inference uses.
        TinfoilProvider()
    except Exception:  # noqa: BLE001 — warmup is advisory, never fatal
        traceback.print_exc(file=sys.stderr)


def main() -> int:
    # Wire per-call telemetry once per process (not per turn) so every
    # chatbot call across the persistent process's lifetime lands in the
    # dedicated chatbot llm-calls.jsonl.
    _wire_call_stats()

    # Persistent request loop. The attested TinfoilAI client is a
    # per-process singleton; keeping this process alive across turns is
    # exactly what amortizes its ~2 s construction (paid once per
    # process, not once per message). The ad-hoc argv path (no stdin)
    # is a single-shot turn and exits; the framed stdin path warms the
    # client up front, then loops until Rust drops the pipe (EOF on app
    # exit or before a re-spawn) — a clean shutdown, return 0.
    if sys.stdin.isatty():
        req = _read_one_request()
        if req is not None and req[0] == "turn":
            # One-shot = a one-turn session: same demarcation contract.
            # The argv path only produces turn frames; a run_available
            # frame here would be a misconfiguration, ignore.
            _start_session()
            _serve_one(*req[1])
        return 0
    _warm_client()
    # A session = this process's lifetime: mint the session id, resolve
    # the bound run once, emit the single session_start marker + the
    # chatbot_bound UI event — all before the first request.
    _start_session()
    while True:
        req = _read_one_request()
        if req is None:
            return 0
        kind, payload = req
        if kind == "turn":
            _serve_one(*payload)
        elif kind == "run_available":
            # Out-of-band corpus-binding push from Rust (#780). Does
            # not consume a turn slot, does not advance the turn
            # counter, does not interrupt an in-flight stream
            # (the stdin read loop is single-threaded — a push line
            # buffered behind an in-flight turn is processed only
            # after that turn's `_serve_one` returns).
            _handle_run_available(payload)
        # "noop" → blank / malformed / unknown-kind line; continue.


if __name__ == "__main__":
    sys.exit(main())
