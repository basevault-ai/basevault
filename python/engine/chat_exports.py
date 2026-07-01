"""
Chat export parsers — Claude.ai web, Claude Code, Codex, ChatGPT.

All adapters emit Document objects with the same shape the rest of the
ingestion pipeline already consumes. The strip rule is: keep ONLY the
text the user typed plus structural metadata (session id, cwd, project
name, timestamps, conversation title/summary). Drop assistant text,
tool I/O (tool_use / tool_result / function_call / exec output),
reasoning blocks, system reminders, pasted file content, and any
bookkeeping events (turn-start, token-count, summary-checkpoint, etc).

Format detection is shape-driven: the dispatcher inspects the parsed
top-level value (for JSON) or the first line's keys (for JSONL) and
picks the matching parser. Filename and parent directory are not
consulted — users drop exports under arbitrary subdir names.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from engine.ingestor import Document, SourceType


# ── Shape sniff ───────────────────────────────────────────────────────────────

def detect_json_format(parsed) -> SourceType | None:
    """Classify a parsed JSON top-level value. Returns None for shapes
    that should be skipped (e.g. the users.json roster file in a
    Claude.ai bundle), or for shapes that look like an unrecognized
    export. Day One is handled by the existing dispatcher and is not
    matched here — chat_exports is additive."""
    if isinstance(parsed, dict):
        # Single Claude.ai project metadata file.
        keys = parsed.keys()
        if {"name", "uuid", "prompt_template"}.issubset(keys) and "chat_messages" not in keys:
            return SourceType.CLAUDE_WEB_PROJECT
        return None

    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        if not isinstance(first, dict):
            return None
        first_keys = first.keys()
        if {"chat_messages", "summary", "uuid", "name"}.issubset(first_keys):
            return SourceType.CLAUDE_WEB_CONVERSATION
        # ChatGPT canonical conversations.json: each item carries a
        # `mapping` dict keyed by node id plus a `title` field.
        if "mapping" in first_keys and "title" in first_keys:
            return SourceType.CHATGPT_CONVERSATION
        # Claude.ai users.json — PII roster, never corpus.
        if {"email_address", "uuid"}.issubset(first_keys) and "chat_messages" not in first_keys:
            return None
    return None


def detect_jsonl_format(first_line: str) -> SourceType | None:
    """Classify a JSONL file from its first line's keys. Order of the
    checks matters only insofar as shapes are distinguishable; the
    asserts are deliberately precise enough that ordering is moot."""
    try:
        d = json.loads(first_line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    keys = set(d.keys())

    # Claude Code typed-prompt history: one entry per typed prompt,
    # always carries `display` + `project` + `sessionId` + `timestamp`.
    if {"display", "project", "sessionId", "timestamp"}.issubset(keys):
        return SourceType.CLAUDE_CODE_HISTORY

    # Claude Code per-session transcript: every line carries `type`
    # and `sessionId` but never `project` at top level (cwd is the
    # per-line field, not project).
    if {"type", "sessionId"}.issubset(keys) and "project" not in keys:
        return SourceType.CLAUDE_CODE_SESSION

    # Codex typed-prompt history: shape is exactly {session_id, text, ts}.
    if {"session_id", "text", "ts"}.issubset(keys):
        return SourceType.CODEX_HISTORY

    # Codex per-session transcript: every line is a {timestamp, type,
    # payload} envelope; payload.type carries the event sub-kind.
    if {"timestamp", "type", "payload"}.issubset(keys):
        return SourceType.CODEX_SESSION

    return None


# ── Small helpers ─────────────────────────────────────────────────────────────

def _iso_date(ts, *, unit_seconds: bool = False) -> str:
    """Best-effort YYYY-MM-DD from a timestamp. Accepts ISO strings,
    integer/float epoch seconds, or epoch milliseconds. Heuristic: any
    numeric > 1e12 is treated as ms; pass unit_seconds=True to skip
    the heuristic when the source contract guarantees seconds."""
    if isinstance(ts, str) and ts:
        return ts[:10]
    try:
        n = float(ts)
    except (TypeError, ValueError):
        return ""
    if not unit_seconds and n > 1e12:
        n = n / 1000.0
    try:
        return datetime.fromtimestamp(n, tz=timezone.utc).date().isoformat()
    except (OSError, ValueError, OverflowError):
        return ""


_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _slug(s: str, limit: int = 60) -> str:
    """Conservative filesystem-safe slug for embedding bucket names in
    file_ids. Truncates to keep downstream IDs readable."""
    return _SLUG_RE.sub("_", s or "_")[:limit] or "_"


def _short_anchor(text: str, limit: int = 80) -> str:
    """First line of `text`, trimmed to a retrieval-friendly anchor."""
    first = (text.split("\n", 1)[0] if text else "").strip()
    return (first[:limit]).rstrip(".")


# ── Claude.ai web export ──────────────────────────────────────────────────────

def _claude_web_human_text(message: dict) -> str:
    """Extract typed text from one Claude.ai chat_message. The content
    field is either a list of {type:'text', text} blocks or absent
    with a top-level `text` field. Non-text blocks (attachments,
    images) are skipped."""
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
    elif isinstance(content, str) and content.strip():
        parts.append(content.strip())
    if not parts:
        top = message.get("text", "")
        if isinstance(top, str) and top.strip():
            parts.append(top.strip())
    return "\n\n".join(parts)


def parse_claude_web_conversations(
    path: Path, file_id: str, content: str,
) -> list[Document]:
    data = json.loads(content)
    if not isinstance(data, list):
        return []
    docs: list[Document] = []
    for idx, conv in enumerate(data):
        if not isinstance(conv, dict):
            continue
        name = (conv.get("name") or "").strip()
        summary = (conv.get("summary") or "").strip()
        uuid = conv.get("uuid", "")
        date = _iso_date(conv.get("created_at", ""))

        human_turns: list[str] = []
        for msg in conv.get("chat_messages") or []:
            if not isinstance(msg, dict):
                continue
            if msg.get("sender") != "human":
                continue
            txt = _claude_web_human_text(msg)
            if txt:
                human_turns.append(txt)

        if not human_turns and not name and not summary:
            continue

        body_parts: list[str] = []
        if name:
            body_parts.append(f"# {name}")
        if summary:
            body_parts.append(f"Summary: {summary}")
        if human_turns:
            body_parts.append("\n\n".join(human_turns))

        conv_short = uuid[:8] if uuid else f"{idx:03d}"
        docs.append(Document(
            id=f"{file_id}::conv_{conv_short}",
            source_path=str(path),
            source_type=SourceType.CLAUDE_WEB_CONVERSATION,
            content="\n\n".join(body_parts).strip(),
            title=name or f"Claude.ai conversation {conv_short}",
            date=date,
            file_id=file_id,
            metadata={
                "conversation_uuid": uuid,
                "n_human_turns": len(human_turns),
            },
        ))
    return docs


def parse_claude_web_project(
    path: Path, file_id: str, content: str,
) -> list[Document]:
    data = json.loads(content)
    if not isinstance(data, dict):
        return []
    name = (data.get("name") or "").strip()
    description = (data.get("description") or "").strip()
    # `prompt_template` is the project-level custom instructions.
    instructions = (data.get("prompt_template") or "").strip()
    uuid = data.get("uuid", "")
    date = _iso_date(data.get("created_at", ""))

    parts: list[str] = []
    if name:
        parts.append(f"# Project: {name}")
    if description:
        parts.append(f"Description: {description}")
    if instructions:
        parts.append(f"Custom instructions:\n{instructions}")
    if not parts:
        return []

    return [Document(
        id=file_id,
        source_path=str(path),
        source_type=SourceType.CLAUDE_WEB_PROJECT,
        content="\n\n".join(parts).strip(),
        title=name or f"Claude.ai project {uuid[:8]}",
        date=date,
        file_id=file_id,
        metadata={"project_uuid": uuid},
    )]


# ── Claude Code per-session transcript ────────────────────────────────────────

def _claude_code_user_text(message: dict) -> str:
    """Pull the typed text from a Claude Code session `type=user`
    event's `message` payload. Two content shapes exist:
      - str  → the typed prompt itself
      - list → only `type=text` blocks count; `type=tool_result`
               blocks are tool I/O re-served as user-role messages
               and are dropped per the strip contract.
    """
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
        return "\n\n".join(parts)
    return ""


def parse_claude_code_session(
    path: Path, file_id: str, content: str,
) -> list[Document]:
    session_id = ""
    cwd = ""
    custom_title = ""
    typed: list[tuple[object, str]] = []  # (timestamp, text)

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(d, dict):
            continue

        if not session_id and isinstance(d.get("sessionId"), str):
            session_id = d["sessionId"]
        if not cwd and isinstance(d.get("cwd"), str):
            cwd = d["cwd"]

        t = d.get("type")
        if t == "custom-title":
            ct = d.get("customTitle")
            if isinstance(ct, str) and ct.strip():
                custom_title = ct.strip()
        elif t == "user":
            text = _claude_code_user_text(d.get("message") or {})
            if text:
                typed.append((d.get("timestamp", 0), text))
        # Everything else — assistant, attachment, file-history-snapshot,
        # last-prompt, pr-link, system, queue-operation, agent-name,
        # permission-mode — is bookkeeping or assistant/tool content
        # and is dropped per the strip contract.

    if not typed:
        return []

    typed.sort(key=lambda x: x[0] if isinstance(x[0], (int, float)) else 0)
    anchor = custom_title or _short_anchor(typed[0][1])
    date = _iso_date(typed[0][0])

    body_parts: list[str] = []
    if anchor:
        body_parts.append(f"# {anchor}")
    if cwd:
        body_parts.append(f"cwd: {cwd}")
    body_parts.append("\n\n".join(t for _, t in typed))

    short_sid = (session_id[:8] if session_id else file_id[:8])
    return [Document(
        id=file_id,
        source_path=str(path),
        source_type=SourceType.CLAUDE_CODE_SESSION,
        content="\n\n".join(body_parts).strip(),
        title=anchor or f"Claude Code session {short_sid}",
        date=date,
        file_id=file_id,
        metadata={
            "session_id": session_id,
            "cwd": cwd,
            "n_typed_turns": len(typed),
        },
    )]


# ── Claude Code typed-prompt history ──────────────────────────────────────────

def _ts_month_key(ts) -> str:
    """YYYY-MM bucket key for a ms-epoch (Claude Code) or s-epoch
    (Codex) timestamp. Falls back to `unknown` for unparseable values
    so they cluster together rather than each becoming its own bucket."""
    if isinstance(ts, str) and len(ts) >= 7:
        return ts[:7]
    try:
        n = float(ts)
    except (TypeError, ValueError):
        return "unknown"
    if n > 1e12:
        n = n / 1000.0
    try:
        return datetime.fromtimestamp(n, tz=timezone.utc).strftime("%Y-%m")
    except (OSError, ValueError, OverflowError):
        return "unknown"


def parse_claude_code_history(
    path: Path, file_id: str, content: str,
) -> list[Document]:
    """Cross-session firehose of typed prompts. One Document per
    calendar MONTH — per-project bucketing produced one Document per
    cwd, which fragments into hundreds of tiny units for any
    long-running BaseVault user. Month bucketing matches the Day One
    year-split precedent at the granularity history files actually
    move at (typed prompts span months, not years).

    Project context is preserved inline as a `[project] prompt` prefix
    on each entry, so the chunker / retrieval pipeline still sees the
    cwd anchor even though it no longer drives Document boundaries.

    `pastedContents` is dropped: per director call, it's overwhelmingly
    code/logs/doc fragments being worked WITH, not personal thought."""
    from collections import OrderedDict
    buckets: "OrderedDict[str, list[tuple[object, str, str]]]" = OrderedDict()

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        display = d.get("display")
        if not isinstance(display, str) or not display.strip():
            continue
        project = d.get("project") or "_unknown"
        if not isinstance(project, str):
            project = "_unknown"
        ts = d.get("timestamp", 0)
        month_key = _ts_month_key(ts)
        buckets.setdefault(month_key, []).append((ts, project, display.strip()))

    if not buckets:
        return []

    docs: list[Document] = []
    for month_key in sorted(buckets.keys()):
        entries = buckets[month_key]
        entries.sort(key=lambda x: x[0] if isinstance(x[0], (int, float)) else 0)
        body = "\n\n".join(f"[{project}] {text}" for _, project, text in entries)
        first_ts = entries[0][0]
        date = _iso_date(first_ts)
        projects_seen = sorted({p for _, p, _ in entries})
        bucket_id = f"{file_id}::month_{month_key}"
        docs.append(Document(
            id=bucket_id,
            source_path=str(path),
            source_type=SourceType.CLAUDE_CODE_HISTORY,
            content=f"# Claude Code typed prompts — {month_key}\n\n{body}".strip(),
            title=f"Claude Code prompts {month_key}",
            date=date,
            file_id=bucket_id,
            metadata={
                "month": month_key,
                "projects": projects_seen,
                "n_prompts": len(entries),
                "origin_file_id": file_id,
            },
        ))
    return docs


# ── Codex per-session transcript ──────────────────────────────────────────────

def parse_codex_session(
    path: Path, file_id: str, content: str,
) -> list[Document]:
    """Codex sessions carry two parallel user-side surfaces:
      - event_msg / payload.type=user_message  → clean typed prompt
      - response_item / payload.type=message,
        payload.role=user                      → the same prompts
                                                 replayed into model
                                                 context, with
                                                 synthetic
                                                 <environment_context>
                                                 blocks interleaved.
    Only the first is human-typed; the second is dropped to avoid the
    env-context injection leaking into corpus."""
    session_id = ""
    cwd = ""
    typed: list[tuple[object, str]] = []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        t = d.get("type")
        payload = d.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}

        if t == "session_meta":
            if not session_id and isinstance(payload.get("id"), str):
                session_id = payload["id"]
            if not cwd and isinstance(payload.get("cwd"), str):
                cwd = payload["cwd"]
            continue

        if t == "event_msg" and payload.get("type") == "user_message":
            msg = payload.get("message")
            if isinstance(msg, str) and msg.strip():
                typed.append((d.get("timestamp", ""), msg.strip()))

    if not typed:
        return []

    typed.sort(key=lambda x: x[0] if isinstance(x[0], str) else (
        x[0] if isinstance(x[0], (int, float)) else 0))
    first_ts = typed[0][0]
    date = _iso_date(first_ts, unit_seconds=True) if isinstance(first_ts, (int, float)) else _iso_date(first_ts)
    anchor = _short_anchor(typed[0][1])

    body_parts: list[str] = []
    if anchor:
        body_parts.append(f"# {anchor}")
    if cwd:
        body_parts.append(f"cwd: {cwd}")
    body_parts.append("\n\n".join(t for _, t in typed))

    short_sid = (session_id[:8] if session_id else file_id[:8])
    return [Document(
        id=file_id,
        source_path=str(path),
        source_type=SourceType.CODEX_SESSION,
        content="\n\n".join(body_parts).strip(),
        title=anchor or f"Codex session {short_sid}",
        date=date,
        file_id=file_id,
        metadata={
            "session_id": session_id,
            "cwd": cwd,
            "n_typed_turns": len(typed),
        },
    )]


# ── Codex typed-prompt history ────────────────────────────────────────────────

def parse_codex_history(
    path: Path, file_id: str, content: str,
) -> list[Document]:
    """Codex history carries no project / cwd field — only session_id
    + text + ts. Bucket by calendar MONTH (same coarsening rationale
    as Claude Code history); short session ids ride as inline prefixes
    so retrieval can still cluster within a session."""
    from collections import OrderedDict
    buckets: "OrderedDict[str, list[tuple[object, str, str]]]" = OrderedDict()

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        text = d.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        sid = d.get("session_id") or "_unknown"
        if not isinstance(sid, str):
            sid = "_unknown"
        ts = d.get("ts", 0)
        month_key = _ts_month_key(ts)
        buckets.setdefault(month_key, []).append((ts, sid, text.strip()))

    if not buckets:
        return []

    docs: list[Document] = []
    for month_key in sorted(buckets.keys()):
        entries = buckets[month_key]
        entries.sort(key=lambda x: x[0] if isinstance(x[0], (int, float)) else 0)
        body = "\n\n".join(
            f"[session {sid[:8]}] {text}" for _, sid, text in entries)
        first_ts = entries[0][0]
        date = _iso_date(first_ts, unit_seconds=True)
        sessions_seen = sorted({s for _, s, _ in entries})
        bucket_id = f"{file_id}::month_{month_key}"
        docs.append(Document(
            id=bucket_id,
            source_path=str(path),
            source_type=SourceType.CODEX_HISTORY,
            content=f"# Codex typed prompts — {month_key}\n\n{body}".strip(),
            title=f"Codex prompts {month_key}",
            date=date,
            file_id=bucket_id,
            metadata={
                "month": month_key,
                "n_sessions": len(sessions_seen),
                "n_prompts": len(entries),
                "origin_file_id": file_id,
            },
        ))
    return docs


# ── ChatGPT web export ────────────────────────────────────────────────────────

def _chatgpt_user_text(message: dict) -> str:
    """Pull typed text from one ChatGPT message node. Content shape is
    {content_type, parts:[str, ...]} for plain-text turns; multimodal
    or tool turns use other content_types and are skipped."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content") or {}
    if not isinstance(content, dict):
        return ""
    if content.get("content_type") not in (None, "text"):
        return ""
    parts = content.get("parts") or []
    if not isinstance(parts, list):
        return ""
    texts: list[str] = []
    for p in parts:
        if isinstance(p, str) and p.strip():
            texts.append(p.strip())
    return "\n\n".join(texts)


def parse_chatgpt_conversations(
    path: Path, file_id: str, content: str,
) -> list[Document]:
    """ChatGPT export shape: top-level list of conversations, each
    with a `mapping` dict whose values are message nodes. We collect
    user-role messages, ignore branching (latest-write-wins per
    node-id is fine for v1; assistant-only branches are dropped by the
    role filter anyway), and sort by message create_time."""
    data = json.loads(content)
    if not isinstance(data, list):
        return []
    docs: list[Document] = []
    for idx, conv in enumerate(data):
        if not isinstance(conv, dict):
            continue
        title = (conv.get("title") or "").strip()
        conv_id = conv.get("conversation_id") or conv.get("id") or ""
        date = _iso_date(conv.get("create_time"), unit_seconds=True)
        mapping = conv.get("mapping") or {}
        if not isinstance(mapping, dict):
            mapping = {}

        typed: list[tuple[float, str]] = []
        for node in mapping.values():
            if not isinstance(node, dict):
                continue
            msg = node.get("message")
            if not isinstance(msg, dict):
                continue
            author = msg.get("author") or {}
            if not isinstance(author, dict) or author.get("role") != "user":
                continue
            text = _chatgpt_user_text(msg)
            if not text:
                continue
            ct = msg.get("create_time")
            ct_val = float(ct) if isinstance(ct, (int, float)) else 0.0
            typed.append((ct_val, text))

        if not typed and not title:
            continue

        typed.sort(key=lambda x: x[0])
        body_parts: list[str] = []
        if title:
            body_parts.append(f"# {title}")
        if typed:
            body_parts.append("\n\n".join(t for _, t in typed))

        short = (conv_id[:8] if isinstance(conv_id, str) and conv_id else f"{idx:03d}")
        docs.append(Document(
            id=f"{file_id}::conv_{short}",
            source_path=str(path),
            source_type=SourceType.CHATGPT_CONVERSATION,
            content="\n\n".join(body_parts).strip(),
            title=title or f"ChatGPT conversation {short}",
            date=date,
            file_id=file_id,
            metadata={
                "conversation_id": conv_id,
                "n_typed_turns": len(typed),
            },
        ))
    return docs


# ── Dispatch entry points ─────────────────────────────────────────────────────

_JSON_PARSERS = {
    SourceType.CLAUDE_WEB_CONVERSATION: parse_claude_web_conversations,
    SourceType.CLAUDE_WEB_PROJECT: parse_claude_web_project,
    SourceType.CHATGPT_CONVERSATION: parse_chatgpt_conversations,
}

_JSONL_PARSERS = {
    SourceType.CLAUDE_CODE_SESSION: parse_claude_code_session,
    SourceType.CLAUDE_CODE_HISTORY: parse_claude_code_history,
    SourceType.CODEX_SESSION: parse_codex_session,
    SourceType.CODEX_HISTORY: parse_codex_history,
}


def parse_chat_json(
    path: Path, file_id: str, content: str,
) -> list[Document] | None:
    """Try the chat-export JSON parsers. Returns the Document list on
    a match, [] if a parser matched but emitted nothing, or None if
    the shape isn't one of ours (caller should fall back to Day One
    or skip)."""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None
    src = detect_json_format(parsed)
    if src is None:
        return None
    parser = _JSON_PARSERS.get(src)
    if parser is None:
        return None
    return parser(path, file_id, content)


def parse_chat_jsonl(
    path: Path, file_id: str, content: str,
) -> list[Document]:
    """Dispatch a .jsonl file to the right chat parser based on its
    first non-empty line. Returns [] for an unrecognized JSONL — the
    caller already accepted .jsonl as a supported ext, so an unknown
    shape silently drops (mirroring the existing .json behaviour for
    non-Day-One JSON)."""
    first_line = ""
    for line in content.splitlines():
        if line.strip():
            first_line = line.strip()
            break
    if not first_line:
        return []
    src = detect_jsonl_format(first_line)
    if src is None:
        return []
    parser = _JSONL_PARSERS.get(src)
    if parser is None:
        return []
    return parser(path, file_id, content)
