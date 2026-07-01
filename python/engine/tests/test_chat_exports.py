"""
Tests for chat_exports.py — Claude.ai web, Claude Code, Codex, ChatGPT
ingestion adapters.

Run from engine/:
    cd engine && pytest tests/test_chat_exports.py

Fixture contents are synthetic. Drop-class fields embed `DROP_*`
markers so the strip-correctness assertions can be surgical: the
assertion is "no DROP_* substring survives into any Document.content."
"""
from __future__ import annotations

import json

import pytest

from engine.ingestor import SourceType, ingest
from engine.chat_exports import (
    detect_json_format,
    detect_jsonl_format,
)


# ── Drop / keep markers ───────────────────────────────────────────────────────

DROP_MARKERS = [
    "DROP_ASSISTANT", "DROP_TOOL_RESULT", "DROP_REASONING",
    "DROP_PASTED", "DROP_FUNCTION_CALL", "DROP_EXEC_OUTPUT",
    "DROP_AGENT_MSG", "DROP_ENV_CONTEXT", "DROP_THINKING",
    "DROP_BOOKKEEPING", "DROP_TOKEN_COUNT", "DROP_FILE_SNAPSHOT",
]


def _assert_no_drop_markers(docs):
    """Every emitted Document's content must be free of every drop
    marker. The marker scheme is the only thing each fixture relies
    on for strip-correctness — if a parser ever fails to drop a
    class, the corresponding marker survives and the assertion fires."""
    for d in docs:
        for m in DROP_MARKERS:
            assert m not in d.content, (
                f"drop marker {m!r} survived into Document "
                f"{d.id!r} ({d.source_type.value})")


# ── Claude.ai web — conversations.json ────────────────────────────────────────

CLAUDE_WEB_CONVERSATIONS = [
    {
        "uuid": "conv-aaaa-1111",
        "name": "KEEP_CONV_TITLE_alpha",
        "summary": "KEEP_CONV_SUMMARY_alpha",
        "account": {"uuid": "acct-1"},
        "created_at": "2024-03-15T10:00:00Z",
        "updated_at": "2024-03-15T10:30:00Z",
        "chat_messages": [
            {
                "uuid": "m-1",
                "sender": "human",
                "parent_message_uuid": None,
                "content": [{"type": "text", "text": "KEEP_HUMAN_TURN_1"}],
                "attachments": [], "files": [], "text": "",
                "created_at": "2024-03-15T10:00:00Z",
                "updated_at": "2024-03-15T10:00:00Z",
            },
            {
                "uuid": "m-2",
                "sender": "assistant",
                "parent_message_uuid": "m-1",
                "content": [{"type": "text", "text": "DROP_ASSISTANT_alpha_reply"}],
                "attachments": [], "files": [], "text": "",
                "created_at": "2024-03-15T10:01:00Z",
                "updated_at": "2024-03-15T10:01:00Z",
            },
            {
                "uuid": "m-3",
                "sender": "human",
                "parent_message_uuid": "m-2",
                "content": [{"type": "text", "text": "KEEP_HUMAN_TURN_2"}],
                "attachments": [], "files": [], "text": "",
                "created_at": "2024-03-15T10:02:00Z",
                "updated_at": "2024-03-15T10:02:00Z",
            },
        ],
    },
    {
        "uuid": "conv-bbbb-2222",
        "name": "KEEP_CONV_TITLE_beta",
        "summary": "",
        "account": {"uuid": "acct-1"},
        "created_at": "2024-04-01T08:00:00Z",
        "updated_at": "2024-04-01T08:30:00Z",
        "chat_messages": [
            {
                "uuid": "m-4",
                "sender": "human",
                "parent_message_uuid": None,
                "content": [{"type": "text", "text": "KEEP_HUMAN_TURN_3"}],
                "attachments": [], "files": [], "text": "",
                "created_at": "2024-04-01T08:00:00Z",
                "updated_at": "2024-04-01T08:00:00Z",
            },
            {
                "uuid": "m-5",
                "sender": "assistant",
                "parent_message_uuid": "m-4",
                "content": [{"type": "text", "text": "DROP_ASSISTANT_beta_reply"}],
                "attachments": [], "files": [], "text": "",
                "created_at": "2024-04-01T08:01:00Z",
                "updated_at": "2024-04-01T08:01:00Z",
            },
        ],
    },
]


@pytest.fixture
def claude_web_conversations_file(tmp_path):
    f = tmp_path / "conversations.json"
    f.write_text(json.dumps(CLAUDE_WEB_CONVERSATIONS), encoding="utf-8")
    return f


class TestClaudeWebConversations:
    def test_one_document_per_conversation(self, claude_web_conversations_file):
        docs = ingest(claude_web_conversations_file)
        assert len(docs) == 2
        assert all(d.source_type == SourceType.CLAUDE_WEB_CONVERSATION for d in docs)

    def test_human_turns_kept(self, claude_web_conversations_file):
        docs = ingest(claude_web_conversations_file)
        all_content = "\n".join(d.content for d in docs)
        for n in (1, 2, 3):
            assert f"KEEP_HUMAN_TURN_{n}" in all_content

    def test_assistant_text_stripped(self, claude_web_conversations_file):
        docs = ingest(claude_web_conversations_file)
        _assert_no_drop_markers(docs)

    def test_title_and_summary_in_content(self, claude_web_conversations_file):
        docs = ingest(claude_web_conversations_file)
        alpha = next(d for d in docs if "alpha" in d.title)
        assert "KEEP_CONV_TITLE_alpha" in alpha.content
        assert "KEEP_CONV_SUMMARY_alpha" in alpha.content
        assert alpha.date == "2024-03-15"

    def test_metadata_captures_uuid_and_turn_count(
        self, claude_web_conversations_file,
    ):
        docs = ingest(claude_web_conversations_file)
        alpha = next(d for d in docs if d.metadata["conversation_uuid"] == "conv-aaaa-1111")
        assert alpha.metadata["n_human_turns"] == 2


# ── Claude.ai web — project metadata file ─────────────────────────────────────

CLAUDE_WEB_PROJECT = {
    "uuid": "proj-cccc-3333",
    "name": "KEEP_PROJECT_NAME",
    "description": "KEEP_PROJECT_DESCRIPTION",
    "prompt_template": "KEEP_PROJECT_INSTRUCTIONS — multi\nline custom rules.",
    "is_private": True,
    "is_starter_project": False,
    "created_at": "2024-01-10T12:00:00Z",
    "updated_at": "2024-01-10T12:00:00Z",
    "creator": {"uuid": "u-1"},
    # docs[] field is dropped for v1 (file-content rather than typed text).
    "docs": [{"uuid": "d-1", "filename": "ref.md", "content": "DROP_PASTED_proj_doc",
              "created_at": "2024-01-10T12:00:00Z"}],
}


@pytest.fixture
def claude_web_project_file(tmp_path):
    f = tmp_path / "project_alpha.json"
    f.write_text(json.dumps(CLAUDE_WEB_PROJECT), encoding="utf-8")
    return f


class TestClaudeWebProject:
    def test_single_document(self, claude_web_project_file):
        docs = ingest(claude_web_project_file)
        assert len(docs) == 1
        assert docs[0].source_type == SourceType.CLAUDE_WEB_PROJECT

    def test_metadata_fields_kept(self, claude_web_project_file):
        d = ingest(claude_web_project_file)[0]
        assert "KEEP_PROJECT_NAME" in d.content
        assert "KEEP_PROJECT_DESCRIPTION" in d.content
        assert "KEEP_PROJECT_INSTRUCTIONS" in d.content

    def test_project_docs_dropped(self, claude_web_project_file):
        _assert_no_drop_markers(ingest(claude_web_project_file))


# ── Claude.ai web — users.json is skipped ─────────────────────────────────────

@pytest.fixture
def claude_web_users_file(tmp_path):
    f = tmp_path / "users.json"
    f.write_text(json.dumps([
        {"email_address": "placeholder@example.com",
         "full_name": "Placeholder User",
         "uuid": "u-1",
         "verified_phone_number": None},
    ]), encoding="utf-8")
    return f


class TestClaudeWebUsersSkipped:
    def test_users_file_emits_no_documents(self, claude_web_users_file):
        # users.json is PII roster — sniff matches it but no parser
        # is registered, so it emits nothing. Day One fallback also
        # returns [] because there's no `entries` key.
        assert ingest(claude_web_users_file) == []


# ── Claude Code — per-session JSONL ───────────────────────────────────────────

CC_SESSION_LINES = [
    # custom-title — the anchor
    {"type": "custom-title", "sessionId": "sess-9999",
     "customTitle": "KEEP_SESSION_TITLE"},
    # bookkeeping events that must drop
    {"type": "agent-name", "sessionId": "sess-9999",
     "agentName": "DROP_BOOKKEEPING_agent_name"},
    {"type": "permission-mode", "sessionId": "sess-9999",
     "permissionMode": "DROP_BOOKKEEPING_perm_mode"},
    # typed user prompt — str content
    {"type": "user", "sessionId": "sess-9999",
     "cwd": "/workspace/proj",
     "timestamp": 1710000000000, "uuid": "u-1",
     "userType": "external", "isSidechain": False,
     "message": {"role": "user", "content": "KEEP_HUMAN_TURN_cc_1"}},
    # assistant — drop
    {"type": "assistant", "sessionId": "sess-9999",
     "cwd": "/workspace/proj",
     "timestamp": 1710000001000, "uuid": "a-1",
     "message": {"role": "assistant",
                 "content": [{"type": "text",
                              "text": "DROP_ASSISTANT_cc_reply"}]}},
    # thinking block — drop
    {"type": "assistant", "sessionId": "sess-9999",
     "cwd": "/workspace/proj",
     "timestamp": 1710000001500, "uuid": "a-2",
     "message": {"role": "assistant",
                 "content": [{"type": "thinking",
                              "thinking": "DROP_THINKING_cc",
                              "signature": "sig"}]}},
    # user role wrapping a tool_result — must drop (it's tool I/O,
    # not typed text)
    {"type": "user", "sessionId": "sess-9999",
     "cwd": "/workspace/proj",
     "timestamp": 1710000002000, "uuid": "u-2",
     "userType": "external", "isSidechain": False,
     "message": {"role": "user", "content": [
         {"type": "tool_result", "tool_use_id": "t-1",
          "content": "DROP_TOOL_RESULT_cc"}]}},
    # second typed user prompt
    {"type": "user", "sessionId": "sess-9999",
     "cwd": "/workspace/proj",
     "timestamp": 1710000003000, "uuid": "u-3",
     "userType": "external", "isSidechain": False,
     "message": {"role": "user", "content": "KEEP_HUMAN_TURN_cc_2"}},
    # attachment / file snapshot / system / last-prompt — all drop
    {"type": "attachment", "sessionId": "sess-9999",
     "cwd": "/workspace/proj",
     "timestamp": 1710000004000, "uuid": "att-1",
     "attachment": "DROP_BOOKKEEPING_attach"},
    {"type": "file-history-snapshot", "messageId": "u-3",
     "isSnapshotUpdate": False,
     "snapshot": "DROP_FILE_SNAPSHOT_cc"},
    {"type": "system", "sessionId": "sess-9999",
     "timestamp": 1710000005000,
     "content": "DROP_BOOKKEEPING_system"},
    {"type": "last-prompt", "sessionId": "sess-9999",
     "leafUuid": "u-3",
     "lastPrompt": "DROP_BOOKKEEPING_last_prompt"},
]


@pytest.fixture
def claude_code_session_file(tmp_path):
    f = tmp_path / "cc-session.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in CC_SESSION_LINES) + "\n",
                 encoding="utf-8")
    return f


class TestClaudeCodeSession:
    def test_single_document(self, claude_code_session_file):
        docs = ingest(claude_code_session_file)
        assert len(docs) == 1
        assert docs[0].source_type == SourceType.CLAUDE_CODE_SESSION

    def test_typed_prompts_kept(self, claude_code_session_file):
        d = ingest(claude_code_session_file)[0]
        assert "KEEP_HUMAN_TURN_cc_1" in d.content
        assert "KEEP_HUMAN_TURN_cc_2" in d.content
        assert "KEEP_SESSION_TITLE" in d.content

    def test_everything_else_stripped(self, claude_code_session_file):
        _assert_no_drop_markers(ingest(claude_code_session_file))

    def test_session_metadata_captured(self, claude_code_session_file):
        d = ingest(claude_code_session_file)[0]
        assert d.metadata["session_id"] == "sess-9999"
        assert d.metadata["cwd"] == "/workspace/proj"
        assert d.metadata["n_typed_turns"] == 2


# ── Claude Code — typed-prompt history JSONL ──────────────────────────────────

# Timestamps in ms-epoch chosen to span two calendar months
# (2024-03 + 2024-04) so per-month bucketing is exercised.
CC_HISTORY_LINES = [
    {"display": "KEEP_HUMAN_TURN_h1",
     "pastedContents": {"1": {"id": "p1", "type": "text",
                              "contentHash": "DROP_PASTED_h1"}},
     "timestamp": 1709942400000,   # 2024-03-09
     "project": "proj_alpha", "sessionId": "s-1"},
    {"display": "KEEP_HUMAN_TURN_h2",
     "pastedContents": {},
     "timestamp": 1710460800000,   # 2024-03-15
     "project": "proj_alpha", "sessionId": "s-1"},
    {"display": "KEEP_HUMAN_TURN_h3",
     "pastedContents": {},
     "timestamp": 1712275200000,   # 2024-04-05
     "project": "proj_beta", "sessionId": "s-2"},
    # empty display drops
    {"display": "",
     "pastedContents": {},
     "timestamp": 1712361600000,   # 2024-04-06
     "project": "proj_beta", "sessionId": "s-2"},
]


@pytest.fixture
def claude_code_history_file(tmp_path):
    f = tmp_path / "cc-history.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in CC_HISTORY_LINES) + "\n",
                 encoding="utf-8")
    return f


class TestClaudeCodeHistory:
    def test_bucketed_by_month(self, claude_code_history_file):
        docs = ingest(claude_code_history_file)
        # Two months represented → two Documents.
        assert len(docs) == 2
        months = {d.metadata["month"] for d in docs}
        assert months == {"2024-03", "2024-04"}

    def test_typed_prompts_kept(self, claude_code_history_file):
        docs = ingest(claude_code_history_file)
        all_content = "\n".join(d.content for d in docs)
        for n in (1, 2, 3):
            assert f"KEEP_HUMAN_TURN_h{n}" in all_content

    def test_pasted_contents_stripped(self, claude_code_history_file):
        _assert_no_drop_markers(ingest(claude_code_history_file))

    def test_project_prefix_inline(self, claude_code_history_file):
        # Project anchor is now an inline prefix on each prompt, not a
        # per-Document axis. Both monthly docs must surface project info.
        docs = {d.metadata["month"]: d for d in ingest(claude_code_history_file)}
        assert "[proj_alpha]" in docs["2024-03"].content
        assert "[proj_beta]" in docs["2024-04"].content

    def test_metadata_counts_and_projects(self, claude_code_history_file):
        docs = {d.metadata["month"]: d for d in ingest(claude_code_history_file)}
        assert docs["2024-03"].metadata["n_prompts"] == 2
        assert docs["2024-03"].metadata["projects"] == ["proj_alpha"]
        assert docs["2024-04"].metadata["n_prompts"] == 1
        assert docs["2024-04"].metadata["projects"] == ["proj_beta"]


# ── Codex — per-session JSONL ─────────────────────────────────────────────────

CODEX_SESSION_LINES = [
    {"timestamp": "2024-03-15T10:00:00Z", "type": "session_meta",
     "payload": {"id": "codex-sess-1", "cwd": "/workspace/codex_proj",
                 "cli_version": "0.x", "model_provider": "openai",
                 "originator": "cli", "source": "DROP_BOOKKEEPING_source",
                 "timestamp": "2024-03-15T10:00:00Z",
                 "git": {}, "base_instructions": "DROP_BOOKKEEPING_base"}},
    # turn_context bookkeeping — drop
    {"timestamp": "2024-03-15T10:00:01Z", "type": "turn_context",
     "payload": {"turn_id": "t-1",
                 "summary": "DROP_BOOKKEEPING_turn_summary",
                 "model": "gpt", "cwd": "/x"}},
    # event_msg/task_started — drop
    {"timestamp": "2024-03-15T10:00:02Z", "type": "event_msg",
     "payload": {"type": "task_started", "turn_id": "t-1",
                 "started_at": "2024-03-15T10:00:02Z",
                 "model_context_window": 100000,
                 "collaboration_mode_kind": "DROP_BOOKKEEPING_mode"}},
    # event_msg/user_message — KEEP
    {"timestamp": "2024-03-15T10:00:03Z", "type": "event_msg",
     "payload": {"type": "user_message", "message": "KEEP_HUMAN_TURN_cx_1",
                 "images": [], "local_images": [], "text_elements": []}},
    # response_item/message role=user — DROP (duplicate + env-ctx)
    {"timestamp": "2024-03-15T10:00:03Z", "type": "response_item",
     "payload": {"type": "message", "role": "user",
                 "content": [{"type": "input_text",
                              "text": "<environment_context>DROP_ENV_CONTEXT_cx</environment_context>"}]}},
    # event_msg/agent_message — DROP
    {"timestamp": "2024-03-15T10:00:04Z", "type": "event_msg",
     "payload": {"type": "agent_message",
                 "message": "DROP_AGENT_MSG_cx",
                 "phase": "final", "memory_citation": None}},
    # response_item/message role=assistant — DROP
    {"timestamp": "2024-03-15T10:00:04Z", "type": "response_item",
     "payload": {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text",
                              "text": "DROP_ASSISTANT_cx_reply"}]}},
    # response_item/reasoning — DROP
    {"timestamp": "2024-03-15T10:00:04Z", "type": "response_item",
     "payload": {"type": "reasoning",
                 "content": [{"text": "DROP_REASONING_cx"}],
                 "encrypted_content": "x", "summary": []}},
    # function_call — DROP
    {"timestamp": "2024-03-15T10:00:05Z", "type": "response_item",
     "payload": {"type": "function_call", "call_id": "fc-1",
                 "name": "shell",
                 "arguments": "DROP_FUNCTION_CALL_cx"}},
    # exec_command_end — DROP
    {"timestamp": "2024-03-15T10:00:05Z", "type": "event_msg",
     "payload": {"type": "exec_command_end", "call_id": "fc-1",
                 "command": ["ls"], "cwd": "/x", "duration": 1, "exit_code": 0,
                 "stdout": "DROP_EXEC_OUTPUT_cx", "stderr": "",
                 "aggregated_output": "DROP_EXEC_OUTPUT_cx_agg",
                 "formatted_output": "DROP_EXEC_OUTPUT_cx_fmt",
                 "parsed_cmd": [], "process_id": 1,
                 "source": "x", "status": "ok", "turn_id": "t-1"}},
    # function_call_output — DROP
    {"timestamp": "2024-03-15T10:00:05Z", "type": "response_item",
     "payload": {"type": "function_call_output", "call_id": "fc-1",
                 "output": "DROP_FUNCTION_CALL_cx_out"}},
    # token_count — DROP
    {"timestamp": "2024-03-15T10:00:06Z", "type": "event_msg",
     "payload": {"type": "token_count",
                 "info": "DROP_TOKEN_COUNT_cx",
                 "rate_limits": {}}},
    # second typed user message
    {"timestamp": "2024-03-15T10:00:07Z", "type": "event_msg",
     "payload": {"type": "user_message", "message": "KEEP_HUMAN_TURN_cx_2",
                 "images": [], "local_images": [], "text_elements": []}},
    # task_complete — DROP
    {"timestamp": "2024-03-15T10:00:08Z", "type": "event_msg",
     "payload": {"type": "task_complete", "turn_id": "t-1",
                 "completed_at": "2024-03-15T10:00:08Z", "duration_ms": 6000,
                 "last_agent_message": "DROP_AGENT_MSG_cx_last"}},
]


@pytest.fixture
def codex_session_file(tmp_path):
    f = tmp_path / "rollout-2024-03-15.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in CODEX_SESSION_LINES) + "\n",
                 encoding="utf-8")
    return f


class TestCodexSession:
    def test_single_document(self, codex_session_file):
        docs = ingest(codex_session_file)
        assert len(docs) == 1
        assert docs[0].source_type == SourceType.CODEX_SESSION

    def test_typed_prompts_kept(self, codex_session_file):
        d = ingest(codex_session_file)[0]
        assert "KEEP_HUMAN_TURN_cx_1" in d.content
        assert "KEEP_HUMAN_TURN_cx_2" in d.content

    def test_all_other_classes_stripped(self, codex_session_file):
        _assert_no_drop_markers(ingest(codex_session_file))

    def test_session_metadata(self, codex_session_file):
        d = ingest(codex_session_file)[0]
        assert d.metadata["session_id"] == "codex-sess-1"
        assert d.metadata["cwd"] == "/workspace/codex_proj"
        assert d.metadata["n_typed_turns"] == 2


# ── Codex — typed-prompt history JSONL ────────────────────────────────────────

# Timestamps in s-epoch chosen to span two calendar months
# (2024-03 + 2024-04) so per-month bucketing is exercised.
CODEX_HISTORY_LINES = [
    {"session_id": "codex-sess-A", "text": "KEEP_HUMAN_TURN_cxh_1",
     "ts": 1709942400},   # 2024-03-09
    {"session_id": "codex-sess-A", "text": "KEEP_HUMAN_TURN_cxh_2",
     "ts": 1710460800},   # 2024-03-15
    {"session_id": "codex-sess-B", "text": "KEEP_HUMAN_TURN_cxh_3",
     "ts": 1712275200},   # 2024-04-05
    # empty text drops
    {"session_id": "codex-sess-B", "text": "", "ts": 1712361600},
]


@pytest.fixture
def codex_history_file(tmp_path):
    f = tmp_path / "codex-history.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in CODEX_HISTORY_LINES) + "\n",
                 encoding="utf-8")
    return f


class TestCodexHistory:
    def test_bucketed_by_month(self, codex_history_file):
        docs = ingest(codex_history_file)
        assert len(docs) == 2
        months = {d.metadata["month"] for d in docs}
        assert months == {"2024-03", "2024-04"}

    def test_typed_prompts_kept(self, codex_history_file):
        docs = ingest(codex_history_file)
        all_content = "\n".join(d.content for d in docs)
        for n in (1, 2, 3):
            assert f"KEEP_HUMAN_TURN_cxh_{n}" in all_content

    def test_session_prefix_inline(self, codex_history_file):
        docs = {d.metadata["month"]: d for d in ingest(codex_history_file)}
        assert "[session codex-se]" in docs["2024-03"].content
        assert "[session codex-se]" in docs["2024-04"].content

    def test_metadata_counts(self, codex_history_file):
        docs = {d.metadata["month"]: d for d in ingest(codex_history_file)}
        assert docs["2024-03"].metadata["n_prompts"] == 2
        assert docs["2024-03"].metadata["n_sessions"] == 1
        assert docs["2024-04"].metadata["n_prompts"] == 1
        assert docs["2024-04"].metadata["n_sessions"] == 1

    def test_no_drop_markers(self, codex_history_file):
        _assert_no_drop_markers(ingest(codex_history_file))


# ── ChatGPT web — conversations.json ──────────────────────────────────────────

CHATGPT_CONVERSATIONS = [
    {
        "title": "KEEP_CHATGPT_TITLE_alpha",
        "create_time": 1710000000.0,
        "update_time": 1710000600.0,
        "conversation_id": "cg-conv-aaaa",
        "mapping": {
            "n1": {
                "id": "n1", "parent": None, "children": ["n2"],
                "message": {
                    "id": "n1",
                    "author": {"role": "system"},
                    "content": {"content_type": "text",
                                "parts": ["DROP_BOOKKEEPING_chatgpt_system"]},
                    "create_time": 1710000000.0,
                },
            },
            "n2": {
                "id": "n2", "parent": "n1", "children": ["n3"],
                "message": {
                    "id": "n2",
                    "author": {"role": "user"},
                    "content": {"content_type": "text",
                                "parts": ["KEEP_HUMAN_TURN_cgt_1"]},
                    "create_time": 1710000010.0,
                },
            },
            "n3": {
                "id": "n3", "parent": "n2", "children": ["n4"],
                "message": {
                    "id": "n3",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text",
                                "parts": ["DROP_ASSISTANT_chatgpt"]},
                    "create_time": 1710000020.0,
                },
            },
            "n4": {
                "id": "n4", "parent": "n3", "children": [],
                "message": {
                    "id": "n4",
                    "author": {"role": "user"},
                    "content": {"content_type": "text",
                                "parts": ["KEEP_HUMAN_TURN_cgt_2"]},
                    "create_time": 1710000030.0,
                },
            },
        },
    },
]


@pytest.fixture
def chatgpt_conversations_file(tmp_path):
    f = tmp_path / "chatgpt-conversations.json"
    f.write_text(json.dumps(CHATGPT_CONVERSATIONS), encoding="utf-8")
    return f


class TestChatGPTConversations:
    def test_document_per_conversation(self, chatgpt_conversations_file):
        docs = ingest(chatgpt_conversations_file)
        assert len(docs) == 1
        assert docs[0].source_type == SourceType.CHATGPT_CONVERSATION

    def test_typed_prompts_kept_and_ordered(self, chatgpt_conversations_file):
        d = ingest(chatgpt_conversations_file)[0]
        assert "KEEP_HUMAN_TURN_cgt_1" in d.content
        assert "KEEP_HUMAN_TURN_cgt_2" in d.content
        # Order matters — first message must appear before the second.
        assert d.content.index("KEEP_HUMAN_TURN_cgt_1") < \
            d.content.index("KEEP_HUMAN_TURN_cgt_2")

    def test_strip_drops_assistant_and_system(self, chatgpt_conversations_file):
        _assert_no_drop_markers(ingest(chatgpt_conversations_file))

    def test_title_and_metadata(self, chatgpt_conversations_file):
        d = ingest(chatgpt_conversations_file)[0]
        assert "KEEP_CHATGPT_TITLE_alpha" in d.content
        assert d.metadata["conversation_id"] == "cg-conv-aaaa"
        assert d.metadata["n_typed_turns"] == 2


# ── Shape sniff edge cases ────────────────────────────────────────────────────

class TestShapeSniff:
    def test_dayone_is_NOT_claimed_by_chat_router(self):
        # An `entries` dict is Day One; chat_exports must return None so
        # the ingestor falls through to the existing Day One parser.
        assert detect_json_format({"entries": []}) is None

    def test_empty_jsonl_line_returns_none(self):
        assert detect_jsonl_format("") is None

    def test_malformed_jsonl_line_returns_none(self):
        assert detect_jsonl_format("not-json") is None

    def test_claude_web_conv_list_detected(self):
        sample = [{
            "uuid": "u", "name": "n", "summary": "s",
            "chat_messages": [], "account": {}, "created_at": "",
            "updated_at": "",
        }]
        assert detect_json_format(sample) == SourceType.CLAUDE_WEB_CONVERSATION

    def test_chatgpt_conv_list_detected(self):
        sample = [{"title": "t", "mapping": {}, "create_time": 0.0}]
        assert detect_json_format(sample) == SourceType.CHATGPT_CONVERSATION

    def test_claude_web_project_dict_detected(self):
        sample = {"uuid": "u", "name": "n", "description": "d",
                  "prompt_template": "p"}
        assert detect_json_format(sample) == SourceType.CLAUDE_WEB_PROJECT

    def test_claude_code_history_jsonl_detected(self):
        line = json.dumps({"display": "x", "pastedContents": {},
                           "timestamp": 1, "project": "p",
                           "sessionId": "s"})
        assert detect_jsonl_format(line) == SourceType.CLAUDE_CODE_HISTORY

    def test_claude_code_session_jsonl_detected(self):
        line = json.dumps({"type": "custom-title", "sessionId": "s",
                           "customTitle": "t"})
        assert detect_jsonl_format(line) == SourceType.CLAUDE_CODE_SESSION

    def test_codex_history_jsonl_detected(self):
        line = json.dumps({"session_id": "s", "text": "x", "ts": 1})
        assert detect_jsonl_format(line) == SourceType.CODEX_HISTORY

    def test_codex_session_jsonl_detected(self):
        line = json.dumps({"timestamp": "t", "type": "session_meta",
                           "payload": {}})
        assert detect_jsonl_format(line) == SourceType.CODEX_SESSION


# ── End-to-end on a directory mixing all formats ──────────────────────────────

class TestMixedDirectoryIngest:
    def test_all_formats_coexist(
        self, claude_web_conversations_file, claude_web_project_file,
        claude_code_session_file, claude_code_history_file,
        codex_session_file, codex_history_file,
        chatgpt_conversations_file, tmp_path,
    ):
        # Each fixture wrote into its own tmp_path subtree; build a
        # single dir with all of them to drive `ingest()` end-to-end.
        bundle = tmp_path / "ai_dumps_mix"
        bundle.mkdir()
        for src in (claude_web_conversations_file, claude_web_project_file,
                    claude_code_session_file, claude_code_history_file,
                    codex_session_file, codex_history_file,
                    chatgpt_conversations_file):
            dest = bundle / src.name
            dest.write_bytes(src.read_bytes())
        docs = ingest(bundle)
        types = {d.source_type for d in docs}
        assert SourceType.CLAUDE_WEB_CONVERSATION in types
        assert SourceType.CLAUDE_WEB_PROJECT in types
        assert SourceType.CLAUDE_CODE_SESSION in types
        assert SourceType.CLAUDE_CODE_HISTORY in types
        assert SourceType.CODEX_SESSION in types
        assert SourceType.CODEX_HISTORY in types
        assert SourceType.CHATGPT_CONVERSATION in types
        _assert_no_drop_markers(docs)
