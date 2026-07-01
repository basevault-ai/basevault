"""Sidecar event-surface guards.

Focused on #558: every ``chatbot_done`` must carry the session's bound
``run`` so the UI can PIN it onto the persisted turn. Without the stamp
a restored citation has no run identity of its own and falls back to
transient live-bind state — dead after a restart, mis-targeted across a
run switch. These exercise the pure-conversation path (no retrieval /
store) so the guard stays isolated from the LLM + vector store.
"""
from __future__ import annotations

from pathlib import Path

from engine import chatbot_sidecar as cs


class _FakeResult:
    def __init__(self, content: str) -> None:
        self.content = content


def _capture_emits(monkeypatch):
    events: list[dict] = []

    def fake_emit(event, **payload):
        payload["event"] = event
        events.append(payload)

    monkeypatch.setattr(cs, "_emit", fake_emit)
    monkeypatch.setattr(cs, "_read_app_config", lambda: {})
    return events


def test_chatbot_done_carries_bound_run_on_conversation_turn(monkeypatch):
    events = _capture_emits(monkeypatch)
    # Plain reply, no tool call → the conversation branch.
    monkeypatch.setattr(
        cs, "_tracked_complete",
        lambda *a, **k: _FakeResult("Sure, happy to chat."),
    )
    monkeypatch.setattr(cs, "_SESSION_BOUND_RUN", "run-A")

    cs._run("hello there", [])

    done = [e for e in events if e["event"] == "chatbot_done"]
    assert len(done) == 1
    assert done[0]["resources"] is None
    # The pin: this turn travels with the run it was answered against.
    assert done[0]["run"] == "run-A"


def test_chatbot_done_carries_bound_run_on_empty_query(monkeypatch):
    events = _capture_emits(monkeypatch)
    monkeypatch.setattr(cs, "_SESSION_BOUND_RUN", "run-B")

    cs._run("   ", [])

    done = [e for e in events if e["event"] == "chatbot_done"]
    assert len(done) == 1
    assert done[0]["run"] == "run-B"


def test_telemetry_dir_scopes_to_active_conversation(monkeypatch, tmp_path):
    # #565: when the shell points at the active conversation's dir, this
    # thread's calls/payloads land THERE — not the shared firehose.
    convo = tmp_path / "2026-05-02T13-30-11Z-conversation-3"
    monkeypatch.setenv("BASEVAULT_CHATBOT_CONVO_DIR", str(convo))
    assert cs._telemetry_dir() == convo


def test_telemetry_dir_falls_back_to_chats_root_when_unscoped(
    monkeypatch, tmp_path
):
    # Unset / blank (ad-hoc or older shell) → _chats_root() (chats are
    # OUTSIDE the logs tree post-#568). BASEVAULT_CHATS_ROOT override is
    # honoured exactly like BASEVAULT_LOGS_ROOT for _logs_root().
    monkeypatch.setenv("BASEVAULT_CHATS_ROOT", str(tmp_path))
    monkeypatch.delenv("BASEVAULT_CHATBOT_CONVO_DIR", raising=False)
    assert cs._telemetry_dir() == tmp_path
    monkeypatch.setenv("BASEVAULT_CHATBOT_CONVO_DIR", "   ")
    assert cs._telemetry_dir() == tmp_path


def test_chats_root_mirrors_logs_root_agent_split(monkeypatch):
    # Same agent/dev split as _logs_root(), just chats/ vs chats-dev/.
    monkeypatch.delenv("BASEVAULT_CHATS_ROOT", raising=False)
    monkeypatch.setenv("BASEVAULT_AGENT", "app")
    assert cs._chats_root() == Path.home() / ".basevault" / "chats"
    monkeypatch.setenv("BASEVAULT_AGENT", "")
    assert cs._chats_root() == Path.home() / ".basevault" / "chats-dev"


def test_chatbot_done_run_is_null_when_no_run_bound(monkeypatch):
    # No processed run → bound run is None; the stamp is still present
    # (the UI distinguishes "no run" from "field missing / legacy").
    events = _capture_emits(monkeypatch)
    monkeypatch.setattr(
        cs, "_tracked_complete",
        lambda *a, **k: _FakeResult("Just talking."),
    )
    monkeypatch.setattr(cs, "_SESSION_BOUND_RUN", None)

    cs._run("hi", [])

    done = [e for e in events if e["event"] == "chatbot_done"]
    assert len(done) == 1
    assert "run" in done[0]
    assert done[0]["run"] is None


def test_yaml_block_uses_explicit_indent_indicator_for_safe_first_line():
    """Per-turn ``llm-payloads.yaml`` uses ``|2-`` (literal block, strip
    trailing newlines, explicit indent indicator 2). Without the
    explicit ``2`` a body whose first content line happens to start
    with extra whitespace would mis-set YAML's auto-detected indent and
    the parser would either reject the doc or strip the wrong number
    of leading spaces from every line. Folded in here per slice-2
    reviewer feedback."""
    from engine.llm import _yaml_block

    # Indent 4 (the actual call-site offset for `request:` /
    # `response:` inside the `calls:` list).
    block = _yaml_block("request", "    starts with leading spaces", indent=4)
    assert "|2-\n" in block
    assert "|-\n" not in block

    # The block round-trips through any standard YAML parser without
    # auto-indent ambiguity. We don't import a YAML library here
    # (yaml.safe_load isn't in the unit-test surface); instead, pin
    # the textual shape of the body lines.
    lines = block.splitlines()
    # `    request: |2-`, then body lines indented 6 spaces.
    assert lines[0] == "    request: |2-"
    assert lines[1] == "          starts with leading spaces"
