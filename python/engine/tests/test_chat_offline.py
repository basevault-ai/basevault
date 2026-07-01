"""Offline validation for the migrated CHAT phase (#912).

Chat is a single LOOPING phase: the kernel runs ``run_main`` once and the
multi-hop ReAct loop (``chatbot_turn.run``) drives itself, taking each
hop's LLM call through the kernel via an injected ``tracked_complete``.

This drives a real turn through ``ChatPhase`` with a scripted provider that
reads the multi-turn messages off ``LlmCall.messages`` and streams a
scripted reply. It proves: the kernel-backed hop call fires, the full
``[{system}, *history, {user}]`` message list reaches the provider, the
``_StreamGate`` streams the prose reply to the UI, and the loop returns the
prose answer in one hop.

Tool-dispatch fidelity (valid tool schema + a real vector store) is the
live chat-eval's job; this isolates the kernel execution + looping wiring.
"""
from __future__ import annotations


from kernel.abstractions import InferenceProvider, LlmResponse
from kernel.enums import PhaseName
from kernel.execution_env import ExecutionEnv

from engine import chatbot_turn
from engine.llm import ModelSpec as LegacyModelSpec, Mode
from engine.phases.chat import run_chat_turn
from engine.phases.model_specs import PipelineModelSpec


class _ScriptedChatProvider(InferenceProvider):
    """Reads the multi-turn messages off ``call.messages`` (list of
    ``(role, content)`` tuples) and streams a scripted reply through
    ``call.stream_handler``."""

    def __init__(self, reply: str):
        self._reply = reply
        self.call_count = 0
        self.seen_messages: list = []
        self._inj = {p: [] for p in PhaseName}

    def name(self):
        return "scripted-chat"

    def run(self, call, execution_env) -> LlmResponse:
        self.call_count += 1
        self.seen_messages.append(call.messages)
        if call.stream_handler:
            call.stream_handler(self._reply)
        return LlmResponse(None, self._reply, None, 0, 0, 0, 0.0, 0.0)

    def inject_errors(self, phase, errors):
        self._inj[phase] += errors


def _ctx(query: str, history=None):
    return chatbot_turn.TurnContext(
        query=query,
        history=history or [],
        turn_index=0,
        session_id="sess-test",
        store_path=None,
        bound_run=None,
        chatbot_config={"model": "scripted", "reasoning": False},
        tracked_complete=lambda *a, **k: None,  # replaced by ChatPhase
        emit=lambda *a, **k: _EVENTS.append((a, k)),
    )


_EVENTS: list = []


def _run(reply: str, query="hello", history=None):
    _EVENTS.clear()
    provider = _ScriptedChatProvider(reply)
    legacy = LegacyModelSpec(
        provider="scripted", model_id="gpt-oss-120b", context_window=131_000
    )
    spec = PipelineModelSpec(provider, legacy.model_id, legacy.context_window, max_parallelism=4)
    env = ExecutionEnv()
    env.register_spec(PhaseName.CHAT, spec, spec, thinking=False)
    result = run_chat_turn(_ctx(query, history), Mode.TEE, env)
    return result, provider


def test_chat_prose_answer_single_hop():
    result, provider = _run("Hi there — here is your answer.")
    # One decision hop, voluntary prose finalize.
    assert provider.call_count == 1
    assert result.hops == 1
    assert result.answer == "Hi there — here is your answer."
    assert result.lookup_fired is False


def test_chat_passes_multi_turn_messages_to_provider():
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    _result, provider = _run("done", query="follow up", history=history)
    msgs = provider.seen_messages[0]
    assert isinstance(msgs, list)
    roles = [m["role"] for m in msgs]  # {role, content} dicts
    # [system, *history(user/assistant), user] — the multi-turn shape now
    # carried verbatim on LlmCall.messages.
    assert roles[0] == "system"
    assert "assistant" in roles  # history carried through
    assert msgs[-1] == {"role": "user", "content": "follow up"}


def test_chat_streams_prose_to_ui():
    _run("streamed prose reply")
    # The _StreamGate streamed the prose to the UI (chatbot_chunk events).
    chunk_events = [k for a, k in _EVENTS if a and a[0] == "chatbot_chunk"]
    assert chunk_events, "prose reply should stream chatbot_chunk events"
