"""Chat phase on the kernel (issue #912).

Chat is a Job with a SINGLE LOOPING phase: the multi-hop ReAct loop
(decide → tool-dispatch → loop → answer) lives in
``chatbot_turn.run(ctx)``. The kernel runs ``run_main`` once; the looping
is the phase's own concern — it drives ``chatbot_turn.run`` directly.

The migration is surgical: ``chatbot_turn.run`` already takes its single
LLM call through an INJECTED ``ctx.tracked_complete`` callable (the only
I/O seam in an otherwise pure loop). This phase injects a KERNEL-backed
``tracked_complete`` so every hop's call routes through
``execution_env.run`` (kernel scheduling + per-call retry), while the
personas, the ``_StreamGate`` (via ``call.stream_handler``), tool parse /
dispatch, accumulation, and diagnostics are all reused VERBATIM.

Multi-turn messages: chat prompts are ``[{system}, *history, {user}]``
(``chatbot.build_chat_prompt``). The kernel's ``LlmCall.messages``
(``list[tuple[role, content]]``, landed on main) carries them verbatim and
the production providers send them, so chat runs on the REAL kernel path —
no flattening (which would not be behavior-neutral; cf. spike finding #4).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import override

from kernel.abstractions import Job, LlmCall, Phase, PhaseResult
from kernel.enums import JobName, LlmStatus, PhaseName
from kernel.execution_env import BoundExecutionEnv, ExecutionEnv

from engine import chatbot_turn
from engine.llm import dynamic_max_tokens, estimate_prompt_tokens


@dataclass
class _ChatResult:
    """Minimal ``CompletionResult`` stand-in — ``chatbot_turn.run`` reads
    only ``.content`` off the ``tracked_complete`` return value."""

    content: str


class ChatPhase(Phase):
    def __init__(self, job: Job):
        super().__init__(job)
        self._ctx = None
        self._mode = None

    @override
    def name(self) -> PhaseName:
        return PhaseName.CHAT

    @override
    def init_from_memory(self, input: PhaseResult) -> bool:
        self._ctx = input.data["ctx"]
        self._mode = input.data["mode"]
        return True

    @override
    def init_from_disk(self) -> None:
        raise NotImplementedError("ChatPhase always inits from memory")

    @override
    def validate(self, payload) -> LlmStatus | None:
        # A chat reply is free-form (tool-call JSON or prose) — any
        # non-empty string is OK; the app loop classifies it. Empty on the
        # wire is caught by compute_status (→ LOAD) before validate runs.
        if not isinstance(payload, str) or not payload.strip():
            return LlmStatus.SUCCESS_EMPTY
        return None

    @override
    def halve_llm_call(self, call: LlmCall) -> list[LlmCall]:
        return []  # chat does not halve

    @override
    def sample_llm_call(self, call: LlmCall) -> LlmCall | None:
        return None  # chat does not sample — stop on the rare sizing failure

    @override
    def checkpoint(self) -> None:
        pass

    def _kernel_tracked_complete(self, execution_env: BoundExecutionEnv):
        """Build a ``tracked_complete`` that routes each hop's call through
        the kernel. Mirrors the sidecar's ``_tracked_complete`` contract
        (``messages, *, _chatbot_stage, _chatbot_category, **kwargs`` →
        object with ``.content``)."""

        def _tc(messages, *, _chatbot_stage="chatbot",
                _chatbot_category="chatbot_answer", on_chunk=None, **_kwargs):
            try:
                payload_tokens = int(estimate_prompt_tokens(messages))
            except Exception:
                payload_tokens = 0
            max_tokens = dynamic_max_tokens(
                payload_tokens, self._mode, stage="chatbot"
            )
            # The hop's full ``[{system}, *history, {user}]`` dict list is
            # already the kernel message shape — carried verbatim.
            call = self.new_call(list(messages), max_tokens, None)
            call.stream_handler = on_chunk
            responses = execution_env.run(call).result()
            content = ""
            for response in responses:
                if isinstance(response.payload, str) and response.payload:
                    content = response.payload
            return _ChatResult(content=content)

        return _tc

    @override
    def run_main(self, execution_env: BoundExecutionEnv | None) -> PhaseResult:
        assert execution_env is not None
        # Inject the kernel-backed tracked_complete and run the existing
        # multi-hop loop verbatim. ctx is frozen → rebuild with replace().
        ctx = replace(
            self._ctx, tracked_complete=self._kernel_tracked_complete(execution_env)
        )
        result = chatbot_turn.run(ctx)
        return PhaseResult({"result": result})


class ChatJob(Job):
    def __init__(self, ctx, mode):
        super().__init__()
        self._ctx = ctx
        self._mode = mode

    @override
    def name(self) -> JobName:
        return JobName.FULL_PIPELINE

    @override
    def generate_phases(self) -> list[Phase]:
        return [ChatPhase(self)]

    def initial_input(self) -> PhaseResult:
        return PhaseResult({"ctx": self._ctx, "mode": self._mode})


def run_chat_turn(ctx, mode, execution_env: ExecutionEnv | None = None):
    """Run one chat turn on the kernel; returns the ``TurnResult``. ``ctx``
    is a ``chatbot_turn.TurnContext`` (its ``tracked_complete`` is replaced
    with the kernel-backed one)."""
    from engine.phases.model_specs import chat_spec_for_mode

    job = ChatJob(ctx, mode)
    if execution_env is None:
        # Chat's model is the chatbot-config model (glm-5-2 default), NOT the
        # gpt-oss stage anchor — same resolution build_stage_env(CHAT) uses.
        spec = chat_spec_for_mode(mode)
        env = ExecutionEnv()
        env.register_spec(PhaseName.CHAT, spec, spec, False)
    else:
        env = execution_env
    result = job.run(job.initial_input(), env)
    return result.data["result"]
