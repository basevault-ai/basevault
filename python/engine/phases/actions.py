"""Actions phase on the kernel (issue #912).

ACTIONS is a single-call DEGRADING stage, structurally parallel to
INSIGHTS: one LLM call plans actions from the insights, and on a sizing
failure the kernel SAMPLES — ``_next_pattern_detail_keys`` sheds
descriptions from the bottom 50% of the *pattern context* block (the
insights themselves stay whole; every pattern stays visible by name +
count). The cap (``action_cap``), prompt build (``_build_prompt`` + the
pattern-context block via ``patterns.build_context_block`` + ``_SYSTEM``/
``_SUBJECT_DISCIPLINE``/``_EVENT_VS_HORIZON``/``_HARM_CLAUSE``/sentiment),
the parser (``_validate_actions_or_raise`` / ``_parse_output``) and the
sampler (``_next_pattern_detail_keys``) are reused VERBATIM from
``actions``.

The source-index-map is invariant under sampling (it indexes the
insights, which never change), so the phase parses the single terminal
leaf against the once-built map. At the pattern-detail floor
``sample_llm_call`` returns ``None`` to stop (mirrors ``halve``'s ``[]``).
"""
from __future__ import annotations

from typing import override

from kernel.abstractions import Job, LlmCall, Phase, PhaseResult
from kernel.enums import JobName, LlmStatus, PhaseName
from kernel.execution_env import BoundExecutionEnv, ExecutionEnv

from engine.actions import (
    _build_prompt,
    _EVENT_VS_HORIZON,
    _HARM_CLAUSE,
    _next_pattern_detail_keys,
    _parse_output,
    _SENTIMENT_BIAS_CLAUSES,
    _SENTIMENT_FRAMING,
    _SUBJECT_DISCIPLINE,
    _SYNTHESIS_INPUT_FLOOR,
    _SYSTEM,
    _validate_actions_or_raise,
    action_cap,
)
from engine.llm import dynamic_max_tokens
from engine.patterns import build_context_block as _patterns_context_block
from engine.tokens import count_tokens


class ActionsPhase(Phase):
    def __init__(self, job: Job):
        super().__init__(job)
        self._data: dict = {}
        self._index_map: list = []

    @override
    def name(self) -> PhaseName:
        return PhaseName.ACTIONS

    @override
    def init_from_memory(self, input: PhaseResult) -> bool:
        d = dict(input.data)
        self._data = d
        self._mode = d["mode"]
        self._insight_output = d["insight_output"]
        self._entities_context = d.get("entities_context")
        self._patterns_by_topic = d.get("patterns_by_topic")
        self._today = d["today"]
        self._max_actions = action_cap(d.get("total_facts", 1))

        sentiment = d.get("sentiment", "neutral")
        subject = d.get("subject", "the author")
        sentiment_clause = _SENTIMENT_BIAS_CLAUSES.get(
            sentiment, _SENTIMENT_BIAS_CLAUSES["neutral"]
        )
        sentiment_block = _SENTIMENT_FRAMING.format(sentiment_clause=sentiment_clause)
        self._system_content = "\n\n".join(
            [
                _SYSTEM,
                _SUBJECT_DISCIPLINE.format(subject=subject),
                _EVENT_VS_HORIZON,
                sentiment_block,
                _HARM_CLAUSE,
            ]
        )
        return True

    @override
    def init_from_disk(self) -> None:
        raise NotImplementedError("ActionsPhase always inits from memory")

    @override
    def validate(self, payload) -> LlmStatus | None:
        from engine.parse_signals import _EmptyResponse, _ParseError, _SuccessEmpty

        if not isinstance(payload, str):
            return LlmStatus.PARSE_ERROR
        try:
            _validate_actions_or_raise(payload)
        except _ParseError:
            return LlmStatus.PARSE_ERROR
        except (_SuccessEmpty, _EmptyResponse):
            return LlmStatus.SUCCESS_EMPTY
        return None

    def _build_call(self, detail_keys, previous: LlmCall | None) -> LlmCall:
        prompt, idx_map = _build_prompt(self._insight_output, self._max_actions)
        ctx = (
            _patterns_context_block(self._patterns_by_topic, detail_keys=detail_keys)
            if self._patterns_by_topic
            else None
        )
        if ctx:
            prompt = ctx + "\n\n" + prompt
        if self._entities_context:
            prompt = self._entities_context + "\n\n" + prompt
        self._index_map = idx_map
        max_tokens = dynamic_max_tokens(
            count_tokens(prompt), self._mode, stage="actions"
        )
        return self.new_call(
            [
                {"role": "system", "content": self._system_content},
                {"role": "user", "content": prompt},
            ],
            max_tokens,
            previous,
            context=detail_keys,
        )

    @override
    def halve_llm_call(self, call: LlmCall) -> list[LlmCall]:
        raise NotImplementedError("actions samples; it never halves")

    @override
    def sample_llm_call(self, call: LlmCall) -> LlmCall | None:
        nxt = _next_pattern_detail_keys(self._patterns_by_topic, call.context)
        if nxt is None or len(nxt) < _SYNTHESIS_INPUT_FLOOR:
            # Pattern detail set floored — return None to stop (kernel
            # resolves with the last response), mirroring halve's [].
            return None
        return self._build_call(nxt, call)

    @override
    def checkpoint(self) -> None:
        pass

    @override
    def run_main(self, execution_env: BoundExecutionEnv | None) -> PhaseResult:
        assert execution_env is not None
        io = self._insight_output
        if (not io.cross_domain and not io.critical) or self._max_actions <= 0:
            return PhaseResult({**self._data, "actions": [], "action_list": []})

        call = self._build_call(None, None)
        responses = execution_env.run(call).result()
        raw = None
        for response in responses:
            if isinstance(response.payload, str) and response.payload.strip():
                raw = response.payload
        if raw is None:
            return PhaseResult({**self._data, "actions": [], "action_list": []})

        parsed, _parse_error = _parse_output(
            raw, self._index_map, self._today, self._max_actions
        )
        return PhaseResult({**self._data, "actions": parsed, "action_list": parsed})


class ActionsJob(Job):
    def __init__(self, initial: dict):
        super().__init__()
        self._initial = initial

    @override
    def name(self) -> JobName:
        return JobName.FULL_PIPELINE

    @override
    def generate_phases(self) -> list[Phase]:
        return [ActionsPhase(self)]

    def initial_input(self) -> PhaseResult:
        return PhaseResult(self._initial)


def generate_actions(
    insight_output,
    mode,
    today,
    execution_env: ExecutionEnv | None = None,
    **kwargs,
) -> list:
    from engine.phases.model_specs import spec_for_mode

    job = ActionsJob(
        {"insight_output": insight_output, "mode": mode, "today": today, **kwargs}
    )
    if execution_env is None:
        spec = spec_for_mode(mode)
        env = ExecutionEnv()
        env.register_spec(PhaseName.ACTIONS, spec, spec, False)
    else:
        env = execution_env
    result = job.run(job.initial_input(), env)
    return result.data["actions"]
