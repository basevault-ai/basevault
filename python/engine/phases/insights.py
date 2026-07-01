"""Insights phase on the kernel (issue #912).

INSIGHTS is a single-call DEGRADING stage: one LLM call synthesizes both
cross-domain and critical insights from the patterns. On a sizing failure
the kernel's degrading ``RetryPolicy`` SAMPLES — ``_next_detail_keys``
sheds descriptions from the bottom 50% of patterns (every pattern stays
visible by id + name; only detail is trimmed), never dropping a pattern.
The cap math (``insight_caps``), prompt build (``_build_prompt`` +
``_SYSTEM``/``_SUBJECT_DISCIPLINE``/``_EVENT_VS_INSIGHT``/sentiment), the
parser (``_validate_insights_or_raise`` / ``_parse_output``), and the
sampler (``_next_detail_keys``) are all reused VERBATIM from ``insights``.

Single-call degrading shape: the kernel's sample retry REPLACES the call
(it doesn't fan out), so ``execution_env.run(call).result()`` returns
exactly one terminal leaf, and the phase parses it against the index-map of
the last-built call (``_next_detail_keys`` keeps the map stable — it only
trims descriptions). At the synthesis floor (``_next_detail_keys`` → None)
``sample_llm_call`` returns ``None`` to stop — the kernel resolves with the
last response (mirrors ``halve_llm_call``'s ``[]``).
"""
from __future__ import annotations

from typing import override

from kernel.abstractions import Job, LlmCall, Phase, PhaseResult
from kernel.enums import LlmStatus, PhaseName
from kernel.execution_env import BoundExecutionEnv

from engine.insights import (
    InsightOutput,
    _build_prompt,
    _CRITICAL_ONLY_NOTE,
    _EVENT_VS_INSIGHT,
    _next_detail_keys,
    _parse_output,
    _SENTIMENT_BIAS_CLAUSES,
    _SENTIMENT_FRAMING,
    _SUBJECT_DISCIPLINE,
    _SYSTEM,
    _validate_insights_or_raise,
    insight_caps,
)
from engine.llm import dynamic_max_tokens
from engine.tokens import count_tokens


class InsightsPhase(Phase):
    def __init__(self, job: Job):
        super().__init__(job)
        self._data: dict = {}
        self._last_index_map: list = []

    @override
    def name(self) -> PhaseName:
        return PhaseName.INSIGHTS

    @override
    def init_from_memory(self, input: PhaseResult) -> bool:
        d = dict(input.data)
        self._data = d
        self._mode = d["mode"]
        self._strong = {t: ps for t, ps in d["patterns_by_topic"].items() if ps}
        self._entities_context = d.get("entities_context")
        total_facts = d.get("total_facts", 1)
        min_topics = d.get("min_topics", 2)
        sentiment = d.get("sentiment", "neutral")
        subject = d.get("subject", "the author")

        self._total_cap, self._cross_cap, self._critical_cap = insight_caps(
            total_facts
        )
        self._critical_only = len(self._strong) < min_topics
        if self._critical_only:
            self._critical_cap = self._total_cap
            self._cross_cap = 0

        sentiment_clause = _SENTIMENT_BIAS_CLAUSES.get(
            sentiment, _SENTIMENT_BIAS_CLAUSES["neutral"]
        )
        sentiment_block = _SENTIMENT_FRAMING.format(sentiment_clause=sentiment_clause)
        self._system_content = (
            _SYSTEM
            + "\n\n" + _SUBJECT_DISCIPLINE.format(subject=subject)
            + "\n\n" + _EVENT_VS_INSIGHT
            + "\n\n" + sentiment_block
        )
        return True

    @override
    def init_from_disk(self) -> None:
        raise NotImplementedError("InsightsPhase always inits from memory")

    @override
    def validate(self, payload) -> LlmStatus | None:
        from engine.parse_signals import _EmptyResponse, _ParseError, _SuccessEmpty

        if not isinstance(payload, str):
            return LlmStatus.PARSE_ERROR
        try:
            _validate_insights_or_raise(payload)
        except _ParseError:
            return LlmStatus.PARSE_ERROR
        except (_SuccessEmpty, _EmptyResponse):
            return LlmStatus.SUCCESS_EMPTY
        return None

    def _build_call(self, detail_keys, previous: LlmCall | None) -> LlmCall:
        prompt, index_map = _build_prompt(
            self._strong, self._cross_cap, self._critical_cap, detail_keys
        )
        if self._entities_context:
            prompt = self._entities_context + "\n\n" + prompt
        if self._critical_only:
            prompt = prompt + _CRITICAL_ONLY_NOTE
        self._last_index_map = index_map
        max_tokens = dynamic_max_tokens(
            count_tokens(prompt), self._mode, stage="insights"
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
        raise NotImplementedError("insights samples; it never halves")

    @override
    def sample_llm_call(self, call: LlmCall) -> LlmCall | None:
        nxt = _next_detail_keys(self._strong, call.context)
        if nxt is None:
            # At the synthesis floor — no further degradation possible.
            # Return None to stop (the kernel resolves with the last
            # response), mirroring halve_llm_call's [].
            return None
        return self._build_call(nxt, call)

    @override
    def checkpoint(self) -> None:
        pass

    @override
    def run_main(self, execution_env: BoundExecutionEnv | None) -> PhaseResult:
        assert execution_env is not None
        if not self._strong or self._total_cap <= 0:
            return PhaseResult(
                {**self._data, "output": InsightOutput(),
                 "insight_output": InsightOutput()}
            )

        call = self._build_call(None, None)
        responses = execution_env.run(call).result()
        raw = None
        for response in responses:
            if isinstance(response.payload, str) and response.payload.strip():
                raw = response.payload
        if raw is None:
            return PhaseResult(
                {**self._data, "output": InsightOutput(),
                 "insight_output": InsightOutput()}
            )

        parsed, _parse_error = _parse_output(
            raw, self._last_index_map, self._cross_cap, self._critical_cap
        )
        return PhaseResult({**self._data, "output": parsed, "insight_output": parsed})


class InsightsJob(Job):
    def __init__(self, initial: dict):
        super().__init__()
        self._initial = initial

    @override
    def name(self):
        from kernel.enums import JobName

        return JobName.FULL_PIPELINE

    @override
    def generate_phases(self) -> list[Phase]:
        return [InsightsPhase(self)]

    def initial_input(self) -> PhaseResult:
        return PhaseResult(self._initial)


def detect_insights(
    patterns_by_topic: dict,
    mode,
    execution_env: BoundExecutionEnv | None = None,
    **kwargs,
) -> InsightOutput:
    from kernel.execution_env import ExecutionEnv

    from engine.phases.model_specs import spec_for_mode

    job = InsightsJob({"patterns_by_topic": patterns_by_topic, "mode": mode, **kwargs})
    if execution_env is None:
        spec = spec_for_mode(mode)
        env = ExecutionEnv()
        env.register_spec(PhaseName.INSIGHTS, spec, spec, False)
    else:
        env = execution_env
    result = job.run(job.initial_input(), env)
    return result.data["output"]
