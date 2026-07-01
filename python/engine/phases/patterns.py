"""Patterns phase on the kernel (issue #912).

PATTERNS is a per-topic DEGRADING stage: one LLM call PER TOPIC turns that
topic's grounded facts into patterns, and on a sizing failure the kernel
SAMPLES via ``_confidence_prioritized_halving`` (drop the lowest-confidence
half of the facts, keeping the ``dedup_to_orig`` map in step). The dedup
pre-pass (``_dedup_facts``), the fit-sampling (``_sample_to_fit`` + ``_cap``
+ ``chunk_cap_for_stage``), the prompt build (``_build_messages``), the
parser (``_validate_patterns_or_raise`` / ``_parse_patterns``) and the
sampler (``_confidence_prioritized_halving``) are reused VERBATIM from
``patterns``.

Concurrency: topics are independent, so the phase pre-passes all topics
then fans their root calls out via ``execution_env.run_all(calls)`` (each
topic's sample cascade still ladders per-call internally). All per-topic
state a sample retry needs travels in ``call.context`` (topic, sampled
facts + map, hard_cap, entities_context), and the terminal dedup→orig map
for parsing is recovered from a per-topic dict (distinct key per topic →
safe under concurrent chains), so no mutable phase state is shared across
topic chains.
"""
from __future__ import annotations

from typing import override

from kernel.abstractions import Job, LlmCall, Phase, PhaseResult
from kernel.enums import JobName, LlmStatus, PhaseName
from kernel.execution_env import BoundExecutionEnv, ExecutionEnv

from engine.phases.telemetry_hook import record_call_counts, set_call_category

from engine.llm import dynamic_max_tokens
from engine.patterns import (
    _build_messages,
    _cap,
    _confidence_prioritized_halving,
    _dedup_facts,
    _llm_chunk_cap_for_stage,
    _parse_patterns,
    _sample_to_fit,
    _validate_patterns_or_raise,
)
from engine.tokens import count_tokens


class PatternsPhase(Phase):
    def __init__(self, job: Job):
        super().__init__(job)
        self._data: dict = {}
        # Per-topic chain state (set before each topic's run).
        self._subject: str = "the author"
        # Per-topic terminal dedup→orig map for parse time. Keyed by topic
        # so concurrent topic chains (run_all) never collide — each topic's
        # sample chain is sequential internally, so the last write per topic
        # is its terminal map.
        self._topic_maps: dict[str, list] = {}

    @override
    def name(self) -> PhaseName:
        return PhaseName.PATTERNS

    @override
    def init_from_memory(self, input: PhaseResult) -> bool:
        d = dict(input.data)
        self._data = d
        self._mode = d["mode"]
        self._facts_by_topic = d["facts_by_topic"]
        self._subject = d.get("subject", "the author")
        self._entities_context_by_topic = d.get("entities_context_by_topic") or {}
        return True

    @override
    def init_from_disk(self) -> None:
        raise NotImplementedError("PatternsPhase always inits from memory")

    @override
    def validate(self, payload) -> LlmStatus | None:
        from engine.parse_signals import _EmptyResponse, _ParseError, _SuccessEmpty

        if not isinstance(payload, str):
            return LlmStatus.PARSE_ERROR
        try:
            _validate_patterns_or_raise(payload)
        except _ParseError:
            return LlmStatus.PARSE_ERROR
        except (_SuccessEmpty, _EmptyResponse):
            return LlmStatus.SUCCESS_EMPTY
        return None

    def _build_call(self, ctx: dict, previous: LlmCall | None) -> LlmCall:
        """Build a patterns call from a per-topic context dict. ALL state a
        sample retry needs travels in ``call.context`` (topic, sampled facts
        + map, hard_cap, entities_context) so parallel topic chains don't
        share mutable phase state."""
        messages = _build_messages(
            ctx["sampled"], ctx["topic"], ctx["hard_cap"], self._subject,
            ctx["entities_context"],
        )
        # ``_build_messages`` already returns ``[{role, content}, …]`` — the
        # kernel message shape — so reuse it verbatim.
        call_messages = list(messages)
        user_prompt = messages[1]["content"]
        # Per-topic terminal map for parse time (distinct key per topic →
        # safe under concurrent run_all).
        self._topic_maps[ctx["topic"]] = ctx["sampled_to_orig"]
        max_tokens = dynamic_max_tokens(
            count_tokens(user_prompt), self._mode, stage="patterns"
        )
        return self.new_call(call_messages, max_tokens, previous, context=ctx)

    @override
    def halve_llm_call(self, call: LlmCall) -> list[LlmCall]:
        raise NotImplementedError("patterns samples; it never halves")

    @override
    def sample_llm_call(self, call: LlmCall) -> LlmCall | None:
        ctx = call.context
        new_sampled, new_to_orig = _confidence_prioritized_halving(
            ctx["sampled"], ctx["sampled_to_orig"]
        )
        if len(new_sampled) >= len(ctx["sampled"]):
            # Fact floor reached (can't shed further) — return None to stop
            # (kernel resolves with the last response), mirroring halve's [].
            return None
        next_ctx = {**ctx, "sampled": new_sampled, "sampled_to_orig": new_to_orig}
        return self._build_call(next_ctx, call)

    @override
    def checkpoint(self) -> None:
        pass

    def _prep_topic(self, topic: str, facts) -> dict | None:
        """Deterministic pre-pass for one topic: dedup → fit-sample → build
        the root context dict. Returns None when the topic has too little to
        synthesize."""
        if not facts or len(facts) < 3:
            return None
        deduped, dedup_to_orig = _dedup_facts(facts)
        if len(deduped) < 3:
            return None
        hard_cap = _cap(len(deduped))
        entities_context = self._entities_context_by_topic.get(topic)
        cap = _llm_chunk_cap_for_stage(self._mode, "patterns")
        sampled_idx = _sample_to_fit(
            deduped, topic, hard_cap, self._subject, entities_context, cap,
        )
        if len(sampled_idx) < 3:
            return None
        return {
            "topic": topic,
            "sampled": [deduped[i] for i in sampled_idx],
            "sampled_to_orig": [dedup_to_orig[i] for i in sampled_idx],
            "hard_cap": hard_cap,
            "entities_context": entities_context,
        }

    @override
    def run_main(self, execution_env: BoundExecutionEnv | None) -> PhaseResult:
        assert execution_env is not None
        # Each topic is an INDEPENDENT call-chain — pre-pass all topics, then
        # fan their root calls out via run_all (each topic's own sample
        # cascade still ladders per-call internally). results[i] <-> topic[i].
        topics: list[str] = []
        calls: list[LlmCall] = []
        for topic, facts in self._facts_by_topic.items():
            ctx = self._prep_topic(topic, facts)
            if ctx is None:
                continue
            topics.append(topic)
            call = self._build_call(ctx, None)
            # Per-call category column = the topic (the legacy patterns label).
            set_call_category(execution_env, call, topic)
            calls.append(call)

        def _parse_topic(topic, responses):
            raw = None
            for response in responses:
                if isinstance(response.payload, str) and response.payload.strip():
                    raw = response.payload
            if raw is None:
                return []
            parsed, _parse_error = _parse_patterns(
                raw, topic, self._topic_maps[topic]
            )
            return parsed or []

        # STREAM per-topic persistence: write each topic's patterns file as its
        # call resolves so the run-tree's Patterns node fills in live (legacy
        # did this via on_topic_done). Fire-and-forget + guarded — a raising
        # Future done-callback would be swallowed and orphan the run. Ordered
        # results are collected below off the blocking results, not callbacks.
        on_topic_done = self._data.get("on_topic_done")

        def _stream_persist(topic, fut):
            if on_topic_done is None:
                return
            try:
                on_topic_done(topic, _parse_topic(topic, fut.result()))
            except Exception:
                pass

        futures = [execution_env.run(c) for c in calls]
        for topic, fut in zip(topics, futures):
            fut.add_done_callback(lambda f, t=topic: _stream_persist(t, f))

        results = [fut.result() for fut in futures]
        out: dict[str, list] = {}
        for call, topic, responses in zip(calls, topics, results):
            parsed = _parse_topic(topic, responses)
            if parsed:
                out[topic] = parsed
            # Per-call counts (facts sampled in → patterns out), parity with
            # legacy patterns' record_stage_counts.
            sampled = call.context.get("sampled", []) if isinstance(call.context, dict) else []
            record_call_counts(
                execution_env, call,
                {"facts": len(sampled)}, {"patterns": len(parsed)},
            )
        return PhaseResult({**self._data, "patterns_by_topic": out})


class PatternsJob(Job):
    def __init__(self, initial: dict):
        super().__init__()
        self._initial = initial

    @override
    def name(self) -> JobName:
        return JobName.FULL_PIPELINE

    @override
    def generate_phases(self) -> list[Phase]:
        return [PatternsPhase(self)]

    def initial_input(self) -> PhaseResult:
        return PhaseResult(self._initial)


def detect_patterns(
    facts_by_topic: dict,
    mode,
    execution_env: ExecutionEnv | None = None,
    **kwargs,
) -> dict:
    from engine.phases.model_specs import spec_for_mode

    job = PatternsJob({"facts_by_topic": facts_by_topic, "mode": mode, **kwargs})
    if execution_env is None:
        spec = spec_for_mode(mode)
        env = ExecutionEnv()
        env.register_spec(PhaseName.PATTERNS, spec, spec, False)
    else:
        env = execution_env
    result = job.run(job.initial_input(), env)
    return result.data["patterns_by_topic"]
