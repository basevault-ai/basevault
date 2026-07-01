"""Entities Job on the kernel (issue #912).

Wires the three entity phases — grouping (deterministic) → summarize (LLM,
halve) → dedupe (LLM, sample) — into a kernel ``Job`` and exposes
``detect_entities`` returning an ``EntitiesOutput``, consumed by the
runner and the evals.
"""
from __future__ import annotations

from typing import override

from kernel.abstractions import Job, Phase, PhaseResult
from kernel.enums import JobName, PhaseName
from kernel.execution_env import ExecutionEnv

from engine.entities import EntitiesOutput
from engine.llm import Mode

from engine.phases.entity_dedupe import EntityDedupePhase
from engine.phases.entity_grouping import EntityGroupingPhase
from engine.phases.entity_summarize import EntitySummarizePhase


class EntitiesJob(Job):
    def __init__(self, initial: dict):
        super().__init__()
        self._initial = initial

    @override
    def name(self) -> JobName:
        return JobName.FULL_PIPELINE

    @override
    def generate_phases(self) -> list[Phase]:
        return [
            EntityGroupingPhase(self),
            EntitySummarizePhase(self),
            EntityDedupePhase(self),
        ]

    def initial_input(self) -> PhaseResult:
        return PhaseResult(self._initial)


def build_entities_env(
    mode: Mode, thinking: bool = False, extra_hooks=None
) -> ExecutionEnv:
    """Entities runs multiple phases (grouping / summarize / dedupe) under one
    env, so it can't use the single-phase ``build_stage_env`` — but it wires
    the SAME telemetry hook (with the dev-tab payload sink), disk cache, and
    optional live-progress hooks so entities reaches parity with the other
    kernel stages (per-call records, cached=true on re-runs, dev payloads)."""
    from engine.phases.telemetry_hook import KernelTelemetryHook
    from engine.phases.kernel_cache import KernelDiskCache
    from engine.phases.model_specs import spec_for_stage
    from engine.llm import _stamp_full_io, _log_call_failure_payload

    # Each entity phase resolves its OWN model: summarize on the entities model
    # (gpt-oss-120b), dedupe on the entities_dedupe model (gemma4-31b) — they
    # run on separate schedulers with their own pacing.
    sum_spec = spec_for_stage(PhaseName.ENTITY_SUMMARIZE, mode)
    dedupe_spec = spec_for_stage(PhaseName.ENTITY_DEDUPE, mode)
    env = ExecutionEnv()
    env.register_spec(PhaseName.ENTITY_SUMMARIZE, sum_spec, sum_spec, thinking)
    env.register_spec(PhaseName.ENTITY_DEDUPE, dedupe_spec, dedupe_spec, thinking)
    env.register_llm_hook(
        KernelTelemetryHook(
            payload_sink=_stamp_full_io,
            failure_payload_sink=_log_call_failure_payload,
            mode=mode,
        )
    )
    for hook in extra_hooks or []:
        env.register_llm_hook(hook)
    env.register_caching_hook(
        KernelDiskCache(stage=PhaseName.ENTITY_SUMMARIZE.stage_name().value)
    )
    return env


def detect_entities(
    facts_by_topic: dict,
    mode: Mode = Mode.TEE,
    model: str | None = None,
    subject: str = "the author",
    manifest_pos: dict | None = None,
    execution_env: ExecutionEnv | None = None,
    on_phase_done=None,
) -> EntitiesOutput:
    """Kernel analogue of ``entities.detect_entities``. Runs the three
    entity phases as a kernel ``Job`` and returns the ``EntitiesOutput``.

    ``on_phase_done(phase, payload)`` fires after grouping ("phase_1") and
    after the summarize batches ("phase_2") so the runner streams the
    per-entity grouping/enrichment files mid-stage — parity with legacy."""
    job = EntitiesJob(
        {
            "facts_by_topic": facts_by_topic,
            "mode": mode,
            "model": model,
            "subject": subject,
            "manifest_pos": manifest_pos,
            "on_phase_done": on_phase_done,
        }
    )
    env = execution_env if execution_env is not None else build_entities_env(mode)
    result = job.run(job.initial_input(), env)
    return result.data["output"]
