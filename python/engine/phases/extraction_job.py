"""Extraction stage Job on the kernel (issue #912).

Wires the three extraction phases — splitter (deterministic) → LLM
(per-doc, halve) → completion (deterministic) — into one kernel ``Job`` and
exposes ``extract_stage`` returning the ``facts_by_topic`` dict
the downstream stages consume. Only ``EXTRACTION_LLM`` does an LLM call, so
only it is registered on the env.
"""
from __future__ import annotations

from typing import override

from kernel.abstractions import Job, Phase, PhaseResult
from kernel.enums import JobName, PhaseName
from kernel.execution_env import ExecutionEnv

from engine.llm import Mode
from engine.splitter import DEFAULT_BUDGET_TOKENS

from engine.phases.extraction_completion import ExtractionCompletionPhase
from engine.phases.extraction_llm import ExtractionPhase
from engine.phases.extraction_splitter import ExtractionSplitterPhase
from engine.phases.model_specs import spec_for_mode


class ExtractionStageJob(Job):
    def __init__(self, initial: dict):
        super().__init__()
        self._initial = initial

    @override
    def name(self) -> JobName:
        return JobName.FULL_PIPELINE

    @override
    def generate_phases(self) -> list[Phase]:
        return [
            ExtractionSplitterPhase(self),
            ExtractionPhase(self),
            ExtractionCompletionPhase(self),
        ]

    def initial_input(self) -> PhaseResult:
        return PhaseResult(self._initial)


def extract_stage(
    all_docs: list,
    mode: Mode = Mode.TEE,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    combine_small: bool = True,
    carve_first: bool = True,
    execution_env: ExecutionEnv | None = None,
) -> dict:
    """Run the full extract stage (split → extract → assemble) on the
    kernel; returns ``facts_by_topic``."""
    job = ExtractionStageJob(
        {
            "all_docs": all_docs,
            "mode": mode,
            "budget_tokens": budget_tokens,
            "combine_small": combine_small,
            "carve_first": carve_first,
        }
    )
    if execution_env is None:
        spec = spec_for_mode(mode)
        env = ExecutionEnv()
        env.register_spec(PhaseName.EXTRACTION_LLM, spec, spec, False)
    else:
        env = execution_env
    result = job.run(job.initial_input(), env)
    return result.data["facts_by_topic"]
