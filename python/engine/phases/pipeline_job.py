"""The production pipeline as ONE flat kernel Job (issue #912).

Architecture decision: production is a single ``PipelineJob`` whose
``generate_phases()`` returns ALL phases in pipeline order — there is NO
Stage abstraction. The kernel already decomposed stage concerns to the
phase level (per-phase ``ModelSpec`` + ``RetryPolicy``), so a Stage layer
would be redundant. Phases group for eval / UI via
``PhaseName.stage_name()`` (→ ``StageName``); no per-phase tag is needed.

The per-stage Jobs (``ExtractionStageJob``, ``EntitiesJob``,
``PatternsJob``, …) stay as EVAL / TEST harnesses — they seed a stage's
inputs directly. The ``PipelineJob`` instead threads ONE accumulating
state dict: every phase reads what it needs from the prior phase's
``PhaseResult`` and returns ``{**state, <its outputs>}``, so each
downstream phase sees all upstream artifacts (facts_by_topic →
entities_output → patterns_by_topic → insight_output → action_list →
embedding_pairs). Inputs not produced upstream (the embeddings plan, the
pending-vision image set) are derived in the consuming phase from the
accumulated state.

Entity grounding: ``EntityDedupePhase`` derives the run-wide + per-topic
``entities_context`` blocks (``entities.derive_entities_context``, the
runner's verbatim logic) into the state, so patterns / insights / actions
are grounded on the pipeline path exactly as the legacy runner grounds them.
"""
from __future__ import annotations

from typing import override

from kernel.abstractions import Job, Phase, PhaseResult
from kernel.enums import JobName, PhaseName
from kernel.execution_env import ExecutionEnv

from engine.llm import Mode
from engine.splitter import DEFAULT_BUDGET_TOKENS

from engine.phases.actions import ActionsPhase
from engine.phases.embeddings import EmbeddingsPhase
from engine.phases.entity_dedupe import EntityDedupePhase
from engine.phases.entity_grouping import EntityGroupingPhase
from engine.phases.entity_summarize import EntitySummarizePhase
from engine.phases.extraction_completion import ExtractionCompletionPhase
from engine.phases.extraction_llm import ExtractionPhase
from engine.phases.extraction_splitter import ExtractionSplitterPhase
from engine.phases.ingestion import IngestionPhase
from engine.phases.insights import InsightsPhase
from engine.phases.model_specs import (
    embedding_spec_for_mode,
    spec_for_mode,
    vision_spec_for_mode,
)
from engine.phases.patterns import PatternsPhase

# Phases whose LLM call routes through the mode's chat anchor spec.
_CHAT_SPEC_PHASES = (
    PhaseName.EXTRACTION_LLM,
    PhaseName.ENTITY_SUMMARIZE,
    PhaseName.ENTITY_DEDUPE,
    PhaseName.PATTERNS,
    PhaseName.INSIGHTS,
    PhaseName.ACTIONS,
)


class PipelineJob(Job):
    def __init__(self, initial: dict):
        super().__init__()
        self._initial = initial

    @override
    def name(self) -> JobName:
        return JobName.FULL_PIPELINE

    @override
    def generate_phases(self) -> list[Phase]:
        return [
            IngestionPhase(self),
            ExtractionSplitterPhase(self),
            ExtractionPhase(self),
            ExtractionCompletionPhase(self),
            EntityGroupingPhase(self),
            EntitySummarizePhase(self),
            EntityDedupePhase(self),
            PatternsPhase(self),
            InsightsPhase(self),
            ActionsPhase(self),
            EmbeddingsPhase(self),
        ]

    def initial_input(self) -> PhaseResult:
        return PhaseResult(self._initial)


def build_pipeline_env(mode: Mode, thinking: bool = False) -> ExecutionEnv:
    """Register the per-phase specs: a dedicated vision spec for INGESTION,
    the embedding spec for EMBEDDINGS, the mode chat anchor for the rest."""
    env = ExecutionEnv()
    chat_spec = spec_for_mode(mode)
    for pn in _CHAT_SPEC_PHASES:
        env.register_spec(pn, chat_spec, chat_spec, thinking)
    vis = vision_spec_for_mode(mode)
    env.register_spec(PhaseName.INGESTION, vis, vis, thinking)
    emb = embedding_spec_for_mode(mode)
    env.register_spec(PhaseName.EMBEDDINGS, emb, emb, False)
    return env


def run_pipeline(
    all_docs: list,
    mode: Mode = Mode.TEE,
    *,
    subject: str = "the author",
    today=None,
    sentiment: str = "neutral",
    total_facts: int = 1,
    min_topics: int = 2,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    manifest_pos: dict | None = None,
    execution_env: ExecutionEnv | None = None,
) -> dict:
    """Run the whole pipeline as one kernel Job; returns the final
    accumulated state dict (facts_by_topic, entities_output,
    patterns_by_topic, insight_output, action_list, embedding_pairs, …)."""
    job = PipelineJob({
        "all_docs": all_docs,
        "mode": mode,
        "subject": subject,
        "today": today,
        "sentiment": sentiment,
        "total_facts": total_facts,
        "min_topics": min_topics,
        "budget_tokens": budget_tokens,
        "manifest_pos": manifest_pos,
    })
    env = execution_env if execution_env is not None else build_pipeline_env(mode)
    return job.run(job.initial_input(), env).data
