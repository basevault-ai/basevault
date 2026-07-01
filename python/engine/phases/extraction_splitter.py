"""Extraction splitter phase on the kernel (issue #912).

EXTRACTION_SPLITTER does NO LLM call. It is the deterministic Phase-1 of
the extract stage: chunk the ingested documents to the per-call token
budget and pack small consecutive entries into batched units, exactly as
the runner does today. Reuses ``splitter.split_documents`` /
``carve_first_batch`` VERBATIM; the migration changes none of the chunking
logic. Emits the prepared ``list[Document]`` for ``ExtractionPhase``.
"""
from __future__ import annotations

from typing import override

from kernel.abstractions import Job, LlmCall, Phase, PhaseResult
from kernel.enums import PhaseName
from kernel.execution_env import BoundExecutionEnv

from engine.splitter import carve_first_batch, split_documents


class ExtractionSplitterPhase(Phase):
    def __init__(self, job: Job):
        super().__init__(job)
        self._data: dict = {}

    @override
    def name(self) -> PhaseName:
        return PhaseName.EXTRACTION_SPLITTER

    @override
    def init_from_memory(self, input: PhaseResult) -> bool:
        self._data = dict(input.data)
        return True

    @override
    def init_from_disk(self) -> None:
        raise NotImplementedError("ExtractionSplitterPhase always inits from memory")

    @override
    def validate(self, payload):  # no LLM call
        return None

    @override
    def halve_llm_call(self, call: LlmCall) -> list[LlmCall]:
        raise NotImplementedError("splitter does no LLM call")

    @override
    def sample_llm_call(self, call: LlmCall) -> LlmCall | None:
        raise NotImplementedError("splitter does no LLM call")

    @override
    def checkpoint(self) -> None:
        pass

    @override
    def run_main(self, execution_env: BoundExecutionEnv | None) -> PhaseResult:
        all_docs = self._data["all_docs"]
        budget = self._data["budget_tokens"]
        combine_small = self._data.get("combine_small", True)
        carve_first = self._data.get("carve_first", True)

        docs = split_documents(
            all_docs, budget_tokens=budget, combine_small=combine_small
        )
        if carve_first:
            docs = carve_first_batch(docs, rest_budget=budget)
        return PhaseResult({**self._data, "docs": docs})
