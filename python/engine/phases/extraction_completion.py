"""Extraction completion phase on the kernel (issue #912).

EXTRACTION_COMPLETION does NO LLM call. It is the deterministic Phase-3 of
the extract stage: fold the flat ``list[ExtractedItem]`` the LLM phase
produced into the ``facts_by_topic`` dict the downstream stages consume —
one item appears under each of its topics, each topic's list sorted by
(evidence file_path, file_offset, summary). Mirrors the runner's in-memory
assembly VERBATIM (the on-disk JSONL re-sort is a persistence detail the
runner still owns).
"""
from __future__ import annotations

from collections import defaultdict
from typing import override

from kernel.abstractions import Job, LlmCall, Phase, PhaseResult
from kernel.enums import PhaseName
from kernel.execution_env import BoundExecutionEnv


def _sort_key(it):
    ev = it.evidence[0] if it.evidence else None
    return (
        (ev.file_path or "") if ev else "",
        (ev.file_offset if ev and ev.file_offset is not None else 0),
        it.summary,
    )


class ExtractionCompletionPhase(Phase):
    def __init__(self, job: Job):
        super().__init__(job)
        self._items: list = []

    @override
    def name(self) -> PhaseName:
        return PhaseName.EXTRACTION_COMPLETION

    @override
    def init_from_memory(self, input: PhaseResult) -> bool:
        self._state = dict(input.data)
        self._items = list(input.data.get("items", []))
        return True

    @override
    def init_from_disk(self) -> None:
        raise NotImplementedError("ExtractionCompletionPhase always inits from memory")

    @override
    def validate(self, payload):  # no LLM call
        return None

    @override
    def halve_llm_call(self, call: LlmCall) -> list[LlmCall]:
        raise NotImplementedError("completion does no LLM call")

    @override
    def sample_llm_call(self, call: LlmCall) -> LlmCall | None:
        raise NotImplementedError("completion does no LLM call")

    @override
    def checkpoint(self) -> None:
        pass

    @override
    def run_main(self, execution_env: BoundExecutionEnv | None) -> PhaseResult:
        by_topic = defaultdict(list)
        for it in self._items:
            for topic in it.topics:
                by_topic[topic].append(it)
        facts_by_topic = {
            topic: sorted(items, key=_sort_key) for topic, items in by_topic.items()
        }
        return PhaseResult({**self._state, "facts_by_topic": facts_by_topic})
