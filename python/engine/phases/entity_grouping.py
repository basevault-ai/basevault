"""Entity grouping phase on the kernel (issue #912).

ENTITY_GROUPING does NO LLM call (``PhaseName.ENTITY_GROUPING`` is absent
from ``does_llm_call``). It holds all the DETERMINISTIC prep the legacy
``entities.detect_entities`` runs before the per-entity LLM batches:

  * ``_group_entities``      — bucket entity mentions into canonical groups
  * ``_resolve_candidates``  — resolve extract's relation candidates to gids
  * ``_build_other_catalogs``/``_build_name_key_map`` — per-group context
  * ``_pack_batches``        — greedy-pack groups into per-call batches

All reused VERBATIM from ``entities`` — the migration changes none of the
grouping logic. The phase emits the packed batches + the candidate / group
maps as its ``PhaseResult`` for ``EntitySummarizePhase`` to consume.
"""
from __future__ import annotations

from typing import override

from kernel.abstractions import Job, LlmCall, Phase, PhaseResult
from kernel.enums import PhaseName
from kernel.execution_env import BoundExecutionEnv

from engine.entities import (
    _build_name_key_map,
    _build_other_catalogs,
    _group_entities,
    _is_bundle,
    _pack_batches,
    _resolve_candidates,
)
from engine.llm import Mode


class EntityGroupingPhase(Phase):
    def __init__(self, job: Job):
        super().__init__(job)
        self._data: dict = {}

    @override
    def name(self) -> PhaseName:
        return PhaseName.ENTITY_GROUPING

    @override
    def init_from_memory(self, input: PhaseResult) -> bool:
        self._data = dict(input.data)
        return True

    @override
    def init_from_disk(self) -> None:
        raise NotImplementedError("EntityGroupingPhase always inits from memory")

    @override
    def validate(self, payload):  # no LLM call
        return None

    @override
    def halve_llm_call(self, call: LlmCall) -> list[LlmCall]:
        raise NotImplementedError("grouping does no LLM call")

    @override
    def sample_llm_call(self, call: LlmCall) -> LlmCall | None:
        raise NotImplementedError("grouping does no LLM call")

    @override
    def checkpoint(self) -> None:
        pass

    @override
    def run_main(self, execution_env: BoundExecutionEnv | None) -> PhaseResult:
        facts_by_topic = self._data["facts_by_topic"]
        mode: Mode = self._data["mode"]
        manifest_pos = self._data.get("manifest_pos")

        groups = _group_entities(facts_by_topic, manifest_pos=manifest_pos)
        if not groups:
            return PhaseResult({**self._data, "groups": [], "empty": True})

        candidates_by_gid, n_resolved, n_dropped = _resolve_candidates(
            facts_by_topic, groups
        )
        _build_other_catalogs(facts_by_topic, groups)
        by_key = _build_name_key_map(groups)
        groups_by_gid = {g.gid: g for g in groups}
        batches = _pack_batches(
            groups,
            candidates_by_gid,
            groups_by_gid,
            mode,
            facts_by_topic=facts_by_topic,
            manifest_pos=manifest_pos,
            by_key=by_key,
        )

        # Live persistence: fire the phase_1 (grouping) marker so the runner
        # writes the per-entity grouping view mid-stage (the run-tree's
        # Entities node fills in before the summarize LLM batches run), same
        # as the prior driver. Guarded — a raising callback must not
        # break the phase.
        on_phase_done = self._data.get("on_phase_done")
        if on_phase_done is not None:
            try:
                from engine.entities import build_phase1_marker_payload
                on_phase_done("phase_1", build_phase1_marker_payload(
                    groups, candidates_by_gid, groups_by_gid,
                    n_resolved, n_dropped, len(batches),
                ))
            except Exception:
                pass

        return PhaseResult(
            {
                **self._data,
                "empty": False,
                "groups": groups,
                "candidates_by_gid": candidates_by_gid,
                "groups_by_gid": groups_by_gid,
                "by_key": by_key,
                "batches": batches,
                "is_bundle": _is_bundle(facts_by_topic),
                "n_resolved": n_resolved,
                "n_dropped": n_dropped,
            }
        )
