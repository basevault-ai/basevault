"""Entity dedupe phase on the kernel (issue #912).

ENTITY_DEDUPE is the degrading LLM fuzzy-merge step: a deterministic
collapse floor (applied by ``EntitySummarizePhase``), then ONE LLM call
proposing alias merges, sized down by SAMPLE-step on a sizing failure
(``_render_dedupe_rows`` renders the bottom fraction of rows minimally —
identity head only — never dropping a row). The kept merges are applied;
subject selection + final sort produce the ``EntitiesOutput``.

KERNEL-NATIVE: the LLM call routes through ``execution_env.run`` (kernel
scheduling + the degrading ``RetryPolicy``); ``sample_llm_call`` rebuilds the
rows at ``sample_step + 1``. The prompt (``_DEDUPE_SYSTEM`` / ``_DEDUPE_TASK``
/ ``_render_dedupe_rows``), the merge filter (``_filter_dedupe_merges`` —
confidence floor + name-overlap gate + cross-type rule), ``_apply_merges_to_records``
and the subject resolution are reused VERBATIM from ``entities``.
"""
from __future__ import annotations

import json
from typing import override

from kernel.abstractions import Job, LlmCall, Phase, PhaseResult
from kernel.enums import JobName, LlmStatus, PhaseName
from kernel.execution_env import BoundExecutionEnv, ExecutionEnv

from engine.phases.telemetry_hook import set_call_category

from engine.entities import (
    EntitiesOutput,
    _apply_merges_to_records,
    _DEDUPE_SYSTEM,
    _DEDUPE_TASK,
    _filter_dedupe_merges,
    _llm_resolve_stage_override,
    _render_dedupe_rows,
    _resolve_subject,
    _scrub_bundle_subject,
    derive_entities_context,
)
from engine.llm import dynamic_max_tokens, strip_fences as _strip_fences
from engine.tokens import count_tokens


class EntityDedupePhase(Phase):
    def __init__(self, job: Job):
        super().__init__(job)
        self._data: dict = {}
        self._records: list = []
        self._model: str = ""

    @override
    def name(self) -> PhaseName:
        return PhaseName.ENTITY_DEDUPE

    @override
    def init_from_memory(self, input: PhaseResult) -> bool:
        self._data = dict(input.data)
        self._records = self._data.get("records", [])
        self._mode = self._data["mode"]
        spec, _ = _llm_resolve_stage_override(self._mode, "entities_dedupe")
        self._model = spec.model_id
        return True

    @override
    def init_from_disk(self) -> None:
        raise NotImplementedError("EntityDedupePhase always inits from memory")

    @override
    def validate(self, payload) -> LlmStatus | None:
        # Mirrors the legacy dedupe parser: empty / empty-merges →
        # success-empty (the intended false-success guard → one retry);
        # non-empty-unparseable / wrong-shape → parse-error.
        if not isinstance(payload, str) or not payload.strip():
            return LlmStatus.SUCCESS_EMPTY
        try:
            data = json.loads(_strip_fences(payload))
        except (json.JSONDecodeError, ValueError):
            return LlmStatus.PARSE_ERROR
        if not isinstance(data, dict) or not isinstance(data.get("merges") or [], list):
            return LlmStatus.PARSE_ERROR
        if not (data.get("merges") or []):
            return LlmStatus.SUCCESS_EMPTY
        return None

    def _build_call(self, sample_step: int, previous: LlmCall | None) -> LlmCall:
        rows = _render_dedupe_rows(self._records, sample_step=sample_step)
        user = _DEDUPE_TASK.format(n=len(self._records), rows=rows)
        max_tokens = dynamic_max_tokens(
            count_tokens(user), self._mode, stage="entities_dedupe"
        )
        return self.new_call(
            [
                {"role": "system", "content": _DEDUPE_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens,
            previous,
            context=sample_step,
        )

    @override
    def halve_llm_call(self, call: LlmCall) -> list[LlmCall]:
        raise NotImplementedError("entity dedupe samples; it never halves")

    @override
    def sample_llm_call(self, call: LlmCall) -> LlmCall | None:
        # Sample-step monotonically increases; the kernel's degrading cap
        # bounds depth. Rows shed detail (never drop a row).
        return self._build_call(call.context + 1, call)

    @override
    def checkpoint(self) -> None:
        pass

    def _finalize(self, merges: list[dict]) -> PhaseResult:
        records = self._records
        relations = self._data.get("relations", [])
        likelihoods = dict(self._data.get("likelihoods", {}))
        subject = self._data.get("subject", "the author")
        is_bundle = self._data.get("is_bundle", False)

        if merges:
            records, relations, id_remap = _apply_merges_to_records(
                records, merges, relations, self._mode
            )
            remapped: dict[str, float] = {}
            for cid, lk in likelihoods.items():
                new_id = id_remap.get(cid, cid)
                remapped[new_id] = max(remapped.get(new_id, 0.0), lk)
            likelihoods = remapped

        subject_ref = _resolve_subject(records, likelihoods, subject)
        records.sort(key=lambda r: (-r.mention_count, r.canonical_name.lower()))
        if subject_ref is not None:
            records.sort(
                key=lambda r: (
                    r.canonical_id != subject_ref.canonical_id,
                    -r.mention_count,
                    r.canonical_name.lower(),
                )
            )
            for r in records:
                if r.canonical_id == subject_ref.canonical_id and not r.role:
                    r.role = "subject"

        out = _scrub_bundle_subject(
            EntitiesOutput(subject=subject_ref, entities=records, relations=relations),
            is_bundle,
        )
        # Derive the grounding blocks the synthesis stages consume (verbatim
        # runner logic) so the flat PipelineJob grounds patterns / insights /
        # actions identically. Per-stage *Job harnesses seed these directly,
        # so this only feeds the pipeline path.
        resolved_subject, entities_context, entities_context_by_topic = (
            derive_entities_context(out, self._data.get("facts_by_topic", {}), subject)
        )
        return PhaseResult({
            **self._data,
            "output": out,
            "entities_output": out,
            "subject": resolved_subject,
            "entities_context": entities_context,
            "entities_context_by_topic": entities_context_by_topic,
        })

    @override
    def run_main(self, execution_env: BoundExecutionEnv | None) -> PhaseResult:
        assert execution_env is not None
        if not self._records:
            return self._finalize([])

        _dedupe_call = self._build_call(0, None)
        # Mark the dedupe call so run-details distinguishes it from the
        # entity summarize batches (legacy used a separate "entities_dedupe"
        # stage; the kernel collapses entity phases to one stage, so the
        # distinction lives in the per-call category column).
        set_call_category(execution_env, _dedupe_call, "entities_dedupe")
        responses = execution_env.run(_dedupe_call).result()
        raw = None
        for response in responses:
            if isinstance(response.payload, str) and response.payload.strip():
                raw = response.payload
        merges: list[dict] = []
        if raw is not None:
            try:
                data = json.loads(_strip_fences(raw))
                raw_merges = data.get("merges") or [] if isinstance(data, dict) else []
            except (json.JSONDecodeError, ValueError):
                raw_merges = []
            if raw_merges:
                merges = _filter_dedupe_merges(
                    raw_merges, self._records, self._mode, self._model
                )
        return self._finalize(merges)


class EntityDedupeJob(Job):
    def __init__(self, initial: dict):
        super().__init__()
        self._initial = initial

    @override
    def name(self) -> JobName:
        return JobName.FULL_PIPELINE

    @override
    def generate_phases(self) -> list[Phase]:
        return [EntityDedupePhase(self)]

    def initial_input(self) -> PhaseResult:
        return PhaseResult(self._initial)


def dedupe_entities(
    records: list,
    mode,
    relations: list | None = None,
    likelihoods: dict | None = None,
    subject: str = "the author",
    is_bundle: bool = False,
    execution_env: ExecutionEnv | None = None,
) -> EntitiesOutput:
    """Run the dedupe phase on the kernel; returns the ``EntitiesOutput``."""
    from engine.phases.model_specs import spec_for_mode

    job = EntityDedupeJob({
        "records": records, "relations": relations or [],
        "likelihoods": likelihoods or {}, "subject": subject,
        "is_bundle": is_bundle, "mode": mode,
    })
    if execution_env is None:
        spec = spec_for_mode(mode)
        env = ExecutionEnv()
        env.register_spec(PhaseName.ENTITY_DEDUPE, spec, spec, False)
    else:
        env = execution_env
    return job.run(job.initial_input(), env).data["output"]
