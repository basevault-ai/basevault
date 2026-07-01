"""Entity summarize phase on the kernel (issue #912).

ENTITY_SUMMARIZE is the per-entity LLM step: each packed batch of
canonical groups is rendered into a prompt, one LLM call annotates every
entity in the batch (canonical name, role, description, subject-likelihood,
consolidated relations), and the kernel's retry ladder HALVES the batch on
a sizing failure (non-degrading — split the work, never sample). The
prompt build (``_render_entity_block`` / ``_build_prompt`` / ``_SYSTEM``),
the parser (``_validate_entities_or_raise`` / ``_parse_per_entity_response``),
the halve (``_halve_batch`` / ``_split_heavy_entity``) and the record
assembly (``_materialize_record`` / ``_disambiguate_ids`` /
``_deterministic_collapse``) are all reused VERBATIM from ``entities``.

Concurrency: batches are independent, so the phase fans all batch
call-chains out via ``execution_env.run_all(calls)`` — the kernel scheduler
runs them concurrently (each batch's halve cascade still ladders per-call
internally), matching the legacy scheduler-pool fan-out. A lock guards the
clone registration into the shared group / candidate maps when batches
halve in parallel.
"""
from __future__ import annotations

import re
import threading
from collections import defaultdict
from typing import override

from kernel.abstractions import Job, LlmCall, Phase, PhaseResult
from kernel.enums import LlmStatus, PhaseName
from kernel.execution_env import BoundExecutionEnv

from engine.phases.telemetry_hook import set_call_category

from engine.entities import (
    EntityRecord,
    RelationEdge,
    _build_name_index,
    _build_prompt,
    _deterministic_collapse,
    _disambiguate_ids,
    _materialize_record,
    _parse_per_entity_response,
    _render_entity_block,
    _resolve_one_candidate,
    _slugify,
    _split_heavy_entity,
    _SYSTEM,
    _validate_entities_or_raise,
)
from engine.llm import dynamic_max_tokens
from engine.tokens import count_tokens

# The parse-signal exception types are imported lazily in ``validate`` from
# the leaf ``parse_signals`` module (the surviving half of legacy ``retry``).


def _halve_batch(batch):
    """Split a batch into two contiguous halves, preserving the upstream
    ordering ``_pack_batches`` established. Returns ``None`` when the batch
    is a single group (can't halve by group). Verbatim from the legacy
    ``detect_entities._halve_batch`` closure."""
    if len(batch) < 2:
        return None
    mid = max(1, len(batch) // 2)
    return batch[:mid], batch[mid:]


class EntitySummarizePhase(Phase):
    def __init__(self, job: Job):
        super().__init__(job)
        self._data: dict = {}
        self._annotations: dict[str, dict] = {}
        # Guards concurrent clone registration into the shared group /
        # candidate maps when batches halve in parallel under run_all.
        self._halve_lock = threading.Lock()

    @override
    def name(self) -> PhaseName:
        return PhaseName.ENTITY_SUMMARIZE

    @override
    def init_from_memory(self, input: PhaseResult) -> bool:
        self._data = dict(input.data)
        # Mutable group map: ``halve_llm_call`` registers clone groups here
        # so the post-run parse + assembly can see them.
        self._groups_by_gid = self._data["groups_by_gid"]
        self._candidates_by_gid = self._data["candidates_by_gid"]
        self._by_key = self._data["by_key"]
        self._facts_by_topic = self._data["facts_by_topic"]
        self._batches = self._data["batches"]
        self._subject = self._data.get("subject", "the author")
        self._is_bundle = self._data.get("is_bundle", False)
        self._mode = self._data["mode"]
        return True

    @override
    def init_from_disk(self) -> None:
        raise NotImplementedError("EntitySummarizePhase always inits from memory")

    @override
    def validate(self, payload) -> LlmStatus | None:
        from engine.parse_signals import _EmptyResponse, _ParseError, _SuccessEmpty

        if not isinstance(payload, str):
            return LlmStatus.PARSE_ERROR
        try:
            _validate_entities_or_raise(payload)
        except _ParseError:
            return LlmStatus.PARSE_ERROR
        except (_SuccessEmpty, _EmptyResponse):
            return LlmStatus.SUCCESS_EMPTY
        return None

    def _blocks_for(self, batch) -> list[str]:
        return [
            _render_entity_block(
                g, self._candidates_by_gid.get(g.gid, []), self._groups_by_gid
            )
            for g in batch
        ]

    def _call_for_batch(
        self, batch, max_tokens: int, previous: LlmCall | None
    ) -> LlmCall:
        prompt = _build_prompt(
            self._blocks_for(batch), self._subject, len(batch), self._is_bundle
        )
        return self.new_call(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens,
            previous,
            context=batch,
        )

    @override
    def halve_llm_call(self, call: LlmCall) -> list[LlmCall]:
        """Non-degrading sizing recovery: split the batch by group, or
        clone-split a single heavy group's facts. Reuses ``_halve_batch``
        / ``_split_heavy_entity`` verbatim; registers clones in the shared
        ``groups_by_gid`` so parse + assembly resolve them."""
        batch = call.context
        halved = _halve_batch(batch)
        if halved is None:
            g = batch[0]
            if len(g.facts) < 2:
                return []
            # Clone-split + registration mutate the shared group / candidate
            # maps; serialize so parallel halving batches don't race.
            with self._halve_lock:
                clones = _split_heavy_entity(
                    g,
                    self._candidates_by_gid,
                    self._groups_by_gid,
                    budget_tokens=10**12,
                    by_key=self._by_key,
                    facts_by_topic=self._facts_by_topic,
                )
                if len(clones) < 2:
                    return []
                for clone in clones:
                    self._groups_by_gid[clone.gid] = clone
            halved = ([clones[0]], [clones[1]])
        left, right = halved
        return [
            self._call_for_batch(left, call.max_tokens, call),
            self._call_for_batch(right, call.max_tokens, call),
        ]

    @override
    def sample_llm_call(self, call: LlmCall) -> LlmCall | None:
        raise NotImplementedError("entity summarize halves; it never samples")

    @override
    def checkpoint(self) -> None:
        pass

    def _parse_leaves(self, responses) -> None:
        batch_all = list(self._groups_by_gid.values())
        for response in responses:
            payload = response.payload
            if not payload or not isinstance(payload, str):
                continue
            parsed, parse_error = _parse_per_entity_response(
                payload, batch_all, self._groups_by_gid
            )
            if parse_error:
                continue
            self._annotations.update(parsed)

    def _assemble(self) -> PhaseResult:
        """Materialize EntityRecords from groups + annotations, then run the
        deterministic collapse floor. Mirrors ``detect_entities`` verbatim."""
        groups = self._data["groups"]
        groups_by_gid = self._groups_by_gid
        annotations = self._annotations

        records: list[EntityRecord] = []
        likelihoods: dict[str, float] = {}
        pre_relations: list[RelationEdge] = []
        relation_evidence: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(
            list
        )
        name_idx = _build_name_index(groups)
        for topic, items in self._facts_by_topic.items():
            for idx, item in enumerate(items):
                rc = item.relation_candidate
                if not isinstance(rc, dict):
                    continue
                from_g = _resolve_one_candidate(rc.get("from", ""), item, name_idx)
                to_g = _resolve_one_candidate(rc.get("to", ""), item, name_idx)
                if from_g is None or to_g is None or from_g.gid == to_g.gid:
                    continue
                from_id = _slugify(from_g.canonical_name)
                to_id = _slugify(to_g.canonical_name)
                relation_evidence[(from_id, to_id)].append((topic, idx))
                relation_evidence[(to_id, from_id)].append((topic, idx))

        for g in groups:
            ann = annotations.get(g.gid, {})
            clones = [
                (cgid, cg)
                for cgid, cg in groups_by_gid.items()
                if cg.parent_gid == g.gid
            ]
            if clones:
                for cgid, clone in sorted(clones, key=lambda x: x[0]):
                    cann = annotations.get(cgid, {})
                    _materialize_record(
                        clone, cann, records, likelihoods, pre_relations, groups_by_gid
                    )
                continue
            _materialize_record(
                g, ann, records, likelihoods, pre_relations, groups_by_gid
            )

        _disambiguate_ids(records)
        for rel in pre_relations:
            from_base = re.sub(r"-\d+$", "", rel.from_id)
            to_base = re.sub(r"-\d+$", "", rel.to_id)
            rel.evidence_fact_refs = sorted(
                relation_evidence.get((from_base, to_base), [])
            )

        records, relations = _deterministic_collapse(records, pre_relations, self._mode)
        rec_ids_now = {r.canonical_id for r in records}
        collapsed_likelihoods: dict[str, float] = {}
        for cid, lk in likelihoods.items():
            if cid in rec_ids_now:
                collapsed_likelihoods[cid] = max(
                    collapsed_likelihoods.get(cid, 0.0), lk
                )

        return PhaseResult(
            {
                **self._data,
                "records": records,
                "relations": relations,
                "likelihoods": collapsed_likelihoods,
            }
        )

    @override
    def run_main(self, execution_env: BoundExecutionEnv | None) -> PhaseResult:
        assert execution_env is not None
        if self._data.get("empty"):
            return PhaseResult({**self._data, "records": [], "relations": [],
                                "likelihoods": {}})

        # Each batch is an INDEPENDENT call-chain — fan them all out via
        # run_all so the kernel scheduler runs them concurrently (each
        # batch's own halve cascade still ladders per-call internally).
        # results[i] <-> calls[i].
        calls = []
        n = len(self._batches)
        for i, batch in enumerate(self._batches):
            payload_tokens = count_tokens("\n".join(self._blocks_for(batch)))
            max_tokens = dynamic_max_tokens(payload_tokens, self._mode, stage="entities")
            call = self._call_for_batch(batch, max_tokens, None)
            # Per-call category column = the legacy "batch-N-of-M" label.
            set_call_category(execution_env, call, f"batch-{i + 1}-of-{n}")
            calls.append(call)

        for responses in execution_env.run_all(calls):
            self._parse_leaves(responses)

        # Live persistence: fire the phase_2 (enrichment) marker so the runner
        # rewrites each per-entity file with its LLM-produced description /
        # role mid-stage, same as the prior driver. Guarded.
        on_phase_done = self._data.get("on_phase_done")
        if on_phase_done is not None:
            try:
                from engine.entities import build_phase2_marker_payload
                on_phase_done("phase_2", build_phase2_marker_payload(
                    self._annotations, self._groups_by_gid, len(self._batches),
                ))
            except Exception:
                pass

        return self._assemble()
