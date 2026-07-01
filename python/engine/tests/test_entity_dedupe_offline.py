"""Offline validation for the kernel-native ENTITY_DEDUPE cascade (#912).

The dedupe LLM call now routes through the kernel (was interim-legacy).
Drives ``EntityDedupePhase`` with a scripted provider to prove: a kept merge
is applied (records collapse), the degrading sample cascade fans on repeated
PARSE_ERROR, and the no-merges case takes exactly one retry (the intended
false-success guard) — using the verbatim ``_filter_dedupe_merges`` +
``_apply_merges_to_records``.
"""
from __future__ import annotations

from engine.entities import EntityRecord
from kernel.abstractions import InferenceProvider, LlmResponse
from kernel.enums import PhaseName
from kernel.execution_env import ExecutionEnv

from engine.llm import ModelSpec as LegacyModelSpec, Mode
from engine.phases.entity_dedupe import EntityDedupeJob
from engine.phases.model_specs import PipelineModelSpec


def _rec(cid, name, etype="person"):
    return EntityRecord(
        canonical_id=cid,
        canonical_name=name,
        entity_type=etype,
        aliases=[],
        role="",
        description="d",
        mention_count=3,
        topics=["work"],
        evidence_fact_refs=[],
    )


class _Scripted(InferenceProvider):
    def __init__(self, responder):
        self._responder = responder
        self.call_count = 0
        self._inj = {p: [] for p in PhaseName}

    def name(self):
        return "scripted-dedupe"

    def run(self, call, execution_env) -> LlmResponse:
        self.call_count += 1
        return LlmResponse(None, self._responder(self.call_count), None, 0, 0, 0, 0.0, 0.0)

    def inject_errors(self, phase, errors):
        self._inj[phase] += errors


def _run(responder, records):
    provider = _Scripted(responder)
    legacy = LegacyModelSpec(
        provider="scripted", model_id="kimi-k2-6", context_window=131_000
    )
    spec = PipelineModelSpec(provider, legacy.model_id, legacy.context_window, max_parallelism=4)
    env = ExecutionEnv()
    env.register_spec(PhaseName.ENTITY_DEDUPE, spec, spec, thinking=False)
    job = EntityDedupeJob({
        "records": records, "relations": [], "likelihoods": {},
        "subject": "the author", "is_bundle": False, "mode": Mode.TEE,
    })
    out = job.run(job.initial_input(), env).data["output"]
    return out, provider.call_count


# High-conf, same-type → bypasses the name-overlap gate → merge kept.
_MERGE = (
    '{"merges": [{"a_id": "bob", "b_id": "bobby", "confidence": 0.95,'
    ' "synthesized_description": "Bob"}]}'
)


def test_dedupe_applies_kept_merge():
    records = [_rec("bob", "Bob"), _rec("bobby", "Bobby")]
    out, calls = _run(lambda n: _MERGE, records)
    assert calls == 1
    # The two records collapsed into one.
    assert len(out.entities) == 1


def test_dedupe_no_merges_takes_one_retry():
    # Empty merges → SUCCESS_EMPTY → one retry (the false-success guard) →
    # settle. Both entities survive.
    records = [_rec("bob", "Bob"), _rec("ann", "Ann")]
    out, calls = _run(lambda n: '{"merges": []}', records)
    assert calls == 2
    assert len(out.entities) == 2


def test_dedupe_parse_error_runs_degrading_cascade():
    records = [_rec("bob", "Bob"), _rec("ann", "Ann")]
    out, calls = _run(lambda n: "not json {{{", records)
    assert calls == 6  # full-retry → model-fallback → SAMPLE×3 → stop
    assert len(out.entities) == 2
