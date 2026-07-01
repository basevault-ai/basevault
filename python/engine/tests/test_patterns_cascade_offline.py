"""Offline validation for the migrated PATTERNS degrading cascade (#912).

PATTERNS is per-topic + degrading (sample via confidence-prioritized
halving). This drives the migrated phase with a scripted provider to prove
the kernel runs one terminal chain per topic and fans the degrading
cascade on repeated PARSE_ERROR exactly as the legacy stage would (6
provider calls per topic, reasoning off), and a clean response is one call.
"""
from __future__ import annotations

from engine.content_extractor import Entity, EntityRef, EvidenceSpan, ExtractedItem
from kernel.abstractions import InferenceProvider, LlmResponse
from kernel.enums import PhaseName
from kernel.execution_env import ExecutionEnv

from engine.llm import ModelSpec as LegacyModelSpec, Mode
from engine.phases.patterns import PatternsJob
from engine.phases.model_specs import PipelineModelSpec


def _item(summary, conf):
    return ExtractedItem(
        item_type="fact",
        summary=summary,
        evidence=[EvidenceSpan(text=summary, source_ref="t")],
        entities=[EntityRef(entity=Entity(name="Alice", entity_type="person"),
                            role="subject")],
        topics=["work"],
        tags=[],
        confidence=conf,
        relation_candidate=None,
        occurred_at=None,
    )


def _facts():
    return {
        "work": [
            _item(f"distinct fact number {i} about work habits", 0.3 + 0.1 * i)
            for i in range(6)
        ]
    }


_VALID = '[{"name": "overcommitment", "description": "d", "source_facts": [1, 2]}]'


class _Scripted(InferenceProvider):
    def __init__(self, responder):
        self._responder = responder
        self.call_count = 0
        self._inj = {p: [] for p in PhaseName}

    def name(self):
        return "scripted"

    def run(self, call, execution_env) -> LlmResponse:
        self.call_count += 1
        return LlmResponse(None, self._responder(self.call_count), None, 0, 0, 0, 0.0, 0.0)

    def inject_errors(self, phase, errors):
        self._inj[phase] += errors


def _run(responder, facts=None):
    provider = _Scripted(responder)
    legacy = LegacyModelSpec(
        provider="scripted", model_id="gpt-oss-120b", context_window=131_000
    )
    spec = PipelineModelSpec(provider, legacy.model_id, legacy.context_window, max_parallelism=4)
    env = ExecutionEnv()
    env.register_spec(PhaseName.PATTERNS, spec, spec, thinking=False)

    job = PatternsJob({"facts_by_topic": facts or _facts(), "mode": Mode.TEE})
    out = job.run(job.initial_input(), env)
    return out.data["patterns_by_topic"], provider.call_count


def _multi_facts(n_topics):
    return {
        f"topic{t}": [
            _item(f"distinct fact {i} in topic {t}", 0.3 + 0.1 * i)
            for i in range(6)
        ]
        for t in range(n_topics)
    }


def test_patterns_fans_topics_via_run_all():
    # 3 independent topics → one root call each, fired concurrently via
    # run_all (each through its own per-call ladder). Clean responses →
    # exactly one call per topic; per-topic dedup maps stay isolated.
    _by_topic, calls = _run(lambda n: _VALID, facts=_multi_facts(3))
    assert calls == 3


def test_patterns_clean_success_single_call():
    # Contract under test = one provider call per topic on a clean (OK)
    # response. Parse fidelity (≥2 mapped source-facts per pattern) is the
    # live L2 eval's job, not this cascade-mechanics test.
    _by_topic, calls = _run(lambda n: _VALID)
    assert calls == 1


def test_patterns_parse_error_runs_degrading_cascade():
    # Always-unparseable, reasoning off → full-retry → model-fallback →
    # SAMPLE. The 6-fact fixture halves once (6→3 via confidence-prioritized
    # halving), then the next SAMPLE can't shrink below the floor so
    # `sample_llm_call` returns None and the kernel STOPS. Calls: root +
    # full-retry + model-fallback + one sampled call = 4. (Pre-async this
    # re-issued the floored call to the 3-sample cap = 6.)
    by_topic, calls = _run(lambda n: "not json {{{")
    assert calls == 4, f"degrading cascade should make 4 calls, made {calls}"
    assert by_topic == {}
