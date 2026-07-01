"""Offline validation for the migrated INSIGHTS degrading cascade (#912).

The extraction spike validated only the NON-degrading (halve) ladder. This
drives the migrated INSIGHTS phase — a single-call DEGRADING stage — with a
scripted provider to prove the kernel's degrading ``RetryPolicy`` fans the
sample cascade exactly as the legacy stage would: on repeated PARSE_ERROR a
degrading stage takes full-retry → model-fallback → SAMPLE×3 → stop (6
provider calls, reasoning off), and a clean response is a single call.

Parse fidelity (the entry-level cross/critical gates) is the live L2 eval's
job; this isolates the retry/sample MECHANICS the migration owns.
"""
from __future__ import annotations


from engine.insights import Pattern
from kernel.abstractions import InferenceProvider, LlmResponse
from kernel.enums import PhaseName
from kernel.execution_env import ExecutionEnv

from engine.llm import ModelSpec as LegacyModelSpec, Mode
from engine.phases.insights import InsightsJob
from engine.phases.model_specs import PipelineModelSpec


def _patterns():
    # 2 topics × 2 patterns → strong has ≥ min_topics (not critical-only).
    return {
        "work": [
            Pattern(name="overcommits", description="says yes too often",
                    domain="work", count=4),
            Pattern(name="late nights", description="works past midnight",
                    domain="work", count=3),
        ],
        "health": [
            Pattern(name="skips meals", description="forgets lunch",
                    domain="health", count=3),
            Pattern(name="poor sleep", description="under six hours",
                    domain="health", count=2),
        ],
    }


_VALID = '{"cross_domain": [], "critical": [{"name": "x"}]}'  # validate → OK


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


def _run(responder):
    provider = _Scripted(responder)
    legacy = LegacyModelSpec(
        provider="scripted", model_id="gpt-oss-120b", context_window=131_000
    )
    spec = PipelineModelSpec(provider, legacy.model_id, legacy.context_window, max_parallelism=4)
    env = ExecutionEnv()
    env.register_spec(PhaseName.INSIGHTS, spec, spec, thinking=False)

    job = InsightsJob({"patterns_by_topic": _patterns(), "mode": Mode.TEE,
                       "total_facts": 200, "min_topics": 2})
    out = job.run(job.initial_input(), env)
    return out.data["output"], provider.call_count


def test_insights_clean_success_single_call():
    output, calls = _run(lambda n: _VALID)
    assert calls == 1
    # Output is a well-formed InsightOutput (entry gates may reject the
    # toy entry → 0 insights; the contract under test is "single call").
    assert hasattr(output, "cross_domain") and hasattr(output, "critical")


def test_insights_parse_error_runs_degrading_cascade():
    # Always-unparseable, reasoning off → full-retry → model-fallback →
    # SAMPLE. With this 4-pattern fixture `_next_detail_keys` is already at
    # the synthesis floor, so the first SAMPLE returns None and the kernel
    # STOPS (resolving with the last response). Calls: root + full-retry +
    # model-fallback = 3. (Pre-async this re-issued the floored call up to
    # the 3-sample cap = 6; the None stop signal is the corrected behavior.)
    output, calls = _run(lambda n: "this is not json {{{")
    assert calls == 3, f"degrading cascade should make 3 calls, made {calls}"
    assert output.cross_domain == [] and output.critical == []
