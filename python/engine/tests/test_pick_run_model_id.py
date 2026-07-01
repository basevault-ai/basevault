"""
Unit tests for runner._pick_run_model_id — the function that picks the
run-row's model label honestly from the active stage map.

A run that collapses to a single model shows that model id; a run whose
stages route to more than one model shows the "per-stage" sentinel
(rendered as "per-stage models TEE" by the UI) rather than claiming any
single model.

No live LLM calls. Run with:
    cd engine && pytest tests/test_pick_run_model_id.py -v
"""
from __future__ import annotations



from engine.llm import Mode
from engine.runner import _pick_run_model_id


class TestPickRunModelId:
    def test_single_model_returns_that_model(self):
        # Every stage uses the same model.
        out = _pick_run_model_id(
            mode=Mode.TEE,
            unique_models=["kimi-k2-6"],
            spec_model_id="kimi-k2-6",
        )
        assert out == "kimi-k2-6"

    def test_multiple_models_returns_per_stage(self):
        # Stages route to more than one model; surface "per-stage" so
        # the UI shows "per-stage models TEE".
        out = _pick_run_model_id(
            mode=Mode.TEE,
            unique_models=["gemma4-31b", "gpt-oss-120b", "kimi-k2-6"],
            spec_model_id="gpt-oss-120b",
        )
        assert out == "per-stage"

    def test_empty_stage_map_falls_back_to_spec(self):
        out = _pick_run_model_id(
            mode=Mode.TEE,
            unique_models=[],
            spec_model_id="gpt-oss-120b",
        )
        assert out == "gpt-oss-120b"

    def test_non_tee_mode_returns_spec(self):
        # Mode.LOCAL / Mode.TEE never read stage_models — the helper
        # short-circuits to the mode anchor.
        out = _pick_run_model_id(
            mode=Mode.LOCAL,
            unique_models=["gpt-oss-120b", "kimi-k2-6"],
            spec_model_id="qwen3.5:9b",
        )
        assert out == "qwen3.5:9b"
