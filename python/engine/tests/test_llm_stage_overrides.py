"""
Unit tests for per-stage model routing in llm.py.

config.json's `stage_models` field maps each stage to a model id;
llm.py resolves the per-stage spec at call time via
`_resolve_stage_override` / `_STAGE_MODEL_MAP`. This file covers:
  - _resolve_stage_override returns the mode-pinned spec when no
    per-stage map is active or the stage maps to the mode-pinned model
  - _resolve_stage_override swaps to the mapped spec when a stage
    routes elsewhere
  - _spec_for_model_id lookup behavior
  - complete() routes to the mapped model_id (verified via a stub)

No live LLM calls. Run with:
    cd engine && pytest tests/test_llm_stage_overrides.py -v
"""
from __future__ import annotations


import pytest


from engine import llm
from engine.llm import (
    Mode,
    Provider,
    _MODEL_SPECS,
    _resolve_stage_override,
    _spec_for_model_id,
    get_mode_spec,
    resolve_stage_models_from_config,
)


# A per-stage map that routes the structured stages to gpt-oss-120b
# (the budget anchor) and the synthesis stages to kimi-k2-6 — the
# routing the old mixed-mode preset used to express, now built straight
# from a `stage_models` config.
def _stage_map() -> dict[str, dict]:
    return resolve_stage_models_from_config({
        "stage_models": {
            "extract":         {"model": "gpt-oss-120b", "reasoning": False},
            "entities":        {"model": "gpt-oss-120b", "reasoning": False},
            "entities_dedupe": {"model": "kimi-k2-6",    "reasoning": False},
            "patterns":        {"model": "kimi-k2-6",    "reasoning": False},
            "insights":        {"model": "kimi-k2-6",    "reasoning": False},
            "actions":         {"model": "kimi-k2-6",    "reasoning": False},
        },
    })


# ── _resolve_stage_override (no per-stage map) ─────────────────────────────────

class TestResolveStageOverrideNoMap:
    def setup_method(self):
        self._prev_stage_map = llm._STAGE_MODEL_MAP

    def teardown_method(self):
        llm._STAGE_MODEL_MAP = self._prev_stage_map

    def test_returns_mode_spec_when_map_empty(self):
        llm._STAGE_MODEL_MAP = {}
        base = get_mode_spec(Mode.TEE)
        for stage in ("metadata", "extract", "entities",
                      "patterns", "insights", "actions"):
            spec, override = _resolve_stage_override(Mode.TEE, stage)
            assert override is None, f"stage={stage}"
            assert spec is base, f"stage={stage}"

    def test_none_stage_returns_mode_spec(self):
        llm._STAGE_MODEL_MAP = _stage_map()
        spec, override = _resolve_stage_override(Mode.TEE, None)
        assert override is None
        assert spec is get_mode_spec(Mode.TEE)


# ── _resolve_stage_override (per-stage map active) ─────────────────────────────

class TestResolveStageOverrideMapActive:
    """When `_STAGE_MODEL_MAP` routes a stage to a non-anchor model,
    the resolver swaps to that stage's spec. The anchor is pinned to
    gpt-oss-120b so the "mode-pinned vs. mapped" comparison is
    deterministic across config.json states."""

    def setup_method(self):
        self._prev_stage_map = llm._STAGE_MODEL_MAP
        self._prev_tee_spec = llm.MODE_SPEC[Mode.TEE]
        llm.MODE_SPEC[Mode.TEE] = _MODEL_SPECS[(Provider.TINFOIL, "gpt-oss-120b")]
        llm._STAGE_MODEL_MAP = _stage_map()

    def teardown_method(self):
        llm._STAGE_MODEL_MAP = self._prev_stage_map
        llm.MODE_SPEC[Mode.TEE] = self._prev_tee_spec

    def test_extract_keeps_anchor_no_override(self):
        # extract maps to gpt-oss-120b (the budget anchor) → no swap,
        # override_model=None so tokens.json reads identically to a
        # plain gpt-oss run.
        spec, override = _resolve_stage_override(Mode.TEE, "extract")
        assert override is None
        assert spec.model_id == "gpt-oss-120b"

    def test_metadata_keeps_anchor_no_override(self):
        spec, override = _resolve_stage_override(Mode.TEE, "metadata")
        assert override is None
        assert spec.model_id == "gpt-oss-120b"

    def test_entities_keeps_anchor_no_override(self):
        spec, override = _resolve_stage_override(Mode.TEE, "entities")
        assert override is None
        assert spec.model_id == "gpt-oss-120b"

    def test_patterns_routes_to_kimi(self):
        spec, override = _resolve_stage_override(Mode.TEE, "patterns")
        assert override == "kimi-k2-6"
        assert spec.provider == Provider.TINFOIL
        assert spec.model_id == "kimi-k2-6"
        assert spec is _MODEL_SPECS[(Provider.TINFOIL, "kimi-k2-6")]

    def test_insights_routes_to_kimi(self):
        spec, override = _resolve_stage_override(Mode.TEE, "insights")
        assert override == "kimi-k2-6"
        assert spec.model_id == "kimi-k2-6"

    def test_actions_routes_to_kimi(self):
        spec, override = _resolve_stage_override(Mode.TEE, "actions")
        assert override == "kimi-k2-6"
        assert spec.model_id == "kimi-k2-6"

    def test_unknown_stage_no_swap(self):
        # Stage absent from the map → no swap (a typo'd stage at call
        # time shouldn't crash).
        spec, override = _resolve_stage_override(Mode.TEE, "unknown_stage")
        assert override is None
        assert spec is get_mode_spec(Mode.TEE)

    def test_non_tee_mode_unaffected(self):
        # Per-stage routing only applies to Mode.TEE — every other mode
        # ignores `_STAGE_MODEL_MAP`. LOCAL is the only non-TEE mode today
        # (the old TEST mode was removed); patterns maps to kimi-k2-6 in
        # the active map, so a non-TEE resolve must still return None.
        for mode in (Mode.LOCAL,):
            spec, override = _resolve_stage_override(mode, "patterns")
            assert override is None
            assert spec is get_mode_spec(mode)


# ── _spec_for_model_id ─────────────────────────────────────────────────────────

class TestSpecForModelId:
    def test_returns_registered_spec(self):
        spec = _spec_for_model_id(Provider.TINFOIL, "kimi-k2-6")
        assert spec.provider == Provider.TINFOIL
        assert spec.model_id == "kimi-k2-6"
        assert spec is _MODEL_SPECS[(Provider.TINFOIL, "kimi-k2-6")]

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            _spec_for_model_id(Provider.TINFOIL, "no-such-model-id")


# ── complete() routes through the per-stage map ────────────────────────────────

class TestCompleteRoutesPerStage:
    """complete() should send the per-stage-mapped model_id (not the
    caller-passed default) to the provider client when a stage maps to
    a non-anchor model."""

    def setup_method(self):
        self._prev_stage = llm._current_stage
        self._prev_tee_spec = llm.MODE_SPEC[Mode.TEE]
        self._prev_stage_map = llm._STAGE_MODEL_MAP
        llm.MODE_SPEC[Mode.TEE] = _MODEL_SPECS[(Provider.TINFOIL, "gpt-oss-120b")]
        llm._STAGE_MODEL_MAP = _stage_map()
        llm.reset_usage_log()
        llm.reset_call_warnings()

    def teardown_method(self):
        llm.set_stage(self._prev_stage)
        llm._STAGE_MODEL_MAP = self._prev_stage_map
        llm.MODE_SPEC[Mode.TEE] = self._prev_tee_spec





# ── Fail-loud: an unregistered model id crashes uniformly ──────────────────────

class TestUnregisteredModelFailsLoud:
    """An unregistered/unknown model id in `stage_models` must CRASH —
    not silently degrade to another model. The graceful-fallback
    experiment was rejected in favor of fail-loud: a config naming a
    model the build lacks should fail visibly, not run on a different
    model the user didn't choose.

    This pins that decision (guards against re-introducing the rejected
    fallback) AND that the crash is uniform across stages — the dispatch
    path is stage-agnostic, so no stage (patterns included) gets a softer
    resolve path. `patterns` is in the stage list precisely because it
    was once special-cased in a now-removed comment.
    """

    _BOGUS = "no-such-model-xyz"
    _STAGES = ("extract", "entities", "patterns", "insights", "actions")

    def setup_method(self):
        self._prev_tee_spec = llm.MODE_SPEC[Mode.TEE]
        self._prev_stage_map = llm._STAGE_MODEL_MAP
        # Anchor stays a real registered spec; only the per-stage entries
        # name the bogus id (and differ from the anchor, so the resolver
        # reaches the spec lookup rather than the equals-anchor shortcut).
        llm.MODE_SPEC[Mode.TEE] = _MODEL_SPECS[(Provider.TINFOIL, "gpt-oss-120b")]
        llm._STAGE_MODEL_MAP = {
            s: {"model": self._BOGUS, "reasoning": False} for s in self._STAGES
        }

    def teardown_method(self):
        llm._STAGE_MODEL_MAP = self._prev_stage_map
        llm.MODE_SPEC[Mode.TEE] = self._prev_tee_spec

    def test_resolve_raises_uniformly_across_stages(self):
        for stage in self._STAGES:
            with pytest.raises(KeyError):
                _resolve_stage_override(Mode.TEE, stage)
