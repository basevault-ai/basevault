"""
Config migration tests for stage_models.

Verifies the ladder in `llm.resolve_stage_models_from_config`:

  1. cfg.stage_models present → use it (sanitized, partial maps filled
     from the broadcast layer).
  2. cfg.tee_model is a single model id → broadcast to every stage,
     reasoning OFF.
  3. neither set (or tee_model is a retired preset id) → ship default
     (see _DEFAULT_STAGE_MODELS — extract+entities on gpt-oss-120b
     reasoning ON, entities_dedupe on gemma4-31b reasoning OFF,
     patterns/insights/actions on kimi-k2-6 reasoning OFF).

No live LLM calls. Run with:
    cd engine && pytest tests/test_config_migration.py -v
"""
from __future__ import annotations



from engine.llm import (
    resolve_stage_models_from_config,
    _DEFAULT_STAGE_MODELS,
    _RETIRED_TEE_MODEL_IDS,
)


# ── Fresh-default ladder ──────────────────────────────────────────────────────

class TestFreshInstall:
    def test_empty_config_returns_ship_default_map(self):
        m = resolve_stage_models_from_config({})
        # Heavy/dominant stages route off the single-enclave Kimi
        # pinch point onto gpt-oss-120b reasoning ON.
        assert m["extract"]["model"] == "gpt-oss-120b"
        assert m["extract"]["reasoning"] is True
        assert m["entities"]["model"] == "gpt-oss-120b"
        assert m["entities"]["reasoning"] is True
        # Synthesis stages run reasoning OFF (reasoning-ON cap-hits/grinds
        # at scale per the large-prompt concurrency bench): gemma on dedupe
        # (best merge correctness; its multi-hour merge hang was reasoning-
        # ON only), kimi on patterns (best per-topic content).
        assert m["entities_dedupe"]["model"] == "gemma4-31b"
        assert m["entities_dedupe"]["reasoning"] is False
        assert m["patterns"]["model"] == "kimi-k2-6"
        assert m["patterns"]["reasoning"] is False
        # Tail synthesis stays on kimi reasoning OFF.
        assert m["insights"]["model"] == "kimi-k2-6"
        assert m["insights"]["reasoning"] is False
        assert m["actions"]["model"] == "kimi-k2-6"
        assert m["actions"]["reasoning"] is False

    def test_ship_default_matches_module_constant(self):
        m = resolve_stage_models_from_config({})
        # Sanity: every stage in the resolved map matches
        # _DEFAULT_STAGE_MODELS exactly.
        for stage, expected in _DEFAULT_STAGE_MODELS.items():
            assert m[stage]["model"] == expected["model"], stage
            assert m[stage]["reasoning"] == expected["reasoning"], stage


# ── Retired preset migration ──────────────────────────────────────────────────

class TestRetiredPresetMigration:
    """Legacy users whose `tee_model` still holds a retired whole-pipeline
    preset id (e.g. "mixed-gpt-oss-kimi-k2-6") and no stage_models field.
    The id is treated as unset → ship default per-stage map, NOT
    broadcast as a (non-existent) backend model to every stage."""

    def test_retired_id_falls_back_to_ship_default(self):
        for retired in _RETIRED_TEE_MODEL_IDS:
            m = resolve_stage_models_from_config({"tee_model": retired})
            for stage, expected in _DEFAULT_STAGE_MODELS.items():
                assert m[stage]["model"] == expected["model"], (retired, stage)
                assert m[stage]["reasoning"] == expected["reasoning"], (retired, stage)

    def test_retired_id_not_broadcast_as_model(self):
        # The bug this guards against: a removed sentinel falling through
        # the single-model broadcast branch and routing every stage to a
        # model id that isn't registered.
        for retired in _RETIRED_TEE_MODEL_IDS:
            m = resolve_stage_models_from_config({"tee_model": retired})
            for stage in ("extract", "entities", "patterns", "insights", "actions"):
                assert m[stage]["model"] != retired, (retired, stage)


# ── Single-model legacy migration ─────────────────────────────────────────────

class TestSingleModelMigration:
    """Legacy users with `tee_model: "kimi-k2-6"` (or any single model)
    and no stage_models. Resolver broadcasts that model to every stage."""

    def test_single_model_broadcasts_to_every_stage(self):
        m = resolve_stage_models_from_config({"tee_model": "kimi-k2-6"})
        for stage in ("extract", "entities",
                      "patterns", "insights", "actions"):
            assert m[stage]["model"] == "kimi-k2-6", stage
            assert m[stage]["reasoning"] is False, stage

    def test_unregistered_model_broadcasts_anyway(self):
        # Resolver doesn't validate against _MODEL_SPECS — it leaves
        # that to per-call resolution so the error surfaces with full
        # context. Migration must not silently drop a config the user
        # set deliberately.
        m = resolve_stage_models_from_config({"tee_model": "future-model-x"})
        assert m["extract"]["model"] == "future-model-x"


# ── stage_models present (new shape) ──────────────────────────────────────────

class TestStageModelsPresent:
    def test_exact_user_map_passes_through(self):
        cfg = {
            "stage_models": {
                "extract":  {"model": "gpt-oss-120b",   "reasoning": False},
                "entities": {"model": "glm-5-2", "reasoning": False},
                "patterns": {"model": "glm-5-2", "reasoning": True},
                "insights": {"model": "kimi-k2-6",       "reasoning": False},
                "actions":  {"model": "glm-5-2", "reasoning": False},
            },
        }
        m = resolve_stage_models_from_config(cfg)
        assert m["extract"]["model"] == "gpt-oss-120b"
        assert m["entities"]["model"] == "glm-5-2"
        assert m["patterns"]["model"] == "glm-5-2"
        assert m["patterns"]["reasoning"] is True
        assert m["insights"]["model"] == "kimi-k2-6"
        assert m["actions"]["model"] == "glm-5-2"

    def test_partial_map_fills_from_broadcast_layer(self):
        # User only set extract; the other stages should fall back to
        # whatever the legacy layer says (here: tee_model="gemma4-31b"
        # → broadcast to every other stage).
        cfg = {
            "tee_model": "gemma4-31b",
            "stage_models": {
                "extract":  {"model": "gpt-oss-120b", "reasoning": False},
            },
        }
        m = resolve_stage_models_from_config(cfg)
        assert m["extract"]["model"] == "gpt-oss-120b"
        # entities/patterns/insights/actions inherit from broadcast.
        for stage in ("entities", "patterns", "insights", "actions"):
            assert m[stage]["model"] == "gemma4-31b", stage

    def test_malformed_entry_is_ignored(self):
        # A genuinely malformed entry — a non-str/non-dict type, or a
        # dict missing "model" — is dropped and falls through to the
        # broadcast layer (here: ship default, tee_model unset). Note a
        # bare *string* is NOT malformed: it's accepted as a model-id
        # shorthand (covered below); an unregistered id surfaces fail-loud
        # at the per-call resolver, not here.
        cfg = {
            "stage_models": {
                "extract":  {"model": "gpt-oss-120b", "reasoning": False},
                "entities": ["not", "a", "dict"],   # wrong type → dropped
                "patterns": {"reasoning": True},     # missing model → dropped
            },
        }
        m = resolve_stage_models_from_config(cfg)
        assert m["extract"]["model"] == "gpt-oss-120b"
        assert m["entities"]["model"] == _DEFAULT_STAGE_MODELS["entities"]["model"]
        assert m["patterns"]["model"] == _DEFAULT_STAGE_MODELS["patterns"]["model"]

    def test_string_entry_is_model_id_shorthand(self):
        # A bare string is shorthand for {"model": <id>, "reasoning": False}
        # — not a malformed entry. The id isn't validated here (the
        # per-call resolver does the spec lookup, fail-loud on unknowns).
        cfg = {
            "stage_models": {
                "entities": "gemma4-31b",
            },
        }
        m = resolve_stage_models_from_config(cfg)
        assert m["entities"] == {"model": "gemma4-31b", "reasoning": False}

    def test_stage_models_supersedes_tee_model(self):
        # User has both fields set; stage_models wins (it's the new
        # source of truth). tee_model gets ignored for routing.
        cfg = {
            "tee_model": "mixed-gpt-oss-kimi-k2-6",
            "stage_models": {
                "extract":  {"model": "gemma4-31b", "reasoning": False},
                "entities": {"model": "gemma4-31b", "reasoning": False},
                "patterns": {"model": "gemma4-31b", "reasoning": False},
                "insights": {"model": "gemma4-31b", "reasoning": False},
                "actions":  {"model": "gemma4-31b", "reasoning": False},
            },
        }
        m = resolve_stage_models_from_config(cfg)
        for stage in ("extract", "entities",
                      "patterns", "insights", "actions"):
            assert m[stage]["model"] == "gemma4-31b", stage


# ── unique_models_in_stage_map (downstream of migration) ─────────────────────

class TestUniqueModels:
    def test_three_distinct_models_dedupe_in_pipeline_order(self):
        from engine.llm import unique_models_in_stage_map
        sm = {
            "extract":  {"model": "gpt-oss-120b",   "reasoning": False},
            "entities": {"model": "kimi-k2-6",      "reasoning": False},
            "patterns": {"model": "glm-5-2", "reasoning": False},
            "insights": {"model": "glm-5-2", "reasoning": False},
            "actions":  {"model": "glm-5-2", "reasoning": False},
        }
        out = unique_models_in_stage_map(sm)
        assert out == ["gpt-oss-120b", "kimi-k2-6", "glm-5-2"]

    def test_empty_map_returns_empty_list(self):
        from engine.llm import unique_models_in_stage_map
        assert unique_models_in_stage_map({}) == []
