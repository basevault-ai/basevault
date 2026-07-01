import { describe, it, expect } from "vitest";
import {
  TEE_PROVIDERS,
  DEFAULT_TEE_PROVIDER,
  defaultModelFor,
  isKnownCombo,
  providerDisplayName,
  modelDisplayName,
  defaultStageModels,
  stageMapFromPreset,
  STAGE_MODEL_OPTIONS,
  REASONING_TOGGLE_MODELS,
  VISIBLE_STAGES,
  optionsForStage,
} from "./teeProviders";

describe("TEE_PROVIDERS shape", () => {
  it("exposes tinfoil as the only attested provider", () => {
    // Production user-data routing is TEE-only or LOCAL — Tinfoil is
    // the only attested TEE backend in the binary.
    expect(Object.keys(TEE_PROVIDERS).sort()).toEqual(["tinfoil"]);
  });

  it("each provider has a defaultModel that exists in its model list", () => {
    for (const [_id, p] of Object.entries(TEE_PROVIDERS)) {
      const ids = p.models.map((m) => m.id);
      expect(ids).toContain(p.defaultModel);
      expect(p.label).toBeTruthy();
      expect(p.keyEnvVar).toBeTruthy();
      expect(p.verifyCommand).toBeTruthy();
    }
  });

  it("DEFAULT_TEE_PROVIDER is a registered provider", () => {
    expect(TEE_PROVIDERS[DEFAULT_TEE_PROVIDER]).toBeDefined();
  });
});

describe("defaultModelFor", () => {
  it("returns the registered default for known providers", () => {
    // Re-rooted on the extract/budget-anchor model after the mixed-mode
    // preset teardown — actual routing keys on stage_models, this only
    // seeds the legacy tee_model + dropdown muscle-memory.
    expect(defaultModelFor("tinfoil")).toBe("gpt-oss-120b");
  });

  it("returns null for unknown providers", () => {
    expect(defaultModelFor("nope")).toBeNull();
    expect(defaultModelFor(undefined)).toBeNull();
    expect(defaultModelFor("")).toBeNull();
  });
});

describe("isKnownCombo", () => {
  it("accepts registered (provider, model) pairs", () => {
    expect(isKnownCombo("tinfoil", "gpt-oss-120b")).toBe(true);
    expect(isKnownCombo("tinfoil", "kimi-k2-6")).toBe(true);
    // glm-5-2 wired into the selector (has a backend ModelSpec).
    expect(isKnownCombo("tinfoil", "glm-5-2")).toBe(true);
  });

  it("rejects the retired whole-pipeline preset id", () => {
    // mixed-gpt-oss-kimi-k2-6 was removed from the model list — it's no
    // longer a selectable combo.
    expect(isKnownCombo("tinfoil", "mixed-gpt-oss-kimi-k2-6")).toBe(false);
  });

  it("rejects unregistered model ids on tinfoil", () => {
    // Tinfoil is the only registered provider; an id not in its model
    // list returns false. Unknown providers are covered separately.
    expect(isKnownCombo("tinfoil", "some/unregistered/model")).toBe(false);
  });

  it("rejects unknown providers", () => {
    expect(isKnownCombo("nope", "gpt-oss-120b")).toBe(false);
  });
});

describe("providerDisplayName", () => {
  it("maps known providers to their human label", () => {
    expect(providerDisplayName("tinfoil")).toBe("tinfoil.sh");
    expect(providerDisplayName("ollama")).toBe("ollama");
    expect(providerDisplayName("mlx")).toBe("mlx");
  });

  it("falls back to the raw id for unknown providers", () => {
    expect(providerDisplayName("custom")).toBe("custom");
  });

  it("returns empty string for null/undefined", () => {
    expect(providerDisplayName(null)).toBe("");
    expect(providerDisplayName(undefined)).toBe("");
    expect(providerDisplayName("")).toBe("");
  });
});

describe("modelDisplayName", () => {
  it("formats known TEE model ids with the TEE suffix", () => {
    expect(modelDisplayName("gpt-oss-120b")).toBe("gpt-oss-120b TEE");
    expect(modelDisplayName("kimi-k2-6")).toBe("kimi-k2.6 TEE");
  });

  it("formats local (ollama) models without a TEE suffix", () => {
    expect(modelDisplayName("qwen3.5:9b")).toBe("qwen3.5:9b");
  });

  it("falls back to the raw id for unregistered models", () => {
    expect(modelDisplayName("some/new/model")).toBe("some/new/model");
  });

  it("returns empty string for null/undefined/empty", () => {
    expect(modelDisplayName(null)).toBe("");
    expect(modelDisplayName(undefined)).toBe("");
    expect(modelDisplayName("")).toBe("");
  });
});

describe("defaultStageModels", () => {
  it("returns one entry per visible stage", () => {
    const map = defaultStageModels();
    const stageIds = VISIBLE_STAGES.map((s) => s.id);
    for (const sid of stageIds) {
      expect(map[sid]).toBeDefined();
      expect(map[sid].model).toBeTruthy();
      expect(typeof map[sid].reasoning).toBe("boolean");
    }
  });

  it("ships extract + entities on gpt-oss-120b reasoning ON (contention escape)", () => {
    // Mirror of llm._DEFAULT_STAGE_MODELS — the heavy/dominant stages
    // route off the single-enclave Kimi pinch point onto gpt-oss-120b
    // (3 Tinfoil enclaves), reasoning ON.
    const map = defaultStageModels();
    expect(map.extract).toEqual({ model: "gpt-oss-120b", reasoning: true });
    expect(map.entities).toEqual({ model: "gpt-oss-120b", reasoning: true });
  });

  it("ships entities_dedupe on gemma4-31b reasoning OFF", () => {
    const map = defaultStageModels();
    expect(map.entities_dedupe).toEqual({ model: "gemma4-31b", reasoning: false });
  });

  it("ships patterns on kimi-k2-6 reasoning OFF", () => {
    const map = defaultStageModels();
    expect(map.patterns).toEqual({ model: "kimi-k2-6", reasoning: false });
  });

  it("ships insights + actions + vision on kimi-k2-6 reasoning OFF", () => {
    const map = defaultStageModels();
    expect(map.insights).toEqual({ model: "kimi-k2-6", reasoning: false });
    expect(map.actions).toEqual({ model: "kimi-k2-6", reasoning: false });
    expect(map.vision).toEqual({ model: "kimi-k2-6", reasoning: false });
  });
});

describe("stageMapFromPreset", () => {
  it("broadcasts a single-model id to every chat stage", () => {
    const map = stageMapFromPreset("kimi-k2-6");
    // Vision is excluded from chat-stage broadcasting — chat routing
    // only, vision keeps its own ship default.
    for (const [stageId, stage] of Object.entries(map)) {
      if (stageId === "vision") continue;
      expect(stage.model).toBe("kimi-k2-6");
      expect(stage.reasoning).toBe(false);
    }
  });

  it("vision keeps its own ship default when a chat id is broadcast", () => {
    const expected = defaultStageModels().vision;
    const map = stageMapFromPreset("gpt-oss-120b");
    expect(map.vision).toEqual(expected);
  });

  it("single-model broadcast rows always start with reasoning OFF", () => {
    const single = stageMapFromPreset("gpt-oss-120b");
    for (const stage of Object.values(single)) {
      expect(stage.reasoning).toBe(false);
    }
  });

  it("migrates a retired whole-pipeline preset id to the ship default map", () => {
    // A legacy config whose tee_model still holds the removed sentinel
    // must NOT broadcast the (non-existent) id to every stage — it falls
    // back to the ship default per-stage map.
    const map = stageMapFromPreset("mixed-gpt-oss-kimi-k2-6");
    expect(map).toEqual(defaultStageModels());
    for (const stage of Object.values(map)) {
      expect(stage.model).not.toBe("mixed-gpt-oss-kimi-k2-6");
    }
  });

  it("migrates an empty id to the ship default map", () => {
    expect(stageMapFromPreset("")).toEqual(defaultStageModels());
  });
});

describe("STAGE_MODEL_OPTIONS / REASONING_TOGGLE_MODELS contract", () => {
  it("every per-stage option is also in the provider model list", () => {
    for (const [provider, options] of Object.entries(STAGE_MODEL_OPTIONS)) {
      const modelIds = TEE_PROVIDERS[provider].models.map((m) => m.id);
      for (const opt of options) {
        expect(modelIds).toContain(opt);
      }
    }
  });

  it("REASONING_TOGGLE_MODELS only contains tinfoil-side ids (UI guarantee)", () => {
    const tinfoilIds = TEE_PROVIDERS.tinfoil.models.map((m) => m.id);
    for (const id of REASONING_TOGGLE_MODELS) {
      expect(tinfoilIds).toContain(id);
    }
  });
});

describe("kimi+glm multi-scheduler preset (#666)", () => {
  it("is a selectable Tinfoil preset but NOT the ship-default", () => {
    const ids = TEE_PROVIDERS.tinfoil.models.map((m) => m.id);
    expect(ids).toContain("kimi+glm");
    // kimi+glm is opt-in; the ship default re-rooted on the budget
    // anchor after the mixed-mode preset teardown.
    expect(defaultModelFor("tinfoil")).toBe("gpt-oss-120b");
  });

  it("renders the within-stage sentinel as the parallel pair", () => {
    expect(modelDisplayName("kimi+glm", "tee")).toBe(
      "kimi-k2.6 TEE + glm-5.2 TEE (parallel)",
    );
  });

  it("is offered in the patterns per-stage dropdown", () => {
    expect(optionsForStage("patterns", "tinfoil")).toContain("kimi+glm");
  });

  it("exposes a reasoning toggle for the sentinel", () => {
    expect(REASONING_TOGGLE_MODELS.has("kimi+glm")).toBe(true);
  });
});
