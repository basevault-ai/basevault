// TEE provider + model registry for the wizard / settings UI.
//
// Keep in sync with `engine/llm.py` — any (provider, model) listed
// here MUST also be a registered combo in `_build_tee_spec`'s registry,
// otherwise picking it from the dropdown will resolve to a best-effort
// spec swap at runtime (and may silently miss provider-specific quirks
// like `extra_body` or `has_reasoning`).
//
// Tinfoil is the only attested TEE backend. The production app routes
// user data only through TEE or LOCAL; no other cloud provider exists
// in the binary.

// Per-provider model order is fastest → slowest. Default is whatever
// `defaultModel` points to.
export const TEE_PROVIDERS = {
  tinfoil: {
    label: "Tinfoil",
    signupUrl: "https://tinfoil.sh",
    keyPlaceholder: "tf_...",
    keyEnvVar: "TINFOIL_API_KEY",
    verifyCommand: "verify_tinfoil_key",
    // Tagline shown under the "Set up Tinfoil" radio in the wizard
    // and Settings. Keep it short — the trust story in one line.
    tagline:
      "End-to-end encrypted to the enclave. Open-source verifier — don't trust, verify.",
    // Registry of known (provider, model) combos — `isKnownCombo` reads
    // this. Every id MUST also be a registered combo in `_build_tee_spec`'s
    // registry (`engine/llm.py::_MODEL_SPECS`).
    models: [
      { id: "kimi+glm",     label: "Kimi K2.6 + GLM 5.2 — parallel dispatch" },
      { id: "gpt-oss-120b", label: "GPT-OSS 120B — fastest & lightest" },
      { id: "gemma4-31b",   label: "Gemma 4 31B — fast & light" },
      { id: "kimi-k2-6",    label: "Kimi K2.6 — slower, thorough" },
      { id: "glm-5-2",      label: "GLM 5.2 — frontier, 384k ctx" },
    ],
    // Seeds the legacy `tee_model` field + dropdown muscle-memory. Actual
    // per-stage routing keys on `stage_models` (= `defaultStageModels()`),
    // so this only matters as the budget anchor — gpt-oss-120b is the
    // extract-stage model `_build_tee_spec()` returns as the sizing spec.
    defaultModel: "gpt-oss-120b",
  },
};

export const DEFAULT_TEE_PROVIDER = "tinfoil";

export function defaultModelFor(providerId) {
  return TEE_PROVIDERS[providerId]?.defaultModel ?? null;
}

export function isKnownCombo(providerId, modelId) {
  const p = TEE_PROVIDERS[providerId];
  if (!p) return false;
  return p.models.some((m) => m.id === modelId);
}

// ── Display helpers for the run row + anywhere we surface engine info ───────

const PROVIDER_DISPLAY = {
  tinfoil: "tinfoil.sh",
  ollama:  "ollama",
  mlx:     "mlx",
};

export function providerDisplayName(provider) {
  return PROVIDER_DISPLAY[provider] || provider || "";
}

// Display names — explicit dict, ollama-ish lowercase format, "TEE"
// suffix on Tinfoil entries. Ollama / MLX (local) don't get the suffix.
// Add a row when you register a new ModelSpec; unknown ids fall back
// to the raw model_id so the UI never shows blank.
const MODEL_DISPLAY_NAMES = {
  // Tinfoil (TEE)
  "gemma4-31b":   "gemma4-31b TEE",
  "kimi-k2-6":    "kimi-k2.6 TEE",
  "gpt-oss-120b": "gpt-oss-120b TEE",
  "glm-5-2":      "glm-5.2 TEE",
  // Synthetic sentinel for a run whose stages route to more than one
  // model. Real routing lives in config.json's stage_models field; this
  // is the run-row label when no single model id describes the run.
  "per-stage":    "per-stage models TEE",
  // Ollama (local — NOT TEE)
  "qwen3.5:9b":   "qwen3.5:9b",
};

// Within-stage parallel-dispatch sentinels — mirror of
// `llm._MULTI_MODEL_SENTINELS`. A sentinel id like "kimi+glm" routes
// the stage across two `Scheduler` instances (one per constituent)
// pulling from a shared producer (#666). Display = the parallel pair
// with a "(parallel)" suffix so the per-stage dropdown reads naturally.
const MULTI_MODEL_SENTINELS = {
  "kimi+glm": ["kimi-k2.6", "glm-5.2"],
};

export function modelDisplayName(modelId, mode) {
  if (!modelId) return "";
  // Within-stage parallel-dispatch sentinel: "kimi-k2.6 TEE + glm-5.2
  // TEE (parallel)".
  if (modelId in MULTI_MODEL_SENTINELS) {
    const isTee = mode === "tee" || mode === undefined;  // undefined = legacy callers pre-mode
    const parts = MULTI_MODEL_SENTINELS[modelId];
    const body = parts.map((p) => (isTee ? `${p} TEE` : p)).join(" + ");
    return `${body} (parallel)`;
  }
  return MODEL_DISPLAY_NAMES[modelId] || modelId;
}

// ── Per-stage models registry (UI-side mirror of llm._STAGE_MODEL_MAP) ───────
//
// The visible per-stage rows in Settings. metadata is intentionally
// absent — it tracks extract's choice transparently in the pipeline
// (same model + reasoning), which keeps the UI from looking redundant.
// `entities_dedupe` runs ONE LLM call after the per-entity batches to
// merge fuzzy duplicates (e.g. "Mom" + "Jane Doe"); separate row
// because it can afford a stronger model than the bulk per-entity calls.
// `vision` transcribes images during ingest; orthogonal to chat
// routing (vision models can't run chat prompts and vice versa), so it
// carries its own ship default in `stageMapFromPreset`.
export const VISIBLE_STAGES = [
  { id: "vision",          label: "Vision",          hint: "image transcription (during ingest)" },
  { id: "extract",         label: "Extract",         hint: "facts from raw text" },
  { id: "entities",        label: "Entities",        hint: "per-entity canonicalization (~sqrt(N) calls)" },
  { id: "entities_dedupe", label: "Entities dedupe", hint: "single cross-entity merge call" },
  { id: "patterns",        label: "Patterns",        hint: "cross-fact themes per topic" },
  { id: "insights",        label: "Insights",        hint: "high-level synthesis" },
  { id: "actions",         label: "Actions",         hint: "concrete recommendations" },
];

// Models eligible for the per-stage dropdown, keyed by stage id.
// Default key is reused by every chat stage that doesn't have its own
// entry — keeps the historical Tinfoil chat list intact while letting
// the vision row surface vision-capable models. Each entry's id MUST
// also appear in MODEL_DISPLAY_NAMES so the dropdown renders a non-
// blank label. Vision-list order is default-first.
export const STAGE_MODEL_OPTIONS_BY_STAGE = {
  tinfoil: {
    default: [
      "gpt-oss-120b",
      "kimi-k2-6",
      "gemma4-31b",
      "glm-5-2",
      // kimi+glm multi-scheduler parallel dispatch (#666) — offered in
      // the shared default options, not carved out as a patterns-only
      // entry. NOT the ship-default (per #703).
      "kimi+glm",
    ],
    vision: [
      "kimi-k2-6",
    ],
  },
};

// Legacy export kept for back-compat with the pre-vision Settings
// callsite + the contract test in teeProviders.test.js. Equivalent
// to STAGE_MODEL_OPTIONS_BY_STAGE.tinfoil.default — chat-stage list
// only. New callsites should prefer optionsForStage(stageId, providerId).
export const STAGE_MODEL_OPTIONS = {
  tinfoil: STAGE_MODEL_OPTIONS_BY_STAGE.tinfoil.default,
};

export function optionsForStage(stageId, providerId) {
  const byStage = STAGE_MODEL_OPTIONS_BY_STAGE[providerId];
  if (!byStage) return [];
  return byStage[stageId] || byStage.default || [];
}

// Models with a reasoning on/off control. Mirror of llm._REASONING_WHITELIST
// — rows whose selected model isn't in this set grey out the reasoning
// toggle. Keep in sync when adding a model with reasoning support to
// llm._reasoning_kwargs.
export const REASONING_TOGGLE_MODELS = new Set([
  "gpt-oss-120b",
  "kimi-k2-6",
  "gemma4-31b",
  "glm-5-2",
  // The kimi+glm sentinel routes to two real-backend models that both
  // honor reasoning; expose the toggle so the patterns row's flag
  // flows through to both sides of the parallel dispatch.
  "kimi+glm",
]);

// Default per-stage map shipped to fresh installs. Mirror of
// llm._DEFAULT_STAGE_MODELS — heavy/dominant stages (extract +
// entities) route to gpt-oss-120b reasoning ON to escape Tinfoil's
// single Kimi enclave on ~88% of pipeline call volume;
// entities_dedupe routes to gemma4-31b reasoning OFF and patterns to
// kimi-k2-6 reasoning OFF — a large-prompt concurrency bench showed
// reasoning-ON at scale cap-hits/grinds on every model for both
// synthesis stages (gemma's multi-hour merge hang was reasoning-ON
// only), while reasoning-OFF is clean (OFF gemma = best dedupe merge
// correctness; OFF kimi = best patterns content); insights + actions
// stay kimi-k2-6 reasoning OFF; vision stays kimi-k2-6 (no dedicated
// vision backend — the Tinfoil vision model was dropped).
export function defaultStageModels() {
  return {
    vision:          { model: "kimi-k2-6",    reasoning: false },
    extract:         { model: "gpt-oss-120b", reasoning: true  },
    entities:        { model: "gpt-oss-120b", reasoning: true  },
    entities_dedupe: { model: "gemma4-31b",   reasoning: false },
    patterns:        { model: "kimi-k2-6",    reasoning: false },
    insights:        { model: "kimi-k2-6",    reasoning: false },
    actions:         { model: "kimi-k2-6",    reasoning: false },
  };
}

// Config ids that once selected a whole-pipeline model preset (mirror of
// llm._RETIRED_TEE_MODEL_IDS). Per-stage `stage_models` is the only
// routing source now; a legacy config whose `tee_model` still holds one
// of these is migrated to the ship per-stage defaults rather than
// broadcasting the id as a (non-existent) backend model.
const RETIRED_TEE_MODEL_IDS = new Set(["mixed-gpt-oss-kimi-k2-6"]);

// Translate a legacy single-model `tee_model` id into a per-stage map,
// used to hydrate the Settings rows for a config that predates the
// `stage_models` field. Broadcasts the id to every chat stage with
// reasoning OFF; vision is not part of chat routing (vision models can't
// run chat prompts) so it stays at the ship default. A retired preset id
// (or empty id) yields the full ship default map.
export function stageMapFromPreset(presetModelId) {
  const visionDefault = defaultStageModels().vision;
  if (!presetModelId || RETIRED_TEE_MODEL_IDS.has(presetModelId)) {
    return defaultStageModels();
  }
  return {
    vision:          { ...visionDefault },
    extract:         { model: presetModelId, reasoning: false },
    entities:        { model: presetModelId, reasoning: false },
    entities_dedupe: { model: presetModelId, reasoning: false },
    patterns:        { model: presetModelId, reasoning: false },
    insights:        { model: presetModelId, reasoning: false },
    actions:         { model: presetModelId, reasoning: false },
  };
}
