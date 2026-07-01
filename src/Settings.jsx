import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { confirm as askConfirm } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import {
  TEE_PROVIDERS,
  DEFAULT_TEE_PROVIDER,
  defaultModelFor,
  isKnownCombo,
  VISIBLE_STAGES,
  STAGE_MODEL_OPTIONS,
  optionsForStage,
  REASONING_TOGGLE_MODELS,
  defaultStageModels,
  stageMapFromPreset,
  modelDisplayName,
} from "./teeProviders";
import { useLocalModel } from "./useLocalModel";
import { localBackendReady } from "./localUsable";
import ErrorWithTrace from "./ErrorWithTrace";

// In a Tauri webview, plain `<a target="_blank">` does nothing — the
// internal navigation is blocked and there's no browser handoff. Use
// the opener plugin to actually open the URL in the system browser.
function externalLinkClick(url) {
  return (e) => {
    e.preventDefault();
    openUrl(url).catch((err) => console.error("openUrl failed:", err));
  };
}

// Ship-default per-stage model map. Frozen snapshot of `defaultStageModels()`
// captured at module load — used both as the target of the
// "Reset to defaults" button and to mark the default option in each
// per-stage selector with a " (default)" suffix.
const SHIP_DEFAULT_STAGE_MODELS = defaultStageModels();

// Ship-default chatbot entry. Persists to the top-level `chatbot`
// config field. Mirrors `chatbot.DEFAULT_CHATBOT_MODEL` /
// `.DEFAULT_CHATBOT_REASONING` — keep this in sync with the Python
// resolver, which is the single source of truth for the ship-default.
// Reasoning ships OFF (reasoning-ON-by-default was too slow on the
// attested route); it stays a fully honored opt-in the toggle controls
// verbatim either way.
const DEFAULT_CHATBOT = { model: "glm-5-2", reasoning: false };

// Default topic taxonomy. Keep in sync with
// `engine/content_extractor.py::_DEFAULT_TOPICS`. Settings → General
// → Categories overrides at runtime via `app_config.categories`; this
// constant is the seed and the target of "Reset to defaults."
// Alphabetical so the seed list is scannable; user-added rows append
// at the end (insertion order is preserved on save).
const DEFAULT_CATEGORIES = [
  "admin", "education", "family", "finance", "health",
  "housing", "legal", "logistics", "other", "relationships",
  "spirituality", "travel", "work",
];

// Validation for an individual category slug. Lowercase ASCII letters,
// digits, and hyphens. Empty strings + duplicates surface as separate
// hints so the user can tell which rule failed.
const CATEGORY_SLUG_RE = /^[a-z0-9][a-z0-9-]*$/;

function categoryError(value, otherValues) {
  const v = value.trim();
  if (!v) return "empty";
  if (!CATEGORY_SLUG_RE.test(v)) return "shape";
  if (otherValues.some((o) => o.trim() === v)) return "duplicate";
  return null;
}

function categoriesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// Keep in sync with engine/runner.py _VALID_SENTIMENTS / _DEFAULT_SENTIMENT.
// Order is the dropdown order — runs from sharpest critique to most
// uplifting, with the default "neutral" centered.
const SENTIMENT_OPTIONS = [
  {
    value: "brutally-honest",
    label: "Brutally honest",
    hint: "Surface failure modes directly; drop hedging qualifiers.",
  },
  {
    value: "critical",
    label: "Critical",
    hint: "Lean into shadow patterns with proportionality.",
  },
  {
    value: "neutral",
    label: "Neutral",
    hint: "Describe what is. No praise, no judgment. (Default)",
  },
  {
    value: "uplifting",
    label: "Uplifting",
    hint: "Lead with capability and trajectory. No fabricated positives.",
  },
  {
    value: "bubbly",
    label: "Bubbly",
    hint: "Maximally encouraging. Defensive patterns reframed as growth.",
  },
];
const DEFAULT_SENTIMENT = "neutral";

// Keep in sync with engine/llm.py (_build_local_spec / DEFAULT_MLX_MODEL)
// and Wizard.jsx. MLX is the primary local path; qwen3.5:9b is the
// Ollama-path model id when the user opts into Ollama.
const DEFAULT_LOCAL_MODEL = "qwen3.5:9b";
const DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-9B-4bit";

/**
 * Settings modal: identity + local + TEE + sentiment + cache. Mirrors
 * the Wizard's section vocabulary so the same idle/verifying/verified/
 * failed state machine and "Reverify" semantics apply across both
 * surfaces.
 *
 * Unlike the wizard, Settings edits are staged and saved on [Save]. The
 * "Run setup" / "Verify key" actions fire immediately and live outside
 * the staging flow — they mutate real state (install software, check
 * keys) that isn't undoable by hitting Cancel.
 *
 * Obsidian setup lives in the first-run Wizard, not Settings: power
 * users who want to point Obsidian at the vault dir do it from the
 * Wizard or by opening the folder directly. The redesigned 3-pane nav
 * (issue #77) renders runs and outputs natively, so most users never
 * need Obsidian for the day-to-day flow.
 */
export default function Settings({ onClose, onChanged, onResetPaneSizes }) {
  const [loaded, setLoaded] = useState(null);       // original dotenv state
  const [config, setConfig] = useState(null);       // original app config

  // Name. Per-provider TEE state lives in `providers` below.
  const [name, setName] = useState("");

  // Local setup status — same state machine as the wizard:
  //   idle → verifying → done | verify-failed → installing → done | install-failed
  const [localStatus, setLocalStatus] = useState("idle");
  const [localError, setLocalError] = useState(null);
  // When the detect-only check fails with a structured remedy
  // (Ollama daemon down / model not pulled), this holds the copyable
  // command so the UI can show a Copy button instead of install prose.
  const [localFixCmd, setLocalFixCmd] = useState(null);
  const [localCmdCopied, setLocalCmdCopied] = useState(false);
  const [localChoice, setLocalChoice] = useState("auto"); // auto | skip
  // Local backend: "mlx" (primary, bundled, no external daemon) or
  // "ollama" (opt-in, only if the user already runs it). Persisted as
  // `local_backend`; resolved by engine/llm._build_local_spec.
  const [localBackend, setLocalBackend] = useState("mlx");
  // Ollama-path model id (mirrors `local_model`). MLX uses the pinned
  // DEFAULT_MLX_MODEL — one shipping model, no dropdown.
  const [localModel, setLocalModel] = useState(DEFAULT_LOCAL_MODEL);
  // MLX model status + download/delete driver (shared with the Wizard).
  const mlx = useLocalModel();

  // Active settings tab. Three tabs: General (name / sentiment /
  // cache / updates), Local (Set Up vs Skip + install + model), and
  // Private Cloud (TEE setup). Independent of the section state above
  // — switching tabs preserves dirty fields.
  const [tab, setTab] = useState("general");

  // TEE (Private Cloud) — radio choice matches wizard.
  // teeChoice : setup | skip
  // teeProvider: tinfoil — Tinfoil is the only attested TEE backend.
  //
  // Per-provider state cluster — schema:
  //   newKey       : string — typed in the password input this session
  //   pendingClear : bool   — Clear was clicked, no replacement typed
  //   model        : modelId — picked in the model selector
  //   verifyStatus : "idle" | "verifying" | "verified" | "failed"
  //                  ("verified" = key authenticated this session; a stored
  //                  key still hydrates teeDone via the `_key_set` fields
  //                  on `loaded`)
  //   verifyError  : null | string  — last error from Connect & verify
  //
  // Verify here answers "does this key authenticate?" only. Trust-chain
  // attestation lives on the upper-right Attest button and the app's
  // startup pass — not on Settings save.
  const [teeChoice, setTeeChoice] = useState("setup");
  const [teeProvider, setTeeProvider] = useState(DEFAULT_TEE_PROVIDER);
  const [providers, setProviders] = useState(() => ({
    tinfoil: { newKey: "", pendingClear: false, model: defaultModelFor("tinfoil"),
               verifyStatus: "idle", verifyError: null },
  }));
  const patchProvider = (id, partial) =>
    setProviders((prev) => ({ ...prev, [id]: { ...prev[id], ...partial } }));

  // Per-stage model + reasoning rows. Five visible stages (extract,
  // entities, patterns, insights, actions); metadata is coupled to
  // extract in the pipeline (see llm._STAGE_MODEL_MAP) so it isn't
  // surfaced as its own row. Edited directly per row.
  //
  // Schema per stage: { model: <id>, reasoning: <bool> }.
  const [stageModels, setStageModels] = useState(() => defaultStageModels());
  const patchStage = (stageId, partial) =>
    setStageModels((prev) => ({
      ...prev,
      [stageId]: { ...prev[stageId], ...partial },
    }));

  // "Reset to defaults": snap every model picker — the per-stage
  // pipeline rows AND the chatbot row — back to the ship defaults
  // (defaultStageModels() + DEFAULT_CHATBOT), after a confirm dialog.
  // Lets users recover the ship-default routing if they've been
  // experimenting. Scope is the model selectors only — sentiment,
  // name, API keys, and local install choice are NOT touched (they're
  // orthogonal state slices, and dropping them would surprise a user
  // clicking "reset models").
  //
  // Uses Tauri's plugin-dialog `confirm` (awaitable Promise<boolean>)
  // instead of window.confirm — the latter doesn't reliably block
  // inside Tauri's webview.
  async function resetModelsToDefaults() {
    const ok = await askConfirm(
      "Reset every model + reasoning toggle (per-stage pipeline rows " +
        "and the chatbot row) to the ship defaults?\n\nYour sentiment, " +
        "name, API keys, and local install choice stay the same.",
      { title: "Reset models to defaults?", kind: "warning" },
    );
    if (!ok) return;
    setStageModels(defaultStageModels());
    setChatbot({ ...DEFAULT_CHATBOT });
  }

  // chatbot surface. One model + reasoning pair used at answer-
  // composition time. Persists to top-level `chatbot` in config.json;
  // Python-side reader is `chatbot.resolve_chatbot_from_config`.
  const [chatbot, setChatbot] = useState(() => ({ ...DEFAULT_CHATBOT }));
  const patchChatbot = (partial) =>
    setChatbot((prev) => ({ ...prev, ...partial }));

  // Sentiment-bias dropdown — controls insight/action tone. Stored
  // as `sentiment_bias` in config.json; runner.py reads it as the
  // default for --sentiment. Patterns are descriptive and are NOT
  // sentiment-tunable.
  const [sentiment, setSentiment] = useState(DEFAULT_SENTIMENT);

  // Topic taxonomy editable rows. Persists to `app_config.categories`
  // (string list); the pipeline reads it via
  // `content_extractor._topics_for_run`. Default-seeded on first open
  // when config carries no `categories` field (or it's empty / wrong-
  // shape) — matches the pipeline's fallback so seed-vs-default reads
  // identically end-to-end.
  const [categories, setCategories] = useState(DEFAULT_CATEGORIES);

  // LLM prompt-hash cache toggle. Stored as `llm_cache_enabled` in
  // config.json (default true). When false, runner.py sets the
  // BASEVAULT_LLM_CACHE_BYPASS=1 env var for the run, which makes
  // llm_cache.lookup/store no-op so every call hits the provider.
  // The cache stats + wipe button below operate on the on-disk cache
  // dir (~/.basevault/cache/) regardless of the toggle's state — a
  // user can wipe even while the cache is disabled.
  const [cacheEnabled, setCacheEnabled] = useState(true);
  // Dev tab: per-stage full prompt + response logging toggles. Shape
  // mirrors the persisted config:
  //   { extract: { input: bool, output: bool }, ... }
  // All default OFF; turning ON makes the runner stamp `full_prompt`
  // and/or `full_response` onto each per-call stat record for that
  // stage. Disk impact warning surfaced in the dev-tab UI.
  const [devFullPromptLogging, setDevFullPromptLogging] = useState(() => {
    const out = {};
    for (const { id } of VISIBLE_STAGES) {
      out[id] = { input: false, output: false };
    }
    return out;
  });
  // Dev tab: timing-trace toggle. When ON, the runner + Rust shell +
  // frontend emit `[LAUNCH_TRACE] <step> t=… wall=…` markers across
  // the launch chain into ~/.basevault/logs/app/app.log. Off by default;
  // flipping this is the only knob — no env var override.
  const [devTracing, setDevTracing] = useState(false);
  // Dev tab: opt into the release-candidate update channel. OFF →
  // updater checks the stable manifest; ON → the rc manifest. Read
  // fresh by Rust on every update check, so no app reload is needed.
  const [includeRC, setIncludeRC] = useState(false);
  // Dev tab: Tinfoil HTTP wire-capture. When ON, the Python pipeline
  // attaches httpx event hooks on the TinfoilAI singleton and writes
  // every chat-completion request + response (full bodies, headers
  // incl. `tinfoil-enclave`, TLS pin) to a per-pid JSONL alongside
  // llm-calls.jsonl. Off by default; same disk-impact + sensitivity
  // tier as full prompt + response logging.
  const [devWireCapture, setDevWireCapture] = useState(false);
  const [cacheStats, setCacheStats] = useState({ entries: 0, bytes: 0 });
  const [wiping, setWiping] = useState(false);

  // Save state
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  // Auto-update state. Status machine:
  //   idle        — nothing in flight, no recent result
  //   checking    — check_update in flight
  //   up-to-date  — last check confirmed running version is latest
  //   available   — last check found a newer version (latestVersion populated)
  //   downloading — download_and_install_update in flight, downloading bundle
  //   installing  — bundle download finished, install (replace .app) in flight
  //   restarting  — install done, app.restart() about to fire
  //   failed      — most recent action errored; updateError populated
  // updateLastAction lets Retry know whether to re-check or re-install.
  const [currentVersion, setCurrentVersion] = useState("");
  const [updateStatus, setUpdateStatus] = useState("idle");
  const [latestVersion, setLatestVersion] = useState("");
  const [updateBody, setUpdateBody] = useState("");
  const [updateError, setUpdateError] = useState(null);
  const [updateProgress, setUpdateProgress] = useState({ downloaded: 0, total: 0 });
  const [updateLastAction, setUpdateLastAction] = useState(null); // "check" | "install"

  // Escape closes Settings (modal convention).
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose?.();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Fetch the running app version once, and subscribe to update-progress
  // events. The Rust side emits these from download_and_install_update so
  // the UI can show bytes/total + phase transitions without polling.
  useEffect(() => {
    // ci-allow:listen-guard — handler writes absolute progress values
    // (downloaded/total/phase), not appends; a StrictMode
    // double-subscribe re-sets the same idempotent state, with no
    // doubled-append risk like a streaming listener.
    invoke("app_version")
      .then((v) => setCurrentVersion(String(v || "")))
      .catch(() => {});
    let unlisten;
    listen("update-progress", (event) => {
      const p = (event && event.payload) || {};
      if (p.phase === "download") {
        setUpdateStatus("downloading");
        setUpdateProgress({
          downloaded: Number(p.downloaded) || 0,
          total: Number(p.total) || 0,
        });
      } else if (p.phase === "downloaded" || p.phase === "installing") {
        setUpdateStatus("installing");
      } else if (p.phase === "restarting") {
        setUpdateStatus("restarting");
      }
    }).then((u) => {
      unlisten = u;
    });
    return () => unlisten?.();
  }, []);

  useEffect(() => {
    let s_loaded = null;
    let cfg_loaded = null;
    function applyHydration() {
      if (!s_loaded || !cfg_loaded) return;
      // Hydrate TEE radio: Tinfoil is the only attested provider, so a
      // usable key → set up; otherwise skip. A build's bundled key
      // counts too — it makes Private Cloud work with no user key, so a
      // keyed build defaults to "set up" rather than wrongly showing the
      // section as opted-out.
      const hasTinfoilKey = !!s_loaded.tinfoil_key_set;
      const hasBundledKey = !!s_loaded.bundled_key_set;
      if (hasTinfoilKey || hasBundledKey) {
        setTeeChoice("setup");
      } else {
        setTeeChoice("skip");
      }
      const cfgProvider = (cfg_loaded.tee_provider || "").trim();
      const cfgModel = (cfg_loaded.tee_model || "").trim();
      // Any non-tinfoil `tee_provider` in the loaded config clamps to
      // Tinfoil — it's the only attested TEE backend in the production
      // binary, so a stale value from a prior layout falls through here.
      const chosenProvider =
        cfgProvider && TEE_PROVIDERS[cfgProvider]
          ? cfgProvider
          : DEFAULT_TEE_PROVIDER;
      setTeeProvider(chosenProvider);
      if (cfgModel && isKnownCombo(chosenProvider, cfgModel)) {
        patchProvider(chosenProvider, { model: cfgModel });
      }

      // Hydrate per-stage rows. Resolution mirrors
      // llm.resolve_stage_models_from_config:
      //   1. cfg.stage_models present → use it (sanitized).
      //   2. cfg.tee_model is a single model id → derive via
      //      stageMapFromPreset (a retired preset id → ship default).
      //   3. Default ship map.
      const cfgStageModels = cfg_loaded.stage_models;
      if (cfgStageModels && typeof cfgStageModels === "object") {
        const sanitized = {};
        for (const { id } of VISIBLE_STAGES) {
          const entry = cfgStageModels[id];
          if (entry && typeof entry === "object" && entry.model) {
            sanitized[id] = {
              model: String(entry.model),
              reasoning: !!entry.reasoning,
            };
          }
        }
        if (Object.keys(sanitized).length > 0) {
          setStageModels((prev) => ({ ...prev, ...sanitized }));
        } else if (cfgModel) {
          setStageModels(stageMapFromPreset(cfgModel));
        }
      } else if (cfgModel) {
        setStageModels(stageMapFromPreset(cfgModel));
      }

      // Hydrate chatbot row. Mirrors `chatbot.resolve_chatbot_from_config`:
      // valid {model, reasoning} → use it; otherwise stay on the ship
      // default (reasoning-OFF). Absent / malformed both fall back silently.
      const cfgChatbot = cfg_loaded.chatbot;
      if (cfgChatbot && typeof cfgChatbot === "object" && cfgChatbot.model) {
        setChatbot({
          model: String(cfgChatbot.model),
          reasoning: !!cfgChatbot.reasoning,
        });
      }
    }

    invoke("get_settings")
      .then((s) => {
        s_loaded = s;
        setLoaded(s);
        applyHydration();
      })
      .catch((e) => setError(e?.message || String(e)));
    invoke("get_config")
      .then((cfg) => {
        const safe = cfg && typeof cfg === "object" ? cfg : {};
        cfg_loaded = safe;
        setConfig(safe);
        setName(safe.subject || "");
        // Local: only hydrate the radio (skip vs auto/manual). Don't
        // hydrate localStatus → "done" on reopen — that flag is also
        // reserved for "user clicked Verify this session." The green
        // chip on reopen reads localPreviouslyVerified from config.
        if (safe.local_setup_mode === "skipped") {
          setLocalChoice("skip");
        }
        // Local backend: MLX is the primary default; only "ollama"
        // flips it. Mirrors _build_local_spec's default.
        if (safe.local_backend === "ollama") {
          setLocalBackend("ollama");
        }
        // Ollama-path model id: hydrate from config; stick with the
        // default when absent. _build_local_spec defaults the same way.
        const lm = (safe.local_model || "").trim();
        if (lm) setLocalModel(lm);
        // Hydrate sentiment dropdown from config. Unknown values
        // fall back to the default — keeps a typo or future-renamed
        // value from leaving the dropdown blank.
        const sentVal = (safe.sentiment_bias || "").trim();
        if (SENTIMENT_OPTIONS.some((o) => o.value === sentVal)) {
          setSentiment(sentVal);
        } else {
          setSentiment(DEFAULT_SENTIMENT);
        }
        // Hydrate categories. Pipeline-side `_topics_for_run` falls
        // back to defaults on absent / empty / wrong-shape; mirror
        // that so a fresh install and a config-with-empty-list show
        // the same seed list to the user.
        const rawCats = safe.categories;
        if (Array.isArray(rawCats)) {
          const cleaned = rawCats
            .filter((c) => typeof c === "string")
            .map((c) => c.trim())
            .filter((c) => c.length > 0);
          setCategories(cleaned.length > 0 ? cleaned : DEFAULT_CATEGORIES);
        } else {
          setCategories(DEFAULT_CATEGORIES);
        }
        // Hydrate cache toggle. Treat undefined as enabled — fresh
        // installs and pre-cache configs both ship with caching ON.
        // Only an explicit `false` flips it off.
        setCacheEnabled(safe.llm_cache_enabled !== false);
        // Hydrate the dev-tab full-prompt-logging map. Each stage gets
        // an {input, output} cell defaulting to false; explicit `true`
        // in config flips the corresponding toggle on.
        const dfp = safe.dev_full_prompt_logging || {};
        const hydrated = {};
        for (const { id } of VISIBLE_STAGES) {
          const entry = (dfp && typeof dfp === "object") ? dfp[id] : null;
          hydrated[id] = {
            input: !!(entry && entry.input === true),
            output: !!(entry && entry.output === true),
          };
        }
        setDevFullPromptLogging(hydrated);
        // Dev tracing toggle. Default OFF; only an explicit `true`
        // turns it on.
        setDevTracing(safe.dev_tracing === true);
        setDevWireCapture(safe.dev_wire_capture === true);
        setIncludeRC(safe.include_release_candidates === true);
        applyHydration();
      })
      .catch(() => {
        cfg_loaded = {};
        setConfig({});
        applyHydration();
      });
  }, []);

  // Fetch cache disk stats on mount so the section can render
  // "X entries (Y MB)" without a click. Refreshed after a wipe so
  // the post-wipe display goes to (0, 0).
  function refreshCacheStats() {
    invoke("get_llm_cache_stats")
      .then((s) => setCacheStats(s || { entries: 0, bytes: 0 }))
      .catch(() => {/* best-effort; section just shows the last known stats */});
  }
  useEffect(() => {
    refreshCacheStats();
    mlx.refresh();
    // Mount-only: mlx.refresh is useCallback-stable; depending on the
    // whole `mlx` object would re-run every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function doWipeCache() {
    setWiping(true);
    try {
      const result = await invoke("wipe_llm_cache");
      setCacheStats({ entries: 0, bytes: 0 });
      // Surface a one-line confirmation; the dialog already showed
      // counts so we don't need a separate toast component.
      console.log(
        `wiped ${result?.entries ?? 0} cache entries (` +
        `${formatBytes(result?.bytes ?? 0)})`,
      );
    } catch (e) {
      setError(`Wipe failed: ${e?.message || e}`);
    } finally {
      setWiping(false);
    }
  }

  async function askWipeCache() {
    // Re-fetch right before the dialog so the entry/byte counts are
    // current (the user might have run a pipeline between the modal
    // open and the click).
    refreshCacheStats();
    const { entries, bytes } = cacheStats;
    if (entries === 0) {
      // Nothing to wipe; silent no-op rather than confirming an empty action.
      return;
    }
    const message =
      `This will delete ${entries} cache entr${entries === 1 ? "y" : "ies"} ` +
      `(${formatBytes(bytes)}). Cache misses on the next run will fire fresh ` +
      `LLM calls — costs and wall-clock time return until the cache repopulates.`;
    // Tauri plugin-dialog `confirm`: returns Promise<boolean>, blocks
    // reliably inside the webview. Same pattern as resetToDefaults
    // below — keeps every Settings-side destructive confirmation on
    // one mechanism instead of half-using App's React modal via a
    // threaded prop. window.confirm doesn't reliably block in Tauri's
    // webview (the reset-to-defaults code documents this same bug).
    const ok = await askConfirm(message, {
      title: "Wipe LLM prompt cache?",
      kind: "warning",
    });
    if (ok) await doWipeCache();
  }

  async function openChatsDir() {
    try {
      await invoke("reveal_chats_dir");
    } catch (e) {
      setError(`Open Chats failed: ${e?.message || e}`);
    }
  }

  // ── Dirty flags ──────────────────────────────────────────────────────────
  const nameDirty = config && name.trim() !== (config.subject || "");

  // Sentiment dirty when the dropdown differs from what's persisted.
  // Treat absent / unknown stored values as "neutral" for comparison.
  const originalSentiment = (() => {
    if (!config) return DEFAULT_SENTIMENT;
    const stored = (config.sentiment_bias || "").trim();
    return SENTIMENT_OPTIONS.some((o) => o.value === stored)
      ? stored
      : DEFAULT_SENTIMENT;
  })();
  const sentimentDirty = sentiment !== originalSentiment;

  // Cache toggle dirty when the checkbox differs from what's persisted.
  // Treat absent as "enabled" (default), matching hydration above.
  const originalCacheEnabled = config ? (config.llm_cache_enabled !== false) : true;
  const cacheEnabledDirty = cacheEnabled !== originalCacheEnabled;

  // Dev tracing dirty when the checkbox differs from what's persisted.
  // Treat absent as false (default OFF).
  const originalDevTracing = !!(config && config.dev_tracing === true);
  const devTracingDirty = devTracing !== originalDevTracing;

  // Wire-capture dirty: same shape as dev_tracing.
  const originalDevWireCapture =
    !!(config && config.dev_wire_capture === true);
  const devWireCaptureDirty = devWireCapture !== originalDevWireCapture;

  // RC-channel opt-in dirty. Treat absent as false (default OFF).
  const originalIncludeRC = !!(config && config.include_release_candidates === true);
  const includeRCDirty = includeRC !== originalIncludeRC;

  // Dev full-prompt-logging dirty: any stage's input or output toggle
  // differs from what's persisted. Treat absent as false (default).
  const devFullPromptDirty = useMemo(() => {
    const persisted = config?.dev_full_prompt_logging || {};
    for (const { id } of VISIBLE_STAGES) {
      const cur = devFullPromptLogging[id] || { input: false, output: false };
      const orig = (persisted && typeof persisted === "object") ? persisted[id] : null;
      const origIn = !!(orig && orig.input === true);
      const origOut = !!(orig && orig.output === true);
      if (cur.input !== origIn || cur.output !== origOut) return true;
    }
    return false;
  }, [config, devFullPromptLogging]);

  // Categories dirty when the rows differ from what's persisted. Treat
  // absent / non-array / empty as the default seed (mirrors hydration
  // and pipeline fallback) so an unedited fresh install never appears
  // dirty. Per-row validation errors (empty, bad shape, dupes) gate
  // the save below.
  const originalCategories = (() => {
    if (!config) return DEFAULT_CATEGORIES;
    const raw = config.categories;
    if (!Array.isArray(raw)) return DEFAULT_CATEGORIES;
    const cleaned = raw
      .filter((c) => typeof c === "string")
      .map((c) => c.trim())
      .filter((c) => c.length > 0);
    return cleaned.length > 0 ? cleaned : DEFAULT_CATEGORIES;
  })();
  const categoriesDirty = !categoriesEqual(categories, originalCategories);
  const categoriesValid = categories.every((c, i) =>
    categoryError(c, categories.filter((_, j) => j !== i)) === null
  ) && categories.length > 0;

  // Tinfoil key dirty: typed a new key OR explicitly hit Clear.
  // Skip→clear is no longer automatic — the key persists across teeChoice
  // toggles so flipping back to Setup doesn't require re-entering it.
  // Use the explicit Clear button to remove a stored key.
  // keyDirty drives the auto-save debounce. Typing a new key counts;
  // pendingClear ALONE (no replacement typed) deliberately doesn't —
  // clearing a stored key is destructive, so we want the in-modal
  // Undo button to stay reachable for the entire session, not for
  // just the 500ms debounce window. The flush-on-unmount path uses
  // keyDirtyOnClose so the clear DOES commit if the user dismisses
  // the modal without undoing.
  const keyDirty = !!providers.tinfoil.newKey.trim();
  const keyDirtyOnClose = keyDirty || providers.tinfoil.pendingClear;

  // Local choice is persisted as local_setup_mode in config.json. Dirty
  // if the radio differs from what's on disk (or is nothing on disk and
  // the user hasn't touched it — treat as clean).
  const originalLocalMode =
    (config && config.local_setup_mode) || null;
  // Ready this session: MLX → model downloaded; Ollama → detect-only
  // verify passed. Mirrors the Wizard so the two surfaces agree that a
  // downloaded MLX model needs no separate verify — the download IS the
  // setup. Without this, downloading MLX left the section stuck "pending"
  // and persisted the stale "skipped", keeping the Local mode disabled.
  const localReady =
    localChoice !== "skip" &&
    localBackendReady({
      backend: localBackend,
      mlxDownloaded: mlx.stat?.downloaded,
      ollamaVerified: localStatus === "done",
    });
  // The radio's "resulting" mode: skip → "skipped"; otherwise "verified"
  // once the backend is ready this session. If the user touched the radio
  // but isn't ready yet, carry whatever was on disk rather than downgrading.
  const resultingLocalMode =
    localChoice === "skip"
      ? "skipped"
      : localReady
      ? "verified"
      : originalLocalMode;
  const localDirty = resultingLocalMode !== originalLocalMode;
  const originalLocalModel = (config && config.local_model) || DEFAULT_LOCAL_MODEL;
  const localModelDirty =
    localChoice !== "skip" &&
    localBackend === "ollama" &&
    localModel.trim() !== originalLocalModel;
  const originalLocalBackend = (config && config.local_backend) || "mlx";
  const localBackendDirty = localBackend !== originalLocalBackend;

  // Active provider's state — every read for "the one currently shown"
  // goes through `cur`. Selectors derived from it (cur.newKey,
  // cur.verifyStatus, ...) read directly; they don't get a parallel
  // local alias unless they're used many times.
  const cur = providers[teeProvider];
  const currentTeeModel = cur.model;
  const originalTeeProvider = (config && config.tee_provider) || null;
  const originalTeeModel = (config && config.tee_model) || null;
  const teeProviderDirty =
    teeChoice === "setup" &&
    (teeProvider !== originalTeeProvider || currentTeeModel !== originalTeeModel);

  // Per-stage rows dirty when ANY (model, reasoning) pair differs from
  // what's on disk — or when stage_models was absent on disk and the
  // current rows differ from what stageMapFromPreset(originalTeeModel)
  // would produce (so a fresh-default user editing a row counts as dirty).
  const originalStageModels = (() => {
    const cfg = config || {};
    if (cfg.stage_models && typeof cfg.stage_models === "object") {
      return cfg.stage_models;
    }
    if (originalTeeModel) {
      return stageMapFromPreset(originalTeeModel);
    }
    return defaultStageModels();
  })();
  const stageModelsDirty = (() => {
    if (!config) return false;
    for (const { id } of VISIBLE_STAGES) {
      const cur = stageModels[id] || {};
      const orig = originalStageModels[id] || {};
      if ((cur.model || "") !== (orig.model || "")) return true;
      if (!!cur.reasoning !== !!orig.reasoning) return true;
    }
    return false;
  })();

  const originalChatbot = (() => {
    const cfg = config || {};
    if (cfg.chatbot && typeof cfg.chatbot === "object" && cfg.chatbot.model) {
      return {
        model: String(cfg.chatbot.model),
        reasoning: !!cfg.chatbot.reasoning,
      };
    }
    return { ...DEFAULT_CHATBOT };
  })();
  const chatbotDirty =
    !!config &&
    ((chatbot.model || "") !== (originalChatbot.model || "") ||
      !!chatbot.reasoning !== !!originalChatbot.reasoning);

  // ── Two sections, one shape ──────────────────────────────────────────
  //
  // Local and TEE share the same conceptual structure:
  //   - Status   : idle | verifying | verified | failed   (in-session)
  //                    localStatus / cur.verifyStatus
  //   - Error    : null | string  (matches the failed status)
  //                    localError / cur.verifyError
  //   - PreviouslyVerified : bool, derived from config.json
  //                          (i.e. "verified by some past session, no
  //                          live click required to keep the section
  //                          usable")
  //   - Done     : bool, drives the green chip on reopen.
  //                Done = (skipped) || (Status === "verified")
  //                       || PreviouslyVerified.
  //   - Reverified : bool, true ONLY on this session's verify click.
  //                  Feeds the Save-enable gate so a re-verify with
  //                  no other config change still allows Save.
  //
  // "Reverified" is intentionally OFF on hydration — clicking verify
  // is the meaningful action; reopening the modal isn't.

  const localPreviouslyVerified = config?.local_setup_mode === "verified";
  // For TEE the "previously verified" signal is "a key for the chosen
  // provider is on file in dotenv, with no in-session intent to clear/
  // replace it." The shape diverges slightly because TEE uniquely has
  // a per-provider input cluster (typed key, pending-clear, model).
  const providerKeyOnFile = !!loaded?.[`${teeProvider}_key_set`];
  const teePreviouslyVerified =
    teeChoice === "setup" &&
    providerKeyOnFile &&
    !cur.pendingClear &&
    !cur.newKey.trim();

  // "Ready this session" plays the role "verified" plays in the other
  // sections: a downloaded MLX model or an Ollama verify done this session.
  const localReverified = localReady;
  const teeReverified = providers.tinfoil.verifyStatus === "verified";

  // App needs at least one of Local or Private Cloud to be usable.
  // Each section is "usable" if reverified this session OR previously
  // verified — re-verifying isn't required to keep the app working.
  const localUsable = localReverified || localPreviouslyVerified;
  // A bundled Tinfoil key makes Private Cloud usable even with no user
  // key on file — it powers inference via the Rust effective-key
  // fallback (user key first, bundled second). Without this, Save and
  // the run-somewhere gate stay disabled on a keyed build until the user
  // needlessly enters a key. Gated on teeChoice==="setup" so an explicit
  // Skip still requires Local, matching teePreviouslyVerified.
  const teeUsable =
    teeReverified ||
    teePreviouslyVerified ||
    (teeChoice === "setup" && !!loaded?.bundled_key_set);
  const atLeastOneUsable = localUsable || teeUsable;

  const verifyHappened = teeReverified || localReverified;
  const canSave =
    !saving &&
    !!loaded &&
    !!config &&
    name.trim().length > 0 &&
    atLeastOneUsable &&
    categoriesValid &&
    (nameDirty || keyDirty || localDirty || localModelDirty || localBackendDirty ||
      teeProviderDirty || stageModelsDirty || chatbotDirty || sentimentDirty ||
      cacheEnabledDirty || devFullPromptDirty || devTracingDirty || devWireCaptureDirty || includeRCDirty || categoriesDirty ||
      verifyHappened);

  // ── Actions ──────────────────────────────────────────────────────────────
  async function save() {
    if (!canSave) return;
    setSaving(true);
    setError(null);
    try {
      // 1. Dotenv: Tinfoil key update. States:
      //   - typed a new key AND verified it → write the new value
      //   - typed but NOT verified → null (don't persist an unchecked
      //     key — a bogus key typed then dismissed must not stick, and
      //     must not clobber a good key already on file)
      //   - pendingClear (and no new typed) → write "" (delete)
      //   - neither              → null (leave the line untouched)
      const tinfoilPs = providers.tinfoil;
      const tinfoilKey =
        tinfoilPs.newKey.trim() && tinfoilPs.verifyStatus === "verified"
          ? tinfoilPs.newKey.trim()
          : tinfoilPs.pendingClear ? "" : null;
      if (tinfoilKey !== null) {
        await invoke("save_settings", { tinfoilKey });
      }

      // 2. Config.json: subject + local_setup_mode + tee_*.
      // Always write subject (even if !nameDirty — costs nothing and
      // keeps config.json self-consistent after any kind of edit).
      // Obsidian fields (obsidian_vault_name / obsidian_vault_path) are
      // owned by the first-run Wizard and the underlying open_vault_for_run
      // command; Settings doesn't touch them anymore (issue #77).
      let fields = { subject: name.trim() };
      if (localDirty) {
        fields.local_setup_mode = resultingLocalMode;
      }
      if (localModelDirty) {
        fields.local_model = localModel.trim();
      }
      if (localBackendDirty) {
        fields.local_backend = localBackend;
      }
      // Only touch tee_provider/tee_model/stage_models when the user is
      // actively in Setup. Skip leaves them alone so re-enabling later
      // hydrates back to the same provider+model+rows.
      if (teeChoice === "setup") {
        fields.tee_provider = teeProvider;
        fields.tee_model = currentTeeModel;
        // Write the per-stage map authoritatively. Pipeline routing
        // (llm.resolve_stage_models_from_config) prefers stage_models
        // over tee_model when both are present; tee_model is kept as the
        // budget-anchor fallback (_build_tee_spec) + the Tauri
        // attestation display field.
        const sanitizedStageModels = {};
        for (const { id } of VISIBLE_STAGES) {
          const entry = stageModels[id];
          if (!entry || !entry.model) continue;
          sanitizedStageModels[id] = {
            model: entry.model,
            reasoning: !!entry.reasoning,
          };
        }
        fields.stage_models = sanitizedStageModels;
      }
      if (chatbotDirty) {
        // Top-level `chatbot` field — an inference-time concern, not a
        // pipeline stage, so it doesn't live inside `stage_models`.
        // Python-side reader is `chatbot.resolve_chatbot_from_config`.
        fields.chatbot = {
          model: chatbot.model,
          reasoning: !!chatbot.reasoning,
        };
      }
      if (sentimentDirty) {
        // Persist sentiment_bias even when "neutral" so the file makes
        // the choice explicit; absent vs neutral both mean the same
        // pipeline default but writing it through helps debug runs
        // that don't see the sentiment effect.
        fields.sentiment_bias = sentiment;
      }
      if (cacheEnabledDirty) {
        // Persist explicitly. The runner reads this on every run start
        // and translates `false` into BASEVAULT_LLM_CACHE_BYPASS=1.
        fields.llm_cache_enabled = cacheEnabled;
      }
      if (devTracingDirty) {
        // Persist explicitly so the absent → false default stays
        // distinguishable from "user toggled off". Rust reads this on
        // every run-pipeline launch and propagates BASEVAULT_DEV_TRACING
        // to the Python subprocess; the Rust + frontend layers read it
        // directly. Frontend markers route through the `record_dev_trace`
        // invoke, which gates on the live config so a toggle takes
        // effect without an app reload.
        fields.dev_tracing = devTracing;
      }
      if (devWireCaptureDirty) {
        // Same persist pattern as dev_tracing. Rust reads this at
        // pipeline-spawn time and sets BASEVAULT_DEV_WIRE_CAPTURE on
        // the subprocess env. Python's llm.py reads the env var at
        // module init; a toggle takes effect on the next run, not
        // mid-run.
        fields.dev_wire_capture = devWireCapture;
      }
      if (includeRCDirty) {
        // Persist explicitly so absent → false stays distinguishable
        // from an explicit opt-out. Rust reads this on every update
        // check to pick the stable vs rc manifest endpoint.
        fields.include_release_candidates = includeRC;
      }
      if (devFullPromptDirty) {
        // Persist the {stage: {input, output}} map. The pipeline reads
        // it on every LLM call and stamps full_prompt/full_response
        // onto the per-call stat record when the corresponding toggle
        // is on. Default-OFF entries are omitted to keep config.json
        // clean — readers treat absent as false.
        const persisted = {};
        for (const { id } of VISIBLE_STAGES) {
          const cur = devFullPromptLogging[id];
          if (!cur) continue;
          if (cur.input || cur.output) {
            persisted[id] = {
              input: !!cur.input,
              output: !!cur.output,
            };
          }
        }
        fields.dev_full_prompt_logging = persisted;
      }
      // Categories: skip the write when the rows match the seed
      // default — pipeline-side `_topics_for_run` already falls back
      // to the same list when `categories` is absent, so persisting
      // it would just bloat config.json. When the rows changed AWAY
      // from a custom list back to defaults, drop the existing key
      // explicitly so future reads see the absent path. Writing only
      // on real edits keeps "I never touched this" distinguishable
      // from "I edited then reset."
      if (categoriesDirty) {
        const trimmed = categories.map((c) => c.trim()).filter(Boolean);
        // null patch value deletes the key server-side (update_config
        // follows RFC 7386), so a reset-to-default reverts to the
        // pipeline fallback without a whole-object snapshot write.
        fields.categories = categoriesEqual(trimmed, DEFAULT_CATEGORIES)
          ? null
          : trimmed;
      }
      // Narrow per-key patch only — update_config merges these into the
      // persisted config under a lock, so an interleaved runtime write
      // (import_dir, export prefs, …) can't be clobbered by this Save.
      await invoke("update_config", { patch: fields });

      onChanged?.();
      // Save-on-change: don't close the modal. Re-hydrate `loaded` and
      // `config` from disk so the dirty-flag derivation flips back to
      // clean and the auto-save effect doesn't fire again on the same
      // edits. The user dismisses via Esc / click-outside.
      try {
        const [s, c] = await Promise.all([
          invoke("get_settings"),
          invoke("get_config").catch(() => ({})),
        ]);
        setLoaded(s);
        setConfig(c && typeof c === "object" ? c : {});
      } catch {
        // best-effort; the next event will re-hydrate
      }
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      setSaving(false);
    }
  }

  // Auto-save: debounce 500ms after the last edit. Lets a user finish
  // typing (especially in the name field) before we round-trip to
  // disk. A fresh edit during the window resets the wait —
  // equivalent to "stop editing for 500ms before we save."
  //
  // Refs let the unmount cleanup below flush a pending save if the
  // user closes the modal (Esc / click-outside) before the debounce
  // fires. Without the flush, dirty edits made within the last
  // 500ms get silently dropped.
  //
  // canFlushRef is a superset of canSave: it includes pendingClear-
  // only state (the Tinfoil-key Clear-without-replacement). We don't
  // want auto-save to trigger on pendingClear-only (the Undo button
  // needs to stay reachable for the modal session) but if the user
  // closes the modal without undoing, the clear should commit.
  const canSaveRef = useRef(canSave);
  const saveRef = useRef(save);
  const canFlushRef = useRef(false);
  canSaveRef.current = canSave;
  saveRef.current = save;
  canFlushRef.current =
    !!loaded &&
    !!config &&
    name.trim().length > 0 &&
    atLeastOneUsable &&
    categoriesValid &&
    (nameDirty || keyDirtyOnClose || localDirty || localModelDirty || localBackendDirty ||
      teeProviderDirty || stageModelsDirty || chatbotDirty || sentimentDirty ||
      cacheEnabledDirty || devFullPromptDirty || devTracingDirty || devWireCaptureDirty || includeRCDirty || categoriesDirty ||
      verifyHappened);
  useEffect(() => {
    if (!canSave) return;
    const id = setTimeout(() => {
      save().catch((e) => console.error("auto-save:", e));
    }, 500);
    return () => clearTimeout(id);
    // canSave already collapses every dirty flag + verifyHappened.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canSave]);
  // Flush-on-unmount: if the modal unmounts while a save is pending
  // — including a Tinfoil-key Clear that hasn't been Undone — fire
  // the save synchronously so in-progress edits aren't lost. Tauri's
  // invoke completes in the background regardless of the React tree
  // being torn down. Uses canFlushRef (superset of canSave) so a
  // pendingClear-only edit still commits on close.
  useEffect(() => {
    return () => {
      if (canFlushRef.current) {
        saveRef.current().catch((e) => console.error("flush-save:", e));
      }
    };
  }, []);

  async function checkForUpdates() {
    setUpdateLastAction("check");
    setUpdateStatus("checking");
    setUpdateError(null);
    try {
      const result = await invoke("check_update");
      if (result && result.available_version) {
        setLatestVersion(String(result.available_version));
        setUpdateBody(result.body ? String(result.body) : "");
        setUpdateStatus("available");
      } else {
        setLatestVersion("");
        setUpdateBody("");
        setUpdateStatus("up-to-date");
      }
    } catch (e) {
      setUpdateError(e?.message || String(e));
      setUpdateStatus("failed");
    }
  }

  async function installUpdate() {
    setUpdateLastAction("install");
    // Optimistic: flip to "downloading" before the first progress event
    // arrives so the user gets immediate feedback on click. The Rust
    // side will update progress via the update-progress listener.
    setUpdateStatus("downloading");
    setUpdateError(null);
    setUpdateProgress({ downloaded: 0, total: 0 });
    try {
      await invoke("download_and_install_update");
      // On macOS app.restart() doesn't return — if we reach here without
      // a restart, the install completed cleanly but the relaunch was
      // skipped. Tell the user to relaunch manually.
      setUpdateStatus("restarting");
    } catch (e) {
      setUpdateError(e?.message || String(e));
      setUpdateStatus("failed");
    }
  }

  // Detect-only verification of the selected backend. setup_local.py
  // never installs/pulls; on failure it emits a structured error whose
  // `command` (when present) is a copyable remedy — surfaced here
  // instead of any install prose.
  async function runLocalCheck() {
    setLocalStatus("verifying");
    setLocalError(null);
    setLocalFixCmd(null);
    setLocalCmdCopied(false);
    let unlisten;
    try {
      unlisten = await listen("setup-progress", (event) => {
        try {
          const p = JSON.parse(event.payload);
          if (p.status === "error") {
            setLocalError(p.message || "setup error");
            if (p.command) setLocalFixCmd(p.command);
          }
        } catch {
          // non-JSON progress lines are fine to ignore
        }
      });
      await invoke("setup_local", { mode: "verify" });
      setLocalStatus("done");
    } catch (e) {
      setLocalStatus("verify-failed");
      setLocalError((prev) => prev || e?.message || String(e));
    } finally {
      unlisten?.();
    }
  }

  function onLocalChoiceChange(next) {
    setLocalChoice(next);
    setLocalStatus("idle");
    setLocalError(null);
    setLocalFixCmd(null);
  }

  // Persist the backend choice immediately, independent of the gated
  // Save. It's a routing preference, not part of the verify/usable
  // gating — and the Wizard is a separate component that only learns
  // it by re-reading config.json, so a deferred write would let the
  // two surfaces disagree.
  async function persistLocalBackend(next) {
    try {
      await invoke("update_config", { patch: { local_backend: next } });
      // Keep in-memory config in sync so localBackendDirty doesn't
      // show a phantom unsaved change after an already-persisted toggle.
      setConfig((c) => ({ ...(c || {}), local_backend: next }));
    } catch {
      /* best-effort; the gated Save still writes it as a fallback */
    }
  }

  function onLocalBackendChange(next) {
    setLocalBackend(next);
    setLocalStatus("idle");
    setLocalError(null);
    setLocalFixCmd(null);
    persistLocalBackend(next);
  }

  async function copyLocalFixCmd() {
    if (!localFixCmd) return;
    try {
      await navigator.clipboard.writeText(localFixCmd);
      setLocalCmdCopied(true);
      setTimeout(() => setLocalCmdCopied(false), 2000);
    } catch {
      // clipboard can fail under gatekeeper; the command is selectable
    }
  }

  async function askDeleteMlx() {
    const sz = mlx.stat?.size_bytes
      ? ` (${formatBytes(mlx.stat.size_bytes)})`
      : "";
    const ok = await askConfirm(
      `Delete the downloaded local model${sz}? You'll need to download ` +
        `it again to run Local mode.`,
      { title: "Delete local model?", kind: "warning" },
    );
    if (ok) await mlx.remove();
  }

  async function askRemoveAllData() {
    const ok = await askConfirm(
      "Remove ALL BaseVault data? This wipes ~/.basevault/ — downloaded " +
        "models, every run's logs and outputs, the exported vault, the " +
        "LLM cache, and your settings. This cannot be undone.",
      { title: "Remove all data?", kind: "warning" },
    );
    if (!ok) return;
    try {
      await invoke("reset_basevault");
      mlx.refresh();
    } catch (e) {
      setError(`Remove all data failed: ${e?.message || e}`);
    }
  }

  // "Does this key authenticate against Tinfoil?" — minimal probe, no
  // trust-chain work. Hits a router enclave's /v1/models with the bearer
  // (typed key wins over dotenv when present); 200 = verified, 401 =
  // bad key, network errors surfaced as-is. The upper-right Attest
  // button owns the Sigstore + measurement + per-model chain.
  async function verifyTeeKey() {
    const provider = teeProvider;
    const ps = providers[provider];
    const newKey = ps.newKey.trim();
    const onFile = !!loaded?.[`${provider}_key_set`];
    if (!newKey && !onFile) {
      patchProvider(provider, {
        verifyStatus: "failed",
        verifyError: "Enter a key first or pick a different provider.",
      });
      return;
    }
    patchProvider(provider, { verifyStatus: "verifying", verifyError: null });
    try {
      // Typed key wins; empty falls back to the dotenv-stored key on
      // the Rust side. Per-provider command id (Tinfoil today; the
      // registry leaves the door open for attested alternatives).
      await invoke(TEE_PROVIDERS[provider].verifyCommand, { key: newKey || undefined });
      patchProvider(provider, { verifyStatus: "verified", verifyError: null });
    } catch (e) {
      const msg = e?.message || String(e);
      patchProvider(provider, { verifyStatus: "failed", verifyError: msg });
    }
  }

  function onTeeChoiceChange(next) {
    setTeeChoice(next);
    // Don't clear verified status on toggle — a stored key should count
    // as verified across Setup/Skip toggles. Only failed/in-flight
    // states get cleared.
    const s = providers.tinfoil.verifyStatus;
    if (s === "failed" || s === "verifying") {
      patchProvider("tinfoil", { verifyStatus: "idle", verifyError: null });
    } else {
      patchProvider("tinfoil", { verifyError: null });
    }
  }

  // Switch the active sub-provider. Per-provider state is preserved
  // (each provider has its own slot in `providers`). Today Tinfoil is
  // the only attested TEE provider so this is effectively a one-element
  // sub-radio, but the structure stays in place to keep the door open
  // for re-adding attested alternatives without rewiring the section.
  function onTeeProviderChange(next) {
    setTeeProvider(next);
  }


  // ── Per-section done flags (for the green chip + section tint) ───────────
  // "Done" means: the section is in a state that carries through to a
  // working setup. Includes both verified-this-session AND
  // verified-in-a-prior-session (read from config) so reopening
  // Settings shows the chips green without forcing a re-verify.
  const nameDone = !!loaded && name.trim().length > 0;
  const localDone =
    localChoice === "skip" ||
    localReverified ||
    localPreviouslyVerified;
  const teeDone = teeChoice === "skip" || teeUsable;

  // ── Render ───────────────────────────────────────────────────────────────
  if (!loaded || !config) {
    return (
      <div className="modal-backdrop" onClick={onClose}>
        <div className="modal" onClick={(e) => e.stopPropagation()}>
          <p>{error || "Loading…"}</p>
        </div>
      </div>
    );
  }

  // Per-provider Setup-JSX helpers. All read from `cur` / write through
  // patchProvider; no parallel state to keep in sync.
  const providerStoredKeyMasked = loaded[`${teeProvider}_key_masked`];
  const showProviderMasked =
    providerKeyOnFile && !cur.pendingClear && !cur.newKey.trim();
  // Keyed build, no user key of their own: surface that Private Cloud is
  // already working off the bundled key rather than leaving the field
  // looking empty/unconfigured. We deliberately do NOT show the bundled
  // key's value here — it's not the user's key, and faking it into the
  // field would re-introduce the user-vs-bundled confusion. The empty
  // input stays so they can enter their own to override.
  const showBundledKeyNotice =
    teeProvider === "tinfoil" &&
    !!loaded?.bundled_key_set &&
    !providerKeyOnFile &&
    !cur.pendingClear &&
    !cur.newKey.trim();
  const showProviderClearedNotice =
    providerKeyOnFile && cur.pendingClear && !cur.newKey.trim();
  const setProviderPendingClear = (v) =>
    patchProvider(teeProvider, { pendingClear: v });
  const setProviderNewKey = (v) =>
    patchProvider(teeProvider, {
      newKey: v,
      ...(v.trim() ? { pendingClear: false } : {}),
      verifyStatus: "idle",
      verifyError: null,
    });

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal wizard-modal"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Fixed (non-scrolling) header: title + tab strip. The
            scrolling body below holds whichever tab's sections are
            active. Splitting at this seam keeps the tabs always in
            view regardless of how far the user scrolls within a
            tab — sticky-positioning had been brittle and would
            occasionally let a tall section push tabs off-screen. */}
        <div className="settings-fixed-head">
          <h1>Settings</h1>
          <nav className="settings-tabs" role="tablist">
            {[
              { id: "general", label: "General" },
              { id: "local", label: "Local" },
              { id: "tee", label: "Private Cloud" },
              { id: "dev", label: "Development" },
            ].map(({ id, label }) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={tab === id}
                className={`settings-tab ${tab === id ? "active" : ""}`}
                onClick={() => setTab(id)}
              >
                {label}
              </button>
            ))}
          </nav>
        </div>

        <div className="settings-scroll-body">
        {tab === "general" && (
        <>
        {/* ── Name ─────────────────────────────────────────────────────── */}
        {/* Status badge tracks the auto-save lifecycle: nameDirty is
            true while the user is typing OR within the 500ms debounce
            before save() fires. After save completes the modal re-
            hydrates from disk, nameDirty flips false, and the badge
            switches to the green "✓ saved" state. */}
        <section
          className={`wizard-section ${nameDone && !nameDirty ? "done" : ""}`}
        >
          <header>
            <h2>Your name</h2>
            <SectionStatus
              done={nameDone && !nameDirty}
              label={nameDirty ? "pending" : "saved"}
              pendingLabel="pending"
            />
          </header>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <p className="field-hint">
            Used so the pipeline writes actions and insights for you.
          </p>
        </section>
        </>
        )}

        {tab === "local" && (
        <>
        {/* ── Card 1: Enable / Disable ─────────────────────────────────── */}
        {/* Green only when the section is actually configured (Enabled
            + a verified/downloaded backend). Skip → grey. */}
        <section
          className={`wizard-section ${
            localChoice === "skip"
              ? "skipped"
              : localDone
              ? "done"
              : ""
          }`}
        >
          <header>
            <h2>Local processing</h2>
            <SectionStatus
              done={localChoice !== "skip" && localDone}
              label={
                localReverified || localPreviouslyVerified
                  ? "ready"
                  : ""
              }
              pendingLabel={localChoice === "skip" ? "disabled" : "pending"}
            />
          </header>
          <p className="field-hint">
            Run the pipeline entirely on your machine — nothing leaves
            the device. Disable to hide the Local runtime mode in the
            top-bar; downloaded models are left intact.
          </p>
          <div className="enable-disable-radios">
            <label className={`enable-radio ${localChoice !== "skip" ? "active" : ""}`}>
              <input
                type="radio"
                name="settings-local-enable"
                checked={localChoice !== "skip"}
                onChange={() => onLocalChoiceChange("auto")}
              />
              <strong>Enable</strong>
            </label>
            <label className={`enable-radio ${localChoice === "skip" ? "active" : ""}`}>
              <input
                type="radio"
                name="settings-local-enable"
                checked={localChoice === "skip"}
                onChange={() => onLocalChoiceChange("skip")}
              />
              <strong>Disable</strong>
            </label>
          </div>
        </section>

        {localChoice !== "skip" && (
        <>
        {/* ── Card 2: Backend ──────────────────────────────────────────── */}
        <section className="wizard-section">
          <header>
            <h2>Inference engine</h2>
          </header>
          <p className="field-hint">
            MLX is bundled with the app — no external install, no daemon.
            Ollama is opt-in for users who already run it; BaseVault
            won't install or configure it for you.
          </p>
          <div className="obsidian-options">
            <label className={`obs-option ${localBackend === "mlx" ? "active" : ""}`}>
              <input
                type="radio"
                name="settings-local-backend"
                checked={localBackend === "mlx"}
                onChange={() => onLocalBackendChange("mlx")}
              />
              <div>
                <strong>MLX</strong> — bundled, recommended
              </div>
            </label>
            <label className={`obs-option ${localBackend === "ollama" ? "active" : ""}`}>
              <input
                type="radio"
                name="settings-local-backend"
                checked={localBackend === "ollama"}
                onChange={() => onLocalBackendChange("ollama")}
              />
              <div>
                <strong>Ollama</strong> — only if you already run it
              </div>
            </label>
          </div>
        </section>

        {/* ── Card 3: Model (backend-specific) ──────────────────────────── */}
        {localBackend === "mlx" ? (
        <section className="wizard-section">
          <header>
            <h2>Local model</h2>
          </header>
          <p className="field-hint">
            {DEFAULT_MLX_MODEL} — the Qwen3.5 9B model, downloaded once
            to ~/.basevault/models/. Multi-GB; downloads in a few
            minutes. Nothing is downloaded until you click below.
          </p>
          {mlx.stat?.downloaded ? (
            <>
              <p className="verify-status verify-ok">
                ✓ Downloaded{mlx.stat.size_bytes
                  ? ` (${formatBytes(mlx.stat.size_bytes)})`
                  : ""}
              </p>
              <div className="row gap">
                <button
                  type="button"
                  onClick={mlx.download}
                  disabled={mlx.busy}
                >
                  {mlx.busy ? "Downloading…" : "Re-download"}
                </button>
                <button
                  type="button"
                  onClick={askDeleteMlx}
                  disabled={mlx.busy}
                >
                  Delete model
                </button>
              </div>
            </>
          ) : (
            <div className="row gap">
              <button
                type="button"
                onClick={mlx.download}
                disabled={mlx.busy}
              >
                {mlx.busy ? "Downloading…" : "Download model"}
              </button>
            </div>
          )}
          {mlx.busy && (
            <p className="verify-status">
              {mlx.pct != null ? `${mlx.pct}% — ` : ""}{mlx.msg || "Working…"}
            </p>
          )}
          {mlx.error && !mlx.busy && (
            <p className="verify-status verify-failed">
              ✗ {String(mlx.error).split("\n").slice(-3).join(" · ")}
            </p>
          )}
        </section>
        ) : (
        <section className="wizard-section">
          <header>
            <h2>Ollama status</h2>
          </header>
          <p className="field-hint">
            BaseVault is only verified with <code>{localModel}</code> on
            the Ollama path — other models may misbehave. We check
            whether the daemon is up and the model is present; we don't
            install or pull anything.
          </p>
          <div className="row gap">
            <button
              type="button"
              onClick={runLocalCheck}
              disabled={localStatus === "verifying"}
            >
              {localStatus === "verifying"
                ? "Checking…"
                : localStatus === "done"
                ? "Verified ✓ — re-check"
                : "Check Ollama"}
            </button>
          </div>
          {localStatus === "verify-failed" && localError && (
            <>
              <p className="verify-status verify-failed">
                ✗ {String(localError).split("\n").slice(-3).join(" · ")}
              </p>
              {localFixCmd && (
                <div className="row gap">
                  <pre className="install-script">
                    <code>{localFixCmd}</code>
                  </pre>
                  <button type="button" onClick={copyLocalFixCmd}>
                    {localCmdCopied ? "Copied ✓" : "Copy"}
                  </button>
                </div>
              )}
            </>
          )}
          {(localReverified || localPreviouslyVerified) && (
            <p className="verify-status verify-ok">
              ✓ Ollama daemon up, {localModel} present
            </p>
          )}
        </section>
        )}

        {/* ── Card 4: Danger zone ──────────────────────────────────────── */}
        <section className="wizard-section">
          <header>
            <h2>Remove all data</h2>
          </header>
          <p className="field-hint">
            macOS doesn't run uninstall hooks, so dragging the app to
            Trash leaves data behind. This wipes everything under
            ~/.basevault/ — downloaded models, all run logs and outputs,
            the exported vault, the LLM cache, and your settings.
          </p>
          <div className="row gap">
            <button type="button" onClick={askRemoveAllData}>
              Remove all BaseVault data…
            </button>
          </div>
        </section>
        </>
        )}
        </>
        )}

        {tab === "tee" && (
        <>
        {/* ── Card 1: Enable / Disable ─────────────────────────────────── */}
        {/* Same green-vs-grey rule as Local: green only when actually
            configured, grey when disabled. */}
        <section
          className={`wizard-section ${
            teeChoice === "skip"
              ? "skipped"
              : teeDone
              ? "done"
              : ""
          }`}
        >
          <header>
            <h2>Private Cloud mode (TEE)</h2>
            <SectionStatus
              done={teeChoice !== "skip" && teeDone}
              label={teeDone ? "ready" : ""}
              pendingLabel={teeChoice === "skip" ? "disabled" : "pending"}
            />
          </header>
          <p className="field-hint">
            Encrypted cloud processing inside a Trusted Execution
            Environment (Tinfoil). Raw files never leave your machine.
          </p>
          <div className="enable-disable-radios">
            <label className={`enable-radio ${teeChoice === "setup" ? "active" : ""}`}>
              <input
                type="radio"
                name="settings-tee-enable"
                checked={teeChoice === "setup"}
                onChange={() => onTeeChoiceChange("setup")}
              />
              <strong>Enable</strong>
            </label>
            <label className={`enable-radio ${teeChoice === "skip" ? "active" : ""}`}>
              <input
                type="radio"
                name="settings-tee-enable"
                checked={teeChoice === "skip"}
                onChange={() => onTeeChoiceChange("skip")}
              />
              <strong>Disable</strong>
            </label>
          </div>
        </section>

        {teeChoice === "setup" && (
        <>
        {/* ── Card 2: Provider key + verify ────────────────────────────── */}
        <section className="wizard-section">
          <header>
            <h2>Provider key</h2>
          </header>
          <div className="tee-provider-config">
                    <div className="tee-provider-tagline">
                      {TEE_PROVIDERS[teeProvider].tagline}{" "}
                      <a
                        className="tee-signup-link"
                        href={TEE_PROVIDERS[teeProvider].signupUrl}
                        onClick={externalLinkClick(
                          TEE_PROVIDERS[teeProvider].signupUrl,
                        )}
                      >
                        Get a key at{" "}
                        {TEE_PROVIDERS[teeProvider].signupUrl.replace(
                          /^https?:\/\//,
                          "",
                        )}{" "}
                        ↗
                      </a>
                    </div>

                    {showBundledKeyNotice && (
                      <p className="field-hint">
                        ✓ Private Cloud is active using BaseVault's built-in
                        key. Enter your own key below to use it instead.
                      </p>
                    )}

                    {/* Stored-key display: masked value + Clear button.
                        Only rendered when a key is on file, not pending
                        clear, and no new key has been typed. */}
                    {showProviderMasked && (
                      <div className="row gap">
                        <code className="masked-key">{providerStoredKeyMasked}</code>
                        <button
                          onClick={() => setProviderPendingClear(true)}
                          disabled={saving}
                        >
                          Clear
                        </button>
                      </div>
                    )}
                    {/* Password input. Hidden when key is masked (intact
                        on file). When pending-clear with no new typed
                        key, an Undo button sits next to the empty input
                        so the user can revert without leaving the row.
                        Save with empty input → key is deleted; type
                        anything → user is replacing, Undo auto-hides. */}
                    {!showProviderMasked && (
                      <div className="row gap">
                        <input
                          type="password"
                          value={cur.newKey}
                          onChange={(e) => setProviderNewKey(e.target.value)}
                          placeholder={TEE_PROVIDERS[teeProvider].keyPlaceholder}
                          autoComplete="off"
                          style={{ flex: 1 }}
                        />
                        {showProviderClearedNotice && (
                          <button
                            onClick={() => setProviderPendingClear(false)}
                            disabled={saving}
                          >
                            Undo
                          </button>
                        )}
                      </div>
                    )}
                    <div className="row gap">
                      <button
                        type="button"
                        onClick={verifyTeeKey}
                        disabled={
                          cur.verifyStatus === "verifying" ||
                          // Disabled when there's nothing to verify:
                          // no typed key, AND either no key on file or
                          // the on-file key is pending clear (would
                          // verify a key the user just discarded).
                          (cur.newKey.trim().length === 0 &&
                            (!providerKeyOnFile || cur.pendingClear))
                        }
                      >
                        {cur.verifyStatus === "verifying"
                          ? "Checking…"
                          : cur.verifyStatus === "verified"
                          ? "Reverify"
                          : "Connect & verify"}
                      </button>
                      {cur.verifyStatus !== "verifying" &&
                        (cur.verifyStatus === "verified" || teePreviouslyVerified) && (
                          <span className="verified-tick">✓ verified</span>
                        )}
                    </div>
                    {cur.verifyStatus === "failed" && cur.verifyError && (
                      <ErrorWithTrace text={cur.verifyError} />
                    )}
          </div>
        </section>

        {/* ── Card 3: Model selection ──────────────────────────────────── */}
        <section className="wizard-section">
          <header>
            <h2>Model selection</h2>
          </header>
          <div className="tee-provider-config">
                    {/* Per-stage model + reasoning rows. Five visible
                        stages; metadata is coupled to extract in the
                        pipeline so it isn't surfaced. The reasoning
                        toggle is greyed out when the selected model
                        doesn't have a verified reasoning control
                        surface (see REASONING_TOGGLE_MODELS / Python
                        _REASONING_WHITELIST). */}
                    {teeProvider === "tinfoil" && (
                      <>
                        <label
                          className="field-label"
                          style={{ marginTop: 16 }}
                        >
                          Per-stage models
                        </label>
                        <p className="field-hint" style={{ marginBottom: 8 }}>
                          Pick a model + reasoning toggle for each pipeline
                          stage. metadata follows extract.
                        </p>
                        <div className="stage-models-grid">
                          {VISIBLE_STAGES.map((stage) => {
                            const row = stageModels[stage.id] || {};
                            return (
                              <StageModelRow
                                key={stage.id}
                                label={stage.label}
                                hint={stage.hint}
                                model={row.model}
                                reasoning={row.reasoning}
                                modelOptions={optionsForStage(stage.id, "tinfoil")}
                                defaultModelId={
                                  SHIP_DEFAULT_STAGE_MODELS[stage.id]?.model
                                }
                                onPatch={(partial) => patchStage(stage.id, partial)}
                              />
                            );
                          })}
                        </div>
                        {/* Chat model — the interactive call the chatbot
                            surface fires per query, distinct from the
                            per-stage pipeline routing above. Persists to
                            its own top-level `chatbot` config field. */}
                        <label
                          className="field-label"
                          style={{ marginTop: 16 }}
                        >
                          Chat model
                        </label>
                        <p className="field-hint" style={{ marginBottom: 8 }}>
                          The model the chatbot surface calls at query
                          time, separate from the per-stage pipeline
                          routing above. Reasoning is off by default;
                          turning it on is slower but more thorough.
                        </p>
                        <StageModelRow
                          label="Chatbot"
                          hint="composes the streamed answer from your retrieved context"
                          model={chatbot.model}
                          reasoning={chatbot.reasoning}
                          modelOptions={optionsForStage("chatbot", "tinfoil")}
                          defaultModelId={DEFAULT_CHATBOT.model}
                          onPatch={patchChatbot}
                          testIdPrefix="chatbot"
                        />
                        <div style={{ marginTop: 12 }}>
                          <button
                            type="button"
                            className="btn-secondary"
                            onClick={resetModelsToDefaults}
                            disabled={saving}
                            data-testid="reset-models-to-defaults"
                            title="Snap every model + reasoning toggle (per-stage rows and chatbot) back to the ship defaults"
                          >
                            Reset to defaults
                          </button>
                        </div>
                      </>
                    )}
          </div>
        </section>
        </>
        )}
        </>
        )}

        {tab === "general" && (
        <>
        {/* ── Sentiment bias ───────────────────────────────────────────── */}
        <section className="wizard-section">
          <header>
            <h2>Insight + action tone</h2>
          </header>
          <p className="field-hint">
            Controls how insights and actions are framed — emphasis and
            word choice, not factual content. Patterns are descriptive
            and stay neutral regardless. The same evidence base
            produces the same factual claims at every setting.
          </p>
          <select
            className="model-select"
            value={sentiment}
            onChange={(e) => setSentiment(e.target.value)}
          >
            {SENTIMENT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
                {opt.value === DEFAULT_SENTIMENT ? " (default)" : ""}
              </option>
            ))}
          </select>
          <p className="field-hint">
            {SENTIMENT_OPTIONS.find((o) => o.value === sentiment)?.hint}
          </p>
        </section>

        {/* ── Topic taxonomy ───────────────────────────────────────────── */}
        <CategoriesSection
          categories={categories}
          setCategories={setCategories}
        />

        {/* ── Pane sizes ─────────────────────────────────────────────── */}
        <section className="wizard-section">
          <header>
            <h2>Pane sizes</h2>
          </header>
          <p className="field-hint">
            Drag the splitters between the runs / file tree / markdown
            panes to resize. Reset returns to the ship-default layout.
          </p>
          <div>
            <button
              type="button"
              onClick={() => onResetPaneSizes?.()}
              disabled={!onResetPaneSizes}
            >
              Reset to default
            </button>
          </div>
        </section>

        {/* ── LLM prompt-hash cache ───────────────────────────────────── */}
        <section className="wizard-section" data-section="llm-cache">
          <header>
            <h2>LLM prompt cache</h2>
          </header>
          <p className="field-hint">
            Disk cache for LLM responses keyed by prompt + model +
            request params. Re-runs on identical inputs make zero LLM
            calls. Lives at <code>~/.basevault/cache/</code>; deletes are
            recoverable only by re-running the pipeline.
          </p>
          <label
            className="checkbox-row"
            style={{ display: "inline-flex", alignItems: "center", gap: 8 }}
          >
            <input
              type="checkbox"
              checked={cacheEnabled}
              onChange={(e) => setCacheEnabled(e.target.checked)}
              data-testid="cache-enabled-checkbox"
            />
            <span>Enable LLM prompt caching (recommended)</span>
          </label>
          <div
            className="cache-stats"
            data-testid="cache-stats"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 12,
              marginTop: 12,
            }}
          >
            <span className="muted">
              {cacheStats.entries.toLocaleString()} cache entr
              {cacheStats.entries === 1 ? "y" : "ies"} on disk
              {" · "}
              {formatBytes(cacheStats.bytes)}
            </span>
            <button
              type="button"
              onClick={askWipeCache}
              disabled={wiping || cacheStats.entries === 0}
              data-testid="wipe-cache-button"
            >
              {wiping ? "Wiping…" : "Wipe cache"}
            </button>
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 12,
              marginTop: 12,
            }}
          >
            <span className="muted">
              Chat transcripts live at <code>~/.basevault/chats/</code>.
            </span>
            <button
              type="button"
              onClick={openChatsDir}
              data-testid="open-chats-button"
            >
              Open Chats
            </button>
          </div>
        </section>

        {/* ── Updates ─────────────────────────────────────────────────── */}
        <section className="wizard-section">
          <header>
            <h2>Updates</h2>
          </header>
          <p className="field-hint">
            Current version: {currentVersion ? `v${currentVersion}` : "unknown"}
          </p>

          {updateStatus === "up-to-date" && (
            <p className="verify-status verify-ok">✓ Up to date</p>
          )}

          {(updateStatus === "idle" || updateStatus === "up-to-date") && (
            <button
              className="btn-secondary"
              onClick={checkForUpdates}
              aria-label="Check for updates"
            >
              Check for updates
            </button>
          )}

          {updateStatus === "checking" && (
            <p className="field-hint">Checking for updates…</p>
          )}

          {updateStatus === "available" && (
            <>
              <p>
                <strong>Update available: v{latestVersion}</strong>
              </p>
              {updateBody && (
                <p
                  className="field-hint"
                  style={{ whiteSpace: "pre-wrap" }}
                >
                  {updateBody}
                </p>
              )}
              <button
                className="btn-primary"
                onClick={installUpdate}
                aria-label="Update now"
              >
                Update now
              </button>
              <p className="field-hint">
                BaseVault will download, install, and restart automatically.
              </p>
            </>
          )}

          {updateStatus === "downloading" && (
            <>
              <p>
                Downloading{latestVersion ? ` v${latestVersion}` : ""}…{" "}
                {formatProgress(updateProgress)}
              </p>
              <p className="field-hint">
                Don't quit BaseVault until the install finishes.
              </p>
            </>
          )}

          {updateStatus === "installing" && (
            <p>
              Installing{latestVersion ? ` v${latestVersion}` : ""}…
            </p>
          )}

          {updateStatus === "restarting" && (
            <p>Restarting BaseVault…</p>
          )}

          {updateStatus === "failed" && (
            <>
              <p className="error">
                {isSignatureError(updateError)
                  ? `⚠ Signature verification failed — refusing to install. The downloaded bundle may be corrupted or tampered with. (${updateError})`
                  : updateLastAction === "install"
                  ? `Update failed: ${updateError || "unknown error"}`
                  : `Couldn't check for updates: ${updateError || "unknown error"}`}
              </p>
              <button
                className="btn-secondary"
                onClick={
                  updateLastAction === "install"
                    ? installUpdate
                    : checkForUpdates
                }
              >
                {updateLastAction === "install" ? "Try again" : "Retry"}
              </button>
            </>
          )}
        </section>
        </>
        )}

        {tab === "dev" && (
        <>
        {/* ── Development banner ────────────────────────────────────── */}
        <section className="wizard-section dev-warning-section">
          <p className="dev-warning-banner">
            <strong>Development settings.</strong> These are not for
            typical use. Changing these settings may affect the app's
            performance, stability, and disk use.
          </p>
        </section>

        {/* ── Timing traces ────────────────────────────────────────── */}
        <section className="wizard-section">
          <header>
            <h2>Timing traces</h2>
          </header>
          <p className="field-hint">
            When ON, the frontend, Rust shell, and Python pipeline emit{" "}
            <code>[LAUNCH_TRACE] &lt;step&gt; t=… wall=…</code> markers
            across the click → run-row-visible chain into the
            application log
            (<code>~/.basevault/logs/app/app.log</code>). Off by default.
            Use the <code>wall</code> field (unix epoch seconds) to
            merge the three timelines:
            {" "}<code>grep '[LAUNCH_TRACE]' app.log | sort -k4</code>.
          </p>
          <label className="dev-tracing-row">
            <input
              type="checkbox"
              checked={devTracing}
              onChange={() => setDevTracing((v) => !v)}
              aria-label="enable timing traces"
            />
            {" "}<span>Emit timing traces to app.log</span>
          </label>
        </section>

        {/* ── Tinfoil wire-capture ────────────────────────────────────── */}
        <section className="wizard-section">
          <header>
            <h2>Tinfoil wire-capture</h2>
          </header>
          <p className="field-hint">
            When ON, every Tinfoil HTTP chat-completion is logged
            byte-level — full request body (incl. the requested
            model), full response body (incl. the returned model),
            all headers (incl. <code>tinfoil-enclave</code> naming the
            backing enclave that served the call), TLS pin, and
            bundle digest. Authorization header is redacted. Output
            sits alongside the call source:{" "}
            <code>&lt;run-dir&gt;/tinfoil-wire.jsonl</code> for
            pipeline runs,{" "}
            <code>&lt;convo-dir&gt;/tinfoil-wire.jsonl</code> for
            chatbot turns, and{" "}
            <code>~/.basevault/sessions/&lt;session&gt;/tinfoil-wire.jsonl</code>
            {" "}for SDK bootstrap + ad-hoc UI attestation. Off by
            default. Same disk-impact and sensitivity tier as full
            prompt + response logging — disable when not actively
            debugging.
          </p>
          <label className="dev-tracing-row">
            <input
              type="checkbox"
              checked={devWireCapture}
              onChange={() => setDevWireCapture((v) => !v)}
              aria-label="enable tinfoil wire-capture"
            />
            {" "}<span>Capture Tinfoil HTTP requests + responses</span>
          </label>
        </section>

        {/* ── Release-candidate update channel ──────────────────────── */}
        <section className="wizard-section">
          <header>
            <h2>Include release candidates in updates</h2>
          </header>
          <p className="field-hint">
            When ON, the updater offers release-candidate builds (tagged
            for testing) in addition to stable releases. Off by default —
            normal users only ever see stable. A later stable supersedes
            a release candidate automatically.
          </p>
          <label className="dev-tracing-row">
            <input
              type="checkbox"
              checked={includeRC}
              onChange={() => setIncludeRC((v) => !v)}
              aria-label="include release candidates in updates"
            />
            {" "}<span>Offer release-candidate builds</span>
          </label>
        </section>

        {/* ── Full prompt + response logging ───────────────────────── */}
        <section className="wizard-section">
          <header>
            <h2>Full prompt + response logging</h2>
          </header>
          <p className="field-hint">
            When ON, the runner stamps the FULL prompt and/or response
            text onto each per-call record (rendered in the Run Details
            modal under expandable sub-sections). Off by default. Disk
            impact is real — extract on long runs can produce 100MB+
            of logged prompts per run. Disable when not actively
            debugging.
          </p>
          <table className="dev-full-prompt-table">
            <thead>
              <tr>
                <th>stage</th>
                <th className="numeric">log input</th>
                <th className="numeric">log output</th>
              </tr>
            </thead>
            <tbody>
              {VISIBLE_STAGES.map(({ id, label }) => {
                const cur = devFullPromptLogging[id]
                  || { input: false, output: false };
                const flip = (which) => {
                  setDevFullPromptLogging((prev) => ({
                    ...prev,
                    [id]: { ...cur, [which]: !cur[which] },
                  }));
                };
                return (
                  <tr key={id}>
                    <td>{label}</td>
                    <td className="numeric">
                      <input
                        type="checkbox"
                        checked={cur.input}
                        onChange={() => flip("input")}
                        aria-label={`log input for ${label}`}
                      />
                    </td>
                    <td className="numeric">
                      <input
                        type="checkbox"
                        checked={cur.output}
                        onChange={() => flip("output")}
                        aria-label={`log output for ${label}`}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
        </>
        )}

        {error && <p className="error">{error}</p>}
        {!atLeastOneUsable && !saving && (
          <p className="field-hint wizard-gate-hint">
            You need at least one of Local or Private Cloud actually set
            up — a pipeline run has to happen somewhere.
          </p>
        )}

        {/* Save-on-change replaces the explicit Save / Cancel /
            Reset buttons. The modal stays open; users dismiss it via
            Esc or click-outside. A subtle "saving…" indicator below
            keeps the round-trip visible without an action surface. */}
        {saving && (
          <p className="settings-saving-indicator">Saving…</p>
        )}
        </div>
      </div>
    </div>
  );
}

// ── Model-picker row (per-stage + chatbot) ────────────────────────────
//
// One row = label/hint, model <select>, reasoning toggle. Used by the
// per-stage grid and the chatbot row. The reasoning toggle
// greys out when the selected model isn't in REASONING_TOGGLE_MODELS,
// and a model change to an unsupported one clamps reasoning OFF (no
// stale `true` written to disk that the pipeline would silently
// ignore). `onPatch` receives a partial `{model?, reasoning?}` so the
// caller controls which state slice gets the update.
function StageModelRow({
  label,
  hint,
  model,
  reasoning,
  modelOptions,
  defaultModelId,
  onPatch,
  testIdPrefix,
}) {
  const reasoningSupported = REASONING_TOGGLE_MODELS.has(model);
  return (
    <div
      className="stage-models-row"
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(140px,1fr) 2fr auto",
        gap: 8,
        alignItems: "center",
        padding: "6px 0",
        borderTop: "1px solid var(--border, #e5e5e5)",
      }}
    >
      <div>
        <strong>{label}</strong>
        <div className="obs-hint" style={{ fontSize: "0.8em" }}>{hint}</div>
      </div>
      <select
        className="model-select"
        data-testid={testIdPrefix ? `${testIdPrefix}-model-select` : undefined}
        value={model || ""}
        onChange={(e) =>
          onPatch({
            model: e.target.value,
            reasoning: REASONING_TOGGLE_MODELS.has(e.target.value)
              ? !!reasoning
              : false,
          })
        }
      >
        {modelOptions.map((id) => {
          const isDefault = id === defaultModelId;
          return (
            <option key={id} value={id}>
              {modelDisplayName(id)}
              {isDefault ? " (default)" : ""}
            </option>
          );
        })}
      </select>
      <label
        title={
          reasoningSupported
            ? "Toggle reasoning ON. Increases latency; opt in deliberately."
            : "This model has no verified reasoning control surface — toggle disabled."
        }
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 4,
          fontSize: "0.85em",
          opacity: reasoningSupported ? 1 : 0.4,
        }}
      >
        <input
          type="checkbox"
          data-testid={
            testIdPrefix ? `${testIdPrefix}-reasoning-toggle` : undefined
          }
          checked={reasoningSupported && !!reasoning}
          disabled={!reasoningSupported}
          onChange={(e) => onPatch({ reasoning: e.target.checked })}
        />
        reasoning
      </label>
    </div>
  );
}


// ── Topic taxonomy editor ─────────────────────────────────────────────
//
// Renders the editable category list. Each row is a text input + Up /
// Down / Remove controls; below the list sit Add and Reset-to-defaults
// buttons. Validation runs per-row (empty / shape / duplicate) and the
// component surfaces the rule that failed inline. The parent's save
// flow gates on `categoriesValid` so a malformed list can't hit disk.
function CategoriesSection({ categories, setCategories }) {
  // Per-row error chip text. Slightly redundant with the regex but
  // friendlier than dumping the source of truth at the user.
  // Note: uppercase is silently down-cased on input, so a "shape"
  // error here means a non-uppercase issue (space, special char,
  // leading hyphen). Wording reflects what's actually invalid by
  // the time the chip renders.
  const errorMessage = (kind) => {
    if (kind === "empty") return "Cannot be empty";
    if (kind === "shape") return "Letters, digits, hyphens only — no spaces";
    if (kind === "duplicate") return "Duplicate of another row";
    return null;
  };

  const update = (i, value) => {
    // Silently down-case on input. Uppercase-only typos would
    // otherwise surface as a "shape" error chip; lowering inline
    // is the more forgiving move and the canonical-form expectation
    // is unambiguous (the slug taxonomy is lowercase by convention).
    // Other shape failures (spaces, special chars) still surface.
    const next = categories.slice();
    next[i] = value.toLowerCase();
    setCategories(next);
  };
  const remove = (i) => {
    if (categories.length <= 1) return;
    const next = categories.slice();
    next.splice(i, 1);
    setCategories(next);
  };
  const add = () => setCategories([...categories, ""]);
  const reset = async () => {
    if (categoriesEqual(categories, DEFAULT_CATEGORIES)) return;
    const ok = await askConfirm(
      "Reset categories to the ship default? Your edits will be discarded.",
      { title: "Reset categories?", kind: "warning" },
    );
    if (ok) setCategories(DEFAULT_CATEGORIES);
  };

  const isDefault = categoriesEqual(categories, DEFAULT_CATEGORIES);

  return (
    <section className="wizard-section" data-section="categories">
      <header>
        <h2>Categories</h2>
      </header>
      <p className="field-hint">
        Topic taxonomy used by the extractor and pattern stages. The
        extractor only emits topics from this list — items the LLM tags
        with anything else are dropped. Edit to fit the shape of your
        own life (e.g. add <code>fundraising</code>, remove
        <code> spirituality</code>). Lowercase letters, digits, and
        hyphens; no spaces.
      </p>
      <ul className="categories-list" data-testid="categories-list">
        {categories.map((value, i) => {
          const others = categories.filter((_, j) => j !== i);
          const err = categoryError(value, others);
          return (
            <li
              key={i}
              className="categories-row"
              data-testid={`category-row-${i}`}
              style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}
            >
              <input
                type="text"
                value={value}
                onChange={(e) => update(i, e.target.value)}
                aria-label={`Category ${i + 1}`}
                data-testid={`category-input-${i}`}
                style={{ flex: 1 }}
              />
              <button
                type="button"
                onClick={() => remove(i)}
                disabled={categories.length <= 1}
                aria-label={`Remove ${value || "row"}`}
                title="Remove"
                data-testid={`category-remove-${i}`}
              >
                ×
              </button>
              {err && (
                <span
                  className="field-hint error"
                  data-testid={`category-error-${i}`}
                  style={{ color: "#c33", fontSize: "0.85em" }}
                >
                  {errorMessage(err)}
                </span>
              )}
            </li>
          );
        })}
      </ul>
      <div className="row gap" style={{ display: "flex", gap: 8, marginTop: 8 }}>
        <button
          type="button"
          onClick={add}
          data-testid="category-add"
        >
          + Add category
        </button>
        <button
          type="button"
          onClick={reset}
          disabled={isDefault}
          data-testid="category-reset"
        >
          Reset to defaults
        </button>
      </div>
    </section>
  );
}


// Compact byte formatter used in the cache-stats display + wipe
// dialog + update download progress. Matches the conventions in the
// rest of the UI (KB / MB thresholds, one decimal). Exported for the
// Vitest coverage in Settings.test.jsx.
export function formatBytes(n) {
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  const KB = 1024;
  const MB = KB * 1024;
  const GB = MB * 1024;
  if (n < KB) return `${n} B`;
  if (n < MB) return `${(n / KB).toFixed(1)} KB`;
  if (n < GB) return `${(n / MB).toFixed(1)} MB`;
  return `${(n / GB).toFixed(2)} GB`;
}

function formatProgress({ downloaded, total }) {
  if (!total) return downloaded ? formatBytes(downloaded) : "starting…";
  const pct = Math.min(100, Math.round((downloaded / total) * 100));
  return `${pct}% (${formatBytes(downloaded)} / ${formatBytes(total)})`;
}

// Heuristic — the updater plugin returns the underlying minisign error
// in the message string. We don't depend on a specific error variant
// because that's a moving target across plugin versions.
function isSignatureError(msg) {
  if (!msg) return false;
  const m = String(msg).toLowerCase();
  return (
    m.includes("signature") ||
    m.includes("minisign") ||
    m.includes("verification")
  );
}

// Duplicated from Wizard.jsx. Could be extracted to a shared module;
// leaving inline for now because Wizard + Settings are the only users.
function SectionStatus({ done, label, pendingLabel }) {
  return (
    <span className={`wizard-status ${done ? "done" : "pending"}`}>
      {done ? `✓ ${label || "done"}` : `· ${pendingLabel || "pending"}`}
    </span>
  );
}

