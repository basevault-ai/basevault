import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { openUrl } from "@tauri-apps/plugin-opener";
import { TEE_PROVIDERS, DEFAULT_TEE_PROVIDER } from "./teeProviders";
import { useLocalModel } from "./useLocalModel";
import { localBackendReady } from "./localUsable";
import ErrorWithTrace from "./ErrorWithTrace";

// Tauri webviews block `<a target="_blank">` — use the opener plugin
// to launch URLs in the system browser.
function externalLinkClick(url) {
  return (e) => {
    e.preventDefault();
    openUrl(url).catch((err) => console.error("openUrl failed:", err));
  };
}

const DEFAULT_VAULT_PATH = "~/Documents/BaseVault";

/**
 * First-run wizard. Four sections; each must reach a "done" state before
 * the user can finish the wizard. Minimum viable path: enter name, then
 * Skip on the other three.
 *
 * Section state machines:
 *   name:     invalid → valid                           (done: valid)
 *   local:    not-downloaded → downloaded                (done: downloaded | skipped)
 *   tee:      idle → verifying → verified | failed       (done: verified | skipped)
 *
 * Skip is always a valid terminal state — users can opt out of every
 * optional section and still finish. The local section uses the
 * bundled MLX path (no external install); Ollama stays an advanced
 * opt-in surfaced only in Settings, never in first-run onboarding.
 */

export default function Wizard({
  onComplete,
  onCancel,
  allowCancel = false,
  // Name-only "Easy Wizard" (issue #609). The whole flow collapses to
  // a single name field: no model section, no key section, and
  // deliberately zero cloud / Tinfoil / API wording (issue #606 — the
  // F&F user must not hit the cloud-shaped-onboarding gut-"no"). It
  // relies on the build's bundled Tinfoil key to make Private Cloud
  // just work; the menu only offers this entry on a build that has
  // one, so the path is always viable when reachable.
  easy = false,
}) {
  // Name
  const [name, setName] = useState("");

  // Local setup
  //   localChoice:  mlx | ollama | skip — bundled model (recommended),
  //     bring-your-own Ollama (detect-only verify), or opt out.
  //   localPrevVerified : was config.local_setup_mode === "verified" on
  //     mount? Drives the green chip on reopen without re-checking.
  const [localChoice, setLocalChoice] = useState("mlx");
  const [localPrevVerified, setLocalPrevVerified] = useState(false);
  // Local backend driver (MLX download/status + Ollama detect-only
  // verify), shared with Settings.
  const mlx = useLocalModel();

  // TEE (Private Cloud)
  // teeChoice: setup | skip — top-level Set up vs Skip radio.
  // teeProvider: tinfoil — Tinfoil is the only attested TEE provider.
  // Model selection is NOT in the wizard — it lives in Settings; the
  // wizard always writes the provider's default model on finish.
  const [teeChoice, setTeeChoice] = useState("setup");
  const [teeProvider, setTeeProvider] = useState(DEFAULT_TEE_PROVIDER);
  // Per-provider state cluster — single source of truth. Reads use
  // providers[id]; updates go through patchProvider(id, partial).
  // Schema mirrors Settings.jsx so the verify-shaped code reads
  // identical across both surfaces.
  //
  // Schema per provider:
  //   newKey       : string — typed in the password input this session
  //   maskedKey    : null | string — pre-existing key on disk, masked
  //                                  (drives the Clear-button display)
  //   onFile       : bool   — a key for this provider was on disk when
  //                           the wizard opened
  //   pendingClear : bool   — Clear was clicked, no replacement typed
  //   verifyStatus : "idle" | "verifying" | "verified" | "failed"
  //                  (in-session click result; "does the key auth")
  //   verifyError  : null | string  — paired with the failed status
  //
  // Trust-chain attestation isn't tracked here — wizard navigation is
  // a save-on-change flow, not a trust action. The upper-right Attest
  // button + app startup pass own attestation state.
  const makeProviderSlot = () => ({
    newKey: "", maskedKey: null, onFile: false, pendingClear: false,
    verifyStatus: "idle", verifyError: null,
  });
  const [providers, setProviders] = useState(() => ({
    tinfoil: makeProviderSlot(),
  }));
  const patchProvider = (id, partial) =>
    setProviders((prev) => ({ ...prev, [id]: { ...prev[id], ...partial } }));

  // Finish state
  const [saving, setSaving] = useState(false);
  const [finishError, setFinishError] = useState(null);

  // True when this build carries a bundled Tinfoil key — Private Cloud
  // then works with no user key, so the TEE section counts as done +
  // configured for the Finish gate even before the user enters a key.
  const [bundledKeySet, setBundledKeySet] = useState(false);

  // Re-run detection: if the user already saved settings/config before,
  // hydrate each section so they don't have to re-enter everything just
  // to tweak one thing. First-run users see empty state; reopeners see
  // their prior choices with done chips.
  const [isRerun, setIsRerun] = useState(false);
  // Body renders only after hydrate() resolves — otherwise the title
  // flickers from "Welcome to BaseVault" to "Onboarding Wizard" as
  // isRerun flips. Settings has the same gate via `loaded && config`.
  const [hydrated, setHydrated] = useState(false);

  // Escape closes the wizard — but ONLY when cancellation is allowed
  // (menu-triggered reopen). First-run (allowCancel=false) blocks Escape
  // so the user can't skip past required setup.
  useEffect(() => {
    if (!allowCancel) return;
    const onKey = (e) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onCancel?.();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [allowCancel, onCancel]);

  useEffect(() => {
    let mounted = true;
    async function hydrate() {
      try {
        const [settings, cfg] = await Promise.all([
          invoke("get_settings").catch(() => ({})),
          invoke("get_config").catch(() => ({})),
        ]);
        if (!mounted) return;
        const cfgSafe = cfg && typeof cfg === "object" ? cfg : {};
        const subject = (cfgSafe.subject || "").trim();
        const hasTinfoilKey = !!settings?.tinfoil_key_set;
        setBundledKeySet(!!settings?.bundled_key_set);

        // "Re-run" controls the modal's title + lead text. It's true if
        // ANY prior setup is on disk — name, or a TEE key. Don't gate
        // on subject alone; a user can land on the wizard because
        // subject is missing while still having keys configured.
        const hasAnyPriorSetup =
          subject.length > 0 || hasTinfoilKey;
        setIsRerun(hasAnyPriorSetup);

        // Hydrate each section independently from its own data source —
        // never gate on another section's data. The wizard fires when
        // ANY required field is missing, but other sections might still
        // be fully configured.
        if (subject) setName(subject);

        // TEE: hydrate provider from config; fall back to "the provider
        // whose key is on file" then to the global default. Model is
        // not surfaced in the wizard — Settings owns it.
        // Tinfoil is the only attested TEE provider; any other stale
        // `tee_provider` value falls back to the default rather than
        // rendering with a missing TEE_PROVIDERS entry (which would
        // crash on tagline/label).
        const cfgProvider = (cfgSafe.tee_provider || "").trim();
        const chosenProvider =
          cfgProvider && TEE_PROVIDERS[cfgProvider]
            ? cfgProvider
            : DEFAULT_TEE_PROVIDER;
        setTeeProvider(chosenProvider);

        if (hasTinfoilKey) {
          setTeeChoice("setup");
          // Hydrate Tinfoil key state. verifyStatus stays "idle" —
          // the green section chip on reopen comes from the
          // teePreviouslyVerified derivation (cur.onFile && !pendingClear
          // && !newKey.trim()), matching Settings' shape.
          patchProvider("tinfoil", {
            onFile: true,
            maskedKey: settings?.tinfoil_key_masked || null,
          });
        } else if (hasAnyPriorSetup) {
          // Returning user with no key on file → likely in "skip" state.
          setTeeChoice("skip");
        }
        // Pure first-run user (no prior setup at all): leave the default
        // initial state so they actively pick.

        // Local: the bundled-model status is read from disk (cheap);
        // the prior session's choice still feeds the green chip via
        // cfgSafe.local_setup_mode.
        mlx.refresh();
        const localMode = cfgSafe.local_setup_mode;
        if (localMode === "verified") {
          setLocalPrevVerified(true);
        } else if (localMode === "skipped") {
          setLocalChoice("skip");
        }
        // Default the radio to the previously-chosen backend (mlx
        // unless the user opted into ollama). Skip wins above.
        if (localMode !== "skipped" && cfgSafe.local_backend === "ollama") {
          setLocalChoice("ollama");
        }
      } catch {
        // best-effort hydration; first-run fallback is the initial state
      } finally {
        if (mounted) setHydrated(true);
      }
    }
    hydrate();
    return () => {
      mounted = false;
    };
    // Mount-only hydration; mlx.refresh is useCallback-stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Three sections, one shape ────────────────────────────────────────
  // Mirrors Settings.jsx — local / tee / obsidian share the same
  // PreviouslyVerified + Reverified + Done vocabulary, so the
  // verify-shaped code reads identical across both surfaces.
  //
  //   Status     : in-session click result (idle | verifying | verified
  //                | failed; local has extra "done"/"installing" states
  //                for install-vs-verify discrimination).
  //   Reverified : status === "verified"/"done" — fires only on a
  //                click in this session.
  //   PreviouslyVerified : derived from disk state — true iff this
  //                section is recorded as set up by some past session.
  //   Done       : skipped OR Reverified OR PreviouslyVerified. Drives
  //                the green chip on reopen.

  const cur = providers[teeProvider];

  const nameDone = name.trim().length > 0;

  // "Ready" this session: MLX → model on disk; Ollama → detect-only
  // verify passed. localPrevVerified covers a prior session having set
  // up either backend (drives the green chip without re-checking).
  const localReady =
    localChoice === "skip"
      ? false
      : localBackendReady({
          backend: localChoice,
          mlxDownloaded: mlx.stat?.downloaded,
          ollamaVerified: mlx.verifyStatus === "done",
        });
  const localPreviouslyVerified = localPrevVerified;
  const localDone =
    localChoice === "skip" || localReady || localPreviouslyVerified;

  const teeReverified = cur.verifyStatus === "verified";
  // For TEE: previously-verified means a key for the chosen provider
  // is on disk AND the user hasn't staged it for clear/replace this
  // session. Same shape Settings uses.
  const teePreviouslyVerified =
    cur.onFile && !cur.pendingClear && !cur.newKey.trim();
  // A bundled key makes Private Cloud usable with no user key, so the
  // section is done/configured automatically on a keyed build (mirrors
  // the Rust effective-key fallback + Settings' atLeastOneUsable).
  const teeDone =
    teeChoice === "skip" || teeReverified || teePreviouslyVerified ||
    (teeChoice !== "skip" && bundledKeySet);

  // The app needs at least one of Local or Private Cloud actually set up
  // to be useful — otherwise there's no mode to run the pipeline in.
  // The "configured" checks gate atLeastOneConfigured (Finish gate).
  // CRITICAL: a previously-verified section that the user is NOW
  // skipping must NOT count — otherwise users can open the wizard
  // with both Local + TEE previously set up, click Skip on both,
  // and Finish anyway with no functional mode. The skip choice
  // overrides any prior setup for the purposes of the gate.
  const localConfigured =
    localChoice !== "skip" && (localReady || localPreviouslyVerified);
  const teeConfigured =
    teeChoice !== "skip" &&
    (teeReverified || teePreviouslyVerified || bundledKeySet);
  const atLeastOneConfigured = localConfigured || teeConfigured;

  const canFinish =
    nameDone &&
    localDone &&
    teeDone &&
    atLeastOneConfigured &&
    !saving;

  // ── Section actions ──────────────────────────────────────────────────────

  function onLocalChoiceChange(next) {
    setLocalChoice(next);
    mlx.setError(null);
    mlx.resetVerify();
    // Persist the backend immediately (skip is about local_setup_mode,
    // not the backend — leave local_backend untouched there). Settings
    // is a separate component that only learns the choice by re-reading
    // config.json, so a finish-only write would let the two disagree.
    if (next === "mlx" || next === "ollama") persistLocalBackend(next);
  }

  async function persistLocalBackend(next) {
    try {
      await invoke("update_config", { patch: { local_backend: next } });
    } catch {
      /* best-effort; the wizard Finish still writes it as a fallback */
    }
  }

  // "Does this key authenticate against Tinfoil?" — minimal probe,
  // no trust-chain work. Mirrors Settings.verifyTeeKey. Trust-chain
  // attestation lives on the upper-right Attest button + app startup.
  async function verifyTee() {
    const provider = teeProvider;
    const cfg = TEE_PROVIDERS[provider];
    const ps = providers[provider];
    const newKey = ps.newKey.trim();
    if (!newKey && !ps.onFile) {
      patchProvider(provider, {
        verifyStatus: "failed",
        verifyError: `Enter a ${cfg.label} API key or skip this step.`,
      });
      return;
    }
    patchProvider(provider, { verifyStatus: "verifying", verifyError: null });
    try {
      await invoke(cfg.verifyCommand, { key: newKey || undefined });
      patchProvider(provider, { verifyStatus: "verified", verifyError: null });
    } catch (e) {
      const msg = e?.message || String(e);
      patchProvider(provider, { verifyStatus: "failed", verifyError: msg });
    }
  }

  function onTeeChoiceChange(next) {
    setTeeChoice(next);
    // Clear in-flight / failed verify states; "verified" stays so a
    // stored key counts as verified across Setup/Skip toggles.
    const s = providers.tinfoil.verifyStatus;
    if (s === "failed" || s === "verifying") {
      patchProvider("tinfoil", { verifyStatus: "idle", verifyError: null });
    } else {
      patchProvider("tinfoil", { verifyError: null });
    }
  }

  // Switch the active sub-provider. Per-provider state stays preserved
  // (each provider has its own slot in `providers`).
  function onTeeProviderChange(next) {
    setTeeProvider(next);
  }

  // Per-provider key edit handler: typing into the password input
  // resets pendingClear (you've started replacing) and any previous
  // verify result so the user can't get past the section with a
  // stale "verified ✓" while editing.
  function onTeeKeyChange(provider, value) {
    patchProvider(provider, {
      newKey: value,
      ...(value.trim() ? { pendingClear: false } : {}),
      verifyStatus: "idle",
      verifyError: null,
    });
  }

  // ── Finish ────────────────────────────────────────────────────────────────
  async function finish() {
    if (!canFinish) return;
    setSaving(true);
    setFinishError(null);
    try {
      // 1. Dotenv: per-provider key updates. Three states (mirror
      // Settings save):
      //   - typed a new key   → write the new value
      //   - pendingClear (no new typed) → write "" (delete)
      //   - neither           → null (leave the line untouched)
      const keyToSave = (id) => {
        const ps = providers[id];
        if (ps.newKey.trim()) return ps.newKey.trim();
        if (ps.pendingClear) return "";
        return null;
      };
      const tinfoilKeyToSave = keyToSave("tinfoil");
      await invoke("save_settings", {
        tinfoilKey: tinfoilKeyToSave,
      });

      // 2. Save config.json: subject, local_setup_mode, and
      // tee_provider/tee_model (only when the user picked Setup, not
      // Skip — Skip leaves whatever was previously chosen alone so a
      // future re-enable hydrates back to the same provider/model).
      // Read-only: needed to default resultingMode when neither TEE nor
      // local was newly configured. The write below is a narrow patch,
      // so this snapshot read can't lost-update concurrent persisters.
      const cfg = (await invoke("get_config").catch(() => ({}))) || {};
      const localMode = localChoice === "skip" ? "skipped" : "verified";
      // Wizard sets only tee_provider; tee_model is owned by Settings
      // (so re-running the wizard never overwrites a model the user
      // explicitly picked in Settings). _build_tee_spec falls back to
      // the provider's default model when tee_model is missing.
      const teeFields =
        teeChoice === "setup" ? { tee_provider: teeProvider } : {};
      // Pick the resulting mode based on what the user just configured.
      // Prefer TEE if they set it up, else local. Don't override an
      // already-persisted choice if neither was newly configured.
      const teeUsable = teeConfigured && teeChoice === "setup";
      let resultingMode = cfg.mode || null;
      if (teeUsable) resultingMode = "tee";
      else if (localChoice !== "skip" && localReady)
        resultingMode = "local";
      const modeField = resultingMode ? { mode: resultingMode } : {};
      // Persist the chosen backend so _build_local_spec routes to it.
      // Skip leaves whatever was previously set.
      const localBackendField =
        localChoice === "skip" ? {} : { local_backend: localChoice };
      await invoke("update_config", {
        patch: {
          subject: name.trim(),
          local_setup_mode: localMode,
          ...localBackendField,
          ...teeFields,
          ...modeField,
        },
      });

      onComplete?.(resultingMode);
    } catch (e) {
      setFinishError(e?.message || String(e));
      setSaving(false);
    }
  }

  // ── Easy (name-only) finish ──────────────────────────────────────────
  // No save_settings call: the Easy path never collects a key — the
  // build's bundled Tinfoil key powers Private Cloud. We persist the
  // name and point the pipeline at Private Cloud (tee + tinfoil). If
  // the user later enters their own key in Settings it transparently
  // wins (effective_tinfoil_key precedence on the Rust side). We don't
  // touch local_* — Easy is purely "name in, ready".
  async function finishEasy() {
    if (!name.trim() || saving) return;
    setSaving(true);
    setFinishError(null);
    try {
      await invoke("update_config", {
        patch: {
          subject: name.trim(),
          mode: "tee",
          tee_provider: "tinfoil",
        },
      });
      onComplete?.("tee");
    } catch (e) {
      setFinishError(e?.message || String(e));
      setSaving(false);
    }
  }

  if (!hydrated) {
    // Don't render content until we know what the user has on disk —
    // otherwise the title flickers from "Welcome to BaseVault" (the
    // first-run default) to "Onboarding Wizard" as isRerun resolves.
    return (
      <div className="modal-backdrop">
        <div className="modal wizard-modal">
          <p>Loading…</p>
        </div>
      </div>
    );
  }

  if (easy) {
    // Name-only flow. Deliberately no model/key sections and no
    // cloud/Tinfoil/API wording (#606). The bundled key (#609) makes
    // the first run just work; the menu only offers this entry on a
    // build that has the key, so the component never has to handle
    // the keyless case.
    const canFinishEasy = name.trim().length > 0 && !saving;
    return (
      <div
        className="modal-backdrop"
        onClick={allowCancel ? () => onCancel?.() : undefined}
      >
        <div
          className="modal wizard-modal wizard-modal-easy"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="settings-fixed-head">
            <h1>Welcome to BaseVault</h1>
            <p className="modal-lead">
              Just enter your name, and you're ready to go.
            </p>
          </div>

          <div className="settings-scroll-body">
            <section
              className={`wizard-section ${name.trim() ? "done" : ""}`}
            >
              <header>
                <h2>Your name</h2>
              </header>
              <input
                type="text"
                autoFocus
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Jane Smith"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && canFinishEasy) finishEasy();
                }}
              />
              <p className="field-hint">
                Used so the pipeline writes actions and insights for you
                specifically — not for whoever else appears in your data.
              </p>
            </section>

            {finishError && <p className="error">{finishError}</p>}
            <div className="modal-actions">
              {allowCancel && (
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => onCancel?.()}
                  disabled={saving}
                >
                  Cancel
                </button>
              )}
              <button
                className="btn-primary"
                disabled={!canFinishEasy}
                onClick={finishEasy}
              >
                {saving ? "Saving…" : "Finish"}
              </button>
            </div>
            {!canFinishEasy && !saving && (
              <p className="field-hint wizard-gate-hint">
                Enter your name to continue.
              </p>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="modal-backdrop"
      onClick={allowCancel ? () => onCancel?.() : undefined}
    >
      <div
        className="modal wizard-modal"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Fixed head: title + lead. Body below scrolls. Same shape
            Settings uses so the wizard doesn't overflow the viewport
            on common laptop heights. */}
        <div className="settings-fixed-head">
          <h1>{isRerun ? "Onboarding Wizard" : "Welcome to BaseVault"}</h1>
          <p className="modal-lead">
            {isRerun
              ? "Your existing settings are loaded. Change anything you want, or click Finish to keep them."
              : "Three quick things before the first run. Your name is required; the other two can be skipped if they don't apply."}
          </p>
        </div>

        <div className="settings-scroll-body">
        {/* ── Section 1: Name ──────────────────────────────────────────── */}
        <section className={`wizard-section ${nameDone ? "done" : ""}`}>
          <header>
            <span className="wizard-step">1</span>
            <h2>Your name</h2>
            <SectionStatus done={nameDone} label="" />
          </header>
          <input
            type="text"
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Jane Smith"
          />
          <p className="field-hint">
            Used so the pipeline writes actions and insights for you
            specifically — not for whoever else appears in your data.
          </p>
        </section>

        {/* ── Section 2: Local ─────────────────────────────────────────── */}
        {/* Grey on Skip, green on actual local-setup completion. */}
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
            <span className="wizard-step">2</span>
            <h2>Local processing</h2>
            <SectionStatus
              done={localChoice !== "skip" && localDone}
              label={
                localChoice === "skip"
                  ? "skipped"
                  : localReady || localPreviouslyVerified
                  ? "ready"
                  : ""
              }
            />
          </header>
          <p className="field-hint">
            Runs the pipeline entirely on your machine — nothing leaves
            the device. MLX is bundled (no external install); Ollama is
            for users who already run it.
          </p>

          <div className="obsidian-options">
            <label className={`obs-option ${localChoice === "mlx" ? "active" : ""}`}>
              <input
                type="radio"
                name="local"
                checked={localChoice === "mlx"}
                onChange={() => onLocalChoiceChange("mlx")}
              />
              <div>
                <strong>MLX</strong> — recommended; download the bundled
                model (multi-GB, a few minutes on first run; nothing
                downloads until you click)
                {localChoice === "mlx" && (
                  <>
                    <div className="row gap">
                      <button
                        type="button"
                        onClick={mlx.download}
                        disabled={mlx.busy || !!mlx.stat?.downloaded}
                      >
                        {mlx.busy
                          ? "Downloading…"
                          : mlx.stat?.downloaded
                          ? "Downloaded ✓"
                          : "Download model"}
                      </button>
                    </div>
                    {mlx.busy && (
                      <p className="verify-status">
                        {mlx.pct != null ? `${mlx.pct}% — ` : ""}
                        {mlx.msg || "Working…"}
                      </p>
                    )}
                    {mlx.error && !mlx.busy && (
                      <p className="verify-status verify-failed">
                        ✗ {String(mlx.error).split("\n").slice(-3).join(" · ")}
                      </p>
                    )}
                    {(mlx.stat?.downloaded || localPreviouslyVerified) &&
                      !mlx.busy && (
                      <p className="verify-status verify-ok">
                        ✓ Local model ready
                      </p>
                    )}
                  </>
                )}
              </div>
            </label>

            <label className={`obs-option ${localChoice === "ollama" ? "active" : ""}`}>
              <input
                type="radio"
                name="local"
                checked={localChoice === "ollama"}
                onChange={() => onLocalChoiceChange("ollama")}
              />
              <div>
                <strong>Ollama</strong> — only if you already run it; we
                verify your environment (no install)
                {localChoice === "ollama" && (
                  <>
                    <div className="row gap">
                      <button
                        type="button"
                        onClick={mlx.verify}
                        disabled={mlx.verifyStatus === "verifying"}
                      >
                        {mlx.verifyStatus === "verifying"
                          ? "Checking…"
                          : mlx.verifyStatus === "done"
                          ? "Verified ✓ — re-check"
                          : "Check Ollama"}
                      </button>
                    </div>
                    {mlx.verifyStatus === "failed" && mlx.verifyError && (
                      <>
                        <p className="verify-status verify-failed">
                          ✗ {String(mlx.verifyError).split("\n").slice(-3).join(" · ")}
                        </p>
                        {mlx.fixCmd && (
                          <div className="row gap">
                            <pre className="install-script">
                              <code>{mlx.fixCmd}</code>
                            </pre>
                            <button type="button" onClick={mlx.copyFixCmd}>
                              {mlx.cmdCopied ? "Copied ✓" : "Copy"}
                            </button>
                          </div>
                        )}
                      </>
                    )}
                    {(mlx.verifyStatus === "done" || localPreviouslyVerified) && (
                      <p className="verify-status verify-ok">
                        ✓ Ollama daemon up, model present
                      </p>
                    )}
                  </>
                )}
              </div>
            </label>

            <label className={`obs-option ${localChoice === "skip" ? "active" : ""}`}>
              <input
                type="radio"
                name="local"
                checked={localChoice === "skip"}
                onChange={() => onLocalChoiceChange("skip")}
              />
              <div>
                <strong>Skip — don't use local mode</strong>
                <p className="obs-hint">
                  The Local option stays visible in the app but will
                  error at runtime if picked before a backend is set up
                  (from Settings).
                </p>
              </div>
            </label>
          </div>
        </section>

        {/* ── Section 3: Private Cloud (TEE) ──────────────────────────── */}
        {/* Grey on Skip, green on actual TEE setup. */}
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
            <span className="wizard-step">3</span>
            <h2>Private Cloud mode (TEE)</h2>
            <SectionStatus
              done={teeChoice !== "skip" && teeDone}
              label={teeDone ? "ready" : ""}
              pendingLabel={teeChoice === "skip" ? "skipped" : "pending"}
            />
          </header>
          <p className="field-hint">
            Cloud processing with privacy guarantees — runs inside a
            Trusted Execution Environment. Pick a provider; we'll use the
            same API key + model for every cloud run.
          </p>

          <div className="obsidian-options">
            <label className={`obs-option ${teeChoice === "setup" ? "active" : ""}`}>
              <input
                type="radio"
                name="tee"
                checked={teeChoice === "setup"}
                onChange={() => onTeeChoiceChange("setup")}
              />
              <div>
                <strong>Set up Private Cloud with Tinfoil</strong>
                {teeChoice === "setup" && (
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

                    {/* Stored-key display: masked + Clear button. Only
                        rendered when a key is on file, not pending
                        clear, and no new key has been typed. Same
                        shape Settings uses. */}
                    {cur.onFile && !cur.pendingClear && !cur.newKey.trim() && (
                      <div className="row gap">
                        <code className="masked-key">
                          {cur.maskedKey || "••••••••"}
                        </code>
                        <button
                          type="button"
                          onClick={() =>
                            patchProvider(teeProvider, {
                              pendingClear: true,
                              verifyStatus: "idle",
                              verifyError: null,
                            })
                          }
                          disabled={saving}
                        >
                          Clear
                        </button>
                      </div>
                    )}
                    {/* Password input. Hidden when the masked-key row is
                        showing. Pending-clear with no typed key shows
                        an Undo button so the user can revert without
                        leaving the row. */}
                    {!(cur.onFile && !cur.pendingClear && !cur.newKey.trim()) && (
                      <div className="row gap">
                        <input
                          type="password"
                          value={cur.newKey}
                          onChange={(e) =>
                            onTeeKeyChange(teeProvider, e.target.value)
                          }
                          placeholder={TEE_PROVIDERS[teeProvider].keyPlaceholder}
                          autoComplete="off"
                          disabled={cur.verifyStatus === "verifying"}
                          style={{ flex: 1 }}
                        />
                        {cur.onFile && cur.pendingClear && !cur.newKey.trim() && (
                          <button
                            type="button"
                            onClick={() =>
                              patchProvider(teeProvider, { pendingClear: false })
                            }
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
                        onClick={verifyTee}
                        disabled={
                          cur.verifyStatus === "verifying" ||
                          // Disabled when there's nothing to verify:
                          // no typed key, AND either no key on file or
                          // the on-file key is pending clear.
                          (cur.newKey.trim().length === 0 &&
                            (!cur.onFile || cur.pendingClear))
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
                )}
              </div>
            </label>

            <label className={`obs-option ${teeChoice === "skip" ? "active" : ""}`}>
              <input
                type="radio"
                name="tee"
                checked={teeChoice === "skip"}
                onChange={() => onTeeChoiceChange("skip")}
              />
              <div>
                <strong>Skip — use Local mode only</strong>
                <p className="obs-hint">
                  Private Cloud will be disabled in the mode selector.
                  You can always add a key later from Settings.
                </p>
              </div>
            </label>
          </div>
        </section>

        {/* ── Finish ───────────────────────────────────────────────────── */}
        {finishError && <p className="error">{finishError}</p>}
        <div className="modal-actions">
          {allowCancel && (
            <button
              type="button"
              className="btn-secondary"
              onClick={() => onCancel?.()}
              disabled={saving}
            >
              Cancel
            </button>
          )}
          <button
            className="btn-primary"
            disabled={!canFinish}
            onClick={finish}
          >
            {saving ? "Saving…" : "Finish"}
          </button>
        </div>
        {!canFinish && !saving && (
          <p className="field-hint wizard-gate-hint">
            {!nameDone
              ? "Enter your name to continue."
              : !localDone
              ? "Set up or skip local processing."
              : !teeDone
              ? "Verify your Private Cloud key or skip this step."
              : !atLeastOneConfigured
              ? "You need at least one of Local or Private Cloud actually set up — a pipeline run has to happen somewhere."
              : ""}
          </p>
        )}
        </div>
      </div>
    </div>
  );
}

function SectionStatus({ done, label, pendingLabel }) {
  return (
    <span className={`wizard-status ${done ? "done" : "pending"}`}>
      {done ? `✓ ${label || "done"}` : `· ${pendingLabel || "pending"}`}
    </span>
  );
}
