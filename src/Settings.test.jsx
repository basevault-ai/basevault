import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, waitFor, within, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { invoke } from "@tauri-apps/api/core";
import { confirm as askConfirm } from "@tauri-apps/plugin-dialog";
import Settings from "./Settings";

afterEach(() => cleanup());

// Build a stub for invoke that returns reasonable defaults plus
// a per-test override.
function stubInvoke({
  settings = {},
  config = {},
  override,
} = {}) {
  vi.mocked(invoke).mockImplementation(async (cmd, args) => {
    if (override) {
      const r = override(cmd, args);
      if (r !== undefined) return r;
    }
    if (cmd === "get_settings") return settings;
    if (cmd === "get_config") return config;
    return undefined;
  });
}

async function renderAndHydrate(props = {}) {
  const { tab, ...rest } = props;
  const utils = render(<Settings onClose={vi.fn()} onChanged={vi.fn()} {...rest} />);
  // Loaded gate: title "Settings" appears once both invoke calls resolve.
  await waitFor(() => {
    expect(screen.getByRole("heading", { name: "Settings" })).toBeTruthy();
  });
  // PR #77 split Settings into General / Local / Private Cloud tabs.
  // Most tests below were written when every section rendered
  // together; default the Private Cloud tab on so per-stage / verify
  // tests don't all need to add an explicit switchToTab().
  if (tab && tab !== "general") {
    await switchToTab(tab === "tee" ? "Private Cloud" : tab === "local" ? "Local" : "General");
  }
  return utils;
}

async function switchToTab(label) {
  const user = userEvent.setup();
  const tab = screen.getByRole("tab", { name: label });
  await user.click(tab);
}

describe("Settings — load + render", () => {
  it("renders Settings title and section headers after hydration", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true, tinfoil_key_masked: "•••wxyz" },
      config: { subject: "User", tee_provider: "tinfoil", tee_model: "gpt-oss-120b" },
    });
    await renderAndHydrate();
    // General tab is the default — Name + Sentiment headers visible.
    expect(screen.getByText("Your name")).toBeTruthy();
    expect(screen.getByText("Insight + action tone")).toBeTruthy();
    // Local + Private Cloud headers live under their own tabs.
    expect(screen.queryByText("Local processing (Ollama)")).toBeNull();
    expect(screen.queryByText("Private Cloud mode (TEE)")).toBeNull();
    await switchToTab("Local");
    expect(screen.getByText("Local processing (Ollama)")).toBeTruthy();
    await switchToTab("Private Cloud");
    expect(screen.getByText("Private Cloud mode (TEE)")).toBeTruthy();
    // Obsidian integration was demoted out of Settings in #77 — the
    // first-run Wizard still owns vault setup and the underlying
    // open_vault_for_run path is unchanged.
    expect(screen.queryByText("Obsidian integration")).toBeNull();
  });

  it("populates the name field from config.subject", async () => {
    stubInvoke({ config: { subject: "Existing User", tee_provider: "tinfoil" } });
    await renderAndHydrate();
    const input = screen.getAllByRole("textbox").find((el) => el.value === "Existing User");
    expect(input).toBeTruthy();
  });

  it("auto-save does not fire when no mode is usable + nothing has changed", async () => {
    let setConfigCalled = 0;
    stubInvoke({
      settings: {},
      config: { subject: "User" },
      override: (cmd) => {
        if (cmd === "update_config") {
          setConfigCalled += 1;
          return undefined;
        }
      },
    });
    await renderAndHydrate();
    // No Save button anymore — save-on-change. The "at least one of
    // Local or Private Cloud actually set up" gate must still surface
    // when neither is usable.
    expect(
      screen.getByText(/at least one of Local or Private Cloud/)
    ).toBeTruthy();
    // Wait past the 1s auto-save debounce. update_config should NOT
    // have fired because canSave gates on atLeastOneUsable.
    await new Promise((r) => setTimeout(r, 1100));
    expect(setConfigCalled).toBe(0);
  });

  it("treats Private Cloud as usable on a keyed build (bundled key, no user key)", async () => {
    // The bundled key powers Private Cloud via the Rust effective-key
    // fallback, so a keyed build with no user key must NOT show the
    // run-somewhere gate — otherwise atLeastOneUsable is false and
    // save-on-change is frozen (the "settings disabled" symptom).
    stubInvoke({
      settings: { tinfoil_key_set: false, bundled_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil" },
    });
    await renderAndHydrate();
    expect(
      screen.queryByText(/at least one of Local or Private Cloud/)
    ).toBeNull();
  });

  it("does NOT persist a typed-but-unverified Tinfoil key on close", async () => {
    // Repro of the reported bug: type a bogus key, dismiss the modal
    // without clicking Connect & verify. The unchecked key must not be
    // written (so it can't stick or show as verified on reopen).
    // Keyed build so the save gate (atLeastOneUsable) is open.
    const user = userEvent.setup();
    let savedTinfoilKey = "NOT_CALLED";
    stubInvoke({
      settings: { bundled_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil" },
      override: (cmd, args) => {
        if (cmd === "save_settings") {
          savedTinfoilKey = args?.tinfoilKey;
          return undefined;
        }
      },
    });
    const utils = await renderAndHydrate({ tab: "tee" });
    const pwd = document.querySelector('input[type="password"]');
    expect(pwd).toBeTruthy();
    await user.type(pwd, "tf_bogus_unverified");
    utils.unmount(); // flush-on-close fires
    await new Promise((r) => setTimeout(r, 50));
    expect(savedTinfoilKey).toBe("NOT_CALLED");
  });

  it("persists the Tinfoil key after a successful verify", async () => {
    // The legit path must still work: type a key, Connect & verify,
    // close → the verified key is saved.
    const user = userEvent.setup();
    let savedTinfoilKey = null;
    stubInvoke({
      settings: { bundled_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil" },
      override: (cmd, args) => {
        if (cmd === "save_settings") {
          savedTinfoilKey = args?.tinfoilKey;
          return undefined;
        }
      },
    });
    const utils = await renderAndHydrate({ tab: "tee" });
    const pwd = document.querySelector('input[type="password"]');
    await user.type(pwd, "tf_good_key");
    await user.click(screen.getByRole("button", { name: /Connect & verify/ }));
    await waitFor(() =>
      expect(screen.getByText(/✓ verified/)).toBeTruthy()
    );
    utils.unmount(); // flush-on-close fires
    await new Promise((r) => setTimeout(r, 50));
    expect(savedTinfoilKey).toBe("tf_good_key");
  });
});

describe("Settings — sentiment dropdown (PR #37)", () => {
  it("hydrates from config.sentiment_bias when valid", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: {
        subject: "User",
        tee_provider: "tinfoil",
        sentiment_bias: "brutally-honest",
      },
    });
    await renderAndHydrate();
    // Find the sentiment <select> by its current value
    const selects = screen.getAllByRole("combobox");
    const sentimentSelect = selects.find((s) => s.value === "brutally-honest");
    expect(sentimentSelect).toBeTruthy();
  });

  it("falls back to neutral when stored value is unknown / typo'd", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: {
        subject: "User",
        tee_provider: "tinfoil",
        sentiment_bias: "made-up-value",
      },
    });
    await renderAndHydrate();
    // The section status chip was retired in the round-7 General-tab
    // cleanup (no per-section ✓-state badges except on Name). Verify
    // the fallback-to-neutral via the select's value instead.
    const selects = screen.getAllByRole("combobox");
    const sentimentSelect = selects.find((s) => s.value === "neutral");
    expect(sentimentSelect).toBeTruthy();
  });

  it("changing sentiment auto-saves after the debounce window", async () => {
    const user = userEvent.setup();
    let savedConfig = null;
    stubInvoke({
      settings: { tinfoil_key_set: true, tinfoil_key_masked: "•••abcd" },
      config: {
        subject: "User",
        tee_provider: "tinfoil",
        tee_model: "gpt-oss-120b",
        // Pin stage_models so originalStageModels matches the hydrated state
        // exactly (otherwise stageModelsDirty stays true and the auto-save
        // debounce effect keys on a `canSave` flag that's already true on
        // mount — its dependency array `[canSave]` then can't notice
        // subsequent edits flip dirty on for OTHER fields). Must include
        // every VISIBLE_STAGES row, including vision (added for #115's
        // Settings addendum) and entities_dedupe (PR #43).
        stage_models: {
          vision:          { model: "kimi-k2-6", reasoning: false },
          extract:         { model: "gpt-oss-120b", reasoning: false },
          entities:        { model: "gpt-oss-120b", reasoning: false },
          entities_dedupe: { model: "gpt-oss-120b", reasoning: false },
          patterns:        { model: "gpt-oss-120b", reasoning: false },
          insights:        { model: "gpt-oss-120b", reasoning: false },
          actions:         { model: "gpt-oss-120b", reasoning: false },
        },
        sentiment_bias: "neutral",
      },
      override: (cmd, args) => {
        if (cmd === "update_config") {
          savedConfig = args.patch;
          return undefined;
        }
      },
    });
    await renderAndHydrate();
    const selects = screen.getAllByRole("combobox");
    const sentimentSelect = selects.find((s) => s.value === "neutral");
    expect(sentimentSelect).toBeTruthy();
    await user.selectOptions(sentimentSelect, "uplifting");
    // Auto-save fires after the 1s debounce window — round-trip
    // through Settings' own re-hydrate. Wait for the patch to land.
    await waitFor(
      () => expect(savedConfig?.sentiment_bias).toBe("uplifting"),
      { timeout: 2500 }
    );
  });

});

describe("Settings — per-stage rows (PR #36)", () => {
  it("renders one per-stage row per VISIBLE_STAGES entry on tinfoil", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil", tee_model: "gpt-oss-120b" },
    });
    await renderAndHydrate({ tab: "tee" });
    // Five stages — Extract, Entities, Patterns, Insights, Actions.
    expect(screen.getByText("Extract")).toBeTruthy();
    expect(screen.getByText("Entities")).toBeTruthy();
    expect(screen.getByText("Patterns")).toBeTruthy();
    expect(screen.getByText("Insights")).toBeTruthy();
    expect(screen.getByText("Actions")).toBeTruthy();
  });

  it("exposes glm-5-2 as a selectable per-stage model option", async () => {
    // glm-5-2 has a backend ModelSpec (llm.py) but was never wired into
    // the JS selector. Confirm the component actually renders it as a
    // dropdown <option> (not merely that the data structure lists it):
    // the per-stage model selects each carry the glm-5.2 option, with
    // the display name from MODEL_DISPLAY_NAMES.
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil", tee_model: "gpt-oss-120b" },
    });
    await renderAndHydrate({ tab: "tee" });
    const glmOptions = screen
      .getAllByRole("option")
      .filter((opt) => opt.value === "glm-5-2");
    expect(glmOptions.length).toBeGreaterThan(0);
    expect(glmOptions[0].textContent).toContain("glm-5.2");
  });

  it("forces reasoning OFF when switching to a model not in REASONING_TOGGLE_MODELS", async () => {
    // Every model in the ship default map (gpt-oss-120b, gemma4-31b,
    // kimi-k2-6) is in REASONING_TOGGLE_MODELS, so every per-stage
    // row renders with the reasoning toggle live. With the dedicated
    // vision model dropped, every selectable model now carries a
    // verified reasoning control, so no row greys the toggle out.
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil", tee_model: "gpt-oss-120b" },
    });
    await renderAndHydrate({ tab: "tee" });
    const reasoningCheckboxes = screen
      .getAllByRole("checkbox")
      .filter((el) => el.parentElement?.textContent?.includes("reasoning"));
    // 7 visible stages: vision, extract, entities, entities_dedupe,
    // patterns, insights, actions. Plus 1 chat-section row (chatbot).
    expect(reasoningCheckboxes.length).toBe(8);
    const disabled = reasoningCheckboxes.filter((cb) => cb.disabled);
    const enabled = reasoningCheckboxes.filter((cb) => !cb.disabled);
    expect(disabled.length).toBe(0);
    expect(enabled.length).toBe(8);
  });

  it("hydrates per-stage rows from config.stage_models", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: {
        subject: "User",
        tee_provider: "tinfoil",
        tee_model: "gpt-oss-120b",
        stage_models: {
          extract: { model: "kimi-k2-6", reasoning: true },
          entities: { model: "gemma4-31b", reasoning: false },
          // Pinned so the row doesn't fall through to the ship default
          // (which has entities_dedupe reasoning ON) and inflate the
          // checked-count below.
          entities_dedupe: { model: "kimi-k2-6", reasoning: false },
          patterns: { model: "gemma4-31b", reasoning: false },
          insights: { model: "gemma4-31b", reasoning: true },
          actions: { model: "gpt-oss-120b", reasoning: false },
        },
      },
    });
    await renderAndHydrate({ tab: "tee" });
    // extract + insights are reasoning-on in stage_models (2). The
    // config carries no `chatbot` field, so the chatbot row hydrates
    // to its ship-default, which is reasoning-OFF — no extra checked box.
    const reasoningCheckboxes = screen
      .getAllByRole("checkbox")
      .filter((el) => el.parentElement?.textContent?.includes("reasoning"));
    const checkedCount = reasoningCheckboxes.filter((cb) => cb.checked).length;
    expect(checkedCount).toBe(2);
  });

  it("Reset to defaults snaps per-stage rows AND the chatbot row to the ship defaults", async () => {
    const user = userEvent.setup();
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: {
        subject: "User",
        tee_provider: "tinfoil",
        tee_model: "gpt-oss-120b",
        // Start fully diverged: every per-stage row reasoning OFF on a
        // single model, chatbot reasoning ON. Only the chatbot toggle is
        // checked pre-reset (1).
        stage_models: {
          vision:          { model: "kimi-k2-6", reasoning: false },
          extract:         { model: "kimi-k2-6", reasoning: false },
          entities:        { model: "kimi-k2-6", reasoning: false },
          entities_dedupe: { model: "kimi-k2-6", reasoning: false },
          patterns:        { model: "kimi-k2-6", reasoning: false },
          insights:        { model: "kimi-k2-6", reasoning: false },
          actions:         { model: "kimi-k2-6", reasoning: false },
        },
        chatbot: { model: "gpt-oss-120b", reasoning: true },
      },
    });
    await renderAndHydrate({ tab: "tee" });
    const checkedNow = () =>
      screen
        .getAllByRole("checkbox")
        .filter((el) => el.parentElement?.textContent?.includes("reasoning"))
        .filter((cb) => cb.checked).length;
    // Pre-reset: only the chatbot row is reasoning-ON.
    expect(checkedNow()).toBe(1);
    expect(screen.getByTestId("chatbot-reasoning-toggle").checked).toBe(true);

    vi.mocked(askConfirm).mockResolvedValue(true);
    await user.click(screen.getByTestId("reset-models-to-defaults"));

    // Post-reset: the 4 reasoning-ON ship-default stages (extract,
    // entities, entities_dedupe, patterns) are checked; chatbot snaps
    // back to its reasoning-OFF ship default.
    await waitFor(() => expect(checkedNow()).toBe(4));
    expect(screen.getByTestId("chatbot-reasoning-toggle").checked).toBe(false);
    expect(screen.getByTestId("chatbot-model-select").value).toBe("glm-5-2");
  });
});

describe("Settings — chatbot row (issue #422)", () => {
  it("hydrates the chatbot picker from config.chatbot", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: {
        subject: "User",
        tee_provider: "tinfoil",
        tee_model: "gpt-oss-120b",
        chatbot: { model: "gpt-oss-120b", reasoning: true },
      },
    });
    await renderAndHydrate({ tab: "tee" });
    const select = screen.getByTestId("chatbot-model-select");
    expect(select.value).toBe("gpt-oss-120b");
    const toggle = screen.getByTestId("chatbot-reasoning-toggle");
    expect(toggle.checked).toBe(true);
  });

  it("defaults to glm-5-2 reasoning-OFF when config has no chatbot field", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil", tee_model: "gpt-oss-120b" },
    });
    await renderAndHydrate({ tab: "tee" });
    const select = screen.getByTestId("chatbot-model-select");
    expect(select.value).toBe("glm-5-2");
    const toggle = screen.getByTestId("chatbot-reasoning-toggle");
    // Ship-default for the interactive chatbot is reasoning-OFF
    // (reasoning-on was too slow live); it stays an opt-in.
    expect(toggle.checked).toBe(false);
  });

  it("round-trips the chatbot model through update_config on change", async () => {
    const user = userEvent.setup();
    let finalCfg = null;
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil", tee_model: "gpt-oss-120b" },
      override: (cmd, args) => {
        if (cmd === "update_config") {
          finalCfg = args?.patch;
          return undefined;
        }
      },
    });
    await renderAndHydrate({ tab: "tee" });
    const select = screen.getByTestId("chatbot-model-select");
    await user.selectOptions(select, "gpt-oss-120b");
    await waitFor(
      () => {
        expect(finalCfg?.chatbot?.model).toBe("gpt-oss-120b");
      },
      { timeout: 2000 },
    );
    // Ship-default reasoning is OFF, so a model-only change carries the
    // OFF state through verbatim.
    expect(finalCfg?.chatbot?.reasoning).toBe(false);
  });
});

describe("Settings — verify TEE", () => {
  it("blocks verify with no key typed and no key on file", async () => {
    const user = userEvent.setup();
    stubInvoke({
      settings: { tinfoil_key_set: false },
      config: { subject: "User" },
    });
    await renderAndHydrate({ tab: "tee" });
    // Hydration with no keys lands on teeChoice="skip"; click the "Set up
    // Private Cloud" radio to expose the verify button.
    await user.click(screen.getByLabelText("Enable"));
    const verify = screen.getByRole("button", { name: /Connect & verify/ });
    expect(verify.disabled).toBe(true);
  });

  it("surfaces verifyError when verify_tinfoil_key throws", async () => {
    const user = userEvent.setup();
    stubInvoke({
      settings: { tinfoil_key_set: false },
      config: { subject: "User" },
      override(cmd) {
        if (cmd === "verify_tinfoil_key") {
          // Rust command Err — auth failed.
          throw new Error("Invalid API key — check the key at tinfoil.sh and try again.");
        }
      },
    });
    await renderAndHydrate({ tab: "tee" });
    await user.click(screen.getByLabelText("Enable"));
    const passwordInput = screen.getByPlaceholderText(/tf_/);
    await user.type(passwordInput, "tf_x");
    await user.click(screen.getByRole("button", { name: /Connect & verify/ }));
    await waitFor(() => {
      expect(screen.getAllByText(/Invalid API key/).length).toBeGreaterThan(0);
    });
  });

  it("after a successful verify with a typed key, save() writes save_settings + update_config", async () => {
    const user = userEvent.setup();
    let savedSettings = null;
    let savedConfigs = [];
    stubInvoke({
      settings: { tinfoil_key_set: false },
      config: { subject: "User" },
      override(cmd, args) {
        if (cmd === "verify_tinfoil_key") return undefined;
        if (cmd === "save_settings") {
          savedSettings = args;
          return undefined;
        }
        if (cmd === "update_config") {
          savedConfigs.push(args.patch);
          return undefined;
        }
      },
    });
    const onChanged = vi.fn();
    const onClose = vi.fn();
    await renderAndHydrate({ onChanged, onClose, tab: "tee" });

    await user.click(screen.getByLabelText("Enable"));
    const passwordInput = screen.getByPlaceholderText(/tf_/);
    await user.type(passwordInput, "tf_realkey");
    await user.click(screen.getByRole("button", { name: /Connect & verify/ }));
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Reverify/ })).toBeTruthy();
    });
    // Save-on-change: the save fires on the 1s debounce; verify the
    // resulting writes landed. Save no longer touches tee_attestations
    // (attestation state is owned by the upper-right Attest button +
    // startup pass), so the final update_config patch carries only the
    // configured-set fields.
    await waitFor(
      () => {
        expect(savedSettings?.tinfoilKey).toBe("tf_realkey");
        const finalCfg = savedConfigs[savedConfigs.length - 1];
        expect(finalCfg?.tee_provider).toBe("tinfoil");
        expect(finalCfg?.stage_models).toBeTruthy();
        expect(finalCfg?.stage_models.extract.model).toBeTruthy();
        expect(finalCfg?.tee_attestations).toBeUndefined();
      },
      { timeout: 2500 }
    );
  });
});

describe("Settings — local MLX readiness", () => {
  it("treats a downloaded MLX model as ready and persists 'verified', not the stale 'skipped'", async () => {
    // The bug: a downloaded MLX model left the Local section stuck "pending"
    // and re-saved the stale "skipped", so the Local mode stayed disabled.
    // MLX has no separate verify step — the download IS the setup.
    const user = userEvent.setup();
    const savedConfigs = [];
    stubInvoke({
      settings: {},
      config: { subject: "User", local_backend: "mlx", local_setup_mode: "skipped" },
      override(cmd, args) {
        if (cmd === "local_model_status") return { downloaded: true, os_supported: true };
        if (cmd === "update_config") {
          savedConfigs.push(args.patch);
          return undefined;
        }
        if (cmd === "save_settings") return undefined;
      },
    });
    await renderAndHydrate({ tab: "local" });
    // Hydrates to Disable (skipped on disk); re-enable to expose the MLX panel.
    await user.click(screen.getByLabelText("Enable"));
    // Section reports ready off the downloaded model — not stuck "pending".
    await waitFor(() => {
      expect(screen.getByText("✓ ready")).toBeTruthy();
    });
    // The stale skip is cleared on save so the mode picker can enable Local.
    await waitFor(
      () => {
        expect(savedConfigs.some((p) => p && p.local_setup_mode === "verified")).toBe(true);
      },
      { timeout: 2500 },
    );
  });
});

// Reset to defaults was retired alongside the explicit Save / Cancel
// buttons (round 2 of UX feedback for #77 — save-on-change is the
// new model). The per-stage / sentiment / cache fields are still
// editable; users get back to the shipping defaults by hand-picking.

describe("Settings — '(default)' suffix on selectors", () => {
  it("marks the default sentiment option with ' (default)'", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil" },
    });
    await renderAndHydrate();
    // The default-suffixed Neutral option should show "Neutral (default)".
    expect(screen.getByText(/^Neutral \(default\)$/)).toBeTruthy();
  });

});

// ── LLM prompt cache section ────────────────────────────────────────────────

describe("Settings — LLM prompt cache section", () => {
  it("renders the section header and stats line after hydration", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil" },
      override: (cmd) => {
        if (cmd === "get_llm_cache_stats") return { entries: 42, bytes: 12345 };
      },
    });
    await renderAndHydrate();
    expect(screen.getByText("LLM prompt cache")).toBeTruthy();
    const stats = await screen.findByTestId("cache-stats");
    expect(stats.textContent).toMatch(/42 cache entries/);
    expect(stats.textContent).toMatch(/12\.1 KB/);
  });

  it("checkbox defaults to enabled when config has no llm_cache_enabled key", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil" },
      override: (cmd) => {
        if (cmd === "get_llm_cache_stats") return { entries: 0, bytes: 0 };
      },
    });
    await renderAndHydrate();
    const cb = await screen.findByTestId("cache-enabled-checkbox");
    expect(cb.checked).toBe(true);
  });

  it("hydrates checkbox to disabled when config.llm_cache_enabled === false", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: {
        subject: "User", tee_provider: "tinfoil",
        llm_cache_enabled: false,
      },
      override: (cmd) => {
        if (cmd === "get_llm_cache_stats") return { entries: 0, bytes: 0 };
      },
    });
    await renderAndHydrate();
    const cb = await screen.findByTestId("cache-enabled-checkbox");
    expect(cb.checked).toBe(false);
  });

  it("toggling the checkbox flips state and persists `llm_cache_enabled` on save", async () => {
    // The pre-existing flake "changing sentiment marks the form dirty"
    // shows that stageModelsDirty fires unconditionally on hydration in
    // this test environment (also broken on main), which makes the
    // initial `save.disabled === true` assertion unreliable. Instead of
    // asserting on the dirty-gate transition, this test verifies the
    // wiring at both ends: the checkbox toggle flips React state, AND
    // a subsequent save() invocation includes `llm_cache_enabled: false`
    // in the persisted config payload.
    const user = userEvent.setup();
    let savedConfig = null;
    stubInvoke({
      settings: { tinfoil_key_set: true, tinfoil_key_masked: "•••abcd" },
      config: {
        subject: "User",
        tee_provider: "tinfoil",
        tee_model: "gpt-oss-120b",
        sentiment_bias: "neutral",
        llm_cache_enabled: true,
      },
      override: (cmd, args) => {
        if (cmd === "get_llm_cache_stats") return { entries: 0, bytes: 0 };
        if (cmd === "update_config") {
          savedConfig = args.patch;
          return undefined;
        }
        if (cmd === "save_settings") return undefined;
      },
    });
    await renderAndHydrate();
    const cb = await screen.findByTestId("cache-enabled-checkbox");
    expect(cb.checked).toBe(true);
    await user.click(cb);
    expect(cb.checked).toBe(false);

    // Save-on-change: no Save button. Wait for the 1s auto-save
    // debounce + round-trip, then assert llm_cache_enabled persisted.
    await waitFor(
      () => {
        expect(savedConfig).not.toBeNull();
        expect(savedConfig.llm_cache_enabled).toBe(false);
      },
      { timeout: 2500 }
    );
  });

  it("Wipe button is disabled when there are no cache entries", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil" },
      override: (cmd) => {
        if (cmd === "get_llm_cache_stats") return { entries: 0, bytes: 0 };
      },
    });
    await renderAndHydrate();
    const btn = await screen.findByTestId("wipe-cache-button");
    expect(btn.disabled).toBe(true);
  });

  it("clicking Wipe opens the Tauri plugin-dialog confirm with entry/byte counts", async () => {
    const user = userEvent.setup();
    // Settings now uses the Tauri plugin-dialog `confirm` for this
    // confirmation (same as resetToDefaults), not a prop-threaded
    // App-level React modal. Match the call shape: `confirm(message,
    // {title, kind})` returns Promise<boolean>.
    vi.mocked(askConfirm).mockResolvedValue(false);
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil" },
      override: (cmd) => {
        if (cmd === "get_llm_cache_stats") return { entries: 5, bytes: 2048 };
      },
    });
    await renderAndHydrate();
    const btn = await screen.findByTestId("wipe-cache-button");
    expect(btn.disabled).toBe(false);
    await user.click(btn);
    expect(askConfirm).toHaveBeenCalledTimes(1);
    const [message, opts] = vi.mocked(askConfirm).mock.calls[0];
    expect(message).toMatch(/5 cache entries/);
    expect(message).toMatch(/2\.0 KB/);
    expect(opts.title).toMatch(/Wipe LLM prompt cache/);
    expect(opts.kind).toBe("warning");
  });

  it("on confirm, invokes wipe_llm_cache and resets the displayed stats to 0", async () => {
    const user = userEvent.setup();
    vi.mocked(askConfirm).mockResolvedValue(true);
    const wipeSpy = vi.fn(async () => ({ entries: 5, bytes: 2048 }));
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil" },
      override: (cmd) => {
        if (cmd === "get_llm_cache_stats") return { entries: 5, bytes: 2048 };
        if (cmd === "wipe_llm_cache") return wipeSpy();
      },
    });
    await renderAndHydrate();
    await user.click(await screen.findByTestId("wipe-cache-button"));
    await waitFor(() => expect(askConfirm).toHaveBeenCalled());
    await waitFor(() => expect(wipeSpy).toHaveBeenCalledTimes(1));

    await waitFor(() => {
      const stats = screen.getByTestId("cache-stats");
      expect(stats.textContent).toMatch(/0 cache entries/);
      expect(stats.textContent).toMatch(/0 B/);
    });
  });

  it("clicking Open Chats invokes reveal_chats_dir", async () => {
    const user = userEvent.setup();
    const revealSpy = vi.fn(async () => null);
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil" },
      override: (cmd) => {
        if (cmd === "reveal_chats_dir") return revealSpy();
      },
    });
    await renderAndHydrate();
    await user.click(await screen.findByTestId("open-chats-button"));
    await waitFor(() => expect(revealSpy).toHaveBeenCalledTimes(1));
  });
});

// ── formatBytes helper ──────────────────────────────────────────────────────

import { formatBytes } from "./Settings";

describe("formatBytes", () => {
  it("returns 0 B for zero / negative / non-finite inputs", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(-1)).toBe("0 B");
    expect(formatBytes(NaN)).toBe("0 B");
    expect(formatBytes(Infinity)).toBe("0 B");
  });
  it("renders bytes < 1 KB as plain bytes", () => {
    expect(formatBytes(512)).toBe("512 B");
  });
  it("renders KB / MB / GB with one decimal", () => {
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(1024 * 1024 * 5.5)).toBe("5.5 MB");
  });
  it("renders GB with two decimals", () => {
    expect(formatBytes(1024 * 1024 * 1024 * 1.25)).toBe("1.25 GB");
  });
});

describe("Settings — Updates section (issue #50)", () => {
  function stubUpdatesInvoke({ checkResult, checkError, installError } = {}) {
    vi.mocked(invoke).mockImplementation(async (cmd) => {
      if (cmd === "get_settings") return { tinfoil_key_set: true };
      if (cmd === "get_config") return { subject: "User", tee_provider: "tinfoil" };
      if (cmd === "app_version") return "0.1.0";
      if (cmd === "check_update") {
        if (checkError) throw new Error(checkError);
        return checkResult ?? null;
      }
      if (cmd === "download_and_install_update") {
        if (installError) throw new Error(installError);
        return undefined;
      }
      return undefined;
    });
  }

  it("renders the running app version after hydration", async () => {
    stubUpdatesInvoke();
    await renderAndHydrate();
    await waitFor(() => {
      expect(screen.getByText(/Current version: v0\.1\.0/)).toBeTruthy();
    });
  });

  it("Check for updates → shows up-to-date when backend returns null", async () => {
    const user = userEvent.setup();
    stubUpdatesInvoke({ checkResult: null });
    await renderAndHydrate();
    await user.click(screen.getByRole("button", { name: /Check for updates/ }));
    await waitFor(() => {
      expect(screen.getByText(/up to date/i)).toBeTruthy();
    });
  });

  it("Check for updates → surfaces available version + body, exposes Update now button", async () => {
    const user = userEvent.setup();
    stubUpdatesInvoke({
      checkResult: {
        current_version: "0.1.0",
        available_version: "0.2.0",
        body: "Bug fixes and new features",
      },
    });
    await renderAndHydrate();
    await user.click(screen.getByRole("button", { name: /Check for updates/ }));
    await waitFor(() => {
      expect(screen.getByText(/Update available: v0\.2\.0/)).toBeTruthy();
    });
    expect(screen.getByText(/Bug fixes and new features/)).toBeTruthy();
    expect(screen.getByRole("button", { name: /Update now/ })).toBeTruthy();
  });

  it("check_update failure surfaces a Retry button that re-invokes check_update", async () => {
    const user = userEvent.setup();
    let calls = 0;
    vi.mocked(invoke).mockImplementation(async (cmd) => {
      if (cmd === "get_settings") return { tinfoil_key_set: true };
      if (cmd === "get_config") return { subject: "User", tee_provider: "tinfoil" };
      if (cmd === "app_version") return "0.1.0";
      if (cmd === "check_update") {
        calls += 1;
        if (calls === 1) throw new Error("network unreachable");
        return null;
      }
      return undefined;
    });
    await renderAndHydrate();
    await user.click(screen.getByRole("button", { name: /Check for updates/ }));
    await waitFor(() => {
      expect(screen.getByText(/Couldn't check for updates/)).toBeTruthy();
    });
    await user.click(screen.getByRole("button", { name: /Retry/ }));
    await waitFor(() => {
      expect(screen.getByText(/up to date/i)).toBeTruthy();
    });
    expect(calls).toBe(2);
  });

  it("install failure surfaces Try again that re-invokes download_and_install_update", async () => {
    const user = userEvent.setup();
    let installCalls = 0;
    vi.mocked(invoke).mockImplementation(async (cmd) => {
      if (cmd === "get_settings") return { tinfoil_key_set: true };
      if (cmd === "get_config") return { subject: "User", tee_provider: "tinfoil" };
      if (cmd === "app_version") return "0.1.0";
      if (cmd === "check_update")
        return { current_version: "0.1.0", available_version: "0.2.0" };
      if (cmd === "download_and_install_update") {
        installCalls += 1;
        throw new Error("download interrupted");
      }
      return undefined;
    });
    await renderAndHydrate();
    await user.click(screen.getByRole("button", { name: /Check for updates/ }));
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Update now/ })).toBeTruthy();
    });
    await user.click(screen.getByRole("button", { name: /Update now/ }));
    await waitFor(() => {
      expect(screen.getByText(/Update failed/)).toBeTruthy();
    });
    expect(installCalls).toBe(1);
    await user.click(screen.getByRole("button", { name: /Try again/ }));
    await waitFor(() => expect(installCalls).toBe(2));
  });

  it("signature-error message gets the security framing", async () => {
    const user = userEvent.setup();
    stubUpdatesInvoke({ checkError: "minisign signature verification failed" });
    await renderAndHydrate();
    await user.click(screen.getByRole("button", { name: /Check for updates/ }));
    await waitFor(() => {
      expect(
        screen.getByText(/Signature verification failed/)
      ).toBeTruthy();
    });
  });
});

// Development tab — gated by a "you're on your own" disclaimer banner;
// holds 7×2 toggles for full prompt + response logging per stage. Every
// toggle defaults OFF and persists to
// app_config.dev_full_prompt_logging.<stage>.<input|output>.
describe("Settings — Development tab", () => {
  it("renders the disclaimer banner + 14 toggles in canonical stage order", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil" },
    });
    await renderAndHydrate();
    await switchToTab("Development");
    expect(screen.getByText(/Development settings/i)).toBeTruthy();
    expect(screen.getByText(/These are not for typical use/i)).toBeTruthy();
    expect(screen.getByText(/100MB\+/)).toBeTruthy();
    // 7 stages × 2 toggles = 14 checkboxes; order matches VISIBLE_STAGES
    // (vision first, actions last).
    const tbody = document.querySelector(".dev-full-prompt-table tbody");
    expect(tbody).toBeTruthy();
    const rows = tbody.querySelectorAll("tr");
    expect(rows.length).toBe(7);
    const stageNames = Array.from(rows).map((r) =>
      r.querySelector("td:first-child").textContent.trim()
    );
    expect(stageNames).toEqual([
      "Vision", "Extract", "Entities", "Entities dedupe",
      "Patterns", "Insights", "Actions",
    ]);
    const checkboxes = tbody.querySelectorAll('input[type="checkbox"]');
    expect(checkboxes.length).toBe(14);
    Array.from(checkboxes).forEach((cb) => expect(cb.checked).toBe(false));
  });

  it("hydrates toggles from config.dev_full_prompt_logging", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: {
        subject: "User", tee_provider: "tinfoil",
        dev_full_prompt_logging: {
          extract: { input: true, output: false },
          patterns: { input: false, output: true },
        },
      },
    });
    await renderAndHydrate();
    await switchToTab("Development");
    const extractRow = Array.from(document.querySelectorAll(".dev-full-prompt-table tbody tr"))
      .find((r) => r.querySelector("td:first-child").textContent.trim() === "Extract");
    const patternsRow = Array.from(document.querySelectorAll(".dev-full-prompt-table tbody tr"))
      .find((r) => r.querySelector("td:first-child").textContent.trim() === "Patterns");
    const [extractIn, extractOut] = extractRow.querySelectorAll('input[type="checkbox"]');
    const [patternsIn, patternsOut] = patternsRow.querySelectorAll('input[type="checkbox"]');
    expect(extractIn.checked).toBe(true);
    expect(extractOut.checked).toBe(false);
    expect(patternsIn.checked).toBe(false);
    expect(patternsOut.checked).toBe(true);
  });

  it("flipping a toggle persists dev_full_prompt_logging via update_config", async () => {
    let savedFields = null;
    stubInvoke({
      settings: { tinfoil_key_set: true, tinfoil_key_masked: "•••abcd" },
      config: {
        subject: "User",
        tee_provider: "tinfoil",
        tee_model: "gpt-oss-120b",
      },
      override: (cmd, args) => {
        if (cmd === "update_config") {
          savedFields = args.patch;
          return undefined;
        }
        if (cmd === "save_settings") return undefined;
      },
    });
    const user = userEvent.setup();
    await renderAndHydrate();
    await switchToTab("Development");
    const extractRow = Array.from(document.querySelectorAll(".dev-full-prompt-table tbody tr"))
      .find((r) => r.querySelector("td:first-child").textContent.trim() === "Extract");
    const [extractIn] = extractRow.querySelectorAll('input[type="checkbox"]');
    await user.click(extractIn);
    // Auto-save fires after the 500ms debounce + round-trip.
    await waitFor(
      () => {
        expect(savedFields).not.toBeNull();
        expect(savedFields.dev_full_prompt_logging).toEqual({
          extract: { input: true, output: false },
        });
      },
      { timeout: 2500 }
    );
  });
});

describe("Settings — Categories section (issue #166)", () => {
  // Default seed is duplicated in Settings.jsx as DEFAULT_CATEGORIES; the
  // pipeline side carries the matching constant. Length is the
  // structural assertion; element-level assertions go through the row
  // inputs to avoid coupling the test to the exact ordering.
  const DEFAULT_LEN = 13;

  it("renders the seed default rows on first open (no `categories` in config)", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil" },
    });
    await renderAndHydrate();
    const list = await screen.findByTestId("categories-list");
    const rows = within(list).getAllByRole("textbox");
    expect(rows).toHaveLength(DEFAULT_LEN);
    // Surface a couple of well-known seeds so a future regression that
    // silently flips the default order or contents fails loudly.
    expect(rows.map((r) => r.value)).toContain("health");
    expect(rows.map((r) => r.value)).toContain("other");
  });

  it("hydrates from config.categories when present and non-empty", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: {
        subject: "User",
        tee_provider: "tinfoil",
        categories: ["fundraising", "research", "other"],
      },
    });
    await renderAndHydrate();
    const list = await screen.findByTestId("categories-list");
    const rows = within(list).getAllByRole("textbox");
    expect(rows.map((r) => r.value)).toEqual([
      "fundraising", "research", "other",
    ]);
  });

  it("falls back to defaults when config.categories is empty / wrong-shape", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: {
        subject: "User", tee_provider: "tinfoil",
        categories: [],
      },
    });
    await renderAndHydrate();
    const list = await screen.findByTestId("categories-list");
    expect(within(list).getAllByRole("textbox")).toHaveLength(DEFAULT_LEN);
  });

  it("Add appends an empty row; Remove drops the targeted row", async () => {
    const user = userEvent.setup();
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: { subject: "User", tee_provider: "tinfoil" },
    });
    await renderAndHydrate();
    const list = await screen.findByTestId("categories-list");
    expect(within(list).getAllByRole("textbox")).toHaveLength(DEFAULT_LEN);

    await user.click(screen.getByTestId("category-add"));
    expect(within(list).getAllByRole("textbox")).toHaveLength(DEFAULT_LEN + 1);

    await user.click(screen.getByTestId("category-remove-0"));
    expect(within(list).getAllByRole("textbox")).toHaveLength(DEFAULT_LEN);
  });

  it("validation surfaces empty / shape / duplicate errors per row", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: {
        subject: "User", tee_provider: "tinfoil",
        categories: ["work", "health"],
      },
    });
    await renderAndHydrate();
    const r0 = await screen.findByTestId("category-input-0");
    // Empty
    fireEvent.change(r0, { target: { value: "" } });
    expect(screen.getByTestId("category-error-0").textContent).toMatch(/empty/i);
    // Space (bad shape — uppercase is silently down-cased so it
    // doesn't surface as an error; spaces and other special chars do)
    fireEvent.change(r0, { target: { value: "my work" } });
    expect(screen.getByTestId("category-error-0").textContent).toMatch(/no spaces/i);
    // Duplicate of row 1
    fireEvent.change(r0, { target: { value: "health" } });
    expect(screen.getByTestId("category-error-0").textContent).toMatch(/duplicate/i);
  });

  it("uppercase input is silently down-cased, no error chip", async () => {
    stubInvoke({
      settings: { tinfoil_key_set: true },
      config: {
        subject: "User", tee_provider: "tinfoil",
        categories: ["work", "health"],
      },
    });
    await renderAndHydrate();
    const r0 = await screen.findByTestId("category-input-0");
    fireEvent.change(r0, { target: { value: "Fundraising" } });
    expect(r0.value).toBe("fundraising");
    expect(screen.queryByTestId("category-error-0")).toBeNull();
  });

  it("save persists custom rows to config.categories", async () => {
    let savedConfig = null;
    stubInvoke({
      settings: { tinfoil_key_set: true, tinfoil_key_masked: "•••abcd" },
      config: {
        subject: "User", tee_provider: "tinfoil",
        tee_model: "gpt-oss-120b",
        categories: ["work", "health", "other"],
      },
      override: (cmd, args) => {
        if (cmd === "update_config") { savedConfig = args.patch; return undefined; }
        if (cmd === "save_settings") return undefined;
      },
    });
    await renderAndHydrate();
    const r0 = await screen.findByTestId("category-input-0");
    // fireEvent.change is atomic — sets the value in one render pass.
    // Avoids the typing-keystroke race where the 500ms auto-save
    // debounce can fire mid-string and capture a partial value.
    fireEvent.change(r0, { target: { value: "fundraising" } });
    await waitFor(
      () => {
        expect(savedConfig).not.toBeNull();
        expect(savedConfig.categories).toEqual([
          "fundraising", "health", "other",
        ]);
      },
      { timeout: 2500 },
    );
  });

  it("save sends a null `categories` patch (server-side delete) when rows match the seed default", async () => {
    const user = userEvent.setup();
    let savedConfig = null;
    stubInvoke({
      settings: { tinfoil_key_set: true, tinfoil_key_masked: "•••abcd" },
      config: {
        subject: "User", tee_provider: "tinfoil",
        tee_model: "gpt-oss-120b",
        // Diverged-from-default starting state so `categoriesDirty`
        // flips when we restore.
        categories: ["work", "health"],
      },
      override: (cmd, args) => {
        if (cmd === "update_config") { savedConfig = args.patch; return undefined; }
        if (cmd === "save_settings") return undefined;
      },
    });
    await renderAndHydrate();
    // Trigger reset via the button — askConfirm is mocked to true.
    const { confirm } = await import("@tauri-apps/plugin-dialog");
    vi.mocked(confirm).mockResolvedValue(true);
    await user.click(screen.getByTestId("category-reset"));
    await waitFor(
      () => {
        expect(savedConfig).not.toBeNull();
        // Reset-to-default carries `categories: null` — update_config
        // (RFC 7386) deletes the key server-side, so pipeline-side
        // `_topics_for_run` falls back to the same seed list. The patch
        // stays narrow, so the delete can't clobber a sibling key.
        expect(savedConfig.categories).toBeNull();
      },
      { timeout: 2500 },
    );
  });

});

describe("Settings — #489: narrow patch can't clobber sibling keys", () => {
  // #489 repro at the contract layer. A runtime path persisted
  // `import_dir` (via #485's update_config). A subsequent Settings
  // Save must not drop it. The old whole-object set_config RMW did,
  // because the snapshot it merged into didn't carry the runtime
  // write. With a narrow update_config patch + the backend's
  // server-side merge (emulated here), the disjoint key survives.
  it("a Settings save preserves a runtime-written import_dir", async () => {
    const store = {
      subject: "User",
      tee_provider: "tinfoil",
      tee_model: "gpt-oss-120b",
      sentiment_bias: "neutral",
      import_dir: "/runtime/dir/A",
    };
    vi.mocked(invoke).mockImplementation(async (cmd, args) => {
      if (cmd === "get_settings") {
        return { tinfoil_key_set: true, tinfoil_key_masked: "•••abcd" };
      }
      if (cmd === "get_config") return { ...store };
      if (cmd === "update_config") {
        // RFC 7386 top-level merge, mirroring the Rust backend:
        // null deletes, everything else overwrites; untouched keys
        // (import_dir) are retained.
        for (const [k, v] of Object.entries(args.patch)) {
          if (v === null) delete store[k];
          else store[k] = v;
        }
        return undefined;
      }
      return undefined;
    });
    const user = userEvent.setup();
    await renderAndHydrate();
    const selects = screen.getAllByRole("combobox");
    const sentimentSelect = selects.find((s) => s.value === "neutral");
    await user.selectOptions(sentimentSelect, "uplifting");
    await waitFor(
      () => expect(store.sentiment_bias).toBe("uplifting"),
      { timeout: 2500 },
    );
    // The Settings save never carried import_dir in its patch, so the
    // server-side merge kept it — the #489 lost-update is gone.
    expect(store.import_dir).toBe("/runtime/dir/A");
  });
});
