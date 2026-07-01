import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { invoke } from "@tauri-apps/api/core";
import Wizard from "./Wizard";

afterEach(() => cleanup());

// Default backend stub: return empty settings + empty config so the wizard
// renders in pure first-run state. Individual tests override before render.
function stubInvoke(handler) {
  vi.mocked(invoke).mockImplementation(async (cmd, args) => {
    if (handler) {
      const r = handler(cmd, args);
      if (r !== undefined) return r;
    }
    if (cmd === "get_settings") return {};
    if (cmd === "get_config") return {};
    return undefined;
  });
}

async function renderAndHydrate(props = {}) {
  const utils = render(<Wizard onComplete={vi.fn()} onCancel={vi.fn()} {...props} />);
  // The wizard shows "Loading…" until both invoke calls resolve. Wait for
  // the title to appear.
  await waitFor(() => {
    expect(screen.queryByText(/Loading/)).toBeNull();
  });
  return utils;
}

describe("Wizard — first-run state", () => {
  it("renders the first-run title before any prior config exists", async () => {
    stubInvoke();
    await renderAndHydrate();
    expect(screen.getByText("Welcome to BaseVault")).toBeTruthy();
  });

  it("hides the Cancel button when allowCancel is false (first-run lock)", async () => {
    stubInvoke();
    await renderAndHydrate({ allowCancel: false });
    expect(screen.queryByRole("button", { name: /Cancel/i })).toBeNull();
  });

  it("shows the Cancel button when allowCancel is true (menu reopen)", async () => {
    stubInvoke();
    await renderAndHydrate({ allowCancel: true });
    expect(screen.getByRole("button", { name: /Cancel/i })).toBeTruthy();
  });

  it("Finish is disabled until at least the name is filled in", async () => {
    stubInvoke();
    await renderAndHydrate();
    const finish = screen.getByRole("button", { name: /Finish/ });
    expect(finish.disabled).toBe(true);
    expect(screen.getByText(/Enter your name to continue/)).toBeTruthy();
  });

  it("blocks finish when name is set but no mode (Local nor TEE) is configured", async () => {
    const user = userEvent.setup();
    stubInvoke();
    await renderAndHydrate();
    await user.type(screen.getByPlaceholderText(/e\.g\. Jane Smith/), "Alice");
    // Now skip local and TEE — gate should swap to the at-least-one-mode message.
    // Obsidian section was retired in #77 so there's no third skip click.
    await user.click(screen.getByLabelText(/Skip — don't use local mode/));
    await user.click(screen.getByLabelText(/Skip — use Local mode only/));
    const finish = screen.getByRole("button", { name: /Finish/ });
    expect(finish.disabled).toBe(true);
    expect(
      screen.getByText(/at least one of Local or Private Cloud/)
    ).toBeTruthy();
  });

  it("allows finish on a keyed build via the bundled key (name + skip local, no user key)", async () => {
    const user = userEvent.setup();
    // Keyed build, no user key, no local model. The bundled key makes
    // Private Cloud usable, so the TEE section counts as configured and
    // Finish must enable — the regular wizard reached via the menu on a
    // keyed build must not be a dead end.
    stubInvoke((cmd) => {
      if (cmd === "get_settings") return { bundled_key_set: true };
    });
    await renderAndHydrate();
    await user.type(screen.getByPlaceholderText(/e\.g\. Jane Smith/), "Alice");
    // Skip local; leave TEE on its default Setup (no user key entered).
    await user.click(screen.getByLabelText(/Skip — don't use local mode/));
    const finish = screen.getByRole("button", { name: /Finish/ });
    expect(finish.disabled).toBe(false);
    expect(
      screen.queryByText(/at least one of Local or Private Cloud/)
    ).toBeNull();
  });
});

describe("Wizard — TEE verify failure modes", () => {
  it("blocks verify with a friendly message when no key is entered + nothing on file", async () => {
    const user = userEvent.setup();
    stubInvoke();
    await renderAndHydrate();
    // TEE section is rendered (Setup is the default). Click Connect & verify
    // with an empty input.
    const verifyBtn = screen.getByRole("button", { name: /Connect & verify/ });
    // Button is disabled because there's no typed key and no key on file.
    expect(verifyBtn.disabled).toBe(true);
  });

  it("calls verify_tinfoil_key and surfaces the error on failure", async () => {
    const user = userEvent.setup();
    stubInvoke((cmd) => {
      if (cmd === "verify_tinfoil_key") {
        // Rust command Err on bad key.
        throw new Error("Invalid API key — check the key at tinfoil.sh and try again.");
      }
    });
    await renderAndHydrate();
    const passwordInput = screen.getByPlaceholderText(/tf_/);
    await user.type(passwordInput, "tf_fakekey_12345");
    await user.click(screen.getByRole("button", { name: /Connect & verify/ }));
    await waitFor(() => {
      expect(screen.getAllByText(/Invalid API key/).length).toBeGreaterThan(0);
    });
    expect(
      screen.getByRole("button", { name: /Connect & verify/ })
    ).toBeTruthy();
  });

  it("surfaces an unexpected backend exception as the error string", async () => {
    const user = userEvent.setup();
    stubInvoke((cmd) => {
      if (cmd === "verify_tinfoil_key") {
        throw new Error("python sidecar crashed");
      }
    });
    await renderAndHydrate();
    await user.type(screen.getByPlaceholderText(/tf_/), "tf_x");
    await user.click(screen.getByRole("button", { name: /Connect & verify/ }));
    await waitFor(() => {
      expect(
        screen.getAllByText(/python sidecar crashed/).length
      ).toBeGreaterThan(0);
    });
  });

  it("a successful key probe flips the button label to Reverify (no attestation persistence)", async () => {
    const user = userEvent.setup();
    let savedConfig = null;
    stubInvoke((cmd, args) => {
      if (cmd === "verify_tinfoil_key") return undefined;
      if (cmd === "update_config") {
        savedConfig = args.patch;
        return undefined;
      }
    });
    await renderAndHydrate();
    await user.type(screen.getByPlaceholderText(/tf_/), "tf_real");
    await user.click(screen.getByRole("button", { name: /Connect & verify/ }));
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Reverify/ })).toBeTruthy();
    });
    // Key-validity flow no longer writes to config; attestation state
    // is owned by the upper-right Attest button + startup pass.
    expect(savedConfig?.tee_attestations).toBeUndefined();
  });
});

describe("Wizard — cancel flow", () => {
  it("Escape triggers onCancel when allowCancel=true", async () => {
    stubInvoke();
    const onCancel = vi.fn();
    await renderAndHydrate({ allowCancel: true, onCancel });
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    });
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("Escape is ignored when allowCancel=false (first-run lock)", async () => {
    stubInvoke();
    const onCancel = vi.fn();
    await renderAndHydrate({ allowCancel: false, onCancel });
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    });
    expect(onCancel).not.toHaveBeenCalled();
  });

  it("clicking Cancel button calls onCancel", async () => {
    const user = userEvent.setup();
    stubInvoke();
    const onCancel = vi.fn();
    await renderAndHydrate({ allowCancel: true, onCancel });
    await user.click(screen.getByRole("button", { name: /Cancel/i }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});

describe("Wizard — re-run hydration", () => {
  it("renders the re-run title when prior config has a subject", async () => {
    stubInvoke((cmd) => {
      if (cmd === "get_config")
        return { subject: "Existing User", obsidian_vault_name: "MyVault" };
      if (cmd === "get_settings") return { tinfoil_key_set: false };
    });
    await renderAndHydrate({ allowCancel: true });
    expect(screen.getByText("Onboarding Wizard")).toBeTruthy();
    expect(
      screen.queryByText("Welcome to BaseVault")
    ).toBeNull();
  });

  it("hydrates the name from prior config", async () => {
    stubInvoke((cmd) => {
      if (cmd === "get_config") return { subject: "Existing User" };
    });
    await renderAndHydrate({ allowCancel: true });
    const nameInput = screen.getByPlaceholderText(/e\.g\. Jane Smith/);
    expect(nameInput.value).toBe("Existing User");
  });

  it("hydrates the on-file Tinfoil key + masked display", async () => {
    stubInvoke((cmd) => {
      if (cmd === "get_settings")
        return {
          tinfoil_key_set: true,
          tinfoil_key_masked: "••••••••wxyz",
        };
      if (cmd === "get_config")
        return { subject: "U", tee_provider: "tinfoil" };
    });
    await renderAndHydrate({ allowCancel: true });
    expect(screen.getByText("••••••••wxyz")).toBeTruthy();
    // Clear button visible alongside masked key
    expect(screen.getByRole("button", { name: /Clear/ })).toBeTruthy();
  });
});

describe("Wizard — finish flow", () => {
  it("saves settings + config and calls onComplete with the resulting mode", async () => {
    const user = userEvent.setup();
    let savedSettings = null;
    let savedConfigs = [];
    stubInvoke((cmd, args) => {
      if (cmd === "verify_attestation") {
        return {
          provider: args.provider,
          model: args.model,
          ok: true,
          fingerprint: "fp",
          error: null,
          ts: 1700000000,
          constituents: [],
        };
      }
      if (cmd === "save_settings") {
        savedSettings = args;
        return undefined;
      }
      if (cmd === "update_config") {
        savedConfigs.push(args.patch);
        return undefined;
      }
    });
    const onComplete = vi.fn();
    await renderAndHydrate({ onComplete });

    await user.type(screen.getByPlaceholderText(/e\.g\. Jane Smith/), "Alice");
    await user.click(screen.getByLabelText(/Skip — don't use local mode/));
    await user.type(screen.getByPlaceholderText(/tf_/), "tf_realkey");
    await user.click(screen.getByRole("button", { name: /Connect & verify/ }));
    // Wait for verify success state
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Reverify/ })).toBeTruthy();
    });
    // Obsidian section retired in #77 — no third skip click needed.

    const finish = screen.getByRole("button", { name: /Finish/ });
    await waitFor(() => expect(finish.disabled).toBe(false));
    await user.click(finish);

    await waitFor(() => {
      expect(onComplete).toHaveBeenCalledWith("tee");
    });
    expect(savedSettings.tinfoilKey).toBe("tf_realkey");
    // Last update_config write is the finish patch. It carries subject
    // + mode + the configured-set fields — a narrow per-key patch, not
    // a whole-object snapshot, so it can't clobber unrelated keys.
    const finalCfg = savedConfigs[savedConfigs.length - 1];
    expect(finalCfg.subject).toBe("Alice");
    expect(finalCfg.tee_provider).toBe("tinfoil");
    expect(finalCfg.mode).toBe("tee");
    expect(finalCfg.local_setup_mode).toBe("skipped");
  });
});

describe("Wizard — Easy (name-only) mode (#609)", () => {
  it("collapses to a single name field with no model/key/cloud wording", async () => {
    stubInvoke();
    await renderAndHydrate({ easy: true, allowCancel: true });

    expect(screen.getByText("Welcome to BaseVault")).toBeTruthy();
    expect(screen.getByPlaceholderText(/e\.g\. Jane Smith/)).toBeTruthy();
    // None of the full-wizard sections — and crucially zero
    // cloud-shaped wording (#606).
    expect(screen.queryByText(/Local processing/)).toBeNull();
    expect(screen.queryByText(/Private Cloud/)).toBeNull();
    expect(screen.queryByText(/Tinfoil/i)).toBeNull();
    expect(screen.queryByText(/API key/i)).toBeNull();
    expect(screen.queryByPlaceholderText(/tf_/)).toBeNull();
  });

  it("Finish persists name + Private Cloud and never calls save_settings", async () => {
    const user = userEvent.setup();
    const savedConfigs = [];
    let saveSettingsCalled = false;
    stubInvoke((cmd, args) => {
      if (cmd === "update_config") {
        savedConfigs.push(args.patch);
        return undefined;
      }
      if (cmd === "save_settings") {
        saveSettingsCalled = true;
        return undefined;
      }
    });
    const onComplete = vi.fn();
    await renderAndHydrate({ easy: true, allowCancel: true, onComplete });

    const finish = screen.getByRole("button", { name: /Finish/ });
    expect(finish.disabled).toBe(true); // name required
    await user.type(screen.getByPlaceholderText(/e\.g\. Jane Smith/), "Alice");
    await waitFor(() => expect(finish.disabled).toBe(false));
    await user.click(finish);

    await waitFor(() => expect(onComplete).toHaveBeenCalledWith("tee"));
    // No key is ever collected on the Easy path — the bundled key
    // powers Private Cloud, so save_settings must not be touched.
    expect(saveSettingsCalled).toBe(false);
    const finalCfg = savedConfigs[savedConfigs.length - 1];
    expect(finalCfg.subject).toBe("Alice");
    expect(finalCfg.mode).toBe("tee");
    expect(finalCfg.tee_provider).toBe("tinfoil");
    // Easy is purely "name in, ready" — it must not touch local_*.
    expect(finalCfg.local_setup_mode).toBeUndefined();
  });
});
