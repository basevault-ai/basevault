import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, within } from "@testing-library/react";
import { act } from "react";
import userEvent from "@testing-library/user-event";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import App, {
  _resetFrozenAbortedCacheForTests,
  findSegmentForward,
  fmtElapsed,
  formatEta,
  formatSizeShort,
  freezeAbortedCallDurations,
  liveElapsedMs,
  MAX_SEARCH_MATCHES,
  normalizeNavRelPath,
  progressChipPct,
  resolveCitation,
  resourceNavTarget,
} from "./App";

afterEach(() => cleanup());

// App.jsx is the top-level mount and fires several invoke calls on bootstrap:
//   needs_wizard, get_settings, get_config, list_runs, verify_attestation
// Rather than hand-coding every reply per test, this stub provides defaults
// that yield a "configured user, main UI visible, no active runs" baseline.
// Tests override individual commands via the `override` callback.
function stubInvoke({
  needsWizard = false,
  settings = { tinfoil_key_set: true },
  config = {
    subject: "User",
    tee_provider: "tinfoil",
    tee_model: "gpt-oss-120b",
    inputs: [],
    mode: "tee",
  },
  runs = [],
  attestation = {
    provider: "tinfoil",
    model: "gpt-oss-120b",
    ok: true,
    fingerprint: "fp",
    error: null,
    ts: 1700000000,
    constituents: [],
  },
  override,
} = {}) {
  vi.mocked(invoke).mockImplementation(async (cmd, args) => {
    if (override) {
      const r = override(cmd, args);
      if (r !== undefined) return r;
    }
    if (cmd === "needs_wizard") return needsWizard;
    if (cmd === "get_settings") return settings;
    if (cmd === "get_config") return config;
    if (cmd === "list_runs") return runs;
    if (cmd === "verify_attestation") return attestation;
    if (cmd === "set_config") return undefined;
    if (cmd === "update_config") return undefined;
    if (cmd === "expand_paths") return [];
    return undefined;
  });
}

async function renderApp(opts) {
  stubInvoke(opts);
  const utils = render(<App />);
  // Bootstrap effects fire on mount; wait for the configLoaded gate to flip
  // by waiting on a stable render anchor (the pipeline-runs pane title).
  await waitFor(() => {
    expect(screen.getByText("Pipeline runs")).toBeTruthy();
  });
  return utils;
}

describe("App — bootstrap + wizard gating", () => {
  it("queries needs_wizard at startup and surfaces the wizard when true", async () => {
    stubInvoke({
      needsWizard: true,
      config: {},
      settings: { tinfoil_key_set: false },
    });
    render(<App />);
    // Bootstrap calls needs_wizard. The full Wizard internals are covered
    // separately in Wizard.test.jsx; here we just verify App's gating
    // wires up correctly: when needs_wizard=true, the modal-backdrop class
    // appears (the Wizard's outermost element) and the main run button
    // is hidden behind it. Anchor on the modal class to avoid coupling
    // to the Wizard's own loading-vs-loaded text states.
    await waitFor(
      () => {
        expect(invoke).toHaveBeenCalledWith("needs_wizard");
        expect(document.querySelector(".wizard-modal")).toBeTruthy();
      },
      { timeout: 3000 }
    );
  });

  it("renders the main UI when needs_wizard=false", async () => {
    await renderApp();
    expect(screen.getByText("Pipeline runs")).toBeTruthy();
    // No wizard modal
    expect(screen.queryByText("Welcome to BaseVault")).toBeNull();
  });

  it("first run on a keyed build (bundled key present) opens the name-only Easy Wizard", async () => {
    // The whole point of the bundled key: a fresh user gets the
    // friction-free name-only flow, never the cloud-shaped regular
    // wizard. Anchor on the Easy modal's distinct class + the absence
    // of the regular wizard's Local section.
    stubInvoke({
      needsWizard: true,
      config: {},
      settings: { tinfoil_key_set: false, bundled_key_set: true },
    });
    render(<App />);
    await waitFor(
      () => {
        expect(document.querySelector(".wizard-modal-easy")).toBeTruthy();
      },
      { timeout: 3000 }
    );
    expect(screen.queryByText("Local processing")).toBeNull();
  });

  it("first run on a keyless build (no bundled key) opens the regular wizard", async () => {
    // No bundled key → the Easy path would be a dead end, so first run
    // falls back to the regular key-asking wizard (unchanged behavior).
    stubInvoke({
      needsWizard: true,
      config: {},
      settings: { tinfoil_key_set: false, bundled_key_set: false },
    });
    render(<App />);
    await waitFor(
      () => {
        expect(screen.getByText("Local processing")).toBeTruthy();
      },
      { timeout: 3000 }
    );
    expect(document.querySelector(".wizard-modal-easy")).toBeNull();
  });

  it("fires verify_attestation on mount as the launch-prewarm trigger (issue #48)", async () => {
    // The prewarm contract for cloud-mode launch latency: App's
    // bootstrap useEffect MUST call verify_attestation on mount,
    // before any user interaction. The Tauri-side script then walks
    // every backend model in stage_models and writes successful
    // attestations to the on-disk cache (`attestation.py:_ok`); the
    // pipeline runner subprocess later hits that warm cache instead
    // of paying the ~2s/model TUF + Sigstore cost on Run-click.
    //
    // This test would catch a regression where someone removes
    // refreshAttestation() from the bootstrap useEffect (or moves
    // it behind a guard), which would silently push the latency
    // back to the 4.5s/click profile from the ehem run.
    await renderApp();
    const attestCalls = vi
      .mocked(invoke)
      .mock.calls.filter((c) => c[0] === "verify_attestation");
    expect(attestCalls.length).toBeGreaterThanOrEqual(1);
  });
});

describe("App — run history list", () => {
  it("shows the empty-state message when list_runs returns no runs", async () => {
    await renderApp({ runs: [] });
    expect(screen.getByText(/No runs yet/i)).toBeTruthy();
  });

  // Bar fill width = round(displayDone / total * 100). Issue #209
  // pinned this as the source of truth for both the bar and the chip
  // text — they're the same fraction stated two ways. `bar_position`
  // (computed by the python tracker) is now only a fallback for the
  // pre-tick window where `total === 0`; once total lands, the
  // rust-derived leaf count drives both. Wall-clock progression
  // between ticks does not advance the fill.
  it("bar fill width = displayDone / total", async () => {
    const FAKE_NOW = Date.parse("2026-04-29T12:00:30Z");
    const realNow = Date.now;
    Date.now = () => FAKE_NOW;
    try {
      await renderApp({
        runs: [
          {
            run_id: "2026-04-29T12-00-00Z-live",
            short_id: "live",
            status: "running",
            mode: "tee",
            inputs: ["/x.md"],
            input_count: 1,
            created_at: "2026-04-29T12:00:00Z",
            progress: {
              stage: "insights",
              completed: 25,
              total: 50,
              // bar_position carried in the payload but no longer
              // drives pct — the fill comes from completed/total.
              bar_position: 0.42,
              eta_seconds: 60,
            },
          },
        ],
      });
      const fill = document.querySelector(".progress-fill");
      expect(fill).toBeTruthy();
      const widthPct = parseFloat(fill.style.width);
      expect(widthPct).toBe(50);
    } finally {
      Date.now = realNow;
    }
  });

  it("renders one row per run with the input title and status badge", async () => {
    await renderApp({
      runs: [
        {
          run_id: "2026-04-29T12-00-00Z-abcd",
          short_id: "abcd",
          status: "completed",
          mode: "tee",
          inputs: ["/Users/u/Documents/notes.md"],
          input_count: 1,
          created_at: "2026-04-29T12:00:00Z",
          duration_ms: 30_000,
          progress: { stage: "done", completed: 50, total: 50 },
          provider: "tinfoil",
          model: "gpt-oss-120b",
          vault_exists: true,
        },
        {
          run_id: "2026-04-29T11-00-00Z-efgh",
          short_id: "efgh",
          status: "failed",
          mode: "local",
          inputs: ["/a/b.txt", "/a/c.txt", "/a/d.txt"],
          input_count: 3,
          created_at: "2026-04-29T11:00:00Z",
          error: "ollama not running\nstack trace …",
        },
      ],
    });
    // Run 1: single input → title is the basename minus the trailing .md
    expect(screen.getByText("notes")).toBeTruthy();
    expect(screen.getByText("#abcd")).toBeTruthy();
    expect(screen.getByText("Completed")).toBeTruthy();
    // Run 2: multiple inputs → "first +N more" pattern
    expect(screen.getByText("b.txt +2 more")).toBeTruthy();
    expect(screen.getByText("Failed")).toBeTruthy();
    // Error first line surfaces; multi-line tail does not (only first line
    // shown in the row, full text is tooltip).
    expect(screen.getByText("ollama not running")).toBeTruthy();
    expect(screen.queryByText(/stack trace/)).toBeNull();
  });
});

describe("App — attaching to an active run", () => {
  it("a running row renders pause + cancel buttons (delete moved to bottom bar)", async () => {
    await renderApp({
      runs: [
        {
          run_id: "2026-04-29T12-00-00Z-runn",
          short_id: "runn",
          status: "running",
          mode: "tee",
          inputs: ["/x/y.md"],
          input_count: 1,
          created_at: "2026-04-29T12:00:00Z",
          progress: { stage: "extract", completed: 10, total: 50 },
        },
      ],
    });
    // Pause + Cancel are the per-row action buttons on a running row.
    // Delete is no longer per-row — bulk Delete (N) lives in the
    // sticky bottom bar, gated on selection size.
    expect(screen.getByTitle("Pause")).toBeTruthy();
    expect(screen.getByTitle("Cancel")).toBeTruthy();
    expect(screen.queryByTitle("Delete")).toBeNull();
    // Stage label rendered from STAGE_LABELS
    expect(screen.getByText(/Extracting facts/)).toBeTruthy();
    // Counts ("X% completed (N / M estimated calls)") sit on their
    // own DOM node (.progress-counts). pct and the call ratio share
    // a denominator: bar_position is `completed / total` on the
    // runner side, so the parens reading is the same fact stated
    // two ways. The denominator is the runner's est_calls sum (a
    // moving target — refines as stages register / re-estimate /
    // halve), so the label reads "estimated calls" to make that
    // explicit.
    const counts = document.querySelector(".progress-counts");
    expect(counts).toBeTruthy();
    expect(counts.textContent).toBe("20% completed (10 / 50 estimated calls)");
  });

  it("clicking Pause invokes pause_run with the run_id", async () => {
    const user = userEvent.setup();
    let pauseCalled = null;
    await renderApp({
      runs: [
        {
          run_id: "2026-04-29T12-00-00Z-runn",
          short_id: "runn",
          status: "running",
          mode: "tee",
          inputs: ["/x/y.md"],
          input_count: 1,
          created_at: "2026-04-29T12:00:00Z",
          progress: { stage: "extract", completed: 10, total: 50 },
        },
      ],
      override(cmd, args) {
        if (cmd === "pause_run") {
          pauseCalled = args;
          return undefined;
        }
      },
    });
    await user.click(screen.getByTitle("Pause"));
    expect(pauseCalled).toEqual({ runId: "2026-04-29T12-00-00Z-runn" });
  });

  it("running detail block stacks: stage line, bar, counts line, elapsed line (issue #161)", async () => {
    // Issue #161 split the old single detail line ("<stage> · X% · ETA")
    // into three rendered elements: stage name above the bar, then two
    // stacked lines under it — counts on top, elapsed/ETA below. Each
    // datum gets its own DOM node so narrow sidebars wrap gracefully.
    await renderApp({
      runs: [
        {
          run_id: "2026-04-29T12-00-00Z-runn",
          short_id: "runn",
          status: "running",
          mode: "tee",
          inputs: ["/x/y.md"],
          input_count: 1,
          created_at: "2026-04-29T12:00:00Z",
          progress: {
            stage: "entities",
            completed: 55,
            total: 130,
            in_flight_calls: 4,
            // bar_position no longer drives pct (issue #209) — pct
            // derives from completed/total to keep bar and chip
            // agreeing by construction. 55/130 ≈ 42%.
            bar_position: 0.42,
            eta_seconds: 220,
            stage_eta_seconds: 70,
            elapsed_in_stage: 30,
          },
          provider: "tinfoil",
        },
      ],
    });
    // Stage label sits on its own line above the bar.
    const stageNode = document.querySelector(".progress-stage");
    expect(stageNode).toBeTruthy();
    expect(stageNode.textContent).toBe("Resolving entities");
    // Counts line — exact match (no trailing ETA jammed in).
    const counts = document.querySelector(".progress-counts");
    expect(counts).toBeTruthy();
    expect(counts.textContent).toBe("42% completed (55 / 130 estimated calls)");
    // Elapsed line — separate node, ETA in parens.
    const elapsed = document.querySelector(".progress-elapsed");
    expect(elapsed).toBeTruthy();
    expect(elapsed.textContent).toMatch(/^Elapsed \S.* \(~3m 40s remaining\)$/);
  });

  // Issue #209 invariant: the chip renders one string ("X% completed
  // (Y / Z calls)") and X must always equal round(Y/Z*100). This used
  // to break — pre-#209 X came from python's bar_position and Y/Z from
  // a different source, producing strings like "42% completed (66 /
  // 130)" where 66/130 = 50.77% (and an existing test snapshotted that
  // lie). The parser below extracts X, Y, Z directly from the rendered
  // DOM and computes round(Y/Z*100) — if the bar/chip math ever
  // diverges again, this test fails loudly. DO NOT replace these
  // assertions with a hardcoded `expect(...).toBe("42% completed
  // (55 / 130 calls)")` snapshot — that's how the lie got locked in
  // last time.
  describe("issue #209 invariant: chip pct === round(Y/Z*100)", () => {
    function parseChip(text) {
      // "42% completed (55 / 130 estimated calls)"
      // "42% completed (55 / 130 estimated calls, +12 retries)"
      // "100% completed (50 / 50 estimated calls)"
      const m = text.match(
        /^(\d+)% completed \((\d+) \/ (\d+) estimated calls(?:, \+(\d+) retries)?\)$/,
      );
      if (!m) {
        throw new Error(`chip text doesn't match expected format: "${text}"`);
      }
      return {
        pct: parseInt(m[1], 10),
        done: parseInt(m[2], 10),
        total: parseInt(m[3], 10),
        retries: m[4] ? parseInt(m[4], 10) : 0,
      };
    }

    // Assert: the X% in the chip text is exactly what the formula
    // would produce given the (Y, Z) also in the chip text. Allows
    // the [0, 99]-clamp + 100-on-done semantics.
    function assertInvariant({ pct, done, total }, { isDone }) {
      if (isDone) {
        expect(pct).toBe(100);
        return;
      }
      const expected =
        done >= total ? 99 : Math.min(99, Math.round((done / total) * 100));
      expect(pct).toBe(expected);
    }

    // Combinations chosen to cover: small/large, retry-storm with
    // displayDone clamped (raw done > total), edge of clamp (done ===
    // total mid-run), zero-retry baseline, with-retries surface.
    const cases = [
      { label: "no retries, mid-run", completed: 10, total: 50, retries: 0 },
      { label: "no retries, near-done", completed: 49, total: 50, retries: 0 },
      { label: "fractional, rounds down", completed: 55, total: 130, retries: 0 },
      { label: "fractional, rounds up", completed: 56, total: 130, retries: 0 },
      { label: "1/1 mid-run pegs to 99", completed: 1, total: 1, retries: 0 },
      { label: "with retries surfaced", completed: 22, total: 42, retries: 57 },
      { label: "pd5w shape", completed: 43, total: 43, retries: 57 },
      // Race window: rust-derived done outpaced total briefly. The
      // displayDone clamp pins to total; the chip must STILL be self-
      // consistent (pct from clamped done, not raw).
      { label: "raw done > total (clamp)", completed: 100, total: 42, retries: 58 },
    ];
    for (const c of cases) {
      it(`running: ${c.label} (${c.completed}/${c.total}, +${c.retries})`, async () => {
        await renderApp({
          runs: [
            {
              run_id: `2026-04-29T12-00-00Z-${c.label.replace(/\s+/g, "")}`,
              short_id: "test",
              status: "running",
              mode: "tee",
              inputs: ["/x.md"],
              input_count: 1,
              created_at: "2026-04-29T12:00:00Z",
              progress: {
                stage: "extract",
                completed: c.completed,
                total: c.total,
                retries: c.retries,
                // Set bar_position to a deliberately-wrong value to
                // prove pct doesn't sneak back in from this field
                // when total > 0. If a regression makes bar_position
                // drive pct again, the assertion below will fail
                // (0.123 → 12%, none of the chip Y/Z values match).
                bar_position: 0.123,
              },
            },
          ],
        });
        const counts = document.querySelector(".progress-counts");
        expect(counts).toBeTruthy();
        const parsed = parseChip(counts.textContent);
        // displayDone-clamp is part of the invariant: even with raw
        // done > total upstream, the chip's Y must be ≤ Z.
        expect(parsed.done).toBeLessThanOrEqual(parsed.total);
        // Retries field round-trips exactly (no clamp).
        expect(parsed.retries).toBe(c.retries);
        // The hard rule.
        assertInvariant(parsed, { isDone: false });
      });
    }

    // Bar fill width must use the same pct, so the green fill agrees
    // with the chip text by construction.
    it("bar fill width === chip pct (same fraction, two renders)", async () => {
      await renderApp({
        runs: [
          {
            run_id: "2026-04-29T12-00-00Z-bar",
            short_id: "bar",
            status: "running",
            mode: "tee",
            inputs: ["/x.md"],
            input_count: 1,
            created_at: "2026-04-29T12:00:00Z",
            progress: {
              stage: "extract",
              completed: 17,
              total: 60,
              bar_position: 0.999, // adversarial
            },
          },
        ],
      });
      const counts = document.querySelector(".progress-counts");
      const fill = document.querySelector(".progress-fill");
      const { pct } = parseChip(counts.textContent);
      const fillPct = parseFloat(fill.style.width);
      expect(fillPct).toBe(pct);
      // Sanity: 17/60 = 28.33% → 28%. NOT 99% (what bar_position
      // would have produced pre-fix).
      expect(pct).toBe(28);
    });
  });

  // Issue #209: progressChipPct is the single source of the chip's
  // percentage formula. Both the chip text and the bar fill width
  // must call it (or render the same value). Direct unit tests on the
  // helper give surgical coverage of the formula independent of the
  // DOM render.
  describe("progressChipPct (issue #209 helper)", () => {
    it("returns 100 when status === 'completed'", () => {
      expect(
        progressChipPct({
          status: "completed",
          stage: "actions",
          displayDone: 5,
          total: 10,
        }),
      ).toBe(100);
    });
    it("returns 100 when stage === 'done' even if status still 'running'", () => {
      // Runner emits stage=done on the final progress_tick before
      // the cycle_end terminator lands; status flips a beat later.
      expect(
        progressChipPct({
          status: "running",
          stage: "done",
          displayDone: 50,
          total: 50,
        }),
      ).toBe(100);
    });
    it("uses displayDone/total when total > 0", () => {
      // Pin the formula: round(22 / 42 * 100) = 52.
      expect(
        progressChipPct({
          status: "running",
          stage: "extract",
          displayDone: 22,
          total: 42,
        }),
      ).toBe(52);
    });
    it("clamps to 99 mid-run even when displayDone === total", () => {
      // Bar can't visually "complete" until the runner emits done.
      expect(
        progressChipPct({
          status: "running",
          stage: "extract",
          displayDone: 50,
          total: 50,
        }),
      ).toBe(99);
    });
    it("returns 0 when total === 0 (pre-tick window)", () => {
      // Before the first progress_tick lands, total is unknown.
      // Issue #209 cleanup removed the bar_position fallback entirely
      // — the bar shows 0% briefly at run start. There's no longer
      // any way to source pct from a separate field; the only inputs
      // are displayDone and total.
      expect(
        progressChipPct({
          status: "running",
          stage: "init",
          displayDone: 0,
          total: 0,
        }),
      ).toBe(0);
    });
    it("ignores extra/unknown fields in the input shape", () => {
      // Defensive: a stale bar_position sneaking back into the call
      // site won't influence pct. The helper signature is closed —
      // only status, stage, displayDone, total are read.
      expect(
        progressChipPct({
          status: "running",
          stage: "extract",
          displayDone: 22,
          total: 42,
          // These are deliberate no-ops; helper must not read them.
          barPosition: 0.999,
          bar_position: 0.999,
          completed: 999,
        }),
      ).toBe(52);
    });
  });

  it("formatEta rolls minutes correctly when seconds quantize to 60s (round 4 fix)", async () => {
    // The round-3 implementation had a split-then-quantize bug:
    // floor(356/60) = 5, then round((356 - 300)/10)*10 = 60 → "5m 60s".
    // Round 4 quantizes total first, then splits, so 356 → 360 → "6m".
    await renderApp({
      runs: [
        {
          run_id: "2026-04-29T12-00-00Z-rne1",
          short_id: "rne1",
          status: "running",
          mode: "tee",
          inputs: ["/x/y.md"],
          input_count: 1,
          created_at: "2026-04-29T12:00:00Z",
          progress: {
            stage: "extract",
            completed: 1,
            total: 100,
            bar_position: 0.05,
            eta_seconds: 356,
          },
        },
      ],
    });
    expect(screen.getByText(/~6m remaining/)).toBeTruthy();
    expect(screen.queryByText(/5m 60s/)).toBeNull();
    expect(screen.queryByText(/5min 60s/)).toBeNull();
  });

  it("run header shows '<files> — <in-flight> in progress' while running", async () => {
    // Round 3: run-meta combines file count + in-flight call count
    // on one line. "in progress" suffix only appears while running
    // (in_flight is meaningless for a done run).
    await renderApp({
      runs: [
        {
          run_id: "2026-04-29T12-00-00Z-runn",
          short_id: "runn",
          status: "running",
          mode: "tee",
          inputs: ["/x/y.md"],
          input_count: 1,
          created_at: "2026-04-29T12:00:00Z",
          progress: {
            stage: "extract",
            completed: 50,
            total: 130,
            in_flight_calls: 5,
            bar_position: 0.32,
            eta_seconds: 220,
          },
        },
      ],
    });
    expect(screen.getByText(/1 file — 5 calls in progress/)).toBeTruthy();
  });

  it("done run shows '<files> — <total> calls' (cumulative, no 'in progress')", async () => {
    await renderApp({
      runs: [
        {
          run_id: "2026-04-29T12-00-00Z-done",
          short_id: "done",
          status: "completed",
          mode: "tee",
          inputs: ["/x/y.md"],
          input_count: 1,
          created_at: "2026-04-29T12:00:00Z",
          progress: { stage: "done", completed: 130, total: 130 },
          provider: "tinfoil",
          model: "gpt-oss-120b",
          vault_exists: true,
        },
      ],
    });
    expect(screen.getByText(/1 file — 130 calls$/)).toBeTruthy();
    expect(screen.queryByText(/in progress/)).toBeNull();
  });

  it("run-engine row shows mode + provider only — no model (issue #161)", async () => {
    // Issue #161 dropped the model name from the run-row card. Per-stage
    // routing (and single-model names) live in the Details modal +
    // RunDetails routing snapshot — surfacing the model here too is
    // redundant. Test mode is NOT TEE → no TEE suffix.
    await renderApp({
      runs: [
        {
          run_id: "2026-04-29T12-00-00Z-runn",
          short_id: "runn",
          status: "running",
          mode: "local",
          inputs: ["/x/y.md"],
          input_count: 1,
          created_at: "2026-04-29T12:00:00Z",
          progress: { stage: "extract", in_flight_calls: 2 },
          provider: "ollama",
          model: "qwen3.5:9b",
        },
      ],
    });
    expect(screen.getByText(/Local \(ollama\)/)).toBeTruthy();
    // Model name is intentionally NOT in the run-row anymore.
    expect(screen.queryByText(/qwen3.5:9b/)).toBeNull();
  });

  it("TEE-mode appends ' TEE' to the provider in the run-engine row (issue #161)", async () => {
    // The TEE marker rides on the provider now (gated on mode === "tee"),
    // not on the model. The model name is not rendered in the run-row.
    await renderApp({
      runs: [
        {
          run_id: "2026-04-29T12-00-00Z-runn",
          short_id: "runn",
          status: "running",
          mode: "tee",
          inputs: ["/x/y.md"],
          input_count: 1,
          created_at: "2026-04-29T12:00:00Z",
          progress: { stage: "extract", in_flight_calls: 1 },
          provider: "tinfoil",
          model: "gpt-oss-120b",
        },
      ],
    });
    expect(screen.getByText(/Private Cloud \(tinfoil\.sh TEE\)/)).toBeTruthy();
    // The model name is intentionally NOT in the run-row.
    expect(screen.queryByText(/gpt-oss-120b/)).toBeNull();
  });

  it("a paused row shows resume but not pause", async () => {
    await renderApp({
      runs: [
        {
          run_id: "2026-04-29T12-00-00Z-pswd",
          short_id: "pswd",
          status: "paused",
          mode: "tee",
          inputs: ["/x/y.md"],
          input_count: 1,
          created_at: "2026-04-29T12:00:00Z",
          progress: { stage: "patterns", completed: 30, total: 50 },
          error: "paused because the app was closed",
        },
      ],
    });
    expect(screen.getByTitle("Resume")).toBeTruthy();
    expect(screen.queryByTitle("Pause")).toBeNull();
    // The paused-due-to-close message reads as a note (not a screaming
    // error) — the source treats it as informational.
    expect(screen.getByText(/paused because the app was closed/)).toBeTruthy();
  });

  it("a completed run does not show a per-row regen-vault button", async () => {
    // The vault is now materialized on demand by the Export flow
    // (which calls regenVault before export_run). The standalone
    // ⟳ button on each completed-but-vaultless row was removed —
    // there's no manual entry point for regen, and that's intentional.
    await renderApp({
      runs: [
        {
          run_id: "2026-04-29T12-00-00Z-done",
          short_id: "done",
          status: "completed",
          mode: "tee",
          inputs: ["/x/y.md"],
          input_count: 1,
          created_at: "2026-04-29T12:00:00Z",
          vault_exists: false,
        },
      ],
    });
    expect(screen.queryByTitle(/Vault missing/)).toBeNull();
    expect(screen.queryByTitle(/regenerate/i)).toBeNull();
  });
});

describe("App — startup gating from env state", () => {
  it("exposes Local + Private Cloud as the only mode tabs", async () => {
    await renderApp({
      settings: { tinfoil_key_set: true },
    });
    // Production routing is TEE-only or LOCAL — no other tabs exist.
    const tabs = screen.getAllByRole("tab");
    const labels = tabs.map((t) => t.textContent);
    expect(labels.some((l) => l.includes("Local"))).toBe(true);
    expect(labels.some((l) => l.includes("Private Cloud"))).toBe(true);
    expect(labels.length).toBe(2);
  });

  it("enables the Private Cloud tab on a keyed build even with no user key (bundled key)", async () => {
    // The bundled key powers Private Cloud, so a fresh F&F user who never
    // entered a key of their own must still get an enabled tee tab —
    // otherwise the selector shows "(disabled)" and the fallback logic
    // kicks them off tee, defeating the whole point of the bundled key.
    // Attestation defaults to ok in renderApp.
    await renderApp({
      settings: {
        tinfoil_key_set: false,
        
        
        bundled_key_set: true,
      },
    });
    const teeTab = screen
      .getAllByRole("tab")
      .find((t) => t.textContent.includes("Private Cloud"));
    expect(teeTab).toBeTruthy();
    expect(teeTab.disabled).toBe(false);
    expect(teeTab.textContent).not.toContain("(disabled)");
  });

  it("keeps the TEE mode tab enabled when attestation fails (non-blocking)", async () => {
    // Attestation is a non-blocking visibility signal now: a failed
    // attestation must NOT disable Private Cloud. The transport-layer
    // enclave pinning is the real per-connection guarantee; the UI just
    // surfaces the check on the always-on indicator.
    await renderApp({
      attestation: {
        provider: "tinfoil",
        model: "gpt-oss-120b",
        ok: false,
        fingerprint: null,
        error: "TLS pin mismatch",
        ts: 1700000000,
        constituents: [],
      },
    });
    const teeTab = screen
      .getAllByRole("tab")
      .find((t) => t.textContent.includes("Private Cloud"));
    expect(teeTab).toBeTruthy();
    expect(teeTab.disabled).toBe(false);
    expect(teeTab.textContent).not.toContain("(disabled)");
  });

  const localTab = () =>
    screen.getAllByRole("tab").find((t) => t.textContent.includes("Local"));

  it("disables the Local tab by default when no local model is downloaded", async () => {
    // #731: Local must be off by default — selecting it with no model
    // dead-ends in the MLX import crash (#732). local_model_status is
    // unstubbed here (returns undefined → not downloaded).
    await renderApp({});
    expect(localTab().disabled).toBe(true);
    expect(localTab().textContent).toContain("(disabled)");
  });

  it("enables the Local tab once a model is downloaded on a supported OS", async () => {
    await renderApp({
      override: (cmd) =>
        cmd === "local_model_status"
          ? { downloaded: true, os_supported: true }
          : undefined,
    });
    expect(localTab().disabled).toBe(false);
    expect(localTab().textContent).not.toContain("(disabled)");
  });

  it("keeps the Local tab disabled when the OS is below the MLX floor", async () => {
    // #732 coordination: a downloaded model on an unsupported macOS still
    // can't load the bundled dylib, so Local stays unselectable.
    await renderApp({
      override: (cmd) =>
        cmd === "local_model_status"
          ? { downloaded: true, os_supported: false }
          : undefined,
    });
    expect(localTab().disabled).toBe(true);
  });
});

describe("App — concurrent click + refresh (issue #54)", () => {
  // The 500ms refreshRuns poll + per-LLM-call pipeline-progress event used
  // to re-render RunRow's children mid-click; mousedown landed on a span
  // that was detached before mouseup, and the browser dropped the click.
  // Refresh-skip is scoped to the active mousedown→mouseup click window
  // only. An earlier shape paused refreshes for the whole time the cursor
  // sat over the pane; that froze the run-row progress bar for fast cached
  // runs (the user naturally hovers to watch the new row tick), so idle
  // hover deliberately no longer suspends the poll.

  const runningRun = {
    run_id: "2026-04-29T12-00-00Z-runn",
    short_id: "runn",
    status: "running",
    mode: "tee",
    inputs: ["/x/y.md"],
    input_count: 1,
    created_at: "2026-04-29T12:00:00Z",
    progress: { stage: "extract", completed: 10, total: 50 },
  };

  it("clicking a run row reliably selects it even with an active poll", async () => {
    // Capture the pipeline-progress handler so we can fire it between
    // mousedown and mouseup — that's the actual race the fix targets:
    // the pre-fix bug let a refresh re-render RunRow's children mid-
    // click, which dropped the click event. The click-window skip
    // (onMouseDown → global mouseup) must suppress the refresh while
    // the button is held.
    let progressHandler = null;
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      if (evt === "pipeline-progress") progressHandler = fn;
      return () => {};
    });
    await renderApp({ runs: [runningRun] });
    await waitFor(() => expect(progressHandler).toBeTruthy());
    // Pre-condition: staging pane is showing (no run selected yet).
    expect(screen.getByText(/Input files/)).toBeTruthy();

    const row = screen.getByText("Running").closest(".run-row");
    expect(row).toBeTruthy();

    // Active race reproduction: mousedown → fire a pipeline-progress
    // event (which would have triggered list_runs + child re-render
    // pre-fix) → mouseup → click. The click-window skip must suppress
    // the refresh; without it, mousedown.target would be detached and
    // the click would silently drop.
    const baseline = vi
      .mocked(invoke)
      .mock.calls.filter((c) => c[0] === "list_runs").length;
    fireEvent.mouseDown(row);
    await act(async () => {
      progressHandler({ payload: "tick" });
    });
    // Verify the progress event did NOT fire a refresh (the fix at work).
    const midClick = vi
      .mocked(invoke)
      .mock.calls.filter((c) => c[0] === "list_runs").length;
    expect(midClick).toBe(baseline);
    fireEvent.mouseUp(row);
    fireEvent.click(row);

    // Selection switched the left pane to the ViewingRunPane: a
    // "Run again" button is its anchor (NewRunPane shows "Run").
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Run again/i })).toBeTruthy();
    });
  });

  it("idle hover over the run list does NOT pause the 500ms poll", async () => {
    // Regression guard for the deliberate narrowing of the refresh-skip
    // to the click window: idle hover must keep the poll running so the
    // run-row progress bar keeps ticking while the user watches a fast
    // cached run. The earlier whole-hover pause froze the bar here.
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      await renderApp({ runs: [runningRun] });
      const baseline = vi
        .mocked(invoke)
        .mock.calls.filter((c) => c[0] === "list_runs").length;

      // No hover yet: the poll fires on each 500ms tick.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1100);
      });
      const afterTicks = vi
        .mocked(invoke)
        .mock.calls.filter((c) => c[0] === "list_runs").length;
      expect(afterTicks).toBeGreaterThan(baseline);

      // Cursor sits over the runs pane (mouseEnter, no button held):
      // the poll must keep firing — idle hover does not suspend it.
      const runsPane = document.querySelector(".runs-pane");
      expect(runsPane).toBeTruthy();
      act(() => {
        fireEvent.mouseEnter(runsPane);
      });
      const beforeHoverTicks = vi
        .mocked(invoke)
        .mock.calls.filter((c) => c[0] === "list_runs").length;
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000);
      });
      const afterHoverTicks = vi
        .mocked(invoke)
        .mock.calls.filter((c) => c[0] === "list_runs").length;
      expect(afterHoverTicks).toBeGreaterThan(beforeHoverTicks);
    } finally {
      vi.useRealTimers();
    }
  });

  it("a pipeline-progress event during idle hover still refreshes; the click window suppresses it", async () => {
    let progressHandler = null;
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      if (evt === "pipeline-progress") progressHandler = fn;
      return () => {};
    });
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      await renderApp({ runs: [runningRun] });
      await waitFor(() => expect(progressHandler).toBeTruthy());
      const listRunsCount = () =>
        vi.mocked(invoke).mock.calls.filter((c) => c[0] === "list_runs").length;

      // Idle hover: a progress event STILL drives a refresh once its
      // coalescing timer fires (only the active click window is skipped,
      // not idle hover).
      const runsPane = document.querySelector(".runs-pane");
      act(() => {
        fireEvent.mouseEnter(runsPane);
      });
      const baseline = listRunsCount();
      await act(async () => {
        progressHandler({ payload: "tick" });
        await vi.advanceTimersByTimeAsync(300);
      });
      expect(listRunsCount()).toBeGreaterThan(baseline);

      // Within a mousedown→mouseup window the same event IS suppressed —
      // that's the #406 click-race protection. The 500ms poll also gates
      // on the click window, so the count stays flat while held.
      const row = screen.getByText("Running").closest(".run-row");
      fireEvent.mouseDown(row);
      const heldBaseline = listRunsCount();
      await act(async () => {
        progressHandler({ payload: "tick" });
        await vi.advanceTimersByTimeAsync(600);
      });
      expect(listRunsCount()).toBe(heldBaseline);
      fireEvent.mouseUp(row);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("App — new run from an existing run swaps selection (issue #477)", () => {
  // Pre-fix: starting a new run from run A (run-details "Run again")
  // left A selected, so the view stayed on the old run and the user
  // had to manually click the freshly-created row. Pin the swap:
  // source deselected, new run sole selection + scrolled into view.
  const sourceRun = {
    run_id: "2026-05-15T10-00-00Z-srca",
    short_id: "srca",
    status: "completed",
    mode: "tee",
    inputs: ["/x/a.md"],
    input_count: 1,
    created_at: "2026-05-15T10:00:00Z",
    progress: { stage: "done", completed: 1, total: 1 },
  };
  const newRun = {
    run_id: "2026-05-15T11-00-00Z-newb",
    short_id: "newb",
    status: "running",
    mode: "tee",
    inputs: ["/x/a.md"],
    input_count: 1,
    created_at: "2026-05-15T11:00:00Z",
    progress: { stage: "extract", completed: 0, total: 1 },
  };

  it("deselects the source run, selects + scrolls the new run into view", async () => {
    let runsState = [sourceRun];
    const scrollSpy = vi.spyOn(Element.prototype, "scrollIntoView");
    await renderApp({
      override: (cmd) => {
        if (cmd === "list_runs") return runsState;
        if (cmd === "run_pipeline") {
          runsState = [newRun, sourceRun];
          return newRun.run_id;
        }
        return undefined;
      },
    });

    // Select the source run → run-details pane (anchor: "Run again").
    fireEvent.click(screen.getByText("#srca").closest(".run-row"));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /Run again/i })).toBeTruthy(),
    );
    expect(
      document
        .querySelector(`[data-run-id="${sourceRun.run_id}"]`)
        .className,
    ).toContain("selected");

    // Run again from this run → selection follows the new run.
    fireEvent.click(screen.getByRole("button", { name: /Run again/i }));
    await waitFor(() => {
      const byId = (id) =>
        document.querySelector(`[data-run-id="${id}"]`);
      expect(byId(newRun.run_id)?.className).toContain("selected");
      expect(byId(sourceRun.run_id)?.className).not.toContain("selected");
    });

    // Gate-2: the newly-selected row was scrolled into view.
    const newRow = document.querySelector(
      `[data-run-id="${newRun.run_id}"]`,
    );
    expect(scrollSpy.mock.contexts).toContain(newRow);
  });
});

describe("App — switching selectedRun resets middle + right panes (issue #83)", () => {
  // Pre-fix bug: clicking from run A to run B left A's tree visible
  // (sometimes briefly, sometimes until a tab-away/back), and the
  // markdown pane kept rendering A's previously-selected file. Live
  // updates for B's in-flight pipeline were also flaky because the
  // pipeline-progress listener was being torn down + re-registered on
  // every selectedRun change, racing against listen()'s pending
  // promise.

  const runA = {
    run_id: "2026-04-29T12-00-00Z-aaaa",
    short_id: "aaaa",
    status: "completed",
    mode: "tee",
    inputs: ["/u/barbellion.txt"],
    input_count: 1,
    created_at: "2026-04-29T12:00:00Z",
    duration_ms: 30_000,
    progress: { stage: "done", completed: 50, total: 50 },
    provider: "tinfoil",
    model: "gpt-oss-120b",
    vault_exists: true,
  };
  const runB = {
    run_id: "2026-04-30T12-00-00Z-bbbb",
    short_id: "bbbb",
    status: "running",
    mode: "tee",
    inputs: ["/u/pepys.txt"],
    input_count: 1,
    created_at: "2026-04-30T12:00:00Z",
    progress: { stage: "extract", completed: 5, total: 50 },
    provider: "tinfoil",
    model: "gpt-oss-120b",
    vault_exists: false,
  };

  // Tree shapes that make A's and B's bleed visually distinguishable
  // by an input filename rendered under the inputs dir.
  const treeA = [
    {
      name: "0-inputs",
      rel_path: "0-inputs",
      is_dir: true,
      children: [
        {
          name: "barbellion.txt.md",
          rel_path: "0-inputs/barbellion.txt.md",
          is_dir: false,
          children: [],
        },
      ],
    },
  ];
  const treeB = [
    {
      name: "0-inputs",
      rel_path: "0-inputs",
      is_dir: true,
      children: [
        {
          name: "pepys.txt.md",
          rel_path: "0-inputs/pepys.txt.md",
          is_dir: false,
          children: [],
        },
      ],
    },
  ];

  it("clearing happens synchronously on click; the new run's tree fetches against the new run_id", async () => {
    // Hold-then-release on B's tree fetch so we can observe the cleared
    // intermediate state. A's tree resolves immediately. Without the
    // synchronous reset, A's tree would stay visible until B's fetch
    // completes — exactly the bug.
    let resolveTreeB;
    const treeBPromise = new Promise((r) => {
      resolveTreeB = r;
    });
    const treeCalls = [];
    await renderApp({
      runs: [runA, runB],
      override: (cmd, args) => {
        if (cmd === "list_run_tree") {
          treeCalls.push(args.runId);
          if (args.runId === runA.run_id) return treeA;
          if (args.runId === runB.run_id) return treeBPromise;
        }
        if (cmd === "read_run_facts") return [];
        return undefined;
      },
    });

    // Click run A's row → A's tree dir renders. 0-inputs is closed by
    // default (the user just queued those files), so click its
    // disclosure to expose the leaf that distinguishes A from B.
    const rowA = screen.getByText("barbellion.txt").closest(".run-row");
    await userEvent.click(rowA);
    await waitFor(() => {
      expect(screen.getByText("0-inputs")).toBeTruthy();
    });
    await userEvent.click(screen.getByText("0-inputs").closest(".tree-row"));
    await waitFor(() => {
      expect(screen.getByText("barbellion.txt.md")).toBeTruthy();
    });
    expect(treeCalls.filter((id) => id === runA.run_id).length).toBeGreaterThanOrEqual(1);

    // Click run B's row → A's tree must clear synchronously, even
    // though B's fetch is still pending (treeBPromise is unresolved).
    const rowB = screen.getByText("pepys.txt").closest(".run-row");
    await userEvent.click(rowB);

    // A's distinctive file must be gone immediately. B's hasn't
    // arrived yet (we haven't resolved treeBPromise). The empty-state
    // placeholder is what should be visible.
    expect(screen.queryByText("barbellion.txt.md")).toBeNull();
    expect(screen.getByText(/No vault output yet/)).toBeTruthy();

    // B's tree fetch should have been issued with B's run_id.
    expect(treeCalls).toContain(runB.run_id);

    // Once B's fetch resolves, B's dir renders — open it to verify
    // the leaf, and confirm A's leaf still doesn't.
    await act(async () => {
      resolveTreeB(treeB);
      await Promise.resolve();
    });
    await waitFor(() => {
      expect(screen.getByText("0-inputs")).toBeTruthy();
    });
    await userEvent.click(screen.getByText("0-inputs").closest(".tree-row"));
    await waitFor(() => {
      expect(screen.getByText("pepys.txt.md")).toBeTruthy();
    });
    expect(screen.queryByText("barbellion.txt.md")).toBeNull();
  });

  it("a stale tree-refresh timer queued against run A doesn't clobber run B's tree after a switch", async () => {
    // Reproduces the in-flight bleed: pipeline-progress event fires
    // for A, schedules a 500ms tree refresh against A. User clicks B
    // before the timer fires. Pre-fix the timer fired with A's run_id
    // and overwrote B's freshly-loaded tree. Post-fix the run-change
    // effect cancels the pending timer AND the timer body re-reads
    // the current run_id from a ref at fire-time.
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      let progressHandler = null;
      vi.mocked(listen).mockImplementation(async (evt, fn) => {
        if (evt === "pipeline-progress") progressHandler = fn;
        return () => {};
      });
      const treeCalls = [];
      await renderApp({
        runs: [runA, runB],
        override: (cmd, args) => {
          if (cmd === "list_run_tree") {
            treeCalls.push(args.runId);
            if (args.runId === runA.run_id) return treeA;
            if (args.runId === runB.run_id) return treeB;
          }
          if (cmd === "read_run_facts") return [];
          return undefined;
        },
      });
      await waitFor(() => expect(progressHandler).toBeTruthy());

      // Select A; let its tree load. 0-inputs is closed by default;
      // open it to expose A's distinctive leaf.
      await userEvent.click(screen.getByText("barbellion.txt").closest(".run-row"));
      await waitFor(() => {
        expect(screen.getByText("0-inputs")).toBeTruthy();
      });
      await userEvent.click(screen.getByText("0-inputs").closest(".tree-row"));
      await waitFor(() => {
        expect(screen.getByText("barbellion.txt.md")).toBeTruthy();
      });

      // Pipeline-progress event arrives for A → schedules a 500ms
      // tree-refresh timer. User immediately switches to B (before
      // the 500ms elapses).
      await act(async () => {
        progressHandler({ payload: "tick" });
      });
      await userEvent.click(screen.getByText("pepys.txt").closest(".run-row"));
      // B's initial fetch fires synchronously and resolves; let
      // microtasks settle.
      await act(async () => {
        await Promise.resolve();
      });
      await waitFor(() => {
        expect(screen.getByText("0-inputs")).toBeTruthy();
      });
      await userEvent.click(screen.getByText("0-inputs").closest(".tree-row"));
      await waitFor(() => {
        expect(screen.getByText("pepys.txt.md")).toBeTruthy();
      });

      // Drain the would-have-been stale 500ms timer. Pre-fix this
      // would issue list_run_tree with A's run_id and clobber B's
      // tree. Post-fix the timer was cancelled at click time, so
      // no extra A-targeted call should appear.
      const aCallsBefore = treeCalls.filter((id) => id === runA.run_id).length;
      await act(async () => {
        await vi.advanceTimersByTimeAsync(700);
      });
      const aCallsAfter = treeCalls.filter((id) => id === runA.run_id).length;
      expect(aCallsAfter).toBe(aCallsBefore);
      // B's file must still be visible.
      expect(screen.getByText("pepys.txt.md")).toBeTruthy();
      expect(screen.queryByText("barbellion.txt.md")).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("the run-switch reset must paint synchronously — no frame of A's content under selectedRun=B", async () => {
    // The existing two tests in this block assert the post-effect
    // state via `await userEvent.click(...)`, which wraps in act()
    // and flushes BOTH useEffect and useLayoutEffect before the
    // assertion runs. That passes whether the reset is wired through
    // useEffect (post-paint, browser shows a stale frame on real
    // Tauri) or useLayoutEffect (pre-paint, never visible). Vitest's
    // act() wrapper masks the difference, so a passing test there
    // does NOT pin the user-observable invariant.
    //
    // The bug as the issue body describes it ("Never run A's tree,
    // even briefly" / "Never auto-renders run A's previously-
    // selected file") is exactly the paint between Render-after-
    // click and Render-after-useEffect: selectedRun=B but
    // selectedFile, runTree, fileContent still point at A. To pin
    // it we need a click that does NOT flush useEffect — a raw DOM
    // .click() bypasses RTL's act() wrapper. React still batches
    // and commits inside the click handler (so useLayoutEffect
    // fires synchronously, producing the post-reset DOM), but
    // useEffect stays queued on the microtask. Asserting between
    // those two checkpoints differentiates the two timings.
    let resolveContentA;
    let resolveContentB;
    await renderApp({
      runs: [runA, runB],
      override: (cmd, args) => {
        if (cmd === "list_run_tree") {
          if (args.runId === runA.run_id) return treeA;
          if (args.runId === runB.run_id) return treeB;
        }
        if (cmd === "read_run_file") {
          if (args.runId === runA.run_id && args.relPath === "0-inputs/barbellion.txt.md") {
            return new Promise((r) => { resolveContentA = r; });
          }
          if (args.runId === runB.run_id) {
            return new Promise((r) => { resolveContentB = r; });
          }
        }
        if (cmd === "read_run_facts") return [];
        return undefined;
      },
    });

    // Open A → expand A's tree → click A's input file → render A's
    // content into the right pane. After this block the DOM contains
    // BARBELLION_BODY_MARKER (right pane) + barbellion.txt.md
    // (middle pane tree node). 0-inputs is closed by default; open
    // it so the leaf is reachable.
    await userEvent.click(screen.getByText("barbellion.txt").closest(".run-row"));
    await waitFor(() => {
      expect(screen.getByText("0-inputs")).toBeTruthy();
    });
    await userEvent.click(screen.getByText("0-inputs").closest(".tree-row"));
    await waitFor(() => {
      expect(screen.getByText("barbellion.txt.md")).toBeTruthy();
    });
    await userEvent.click(screen.getByText("barbellion.txt.md"));
    // Resolve A's file body so the right pane actually paints content.
    await act(async () => {
      resolveContentA("BARBELLION_BODY_MARKER");
      await Promise.resolve();
    });
    await waitFor(() => {
      expect(screen.getByText(/BARBELLION_BODY_MARKER/)).toBeTruthy();
    });

    // Raw DOM .click() — bypasses RTL's act() wrapper, so React's
    // event handler runs but the scheduled render + effects are
    // not force-flushed. A single microtask tick (Promise.resolve)
    // is enough for React to commit the render and synchronously
    // run useLayoutEffect (which itself triggers a re-render that
    // commits in the same microtask). useEffect waits for a
    // macrotask (setTimeout / MessageChannel) and is therefore
    // NOT flushed at this checkpoint. Pre-fix the reset was a
    // useEffect, so after this checkpoint the DOM reflects only
    // the click-induced render: selectedRun=B, but selectedFile,
    // runTree, and fileContent all still point at A — A's tree
    // node + A's right-pane body marker are both painted. Post-
    // fix the reset is a useLayoutEffect, so the post-reset DOM
    // (selectedFile=null, runTree=[]) commits inside the
    // microtask and the assertions hold.
    //
    // React would emit a "not wrapped in act(...)" warning here
    // because the click is intentionally outside an act() boundary
    // — wrapping it would flush BOTH effect types and re-mask the
    // pre-fix vs post-fix timing distinction. Filter that one
    // string out so this test does not pollute the suite output;
    // any other console.error still surfaces. Future maintainers:
    // do NOT "fix" this by wrapping the click in act(), it would
    // un-pin the regression.
    const consoleErrorSpy = vi
      .spyOn(console, "error")
      .mockImplementation((first, ...rest) => {
        if (typeof first === "string" && first.includes("not wrapped in act")) return;
        // Surface anything else through the original implementation
        // so real errors still fail the test.
        const orig = consoleErrorSpy.getMockImplementation();
        if (orig && orig !== consoleErrorSpy) orig(first, ...rest);
      });
    const rowB = screen.getByText("pepys.txt").closest(".run-row");
    rowB.click();
    await Promise.resolve();

    // Pre-useEffect-flush assertion. This is the frame the user
    // actually sees on real Tauri (WKWebView paints between the
    // click and the next macrotask).
    expect(screen.queryByText(/BARBELLION_BODY_MARKER/)).toBeNull();
    expect(screen.queryByText("barbellion.txt.md")).toBeNull();

    // Drain the rest of the effect chain (file-content effect,
    // facts cache effect) and re-assert. This is the existing
    // tests' assertion shape — passes under either timing — kept
    // here so a regression that breaks the long-run invariant
    // also surfaces.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(screen.queryByText("barbellion.txt.md")).toBeNull();
    expect(screen.queryByText(/BARBELLION_BODY_MARKER/)).toBeNull();
    // Defensive: even if B's file body resolves later (it won't,
    // the user hasn't picked a file in B yet), the render gate
    // must not resurrect A's content.
    if (resolveContentB) resolveContentB("PEPYS_BODY_MARKER");
    consoleErrorSpy.mockRestore();
  });
});

// Pin the cmd+F segment locator behaviour. Issue #80: the original
// implementation walked segments linearly per match, making the search
// O(M·S) and stalling the main thread for seconds on large inputs.
// The forward-cursor variant must return the same segment indices as
// a naive linear scan and amortize to O(M+S) when matches are sorted.
describe("findSegmentForward", () => {
  function build(spans) {
    let acc = 0;
    return spans.map((len) => {
      const seg = { node: { len }, start: acc, end: acc + len };
      acc += len;
      return seg;
    });
  }

  it("returns the index of the segment containing the offset", () => {
    const segs = build([5, 5, 5]);
    expect(findSegmentForward(segs, 0)).toBe(0);
    expect(findSegmentForward(segs, 4)).toBe(0);
    expect(findSegmentForward(segs, 5)).toBe(1);
    expect(findSegmentForward(segs, 9)).toBe(1);
    expect(findSegmentForward(segs, 10)).toBe(2);
    expect(findSegmentForward(segs, 14)).toBe(2);
  });

  it("returns -1 when the offset is past the last segment", () => {
    const segs = build([3, 3]);
    expect(findSegmentForward(segs, 6)).toBe(-1);
    expect(findSegmentForward(segs, 100)).toBe(-1);
  });

  it("never returns an index below `fromIndex`", () => {
    const segs = build([5, 5, 5]);
    // Even though offset 0 is in segment 0, a cursor advanced past it
    // (e.g. by an earlier match) should not rewind. The caller is
    // responsible for monotonic offsets; the helper trusts the cursor.
    expect(findSegmentForward(segs, 0, 1)).toBe(1);
    expect(findSegmentForward(segs, 0, 2)).toBe(2);
  });

  it("matches a naive linear scan across a monotonically advancing cursor", () => {
    // 100 segments of varying length; the offsets we query advance
    // monotonically. The cursor-threaded forward search must return
    // the same indices as a fresh-from-zero scan would.
    const segs = build(Array.from({ length: 100 }, (_, i) => 1 + (i % 7)));
    const offsets = [];
    for (let i = 0; i < 200; i++) {
      offsets.push(Math.floor(((segs.at(-1).end - 1) * i) / 199));
    }
    let cursor = 0;
    for (const off of offsets) {
      const cursored = findSegmentForward(segs, off, cursor);
      const fromZero = findSegmentForward(segs, off, 0);
      expect(cursored).toBe(fromZero);
      cursor = cursored;
    }
  });

  it("MAX_SEARCH_MATCHES is bounded so the UI never tries to paint the unbounded set", () => {
    expect(MAX_SEARCH_MATCHES).toBeGreaterThan(0);
    expect(MAX_SEARCH_MATCHES).toBeLessThanOrEqual(50000);
  });
});

describe("formatSizeShort", () => {
  // Spec from issue #120: kb / mb only (lowercase, no GB), single
  // decimal digit when the integer part is 0-9, otherwise integer-
  // rounded. 0 bytes is the lone special case (no decimal).
  it("renders zero bytes as `0kb` (the only no-decimal case)", () => {
    expect(formatSizeShort(0)).toBe("0kb");
  });

  it("renders sub-1kb non-zero values as `1.0kb` (rounds 1023 up via toFixed)", () => {
    // 1023/1024 = 0.999..., toFixed(1) = '1.0' — matches the spec's
    // "round up so anything non-zero shows non-zero" intent.
    expect(formatSizeShort(1023)).toBe("1.0kb");
  });

  it("renders 1kb exact as `1.0kb`", () => {
    expect(formatSizeShort(1024)).toBe("1.0kb");
  });

  it("renders ~4.2kb in the single-digit decimal form", () => {
    // 4096 = 4.0kb exactly; 4300 / 1024 = 4.199 → '4.2kb'.
    expect(formatSizeShort(4096)).toBe("4.0kb");
    expect(formatSizeShort(4300)).toBe("4.2kb");
    // 4200 from the issue's loose ~4200 example computes to 4.1kb
    // (4200/1024 = 4.101). The "~" in the spec is approximate; the
    // assertion below pins the actual computed rendering.
    expect(formatSizeShort(4200)).toBe("4.1kb");
  });

  it("drops the decimal once the kb integer part has two digits", () => {
    expect(formatSizeShort(10240)).toBe("10kb");
    expect(formatSizeShort(105 * 1024)).toBe("105kb");
  });

  it("renders 1mb exact as `1.0mb`", () => {
    expect(formatSizeShort(1048576)).toBe("1.0mb");
  });

  it("renders single-digit mb with one decimal", () => {
    // 1.4 mb → 1.4mb; 9.0 mb → 9.0mb (still single-digit).
    expect(formatSizeShort(Math.round(1.4 * 1024 * 1024))).toBe("1.4mb");
    expect(formatSizeShort(9 * 1024 * 1024)).toBe("9.0mb");
  });

  it("drops the decimal once the mb integer part has two-or-more digits", () => {
    expect(formatSizeShort(51 * 1024 * 1024)).toBe("51mb");
    expect(formatSizeShort(145 * 1024 * 1024)).toBe("145mb");
  });

  it("never emits `10.0kb` at the kb→two-digit-int boundary", () => {
    // Round-then-compare: a value like 9.97kb rounds to 10.0 via
    // toFixed. The implementation must catch that and fall through
    // to the integer-rounded path so the rendering stays consistent.
    const bytes = Math.round(9.97 * 1024);
    expect(formatSizeShort(bytes)).toBe("10kb");
  });

  it("treats negatives / NaN / null as zero (defensive)", () => {
    expect(formatSizeShort(-1)).toBe("0kb");
    expect(formatSizeShort(NaN)).toBe("0kb");
    expect(formatSizeShort(null)).toBe("0kb");
    expect(formatSizeShort(undefined)).toBe("0kb");
  });
});

describe("App — displayBasename strips trailing `.md`", () => {
  // Renderer-level assertion: a picked path that ends in .md should
  // render without the extension across every user-visible surface.
  // The staging row is the simplest anchor for the assertion (the
  // run-row title + RunDetails inputs + pane title route through the
  // same helper).
  it("staging row renders the basename minus `.md`", async () => {
    const persistedInputs = ["/Users/x/notes/journal.md"];
    await renderApp({
      config: {
        subject: "User",
        tee_provider: "tinfoil",
        tee_model: "gpt-oss-120b",
        inputs: persistedInputs,
        mode: "tee",
      },
      override: (cmd, args) => {
        if (cmd === "stat_paths") {
          return (args?.paths || []).map((p) => ({ path: p, size_bytes: 0 }));
        }
        return undefined;
      },
    });
    await waitFor(() => {
      const path = document.querySelector(".inputs-list .input-path");
      expect(path?.textContent).toBe("journal");
    });
  });

  it("staging row preserves non-`.md` extensions", async () => {
    const persistedInputs = ["/Users/x/photo.jpg"];
    await renderApp({
      config: {
        subject: "User",
        tee_provider: "tinfoil",
        tee_model: "gpt-oss-120b",
        inputs: persistedInputs,
        mode: "tee",
      },
      override: (cmd, args) => {
        if (cmd === "stat_paths") {
          return (args?.paths || []).map((p) => ({ path: p, size_bytes: 0 }));
        }
        return undefined;
      },
    });
    await waitFor(() => {
      const path = document.querySelector(".inputs-list .input-path");
      expect(path?.textContent).toBe("photo.jpg");
    });
  });
});

describe("App — run tree surfaces 0-inputs file sizes", () => {
  // The Rust list_run_tree attaches `size_bytes` only to leaves under
  // 0-inputs/. The React TreeNode must render that as a `.tree-size`
  // span next to the file name; non-input leaves (facts, entities …)
  // omit the span — they already carry their own count badge in the
  // node name.
  it("renders a `.tree-size` span next to a 0-inputs file leaf", async () => {
    const completedRun = {
      run_id: "2026-04-29T12-00-00Z-trsz",
      short_id: "trsz",
      status: "completed",
      mode: "tee",
      inputs: ["/u/big.md"],
      input_count: 1,
      created_at: "2026-04-29T12:00:00Z",
      duration_ms: 30_000,
      progress: { stage: "done", completed: 50, total: 50 },
      vault_exists: true,
    };
    const tree = [
      {
        name: "Inputs (1)",
        rel_path: "0-inputs",
        is_dir: true,
        children: [
          {
            name: "Big",
            rel_path: "0-inputs/big.md",
            is_dir: false,
            children: [],
            size_bytes: 4300,
          },
        ],
      },
      {
        name: "Facts (1)",
        rel_path: "1-facts",
        is_dir: true,
        children: [
          { name: "Work (1)", rel_path: "1-facts/work.md", is_dir: false, children: [] },
        ],
      },
    ];
    await renderApp({
      runs: [completedRun],
      override: (cmd, args) => {
        if (cmd === "list_run_tree" && args.runId === completedRun.run_id) return tree;
        if (cmd === "read_run_facts") return [];
        return undefined;
      },
    });
    await userEvent.click(screen.getByText("big").closest(".run-row"));
    // 0-inputs is closed by default; open it to expose the leaf
    // that carries the .tree-size badge.
    await waitFor(() => {
      expect(screen.getByText("Inputs (1)")).toBeTruthy();
    });
    await userEvent.click(screen.getByText("Inputs (1)").closest(".tree-row"));
    await waitFor(() => {
      // 0-inputs leaf carries the size badge.
      const sizes = document.querySelectorAll(".tree-size");
      expect(sizes.length).toBe(1);
      expect(sizes[0].textContent).toBe("4.2kb");
    });
  });
});

describe("App — staging area shows file sizes per row", () => {
  // Bootstrap with a persisted inputs list; the bootstrap effect
  // calls stat_paths(...) on those paths and the staging row should
  // render the formatted size next to the basename. Anchor on the
  // size text; the basename is verified separately by the
  // pre-existing staging tests.
  it("renders a `.input-size` span sourced from stat_paths", async () => {
    const persistedInputs = ["/Users/x/notes/note.md"];
    await renderApp({
      config: {
        subject: "User",
        tee_provider: "tinfoil",
        tee_model: "gpt-oss-120b",
        inputs: persistedInputs,
        mode: "tee",
      },
      override: (cmd, args) => {
        if (cmd === "stat_paths") {
          // Echo whichever paths the App requested with a fixed size.
          const paths = (args?.paths || []);
          return paths.map((p) => ({ path: p, size_bytes: 4300 }));
        }
        return undefined;
      },
    });
    // The size span renders inside the staging .input-row li. Wait
    // for the async stat_paths fetch to settle (state update).
    await waitFor(() => {
      const sizes = document.querySelectorAll(".inputs-list .input-size");
      expect(sizes.length).toBe(1);
      expect(sizes[0].textContent).toBe("4.2kb");
    });
  });
});

describe("App — run tree default-open + auto-open new dirs", () => {
  // The tree pane should open every top-level stage dir by default on
  // run-select — 1-facts / 3-patterns / 2-entities — so the
  // per-entity_type breakdown + counts are visible without a click
  // (#603). The type groups *under* 2-entities stay closed, like fact
  // topics: the user clicks a type to see its entities. Only 0-inputs
  // stays collapsed (the user just queued those files — they already
  // know what's in there). When a top-level stage dir materializes
  // mid-run via the pipeline-progress refresh path, it must auto-open
  // (except 0-inputs); type groups are never auto-opened. User-driven
  // closes must persist across refreshes.

  const completedRun = {
    run_id: "2026-04-29T12-00-00Z-tree",
    short_id: "tree",
    status: "completed",
    mode: "tee",
    inputs: ["/u/note.md"],
    input_count: 1,
    created_at: "2026-04-29T12:00:00Z",
    duration_ms: 30_000,
    progress: { stage: "done", completed: 50, total: 50 },
    vault_exists: true,
  };

  const fullTree = [
    {
      name: "Inputs (1)",
      rel_path: "0-inputs",
      is_dir: true,
      children: [
        { name: "Note", rel_path: "0-inputs/note.md", is_dir: false, children: [] },
      ],
    },
    {
      name: "Facts (3)",
      rel_path: "1-facts",
      is_dir: true,
      children: [
        { name: "Work (3)", rel_path: "1-facts/work.md", is_dir: false, children: [] },
      ],
    },
    {
      name: "Entities (1)",
      rel_path: "2-entities",
      is_dir: true,
      children: [
        {
          name: "Person (1)",
          rel_path: "2-entities/__type__/person",
          is_dir: true,
          children: [
            { name: "Alice (5)", rel_path: "2-entities/alice.md", is_dir: false, children: [] },
          ],
        },
      ],
    },
    {
      name: "Patterns (2)",
      rel_path: "3-patterns",
      is_dir: true,
      children: [
        { name: "Work (2)", rel_path: "3-patterns/work.md", is_dir: false, children: [] },
      ],
    },
  ];

  it("opens 1-facts / 3-patterns / 2-entities by default; type groups + 0-inputs stay collapsed", async () => {
    await renderApp({
      runs: [completedRun],
      override: (cmd, args) => {
        if (cmd === "list_run_tree" && args.runId === completedRun.run_id) {
          return fullTree;
        }
        if (cmd === "read_run_facts") return [];
        return undefined;
      },
    });
    await userEvent.click(screen.getByText("note").closest(".run-row"));
    await waitFor(() => {
      // Children of opened dirs render → these leaf names appear.
      expect(screen.getByText("Work (3)")).toBeTruthy();
      expect(screen.getByText("Work (2)")).toBeTruthy();
    });
    // #603: 2-entities defaults open → the Person type node + count
    // is visible without a click.
    expect(screen.getByText("Person (1)")).toBeTruthy();
    // …but the Person type group itself stays closed (like a fact
    // topic) → its entity leaf is NOT in the DOM until clicked.
    expect(screen.queryByText("Alice (5)")).toBeNull();
    // 0-inputs stays collapsed → "Note" leaf must NOT be in the DOM.
    expect(screen.queryByText("Note")).toBeNull();
  });

  it("auto-opens a stage dir that materializes on a refresh", async () => {
    // Simulate an in-flight run: first list_run_tree returns inputs
    // only; pipeline-progress fires; second list_run_tree returns
    // inputs + facts. The newly-arrived facts dir must auto-open.
    let progressHandler = null;
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      if (evt === "pipeline-progress") progressHandler = fn;
      return () => {};
    });
    let treeCallCount = 0;
    const inputsOnly = [fullTree[0]];
    const inputsAndFacts = [fullTree[0], fullTree[1]];
    await renderApp({
      runs: [{ ...completedRun, status: "running" }],
      override: (cmd, args) => {
        if (cmd === "list_run_tree" && args.runId === completedRun.run_id) {
          treeCallCount += 1;
          return treeCallCount === 1 ? inputsOnly : inputsAndFacts;
        }
        if (cmd === "read_run_facts") return [];
        return undefined;
      },
    });
    await waitFor(() => expect(progressHandler).toBeTruthy());
    await userEvent.click(screen.getByText("note").closest(".run-row"));
    // Wait for first tree fetch to land. 0-inputs is closed by
    // default, so anchor on the dir row itself rather than the leaf.
    await waitFor(() => expect(screen.getByText("Inputs (1)")).toBeTruthy());
    // Facts dir not in tree yet.
    expect(screen.queryByText("Work (3)")).toBeNull();

    // Pipeline-progress event → debounced tree refresh (500ms timer
    // inside scheduleTreeRefresh). Fire the event then wait for the
    // refresh to land + auto-open the new dir.
    await act(async () => {
      progressHandler({});
    });
    await waitFor(
      () => expect(screen.getByText("Work (3)")).toBeTruthy(),
      { timeout: 2000 },
    );
  });

  it("auto-opens 2-entities on a refresh but leaves its type groups closed", async () => {
    // In-flight: 2-entities doesn't exist on the first fetch (entities
    // stage not reached); it arrives on a later refresh. The
    // 2-entities parent must auto-open so the per-type breakdown +
    // count shows, but the Person type group under it stays closed —
    // the user clicks a type to drill in, like a fact topic (#603).
    let progressHandler = null;
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      if (evt === "pipeline-progress") progressHandler = fn;
      return () => {};
    });
    let treeCallCount = 0;
    const inputsOnly = [fullTree[0]];
    const inputsAndEntities = [fullTree[0], fullTree[2]];
    await renderApp({
      runs: [{ ...completedRun, status: "running" }],
      override: (cmd, args) => {
        if (cmd === "list_run_tree" && args.runId === completedRun.run_id) {
          treeCallCount += 1;
          return treeCallCount === 1 ? inputsOnly : inputsAndEntities;
        }
        if (cmd === "read_run_facts") return [];
        return undefined;
      },
    });
    await waitFor(() => expect(progressHandler).toBeTruthy());
    await userEvent.click(screen.getByText("note").closest(".run-row"));
    await waitFor(() => expect(screen.getByText("Inputs (1)")).toBeTruthy());
    // Entities not in tree yet.
    expect(screen.queryByText("Person (1)")).toBeNull();

    await act(async () => {
      progressHandler({});
    });
    // 2-entities auto-opens → the Person type node is visible…
    await waitFor(
      () => expect(screen.getByText("Person (1)")).toBeTruthy(),
      { timeout: 2000 },
    );
    // …but the type group itself stays closed → leaf hidden.
    expect(screen.queryByText("Alice (5)")).toBeNull();
  });

  it("auto-opens 4-insights.md on a refresh — insights expand without a click as the stage lands", async () => {
    // In-flight: 4-insights.md is a top-level FILE (not a dir) that
    // expands to its items. It arrives on a later refresh and must
    // auto-open like the stage dirs, so the insights are visible the
    // moment the stage completes — no click. (Same treeDefaultDirs seed
    // that drives the initial-select path also drives this refresh
    // path, so both stay in lockstep.)
    let progressHandler = null;
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      if (evt === "pipeline-progress") progressHandler = fn;
      return () => {};
    });
    let treeCallCount = 0;
    const insightsNode = {
      name: "Insights (1)",
      rel_path: "4-insights.md",
      is_dir: false,
      children: [],
    };
    const inputsOnly = [fullTree[0]];
    const inputsAndInsights = [fullTree[0], insightsNode];
    await renderApp({
      runs: [{ ...completedRun, status: "running" }],
      override: (cmd, args) => {
        if (cmd === "list_run_tree" && args.runId === completedRun.run_id) {
          treeCallCount += 1;
          return treeCallCount === 1 ? inputsOnly : inputsAndInsights;
        }
        if (cmd === "read_run_facts") return [];
        if (cmd === "read_run_insights") {
          return {
            cross_domain: [
              { name: "Cross-A", kind: "cross_domain", domains: ["work"] },
            ],
            critical: [],
          };
        }
        if (cmd === "read_run_actions") return null;
        return undefined;
      },
    });
    await waitFor(() => expect(progressHandler).toBeTruthy());
    await userEvent.click(screen.getByText("note").closest(".run-row"));
    await waitFor(() => expect(screen.getByText("Inputs (1)")).toBeTruthy());
    // Insights file not in the tree yet.
    expect(screen.queryByText("Insights (1)")).toBeNull();

    await act(async () => {
      progressHandler({});
    });
    // 4-insights.md auto-opens on arrival → its entry shows with no
    // click. Scope to the tree pane (the markdown body can render the
    // same insight name as a heading).
    await waitFor(
      () =>
        expect(
          within(document.querySelector(".tree-pane")).getByText("Cross-A"),
        ).toBeTruthy(),
      { timeout: 2000 },
    );
  });

  it("clicking a type group expands it to reveal its entities", async () => {
    // The type group defaults closed (per #603 it mirrors a fact
    // topic). It must still be a working disclosure: clicking the
    // Person row reveals its nested entity leaves.
    await renderApp({
      runs: [completedRun],
      override: (cmd, args) => {
        if (cmd === "list_run_tree" && args.runId === completedRun.run_id) {
          return fullTree;
        }
        if (cmd === "read_run_facts") return [];
        return undefined;
      },
    });
    await userEvent.click(screen.getByText("note").closest(".run-row"));
    await waitFor(() => expect(screen.getByText("Person (1)")).toBeTruthy());
    // Closed by default → leaf hidden.
    expect(screen.queryByText("Alice (5)")).toBeNull();
    // Click the type group disclosure row → entity leaf appears.
    const personRow = screen.getByText("Person (1)").closest(".tree-row");
    await userEvent.click(personRow);
    await waitFor(() => expect(screen.getByText("Alice (5)")).toBeTruthy());
  });

  it("a user-driven close persists across the next refresh", async () => {
    // Initial load includes 1-facts (auto-opened by default). User
    // clicks the disclosure to close it. Refresh fires; 1-facts
    // must STAY closed (seenTopDirsRef gates the auto-open path so
    // a once-handled dir is never re-opened).
    let progressHandler = null;
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      if (evt === "pipeline-progress") progressHandler = fn;
      return () => {};
    });
    await renderApp({
      runs: [completedRun],
      override: (cmd, args) => {
        if (cmd === "list_run_tree" && args.runId === completedRun.run_id) {
          return fullTree;
        }
        if (cmd === "read_run_facts") return [];
        return undefined;
      },
    });
    await waitFor(() => expect(progressHandler).toBeTruthy());
    await userEvent.click(screen.getByText("note").closest(".run-row"));
    await waitFor(() => expect(screen.getByText("Work (3)")).toBeTruthy());

    // Close 1-facts: the click handler is on .tree-row (parent of
    // the .tree-name span). Click the row so the toggle fires.
    const factsRow = screen.getByText("Facts (3)").closest(".tree-row");
    await userEvent.click(factsRow);
    await waitFor(() => expect(screen.queryByText("Work (3)")).toBeNull());

    // Refresh: 1-facts stays closed (user closed it; no re-open via
    // seenTopDirsRef), while the untouched default-open 2-entities
    // parent stays open → its Person type node is still visible
    // (the type group itself stays closed throughout, so its leaf
    // never appears).
    await act(async () => {
      progressHandler({});
    });
    await new Promise((r) => setTimeout(r, 700)); // beyond the 500ms debounce
    expect(screen.queryByText("Work (3)")).toBeNull();
    expect(screen.getByText("Person (1)")).toBeTruthy();
    expect(screen.queryByText("Alice (5)")).toBeNull();
  });
});

describe("App — bulk Delete invokes delete_run", () => {
  // Issue #112: deleting an in-flight run must reach delete_run with the
  // right runId. The Tauri command itself does the SIGTERM-then-rm-rf
  // dance (covered in the Rust tests); here we just assert the FE wires
  // the click → confirm → invoke chain correctly for a running row, the
  // case that previously left the subprocess alive.
  const runningRun = {
    run_id: "2026-04-29T12-00-00Z-infl",
    short_id: "infl",
    status: "running",
    mode: "tee",
    inputs: ["/x/y.md"],
    input_count: 1,
    created_at: "2026-04-29T12:00:00Z",
    progress: { stage: "extract", completed: 10, total: 50 },
  };

  it("selecting a running run + clicking Delete invokes delete_run with its runId", async () => {
    const user = userEvent.setup();
    const deleteCalls = [];
    await renderApp({
      runs: [runningRun],
      override: (cmd, args) => {
        if (cmd === "delete_run") {
          deleteCalls.push(args);
          return undefined;
        }
      },
    });

    // Select the run row (plain click — single selection). Run title
    // strips the trailing .md so the basename renders as "y".
    await user.click(screen.getByText("y"));

    // The bulk Delete (1) button should appear in the run-selection footer.
    const deleteBtn = await screen.findByRole("button", { name: /Delete \(1\)/ });
    await user.click(deleteBtn);

    // Confirm dialog appears; click "Yes, delete".
    const confirmBtn = await screen.findByRole("button", { name: /Yes, delete/ });
    await user.click(confirmBtn);

    await waitFor(() =>
      expect(deleteCalls).toEqual([{ runId: "2026-04-29T12-00-00Z-infl" }])
    );
  });
});

describe("App — pause/resume polling (no blind wait, no double-click)", () => {
  const runningRun = {
    run_id: "2026-04-29T12-00-00Z-pres",
    short_id: "pres",
    status: "running",
    mode: "tee",
    inputs: ["/x/y.md"],
    input_count: 1,
    created_at: "2026-04-29T12:00:00Z",
    progress: { stage: "extract", completed: 10, total: 50 },
  };

  it("doPause: shows 'Pausing…', polls until status flips to 'paused', then surfaces Resume", async () => {
    const user = userEvent.setup();
    let pauseInvoked = false;
    let runStatus = "running";
    await renderApp({
      runs: [runningRun],
      override: (cmd, args) => {
        if (cmd === "pause_run") {
          pauseInvoked = true;
          // Simulate the runner committing status=paused on disk
          // sometime AFTER the invoke returns. The frontend's
          // pollUntilRunStatus polls list_runs every 250ms; flip
          // the status mid-poll so the next list_runs sees paused.
          setTimeout(() => { runStatus = "paused"; }, 100);
          return undefined;
        }
        if (cmd === "list_runs") {
          return [{ ...runningRun, status: runStatus }];
        }
        return undefined;
      },
    });
    await user.click(screen.getByTitle("Pause"));
    expect(pauseInvoked).toBe(true);
    // Optimistic "Pausing…" badge appears immediately.
    await waitFor(() => expect(screen.getByText(/Pausing…/)).toBeTruthy());
    // Resume button NOT visible during the in-flight transition.
    expect(screen.queryByTitle("Resume")).toBeNull();
    // After the runner commits paused, the badge flips and Resume appears.
    await waitFor(
      () => expect(screen.getByTitle("Resume")).toBeTruthy(),
      { timeout: 3000 },
    );
    expect(screen.queryByText(/Pausing…/)).toBeNull();
  });

  it("doResume: shows 'Resuming…' until status=running; second click can't fire (button suppressed)", async () => {
    const user = userEvent.setup();
    let resumeCallCount = 0;
    let runStatus = "paused";
    const pausedRun = { ...runningRun, status: "paused" };
    await renderApp({
      runs: [pausedRun],
      override: (cmd, args) => {
        if (cmd === "resume_run") {
          resumeCallCount += 1;
          setTimeout(() => { runStatus = "running"; }, 200);
          return undefined;
        }
        if (cmd === "list_runs") {
          return [{ ...pausedRun, status: runStatus }];
        }
        return undefined;
      },
    });
    await user.click(screen.getByTitle("Resume"));
    expect(resumeCallCount).toBe(1);
    // "Resuming…" badge + Resume button hidden during polling.
    await waitFor(() => expect(screen.getByText(/Resuming…/)).toBeTruthy());
    expect(screen.queryByTitle("Resume")).toBeNull();
    // The user's second click in the gap pre-fix would fire resume_run
    // again and surface "already running". Post-fix the button isn't
    // there, so there's nothing to click — guarantee the count stays 1.
    expect(resumeCallCount).toBe(1);
    // After status flips, Pause re-appears (running) and the badge clears.
    await waitFor(
      () => expect(screen.getByTitle("Pause")).toBeTruthy(),
      { timeout: 3000 },
    );
    expect(screen.queryByText(/Resuming…/)).toBeNull();
  });
});

describe("normalizeNavRelPath", () => {
  // The fact serializer stores `evidence.file_path` as the original
  // input filename (e.g. "notes.txt", "README.md"). Stage 0 always
  // appends `.md` to the on-disk preprocessed copy, so the click
  // target needs `.md` post-fixed regardless of the original
  // extension. Direct stage-view paths (`1-facts/<topic>.md`,
  // `4-insights.md`) already match disk and only need `.md`
  // appended when missing.
  it("appends .md to 0-inputs/ paths whose original filename is not .md", () => {
    expect(normalizeNavRelPath("0-inputs/notes.txt")).toBe("0-inputs/notes.txt.md");
    expect(normalizeNavRelPath("0-inputs/image.jpg")).toBe("0-inputs/image.jpg.md");
  });

  it("appends .md to 0-inputs/ paths whose original filename IS .md (the README.md bug)", () => {
    // Pre-fix: the "skip if endsWith .md" branch left this unchanged
    // and read_run_file canonicalize-failed because the on-disk file
    // is README.md.md (Stage 0 doubles the suffix unconditionally).
    expect(normalizeNavRelPath("0-inputs/README.md")).toBe("0-inputs/README.md.md");
    expect(normalizeNavRelPath("0-inputs/notes.md")).toBe("0-inputs/notes.md.md");
  });

  it("leaves stage views (1-facts/<topic>.md etc.) untouched", () => {
    expect(normalizeNavRelPath("1-facts/work.md")).toBe("1-facts/work.md");
    expect(normalizeNavRelPath("2-entities/alice.md")).toBe("2-entities/alice.md");
    expect(normalizeNavRelPath("4-insights.md")).toBe("4-insights.md");
  });

  it("appends .md to non-0-inputs paths missing the extension (wikilink shorthand)", () => {
    expect(normalizeNavRelPath("1-facts/work")).toBe("1-facts/work.md");
    expect(normalizeNavRelPath("4-insights")).toBe("4-insights.md");
  });

  it("passes empty / null relPath through unchanged", () => {
    expect(normalizeNavRelPath("")).toBe("");
    expect(normalizeNavRelPath(null)).toBe(null);
    expect(normalizeNavRelPath(undefined)).toBe(undefined);
  });
});

describe("resourceNavTarget", () => {
  // record_id formats are minted by the embeddings stage. fact/pattern
  // anchors are NOT positional (type-scoped in the rendered markdown),
  // so the deterministic target leaves anchor "" + carries topic/idx
  // for resolveCitation to upgrade. chunk DOES carry a positional
  // anchor: its source char-offset (`<file>@<offset>`), surfaced as
  // the same `offset-<N>` scheme the input view uses for fact-evidence
  // back-links, so a chunk citation scrolls to the grounded passage
  // instead of the file top. insight/action anchors are positional
  // and final here.
  it("maps a chunk id to the input file at its source char-offset", () => {
    expect(resourceNavTarget("chunk", "notes.txt@1024")).toEqual({
      relPath: "0-inputs/notes.txt",
      anchor: "offset-1024",
      chunkOffset: 1024,
    });
  });

  it("falls back to file-level for a chunk id with a non-numeric offset", () => {
    expect(resourceNavTarget("chunk", "notes.txt@bogus")).toEqual({
      relPath: "0-inputs/notes.txt",
      anchor: "",
    });
  });

  it("maps fact / pattern ids to the topic file, anchor deferred", () => {
    expect(resourceNavTarget("fact", "work:0")).toEqual({
      relPath: "1-facts/work", anchor: "", topic: "work", idx: 0,
    });
    expect(resourceNavTarget("pattern", "health:4")).toEqual({
      relPath: "3-patterns/health", anchor: "", topic: "health", idx: 4,
    });
  });

  it("maps insight ids (scope:idx) with the cross_domain slug fix", () => {
    expect(resourceNavTarget("insight", "cross_domain:0")).toEqual({
      relPath: "4-insights", anchor: "cross-domain-1",
      scope: "cross_domain", idx: 0,
    });
    expect(resourceNavTarget("insight", "critical:2")).toEqual({
      relPath: "4-insights", anchor: "critical-3",
      scope: "critical", idx: 2,
    });
  });

  it("maps action / entity ids", () => {
    expect(resourceNavTarget("action", "3")).toEqual({
      relPath: "5-actions", anchor: "action-4", idx: 3,
    });
    expect(resourceNavTarget("entity", "alice-smith")).toEqual({
      relPath: "2-entities/alice-smith", anchor: "",
    });
  });

  it("maps a document id to its source file at the top (no anchor)", () => {
    // A document citation opens the whole file at the top — record_id is
    // the file_id, same 0-inputs target a chunk uses but with no offset.
    expect(resourceNavTarget("document", "alice-trip.txt")).toEqual({
      relPath: "0-inputs/alice-trip.txt", anchor: "",
    });
    expect(resourceNavTarget("document", "journals/2024/notes.md")).toEqual({
      relPath: "0-inputs/journals/2024/notes.md", anchor: "",
    });
  });

  it("tolerates a topic that itself contains a colon (rsplit on last)", () => {
    expect(resourceNavTarget("fact", "a:b:7")).toEqual({
      relPath: "1-facts/a:b", anchor: "", topic: "a:b", idx: 7,
    });
  });

  it("returns null for an unknown kind or a malformed id", () => {
    expect(resourceNavTarget("mystery", "x")).toBeNull();
    expect(resourceNavTarget("fact", "no-index")).toBeNull();
    expect(resourceNavTarget("chunk", "no-at-sign")).toBeNull();
    expect(resourceNavTarget("action", "notanumber")).toBeNull();
    expect(resourceNavTarget(null, "x")).toBeNull();
    expect(resourceNavTarget("fact", "")).toBeNull();
  });
});

describe("resolveCitation", () => {
  // The model cites by CONTEXT position; cited_refs already maps that
  // to the right (kind, record_id). The bug this guards: the rendered
  // markdown anchor is type-scoped (emotion-3), NOT fact-{idx+1}, so
  // the resolver must reuse the canonical obsidian scheme over the
  // bound run's ordered topic list — and degrade to the file head,
  // never a wrong/blank jump. Each test uses a distinct runId: the
  // resolver's fetch cache is module-scoped + keyed by (runId, kind,
  // topic), immutable per run by design, so collisions would mask the
  // per-test mock.
  it("upgrades a fact to the real type-scoped anchor + a titled label", async () => {
    const facts = [
      { item_type: "emotion", summary: "felt uneasy" },
      { item_type: "decision", summary: "chose to leave" },
      { item_type: "emotion", summary: "relief afterwards" },
    ];
    const invoke = vi.fn().mockResolvedValue(facts);
    const r = await resolveCitation("run-fact", "fact", "work:2", invoke);
    expect(invoke).toHaveBeenCalledWith("read_run_facts_for_topic", {
      runId: "run-fact", topic: "work",
    });
    // facts[2] is the 2nd "emotion" → emotion-2, NOT fact-3.
    expect(r).toEqual({
      relPath: "1-facts/work",
      anchor: "emotion-2",
      label: "fact · work · relief afterwards",
    });
  });

  it("upgrades a pattern with the kind-slug counter scheme", async () => {
    const pats = [
      { kind: "Recurring Theme", name: "overwork" },
      { kind: "Recurring Theme", name: "burnout cycle" },
    ];
    const invoke = vi.fn().mockResolvedValue(pats);
    const r = await resolveCitation("run-fact", "pattern", "work:1", invoke);
    expect(r.anchor).toBe("recurring-theme-2");
    expect(r.label).toBe("pattern · work · burnout cycle");
  });

  it("labels insight/action/entity from their run data, anchor unchanged", async () => {
    const invoke = vi.fn((cmd) => {
      if (cmd === "read_run_insights")
        return Promise.resolve({ critical: [{ name: "core tension" }] });
      if (cmd === "read_run_actions")
        return Promise.resolve({ actions: [{}, {}, {}, { recommendation: "ship it" }] });
      if (cmd === "read_run_entities")
        return Promise.resolve({
          subject: "e1",
          entities: [{ canonical_id: "e1", canonical_name: "Alice" }],
          relations: [],
        });
      return Promise.resolve(null);
    });
    expect(await resolveCitation("r", "insight", "critical:0", invoke))
      .toEqual({ relPath: "4-insights", anchor: "critical-1",
        label: "insight · core tension" });
    expect(await resolveCitation("r", "action", "3", invoke))
      .toEqual({ relPath: "5-actions", anchor: "action-4",
        label: "action · ship it" });
    expect(await resolveCitation("r", "entity", "e1", invoke))
      .toEqual({ relPath: "2-entities/e1", anchor: "",
        label: "entity · Alice" });
  });

  it("degrades to the file head + record_id label when run data is missing", async () => {
    const invoke = vi.fn().mockRejectedValue(new Error("not loaded"));
    const r = await resolveCitation("run-miss", "fact", "work:99", invoke);
    expect(r).toEqual({
      relPath: "1-facts/work", anchor: "", label: "fact · work:99",
    });
  });

  it("degrades to the file head, record_id label, when idx past list end", async () => {
    const invoke = vi.fn().mockResolvedValue([{ summary: "only one" }]);
    const r = await resolveCitation("run-end", "fact", "work:5", invoke);
    expect(r).toEqual({
      relPath: "1-facts/work", anchor: "", label: "fact · work:5",
    });
  });

  it("returns null for a malformed id and a file label with no run", async () => {
    expect(await resolveCitation("r", "bogus", "x", vi.fn())).toBeNull();
    // No run → no stage read, but the chunk's source offset is still
    // recoverable from the record_id, so the citation still targets the
    // grounded passage (anchor + chunkOffset), not the file top.
    expect(await resolveCitation(null, "chunk", "f.txt@10", vi.fn()))
      .toEqual({ relPath: "0-inputs/f.txt", anchor: "offset-10",
        chunkOffset: 10, label: "chunk · f.txt@10" });
  });
});

// ── Chat chunk citation → grounded passage (#536) ──────────────────────────
//
// WIRING GUARD ONLY (jsdom — scrollTo is a no-op here; this asserts the
// resolveCitation → handleMarkdownNavigate → InputFileView path puts an
// `offset-<N>` cited-chunk anchor on the grounded passage so the
// scroll-to-anchor effect HAS a target). The real broken→fixed
// acceptance is the real-browser-engine repro + the packaged WebKit
// .app click — a jsdom pass is necessary-but-NOT-sufficient.

describe("App — chat chunk citation anchors the grounded passage (#536)", () => {
  const run = {
    run_id: "2026-05-16T00-00-00Z-chnk", short_id: "chnk",
    status: "completed", mode: "tee", inputs: ["/u/notes.txt"],
    input_count: 1, created_at: "2026-05-16T00:00:00Z", duration_ms: 1000,
    progress: { stage: "done", completed: 1, total: 1 },
    provider: "tinfoil", model: "gpt-oss-120b", vault_exists: true,
  };
  // "HEADER LINE" is 11 chars, then "\n\n" (11,12); the grounded
  // passage starts at offset 13.
  const content =
    "HEADER LINE\n\nGROUNDED PASSAGE about taxes\n\nTAIL PARAGRAPH";

  it("a chunk citation opens the source file with an offset-anchored, highlighted passage (not the file top)", async () => {
    const handlers = {};
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      handlers[evt] = fn;
      return () => {};
    });

    await renderApp({
      runs: [run],
      override: (cmd, args) => {
        if (cmd === "list_run_tree") return [];
        if (cmd === "read_run_facts") return [];
        if (cmd === "read_run_file"
            && args.relPath === "0-inputs/notes.txt.md") return content;
        if (cmd === "chatbot_list_runs") return [
          { run_id: run.run_id, label: "notes", store_path: "/p", bound: true },
        ];
        if (cmd === "chatbot_select_run") return undefined;
        if (cmd === "chatbot") return undefined;
        return undefined;
      },
    });

    await userEvent.click(document.querySelector(`[data-run-id="${run.run_id}"]`));
    await userEvent.click(screen.getByTestId("chatbot-helper-pill"));
    await waitFor(() => expect(handlers["chatbot-event"]).toBeTruthy());
    act(() => {
      handlers["chatbot-event"]({ payload: { event: "chatbot_bound", run: run.run_id } });
    });
    await userEvent.type(
      screen.getByTestId("chatbot-helper-input"), "what about taxes?",
    );
    await userEvent.click(screen.getByTestId("chatbot-helper-send"));
    act(() => {
      handlers["chatbot-event"]({ payload: { event: "chatbot_chunk", delta: "Per [1]" } });
    });
    act(() => {
      handlers["chatbot-event"]({
        payload: {
          event: "chatbot_done",
          resources: [{ index: 1, kind: "chunk", record_id: "notes.txt@13" }],
        },
      });
    });

    const openBtn = await screen.findByTestId("chatbot-helper-resource-open-1");
    await userEvent.click(openBtn);

    // The grounded passage carries the `offset-13` anchor (the
    // scroll-to-anchor effect's target) AND the cited-chunk highlight —
    // pre-fix the chunk resolved to anchor "" and opened at the top.
    await waitFor(() => {
      const mark = document.querySelector("mark.cited-chunk#offset-13");
      expect(mark).toBeTruthy();
      expect(mark.textContent).toContain("GROUNDED PASSAGE about taxes");
    });
    // The header before the passage is NOT inside the highlight.
    const mark = document.querySelector("mark.cited-chunk#offset-13");
    expect(mark.textContent).not.toContain("HEADER LINE");
  });

  it("cross-run: a chunk citation switches to the bound run AND lands the file populated (not empty)", async () => {
    // Regression guard for the #539-family wipe in the cross-run case:
    // with the WRONG run globally selected, clicking a chunk citation
    // switches to the bound run (director-wanted) but the
    // selectedRun-change reset used to setSelectedFile(null) in the
    // same commit → file pane empty. jsdom can't assert the scroll
    // (no-op); it CAN assert run-switch + file populated + anchor
    // present, which is exactly what regressed.
    const runB = {
      run_id: "2026-05-16T00-00-00Z-othr", short_id: "othr",
      status: "completed", mode: "tee", inputs: ["/u/other.txt"],
      input_count: 1, created_at: "2026-05-16T00:00:00Z", duration_ms: 1,
      progress: { stage: "done", completed: 1, total: 1 },
      provider: "tinfoil", model: "gpt-oss-120b", vault_exists: true,
    };
    const handlers = {};
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      handlers[evt] = fn;
      return () => {};
    });
    await renderApp({
      runs: [run, runB],
      override: (cmd, args) => {
        if (cmd === "list_run_tree") return [];
        if (cmd === "read_run_facts") return [];
        // Content ONLY for the bound run's input file — if the file
        // resolved against the wrong run (or got wiped) the passage
        // text would be absent.
        if (cmd === "read_run_file"
            && args.runId === run.run_id
            && args.relPath === "0-inputs/notes.txt.md") return content;
        if (cmd === "chatbot_list_runs") return [
          { run_id: run.run_id, label: "notes", store_path: "/p", bound: true },
          { run_id: runB.run_id, label: "other", store_path: "/q", bound: false },
        ];
        if (cmd === "chatbot_select_run") return undefined;
        if (cmd === "chatbot") return undefined;
        return undefined;
      },
    });

    // Globally select the WRONG run (B).
    await userEvent.click(document.querySelector(`[data-run-id="${runB.run_id}"]`));
    await waitFor(() => {
      expect(
        document.querySelector(`[data-run-id="${runB.run_id}"]`).className,
      ).toContain("selected");
    });

    await userEvent.click(screen.getByTestId("chatbot-helper-pill"));
    await waitFor(() => expect(handlers["chatbot-event"]).toBeTruthy());
    // Chat is bound to run A (≠ the globally-selected B).
    act(() => {
      handlers["chatbot-event"]({ payload: { event: "chatbot_bound", run: run.run_id } });
    });
    await userEvent.type(
      screen.getByTestId("chatbot-helper-input"), "taxes?",
    );
    await userEvent.click(screen.getByTestId("chatbot-helper-send"));
    act(() => {
      handlers["chatbot-event"]({ payload: { event: "chatbot_chunk", delta: "Per [1]" } });
    });
    act(() => {
      handlers["chatbot-event"]({
        payload: {
          event: "chatbot_done",
          resources: [{ index: 1, kind: "chunk", record_id: "notes.txt@13" }],
        },
      });
    });
    const openBtn = await screen.findByTestId("chatbot-helper-resource-open-1");
    await userEvent.click(openBtn);

    // Director-wanted run-switch happened: bound run A is now selected,
    // wrong run B is not.
    await waitFor(() => {
      expect(
        document.querySelector(`[data-run-id="${run.run_id}"]`).className,
      ).toContain("selected");
    });
    expect(
      document.querySelector(`[data-run-id="${runB.run_id}"]`).className,
    ).not.toContain("selected");
    // The file LANDED POPULATED (the defect: it was empty) — passage
    // text + the offset anchor are present in the markdown pane.
    await waitFor(() => {
      const mark = document.querySelector("mark.cited-chunk#offset-13");
      expect(mark).toBeTruthy();
      expect(mark.textContent).toContain("GROUNDED PASSAGE about taxes");
    });
  });

  it("a genuine manual run-switch still clears the prior run's open file (anti-bleed intact)", async () => {
    // The conditional clear must NOT weaken issue #83: clicking a
    // different run in the Runs pane (no citation/navigateToFile) must
    // still drop the previously-open file, which belongs to the OLD
    // run.
    const runB = {
      run_id: "2026-05-16T00-00-00Z-bbb2", short_id: "bbb2",
      status: "completed", mode: "tee", inputs: ["/u/other.txt"],
      input_count: 1, created_at: "2026-05-16T00:00:00Z", duration_ms: 1,
      progress: { stage: "done", completed: 1, total: 1 },
      provider: "tinfoil", model: "gpt-oss-120b", vault_exists: true,
    };
    const handlers = {};
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      handlers[evt] = fn;
      return () => {};
    });
    await renderApp({
      runs: [run, runB],
      override: (cmd, args) => {
        if (cmd === "list_run_tree") return [];
        if (cmd === "read_run_facts") return [];
        if (cmd === "read_run_file"
            && args.runId === run.run_id
            && args.relPath === "0-inputs/notes.txt.md") return content;
        if (cmd === "chatbot_list_runs") return [
          { run_id: run.run_id, label: "notes", store_path: "/p", bound: true },
        ];
        if (cmd === "chatbot_select_run") return undefined;
        if (cmd === "chatbot") return undefined;
        return undefined;
      },
    });
    // Open run A's input file via a chunk citation (bound==selected: A).
    await userEvent.click(document.querySelector(`[data-run-id="${run.run_id}"]`));
    await userEvent.click(screen.getByTestId("chatbot-helper-pill"));
    await waitFor(() => expect(handlers["chatbot-event"]).toBeTruthy());
    act(() => {
      handlers["chatbot-event"]({ payload: { event: "chatbot_bound", run: run.run_id } });
    });
    await userEvent.type(screen.getByTestId("chatbot-helper-input"), "taxes?");
    await userEvent.click(screen.getByTestId("chatbot-helper-send"));
    act(() => {
      handlers["chatbot-event"]({ payload: { event: "chatbot_chunk", delta: "Per [1]" } });
    });
    act(() => {
      handlers["chatbot-event"]({
        payload: {
          event: "chatbot_done",
          resources: [{ index: 1, kind: "chunk", record_id: "notes.txt@13" }],
        },
      });
    });
    const openBtn = await screen.findByTestId("chatbot-helper-resource-open-1");
    await userEvent.click(openBtn);
    await waitFor(() => {
      expect(document.querySelector("mark.cited-chunk#offset-13")).toBeTruthy();
    });

    // Now MANUALLY switch to run B (no citation). The stale A file
    // must be dropped — pane no longer shows the A passage.
    await userEvent.click(document.querySelector(`[data-run-id="${runB.run_id}"]`));
    await waitFor(() => {
      expect(document.querySelector("mark.cited-chunk#offset-13")).toBeNull();
    });
    expect(screen.queryByText(/GROUNDED PASSAGE about taxes/)).toBeNull();
  });
});

// ── Run-details modal (issue #104) ─────────────────────────────────────────
//
// The Details (ⓘ) button next to each run name opens a modal that
// reads `read_run_llm_stats(run_id)` and renders the per-stage outcome
// summary + per-call rows with retry-chain expand/collapse. These
// tests pin the wiring so a regression in either the trigger or the
// modal contents fails loud.

describe("App — run-details modal (issue #104)", () => {
  const completedRun = {
    run_id: "2026-04-29T12-00-00Z-d104",
    short_id: "d104",
    status: "completed",
    mode: "tee",
    inputs: ["/x/y.md"],
    input_count: 1,
    created_at: "2026-04-29T12:00:00Z",
    duration_ms: 30_000,
    progress: { stage: "done", completed: 50, total: 50 },
  };

  const sampleStats = {
    schema: "llm-stats/v1",
    run_id: "2026-04-29T12-00-00Z-d104",
    totals: {
      calls: 3,
      outcomes: {
        success: 1, success_empty: 2, blanked: 0,
        parse_error: 0, timeout: 0, other_failure: 0,
      },
    },
    per_stage: {
      patterns: {
        name: "patterns",
        calls_total: 3,
        calls_failed: 0,
        calls_aborted: 0,
        calls_cached: 0,
        outcomes: {
          success: 1, success_empty: 2, blanked: 0,
          parse_error: 0, timeout: 0, other_failure: 0,
        },
        models_used: ["fixture"],
        prompt_tokens: { total: 3000, avg: 1000, median: 1000, min: 1000, max: 1000 },
        completion_tokens: { total: 600, avg: 200, median: 200, min: 200, max: 200 },
        duration_ms: { total: 3000, avg: 1000, median: 1000, min: 1000, max: 1000 },
      },
    },
    calls: [
      // Retry chain: 0001 → 0002 (retry of 0001), 0002 succeeds.
      { call_id: "0001", stage: "patterns", category: "work", model: "fixture",
        outcome: "timeout", attempt: 1, retry_of_call_id: null, cached: false,
        prompt_tokens: 1000, completion_tokens: 0, duration_ms: 60_000,
        template_hash: "abc123abcdef", output: null,
        error: { class: "APITimeoutError", message: "timed out" } },
      { call_id: "0002", stage: "patterns", category: "work", model: "fixture",
        outcome: "success", attempt: 2, retry_of_call_id: "0001", cached: false,
        prompt_tokens: 1000, completion_tokens: 200, duration_ms: 1000,
        template_hash: "abc123abcdef", output: { patterns: 2 },
        error: null },
      // Single call, success_empty (model returned []).
      { call_id: "0003", stage: "patterns", category: "health", model: "fixture",
        outcome: "success_empty", attempt: 1, retry_of_call_id: null, cached: false,
        prompt_tokens: 1000, completion_tokens: 200, duration_ms: 1000,
        template_hash: "abc123abcdef", output: { patterns: 0 },
        error: null },
    ],
  };

  // The Details (⋯) button lives in the MIDDLE pane (RunTreePane)
  // next to "click to rename", not in the runs-list RunRow. Tests
  // need to first click the run row to make it the current run, so
  // RunTreePane mounts and the Details button appears. This helper
  // wraps that two-step. Uses the status badge as a stable click
  // target — `.run-title` text varies per fixture; "Completed" /
  // "Running" is on every row of that status.
  async function openRunDetailsModal(user) {
    const row = screen.getByText("Completed").closest(".run-row");
    await user.click(row);
    await waitFor(() => expect(screen.getByTitle(/Open run details/)).toBeTruthy());
    await user.click(screen.getByTitle(/Open run details/));
  }

  it("Details (⋯) button appears in the middle pane next to the run name", async () => {
    const user = userEvent.setup();
    await renderApp({ runs: [completedRun] });
    // Pre-click: button NOT visible (no run selected → middle pane is StagingPane).
    expect(screen.queryByTitle(/Open run details/)).toBeNull();
    const row = screen.getByText("Completed").closest(".run-row");
    await user.click(row);
    // Post-click: middle pane swapped to RunTreePane; button is there.
    await waitFor(() =>
      expect(screen.getByTitle(/Open run details/)).toBeTruthy()
    );
  });

  it("clicking Details opens the modal and invokes read_run_llm_stats", async () => {
    const user = userEvent.setup();
    let statsCalledWith = null;
    await renderApp({
      runs: [completedRun],
      override: (cmd, args) => {
        if (cmd === "read_run_llm_stats") {
          statsCalledWith = args;
          return sampleStats;
        }
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => {
      expect(screen.getByText(/Run details — /)).toBeTruthy();
    });
    expect(statsCalledWith).toEqual({ runId: "2026-04-29T12-00-00Z-d104" });
    // Per-stage summary table renders with outcome buckets + drop count.
    expect(screen.getByText("Per-stage summary")).toBeTruthy();
    // "patterns" appears multiple times (stage row + per-call rows);
    // assert at least one match rather than uniqueness.
    expect(screen.getAllByText("patterns").length).toBeGreaterThan(0);
  });

  it("modal renders per-call rows and collapses retry chains by default", async () => {
    const user = userEvent.setup();
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return sampleStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    // Three calls but two chains. Default-collapsed: only chain-final
    // row + single call visible.
    expect(screen.getByText(/0002/)).toBeTruthy();
    expect(screen.getByText(/0003/)).toBeTruthy();
    expect(screen.getByText(/\(try 2\)/)).toBeTruthy();
    expect(screen.queryByText(/0001/)).toBeNull();
  });

  it("clicking a chain root expands the retry attempts", async () => {
    const user = userEvent.setup();
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return sampleStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/\(try 2\)/)).toBeTruthy());
    const chainRoot = screen.getByText(/\(try 2\)/).closest("tr");
    expect(chainRoot).toBeTruthy();
    await user.click(chainRoot);
    expect(screen.getByText(/0001/)).toBeTruthy();
    expect(screen.getByText(/0002/)).toBeTruthy();
  });

  // Issue #105 v3 follow-up #2: extract halve trees produce ONE root
  // with TWO leaves (parallel halve children). Pre-fix the modal
  // grouped by root and showed only the chain's last attempt by
  // attempt-asc, silently dropping a sibling half. Post-fix each
  // leaf gets its own row.
  it("halve tree renders both leaves as separate rows by default", async () => {
    const user = userEvent.setup();
    const halveStats = {
      ...sampleStats,
      calls: [
        // Root (split_00) timed out → halved.
        { call_id: "0001", stage: "extract", category: "split_00",
          model: "fixture", outcome: "timeout", attempt: 1,
          retry_of_call_id: null, cached: false, prompt_tokens: 1000,
          completion_tokens: 0, duration_ms: 60_000,
          template_hash: "h1", output: null,
          error: { class: "APITimeoutError", message: "t" } },
        // Half-1 succeeded.
        { call_id: "0002", stage: "extract", category: "split_00/half-1",
          model: "fixture", outcome: "success", attempt: 1,
          retry_of_call_id: "0001", cached: false, prompt_tokens: 500,
          completion_tokens: 100, duration_ms: 800,
          template_hash: "h1", output: { facts: 3 }, error: null },
        // Half-2 succeeded.
        { call_id: "0003", stage: "extract", category: "split_00/half-2",
          model: "fixture", outcome: "success", attempt: 1,
          retry_of_call_id: "0001", cached: false, prompt_tokens: 500,
          completion_tokens: 100, duration_ms: 900,
          template_hash: "h1", output: { facts: 4 }, error: null },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return halveStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    // Both leaves visible (0002, 0003); root (0001) hidden until expand.
    expect(screen.getByText(/0002/)).toBeTruthy();
    expect(screen.getByText(/0003/)).toBeTruthy();
    expect(screen.queryByText(/0001/)).toBeNull();
    // Each leaf shows a "(try 2)" suffix because both chains have
    // the same shape (root + leaf, leaf at position 2). User dropped
    // the "×N" prefix in favor of (try N) per b889 feedback.
    const tryMarks = screen.getAllByText(/\(try 2\)/);
    expect(tryMarks.length).toBe(2);
  });

  it("clicking a halve-tree leaf expands its ancestor chain (oldest-first)", async () => {
    const user = userEvent.setup();
    const halveStats = {
      ...sampleStats,
      calls: [
        { call_id: "0001", stage: "extract", category: "split_00",
          model: "fixture", outcome: "timeout", attempt: 1,
          retry_of_call_id: null, cached: false, prompt_tokens: 1000,
          completion_tokens: 0, duration_ms: 60_000,
          template_hash: "h1", output: null,
          error: { class: "APITimeoutError", message: "t" } },
        { call_id: "0002", stage: "extract", category: "split_00/half-1",
          model: "fixture", outcome: "success", attempt: 1,
          retry_of_call_id: "0001", cached: false, prompt_tokens: 500,
          completion_tokens: 100, duration_ms: 800,
          template_hash: "h1", output: { facts: 3 }, error: null },
        { call_id: "0003", stage: "extract", category: "split_00/half-2",
          model: "fixture", outcome: "success", attempt: 1,
          retry_of_call_id: "0001", cached: false, prompt_tokens: 500,
          completion_tokens: 100, duration_ms: 900,
          template_hash: "h1", output: { facts: 4 }, error: null },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return halveStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(
      () => expect(screen.getAllByText(/\(try 2\)/).length).toBe(2));
    // Click on the first half (0002) — its ancestor chain expands to
    // show 0001 above it.
    const leafRow = screen.getByText(/0002/).closest("tr");
    await user.click(leafRow);
    // 0001 now visible; 0003 still visible (it's the second leaf).
    expect(screen.getByText(/0001/)).toBeTruthy();
    expect(screen.getByText(/0003/)).toBeTruthy();
  });

  // #963: leaf ordering must key on each chain's ROOT so two leaves
  // sharing a root render adjacently even when a second root's leaves
  // are interleaved in the calls array. Chains are newest-first (leaf
  // at index 0, root last); the pre-fix code keyed on chain[0] (the
  // leaf), so every leaf became its own "root" group and same-root
  // halve siblings split apart. Two roots (A=0001, B=0002), each halved
  // into two successful leaves, interleaved on the wire.
  it("groups halve siblings by root when two roots interleave (#963)", async () => {
    const user = userEvent.setup();
    const twoRootStats = {
      ...sampleStats,
      calls: [
        { call_id: "0001", stage: "extract", category: "split_00",
          model: "fixture", outcome: "timeout", attempt: 1,
          retry_of_call_id: null, cached: false, prompt_tokens: 1000,
          completion_tokens: 0, duration_ms: 60_000,
          template_hash: "h1", output: null,
          error: { class: "APITimeoutError", message: "t" } },
        { call_id: "0002", stage: "extract", category: "split_01",
          model: "fixture", outcome: "timeout", attempt: 1,
          retry_of_call_id: null, cached: false, prompt_tokens: 1000,
          completion_tokens: 0, duration_ms: 60_000,
          template_hash: "h1", output: null,
          error: { class: "APITimeoutError", message: "t" } },
        // A's two halves, then B's two halves — interleaved so a
        // leaf-keyed order would emit 0003,0004,0005,0006 (A,B,A,B).
        { call_id: "0003", stage: "extract", category: "split_00/half-1",
          model: "fixture", outcome: "success", success: true, attempt: 2,
          retry_of_call_id: "0001", cached: false, prompt_tokens: 500,
          completion_tokens: 100, duration_ms: 800,
          template_hash: "h1", output: { facts: 3 }, error: null },
        { call_id: "0004", stage: "extract", category: "split_01/half-1",
          model: "fixture", outcome: "success", success: true, attempt: 2,
          retry_of_call_id: "0002", cached: false, prompt_tokens: 500,
          completion_tokens: 100, duration_ms: 850,
          template_hash: "h1", output: { facts: 4 }, error: null },
        { call_id: "0005", stage: "extract", category: "split_00/half-2",
          model: "fixture", outcome: "success", success: true, attempt: 2,
          retry_of_call_id: "0001", cached: false, prompt_tokens: 500,
          completion_tokens: 100, duration_ms: 900,
          template_hash: "h1", output: { facts: 5 }, error: null },
        { call_id: "0006", stage: "extract", category: "split_01/half-2",
          model: "fixture", outcome: "success", success: true, attempt: 2,
          retry_of_call_id: "0002", cached: false, prompt_tokens: 500,
          completion_tokens: 100, duration_ms: 950,
          template_hash: "h1", output: { facts: 6 }, error: null },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return twoRootStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    const leafIds = Array.from(
      document.querySelectorAll(".run-details-call-row")
    )
      .map((r) => r.children[0].textContent.match(/(\d{4})/)?.[1])
      .filter(Boolean);
    // Roots (0001, 0002) are hidden until expand; only the four leaves
    // render. Same-root siblings adjacent, root A before root B.
    expect(leafIds).toEqual(["0003", "0005", "0004", "0006"]);
  });

  // #963: a deep halve tree's leaves SHARE a trunk (root → first split). When
  // two leaves are expanded at once, that shared trunk must be drawn ONCE, not
  // once per expanded leaf (the "stray rows" pile-up). 0003 splits into {0004
  // half-1, 0005 half-2}; 0004 splits into {0006, 0007}. Leaves 0006 + 0007
  // both share 0004→0003→0002→0001.
  it("draws a shared retry trunk once when two leaves are expanded (#963)", async () => {
    const user = userEvent.setup();
    const mk = (id, ro, cat, outcome) => ({
      call_id: id, stage: "extract", category: cat, model: "fixture",
      outcome, attempt: 1, retry_of_call_id: ro, cached: false,
      prompt_tokens: 1000, completion_tokens: outcome === "ok" ? 100 : 0,
      duration_ms: 1000, template_hash: "h1",
      output: outcome === "ok" ? { facts: 1 } : null,
      error: outcome === "ok" ? null : { class: "X", message: "x" },
    });
    const treeStats = {
      ...sampleStats,
      calls: [
        mk("0001", null, "split_00::head", "failed (load)"),
        mk("0002", "0001", "split_00::head - retry/load", "failed (other)"),
        mk("0003", "0002", "split_00::head - retry/other", "failed (sizing)"),
        mk("0004", "0003", "split_00::head/half-1 - retry/sizing", "failed (sizing)"),
        mk("0005", "0003", "split_00::head/half-2 - retry/sizing", "ok"),
        mk("0006", "0004", "split_00::head/half-1/half-1 - retry/sizing", "ok"),
        mk("0007", "0004", "split_00::head/half-1/half-2 - retry/sizing", "ok"),
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return treeStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    // Expand both sibling branch-leaves under 0004.
    await user.click(screen.getByText(/0006/).closest("tr"));
    await user.click(screen.getByText(/0007/).closest("tr"));
    const rowIds = Array.from(document.querySelectorAll(".run-details-call-row"))
      .map((r) => r.children[0].textContent.match(/(\d{4})/)?.[1])
      .filter(Boolean);
    const count = (id) => rowIds.filter((x) => x === id).length;
    // Shared trunk (0004, 0003, 0002, 0001) drawn exactly once; both unique
    // leaves (0006, 0007) present.
    expect(count("0004")).toBe(1);
    expect(count("0003")).toBe(1);
    expect(count("0002")).toBe(1);
    expect(count("0001")).toBe(1);
    expect(count("0006")).toBe(1);
    expect(count("0007")).toBe(1);
  });

  // #761: the many per-chunk embedding calls fold under ONE collapsible
  // parent row (default collapsed) showing pass/fail + a token sum,
  // reusing the retry collapse/expand machinery.
  const embStats = {
    ...sampleStats,
    calls: [
      // A non-embedding call stays a normal top-level row.
      { call_id: "0001", stage: "extract", category: "split_00",
        model: "fixture", outcome: "success", success: true, attempt: 1,
        retry_of_call_id: null, cached: false, prompt_tokens: 100,
        content_tokens: 50, duration_ms: 800, template_hash: "h1",
        output: { facts: 2 }, error: null },
      { call_id: "0002", stage: "embeddings", category: "embedding-0",
        model: "nomic-embed-text", outcome: "success", success: true,
        attempt: 1, retry_of_call_id: null, cached: false,
        prompt_tokens: 30, duration_ms: 50,
        output: null, error: null },
      { call_id: "0003", stage: "embeddings", category: "embedding-1",
        model: "nomic-embed-text", outcome: "success", success: true,
        attempt: 1, retry_of_call_id: null, cached: true,
        prompt_tokens: 40, duration_ms: 0,
        output: null, error: null },
      { call_id: "0004", stage: "embeddings", category: "embedding-2",
        model: "nomic-embed-text", outcome: "failed (load)", success: false,
        attempt: 1, retry_of_call_id: null, cached: false,
        prompt_tokens: 20, duration_ms: 100,
        output: null, error: { class: "X", message: "boom" } },
    ],
  };

  it("embeddings collapse under a parent row (default collapsed) with pass/fail + token sum", async () => {
    const user = userEvent.setup();
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return embStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    // Parent summary: 2 passed / 1 failed, token sum 30+40+20 = 90.
    expect(screen.getByText(/2 passed \/ 1 failed/)).toBeTruthy();
    expect(screen.getByText(/90 tok/)).toBeTruthy();
    expect(screen.getByText(/embeddings \(3\)/)).toBeTruthy();
    // Collapsed by default: individual embedding rows hidden; the
    // non-embedding call still renders.
    expect(screen.queryByText(/0002/)).toBeNull();
    expect(screen.queryByText(/0004/)).toBeNull();
    expect(screen.getByText(/0001/)).toBeTruthy();
  });

  it("expanding the embeddings parent reveals the individual calls", async () => {
    const user = userEvent.setup();
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return embStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/embeddings \(3\)/)).toBeTruthy());
    const parentRow = screen.getByText(/embeddings \(3\)/).closest("tr");
    await user.click(parentRow);
    // Individual embedding rows now visible.
    expect(screen.getByText(/0002/)).toBeTruthy();
    expect(screen.getByText(/0004/)).toBeTruthy();
  });

  it("(try N) suffix uses chain-position not c.attempt", async () => {
    // Regression for msgj b889 screenshot: ancestor rows 0003 and 0010
    // both showed "(try 1)" because the JSX was reading c.attempt
    // (wrapper-INTERNAL counter, resets to 1 on each fresh wrapper
    // invocation). For halve children the wrapper restarts, so two
    // ancestor rows in the same chain both had attempt=1. Now (try N)
    // is computed from chain position (1-based, oldest-first).
    const user = userEvent.setup();
    const recursiveStats = {
      ...sampleStats,
      calls: [
        // Root timed out.
        { call_id: "0001", stage: "extract", category: "split_00",
          model: "fixture", outcome: "timeout", attempt: 1,
          retry_of_call_id: null, cached: false, prompt_tokens: 1000,
          completion_tokens: 0, duration_ms: 60_000,
          template_hash: "h1", output: null,
          error: { class: "APITimeoutError", message: "t" } },
        // Halve depth-1 child also timed out. Wrapper-internal
        // attempt=1 (fresh wrapper for this child), but its CHAIN
        // position is 2.
        { call_id: "0003", stage: "extract", category: "split_00/half-1",
          model: "fixture", outcome: "timeout", attempt: 1,
          retry_of_call_id: "0001", cached: false, prompt_tokens: 500,
          completion_tokens: 0, duration_ms: 60_000,
          template_hash: "h1", output: null,
          error: { class: "APITimeoutError", message: "t" } },
        // Halve depth-2 child succeeded — chain position 3.
        { call_id: "0010", stage: "extract", category: "split_00/half-1/half-1",
          model: "fixture", outcome: "success", attempt: 1,
          retry_of_call_id: "0003", cached: false, prompt_tokens: 250,
          completion_tokens: 100, duration_ms: 800,
          template_hash: "h1", output: { facts: 2 }, error: null },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return recursiveStats;
      },
    });
    await openRunDetailsModal(user);
    // Click leaf to expand.
    await waitFor(() => expect(screen.getByText(/0010/)).toBeTruthy());
    const leafRow = screen.getByText(/0010/).closest("tr");
    await user.click(leafRow);
    // After expand: 0001 → "(try 1)", 0003 → "(try 2)", 0010 → "(try 3)".
    // Pre-fix every row showed (try 1) because all had attempt=1.
    expect(screen.getByText(/0001/).textContent).toMatch(/\(try 1\)/);
    expect(screen.getByText(/0003/).textContent).toMatch(/\(try 2\)/);
    expect(screen.getByText(/0010/).textContent).toMatch(/\(try 3\)/);
  });

  it("sampled outcome renders distinct pill style and label", async () => {
    const user = userEvent.setup();
    const sampledStats = {
      ...sampleStats,
      calls: [
        // Parent failed; sample-1 succeeded with reduced facts. The
        // pipeline's _apply_chain_aware_outcomes converts the leaf's
        // outcome to "sampled".
        { call_id: "0001", stage: "patterns", category: "logistics",
          model: "fixture", outcome: "timeout", attempt: 1,
          retry_of_call_id: null, cached: false, prompt_tokens: 1000,
          completion_tokens: 0, duration_ms: 60_000,
          template_hash: "h1", output: null,
          error: { class: "APITimeoutError", message: "t" } },
        { call_id: "0002", stage: "patterns",
          category: "logistics/sample-1",
          model: "fixture", outcome: "sampled", attempt: 1,
          retry_of_call_id: "0001", cached: false, prompt_tokens: 500,
          completion_tokens: 100, duration_ms: 800,
          template_hash: "h1", output: { patterns: 1 }, error: null },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return sampledStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/0002/)).toBeTruthy());
    // The leaf (0002) renders the "sampled" outcome pill.
    const leafRow = screen.getByText(/0002/).closest("tr");
    const pill = leafRow.querySelector(".outcome-pill.outcome-sampled");
    expect(pill).toBeTruthy();
    expect(pill.textContent).toBe("sampled");
  });

  it("Escape key closes the modal", async () => {
    const user = userEvent.setup();
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return sampleStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Run details — /)).toBeTruthy());
    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByText(/Run details — /)).toBeNull());
  });

  // ⚠ icon, drops column, and warning-render edge cases.

  it("⚠ icon renders on a non-success leaf (timeout)", async () => {
    const user = userEvent.setup();
    const failedStats = {
      ...sampleStats,
      calls: [
        { call_id: "0001", stage: "patterns", category: "logistics",
          model: "fixture", outcome: "timeout", attempt: 1,
          retry_of_call_id: null, cached: false, prompt_tokens: 1000,
          completion_tokens: 0, duration_ms: 60_000,
          template_hash: "h1", output: null,
          error: { class: "APITimeoutError", message: "t" } },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return failedStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/0001/)).toBeTruthy());
    const row = screen.getByText(/0001/).closest("tr");
    const icon = row.querySelector(".run-details-warn-icon");
    expect(icon).toBeTruthy();
    expect(icon.textContent.trim()).toBe("⚠");
  });

  it("⚠ icon does NOT render on a pending leaf", async () => {
    // Regression: pre-fix the JSX condition was outcome !== "success",
    // which marked in-flight (outcome="pending") leaves as warnings.
    // Fix excludes "pending" too — running calls don't flag.
    const user = userEvent.setup();
    const pendingStats = {
      ...sampleStats,
      calls: [
        { call_id: "0001", stage: "extract", category: "split_00",
          model: "fixture", outcome: "pending",
          retry_of_call_id: null, cached: false, prompt_tokens: 1000,
          completion_tokens: 0, duration_ms: null,
          template_hash: "h1", output: null, error: null,
          pending: true, success: null },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return pendingStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/0001/)).toBeTruthy());
    const row = screen.getByText(/0001/).closest("tr");
    expect(row.querySelector(".run-details-warn-icon")).toBeNull();
  });

  it("⚠ icon does NOT render on a success leaf", async () => {
    const user = userEvent.setup();
    const okStats = {
      ...sampleStats,
      calls: [
        { call_id: "0001", stage: "patterns", category: "work",
          model: "fixture", outcome: "success", attempt: 1,
          retry_of_call_id: null, cached: false, prompt_tokens: 1000,
          completion_tokens: 200, duration_ms: 1000,
          template_hash: "h1", output: { patterns: 2 }, error: null },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return okStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/0001/)).toBeTruthy());
    const row = screen.getByText(/0001/).closest("tr");
    expect(row.querySelector(".run-details-warn-icon")).toBeNull();
  });

  it("⚠ icon renders BEFORE the expand chevron (leftmost glyph)", async () => {
    // User: "exclamation mark … in front of the down expand arrow."
    // Pin the cell text order so a future JSX shuffle doesn't put
    // the chevron back ahead of the warning.
    const user = userEvent.setup();
    const halveTimeoutStats = {
      ...sampleStats,
      calls: [
        { call_id: "0001", stage: "extract", category: "split_00",
          model: "fixture", outcome: "timeout", attempt: 1,
          retry_of_call_id: null, cached: false, prompt_tokens: 1000,
          completion_tokens: 0, duration_ms: 60_000,
          template_hash: "h1", output: null,
          error: { class: "APITimeoutError", message: "t" } },
        // Halve child also failed → leaf chain has 2 calls, leaf is
        // a timeout → both expand chevron AND warning icon render.
        { call_id: "0002", stage: "extract",
          category: "split_00/half-1", model: "fixture",
          outcome: "timeout", attempt: 1,
          retry_of_call_id: "0001", cached: false, prompt_tokens: 500,
          completion_tokens: 0, duration_ms: 60_000,
          template_hash: "h1", output: null,
          error: { class: "APITimeoutError", message: "t" } },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return halveTimeoutStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/0002/)).toBeTruthy());
    const leafRow = screen.getByText(/0002/).closest("tr");
    const cellText = leafRow.querySelector("td").textContent;
    // Order check: the ⚠ glyph must appear before the ▸ (or ▾)
    // chevron in the call cell's text content.
    const warnIdx = cellText.indexOf("⚠");
    const chevIdx = Math.max(
      cellText.indexOf("▸"),
      cellText.indexOf("▾"),
    );
    expect(warnIdx).toBeGreaterThanOrEqual(0);
    expect(chevIdx).toBeGreaterThanOrEqual(0);
    expect(warnIdx).toBeLessThan(chevIdx);
  });

  it("per-call and per-stage tables both omit any 'drops' / 'singleton drops' column", async () => {
    // Drop counts aren't surfaced anywhere in the run-details view.
    const user = userEvent.setup();
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return sampleStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    const perCallSection = screen.getByText(/Per-call/).parentElement;
    const ths = Array.from(
      perCallSection.querySelectorAll(".run-details-calls thead th")
    );
    const headers = ths.map((th) => th.textContent.trim());
    expect(headers).not.toContain("drops");
    const stageHeaders = Array.from(
      document.querySelectorAll(".run-details-stages thead th")
    ).map((th) => th.textContent.trim());
    expect(stageHeaders).not.toContain("singleton drops");
  });

  it("'wait time' column: TTFT for wire calls; '—' for cached / pending / aborted / skipped rows", async () => {
    // Spec (#346): the column reads c.ttft_ms but renders '—' for
    // rows where the value isn't a meaningful "how long before
    // anything happened" measurement — including aborted rows that
    // may still carry a stamped ttft_ms because the first token
    // arrived before the abort fired.
    const user = userEvent.setup();
    const statsWithWaitTimes = {
      ...sampleStats,
      calls: [
        // Normal wire call: sub-second TTFT renders as "Nms".
        { call_id: "1001", stage: "patterns", category: "work", model: "fixture",
          outcome: "success", attempt: 1, retry_of_call_id: null, cached: false,
          prompt_tokens: 1000, completion_tokens: 200, duration_ms: 1500,
          ttft_ms: 250, template_hash: "h", output: { patterns: 1 }, error: null },
        // Normal wire call: TTFT past 1s renders as "Ns" via fmtMs.
        { call_id: "1002", stage: "patterns", category: "work", model: "fixture",
          outcome: "success", attempt: 1, retry_of_call_id: null, cached: false,
          prompt_tokens: 1000, completion_tokens: 200, duration_ms: 4000,
          ttft_ms: 1500, template_hash: "h", output: { patterns: 1 }, error: null },
        // Cached row — stamped ttft_ms ignored, '—' wins.
        { call_id: "1003", stage: "patterns", category: "work", model: "fixture",
          outcome: "success", attempt: 1, retry_of_call_id: null, cached: true,
          prompt_tokens: 1000, completion_tokens: 200, duration_ms: 5,
          ttft_ms: 3, template_hash: "h", output: { patterns: 1 }, error: null },
        // Pending row.
        { call_id: "1004", stage: "patterns", category: "work", model: "fixture",
          outcome: "pending", attempt: 1, retry_of_call_id: null, cached: false,
          prompt_tokens: 1000, completion_tokens: 0, duration_ms: null,
          ttft_ms: null, template_hash: "h", output: null, error: null },
        // Aborted row WITH a stamped ttft_ms — spec says '—' anyway.
        { call_id: "1005", stage: "patterns", category: "work", model: "fixture",
          outcome: "aborted", attempt: 1, retry_of_call_id: null, cached: false,
          prompt_tokens: 1000, completion_tokens: 0, duration_ms: 12000,
          ttft_ms: 800, template_hash: "h", output: null, error: null },
        // Skipped row.
        { call_id: "1006", stage: "patterns", category: "work", model: "fixture",
          outcome: "skipped", attempt: 1, retry_of_call_id: null, cached: false,
          prompt_tokens: 1000, completion_tokens: 0, duration_ms: null,
          ttft_ms: null, template_hash: "h", output: null, error: null },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return statsWithWaitTimes;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());

    const perCallSection = screen.getByText(/Per-call/).parentElement;
    const headers = Array.from(
      perCallSection.querySelectorAll(".run-details-calls thead th")
    ).map((th) => th.textContent.trim());
    expect(headers).toContain("wait time");
    // Position: before "dur".
    expect(headers.indexOf("wait time")).toBeLessThan(headers.indexOf("dur"));

    const waitIdx = headers.indexOf("wait time");
    const rows = Array.from(
      perCallSection.querySelectorAll(".run-details-call-row")
    );
    const byCallId = new Map();
    rows.forEach((r) => {
      const m = r.children[0].textContent.match(/(\d{4})/);
      if (m) byCallId.set(m[1], r);
    });
    const waitCell = (id) =>
      byCallId.get(id).children[waitIdx].textContent.trim();
    expect(waitCell("1001")).toBe("250ms");
    expect(waitCell("1002")).toBe("2s");
    expect(waitCell("1003")).toBe("—");
    expect(waitCell("1004")).toBe("—");
    expect(waitCell("1005")).toBe("—");
    expect(waitCell("1006")).toBe("—");
  });

  it("cache columns: 'cache hit' shows historical; 'cache' shows live with copy+trash buttons", async () => {
    const user = userEvent.setup();
    // 0002 (the chain's final attempt): served from cache during run
    // AND still on disk now → cache hit yes, cache cell has copy +
    // trash. 0003 (standalone): NOT served from cache, NOT on disk
    // now → cache hit no, cache cell renders "–", no buttons.
    const statsWithCacheState = {
      ...sampleStats,
      calls: [
        { ...sampleStats.calls[0], cache_key: "key-0001",
          cached: false, cached_now: false },
        { ...sampleStats.calls[1], cache_key: "key-0002",
          cached: true, cached_now: true },
        { ...sampleStats.calls[2], cache_key: "key-0003",
          cached: false, cached_now: false },
      ],
    };
    let bustArgs = null;
    let readArgs = null;
    await renderApp({
      runs: [completedRun],
      override: (cmd, args) => {
        if (cmd === "read_run_llm_stats") return statsWithCacheState;
        if (cmd === "bust_llm_cache_entry") {
          bustArgs = args;
          return true;
        }
        if (cmd === "read_llm_cache_entry") {
          readArgs = args;
          return "{\"response\":\"cached body\"}";
        }
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());

    // Two visible rows after collapsing the 0001→0002 retry chain:
    //   row 0002 — cache hit yes / cache copy+trash buttons
    //   row 0003 — cache hit no  / cache "–"          / no buttons
    const cacheHitCells = document.querySelectorAll(".cache-hit-cell");
    const cacheCells = document.querySelectorAll(".in-cache-cell");
    expect(cacheHitCells.length).toBe(2);
    expect(cacheCells.length).toBe(2);
    // First visible row is 0002 (final of chain). Second is 0003.
    expect(cacheHitCells[0].textContent.trim()).toBe("yes");
    expect(cacheHitCells[1].textContent.trim()).toBe("no");
    // 0002 cell has a copy button (no "yes" label) + bust ✕.
    expect(screen.getByTestId("cache-copy-0002")).toBeTruthy();
    // 0003 cell is just the "–" placeholder, no buttons.
    expect(cacheCells[1].textContent.trim()).toBe("–");
    expect(screen.queryByTestId("cache-copy-0003")).toBeNull();

    // Trash button: only on the 0002 row.
    const trashButtons = document.querySelectorAll(
      ".run-details-bust-btn"
    );
    expect(trashButtons.length).toBe(1);

    // Copy button reads the on-disk cache entry via the Tauri command
    // and writes its bytes to the clipboard. Stub via defineProperty
    // (navigator.clipboard is a getter-only prop) and fire via
    // fireEvent — userEvent.setup() installs its own clipboard stub
    // that detaches our spy.
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    fireEvent.click(screen.getByTestId("cache-copy-0002"));
    expect(readArgs).toEqual({
      stage: "patterns",
      cacheKey: "key-0002",
    });
    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith("{\"response\":\"cached body\"}");
    });

    // Bust click still flips optimistically: 0002's cache cell goes
    // from copy+trash → "–", and "cache hit" stays yes (historical).
    await user.click(trashButtons[0]);
    expect(bustArgs).toEqual({
      stage: "patterns",
      cacheKey: "key-0002",
    });
    await waitFor(() => {
      expect(
        document.querySelectorAll(".run-details-bust-btn").length,
      ).toBe(0);
      const cacheAfter = document.querySelectorAll(".in-cache-cell");
      expect(cacheAfter[0].textContent.trim()).toBe("–");
      const cacheHitAfter = document.querySelectorAll(".cache-hit-cell");
      expect(cacheHitAfter[0].textContent.trim()).toBe("yes");
    });
  });

  it("listens for llm-cache-changed event and refetches stats", async () => {
    const user = userEvent.setup();
    let listenHandler = null;
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      if (evt === "llm-cache-changed") listenHandler = fn;
      return () => {};
    });
    let fetchCount = 0;
    let cachedNowState = true;
    const cachedRow = {
      ...sampleStats.calls[1],
      cache_key: "key-0002",
      cached: true,
      cached_now: true,
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") {
          fetchCount += 1;
          return {
            ...sampleStats,
            calls: [
              { ...sampleStats.calls[0], cache_key: "key-0001",
                cached: false, cached_now: false },
              { ...cachedRow, cached_now: cachedNowState },
              { ...sampleStats.calls[2], cache_key: "key-0003",
                cached: false, cached_now: false },
            ],
          };
        }
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    expect(fetchCount).toBe(1);
    // Pre-event: 0002 has trash button visible (cache: yes).
    expect(
      document.querySelectorAll(".run-details-bust-btn").length,
    ).toBe(1);

    // Simulate a cache wipe out-of-band (e.g. user clicked Wipe in
    // Settings while the modal is open). Server now returns
    // cached_now=false for the previously-cached row.
    cachedNowState = false;
    expect(listenHandler).toBeTruthy();
    await act(async () => {
      listenHandler({ event: "llm-cache-changed", payload: null });
    });
    await waitFor(() => expect(fetchCount).toBe(2));
    // Post-event: 0002's cache cell flipped to "–", trash button gone.
    await waitFor(() => {
      expect(
        document.querySelectorAll(".run-details-bust-btn").length,
      ).toBe(0);
    });
    const cacheCells = document.querySelectorAll(".in-cache-cell");
    expect(cacheCells[0].textContent.trim()).toBe("–");
    // "cache hit" stays yes — historical, unaffected by wipe.
    const cacheHit = document.querySelectorAll(".cache-hit-cell");
    expect(cacheHit[0].textContent.trim()).toBe("yes");
  });

  it("per-stage summary table renders stages in canonical pipeline order, not alphabetical", async () => {
    const user = userEvent.setup();
    // Multi-stage stats fixture. Insert stages in alphabetical order
    // on purpose — the rendered order MUST come out as the pipeline
    // order (vision → extract → entities → entities_dedupe → patterns
    // → insights → actions). If the modal sorts alphabetically the
    // first row will be "actions"; if it sorts by pipeline order the
    // first row will be "vision".
    const multiStageStats = {
      ...sampleStats,
      per_stage: {
        actions:        { name: "actions",        calls_total: 1, calls_cached: 0,
                          outcomes: { success: 1, success_empty: 0, blanked: 0, parse_error: 0, timeout: 0, other_failure: 0 },
                          duration_ms: { median: 100 } },
        entities:       { name: "entities",       calls_total: 1, calls_cached: 0,
                          outcomes: { success: 1, success_empty: 0, blanked: 0, parse_error: 0, timeout: 0, other_failure: 0 },
                          duration_ms: { median: 100 } },
        entities_dedupe:{ name: "entities_dedupe",calls_total: 1, calls_cached: 0,
                          outcomes: { success: 1, success_empty: 0, blanked: 0, parse_error: 0, timeout: 0, other_failure: 0 },
                          duration_ms: { median: 100 } },
        extract:        { name: "extract",        calls_total: 1, calls_cached: 0,
                          outcomes: { success: 1, success_empty: 0, blanked: 0, parse_error: 0, timeout: 0, other_failure: 0 },
                          duration_ms: { median: 100 } },
        insights:       { name: "insights",       calls_total: 1, calls_cached: 0,
                          outcomes: { success: 1, success_empty: 0, blanked: 0, parse_error: 0, timeout: 0, other_failure: 0 },
                          duration_ms: { median: 100 } },
        patterns:       { name: "patterns",       calls_total: 1, calls_cached: 0,
                          outcomes: { success: 1, success_empty: 0, blanked: 0, parse_error: 0, timeout: 0, other_failure: 0 },
                          duration_ms: { median: 100 } },
        vision:         { name: "vision",         calls_total: 1, calls_cached: 0,
                          outcomes: { success: 1, success_empty: 0, blanked: 0, parse_error: 0, timeout: 0, other_failure: 0 },
                          duration_ms: { median: 100 } },
      },
      calls: [],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return multiStageStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText("Per-stage summary")).toBeTruthy());
    // Read the rendered stage column from the per-stage table.
    const tbody = screen.getByText("Per-stage summary").parentElement
      .querySelector(".run-details-stages tbody");
    const stageNames = Array.from(tbody.querySelectorAll("tr td:first-child"))
      .map((td) => td.textContent.trim());
    expect(stageNames).toEqual([
      "vision", "extract", "entities", "entities_dedupe",
      "patterns", "insights", "actions",
    ]);
  });

  // Cache summary line at top of the modal — two branches.
  it("cache summary renders hit-rate when run_config.llm_cache_enabled is true (or null)", async () => {
    const user = userEvent.setup();
    // 3 calls in patterns, 2 cached → 67% hit rate.
    const statsWithCacheStats = {
      ...sampleStats,
      per_stage: {
        patterns: {
          ...sampleStats.per_stage.patterns,
          calls_cached: 2,
          calls_total: 3,
        },
      },
    };
    const runWithCacheOn = {
      ...completedRun,
      run_config: { llm_cache_enabled: true },
    };
    await renderApp({
      runs: [runWithCacheOn],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return statsWithCacheStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText("Per-stage summary")).toBeTruthy());
    const summary = document.querySelector(".cache-summary-line");
    expect(summary).toBeTruthy();
    expect(summary.classList.contains("bypassed")).toBe(false);
    expect(summary.textContent).toContain("2");
    expect(summary.textContent).toContain("3");
    expect(summary.textContent).toContain("67%");
  });

  it("cache summary renders bypassed label when run_config.llm_cache_enabled is false", async () => {
    const user = userEvent.setup();
    const runBypassed = {
      ...completedRun,
      run_config: { llm_cache_enabled: false },
    };
    await renderApp({
      runs: [runBypassed],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return sampleStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText("Per-stage summary")).toBeTruthy());
    const summary = document.querySelector(".cache-summary-line");
    expect(summary).toBeTruthy();
    expect(summary.classList.contains("bypassed")).toBe(true);
    expect(summary.textContent).toMatch(/bypassed/i);
    expect(summary.textContent).toContain("BASEVAULT_LLM_CACHE_BYPASS");
  });

  // The per-stage rollup carries streaming distributions (ttft_ms,
  // reasoning_tokens, etc); the modal surfaces "p50 ttft" but no
  // longer renders a per-stage reasoning sum — reasoning is a per-call
  // signal, surfaced on the per-call rows alongside payload, since the
  // stage-level sum hid the cost asymmetry between calls.
  it("per-stage table shows p50 ttft and does NOT carry a reasoning column", async () => {
    const user = userEvent.setup();
    const stats = {
      ...sampleStats,
      per_stage: {
        patterns: {
          ...sampleStats.per_stage.patterns,
          ttft_ms: { total: 6000, avg: 2000, median: 1900, min: 1700, max: 2400 },
          reasoning_tokens: { total: 1234, avg: 411, median: 380, min: 0, max: 600 },
        },
      },
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return stats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText("Per-stage summary")).toBeTruthy());
    const stagesTable = document.querySelector(".run-details-stages");
    const headers = Array.from(stagesTable.querySelectorAll("thead th"))
      .map((th) => th.textContent.trim());
    expect(headers).toContain("p50 ttft");
    expect(headers).not.toContain("reasoning");
    // Row cells indexed by header; verify p50 ttft still renders.
    const cells = Array.from(stagesTable.querySelectorAll("tbody tr td"))
      .map((td) => td.textContent.trim());
    const cellByHeader = Object.fromEntries(headers.map((h, i) => [h, cells[i]]));
    // fmtMs returns "2s" for 1900ms (formatDuration rounds whole-secs).
    expect(cellByHeader["p50 ttft"]).toBe("2s");
  });

  it("per-call table replaces 'completion' with 'payload' + 'reasoning' (= completion by construction)", async () => {
    const user = userEvent.setup();
    // payload = content_tokens; reasoning = reasoning_tokens; their sum
    // equals completion_tokens by construction in llm._record_usage.
    const stats = {
      ...sampleStats,
      calls: [
        {
          ...sampleStats.calls[1],
          call_id: "0099",
          retry_of_call_id: null,
          prompt_tokens: 4500,
          completion_tokens: 1300,
          content_tokens: 800,
          reasoning_tokens: 500,
          duration_ms: 1200,
          stage: "patterns",
          category: "topic-x",
        },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return stats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    const callsTable = document.querySelector(".run-details-calls table");
    const headers = Array.from(callsTable.querySelectorAll("thead th"))
      .map((th) => th.textContent.trim());
    expect(headers).toContain("payload");
    expect(headers).toContain("reasoning");
    expect(headers).not.toContain("completion");
    const row = callsTable.querySelector(".run-details-call-row");
    const cells = Array.from(row.querySelectorAll("td"))
      .map((td) => td.textContent.trim());
    const cellByHeader = Object.fromEntries(headers.map((h, i) => [h, cells[i]]));
    expect(cellByHeader["payload"]).toBe("800");
    expect(cellByHeader["reasoning"]).toBe("500");
  });

  it("per-row details expander surfaces the PR #158 streaming fields when present", async () => {
    const user = userEvent.setup();
    const stats = {
      ...sampleStats,
      calls: [
        {
          ...sampleStats.calls[1],
          call_id: "0042",
          ttft_ms: 1850,
          ttfr_ms: 950,
          last_token_ms: 4200,
          finish_reason: "stop",
          reasoning_tokens: 512,
          reasoning_tokens_source: "estimated",
          content_tokens: 800,
          max_tokens_reserved: 8192,
        },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return stats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    await user.click(document.querySelector(".run-details-detail-toggle"));
    const panel = document.querySelector(".run-details-detail-panel");
    expect(panel).toBeTruthy();
    const text = panel.textContent;
    expect(text).toContain("1850");
    expect(text).toContain("950");
    expect(text).toContain("4200");
    expect(text).toContain("stop");
    expect(text).toContain("512");
    expect(text).toContain("estimated");
    expect(text).toContain("800");
    expect(text).toContain("8192");
    const dts = Array.from(panel.querySelectorAll("dt")).map((el) => el.textContent);
    expect(dts).toContain("reasoning_tokens_source");
  });

  // 🧠 marker after the model name when request_extras.reasoning is
  // true. Mirrors the run-row reasoning badge so the UI vocabulary is
  // consistent across the modal and the runs list.
  it("renders 🧠 after model in per-call rows when reasoning=true", async () => {
    const user = userEvent.setup();
    const stats = {
      ...sampleStats,
      calls: [
        {
          ...sampleStats.calls[1],
          call_id: "0010",
          model: "kimi-k2-6",
          request_extras: { temperature: 0.0, reasoning: true },
        },
        {
          ...sampleStats.calls[2],
          call_id: "0011",
          model: "kimi-k2-6",
          request_extras: { temperature: 0.0, reasoning: false },
        },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return stats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    // Exactly one badge inside the per-call table — only the
    // reasoning=true row carries it; reasoning=false stays clean.
    const callsTable = document.querySelector(".run-details-calls table");
    const badges = callsTable.querySelectorAll(".per-call-reasoning-badge");
    expect(badges.length).toBe(1);
    expect(badges[0].textContent).toContain("🧠");
    expect(badges[0].getAttribute("title")).toMatch(/reasoning enabled/);
  });

  it("omits 🧠 marker when request_extras is missing (older calls)", async () => {
    const user = userEvent.setup();
    const stats = {
      ...sampleStats,
      calls: [
        {
          ...sampleStats.calls[1],
          call_id: "0020",
          model: "kimi-k2-6",
          request_extras: null,
        },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return stats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    const callsTable = document.querySelector(".run-details-calls table");
    expect(callsTable.querySelectorAll(".per-call-reasoning-badge").length).toBe(0);
  });

  // Per-row details expander surfaces the long-tail fields under each
  // call row — temperature, reasoning, budget, template_hash, error
  // trace, raw attempt + retry_of_call_id, plus the streaming-token
  // fields that land later (ttft_ms, finish_reason, etc.). Default-
  // collapsed; clicking the ▸ button expands a colspan'd row beneath
  // with a key/value grid; clicking again collapses.
  it("per-row details expander toggles a panel with long-tail fields", async () => {
    const user = userEvent.setup();
    const stats = {
      ...sampleStats,
      calls: [
        {
          ...sampleStats.calls[1],
          call_id: "0099",
          template_hash: "abc123abcdef",
          budget: { stage_cap: 41833, max_output: 83666, scaffolding: 2500 },
          request_extras: { temperature: 0.0, reasoning: false },
        },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return stats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    // Default-collapsed: no detail panel in DOM.
    expect(document.querySelector(".run-details-detail-panel")).toBeNull();
    // The ▸ button on the row.
    const toggle = document.querySelector(".run-details-detail-toggle");
    expect(toggle).toBeTruthy();
    await user.click(toggle);
    // Panel renders with the expected fields.
    const panel = document.querySelector(".run-details-detail-panel");
    expect(panel).toBeTruthy();
    const dts = Array.from(panel.querySelectorAll("dt")).map((el) => el.textContent);
    expect(dts).toContain("temperature");
    expect(dts).toContain("reasoning");
    expect(dts).toContain("stage_cap");
    expect(dts).toContain("max_output");
    expect(dts).toContain("scaffolding");
    expect(dts).toContain("template_hash");
    expect(dts).toContain("attempt");
    expect(dts).toContain("retry_of_call_id");
    // Concrete values rendered.
    expect(panel.textContent).toContain("41833");
    expect(panel.textContent).toContain("abc123abcdef");
    // Click again → collapse.
    await user.click(toggle);
    expect(document.querySelector(".run-details-detail-panel")).toBeNull();
  });

  it("per-row details expander shows error block on failure rows with traceback", async () => {
    const user = userEvent.setup();
    const stats = {
      ...sampleStats,
      calls: [
        {
          ...sampleStats.calls[0],
          call_id: "0050",
          outcome: "timeout",
          error: {
            class: "APITimeoutError",
            message: "request timed out after 60s",
            traceback: "Traceback (most recent call last):\n  ...",
          },
        },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return stats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    await user.click(document.querySelector(".run-details-detail-toggle"));
    const errBlock = document.querySelector(".run-details-detail-error");
    expect(errBlock).toBeTruthy();
    expect(errBlock.textContent).toContain("APITimeoutError");
    expect(errBlock.textContent).toContain("request timed out after 60s");
    // Traceback hidden behind a <details> until clicked.
    const tbDetails = errBlock.querySelector("details");
    expect(tbDetails).toBeTruthy();
    expect(tbDetails.open).toBe(false);
    await user.click(tbDetails.querySelector("summary"));
    await waitFor(() => {
      expect(errBlock.querySelector(".run-details-detail-error-tb").textContent)
        .toContain("Traceback (most recent call last)");
    });
  });

  it("details expander shows the kernel status block for a no-exception failure (#963)", async () => {
    // A `from_status` failure (injected/real LOAD): success=false, NO
    // error object. The red block must still render — naming the kernel's
    // llm_status — rather than leaving the panel silent about why it failed.
    const user = userEvent.setup();
    const stats = {
      ...sampleStats,
      calls: [
        {
          ...sampleStats.calls[0],
          call_id: "0051",
          outcome: "failed (load)",
          success: false,
          llm_status: "LOAD",
          error: null,
        },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return stats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    await user.click(document.querySelector(".run-details-detail-toggle"));
    const errBlock = document.querySelector(".run-details-detail-error");
    expect(errBlock).toBeTruthy();
    expect(errBlock.textContent).toContain("LOAD");
    expect(errBlock.textContent).toContain("no exception raised");
    // No exception → no traceback <details> in this block.
    expect(errBlock.querySelector("details")).toBeNull();
  });

  it("details expander shows NO error block for a clean success", async () => {
    // Guard the gate: a successful call (no error, success=true) must not
    // grow a spurious red block from the new no-exception-failure branch.
    const user = userEvent.setup();
    const stats = {
      ...sampleStats,
      calls: [
        {
          ...sampleStats.calls[0],
          call_id: "0052",
          outcome: "success",
          success: true,
          llm_status: "OK",
          error: null,
        },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return stats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    await user.click(document.querySelector(".run-details-detail-toggle"));
    expect(document.querySelector(".run-details-detail-panel")).toBeTruthy();
    expect(document.querySelector(".run-details-detail-error")).toBeNull();
  });

  // Full prompt + response logs section — surfaces calls where the
  // dev-tab toggle was ON during the run. Hidden when no call carries
  // the fields (the common case).
  it("does not render Full prompt + response logs section when no call has the fields", async () => {
    const user = userEvent.setup();
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return sampleStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    expect(screen.queryByText(/Full prompt . response logs/)).toBeNull();
  });

  it("renders Full prompt + response logs entries for calls with the fields, expandable on click", async () => {
    const user = userEvent.setup();
    const statsWithLog = {
      ...sampleStats,
      calls: [
        {
          ...sampleStats.calls[1],
          full_prompt: [
            { role: "system", content: "be helpful" },
            { role: "user", content: "do the thing" },
          ],
          full_response: "the response text",
        },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return statsWithLog;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Full prompt . response logs/)).toBeTruthy());
    // One <details> entry, collapsed by default — content not in DOM yet.
    const entry = document.querySelector(".full-prompt-entry");
    expect(entry).toBeTruthy();
    expect(entry.open).toBe(false);
    expect(document.querySelector(".full-prompt-content")).toBeNull();
    // Click the summary → expand.
    await user.click(entry.querySelector("summary"));
    await waitFor(() => {
      expect(document.querySelectorAll(".full-prompt-content").length).toBe(3);
    });
    const blocks = Array.from(document.querySelectorAll(".full-prompt-content"))
      .map((b) => b.textContent);
    expect(blocks[0]).toContain("be helpful");
    expect(blocks[1]).toContain("do the thing");
    expect(blocks[2]).toContain("the response text");
  });

  it("Full prompt entry has a corner CopyButton that copies prompt + response", async () => {
    const user = userEvent.setup();
    const statsWithLog = {
      ...sampleStats,
      calls: [
        {
          ...sampleStats.calls[1],
          full_prompt: [
            { role: "system", content: "be helpful" },
            { role: "user", content: "do the thing" },
          ],
          full_response: "the response text",
        },
      ],
    };
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return statsWithLog;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Full prompt . response logs/)).toBeTruthy());
    const entry = document.querySelector(".full-prompt-entry");
    await user.click(entry.querySelector("summary"));
    const callId = statsWithLog.calls[0].call_id;
    const copyBtn = screen.getByTestId(`full-prompt-copy-${callId}`);
    // fireEvent (not userEvent) for the same reason as the cache-cell
    // copy test above: userEvent.setup() detaches our clipboard spy.
    fireEvent.click(copyBtn);
    expect(writeText).toHaveBeenCalledTimes(1);
    const text = writeText.mock.calls[0][0];
    // Plain-text dump: prompt header, both messages by role, then the
    // response. No JSON noise.
    expect(text).toContain("=== PROMPT ===");
    expect(text).toContain("--- SYSTEM ---");
    expect(text).toContain("be helpful");
    expect(text).toContain("--- USER ---");
    expect(text).toContain("do the thing");
    expect(text).toContain("=== RESPONSE ===");
    expect(text).toContain("the response text");
  });

  // Vision-stage calls stamp `content` as an OpenAI-multimodal array
  // (text part + image_url part with a base64 data URL). Rendering
  // that as a plain string used to crash React with "objects are
  // not valid as a child"; the viewer now renders text parts as
  // <pre> and image parts as <img>.
  it("renders multimodal vision content (text + image_url) without crashing", async () => {
    const user = userEvent.setup();
    const dataUrl =
      "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAUDBAQEAwUEBAQF";
    const statsWithVisionLog = {
      ...sampleStats,
      calls: [
        {
          ...sampleStats.calls[1],
          full_prompt: [
            {
              role: "user",
              content: [
                { type: "text", text: "Describe this image." },
                { type: "image_url", image_url: { url: dataUrl } },
              ],
            },
          ],
          full_response: "A scene with a tattoo reading '11:59'.",
        },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return statsWithVisionLog;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Full prompt . response logs/)).toBeTruthy());
    const entry = document.querySelector(".full-prompt-entry");
    expect(entry).toBeTruthy();
    await user.click(entry.querySelector("summary"));
    await waitFor(() => {
      expect(document.querySelector(".full-prompt-image")).toBeTruthy();
    });
    const textBlocks = Array.from(document.querySelectorAll(".full-prompt-content"))
      .map((b) => b.textContent);
    expect(textBlocks).toContain("Describe this image.");
    expect(textBlocks).toContain("A scene with a tattoo reading '11:59'.");
    const img = document.querySelector(".full-prompt-image");
    expect(img.getAttribute("src")).toBe(dataUrl);
  });

  // After a bust completes, the row picks up a `.cache-busted` visual
  // dim class while the override survives (until the refetch
  // triggered by llm-cache-changed sweeps the override).
  it("post-bust row picks up .cache-busted class for visual feedback", async () => {
    const user = userEvent.setup();
    const baseStats = {
      ...sampleStats,
      calls: [
        { ...sampleStats.calls[1], cache_key: "key-0002",
          cached: true, cached_now: true },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return baseStats;
        if (cmd === "bust_llm_cache_entry") return true;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    await user.click(document.querySelector(".run-details-bust-btn"));
    await waitFor(() => {
      const row = document.querySelector(".run-details-call-row");
      expect(row.className).toContain("cache-busted");
    });
  });

  // cap_hit outcome: per-call rows for calls whose Python classifier
  // tagged them outcome="cap_hit (sizing)" (finish_reason="length" +
  // success) render the sizing-orange pill with the verbatim label.
  // The failure-class suffix is itself the user-visible signal, so
  // the pill keeps the parenthetical form rather than abbreviating
  // to "cap hit".
  it("renders 'cap_hit (sizing)' pill on per-call rows", async () => {
    const user = userEvent.setup();
    const stats = {
      ...sampleStats,
      calls: [
        {
          ...sampleStats.calls[1],
          call_id: "0042",
          outcome: "cap_hit (sizing)",
          finish_reason: "length",
        },
      ],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return stats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Per-call/)).toBeTruthy());
    const callsTable = document.querySelector(".run-details-calls table");
    // CSS sanitizes the parenthetical/space into a flat class name:
    // "cap_hit (sizing)" → "outcome-cap_hit_sizing".
    const pill = callsTable.querySelector(".outcome-pill.outcome-cap_hit_sizing");
    expect(pill).toBeTruthy();
    expect(pill.textContent).toBe("cap_hit (sizing)");
  });

  // Per-stage rollup: when a stage has cap_hit (sizing) calls, the
  // sizing column surfaces the count alongside the other retry-class
  // buckets. The taxonomy collapses cap_hit / timeout (sizing) /
  // parse_error (sizing) / interrupted (sizing) / failed (sizing)
  // into a single "Sizing Failures" column in the per-stage summary
  // (per STAGE_OUTCOME_COLUMNS), so the assertion checks the sizing
  // column total rather than a cap-hit-specific one.
  it("per-stage summary shows sizing count column with cap_hit (sizing) included", async () => {
    const user = userEvent.setup();
    const stats = {
      ...sampleStats,
      per_stage: {
        patterns: {
          ...sampleStats.per_stage.patterns,
          outcomes: {
            success: 2,
            "cap_hit (sizing)": 3,
            success_empty: 0,
            "parse_error (sizing)": 0,
            "timeout (sizing)": 0, "timeout (load)": 0,
            "failed (sizing)": 0, "failed (load)": 0,
            "failed (other)": 0,
          },
        },
      },
      calls: [],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return stats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText("Per-stage summary")).toBeTruthy());
    const stagesTable = document.querySelector(".run-details-stages");
    const headers = Array.from(stagesTable.querySelectorAll("thead th"))
      .map((th) => th.textContent.trim());
    const sizingIdx = headers.indexOf("Sizing Failures");
    expect(sizingIdx).toBeGreaterThan(-1);
    const cells = Array.from(stagesTable.querySelectorAll("tbody tr td"))
      .map((td) => td.textContent.trim());
    expect(cells[sizingIdx]).toBe("3");
  });

  // Sanity: sum of per-stage `cap_hit (sizing)` outcomes equals the
  // run-level warnings.cap_hits aggregate. Pins the contract that the
  // outcome bucket and the legacy aggregate counter are the same thing
  // observed at different layers — divergence would mean a bug in
  // either the classifier or the warning emitter.
  it("sum of per-stage cap_hit (sizing) outcomes equals run.warnings.cap_hits", async () => {
    const user = userEvent.setup();
    const runWithCapWarnings = {
      ...completedRun,
      warnings: { cap_hits: 4, empty_responses: 0, input_overflows: 0 },
    };
    const stats = {
      ...sampleStats,
      per_stage: {
        patterns: {
          ...sampleStats.per_stage.patterns,
          outcomes: {
            success: 1,
            "cap_hit (sizing)": 3,
            success_empty: 0,
          },
        },
        insights: {
          name: "insights", calls_total: 1, calls_cached: 0,
          outcomes: {
            success: 0,
            "cap_hit (sizing)": 1,
            success_empty: 0,
          },
          duration_ms: { median: 100 },
        },
      },
      calls: [],
    };
    await renderApp({
      runs: [runWithCapWarnings],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return stats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText("Per-stage summary")).toBeTruthy());
    const perStageSum = Object.values(stats.per_stage)
      .reduce((acc, b) => acc + (b.outcomes?.["cap_hit (sizing)"] || 0), 0);
    expect(perStageSum).toBe(runWithCapWarnings.warnings.cap_hits);
  });

  // Issue #165 layer 4: the routing-snapshot panel (was the
  // collapsible "Run details" panel above the file tree) folded into
  // the modal header above the cache hit-rate line. Always-visible.
  it("renders RunSummarySection at the top of the modal above cache summary", async () => {
    const user = userEvent.setup();
    const runWithRouting = {
      ...completedRun,
      run_config: {
        stage_models: { extract: "kimi-k2-6", patterns: "qwen3" },
        stage_reasoning: { extract: false, patterns: true },
        temperature: 0.0,
        sentiment: "neutral",
        llm_cache_enabled: true,
        pipeline_git_sha: "deadbeefcafe",
        app_version: "0.1.0",
      },
    };
    await renderApp({
      runs: [runWithRouting],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return sampleStats;
      },
    });
    await openRunDetailsModal(user);
    await waitFor(() => expect(screen.getByText(/Run details — /)).toBeTruthy());
    // Routing table is rendered.
    const routing = document.querySelector(".run-details-routing");
    expect(routing).toBeTruthy();
    expect(routing.textContent).toContain("kimi-k2-6");
    expect(routing.textContent).toContain("qwen3");
    // RunSummarySection sits ABOVE the cache summary line.
    const summary = document.querySelector(".run-summary-header");
    const cacheLine = document.querySelector(".run-details-cache-summary");
    expect(summary).toBeTruthy();
    expect(cacheLine).toBeTruthy();
    // DOM order: summary precedes cache line.
    expect(
      summary.compareDocumentPosition(cacheLine)
        & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("in-tree RunDetails panel no longer renders in the run-tree pane", async () => {
    // Pre-#165 the RunTreePane mounted a `<RunDetails>` collapsible
    // panel above the file tree. Post-#165 that surface lives only in
    // the modal header — the tree pane is just file tree + run
    // controls. Assert the toggle button is gone.
    const user = userEvent.setup();
    await renderApp({ runs: [completedRun] });
    const row = screen.getByText("Completed").closest(".run-row");
    await user.click(row);
    await waitFor(() => expect(screen.getByTitle(/Open run details/)).toBeTruthy());
    // Modal not open yet; the in-tree panel's old toggle button must
    // not exist anywhere on the screen.
    expect(document.querySelector(".run-details-toggle")).toBeNull();
  });

  // Issue #165: the modal opens for in-flight runs and reads the
  // jsonl-materialized in-flight stats payload from read_run_llm_stats.
  // Header carries the [RUNNING] tag so a viewer doesn't mistake a
  // mid-run snapshot for a completed-run summary.
  it("opens for a running run and surfaces [RUNNING] in the header", async () => {
    const user = userEvent.setup();
    const runningRun = {
      ...completedRun,
      status: "running",
      created_at: "2026-04-29T12:00:00Z",
      duration_ms: null,
    };
    // In-flight materialized payload: schema "llm-stats/v1-inflight"
    // (same field shape as the post-run rollup so the JSX path doesn't
    // fork). One stage with one in-flight (cancelled-shaped) call.
    const inflightStats = {
      schema: "llm-stats/v1-inflight",
      in_flight: true,
      totals: { calls: 1, successful: 0, failed: 0, cancelled: 1 },
      per_stage: {
        extract: {
          name: "extract",
          calls_total: 1,
          calls_failed: 0,
          calls_aborted: 0,
          calls_cached: 0,
          outcomes: {
            success: 1, success_empty: 0, blanked: 0,
            parse_error: 0, timeout: 0, other_failure: 0,
          },
          models_used: ["fixture"],
          duration_ms: { median: 1000 },
          ttft_ms: { median: 250 },
          reasoning_tokens: { total: 0 },
        },
      },
      calls: [{
        call_id: "0001", stage: "extract", category: "x", model: "fixture",
        outcome: "success", success: true, attempt: 1, retry_of_call_id: null,
        cached: false, cached_now: false,
        prompt_tokens: 100, completion_tokens: 50, duration_ms: 1000,
        output: { facts: 5 }, error: null,
      }],
    };
    // Replace "Completed" with "Running" in the row-finder helper —
    // running runs render the "Running" status badge instead.
    await renderApp({
      runs: [runningRun],
      override: (cmd) => {
        if (cmd === "read_run_llm_stats") return inflightStats;
      },
    });
    const row = screen.getByText("Running").closest(".run-row");
    await user.click(row);
    await waitFor(() => expect(screen.getByTitle(/Open run details/)).toBeTruthy());
    await user.click(screen.getByTitle(/Open run details/));
    await waitFor(() => expect(screen.getByText(/Run details — /)).toBeTruthy());
    // Header tag flags the run as in-flight.
    const header = document.querySelector(".modal-header h2");
    expect(header.textContent).toMatch(/\[RUNNING\]/);
    // The materialized payload renders the per-stage table.
    await waitFor(() => expect(screen.getByText("Per-stage summary")).toBeTruthy());
    const stagesTable = document.querySelector(".run-details-stages");
    expect(stagesTable.textContent).toContain("extract");
    // Per-call table renders the materialized record.
    expect(document.querySelector(".run-details-call-row")).toBeTruthy();
  });

  // Issue #678: the optimistic skip ✕ click must NOT be wiped by the
  // next refetch when the canonical record hasn't caught up yet. Pre-
  // fix `refetchStats` unconditionally cleared `skipOverride` on every
  // poll tick. For a slow pre-first-token backend (kimi-k2-6 on Tinfoil
  // can sit silent tens of seconds before the first chunk), the marker
  // → end-event chain hadn't propagated to the rollup by the time the
  // next pipeline-progress refetch landed, so the pill flipped back to
  // "pending" until the end-event eventually wrote. Post-fix the
  // override holds while canonical state still says pending and gives
  // way once canonical reaches a terminal label.
  it("optimistic skip override holds across refetch until canonical state shows non-pending (issue #678)", async () => {
    const user = userEvent.setup();
    let progressHandler = null;
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      if (evt === "pipeline-progress") progressHandler = fn;
      return () => {};
    });

    const runningRun = {
      ...completedRun,
      status: "running",
      progress: { stage: "patterns", completed: 0, total: 1 },
    };

    // One in-flight pending call. The stats object is captured by
    // reference into the read_run_llm_stats override below, so the test
    // can mutate `calls[0].outcome` to simulate the canonical record
    // catching up between refetches.
    const stats = {
      schema: "llm-stats/v1",
      run_id: runningRun.run_id,
      totals: {
        calls: 1,
        outcomes: {
          success: 0, success_empty: 0, blanked: 0,
          parse_error: 0, timeout: 0, other_failure: 0,
        },
      },
      per_stage: {
        patterns: {
          name: "patterns",
          calls_total: 1,
          calls_failed: 0,
          calls_aborted: 0,
          calls_cached: 0,
          outcomes: {
            success: 0, success_empty: 0, blanked: 0,
            parse_error: 0, timeout: 0, other_failure: 0,
          },
          models_used: ["fixture"],
          prompt_tokens: { total: 0, avg: 0, median: 0, min: 0, max: 0 },
          completion_tokens: { total: 0, avg: 0, median: 0, min: 0, max: 0 },
          duration_ms: { total: 0, avg: 0, median: 0, min: 0, max: 0 },
        },
      },
      calls: [
        {
          call_id: "0007", stage: "patterns", category: "work",
          model: "fixture", outcome: "pending", attempt: 1,
          retry_of_call_id: null, cached: false, prompt_tokens: 0,
          completion_tokens: 0, duration_ms: null,
          template_hash: "h", output: null, error: null,
          cache_key: null, cached_now: false,
        },
      ],
    };

    let skipCallInvokedWith = null;
    await renderApp({
      runs: [runningRun],
      override: (cmd, args) => {
        if (cmd === "read_run_llm_stats") {
          // Deep-clone so each refetch sees an independent payload
          // (the modal mutates the stats object during cell renders
          // for the cached_now overlay).
          return JSON.parse(JSON.stringify(stats));
        }
        if (cmd === "skip_call") {
          skipCallInvokedWith = args;
          return null;
        }
      },
    });

    // Open the modal. The shared helper keys off "Completed", so open
    // via the "Running" badge instead.
    const row = screen.getByText("Running").closest(".run-row");
    await user.click(row);
    await waitFor(() =>
      expect(screen.getByTitle(/Open run details/)).toBeTruthy()
    );
    await user.click(screen.getByTitle(/Open run details/));
    await waitFor(() => expect(screen.getByText(/0007/)).toBeTruthy());

    // Scope all outcome-pill assertions to the per-call row — the
    // top-of-modal "stage: patterns" header also renders "pending".
    const callRow = () =>
      document.querySelector('[data-call-id="0007"]');
    const pillText = () =>
      callRow()?.querySelector(".outcome-pill")?.textContent?.trim();

    // Pre-click sanity: the per-call pill shows pending.
    expect(pillText()).toBe("pending");

    // Click the skip ✕. Override flips outcome to "skipped"
    // optimistically; the canonical stats record is still pending.
    const skipBtn = screen.getByLabelText("Skip this call");
    await user.click(skipBtn);
    expect(skipCallInvokedWith).toEqual({
      runId: runningRun.run_id, callId: "0007",
    });
    await waitFor(() => expect(pillText()).toBe("skipped"));

    // Fire pipeline-progress to drive a refetch. Stats still report
    // outcome="pending" for 0007. Pre-fix the unconditional override
    // wipe would flip the pill back here; post-fix the override is
    // pruned only against ids that hit a non-pending canonical outcome,
    // so 0007 stays in the override and the pill stays "skipped". 600ms
    // wait covers the modal's pipeline-progress coalesce timer (500ms).
    await act(async () => {
      progressHandler({ payload: "tick" });
      await new Promise((r) => setTimeout(r, 600));
    });
    expect(pillText()).toBe("skipped");

    // Canonical state now catches up — mutate the shared stats so the
    // next refetch reads outcome="skipped" off disk. The optimistic
    // override is pruned (id no longer pending in the payload) and the
    // canonical record carries the pill instead. Same label, now from
    // the authoritative source.
    stats.calls[0].outcome = "skipped";
    await act(async () => {
      progressHandler({ payload: "tick" });
      await new Promise((r) => setTimeout(r, 600));
    });
    expect(pillText()).toBe("skipped");
  });
});

// Inline rename surface lives on the run-view header (RunHeaderName,
// reached by clicking a run row). The bug being pinned: an alias-less
// run used to open an empty input, leaving the user nothing to edit
// from. The fix pre-fills the input with `alias || short_id`, and the
// existing focus-then-select makes the value selected so a keystroke
// replaces it cleanly.
describe("App — run header rename pre-fill (issue #155)", () => {
  const baseRun = {
    run_id: "2026-04-29T12-00-00Z-8bv3",
    short_id: "8bv3",
    status: "completed",
    mode: "tee",
    inputs: ["/x/y.md"],
    input_count: 1,
    created_at: "2026-04-29T12:00:00Z",
    duration_ms: 30_000,
    progress: { stage: "done", completed: 50, total: 50 },
  };

  async function enterRenameMode(user) {
    const row = screen.getByText("Completed").closest(".run-row");
    await user.click(row);
    // RunHeaderName mounts inside RunTreePane after the run is current.
    await waitFor(() =>
      expect(document.querySelector(".run-header-name")).toBeTruthy()
    );
    await user.click(document.querySelector(".run-header-name"));
    await waitFor(() =>
      expect(document.querySelector(".run-header-rename")).toBeTruthy()
    );
    return document.querySelector(".run-header-rename");
  }

  it("alias-less run pre-fills the rename input with short_id", async () => {
    const user = userEvent.setup();
    await renderApp({ runs: [baseRun] });
    const input = await enterRenameMode(user);
    expect(input.value).toBe("8bv3");
  });

  it("aliased run pre-fills the rename input with the alias", async () => {
    const user = userEvent.setup();
    await renderApp({
      runs: [baseRun],
      // Aliases live in config.run_aliases; App reads them at bootstrap
      // and threads the lookup down to RunHeaderName via the alias prop.
      config: {
        subject: "User",
        tee_provider: "tinfoil",
        tee_model: "gpt-oss-120b",
        inputs: [],
        mode: "tee",
        run_aliases: { [baseRun.run_id]: "my-favorite-run" },
      },
    });
    const input = await enterRenameMode(user);
    expect(input.value).toBe("my-favorite-run");
  });

  // Issue #181: opening the rename input pre-fills with `short_id` (or
  // alias). If the user clicks the name and then clicks away without
  // typing, the unchanged draft should NOT be committed — otherwise the
  // pre-filled short_id would persist as the alias and the run would
  // visibly rename itself to e.g. `8bv3`.
  function setConfigCallsWithAliases() {
    return vi.mocked(invoke).mock.calls.filter(
      (c) => c[0] === "update_config" && c[1]?.patch?.run_aliases !== undefined,
    );
  }

  it("opening rename and blurring without typing does not commit a rename", async () => {
    const user = userEvent.setup();
    await renderApp({ runs: [baseRun] });
    const input = await enterRenameMode(user);
    expect(input.value).toBe("8bv3");
    // Snapshot the pre-blur set_config-with-aliases count so any
    // unrelated bootstrap config writes don't pollute the assertion.
    const before = setConfigCallsWithAliases().length;
    fireEvent.blur(input);
    await waitFor(() =>
      expect(document.querySelector(".run-header-rename")).toBeNull()
    );
    expect(setConfigCallsWithAliases().length).toBe(before);
  });

  it("typing a new value and blurring commits the rename", async () => {
    const user = userEvent.setup();
    await renderApp({ runs: [baseRun] });
    const input = await enterRenameMode(user);
    await user.clear(input);
    await user.type(input, "renamed-run");
    const before = setConfigCallsWithAliases().length;
    fireEvent.blur(input);
    await waitFor(() =>
      expect(setConfigCallsWithAliases().length).toBeGreaterThan(before)
    );
    const last = setConfigCallsWithAliases().at(-1);
    expect(last[1].patch.run_aliases[baseRun.run_id]).toBe("renamed-run");
  });

  // Perma-id binding (#574): in the run LIST a rename is purely
  // cosmetic — the 4-letter #<short_id> stays visible on line 1 even
  // once an alias is set (the run-view header keeps its prior compact
  // behaviour by design; the always-visible layout is the list +
  // chat picker).
  it("renamed run still shows #short_id in the run list", async () => {
    await renderApp({
      runs: [baseRun],
      config: {
        subject: "User",
        tee_provider: "tinfoil",
        tee_model: "gpt-oss-120b",
        inputs: [],
        mode: "tee",
        run_aliases: { [baseRun.run_id]: "my-favorite-run" },
      },
    });
    const row = screen.getByText("my-favorite-run").closest(".run-row");
    const chip = row.querySelector(".run-short-id");
    expect(chip).toBeTruthy();
    expect(chip.textContent).toBe("#8bv3");
  });

});

describe("App — drag-and-drop file ingestion (issue #139)", () => {
  // Capture the callback the App registers with onDragDropEvent so we
  // can synthesize OS-level drag-drop events in tests (jsdom can't fire
  // them; the real handler is owned by the Tauri webview).
  function stubDragDrop() {
    let dropHandler = null;
    const onDragDropEvent = vi.fn(async (cb) => {
      dropHandler = cb;
      return () => {};
    });
    vi.mocked(getCurrentWebview).mockReturnValue({ onDragDropEvent });
    return {
      getHandler: () => dropHandler,
      fire: async (payload) => {
        if (!dropHandler) throw new Error("onDragDropEvent not registered yet");
        await act(async () => {
          dropHandler({ payload });
        });
      },
    };
  }

  it("dropping files routes them through expand_paths and adds them to the staging list", async () => {
    const dd = stubDragDrop();
    await renderApp({
      override: (cmd, args) => {
        if (cmd === "expand_paths") {
          // Simulate a directory in the drop expanding into two files.
          return args.paths.flatMap((p) =>
            p.endsWith("/dir") ? [`${p}/a.md`, `${p}/b.md`] : [p],
          );
        }
      },
    });
    await waitFor(() => expect(dd.getHandler()).toBeTruthy());

    // Pre-condition: empty staging list.
    expect(screen.getByText(/Input files/i).textContent).toBe("Input files");

    await dd.fire({
      type: "drop",
      paths: ["/Users/u/notes.md", "/Users/u/dir"],
    });

    // expand_paths invoked with the raw drop payload.
    expect(invoke).toHaveBeenCalledWith("expand_paths", {
      paths: ["/Users/u/notes.md", "/Users/u/dir"],
    });
    // 1 file passed through + 2 from directory expansion = 3 rows.
    await waitFor(() =>
      expect(screen.getByText(/Input files \(3\)/)).toBeTruthy(),
    );
  });

  it("drop while a run is selected clears the selection so the staging view returns", async () => {
    const dd = stubDragDrop();
    const completedRun = {
      run_id: "2026-04-29T12-00-00Z-abcd",
      short_id: "abcd",
      status: "completed",
      mode: "tee",
      inputs: ["/Users/u/notes.md"],
      input_count: 1,
      created_at: "2026-04-29T12:00:00Z",
      duration_ms: 30_000,
      progress: { stage: "done", completed: 50, total: 50 },
      provider: "tinfoil",
      model: "gpt-oss-120b",
      vault_exists: true,
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd, args) => {
        if (cmd === "expand_paths") return args.paths;
      },
    });
    await waitFor(() => expect(dd.getHandler()).toBeTruthy());

    // Select the run by clicking its row. Once selected the StagingPane
    // collapses to ViewingRunPane (anchor: "Run again" button).
    const row = screen.getByText("Completed").closest(".run-row");
    expect(row).toBeTruthy();
    fireEvent.click(row);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Run again/i })).toBeTruthy();
    });

    // Drop a file. This should add the input AND clear the run
    // selection so the staging view returns.
    await dd.fire({ type: "drop", paths: ["/Users/u/dropped.md"] });

    await waitFor(() => {
      // Run-again button is gone (no run selected anymore).
      expect(screen.queryByRole("button", { name: /Run again/i })).toBeNull();
      // Staging pane shows the new file as a row.
      expect(screen.getByText(/Input files \(1\)/)).toBeTruthy();
    });
  });

  it("enter/over/leave events toggle the drop overlay without ingesting paths", async () => {
    const dd = stubDragDrop();
    await renderApp();
    await waitFor(() => expect(dd.getHandler()).toBeTruthy());

    // No overlay before any drag event.
    expect(document.querySelector(".drop-overlay")).toBeNull();

    await dd.fire({ type: "enter", paths: ["/x/a.md"] });
    expect(document.querySelector(".drop-overlay")).toBeTruthy();

    await dd.fire({ type: "over", paths: ["/x/a.md"] });
    expect(document.querySelector(".drop-overlay")).toBeTruthy();

    await dd.fire({ type: "leave" });
    expect(document.querySelector(".drop-overlay")).toBeNull();

    // Enter/over/leave never call expand_paths — only drop does.
    expect(
      vi.mocked(invoke).mock.calls.filter((c) => c[0] === "expand_paths").length,
    ).toBe(0);
  });

  it("drop with an empty paths array is a no-op (does not call expand_paths or clear selection)", async () => {
    const dd = stubDragDrop();
    await renderApp();
    await waitFor(() => expect(dd.getHandler()).toBeTruthy());

    await dd.fire({ type: "drop", paths: [] });

    expect(
      vi.mocked(invoke).mock.calls.filter((c) => c[0] === "expand_paths").length,
    ).toBe(0);
    // Staging list is still empty.
    expect(screen.getByText("Input files").textContent).toBe("Input files");
  });
});

// ── Staging-time filter + 'X files ignored' modal (issue #156) ───────────
//
// Validation runs at staging time across all three entry paths (file
// picker, folder picker, drag-drop) so users see exactly which files
// the pipeline would silently skip — instead of finding out via
// run.log lines after a Run. The shared `stageCandidates` path stats
// candidates first (so size-based rejections use real bytes), then
// validateInputs splits accepted vs rejected; only accepted go into
// the staging list, rejected populate a modal grouped by reason.
describe("App — staging-time filter + ignored-files modal (issue #156)", () => {
  function stubDragDrop() {
    let dropHandler = null;
    const onDragDropEvent = vi.fn(async (cb) => {
      dropHandler = cb;
      return () => {};
    });
    vi.mocked(getCurrentWebview).mockReturnValue({ onDragDropEvent });
    return {
      getHandler: () => dropHandler,
      fire: async (payload) => {
        if (!dropHandler) throw new Error("onDragDropEvent not registered yet");
        await act(async () => {
          dropHandler({ payload });
        });
      },
    };
  }

  // Echo every requested path with the size declared in `sizes`
  // (defaults to 100 bytes when unspecified). Lets a test inject a
  // single oversized file without enumerating the rest.
  function statEcho(sizes = {}) {
    return (cmd, args) => {
      if (cmd === "stat_paths") {
        return (args?.paths || []).map((p) => ({
          path: p,
          size_bytes: sizes[p] ?? 100,
        }));
      }
    };
  }

  it("file picker: an unsupported .dmg is filtered out and surfaces in the modal", async () => {
    vi.mocked(openDialog).mockResolvedValueOnce(["/u/notes.md", "/u/installer.dmg"]);
    await renderApp({ override: statEcho() });

    await userEvent.click(screen.getByTitle("Add file(s)"));

    // Only the .md lands in staging.
    await waitFor(() => {
      expect(screen.getByText(/Input files \(1\)/)).toBeTruthy();
    });
    // Modal opens listing the rejected file.
    const modal = await screen.findByText("1 file ignored");
    expect(modal).toBeTruthy();
    // Reason group renders, with the file basename inside.
    expect(screen.getByText(/unsupported extension \(\.dmg\)/)).toBeTruthy();
    expect(screen.getByText("installer.dmg")).toBeTruthy();
  });

  it("file picker: oversized file gets 'too large' reason with correct MB", async () => {
    vi.mocked(openDialog).mockResolvedValueOnce(["/u/big.pdf"]);
    await renderApp({
      override: statEcho({ "/u/big.pdf": 50 * 1024 * 1024 }),
    });

    await userEvent.click(screen.getByTitle("Add file(s)"));

    await screen.findByText("1 file ignored");
    expect(screen.getByText(/too large \(50\.0 MB > 40 MB limit\)/)).toBeTruthy();
    // Nothing accepted.
    expect(screen.queryByText(/Input files \([1-9]/)).toBeNull();
  });

  it("folder picker: .DS_Store + .html stripped after expand_paths; modal groups by reason", async () => {
    vi.mocked(openDialog).mockResolvedValueOnce("/u/folder");
    await renderApp({
      override: (cmd, args) => {
        if (cmd === "expand_paths") {
          return ["/u/folder/notes.md", "/u/folder/.DS_Store", "/u/folder/page.html"];
        }
        if (cmd === "stat_paths") {
          return (args?.paths || []).map((p) => ({ path: p, size_bytes: 100 }));
        }
      },
    });

    await userEvent.click(screen.getByTitle("Add folder (recursive)"));

    await waitFor(() => expect(screen.getByText(/Input files \(1\)/)).toBeTruthy());
    // Folder load → title leads with the imported count.
    await screen.findByText("1 file imported, 2 files ignored");
    expect(screen.getByText(/system file \(\.DS_Store\)/)).toBeTruthy();
    expect(screen.getByText(/excluded format \(\.html\)/)).toBeTruthy();
  });

  it("drag-drop: rejected files surface in the modal and stay out of the staging list", async () => {
    const dd = stubDragDrop();
    await renderApp({
      override: (cmd, args) => {
        if (cmd === "expand_paths") return args.paths;
        if (cmd === "stat_paths") {
          return (args?.paths || []).map((p) => ({ path: p, size_bytes: 100 }));
        }
      },
    });
    await waitFor(() => expect(dd.getHandler()).toBeTruthy());

    await dd.fire({
      type: "drop",
      paths: ["/u/a.md", "/u/b.dmg", "/u/.gitignore"],
    });

    await waitFor(() => expect(screen.getByText(/Input files \(1\)/)).toBeTruthy());
    await screen.findByText("2 files ignored");
    expect(screen.getByText(/unsupported extension \(\.dmg\)/)).toBeTruthy();
    expect(screen.getByText(/hidden file/)).toBeTruthy();
  });

  it("all-accepted batch does NOT render the modal", async () => {
    const dd = stubDragDrop();
    await renderApp({
      override: (cmd, args) => {
        if (cmd === "expand_paths") return args.paths;
        if (cmd === "stat_paths") {
          return (args?.paths || []).map((p) => ({ path: p, size_bytes: 100 }));
        }
      },
    });
    await waitFor(() => expect(dd.getHandler()).toBeTruthy());

    await dd.fire({ type: "drop", paths: ["/u/a.md", "/u/b.txt"] });

    await waitFor(() => expect(screen.getByText(/Input files \(2\)/)).toBeTruthy());
    expect(screen.queryByText(/files? ignored$/)).toBeNull();
  });

  it("all-rejected batch: 0 in staging, modal lists all rejections", async () => {
    const dd = stubDragDrop();
    await renderApp({
      override: (cmd, args) => {
        if (cmd === "expand_paths") return args.paths;
        if (cmd === "stat_paths") {
          return (args?.paths || []).map((p) => ({ path: p, size_bytes: 100 }));
        }
      },
    });
    await waitFor(() => expect(dd.getHandler()).toBeTruthy());

    await dd.fire({
      type: "drop",
      paths: ["/u/.DS_Store", "/u/setup.pkg", "/u/.gitignore"],
    });

    await screen.findByText("3 files ignored");
    // Staging list stays empty (no Input-files-(N) badge for N>0).
    expect(screen.queryByText(/Input files \([1-9]/)).toBeNull();
  });

  it("modal groups multiple rejections under one section per reason", async () => {
    const dd = stubDragDrop();
    await renderApp({
      override: (cmd, args) => {
        if (cmd === "expand_paths") return args.paths;
        if (cmd === "stat_paths") {
          return (args?.paths || []).map((p) => ({ path: p, size_bytes: 100 }));
        }
      },
    });
    await waitFor(() => expect(dd.getHandler()).toBeTruthy());

    await dd.fire({
      type: "drop",
      paths: [
        "/u/a/.DS_Store",
        "/u/b/.DS_Store",
        "/u/c/.DS_Store",
        "/u/x.dmg",
        "/u/y.dmg",
      ],
    });

    await screen.findByText("5 files ignored");
    // Two distinct reason groups → two h3 headers.
    const headers = document.querySelectorAll(".ignored-files-group h3");
    expect(headers.length).toBe(2);
    // Each section labels its own count next to the reason.
    expect(screen.getByText(/system file \(\.DS_Store\)/).textContent).toMatch(/\(3\)/);
    expect(screen.getByText(/unsupported extension \(\.dmg\)/).textContent).toMatch(/\(2\)/);
  });

  it("modal closes on OK button and on Escape", async () => {
    const dd = stubDragDrop();
    await renderApp({
      override: (cmd, args) => {
        if (cmd === "expand_paths") return args.paths;
        if (cmd === "stat_paths") {
          return (args?.paths || []).map((p) => ({ path: p, size_bytes: 100 }));
        }
      },
    });
    await waitFor(() => expect(dd.getHandler()).toBeTruthy());

    // First drop → open modal, click OK to close.
    await dd.fire({ type: "drop", paths: ["/u/a.dmg"] });
    await screen.findByText("1 file ignored");
    await userEvent.click(screen.getByRole("button", { name: "OK" }));
    await waitFor(() => expect(screen.queryByText(/file ignored/)).toBeNull());

    // Second drop → open again, dismiss with Escape.
    await dd.fire({ type: "drop", paths: ["/u/b.dmg"] });
    await screen.findByText("1 file ignored");
    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(screen.queryByText(/file ignored/)).toBeNull());
  });

  it("a `.txt` under a dot-prefixed directory is accepted (basename-only hidden check)", async () => {
    const dd = stubDragDrop();
    await renderApp({
      override: (cmd, args) => {
        if (cmd === "expand_paths") return args.paths;
        if (cmd === "stat_paths") {
          return (args?.paths || []).map((p) => ({ path: p, size_bytes: 100 }));
        }
      },
    });
    await waitFor(() => expect(dd.getHandler()).toBeTruthy());

    await dd.fire({ type: "drop", paths: ["/u/.cache/foo.txt"] });

    await waitFor(() => expect(screen.getByText(/Input files \(1\)/)).toBeTruthy());
    expect(screen.queryByText(/file ignored/)).toBeNull();
  });
});

// Issue #467: adding files scrolls the staging list to the bottom and
// flags the just-added rows with the shared transient-highlight (the
// same affordance scroll-to-anchor jumps use). The highlight must land
// on genuinely-new rows only — re-adding an already-staged path is a
// no-op, so its row must not re-flash.
describe("App — added files scroll into view + transient-highlight (issue #467)", () => {
  function statEcho() {
    return (cmd, args) => {
      if (cmd === "stat_paths") {
        return (args?.paths || []).map((p) => ({ path: p, size_bytes: 100 }));
      }
    };
  }

  it("just-added rows get .transient-highlight and the list scrolls to bottom", async () => {
    vi.mocked(openDialog).mockResolvedValueOnce(["/u/a.md", "/u/b.md"]);
    await renderApp({ override: statEcho() });

    await userEvent.click(screen.getByTitle("Add file(s)"));

    await waitFor(() => {
      expect(screen.getByText(/Input files \(2\)/)).toBeTruthy();
    });
    const list = document.querySelector(".inputs-list");
    await waitFor(() => {
      const rows = list.querySelectorAll("[data-input-path]");
      expect(rows.length).toBe(2);
      for (const row of rows) {
        expect(row.classList.contains("transient-highlight")).toBe(true);
      }
    });
    // Scrolled to the bottom (scrollHeight is 0 under jsdom's no-layout,
    // so this pins "scroll was driven to the end", not a pixel value).
    expect(list.scrollTop).toBe(list.scrollHeight);
  });

  it("re-adding an already-staged path does not re-flash it; only the new row highlights", async () => {
    vi.mocked(openDialog).mockResolvedValueOnce(["/u/a.md"]);
    await renderApp({ override: statEcho() });

    await userEvent.click(screen.getByTitle("Add file(s)"));
    await waitFor(() => {
      expect(screen.getByText(/Input files \(1\)/)).toBeTruthy();
    });
    const list = document.querySelector(".inputs-list");
    // Clear the first-add highlight so the second add's effect is the
    // only thing that can re-apply the class (the CSS `forwards` would
    // otherwise leave it on the DOM and mask the assertion).
    for (const row of list.querySelectorAll("[data-input-path]")) {
      row.classList.remove("transient-highlight");
    }

    // a.md is already staged; only b.md is genuinely new.
    vi.mocked(openDialog).mockResolvedValueOnce(["/u/a.md", "/u/b.md"]);
    await userEvent.click(screen.getByTitle("Add file(s)"));

    await waitFor(() => {
      expect(screen.getByText(/Input files \(2\)/)).toBeTruthy();
    });
    const rowA = list.querySelector('[data-input-path="/u/a.md"]');
    const rowB = list.querySelector('[data-input-path="/u/b.md"]');
    await waitFor(() => {
      expect(rowB.classList.contains("transient-highlight")).toBe(true);
    });
    expect(rowA.classList.contains("transient-highlight")).toBe(false);
  });

  // The modal title is the H2; the per-reason group also contains the
  // string "already imported", so assert the exact heading rather than
  // a substring (which would match both nodes).
  const modalTitle = () =>
    document.querySelector(".ignored-files-modal .modal-header h2")
      ?.textContent;

  it("re-adding already-staged files via the file picker: title omits the imported count", async () => {
    vi.mocked(openDialog).mockResolvedValueOnce(["/u/a.md", "/u/b.md"]);
    await renderApp({ override: statEcho() });

    await userEvent.click(screen.getByTitle("Add file(s)"));
    await waitFor(() => {
      expect(screen.getByText(/Input files \(2\)/)).toBeTruthy();
    });

    // Re-add a.md (already staged) + c.md (new) via the file picker.
    // a.md surfaces as a duplicate; c.md lands in the list. File pick
    // is not a folder load, so the "imported" clause is suppressed.
    vi.mocked(openDialog).mockResolvedValueOnce(["/u/a.md", "/u/c.md"]);
    await userEvent.click(screen.getByTitle("Add file(s)"));

    await waitFor(() => {
      expect(modalTitle()).toBe("1 file already imported");
    });
    expect(screen.getByText("a.md")).toBeTruthy();
    await waitFor(() => {
      expect(screen.getByText(/Input files \(3\)/)).toBeTruthy();
    });
  });

  it("folder load: title reports imported, already-imported, and ignored counts", async () => {
    // Pre-stage dup.md (file pick) so the folder load sees it as a
    // duplicate by exact path.
    vi.mocked(openDialog).mockResolvedValueOnce(["/u/f/dup.md"]);
    await renderApp({
      override: (cmd, args) => {
        if (cmd === "stat_paths") {
          return (args?.paths || []).map((p) => ({ path: p, size_bytes: 100 }));
        }
        if (cmd === "expand_paths") {
          return ["/u/f/dup.md", "/u/f/new.md", "/u/f/skip.dmg"];
        }
      },
    });
    await userEvent.click(screen.getByTitle("Add file(s)"));
    await waitFor(() => {
      expect(screen.getByText(/Input files \(1\)/)).toBeTruthy();
    });

    // One folder load hits all three: dup.md already staged, new.md
    // genuinely new (imported), skip.dmg rejected.
    vi.mocked(openDialog).mockResolvedValueOnce("/u/f");
    await userEvent.click(screen.getByTitle("Add folder (recursive)"));
    await waitFor(() => {
      expect(modalTitle()).toBe(
        "1 file imported, 1 file already imported, 1 file ignored",
      );
    });
    await waitFor(() => {
      expect(screen.getByText(/Input files \(2\)/)).toBeTruthy();
    });
  });

  it("a pure-duplicate batch does not scroll or re-highlight any row", async () => {
    vi.mocked(openDialog).mockResolvedValueOnce(["/u/a.md"]);
    await renderApp({ override: statEcho() });

    await userEvent.click(screen.getByTitle("Add file(s)"));
    await waitFor(() => {
      expect(screen.getByText(/Input files \(1\)/)).toBeTruthy();
    });
    const list = document.querySelector(".inputs-list");
    for (const row of list.querySelectorAll("[data-input-path]")) {
      row.classList.remove("transient-highlight");
    }

    // Re-add only the already-staged path: the list is unchanged, the
    // modal opens, and no row gets re-flagged.
    vi.mocked(openDialog).mockResolvedValueOnce(["/u/a.md"]);
    await userEvent.click(screen.getByTitle("Add file(s)"));

    await waitFor(() => {
      expect(modalTitle()).toBe("1 file already imported");
    });
    expect(screen.getByText(/Input files \(1\)/)).toBeTruthy();
    const rowA = list.querySelector('[data-input-path="/u/a.md"]');
    expect(rowA.classList.contains("transient-highlight")).toBe(false);
  });
});

// Issue #159: live elapsed-time counter + barber-pole "alive" indicator
// on the progress bar. Covers the two helpers (fmtElapsed, liveElapsedMs)
// plus the App-level wiring: shared 1Hz ticker gated on running runs,
// reduced-motion respect, no-tick-when-idle.

describe("fmtElapsed (issue #159)", () => {
  it("formats sub-minute as Ns (no padding, no quantization)", () => {
    expect(fmtElapsed(0)).toBe("0s");
    expect(fmtElapsed(1_000)).toBe("1s");
    expect(fmtElapsed(42_000)).toBe("42s");
    expect(fmtElapsed(59_999)).toBe("59s");
  });
  it("formats sub-hour as `Mm` or `Mm Ss`", () => {
    expect(fmtElapsed(60_000)).toBe("1m");
    expect(fmtElapsed(8 * 60_000 + 23_000)).toBe("8m 23s");
    expect(fmtElapsed(60 * 60_000 - 1)).toBe("59m 59s");
  });
  it("formats >=1h as `Hh Mm Ss` — seconds always shown so it visibly ticks past 1h (issue #586)", () => {
    expect(fmtElapsed(60 * 60_000)).toBe("1h 0m 0s");
    expect(fmtElapsed(60 * 60_000 + 12 * 60_000)).toBe("1h 12m 0s");
    expect(fmtElapsed(5 * 3600_000 + 18 * 60_000 + 42_000)).toBe("5h 18m 42s");
    // The exact symptom from the issue: 286m29s must roll to hours.
    expect(fmtElapsed((286 * 60 + 29) * 1000)).toBe("4h 46m 29s");
  });
  it("returns null for invalid inputs (so callers can skip rendering)", () => {
    expect(fmtElapsed(null)).toBeNull();
    expect(fmtElapsed(undefined)).toBeNull();
    expect(fmtElapsed(NaN)).toBeNull();
  });
  it("clamps negative values to 0s (defensive against clock skew)", () => {
    expect(fmtElapsed(-5)).toBe("0s");
  });
});

describe("liveElapsedMs (issue #159 + #617)", () => {
  it("returns backend duration_ms verbatim when set (running)", () => {
    // #617: running runs no longer tick from `now − created_at` —
    // they display the backend's active-runtime duration_ms (already
    // accumulator-derived). The created_at is irrelevant when
    // duration_ms is present.
    const run = {
      status: "running",
      created_at: "2026-04-29T12:00:00Z",
      duration_ms: 7_500,
    };
    expect(liveElapsedMs(run, Date.parse("2026-04-29T13:00:00Z"))).toBe(7_500);
  });
  it("returns backend duration_ms verbatim when set (paused)", () => {
    const run = {
      status: "paused",
      created_at: "2026-04-29T12:00:00Z",
      duration_ms: 42_000,
    };
    expect(liveElapsedMs(run, Date.parse("2026-04-29T13:00:00Z"))).toBe(42_000);
  });
  it("freezes for completed runs at duration_ms regardless of nowMs", () => {
    const run = {
      status: "completed",
      created_at: "2026-04-29T12:00:00Z",
      duration_ms: 30_000,
    };
    // Advance nowMs by an hour — frozen value must NOT change.
    expect(liveElapsedMs(run, Date.parse("2026-04-29T13:00:00Z"))).toBe(30_000);
  });
  it("freezes for cancelled runs at duration_ms (same rule as completed)", () => {
    const run = {
      status: "cancelled",
      created_at: "2026-04-29T12:00:00Z",
      duration_ms: 12_345,
    };
    expect(liveElapsedMs(run, Date.parse("2026-04-29T13:00:00Z"))).toBe(12_345);
  });
  it("falls back to (nowMs − created_at) when running with NO duration_ms yet (pre-cycle_start window)", () => {
    // A fresh run whose Python sidecar hasn't emitted its first
    // cycle_start yet has no backend duration_ms. We tick from
    // created_at so the display isn't blank during the startup
    // window. Once cycle_start lands the next list_runs poll will
    // supply duration_ms and the fallback stops firing.
    const created = Date.parse("2026-04-29T12:00:00Z");
    const run = { status: "running", created_at: "2026-04-29T12:00:00Z" };
    expect(liveElapsedMs(run, created + 30_000)).toBe(30_000);
    expect(liveElapsedMs(run, created + 31_000)).toBe(31_000);
  });
  it("returns null when running with no duration_ms AND no parseable created_at", () => {
    const run = { status: "running", created_at: "not-a-date" };
    expect(liveElapsedMs(run, Date.now())).toBeNull();
  });
  it("returns null for non-running run with no duration_ms (nothing to display)", () => {
    const run = { status: "paused", created_at: "2026-04-29T12:00:00Z" };
    expect(liveElapsedMs(run, Date.now())).toBeNull();
  });
  it("returns null when run is undefined/null", () => {
    expect(liveElapsedMs(null, Date.now())).toBeNull();
    expect(liveElapsedMs(undefined, Date.now())).toBeNull();
  });
  it("duration_ms takes precedence over the running branch (resume jump regression guard, #617)", () => {
    // After pause + resume, the backend reports cumulative active-
    // runtime via duration_ms. Even though status=running, we
    // display the backend value (NOT now − created_at, which would
    // include the pause window — the original #617 bug).
    const run = {
      status: "running",
      created_at: "2026-04-29T11:00:00Z",   // 1h before "now"
      duration_ms: 5_000,                    // ...but backend says only 5s active
    };
    const nowMs = Date.parse("2026-04-29T12:00:00Z");
    expect(liveElapsedMs(run, nowMs)).toBe(5_000);
  });
});

describe("freezeAbortedCallDurations (#617)", () => {
  // The per-call modal's `outcome=aborted` rows display c.duration_ms,
  // which the backend overlay re-stamps on every read against `now`.
  // UI-only freeze: snapshot the first-observed duration_ms for an
  // aborted (runId, callId) pair and use that forever. Aborted never
  // updates its timer.

  // Each test starts from a clean cache so order doesn't leak.
  // eslint-disable-next-line no-undef
  beforeEach(() => _resetFrozenAbortedCacheForTests());

  it("snapshots duration_ms on first observation of an aborted call", () => {
    const stats = {
      calls: [
        { call_id: "0001", outcome: "aborted", duration_ms: 12_345 },
      ],
    };
    const out = freezeAbortedCallDurations(stats, "run-A");
    expect(out.calls[0].duration_ms).toBe(12_345);
  });

  it("uses cached snapshot on subsequent observations regardless of backend value", () => {
    freezeAbortedCallDurations(
      { calls: [{ call_id: "0001", outcome: "aborted", duration_ms: 12_345 }] },
      "run-A",
    );
    // Backend hands back a different value on the next poll (e.g.
    // overlay_live_token_state re-stamped now − started_at_iso).
    const out = freezeAbortedCallDurations(
      { calls: [{ call_id: "0001", outcome: "aborted", duration_ms: 99_999 }] },
      "run-A",
    );
    // Cached value wins; the new 99_999 is ignored.
    expect(out.calls[0].duration_ms).toBe(12_345);
  });

  it("does NOT freeze calls with other outcomes (pending, success, skipped, etc.)", () => {
    const stats = {
      calls: [
        { call_id: "0001", outcome: "pending", duration_ms: 1_000 },
        { call_id: "0002", outcome: "success", duration_ms: 2_000 },
        { call_id: "0003", outcome: "skipped", duration_ms: 3_000 },
        { call_id: "0004", outcome: "failed (other)", duration_ms: 4_000 },
      ],
    };
    const out = freezeAbortedCallDurations(stats, "run-A");
    // None of these snapshots into the aborted cache; passthrough.
    expect(out.calls.map((c) => c.duration_ms)).toEqual([1_000, 2_000, 3_000, 4_000]);
    // And a subsequent call with new values should pass through too.
    const out2 = freezeAbortedCallDurations(
      {
        calls: [
          { call_id: "0001", outcome: "pending", duration_ms: 1_500 },
          { call_id: "0002", outcome: "success", duration_ms: 2_500 },
        ],
      },
      "run-A",
    );
    expect(out2.calls.map((c) => c.duration_ms)).toEqual([1_500, 2_500]);
  });

  it("scopes cache by (runId, callId) — same call_id across two runs cache independently", () => {
    freezeAbortedCallDurations(
      { calls: [{ call_id: "0001", outcome: "aborted", duration_ms: 100 }] },
      "run-A",
    );
    const outB = freezeAbortedCallDurations(
      { calls: [{ call_id: "0001", outcome: "aborted", duration_ms: 200 }] },
      "run-B",
    );
    // run-B's 0001 wasn't in cache yet — its first observation snapshots
    // independently from run-A's 0001.
    expect(outB.calls[0].duration_ms).toBe(200);
    // And run-A's still pinned to 100.
    const outA2 = freezeAbortedCallDurations(
      { calls: [{ call_id: "0001", outcome: "aborted", duration_ms: 999 }] },
      "run-A",
    );
    expect(outA2.calls[0].duration_ms).toBe(100);
  });

  it("preserves other call fields and leaves non-aborted rows untouched", () => {
    freezeAbortedCallDurations(
      { calls: [{ call_id: "0001", outcome: "aborted", duration_ms: 100 }] },
      "run-A",
    );
    const out = freezeAbortedCallDurations(
      {
        calls: [
          { call_id: "0001", outcome: "aborted", duration_ms: 9999, ttft_ms: 42, model: "x" },
          { call_id: "0002", outcome: "success", duration_ms: 2_000 },
        ],
      },
      "run-A",
    );
    expect(out.calls[0]).toEqual({
      call_id: "0001",
      outcome: "aborted",
      duration_ms: 100,
      ttft_ms: 42,
      model: "x",
    });
    expect(out.calls[1]).toEqual({
      call_id: "0002",
      outcome: "success",
      duration_ms: 2_000,
    });
  });

  it("no-op on missing/invalid stats", () => {
    expect(freezeAbortedCallDurations(null, "run-A")).toBeNull();
    expect(freezeAbortedCallDurations({}, "run-A")).toEqual({});
    expect(freezeAbortedCallDurations({ calls: [] }, "run-A")).toEqual({ calls: [] });
    // Missing runId: passthrough (we'd cache under an unknown key).
    const stats = { calls: [{ call_id: "0001", outcome: "aborted", duration_ms: 100 }] };
    expect(freezeAbortedCallDurations(stats, undefined)).toBe(stats);
  });

  it("first observation with non-finite duration_ms doesn't poison the cache", () => {
    // If the backend hasn't computed duration_ms yet (null), DON'T cache.
    // Wait until a finite value lands, then snapshot that.
    const out1 = freezeAbortedCallDurations(
      { calls: [{ call_id: "0001", outcome: "aborted", duration_ms: null }] },
      "run-A",
    );
    expect(out1.calls[0].duration_ms).toBeNull();
    // Later poll lands a real value — cache snapshots it.
    const out2 = freezeAbortedCallDurations(
      { calls: [{ call_id: "0001", outcome: "aborted", duration_ms: 500 }] },
      "run-A",
    );
    expect(out2.calls[0].duration_ms).toBe(500);
    // Subsequent polls keep the snapshot.
    const out3 = freezeAbortedCallDurations(
      { calls: [{ call_id: "0001", outcome: "aborted", duration_ms: 99_999 }] },
      "run-A",
    );
    expect(out3.calls[0].duration_ms).toBe(500);
  });
});

describe("App — live elapsed counter (issue #159)", () => {
  // Fixture builders — runningRun's created_at is whatever the caller
  // sets so each test can pin elapsed deterministically.
  function makeRunning({ id = "aaaa", createdAt, mode = "tee" } = {}) {
    return {
      run_id: `2026-04-29T12-00-00Z-${id}`,
      short_id: id,
      status: "running",
      mode,
      inputs: ["/x.md"],
      input_count: 1,
      created_at: createdAt,
      progress: { stage: "extract", completed: 5, total: 50 },
    };
  }
  function makeCompleted({ id = "dddd", durationMs = 30_000 } = {}) {
    return {
      run_id: `2026-04-29T12-00-00Z-${id}`,
      short_id: id,
      status: "completed",
      mode: "tee",
      inputs: ["/x.md"],
      input_count: 1,
      created_at: "2026-04-29T12:00:00Z",
      duration_ms: durationMs,
      progress: { stage: "done", completed: 50, total: 50 },
    };
  }

  it("running row shows `Elapsed 30s` and ticks to `31s` after 1s", async () => {
    const fixedNow = Date.parse("2026-04-29T12:00:30.000Z");
    vi.useFakeTimers({ now: fixedNow, shouldAdvanceTime: true });
    try {
      const run = makeRunning({ createdAt: "2026-04-29T12:00:00.000Z" });
      await renderApp({ runs: [run] });
      await waitFor(() => {
        expect(screen.getByText(/Elapsed 30s/)).toBeTruthy();
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1100);
      });
      expect(screen.getByText(/Elapsed 31s/)).toBeTruthy();
    } finally {
      vi.useRealTimers();
    }
  });

  it("running row's `.run-dur` (the 'running' time) === the live Elapsed value, ticking on the same tick (issue #586)", async () => {
    // created_at 1h05m07s before fixedNow — past 1h so the hour-rolled
    // `Hh Mm Ss` format is exercised and the seconds digit must tick.
    const fixedNow = Date.parse("2026-04-29T13:05:07.000Z");
    vi.useFakeTimers({ now: fixedNow, shouldAdvanceTime: true });
    try {
      const run = makeRunning({ createdAt: "2026-04-29T12:00:00.000Z" });
      await renderApp({ runs: [run] });
      const runDur = await waitFor(() => {
        const el = document.querySelector(".run-dur");
        expect(el).toBeTruthy();
        return el;
      });
      // Hour-rolled, seconds shown — never a raw "65m..." minute form.
      expect(runDur.textContent).toBe("1h 5m 7s");
      // And it is the SAME string as the Elapsed label under the bar.
      expect(screen.getByText(/Elapsed 1h 5m 7s/)).toBeTruthy();
      // One second later both advance together (same clock, not stale).
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1100);
      });
      expect(document.querySelector(".run-dur").textContent).toBe("1h 5m 8s");
      expect(screen.getByText(/Elapsed 1h 5m 8s/)).toBeTruthy();
    } finally {
      vi.useRealTimers();
    }
  });

  it("completed-only run list mounts NO 1Hz interval (idle CPU when nothing's in flight)", async () => {
    const setIntervalSpy = vi.spyOn(globalThis, "setInterval");
    try {
      await renderApp({ runs: [makeCompleted()] });
      // The 500ms list_runs poll IS expected; the 1Hz elapsed ticker is NOT.
      const oneHertz = setIntervalSpy.mock.calls.filter((c) => c[1] === 1000);
      expect(oneHertz.length).toBe(0);
      // And the visible Elapsed label never appears for completed rows
      // (the run-dur span still derives from duration_ms via
      // liveElapsedMs, just formatted through the shared fmtElapsed).
      expect(screen.queryByText(/Elapsed /)).toBeNull();
    } finally {
      setIntervalSpy.mockRestore();
    }
  });

  it("two running runs share a single 1Hz ticker; both labels tick together", async () => {
    const fixedNow = Date.parse("2026-04-29T12:00:10.000Z");
    vi.useFakeTimers({ now: fixedNow, shouldAdvanceTime: true });
    const setIntervalSpy = vi.spyOn(globalThis, "setInterval");
    try {
      const a = makeRunning({ id: "aaaa", createdAt: "2026-04-29T12:00:00.000Z" });
      const b = makeRunning({ id: "bbbb", createdAt: "2026-04-29T12:00:05.000Z" });
      await renderApp({ runs: [a, b] });
      await waitFor(() => {
        expect(screen.getByText(/Elapsed 10s/)).toBeTruthy();
        expect(screen.getByText(/Elapsed 5s/)).toBeTruthy();
      });
      // Exactly ONE 1Hz interval is mounted across both rows.
      const oneHertzCalls = setIntervalSpy.mock.calls.filter((c) => c[1] === 1000);
      expect(oneHertzCalls.length).toBe(1);
      // Advance 2s — both tick on the shared timer.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2100);
      });
      expect(screen.getByText(/Elapsed 12s/)).toBeTruthy();
      expect(screen.getByText(/Elapsed 7s/)).toBeTruthy();
    } finally {
      setIntervalSpy.mockRestore();
      vi.useRealTimers();
    }
  });

  it("respects prefers-reduced-motion: reduce — drops the .is-running animation class", async () => {
    const orig = window.matchMedia;
    window.matchMedia = vi.fn().mockImplementation((q) => ({
      matches: q === "(prefers-reduced-motion: reduce)",
      media: q,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      onchange: null,
      dispatchEvent: vi.fn(),
    }));
    try {
      const run = makeRunning({ createdAt: "2026-04-29T12:00:00.000Z" });
      await renderApp({ runs: [run] });
      const fill = document.querySelector(".progress-fill");
      expect(fill).toBeTruthy();
      // Class is suppressed in JSX (defense in depth: CSS @media also
      // disables animation if the class slips through somehow).
      expect(fill.classList.contains("is-running")).toBe(false);
      // Sanity: progress-fill-running is still applied — only the
      // animation class is the one being gated by motion preference.
      expect(fill.classList.contains("progress-fill-running")).toBe(true);
    } finally {
      window.matchMedia = orig;
    }
  });

  it("running row WITHOUT reduced motion has the .is-running animation class", async () => {
    const run = makeRunning({ createdAt: "2026-04-29T12:00:00.000Z" });
    await renderApp({ runs: [run] });
    const fill = await waitFor(() => {
      const el = document.querySelector(".progress-fill");
      expect(el).toBeTruthy();
      return el;
    });
    expect(fill.classList.contains("is-running")).toBe(true);
  });
});

// Issue #161: tighten the run-row card. Provider line drops the
// model segment (it's redundant with the Details modal + RunDetails
// routing snapshot); progress info splits across stacked lines so
// each datum gets its own DOM node — narrow sidebars wrap gracefully
// and tests can pin exact values per line.

describe("App — run-row layout (issue #161)", () => {
  it("paused runs render the stacked layout (stage line + counts) without an Elapsed/ETA line when neither is computable", async () => {
    // Paused runs are still in the running-or-paused branch but the
    // ticker freezes — the Elapsed line should be omitted entirely
    // when there's nothing to put in it (no fmtElapsed-able value
    // and no ETA), rather than render an empty span.
    await renderApp({
      runs: [
        {
          run_id: "2026-04-29T12-00-00Z-pp01",
          short_id: "pp01",
          status: "paused",
          mode: "tee",
          inputs: ["/x.md"],
          input_count: 1,
          // unparseable created_at + no duration_ms ⇒ liveElapsedMs returns null
          created_at: "not-a-date",
          progress: { stage: "extract", completed: 4, total: 50 },
          provider: "tinfoil",
        },
      ],
    });
    const stage = document.querySelector(".progress-stage");
    expect(stage.textContent).toBe("Extracting facts");
    const counts = document.querySelector(".progress-counts");
    expect(counts.textContent).toBe("8% completed (4 / 50 estimated calls)");
    // Elapsed line is omitted — no node, not an empty one.
    expect(document.querySelector(".progress-elapsed")).toBeNull();
  });

  it("local-mode running run with no provider just shows the mode label (no parens, no TEE)", async () => {
    // Defensive: a run with mode but no provider/model shouldn't
    // render dangling parens. The new render path keys parens off
    // run.provider; absent provider ⇒ no parens, no TEE.
    await renderApp({
      runs: [
        {
          run_id: "2026-04-29T12-00-00Z-loc1",
          short_id: "loc1",
          status: "running",
          mode: "local",
          inputs: ["/x.md"],
          input_count: 1,
          created_at: "2026-04-29T12:00:00Z",
          progress: { stage: "extract", completed: 1, total: 10 },
          // provider / model deliberately omitted
        },
      ],
    });
    const engine = document.querySelector(".run-engine");
    expect(engine).toBeTruthy();
    expect(engine.textContent.trim()).toBe("Local");
  });
});

// Issue #98: clicking 4-insights.md / 5-actions.md in the tree expands
// the file node into one nested entry per item (cross-domain N + critical
// N for insights; action N for actions). Clicking an entry sets the
// selected file's anchor → MarkdownPreview's existing scroll-to-anchor +
// transient-highlight primitive picks it up. Other markdown files (facts,
// patterns, README) keep the existing render: no nested entries.
describe("App — insights/actions tree-nest (issue #98)", () => {
  const completedRun = {
    run_id: "2026-04-29T12-00-00Z-nest",
    short_id: "nest",
    status: "completed",
    mode: "tee",
    inputs: ["/u/note.md"],
    input_count: 1,
    created_at: "2026-04-29T12:00:00Z",
    duration_ms: 30_000,
    progress: { stage: "done", completed: 50, total: 50 },
    vault_exists: true,
  };

  const treeWithStages = [
    {
      name: "Patterns (1)",
      rel_path: "3-patterns",
      is_dir: true,
      children: [
        {
          name: "Work (1)",
          rel_path: "3-patterns/work.md",
          is_dir: false,
          children: [],
        },
      ],
    },
    {
      name: "Insights (3)",
      rel_path: "4-insights.md",
      is_dir: false,
      children: [],
    },
    {
      name: "Actions (2)",
      rel_path: "5-actions.md",
      is_dir: false,
      children: [],
    },
  ];

  const insightsPayload = {
    cross_domain: [
      { name: "Cross-A", kind: "cross_domain", domains: ["work"] },
      { name: "Cross-B", kind: "cross_domain", domains: ["health"] },
    ],
    critical: [{ name: "Crit-A", kind: "critical", domains: ["work"] }],
  };
  const actionsPayload = {
    actions: [
      { recommendation: "Take action one", horizon: "short" },
      { recommendation: "Take action two", horizon: "long" },
    ],
  };

  // Helper: scope text matchers to the tree pane so we don't collide
  // with the markdown body, which renders the same labels as H2/H3.
  const treeWithin = () => within(document.querySelector(".tree-pane"));

  it("renders 4-insights.md expanded by default — one entry per cross-domain + critical insight with no click; entry click jumps with anchor", async () => {
    await renderApp({
      runs: [completedRun],
      override: (cmd, args) => {
        if (cmd === "list_run_tree" && args.runId === completedRun.run_id) {
          return treeWithStages;
        }
        if (cmd === "read_run_facts") return [];
        if (cmd === "read_run_insights") return insightsPayload;
        if (cmd === "read_run_actions") return actionsPayload;
        return undefined;
      },
    });
    await userEvent.click(screen.getByText("note").closest(".run-row"));
    // Insights is open by default on run-select: the nested entries
    // render with NO click (treeDefaultDirs seeds 4-insights.md open).
    await waitFor(() => {
      expect(treeWithin().getByText("Cross-A")).toBeTruthy();
      expect(treeWithin().getByText("Cross-B")).toBeTruthy();
      expect(treeWithin().getByText("Crit-A")).toBeTruthy();
    });
    // Entries render under the insights file's own .tree-file-entries
    // with one .tree-name each. Scope to that file node — 5-actions.md
    // is ALSO open by default now, so its entries share the tree pane.
    const insightsFile = treeWithin()
      .getByText("Insights (3)")
      .closest(".tree-file");
    const entries = insightsFile.querySelectorAll(
      ".tree-file-entries .tree-file-entry",
    );
    expect(entries.length).toBe(3);
    // Order matches obsidianRenderer: cross-domain first, then critical.
    const labels = Array.from(entries).map(
      (li) => li.querySelector(".tree-name").textContent,
    );
    expect(labels).toEqual(["Cross-A", "Cross-B", "Crit-A"]);
    // Numbering is CONTINUOUS across cross-domain → critical (so the
    // critical entry is "3", not "1"). Anchors stay scoped per-section
    // (`critical-1`, not `critical-3`) to match the renderer.
    const nums = Array.from(entries).map(
      (li) => li.querySelector(".tree-num")?.textContent,
    );
    expect(nums).toEqual(["1", "2", "3"]);

    // Clicking an entry routes to setSelectedFile with the matching
    // anchor — Cross-B is the second cross-domain insight, so the
    // anchor is `cross-domain-2` (mirrors insightAnchor()).
    await userEvent.click(
      treeWithin().getByText("Cross-B").closest(".tree-row"),
    );
    // The entry shows .selected when the anchor matches.
    await waitFor(() => {
      const selectedEntry = document.querySelector(
        ".tree-file-entry.selected .tree-name",
      );
      expect(selectedEntry?.textContent).toBe("Cross-B");
    });
  });

  it("the collapse toggle still works — clicking the open-by-default file row collapses, then re-expands the nested entries", async () => {
    await renderApp({
      runs: [completedRun],
      override: (cmd, args) => {
        if (cmd === "list_run_tree" && args.runId === completedRun.run_id) {
          return treeWithStages;
        }
        if (cmd === "read_run_facts") return [];
        if (cmd === "read_run_insights") return insightsPayload;
        if (cmd === "read_run_actions") return actionsPayload;
        return undefined;
      },
    });
    await userEvent.click(screen.getByText("note").closest(".run-row"));
    // Open by default → entries present on select, no click.
    await waitFor(() => {
      expect(treeWithin().getByText("Cross-A")).toBeTruthy();
    });
    // First click on the file row — collapse. Entries leave the DOM.
    await userEvent.click(
      treeWithin().getByText("Insights (3)").closest(".tree-row"),
    );
    await waitFor(() => {
      expect(treeWithin().queryByText("Cross-A")).toBeNull();
    });
    // Second click — expand again.
    await userEvent.click(
      treeWithin().getByText("Insights (3)").closest(".tree-row"),
    );
    await waitFor(() => {
      expect(treeWithin().getByText("Cross-A")).toBeTruthy();
    });
  });

  it("renders 5-actions.md expanded by default — one entry per action with no click; anchors match action-N", async () => {
    await renderApp({
      runs: [completedRun],
      override: (cmd, args) => {
        if (cmd === "list_run_tree" && args.runId === completedRun.run_id) {
          return treeWithStages;
        }
        if (cmd === "read_run_facts") return [];
        if (cmd === "read_run_insights") return insightsPayload;
        if (cmd === "read_run_actions") return actionsPayload;
        return undefined;
      },
    });
    await userEvent.click(screen.getByText("note").closest(".run-row"));
    // Actions is open by default → entries render with no click.
    await waitFor(() => {
      expect(treeWithin().getByText("Take action one")).toBeTruthy();
      expect(treeWithin().getByText("Take action two")).toBeTruthy();
    });
    // Each entry carries its action index in .tree-num. Scope to the
    // actions file node — 4-insights.md is ALSO open by default now.
    const actionsFile = treeWithin()
      .getByText("Actions (2)")
      .closest(".tree-file");
    const actionNums = Array.from(
      actionsFile.querySelectorAll(".tree-file-entries .tree-num"),
    ).map((n) => n.textContent);
    expect(actionNums).toEqual(["1", "2"]);
    // Click the second action → anchor `action-2`.
    await userEvent.click(
      treeWithin().getByText("Take action two").closest(".tree-row"),
    );
    await waitFor(() => {
      const selectedEntry = document.querySelector(
        ".tree-file-entry.selected .tree-name",
      );
      expect(selectedEntry?.textContent).toBe("Take action two");
    });
  });

  it("tree-entry clicks do NOT trigger the .transient-highlight (TOC-style nav: no flash on every click)", async () => {
    await renderApp({
      runs: [completedRun],
      override: (cmd, args) => {
        if (cmd === "list_run_tree" && args.runId === completedRun.run_id) {
          return treeWithStages;
        }
        if (cmd === "read_run_facts") return [];
        if (cmd === "read_run_insights") return insightsPayload;
        if (cmd === "read_run_actions") return actionsPayload;
        return undefined;
      },
    });
    await userEvent.click(screen.getByText("note").closest(".run-row"));
    // Actions is open by default → the nested entry is present with no
    // click. Click a nested entry → scroll-to-anchor should fire BUT
    // without the transient highlight.
    await waitFor(() => {
      expect(treeWithin().getByText("Take action one")).toBeTruthy();
    });
    await userEvent.click(
      treeWithin().getByText("Take action one").closest(".tree-row"),
    );
    // The matching section in the markdown body must NOT receive
    // .transient-highlight; the entry row itself uses .selected.
    await waitFor(() => {
      const selectedEntry = document.querySelector(
        ".tree-file-entry.selected .tree-name",
      );
      expect(selectedEntry?.textContent).toBe("Take action one");
    });
    // Section id="action-1" lives in the markdown body; it must NOT
    // carry .transient-highlight after a tree click.
    const section = document.querySelector("[id=\"action-1\"]");
    expect(section).toBeTruthy();
    const ancestor = section.closest(".md-section");
    expect(ancestor?.classList.contains("transient-highlight")).toBe(false);
    expect(section.classList.contains("transient-highlight")).toBe(false);
  });

  it("non-insight/action markdown files render with NO nested entries", async () => {
    await renderApp({
      runs: [completedRun],
      override: (cmd, args) => {
        if (cmd === "list_run_tree" && args.runId === completedRun.run_id) {
          return treeWithStages;
        }
        if (cmd === "read_run_facts") return [];
        if (cmd === "read_run_insights") return insightsPayload;
        if (cmd === "read_run_actions") return actionsPayload;
        return undefined;
      },
    });
    await userEvent.click(screen.getByText("note").closest(".run-row"));
    await waitFor(() => {
      expect(screen.getByText("Work (1)")).toBeTruthy();
    });
    // Click a patterns file (not insights/actions) — no nested entries.
    await userEvent.click(screen.getByText("Work (1)").closest(".tree-row"));
    // Give the headings effect a chance to settle, then assert nothing
    // rendered under the patterns file.
    await waitFor(() => {
      const work = screen.getByText("Work (1)").closest(".tree-file");
      expect(work.querySelector(".tree-file-entries")).toBeNull();
    });
  });

  it("long heading text uses the same .tree-name truncation as input filenames", async () => {
    // The input-filename truncation rule on this tree is the
    // `.tree-name` class with CSS `overflow: hidden; text-overflow:
    // ellipsis;` (App.css). Reusing the class on entry labels means
    // they truncate by the same rule — assert the class is applied
    // and the full text lives in the title attribute so hover reveals
    // it.
    const longName = "A".repeat(200);
    const customInsights = {
      cross_domain: [{ name: longName, kind: "cross_domain", domains: [] }],
      critical: [],
    };
    await renderApp({
      runs: [completedRun],
      override: (cmd, args) => {
        if (cmd === "list_run_tree" && args.runId === completedRun.run_id) {
          return [
            {
              name: "Insights (1)",
              rel_path: "4-insights.md",
              is_dir: false,
              children: [],
            },
          ];
        }
        if (cmd === "read_run_facts") return [];
        if (cmd === "read_run_insights") return customInsights;
        if (cmd === "read_run_actions") return null;
        return undefined;
      },
    });
    await userEvent.click(screen.getByText("note").closest(".run-row"));
    // Insights is open by default → the entry renders with no click.
    await waitFor(() => {
      const entryName = document.querySelector(
        ".tree-file-entry .tree-name",
      );
      expect(entryName).toBeTruthy();
      expect(entryName.textContent).toBe(longName);
    });
    // Hover surface for the full label: title attribute on the row.
    const entryRow = document.querySelector(".tree-file-entry .tree-row");
    expect(entryRow.getAttribute("title")).toBe(longName);
  });
});

// Native File menu (Rust emits menu-add-file / menu-add-folder /
// menu-export-selected on click). The privacy-gate regression: an
// earlier wiring pointed Export Selected at a removed `exportSelectedRuns`
// (undefined → throws at fire time, modal never opens); the contract is
// that the menu routes into the same `openPrivacyExport` front-door the
// bottom-bar Export button uses, so the Privacy Level cut can't be
// bypassed from the menu.
describe("App — native File menu", () => {
  const completedRun = {
    run_id: "2026-04-29T12-00-00Z-fmnu",
    short_id: "fmnu",
    status: "completed",
    mode: "tee",
    inputs: ["/u/big.md"],
    input_count: 1,
    created_at: "2026-04-29T12:00:00Z",
    duration_ms: 30_000,
    progress: { stage: "done", completed: 50, total: 50 },
    vault_exists: true,
  };

  function captureMenuHandlers() {
    const handlers = {};
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      handlers[evt] = fn;
      return () => {};
    });
    return handlers;
  }

  it("registers listeners for all three native File-menu events", async () => {
    const handlers = captureMenuHandlers();
    await renderApp({ runs: [completedRun] });
    await waitFor(() => {
      expect(typeof handlers["menu-add-file"]).toBe("function");
      expect(typeof handlers["menu-add-folder"]).toBe("function");
      expect(typeof handlers["menu-export-selected"]).toBe("function");
    });
  });

  it("Export Selected opens the Privacy Level modal for the selected run (no bypass)", async () => {
    const handlers = captureMenuHandlers();
    await renderApp({
      runs: [completedRun],
      override: (cmd) => {
        if (cmd === "export_default_dir") return "/tmp/exports";
        return undefined;
      },
    });

    // Select the run so openPrivacyExport has a non-empty selection.
    await userEvent.click(screen.getByText("big").closest(".run-row"));

    // Not open before the menu fires — the menu must not auto-open it
    // and nothing else opens it on mount.
    expect(screen.queryByText("Privacy Level")).toBeNull();

    // Simulate the Rust File ▸ Export Selected click emitting the event.
    await act(async () => {
      await handlers["menu-export-selected"]();
    });

    // The same modal the bottom-bar Export button opens (its <h1>),
    // proving the menu routes through openPrivacyExport, not a bypass.
    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: "Privacy Level" }),
      ).toBeTruthy();
    });
  });

  it("Export Selected with no run selected does not open the modal", async () => {
    const handlers = captureMenuHandlers();
    await renderApp({ runs: [completedRun] });
    await act(async () => {
      await handlers["menu-export-selected"]();
    });
    expect(screen.queryByText("Privacy Level")).toBeNull();
  });

  it("syncs the native Export Selected enabled state with the run selection", async () => {
    captureMenuHandlers();
    await renderApp({ runs: [completedRun] });

    const enabledArgs = () =>
      vi
        .mocked(invoke)
        .mock.calls.filter((c) => c[0] === "set_export_menu_enabled")
        .map((c) => c[1].enabled);

    // No selection on mount → disabled.
    await waitFor(() => expect(enabledArgs()).toContain(false));
    expect(enabledArgs().every((e) => e === false)).toBe(true);

    // Selecting a run → enabled true.
    await userEvent.click(screen.getByText("big").closest(".run-row"));
    await waitFor(() => expect(enabledArgs()).toContain(true));
  });

  it("keeps export disabled when only an unfinished run is selected", async () => {
    const handlers = captureMenuHandlers();
    const runningRun = {
      ...completedRun,
      run_id: "2026-04-29T12-00-00Z-rung",
      short_id: "rung",
      status: "running",
      inputs: ["/u/wip.md"],
      progress: { stage: "extract", completed: 5, total: 50 },
    };
    await renderApp({ runs: [runningRun] });

    await userEvent.click(screen.getByText("wip").closest(".run-row"));

    // Native menu item: never enabled for an all-unfinished selection.
    const enabledArgs = vi
      .mocked(invoke)
      .mock.calls.filter((c) => c[0] === "set_export_menu_enabled")
      .map((c) => c[1].enabled);
    expect(enabledArgs.every((e) => e === false)).toBe(true);

    // Bottom-bar Export button: present (a run is selected) but
    // disabled, and labeled just "Export" — no count for a selection
    // with nothing exportable.
    const exportBtn = screen.getByRole("button", { name: "Export" });
    expect(exportBtn.disabled).toBe(true);

    // Native menu fire is a no-op — no Privacy modal for an
    // unfinished-only selection.
    await act(async () => {
      await handlers["menu-export-selected"]();
    });
    expect(screen.queryByText("Privacy Level")).toBeNull();
  });

  it("Export button counts only the finished runs in the selection", async () => {
    captureMenuHandlers();
    const runningRun = {
      ...completedRun,
      run_id: "2026-04-29T12-00-00Z-rng2",
      short_id: "rng2",
      status: "running",
      inputs: ["/u/wip.md"],
      progress: { stage: "extract", completed: 5, total: 50 },
    };
    await renderApp({ runs: [runningRun, completedRun] });

    // Select the unfinished run only → "Export", disabled.
    await userEvent.click(screen.getByText("wip").closest(".run-row"));
    expect(
      screen.getByRole("button", { name: "Export" }).disabled,
    ).toBe(true);

    // Add the finished run to the selection (Cmd-click) → "Export (1)",
    // enabled — the unfinished one is not counted.
    await userEvent.click(screen.getByText("big").closest(".run-row"), {
      metaKey: true,
    });
    const btn = await screen.findByRole("button", { name: "Export (1)" });
    expect(btn.disabled).toBe(false);
  });
});

describe("App — last import dir persists independently of export (issue #475)", () => {
  // A live in-memory config that emulates the Rust backend: get_config
  // returns the current object, update_config shallow-merges its patch
  // server-side (the lock-serialized atomic merge). This lets the test
  // pin the actual #475 repro — import_dir surviving an interleaved
  // export — rather than just inspecting the client write shape. Every
  // update_config patch is also captured so we can assert each writer
  // sends a narrow per-key patch (structurally unable to clobber a
  // sibling), which is the fix: no client-side whole-object RMW.
  function harness(extra, seed = {}) {
    const cfg = {
      subject: "User",
      tee_provider: "tinfoil",
      tee_model: "gpt-oss-120b",
      inputs: [],
      mode: "tee",
      ...seed,
    };
    const patches = [];
    const override = (cmd, args) => {
      if (cmd === "get_config") return { ...cfg };
      if (cmd === "stat_paths") {
        return (args?.paths || []).map((p) => ({ path: p, size_bytes: 100 }));
      }
      if (cmd === "update_config") {
        patches.push(args?.patch);
        Object.assign(cfg, args?.patch || {});
        return undefined;
      }
      const r = extra?.(cmd, args);
      if (r !== undefined) return r;
    };
    return { cfg, patches, override };
  }
  const importPatch = (patches) =>
    patches.filter((p) => p && typeof p.import_dir === "string").at(-1);

  it("Add-file / Add-folder dialogs open at the restored import_dir", async () => {
    const { override } = harness(undefined, {
      import_dir: "/notes",
      export_dir: "/desktop",
    });
    vi.mocked(openDialog).mockResolvedValue(null);
    await renderApp({ override });

    await userEvent.click(screen.getByTitle("Add file(s)"));
    await userEvent.click(screen.getByTitle("Add folder (recursive)"));

    const dirs = vi
      .mocked(openDialog)
      .mock.calls.map(([opts]) => opts?.defaultPath);
    // Both pickers seed from the persisted import dir, not /desktop.
    expect(dirs).toContain("/notes");
    expect(dirs).not.toContain("/desktop");
  });

  it("a file import writes a narrow patch; an export between imports can't drop it (#475 repro)", async () => {
    const { cfg, patches, override } = harness(undefined, {
      export_dir: "/desktop",
    });
    vi.mocked(openDialog).mockResolvedValueOnce(["/src/a.md"]);
    await renderApp({ override });

    await userEvent.click(screen.getByTitle("Add file(s)"));
    await waitFor(() => {
      expect(screen.getByText(/Input files \(1\)/)).toBeTruthy();
    });

    await waitFor(() => expect(importPatch(patches)).toBeTruthy());
    // The write is a per-key patch — it carries import_dir and nothing
    // about export. The old whole-object RMW is what let an interleaved
    // export drop import_dir; a narrow patch structurally can't.
    expect(importPatch(patches)).toEqual({ import_dir: "/src" });
    expect("export_dir" in importPatch(patches)).toBe(false);
    // Backend merge keeps both keys: next launch restores /src, and the
    // export dir is untouched.
    expect(cfg.import_dir).toBe("/src");
    expect(cfg.export_dir).toBe("/desktop");
  });

  it("multi-pick across dirs keys off the last-selected file's dir", async () => {
    const { patches, override } = harness();
    vi.mocked(openDialog).mockResolvedValueOnce(["/a/x.md", "/b/y.md"]);
    await renderApp({ override });

    await userEvent.click(screen.getByTitle("Add file(s)"));
    await waitFor(() => {
      expect(screen.getByText(/Input files \(2\)/)).toBeTruthy();
    });

    await waitFor(() =>
      expect(importPatch(patches)?.import_dir).toBe("/b"),
    );
  });

  it("a folder pick records the folder itself, not export_dir", async () => {
    const { cfg, patches, override } = harness(
      (cmd, args) => {
        if (cmd === "expand_paths") {
          return (args?.paths || []).flatMap((p) => [`${p}/one.md`]);
        }
      },
      { export_dir: "/desktop" },
    );
    vi.mocked(openDialog).mockResolvedValueOnce("/picked/folder");
    await renderApp({ override });

    await userEvent.click(screen.getByTitle("Add folder (recursive)"));
    await waitFor(() => {
      expect(screen.getByText(/Input files \(1\)/)).toBeTruthy();
    });

    await waitFor(() =>
      expect(importPatch(patches)?.import_dir).toBe("/picked/folder"),
    );
    expect("export_dir" in importPatch(patches)).toBe(false);
    expect(cfg.export_dir).toBe("/desktop");
  });
});
