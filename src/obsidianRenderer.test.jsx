/**
 * Golden-equivalence parity tests for obsidianRenderer.js.
 *
 * The fixtures under `engine/tests/fixtures/obsidian_renderer/<size>/`
 * pin the byte-for-byte output of the legacy Python `vault_exporter.py`
 * exporter — they were generated once at the cutover with
 * `dump_obsidian_fixtures.py` running both the Python exporter and the
 * JS renderer side-by-side, then committed when parity held. The
 * exporter has since been deleted; these committed goldens are now the
 * canonical regression tier — any drift in `exportRun`'s output trips
 * a diff.
 *
 * Three sizes mirror the issue's acceptance criteria: small (single
 * fact, no insights/actions), medium (multi-topic + entities +
 * cross-domain insight), large (Pepys-style with hundreds of facts +
 * hallucinated/broken refs + escape characters).
 */
import { describe, test, expect } from "vitest";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join, relative } from "node:path";

import { exportRun } from "./obsidianRenderer.js";

const REPO_ROOT = join(__dirname, "..");
const FIXTURE_DIR = join(
  REPO_ROOT, "engine", "tests", "fixtures", "obsidian_renderer",
);

const FIXTURES = ["small", "medium", "large"];

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...walk(p));
    } else if (entry.isFile()) {
      out.push(p);
    }
  }
  return out;
}

function loadFixtureInputs(fixtureName) {
  const root = join(FIXTURE_DIR, fixtureName, "inputs");
  const facts = JSON.parse(readFileSync(join(root, "facts.json"), "utf8"));
  const entities = JSON.parse(readFileSync(join(root, "entities.json"), "utf8"));
  const patterns = JSON.parse(readFileSync(join(root, "patterns.json"), "utf8"));
  const insights = JSON.parse(readFileSync(join(root, "insights.json"), "utf8"));
  const actions = JSON.parse(readFileSync(join(root, "actions.json"), "utf8"));
  const stats = JSON.parse(readFileSync(join(root, "stats.json"), "utf8"));

  // Preprocessed inputs: { fileId: text } where fileId has no .md suffix.
  const preDir = join(root, "preprocessed");
  const inputs = {};
  if (existsSync(preDir)) {
    for (const p of walk(preDir)) {
      const rel = relative(preDir, p);
      const fileId = rel.replace(/\.md$/, "");
      inputs[fileId] = readFileSync(p, "utf8");
    }
  }
  // Mirror the prod regenVault wiring: runName == runId so wikilinks
  // come out prefixed `<runId>/…`. The fixture's runId is fixed per
  // size so the goldens that bake it in stay reproducible.
  const runName = `2026-05-02T12-10-32Z-${fixtureName}`;
  return {
    factsByTopic: facts,
    entitiesPayload: entities,
    patternsByTopic: patterns,
    insightsPayload: insights,
    actionsList: actions.actions || [],
    stats,
    inputs,
    runName,
  };
}

function loadGolden(fixtureName) {
  const root = join(FIXTURE_DIR, fixtureName, "golden");
  const out = {};
  if (!existsSync(root)) return out;
  for (const p of walk(root)) {
    const rel = relative(root, p).replace(/\\/g, "/");
    out[rel] = readFileSync(p, "utf8");
  }
  return out;
}

function diffFiles(a, b) {
  if (a === b) return null;
  const aLines = a.split("\n");
  const bLines = b.split("\n");
  for (let i = 0; i < Math.max(aLines.length, bLines.length); i++) {
    if (aLines[i] !== bLines[i]) {
      return `line ${i + 1}\n  expected: ${JSON.stringify(bLines[i])}\n  got:      ${JSON.stringify(aLines[i])}`;
    }
  }
  return `length differs (got ${aLines.length}, expected ${bLines.length})`;
}

describe("obsidianRenderer.exportRun parity with vault_exporter.py goldens", () => {
  for (const fx of FIXTURES) {
    test(`fixture: ${fx}`, () => {
      const payload = loadFixtureInputs(fx);
      const got = exportRun(payload);
      const golden = loadGolden(fx);

      // Same set of file paths.
      const gotKeys = Object.keys(got).sort();
      const goldenKeys = Object.keys(golden).sort();
      expect(gotKeys, `[${fx}] file set differs`).toEqual(goldenKeys);

      // Each file byte-identical.
      const mismatches = [];
      for (const key of goldenKeys) {
        if (got[key] !== golden[key]) {
          mismatches.push(`${key}: ${diffFiles(got[key], golden[key])}`);
        }
      }
      expect(mismatches, `[${fx}] file content mismatches`).toEqual([]);
    });
  }
});

// The cut is a contiguous prefix of the slider; every reference into an
// excluded level must collapse to a provenance count, never a dangling
// wikilink. These pin that boundary behavior, not the renderer's
// internal shape.
describe("obsidianRenderer privacy cut", () => {
  const EXCLUDED_DIRS = {
    insights: ["3-patterns", "2-entities", "1-facts", "0-inputs"],
    patterns: ["2-entities", "1-facts", "0-inputs"],
    entities: ["1-facts", "0-inputs"],
    facts: ["0-inputs"],
    raw: [],
  };

  test("default render equals explicit privacyLevel raw", () => {
    const payload = loadFixtureInputs("medium");
    expect(exportRun({ ...payload, privacyLevel: "raw" })).toEqual(
      exportRun(payload),
    );
  });

  test("floor cut emits only Actions + Insights + index", () => {
    const payload = loadFixtureInputs("medium");
    const got = exportRun({ ...payload, privacyLevel: "insights" });
    expect(Object.keys(got).sort()).toEqual(
      ["4-insights.md", "5-actions.md", "index.md"],
    );
  });

  for (const [level, excludedDirs] of Object.entries(EXCLUDED_DIRS)) {
    test(`cut "${level}" leaves no wikilink into an excluded level`, () => {
      const payload = loadFixtureInputs("medium");
      const got = exportRun({ ...payload, privacyLevel: level });
      const prefix = `[[${payload.runName}/`;
      for (const [file, content] of Object.entries(got)) {
        for (const dir of excludedDirs) {
          expect(
            content.includes(`${prefix}${dir}`),
            `${file} links into excluded ${dir} at cut ${level}`,
          ).toBe(false);
        }
      }
    });
  }

  test("raw excluded scrubs verbatim evidence quotes from facts", () => {
    const payload = loadFixtureInputs("medium");
    // Sanity: with raw included the facts DO carry blockquoted excerpts.
    const full = exportRun({ ...payload, privacyLevel: "raw" });
    const factFilesFull = Object.entries(full).filter(([f]) => f.startsWith("1-facts/"));
    expect(factFilesFull.some(([, c]) => /\n\s*> /.test(c))).toBe(true);
    // At cut "facts" (raw below the cut) no fact file may ship a
    // verbatim quote — the Source block is a count instead.
    const got = exportRun({ ...payload, privacyLevel: "facts" });
    for (const [file, content] of Object.entries(got)) {
      if (!file.startsWith("1-facts/")) continue;
      expect(/\n\s*> /.test(content), `${file} still ships a quote`).toBe(false);
      expect(content).toMatch(/source excerpts? below the privacy cut \(not exported\)/);
    }
  });

  test("excluded provenance collapses to a count line", () => {
    const payload = loadFixtureInputs("medium");
    const got = exportRun({ ...payload, privacyLevel: "insights" });
    // Medium has a cross-domain insight sourced from patterns; with
    // patterns below the cut its provenance is a count, not a walk.
    expect(got["4-insights.md"]).toMatch(/below the privacy cut \(not exported\)/);
  });

  // Top-level file "bucket" of a manifest key: the dir for dir-scoped
  // pages (`1-facts/foo.md` → `1-facts`), the filename for root pages.
  const bucket = (k) => (k.includes("/") ? k.split("/")[0] : k);
  const buckets = (manifest) =>
    [...new Set(Object.keys(manifest).map(bucket))].sort();

  // The cut is a contiguous prefix: each level adds exactly its own
  // bucket on top of everything more-private. Pins both over-emit
  // (a level that shouldn't be there) and under-emit.
  const EXPECTED_BUCKETS = {
    insights: ["4-insights.md", "5-actions.md", "index.md"],
    patterns: ["3-patterns", "4-insights.md", "5-actions.md", "index.md"],
    entities: ["2-entities", "3-patterns", "4-insights.md", "5-actions.md", "index.md"],
    facts: ["1-facts", "2-entities", "3-patterns", "4-insights.md", "5-actions.md", "index.md"],
    raw: ["0-inputs", "1-facts", "2-entities", "3-patterns", "4-insights.md", "5-actions.md", "index.md"],
  };

  for (const [level, expected] of Object.entries(EXPECTED_BUCKETS)) {
    test(`cut "${level}" emits exactly the expected file set`, () => {
      const got = exportRun({ ...loadFixtureInputs("medium"), privacyLevel: level });
      expect(buckets(got)).toEqual([...expected].sort());
    });
  }

  test("file set grows monotonically as the cut loosens (contiguous prefix)", () => {
    const order = ["insights", "patterns", "entities", "facts", "raw"];
    const payload = loadFixtureInputs("medium");
    let prev = null;
    for (const level of order) {
      const cur = new Set(buckets(exportRun({ ...payload, privacyLevel: level })));
      if (prev) {
        for (const b of prev) {
          expect(cur.has(b), `cut "${level}" dropped bucket ${b}`).toBe(true);
        }
      }
      prev = cur;
    }
  });

  for (const level of ["insights", "patterns", "entities", "facts"]) {
    test(`cut "${level}" surfaces a boundary count somewhere; "raw" has none`, () => {
      const payload = loadFixtureInputs("medium");
      const re = /below the privacy cut \(not exported\)/;
      const cut = exportRun({ ...payload, privacyLevel: level });
      expect(Object.values(cut).some((c) => re.test(c)), `no count line at cut ${level}`).toBe(true);
      const full = exportRun({ ...payload, privacyLevel: "raw" });
      expect(Object.values(full).some((c) => re.test(c))).toBe(false);
    });
  }

  test("boundary count VALUE matches the source cardinality", () => {
    const payload = loadFixtureInputs("medium");
    // At cut "insights", a pattern-sourced insight reports exactly its
    // own source_patterns count — the value, not just the phrase.
    const groups = [
      ...(payload.insightsPayload?.cross_domain || []),
      ...(payload.insightsPayload?.critical || []),
    ];
    const ins = groups.find((g) => (g.source_patterns || []).length > 0);
    expect(ins, "fixture has no pattern-sourced insight").toBeTruthy();
    const got = exportRun({ ...payload, privacyLevel: "insights" });
    const n = ins.source_patterns.length;
    expect(got["4-insights.md"]).toContain(
      `${n} pattern${n === 1 ? "" : "s"} below the privacy cut (not exported)`,
    );
  });

  test('entity "Cited in facts" collapses to a count when facts are below the cut', () => {
    const payload = loadFixtureInputs("medium");
    const got = exportRun({ ...payload, privacyLevel: "entities" });
    const ents = Object.entries(got).filter(([f]) => f.startsWith("2-entities/"));
    expect(ents.length).toBeGreaterThan(0);
    const cited = ents.filter(([, c]) => c.includes("## Cited in facts"));
    expect(cited.length, "fixture has no entity citing facts").toBeGreaterThan(0);
    for (const [file, c] of cited) {
      expect(/\[\[.*1-facts/.test(c), `${file} still links into 1-facts`).toBe(false);
      expect(c).toMatch(/facts? below the privacy cut \(not exported\)/);
    }
  });
});

describe("exportRun: insight numbering + clickable action refs", () => {
  function _ins(name) {
    return { name, description: "d", mechanism: "m", implication: "i",
             domains: ["work"], proposed_actions: [], source_patterns: [] };
  }
  const payload = {
    factsByTopic: {}, entitiesPayload: { entities: [], relations: [] },
    patternsByTopic: {},
    insightsPayload: {
      cross_domain: [_ins("Alpha loop"), _ins("Beta loop")],
      critical: [_ins("Gamma risk")],
    },
    actionsList: [{
      recommendation: "Ship it", objective: "obj",
      why: "Insight [1] sets it up; insight [3] is the risk; arr[2] is noise.",
      immediate_action: "", habit: "", success_metric: "",
      horizon: "short", review_date: "2026-06-01", kind: "build",
      regret_reduction: 0.5, leverage: 0.5, consequence: 0.5,
      generativity: 0.5, decisiveness: 0.5, time_to_feedback: 0.5,
      constraint_fit: 0.5, confidence: 0.8,
      source_insights: [["cross_domain", 0, 0.9]],
    }],
    stats: {}, inputs: {}, runName: "2026-05-02T12-10-32Z-x",
  };

  test("insight headings are numbered continuously across scopes", () => {
    const out = exportRun(payload)["4-insights.md"];
    // cross-domain 1,2 then critical continues at 3 (not restarting at 1).
    expect(out).toContain("### [1] Alpha loop");
    expect(out).toContain("### [2] Beta loop");
    expect(out).toContain("### [3] Gamma risk");
    // Block-ref anchors stay per-scope as stable link targets.
    expect(out).toContain("^cross-domain-1");
    expect(out).toContain("^critical-1");
  });

  test("action-prose [N] refs become wikilinks to the insight", () => {
    const out = exportRun(payload)["5-actions.md"];
    // [1] → cross-domain-1; [3] → critical-1 (continuous index → per-scope
    // anchor). Display is the continuous number; `arr[2]` is left alone.
    expect(out).toMatch(/Insight \[\[[^\]]*4-insights#\^cross-domain-1\|1\]\]/);
    expect(out).toMatch(/insight \[\[[^\]]*4-insights#\^critical-1\|3\]\]/);
    expect(out).toContain("arr[2]");
  });
});
