import { describe, it, expect, afterEach, vi } from "vitest";
import { render, screen, cleanup, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { invoke } from "@tauri-apps/api/core";
import {
  FactsView,
  EntityView,
  PatternsView,
  InsightsView,
  ActionsView,
} from "./RunViews";

afterEach(() => cleanup());

// ── Stub helpers ───────────────────────────────────────────────────────────
// Tests mock invoke() to return canned data per Tauri command. Each
// stage view fires multiple invokes on mount — facts pull patterns +
// entities, patterns pull facts + insights, etc. The default returns
// keep unrelated commands quiet so a test only needs to declare the
// data it actually asserts on.
function stubInvoke(canned = {}) {
  vi.mocked(invoke).mockImplementation(async (cmd, args) => {
    if (canned[cmd] !== undefined) {
      return typeof canned[cmd] === "function"
        ? canned[cmd](args)
        : canned[cmd];
    }
    // Sensible empties — every per-stage command returns either a
    // list, a dict-of-lists, or a payload object. Returning empty
    // shapes lets components render their "no data yet" states without
    // throwing.
    if (cmd === "read_run_facts_for_topic") return [];
    if (cmd === "read_run_facts_all") return {};
    if (cmd === "read_run_patterns_for_topic") return [];
    if (cmd === "read_run_patterns_all") return {};
    if (cmd === "read_run_entities") return null;
    if (cmd === "read_run_entity") return null;
    if (cmd === "read_run_insights") return null;
    if (cmd === "read_run_actions") return null;
    return undefined;
  });
}

// Wait for a stage view's first useEffect tick to land. The
// components fire async invoke()s on mount; tests assert on the
// post-resolution DOM after one microtask flush.
async function flush() {
  await new Promise((r) => setTimeout(r, 0));
}

// ── FactsView ──────────────────────────────────────────────────────────────

describe("FactsView", () => {
  const sampleFact = {
    summary: "Alice signed the contract",
    type: "fact",
    occurred_at: "2026-04-15",
    confidence: 0.9,
    entities: [
      { name: "Alice", entity_type: "person", role: "subject" },
      { name: "Bob",   entity_type: "person", role: "witness" },
    ],
    evidence: [
      {
        text: "Alice signed the contract.",
        file_path: "journal.txt",
        file_offset: 100,
        file_length: 26,
      },
    ],
  };

  const entitiesPayload = {
    entities: [
      { canonical_id: "alice", canonical_name: "Alice",
        entity_type: "person", aliases: [] },
    ],
    relations: [],
  };

  it("puts the formatted date in the title and explicit confidence (no dot) in the subtitle", async () => {
    stubInvoke({ read_run_facts_for_topic: [sampleFact] });
    const { container } = render(
      <FactsView runId="r1" topic="work" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    const heading = screen.getByRole("heading", { level: 2 });
    // Number stays; the grey date leads the title, then the summary.
    expect(heading.querySelector(".md-h-num")?.textContent).toBe("1");
    expect(heading.querySelector(".fact-date")?.textContent).toBe("Apr 15"); // 2026 == current year
    expect(heading.querySelector(".prov-dot")).toBeNull(); // no dot in the title
    expect(heading.textContent).toContain("Alice signed the contract");
    // Subtitle: type + explicit confidence text; no dot anywhere, no date.
    const meta = container.querySelector(".md-meta-line");
    expect(meta.querySelector(".prov-dot")).toBeNull();
    expect(meta.textContent).toContain("fact · confidence 0.90");
    expect(meta.textContent).not.toContain("Apr 15");
  });

  it("undated facts: keep 'undated' in the subtitle, with the raw occurred_at_text in parens when present", async () => {
    stubInvoke({
      read_run_facts_for_topic: [
        { summary: "No date at all", type: "event", confidence: 0.8 },
        { summary: "Partial mention only", type: "event", occurred_at_text: "the 3rd", confidence: 0.8 },
      ],
    });
    const { container } = render(
      <FactsView runId="r1" topic="work" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    const headings = screen.getAllByRole("heading", { level: 2 });
    // No date in either title (both undated).
    expect(headings[0].querySelector(".fact-date")).toBeNull();
    expect(headings[1].querySelector(".fact-date")).toBeNull();
    const metas = container.querySelectorAll(".md-meta-line");
    expect(metas[0].textContent).toContain("undated · event");
    expect(metas[0].textContent).not.toContain("(");
    expect(metas[1].textContent).toContain('undated ("the 3rd") · event'); // raw mention quoted
  });

  it("orders facts newest-first by occurred_at; undated sink to the bottom", async () => {
    stubInvoke({
      read_run_facts_for_topic: [
        { summary: "Oldest", type: "fact", occurred_at: "2020-01-01", confidence: 0.9 },
        { summary: "Newest", type: "fact", occurred_at: "2024-12-31", confidence: 0.9 },
        { summary: "Undated", type: "fact", confidence: 0.9 },
        { summary: "Middle", type: "fact", occurred_at: "2022-06-15", confidence: 0.9 },
      ],
    });
    render(
      <FactsView runId="r1" topic="work" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    const titles = screen
      .getAllByRole("heading", { level: 2 })
      .map((h) => h.textContent);
    expect(titles[0]).toContain("Newest");
    expect(titles[1]).toContain("Middle");
    expect(titles[2]).toContain("Oldest");
    expect(titles[3]).toContain("Undated");
  });

  it("formats the title date with the shared formatter and is timezone-safe (no off-by-one)", async () => {
    stubInvoke({
      read_run_facts_for_topic: [
        // A bare YYYY-MM-DD must render as the date GIVEN, not the day
        // before it — the UTC-midnight parse bug renders "Jun 17" in any
        // zone behind UTC. This row is the regression guard.
        { summary: "Past-year fact", type: "fact", occurred_at: "2011-06-18", confidence: 0.9 },
      ],
    });
    render(
      <FactsView runId="r1" topic="work" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    const date = screen
      .getByRole("heading", { level: 2 })
      .querySelector(".fact-date");
    expect(date.textContent).toBe("Jun 18, 2011"); // year kept (not current), not "Jun 17"
  });

  it("includes the topic in the title", async () => {
    stubInvoke({ read_run_facts_for_topic: [sampleFact] });
    render(
      <FactsView runId="r1" topic="work" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    const h1 = screen.getByRole("heading", { level: 1 });
    expect(h1.textContent).toContain("Work");
  });

  it("shows the OTHER categories a fact belongs to after the fact text", async () => {
    const multiCatFact = {
      summary: "Repulsion at the Vatican paintings.",
      type: "emotion",
      confidence: 0.9,
      occurred_at: "2019-05-21",
      topics: ["spirituality", "travel"],
      entities: [],
      evidence: [],
    };
    stubInvoke({ read_run_facts_for_topic: [multiCatFact] });
    render(
      <FactsView runId="r1" topic="spirituality" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    const h2 = screen.getByRole("heading", { level: 2 });
    // The viewing category is implicit; only the OTHER one is appended.
    expect(h2.textContent).toContain("[Travel]");
    expect(h2.textContent).not.toContain("Spirituality");
  });

  it("omits the category tag when a fact is only in the current category", async () => {
    const singleCatFact = {
      summary: "A single-category fact.",
      type: "fact",
      confidence: 0.9,
      topics: ["work"],
      entities: [],
      evidence: [],
    };
    stubInvoke({ read_run_facts_for_topic: [singleCatFact] });
    const { container } = render(
      <FactsView runId="r1" topic="work" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    expect(container.querySelector(".md-section-heading .prov-tag")).toBeNull();
  });

  it("resolves a fact's entity ref to a wikilink via the entities catalog", async () => {
    stubInvoke({
      read_run_facts_for_topic: [sampleFact],
      read_run_entities: entitiesPayload,
    });
    const onNavigate = vi.fn();
    const { container } = render(
      <FactsView runId="r1" topic="work" refreshTick={0} onNavigate={onNavigate} />,
    );
    await flush();
    // Alice resolved → wikilink. Bob unresolved → plain text.
    const aliceLink = within(container).getByRole("link", { name: "Alice" });
    expect(aliceLink).toBeTruthy();
    expect(within(container).queryByRole("link", { name: "Bob" })).toBeNull();
    // Plain text is still in the DOM for Bob.
    expect(container.textContent).toContain("Bob");
  });

  it("renders Source: <plain filename> + clickable blockquote that navigates", async () => {
    stubInvoke({ read_run_facts_for_topic: [sampleFact] });
    const onNavigate = vi.fn();
    const user = userEvent.setup();
    const { container } = render(
      <FactsView runId="r1" topic="work" refreshTick={0} onNavigate={onNavigate} />,
    );
    await flush();
    const filenameLabel = container.querySelector(".prov-fact-source-filename");
    expect(filenameLabel.textContent).toBe("journal.txt");
    expect(filenameLabel.tagName).toBe("SPAN"); // NOT an anchor
    const quote = container.querySelector(".prov-fact-source-quote");
    expect(quote.textContent).toContain("Alice signed the contract");
    await user.click(quote);
    expect(onNavigate).toHaveBeenCalledTimes(1);
    expect(onNavigate).toHaveBeenCalledWith({
      runId: "r1",
      relPath: "0-inputs/journal.txt",
      anchor: "offset-100",
    });
  });

  it("clicking the filename label does NOT navigate (only the quote does)", async () => {
    stubInvoke({ read_run_facts_for_topic: [sampleFact] });
    const onNavigate = vi.fn();
    const user = userEvent.setup();
    const { container } = render(
      <FactsView runId="r1" topic="work" refreshTick={0} onNavigate={onNavigate} />,
    );
    await flush();
    const filenameLabel = container.querySelector(".prov-fact-source-filename");
    await user.click(filenameLabel);
    expect(onNavigate).not.toHaveBeenCalled();
  });

  it("shows the empty-state placeholder when no facts have landed yet", async () => {
    stubInvoke({ read_run_facts_for_topic: [] });
    render(
      <FactsView runId="r1" topic="work" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    expect(screen.getByText(/extraction in progress or topic empty/i))
      .toBeTruthy();
  });
});

// ── PatternsView ───────────────────────────────────────────────────────────

describe("PatternsView", () => {
  const samplePattern = {
    name: "Pattern A",
    description: "A description.",
    kind: "behavior",
    count: 2,
    source_facts: [[0, 0.85]],
  };
  const sampleFact = {
    summary: "Backing fact summary",
    type: "fact",
    confidence: 0.8,
    evidence: [
      {
        text: "verbatim quote from raw input file",
        file_path: "journal.txt",
        file_offset: 200,
      },
    ],
  };

  it("numbers each pattern heading sequentially", async () => {
    stubInvoke({
      read_run_patterns_for_topic: [samplePattern, samplePattern, samplePattern],
      read_run_facts_for_topic: [sampleFact],
    });
    render(
      <PatternsView runId="r1" topic="work" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    const h2s = screen.getAllByRole("heading", { level: 2 });
    expect(h2s.map((h) => h.querySelector(".md-h-num")?.textContent))
      .toEqual(["1", "2", "3"]);
  });

  it("renders a leaf row that is whole-row clickable into the raw input", async () => {
    stubInvoke({
      read_run_patterns_for_topic: [samplePattern],
      read_run_facts_for_topic: [sampleFact],
    });
    const onNavigate = vi.fn();
    const user = userEvent.setup();
    const { container } = render(
      <PatternsView runId="r1" topic="work" refreshTick={0} onNavigate={onNavigate} />,
    );
    await flush();
    const leaf = container.querySelector(".prov-leaf");
    expect(leaf).not.toBeNull();
    expect(leaf.classList.contains("prov-leaf-clickable")).toBe(true);
    expect(leaf.getAttribute("role")).toBe("link");
    // Quote on the left, filename pushed right (grey, NOT a link).
    expect(leaf.querySelector(".prov-quote").textContent).toContain(
      "verbatim quote from raw input file",
    );
    const filenameSpan = leaf.querySelector(".prov-file");
    expect(filenameSpan.textContent).toBe("journal.txt");
    expect(filenameSpan.tagName).toBe("SPAN"); // not an <a>
    // Whole row click → navigate.
    await user.click(leaf);
    expect(onNavigate).toHaveBeenCalledWith({
      runId: "r1",
      relPath: "0-inputs/journal.txt",
      anchor: "offset-200",
    });
  });

  it("renders a 'Provenance:' inline summary line with per-file shares pills", async () => {
    stubInvoke({
      read_run_patterns_for_topic: [samplePattern],
      read_run_facts_for_topic: [sampleFact],
    });
    const { container } = render(
      <PatternsView runId="r1" topic="work" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    const inline = container.querySelector(".md-provenance-inline");
    expect(inline).not.toBeNull();
    // Chevron + "Provenance:" label.
    expect(inline.querySelector(".md-provenance-chevron").textContent).toBe("▼");
    expect(inline.querySelector(".md-provenance-label").textContent).toBe(
      "Provenance:",
    );
    // One file → one pill at 100%.
    const pill = container.querySelector(".md-provenance-pill");
    expect(pill).not.toBeNull();
    expect(pill.textContent).toContain("100%");
    expect(pill.textContent).toContain("journal.txt");
  });

  it("orders source facts by AGGREGATED composite (edge × extraction), not last-hop edge", async () => {
    // fact[1] has the higher match edge (0.95 > 0.90) but lower
    // extraction (0.80), so its composite (0.76) is BELOW fact[0]'s
    // (0.99 × 0.90 = 0.89). Sorting by last-hop edge would render
    // fact[1] first → a 0.76 dot above a 0.89 dot (the reported
    // inversion). The aggregated sort puts fact[0] on top.
    stubInvoke({
      read_run_patterns_for_topic: [
        { name: "P", kind: "behavior", count: 1, source_facts: [[0, 0.9], [1, 0.95]] },
      ],
      read_run_facts_for_topic: [
        { summary: "High composite fact", type: "fact", confidence: 0.99,
          evidence: [{ text: "q0", file_path: "a.txt", file_offset: 1 }] },
        { summary: "Low composite fact", type: "fact", confidence: 0.8,
          evidence: [{ text: "q1", file_path: "b.txt", file_offset: 2 }] },
      ],
    });
    const { container } = render(
      <PatternsView runId="r1" topic="work" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    const factRows = [...container.querySelectorAll(".prov-row.prov-mid")];
    expect(factRows.map((r) => r.querySelector(".prov-summary").textContent))
      .toEqual(["High composite fact", "Low composite fact"]);
    // Dots now read monotonically descending (no 0.76-above-0.89 flip).
    expect(factRows.map((r) => r.getAttribute("data-score")))
      .toEqual(["0.89", "0.76"]);
  });

  it("prefixes each source-fact row with its grey formatted date (without reordering by date)", async () => {
    stubInvoke({
      read_run_patterns_for_topic: [
        { name: "P", kind: "behavior", count: 1, source_facts: [[0, 0.95]] },
      ],
      read_run_facts_for_topic: [
        { summary: "A dated backing fact", type: "event", occurred_at: "1660-07-04", confidence: 0.9,
          evidence: [{ text: "q", file_path: "a.txt", file_offset: 1 }] },
      ],
    });
    const { container } = render(
      <PatternsView runId="r1" topic="work" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    const row = container.querySelector(".prov-row.prov-mid");
    expect(row.querySelector(".fact-date").textContent).toBe("Jul 4, 1660");
    expect(row.querySelector(".prov-summary").textContent).toContain("A dated backing fact");
  });
});

// ── EntityView ─────────────────────────────────────────────────────────────

describe("EntityView", () => {
  const entity = {
    canonical_id: "her",
    canonical_name: "her", // lowercase from LLM — display title-cases
    entity_type: "person",
    description: "A pronoun-only mention.",
    aliases: ["she", "the woman"],
    mention_count: 5,
    evidence_fact_refs: [],
  };

  it("title-cases canonical_name in the heading even when LLM returned lowercase", async () => {
    stubInvoke({ read_run_entity: entity });
    render(
      <EntityView runId="r1" entityId="her" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    const h1 = screen.getByRole("heading", { level: 1 });
    expect(h1.textContent).toBe("Her");
  });

  it("renders relation rows with title-cased endpoint names", async () => {
    const allEnts = {
      entities: [
        { canonical_id: "her",      canonical_name: "her" },
        { canonical_id: "daughter", canonical_name: "daughter" },
      ],
      relations: [
        { from: "her", to: "daughter", relation: "mother", confidence: 0.9 },
      ],
    };
    stubInvoke({ read_run_entity: entity, read_run_entities: allEnts });
    render(
      <EntityView runId="r1" entityId="her" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    // Scope to the relation row — "Her" also appears in the page
    // <h1>, and Aliases renders its own <ul> first.
    const relationLi = screen.getByText("mother").closest("li");
    expect(relationLi.textContent).toContain("Her");
    expect(relationLi.textContent).toContain("Daughter");
    expect(relationLi.textContent).toMatch(/confidence\s+0\.90/);
  });

  it("cites a cross-category fact once, tagged with all its categories (alphabetical)", async () => {
    // The pipeline collapses the fan-out to the alphabetically-first
    // category, so the entity carries a single ref; the row shows the
    // fact's full category list, alphabetical, in one merged tag.
    const citedEntity = {
      ...entity,
      evidence_fact_refs: [["spirituality", 0]],
    };
    const factsAll = {
      spirituality: [
        {
          summary: "Repulsion at the Vatican paintings.",
          type: "emotion",
          confidence: 0.9,
          topics: ["travel", "spirituality"],
        },
      ],
    };
    stubInvoke({ read_run_entity: citedEntity, read_run_facts_all: factsAll });
    const { container } = render(
      <EntityView runId="r1" entityId="her" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    expect(container.textContent).toContain("[Spirituality, Travel · Emotion]");
  });

  it("exposes per-relation evidence facts behind a 'see N sources' / 'hide N sources' toggle that resolves [topic, fact_idx] refs to fact summaries", async () => {
    // A relation backed by two facts in two topics — the materialized
    // shape that lives in phase_3_marker.json after Stage 2 Phase 3.
    const allEnts = {
      entities: [
        { canonical_id: "her",      canonical_name: "her" },
        { canonical_id: "daughter", canonical_name: "daughter" },
      ],
      relations: [
        {
          from: "her",
          to: "daughter",
          relation: "mother",
          confidence: 0.92,
          evidence_fact_refs: [
            ["family", 0],
            ["health", 1],
          ],
        },
      ],
    };
    const factsAll = {
      family: [
        { summary: "Her daughter visited on Sunday.", type: "event", confidence: 0.95 },
      ],
      health: [
        { summary: "A throwaway fact in slot 0.", type: "fact", confidence: 0.5 },
        { summary: "Her daughter caught a cold.", type: "event", confidence: 0.8 },
      ],
    };
    stubInvoke({
      read_run_entity: entity,
      read_run_entities: allEnts,
      read_run_facts_all: factsAll,
    });
    const user = userEvent.setup();
    const { container } = render(
      <EntityView runId="r1" entityId="her" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    // Toggle exists with "see N sources" label and is collapsed by default.
    const toggle = screen.getByRole("button", { name: /see 2 sources/ });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    // No evidence rows under the relation pre-click.
    expect(container.querySelector(".rel-evidence-list")).toBeNull();
    // Expand — label flips to "hide N sources".
    await user.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(toggle.textContent.replace(/\s+/g, " ").trim()).toBe("hide 2 sources");
    const list = container.querySelector(".rel-evidence-list");
    expect(list).not.toBeNull();
    // Both refs resolve to their fact summaries (skipping the unrelated
    // health[0] entry — health[1] is the one cited).
    expect(within(list).getByText("Her daughter visited on Sunday.")).toBeTruthy();
    expect(within(list).getByText("Her daughter caught a cold.")).toBeTruthy();
    expect(within(list).queryByText("A throwaway fact in slot 0.")).toBeNull();
    // Re-click collapses; label flips back to "see N sources".
    await user.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(toggle.textContent.replace(/\s+/g, " ").trim()).toBe("see 2 sources");
    expect(container.querySelector(".rel-evidence-list")).toBeNull();
  });

  it("omits the evidence toggle when a relation has no evidence_fact_refs", async () => {
    const allEnts = {
      entities: [
        { canonical_id: "her",      canonical_name: "her" },
        { canonical_id: "daughter", canonical_name: "daughter" },
      ],
      relations: [
        { from: "her", to: "daughter", relation: "mother", confidence: 0.9 },
      ],
    };
    stubInvoke({ read_run_entity: entity, read_run_entities: allEnts });
    render(
      <EntityView runId="r1" entityId="her" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    expect(screen.queryByRole("button", { name: /see \d+ sources?/ })).toBeNull();
  });

  it("caps Cited-in-facts at 30 and reveals all rows when '+N more' is clicked", async () => {
    // 35 refs across two topics, all resolvable.
    const refs = Array.from({ length: 35 }, (_, i) => ["work", i]);
    const ent = { ...entity, evidence_fact_refs: refs };
    const factsAll = {
      work: Array.from({ length: 35 }, (_, i) => ({
        summary: `Fact ${i + 1}`,
        type: "fact",
        confidence: 0.9,
      })),
    };
    stubInvoke({ read_run_entity: ent, read_run_facts_all: factsAll });
    const user = userEvent.setup();
    const { container } = render(
      <EntityView runId="r1" entityId="her" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    // Default: 30 visible + 1 "+5 more" expander row.
    let rows = container.querySelectorAll(".prov-list .prov-item-mid");
    expect(rows).toHaveLength(30);
    const moreBtn = screen.getByRole("button", { name: /\+5 more/ });
    expect(moreBtn).toBeTruthy();
    await user.click(moreBtn);
    rows = container.querySelectorAll(".prov-list .prov-item-mid");
    expect(rows).toHaveLength(35);
    // Expander gone after click.
    expect(screen.queryByRole("button", { name: /\+5 more/ })).toBeNull();
  });

  it("orders Cited-in-facts newest-first and prefixes each with its grey date", async () => {
    const ent = {
      ...entity,
      evidence_fact_refs: [["work", 0], ["work", 1], ["work", 2]],
    };
    const factsAll = {
      work: [
        { summary: "Fact 2020", type: "fact", occurred_at: "2020-03-03", confidence: 0.9 },
        { summary: "Fact 2024", type: "fact", occurred_at: "2024-08-08", confidence: 0.9 },
        { summary: "Fact undated", type: "fact", confidence: 0.9 },
      ],
    };
    stubInvoke({ read_run_entity: ent, read_run_facts_all: factsAll });
    const { container } = render(
      <EntityView runId="r1" entityId="her" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    const rows = [...container.querySelectorAll(".prov-list .prov-item-mid .prov-summary")];
    expect(rows.map((r) => r.textContent)).toEqual([
      expect.stringContaining("Fact 2024"),
      expect.stringContaining("Fact 2020"),
      expect.stringContaining("Fact undated"),
    ]);
    // Dated rows carry a grey .fact-date; the undated one doesn't.
    const dates = rows.map((r) => r.querySelector(".fact-date")?.textContent ?? null);
    expect(dates).toEqual(["Aug 8, 2024", "Mar 3, 2020", null]);
  });

  it("renders consolidating banner + Live mentions list when read_run_entity returns the synthesized in-flight shape", async () => {
    // Mid-Stage-1 synthesized shape from #146: no canonical record on
    // disk yet, so the panel must show the live mention stream + a
    // consolidating note instead of "0 mentions" with an empty body.
    const consolidating = {
      _state: "consolidating",
      canonical_name: "Author",
      entity_type: "person",
      mention_count: 4,
      topics: ["work", "health"],
      mentions: [
        // Each mention carries its own extraction confidence; the
        // live-mention dot maps it through the same gradient
        // ConfidenceDot used elsewhere. One mention omits confidence
        // to exercise the unknown-dot fallback.
        { name: "Author", entity_type: "person", role: "subject",
          topic: "work", fact_summary: "Author drafted chapter 3.",
          confidence: 0.97 },
        { name: "Author", entity_type: "person", role: "subject",
          topic: "health", fact_summary: "Author skipped breakfast.",
          confidence: 0.65 },
        { name: "Author", entity_type: "person", role: "subject",
          topic: "work", fact_summary: "Author met with editor.",
          confidence: 0.2 },
        { name: "Author", entity_type: "person", role: "subject",
          topic: "work", fact_summary: "Author paused for coffee." },
      ],
      evidence_fact_refs: [],
      aliases: [],
    };
    stubInvoke({ read_run_entity: consolidating });
    render(
      <EntityView runId="r1" entityId="author" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    // Heading title-cases the synthesized canonical_name.
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("Author");
    // Meta line includes the consolidating badge AND the count that
    // matches the tree's "(3)" badge for this on-disk file.
    const meta = screen.getByText(/consolidating/i);
    expect(meta.textContent).toContain("person");
    expect(meta.textContent).toContain("consolidating");
    expect(meta.textContent).toContain("4 mentions");
    // Explainer is rendered.
    expect(screen.getByText(/still being consolidated/i)).toBeTruthy();
    // Live mentions section + 4 rows.
    const heading = screen.getByRole("heading", { level: 2,
      name: /Live mentions \(4\)/ });
    expect(heading).toBeTruthy();
    expect(screen.getByText("Author drafted chapter 3.")).toBeTruthy();
    expect(screen.getByText("Author skipped breakfast.")).toBeTruthy();
    expect(screen.getByText("Author met with editor.")).toBeTruthy();
    expect(screen.getByText("Author paused for coffee.")).toBeTruthy();
    // Each mention row carries a data-score derived from its
    // confidence (rounded pct ÷ 100); the dot is the gradient
    // ConfidenceDot, and missing-confidence rows render the
    // unknown-dot variant.
    const rows = document.querySelectorAll(
      ".prov-list .prov-item-mid .prov-row",
    );
    expect(rows).toHaveLength(4);
    expect(rows[0].getAttribute("data-score")).toBe("0.97");
    expect(rows[1].getAttribute("data-score")).toBe("0.65");
    expect(rows[2].getAttribute("data-score")).toBe("0.20");
    expect(rows[3].getAttribute("data-score")).toBeNull();
    const dots = document.querySelectorAll(
      ".prov-list .prov-item-mid .prov-dot",
    );
    expect(dots).toHaveLength(4);
    expect(dots[3].className).toContain("prov-dot-unknown");
  });

  it("resolves a consolidating entity's live-mention click to the fact's anchor (not the topic head)", async () => {
    // #868: a consolidating entity renders the in-flight mention stream,
    // whose lines carry the fact's summary + evidence but NOT its topic
    // index. The row must resolve back to the fact in read_run_facts_all
    // (keyed on file_path + file_offset + summary) and navigate with the
    // computed `type-N` anchor — otherwise the click lands on the topic
    // head with an empty anchor and the page never scrolls to the fact.
    const consolidating = {
      _state: "consolidating",
      canonical_name: "Author",
      entity_type: "person",
      mention_count: 2,
      topics: ["work"],
      mentions: [
        { name: "Author", entity_type: "person", role: "subject",
          topic: "work", fact_summary: "Author met with editor.",
          confidence: 0.9,
          evidence: [{ file_path: "journal.txt", file_offset: 200 }] },
        // No matching fact in read_run_facts_all → falls back to "".
        { name: "Author", entity_type: "person", role: "subject",
          topic: "work", fact_summary: "Author drafted chapter 3.",
          confidence: 0.8,
          evidence: [{ file_path: "journal.txt", file_offset: 999 }] },
      ],
      evidence_fact_refs: [],
      aliases: [],
    };
    // The fact at idx 1 is the "met with editor" mention's fact. With one
    // prior "event" and this "fact"-typed item, computeFactAnchors gives
    // it the anchor "fact-1".
    const factsAll = {
      work: [
        { summary: "Earlier event", type: "event", confidence: 0.9,
          evidence: [{ file_path: "journal.txt", file_offset: 10 }] },
        { summary: "Author met with editor.", type: "fact", confidence: 0.9,
          evidence: [{ file_path: "journal.txt", file_offset: 200 }] },
      ],
    };
    stubInvoke({ read_run_entity: consolidating, read_run_facts_all: factsAll });
    const onNavigate = vi.fn();
    const user = userEvent.setup();
    render(
      <EntityView runId="r1" entityId="author" refreshTick={0} onNavigate={onNavigate} />,
    );
    await flush();
    // Matched mention → navigates to the fact's anchor.
    await user.click(screen.getByText("Author met with editor."));
    expect(onNavigate).toHaveBeenCalledWith({
      runId: "r1",
      relPath: "1-facts/work.md",
      anchor: "fact-1",
    });
    // Unmatched mention → graceful fall back to the topic head.
    onNavigate.mockClear();
    await user.click(screen.getByText("Author drafted chapter 3."));
    expect(onNavigate).toHaveBeenCalledWith({
      runId: "r1",
      relPath: "1-facts/work.md",
      anchor: "",
    });
  });

  it("does NOT render consolidating banner for a normal canonical record", async () => {
    // Regression guard: canonical records (no _state) keep the original
    // panel UI, including the role-in-meta and no Live-mentions section.
    stubInvoke({ read_run_entity: entity });
    render(
      <EntityView runId="r1" entityId="her" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    expect(screen.queryByText(/still being consolidated/i)).toBeNull();
    expect(screen.queryByRole("heading", { level: 2, name: /Live mentions/ }))
      .toBeNull();
  });

  it("resets the show-all expander when navigating to a different entity", async () => {
    const refs = Array.from({ length: 35 }, (_, i) => ["work", i]);
    const factsAll = {
      work: Array.from({ length: 35 }, (_, i) => ({
        summary: `Fact ${i + 1}`,
        type: "fact",
        confidence: 0.9,
      })),
    };
    let currentId = "her";
    vi.mocked(invoke).mockImplementation(async (cmd) => {
      if (cmd === "read_run_entity") {
        return { ...entity, canonical_id: currentId,
                 canonical_name: currentId, evidence_fact_refs: refs };
      }
      if (cmd === "read_run_facts_all") return factsAll;
      if (cmd === "read_run_entities") return null;
      return undefined;
    });
    const user = userEvent.setup();
    const { container, rerender } = render(
      <EntityView runId="r1" entityId="her" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    await user.click(screen.getByRole("button", { name: /\+5 more/ }));
    expect(container.querySelectorAll(".prov-list .prov-item-mid"))
      .toHaveLength(35);
    // Navigate to a different entity → expander state must reset.
    currentId = "daughter";
    rerender(
      <EntityView runId="r1" entityId="daughter" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    expect(container.querySelectorAll(".prov-list .prov-item-mid"))
      .toHaveLength(30);
    expect(screen.getByRole("button", { name: /\+5 more/ })).toBeTruthy();
  });
});

// ── InsightsView ───────────────────────────────────────────────────────────

describe("InsightsView", () => {
  const insightsPayload = {
    cross_domain: [
      { name: "Cross 1", description: "d", mechanism: "m", implication: "i",
        domains: ["work"], proposed_actions: [], source_patterns: [] },
    ],
    critical: [
      { name: "Crit A", description: "d", mechanism: "m", implication: "i",
        domains: ["health"], proposed_actions: [], source_patterns: [] },
      { name: "Crit B", description: "d", mechanism: "m", implication: "i",
        domains: ["health"], proposed_actions: [], source_patterns: [] },
    ],
  };

  it("numbers insights continuously across cross-domain → critical", async () => {
    stubInvoke({ read_run_insights: insightsPayload });
    render(
      <InsightsView runId="r1" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    expect(screen.getByRole("heading", { level: 2, name: "Cross-domain" }))
      .toBeTruthy();
    expect(screen.getByRole("heading", { level: 2, name: "Critical" }))
      .toBeTruthy();
    // The visible count is continuous: critical picks up where cross-domain
    // left off (1 cross-domain → [1]; 2 critical → [2], [3]), matching the
    // actions-prompt `[N]` enumeration and the action-body insight refs.
    const h3s = screen.getAllByRole("heading", { level: 3 });
    expect(h3s.map((h) => h.querySelector(".md-h-num")?.textContent))
      .toEqual(["1", "2", "3"]);
  });

  it("shows 'no insights detected' when both buckets are empty", async () => {
    stubInvoke({
      read_run_insights: { cross_domain: [], critical: [] },
    });
    render(
      <InsightsView runId="r1" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    expect(screen.getByText(/no insights detected/i)).toBeTruthy();
  });
});

// ── ActionsView ────────────────────────────────────────────────────────────

describe("ActionsView", () => {
  const actionsPayload = {
    actions: [
      {
        recommendation: "Do the thing",
        objective: "obj", why: "why",
        immediate_action: "now",
        habit: "daily", success_metric: "sm",
        horizon: "short", score: 0.7, confidence: 0.8,
        kind: "discrete",
        regret_reduction: 0.5, leverage: 0.6, consequence: 0.4,
        generativity: 0.3, decisiveness: 0.7, time_to_feedback: 0.5,
        constraint_fit: 0.6,
        source_insights: [],
      },
      {
        recommendation: "Do the second thing",
        objective: "", why: "", immediate_action: "", habit: "",
        success_metric: "", source_insights: [],
      },
    ],
  };

  it("numbers action headings sequentially", async () => {
    stubInvoke({ read_run_actions: actionsPayload });
    render(
      <ActionsView runId="r1" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    const h2s = screen.getAllByRole("heading", { level: 2 });
    expect(h2s.map((h) => h.querySelector(".md-h-num")?.textContent))
      .toEqual(["1", "2"]);
  });

  it("renders 'no actions generated' when the actions list is empty", async () => {
    stubInvoke({ read_run_actions: { actions: [] } });
    render(
      <ActionsView runId="r1" refreshTick={0} onNavigate={vi.fn()} />,
    );
    await flush();
    expect(screen.getByText(/no actions generated/i)).toBeTruthy();
  });

  it("turns positional [N] insight refs in action prose into clickable insight links", async () => {
    const payload = {
      actions: [{
        recommendation: "Ship it", objective: "obj",
        why: "Insight [1] shows the loop; insight [2] confirms it; arr[3] is noise.",
        immediate_action: "", habit: "", success_metric: "",
        source_insights: [],
      }],
    };
    // One cross-domain + one critical insight → continuous index:
    // [1] = cross-domain[0], [2] = critical[0].
    const insights = {
      cross_domain: [{ name: "Toxic loop" }],
      critical: [{ name: "Hidden constraint" }],
    };
    stubInvoke({ read_run_actions: payload, read_run_insights: insights });
    const onNavigate = vi.fn();
    const user = userEvent.setup();
    const { container } = render(
      <ActionsView runId="r1" refreshTick={0} onNavigate={onNavigate} />,
    );
    await flush();

    const refs = container.querySelectorAll(".md-insight-ref");
    // Only [1] and [2] linkify; `arr[3]` is array syntax, left as text.
    expect(Array.from(refs).map((a) => a.textContent)).toEqual(["[1]", "[2]"]);

    await user.click(refs[1]); // [2] → the critical insight
    expect(onNavigate).toHaveBeenCalledTimes(1);
    expect(onNavigate).toHaveBeenCalledWith({
      runId: "r1",
      relPath: "4-insights.md",
      anchor: "critical-1",
    });
  });
});
