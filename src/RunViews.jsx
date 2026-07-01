/**
 * Per-stage view components for in-app run browsing.
 *
 * Each component fetches structured data via Tauri commands and renders
 * JSX directly — no markdown intermediate. The visible shape mirrors
 * the on-disk Obsidian markdown the export path produces (anchors and
 * wikilinks both come out of `obsidianRenderer.js`):
 * same primitives (ConfidenceDot, ProvRow with root/mid/leaf levels,
 * per-file shares pills, collapsible Provenance, FactSourceBlock with
 * vertical-bar layout), same composite confidence math (extraction ×
 * match), same self-link handling.
 *
 * Navigation contract: every cross-stage link calls
 *   onNavigate({ runId, relPath, anchor })
 * — the same shape App.jsx's handleMarkdownNavigate already accepts.
 *
 * In-flight refresh: each component re-fetches when `refreshTick`
 * changes (App.jsx bumps it on pipeline-progress events).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { prettyDate } from "./dateFormat";

// ── Tiny helpers ───────────────────────────────────────────────────────────

const STAGE_DIRS = {
  1: "1-facts",
  2: "2-entities",
  3: "3-patterns",
  4: "4-insights",
  5: "5-actions",
};

// Title-case a slug. Underscores become spaces (separator), hyphens
// stay (compound words). Mirror of MarkdownView.titleCaseSlug.
//   `cross-domain`  → "Cross-Domain"
//   `mental_health` → "Mental Health"
//   `open_loop`     → "Open Loop"
function titleCaseSlug(slug) {
  if (!slug) return "";
  return slug
    .split("_")
    .map((seg) =>
      seg
        .split("-")
        .map((w) => (w ? w.charAt(0).toUpperCase() + w.slice(1) : ""))
        .join("-"),
    )
    .join(" ");
}

function singularizeTopic(topic) {
  const t = titleCaseSlug(topic);
  if (t.endsWith("s") && !t.endsWith("ss")) return t.slice(0, -1);
  return t;
}

// A fact's (or live-mention's) ISO date key for chronological sorting,
// or null when undated. `occurred_at` is `YYYY-MM-DD`, so a plain
// lexicographic string compare IS a chronological compare.
function factDateKey(x) {
  const v = x && x.occurred_at;
  return typeof v === "string" && v.trim() ? v.trim() : null;
}

// Indices of `items` in reverse-chronological display order (newest
// `occurred_at` first). Undated items sink to the bottom; equal keys
// (and the undated tail) keep their original order (stable). Returns
// indices into the ORIGINAL array so callers that key anchors /
// cited-by maps by index keep working — only the render *sequence*
// changes, never the index↔anchor correspondence.
function reverseChronIndices(items, keyOf) {
  return (items || [])
    .map((it, i) => ({ i, key: keyOf(it) }))
    .sort((a, b) => {
      if (a.key === b.key) return a.i - b.i;
      if (a.key === null) return 1;
      if (b.key === null) return -1;
      return a.key < b.key ? 1 : -1;
    })
    .map((e) => e.i);
}

// Pull (topic, fact_idx) out of a fact ref, tolerating both the array
// shape `[topic, idx]` and any object-with-numeric-keys variant.
function refTopicIdx(ref) {
  const t = Array.isArray(ref) ? ref[0] : ref?.[0];
  const fIdx = Array.isArray(ref) ? ref[1] : ref?.[1];
  return [t, fIdx];
}

// Sort a pattern's `[factIdx, edgeConf]` source-fact refs by the
// AGGREGATED composite confidence (edge × extraction) descending — the
// same value the fact's ConfidenceDot already shows. Sorting by the
// last-hop edge alone drifts out of step with the dot and yields
// visible inversions (a 0.74 composite sitting above a 0.77). The
// parent path confidence is a constant factor across these siblings,
// so it doesn't change their relative order and is omitted here.
function sortSourceFactsByComposite(sourceFacts, topicFacts) {
  const ext = (idx) => {
    const c = topicFacts?.[idx]?.confidence;
    return typeof c === "number" ? c : 1;
  };
  return [...(sourceFacts || [])].sort(
    (a, b) => b[1] * ext(b[0]) - a[1] * ext(a[0]),
  );
}

// Type-scoped anchor for one fact within its topic file: "fact-1",
// "event-3", "emotion-2". Mirrors obsidianRenderer.computeFactAnchors.
function computeFactAnchors(facts) {
  const out = new Map();
  const counters = new Map();
  facts.forEach((fact, i) => {
    const type = fact.type || fact.item_type || "fact";
    counters.set(type, (counters.get(type) || 0) + 1);
    out.set(i, `${type}-${counters.get(type)}`);
  });
  return out;
}

// Match key for resolving an in-flight entity mention back to its fact's
// position in the topic's facts list. A fact and its per-mention stream
// line are both serialized from the same ExtractedItem, so the first
// evidence span's file_path + file_offset plus the summary pin the fact
// uniquely — the same fields the on-disk fact ordering keys on.
function factMatchKey(filePath, fileOffset, summary) {
  return `${filePath || ""}|${fileOffset ?? ""}|${summary || ""}`;
}

function computePatternAnchors(patterns) {
  const out = new Map();
  const counters = new Map();
  patterns.forEach((p, i) => {
    const k = (p.kind || "pattern").toLowerCase().replace(/\s+/g, "-");
    counters.set(k, (counters.get(k) || 0) + 1);
    out.set(i, `${k}-${counters.get(k)}`);
  });
  return out;
}

function insightAnchor(kind, idx) {
  const k = kind === "cross_domain" ? "cross-domain" : kind;
  return `${k}-${idx + 1}`;
}

function actionAnchor(idx) {
  return `action-${idx + 1}`;
}

// Reverse maps for cited-by sections.
function buildFactsCitedBy(patternsAll) {
  // (topic, fact_idx) → [(topic, pat_idx, conf)] sorted by -conf
  const out = {};
  Object.entries(patternsAll || {}).forEach(([topic, pats]) => {
    (pats || []).forEach((pat, pIdx) => {
      (pat.source_facts || []).forEach(([fIdx, conf]) => {
        const key = `${topic}#${fIdx}`;
        if (!out[key]) out[key] = [];
        out[key].push([topic, pIdx, conf]);
      });
    });
  });
  Object.values(out).forEach((list) =>
    list.sort((a, b) => b[2] - a[2] || a[0].localeCompare(b[0]) || a[1] - b[1])
  );
  return out;
}

function buildPatternsCitedBy(insightOutput) {
  const out = {};
  ["cross_domain", "critical"].forEach((kind) => {
    const list = insightOutput?.[kind] || [];
    list.forEach((ins, iIdx) => {
      (ins.source_patterns || []).forEach(([topic, pIdx, conf]) => {
        const key = `${topic}#${pIdx}`;
        if (!out[key]) out[key] = [];
        out[key].push([kind, topic, iIdx, conf]);
      });
    });
  });
  Object.values(out).forEach((list) =>
    list.sort((a, b) => b[3] - a[3] || a[0].localeCompare(b[0]) || a[2] - b[2])
  );
  return out;
}

function buildInsightsBackEdges(actionsList) {
  const out = {};
  (actionsList || []).forEach((a, aIdx) => {
    (a.source_insights || []).forEach(([kind, iIdx, conf]) => {
      const key = `${kind}#${iIdx}`;
      if (!out[key]) out[key] = [];
      out[key].push([aIdx, conf]);
    });
  });
  Object.values(out).forEach((list) =>
    list.sort((a, b) => b[1] - a[1] || a[0] - b[0])
  );
  return out;
}

// Look up a fact's extraction confidence by (topic, anchor).
// MarkdownView's lookupExtractionConfidence parses the link target;
// here we have (topic, anchor) directly. Returns 1.0 when the fact
// can't be resolved — graceful degrade matches MarkdownView's
// behavior so composite scores don't disappear for in-flight runs.
function lookupExtractionConfidence(topic, anchor, factsAll, factAnchorsByTopic) {
  if (!topic || !anchor) return 1.0;
  const facts = factsAll?.[topic];
  if (!facts || facts.length === 0) return 1.0;
  const anchors = factAnchorsByTopic?.[topic];
  if (!anchors) return 1.0;
  for (const [idx, a] of anchors.entries()) {
    if (a === anchor) {
      const conf = facts[idx]?.confidence;
      return typeof conf === "number" ? conf : 1.0;
    }
  }
  return 1.0;
}

// ── ConfidenceDot ──────────────────────────────────────────────────────────
// Direct port of MarkdownView.ConfidenceDot. Hue stops:
//   pct ≤ 30  → full red (hue=0).
//   pct  30..100 → red → amber → green linearly (hue 0..120).
// Opacity 50% → 100% to double-encode magnitude.
function ConfidenceDot({ pct }) {
  if (pct == null) {
    return (
      <span
        className="prov-dot prov-dot-unknown"
        style={{ background: "hsl(0, 0%, 60%)", opacity: 0.35 }}
        aria-hidden="true"
      />
    );
  }
  const clamped = Math.max(0, Math.min(100, pct));
  const hue = clamped <= 30 ? 0 : ((clamped - 30) / 70) * 120;
  const opacity = 0.5 + (clamped / 100) * 0.5;
  return (
    <span
      className="prov-dot"
      style={{
        background: `hsl(${hue.toFixed(0)}, 65%, 45%)`,
        opacity,
      }}
      aria-hidden="true"
    />
  );
}

// ── Wikilink + self-link handling ──────────────────────────────────────────

function Wikilink({ runId, relPath, anchor, currentRunId, currentRelPath, onNavigate, children, className }) {
  const normalizedRel = relPath ? relPath.replace(/\.md$/, "") : null;
  const normalizedCurrent = currentRelPath ? currentRelPath.replace(/\.md$/, "") : null;
  const isSelf =
    runId &&
    currentRunId === runId &&
    normalizedRel &&
    normalizedRel === normalizedCurrent;
  if (isSelf) {
    return (
      <span className="md-wikilink-self" title="(this page)">
        {children}
      </span>
    );
  }
  const target = { runId, relPath, anchor: anchor || "" };
  return (
    <a
      className={className || "md-wikilink md-link"}
      href="#"
      onClick={(e) => {
        e.preventDefault();
        onNavigate?.(target);
      }}
    >
      {children}
    </a>
  );
}

// Positional `[N]` insight references the actions LLM emits in prose.
// The negative lookbehind keeps array-index syntax like `arr[5]` out of
// the pool — same discriminator the chatbot citation scanner uses.
const INSIGHT_REF_RE = /(?<![A-Za-z0-9])\[(\d+)\]/g;

// Render action prose, turning each positional `[N]` insight reference
// into a clickable wikilink to the insight it points at. `[N]` is the
// actions-prompt enumeration index — cross-domain first, then critical,
// the same continuous numbering the insights tree uses — so it maps to
// exactly one insight regardless of scope. The bracket text is kept
// verbatim (we surface the reference, we don't rewrite it; the
// embedding/CONTEXT side resolves it to the title instead). Refs outside
// the enumeration and array-index syntax are left as plain text.
function linkifyInsightRefs(text, { insights, runId, onNavigate }) {
  if (!text || typeof text !== "string") return text;
  const cross = insights?.cross_domain || [];
  const crit = insights?.critical || [];
  const nCross = cross.length;
  const nTotal = nCross + crit.length;
  if (nTotal === 0) return text;

  const out = [];
  let last = 0;
  let key = 0;
  INSIGHT_REF_RE.lastIndex = 0;
  let m;
  while ((m = INSIGHT_REF_RE.exec(text)) !== null) {
    const n = Number(m[1]);
    if (n < 1 || n > nTotal) continue; // not an insight ref — leave as text
    const kind = n <= nCross ? "cross_domain" : "critical";
    const localIdx = n <= nCross ? n - 1 : n - 1 - nCross;
    if (m.index > last) out.push(text.slice(last, m.index));
    out.push(
      <Wikilink
        key={`iref-${key++}`}
        runId={runId}
        relPath={`${STAGE_DIRS[4]}.md`}
        anchor={insightAnchor(kind, localIdx)}
        currentRunId={runId}
        currentRelPath={`${STAGE_DIRS[5]}.md`}
        onNavigate={onNavigate}
        className="md-wikilink md-link md-insight-ref"
      >
        {m[0]}
      </Wikilink>
    );
    last = m.index + m[0].length;
  }
  if (out.length === 0) return text; // no resolvable refs
  if (last < text.length) out.push(text.slice(last));
  return out;
}

// ── Section wrapper ────────────────────────────────────────────────────────
// Mirrors MarkdownView.Section: a <section class="md-section"> wraps
// the heading + body so the transient-highlight (`element.closest('.md-section')`)
// can fade the whole block when a wikilink lands here. Heading id =
// the type-scoped anchor; the visible bracket prefix is suppressed
// (MarkdownView strips the same `[fact-1]` prefix from rendered text).
function Section({ anchor, level = 2, displayNum, title, meta, children }) {
  const Tag = `h${level}`;
  return (
    <section className="md-section" id={anchor || undefined}>
      <Tag id={anchor || undefined} className={`md-h md-h${level} md-section-heading`}>
        {displayNum != null && (
          <span className="md-h-num">{displayNum}</span>
        )}
        {title}
      </Tag>
      <div className="md-section-body">
        {meta ? <p className="md-p md-meta-line"><em>{meta}</em></p> : null}
        {children}
      </div>
    </section>
  );
}

// ── Provenance rows (root / mid / leaf) ────────────────────────────────────

// Root row: bold summary + `[Type Tag]` + ConfidenceDot, whole row clickable.
function ProvRowRoot({ pct, summary, typeTag, navTarget, onNavigate, rowTitle }) {
  const navigate = () => navTarget && onNavigate?.(navTarget);
  const onKeyDown = (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      navigate();
    }
  };
  return (
    <div
      className="prov-row prov-root"
      role="link"
      tabIndex={0}
      onClick={navigate}
      onKeyDown={onKeyDown}
      data-score={pct != null ? (pct / 100).toFixed(2) : undefined}
      title={rowTitle}
    >
      <ConfidenceDot pct={pct} />
      <span className="prov-summary">
        <strong>{summary}</strong>
        {typeTag ? <span className="prov-tag"> [{typeTag}]</span> : null}
      </span>
    </div>
  );
}

// Mid row: plain summary + ConfidenceDot, whole row clickable.
function ProvRowMid({ pct, summary, navTarget, onNavigate, rowTitle }) {
  const navigate = () => navTarget && onNavigate?.(navTarget);
  const onKeyDown = (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      navigate();
    }
  };
  return (
    <div
      className="prov-row prov-mid"
      role="link"
      tabIndex={0}
      onClick={navigate}
      onKeyDown={onKeyDown}
      data-score={pct != null ? (pct / 100).toFixed(2) : undefined}
      title={rowTitle}
    >
      <ConfidenceDot pct={pct} />
      <span className="prov-summary">{summary}</span>
    </div>
  );
}

// Leaf row: vertical grey bar + italic quote + filename pushed to
// the right edge. Filename renders as muted grey text, NOT a blue
// link — the whole row is the click target in patterns / insights /
// actions. In facts (clickable={false}) the row is plain readable
// markup with no nav: facts already cite their source inline, an
// extra hop into the raw input file isn't useful from there.
function ProvRowLeaf({ quote, filename, navTarget, onNavigate, clickable = true }) {
  const navigate = () => clickable && navTarget && onNavigate?.(navTarget);
  const onKeyDown = (e) => {
    if (!clickable) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      navigate();
    }
  };
  const interactive = clickable && navTarget;
  return (
    <div
      className={`prov-leaf${interactive ? " prov-leaf-clickable" : ""}`}
      role={interactive ? "link" : undefined}
      tabIndex={interactive ? 0 : undefined}
      onClick={interactive ? navigate : undefined}
      onKeyDown={interactive ? onKeyDown : undefined}
      title={filename}
    >
      <em className="prov-quote">{`"${quote}"`}</em>
      <span className="prov-file">{filename}</span>
    </div>
  );
}

function rowTitleFor(extractionPct, matchPct, compositePct) {
  if (compositePct == null) return undefined;
  if (matchPct != null && extractionPct != null && extractionPct < 100) {
    return `${compositePct}% confidence (${extractionPct}% extraction × ${matchPct}% match)`;
  }
  return `${compositePct}% match`;
}

// One fact-reference row. Looks up the fact by (topic, fact_idx) in the
// in-memory facts map and renders it as a ProvRowMid that links to the
// fact's anchor inside its topic .md. Returns null when the ref doesn't
// resolve (topic missing, idx out of range, or facts not yet loaded) —
// the caller doesn't render an empty <li>.
function FactRefRow({ refTopic, refIdx, factsAll, factAnchorsByTopic, runId, onNavigate }) {
  if (!refTopic || refIdx == null) return null;
  const fact = factsAll?.[refTopic]?.[refIdx];
  if (!fact) return null;
  const anchor = factAnchorsByTopic[refTopic]?.get(refIdx) || `fact-${refIdx + 1}`;
  const navTarget = { runId, relPath: `${STAGE_DIRS[1]}/${refTopic}.md`, anchor };
  const extractionConfRaw = typeof fact.confidence === "number" ? fact.confidence : 1.0;
  const pct = Math.round(extractionConfRaw * 100);
  const summary = fact.summary || "(no summary)";
  const dateLabel = prettyDate(fact.occurred_at);
  // A fact extracted under several categories is cited once here (the
  // pipeline collapses the fan-out to the alphabetically-first category);
  // show the fact's full category list, alphabetical, in one merged tag.
  const typeLabel = titleCaseSlug(fact.type || fact.item_type || "fact");
  const cats = Array.isArray(fact.topics) && fact.topics.length
    ? [...fact.topics].sort()
    : [refTopic];
  const catLabel = cats.map(titleCaseSlug).join(", ");
  const tag = cats.length > 1
    ? `[${catLabel} · ${typeLabel}]`
    : `[${catLabel} ${typeLabel}]`;
  return (
    <ProvRowMid
      pct={pct}
      summary={
        <>
          {dateLabel ? <span className="fact-date">{dateLabel}</span> : null}
          {summary}{" "}
          <span className="prov-tag">{tag}</span>
        </>
      }
      navTarget={navTarget}
      onNavigate={onNavigate}
      rowTitle={`${pct}% extraction confidence`}
    />
  );
}

// ── ProvenanceShares: per-file ranked pills ────────────────────────────────
// Aggregates leaf weight share (extraction × match) per raw-input
// file and renders one pill per file, opacity-coded by share. Click a
// pill → navigate to the file.
function ProvenanceShares({ leaves, onNavigate }) {
  if (leaves.length === 0) {
    return (
      <span className="field-hint" style={{ marginLeft: 4 }}>
        (no raw-input refs)
      </span>
    );
  }
  return leaves.map((entry, i) => {
    const pct = Math.round(entry.share * 100);
    const opacity = 0.4 + 0.6 * Math.min(1, entry.share);
    const onClick = entry.navTarget
      ? () => onNavigate?.(entry.navTarget)
      : undefined;
    return (
      <span key={entry.file} className="md-provenance-pill">
        {i > 0 && (
          <span className="md-provenance-sep" aria-hidden="true">
            {" · "}
          </span>
        )}
        <span style={{ opacity }} className="md-provenance-pct">
          {pct}%
        </span>{" "}
        <span
          className={`md-provenance-file ${onClick ? "is-link" : ""}`}
          onClick={onClick}
          role={onClick ? "link" : undefined}
          tabIndex={onClick ? 0 : undefined}
          onKeyDown={
            onClick
              ? (e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onClick();
                  }
                }
              : undefined
          }
          style={{ opacity }}
          title={entry.file}
        >
          {entry.file}
        </span>
      </span>
    );
  });
}

// Compute weighted shares per raw-input file from a flat list of leaf
// rows. Mirrors MarkdownView.computeProvenanceShares: weight =
// extraction_confidence × match_score, ranked by descending share.
function computeShares(leaves) {
  const navByFile = new Map();
  const weightByFile = new Map();
  let total = 0;
  for (const lf of leaves) {
    const weight = lf.extractionConf * lf.matchScore;
    if (weight <= 0) continue;
    total += weight;
    weightByFile.set(lf.file, (weightByFile.get(lf.file) || 0) + weight);
    if (!navByFile.has(lf.file) && lf.navTarget) {
      navByFile.set(lf.file, { ...lf.navTarget, anchor: "" });
    }
  }
  if (total === 0) return [];
  return [...weightByFile.entries()]
    .map(([file, w]) => ({
      file,
      share: w / total,
      navTarget: navByFile.get(file) || null,
    }))
    .sort((a, b) => b.share - a.share);
}

// ── Provenance wrapper (chevron + collapsible tree) ────────────────────────
function Provenance({
  collapseKey,
  collapsedSlugs,
  toggleSection,
  leaves,
  onNavigate,
  children,
}) {
  const collapsed = collapseKey ? !!collapsedSlugs?.has(collapseKey) : false;
  const toggle = () => collapseKey && toggleSection?.(collapseKey);
  const onLabelKeyDown = (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggle();
    }
  };
  const shares = useMemo(() => computeShares(leaves || []), [leaves]);
  return (
    <div className="md-provenance">
      <p className="md-p md-provenance-inline">
        <span
          className="md-provenance-toggle"
          role="button"
          tabIndex={0}
          onClick={toggle}
          onKeyDown={onLabelKeyDown}
          aria-expanded={!collapsed}
          title="% of evidence weight (extraction × match)"
        >
          <span className="md-provenance-chevron" aria-hidden="true">
            {collapsed ? "▶" : "▼"}
          </span>
          <span className="md-provenance-label">Provenance:</span>
        </span>{" "}
        <ProvenanceShares leaves={shares} onNavigate={onNavigate} />
      </p>
      {!collapsed && children}
    </div>
  );
}

// ── FactSourceBlock: `Source: filename` header + clickable quote ──────────
// Header line stays as plain text — `Source:` label + filename in
// muted grey, no link styling. The verbatim quote in the blockquote
// underneath IS the click target; clicking it navigates to the raw
// input at the offset. Header filename is intentionally NOT a link
// so the labelling stays readable without the blue-pill noise.
function FactSourceBlock({ runId, evidence, onNavigate }) {
  if (!evidence || evidence.length === 0) return null;
  return (
    <div className="md-fact-source">
      {evidence.map((ev, i) => {
        const quote = (ev.text || "").trim();
        const filename = ev.file_path || "";
        const offset = ev.file_offset;
        const anchor = offset != null ? `offset-${offset}` : "";
        const navTarget = filename
          ? { runId, relPath: `0-inputs/${filename}`, anchor }
          : null;
        const navigate = () => navTarget && onNavigate?.(navTarget);
        const onKeyDown = (e) => {
          if (!navTarget) return;
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            navigate();
          }
        };
        return (
          <div key={i} className="prov-fact-source">
            {filename && (
              <p className="prov-fact-source-header md-p">
                <strong>Source:</strong>{" "}
                <span className="prov-fact-source-filename">{filename}</span>
              </p>
            )}
            {quote && (
              <blockquote
                className={`md-blockquote prov-fact-source-quote${
                  navTarget ? " prov-fact-source-quote-clickable" : ""
                }`}
                role={navTarget ? "link" : undefined}
                tabIndex={navTarget ? 0 : undefined}
                onClick={navTarget ? navigate : undefined}
                onKeyDown={navTarget ? onKeyDown : undefined}
                title={navTarget ? `Open ${filename}` : undefined}
              >
                {quote}
              </blockquote>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Provenance tree builder ────────────────────────────────────────────────
// Walks the source graph and emits a nested <ul class="prov-list"> of
// rows + sub-trees. Each invocation returns one nested level. The
// `leavesOut` array accumulates leaf metadata (file, share weights)
// as we walk so the parent Provenance wrapper can render its inline
// shares pills above the tree.
//
// Confidence math (matches MarkdownView):
//   - edge_conf : pattern→fact / insight→pattern / action→insight strength
//   - match_pct : path multiplier (parentPathConf × edge_conf × 100)
//   - extraction: lookup by (topic, fact_anchor); 1.0 if not a fact target
//   - composite : extraction × match  → ConfidenceDot color + row tooltip

function emitFactSubtree({
  runId, topic, factIdx, fact, factAnchor,
  parentPathConf, factsAll, factAnchorsByTopic,
  onNavigate, leavesOut,
}) {
  const matchPct = Math.round(parentPathConf * 100);
  const extractionConfRaw =
    typeof fact.confidence === "number" ? fact.confidence : 1.0;
  const extractionPct = Math.round(extractionConfRaw * 100);
  const compositePct = Math.round(extractionConfRaw * parentPathConf * 100);
  const summary = fact.summary || "";
  const dateLabel = prettyDate(fact.occurred_at);
  const navTarget = {
    runId,
    relPath: `${STAGE_DIRS[1]}/${topic}.md`,
    anchor: factAnchor,
  };
  const rowTitle = rowTitleFor(extractionPct, matchPct, compositePct);

  const evidence = fact.evidence || [];
  const childItems = evidence
    .map((ev, evIdx) => {
      if (!ev.file_path) return null;
      const offset = ev.file_offset;
      const filename = ev.file_path;
      const anchor = offset != null ? `offset-${offset}` : "";
      const evNav = { runId, relPath: `0-inputs/${filename}`, anchor };
      const quote = (ev.text || "").trim().replace(/\n/g, " ");
      // Each evidence row carries the same composite — extraction
      // belongs to the fact, not the snippet. Match propagates through.
      leavesOut.push({
        file: filename,
        extractionConf: extractionConfRaw,
        matchScore: parentPathConf,
        navTarget: evNav,
      });
      return (
        <li key={`ev-${evIdx}`} className="prov-item prov-item-leaf">
          <ProvRowLeaf
            quote={quote}
            filename={filename}
            navTarget={evNav}
            onNavigate={onNavigate}
          />
        </li>
      );
    })
    .filter(Boolean);

  return (
    <li key={`f-${topic}-${factIdx}`} className="prov-item prov-item-mid">
      <ProvRowMid
        pct={compositePct}
        summary={
          <>
            {dateLabel ? <span className="fact-date">{dateLabel}</span> : null}
            {summary}
          </>
        }
        navTarget={navTarget}
        onNavigate={onNavigate}
        rowTitle={rowTitle}
      />
      {childItems.length > 0 && (
        <ul className="prov-list prov-depth-1">{childItems}</ul>
      )}
    </li>
  );
}

function emitPatternSubtree({
  runId, topic, patIdx, pat, patAnchor,
  parentPathConf, factsAll, factAnchorsByTopic,
  onNavigate, leavesOut,
}) {
  const matchPct = Math.round(parentPathConf * 100);
  const compositePct = matchPct; // patterns: no extraction conf
  const navTarget = {
    runId,
    relPath: `${STAGE_DIRS[3]}/${topic}.md`,
    anchor: patAnchor,
  };
  const summary = pat.name || pat.description || "(unnamed pattern)";
  const typeTag = `${titleCaseSlug(topic)} ${titleCaseSlug(
    (pat.kind || "pattern").toLowerCase().replace(/\s+/g, "-")
  )}`;
  const rowTitle = rowTitleFor(null, matchPct, compositePct);

  const sortedFacts = sortSourceFactsByComposite(
    pat.source_facts,
    factsAll?.[topic] || [],
  );
  const childItems = sortedFacts
    .map(([fIdx, edgeConf]) => {
      const topicFacts = factsAll?.[topic] || [];
      if (fIdx < 0 || fIdx >= topicFacts.length) return null;
      const fact = topicFacts[fIdx];
      const fAnchor =
        factAnchorsByTopic?.[topic]?.get(fIdx) || `fact-${fIdx + 1}`;
      return emitFactSubtree({
        runId,
        topic,
        factIdx: fIdx,
        fact,
        factAnchor: fAnchor,
        parentPathConf: parentPathConf * edgeConf,
        factsAll,
        factAnchorsByTopic,
        onNavigate,
        leavesOut,
      });
    })
    .filter(Boolean);

  return (
    <li key={`p-${topic}-${patIdx}`} className="prov-item prov-item-root">
      <ProvRowRoot
        pct={compositePct}
        summary={summary}
        typeTag={typeTag}
        navTarget={navTarget}
        onNavigate={onNavigate}
        rowTitle={rowTitle}
      />
      {childItems.length > 0 && (
        <ul className="prov-list prov-depth-1">{childItems}</ul>
      )}
    </li>
  );
}

function emitInsightSubtree({
  runId, kind, iIdx, ins,
  parentPathConf, patternsAll, patternAnchorsByTopic,
  factsAll, factAnchorsByTopic,
  onNavigate, leavesOut,
}) {
  const matchPct = Math.round(parentPathConf * 100);
  const compositePct = matchPct;
  const iAnchor = insightAnchor(kind, iIdx);
  const navTarget = {
    runId,
    relPath: `${STAGE_DIRS[4]}.md`,
    anchor: iAnchor,
  };
  const summary = ins.name || ins.description || "(unnamed insight)";
  const typeTag = `${titleCaseSlug(
    kind === "cross_domain" ? "cross-domain" : kind
  )} Insight`;
  const rowTitle = rowTitleFor(null, matchPct, compositePct);

  const sortedPats = [...(ins.source_patterns || [])].sort(
    (a, b) => b[2] - a[2]
  );
  const childItems = sortedPats
    .map(([topic, pIdx, edgeConf]) => {
      const topicPats = patternsAll?.[topic] || [];
      if (pIdx < 0 || pIdx >= topicPats.length) return null;
      const pat = topicPats[pIdx];
      const pAnchor =
        patternAnchorsByTopic?.[topic]?.get(pIdx) || `pattern-${pIdx + 1}`;
      return emitPatternSubtree({
        runId,
        topic,
        patIdx: pIdx,
        pat,
        patAnchor: pAnchor,
        parentPathConf: parentPathConf * edgeConf,
        factsAll,
        factAnchorsByTopic,
        onNavigate,
        leavesOut,
      });
    })
    .filter(Boolean);

  return (
    <li key={`i-${kind}-${iIdx}`} className="prov-item prov-item-root">
      <ProvRowRoot
        pct={compositePct}
        summary={summary}
        typeTag={typeTag}
        navTarget={navTarget}
        onNavigate={onNavigate}
        rowTitle={rowTitle}
      />
      {childItems.length > 0 && (
        <ul className="prov-list prov-depth-1">{childItems}</ul>
      )}
    </li>
  );
}

// ── Page-level toolbar (Collapse all / Expand all) ─────────────────────────
function PageToolbar({ collapseAll, expandAll }) {
  return (
    <div className="md-page-toolbar">
      <button type="button" className="md-page-toolbar-btn" onClick={collapseAll}>
        Collapse all
      </button>
      <button type="button" className="md-page-toolbar-btn" onClick={expandAll}>
        Expand all
      </button>
    </div>
  );
}

// Hook that owns the per-view collapse state. Returns the set + the
// three handlers and a stable allKeys updater so the toolbar's
// "Collapse all" can fold every block on the page.
function useCollapseState() {
  const [collapsedSlugs, setCollapsedSlugs] = useState(() => new Set());
  const toggleSection = useCallback((key) => {
    setCollapsedSlugs((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);
  const collapseAll = useCallback((keys) => {
    setCollapsedSlugs(new Set(keys));
  }, []);
  const expandAll = useCallback(() => {
    setCollapsedSlugs(new Set());
  }, []);
  return { collapsedSlugs, toggleSection, collapseAll, expandAll };
}

// ── FactsView ──────────────────────────────────────────────────────────────

export function FactsView({ runId, topic, refreshTick, onNavigate }) {
  const [facts, setFacts] = useState([]);
  const [patternsAll, setPatternsAll] = useState({});
  const [entitiesPayload, setEntitiesPayload] = useState(null);

  useEffect(() => {
    let cancelled = false;
    invoke("read_run_facts_for_topic", { runId, topic })
      .then((r) => { if (!cancelled) setFacts(Array.isArray(r) ? r : []); })
      .catch(() => { if (!cancelled) setFacts([]); });
    invoke("read_run_patterns_all", { runId })
      .then((r) => { if (!cancelled) setPatternsAll(r && typeof r === "object" ? r : {}); })
      .catch(() => { if (!cancelled) setPatternsAll({}); });
    invoke("read_run_entities", { runId })
      .then((r) => { if (!cancelled) setEntitiesPayload(r || null); })
      .catch(() => { if (!cancelled) setEntitiesPayload(null); });
    return () => { cancelled = true; };
  }, [runId, topic, refreshTick]);

  const factAnchors = useMemo(() => computeFactAnchors(facts), [facts]);
  // Display the facts newest-first. Anchors + cited-by stay keyed by the
  // original JSONL index (computeFactAnchors above runs on `facts`), so
  // reordering the render sequence here never moves a wikilink target.
  const displayOrder = useMemo(
    () => reverseChronIndices(facts, factDateKey),
    [facts],
  );
  const citedByMap = useMemo(() => buildFactsCitedBy(patternsAll), [patternsAll]);
  const patternAnchorsByTopic = useMemo(() => {
    const out = {};
    Object.entries(patternsAll).forEach(([t, pats]) => {
      out[t] = computePatternAnchors(pats || []);
    });
    return out;
  }, [patternsAll]);
  // (normalized name + entity_type) → {canonical_id, canonical_name}.
  // Aliases map to the same record, so a fact's raw entity ref can
  // resolve via either the canonical name or any alias the entities
  // stage glued onto the group. Without this index the wikilink target
  // is a guess; with it, the link lands on the actual entity page.
  const entityIndex = useMemo(() => {
    const idx = new Map();
    const ents = entitiesPayload?.entities || [];
    const norm = (s) => (s || "").toLowerCase().trim().replace(/\s+/g, " ");
    for (const e of ents) {
      const type = (e.entity_type || "").toLowerCase();
      const record = {
        canonical_id: e.canonical_id,
        canonical_name: e.canonical_name,
      };
      if (e.canonical_name) {
        idx.set(`${norm(e.canonical_name)}|${type}`, record);
        // Also key by name-only as a fallback when fact's entity_type
        // disagrees with the catalog's (e.g., "person" vs "other").
        idx.set(`${norm(e.canonical_name)}|`, record);
      }
      for (const a of e.aliases || []) {
        idx.set(`${norm(a)}|${type}`, record);
        idx.set(`${norm(a)}|`, record);
      }
    }
    return idx;
  }, [entitiesPayload]);
  const resolveEntityRef = (rawName, entityType) => {
    const norm = (s) => (s || "").toLowerCase().trim().replace(/\s+/g, " ");
    const t = (entityType || "").toLowerCase();
    return (
      entityIndex.get(`${norm(rawName)}|${t}`) ||
      entityIndex.get(`${norm(rawName)}|`) ||
      null
    );
  };

  return (
    <article className="md-view">
      <h1 className="md-h md-h1">{titleCaseSlug(topic)} ({facts.length})</h1>
      {displayOrder.map((idx, pos) => {
        const fact = facts[idx];
        const anchor = factAnchors.get(idx);
        const summary = fact.summary || "";
        const itemType = fact.type || fact.item_type || "fact";
        const conf = (typeof fact.confidence === "number" ? fact.confidence : 1).toFixed(2);
        const dateLabel = prettyDate(fact.occurred_at);
        const occurredText = (fact.occurred_at_text || "").trim();
        // Title carries the formatted date (grey) in front of the summary
        // when the fact is dated. Subtitle shows the type + explicit
        // confidence; for an undated fact it keeps the word "undated" up
        // front, with the raw date mention quoted in parens when one
        // exists but couldn't be resolved to a calendar date.
        // The same fact can be filed under several categories; surface the
        // OTHER categories it belongs to after the fact text.
        const otherCats = (Array.isArray(fact.topics) ? fact.topics : [])
          .filter((t) => t && t !== topic)
          .sort();
        const title = (
          <>
            {dateLabel ? <span className="fact-date">{dateLabel}</span> : null}
            {summary}
            {otherCats.length ? (
              <>
                {" "}
                <span className="prov-tag">[{otherCats.map(titleCaseSlug).join(", ")}]</span>
              </>
            ) : null}
          </>
        );
        const undatedMarker = dateLabel
          ? ""
          : `undated${occurredText ? ` ("${occurredText}")` : ""} · `;
        const meta = `${undatedMarker}${itemType} · confidence ${conf}`;
        const ents = fact.entities || [];
        const cited = citedByMap[`${topic}#${idx}`] || [];
        return (
          <Section key={idx} anchor={anchor} displayNum={pos + 1} title={title} meta={meta}>
            {ents.length > 0 ? (
              <p className="md-p">
                <strong>Entities:</strong>{" "}
                {ents.map((r, i) => {
                  const rawName = r.name || r.canonical_name || "(unknown)";
                  const role = r.role || "mentioned";
                  const resolved = resolveEntityRef(rawName, r.entity_type || r.type);
                  const display = resolved
                    ? titleCaseSlug(resolved.canonical_name)
                    : rawName;
                  return (
                    <span key={i}>
                      {i > 0 ? ", " : null}
                      {resolved ? (
                        <Wikilink
                          runId={runId}
                          relPath={`${STAGE_DIRS[2]}/${resolved.canonical_id}.md`}
                          currentRunId={runId}
                          currentRelPath={`${STAGE_DIRS[1]}/${topic}.md`}
                          onNavigate={onNavigate}
                        >
                          {display}
                        </Wikilink>
                      ) : (
                        <span>{display}</span>
                      )}
                      {role !== "mentioned" ? <span className="prov-tag"> ({role})</span> : null}
                    </span>
                  );
                })}
              </p>
            ) : null}
            <FactSourceBlock
              runId={runId}
              evidence={fact.evidence || []}
              onNavigate={onNavigate}
            />
            {cited.length > 0 ? (
              <>
                <p className="md-p"><strong>Cited by:</strong></p>
                <ul className="prov-list">
                  {cited.map(([t, pIdx, edgeConf], i) => {
                    const a = patternAnchorsByTopic[t]?.get(pIdx) || `pattern-${pIdx + 1}`;
                    const pats = patternsAll?.[t] || [];
                    const pat = pats[pIdx];
                    if (!pat) return null;
                    const pct = Math.round(edgeConf * 100);
                    const navTarget = {
                      runId,
                      relPath: `${STAGE_DIRS[3]}/${t}.md`,
                      anchor: a,
                    };
                    const typeTag = `${titleCaseSlug(t)} ${titleCaseSlug(
                      (pat.kind || "pattern").toLowerCase().replace(/\s+/g, "-")
                    )}`;
                    return (
                      <li key={i} className="prov-item prov-item-root">
                        <ProvRowRoot
                          pct={pct}
                          summary={pat.name || pat.description || "(unnamed pattern)"}
                          typeTag={typeTag}
                          navTarget={navTarget}
                          onNavigate={onNavigate}
                          rowTitle={`${pct}% match`}
                        />
                      </li>
                    );
                  })}
                </ul>
              </>
            ) : null}
          </Section>
        );
      })}
      {facts.length === 0 ? (
        <p className="md-empty"><em>(No facts yet — extraction in progress or topic empty.)</em></p>
      ) : null}
    </article>
  );
}

// ── EntityView ─────────────────────────────────────────────────────────────

export function EntityView({ runId, entityId, refreshTick, onNavigate }) {
  const [entity, setEntity] = useState(null);
  const [allEntities, setAllEntities] = useState(null);
  const [factsAll, setFactsAll] = useState({});
  // "Cited in facts" expander: capped at REF_LIMIT by default so a
  // 200-ref entity doesn't dominate the page; click "+N more" to load
  // the full list. Resets when the user navigates to a different entity.
  const [showAllRefs, setShowAllRefs] = useState(false);
  // Per-relation evidence disclosure. A Set of relation indices that
  // are currently expanded. Default-collapsed because high-degree
  // entities (Walter Scott had 58 relations on a single run) would
  // otherwise drown the page in expanded evidence rows.
  const [expandedRelations, setExpandedRelations] = useState(() => new Set());
  useEffect(() => {
    setShowAllRefs(false);
    setExpandedRelations(new Set());
  }, [runId, entityId]);

  useEffect(() => {
    let cancelled = false;
    invoke("read_run_entity", { runId, entityId })
      .then((r) => { if (!cancelled) setEntity(r || null); })
      .catch(() => { if (!cancelled) setEntity(null); });
    invoke("read_run_entities", { runId })
      .then((r) => { if (!cancelled) setAllEntities(r || null); })
      .catch(() => { if (!cancelled) setAllEntities(null); });
    invoke("read_run_facts_all", { runId })
      .then((r) => { if (!cancelled) setFactsAll(r && typeof r === "object" ? r : {}); })
      .catch(() => { if (!cancelled) setFactsAll({}); });
    return () => { cancelled = true; };
  }, [runId, entityId, refreshTick]);

  const idToName = useMemo(() => {
    const map = {};
    (allEntities?.entities || []).forEach((e) => {
      map[e.canonical_id] = e.canonical_name;
    });
    return map;
  }, [allEntities]);

  const relations = useMemo(() => {
    if (!entity || !allEntities?.relations) return [];
    return (allEntities.relations || [])
      .filter((r) => r.from === entity.canonical_id || r.to === entity.canonical_id)
      .slice()
      .sort((a, b) => (b.confidence ?? 1) - (a.confidence ?? 1));
  }, [entity, allEntities]);

  const factAnchorsByTopic = useMemo(() => {
    const out = {};
    Object.entries(factsAll).forEach(([t, facts]) => {
      out[t] = computeFactAnchors(facts || []);
    });
    return out;
  }, [factsAll]);

  // (topic → Map(matchKey → fact idx)). Lets a consolidating entity's
  // live-mention rows resolve to their fact's anchor: the in-flight
  // mention stream carries the fact's summary + evidence but NOT its
  // topic index, so without this a live-mention click navigates to the
  // topic head with an empty anchor and never scrolls to the fact.
  // Rebuilt whenever factsAll re-fetches (each refreshTick), so it stays
  // correct even after Phase 3's in-place re-sort shifts indices — the
  // key fields are immutable extraction output, only the index moves.
  const factIdxByKey = useMemo(() => {
    const out = {};
    Object.entries(factsAll).forEach(([t, facts]) => {
      const m = new Map();
      (facts || []).forEach((f, i) => {
        const ev = Array.isArray(f.evidence) ? f.evidence[0] : null;
        m.set(factMatchKey(ev?.file_path, ev?.file_offset, f.summary), i);
      });
      out[t] = m;
    });
    return out;
  }, [factsAll]);

  if (!entity) {
    return (
      <article className="md-view">
        <h1 className="md-h md-h1">{titleCaseSlug(entityId)}</h1>
        <p className="md-empty"><em>(Entity not yet resolved — entities stage in progress.)</em></p>
      </article>
    );
  }

  const isConsolidating = entity._state === "consolidating";
  const role = entity.role || "";
  const desc = (entity.description || "").trim();
  const meta = isConsolidating
    ? `${entity.entity_type || "entity"} · consolidating · ${
        entity.mention_count || 0
      } mentions`
    : `${entity.entity_type || "entity"}${role ? ` · ${role}` : ""} · ${
        entity.mention_count || 0
      } mentions`;
  const refs = entity.evidence_fact_refs || [];
  const mentions = entity.mentions || [];
  const REF_LIMIT = 30;
  // Fact lists under an entity render newest-first (by the cited fact's
  // occurred_at), matching the facts view. The cap slices the list
  // *after* the reverse-chron sort, so the 30 shown are the 30 most
  // recent. Resolve each ref's date through factsAll; undated sink last.
  const refDateKey = (ref) => {
    const [t, f] = refTopicIdx(ref);
    return factDateKey(factsAll?.[t]?.[f]);
  };
  const orderedRefs = reverseChronIndices(refs, refDateKey).map((i) => refs[i]);
  const orderedMentions = reverseChronIndices(mentions, factDateKey).map(
    (i) => mentions[i],
  );

  return (
    <article className="md-view">
      <h1 className="md-h md-h1">{titleCaseSlug(entity.canonical_name)}</h1>
      <p className="md-p md-meta-line"><em>{meta}</em></p>
      {isConsolidating ? (
        <p className="md-p md-consolidating-banner">
          <em>(Entity is still being consolidated. Live mentions from
          extraction shown below — final canonical record with description,
          aliases, and relations will appear when entities stage finishes.)</em>
        </p>
      ) : null}
      {desc ? <p className="md-p">{desc}</p> : null}

      {(entity.aliases || []).length > 0 ? (
        <>
          <h2 className="md-h md-h2">Aliases</h2>
          <ul>{entity.aliases.map((a, i) => <li key={i}>{a}</li>)}</ul>
        </>
      ) : null}

      {relations.length > 0 ? (
        <>
          <h2 className="md-h md-h2">Relations</h2>
          <ul>
            {relations.map((r, i) => {
              const fromName = titleCaseSlug(idToName[r.from] || r.from);
              const toName = titleCaseSlug(idToName[r.to] || r.to);
              const conf = typeof r.confidence === "number" ? r.confidence : 1;
              const evidenceRefs = Array.isArray(r.evidence_fact_refs)
                ? r.evidence_fact_refs
                : [];
              const expanded = expandedRelations.has(i);
              const toggleEvidence = () =>
                setExpandedRelations((prev) => {
                  const next = new Set(prev);
                  if (next.has(i)) next.delete(i);
                  else next.add(i);
                  return next;
                });
              return (
                <li key={i}>
                  <Wikilink
                    runId={runId}
                    relPath={`${STAGE_DIRS[2]}/${r.from}.md`}
                    currentRunId={runId}
                    currentRelPath={`${STAGE_DIRS[2]}/${entity.canonical_id}.md`}
                    onNavigate={onNavigate}
                  >
                    {fromName}
                  </Wikilink>{" "}
                  is <em>{r.relation}</em> of{" "}
                  <Wikilink
                    runId={runId}
                    relPath={`${STAGE_DIRS[2]}/${r.to}.md`}
                    currentRunId={runId}
                    currentRelPath={`${STAGE_DIRS[2]}/${entity.canonical_id}.md`}
                    onNavigate={onNavigate}
                  >
                    {toName}
                  </Wikilink>{" "}
                  <span className="prov-tag">(confidence {conf.toFixed(2)})</span>
                  {evidenceRefs.length > 0 ? (
                    <>
                      {" "}
                      <button
                        type="button"
                        className="md-show-more rel-evidence-toggle"
                        aria-expanded={expanded}
                        onClick={toggleEvidence}
                      >
                        {expanded ? "hide" : "see"}{" "}
                        {evidenceRefs.length}{" "}
                        {evidenceRefs.length === 1 ? "source" : "sources"}
                      </button>
                      {expanded ? (
                        <ul className="prov-list rel-evidence-list">
                          {reverseChronIndices(evidenceRefs, refDateKey)
                            .map((k) => evidenceRefs[k])
                            .map((ref, j) => {
                            const [t, fIdx] = refTopicIdx(ref);
                            if (!t || fIdx == null) return null;
                            if (!factsAll?.[t]?.[fIdx]) return null;
                            return (
                              <li key={j} className="prov-item prov-item-mid">
                                <FactRefRow
                                  refTopic={t}
                                  refIdx={fIdx}
                                  factsAll={factsAll}
                                  factAnchorsByTopic={factAnchorsByTopic}
                                  runId={runId}
                                  onNavigate={onNavigate}
                                />
                              </li>
                            );
                          })}
                        </ul>
                      ) : null}
                    </>
                  ) : null}
                </li>
              );
            })}
          </ul>
        </>
      ) : null}

      {refs.length > 0 ? (
        <>
          <h2 className="md-h md-h2">Cited in facts ({refs.length})</h2>
          <ul className="prov-list">
            {(showAllRefs ? orderedRefs : orderedRefs.slice(0, REF_LIMIT)).map((ref, i) => {
              const [t, fIdx] = refTopicIdx(ref);
              if (!t || fIdx == null) return null;
              if (!factsAll?.[t]?.[fIdx]) return null;
              return (
                <li key={i} className="prov-item prov-item-mid">
                  <FactRefRow
                    refTopic={t}
                    refIdx={fIdx}
                    factsAll={factsAll}
                    factAnchorsByTopic={factAnchorsByTopic}
                    runId={runId}
                    onNavigate={onNavigate}
                  />
                </li>
              );
            })}
            {refs.length > REF_LIMIT && !showAllRefs ? (
              <li className="prov-item">
                <button
                  type="button"
                  className="md-show-more"
                  onClick={() => setShowAllRefs(true)}
                >
                  +{refs.length - REF_LIMIT} more
                </button>
              </li>
            ) : null}
          </ul>
        </>
      ) : null}

      {isConsolidating && mentions.length > 0 ? (
        <>
          <h2 className="md-h md-h2">Live mentions ({mentions.length})</h2>
          <ul className="prov-list">
            {(showAllRefs ? orderedMentions : orderedMentions.slice(0, REF_LIMIT)).map((m, i) => {
              const summary = m.fact_summary || "(no summary)";
              const topic = m.topic || (Array.isArray(m.topics) ? m.topics[0] : "") || "";
              const mRole = m.role || "";
              const dateLabel = prettyDate(m.occurred_at);
              // Resolve this mention back to its fact so the click lands
              // on (and highlights) the exact fact. Fall back to the
              // topic head (empty anchor) only when the mention can't be
              // matched — e.g. the fact isn't flushed to the topic file
              // yet on a still-extracting run.
              const mEv = Array.isArray(m.evidence) ? m.evidence[0] : null;
              const mFactIdx = topic
                ? factIdxByKey[topic]?.get(
                    factMatchKey(mEv?.file_path, mEv?.file_offset, m.fact_summary),
                  )
                : undefined;
              const mAnchor =
                (mFactIdx != null && factAnchorsByTopic[topic]?.get(mFactIdx)) || "";
              const navTarget = topic ? {
                runId,
                relPath: `${STAGE_DIRS[1]}/${topic}.md`,
                anchor: mAnchor,
              } : null;
              const mPct = typeof m.confidence === "number"
                ? Math.round(m.confidence * 100)
                : null;
              return (
                <li key={i} className="prov-item prov-item-mid">
                  <ProvRowMid
                    pct={mPct}
                    summary={
                      <>
                        {dateLabel ? <span className="fact-date">{dateLabel}</span> : null}
                        {summary}
                        {topic ? (
                          <>
                            {" "}
                            <span className="prov-tag">
                              [{titleCaseSlug(topic)}]
                            </span>
                          </>
                        ) : null}
                        {mRole ? (
                          <>
                            {" "}
                            <span className="prov-tag">({mRole})</span>
                          </>
                        ) : null}
                      </>
                    }
                    navTarget={navTarget}
                    onNavigate={onNavigate}
                    rowTitle={mPct != null ? `${mPct}% extraction confidence` : "live mention"}
                  />
                </li>
              );
            })}
            {mentions.length > REF_LIMIT && !showAllRefs ? (
              <li className="prov-item">
                <button
                  type="button"
                  className="md-show-more"
                  onClick={() => setShowAllRefs(true)}
                >
                  +{mentions.length - REF_LIMIT} more
                </button>
              </li>
            ) : null}
          </ul>
        </>
      ) : null}
    </article>
  );
}

// ── PatternsView ───────────────────────────────────────────────────────────

export function PatternsView({ runId, topic, refreshTick, onNavigate }) {
  const [patterns, setPatterns] = useState([]);
  const [factsForTopic, setFactsForTopic] = useState([]);
  const [insights, setInsights] = useState(null);

  useEffect(() => {
    let cancelled = false;
    invoke("read_run_patterns_for_topic", { runId, topic })
      .then((r) => { if (!cancelled) setPatterns(Array.isArray(r) ? r : []); })
      .catch(() => { if (!cancelled) setPatterns([]); });
    invoke("read_run_facts_for_topic", { runId, topic })
      .then((r) => { if (!cancelled) setFactsForTopic(Array.isArray(r) ? r : []); })
      .catch(() => { if (!cancelled) setFactsForTopic([]); });
    invoke("read_run_insights", { runId })
      .then((r) => { if (!cancelled) setInsights(r || null); })
      .catch(() => { if (!cancelled) setInsights(null); });
    return () => { cancelled = true; };
  }, [runId, topic, refreshTick]);

  const patternAnchors = useMemo(() => computePatternAnchors(patterns), [patterns]);
  const citedByMap = useMemo(() => buildPatternsCitedBy(insights), [insights]);
  const factsAll = useMemo(() => ({ [topic]: factsForTopic }), [topic, factsForTopic]);
  const factAnchorsByTopic = useMemo(() => ({
    [topic]: computeFactAnchors(factsForTopic),
  }), [topic, factsForTopic]);

  // Page-level collapse state.
  const { collapsedSlugs, toggleSection, collapseAll, expandAll } = useCollapseState();
  const provKeys = patterns.map((_, idx) => `prov-pattern-${idx}`);

  return (
    <article className="md-view">
      {provKeys.length >= 2 && (
        <PageToolbar
          collapseAll={() => collapseAll(provKeys)}
          expandAll={expandAll}
        />
      )}
      <h1 className="md-h md-h1">{titleCaseSlug(topic)} — patterns ({patterns.length})</h1>
      {patterns.map((pat, idx) => {
        const anchor = patternAnchors.get(idx);
        const kind = pat.kind || "pattern";
        const meta = `${kind} · count ${pat.count || 1} · domain ${topic}`;
        const cited = citedByMap[`${topic}#${idx}`] || [];
        const collapseKey = `prov-pattern-${idx}`;

        // Build the children (per-source-fact subtree) + accumulate
        // leaves so the inline shares pills are accurate per pattern.
        // Sort by aggregated composite (edge × extraction) so row order
        // matches the confidence dots — last-hop edge alone inverts them.
        const sortedFacts = sortSourceFactsByComposite(
          pat.source_facts,
          factsForTopic,
        );
        const leavesOut = [];
        const childItems = sortedFacts
          .map(([fIdx, edgeConf]) => {
            const fact = factsForTopic[fIdx];
            if (!fact) return null;
            const fAnchor =
              factAnchorsByTopic[topic]?.get(fIdx) || `fact-${fIdx + 1}`;
            return emitFactSubtree({
              runId,
              topic,
              factIdx: fIdx,
              fact,
              factAnchor: fAnchor,
              parentPathConf: edgeConf,
              factsAll,
              factAnchorsByTopic,
              onNavigate,
              leavesOut,
            });
          })
          .filter(Boolean);

        return (
          <Section key={idx} anchor={anchor} displayNum={idx + 1} title={pat.name} meta={meta}>
            {pat.description ? <p className="md-p">{pat.description}</p> : null}
            {childItems.length > 0 && (
              <Provenance
                collapseKey={collapseKey}
                collapsedSlugs={collapsedSlugs}
                toggleSection={toggleSection}
                leaves={leavesOut}
                onNavigate={onNavigate}
              >
                <ul className="prov-list prov-depth-0">{childItems}</ul>
              </Provenance>
            )}
            {cited.length > 0 ? (
              <>
                <p className="md-p"><strong>Cited by:</strong></p>
                <ul className="prov-list">
                  {cited.map(([kind, _topic, iIdx, edgeConf], i) => {
                    const a = insightAnchor(kind, iIdx);
                    const insList =
                      kind === "cross_domain"
                        ? insights?.cross_domain
                        : insights?.critical;
                    const ins = insList?.[iIdx];
                    if (!ins) return null;
                    const pct = Math.round(edgeConf * 100);
                    const navTarget = {
                      runId,
                      relPath: `${STAGE_DIRS[4]}.md`,
                      anchor: a,
                    };
                    const typeTag = `${titleCaseSlug(
                      kind === "cross_domain" ? "cross-domain" : kind
                    )} Insight`;
                    return (
                      <li key={i} className="prov-item prov-item-root">
                        <ProvRowRoot
                          pct={pct}
                          summary={ins.name || ins.description || "(unnamed insight)"}
                          typeTag={typeTag}
                          navTarget={navTarget}
                          onNavigate={onNavigate}
                          rowTitle={`${pct}% match`}
                        />
                      </li>
                    );
                  })}
                </ul>
              </>
            ) : null}
            {pat.hallucinated_ref_count ? (
              <p className="md-warn">
                <em>⚠ {pat.hallucinated_ref_count} hallucinated refs</em>
              </p>
            ) : null}
          </Section>
        );
      })}
      {patterns.length === 0 ? (
        <p className="md-empty">
          <em>(No patterns yet — patterns stage in progress or topic produced none.)</em>
        </p>
      ) : null}
    </article>
  );
}

// ── InsightsView ───────────────────────────────────────────────────────────

export function InsightsView({ runId, refreshTick, onNavigate }) {
  const [insights, setInsights] = useState(null);
  const [patternsAll, setPatternsAll] = useState({});
  const [factsAll, setFactsAll] = useState({});
  const [actions, setActions] = useState(null);

  useEffect(() => {
    let cancelled = false;
    invoke("read_run_insights", { runId })
      .then((r) => { if (!cancelled) setInsights(r || null); })
      .catch(() => { if (!cancelled) setInsights(null); });
    invoke("read_run_patterns_all", { runId })
      .then((r) => { if (!cancelled) setPatternsAll(r || {}); })
      .catch(() => { if (!cancelled) setPatternsAll({}); });
    invoke("read_run_facts_all", { runId })
      .then((r) => { if (!cancelled) setFactsAll(r || {}); })
      .catch(() => { if (!cancelled) setFactsAll({}); });
    invoke("read_run_actions", { runId })
      .then((r) => { if (!cancelled) setActions(r || null); })
      .catch(() => { if (!cancelled) setActions(null); });
    return () => { cancelled = true; };
  }, [runId, refreshTick]);

  const cross = insights?.cross_domain || [];
  const crit = insights?.critical || [];
  const backEdges = useMemo(
    () => buildInsightsBackEdges(actions?.actions || []),
    [actions]
  );

  const patternAnchorsByTopic = useMemo(() => {
    const out = {};
    Object.entries(patternsAll).forEach(([t, pats]) => {
      out[t] = computePatternAnchors(pats || []);
    });
    return out;
  }, [patternsAll]);
  const factAnchorsByTopic = useMemo(() => {
    const out = {};
    Object.entries(factsAll).forEach(([t, facts]) => {
      out[t] = computeFactAnchors(facts || []);
    });
    return out;
  }, [factsAll]);

  const { collapsedSlugs, toggleSection, collapseAll, expandAll } = useCollapseState();
  const provKeys = [
    ...cross.map((_, idx) => `prov-cross-${idx}`),
    ...crit.map((_, idx) => `prov-crit-${idx}`),
  ];

  const renderInsightSection = (ins, kind, idx) => {
    const anchor = insightAnchor(kind, idx);
    // Visible number is CONTINUOUS across cross-domain → critical (critical
    // picks up where cross-domain left off), matching the actions-prompt
    // `[N]` enumeration, the action-body insight refs, and the run tree's
    // running scan-index. The anchor stays per-scope (`critical-1`) as a
    // stable navigation id — only the displayed count is continuous.
    const displayNum = (kind === "cross_domain" ? idx : cross.length + idx) + 1;
    const collapseKey =
      kind === "cross_domain" ? `prov-cross-${idx}` : `prov-crit-${idx}`;
    const domains = (ins.domains || []).join(", ") || "—";
    const meta = `${
      kind === "cross_domain" ? "cross-domain" : kind
    } · domains: ${domains}`;
    const leadsTo = backEdges[`${kind}#${idx}`] || [];

    // Build the per-source-pattern subtree once, accumulate leaves.
    const sortedPats = [...(ins.source_patterns || [])].sort(
      (a, b) => b[2] - a[2]
    );
    const leavesOut = [];
    const childItems = sortedPats
      .map(([topic, pIdx, edgeConf]) => {
        const topicPats = patternsAll?.[topic] || [];
        const pat = topicPats[pIdx];
        if (!pat) return null;
        const pAnchor =
          patternAnchorsByTopic?.[topic]?.get(pIdx) || `pattern-${pIdx + 1}`;
        return emitPatternSubtree({
          runId,
          topic,
          patIdx: pIdx,
          pat,
          patAnchor: pAnchor,
          parentPathConf: edgeConf,
          factsAll,
          factAnchorsByTopic,
          onNavigate,
          leavesOut,
        });
      })
      .filter(Boolean);

    return (
      <Section key={`${kind}-${idx}`} anchor={anchor} level={3} displayNum={displayNum} title={ins.name} meta={meta}>
        {ins.description ? <p className="md-p">{ins.description}</p> : null}
        {ins.mechanism ? (
          <p className="md-p"><strong>Mechanism:</strong> {ins.mechanism}</p>
        ) : null}
        {ins.implication ? (
          <p className="md-p"><strong>Implication:</strong> {ins.implication}</p>
        ) : null}
        {(ins.proposed_actions || []).length > 0 ? (
          <>
            <p className="md-p"><strong>Proposed actions:</strong></p>
            <ul>{ins.proposed_actions.map((a, i) => <li key={i}>{a}</li>)}</ul>
          </>
        ) : null}
        {childItems.length > 0 && (
          <Provenance
            collapseKey={collapseKey}
            collapsedSlugs={collapsedSlugs}
            toggleSection={toggleSection}
            leaves={leavesOut}
            onNavigate={onNavigate}
          >
            <ul className="prov-list prov-depth-0">{childItems}</ul>
          </Provenance>
        )}
        {leadsTo.length > 0 ? (
          <>
            <p className="md-p"><strong>Cited by:</strong></p>
            <ul className="prov-list">
              {leadsTo.map(([aIdx, edgeConf], i) => {
                const a = actionAnchor(aIdx);
                const action = actions?.actions?.[aIdx];
                if (!action) return null;
                const pct = Math.round(edgeConf * 100);
                const navTarget = {
                  runId,
                  relPath: `${STAGE_DIRS[5]}.md`,
                  anchor: a,
                };
                const typeTag = `${titleCaseSlug(action.kind || "action")} Action`;
                return (
                  <li key={i} className="prov-item prov-item-root">
                    <ProvRowRoot
                      pct={pct}
                      summary={action.recommendation || "(unnamed action)"}
                      typeTag={typeTag}
                      navTarget={navTarget}
                      onNavigate={onNavigate}
                      rowTitle={`${pct}% match`}
                    />
                  </li>
                );
              })}
            </ul>
          </>
        ) : null}
        {ins.hallucinated_ref_count ? (
          <p className="md-warn">
            <em>⚠ {ins.hallucinated_ref_count} hallucinated refs</em>
          </p>
        ) : null}
      </Section>
    );
  };

  if (!insights) {
    return (
      <article className="md-view">
        <h1 className="md-h md-h1">Insights</h1>
        <p className="md-empty"><em>(Insights stage hasn&apos;t run yet.)</em></p>
      </article>
    );
  }

  return (
    <article className="md-view">
      {provKeys.length >= 2 && (
        <PageToolbar
          collapseAll={() => collapseAll(provKeys)}
          expandAll={expandAll}
        />
      )}
      <h1 className="md-h md-h1">Insights ({cross.length + crit.length})</h1>
      <p className="md-p">
        <em>
          Higher-level patterns synthesized across topics.{" "}
          <strong>{cross.length}</strong> cross-domain,{" "}
          <strong>{crit.length}</strong> critical.
        </em>
      </p>
      {cross.length > 0 ? (
        <>
          <h2 className="md-h md-h2" id="cross-domain">Cross-domain</h2>
          {cross.map((ins, idx) => renderInsightSection(ins, "cross_domain", idx))}
        </>
      ) : null}
      {crit.length > 0 ? (
        <>
          <h2 className="md-h md-h2" id="critical">Critical</h2>
          {crit.map((ins, idx) => renderInsightSection(ins, "critical", idx))}
        </>
      ) : null}
      {cross.length === 0 && crit.length === 0 ? (
        <p className="md-empty"><em>(no insights detected)</em></p>
      ) : null}
    </article>
  );
}

// ── ActionsView ────────────────────────────────────────────────────────────

export function ActionsView({ runId, refreshTick, onNavigate }) {
  const [actionsPayload, setActionsPayload] = useState(null);
  const [insights, setInsights] = useState(null);
  const [patternsAll, setPatternsAll] = useState({});
  const [factsAll, setFactsAll] = useState({});

  useEffect(() => {
    let cancelled = false;
    invoke("read_run_actions", { runId })
      .then((r) => { if (!cancelled) setActionsPayload(r || null); })
      .catch(() => { if (!cancelled) setActionsPayload(null); });
    invoke("read_run_insights", { runId })
      .then((r) => { if (!cancelled) setInsights(r || null); })
      .catch(() => { if (!cancelled) setInsights(null); });
    invoke("read_run_patterns_all", { runId })
      .then((r) => { if (!cancelled) setPatternsAll(r || {}); })
      .catch(() => { if (!cancelled) setPatternsAll({}); });
    invoke("read_run_facts_all", { runId })
      .then((r) => { if (!cancelled) setFactsAll(r || {}); })
      .catch(() => { if (!cancelled) setFactsAll({}); });
    return () => { cancelled = true; };
  }, [runId, refreshTick]);

  const actions = actionsPayload?.actions || [];

  const patternAnchorsByTopic = useMemo(() => {
    const out = {};
    Object.entries(patternsAll).forEach(([t, pats]) => {
      out[t] = computePatternAnchors(pats || []);
    });
    return out;
  }, [patternsAll]);
  const factAnchorsByTopic = useMemo(() => {
    const out = {};
    Object.entries(factsAll).forEach(([t, facts]) => {
      out[t] = computeFactAnchors(facts || []);
    });
    return out;
  }, [factsAll]);

  const { collapsedSlugs, toggleSection, collapseAll, expandAll } = useCollapseState();
  const provKeys = actions.map((_, idx) => `prov-action-${idx}`);

  const renderActionSection = (action, idx) => {
    const anchor = actionAnchor(idx);
    const horizon = action.horizon || "—";
    const score = (action.score ?? 0).toFixed(2);
    const conf = (typeof action.confidence === "number" ? action.confidence : 1).toFixed(2);
    const meta = `${horizon} horizon · score ${score} · confidence ${conf}${
      action.review_date ? ` · review ${action.review_date}` : ""
    }`;
    const collapseKey = `prov-action-${idx}`;
    const refCtx = { insights, runId, onNavigate };

    const sortedSources = [...(action.source_insights || [])].sort(
      (a, b) => b[2] - a[2]
    );
    const leavesOut = [];
    const childItems = sortedSources
      .map(([kind, iIdx, edgeConf]) => {
        const insList =
          kind === "cross_domain" ? insights?.cross_domain : insights?.critical;
        const ins = insList?.[iIdx];
        if (!ins) return null;
        return emitInsightSubtree({
          runId,
          kind,
          iIdx,
          ins,
          parentPathConf: edgeConf,
          patternsAll,
          patternAnchorsByTopic,
          factsAll,
          factAnchorsByTopic,
          onNavigate,
          leavesOut,
        });
      })
      .filter(Boolean);

    return (
      <Section key={idx} anchor={anchor} displayNum={idx + 1} title={action.recommendation} meta={meta}>
        {action.objective ? (
          <p className="md-p"><strong>Objective:</strong> {linkifyInsightRefs(action.objective, refCtx)}</p>
        ) : null}
        {action.why ? (
          <p className="md-p"><strong>Why:</strong> {linkifyInsightRefs(action.why, refCtx)}</p>
        ) : null}
        {action.immediate_action ? (
          <p className="md-p">
            <strong>Immediate action (7d):</strong> {linkifyInsightRefs(action.immediate_action, refCtx)}
          </p>
        ) : null}
        {action.habit ? (
          <p className="md-p"><strong>Habit:</strong> {linkifyInsightRefs(action.habit, refCtx)}</p>
        ) : null}
        {action.success_metric ? (
          <p className="md-p">
            <strong>Success metric:</strong> {linkifyInsightRefs(action.success_metric, refCtx)}
          </p>
        ) : null}
        <p className="md-p">
          <strong>Scores:</strong>{" "}
          {action.kind ? <>{action.kind} · </> : null}
          regret {(action.regret_reduction ?? 0).toFixed(2)} · leverage{" "}
          {(action.leverage ?? 0).toFixed(2)} · consequence{" "}
          {(action.consequence ?? 0).toFixed(2)} · generativity{" "}
          {(action.generativity ?? 0).toFixed(2)} · decisiveness{" "}
          {(action.decisiveness ?? 0).toFixed(2)} · feedback-speed{" "}
          {(action.time_to_feedback ?? 0).toFixed(2)} · constraint-fit{" "}
          {(action.constraint_fit ?? 0).toFixed(2)}
        </p>
        {childItems.length > 0 && (
          <Provenance
            collapseKey={collapseKey}
            collapsedSlugs={collapsedSlugs}
            toggleSection={toggleSection}
            leaves={leavesOut}
            onNavigate={onNavigate}
          >
            <ul className="prov-list prov-depth-0">{childItems}</ul>
          </Provenance>
        )}
        {action.hallucinated_ref_count ? (
          <p className="md-warn">
            <em>⚠ {action.hallucinated_ref_count} hallucinated refs</em>
          </p>
        ) : null}
      </Section>
    );
  };

  return (
    <article className="md-view">
      {provKeys.length >= 2 && (
        <PageToolbar
          collapseAll={() => collapseAll(provKeys)}
          expandAll={expandAll}
        />
      )}
      <h1 className="md-h md-h1">Actions ({actions.length})</h1>
      <p className="md-p">
        <em>
          Prioritized from the proposed actions across all insights.{" "}
          <code>review_date</code> is the next check-in, not a completion deadline.
        </em>
      </p>
      {actions.length === 0 ? (
        <p className="md-empty"><em>(no actions generated)</em></p>
      ) : (
        actions.map((a, idx) => renderActionSection(a, idx))
      )}
    </article>
  );
}
