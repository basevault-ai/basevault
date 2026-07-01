/**
 * Pure Obsidian-flavored markdown renderer for completed pipeline runs.
 *
 * `exportRun(payload)` returns `{ [relPath]: string }` — a manifest of
 * file paths and their full contents. The caller writes them to disk
 * (or anywhere else). No fs / Tauri side effects here.
 *
 * Output is byte-identical to vault_exporter.py's `VaultExporter.export_run`
 * given the JSON-serialized equivalents of the same dataclass payloads;
 * the goldens used to pin parity live under
 * `engine/tests/fixtures/obsidian_renderer/<size>/`.
 *
 * JSON shape contract — what each `read_run_*` Tauri command returns:
 *   facts (per topic): array of {type, summary, occurred_at, occurred_at_text,
 *     entities[], evidence[{text, ref, start_char, end_char, file_path,
 *     file_offset, file_length, approximate}], topics[], tags[],
 *     confidence, relation_candidate}
 *   entities aggregate: {subject:{canonical_id,display,source}, entities:[...],
 *     relations:[{from, to, relation, confidence, evidence_fact_refs}]}
 *   patterns (per topic): array of {name, description, kind, count,
 *     source_facts:[[idx, conf]], hallucinated_ref_count}
 *   insights: {cross_domain:[...], critical:[...]} where each insight is
 *     {name, description, mechanism, implication, domains, proposed_actions,
 *     source_patterns:[[topic, idx, conf]], hallucinated_ref_count}
 *   actions: {actions:[{recommendation, kind, objective, why, immediate_action,
 *     habit, success_metric, horizon, review_date, regret_reduction, leverage,
 *     consequence, generativity, decisiveness, time_to_feedback,
 *     constraint_fit, confidence, score, source_insights:[[kind, idx, conf]],
 *     hallucinated_ref_count}]}
 */

const STAGE_DIRS = { 2: "1-facts", 3: "2-entities", 4: "3-patterns" };

// Privacy slider order (left → right = increasingly raw / less private).
// `privacyLevel` names the RIGHTMOST included level; the cut is the
// contiguous prefix `idx(level) <= idx(privacyLevel)`. `"raw"` (the
// default) includes everything, keeping the no-arg render byte-identical
// to the committed goldens. The provenance graph is a chain
// actions→insights→patterns→facts→raw with entities as a lateral index
// off facts, so at any cut exactly one level is the first-excluded one —
// "the immediately-next excluded level" a boundary count refers to.
const PRIVACY_LEVELS = ["actions", "insights", "patterns", "entities", "facts", "raw"];
const PRIVACY_IDX = Object.fromEntries(PRIVACY_LEVELS.map((l, i) => [l, i]));

function plural(n, noun) {
  return `${n} ${noun}${n === 1 ? "" : "s"}`;
}

// ── Anchors ────────────────────────────────────────────────────────────────

// Type-scoped 1-indexed anchor for a fact in its topic file. Untyped → fact-N.
export function computeFactAnchorsForTopic(facts) {
  const out = [];
  const counters = new Map();
  for (let i = 0; i < facts.length; i++) {
    const t = facts[i]?.type || facts[i]?.item_type || "fact";
    counters.set(t, (counters.get(t) || 0) + 1);
    out.push(`${t}-${counters.get(t)}`);
  }
  return out;
}

function computeFactAnchors(factsByTopic) {
  const out = {};
  for (const [topic, facts] of Object.entries(factsByTopic || {})) {
    out[topic] = computeFactAnchorsForTopic(facts || []);
  }
  return out;
}

export function computePatternAnchorsForTopic(patterns) {
  const out = [];
  const counters = new Map();
  for (let i = 0; i < patterns.length; i++) {
    const k = (patterns[i]?.kind || "pattern").toLowerCase().replace(/\s+/g, "-");
    counters.set(k, (counters.get(k) || 0) + 1);
    out.push(`${k}-${counters.get(k)}`);
  }
  return out;
}

function computePatternAnchors(patternsByTopic) {
  const out = {};
  for (const [topic, pats] of Object.entries(patternsByTopic || {})) {
    out[topic] = computePatternAnchorsForTopic(pats || []);
  }
  return out;
}

export function insightAnchor(kind, idx) {
  const k = kind === "cross_domain" ? "cross-domain" : kind;
  return `${k}-${idx + 1}`;
}

export function actionAnchor(idx) {
  return `action-${idx + 1}`;
}

// ── Cited-by / back-edge maps ─────────────────────────────────────────────

function buildFactsCitedBy(patternsByTopic) {
  const out = new Map();
  for (const [topic, pats] of Object.entries(patternsByTopic || {})) {
    for (let pIdx = 0; pIdx < (pats || []).length; pIdx++) {
      const sf = pats[pIdx]?.source_facts || [];
      for (const [fIdx, conf] of sf) {
        const key = `${topic} ${fIdx}`;
        if (!out.has(key)) out.set(key, []);
        out.get(key).push([topic, pIdx, conf]);
      }
    }
  }
  for (const list of out.values()) {
    list.sort((a, b) => (b[2] - a[2]) || a[0].localeCompare(b[0]) || (a[1] - b[1]));
  }
  return out;
}

function buildPatternsCitedBy(insightOutput) {
  const out = new Map();
  const groups = [
    ["cross_domain", insightOutput?.cross_domain || []],
    ["critical", insightOutput?.critical || []],
  ];
  for (const [kind, list] of groups) {
    for (let iIdx = 0; iIdx < list.length; iIdx++) {
      for (const [topic, pIdx, conf] of list[iIdx].source_patterns || []) {
        const key = `${topic} ${pIdx}`;
        if (!out.has(key)) out.set(key, []);
        out.get(key).push([kind, topic, iIdx, conf]);
      }
    }
  }
  for (const list of out.values()) {
    list.sort((a, b) => (b[3] - a[3]) || a[0].localeCompare(b[0]) || (a[2] - b[2]));
  }
  return out;
}

function buildInsightsBackEdges(actionsList) {
  const out = new Map();
  for (let aIdx = 0; aIdx < (actionsList || []).length; aIdx++) {
    for (const [kind, iIdx, conf] of actionsList[aIdx].source_insights || []) {
      const key = `${kind} ${iIdx}`;
      if (!out.has(key)) out.set(key, []);
      out.get(key).push([aIdx, conf]);
    }
  }
  for (const list of out.values()) {
    list.sort((a, b) => (b[1] - a[1]) || (a[0] - b[0]));
  }
  return out;
}

// ── Tiny string helpers ───────────────────────────────────────────────────

// HTML-like tags break Obsidian rendering — wrap `<foo>` / `<name:value>` /
// `<tag attr="v">` (when not already inline-coded) in backticks. Mirrors
// vault_exporter._escape_tags. Negative look-around can't be used here
// because not all engines support it, so we strip it manually.
const TAG_RE = /<(\/?[A-Za-z][A-Za-z0-9]*(?::[A-Za-z0-9_.+-]+)?(?:\s[^<>`]*)?)>/g;
function escapeTags(text) {
  if (!text) return text;
  let out = "";
  let i = 0;
  while (i < text.length) {
    TAG_RE.lastIndex = i;
    const m = TAG_RE.exec(text);
    if (!m) {
      out += text.slice(i);
      break;
    }
    const start = m.index;
    const end = start + m[0].length;
    const before = start > 0 ? text[start - 1] : "";
    const after = end < text.length ? text[end] : "";
    out += text.slice(i, start);
    if (before === "`" || after === "`") {
      out += m[0];
    } else {
      out += `\`<${m[1]}>\``;
    }
    i = end;
  }
  return out;
}

// Mirror of vault_exporter._escape_quote_for_bullet. Backslashes first
// (so an already-escaped `\"` doesn't double-escape), then literal `"`.
function escapeQuoteForBullet(text) {
  return text.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

// Crude singularizer — "relationships" → "Relationship", "health" → "Health".
// Matches vault_exporter._singularize_topic.
function singularizeTopic(topic) {
  let t = pyTitle(topic);
  if (t.endsWith("s") && !t.endsWith("ss")) t = t.slice(0, -1);
  return t;
}

// Python-flavored str.title(): word boundary == any non-alphanumeric.
// Letters at the start of each alpha-run get upper-cased; rest lowercased.
// Matches `relationships` → `Relationships`, `cross-domain` → `Cross-Domain`.
function pyTitle(text) {
  if (!text) return "";
  return text.replace(/[a-zA-Z]+/g, (w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase());
}

// `_prov_display(topic, anchor, stage)` — matches Python.
// `(relationships, fear-2)` → `Relationship Fear 2`.
// `(None, critical-1, 'insight')` → `Insight Critical 1`.
// `(None, action-1, 'action')` → `Action 1`.
const ANCHOR_RE = /^(.+)-(\d+)$/;
function provDisplay(topic, anchor, stage = null) {
  const m = ANCHOR_RE.exec(anchor);
  if (!m) return anchor;
  const typeRaw = m[1];
  const n = m[2];
  // Python: type_raw.replace("-", " ").replace("_", " ").title().
  const typeTitle = pyTitle(typeRaw.replace(/-/g, " ").replace(/_/g, " "));
  if (topic == null) {
    if (stage) return `${pyTitle(stage)} ${typeTitle} ${n}`;
    return `${typeTitle} ${n}`;
  }
  return `${singularizeTopic(topic)} ${typeTitle} ${n}`;
}

// Build a wikilink. relTarget = `1-facts/relationships#^emotion-3` etc.
function link(relTarget, display) {
  if (display == null) return `[[${relTarget}]]`;
  return `[[${relTarget}|${display}]]`;
}

// Python's `int(round(x))` uses banker's rounding (round-half-to-even).
// Math.round in JS rounds halves toward +infinity. Banker's only kicks
// in at an *exact* midpoint — e.g. 76.5 (exactly representable as
// 153/2) ties to 76, but 58.50000000000001 (the float repr of
// 0.9 * 0.65 * 100) is strictly > 58.5 and rounds to 59. Tolerance-based
// tie detection over-fires on the latter.
function formatPct(conf) {
  const x = conf * 100;
  const sign = x < 0 ? -1 : 1;
  const abs = Math.abs(x);
  const floor = Math.floor(abs);
  const diff = abs - floor;
  let rounded;
  if (diff > 0.5) rounded = floor + 1;
  else if (diff < 0.5) rounded = floor;
  else rounded = (floor % 2 === 0) ? floor : floor + 1;
  return `${sign * rounded}`;
}

// Mirror of Python's `f"{x:.2f}"`. Python rounds half-to-even on the
// binary float; JS `toFixed` is round-half-away-from-zero per spec. The
// exact-midpoint case is rare (most 0.05-multiple inputs aren't exactly
// representable so neither path ties), but the score property's
// 0.075-coefficient sums can produce true ties.
function fmt2(x) {
  if (x == null || Number.isNaN(x)) return "0.00";
  const sign = x < 0 ? -1 : 1;
  const abs = Math.abs(x);
  const scaled = abs * 100;
  const floor = Math.floor(scaled);
  const diff = scaled - floor;
  let rounded;
  if (diff > 0.5) rounded = floor + 1;
  else if (diff < 0.5) rounded = floor;
  else rounded = (floor % 2 === 0) ? floor : floor + 1;
  return (sign * rounded / 100).toFixed(2);
}

// One bullet in a provenance graph tree.
function graphLine(indent, linkText, pathConf, snippet, approximate = false) {
  const pad = "  ".repeat(indent);
  const marker = approximate ? "≈ " : "";
  const pct = formatPct(pathConf);
  if (snippet) {
    let quote = escapeTags(snippet.trim().replace(/\n/g, " "));
    quote = escapeQuoteForBullet(quote);
    return `${pad}- **${pct}% match**  ${marker}${linkText}: *"${quote}"*`;
  }
  return `${pad}- **${pct}% match**  ${marker}${linkText}`;
}

// ── Renderer state ────────────────────────────────────────────────────────

class Renderer {
  constructor({
    factsByTopic = {},
    entitiesPayload = null,
    patternsByTopic = {},
    insightsPayload = null,
    actionsList = [],
    stats = {},
    runName = "",
    inputs = null,  // optional preprocessed input file map { relPath: text }
    privacyLevel = "raw",
  } = {}) {
    const cut = PRIVACY_IDX[privacyLevel] ?? PRIVACY_IDX.raw;
    this.privacyLevel = privacyLevel;
    // True iff `level`'s files are exported under this cut. Cross-stage
    // links/walks route through this so a reference into an excluded
    // level collapses to a provenance count instead of dangling.
    this.inc = (level) => PRIVACY_IDX[level] <= cut;
    this.factsByTopic = factsByTopic;
    this.entitiesPayload = entitiesPayload;
    this.patternsByTopic = patternsByTopic;
    this.insightsPayload = insightsPayload || { cross_domain: [], critical: [] };
    this.actionsList = actionsList;
    this.stats = stats;
    this.runName = runName;
    this.inputs = inputs || {};

    this.factAnchorsByTopic = computeFactAnchors(factsByTopic);
    this.patternAnchorsByTopic = computePatternAnchors(patternsByTopic);
    this.factsCitedBy = buildFactsCitedBy(patternsByTopic);
    this.patternsCitedBy = buildPatternsCitedBy(this.insightsPayload);
    this.insightsBackEdges = buildInsightsBackEdges(actionsList);
  }

  prefix() {
    return this.runName ? `${this.runName}/` : "";
  }

  topicTarget(stage, topic, anchor = "") {
    let t = `${this.prefix()}${STAGE_DIRS[stage]}/${topic}`;
    if (anchor) t += `#${anchor}`;
    return t;
  }

  inputTarget(filePath, anchor = "") {
    let t = `${this.prefix()}0-inputs/${filePath}`;
    if (anchor) t += `#${anchor}`;
    return t;
  }

  entityTarget(canonicalId, anchor = "") {
    let t = `${this.prefix()}2-entities/${canonicalId}`;
    if (anchor) t += `#${anchor}`;
    return t;
  }

  insightsTarget(anchor = "") {
    let t = `${this.prefix()}4-insights`;
    if (anchor) t += `#${anchor}`;
    return t;
  }

  actionsTarget(anchor = "") {
    let t = `${this.prefix()}5-actions`;
    if (anchor) t += `#${anchor}`;
    return t;
  }

  factAnchor(topic, idx) {
    const list = this.factAnchorsByTopic[topic];
    if (!list) return `fact-${idx + 1}`;
    return list[idx] || `fact-${idx + 1}`;
  }

  patternAnchor(topic, idx) {
    const list = this.patternAnchorsByTopic[topic];
    if (!list) return `pattern-${idx + 1}`;
    return list[idx] || `pattern-${idx + 1}`;
  }

  // ── Facts page ─────────────────────────────────────────────────────────
  renderFactsTopic(topic, facts, citedByMap) {
    const lines = [
      "---",
      `topic: ${topic}`,
      `layer: facts`,
      `count: ${facts.length}`,
      "---",
      "",
      `# ${pyTitle(topic)} — facts`,
      "",
    ];
    for (let idx = 0; idx < facts.length; idx++) {
      const fact = facts[idx];
      const cited = citedByMap.get(idx) || [];
      this._appendFactSection(lines, topic, idx, fact, cited);
      lines.push("");
    }
    return lines.join("\n");
  }

  _appendFactSection(lines, topic, idx, fact, citedBy) {
    const anchor = this.factAnchor(topic, idx);
    lines.push(`## [${anchor}] ${fact.summary}`);
    const occurred = fact.occurred_at || "undated";
    const itemType = fact.type || fact.item_type || "fact";
    const conf = typeof fact.confidence === "number" ? fact.confidence : 1.0;
    lines.push(`*${occurred} · ${itemType} · confidence ${fmt2(conf)}* ^${anchor}`);
    lines.push("");
    if ((fact.entities || []).length > 0) {
      const ents = fact.entities.map((r) => {
        const name = r.entity?.name || r.name || "";
        const t = r.entity?.entity_type || r.type || "";
        const role = r.role || "";
        return `${name} (${t}/${role})`;
      }).join(", ");
      lines.push(`**Entities:** ${ents}`);
    }
    if ((fact.topics || []).length > 0) {
      lines.push(`**Topics:** ${fact.topics.map((t) => `\`${t}\``).join(", ")}`);
    }
    if ((fact.tags || []).length > 0) {
      lines.push(`**Tags:** ${fact.tags.join(", ")}`);
    }
    lines.push("");
    lines.push("**Source:**");
    if (!this.inc("raw")) {
      // The evidence quote is a verbatim excerpt of a raw document, so
      // it crosses the same boundary the source file does — collapse the
      // whole Source block to a count rather than leaking the quotes.
      const n = (fact.evidence || []).length;
      lines.push(`- *↳ ${plural(n, "source excerpt")} below the privacy cut (not exported)*`);
    } else {
      for (const ev of fact.evidence || []) {
        const quote = escapeTags((ev.text || "").trim().replace(/\n/g, " "));
        const prefix = ev.approximate ? "≈ " : "";
        if (ev.file_path) {
          let target, display;
          if (ev.file_offset != null && ev.file_length != null) {
            target = this.inputTarget(ev.file_path, `^offset-${ev.file_offset}`);
            display = `${ev.file_path} @ ${ev.file_offset}:${ev.file_length}`;
          } else {
            target = this.inputTarget(ev.file_path);
            display = ev.file_path;
          }
          lines.push(`- ${prefix}${link(target, display)}`);
          lines.push(`  > ${quote}`);
        } else {
          lines.push(`- ${prefix}> ${quote}`);
        }
      }
    }
    if (citedBy.length > 0) {
      lines.push("");
      lines.push("**Cited by (patterns):**");
      for (const [t, pIdx, conf] of citedBy) {
        const a = this.patternAnchor(t, pIdx);
        const target = this.topicTarget(4, t, `^${a}`);
        lines.push(`- ${link(target, `patterns/${t}/${a}`)} · confidence ${fmt2(conf)}`);
      }
    }
    lines.push("");
  }

  // ── Patterns page ──────────────────────────────────────────────────────
  renderPatternsTopic(topic, pats, topicFacts, citedByMap) {
    const lines = [
      "---",
      `topic: ${topic}`,
      `layer: patterns`,
      `count: ${pats.length}`,
      "---",
      "",
      `# ${pyTitle(topic)} — patterns`,
      "",
    ];
    for (let idx = 0; idx < pats.length; idx++) {
      const pat = pats[idx];
      const cited = citedByMap.get(idx) || [];
      this._appendPatternSection(lines, topic, idx, pat, topicFacts, cited);
      lines.push("");
    }
    return lines.join("\n");
  }

  _appendPatternSection(lines, topic, idx, pat, topicFacts, citedBy) {
    const anchor = this.patternAnchor(topic, idx);
    lines.push(`## [${anchor}] ${pat.name}`);
    const kindStr = pat.kind || "pattern";
    // Python: f"*{kind_str} · count {pat.count} · domain {pat.domain}*"
    // — but pat.domain is missing from JSON. The runner-side path always
    //   uses domain == topic (it's how patterns are bucketed), and we
    //   key the bucket by topic anyway, so use topic here for parity
    //   with the dataclass-fed Python invocation.
    const domain = pat.domain || topic;
    lines.push(`*${kindStr} · count ${pat.count} · domain ${domain}* ^${anchor}`);
    lines.push("");
    lines.push(pat.description || "");
    if ((pat.source_facts || []).length > 0) {
      lines.push("");
      lines.push("**Provenance:**");
      lines.push(...this._walkFactsForPattern(topic, pat, 1.0, 0));
    }
    if (citedBy.length > 0) {
      lines.push("");
      lines.push("**Cited by (insights):**");
      for (const [kind, _t, iIdx, conf] of citedBy) {
        const a = insightAnchor(kind, iIdx);
        lines.push(`- ${link(this.insightsTarget(`^${a}`), `insights/${a}`)} · confidence ${fmt2(conf)}`);
      }
    }
    if (pat.hallucinated_ref_count) {
      lines.push("");
      lines.push(`*⚠ ${pat.hallucinated_ref_count} hallucinated refs*`);
    }
    lines.push("");
  }

  // ── Insights page ──────────────────────────────────────────────────────
  renderInsightsFile() {
    const ins = this.insightsPayload || { cross_domain: [], critical: [] };
    const cross = ins.cross_domain || [];
    const crit = ins.critical || [];
    const nCross = cross.length;
    const nCrit = crit.length;
    const nTopics = Object.values(this.patternsByTopic).filter((ps) => (ps || []).length > 0).length;
    const lines = [
      "---",
      "layer: insights",
      `cross_domain: ${nCross}`,
      `critical: ${nCrit}`,
      "---",
      "",
      "# Insights",
      "",
      `> Higher-level patterns synthesized across ${nTopics} topic${nTopics === 1 ? "" : "s"}. **${nCross}** cross-domain, **${nCrit}** critical.`,
      "",
    ];
    if (nCross || nCrit) {
      const navItems = [];
      if (nCross) navItems.push("[Cross-domain](#cross-domain)");
      if (nCrit) navItems.push("[Critical](#critical)");
      navItems.push(link(this.actionsTarget(), "→ Actions"));
      lines.push("Jump to: " + navItems.join(" · "));
      lines.push("");
    }
    if (nCross) {
      lines.push("## Cross-domain");
      lines.push("");
      for (let idx = 0; idx < nCross; idx++) {
        const back = this.insightsBackEdges.get(`cross_domain ${idx}`) || [];
        // Continuous number across cross-domain → critical (see below).
        this._appendInsightSection(lines, cross[idx], "cross_domain", idx, back, idx + 1);
        lines.push("");
      }
    }
    if (nCrit) {
      lines.push("## Critical");
      lines.push("");
      for (let idx = 0; idx < nCrit; idx++) {
        const back = this.insightsBackEdges.get(`critical ${idx}`) || [];
        // Critical numbering picks up where cross-domain left off.
        this._appendInsightSection(lines, crit[idx], "critical", idx, back, nCross + idx + 1);
        lines.push("");
      }
    }
    if (!nCross && !nCrit) {
      lines.push("*(no insights detected)*");
      lines.push("");
    }
    return lines.join("\n");
  }

  _appendInsightSection(lines, ins, kind, idx, leadsToActions, displayNum) {
    const anchor = insightAnchor(kind, idx);
    // Heading shows the CONTINUOUS number (cross-domain then critical, no
    // per-scope restart) so it matches the actions-prompt `[N]` enumeration
    // and the action-body insight refs. The `^anchor` block ref stays
    // per-scope (`critical-1`) as the stable wikilink target.
    lines.push(`### [${displayNum}] ${ins.name}`);
    const domainsStr = (ins.domains || []).length ? ins.domains.join(", ") : "—";
    lines.push(`*${kind.replace(/_/g, "-")} · domains: ${domainsStr}* ^${anchor}`);
    lines.push("");
    lines.push(ins.description || "");
    lines.push("");
    lines.push(`**Mechanism:** ${ins.mechanism || ""}`);
    lines.push("");
    lines.push(`**Implication:** ${ins.implication || ""}`);
    if ((ins.proposed_actions || []).length > 0) {
      lines.push("");
      lines.push("**Proposed actions:**");
      for (const a of ins.proposed_actions) {
        lines.push(`- ${a}`);
      }
    }
    if ((ins.source_patterns || []).length > 0) {
      lines.push("");
      lines.push("**Provenance:**");
      lines.push(...this._walkPatternsForInsight(ins, 1.0, 0));
    }
    if (leadsToActions.length > 0) {
      lines.push("");
      lines.push("**Leads to actions:**");
      for (const [aIdx, conf] of leadsToActions) {
        const a = actionAnchor(aIdx);
        lines.push(`- ${link(this.actionsTarget(`^${a}`), `actions/${a}`)} · confidence ${fmt2(conf)}`);
      }
    }
    if (ins.hallucinated_ref_count) {
      lines.push("");
      lines.push(`*⚠ ${ins.hallucinated_ref_count} hallucinated refs*`);
    }
    lines.push("");
  }

  // ── Actions page ───────────────────────────────────────────────────────
  renderActionsFile() {
    const lines = [
      "---",
      "layer: actions",
      `count: ${this.actionsList.length}`,
      "---",
      "",
      "# Actions",
      "",
      "Prioritized from the proposed actions across all insights.",
      "`review_date` is the next **check-in**, not a completion deadline.",
      "",
    ];
    if (this.actionsList.length === 0) {
      lines.push("*(no actions generated)*");
      lines.push("");
      return lines.join("\n");
    }
    for (let idx = 0; idx < this.actionsList.length; idx++) {
      this._appendActionSection(lines, idx, this.actionsList[idx]);
      lines.push("");
    }
    return lines.join("\n");
  }

  // Turn positional `[N]` insight references in action prose into
  // wikilinks to the insight they point at. `[N]` is the continuous
  // cross-domain++critical enumeration (the same index the insight
  // headings now show + the actions prompt numbers by), so it maps to
  // one insight regardless of scope. Obsidian wikilink display can't
  // contain `]`, so the link text is the bare continuous number `N`
  // (the brackets can't survive inside `[[…|…]]`). The `^anchor` block
  // ref is the stable per-scope target. Out-of-range / array-index refs
  // are left untouched.
  _linkifyInsightRefs(text) {
    if (!text) return text;
    const cross = this.insightsPayload?.cross_domain || [];
    const crit = this.insightsPayload?.critical || [];
    const nCross = cross.length;
    const nTotal = nCross + crit.length;
    if (!nTotal) return text;
    return text.replace(/(?<![A-Za-z0-9])\[(\d+)\]/g, (tok, d) => {
      const n = Number(d);
      if (n < 1 || n > nTotal) return tok;
      const kind = n <= nCross ? "cross_domain" : "critical";
      const localIdx = n <= nCross ? n - 1 : n - 1 - nCross;
      return link(this.insightsTarget(`^${insightAnchor(kind, localIdx)}`), String(n));
    });
  }

  _appendActionSection(lines, idx, action) {
    const anchor = actionAnchor(idx);
    lines.push(`## [${anchor}] ${action.recommendation}`);
    const horizon = action.horizon || "";
    const score = computeActionScore(action);
    const conf = typeof action.confidence === "number" ? action.confidence : 1.0;
    lines.push(`*${horizon} horizon · score ${fmt2(score)} · confidence ${fmt2(conf)} · review ${action.review_date}* ^${anchor}`);
    lines.push("");
    lines.push(`**Objective:** ${this._linkifyInsightRefs(action.objective || "")}`);
    lines.push("");
    lines.push(`**Why:** ${this._linkifyInsightRefs(action.why || "")}`);
    lines.push("");
    lines.push(`**Immediate action (7d):** ${this._linkifyInsightRefs(action.immediate_action || "")}`);
    lines.push("");
    lines.push(`**Habit:** ${this._linkifyInsightRefs(action.habit || "")}`);
    lines.push("");
    lines.push(`**Success metric:** ${this._linkifyInsightRefs(action.success_metric || "")}`);
    lines.push("");
    const kindStr = action.kind ? `${action.kind} · ` : "";
    lines.push(
      `**Scores:** ${kindStr}` +
      `regret ${fmt2(action.regret_reduction)} · ` +
      `leverage ${fmt2(action.leverage)} · ` +
      `consequence ${fmt2(action.consequence)} · ` +
      `generativity ${fmt2(action.generativity)} · ` +
      `decisiveness ${fmt2(action.decisiveness)} · ` +
      `feedback-speed ${fmt2(action.time_to_feedback)} · ` +
      `constraint-fit ${fmt2(action.constraint_fit)}`
    );
    if ((action.source_insights || []).length > 0) {
      lines.push("");
      lines.push("**Provenance:**");
      // Sort by -edge_conf.
      const sorted = [...action.source_insights].sort((a, b) => b[2] - a[2]);
      for (const [kind, iIdx, edgeConf] of sorted) {
        const iAnchor = insightAnchor(kind, iIdx);
        const list = kind === "cross_domain"
          ? (this.insightsPayload.cross_domain || [])
          : (this.insightsPayload.critical || []);
        if (iIdx < 0 || iIdx >= list.length) {
          lines.push(`- **[Insight]** \`${this.insightsTarget(`^${iAnchor}`)}\` · edge ${fmt2(edgeConf)} · *broken*`);
          continue;
        }
        const ins = list[iIdx];
        const linkText = link(
          this.insightsTarget(`^${iAnchor}`),
          provDisplay(null, iAnchor, "insight"),
        );
        lines.push(graphLine(0, linkText, edgeConf, ins.description || ins.name || ""));
        lines.push(...this._walkPatternsForInsight(ins, edgeConf, 1));
      }
    }
    if (action.hallucinated_ref_count) {
      lines.push("");
      lines.push(`*⚠ ${action.hallucinated_ref_count} hallucinated refs*`);
    }
    lines.push("");
  }

  // ── Graph walks ────────────────────────────────────────────────────────
  // One boundary line: the cut crosses here, so the subtree below is
  // replaced by a count of the immediately-next (excluded) level. No
  // dangling wikilink — the reader still sees that depth existed.
  _boundaryLine(indent, n, noun) {
    return `${"  ".repeat(indent)}- *↳ ${plural(n, noun)} below the privacy cut (not exported)*`;
  }

  _walkRawForFact(fact, pathConf, indent) {
    if (!this.inc("raw")) {
      const n = (fact.evidence || []).filter((ev) => ev.file_path).length;
      return n > 0 ? [this._boundaryLine(indent, n, "source excerpt")] : [];
    }
    const out = [];
    for (const ev of fact.evidence || []) {
      if (!ev.file_path) continue;
      let target, display;
      if (ev.file_offset != null && ev.file_length != null) {
        target = this.inputTarget(ev.file_path, `^offset-${ev.file_offset}`);
        display = `${ev.file_path} @ ${ev.file_offset}:${ev.file_length}`;
      } else {
        target = this.inputTarget(ev.file_path);
        display = ev.file_path;
      }
      const linkText = link(target, display);
      const quote = (ev.text || "").trim().replace(/\n/g, " ");
      out.push(graphLine(indent, linkText, pathConf, quote, !!ev.approximate));
    }
    return out;
  }

  _walkFactsForPattern(topic, pat, parentPathConf, indent) {
    if (!this.inc("facts")) {
      const n = (pat.source_facts || []).length;
      return n > 0 ? [this._boundaryLine(indent, n, "fact")] : [];
    }
    const out = [];
    const topicFacts = this.factsByTopic[topic] || [];
    const sorted = [...(pat.source_facts || [])].sort((a, b) => b[1] - a[1]);
    for (const [fIdx, edgeConf] of sorted) {
      const pad = "  ".repeat(indent);
      if (fIdx < 0 || fIdx >= topicFacts.length) {
        out.push(`${pad}- **[Fact]** \`${this.topicTarget(2, topic, `^fact-${fIdx + 1}`)}\` · edge ${fmt2(edgeConf)} · *broken*`);
        continue;
      }
      const fact = topicFacts[fIdx];
      const pathConf = parentPathConf * edgeConf;
      const anchor = this.factAnchor(topic, fIdx);
      const linkText = link(
        this.topicTarget(2, topic, `^${anchor}`),
        provDisplay(topic, anchor),
      );
      out.push(graphLine(indent, linkText, pathConf, fact.summary || ""));
      out.push(...this._walkRawForFact(fact, pathConf, indent + 1));
    }
    return out;
  }

  _walkPatternsForInsight(ins, parentPathConf, indent) {
    if (!this.inc("patterns")) {
      const n = (ins.source_patterns || []).length;
      return n > 0 ? [this._boundaryLine(indent, n, "pattern")] : [];
    }
    const out = [];
    const sorted = [...(ins.source_patterns || [])].sort((a, b) => b[2] - a[2]);
    for (const [topic, pIdx, edgeConf] of sorted) {
      const pad = "  ".repeat(indent);
      const topicPats = this.patternsByTopic[topic] || [];
      if (pIdx < 0 || pIdx >= topicPats.length) {
        out.push(`${pad}- **[Pattern]** \`${this.topicTarget(4, topic, `^pattern-${pIdx + 1}`)}\` · edge ${fmt2(edgeConf)} · *broken*`);
        continue;
      }
      const pat = topicPats[pIdx];
      const patPathConf = parentPathConf * edgeConf;
      const anchor = this.patternAnchor(topic, pIdx);
      const linkText = link(
        this.topicTarget(4, topic, `^${anchor}`),
        provDisplay(topic, anchor),
      );
      out.push(graphLine(indent, linkText, patPathConf, pat.description || pat.name || ""));
      out.push(...this._walkFactsForPattern(topic, pat, patPathConf, indent + 1));
    }
    return out;
  }

  // ── Entity file ────────────────────────────────────────────────────────
  renderEntityFile(ent, isSubject, relations, idToName) {
    const role = ent.role || (isSubject ? "subject" : "");
    const lines = [
      "---",
      "layer: entity",
      `canonical_id: ${ent.canonical_id}`,
      `entity_type: ${ent.entity_type}`,
    ];
    if (role) lines.push(`role: ${role}`);
    lines.push("---", "", `# ${ent.canonical_name}`, "");
    const desc = (ent.description || "").trim();
    lines.push(`*${desc || "(no description)"}* ^entity`);
    lines.push("");

    if ((ent.aliases || []).length > 0) {
      lines.push("## Aliases");
      for (const a of ent.aliases) lines.push(`- ${a}`);
      lines.push("");
    }

    if (relations.length > 0) {
      lines.push("## Relations");
      // Sort: -confidence, relation, from, to.
      const sorted = [...relations].sort((a, b) => {
        const dc = (b.confidence ?? 1) - (a.confidence ?? 1);
        if (dc !== 0) return dc;
        const dr = (a.relation || "").localeCompare(b.relation || "");
        if (dr !== 0) return dr;
        const df = (a.from_id || a.from || "").localeCompare(b.from_id || b.from || "");
        if (df !== 0) return df;
        return (a.to_id || a.to || "").localeCompare(b.to_id || b.to || "");
      });
      for (const r of sorted) {
        const fromId = r.from_id ?? r.from;
        const toId = r.to_id ?? r.to;
        const fromDisplay = idToName[fromId] || fromId;
        const toDisplay = idToName[toId] || toId;
        const fromTarget = this.entityTarget(fromId, "^entity");
        const toTarget = this.entityTarget(toId, "^entity");
        lines.push(`- ${link(fromTarget, fromDisplay)} is ${r.relation} of ${link(toTarget, toDisplay)} (confidence ${fmt2(r.confidence ?? 1)})`);
      }
      lines.push("");
    }

    const refs = ent.evidence_fact_refs || [];
    if (refs.length > 0 && !this.inc("facts")) {
      lines.push("## Cited in facts");
      lines.push(`- *↳ ${plural(refs.length, "fact")} below the privacy cut (not exported)*`);
      lines.push("");
    } else if (refs.length > 0) {
      lines.push("## Cited in facts");
      const shown = refs.slice(0, 30);
      for (const ref of shown) {
        const topic = Array.isArray(ref) ? ref[0] : ref?.[0];
        const fIdx = Array.isArray(ref) ? ref[1] : ref?.[1];
        const anchor = this.factAnchor(topic, fIdx);
        const target = this.topicTarget(2, topic, `^${anchor}`);
        lines.push(`- ${link(target, `${topic}/${anchor}`)}`);
      }
      if (refs.length > 30) {
        lines.push(`- *(+${refs.length - 30} more)*`);
      }
      lines.push("");
    }

    return lines.join("\n");
  }

  // ── Index page ─────────────────────────────────────────────────────────
  renderIndexFile() {
    const lines = ["# Run summary\n"];
    for (const [k, v] of Object.entries(this.stats || {})) {
      lines.push(`- **${k}**: ${v}`);
    }
    lines.push("");
    lines.push("## Top-level artifacts");
    lines.push("");
    lines.push(`- ${link(this.actionsTarget(), "Actions")}`);
    lines.push(`- ${link(this.insightsTarget(), "Insights")}`);
    if (this.inc("patterns")) lines.push("- 3-patterns/ — one file per topic");
    if (this.inc("entities")) lines.push("- 2-entities/ — one file per canonical entity");
    if (this.inc("facts")) lines.push("- 1-facts/ — one file per topic");
    if (this.inc("raw")) lines.push("- 0-inputs/ — preprocessed source files");
    return lines.join("\n") + "\n";
  }

  // ── 0-inputs rewrite with footnote markers ─────────────────────────────
  renderInputsFiles() {
    const out = {};
    const inputs = this.inputs || {};
    if (!inputs || Object.keys(inputs).length === 0) return out;

    // Build span map: file_id → array of {topic, factIdx, offset, length}
    const spansByFile = new Map();
    for (const [topic, items] of Object.entries(this.factsByTopic)) {
      for (let factIdx = 0; factIdx < items.length; factIdx++) {
        const fact = items[factIdx];
        for (const ev of fact.evidence || []) {
          if (!ev.file_path || ev.file_offset == null || ev.file_length == null) continue;
          if (!spansByFile.has(ev.file_path)) spansByFile.set(ev.file_path, []);
          spansByFile.get(ev.file_path).push({
            topic,
            factIdx,
            offset: ev.file_offset,
            length: ev.file_length,
          });
        }
      }
    }

    const fileIds = Object.keys(inputs).sort();
    for (const fileId of fileIds) {
      const text = inputs[fileId];
      const spans = spansByFile.get(fileId) || [];
      const spansForward = [...spans].sort((a, b) => (a.offset + a.length) - (b.offset + b.length));

      const PUNCT = ".!?,;:\"')]}";
      const advancePastPunct = (pos) => {
        while (pos < text.length && PUNCT.indexOf(text[pos]) !== -1) pos++;
        return pos;
      };
      const lineEndFrom = (pos) => {
        const nl = text.indexOf("\n", pos);
        return nl !== -1 ? nl : text.length;
      };

      const markerBuckets = new Map();
      const anchorBuckets = new Map();
      // Maintain insertion order of offsets within each anchor bucket so the
      // earliest-end-offset (= first-inserted) is preferred when resolving
      // duplicates — matches Python's spans_forward iteration.
      for (const span of spansForward) {
        const spanEnd = span.offset + span.length;
        const markerPos = advancePastPunct(spanEnd);
        const anchorPos = lineEndFrom(spanEnd);
        if (!markerBuckets.has(markerPos)) markerBuckets.set(markerPos, []);
        markerBuckets.get(markerPos).push([span.topic, span.factIdx]);
        if (!anchorBuckets.has(anchorPos)) anchorBuckets.set(anchorPos, []);
        anchorBuckets.get(anchorPos).push(span.offset);
      }

      const insertsByPos = new Map();
      const seenAnchors = new Set();
      const anchorPosDesc = [...anchorBuckets.keys()].sort((a, b) => b - a);
      for (const pos of anchorPosDesc) {
        let chosen = null;
        for (const s of anchorBuckets.get(pos)) {
          if (!seenAnchors.has(s)) {
            seenAnchors.add(s);
            chosen = s;
            break;
          }
        }
        if (chosen != null) {
          insertsByPos.set(pos, (insertsByPos.get(pos) || "") + ` ^offset-${chosen}`);
        }
      }

      // Wrap each cited line in blank lines.
      const wrappedPositions = new Set();
      for (const lineEnd of anchorBuckets.keys()) {
        const lineStart = text.lastIndexOf("\n", lineEnd - 1) + 1;
        if (lineStart >= 2 && text[lineStart - 2] !== "\n" && !wrappedPositions.has(lineStart)) {
          insertsByPos.set(lineStart, "\n" + (insertsByPos.get(lineStart) || ""));
          wrappedPositions.add(lineStart);
        }
        const suffixPos = lineEnd + 1;
        if (suffixPos < text.length && text[suffixPos] !== "\n" && !wrappedPositions.has(suffixPos)) {
          insertsByPos.set(suffixPos, "\n" + (insertsByPos.get(suffixPos) || ""));
          wrappedPositions.add(suffixPos);
        }
      }

      // Inline marker text with semantic display.
      for (const [pos, items] of markerBuckets.entries()) {
        const markerText = items.map(([t, idx]) => {
          const anchor = this.factAnchor(t, idx);
          return link(this.topicTarget(2, t, `^${anchor}`), `[${t}/${anchor}]`);
        }).join("");
        insertsByPos.set(pos, ` ${markerText}` + (insertsByPos.get(pos) || ""));
      }

      let body = text;
      const positions = [...insertsByPos.keys()].sort((a, b) => b - a);
      for (const pos of positions) {
        const insertAt = Math.min(pos, body.length);
        body = body.slice(0, insertAt) + insertsByPos.get(pos) + body.slice(insertAt);
      }
      body = escapeTags(body);

      const frontmatter = [
        "---",
        "layer: input",
        `file_id: ${fileId}`,
        `cited_facts: ${spansForward.length}`,
        "---",
        "",
        `# ${fileId}`,
        "",
      ];
      out[`0-inputs/${fileId}.md`] = frontmatter.join("\n") + body;
    }
    return out;
  }
}

// Action.score is a Python @property recomputed from the weighted axes;
// `runner.py` writes it to the action JSON but on-the-fly recompute keeps
// us byte-identical even if a future schema change drops the field.
function computeActionScore(action) {
  if (typeof action.score === "number") return action.score;
  return (
    0.25 * (action.regret_reduction || 0) +
    0.20 * (action.leverage || 0) +
    0.15 * (action.consequence || 0) +
    0.15 * (action.generativity || 0) +
    0.10 * (action.decisiveness || 0) +
    0.075 * (action.time_to_feedback || 0) +
    0.075 * (action.constraint_fit || 0)
  );
}

// ── Public API ────────────────────────────────────────────────────────────

/**
 * Render the whole vault for a finished run. Returns a manifest of
 * `{ relPath: content }`. The caller writes them out.
 *
 * `inputs` is optional: a `{ fileId: text }` map of preprocessed
 * markdown files (no .md extension on the key). When omitted, no
 * `0-inputs/*.md` files are emitted (parity with vault_exporter when
 * `preprocessed_dir` is missing).
 *
 * `runName` is the relative wikilink prefix from the Obsidian vault
 * root to this run's dir. Empty when the run dir IS the vault root.
 *
 * `privacyLevel` names the rightmost included level on the privacy
 * slider (see PRIVACY_LEVELS). Defaults to `"raw"` (full export). At a
 * lower cut the excluded levels' files are not emitted and every
 * reference crossing the cut is rendered as a provenance count, so the
 * manifest is internally consistent — no dangling wikilinks.
 */
export function exportRun({
  factsByTopic = {},
  entitiesPayload = null,
  patternsByTopic = {},
  insightsPayload = null,
  actionsList = [],
  stats = {},
  runName = "",
  inputs = null,
  privacyLevel = "raw",
} = {}) {
  const r = new Renderer({
    factsByTopic, entitiesPayload, patternsByTopic,
    insightsPayload, actionsList, stats, runName, inputs, privacyLevel,
  });
  const out = {};

  // 1-facts/<topic>.md — sorted by topic.
  const factTopics = r.inc("facts") ? Object.keys(factsByTopic).sort() : [];
  for (const topic of factTopics) {
    const facts = factsByTopic[topic] || [];
    const citedByMap = new Map();
    for (let i = 0; i < facts.length; i++) {
      const cited = r.factsCitedBy.get(`${topic} ${i}`) || [];
      citedByMap.set(i, cited);
    }
    out[`1-facts/${topic}.md`] = r.renderFactsTopic(topic, facts, citedByMap);
  }

  // 0-inputs/<file>.md — only when inputs were passed in.
  if (r.inc("raw")) Object.assign(out, r.renderInputsFiles());

  // 2-entities/<id>.md — only when there are entity records.
  const entities = (r.inc("entities") && entitiesPayload?.entities) || [];
  if (entities.length > 0) {
    const subjectId = entitiesPayload?.subject?.canonical_id || null;
    const allRelations = entitiesPayload?.relations || [];
    const byEntity = new Map();
    for (const rel of allRelations) {
      const fromId = rel.from_id ?? rel.from;
      const toId = rel.to_id ?? rel.to;
      if (!byEntity.has(fromId)) byEntity.set(fromId, []);
      byEntity.get(fromId).push(rel);
      if (toId !== fromId) {
        if (!byEntity.has(toId)) byEntity.set(toId, []);
        byEntity.get(toId).push(rel);
      }
    }
    const idToName = {};
    for (const e of entities) idToName[e.canonical_id] = e.canonical_name;
    for (const ent of entities) {
      const isSubject = ent.canonical_id === subjectId;
      const rels = byEntity.get(ent.canonical_id) || [];
      out[`2-entities/${ent.canonical_id}.md`] = r.renderEntityFile(ent, isSubject, rels, idToName);
    }
  }

  // 3-patterns/<topic>.md — sorted by topic, skip empty topics.
  const patternTopics = r.inc("patterns") ? Object.keys(patternsByTopic).sort() : [];
  for (const topic of patternTopics) {
    const pats = patternsByTopic[topic] || [];
    if (pats.length === 0) continue;
    const citedByMap = new Map();
    for (let i = 0; i < pats.length; i++) {
      const cited = r.patternsCitedBy.get(`${topic} ${i}`) || [];
      citedByMap.set(i, cited);
    }
    const topicFacts = factsByTopic[topic] || [];
    out[`3-patterns/${topic}.md`] = r.renderPatternsTopic(topic, pats, topicFacts, citedByMap);
  }

  // 4-insights.md
  out["4-insights.md"] = r.renderInsightsFile();

  // 5-actions.md
  out["5-actions.md"] = r.renderActionsFile();

  // index.md
  out["index.md"] = r.renderIndexFile();

  return out;
}

/**
 * Vault-root README — listing of runs newest first. Caller hands us a
 * sorted list of run names (lexically reverse-chrono — `<iso-z>-<suffix>`)
 * plus the count, and we return the README contents.
 *
 * Mirrors `vault_exporter.write_vault_readme` minus the disk traversal:
 * the caller scans the vault root and passes the run names in.
 */
export function renderVaultReadme(runNames, hasInsightsByRun = {}) {
  const lines = ["# BaseVault runs\n"];
  lines.push(`Total runs: ${runNames.length}\n`);
  lines.push("## Most recent\n");
  const top = runNames.slice(0, 20);
  for (const name of top) {
    const target = hasInsightsByRun[name] ? `${name}/4-insights` : name;
    lines.push(`- [[${target}|${name}]]`);
  }
  if (runNames.length > 20) {
    lines.push(`\n*(+${runNames.length - 20} older runs)*`);
  }
  return lines.join("\n") + "\n";
}

/**
 * Materialize `<vault_root>/<runId>/` for a completed run by fetching
 * each stage's payload over the existing `read_run_*` Tauri commands,
 * calling `exportRun`, and writing the manifest via `write_run_vault`.
 *
 * Replaces the legacy `regen_vault` Tauri command which spawned Python
 * (~1-2s startup per export). Pure JS now: typically <100ms total.
 *
 * Throws on any I/O failure with the offending stage in the message
 * so the UI can surface it.
 */
export async function regenVault({ runId, invoke, privacyLevel = "raw" }) {
  if (!runId) throw new Error("regenVault: runId required");
  if (!invoke) throw new Error("regenVault: invoke required");

  const [factsByTopic, entitiesPayload, patternsByTopic, insightsPayload, actionsPayload, inputs] =
    await Promise.all([
      invoke("read_run_facts_all", { runId }).catch(() => ({})),
      invoke("read_run_entities", { runId }).catch(() => null),
      invoke("read_run_patterns_all", { runId }).catch(() => ({})),
      invoke("read_run_insights", { runId }).catch(() => null),
      invoke("read_run_actions", { runId }).catch(() => ({ actions: [] })),
      invoke("read_run_preprocessed_inputs", { runId }).catch(() => ({})),
    ]);

  const actionsList = (actionsPayload && Array.isArray(actionsPayload.actions))
    ? actionsPayload.actions
    : [];

  // Stats payload mirrors what the Python `regen_vault_only` writes
  // into `index.md`'s frontmatter — derived counts only, no LLM output.
  const stats = {
    facts: Object.values(factsByTopic || {}).reduce((n, list) => n + (list || []).length, 0),
    entities: (entitiesPayload?.entities || []).length,
    relations: (entitiesPayload?.relations || []).length,
    patterns: Object.values(patternsByTopic || {}).reduce((n, list) => n + (list || []).length, 0),
    insights: (insightsPayload?.cross_domain || []).length + (insightsPayload?.critical || []).length,
    actions: actionsList.length,
  };

  const files = exportRun({
    factsByTopic: factsByTopic || {},
    entitiesPayload: entitiesPayload || null,
    patternsByTopic: patternsByTopic || {},
    insightsPayload: insightsPayload || null,
    actionsList,
    stats,
    // Prefix every wikilink with the run id so links stay unambiguous
    // when the user has multiple runs in the same Obsidian vault —
    // bare `[[5-actions|Actions]]` would resolve to whichever run
    // Obsidian indexed first. With the prefix:
    //   `[[2026-05-02T12-10-32Z-mtq2/5-actions|Actions]]`.
    runName: runId,
    inputs,
    privacyLevel,
  });

  // write_run_vault wipes the per-level subdirs + single-file pages
  // before rewriting, so a lower cut (fewer manifest keys) leaves no
  // stale 0-inputs/ etc. from an earlier fuller export of this run.
  await invoke("write_run_vault", { runId, files });
}

// Internal helpers exported for tests + the in-app RunViews helpers.
export const __testing = {
  computeFactAnchorsForTopic,
  computeFactAnchors,
  computePatternAnchorsForTopic,
  computePatternAnchors,
  insightAnchor,
  actionAnchor,
  buildFactsCitedBy,
  buildPatternsCitedBy,
  buildInsightsBackEdges,
  escapeTags,
  escapeQuoteForBullet,
  pyTitle,
  singularizeTopic,
  provDisplay,
  fmt2,
  link,
  graphLine,
  computeActionScore,
};
