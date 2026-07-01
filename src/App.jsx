import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { open as openDialog, confirm as askConfirm } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import Wizard from "./Wizard";
import Settings from "./Settings";
import AttestationPanel from "./AttestationPanel";
import { CopyButton, copyToClipboard } from "./CopyButton";
import { TracebackDetails } from "./ErrorWithTrace";
import { prettyDateTime } from "./dateFormat";
import ChatbotHelper from "./ChatbotHelper";
import baseVaultLogo from "./assets/basevault-logo.svg";
import {
  FactsView,
  EntityView,
  PatternsView,
  InsightsView,
  ActionsView,
} from "./RunViews";
import {
  regenVault,
  computeFactAnchorsForTopic,
  computePatternAnchorsForTopic,
} from "./obsidianRenderer";
import { providerDisplayName, modelDisplayName } from "./teeProviders";
import { validateInputs } from "./validateInputs";
import { localModeUsable } from "./localUsable";
import "./App.css";

// ── DEV TRACING ──────────────────────────────────────────────────────────
// Opt-in timing instrumentation gated by the `dev_tracing` config flag
// (Settings → Development). `record_dev_trace` ships the line to Rust's
// info! sink so the .app log captures it alongside Rust + Python markers
// — toggle the setting, click Run, grep app.log. The first emit per
// layer becomes that layer's t=0; `wall` is unix epoch seconds for
// cross-layer correlation.
//
// The Rust `record_dev_trace` command short-circuits when the config
// flag is off, so the JS side does not cache the flag locally — that
// caching pattern raced against the click (initial async invoke had not
// resolved when doRun ran, leaving _DEV_TRACING_ENABLED false through
// the entire click → row chain even when the setting was on). The Rust
// gate is the single source of truth; every ltrace pays one IPC call,
// and ltrace fires only at coarse boundaries (click, listener fire,
// coalesce fire, refresh entry/exit) so the per-marker IPC cost is
// negligible.
let _DEV_TRACING_T0_MS = 0;
function ltrace(step) {
  const now = performance.now();
  if (_DEV_TRACING_T0_MS === 0) _DEV_TRACING_T0_MS = now;
  const t = ((now - _DEV_TRACING_T0_MS) / 1000).toFixed(3);
  const wall = (Date.now() / 1000).toFixed(3);
  const line = `[LAUNCH_TRACE] ${step} t=${t} wall=${wall}`;
  invoke("record_dev_trace", { line }).catch(() => {});
}
// ── /DEV TRACING ─────────────────────────────────────────────────────────

// Mode metadata. The CSS class drives the Run button color:
//   green = local (nothing leaves)
//   blue  = tee   (encrypted to Tinfoil TEE, attested)
const MODE_META = {
  local: {
    label: "Local",
    btnClass: "run-btn-local",
    trust: "🔒 Nothing leaves your machine. All processing is local.",
  },
  tee: {
    label: "Private Cloud",
    btnClass: "run-btn-tee",
    trust: "🔒 Private Cloud — only encrypted, anonymized data reaches a Trusted Execution Environment (TEE). Raw files never leave your machine. Attested backend: Tinfoil.",
  },
};

const CONFIG_PERSISTED_KEYS = ["inputs", "mode"];

// Privacy slider stops, left → right = increasingly raw / less private.
// One tick per pipeline level; `name` is the label drawn above the tick.
// `level` is the renderer's privacyLevel (rightmost-included). Actions +
// Insights are the fixed floor: both always exported, the handle cannot
// rest below Insights. `raw` is weighted — it requires an explicit
// acknowledgement. Counts come from the run(s) being exported.
const PRIVACY_SLIDER = [
  { level: "actions", name: "Actions", countKey: "actions", noun: "action", floor: true },
  { level: "insights", name: "Insights", countKey: "insights", noun: "insight", floor: true },
  { level: "patterns", name: "Patterns", countKey: "patterns", noun: "pattern" },
  { level: "entities", name: "Entities", countKey: "entities", noun: "entity", nounPlural: "entities" },
  { level: "facts", name: "Facts", countKey: "facts", noun: "fact" },
  { level: "raw", name: "Raw Inputs", countKey: "rawDocs", noun: "source document", raw: true },
];
// Handle cannot rest below "Insights" (idx 1): Actions + Insights is the
// minimum coherent shareable unit, always included.
const PRIVACY_FLOOR_INDEX = 1;

const STAGE_LABELS = {
  init: "Initializing pipeline",
  // `preflight` covers the run-head wait while the runner pre-warms
  // attestation (cold Sigstore TUF fetch is ~30s on first use of a
  // model) plus the ~320ms heavy-import block before that. Surfaced
  // immediately after run.json hits disk so the run row + elapsed
  // ticker render within ~100ms of the click instead of waiting for
  // imports + attestation to finish (which left the button feeling
  // frozen for 2-7s). Phrased as "Verifying inference environment"
  // rather than "TEE attestation" because the same stage covers the
  // whole pre-LLM-call setup; the attestation surface modal still
  // says "TEE attestation" specifically.
  preflight: "Verifying inference environment",
  ingest: "Ingesting inputs",
  // `split` (formerly its own bar label) is Phase 1 of Stage 1
  // (Extraction) per the pipeline-stages spec — the runner now
  // transitions straight to "extract" when the splitter starts, so
  // this entry is unreached but kept as a graceful fallback for
  // older runs replayed in the UI.
  split: "Extracting facts",
  metadata: "Extracting metadata",
  // `vision` runs during ingestion — image LLM calls (describe_image)
  // landed via #121 and are now part of PIPELINE_STAGES. Surface a
  // friendly label so the run-row progress line and Details modal
  // don't show the bare stage id.
  vision: "Transcribing images",
  extract: "Extracting facts",
  vault_export: "Writing vault",
  entities: "Resolving entities",
  entities_dedupe: "Deduplicating entities",
  patterns: "Detecting patterns",
  insights: "Synthesizing insights",
  actions: "Prioritizing actions",
  embeddings: "Generating embeddings",
  done: "Done",
};

const STATUS_LABELS = {
  running: "Running",
  paused: "Paused",
  cancelled: "Cancelled",
  failed: "Failed",
  completed: "Completed",
};

// Default split widths in pixels — runs left, tree middle, markdown
// the rest. 320px each on the two left columns gives the runs pane
// room for two-line metadata (status + mode + ETA) AND the file tree
// enough room for nested provenance file names. Markdown pane gets
// the rest (~800px on the 1440px default window).
const DEFAULT_SPLIT = { runs: 320, tree: 320 };

// How long to wait for the runner to commit its new status (paused
// or running) on disk before we give up polling. Real transitions
// usually settle in under a second; the budget is loose so a slow
// Python startup on a cold filesystem doesn't strand the UI.
const PAUSE_RESUME_POLL_TIMEOUT_MS = 8000;
const PAUSE_RESUME_POLL_INTERVAL_MS = 250;

// Mirror a value with a debounce so an effect that depends on it
// doesn't fire on every keystroke. Used by the cmd+F search to
// throttle DOM walks on large input files.
function useDebouncedValue(value, delayMs) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    if (debounced === value) return;
    const id = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, delayMs]);
  return debounced;
}

// Cmd+F match cap. Long input files (e.g. 4 MiB raw text) can produce
// hundreds of thousands of regex matches for a single-character query;
// without a cap, downstream Range allocation, Highlight construction,
// and CSS paint stall the main thread for seconds and the spread-form
// `new Highlight(...ranges)` constructor risks exceeding the JS engine
// argument-count limit. 5000 is large enough that real find-in-document
// usage is unaffected and matches the conventional browser/editor
// behaviour of stopping after the first N hits.
export const MAX_SEARCH_MATCHES = 5000;

// Map a text-content offset back to the index of the (sorted, contiguous)
// segment that contains it. Segments are produced by walking text nodes
// in document order and concatenating their content, so they are sorted
// by `start` AND contiguous (`segments[i].start === segments[i-1].end`).
// Regex matches arrive in increasing-offset order, so the caller can
// thread a single `fromIndex` cursor across matches and the total work
// is O(M + S) instead of O(M·S). Returns -1 when the offset is past the
// end of the segments array.
export function findSegmentForward(segments, offset, fromIndex = 0) {
  let i = fromIndex < 0 ? 0 : fromIndex;
  while (i < segments.length && segments[i].end <= offset) i += 1;
  if (i >= segments.length) return -1;
  return i;
}

function basename(p) {
  if (!p) return "";
  const segs = p.split("/");
  return segs[segs.length - 1] || p;
}

// POSIX parent directory of an absolute path (BaseVault is Mac/Linux
// only). Returns "" when there's no parent to speak of so callers can
// fall back to the OS default.
function dirname(p) {
  if (!p) return "";
  const i = p.lastIndexOf("/");
  return i > 0 ? p.slice(0, i) : "";
}

// User-facing filename: basename minus a trailing `.md`. The pipeline
// normalizes every ingested input to a `.md` file on disk, so showing
// the extension is noise — all four UI surfaces that render a path
// (staging, RunDetails inputs, run row title, markdown pane title)
// route through here. Non-`.md` extensions are preserved (e.g. an
// `.txt` original that hasn't been ingested yet).
function displayBasename(p) {
  const b = basename(p);
  return b.endsWith(".md") ? b.slice(0, -3) : b;
}

function runTitle(run) {
  if (!run?.inputs?.length) return "No inputs";
  const first = displayBasename(run.inputs[0]);
  if (run.inputs.length === 1) return first;
  return `${first} +${run.inputs.length - 1} more`;
}

function formatDuration(ms) {
  if (ms === null || ms === undefined) return "—";
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  return r ? `${m}m${r}s` : `${m}m`;
}

// Compact byte-size formatter for input lists (staging + RunDetails).
// Two tiers only — kb / mb — switching at 1024×1024 bytes. No GB tier
// (corpora over 1gb render as `1024mb+` and that's fine, they're
// pathological in this app's use case). One decimal digit when the
// integer part is a single digit (e.g. `1.0kb`, `4.2kb`, `1.4mb`);
// otherwise integer-rounded (`10kb`, `105kb`, `51mb`). 0 bytes is the
// only special case — renders as `0kb` (not `0.0kb`).
//
// Branch order matters at the 9.95-rounded boundary: round to one
// decimal FIRST, then test `< 10`. A raw 9.97 has int part 9 (would
// trigger the decimal branch) but rounds to 10.0 — the int branch is
// the right rendering ("10kb") so callers don't see "10.0kb".
export function formatSizeShort(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0kb";
  const KB = 1024;
  const MB = 1024 * 1024;
  if (bytes < MB) {
    const v = bytes / KB;
    const rounded1 = Math.round(v * 10) / 10;
    if (rounded1 < 10) return `${rounded1.toFixed(1)}kb`;
    return `${Math.round(v)}kb`;
  }
  const v = bytes / MB;
  const rounded1 = Math.round(v * 10) / 10;
  if (rounded1 < 10) return `${rounded1.toFixed(1)}mb`;
  return `${Math.round(v)}mb`;
}

// Render the time-based ETA for the progress bar's secondary label
// ("~2m 30s remaining"). Bucketed: "<5s remaining", "Xs", "Xm Ys",
// "Xh Ym".
//
// QUANTIZE-THEN-SPLIT, not split-then-quantize. Round 3 had a
// rollover bug where 356s rendered as "5min 60s": flooring 356/60
// gave 5 minutes, then rounding the 56s remainder to a 10s grid
// produced 60s — printed as "5m 60s" instead of rolling over to
// "6m". The fix is to quantize the total to the grid FIRST, then
// split into h/m/s — that way the remainder is always < grid_size
// and rollover is structurally impossible.
//
// Grid step picked from the raw seconds magnitude: 5s under a
// minute, 10s under an hour, 60s above.
export function formatEta(seconds) {
  if (seconds === null || seconds === undefined) return null;
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  if (seconds < 5) return "<5s remaining";
  let step;
  if (seconds < 60) step = 5;
  else if (seconds < 3600) step = 10;
  else step = 60;
  const q = Math.round(seconds / step) * step;
  if (q < 60) return `~${q}s remaining`;
  if (q < 3600) {
    const m = Math.floor(q / 60);
    const r = q - m * 60;
    return r ? `~${m}m ${r}s remaining` : `~${m}m remaining`;
  }
  const h = Math.floor(q / 3600);
  const m = Math.floor((q - h * 3600) / 60);
  return m ? `~${h}h ${m}m remaining` : `~${h}h remaining`;
}

// Run list / run header timestamps go through the ONE shared
// formatter (same as chats + attestation) — see dateFormat.js.
function formatDate(iso) {
  return prettyDateTime(iso);
}

// Live elapsed-time formatter for in-flight runs (issue #159). Distinct
// from formatEta on purpose: ETA is bucketed/quantized to avoid jitter
// on a noisy estimate, but elapsed is an exact fact and should tick
// each second. Distinct from formatDuration too — that one is the
// compact "30s" / "5m30s" form used for finished-run summaries; this
// one matches the "8m 23s" / "1h 12m 42s" shape the brief calls for.
//
// Seconds are always shown once we cross an hour ("5h 18m 42s", never
// "5h 18m" or a raw "286m29s") — past 1h the user needs the seconds
// digit visibly ticking to read the run as alive rather than stuck
// (issue #586).
export function fmtElapsed(ms) {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return null;
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s`;
  if (s < 3600) {
    const m = Math.floor(s / 60);
    const r = s - m * 60;
    return r ? `${m}m ${r}s` : `${m}m`;
  }
  const h = Math.floor(s / 3600);
  const m = Math.floor((s - h * 3600) / 60);
  const r = s - h * 3600 - m * 60;
  return `${h}h ${m}m ${r}s`;
}

// Resolve a run's elapsed-time-in-ms. Both running and non-running
// states display the backend's `duration_ms`, which is active runtime
// across all cycles (`derive_run_state` sums closed (cycle_start,
// terminator) pairs and extends by `(now - latest_open_start)` while
// running). Pause windows are excluded by construction — the gap
// between a pause's terminator and the next resume's cycle_start sits
// outside any open accumulator slot.
//
// Fallback: a fresh run that hasn't emitted its first cycle_start yet
// has no backend duration_ms. Tick from `created_at` against `nowMs`
// (the 1Hz wallclock ticker) so the display isn't blank during the
// pre-cycle_start startup window. Once cycle_start lands the next
// list_runs poll switches to the backend value automatically.
export function liveElapsedMs(run, nowMs) {
  if (!run) return null;
  if (Number.isFinite(run.duration_ms)) return run.duration_ms;
  if (run.status === "running") {
    const t = Date.parse(run.created_at);
    if (!Number.isFinite(t)) return null;
    return Math.max(0, nowMs - t);
  }
  return null;
}

// Single shared 1Hz wallclock (issue #159). Active flag gates the
// setInterval registration: when no run is running, the interval is
// never mounted — idle CPU when nothing's in flight. Re-snapshots
// Date.now() on activation so the first render after a run starts
// shows the correct elapsed instead of a stale value frozen at mount.
//
// Independent of the pipeline-progress event coalescer (progressTick):
// progressTick fires on per-call returns (gaps of minutes), this one
// fires on each second of wallclock so the user sees the run is alive.
export function useElapsedNow(active) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return undefined;
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

// Track the user's prefers-reduced-motion preference. Used to drop
// the .is-running class from the progress fill so the barber-pole
// animation is suppressed at the JSX level, not just via a CSS @media
// fallback — the brief calls for class-omission so a11y audits and
// the dedicated unit test can both assert on it directly.
export function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(() => {
    if (typeof window === "undefined" || !window.matchMedia) return false;
    try {
      return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    } catch {
      return false;
    }
  });
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return undefined;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = (e) => setReduced(e.matches);
    if (mq.addEventListener) mq.addEventListener("change", onChange);
    else if (mq.addListener) mq.addListener(onChange);
    return () => {
      if (mq.removeEventListener) mq.removeEventListener("change", onChange);
      else if (mq.removeListener) mq.removeListener(onChange);
    };
  }, []);
  return reduced;
}

// Map a click-target relPath onto its on-disk shape under a run dir.
// Stage 0 unconditionally appends `.md` to every preprocessed source
// regardless of original extension, so `0-inputs/README.md` (a fact
// or wikilink target carrying the original filename) lives on disk
// as `0-inputs/README.md.md`. A naive "append .md if missing" guard
// fixes `notes.txt` → `notes.txt.md` but leaves `README.md` pointing
// at a nonexistent file — the user-visible bug. For non-`0-inputs/`
// stages the filename already matches disk, so we only append when
// missing (e.g. `1-facts/work` → `1-facts/work.md`).
export function normalizeNavRelPath(relPath) {
  if (!relPath) return relPath;
  if (relPath.startsWith("0-inputs/")) return `${relPath}.md`;
  return relPath.endsWith(".md") ? relPath : `${relPath}.md`;
}

// Parse a chatbot citation ({kind, record_id}) into the deterministic
// part of its run-view target: the file relPath plus, where the anchor
// scheme is positional (insight/action) or file-level (entity/chunk),
// the anchor. record_id formats are minted by the embeddings stage.
//
// fact / pattern anchors are NOT positional: the rendered markdown uses
// the type-scoped scheme (obsidianRenderer.computeFact/PatternAnchors —
// `emotion-3`, not `fact-15`), so record_id's raw index cannot be
// turned into an anchor without the topic's ordered item list. Those
// come back with anchor "" (opens the file head — never blank) plus
// `topic`/`idx` so `resolveCitation` can upgrade them. chunk has no
// per-stride anchor in the input file (only fact-evidence offsets get
// `^offset-N`), so it is also file-level. relPath is left
// un-normalized — handleMarkdownNavigate runs normalizeNavRelPath.
// Returns null for an unknown kind / malformed id so the click no-ops
// rather than jumping somewhere wrong.
export function resourceNavTarget(kind, recordId) {
  if (!kind || !recordId) return null;
  if (kind === "chunk") {
    const at = recordId.lastIndexOf("@");
    if (at < 0) return null;
    const relPath = `0-inputs/${recordId.slice(0, at)}`;
    // The chunk's start char-offset into the source file IS recoverable
    // from the record_id (`<file_id>@<char_offset>`, minted by the
    // embeddings stage). Prior code dropped it and returned anchor "",
    // so a chunk citation opened the source file at the top instead of
    // the grounded passage. Emit the same `offset-<N>` anchor scheme
    // the input view already uses for fact-evidence back-links, and
    // carry the raw offset so the input view can mark + highlight the
    // passage (it has no fact-evidence mark at an arbitrary chunk
    // offset on its own).
    const off = Number(recordId.slice(at + 1));
    if (!Number.isInteger(off) || off < 0) {
      return { relPath, anchor: "" };
    }
    return { relPath, anchor: `offset-${off}`, chunkOffset: off };
  }
  if (kind === "document") {
    // A document record IS a whole source file; its record_id is the
    // file_id, so it maps to the same `0-inputs/<file>` target a chunk
    // does — but with no offset, so clicking a document citation opens
    // the file at the top (no passage to highlight).
    return { relPath: `0-inputs/${recordId}`, anchor: "" };
  }
  if (kind === "fact" || kind === "pattern") {
    const sep = recordId.lastIndexOf(":");
    if (sep < 0) return null;
    const topic = recordId.slice(0, sep);
    const idx = Number(recordId.slice(sep + 1));
    if (!topic || !Number.isInteger(idx)) return null;
    const dir = kind === "fact" ? "1-facts" : "3-patterns";
    return { relPath: `${dir}/${topic}`, anchor: "", topic, idx };
  }
  if (kind === "insight") {
    const sep = recordId.lastIndexOf(":");
    if (sep < 0) return null;
    const scope = recordId.slice(0, sep);
    const idx = Number(recordId.slice(sep + 1));
    if (!Number.isInteger(idx)) return null;
    const slug = scope === "cross_domain" ? "cross-domain" : scope;
    return { relPath: "4-insights", anchor: `${slug}-${idx + 1}`, scope, idx };
  }
  if (kind === "action") {
    const idx = Number(recordId);
    if (!Number.isInteger(idx)) return null;
    return { relPath: "5-actions", anchor: `action-${idx + 1}`, idx };
  }
  if (kind === "entity") {
    return { relPath: `2-entities/${recordId}`, anchor: "" };
  }
  return null;
}

// The two top-level run-tree nodes that are FILES yet expand to reveal
// their items (each insight / each action renders as a nested entry).
// Named once here so the open-by-default seed, the expandable check,
// the per-item headings builder, and the file-view dispatch can't
// drift apart.
const INSIGHTS_FILE = "4-insights.md";
const ACTIONS_FILE = "5-actions.md";
const EXPANDABLE_TREE_FILES = [INSIGHTS_FILE, ACTIONS_FILE];

// Run-tree nodes eligible for default-open behaviour: every top-level
// dir except `0-inputs` (the source files the user just queued — they
// already know what's there), plus the expandable insights/actions
// files. The `2-entities` parent defaults open like facts/patterns so
// the per-`entity_type` breakdown + counts are visible without a click
// (#603); the type groups *themselves* stay closed — the user clicks a
// type to see its entities, exactly as they click a fact topic.
// `4-insights.md` / `5-actions.md` are top-level FILES, not dirs, but
// they expand to their items, so seed them open too — insights and
// actions are then visible on run-select without a click, matching the
// stage dirs. Returns `{ all, toOpen }`: `all` (incl. `0-inputs`) seeds
// `seenTopDirsRef` so the refresh path won't re-open a node the user
// explicitly closed; `toOpen` is `all` minus `0-inputs`.
function treeDefaultDirs(tree) {
  const dirs = tree.filter((n) => n.is_dir).map((n) => n.rel_path);
  const expandableFiles = tree
    .filter((n) => !n.is_dir && EXPANDABLE_TREE_FILES.includes(n.rel_path))
    .map((n) => n.rel_path);
  const all = [...dirs, ...expandableFiles];
  return { all, toOpen: all.filter((r) => r !== "0-inputs") };
}

// Run-view data is immutable per (run, kind, topic); cache fetches so a
// turn citing several entries in one topic hits the backend once.
const _citationDataCache = new Map();
function _citationData(runId, key, fetcher) {
  const ck = `${runId} ${key}`;
  if (!_citationDataCache.has(ck)) {
    // Evict on failure so a transient backend error doesn't stick a
    // rejected promise for the rest of the session; re-throw so the
    // caller still degrades to the file-level target.
    _citationDataCache.set(
      ck,
      fetcher().catch((e) => { _citationDataCache.delete(ck); throw e; }),
    );
  }
  return _citationDataCache.get(ck);
}

// Resolve a citation to its real navigation anchor + a human label
// (`kind · topic · title`). Upgrades fact/pattern to the exact
// type-scoped markdown anchor by loading the bound run's ordered topic
// list and reusing the canonical obsidianRenderer scheme — no
// re-derivation, no schema change. Every failure path (run not loaded,
// fetch error, index past end, unknown kind) degrades to the
// deterministic file-level target with a record_id label, so a
// citation is legible and never lands blank.
export async function resolveCitation(runId, kind, recordId, invoke) {
  const base = resourceNavTarget(kind, recordId);
  if (!base) return null;
  const out = {
    relPath: base.relPath,
    anchor: base.anchor,
    label: `${kind} · ${recordId}`,
  };
  // Carry the chunk's source char-offset through so the input view can
  // anchor + highlight the grounded passage (set by resourceNavTarget's
  // chunk branch; absent for every other kind).
  if (base.chunkOffset != null) out.chunkOffset = base.chunkOffset;
  if (!runId) return out;
  try {
    if (kind === "fact" || kind === "pattern") {
      const cmd = kind === "fact"
        ? "read_run_facts_for_topic"
        : "read_run_patterns_for_topic";
      const list = await _citationData(
        runId, `${kind}:${base.topic}`,
        () => invoke(cmd, { runId, topic: base.topic }),
      );
      if (Array.isArray(list) && base.idx < list.length) {
        const item = list[base.idx];
        const anchors = kind === "fact"
          ? computeFactAnchorsForTopic(list)
          : computePatternAnchorsForTopic(list);
        out.anchor = anchors[base.idx] || "";
        const title = kind === "fact"
          ? item?.summary
          : item?.name || item?.description;
        out.label = `${kind} · ${base.topic} · ${title || recordId}`;
      } else {
        // Index past the topic list (e.g. an empty topic in this run):
        // keep the file-head target, but carry the record_id so the
        // row isn't a bare "kind · topic" with no handle.
        out.label = `${kind} · ${recordId}`;
      }
    } else if (kind === "insight") {
      const ins = await _citationData(
        runId, "insights", () => invoke("read_run_insights", { runId }),
      );
      const list = base.scope === "cross_domain"
        ? ins?.cross_domain : ins?.critical;
      const title = list?.[base.idx]?.name || list?.[base.idx]?.description;
      out.label = `insight · ${title || recordId}`;
    } else if (kind === "action") {
      const act = await _citationData(
        runId, "actions", () => invoke("read_run_actions", { runId }),
      );
      const title = act?.actions?.[base.idx]?.recommendation;
      out.label = `action · ${title || recordId}`;
    } else if (kind === "entity") {
      const ents = await _citationData(
        runId, "entities", () => invoke("read_run_entities", { runId }),
      );
      // read_run_entities returns {subject, entities[], relations[]} —
      // the canonical record carries `canonical_name`.
      const ent = Array.isArray(ents?.entities)
        ? ents.entities.find((e) => e?.canonical_id === recordId) : null;
      out.label = `entity · ${ent?.canonical_name || recordId}`;
    } else if (kind === "chunk") {
      out.label = `chunk · ${base.relPath.replace(/^0-inputs\//, "")}`;
    }
  } catch {
    // Keep the deterministic file-level target + record_id label.
  }
  return out;
}

export default function App() {
  const [mode, setMode] = useState("local");
  const [inputs, setInputs] = useState([]);
  // Mirrors `inputs` for callbacks that need the current list without
  // re-creating per change (stageCandidates) — lets duplicate / new
  // detection run as a pure pre-state read instead of a side effect
  // inside the setInputs updater.
  const inputsRef = useRef(inputs);
  useEffect(() => {
    inputsRef.current = inputs;
  }, [inputs]);
  const [selectedInputs, setSelectedInputs] = useState(() => new Set());
  // The just-added input paths (computed against prior state, so it's
  // exactly the genuinely-new rows — re-adds of an existing path don't
  // count). StagingPane reads this on each `inputsHighlightTick` bump
  // to scroll the list to the new rows and flag them with the shared
  // transient-highlight. A ref (not state) because it's read at
  // effect-fire time, not rendered.
  const newlyAddedInputsRef = useRef([]);
  const [inputsHighlightTick, setInputsHighlightTick] = useState(0);
  // Per-path size cache for the staging list, keyed by absolute path.
  // Populated by stat_paths() after pickFiles / pickFolder, consumed
  // by StagingPane to render `<basename> · <kb|mb>` next to each row.
  // Missing entries render no size (the row still shows the basename).
  const [inputSizes, setInputSizes] = useState({});

  const [runs, setRuns] = useState([]);
  // 1Hz wallclock for the live elapsed-time labels (issue #159). The
  // hook gates its setInterval on `hasRunningRuns` so an idle UI
  // mounts no timer.
  const hasRunningRuns = useMemo(
    () => runs.some((r) => r.status === "running"),
    [runs],
  );
  const elapsedNow = useElapsedNow(hasRunningRuns);
  const prefersReducedMotion = usePrefersReducedMotion();
  // User-editable display aliases for runs. Keyed by run_id → string.
  // Persisted to config.run_aliases. The canonical short_id stays
  // unchanged; the alias is purely cosmetic.
  const [runAliases, setRunAliases] = useState({});
  // Resolve a run's user-visible NAME the same way the runs pane does:
  // the rename when one is set, else the 4-letter short_id. Shared with
  // the chat run selector so it labels runs identically (no second copy
  // of the rename rule, no dir-name re-parse). Empty string when the
  // run isn't in the current list and carries no short_id.
  const runShortById = useMemo(() => {
    const m = {};
    for (const r of runs) if (r.run_id) m[r.run_id] = r.short_id || "";
    return m;
  }, [runs]);
  const resolveRunName = useCallback(
    (runId) => runAliases[runId] || runShortById[runId] || "",
    [runAliases, runShortById],
  );
  // Finder-style multi-select for runs:
  //   - selectedRunIds : the set of currently-selected runs (any count)
  //   - currentRunId   : the last-clicked run, drives the right-pane
  //                      content view even when many are selected
  //   - runAnchorIndex : index of the last simple-click; shift+click
  //                      extends from this anchor to the clicked row
  const [selectedRunIds, setSelectedRunIds] = useState(() => new Set());
  const [currentRunId, setCurrentRunId] = useState(null);
  const [runAnchorIndex, setRunAnchorIndex] = useState(null);
  // Scrolls the run-list row into view after a programmatic selection
  // (run-again-from-a-run). Holds the target run id until its row has
  // mounted in `runsListRef`; the layout effect below consumes it.
  const runsListRef = useRef(null);
  const pendingSelectScrollRef = useRef(null);

  const [wizardOpen, setWizardOpen] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [confirmDialog, setConfirmDialog] = useState(null);
  // Run-details modal (issue #104). One open at a time; stores the
  // run_id whose per-call stats are being viewed, or null.
  const [detailsModalRunId, setDetailsModalRunId] = useState(null);
  // Rejected-files modal (issue #156). Holds the most recent batch of
  // rejected paths from validateInputs (picker / folder / drop). null
  // when no modal is open. Cleared on close; replaced when the next
  // staging action produces rejections.
  const [rejectedFiles, setRejectedFiles] = useState(null);
  // Privacy Level export gate. `privacyModal` holds the
  // run-ids being exported + per-level counts while the modal is open;
  // `successModal` holds the destination path of a finished export for
  // the Day-One-style confirmation. `exportDir` + `privacyLevel` are
  // session-stable, persisted into the global app config.
  const [privacyModal, setPrivacyModal] = useState(null);
  const [successModal, setSuccessModal] = useState(null);
  const [exportDir, setExportDir] = useState(null);
  // Directory of the most recent successful import, persisted into the
  // global app config under its own key. Kept distinct from exportDir
  // so an export between two imports doesn't move where the next
  // Add-file / Add-folder dialog opens.
  const [importDir, setImportDir] = useState(null);
  const [privacyLevel, setPrivacyLevel] = useState("insights");
  // True while a drag of file(s) is hovering the window. Drives the
  // drop-zone overlay. Tauri's webview swallows the OS drag events
  // before the browser sees them, so we can't rely on React's
  // onDragOver/onDrop on the root element — the truth comes from
  // getCurrentWebview().onDragDropEvent below.
  const [isDragOver, setIsDragOver] = useState(false);

  // Run-view tree + markdown preview state.
  const [runTree, setRunTree] = useState([]);
  const [openDirs, setOpenDirs] = useState(() => new Set());
  // Tracks which top-level dirs we've already auto-opened (or
  // explicitly skipped, in entities' case). Lets the tree-refresh
  // path open NEW dirs as they materialize during an in-flight run
  // without re-opening dirs the user explicitly closed. Reset on
  // selectedRun change.
  const seenTopDirsRef = useRef(new Set());
  // Selected file: { runId, relPath } in run-view, or null.
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileContent, setFileContent] = useState("");
  const [fileError, setFileError] = useState(null);
  // Bumped on every pipeline-progress burst. Per-stage view components
  // (FactsView, EntityView, …) take this as a prop and re-fetch their
  // structured data when it changes — so a file held open during an
  // in-flight stage updates as new data lands on disk. The 0-inputs
  // branch still goes through fileContent + read_run_file.
  const [refreshTick, setRefreshTick] = useState(0);
  // Mirror selectedFile into a ref so the pipeline-progress-driven
  // file refresh can re-read the right path without depending on a
  // closure snapshot. Updated below in a useEffect.
  const selectedFileRef = useRef(selectedFile);
  useEffect(() => {
    selectedFileRef.current = selectedFile;
  }, [selectedFile]);
  // Per-run facts cache. Keyed by run_id; the value is the flat list
  // returned from read_run_facts (each entry = {id, topic, item_type,
  // summary, evidence:[{file_path, file_offset, file_length}]}). Used
  // to overlay clickable <mark> elements on raw inputs (0-inputs/*).
  // Empty array means the run has no fact data on disk yet (early in
  // the pipeline, or intermediate/ was deleted) — the input view
  // gracefully degrades to plain text.
  const [runFacts, setRunFacts] = useState({});

  // Browser-style nav history for the markdown pane. Tracks the
  // sequence of files the user has opened in this session. Tree
  // clicks + wiki-link clicks push to history via navigateToFile;
  // the Back/Forward buttons in the markdown pane don't push — they
  // step the index and update selectedFile directly, bypassing the
  // history-push path so they can't create duplicate entries.
  const [fileHistory, setFileHistory] = useState([]);
  const [fileHistoryIdx, setFileHistoryIdx] = useState(-1);
  // navigateToFile: route every "open this file" through here so
  // history is the only entry point that updates selectedFile (apart
  // from the run-switch reset below). Truncates forward history if
  // the user navigates after going Back — same model as a browser.
  const navigateToFile = useCallback((file) => {
    setFileHistory((prev) => {
      const truncated = prev.slice(0, fileHistoryIdx + 1);
      // De-dupe on file+anchor: re-clicking the same anchor in the
      // same file shouldn't create a history entry, but jumping to a
      // different anchor in the same file IS a navigation step worth
      // recording (so back goes back to the previous fact).
      const last = truncated[truncated.length - 1];
      if (
        last &&
        last.runId === file.runId &&
        last.relPath === file.relPath &&
        (last.anchor || "") === (file.anchor || "")
      ) {
        return truncated;
      }
      const next = [...truncated, file];
      setFileHistoryIdx(next.length - 1);
      return next;
    });
    setSelectedFile(file);
  }, [fileHistoryIdx]);
  const fileNavBack = useCallback(() => {
    if (fileHistoryIdx <= 0) return;
    setFileHistoryIdx((i) => i - 1);
    setSelectedFile(fileHistory[fileHistoryIdx - 1]);
  }, [fileHistory, fileHistoryIdx]);
  const fileNavForward = useCallback(() => {
    if (fileHistoryIdx >= fileHistory.length - 1) return;
    setFileHistoryIdx((i) => i + 1);
    setSelectedFile(fileHistory[fileHistoryIdx + 1]);
  }, [fileHistory, fileHistoryIdx]);

  const [envFlags, setEnvFlags] = useState({
    tinfoil_key_set: false,
    tee_key_set: false,
    // Default-off: Local is unavailable until refreshEnv confirms the
    // backend is actually usable (downloaded model + supported OS for MLX).
    local_usable: false,
    local_disabled_reason: null,
  });

  const [pausingRunId, setPausingRunId] = useState(null);
  const [resumingRunId, setResumingRunId] = useState(null);
  const [startError, setStartError] = useState(null);
  const [configLoaded, setConfigLoaded] = useState(false);

  const [attestation, setAttestation] = useState(null);
  // Tracked separately from `attestation` so a Recheck doesn't unmount
  // the previous result's chain — the modal stays the same size while
  // verify_attestation is in flight; AttestationPanel just hides the
  // chain content via visibility:hidden.
  const [attestationChecking, setAttestationChecking] = useState(false);
  // Top-right attestation modal — opens on click + auto-opens on a
  // fresh ok→fail transition (per brief: "modal opens automatically
  // on red"). The auto-open only fires once per fail; the user can
  // close it without it bouncing back.
  const [attModalOpen, setAttModalOpen] = useState(false);
  const lastAttOkRef = useRef(null);

  // Shared JS entrypoint for sanctioned attestation call sites #1
  // (app-startup verify) and #2 (Settings re-check button) — both
  // invoke the one `verify_attestation` command. Site #3 (the hourly
  // background re-attest) lives Rust-side and reuses the same command
  // path. Attestation runs from exactly these three places and nowhere
  // else; do not add other invocations of `verify_attestation`.
  const refreshAttestation = useCallback(async () => {
    setAttestationChecking(true);
    try {
      const r = await invoke("verify_attestation");
      setAttestation(r);
    } catch (e) {
      setAttestation({
        provider: "",
        model: "",
        ok: false,
        fingerprint: null,
        error: String(e?.message || e),
        ts: 0,
      });
    } finally {
      setAttestationChecking(false);
    }
  }, []);

  const refreshEnv = useCallback(async () => {
    try {
      const [s, cfg, localStat] = await Promise.all([
        invoke("get_settings"),
        invoke("get_config").catch(() => ({})),
        invoke("local_model_status").catch(() => null),
      ]);
      const localMode =
        cfg && typeof cfg === "object" ? cfg.local_setup_mode || null : null;
      const backend = String(
        (cfg && typeof cfg === "object" && cfg.local_backend) || "mlx",
      )
        .trim()
        .toLowerCase();
      const osSupported = !!localStat?.os_supported;
      const localUsable = localModeUsable({
        setupMode: localMode,
        backend,
        modelDownloaded: !!localStat?.downloaded,
        osSupported,
      });
      // Why Local is greyed out, computed where the backend + status are in
      // hand. The OS floor only applies to the bundled MLX path; Ollama and
      // the other reasons point the user to Settings to finish setup.
      const localDisabledReason = localUsable
        ? null
        : backend !== "ollama" && !osSupported
        ? "On-device model isn't available on this system."
        : "Set up a local model in Settings to enable Local mode.";
      // Private Cloud is available when EITHER the user entered their own
      // Tinfoil key OR the build carries a bundled one — mirrors the Rust
      // effective-key precedence (user key first, bundled fallback). Without
      // the bundled half, a keyed build with no user key wrongly shows
      // Private Cloud "(disabled)" and the fallback below kicks the user off
      // tee, even though inference would work.
      const teeKeySet = !!s.tinfoil_key_set || !!s.bundled_key_set;
      setEnvFlags({
        tinfoil_key_set: !!s.tinfoil_key_set,
        tee_key_set: teeKeySet,
        local_usable: localUsable,
        local_disabled_reason: localDisabledReason,
      });
      const localAvailable = localUsable;
      const desired = mode;
      let fallback = null;
      if (desired === "tee" && !teeKeySet) fallback = "local";
      else if (desired === "local" && !localAvailable) fallback = "tee";
      if (fallback) {
        if (fallback === "local" && !localAvailable) {
          fallback = teeKeySet ? "tee" : null;
        } else if (fallback === "tee" && !teeKeySet) {
          fallback = null;
        }
        if (fallback) setMode(fallback);
      }
    } catch (e) {
      console.error("get_settings failed:", e);
    }
  }, [mode]);

  // Suspends refreshRuns for the duration of a click on the runs pane.
  // Without this, the 500ms poll + per-LLM-call pipeline-progress
  // event re-render RunRow's children mid-click; mousedown lands on a
  // span that gets detached before mouseup, and the browser drops the
  // click. Scoped to the active mousedown→mouseup window only — idle
  // hover does NOT suspend refreshes (an earlier mouseEnter/Leave
  // shape held this flag for the entire time the cursor sat in the
  // pane, which froze the run row's progress bar for fast cached
  // runs where the user naturally hovered the pane to watch the new
  // row tick).
  const hoveringRunsRef = useRef(false);

  const refreshRuns = useCallback(async () => {
    ltrace("refreshruns_fired");
    try {
      const list = await invoke("list_runs");
      ltrace("runs_received");
      setRuns(Array.isArray(list) ? list : []);
    } catch (e) {
      console.error("list_runs failed:", e);
    }
  }, []);

  // Finder-style row click. Modifier semantics:
  //   plain click  : select only this; reset anchor to this row
  //   cmd/ctrl     : toggle this row in the existing selection;
  //                  move anchor to this row
  //   shift        : range-select from runAnchorIndex to this row,
  //                  UNIONED with the existing selection (Finder
  //                  preserves cmd-click picks when you then
  //                  shift-click; clobbering them was a bug)
  // currentRunId always tracks the *last clicked* row, even on cmd-
  // click that deselects — that way the right-pane content view
  // follows what the user just touched.
  const handleSelectRun = useCallback(
    (run, index, e) => {
      // dev_tracing: marks the moment JS processes the row click.
      // Pair with `row_rendered` to distinguish "click event arrived
      // promptly" from "JS thread blocked, click queued for seconds."
      // If row_clicked lands within ms of the user's click, the JS
      // thread is responsive and any perceived freeze is downstream
      // (Rust invokes, Tauri runtime, native compositor); if it
      // lands seconds late, the WebView itself was contended.
      ltrace("row_clicked");
      const shift = !!e?.shiftKey;
      const cmd = !!(e?.metaKey || e?.ctrlKey);
      if (shift && runAnchorIndex !== null) {
        const start = Math.min(runAnchorIndex, index);
        const end = Math.max(runAnchorIndex, index);
        setSelectedRunIds((prev) => {
          // Union prior selection with the range. Finder behavior:
          // a cmd-click pick at row 1 + shift-click at row 5 should
          // yield {1, 2, 3, 4, 5} — not {2, 3, 4, 5} as the previous
          // shape produced. The anchor stays put for further shift-
          // click extension.
          const next = new Set(prev);
          for (let i = start; i <= end; i++) {
            const r = runs[i];
            if (r) next.add(r.run_id);
          }
          return next;
        });
      } else if (cmd) {
        setSelectedRunIds((prev) => {
          const next = new Set(prev);
          if (next.has(run.run_id)) next.delete(run.run_id);
          else next.add(run.run_id);
          return next;
        });
        setRunAnchorIndex(index);
      } else {
        // Plain click. If this row is already the SOLE selection,
        // toggle off → back to staging mode (user feedback: clicking
        // the lone selected run should release it). Cmd+click is
        // still the way to add/remove from a multi-row selection.
        if (
          selectedRunIds.size === 1 &&
          selectedRunIds.has(run.run_id)
        ) {
          setSelectedRunIds(new Set());
          setCurrentRunId(null);
          setRunAnchorIndex(null);
          return;
        }
        setSelectedRunIds(new Set([run.run_id]));
        setRunAnchorIndex(index);
      }
      setCurrentRunId(run.run_id);
    },
    [runs, runAnchorIndex, selectedRunIds]
  );

  // Cmd+A / Ctrl+A: select every visible run. Skipped when focus is
  // anywhere a user might be typing — INPUT/TEXTAREA/SELECT or a
  // contenteditable subtree (e.g. the rename-run inline editor below,
  // or a future search box). Without the contenteditable check we'd
  // hijack cmd+A in any rich-text field that's added later.
  useEffect(() => {
    function onKey(e) {
      if (!(e.metaKey || e.ctrlKey)) return;
      if (e.key !== "a" && e.key !== "A") return;
      const el = document.activeElement;
      if (el) {
        const tag = el.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        if (el.isContentEditable) return;
      }
      if (runs.length === 0) return;
      e.preventDefault();
      setSelectedRunIds(new Set(runs.map((r) => r.run_id)));
      setCurrentRunId(runs[runs.length - 1].run_id);
      setRunAnchorIndex(runs.length - 1);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [runs]);

  // Rename the display label of a run. Empty / whitespace-only input
  // clears the alias (back to "Run #shortId"). Persists immediately
  // to config.run_aliases — re-using the same merge-then-set_config
  // pattern as the inputs/mode persistence effect.
  const renameRun = useCallback((runId, label) => {
    const trimmed = (label || "").trim();
    setRunAliases((prev) => {
      const next = { ...prev };
      if (trimmed) next[runId] = trimmed;
      else delete next[runId];
      invoke("update_config", { patch: { run_aliases: next } }).catch((e) =>
        console.error("update_config (run_aliases):", e)
      );
      return next;
    });
  }, []);

  function clearRunSelection() {
    setSelectedRunIds(new Set());
    setCurrentRunId(null);
    setRunAnchorIndex(null);
  }

  // Bring a programmatically-selected run into view. `block: "nearest"`
  // is a no-op when the row is already visible (the common case — a new
  // run lands at the top of the list), so it never jumps the list when
  // it doesn't need to. Re-runs on every `runs` change until the target
  // row has mounted, then clears the pending id.
  useLayoutEffect(() => {
    const id = pendingSelectScrollRef.current;
    if (!id) return;
    const row = runsListRef.current?.querySelector(
      `[data-run-id="${id}"]`,
    );
    if (!row) return;
    row.scrollIntoView({ block: "nearest" });
    pendingSelectScrollRef.current = null;
  }, [runs]);

  // The "current" run drives the middle/right pane content. Derived
  // from currentRunId so it auto-resolves when runs refreshes (e.g. a
  // poll tick picks up freshly-derived state from llm-calls.jsonl).
  // When the current run is deleted from disk, the lookup returns
  // null and the panes fall back to staging mode.
  const selectedRun = useMemo(
    () => (currentRunId ? runs.find((r) => r.run_id === currentRunId) || null : null),
    [currentRunId, runs]
  );

  // When the run's vault has exactly one file under 0-inputs/, every
  // evidence span on every fact references it (the pipeline tends to
  // emit a canonical "input.txt" or similar in evidence file_path,
  // independent of the user's filename on disk). Pre-compute the
  // sole-input rel path so InputFileView can use it as a fallback
  // match — without this, the strict file_path equality fails on
  // single-file runs with non-canonical input names.
  const soleInputFile = useMemo(() => {
    const inputsDir = (runTree || []).find(
      (n) => n.is_dir && n.rel_path === "0-inputs",
    );
    if (!inputsDir) return null;
    const files = (inputsDir.children || []).filter((c) => !c.is_dir);
    return files.length === 1 ? files[0].rel_path : null;
  }, [runTree]);

  // Wiki-link + fact-source navigation: every "open this file" click
  // on a fact's source quote, a provenance tree leaf, or an inline
  // wikilink lands here with the user-facing filename. The pipeline
  // writes Stage 0 output as `<name>.md` for ALL files (e.g.
  // `barbellion-disappointed-man.txt.md`, `README.md.md`) but the
  // click target carries the original name. `normalizeNavRelPath`
  // maps it onto the on-disk shape. The anchor (with `^` already
  // stripped by splitWikiTarget) flows through to the right pane,
  // which scrolls to the matching element after the file renders.
  const handleMarkdownNavigate = useCallback(
    ({ runId, relPath, anchor, chunkOffset, chunkLen }) => {
      if (!runId || !relPath) return;
      const normalized = normalizeNavRelPath(relPath);
      if (selectedRun?.run_id !== runId) {
        const target = runs.find((r) => r.run_id === runId);
        if (!target) return; // run isn't loaded; nothing to navigate to
        setSelectedRunIds(new Set([runId]));
        setCurrentRunId(runId);
      }
      navigateToFile({
        runId,
        relPath: normalized,
        anchor: anchor || "",
        // Only chunk citations carry this; it tells the input view to
        // mark + highlight the grounded passage at this source offset.
        ...(chunkOffset != null ? { chunkOffset } : {}),
        // The chunk's raw char length, when the embeddings run
        // persisted it: lets the input view highlight the whole chunk
        // span instead of approximating to the paragraph end.
        ...(chunkLen != null ? { chunkLen } : {}),
      });
    },
    [runs, selectedRun, navigateToFile]
  );

  // External URL clicks: `<a target="_blank">` does nothing inside a
  // Tauri webview (no system browser handoff). Route through the
  // opener plugin instead. Anchor (`#…`) clicks are NOT routed here —
  // those scroll within the rendered document via heading slug ids.
  const handleOpenUrl = useCallback((url) => {
    openUrl(url).catch((e) => console.error("openUrl:", e));
  }, []);

  // Initial bootstrap.
  useEffect(() => {
    refreshEnv();
    refreshRuns();
    // Sanctioned attestation call site #1: app-startup verify.
    refreshAttestation();
    // First-run gate + config/settings, resolved together so the
    // wizard variant is known before it opens. On a build that carries
    // a bundled Tinfoil key, a fresh user gets the name-only Easy
    // Wizard (so onboarding never looks cloud-shaped); keyless builds
    // fall back to the regular key-asking wizard. Resolving these in
    // one pass also avoids briefly opening the regular wizard before
    // the bundled-key flag arrives.
    Promise.all([
      invoke("needs_wizard").catch(() => false),
      invoke("get_config").catch(() => ({})),
      invoke("get_settings").catch(() => ({})),
    ])
      .then(([needsWiz, cfg, s]) => {
        setWizardOpen(
          needsWiz ? { firstRun: true, easy: !!s?.bundled_key_set } : false
        );
        const cfgSafe = cfg && typeof cfg === "object" ? cfg : {};
        if (Array.isArray(cfgSafe.inputs)) {
          setInputs(cfgSafe.inputs);
          fetchSizesFor(cfgSafe.inputs);
        }
        if (cfgSafe.run_aliases && typeof cfgSafe.run_aliases === "object") {
          setRunAliases(cfgSafe.run_aliases);
        }
        if (typeof cfgSafe.export_dir === "string" && cfgSafe.export_dir) {
          setExportDir(cfgSafe.export_dir);
        }
        if (typeof cfgSafe.import_dir === "string" && cfgSafe.import_dir) {
          setImportDir(cfgSafe.import_dir);
        }
        if (PRIVACY_SLIDER.some((s) => s.level === cfgSafe.privacy_level)) {
          setPrivacyLevel(cfgSafe.privacy_level);
        }
        if (typeof cfgSafe.mode === "string" && MODE_META[cfgSafe.mode]) {
          setMode(cfgSafe.mode);
        } else {
          const hasTee = !!s?.tinfoil_key_set || !!s?.bundled_key_set;
          if (hasTee) setMode("tee");
        }
      })
      .catch((e) => {
        console.error("bootstrap:", e);
        setWizardOpen(false);
      })
      .finally(() => setConfigLoaded(true));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!configLoaded) return;
    let cancelled = false;
    invoke("update_config", { patch: { inputs, mode } }).catch((e) => {
      if (!cancelled) console.error("update_config (inputs/mode):", e);
    });
    return () => {
      cancelled = true;
    };
  }, [inputs, mode, configLoaded]);

  useEffect(() => {
    // ci-allow:listen-guard — handler only flips a boolean modal flag;
    // a StrictMode double-subscribe double-sets the same idempotent
    // state, with no doubled-append risk like a streaming listener.
    let unlisten;
    listen("open-settings", () => setSettingsOpen(true)).then((u) => {
      unlisten = u;
    });
    return () => unlisten?.();
  }, []);

  useEffect(() => {
    // ci-allow:listen-guard — handler only flips idempotent modal
    // flags; a StrictMode double-subscribe is benign here (no streaming
    // append).
    let unlisten;
    listen("open-wizard", () => {
      setSettingsOpen(false);
      setWizardOpen({ firstRun: false });
    }).then((u) => {
      unlisten = u;
    });
    return () => unlisten?.();
  }, []);

  useEffect(() => {
    // Name-only Easy Wizard (issue #609). Same idempotent-modal-flag
    // pattern as open-wizard; the menu entry only exists on a build
    // with a bundled key, so this fires only where the flow is viable.
    // ci-allow:listen-guard
    let unlisten;
    listen("open-easy-wizard", () => {
      setSettingsOpen(false);
      setWizardOpen({ firstRun: false, easy: true });
    }).then((u) => {
      unlisten = u;
    });
    return () => unlisten?.();
  }, []);

  useEffect(() => {
    // Hourly background re-attest push (#926). The Rust timer re-runs
    // attestation every hour and emits the fresh result; without this
    // the indicator would freeze at the startup verify and read as
    // "stale". Idempotent state set — a StrictMode double-subscribe just
    // re-sets the same object. ci-allow:listen-guard
    let unlisten;
    listen("attestation-updated", (event) => {
      if (event?.payload) setAttestation(event.payload);
    }).then((u) => {
      unlisten = u;
    });
    return () => unlisten?.();
  }, []);

  // Auto-open the attestation modal on the first ok→fail transition.
  // Tracking the previous ok-state via a ref (not state) keeps the
  // open-once semantic from regressing on mode switches that re-fetch
  // the same fail.
  useEffect(() => {
    if (attestation === null) return;
    const prev = lastAttOkRef.current;
    const now = !!attestation.ok;
    if (prev === true && !now) setAttModalOpen(true);
    lastAttOkRef.current = now;
  }, [attestation]);

  // Bounded auto-recheck on a transient (recoverable) attestation
  // result. The Python side already retried the sigstore TUF-race
  // under a cross-process lock; by the time it returns `transient`
  // the racing sibling process is gone, so a short-delayed fresh
  // verify almost always recovers. Capped (and never auto-rechecks a
  // real ✗ failure) so a genuine prolonged outage doesn't spin.
  const attTransientRetriesRef = useRef(0);
  useEffect(() => {
    if (attestation === null) return;
    if (attestation.ok || !attestation.transient) {
      attTransientRetriesRef.current = 0;
      return;
    }
    if (attTransientRetriesRef.current >= 3) return;
    const n = (attTransientRetriesRef.current += 1);
    const t = setTimeout(refreshAttestation, 1500 * n);
    return () => clearTimeout(t);
  }, [attestation, refreshAttestation]);

  const activeModes = new Set(
    runs.filter((r) => r.status === "running").map((r) => r.mode).filter(Boolean)
  );
  const hasActive = activeModes.size > 0;
  // While any run is active, poll the runs list on a fixed cadence so
  // status/progress stays live even through a quiet stretch with no
  // pipeline-progress events. The poll funnels through the SAME
  // coalescer as the event path (`scheduleRefreshersRef.current.runs`)
  // instead of calling `refreshRuns()` directly. The shared single
  // in-flight timer then dedupes a poll tick against an event burst,
  // so `list_runs` fires at most once per coalescing window no matter
  // how many sources asked. Calling `refreshRuns()` directly here was
  // the defect: the poll and the event coalescer issued independent,
  // uncoordinated `list_runs` round-trips on every tick (~2/s of poll
  // refreshes stacked on top of the event-driven ones), each a full
  // Rust round-trip + runs-pane re-render. Hover-skip is enforced
  // inside the coalescer's timer at fire-time, so it's dropped here.
  useEffect(() => {
    if (!hasActive) return;
    const id = setInterval(() => {
      scheduleRefreshersRef.current.runs();
    }, 500);
    return () => clearInterval(id);
  }, [hasActive]);

  // Coalesce pipeline-progress events. The runner fires one event per
  // JSON pipeline line (LLM-call boundary, stage transition, live-token
  // heartbeat); with parallel batches in flight that can be 10+/sec.
  // Each event without throttling re-fetches list_runs, the run tree,
  // and re-renders every RunRow — which made the app feel frozen
  // during long runs. Throttle approach: a single in-flight timer per
  // refresher; subsequent events (and poll ticks — see above) while
  // the timer is queued are dropped. One coalesced refresh per window
  // serves every source.
  const treeRefreshTimerRef = useRef(null);
  const runsRefreshTimerRef = useRef(null);
  const factsRefreshTimerRef = useRef(null);
  const fileRefreshTimerRef = useRef(null);
  // Hold a stable ref to the currently-selected run_id so the timer
  // body can re-check at fire-time instead of relying on whatever
  // run was selected when the closure was scheduled. Without this,
  // a pipeline-progress event for run A schedules a refresh, the
  // user switches to run B before the 500ms timer fires, the timer
  // fires against A's run_id, and the result clobbers B's tree.
  const selectedRunIdRef = useRef(selectedRun?.run_id ?? null);
  useEffect(() => {
    selectedRunIdRef.current = selectedRun?.run_id ?? null;
  }, [selectedRun?.run_id]);
  const scheduleTreeRefresh = useCallback(() => {
    if (treeRefreshTimerRef.current) return;
    treeRefreshTimerRef.current = setTimeout(() => {
      treeRefreshTimerRef.current = null;
      const runId = selectedRunIdRef.current;
      if (!runId) return;
      invoke("list_run_tree", { runId })
        .then((t) => {
          // Re-check at resolve-time: if the user switched runs while
          // the fetch was in flight, drop the stale result.
          if (selectedRunIdRef.current !== runId) return;
          const tree = Array.isArray(t) ? t : [];
          setRunTree(tree);
          // Auto-open any NEWLY-APPEARED top-level dir. On in-flight
          // runs the tree gains 1-facts, 3-patterns, the 2-entities
          // parent, etc. as each stage materializes; without this
          // they'd render collapsed because only the initial-select
          // effect seeded openDirs. (Type groups nested under
          // 2-entities are NOT auto-opened — they default closed like
          // fact topics; treeDefaultDirs only walks top-level dirs.)
          // `seenTopDirsRef` tracks which dirs we've already
          // auto-handled, so a dir the user explicitly closes after
          // auto-open stays closed across refreshes.
          const seen = seenTopDirsRef.current;
          const { all } = treeDefaultDirs(tree);
          const newlyArrived = all.filter((r) => !seen.has(r));
          if (newlyArrived.length > 0) {
            for (const r of newlyArrived) seen.add(r);
            const toOpen = newlyArrived.filter((r) => r !== "0-inputs");
            if (toOpen.length > 0) {
              setOpenDirs((prev) => {
                const next = new Set(prev);
                for (const r of toOpen) next.add(r);
                return next;
              });
            }
          }
        })
        .catch(() => {});
    }, 500);
  }, []);
  const scheduleRunsRefresh = useCallback(() => {
    if (runsRefreshTimerRef.current) {
      ltrace("scheduleruns_skipped_pending");
      return;
    }
    ltrace("scheduleruns_armed");
    runsRefreshTimerRef.current = setTimeout(() => {
      runsRefreshTimerRef.current = null;
      ltrace(
        hoveringRunsRef.current
          ? "scheduleruns_fired_hovering"
          : "scheduleruns_fired",
      );
      if (!hoveringRunsRef.current) refreshRuns();
    }, 250);
  }, [refreshRuns]);
  // Re-fetch runFacts for the in-flight run on each pipeline-progress
  // burst. read_run_facts now reads the per-topic JSONL bucket files
  // directly (in-flight + canonical), so this lights up InputFileView
  // marks as new facts arrive during extract.
  const scheduleFactsRefresh = useCallback(() => {
    if (factsRefreshTimerRef.current) return;
    factsRefreshTimerRef.current = setTimeout(() => {
      factsRefreshTimerRef.current = null;
      const runId = selectedRunIdRef.current;
      if (!runId) return;
      invoke("read_run_facts", { runId })
        .then((facts) => {
          if (selectedRunIdRef.current !== runId) return;
          setRunFacts((prev) => ({
            ...prev,
            [runId]: Array.isArray(facts) ? facts : [],
          }));
        })
        .catch(() => {});
    }, 500);
  }, []);
  // Refresh the open file on each pipeline-progress burst. Stage views
  // (FactsView, EntityView, …) react to refreshTick by re-fetching
  // their structured data from per-stage Tauri commands. Raw inputs
  // still flow through read_run_file → fileContent.
  const scheduleFileRefresh = useCallback(() => {
    if (fileRefreshTimerRef.current) return;
    fileRefreshTimerRef.current = setTimeout(() => {
      fileRefreshTimerRef.current = null;
      const file = selectedFileRef.current;
      if (!file) return;
      // Bump for any open file: stage views consume this to re-fetch.
      setRefreshTick((t) => t + 1);
      if (!/^0-inputs\//.test(file.relPath)) return;
      invoke("read_run_file", {
        runId: file.runId,
        relPath: file.relPath,
      })
        .then((text) => {
          // Drop stale results if the user clicked away mid-fetch.
          const live = selectedFileRef.current;
          if (!live || live.runId !== file.runId || live.relPath !== file.relPath) {
            return;
          }
          setFileContent(typeof text === "string" ? text : "");
        })
        .catch(() => {});
    }, 500);
  }, []);
  useEffect(() => () => {
    if (treeRefreshTimerRef.current) {
      clearTimeout(treeRefreshTimerRef.current);
      treeRefreshTimerRef.current = null;
    }
    if (runsRefreshTimerRef.current) {
      clearTimeout(runsRefreshTimerRef.current);
      runsRefreshTimerRef.current = null;
    }
    if (factsRefreshTimerRef.current) {
      clearTimeout(factsRefreshTimerRef.current);
      factsRefreshTimerRef.current = null;
    }
    if (fileRefreshTimerRef.current) {
      clearTimeout(fileRefreshTimerRef.current);
      fileRefreshTimerRef.current = null;
    }
  }, []);

  // Stable handler for `pipeline-progress` so the listen() registration
  // only happens once at mount. Earlier this effect listed
  // `scheduleTreeRefresh` in its deps; every time selectedRun changed,
  // the schedule callback was re-created, this effect re-ran, the old
  // listener was torn down and a new one registered. listen() is async,
  // so during the gap the user could miss progress events for the
  // newly-selected in-flight run — symptom: "live updates not
  // consistently appearing." Reading through refs sidesteps the race.
  const scheduleRefreshersRef = useRef({
    runs: scheduleRunsRefresh,
    tree: scheduleTreeRefresh,
    facts: scheduleFactsRefresh,
    file: scheduleFileRefresh,
  });
  useEffect(() => {
    scheduleRefreshersRef.current = {
      runs: scheduleRunsRefresh,
      tree: scheduleTreeRefresh,
      facts: scheduleFactsRefresh,
      file: scheduleFileRefresh,
    };
  }, [scheduleRunsRefresh, scheduleTreeRefresh, scheduleFactsRefresh, scheduleFileRefresh]);
  useEffect(() => {
    let unlisten;
    let cancelled = false;
    ltrace("pipeline_progress_listener_registering");
    listen("pipeline-progress", () => {
      ltrace("pipeline_progress_event");
      // All refreshers go through their own timer-coalesce; no
      // direct invoke fires from the event handler itself.
      scheduleRefreshersRef.current.runs();
      scheduleRefreshersRef.current.tree();
      scheduleRefreshersRef.current.facts();
      scheduleRefreshersRef.current.file();
    }).then((u) => {
      ltrace("pipeline_progress_listener_registered");
      if (cancelled) {
        u?.();
      } else {
        unlisten = u;
      }
    });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  // Reset middle + right pane state on every selectedRun change so
  // the previous run's tree, file-tree expansion, selected file, and
  // back/forward history can't bleed into the new run's view. Keyed
  // on `selectedRun?.run_id` (not the object reference) so unrelated
  // re-renders that produce a fresh `selectedRun` object via
  // `useMemo` don't re-fire. Cancels any pending tree-refresh timer
  // queued against the prior run before kicking off the fresh fetch.
  //
  // Must be useLayoutEffect, not useEffect. The reset state writes
  // need to land BEFORE the browser paints the click-induced render;
  // a useEffect runs post-paint, so the run-switch click commits a
  // frame where selectedRun = B but selectedFile, runTree, and
  // fileContent still reference run A. On real Tauri (WKWebView with
  // unbatched paint cadence) that frame is observable as A's tree +
  // A's markdown body bleeding into the new run's panes — exactly
  // the "Never run A's tree, even briefly" / "Never auto-renders run
  // A's previously-selected file" wording in the issue. Vitest's
  // act() wrapper masks this because it flushes effects inside
  // fireEvent / userEvent.click, so behavioural tests pass under
  // either timing.
  useLayoutEffect(() => {
    if (treeRefreshTimerRef.current) {
      clearTimeout(treeRefreshTimerRef.current);
      treeRefreshTimerRef.current = null;
    }
    setRunTree([]);
    setOpenDirs(new Set());
    seenTopDirsRef.current = new Set();
    // Anti-bleed (issue #83): a manual run-switch must drop the prior
    // run's open file. But a chat citation into a DIFFERENT (bound)
    // run switches the selected run AND sets `selectedFile` to a
    // target that already belongs to the run we're switching TO, in
    // the same commit — that file is the intended destination, not
    // stale bleed. Clearing it there is the cross-run "file lands
    // empty" defect. Only drop the file when it belongs to a
    // different run than the one now selected; a genuine manual
    // switch still clears (its file points at the OLD run).
    if (selectedFile && selectedFile.runId !== selectedRun?.run_id) {
      setSelectedFile(null);
    }
    setFileHistory([]);
    setFileHistoryIdx(-1);
    if (!selectedRun) return;
    let cancelled = false;
    invoke("list_run_tree", { runId: selectedRun.run_id })
      .then((t) => {
        if (cancelled) return;
        const tree = Array.isArray(t) ? t : [];
        setRunTree(tree);
        // Default every top-level dir open EXCEPT inputs (the source
        // files the user already knows about — they just queued the
        // run). The 2-entities parent now auto-opens like
        // facts/patterns so the per-type breakdown + counts show
        // without a click (#603); the type groups under it stay
        // closed (user clicks a type, like a fact topic). Seed
        // seenTopDirsRef with every top-level dir we saw at first
        // fetch (incl. inputs + the entities parent), so the refresh
        // path only auto-opens dirs that arrive LATER (in-flight
        // runs) and never re-opens one the user explicitly closed.
        const { all, toOpen } = treeDefaultDirs(tree);
        seenTopDirsRef.current = new Set(all);
        setOpenDirs(new Set(toOpen));
      })
      .catch(() => {
        if (!cancelled) setRunTree([]);
      });
    // Fetch the run's facts on selection. For finished runs the cache
    // makes re-clicks instant; for in-flight runs we always re-fetch
    // so the InputFileView marks pick up newly-extracted facts. The
    // pipeline-progress-driven scheduleFactsRefresh keeps it fresh
    // while the user holds this run selected.
    const isRunning = selectedRun.status === "running";
    if (isRunning || !(selectedRun.run_id in runFacts)) {
      invoke("read_run_facts", { runId: selectedRun.run_id })
        .then((facts) => {
          if (cancelled) return;
          setRunFacts((prev) => ({
            ...prev,
            [selectedRun.run_id]: Array.isArray(facts) ? facts : [],
          }));
        })
        .catch(() => {
          if (cancelled) return;
          setRunFacts((prev) => ({ ...prev, [selectedRun.run_id]: [] }));
        });
    }
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRun?.run_id]);

  // Load file content when the user clicks a file in the run tree.
  // Only 0-inputs/* paths flow through read_run_file → fileContent.
  // Stage views (1-facts, 2-entities, 3-patterns, 4-insights, 5-actions)
  // fetch their own structured data via per-stage Tauri commands —
  // see MarkdownPreview's dispatch.
  useEffect(() => {
    if (!selectedFile) {
      setFileContent("");
      setFileError(null);
      return;
    }
    if (!/^0-inputs\//.test(selectedFile.relPath)) {
      setFileContent("");
      setFileError(null);
      return;
    }
    let cancelled = false;
    invoke("read_run_file", {
      runId: selectedFile.runId,
      relPath: selectedFile.relPath,
    })
      .then((text) => {
        if (cancelled) return;
        setFileContent(typeof text === "string" ? text : "");
        setFileError(null);
      })
      .catch((e) => {
        if (cancelled) return;
        setFileContent("");
        setFileError(String(e?.message || e));
      });
    return () => {
      cancelled = true;
    };
  }, [selectedFile]);

  // Fetch size for newly-added paths and merge into the inputSizes
  // cache. Failures (missing path, permission error) collapse to 0
  // server-side so the UI never blanks out — a missing entry just
  // omits the size span on that row.
  const fetchSizesFor = useCallback(async (paths) => {
    if (!paths.length) return [];
    try {
      const results = await invoke("stat_paths", { paths });
      setInputSizes((prev) => {
        const next = { ...prev };
        for (const r of results || []) {
          if (r && typeof r.path === "string") next[r.path] = r.size_bytes ?? 0;
        }
        return next;
      });
      return results || [];
    } catch {
      // stat_paths is purely cosmetic — silently skip on error.
      return [];
    }
  }, []);

  // Stage-time validation common to all three entry paths (picker,
  // folder, drop). Stat the candidates so the size check runs against
  // real bytes, then filter through validateInputs. Accepted paths
  // merge into the staging list; rejected paths populate the modal
  // (issue #156). Centralizing here keeps the three entry points in
  // sync — a new rejection rule lands once and applies everywhere.
  const stageCandidates = useCallback(async (candidates, { folder = false } = {}) => {
    if (!candidates?.length) return;
    const stats = await fetchSizesFor(candidates);
    // stat_paths returns [{ path, size_bytes }, …]. Build a lookup so
    // validateInputs sees the same size the staging row will show, and
    // fall back to undefined when a stat result is missing — the
    // validator skips the size check in that case (still classifies on
    // extension/system rules so we don't leak `.DS_Store` into staging
    // when stat_paths failed).
    const sizeByPath = new Map();
    for (const r of Array.isArray(stats) ? stats : []) {
      if (r && typeof r.path === "string") sizeByPath.set(r.path, r.size_bytes);
    }
    const pathsWithSizes = candidates.map((p) => ({ path: p, size: sizeByPath.get(p) }));
    const { accepted, rejected } = validateInputs(pathsWithSizes);
    // Split accepted into genuinely-new vs already-staged. Read prior
    // state via inputsRef so this is a pure pre-state computation (no
    // side effect inside the setInputs updater). Paths already in the
    // list surface in the "already imported" modal group; within-batch
    // repeats of a not-yet-staged path collapse silently to one row.
    const existing = new Set(inputsRef.current);
    const seen = new Set(existing);
    const added = [];
    const dupes = [];
    for (const p of accepted) {
      if (existing.has(p)) {
        dupes.push(p);
      } else if (!seen.has(p)) {
        seen.add(p);
        added.push(p);
      }
    }
    if (added.length) {
      newlyAddedInputsRef.current = added;
      // Append in arrival order so display-order == append-order: the
      // new rows land contiguously at the bottom, which StagingPane
      // scrolls to and transient-highlights. The defensive dedupe
      // guards a stale-ref race without re-introducing a side effect.
      setInputs((prev) => {
        const s = new Set(prev);
        const fresh = added.filter((p) => !s.has(p));
        return fresh.length ? [...prev, ...fresh] : prev;
      });
      // Only signal scroll+highlight when at least one file is
      // genuinely new — a pure-duplicate batch leaves the list put.
      setInputsHighlightTick((t) => t + 1);
    }
    const notAdded = dupes.length
      ? [
          ...rejected,
          ...dupes.map((p) => ({ path: p, reason: "already imported" })),
        ]
      : rejected;
    // The "X imported" summary is only meaningful for a folder load
    // (bulk, opaque) — a file pick is explicit about what was chosen.
    if (notAdded.length) {
      setRejectedFiles({ items: notAdded, imported: added.length, folder });
    }
  }, [fetchSizesFor]);

  // Record the source directory of a successful import. update_config
  // merges import_dir server-side under a lock, so an export persisting
  // its own keys between two imports can't drop this one. Last-wins: a
  // single slot, overwritten by each import.
  const persistImportDir = useCallback(
    (dir) => {
      if (!dir || dir === importDir) return;
      setImportDir(dir);
      invoke("update_config", { patch: { import_dir: dir } }).catch((e) =>
        console.error("update_config (import dir):", e)
      );
    },
    [importDir]
  );

  // ── Input picker ──────────────────────────────────────────────────────────
  async function pickFiles() {
    const sel = await openDialog({
      multiple: true,
      directory: false,
      defaultPath: importDir || undefined,
    });
    if (!sel) return;
    const arr = Array.isArray(sel) ? sel : [sel];
    await stageCandidates(arr);
    // Multi-pick can span directories; key off the last-selected file
    // (most recent click, best predictor of where the next add goes).
    persistImportDir(dirname(arr[arr.length - 1]));
  }

  async function pickFolder() {
    const sel = await openDialog({
      directory: true,
      defaultPath: importDir || undefined,
    });
    if (!sel) return;
    const files = await invoke("expand_paths", { paths: [sel] });
    await stageCandidates(Array.isArray(files) ? files : [], { folder: true });
    // The picked folder is itself the source directory.
    persistImportDir(sel);
  }

  function deselectAllInputs() {
    setSelectedInputs(new Set());
    setLastSelectedIndex(null);
  }

  // Ingest paths delivered by the Tauri drag-drop event. The drop can
  // mix files and directories (Finder lets you grab both), so route
  // through expand_paths the same way pickFolder does — that flattens
  // dirs and passes files through unchanged. Stage-time validation
  // (whitelist + size + system files) happens in stageCandidates so
  // the picker and drop path share the same rejection modal.
  const ingestDroppedPaths = useCallback(async (paths) => {
    if (!paths?.length) return;
    let expanded = paths;
    try {
      const out = await invoke("expand_paths", { paths });
      if (Array.isArray(out)) expanded = out;
    } catch (e) {
      // Fall back to raw paths so a transient IPC failure doesn't
      // silently swallow the drop.
      console.error("expand_paths (drop):", e);
    }
    if (!expanded.length) return;
    // A drop is a "folder" load when expand_paths flattened a
    // directory — i.e. the expanded set differs from what was dropped.
    const folder =
      expanded.length !== paths.length ||
      expanded.some((p, i) => p !== paths[i]);
    await stageCandidates(expanded, { folder });
    // Same last-wins rule as the file picker: parent of the last
    // expanded path. For a dropped folder that's a directory inside
    // it — still a real source dir near where the user was working.
    persistImportDir(dirname(expanded[expanded.length - 1]));
    // Drop signals "I want to add input" — return to the neutral
    // staging view so the user sees the inputs list, not whatever run
    // they were inspecting. No-op when nothing is selected.
    setSelectedRunIds(new Set());
    setCurrentRunId(null);
    setRunAnchorIndex(null);
    setDetailsModalRunId(null);
  }, [stageCandidates, persistImportDir]);

  // Tauri 2 webview drag-drop event. Tauri 2 default is dragDropEnabled,
  // which routes the OS-level drop to this listener with absolute file
  // paths in the payload (browser DataTransfer is empty under Tauri).
  useEffect(() => {
    let unlisten;
    let cancelled = false;
    (async () => {
      try {
        const webview = getCurrentWebview();
        const off = await webview.onDragDropEvent((event) => {
          const t = event?.payload?.type;
          if (t === "enter" || t === "over") {
            setIsDragOver(true);
          } else if (t === "leave") {
            setIsDragOver(false);
          } else if (t === "drop") {
            setIsDragOver(false);
            const paths = event?.payload?.paths || [];
            ingestDroppedPaths(paths);
          }
        });
        if (cancelled) off?.();
        else unlisten = off;
      } catch (e) {
        console.error("onDragDropEvent:", e);
      }
    })();
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [ingestDroppedPaths]);

  // Shift-click anchor for range select — preserved from the pre-3pane
  // layout. Without this, picking 50 files out of 200 means 50 clicks
  // (Finder muscle memory expects shift-click ranges).
  const [lastSelectedIndex, setLastSelectedIndex] = useState(null);
  function handleStagingRowClick(index, path, shiftKey) {
    if (shiftKey && lastSelectedIndex !== null && inputs.length > 0) {
      const start = Math.min(lastSelectedIndex, index);
      const end = Math.max(lastSelectedIndex, index);
      setSelectedInputs((prev) => {
        const next = new Set(prev);
        for (let i = start; i <= end; i++) {
          const p = inputs[i];
          if (p !== undefined) next.add(p);
        }
        return next;
      });
    } else {
      setSelectedInputs((prev) => {
        const next = new Set(prev);
        if (next.has(path)) next.delete(path);
        else next.add(path);
        return next;
      });
      setLastSelectedIndex(index);
    }
  }

  function askRemoveSelectedInputs() {
    const n = selectedInputs.size;
    if (n === 0) return;
    setConfirmDialog({
      title: n === 1 ? "Remove this file?" : `Remove ${n} files?`,
      message:
        "They'll be dropped from the Inputs list. The files themselves aren't deleted from disk.",
      confirmLabel: "Yes, remove",
      onConfirm: () => {
        setInputs((prev) => prev.filter((p) => !selectedInputs.has(p)));
        setInputSizes((prev) => {
          const next = { ...prev };
          for (const p of selectedInputs) delete next[p];
          return next;
        });
        setSelectedInputs(new Set());
        setLastSelectedIndex(null);
        setConfirmDialog(null);
      },
    });
  }


  // ── Run lifecycle ─────────────────────────────────────────────────────────
  // dev_tracing: pendingRowRef + baselineLenRef are read by the
  // row_rendered useEffect below to fire exactly once per click — when
  // runs.length first exceeds the at-click count.
  const ltPendingRowRef = useRef(false);
  const ltBaselineLenRef = useRef(0);
  // `fromRunId` is set only when the run was started from an existing
  // run (the run-details "run again with these inputs" action). On
  // success the new run becomes the sole selection so the view follows
  // it with no manual click; the source run is dropped. The staging
  // path passes no fromRunId and keeps its selection untouched.
  async function doRun(pathsOverride, fromRunId) {
    ltrace("click");
    ltBaselineLenRef.current = runs.length;
    ltPendingRowRef.current = true;
    ltFirstStatusSeenRef.current = false;
    setStartError(null);
    const paths =
      pathsOverride && pathsOverride.length
        ? pathsOverride
        : selectedInputs.size > 0
        ? inputs.filter((p) => selectedInputs.has(p))
        : inputs;
    if (!paths.length) return;
    try {
      const newRunId = await invoke("run_pipeline", { paths, mode });
      ltrace("run_pipeline_invoke_returned");
      if (fromRunId && typeof newRunId === "string" && newRunId) {
        setSelectedRunIds(new Set([newRunId]));
        setCurrentRunId(newRunId);
        // Stale anchor would point at the now-deselected source row.
        setRunAnchorIndex(null);
        // Row isn't in `runs` until a refresh lands; the scroll effect
        // keyed on `runs` picks this up once the row mounts.
        pendingSelectScrollRef.current = newRunId;
      }
      refreshRuns();
    } catch (e) {
      setStartError(e?.message || String(e));
    }
  }
  // dev_tracing: tracks the row's first appearance + its first observed
  // status. Two separate markers:
  //   row_rendered          — first commit where runs.length > baseline
  //   row_first_status_<s>  — first commit where the new row has status s
  // Splits "row appeared at all" from "row appeared with a non-terminal
  // status the user can see ticking." On cached runs the gap between
  // those two — or the lack of any non-terminal status — is what #406
  // is investigating.
  const ltFirstStatusSeenRef = useRef(false);
  useEffect(() => {
    if (ltPendingRowRef.current && runs.length > ltBaselineLenRef.current) {
      ltrace("row_rendered");
      ltPendingRowRef.current = false;
    }
    if (!ltFirstStatusSeenRef.current && runs.length > ltBaselineLenRef.current) {
      const newest = runs[0];
      if (newest && newest.status) {
        // Sanitize: status strings are currently alphabetic
        // (running / completed / paused / cancelled / failed /
        // unknown) but retry-classification vocabulary could carry
        // parens / spaces; a marker name with literal spaces breaks
        // downstream grep-and-split tooling. Strip everything that
        // isn't alphanumeric or underscore.
        const safe = String(newest.status).replace(/[^A-Za-z0-9_]+/g, "_");
        ltrace(`row_first_status_${safe}`);
        ltFirstStatusSeenRef.current = true;
      }
    }
  }, [runs]);

  // Poll list_runs until the given run hits the desired status (or
  // times out). Returns the matched run, or null on timeout. Each
  // tick refreshes the runs list so the row updates in place — no
  // blind setTimeout wait, no missed state.
  const pollUntilRunStatus = useCallback(async (runId, predicate) => {
    const deadline = Date.now() + PAUSE_RESUME_POLL_TIMEOUT_MS;
    while (Date.now() < deadline) {
      let list = null;
      try {
        list = await invoke("list_runs");
        if (Array.isArray(list)) setRuns(list);
      } catch (e) {
        // Tolerate transient IPC failures; just retry on the next tick.
      }
      const r = (list || []).find((x) => x.run_id === runId);
      if (r && predicate(r)) return r;
      await new Promise((r2) => setTimeout(r2, PAUSE_RESUME_POLL_INTERVAL_MS));
    }
    return null;
  }, []);

  async function doPause(runId) {
    setPausingRunId(runId);
    try {
      await invoke("pause_run", { runId });
      // Poll until the runner commits status=paused on disk. Real
      // transitions are sub-second; previously we waited a blind 3s
      // which made the action buttons disappear long after the
      // backend had already settled.
      await pollUntilRunStatus(runId, (r) => r.status === "paused");
    } catch (e) {
      console.error("pause_run:", e);
    } finally {
      setPausingRunId(null);
      refreshRuns();
    }
  }

  async function doResume(runId) {
    setResumingRunId(runId);
    try {
      await invoke("resume_run", { runId });
      // Wait for the spawned Python sidecar to flip status=running
      // before clearing the spinner. Without this, the UI stays
      // stuck on "Paused" between invoke() returning and the first
      // pipeline-progress event landing — which is exactly when users
      // re-click Resume and trigger "already running" errors.
      await pollUntilRunStatus(runId, (r) => r.status === "running");
    } catch (e) {
      alert(`Resume failed: ${e?.message || e}`);
    } finally {
      setResumingRunId(null);
      refreshRuns();
    }
  }

  function askCancel(runId) {
    setConfirmDialog({
      title: "Cancel run?",
      message:
        "This is unrecoverable — you won't be able to resume. The run's output so far stays on disk until you delete it.",
      confirmLabel: "Yes, cancel",
      onConfirm: async () => {
        try {
          await invoke("cancel_run", { runId });
        } catch (e) {
          console.error("cancel_run:", e);
        }
        setConfirmDialog(null);
        refreshRuns();
      },
    });
  }

  function askDelete(runId) {
    setConfirmDialog({
      title: "Delete run?",
      message:
        "This removes both the logs and the vault output for this run. Unrecoverable.",
      confirmLabel: "Yes, delete",
      onConfirm: async () => {
        try {
          await invoke("delete_run", { runId });
        } catch (e) {
          console.error("delete_run:", e);
        }
        setConfirmDialog(null);
        if (selectedRun?.run_id === runId) clearRunSelection();
        setSelectedRunIds((prev) => {
          if (!prev.has(runId)) return prev;
          const next = new Set(prev);
          next.delete(runId);
          return next;
        });
        refreshRuns();
      },
    });
  }

  function askDeleteSelectedRuns() {
    const ids = [...selectedRunIds];
    if (ids.length === 0) return;
    setConfirmDialog({
      title: ids.length === 1 ? "Delete this run?" : `Delete ${ids.length} runs?`,
      message:
        "Removes both the logs and the vault output for each selected run. Unrecoverable.",
      confirmLabel: ids.length === 1 ? "Yes, delete" : `Yes, delete ${ids.length}`,
      onConfirm: async () => {
        for (const id of ids) {
          try {
            await invoke("delete_run", { runId: id });
          } catch (e) {
            console.error("delete_run:", e);
          }
        }
        if (selectedRun && ids.includes(selectedRun.run_id)) {
          setCurrentRunId(null);
        }
        setSelectedRunIds(new Set());
        setRunAnchorIndex(null);
        setConfirmDialog(null);
        refreshRuns();
      },
    });
  }

  // Persist the destination + chosen privacy level into the global app
  // config so they survive restart. update_config merges these keys
  // server-side — same atomic-partial-write path as run_aliases /
  // inputs / mode / import_dir; config.json at the state root, not the
  // per-run state sidecar.
  function persistExportPrefs(dir, level) {
    invoke("update_config", {
      patch: { export_dir: dir, privacy_level: level },
    }).catch((e) => console.error("update_config (export prefs):", e));
  }

  // Open the Privacy Level gate in front of the export. Resolves the
  // persisted destination (defaulting to ~/Documents/BaseVault) and
  // loads per-level counts for the selected run(s) so the modal can
  // show the privacy/completeness tradeoff at the moment of choice.
  async function openPrivacyExport() {
    const ids = [...selectedRunIds];
    if (ids.length === 0) return;
    // Both entry points (button + native menu) are gated on
    // canExportSelection; guard here too so a stale-enabled native
    // menu click can't export an all-unfinished selection.
    const exportable = ids.filter((id) => {
      const r = runs.find((x) => x.run_id === id);
      return r && r.status !== "running" && r.status !== "paused";
    });
    if (exportable.length === 0) return;
    let dir = exportDir;
    if (!dir) {
      try {
        dir = await invoke("export_default_dir");
        setExportDir(dir);
      } catch (e) {
        alert(`Couldn't initialize export dir: ${e?.message || e}`);
        return;
      }
    }
    // Aggregate counts across the selection — one privacy level + one
    // destination apply to the whole batch.
    const counts = { actions: 0, insights: 0, patterns: 0, entities: 0, facts: 0, rawDocs: 0 };
    await Promise.all(ids.map(async (id) => {
      const [facts, ents, pats, ins, acts, inputs] = await Promise.all([
        invoke("read_run_facts_all", { runId: id }).catch(() => ({})),
        invoke("read_run_entities", { runId: id }).catch(() => null),
        invoke("read_run_patterns_all", { runId: id }).catch(() => ({})),
        invoke("read_run_insights", { runId: id }).catch(() => null),
        invoke("read_run_actions", { runId: id }).catch(() => ({ actions: [] })),
        invoke("read_run_preprocessed_inputs", { runId: id }).catch(() => ({})),
      ]);
      counts.facts += Object.values(facts || {}).reduce((n, l) => n + (l || []).length, 0);
      counts.entities += (ents?.entities || []).length;
      counts.patterns += Object.values(pats || {}).reduce((n, l) => n + (l || []).length, 0);
      counts.insights += (ins?.cross_domain || []).length + (ins?.critical || []).length;
      counts.actions += Array.isArray(acts?.actions) ? acts.actions.length : 0;
      counts.rawDocs += Object.keys(inputs || {}).length;
    }));
    setPrivacyModal({ ids, counts });
  }

  // Run the export at the chosen privacy level + destination. The cut
  // is applied during manifest assembly inside regenVault → exportRun;
  // export_run then mirrors the (already-filtered) vault dir.
  async function runPrivacyExport() {
    const modal = privacyModal;
    if (!modal) return;
    const { ids } = modal;
    const dest = exportDir;
    const level = privacyLevel;
    persistExportPrefs(dest, level);
    setPrivacyModal(null);
    const failures = [];
    for (const id of ids) {
      // run_id is canonical `<iso-z>-<short_id>` (e.g.
      // 2026-04-30T15-43-57Z-xqp4). Strip the trailing `-<short_id>`
      // to derive the timestamp prefix when an alias is set, so the
      // exported subdir keeps a sortable name (`<iso-z>-<alias>`).
      // No alias → use the full run_id verbatim.
      const alias = runAliases[id];
      const tsPrefix = id.lastIndexOf("-") > 0 ? id.slice(0, id.lastIndexOf("-")) : id;
      const label = alias ? `${tsPrefix}-${alias}` : id;
      // Pre-flight: ask Rust where the export will land and whether
      // anything already lives there. Pre-checking + prompting before
      // the destructive copy keeps the Replace/Skip prompt out of the
      // error-handling path (where any IPC quirk could swallow the
      // EXISTS marker silently).
      let overwrite = false;
      try {
        const [targetPath, exists] = await invoke("export_target_path", {
          runId: id,
          dest,
          label,
        });
        if (exists) {
          const replace = await askConfirm(
            `"${targetPath}" already exists. Replacing will delete the existing folder and its contents.`,
            {
              title: "Replace existing export?",
              okLabel: "Replace",
              cancelLabel: "Skip",
              kind: "warning",
            },
          );
          if (!replace) continue;
          overwrite = true;
        }
      } catch (e) {
        failures.push(`${label}: ${String(e?.message || e)}`);
        continue;
      }
      try {
        // Materialize the vault dir on demand. The runtime pipeline
        // no longer auto-emits a vault — `regenVault` is the only
        // path that produces ~/Documents/BaseVault/<run-name>/, and
        // export_run expects the dir to already exist on disk.
        await regenVault({ runId: id, invoke, privacyLevel: level });
        await invoke("export_run", { runId: id, dest, label, overwrite });
      } catch (e) {
        failures.push(`${label}: ${String(e?.message || e)}`);
      }
    }
    if (failures.length) {
      alert(`Export had errors:\n\n${failures.join("\n")}`);
      return;
    }
    setSuccessModal({ dest });
  }

  // Native File-menu items (Rust emits these on click). Add File /
  // Add Folder reuse the in-app pickers, then deselect the run so the
  // user lands back on the staging/file view to see what was added —
  // the same "return to neutral view" the drag-drop ingest path does.
  // Export Selected routes into openPrivacyExport — the same front-door
  // the bottom-bar Export button uses, which opens the Privacy Level
  // modal before any copy. One shared path: the menu can't bypass the
  // privacy cut. Read through a ref kept fresh each render: the
  // listen() registration stays mount-once while the handlers still
  // see current state (openPrivacyExport reads the live selection at
  // fire time).
  const menuActionsRef = useRef(null);
  menuActionsRef.current = {
    addFile: async () => {
      await pickFiles();
      clearRunSelection();
    },
    addFolder: async () => {
      await pickFolder();
      clearRunSelection();
    },
    exportSelected: openPrivacyExport,
  };
  useEffect(() => {
    const unlisteners = [];
    let cancelled = false;
    const wire = (event, run) =>
      listen(event, () => run()).then((u) => {
        if (cancelled) u?.();
        else unlisteners.push(u);
      });
    wire("menu-add-file", () => menuActionsRef.current.addFile());
    wire("menu-add-folder", () => menuActionsRef.current.addFolder());
    wire("menu-export-selected", () => menuActionsRef.current.exportSelected());
    return () => {
      cancelled = true;
      unlisteners.forEach((u) => u?.());
    };
  }, []);

  // Export needs finished output: a still-running or paused run has
  // nothing to export. Count only finished selected runs (completed /
  // failed / cancelled — all terminal); that count gates both the
  // native menu item and the bottom-bar Export button (kept in
  // lockstep) and is what the button label shows.
  const exportableSelectedCount = useMemo(
    () =>
      [...selectedRunIds].filter((id) => {
        const r = runs.find((x) => x.run_id === id);
        return r && r.status !== "running" && r.status !== "paused";
      }).length,
    [selectedRunIds, runs],
  );
  const canExportSelection = exportableSelectedCount > 0;

  // Mirror that onto the native File ▸ Export Selected item (it starts
  // disabled in Rust).
  useEffect(() => {
    invoke("set_export_menu_enabled", {
      enabled: canExportSelection,
    }).catch(() => {});
  }, [canExportSelection]);

  // ── Derived UI state ──────────────────────────────────────────────────────
  const sameModeActive = activeModes.has(mode);
  const modeAvailable = {
    local: envFlags.local_usable,
    // Cloud availability gates on the key ONLY. Attestation is a
    // non-blocking visibility signal — a failed / in-flight attestation
    // does not disable Private Cloud or block a run. The real
    // per-connection guarantee stays at the transport layer (the
    // kernel's attested provider pins the enclave TLS key and refuses a
    // non-matching enclave); the UI just surfaces what it sees.
    tee: envFlags.tee_key_set,
  };

  const canStartNewRun =
    !sameModeActive &&
    modeAvailable[mode] &&
    (selectedRun ? (selectedRun.inputs?.length || 0) > 0 : inputs.length > 0);

  let runLabel;
  if (selectedRun) {
    const n = selectedRun.inputs?.length || 0;
    runLabel = `Run again with ${n} file${n === 1 ? "" : "s"}`;
  } else if (inputs.length === 0) {
    runLabel = "Run pipeline";
  } else if (selectedInputs.size === 0) {
    runLabel = `Run with all ${inputs.length} file${inputs.length === 1 ? "" : "s"}`;
  } else if (selectedInputs.size === 1) {
    runLabel = "Run with 1 file";
  } else {
    runLabel = `Run with ${selectedInputs.size} files`;
  }

  // ── Pane width state — three columns; the markdown pane stretches.
  const [splitWidths, setSplitWidths] = useState(DEFAULT_SPLIT);
  const dragRef = useRef(null);
  const onDragStart = (which) => (e) => {
    dragRef.current = {
      which,
      startX: e.clientX,
      runs: splitWidths.runs,
      tree: splitWidths.tree,
    };
    e.preventDefault();
  };
  useEffect(() => {
    function move(e) {
      const drag = dragRef.current;
      if (!drag) return;
      const dx = e.clientX - drag.startX;
      setSplitWidths((prev) => {
        if (drag.which === "runs") {
          const next = Math.max(180, Math.min(480, drag.runs + dx));
          return { ...prev, runs: next };
        }
        const next = Math.max(220, Math.min(560, drag.tree + dx));
        return { ...prev, tree: next };
      });
    }
    function up() {
      dragRef.current = null;
    }
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, []);

  return (
    <main className="app">
      {/* The whole top bar is the privacy-tinted trust-bar — color
          spans the full width, mode-group floats on the left, lock +
          copy in the middle, app-state buttons (Attested when on
          Private Cloud, Settings always) float right inside the same
          tint. CSS transition on background makes mode flips read as
          the bar's color shifting. */}
      <header className={`top-bar trust-bar trust-bar-${mode}`}>
        <div className="mode-group" role="tablist">
          {Object.entries(MODE_META).map(([key, meta]) => {
            const available = modeAvailable[key] ?? false;
            const teeKeyAvailable = envFlags.tee_key_set;
            const disabled = !available;
            // Cloud disables only when no key is set — attestation no
            // longer gates it, so there is no "attestation failed" mode
            // tooltip (that context lives on the always-on indicator).
            const teeTooltip = !teeKeyAvailable
              ? "Set a Tinfoil API key in Settings to enable Private Cloud mode"
              : undefined;
            const tooltip = disabled
              ? key === "tee"
                ? teeTooltip
                : key === "local"
                ? envFlags.local_disabled_reason ||
                  "Set up a local model in Settings to enable Local mode."
                : undefined
              : undefined;
            return (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={mode === key}
                className={`mode-btn mode-btn-${key} ${
                  mode === key ? "active" : ""
                }`}
                disabled={disabled}
                title={tooltip}
                onClick={() => !disabled && setMode(key)}
              >
                {meta.label}
                {disabled && " (disabled)"}
              </button>
            );
          })}
        </div>
        <span className="trust-text">{MODE_META[mode].trust}</span>
        <div className="top-bar-right">
          {/* Attestation indicator — shown in Private Cloud mode, and
              also whenever there's a non-clean attestation state so the
              failure context never vanishes after switching to Local
              (#899). It's a de-emphasized status light, not a gate. */}
          {(mode === "tee" || (attestation && !attestation.ok)) && (
            <AttestationIndicator
              attestation={attestation}
              checking={attestationChecking}
              onClick={() => setAttModalOpen(true)}
            />
          )}
          <button
            type="button"
            className="settings-btn"
            onClick={() => setSettingsOpen(true)}
            title="Settings"
          >
            <span className="settings-btn-icon">⚙</span>
            <span className="settings-btn-text">Settings</span>
          </button>
        </div>
      </header>

      <div className="body body-3pane">
        <aside
          className="pane runs-pane"
          style={{ width: splitWidths.runs }}
          onMouseDown={() => {
            // Suspend list refreshes for the duration of this click —
            // re-rendering RunRow's children between mousedown and
            // mouseup detaches the click target and the browser drops
            // the click (#406). Scoped to the active mousedown→mouseup
            // window only; idle hover does NOT suspend refreshes, so
            // the bar's stage label + bar position continue to update
            // through cached runs even when the cursor sits over the
            // pane.
            //
            // Global mouseup listener catches the drag-off-pane case:
            // if the user mousedowns inside and releases outside, the
            // aside's onMouseUp never fires, but document-level mouseup
            // does. Self-removes after firing once.
            hoveringRunsRef.current = true;
            const onUp = () => {
              hoveringRunsRef.current = false;
              refreshRuns();
              document.removeEventListener("mouseup", onUp);
            };
            document.addEventListener("mouseup", onUp);
          }}
        >
          <div className="pane-header">
            <span className="pane-title">Pipeline runs</span>
            {selectedRunIds.size > 0 && (
              <span className="pane-selection-count">
                {selectedRunIds.size} selected
              </span>
            )}
          </div>
          {runs.length === 0 ? (
            <p className="empty">No runs yet. Pick files in the middle pane and run the pipeline.</p>
          ) : (
            <ul className="runs-list" ref={runsListRef}>
              {runs.map((run, idx) => (
                <RunRow
                  key={run.run_id}
                  run={run}
                  alias={runAliases[run.run_id] || ""}
                  selected={selectedRunIds.has(run.run_id)}
                  current={currentRunId === run.run_id}
                  isPausing={pausingRunId === run.run_id}
                  isResuming={resumingRunId === run.run_id}
                  nowMs={elapsedNow}
                  reducedMotion={prefersReducedMotion}
                  onSelect={(e) => handleSelectRun(run, idx, e)}
                  onPause={() => doPause(run.run_id)}
                  onResume={() => doResume(run.run_id)}
                  onCancel={() => askCancel(run.run_id)}
                  onDelete={() => askDelete(run.run_id)}
                />
              ))}
            </ul>
          )}
          {/* Run-selection actions only render when something is
              actually selected. The primary Run-again button moved to
              the FILES pane footer (so the Run button always lives in
              the same physical slot regardless of whether the user is
              staging files or viewing a run); only the bulk actions
              live here. */}
          {selectedRunIds.size > 0 && (
            <div className="pane-footer run-selection-actions">
              <div className="actions-row">
                <button
                  type="button"
                  className="nav-btn"
                  onClick={clearRunSelection}
                  title="Deselect all and return to input files"
                >
                  Clear
                </button>
                <button
                  type="button"
                  className="nav-btn"
                  onClick={openPrivacyExport}
                  disabled={!canExportSelection}
                  title={
                    canExportSelection
                      ? "Pick a destination dir, copy each selected run's vault output there"
                      : "Select at least one finished run to export"
                  }
                >
                  {canExportSelection
                    ? `Export (${exportableSelectedCount})`
                    : "Export"}
                </button>
                <button
                  type="button"
                  className="btn-danger"
                  onClick={askDeleteSelectedRuns}
                >
                  Delete ({selectedRunIds.size})
                </button>
              </div>
              {selectedRun && sameModeActive && (
                <p className="hint">
                  A {MODE_META[mode].label} run is already active — pause
                  or cancel it, or switch modes to run in parallel.
                </p>
              )}
            </div>
          )}
        </aside>

        <div
          className="splitter"
          onMouseDown={onDragStart("runs")}
          aria-hidden="true"
        />

        <section className="pane tree-pane" style={{ width: splitWidths.tree }}>
          {selectedRun ? (
            <RunTreePane
              run={selectedRun}
              tree={runTree}
              openDirs={openDirs}
              setOpenDirs={setOpenDirs}
              selectedFile={selectedFile}
              setSelectedFile={(file) =>
                file ? navigateToFile(file) : setSelectedFile(null)
              }
              alias={runAliases[selectedRun.run_id] || ""}
              onRename={(label) => renameRun(selectedRun.run_id, label)}
              mode={mode}
              canRun={canStartNewRun}
              runLabel={runLabel}
              onRun={() => doRun(selectedRun.inputs, selectedRun.run_id)}
              sameModeActive={sameModeActive}
              onOpenDetails={() => setDetailsModalRunId(selectedRun.run_id)}
              refreshTick={refreshTick}
            />
          ) : (
            <StagingPane
              inputs={inputs}
              inputSizes={inputSizes}
              newlyAddedInputsRef={newlyAddedInputsRef}
              inputsHighlightTick={inputsHighlightTick}
              selectedInputs={selectedInputs}
              handleRowClick={handleStagingRowClick}
              deselectAllInputs={deselectAllInputs}
              askDeleteSelected={askRemoveSelectedInputs}
              pickFiles={pickFiles}
              pickFolder={pickFolder}
              mode={mode}
              canRun={canStartNewRun}
              runLabel={runLabel}
              onRun={() => doRun()}
              sameModeActive={sameModeActive}
              startError={startError}
            />
          )}
        </section>

        <div
          className="splitter"
          onMouseDown={onDragStart("tree")}
          aria-hidden="true"
        />

        <section className="pane md-pane">
          <MarkdownPreview
            selectedRun={selectedRun}
            selectedFile={selectedFile}
            content={fileContent}
            error={fileError}
            inputsCount={inputs.length}
            runsCount={runs.length}
            onNavigate={handleMarkdownNavigate}
            onOpenUrl={handleOpenUrl}
            canBack={fileHistoryIdx > 0}
            canForward={fileHistoryIdx < fileHistory.length - 1}
            onBack={fileNavBack}
            onForward={fileNavForward}
            runFacts={selectedRun ? runFacts[selectedRun.run_id] || [] : []}
            soleInputFile={soleInputFile}
            refreshTick={refreshTick}
          />
        </section>
      </div>

      {wizardOpen && (
        <Wizard
          easy={!!wizardOpen.easy}
          allowCancel={!wizardOpen.firstRun}
          onComplete={(resultingMode) => {
            setWizardOpen(false);
            refreshEnv();
            refreshAttestation();
            if (resultingMode && MODE_META[resultingMode]) {
              setMode(resultingMode);
            }
          }}
          onCancel={() => {
            setWizardOpen(false);
          }}
        />
      )}
      {settingsOpen && (
        <Settings
          onClose={() => setSettingsOpen(false)}
          onChanged={() => {
            refreshEnv();
          }}
          onResetPaneSizes={() => {
            // Reset both the in-app pane widths AND the OS-level
            // window size to the ship default (1440×900). The window
            // resize lives Rust-side so we don't have to broaden
            // capabilities/default.json with `core:window:allow-set-size`
            // — invoking commands is already permitted.
            setSplitWidths(DEFAULT_SPLIT);
            invoke("reset_window_size").catch((e) =>
              console.error("reset_window_size:", e)
            );
          }}
        />
      )}
      {attModalOpen && (
        <AttestationModal
          attestation={attestation}
          checking={attestationChecking}
          // Sanctioned attestation call site #2: Settings re-check
          // button (the modal's Recheck control → refreshAttestation).
          onRecheck={refreshAttestation}
          onClose={() => setAttModalOpen(false)}
        />
      )}
      {confirmDialog && (
        <ConfirmDialog
          title={confirmDialog.title}
          message={confirmDialog.message}
          confirmLabel={confirmDialog.confirmLabel}
          onConfirm={confirmDialog.onConfirm}
          onCancel={() => setConfirmDialog(null)}
        />
      )}
      {detailsModalRunId && (
        <RunDetailsModal
          runId={detailsModalRunId}
          run={runs.find((r) => r.run_id === detailsModalRunId)}
          nowMs={elapsedNow}
          onClose={() => setDetailsModalRunId(null)}
        />
      )}
      {rejectedFiles && rejectedFiles.items.length > 0 && (
        <IgnoredFilesModal
          data={rejectedFiles}
          onClose={() => setRejectedFiles(null)}
        />
      )}
      {privacyModal && (
        <PrivacyLevelModal
          counts={privacyModal.counts}
          runCount={privacyModal.ids.length}
          privacyLevel={privacyLevel}
          setPrivacyLevel={setPrivacyLevel}
          exportDir={exportDir}
          onPickDir={async () => {
            try {
              const picked = await invoke("pick_export_dir", { defaultPath: exportDir });
              if (picked) setExportDir(picked);
            } catch (e) {
              console.error("pick_export_dir:", e);
            }
          }}
          onExport={runPrivacyExport}
          onCancel={() => setPrivacyModal(null)}
        />
      )}
      {successModal && (
        <ExportSuccessModal
          dest={successModal.dest}
          onClose={() => setSuccessModal(null)}
        />
      )}
      {isDragOver && (
        <div className="drop-overlay" aria-hidden="true">
          <div className="drop-overlay-card">
            <div className="drop-overlay-icon">⤓</div>
            <div className="drop-overlay-title">Release to add files</div>
            <div className="drop-overlay-sub">
              Folders are expanded; rejected files surface the same error as
              the file picker.
            </div>
          </div>
        </div>
      )}
      {/* Floating chat helper — fixed bottom-right, overlays every view
          (Runs / Wizard / Settings / Attestation modal). Non-modal:
          the rest of the UI stays interactive behind it. Ships in
          release builds: the #451 dev-only gate (hide until the
          conversational persona + transcript landed) was removed by
          director decision once #488 delivered that keystone. */}
      <ChatbotHelper
        resolveRunName={resolveRunName}
        // Attestation does not gate chat send — it's a non-blocking
        // visibility signal on the top-bar indicator, so no mode /
        // attestation props are threaded down here.
        // Reuse the app's ONE confirm dialog (run cancel/delete, input
        // removal). The chat panel is a child of App, so the single
        // source is threaded down — no second modal, no duplicated
        // modal state. The wrapper auto-dismisses on confirm so callers
        // only supply title/message/confirmLabel/onConfirm.
        requestConfirm={(opts) =>
          setConfirmDialog({
            ...opts,
            onConfirm: async () => {
              await opts.onConfirm?.();
              setConfirmDialog(null);
            },
          })
        }
        resolveResource={(resource, runId) =>
          resolveCitation(runId, resource.kind, resource.record_id, invoke)
        }
        onOpenResource={(resolved, runId) => {
          if (resolved?.relPath) {
            handleMarkdownNavigate({
              runId,
              relPath: resolved.relPath,
              anchor: resolved.anchor || "",
              chunkOffset: resolved.chunkOffset,
              // The raw chunk length rides the backend resource (not
              // the record_id, which has only the start offset), so
              // it's read straight off the merged resource here.
              chunkLen: resolved.chunk_len,
            });
          }
        }}
      />
    </main>
  );
}

function StagingPane({
  inputs,
  inputSizes,
  newlyAddedInputsRef,
  inputsHighlightTick,
  selectedInputs,
  handleRowClick,
  deselectAllInputs,
  askDeleteSelected,
  pickFiles,
  pickFolder,
  mode,
  canRun,
  runLabel,
  onRun,
  sameModeActive,
  startError,
}) {
  const hasSelection = selectedInputs.size > 0;

  // After files are added, scroll the list so the new rows are in
  // view (they're appended at the bottom — see stageCandidates) and
  // flag them with the shared transient-highlight, the same affordance
  // a scroll-to-anchor jump uses. Layout effect so the scroll lands
  // before paint (no flash of the pre-scroll list). The
  // remove → reflow → re-add restarts the CSS animation if the same
  // row is re-flagged before its previous fade finished.
  const inputsListRef = useRef(null);
  useLayoutEffect(() => {
    const list = inputsListRef.current;
    const added = newlyAddedInputsRef?.current;
    if (!list || !added?.length) return;
    list.scrollTop = list.scrollHeight;
    const addedSet = new Set(added);
    for (const row of list.querySelectorAll("[data-input-path]")) {
      if (!addedSet.has(row.dataset.inputPath)) continue;
      row.classList.remove("transient-highlight");
      void row.offsetWidth; // force reflow so re-add restarts the fade
      row.classList.add("transient-highlight");
    }
  }, [inputsHighlightTick, newlyAddedInputsRef]);

  return (
    <>
      <div className="pane-header staging-header">
        <span className="pane-title">
          Input files{inputs.length > 0 ? ` (${inputs.length})` : ""}
        </span>
      </div>
      <div className="staging-actions">
        <button type="button" onClick={pickFiles} title="Add file(s)">
          📄 Add file
        </button>
        <button type="button" onClick={pickFolder} title="Add folder (recursive)">
          📁 Add folder
        </button>
      </div>
      {inputs.length === 0 ? (
        <div className="empty drop-hint">
          <p>No files yet.</p>
          <p>Use <strong>Add file</strong> or <strong>Add folder</strong> above to get started.</p>
        </div>
      ) : (
        <>
          {inputs.length >= 5 && (
            <p className="hint inputs-shift-hint">
              Tip: shift-click to select a range.
            </p>
          )}
          <ul className="inputs-list tree-staging" ref={inputsListRef}>
            {inputs.map((p, i) => {
              const size = inputSizes?.[p];
              return (
                <li
                  key={p}
                  data-input-path={p}
                  className={`input-row ${selectedInputs.has(p) ? "selected" : ""}`}
                  onClick={(e) => handleRowClick(i, p, e.shiftKey)}
                >
                  <input
                    type="checkbox"
                    checked={selectedInputs.has(p)}
                    onChange={() => {}}
                    tabIndex={-1}
                  />
                  <span className="input-path" title={p}>
                    {displayBasename(p)}
                  </span>
                  {typeof size === "number" && (
                    <span className="input-size">{formatSizeShort(size)}</span>
                  )}
                </li>
              );
            })}
          </ul>
        </>
      )}
      <div className="pane-footer staging-footer">
        <button
          type="button"
          className={`run-btn ${MODE_META[mode].btnClass}`}
          disabled={!canRun}
          onClick={onRun}
        >
          {runLabel}
        </button>
        {hasSelection && (
          <div className="staging-selection-actions">
            <button
              type="button"
              className="nav-btn"
              onClick={deselectAllInputs}
            >
              Clear
            </button>
            <button
              type="button"
              className="btn-danger"
              onClick={askDeleteSelected}
              title={`Remove ${selectedInputs.size} from list (files stay on disk)`}
            >
              Delete ({selectedInputs.size})
            </button>
          </div>
        )}
        {sameModeActive && (
          <p className="hint">
            A {MODE_META[mode].label} run is already active — pause or cancel
            it, or switch to another mode to run in parallel.
          </p>
        )}
        {startError && <p className="error-text">{startError}</p>}
      </div>
    </>
  );
}

// Inline rename for the run-view header. The 4-letter short_id is the
// canonical reference (used to locate the run dir + log paths); the alias
// persisted to config.run_aliases is the user-visible label. Once an
// alias is set the short_id stops surfacing in the UI entirely — only
// the alias renders, and the rename input pre-fills with the alias.
// (The run LIST and chat picker carry the always-visible #<id> two-line
// layout; the header keeps its prior compact behaviour by design.)
function RunHeaderName({ run, alias, onRename }) {
  const [editing, setEditing] = useState(false);
  // Pre-fill with the alias; fall back to short_id so an alias-less
  // run opens the input with the visible label (e.g. `8bv3`) rather
  // than empty — auto-select-on-focus then makes it edit-or-replace.
  const [draft, setDraft] = useState(alias || run.short_id || "");
  const inputRef = useRef(null);
  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);
  // Re-sync draft when the alias changes from outside (e.g. switching
  // to a different run) so the prefilled value matches the current run.
  useEffect(() => {
    if (!editing) setDraft(alias || run.short_id || "");
  }, [alias, editing, run.short_id]);
  const shortLabel = run.short_id ? `#${run.short_id}` : "(no id)";
  const display = alias || `Run ${shortLabel}`;
  // Skip the onRename call when the draft is unchanged from what the
  // input opened with — clicking the name and clicking away (no typing)
  // would otherwise commit the pre-filled short_id as the alias (#181).
  const initial = alias || run.short_id || "";
  const commit = () => {
    if (draft.trim() !== initial) onRename(draft);
    setEditing(false);
  };
  if (editing) {
    return (
      <input
        ref={inputRef}
        className="run-header-rename"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            commit();
          } else if (e.key === "Escape") {
            e.stopPropagation();
            setDraft(alias || "");
            setEditing(false);
          }
        }}
        onBlur={commit}
        placeholder={`Run ${shortLabel}`}
      />
    );
  }
  return (
    <button
      type="button"
      className="run-header-name"
      onClick={() => setEditing(true)}
      title={alias ? "Click to rename" : "Click to rename"}
    >
      <span className="run-header-name-text">{display}</span>
      {!alias && (
        <span className="run-header-name-hint">click to rename</span>
      )}
    </button>
  );
}

// Stage order for routing-table rendering. Mirrors the pipeline's
// canonical stage list (progress.PIPELINE_STAGES) so the UI table
// reads top-to-bottom in the order calls actually fire.
const RUN_DETAILS_STAGE_ORDER = [
  "vision",           // image transcription, runs during ingest (#121)
  "extract",
  "entities",
  "entities_dedupe",
  "patterns",
  "insights",
  "actions",
];

// List the stages that had reasoning enabled for a run. Source: the
// snapshotted `run_config.stage_reasoning` map written at run start.
// Empty array when no reasoning was enabled or the field is absent
// (pre-this-PR runs / non-TEE runs).
function reasoningStages(run) {
  const map = run?.run_config?.stage_reasoning;
  if (!map || typeof map !== "object") return [];
  return RUN_DETAILS_STAGE_ORDER.filter((s) => map[s] === true);
}

// Format a sha256 / git sha for display: first 8 chars, lowercase.
function shortSha(sha) {
  if (!sha) return "";
  return String(sha).toLowerCase().slice(0, 8);
}

// Run summary section — surfaces the snapshotted run_config (per-stage
// routing, reasoning, temperature, pipeline version, sentiment, bundle
// mode, subject resolution, input file list).
//
// Issue #165 folded this into the RunDetailsModal header (above the
// cache hit-rate summary). Pre-#165 it was a collapsible panel above
// the file tree in RunTreePane; it's modal-only now since the modal
// is opt-in surface, the routing data is high-value, and the on-screen
// real estate justifies always-visible.
//
// Source: `run.run_config` written by the runner at run start (post-
// #165 the entities-stage subject_resolution + bundle_mode are
// overlaid by the Tauri side from llm-calls.jsonl `entities_decision`
// events so they appear on running runs too). Pre-#118 runs lack the
// field; falls back to a one-line provider + model summary.
function RunSummarySection({ run }) {
  const cfg = run?.run_config || null;
  const stageModels = cfg?.stage_models || null;
  const stageReasoning = cfg?.stage_reasoning || {};
  const temperature = cfg?.temperature ?? 0;
  const subjectRes = cfg?.subject_resolution || null;
  const bundleMode = cfg?.bundle_mode;
  const sentiment = cfg?.sentiment;
  const llmCacheEnabled = cfg?.llm_cache_enabled;
  const gitSha = cfg?.pipeline_git_sha;
  const appVersion = cfg?.app_version;
  // Inputs deliberately omitted — the staging pane and run-tree pane
  // already surface input files, and duplicating them in the modal
  // header just adds noise.

  const hasRouting = !!stageModels;

  return (
    <div className="run-summary-section">
      {hasRouting ? (
        <table className="run-details-routing">
          <thead>
            <tr>
              <th>Stage</th>
              <th>Model</th>
              <th>Reasoning</th>
              <th>Temp</th>
            </tr>
          </thead>
          <tbody>
            {RUN_DETAILS_STAGE_ORDER.map((s) => {
              const m = stageModels[s];
              if (!m) return null;
              const r = stageReasoning[s] === true;
              return (
                <tr key={s} className={r ? "reasoning-on" : ""}>
                  <td>{s}</td>
                  <td>{m}</td>
                  <td>{r ? "ON" : "off"}</td>
                  <td>{Number(temperature).toFixed(1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : (
        <p className="run-details-empty">
          {run?.provider || run?.model
            ? `No run_config recorded (older run). provider=${run.provider || "?"} model=${run.model || "?"}`
            : "No run_config recorded (older run)."}
        </p>
      )}
      <dl className="run-details-meta">
        {appVersion && (
          <>
            <dt>App version</dt>
            <dd>{appVersion}</dd>
          </>
        )}
        {gitSha && (
          <>
            <dt>Pipeline git</dt>
            <dd title={gitSha}>{shortSha(gitSha)}</dd>
          </>
        )}
        {sentiment && (
          <>
            <dt>Sentiment</dt>
            <dd>{sentiment}</dd>
          </>
        )}
        {llmCacheEnabled !== undefined && (
          <>
            <dt>LLM cache</dt>
            <dd>{llmCacheEnabled ? "enabled" : "bypass"}</dd>
          </>
        )}
        {bundleMode !== undefined && (
          <>
            <dt>Bundle mode</dt>
            <dd>{bundleMode ? "yes (per-file narrators)" : "no"}</dd>
          </>
        )}
        {subjectRes && (
          <>
            <dt>Subject</dt>
            <dd
              title={`canonical_id=${subjectRes.canonical_id || "?"}\nsource=${subjectRes.source || "?"}`}
            >
              {subjectRes.display || subjectRes.canonical_id || "?"}{" "}
              <span className="run-details-meta-note">
                ({subjectRes.source || "?"})
              </span>
            </dd>
          </>
        )}
      </dl>
    </div>
  );
}

function RunTreePane({
  run,
  tree,
  openDirs,
  setOpenDirs,
  selectedFile,
  setSelectedFile,
  alias,
  onRename,
  mode,
  canRun,
  runLabel,
  onRun,
  sameModeActive,
  onOpenDetails,
  refreshTick,
}) {
  // Fetch insights/actions structured data so the tree can show one
  // nested entry per item under `4-insights.md` / `5-actions.md`.
  // Anchors mirror obsidianRenderer (`cross-domain-N`, `critical-N`,
  // `action-N`); clicking an entry routes through setSelectedFile,
  // which feeds MarkdownPreview's existing scroll-to-anchor +
  // transient-highlight primitive.
  const runId = run?.run_id;
  const [headingsByPath, setHeadingsByPath] = useState({});
  useEffect(() => {
    if (!runId) {
      setHeadingsByPath({});
      return undefined;
    }
    let cancelled = false;
    Promise.allSettled([
      invoke("read_run_insights", { runId }),
      invoke("read_run_actions", { runId }),
    ]).then(([insRes, actRes]) => {
      if (cancelled) return;
      const out = {};
      const insights = insRes.status === "fulfilled" ? insRes.value : null;
      const actions = actRes.status === "fulfilled" ? actRes.value : null;
      if (insights) {
        const cross = insights.cross_domain || [];
        const crit = insights.critical || [];
        // Numbering is continuous across cross-domain → critical so
        // the tree mirrors the running scan-index a user uses to
        // refer to an insight ("insight 5"), regardless of which
        // sub-section it lives in.
        let n = 0;
        const entries = [
          ...cross.map((ins, idx) => ({
            anchor: `cross-domain-${idx + 1}`,
            num: ++n,
            label: ins?.name || "(unnamed insight)",
          })),
          ...crit.map((ins, idx) => ({
            anchor: `critical-${idx + 1}`,
            num: ++n,
            label: ins?.name || "(unnamed insight)",
          })),
        ];
        if (entries.length) out[INSIGHTS_FILE] = entries;
      }
      if (actions) {
        const list = actions.actions || [];
        const entries = list.map((a, idx) => ({
          anchor: `action-${idx + 1}`,
          num: idx + 1,
          label: a?.recommendation || "(unnamed action)",
        }));
        if (entries.length) out[ACTIONS_FILE] = entries;
      }
      setHeadingsByPath(out);
    });
    return () => {
      cancelled = true;
    };
  }, [runId, refreshTick]);

  return (
    <>
      <div className="pane-header run-view-header">
        <RunHeaderName run={run} alias={alias} onRename={onRename} />
        <button
          type="button"
          className="run-details-btn"
          title="Open run details — per-call stats, retry chains, outcome buckets"
          aria-label="Open run details"
          onClick={onOpenDetails}
        >
          ⋯
        </button>
      </div>
      {tree.length === 0 ? (
        <p className="empty">
          No vault output yet. Files will appear here as the pipeline writes
          them.
        </p>
      ) : (
        <ul className="run-tree">
          {tree.map((node) => (
            <TreeNode
              key={node.rel_path}
              node={node}
              depth={0}
              runId={run.run_id}
              openDirs={openDirs}
              setOpenDirs={setOpenDirs}
              selectedFile={selectedFile}
              setSelectedFile={setSelectedFile}
              headingsByPath={headingsByPath}
            />
          ))}
        </ul>
      )}
      {/* Primary Run button lives here so the visual slot stays the
          same as the staging-mode "Run with all N files" — the action
          changes with context but the location doesn't. */}
      <div className="pane-footer staging-footer">
        <button
          type="button"
          className={`run-btn ${MODE_META[mode].btnClass}`}
          disabled={!canRun}
          onClick={onRun}
        >
          {runLabel}
        </button>
        {sameModeActive && (
          <p className="hint">
            A {MODE_META[mode].label} run is already active — pause or
            cancel it, or switch to another mode to run in parallel.
          </p>
        )}
      </div>
    </>
  );
}

function TreeNode({
  node,
  depth,
  runId,
  openDirs,
  setOpenDirs,
  selectedFile,
  setSelectedFile,
  headingsByPath,
}) {
  const isOpen = openDirs.has(node.rel_path);
  if (node.is_dir) {
    return (
      <li className="tree-node tree-dir">
        <div
          className="tree-row"
          style={{ paddingLeft: 8 + depth * 14 }}
          onClick={() => {
            setOpenDirs((prev) => {
              const next = new Set(prev);
              if (next.has(node.rel_path)) next.delete(node.rel_path);
              else next.add(node.rel_path);
              return next;
            });
          }}
        >
          <span className="tree-twirl">{isOpen ? "▾" : "▸"}</span>
          <span className="tree-name">{node.name}</span>
        </div>
        {isOpen && node.children.length > 0 && (
          <ul className="tree-children">
            {node.children.map((c) => (
              <TreeNode
                key={c.rel_path}
                node={c}
                depth={depth + 1}
                runId={runId}
                openDirs={openDirs}
                setOpenDirs={setOpenDirs}
                selectedFile={selectedFile}
                setSelectedFile={setSelectedFile}
                headingsByPath={headingsByPath}
              />
            ))}
          </ul>
        )}
      </li>
    );
  }
  const isSelected =
    selectedFile?.runId === runId && selectedFile?.relPath === node.rel_path;
  const entries = headingsByPath?.[node.rel_path];
  const hasEntries = Array.isArray(entries) && entries.length > 0;
  // Toggling on the file row matches the dir-click pattern: re-clicking
  // the row collapses, just like a dir. We also toggle when entries
  // haven't loaded yet for the special insights/actions paths so the
  // expand state is set BEFORE the data lands — otherwise the user
  // would have to click twice to see entries on a fresh run-select.
  const isExpandable =
    hasEntries || EXPANDABLE_TREE_FILES.includes(node.rel_path);
  const toggleOpen = () => {
    setOpenDirs((prev) => {
      const next = new Set(prev);
      if (next.has(node.rel_path)) next.delete(node.rel_path);
      else next.add(node.rel_path);
      return next;
    });
  };
  return (
    <li className={`tree-node tree-file ${isSelected ? "selected" : ""}`}>
      <div
        className="tree-row"
        style={{ paddingLeft: 8 + depth * 14 }}
        onClick={() => {
          if (isExpandable) toggleOpen();
          // Run-tree file click is browsing, not a hop: suppress the
          // transient highlight (`highlight: false`) the same way the
          // tree anchor-entry click does, so opening an entity (or any
          // file) from the tree doesn't flash — only wikilink /
          // citation / cross-link hops do. No-op for anchorless
          // non-entity files (they top-reset without highlighting
          // regardless); load-bearing only for the entity page.
          setSelectedFile({ runId, relPath: node.rel_path, highlight: false });
        }}
      >
        {isExpandable ? (
          <span
            className="tree-twirl"
            onClick={(e) => {
              e.stopPropagation();
              toggleOpen();
            }}
          >
            {isOpen ? "▾" : "▸"}
          </span>
        ) : (
          <span className="tree-bullet">·</span>
        )}
        <span className="tree-name">{node.name}</span>
        {typeof node.size_bytes === "number" && (
          <span className="tree-size">{formatSizeShort(node.size_bytes)}</span>
        )}
      </div>
      {hasEntries && isOpen && (
        <ul className="tree-children tree-file-entries">
          {entries.map((entry) => {
            const entrySelected =
              isSelected && (selectedFile?.anchor || "") === entry.anchor;
            return (
              <li
                key={entry.anchor}
                className={`tree-node tree-file-entry ${entrySelected ? "selected" : ""}`}
              >
                <div
                  className="tree-row"
                  style={{ paddingLeft: 8 + (depth + 1) * 14 }}
                  onClick={(e) => {
                    e.stopPropagation();
                    // Tree-driven anchor jumps skip the smooth-scroll
                    // animation AND the transient highlight: this is
                    // TOC-style navigation, not a wikilink hop, so a
                    // flash on every entry click would be noisy.
                    // Wikilinks keep the default smooth + highlight.
                    setSelectedFile({
                      runId,
                      relPath: node.rel_path,
                      anchor: entry.anchor,
                      scrollBehavior: "auto",
                      highlight: false,
                    });
                  }}
                  title={entry.label}
                >
                  {entry.num != null ? (
                    <span className="tree-num">{entry.num}</span>
                  ) : (
                    <span className="tree-bullet">·</span>
                  )}
                  <span className="tree-name">{entry.label}</span>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </li>
  );
}

// Match an evidence span's file_path against the currently-rendered
// 0-inputs/ file. The vault writer post-fixes `.md` and preserves the
// user's original filename; the pipeline's evidence often emits a
// generic canonical name (e.g. "input.txt", "input.md") that doesn't
// equal the on-disk vault filename. So the matching strategy is:
//
//   1. Exact match: evidence file_path === basename(relPath)
//   2. Stem match : evidence file_path === basename without trailing .md
//   3. Single-file fallback: when the run has exactly one file under
//      0-inputs/, every evidence span references THAT file regardless
//      of the canonical name the pipeline used.
function inputFileMatchesEvidence(relPath, evidenceFilePath, soleInputFile) {
  if (!evidenceFilePath) return false;
  const base = relPath.replace(/^0-inputs\//, "");
  if (evidenceFilePath === base) return true;
  const stem = base.replace(/\.md$/, "");
  if (evidenceFilePath === stem) return true;
  // Single-input fallback: if the rendered file IS the only file
  // under 0-inputs/, treat every span as belonging to it.
  if (soleInputFile && relPath === soleInputFile) return true;
  return false;
}

// Wrap content in <mark> elements at evidence-span byte offsets.
// Sweep-line algorithm: build a list of boundary events (open-N at
// offset, close-N at offset+length), sort, then walk the content
// linearly emitting plain spans + nested marks. Overlapping spans
// produce nested <mark> elements (CSS handles the visual stacking).
function wrapWithMarks(content, spans, onClickMark) {
  if (!spans.length) return [content];
  // Build (offset, kind, ord, span) tuples. `ord` enforces a stable
  // order when two events share an offset: closes before opens, so
  // a span that ends at the same position another starts doesn't
  // visually merge.
  const events = [];
  spans.forEach((s, idx) => {
    events.push({ offset: s.file_offset, kind: "open", ord: idx, span: s });
    events.push({ offset: s.file_offset + s.file_length, kind: "close", ord: idx, span: s });
  });
  events.sort((a, b) => {
    if (a.offset !== b.offset) return a.offset - b.offset;
    if (a.kind !== b.kind) return a.kind === "close" ? -1 : 1;
    return a.ord - b.ord;
  });
  const out = [];
  let cursor = 0;
  let openStack = [];
  let key = 0;
  const totalLen = content.length;
  // Track which `offset-<N>` anchor IDs we've already emitted so a
  // single span doesn't get an id on multiple flush segments — id
  // duplication is invalid HTML and querySelector("#offset-N") would
  // only return the first. The id goes on the FIRST mark that uses
  // a given span. Subsequent flushes for the same span omit it.
  const emittedAnchorIds = new Set();
  // Helper: emit a leaf text node at the deepest open mark, or as
  // raw text if nothing's open. The deepest mark is the one most
  // recently opened — owns the click target visually.
  const flush = (toOffset) => {
    if (toOffset <= cursor) return;
    const slice = content.slice(cursor, Math.min(toOffset, totalLen));
    if (!slice) {
      cursor = toOffset;
      return;
    }
    if (openStack.length === 0) {
      out.push(slice);
    } else {
      const deepest = openStack[openStack.length - 1];
      // Anchor IDs match the wikilink convention the pipeline emits
      // in 1-facts/<topic>.md → `#^offset-<file_offset>`. Pipeline
      // strips the leading `^` at parse time, so the id we emit is
      // `offset-<N>`. Bound to the first mark for a span so the
      // back-link from a fact lands at the START of the highlighted
      // region, not at whatever segment happens to be visible.
      const anchorId = `offset-${deepest.file_offset}`;
      const wantsAnchor = !emittedAnchorIds.has(anchorId);
      if (wantsAnchor) emittedAnchorIds.add(anchorId);
      if (deepest.cited) {
        // Chat chunk-citation passage: same `offset-<N>` anchor so the
        // existing scroll-to-anchor + transient-highlight effect lands
        // here, but it's a non-clickable highlight (no fact to route
        // to) — distinct class so it can be styled apart from the
        // fact-source marks.
        out.push(
          <mark
            key={`cm-${key++}`}
            id={wantsAnchor ? anchorId : undefined}
            className="cited-chunk"
            title="Cited in chat"
          >
            {slice}
          </mark>
        );
      } else {
        out.push(
          <mark
            key={`fm-${key++}`}
            id={wantsAnchor ? anchorId : undefined}
            className="fact-source"
            data-fact-id={deepest.id}
            data-topic={deepest.topic}
            title={deepest.summary || `${deepest.item_type} (${deepest.topic})`}
            onClick={(e) => {
              e.stopPropagation();
              onClickMark(deepest);
            }}
          >
            {slice}
          </mark>
        );
      }
    }
    cursor = toOffset;
  };
  for (const ev of events) {
    if (ev.offset > cursor) flush(ev.offset);
    if (ev.kind === "open") {
      openStack.push(ev.span);
    } else {
      // Close: pop the matching span (could be nested deeper than the
      // top in pathological cases; filter by id reference).
      const idx = openStack.lastIndexOf(ev.span);
      if (idx >= 0) openStack.splice(idx, 1);
    }
  }
  if (cursor < totalLen) flush(totalLen);
  return out;
}

function InputFileView({
  relPath, content, facts, onNavigateToFact, runId, soleInputFile,
  citedChunkOffset, citedChunkLen,
}) {
  // Collect every (fact, evidence-span) pair whose file_path resolves
  // to this input file. We carry the (id, topic, item_type, summary)
  // in each span object so the renderer can show a tooltip + route
  // the click without a second lookup. Plus, when the user arrived
  // here by clicking a chat CHUNK citation, a synthetic span for the
  // grounded passage so it gets the same `offset-<N>` anchor + the
  // transient-highlight the scroll-to-anchor effect already applies —
  // chunks have no fact-evidence mark at their offset on their own.
  const spans = useMemo(() => {
    const out = [];
    if (Array.isArray(facts)) {
      for (const fact of facts) {
        for (const ev of fact.evidence || []) {
          if (!inputFileMatchesEvidence(relPath, ev.file_path, soleInputFile)) continue;
          const start = Number(ev.file_offset) || 0;
          const len = Number(ev.file_length) || 0;
          if (len <= 0) continue;
          out.push({
            file_offset: start,
            file_length: len,
            id: fact.id,
            topic: fact.topic,
            item_type: fact.item_type,
            summary: fact.summary,
          });
        }
      }
    }
    const off = Number(citedChunkOffset);
    if (Number.isInteger(off) && off >= 0 && off < content.length) {
      const len = Number(citedChunkLen);
      let end;
      if (Number.isInteger(len) && len > 0) {
        // The embeddings run persisted the chunk's raw char length, so
        // highlight the WHOLE chunk span `[off, off + len)` — the
        // faithful, exact bound.
        end = Math.min(off + len, content.length);
      } else {
        // Fallback for embeddings that predate the persisted length
        // (old / cache-wiped runs): approximate to the end of the
        // paragraph (next blank line), capped so a file with no blank
        // lines doesn't highlight the whole document.
        const PASSAGE_MAX = 1200;
        end = content.indexOf("\n\n", off);
        if (end < 0) end = content.length;
        end = Math.min(end, off + PASSAGE_MAX, content.length);
      }
      out.push({
        file_offset: off,
        file_length: Math.max(1, end - off),
        cited: true,
      });
    }
    return out;
  }, [facts, relPath, soleInputFile, citedChunkOffset, citedChunkLen, content]);

  const onClickMark = useCallback(
    (span) => {
      if (!onNavigateToFact || !span?.topic || !span?.id) return;
      onNavigateToFact({
        runId,
        relPath: `1-facts/${span.topic}.md`,
        anchor: span.id,
      });
    },
    [onNavigateToFact, runId]
  );

  const wrapped = useMemo(
    () => wrapWithMarks(content, spans, onClickMark),
    [content, spans, onClickMark]
  );

  return (
    <pre className="md-raw input-source">
      {wrapped.length === 0 ? content : wrapped}
    </pre>
  );
}

function MarkdownPreview({
  selectedRun,
  selectedFile,
  content,
  error,
  inputsCount,
  runsCount,
  onNavigate,
  onOpenUrl,
  canBack,
  canForward,
  onBack,
  onForward,
  runFacts,
  soleInputFile,
  refreshTick,
}) {
  // Search-in-pane state (cmd+f / ctrl+f). Visible textbox renders
  // above the content when `searchOpen` is true. Matches are
  // surfaced via the CSS Custom Highlights API (`::highlight()`)
  // rather than DOM-mutating wrappers, so the underlying React
  // tree (which has its own anchor IDs, mark elements, etc.) is
  // never touched.
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchMatches, setSearchMatches] = useState([]);
  const [searchIdx, setSearchIdx] = useState(0);
  const [searchCapped, setSearchCapped] = useState(false);
  const searchInputRef = useRef(null);

  // Scroll-to-anchor + transient highlight, split into two effects:
  //
  // (1) Navigate effect — fires on selectedFile-identity change.
  //     Entity files (`2-entities/<slug>.md`) and anchorless files
  //     scroll to top, no highlight. Anchored jumps land the target
  //     at ~1/3 of the viewport (heading high enough that the next
  //     ~2/3 of body copy is visible). `selectedFile.scrollBehavior`
  //     picks the animation (default "auto"/instant — smooth lands
  //     on stale composite in WKWebView, see nudgePaint); `.highlight = false`
  //     suppresses the fade (used for TOC-style tree clicks).
  //     A MutationObserver re-applies the scroll on every DOM
  //     mutation under scrollRef for a 3s safety window so siblings
  //     rendering after the first scroll keep the target anchored.
  //
  // (2) Retry effect — fires on each `refreshTick` bump (pipeline-
  //     progress-driven stage refresh). Only invokes the navigate
  //     effect's `apply()` while the anchor is still unresolved.
  //     Once `apply()` hits its target it nulls `pendingApplyRef`
  //     so subsequent `refreshTick` bumps become no-ops. Without
  //     this gating, every in-flight pipeline-progress burst would
  //     re-scroll the open file — entity files would snap to top,
  //     anchored files would re-snap to the anchor's 1/3 position —
  //     leaving the user unable to scroll past row 1 during a run
  //     (issue #344).
  const scrollRef = useRef(null);
  const pendingApplyRef = useRef(null);
  useEffect(() => {
    pendingApplyRef.current = null;
    if (!selectedFile) return;
    const isRawInput = /^0-inputs\//.test(selectedFile.relPath);
    if (isRawInput && !content) return; // input view waits on read_run_file
    const isEntityFile = /(?:^|\/)2-entities\//.test(selectedFile.relPath);
    const anchor = selectedFile.anchor;
    const root = scrollRef.current;
    if (!root) return;
    if (isEntityFile && selectedFile.highlight !== false) {
      // Entity highlight-then-fade with NO scroll. The entity view is a
      // single file-level page (no anchor), so it can't ride the
      // scroll-to-anchor path below — but landing on an entity should
      // give the same 2.8s transient ring a cited fact's heading gets,
      // for ANY deliberate hop here: a chat entity citation, a
      // fact→entity wikilink, an entity-relation link, etc. The gate is
      // the SAME rule the anchored fact highlight uses
      // (`highlight !== false`): wikilink / citation / cross-link hops
      // flash; tree- and TOC-style browsing sets `highlight: false`
      // (the run-tree file click + the tree anchor-entry click) so
      // plain browsing stays flash-free — matching how opening a facts
      // file from the tree doesn't flash while a fact wikilink does.
      //
      // EntityView fetches async so its content isn't in the DOM on the
      // first run; reuse the same MutationObserver + refreshTick retry
      // + 3s safety the anchored path uses, applying the ring (no
      // scrollTo) once the page lands. (During an in-progress run the
      // entity may briefly render as the "not yet resolved"
      // placeholder, which also carries the title heading, so the ring
      // still shows.)
      //
      // Target the entity TITLE heading, NOT the `.md-view` container.
      // `.transient-highlight`'s `margin-left:-8px` + `inset 4px 0 0`
      // ring is designed to bleed past indented markdown content; on
      // `.md-view` — the full-width flex child flush inside the
      // x-clipping `.md-scroll` — that negative bleed pushes the ring
      // under the container edge and it renders INVISIBLE (the exact
      // pathology the `.input-row.transient-highlight` override already
      // documents). The `.md-h1` sits inside the article's left
      // padding, so the ring clears it and shows. jsdom has no layout /
      // box-shadow / clipping, so this only reproduces in WebKit.
      let highlighted = false;
      const apply = () => {
        if (highlighted) return;
        const heading =
          root.querySelector(".md-view .md-h1") ||
          root.querySelector(".md-view");
        if (!heading) return;
        highlighted = true;
        heading.classList.remove("transient-highlight");
        // eslint-disable-next-line no-unused-expressions
        heading.offsetWidth; // force reflow to restart the animation
        heading.classList.add("transient-highlight");
        pendingApplyRef.current = null;
      };
      apply();
      pendingApplyRef.current = apply;
      let observer = null;
      if (typeof MutationObserver !== "undefined") {
        observer = new MutationObserver(() => apply());
        observer.observe(root, { childList: true, subtree: true });
      }
      const safety = setTimeout(() => {
        observer?.disconnect();
        observer = null;
        pendingApplyRef.current = null;
      }, 3000);
      return () => {
        observer?.disconnect();
        clearTimeout(safety);
        pendingApplyRef.current = null;
      };
    }
    if (!anchor || isEntityFile) {
      const raf = requestAnimationFrame(() => {
        root.scrollTop = 0;
      });
      return () => cancelAnimationFrame(raf);
    }
    // Default to instant: WKWebView's smooth-scroll animation runs
    // ~300ms behind the React commit and lands the user on a stale
    // composite when the giant FactsView mutation arrives in the same
    // frame — see the nudgePaint comment below. Callers can still pass
    // an explicit `scrollBehavior` if they want the animation back.
    const requestedBehavior = selectedFile.scrollBehavior || "auto";
    const shouldHighlight = selectedFile.highlight !== false;
    const computeTarget = (el) => {
      const elRect = el.getBoundingClientRect();
      const rootRect = root.getBoundingClientRect();
      const elTopInRoot = elRect.top - rootRect.top + root.scrollTop;
      return Math.max(0, elTopInRoot - root.clientHeight / 3);
    };
    // WKWebView leaves the scroll container's compositor layer stale
    // after the huge DOM mutation that lands when async FactsView data
    // (and the matching jump-to-anchor scroll) arrives in one frame —
    // DOM, layout, and hit-testing all update (links are clickable,
    // cursor shape changes) but pixels don't repaint until any pointer
    // event reaches the WebView. User perceives the post-load viewport
    // as blank until they scroll or hover.
    //
    // The fix has two parts:
    //   1) First-hit uses `auto` (instant), not `smooth` — the smooth
    //      animation runs ~300ms behind the React commit, painting the
    //      pre-commit viewport into the layer and leaving the post-
    //      commit layer stale.
    //   2) After every scrollTo, toggle will-change:transform on root
    //      itself for one frame. Sticky positioning is position-based
    //      and unaffected by transform-related layer hints, so this
    //      doesn't break .md-pane-header pinning. The toggle forces a
    //      fresh composite of the scroll container's painted pixels,
    //      which is what a real user scroll would naturally trigger.
    const nudgePaint = () => {
      root.style.willChange = "transform";
      requestAnimationFrame(() => {
        root.style.willChange = "";
      });
    };
    let firstHit = false;
    let lastTarget = -1;
    const apply = () => {
      const el = root.querySelector(
        `[id="${anchor.replace(/"/g, '\\"')}"]`,
      );
      if (!el) return;
      const target = computeTarget(el);
      if (!firstHit) {
        firstHit = true;
        root.scrollTo({ top: target, behavior: requestedBehavior });
        nudgePaint();
        if (shouldHighlight) {
          // Highlight the WHOLE section (heading + body) when the
          // anchor points at a section heading; otherwise the bare
          // paragraph.
          let highlightTarget = el;
          const sectionAncestor = el.closest(".md-section");
          if (sectionAncestor) {
            const sectionHeading = sectionAncestor.querySelector(
              ":scope > .md-section-heading",
            );
            if (sectionHeading && sectionHeading.id === anchor) {
              highlightTarget = sectionAncestor;
            }
          }
          highlightTarget.classList.remove("transient-highlight");
          // eslint-disable-next-line no-unused-expressions
          highlightTarget.offsetWidth;
          highlightTarget.classList.add("transient-highlight");
        }
        lastTarget = target;
        // Anchor resolved — retry effect short-circuits from here on.
        // MutationObserver still re-aligns within its 3s window for
        // siblings growing under the user.
        pendingApplyRef.current = null;
        return;
      }
      // Re-aligns after the first hit: only if the target moved more
      // than a px (avoid pointless calls). Always instant — competing
      // smooth animations would visibly fight each other; the first
      // smooth scroll has already played.
      if (Math.abs(target - lastTarget) < 2) return;
      root.scrollTo({ top: target, behavior: "auto" });
      nudgePaint();
      lastTarget = target;
    };
    apply();
    // Expose to the retry effect so refreshTick bumps can keep trying
    // until the anchor lands in the DOM. apply() nulls the ref itself
    // on firstHit, capping retries.
    pendingApplyRef.current = apply;
    let observer = null;
    if (typeof MutationObserver !== "undefined") {
      observer = new MutationObserver(() => apply());
      observer.observe(root, { childList: true, subtree: true });
    }
    const safety = setTimeout(() => {
      observer?.disconnect();
      observer = null;
      pendingApplyRef.current = null;
    }, 3000);
    return () => {
      observer?.disconnect();
      clearTimeout(safety);
      pendingApplyRef.current = null;
    };
  }, [
    content,
    selectedFile?.relPath,
    selectedFile?.anchor,
    selectedFile?.runId,
  ]);
  useEffect(() => {
    pendingApplyRef.current?.();
  }, [refreshTick]);

  // Cmd+F / Ctrl+F to open the search bar. Esc closes. Listener is
  // mounted on the window but only fires when this pane is the
  // user's intent — we accept the small global-hotkey footprint
  // since the markdown pane is the only place text-search makes
  // sense.
  useEffect(() => {
    function onKey(e) {
      const isFind = (e.metaKey || e.ctrlKey) && e.key === "f";
      if (isFind) {
        if (!selectedFile) return;
        e.preventDefault();
        setSearchOpen(true);
        // Defer the focus to next tick so the input has mounted.
        requestAnimationFrame(() => searchInputRef.current?.select());
        return;
      }
      if (e.key === "Escape" && searchOpen) {
        setSearchOpen(false);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedFile, searchOpen]);

  // Recompute match Ranges whenever the (debounced) query, content,
  // or open state changes. Walking strategy: collect every text node
  // under scrollRef into a single concatenated string with a
  // (start, end) index per node. Run the regex once on the concat,
  // then map each match back to a Range that may span multiple
  // adjacent text nodes — necessary because the input view splits
  // text into many fragments interleaved with fact-source <mark>
  // children, and a per-node regex would miss queries that straddle
  // a mark boundary (e.g. "DISAPPOINTED" with "DIS" outside and
  // "APPOINTED" inside a mark would silently disappear).
  const debouncedQuery = useDebouncedValue(
    searchOpen ? searchQuery : "",
    150,
  );
  useEffect(() => {
    if (!searchOpen || !debouncedQuery || !scrollRef.current) {
      setSearchMatches([]);
      setSearchIdx(0);
      setSearchCapped(false);
      if (typeof CSS !== "undefined" && CSS.highlights) {
        CSS.highlights.delete("md-search");
        CSS.highlights.delete("md-search-current");
      }
      return;
    }
    const root = scrollRef.current;
    const escaped = debouncedQuery.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(escaped, "gi");
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => {
        // Skip text inside the search bar itself + inside the
        // top header chrome so a query that matches a button label
        // doesn't surface as a match.
        let parent = n.parentNode;
        while (parent) {
          if (parent.classList?.contains("md-pane-header")) return NodeFilter.FILTER_REJECT;
          if (parent.classList?.contains("md-search-bar")) return NodeFilter.FILTER_REJECT;
          parent = parent.parentNode;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    // Collect nodes + concat. We don't walk-and-match in one pass
    // because we need stable indices into the concatenated string.
    const segments = [];
    let concat = "";
    let node;
    while ((node = walker.nextNode())) {
      const start = concat.length;
      const text = node.textContent;
      concat += text;
      segments.push({ node, start, end: start + text.length });
    }
    const ranges = [];
    // Two cursors thread monotonically across the segments array as
    // the regex iterator walks the concat. Both `m.index` and
    // `m.index + m[0].length` are non-decreasing across iterations
    // (regex /g over non-empty matches), so the cursors only ever
    // advance — making the per-match locator amortized O(1).
    let startCursor = 0;
    let endCursor = 0;
    let capped = false;
    let m;
    while ((m = re.exec(concat)) !== null) {
      if (m[0].length === 0) {
        re.lastIndex += 1;
        continue;
      }
      if (ranges.length >= MAX_SEARCH_MATCHES) {
        capped = true;
        break;
      }
      const startAbs = m.index;
      const endAbs = m.index + m[0].length;
      const startSeg = findSegmentForward(segments, startAbs, startCursor);
      if (startSeg < 0) break;
      startCursor = startSeg;
      if (endCursor < startSeg) endCursor = startSeg;
      // End offset is exclusive; the segment that contains it is the
      // one whose end is strictly greater than endAbs. Walk forward
      // from endCursor (which is at least startSeg).
      while (endCursor < segments.length - 1 && segments[endCursor].end < endAbs) {
        endCursor += 1;
      }
      try {
        const range = document.createRange();
        range.setStart(segments[startSeg].node, startAbs - segments[startSeg].start);
        range.setEnd(segments[endCursor].node, endAbs - segments[endCursor].start);
        ranges.push(range);
      } catch {
        // Range construction can throw on detached nodes between
        // walk + use; skip the offender and keep going.
      }
    }
    setSearchMatches(ranges);
    setSearchIdx(0);
    setSearchCapped(capped);
  }, [searchOpen, debouncedQuery, content, selectedFile?.relPath]);

  // Paint matches via CSS Custom Highlights API. The all-set excludes
  // the active range (rather than including it and relying on
  // `Highlight.priority`) so the active colour swap doesn't depend on
  // whether the underlying engine implements priority — earlier
  // WKWebView versions do not, and a wrong colour on the active match
  // destroys the Up/Down feedback. Rebuild on every (matches, idx)
  // change; at MAX_SEARCH_MATCHES the rebuild is a few ms, invisible
  // next to the regex/locator path.
  useEffect(() => {
    if (typeof CSS === "undefined" || !CSS.highlights) return;
    if (!searchMatches.length) {
      CSS.highlights.delete("md-search");
      CSS.highlights.delete("md-search-current");
      return;
    }
    // Constructor's varargs form `new Highlight(...ranges)` would spread
    // up to MAX_SEARCH_MATCHES arguments and trip engine arg-list limits;
    // build empty and add ranges in a loop.
    const all = new Highlight();
    for (let i = 0; i < searchMatches.length; i++) {
      if (i !== searchIdx) all.add(searchMatches[i]);
    }
    CSS.highlights.set("md-search", all);

    const current = searchMatches[searchIdx];
    if (!current) {
      CSS.highlights.delete("md-search-current");
      return;
    }
    CSS.highlights.set("md-search-current", new Highlight(current));
    // Centre the Range itself in the scroll container, not its parent.
    // `parent.scrollIntoView({block:"center"})` centres a paragraph-
    // sized element, which on long paragraphs can leave the actual
    // match offscreen at the top or bottom edge of the now-centred
    // parent. Manual `scrollBy(delta)` against the Range's own rect
    // always lands the match at viewport centre.
    try {
      const rect = current.getBoundingClientRect();
      const container = scrollRef.current;
      if (container && rect.height > 0) {
        const cRect = container.getBoundingClientRect();
        const target = rect.top + rect.height / 2 - cRect.top - cRect.height / 2;
        if (Math.abs(target) > 1) {
          container.scrollBy({ top: target, behavior: "auto" });
        }
      }
    } catch {
      // getBoundingClientRect can throw on detached ranges; bail.
    }
  }, [searchMatches, searchIdx]);

  // Clean up CSS highlights when the pane unmounts or the file
  // changes so stale highlights don't bleed across navigations.
  useEffect(() => {
    return () => {
      if (typeof CSS !== "undefined" && CSS.highlights) {
        CSS.highlights.delete("md-search");
        CSS.highlights.delete("md-search-current");
      }
    };
  }, []);

  const cycleMatch = useCallback((dir) => {
    setSearchIdx((prev) => {
      if (!searchMatches.length) return 0;
      const next = (prev + dir + searchMatches.length) % searchMatches.length;
      return next;
    });
  }, [searchMatches]);

  // Browser-style nav buttons. Always rendered when a file is open
  // so users have a consistent control surface; disabled at history
  // boundaries.
  const navHeader = selectedFile ? (
    <div className="md-pane-header">
      <div className="md-nav">
        <button
          type="button"
          className="md-nav-btn"
          disabled={!canBack}
          onClick={onBack}
          title="Back"
          aria-label="Back"
        >
          ‹
        </button>
        <button
          type="button"
          className="md-nav-btn"
          disabled={!canForward}
          onClick={onForward}
          title="Forward"
          aria-label="Forward"
        >
          ›
        </button>
      </div>
      <span className="pane-title">{displayBasename(selectedFile.relPath)}</span>
      {searchOpen && (
        <div className="md-search-bar">
          <input
            ref={searchInputRef}
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                cycleMatch(e.shiftKey ? -1 : 1);
              } else if (e.key === "Escape") {
                e.preventDefault();
                setSearchOpen(false);
              }
            }}
            placeholder="Search…"
            aria-label="Find in document"
          />
          <span
            className="md-search-count"
            title={searchCapped ? `Showing first ${MAX_SEARCH_MATCHES} matches; refine the query to narrow.` : undefined}
          >
            {searchMatches.length === 0
              ? searchQuery
                ? "0/0"
                : ""
              : `${searchIdx + 1}/${searchMatches.length}${searchCapped ? "+" : ""}`}
          </span>
          <button
            type="button"
            className="md-search-btn"
            disabled={searchMatches.length === 0}
            // mousedown's default action is to move focus to the button;
            // suppressing it keeps the search input focused so Enter /
            // Shift+Enter still chain after a click.
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => cycleMatch(-1)}
            title="Previous match (Shift+Enter)"
            aria-label="Previous match"
          >
            ▲
          </button>
          <button
            type="button"
            className="md-search-btn"
            disabled={searchMatches.length === 0}
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => cycleMatch(1)}
            title="Next match (Enter)"
            aria-label="Next match"
          >
            ▼
          </button>
          <button
            type="button"
            className="md-search-btn md-search-btn-close"
            onClick={() => setSearchOpen(false)}
            title="Close (Esc)"
            aria-label="Close search"
          >
            ✕
          </button>
        </div>
      )}
    </div>
  ) : null;
  if (selectedFile && error) {
    return (
      <div className="md-scroll" ref={scrollRef}>
        {navHeader}
        <div className="md-empty">
          <p className="error-text">Couldn't read file: {error}</p>
        </div>
      </div>
    );
  }
  if (selectedFile) {
    const rel = selectedFile.relPath;
    const isRawInput = /^0-inputs\//.test(rel);
    const factTopic = rel.match(/^1-facts\/(.+)\.md$/)?.[1] || null;
    const entityId = rel.match(/^2-entities\/(.+)\.md$/)?.[1] || null;
    const patternTopic = rel.match(/^3-patterns\/(.+)\.md$/)?.[1] || null;
    const isInsights = rel === INSIGHTS_FILE;
    const isActions = rel === ACTIONS_FILE;

    let body = null;
    if (isRawInput) {
      // Raw input view depends on read_run_file content; wait on it.
      if (!content) {
        return (
          <div className="md-scroll" ref={scrollRef}>
            {navHeader}
          </div>
        );
      }
      body = (
        <InputFileView
          relPath={rel}
          content={content}
          facts={runFacts || []}
          onNavigateToFact={onNavigate}
          runId={selectedFile.runId}
          soleInputFile={soleInputFile}
          citedChunkOffset={selectedFile.chunkOffset}
          citedChunkLen={selectedFile.chunkLen}
        />
      );
    } else if (factTopic) {
      body = (
        <FactsView
          runId={selectedFile.runId}
          topic={factTopic}
          refreshTick={refreshTick}
          onNavigate={onNavigate}
        />
      );
    } else if (entityId) {
      body = (
        <EntityView
          runId={selectedFile.runId}
          entityId={entityId}
          refreshTick={refreshTick}
          onNavigate={onNavigate}
        />
      );
    } else if (patternTopic) {
      body = (
        <PatternsView
          runId={selectedFile.runId}
          topic={patternTopic}
          refreshTick={refreshTick}
          onNavigate={onNavigate}
        />
      );
    } else if (isInsights) {
      body = (
        <InsightsView
          runId={selectedFile.runId}
          refreshTick={refreshTick}
          onNavigate={onNavigate}
        />
      );
    } else if (isActions) {
      body = (
        <ActionsView
          runId={selectedFile.runId}
          refreshTick={refreshTick}
          onNavigate={onNavigate}
        />
      );
    } else if (content) {
      // Unknown path with raw text content (only `0-inputs/*` reaches
      // here today, and that's caught by the InputFileView branch
      // above). Show the raw payload so the user isn't left staring at
      // an empty pane.
      body = <pre className="md-raw">{content}</pre>;
    } else {
      body = (
        <div className="md-empty">
          <p>No view registered for {rel}.</p>
        </div>
      );
    }

    return (
      <div className="md-scroll" ref={scrollRef}>
        {navHeader}
        {body}
      </div>
    );
  }
  if (selectedRun) {
    return (
      <div className="md-empty">
        <p>Pick a file from the run tree to preview.</p>
      </div>
    );
  }
  if (runsCount === 0 && inputsCount === 0) {
    return (
      <div className="md-empty">
        <h3>Ready when you are.</h3>
        <p>
          Add files in the <strong>middle pane</strong>, then run the pipeline.
          Outputs will appear here.
        </p>
      </div>
    );
  }
  return (
    <div className="md-empty">
      <p>Select a run on the left to browse its output.</p>
    </div>
  );
}

function AttestationIndicator({ attestation, checking, onClick }) {
  // De-emphasized status light (not a hero badge): the NAME is always
  // "Attestations"; a lock icon + subtle color convey state, and the
  // tooltip carries the detail. Attestation is non-blocking — this only
  // surfaces the last check, it gates nothing.
  let icon;
  let className = "att-indicator";
  let label;
  const text = "Attestations";
  if (checking || attestation === null) {
    icon = "⏳";
    className += " att-checking";
    label = "Verifying attestations…";
  } else if (attestation.ok) {
    icon = "🔒";
    className += " att-ok";
    label = "Attestations verified — click for details";
  } else if (attestation.transient) {
    icon = "⏳";
    className += " att-checking";
    label = "Verifying attestations… (temporary hiccup, retrying)";
  } else {
    icon = "🔓";
    className += " att-fail";
    label = `Attestation failed — click for details: ${attestation.error || "unknown"}`;
  }
  return (
    <button
      type="button"
      className={className}
      onClick={onClick}
      title={label}
      aria-label={label}
    >
      <span className="att-icon">{icon}</span>
      <span className="att-text">{text}</span>
    </button>
  );
}

function AttestationModal({ attestation, checking, onRecheck, onClose }) {
  // ``checking`` from caller drives the spinner without unmounting the
  // previous result's chain — preserves modal size during a Recheck.
  // ``attestation === null`` only happens before the first verify
  // resolves; treat that as checking too.
  const isChecking = checking || attestation === null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal attestation-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2>Attestations</h2>
          <button
            type="button"
            className="modal-close"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <p className="field-hint">
          Verifies that the Private Cloud backend is running unmodified inside
          a Trusted Execution Environment. Probed automatically on launch.
        </p>
        <AttestationPanel
          status={attestation}
          checking={isChecking}
          onRecheck={onRecheck}
        />
        <div className="modal-actions">
          <button type="button" className="btn-primary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Single source of truth for the run-row chip's percentage.
 *
 * Issue #209 — DO NOT inline this formula at the call site, and DO NOT
 * source pct from anywhere other than displayDone/total. The chip
 * renders one string ("X% completed (Y / Z calls)") and the X must
 * always equal round(Y/Z*100). When this helper is the only place pct
 * is computed, the chip and the bar can't drift apart.
 *
 * Pre-#209 there was a `bar_position` field emitted by the python
 * tracker (= `tracker.completed_calls / est_calls`) that drove the
 * bar's pct independently of the chip's count. It diverged from
 * rust's leaf count on retry-storm runs and produced "50% completed
 * (22 / 42 calls)". The cleanup removed `bar_position` from the
 * payload entirely; this helper has only one input source and no
 * fallback path that could accidentally re-introduce drift.
 *
 * Inputs:
 *   status      run.status — "completed" forces 100.
 *   stage       run.progress.stage — "done" also forces 100 (the
 *               runner emits this on natural completion before the
 *               status flips).
 *   displayDone the clamped leaf count from rust derivation.
 *   total       the python tracker's est_calls sum (denominator).
 *
 * Output: integer in [0, 100]. Capped at 99 mid-run so the bar can't
 * read 100% before the runner declares done. Returns 0 when total === 0
 * (pre-tick window before the first progress_tick lands).
 */
export function progressChipPct({ status, stage, displayDone, total }) {
  if (status === "completed" || stage === "done") return 100;
  if (total <= 0) return 0;
  if (displayDone >= total) return 99;
  return Math.min(99, Math.round((displayDone / total) * 100));
}

function RunRow({
  run,
  alias,
  selected,
  current,
  isPausing,
  isResuming,
  nowMs,
  reducedMotion,
  onSelect,
  onPause,
  onResume,
  onCancel,
  onDelete,
}) {
  const total = run.progress?.total || 0;
  const done = run.progress?.completed || 0;
  // Defensive clamp: rust-derived `done` counts LEAF end events
  // (issue #209). There's a transient window between a parent's `end`
  // landing and its child retry's `begin` landing where the parent
  // briefly looks like a leaf — `done` could outpace `total` until the
  // child's begin demotes the parent. Clamp on the display side so the
  // chip never reads `100 / 42`. The leaf-count math is the real fix;
  // this is belt-and-suspenders.
  const displayDone = total > 0 ? Math.min(done, total) : done;
  const retries = run.progress?.retries || 0;
  const etaSeconds = run.progress?.eta_seconds;
  const elapsedLabel = fmtElapsed(liveElapsedMs(run, nowMs));
  // ───────────────────────────────────────────────────────────────────
  // INVARIANT — DO NOT BREAK (issue #209)
  // ───────────────────────────────────────────────────────────────────
  // The chip text below renders ONE string: `${pct}% completed
  // (${displayDone} / ${total} calls[, +${retries} retries])`. The
  // user reads this as a single fact stated two ways. The `pct` and
  // the `(displayDone / total)` MUST be the same fraction. If they
  // can differ, the UI lies (we shipped "50% completed (22 / 42 calls)"
  // for months — pre-#209 the bar pulled from python's `bar_position`
  // and the chip text from rust's leaf count, two independent
  // numerators feeding one string).
  //
  // Hard rule when changing this block:
  //   pct === Math.round(displayDone / total * 100)
  //   (modulo the [0, 99] clamp + the explicit 100 for done state)
  //
  // `progressChipPct` is the SINGLE source of the bar's percentage —
  // call it from both the bar-fill width and the chip text. Don't
  // inline a divide here, don't pull pct from any other field, don't
  // add a fallback that reads a separate "convenient" pre-computed
  // value (the cleanup removed `bar_position` from the payload to
  // make this impossible). The DOM-parsing test in App.test.jsx
  // (`chip pct always equals round(done/total*100)`) will fail
  // loudly if the bar and chip diverge.
  // ───────────────────────────────────────────────────────────────────
  const pct = progressChipPct({
    status: run.status,
    stage: run.progress?.stage,
    displayDone,
    total,
  });
  const stageLabel =
    STAGE_LABELS[run.progress?.stage] || run.progress?.stage || "";
  const etaLabel = formatEta(etaSeconds);
  const statusLabel = STATUS_LABELS[run.status] || run.status || "";

  return (
    <li
      data-run-id={run.run_id}
      className={`run-row status-${run.status || "unknown"} mode-${
        run.mode || "local"
      } ${selected ? "selected" : ""} ${current ? "current" : ""}`}
      onClick={(e) => onSelect(e)}
    >
      {/* Two-line entry (director layout, shared with the chat picker):
          line 1 = creation date (left) + the immutable #<4-letter id>
          pinned top-right, ALWAYS shown; line 2 = the renameable name /
          alias, secondary. The perma-id is primary and survives rename
          (rename only edits the alias on line 2); the dir-name-suffix
          fallback (Rust read path) guarantees short_id is populated. */}
      <div className="run-header">
        <div className="run-id-line">
          <span className="run-date">{formatDate(run.created_at)}</span>
          {run.short_id && (
            <span
              className="run-short-id"
              title="Run ID — quote these 4 letters to find this run in logs / vault"
            >
              #{run.short_id}
            </span>
          )}
        </div>
        <span
          className="run-title"
          title={(run.inputs || []).join("\n")}
        >
          {alias || runTitle(run)}
        </span>
      </div>
      <div className="run-meta">
        <span
          className={`status-badge status-${
            isPausing ? "pausing" : isResuming ? "resuming" : run.status || "unknown"
          } mode-${run.mode || "local"}`}
        >
          {isPausing ? "Pausing…" : isResuming ? "Resuming…" : statusLabel}
        </span>
        {/* The "running" time sits next to the status badge. It reuses
            the SAME live-elapsed value rendered under the progress bar
            (elapsedLabel, computed once above from
            liveElapsedMs(run, nowMs)) — one clock, not two. Running ==
            elapsed by construction, ticking on the same 1Hz wallclock,
            hour-rolled "Hh Mm Ss" (issue #586). Finished runs fall back
            to duration_ms through liveElapsedMs; null → "—". */}
        <span className="run-dur">{elapsedLabel || "—"}</span>
        <span className="run-count" title="files — calls in flight (running) or productive leaves (done); +N retries surfaces the retry storm">
          {`${run.input_count || 0} file${run.input_count === 1 ? "" : "s"}`}
          {run.status === "running" || run.status === "paused"
            ? ` — ${run.progress?.in_flight_calls || 0} call${
                (run.progress?.in_flight_calls || 0) === 1 ? "" : "s"
              } in progress`
            : run.progress?.completed > 0
              ? ` — ${run.progress.completed} call${
                  run.progress.completed === 1 ? "" : "s"
                }${retries > 0 ? ` (+${retries} retries)` : ""}`
              : ""}
        </span>
      </div>
      {(run.provider || run.model || run.mode) && (
        <div className="run-meta run-engine-row">
          <span
            className="run-engine"
            title={`mode=${run.mode || "?"}\nprovider=${run.provider || "?"}\nmodel=${run.model || "?"}`}
          >
            {MODE_META[run.mode]?.label || run.mode || ""}
            {/* Issue #161: drop the ` / <modelDisplay>` segment from
                the run-row card. Per-stage routing (and the model
                name for single-model runs) is the source-of-truth in
                the Details modal + RunDetails routing snapshot, so
                showing it here too is redundant churn. The `TEE`
                marker that used to ride on the model now rides on
                the provider, gated on tee-mode. */}
            {run.provider
              ? ` (${providerDisplayName(run.provider)}${run.mode === "tee" ? " TEE" : ""})`
              : ""}
          </span>
          {(() => {
            const stages = reasoningStages(run);
            if (stages.length === 0) return null;
            return (
              <span
                className="run-reasoning-badge"
                title={`Reasoning enabled on: ${stages.join(", ")}`}
              >
                🧠
              </span>
            );
          })()}
        </div>
      )}
      {(run.status === "running" || run.status === "paused") && (
        <div className="run-progress">
          {/* Issue #161 layout: stage on its own line, progress bar,
              then two stacked detail lines (counts on top, elapsed +
              ETA below). The previous one-line "stage · pct · elapsed
              · ETA" jammed too much into a narrow sidebar. */}
          <div className="progress-stage">{stageLabel}</div>
          <div className="progress-track">
            <div
              className={`progress-fill progress-fill-${run.status} mode-${
                run.mode || "local"
              }${run.status === "running" && !reducedMotion ? " is-running" : ""}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="progress-detail">
            <span
              className="progress-counts"
              title={
                retries > 0
                  ? `${retries} retry call${retries === 1 ? "" : "s"} ` +
                    `not counted toward leaves (load backoff, halve ` +
                    `cascade intermediates, sample iterations, ` +
                    `reasoning-off retries). The X / Y count is productive ` +
                    `leaves on both sides.`
                  : undefined
              }
            >
              {pct}% completed
              {total > 0 ? ` (${displayDone} / ${total} estimated calls` : ""}
              {total > 0 && retries > 0 ? `, +${retries} retries` : ""}
              {total > 0 ? ")" : ""}
            </span>
            {(elapsedLabel || etaLabel) && (
              <span className="progress-elapsed">
                {elapsedLabel ? `Elapsed ${elapsedLabel}` : ""}
                {elapsedLabel && etaLabel ? " " : ""}
                {etaLabel ? `(${etaLabel})` : ""}
              </span>
            )}
          </div>
        </div>
      )}
      {run.error && (
        <div
          className={
            run.status === "cancelled" || run.status === "paused"
              ? "run-note"
              : "run-error"
          }
          title={run.error}
        >
          {String(run.error).split("\n")[0]}
        </div>
      )}
      {(run.warnings?.cap_hits > 0
        || run.warnings?.empty_responses > 0
        || run.warnings?.interrupted > 0
        || run.warnings?.input_overflows > 0
        || run.warnings?.timeouts > 0
        || run.warnings?.parse_errors > 0
        || run.warnings?.sampled > 0
        || run.warnings?.failed > 0) && (() => {
        // Counts only the ⚠-firing leaves: actual failures +
        // input_overflow + sampled (sizing retry that dropped
        // inputs). `success_empty` and `success_reasoning_off`
        // are caveat-yellow but DATA-COMPLETE — they don't fire
        // the ⚠ icon and don't count toward the run badge.
        const cap = run.warnings.cap_hits || 0;
        const empty = run.warnings.empty_responses || 0;
        const intr = run.warnings.interrupted || 0;
        const over = run.warnings.input_overflows || 0;
        const tout = run.warnings.timeouts || 0;
        const perr = run.warnings.parse_errors || 0;
        const samp = run.warnings.sampled || 0;
        const failed = run.warnings.failed || 0;
        const total = (
          cap + empty + intr + over + tout + perr + samp + failed
        );
        const parts = [];
        if (cap) parts.push(`${cap} hit output cap`);
        if (empty) parts.push(`${empty} returned empty`);
        if (intr) parts.push(`${intr} stream interrupted`);
        if (over) parts.push(`${over} input overflowed`);
        if (tout) parts.push(`${tout} timed out`);
        if (perr) parts.push(`${perr} parse error${perr === 1 ? "" : "s"}`);
        if (samp) parts.push(`${samp} succeeded with reduced inputs`);
        if (failed) parts.push(`${failed} failed`);
        return (
          <div
            className="run-warning"
            title={
              `${total} LLM call(s) flagged. Open the run (click ⋯) ` +
              `for per-call detail.`
            }
          >
            ⚠ {total} call{total === 1 ? "" : "s"} flagged —
            {" "}results may be incomplete
          </div>
        );
      })()}
      {/* Per-row icon buttons for in-flight controls only. Trash + book
          were removed (the bottom-of-pane Delete handles bulk delete;
          the file-tree pane replaces "Open vault in Obsidian"). The
          regen-vault arrow stays — it's a recovery action with no
          equivalent elsewhere. */}
      <div className="run-actions" onClick={(e) => e.stopPropagation()}>
          {run.status === "running" && !isPausing && !isResuming && (
            <>
              <button type="button" className="icon-btn" title="Pause" onClick={onPause}>⏸</button>
              <button type="button" className="icon-btn" title="Cancel" onClick={onCancel}>✕</button>
            </>
          )}
          {run.status === "paused" && !isPausing && !isResuming && (
            <>
              <button type="button" className="icon-btn" title="Resume" onClick={onResume}>▶</button>
              <button type="button" className="icon-btn" title="Cancel" onClick={onCancel}>✕</button>
            </>
          )}
          {(isPausing || isResuming) && (
            // In-flight transition: keep Cancel reachable so the user
            // isn't stranded if the runner hangs. Pause / Resume are
            // suppressed (clicking again would only race with the
            // call already in flight).
            <button type="button" className="icon-btn" title="Cancel" onClick={onCancel}>✕</button>
          )}
        </div>
    </li>
  );
}

function ConfirmDialog({ title, message, confirmLabel, onConfirm, onCancel }) {
  // Escape closes the dialog — matches the Wizard / Attestation modals.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onCancel?.();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);
  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div className="modal confirm-modal" onClick={(e) => e.stopPropagation()}>
        <h2>{title}</h2>
        <p>{message}</p>
        <div className="modal-actions">
          <button type="button" className="btn-secondary" onClick={onCancel}>
            No, keep it
          </button>
          <button
            type="button"
            className="btn-primary btn-danger"
            onClick={onConfirm}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

// Privacy Level gate shown before any export. One ordinal
// slider: the handle position is the privacy statement. The floor
// (Actions + Insights) is always included and cannot be dragged below;
// the rightmost stop (Raw Inputs) crosses the trust boundary and is
// gated behind an explicit acknowledgement.
function PrivacyLevelModal({
  counts, runCount, privacyLevel, setPrivacyLevel,
  exportDir, onPickDir, onExport, onCancel,
}) {
  const [rawAck, setRawAck] = useState(false);
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") { e.stopPropagation(); onCancel?.(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  let idx = PRIVACY_SLIDER.findIndex((s) => s.level === privacyLevel);
  if (idx < 0) idx = PRIVACY_FLOOR_INDEX;
  const cur = PRIVACY_SLIDER[idx];
  const rawSelected = !!cur.raw;
  const exportDisabled = rawSelected && !rawAck;

  const plur = (n, s) => (n === 1 ? s.noun : s.nounPlural || `${s.noun}s`);

  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div
        className="modal privacy-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <h1>Privacy Level</h1>
        <p className="modal-lead">
          Filter your personal data before sharing with other apps,
          people, or AI agents.
          {runCount > 1 ? ` Applies to all ${runCount} selected runs.` : ""}
        </p>

        <div className="privacy-bar">
          <div className="privacy-ticks">
            {PRIVACY_SLIDER.map((s, i) => {
              const last = PRIVACY_SLIDER.length - 1;
              const tx = i === 0 ? "0" : i === last ? "-100%" : "-50%";
              return (
                <span
                  key={s.level}
                  className={
                    i < PRIVACY_FLOOR_INDEX
                      ? "disabled"
                      : i <= idx ? "included" : "excluded"
                  }
                  style={{ left: `${(i / last) * 100}%`, transform: `translateX(${tx})` }}
                >
                  {/* Both floor levels carry the lock (always exported);
                      only Actions is non-selectable — Insights stays the
                      leftmost selectable tick. */}
                  {s.floor ? `🔒 ${s.name}` : s.name}
                </span>
              );
            })}
          </div>
          <input
            type="range"
            className="privacy-slider"
            min={0}
            max={PRIVACY_SLIDER.length - 1}
            step={1}
            value={idx}
            onChange={(e) => {
              // 6 evenly-spaced ticks align with the 6 labels; the
              // handle just can't rest below the Actions+Insights floor.
              const v = Math.max(PRIVACY_FLOOR_INDEX, Number(e.target.value));
              setPrivacyLevel(PRIVACY_SLIDER[v].level);
              if (!PRIVACY_SLIDER[v].raw) setRawAck(false);
            }}
          />
          <div className="privacy-ends">
            <span>most private</span>
            <span>least private (incl. raw documents)</span>
          </div>
        </div>

        <ul className="privacy-preview">
          {PRIVACY_SLIDER.map((s, i) => {
            const included = i <= idx;
            const n = counts[s.countKey];
            return (
              <li
                key={s.level}
                className={included ? "included" : "excluded"}
              >
                <span className="privacy-row-label">{s.name}</span>
                <span className="privacy-row-count">
                  {n} {plur(n, s)}
                  {s.floor ? " · always included" : ""}
                </span>
              </li>
            );
          })}
        </ul>

        {rawSelected && (
          <div className="privacy-warning">
            <p>
              ⚠ Raw source documents will leave this machine. This is the
              genuinely irreversible step — the unprocessed personal
              substrate, not just the derived layers.
            </p>
            <label>
              <input
                type="checkbox"
                checked={rawAck}
                onChange={(e) => setRawAck(e.target.checked)}
              />
              I understand
            </label>
          </div>
        )}

        <div className="privacy-dest">
          <span className="privacy-dest-label">Destination</span>
          <code title={exportDir || ""}>{exportDir || "…"}</code>
          <button type="button" className="btn-secondary" onClick={onPickDir}>
            Change…
          </button>
        </div>

        <div className="modal-actions">
          <button type="button" className="btn-secondary" onClick={onCancel}>
            Cancel
          </button>
          <button
            type="button"
            className="btn-primary"
            disabled={exportDisabled}
            onClick={onExport}
          >
            Export
          </button>
        </div>
      </div>
    </div>
  );
}

// Day-One-style "Export completed" confirmation: centered app logo,
// bold title, stacked primary + secondary buttons.
function ExportSuccessModal({ dest, onClose }) {
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") { e.stopPropagation(); onClose?.(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal success-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <img className="success-logo" src={baseVaultLogo} alt="" />
        <h1>Export successful</h1>
        <div className="success-actions">
          <button type="button" className="btn-primary" onClick={onClose}>
            OK
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={() =>
              invoke("reveal_in_finder", { path: dest }).catch((e) =>
                console.error("reveal_in_finder:", e)
              )
            }
          >
            Show in Finder
          </button>
        </div>
      </div>
    </div>
  );
}

// Display labels for per-call outcome buckets (issue #104, expanded
// with `cap_hit` per #175 + `pending` per #165's in-flight
// materializer + `aborted` per #206's call-classification fix).
// Mirror python's OUTCOME_* constants in engine/runner.py —
// keep the keys in sync if you add a new outcome there.
//
// `cap_hit` (#175): the model finished cleanly but with
// `finish_reason="length"` — output truncated at the max_tokens
// boundary. Distinct from `success` so the user can tell at a glance
// the response is incomplete by construction.
//
// `pending`: in-flight-only — a call whose `begin` event landed but
// `end` hasn't yet, on a run that's still alive. Semantics: not a
// failure, not a success yet — just waiting. Renders neutral.
// Short display labels for the canonical success / pending / aborted
// outcomes. Failure outcomes are passed through verbatim
// ("cap_hit (sizing)", "failed (load)", etc.) — no abbreviation
// since the classification suffix is itself the user-visible signal.
//
// `aborted`: begin landed, end never did, and we've stopped waiting —
// paused / cancelled / errored / superseded by resume. Distinct from
// `pending` (still waiting) and from any `failed (X)` (call didn't
// error). No warning glyph: aborted calls redo cheaply via the LLM
// cache, nothing for the user to act on.
const OUTCOME_LABELS = {
  success: "ok",
  success_empty: "empty (success)",
  success_sampled: "sampled (success)",
  success_reasoning_off: "reasoning_off (success)",
  pending: "pending",
  aborted: "aborted",
  skipped: "skipped",
};

// Outcomes that suppress the leaf-warning ⚠ icon. `success_empty`
// and `success_reasoning_off` are yellow (caveat) but the data is
// COMPLETE — empty is a legit `[]` answer; reasoning-off processed
// the full input. `success_sampled` keeps the ⚠ because inputs
// were dropped (the sizing sample cascade kept only the surviving
// 50%). `skipped` is an intentional human action — no warning, no
// provider-shape signal.
const NO_WARN_ICON = new Set([
  "success", "pending", "success_empty", "success_reasoning_off",
  "skipped",
]);

const OUTCOME_ORDER = [
  "success", "pending", "aborted", "skipped", "success_empty",
  "success_sampled", "success_reasoning_off",
  "cap_hit (sizing)",
  "timeout (sizing)", "timeout (load)",
  "parse_error (sizing)", "empty_response (load)",
  "interrupted (sizing)", "interrupted (load)",
  "failed (load)", "failed (sizing)", "failed (other)",
];

// Per-stage summary columns. Success outcomes stay specific; failure
// outcomes roll up into the three retry strategies (load / sizing /
// other). Per-call rows + outcome pills still use the specific failure
// names (cap_hit, timeout (X), parse_error, empty_response,
// interrupted (X), failed (X)) — the per-stage summary trades that
// detail for uniform 3-strategy totals so each strategy column counts
// ALL failures of that kind, not a leftover bucket.
const STAGE_OUTCOME_COLUMNS = [
  { key: "success", label: "ok" },
  { key: "pending", label: "pending" },
  { key: "success_empty", label: "empty (success)" },
  { key: "success_sampled", label: "sampled (success)" },
  { key: "success_reasoning_off", label: "reasoning_off (success)" },
  {
    key: "all_load",
    label: "Load Failures",
    sumOf: [
      "timeout (load)", "interrupted (load)",
      "empty_response (load)", "failed (load)",
    ],
  },
  {
    key: "all_sizing",
    label: "Sizing Failures",
    sumOf: [
      "cap_hit (sizing)", "timeout (sizing)",
      "interrupted (sizing)", "parse_error (sizing)",
      "failed (sizing)",
    ],
  },
  { key: "all_other", label: "Other Failures", sumOf: ["failed (other)"] },
];

function _stageOutcomeCount(col, oc) {
  if (col.sumOf) {
    return col.sumOf.reduce((acc, k) => acc + (oc[k] || 0), 0);
  }
  return oc[col.key] || 0;
}

// Outcome keys can contain spaces / parens (e.g. "cap_hit (sizing)").
// HTML class attributes split on whitespace so we normalize to
// snake-case for the `outcome-...` className. Used for both row
// styling and the outcome pill.
function _outcomeClass(outcome) {
  return (outcome || "")
    .replace(/[\s()/]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .toLowerCase();
}

function fmtMs(ms) {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1000) return `${ms}ms`;
  return formatDuration(ms);
}

function fmtNum(n) {
  if (n === null || n === undefined) return "—";
  return Number(n).toLocaleString();
}

// Module-level cache: pin per-call duration_ms for aborted calls
// (#617). Backend's per-call materializer + overlay_live_token_state
// re-stamps duration_ms = now - started_at_iso for in-flight
// (success.is_null()) calls on every read, and aborted calls have
// success=null by definition (begin landed, end never did). UI-only
// fix: snapshot the first-observed duration_ms for an aborted call
// and use that forever. Once a (runId, callId) pair lands here, the
// displayed value never updates again — Aborted never updates its
// timer. Period.
//
// Keyed by `${runId}\x00${callId}` (null byte separator avoids any
// run_id / call_id formatting collision). Self-bounded by the modal's
// lifetime in practice; small (a few dozen aborted calls per run max).
const FROZEN_ABORTED_CALL_DURATION_MS = new Map();
function _abortedKey(runId, callId) {
  return `${runId}\x00${callId}`;
}
export function freezeAbortedCallDurations(stats, runId) {
  if (!stats || !Array.isArray(stats.calls) || !runId) return stats;
  let mutated = false;
  const calls = stats.calls.map((c) => {
    if (!c || c.outcome !== "aborted" || !c.call_id) return c;
    const k = _abortedKey(runId, c.call_id);
    const cached = FROZEN_ABORTED_CALL_DURATION_MS.get(k);
    if (cached !== undefined) {
      if (c.duration_ms !== cached) {
        mutated = true;
        return { ...c, duration_ms: cached };
      }
      return c;
    }
    if (Number.isFinite(c.duration_ms)) {
      FROZEN_ABORTED_CALL_DURATION_MS.set(k, c.duration_ms);
    }
    return c;
  });
  return mutated ? { ...stats, calls } : stats;
}

// Test seam: reset the frozen-aborted cache so unit tests don't leak.
export function _resetFrozenAbortedCacheForTests() {
  FROZEN_ABORTED_CALL_DURATION_MS.clear();
}

// Render the per-run llm-stats.json into the dense detail surface
// (issue #104 part 2). Modal pattern matches Wizard / Attestation /
// Confirm: backdrop click + Escape close, body click stops propagation.
// Dense rows: one row per call, retry chains collapse to the FIRST
// attempt and expand on click. Per-stage summary shows outcome buckets
// + cached vs non-cached + drop counts.
export function RunDetailsModal({ runId, run, nowMs, onClose }) {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  // Track which retry-chain root call_ids the user has clicked-to-expand.
  const [expandedChains, setExpandedChains] = useState(() => new Set());
  // Per-call cached_now overrides — local-state shim that lets a
  // post-bust click flip a row's button off without re-fetching the
  // whole rollup. Keys are call_id strings; values are bool. Missing
  // keys fall back to the call's `cached_now` from the loaded stats.
  const [cachedNowOverride, setCachedNowOverride] = useState(() => new Map());
  // Issue #333: optimistic local-state for per-call skip ✕ clicks.
  // Set<call_id>. On click we flip the row's pill to "skipped"
  // immediately; the next stats refetch reads the marker-dir-stamped
  // outcome from disk and supersedes this shim.
  const [skipOverride, setSkipOverride] = useState(() => new Set());
  // Set<call_id> for which the per-row details expander is open. Each
  // expander reveals the call's "long tail" fields (budget, request
  // extras, error trace, streaming-token timings when those land) as
  // a colspan'd row beneath its main row. Orthogonal to retry-chain
  // expand: a chain root can be open as a chain AND have its detail
  // panel open at the same time.
  const [expandedDetails, setExpandedDetails] = useState(() => new Set());

  // Expand/collapse in the calls list inserts or removes rows. The
  // modal is the scroll container, and WebKit (the macOS Tauri
  // webview) has no CSS scroll anchoring, so the clicked row would
  // shift in the viewport on expand. Pin it: record the clicked
  // row's offset from the scroller top before the toggle, restore
  // scrollTop after the DOM updates (useLayoutEffect, pre-paint) so
  // the row stays put — first expand and every one after.
  const modalScrollRef = useRef(null);
  const pinRef = useRef(null);
  const capturePin = (callId) => {
    const scroller = modalScrollRef.current;
    const row = scroller?.querySelector(
      `[data-call-id="${CSS.escape(callId)}"]`,
    );
    if (!row) return;
    pinRef.current = {
      callId,
      offset:
        row.getBoundingClientRect().top
        - scroller.getBoundingClientRect().top,
    };
  };
  useLayoutEffect(() => {
    const pin = pinRef.current;
    pinRef.current = null;
    const scroller = modalScrollRef.current;
    if (!pin || !scroller) return;
    const row = scroller.querySelector(
      `[data-call-id="${CSS.escape(pin.callId)}"]`,
    );
    if (!row) return;
    const offset =
      row.getBoundingClientRect().top
      - scroller.getBoundingClientRect().top;
    scroller.scrollTop += offset - pin.offset;
  }, [expandedChains, expandedDetails]);

  // Loader factored out of the mount-effect so cache-changed and
  // pipeline-progress event listeners below can re-fire it without
  // duplicating the setLoading / setError / setStats triplet.
  //
  // `silent=true` skips the loading-spinner flicker: live re-renders
  // on pipeline-progress (issue #165) keep the table on screen while
  // the new payload streams in. The mount-effect call uses silent=false
  // so the first paint shows the spinner.
  const refetchStats = useCallback((silent = false) => {
    let cancelled = false;
    if (!silent) setLoading(true);
    setError(null);
    invoke("read_run_llm_stats", { runId })
      .then((r) => {
        if (cancelled) return;
        const frozen = freezeAbortedCallDurations(r, runId);
        setStats(frozen);
        // Drop any optimistic cached_now overrides — fresh server data
        // is now the source of truth (covers both this-modal busts and
        // out-of-band wipes via Settings).
        setCachedNowOverride(new Map());
        // Prune the optimistic skip override against the fresh canonical
        // payload: a clicked-skip id stays in the override until the
        // materializer has stamped a non-pending outcome for it. Wiping
        // unconditionally on every poll (the older behavior) flickered
        // the row back to "pending" for the gap between the click and
        // the end-event landing on disk — which on a slow pre-first-
        // token call could be many seconds. Dropping the id only on
        // terminal-state arrival keeps the user's intent visible without
        // ever contradicting canonical state once it lands.
        const pendingIds = new Set();
        const collectPending = (calls) => {
          if (!Array.isArray(calls)) return;
          for (const c of calls) {
            if (c && c.outcome === "pending") pendingIds.add(c.call_id);
          }
        };
        collectPending(frozen?.calls);
        setSkipOverride((prev) => {
          if (prev.size === 0) return prev;
          const next = new Set();
          for (const id of prev) {
            if (pendingIds.has(id)) next.add(id);
          }
          return next;
        });
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(String(e?.message || e));
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  useEffect(() => {
    return refetchStats(false);
  }, [refetchStats]);

  // Refetch on cache-changed events — fired by `wipe_llm_cache` and
  // `bust_llm_cache_entry` from Rust. Without this, the modal's
  // "cache" column stays cached after the user wipes the cache via
  // Settings (or busts an entry from a different open modal).
  useEffect(() => {
    let unlisten = null;
    let cancelled = false;
    listen("llm-cache-changed", () => {
      // Ignore events that arrive after unmount / runId change.
      if (cancelled) return;
      refetchStats(false);
    }).then((u) => {
      if (cancelled) {
        u();
      } else {
        unlisten = u;
      }
    });
    return () => {
      cancelled = true;
      if (unlisten) unlisten();
    };
  }, [refetchStats]);

  // Live re-render for in-flight runs (issue #165). When the modal is
  // open AND the run is still alive, refetch via two complementary
  // paths:
  //
  //   1. `pipeline-progress` event listener — fast path. Fires for
  //      every stdout line the runner emits (begin / end / progress_
  //      tick / live_tokens heartbeats). 500ms-coalesced via a single
  //      in-flight timer so a 50-events/sec burst collapses into one
  //      refetch per tick.
  //   2. 1s setInterval — slow-path floor. The Rust-side overlay
  //      recomputes `duration_ms = now - started_at_iso` for every
  //      in-flight rec on each read, so the per-call elapsed ticks
  //      regardless of whether any stdout activity is happening.
  //      Covers the dead zones:
  //        * TTFT phase on streaming providers — the `for chunk in
  //          stream` loop is blocked waiting on the first chunk, so
  //          live_tokens heartbeats can't fire from Python.
  //        * Tinfoil server-side reasoning — provider sits silent
  //          for 10-30s before any visible content streams.
  //        * Pool full + no completions — scheduler isn't dispatching
  //          new calls, in-flight calls are mid-stream, the only
  //          jsonl-mtime / stdout activity is the live_tokens
  //          heartbeats from streaming calls (gated on tokens having
  //          flowed at all, so still nothing during TTFT).
  //      `read_run_llm_stats` is mtime-cached + in-memory-overlay,
  //      so the per-tick cost is sub-ms; at 1Hz this is well under
  //      what the modal would do under heavy event load anyway.
  const isLive = run?.status === "running" || run?.status === "paused";
  const inflightRefetchTimerRef = useRef(null);
  useEffect(() => {
    if (!isLive) return;
    let unlisten = null;
    let cancelled = false;
    const scheduleRefetch = () => {
      if (inflightRefetchTimerRef.current) return;
      inflightRefetchTimerRef.current = setTimeout(() => {
        inflightRefetchTimerRef.current = null;
        if (cancelled) return;
        refetchStats(true);
      }, 500);
    };
    listen("pipeline-progress", () => {
      if (cancelled) return;
      scheduleRefetch();
    }).then((u) => {
      if (cancelled) {
        u();
      } else {
        unlisten = u;
      }
    });
    const tickId = setInterval(() => {
      if (cancelled) return;
      refetchStats(true);
    }, 1000);
    return () => {
      cancelled = true;
      clearInterval(tickId);
      if (inflightRefetchTimerRef.current) {
        clearTimeout(inflightRefetchTimerRef.current);
        inflightRefetchTimerRef.current = null;
      }
      if (unlisten) unlisten();
    };
  }, [isLive, refetchStats]);

  // Issue #237 follow-up: fire one final refetch on the
  // running→terminal transition. Without this, the LAST stage's
  // end event can race with the run-terminal status flip:
  //   1. actions end event written (mtime invalidates materializer cache)
  //   2. pipeline-progress fires → modal schedules a 500ms refetch
  //   3. cycle_end written → status flips → isLive becomes false
  //   4. live-subscription cleanup CANCELS the scheduled refetch
  //   5. modal stays stale, shows actions as pending until manual reopen
  // The down-transition refetch closes the gap by reading the
  // post-cycle_end materializer state once. Tracks previous isLive
  // via a ref so the effect ONLY fires on the down-transition (not
  // on first mount when isLive is already false).
  const prevIsLiveRef = useRef(isLive);
  useEffect(() => {
    if (prevIsLiveRef.current === true && isLive === false) {
      refetchStats(true);
    }
    prevIsLiveRef.current = isLive;
  }, [isLive, refetchStats]);

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

  // While the run is alive, prefer the live-ticking elapsed (issue
  // #159) over the stale duration_ms recorded at the last list_runs
  // refresh — it makes the modal header confirm the run is moving.
  const elapsedMs = liveElapsedMs(run, nowMs);
  const elapsedHeader =
    run && run.status === "running"
      ? fmtElapsed(elapsedMs)
      : formatDuration(run?.duration_ms);
  // Surface a [RUNNING] tag prominently when the run is in flight — the
  // modal opens for in-flight runs since #165, and the legacy header
  // line read like a completed-run summary which was confusing.
  const statusTag = run?.status === "running"
    ? "[RUNNING] · Elapsed: "
    : "";
  const headerLabel = run
    ? `${run.short_id ? `#${run.short_id}` : run.run_id} — ${statusTag}${run.status === "running" ? "" : (run.status || "?") + " · "}${elapsedHeader || "—"}`
    : runId;

  // Issue #105 v3 follow-up #2: render LEAVES, not chain roots.
  // A leaf is a call whose call_id is NOT referenced as anyone's
  // retry_of_call_id — the terminal node of a chain. Halving makes
  // chains a TREE (one root, multiple leaves via parallel halve
  // children); pre-fix the modal flattened the tree to a single
  // ordered list and showed only the last attempt by attempt-asc
  // order, which silently dropped sibling halves. Now each leaf
  // gets its own row; clicking expands to the oldest-first ancestor
  // chain ABOVE the leaf.
  const calls = (stats && Array.isArray(stats.calls)) ? stats.calls : [];
  const callById = useMemo(() => {
    const m = new Map();
    for (const c of calls) m.set(c.call_id, c);
    return m;
  }, [calls]);
  // Set of call_ids that are anyone's parent. Anything not in this
  // set is a leaf.
  const childOf = useMemo(() => {
    const s = new Set();
    for (const c of calls) {
      if (c.retry_of_call_id) s.add(c.retry_of_call_id);
    }
    return s;
  }, [calls]);
  // Ancestor chains keyed by LEAF call_id. NEWEST-first (leaf →
  // ... → root inclusive) so the expanded render shows the most
  // recent attempt at the top — when debugging the user wants to
  // see what the model did LAST, not scan past N earlier retries
  // to find it. Pre-fix used `unshift` (oldest-first); the user
  // explicitly asked for reverse chronological.
  const ancestorsByLeaf = useMemo(() => {
    const out = new Map();
    for (const c of calls) {
      if (childOf.has(c.call_id)) continue;  // not a leaf
      const chain = [];
      const seen = new Set();
      let cur = c;
      while (cur && !seen.has(cur.call_id)) {
        seen.add(cur.call_id);
        chain.push(cur);  // newest-first (leaf at index 0)
        if (!cur.retry_of_call_id) break;
        cur = callById.get(cur.retry_of_call_id);
      }
      out.set(c.call_id, chain);
    }
    return out;
  }, [calls, childOf, callById]);
  // Order leaves by their root's first-seen position so two leaves
  // sharing a root render adjacently (e.g. extract halve children).
  const leafOrder = useMemo(() => {
    const rootOrderIndex = new Map();
    for (let i = 0; i < calls.length; i++) {
      const c = calls[i];
      const chain = ancestorsByLeaf.get(c.call_id);
      if (!chain || chain.length === 0) continue;
      // Chains are newest-first (leaf at index 0); the ROOT is the
      // LAST element. Key sibling order on the root so two leaves
      // sharing a root (e.g. extract halve children, retry stacks)
      // render adjacently.
      const rootId = chain[chain.length - 1].call_id;
      if (!rootOrderIndex.has(rootId)) {
        rootOrderIndex.set(rootId, i);
      }
    }
    const leaves = [];
    for (const c of calls) {
      if (childOf.has(c.call_id)) continue;
      leaves.push(c);
    }
    leaves.sort((a, b) => {
      const aChain = ancestorsByLeaf.get(a.call_id);
      const bChain = ancestorsByLeaf.get(b.call_id);
      const aRoot = aChain?.[aChain.length - 1]?.call_id;
      const bRoot = bChain?.[bChain.length - 1]?.call_id;
      const ai = rootOrderIndex.get(aRoot) ?? 0;
      const bi = rootOrderIndex.get(bRoot) ?? 0;
      if (ai !== bi) return ai - bi;
      // Same root, multiple leaves (halve siblings) — use call_id
      // for stable order.
      return String(a.call_id).localeCompare(String(b.call_id));
    });
    return leaves.map(c => c.call_id);
  }, [calls, childOf, ancestorsByLeaf]);

  const perStage = (stats && stats.per_stage) || {};
  // Sort by canonical pipeline order, NOT alphabetical. Stages not in
  // RUN_DETAILS_STAGE_ORDER (defensive: future stage someone added
  // without updating the list) fall to the end in their natural order.
  const perStageEntries = Object.entries(perStage).sort(([a], [b]) => {
    const ai = RUN_DETAILS_STAGE_ORDER.indexOf(a);
    const bi = RUN_DETAILS_STAGE_ORDER.indexOf(b);
    if (ai === -1 && bi === -1) return 0;
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  });

  // Cache summary at top of modal. Two branches:
  //   Bypassed: run_config.llm_cache_enabled was false at run start
  //     (Settings checkbox off OR BASEVAULT_LLM_CACHE_BYPASS env var).
  //     The hit-rate is meaningless — every call was a forced miss.
  //   Hit rate: sum calls_cached / calls_total across stages, render as
  //     "X of Y calls served from cache (Z% hit rate)".
  // run_config sits under the run record (passed in from App.jsx),
  // not under the stats payload — the runner stamps it to config.json
  // at start. Defensive against null for older runs that pre-date the
  // run_config snapshot.
  const cacheSummary = useMemo(() => {
    const bypassed = run?.run_config?.llm_cache_enabled === false;
    let total = 0;
    let cached = 0;
    for (const [, b] of perStageEntries) {
      total += (b.calls_total || 0);
      cached += (b.calls_cached || 0);
    }
    const pct = total > 0 ? Math.round((cached / total) * 100) : 0;
    return { bypassed, total, cached, pct };
  }, [run, perStageEntries]);

  const toggleDetails = (callId) => {
    capturePin(callId);
    setExpandedDetails((prev) => {
      const next = new Set(prev);
      if (next.has(callId)) next.delete(callId);
      else next.add(callId);
      return next;
    });
  };

  const toggleChain = (leafId) => {
    capturePin(leafId);
    setExpandedChains((prev) => {
      const next = new Set(prev);
      // Keyed on the clicked leaf id. Each leaf expands/collapses
      // independently — clicking a call shows the attempts that preceded
      // it; halve siblings are separate leaf rows and toggle separately.
      if (next.has(leafId)) next.delete(leafId);
      else next.add(leafId);
      return next;
    });
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        ref={modalScrollRef}
        className="modal run-details-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2>Run details — {headerLabel}</h2>
          <button
            type="button"
            className="modal-close"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        {loading && <p className="field-hint">Loading…</p>}
        {error && (
          <p className="field-hint">
            Failed to load stats: {error}
          </p>
        )}
        {!loading && !error && !stats && (
          <p className="field-hint">
            No LLM calls have landed yet for this run — per-call detail
            will appear as the pipeline progresses.
          </p>
        )}
        {/* Run summary header (#165): always-visible at the top of
            the modal, above the cache hit-rate line. Renders even
            before stats finish loading so the routing snapshot is
            visible immediately on open. Folded in from the in-tree
            RunDetails panel that used to live above the file tree. */}
        {run && (
          <section className="run-details-section run-summary-header">
            <RunSummarySection run={run} />
          </section>
        )}
        {stats && (
          <>
            <section className="run-details-section run-details-cache-summary">
              {cacheSummary.bypassed ? (
                <p className="cache-summary-line bypassed">
                  Cache: bypassed (BASEVAULT_LLM_CACHE_BYPASS set or Settings cache toggle off)
                </p>
              ) : (
                <p className="cache-summary-line">
                  Cache:{" "}
                  <strong>{cacheSummary.cached.toLocaleString()}</strong> of{" "}
                  <strong>{cacheSummary.total.toLocaleString()}</strong> calls served from cache
                  {cacheSummary.total > 0 ? (
                    <> (<strong>{cacheSummary.pct}%</strong> hit rate)</>
                  ) : null}
                </p>
              )}
            </section>
            <section className="run-details-section">
              <h3>Per-stage summary</h3>
              {perStageEntries.length === 0 ? (
                <p className="field-hint">No per-stage data.</p>
              ) : (
                // Outcome columns: 5 success buckets + 4 strategy
                // failure totals. Wrap in a horizontal-scroll container
                // so the leftmost columns (stage / calls / cached) stay
                // pinned-feeling on narrower windows.
                <div className="run-details-stages-scroll">
                <table className="run-details-stages">
                  <thead>
                    <tr>
                      <th>stage</th>
                      <th className="numeric">calls</th>
                      <th className="numeric">cached</th>
                      {STAGE_OUTCOME_COLUMNS.map((col) => (
                        <th
                          key={col.key}
                          className="numeric"
                          title={col.sumOf ? col.sumOf.join(" + ") : col.key}
                        >
                          {col.label}
                        </th>
                      ))}
                      <th className="numeric">p50 dur</th>
                      <th className="numeric" title="p50 time-to-first-token across calls in this stage">p50 ttft</th>
                    </tr>
                  </thead>
                  <tbody>
                    {perStageEntries.map(([stageName, b]) => {
                      const oc = b.outcomes || {};
                      const dur = (b.duration_ms || {});
                      const ttft = (b.ttft_ms || {});
                      return (
                        <tr key={stageName}>
                          <td>{stageName}</td>
                          <td className="numeric">{fmtNum(b.calls_total)}</td>
                          <td className="numeric">{fmtNum(b.calls_cached || 0)}</td>
                          {STAGE_OUTCOME_COLUMNS.map((col) => (
                            <td key={col.key} className="numeric">
                              {fmtNum(_stageOutcomeCount(col, oc))}
                            </td>
                          ))}
                          <td className="numeric">{fmtMs(dur.median)}</td>
                          <td className="numeric">{fmtMs(ttft.median)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                </div>
              )}
            </section>
            <section className="run-details-section">
              <h3>
                Per-call ({calls.length} call{calls.length === 1 ? "" : "s"}
                {leafOrder.length !== calls.length
                  ? `, ${leafOrder.length} chain${leafOrder.length === 1 ? "" : "s"}`
                  : ""})
              </h3>
              {calls.length === 0 ? (
                <p className="field-hint">No calls recorded yet.</p>
              ) : (
                <div className="run-details-calls">
                  <table>
                    <thead>
                      <tr>
                        <th>call</th>
                        <th>stage</th>
                        <th>name</th>
                        <th>model</th>
                        <th>outcome</th>
                        <th className="numeric">prompt</th>
                        <th className="numeric" title="completion content tokens (= completion − reasoning)">payload</th>
                        <th className="numeric" title="reasoning tokens (api / streamed / estimated). payload + reasoning = completion by construction">reasoning</th>
                        <th className="numeric" title="time between request-start and first-token-received (ms / s)">wait time</th>
                        <th className="numeric">dur</th>
                        <th>cache hit</th>
                        <th>cache</th>
                        <th>more</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(() => {
                        // Embeddings collapse under ONE parent row, reusing
                        // the retry collapse/expand machinery (expandedChains
                        // + toggleChain + ▸/▾ + the row CSS). Retries
                        // collapse-to-leaf / expand-up; embeddings need the
                        // inverse — a parent SUMMARY that hides the per-chunk
                        // calls until expanded. Default collapsed; the parent
                        // shows N passed / M failed + a prompt-token sum. The
                        // header row is emitted once, just before the first
                        // embeddings leaf; the individual embeddings rows are
                        // gated behind its expand state. Non-embeddings leaves
                        // render exactly as before (empty `pre`).
                        const EMB_KEY = "__embeddings_parent__";
                        const embExpanded = expandedChains.has(EMB_KEY);
                        const embCalls = leafOrder
                          .map((id) => callById.get(id))
                          .filter((c) => c && c.stage === "embeddings");
                        const embPassed = embCalls
                          .filter((c) => c.success === true).length;
                        const embFailed = embCalls
                          .filter((c) => c.success === false).length;
                        const embTokenSum = embCalls
                          .reduce((a, c) => a + (c.prompt_tokens || 0), 0);
                        let embHeaderDone = false;
                        const embHeaderRow = () => (
                          <tr
                            key={EMB_KEY}
                            data-call-id={EMB_KEY}
                            className="run-details-call-row chain-leaf embeddings-parent-row"
                            onClick={() => toggleChain(EMB_KEY)}
                            title={embExpanded
                              ? "Click to collapse embedding calls"
                              : `Click to expand ${embCalls.length} embedding calls`}
                          >
                            <td>{embExpanded ? "▾ " : "▸ "}
                              {`embeddings (${embCalls.length})`}</td>
                            <td>embeddings</td>
                            <td>—</td>
                            <td>{embCalls[0]
                              ? (modelDisplayName(embCalls[0].model, embCalls[0].mode)
                                 || embCalls[0].model || "—")
                              : "—"}</td>
                            <td>
                              <span className="outcome-cell-content">
                                <span className="embeddings-parent-stats">
                                  {embPassed} passed / {embFailed} failed
                                  {" · "}{fmtNum(embTokenSum)} tok
                                </span>
                              </span>
                            </td>
                            <td className="numeric">—</td>
                            <td className="numeric">—</td>
                            <td className="numeric">—</td>
                            <td className="numeric">—</td>
                            <td className="numeric">—</td>
                            <td>—</td>
                            <td>—</td>
                            <td>—</td>
                          </tr>
                        );
                        // A call is drawn at most ONCE across the whole list.
                        // Leaves of a deep halve tree share the trunk
                        // (root → … → first split); without this, every
                        // EXPANDED leaf redraws that shared spine and the same
                        // 0001/0002/0003 rows pile up. The first expanded leaf
                        // to reach an ancestor renders it; later leaves skip the
                        // already-drawn part (their own unique tail still shows).
                        const renderedIds = new Set();
                        return leafOrder.flatMap((leafId) => {
                        const isEmb =
                          (callById.get(leafId)?.stage) === "embeddings";
                        const pre = [];
                        if (isEmb && !embHeaderDone) {
                          embHeaderDone = true;
                          pre.push(embHeaderRow());
                        }
                        // Collapsed: emit the header once, hide individuals.
                        if (isEmb && !embExpanded) return pre;
                        const chain = ancestorsByLeaf.get(leafId) || [];
                        // chain is NEWEST-first: [leaf, prev, ...,
                        // root]. Leaf is at index 0; root at end.
                        const leaf = chain[0] || callById.get(leafId);
                        if (!leaf) return pre;
                        const expandable = chain.length > 1;
                        // Keyed on the LEAF id alone, so only the chain whose
                        // leaf was clicked expands. Each leaf is independent —
                        // halve siblings (separate leaf rows) never co-expand,
                        // so a shared ancestor is never rendered twice. Empty
                        // set → collapsed by default.
                        const expanded = expandedChains.has(leafId);
                        // Default render: leaf only. Expanded: leaf
                        // STAYS at the top, then prior attempts in
                        // newest-first order — try N-1, N-2, ..., 1.
                        const ancestorsNewestFirst = chain.slice(1);
                        const visibleAll = expanded
                          ? [leaf, ...ancestorsNewestFirst]
                          : [leaf];
                        // Drop any row already drawn under an earlier leaf (the
                        // shared trunk of a halve tree) so it isn't repeated.
                        const visible = visibleAll.filter((c) => {
                          if (renderedIds.has(c.call_id)) return false;
                          renderedIds.add(c.call_id);
                          return true;
                        });
                        if (visible.length === 0) return pre;
                        // (try N) is the call's CHRONOLOGICAL
                        // position: 1 = original (root), N = leaf.
                        // chain is newest-first, so the leaf at
                        // index 0 gets try=chain.length, root at
                        // index chain.length-1 gets try=1.
                        const tryByCallId = new Map(
                          chain.map((c, i) => [c.call_id, chain.length - i])
                        );
                        const __leafRows = visible.flatMap((c, idx) => {
                          // Issue #333: optimistic local-state flip
                          // overrides the materializer's outcome until
                          // the next refetch reads the marker-dir
                          // result. Defended against stale UI by the
                          // refetch dropping `skipOverride` to an
                          // empty Set.
                          const outcomeKey = skipOverride.has(c.call_id)
                            ? "skipped"
                            : c.outcome;
                          const isLeafRow = idx === 0;
                          const detailsOpen = expandedDetails.has(c.call_id);
                          // Two independent cache facts on each row,
                          // surfaced as separate columns:
                          //   c.cached     — RUN-TIME hit. Historical
                          //                  fact about how this call
                          //                  was served; never changes
                          //                  after the run. Renders as
                          //                  the "cache hit" yes/no.
                          //   c.cached_now — IS the response still in
                          //                  the cache right now? Live
                          //                  filesystem check from
                          //                  read_run_llm_stats.
                          //                  Renders as "cache": copy
                          //                  button (clipboards the
                          //                  payload JSON) + bust ✕.
                          // The copy + bust buttons render only when
                          // cache=yes — busting a key that isn't
                          // there is allowed at the API layer (it's
                          // idempotent), but hiding the affordance is
                          // honest about what state the click would
                          // actually change. Un-cached rows show "–".
                          const cachedNow = cachedNowOverride.has(c.call_id)
                            ? cachedNowOverride.get(c.call_id)
                            : !!c.cached_now;
                          const canBust = !!c.cache_key;
                          // Visual dim when the user busted this row's
                          // cache in the current modal session — the
                          // override map says "no" while the
                          // historical `cached` says "yes". Survives
                          // the trash-button vanishing so feedback
                          // doesn't evaporate.
                          const justBusted =
                            cachedNowOverride.get(c.call_id) === false
                            && c.cached === true;
                          // Click target = the LEAF row (the bottom
                          // row when expanded, the only row when
                          // collapsed). Ancestor rows are read-only.
                          const isClickable = expandable && isLeafRow;
                          return [
                            <tr
                              key={c.call_id}
                              data-call-id={c.call_id}
                              className={
                                "run-details-call-row "
                                + `outcome-${_outcomeClass(outcomeKey)} `
                                + (expandable && isLeafRow ? "chain-leaf " : "")
                                + (expandable && !isLeafRow ? "chain-ancestor " : "")
                                + (justBusted ? "cache-busted " : "")
                              }
                              onClick={
                                isClickable
                                  ? () => toggleChain(leafId)
                                  : undefined
                              }
                              title={
                                isClickable
                                  ? (expanded
                                      ? "Click to collapse retry chain"
                                      : `Click to expand ${chain.length} attempts`)
                                  : undefined
                              }
                            >
                              <td>
                                {/* ⚠ marker on flagged LEAVES only.
                                    Decision tree per orchestrator:
                                    leaf outcome ≠ success AND ≠
                                    pending → ⚠. Else nothing.
                                    Ancestor rows (visible only when
                                    expanded) keep their own outcome
                                    pill but skip the icon — warning
                                    belongs to the chain's terminal
                                    state, not each intermediate
                                    failure. Renders BEFORE the ▸/▾
                                    expand arrow so the row's
                                    leftmost glyph is the user's
                                    attention anchor. Yellow ⚠ kept
                                    for consistency with the rest of
                                    the warning surface in the app;
                                    visibility comes from the CSS
                                    (font-size 1.35em + bold). */}
                                {isLeafRow
                                    && !NO_WARN_ICON.has(outcomeKey) ? (
                                  <span
                                    className="run-details-warn-icon"
                                    title={`leaf outcome: ${OUTCOME_LABELS[outcomeKey] || outcomeKey}`}
                                  >
                                    ⚠
                                  </span>
                                ) : null}
                                {expandable && isLeafRow
                                  ? (expanded ? "▾ " : "▸ ")
                                  : ""}
                                {c.call_id}
                                {/* (try N) suffix: chain position
                                    (1-based, oldest-first). The leaf
                                    of an N-attempt chain shows
                                    "(try N)"; the original parent
                                    shows "(try 1)". User: "Drop the
                                    3x ahead of it … show (try X) for
                                    each item that was retried." */}
                                {expandable
                                  ? ` (try ${tryByCallId.get(c.call_id) ?? "?"})`
                                  : ""}
                              </td>
                              <td>{c.stage || "—"}</td>
                              <td>{c.category || "—"}</td>
                              <td>
                                {modelDisplayName(c.model, c.mode) || c.model || "—"}
                                {c.request_extras?.reasoning === true ? (
                                  <span
                                    className="run-reasoning-badge per-call-reasoning-badge"
                                    title="reasoning enabled"
                                  >
                                    🧠
                                  </span>
                                ) : null}
                              </td>
                              <td>
                                {/* Wrapper keeps pill + skip ✕ as a single
                                    non-wrapping inline group. Without it,
                                    the inter-element whitespace is a wrap
                                    opportunity and the ✕ falls to a second
                                    line at narrow column widths, lifting
                                    pending-row height and orphaning the ✕
                                    next to the wrong row. */}
                                <span className="outcome-cell-content">
                                  <span className={`outcome-pill outcome-${_outcomeClass(outcomeKey)}`}>
                                    {OUTCOME_LABELS[outcomeKey] || outcomeKey}
                                  </span>
                                  {outcomeKey === "pending" ? (
                                    <button
                                      type="button"
                                      className="run-details-skip-btn"
                                      title={
                                        "Skip this call. Terminal — the call "
                                        + "is given up on (no retry). The "
                                        + "stage proceeds with whatever the "
                                        + "missing call would have produced."
                                      }
                                      aria-label="Skip this call"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        setSkipOverride((prev) => {
                                          const next = new Set(prev);
                                          next.add(c.call_id);
                                          return next;
                                        });
                                        invoke("skip_call", {
                                          runId,
                                          callId: c.call_id,
                                        }).catch((err) => {
                                          console.error(
                                            "skip_call failed:", err,
                                          );
                                          setSkipOverride((prev) => {
                                            const next = new Set(prev);
                                            next.delete(c.call_id);
                                            return next;
                                          });
                                        });
                                      }}
                                    >
                                      ✕
                                    </button>
                                  ) : null}
                                </span>
                              </td>
                              <td className="numeric">{fmtNum(c.prompt_tokens)}</td>
                              <td className="numeric">{fmtNum(c.content_tokens)}</td>
                              <td className="numeric">{fmtNum(c.reasoning_tokens)}</td>
                              {/* Wait = request-start → first-token. '—'
                                  for rows with no clean wire-call wait
                                  signal: cache hits never touched the wire;
                                  aborted/skipped have no user-meaningful
                                  first-token reading (an aborted row may
                                  carry a stamped ttft_ms anyway, but the
                                  wait is undefined as "how long before
                                  anything happened").

                                  A pending (in-flight) call is NOT blank:
                                  the wait is shown live. Once the first
                                  token lands, ttft_ms is on the record
                                  (via the mid-flight stream_progress
                                  event) — show the true, now-frozen wait.
                                  Before any token, the accumulating wait
                                  IS the live elapsed-since-start (the same
                                  value the `dur` column ticks, overlaid
                                  every refresh) — show it muted with a
                                  trailing ellipsis so it reads as a
                                  still-running count, not a final number. */}
                              <td className="numeric">{
                                c.cached
                                || outcomeKey === "aborted"
                                || outcomeKey === "skipped"
                                  ? "—"
                                  : outcomeKey === "pending"
                                    ? (c.ttft_ms !== null && c.ttft_ms !== undefined
                                        ? fmtMs(c.ttft_ms)
                                        : (c.duration_ms !== null && c.duration_ms !== undefined
                                            ? <span
                                                className="muted"
                                                title="Call is waiting — accumulating wait so far (request-start → now). Settles to the true wait once the first token arrives."
                                              >{fmtMs(c.duration_ms)}…</span>
                                            : "—"))
                                    : fmtMs(c.ttft_ms)
                              }</td>
                              <td className="numeric">{fmtMs(c.duration_ms)}</td>
                              <td
                                className={`cache-hit-cell ${c.cached ? "yes" : "no"}`}
                                title={
                                  c.cached
                                    ? "This call was served from the prompt-hash cache during the run — no provider call was made."
                                    : "This call was a cache miss; the provider was hit."
                                }
                              >
                                {c.cached ? "yes" : "no"}
                              </td>
                              <td
                                className={`in-cache-cell ${cachedNow ? "yes" : "no"}`}
                                title={
                                  cachedNow
                                    ? "The response is currently stored in the prompt-hash cache. The next matching call will be served from disk."
                                    : "Nothing is stored under this call's cache key right now (never stored, or already busted/wiped)."
                                }
                              >
                                {/* Layout container is an inner span so the
                                    `<td>` itself stays a normal table-cell
                                    and inherits the row's vertical-align.
                                    Putting display:grid on the `<td>`
                                    directly took it out of the row's
                                    centering, so this cell sat at the top
                                    of any tall row. */}
                                <span className="in-cache-cell-content">
                                  {cachedNow && canBust ? (
                                    <>
                                      {/* Span wrapper catches the click
                                          before it bubbles to the row's
                                          chain-toggle handler. CopyButton
                                          itself doesn't take an event arg
                                          so we can't stopPropagation from
                                          inside its onClick. */}
                                      <span onClick={(e) => e.stopPropagation()}>
                                        <CopyButton
                                          inline
                                          borderless
                                          onClick={() => copyCacheEntryToClipboard(
                                            c.stage || "_unknown",
                                            c.cache_key,
                                          )}
                                          label="Copy cache payload"
                                          testId={`cache-copy-${c.call_id}`}
                                        />
                                      </span>
                                      <button
                                        type="button"
                                        className="run-details-bust-btn"
                                        title={
                                          "Remove this call's response from the "
                                          + "prompt-hash cache. The next run that "
                                          + "would have hit this entry will fall "
                                          + "through to the provider."
                                        }
                                        aria-label="Remove from cache"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          // Optimistic flip: this row's
                                          // "cache" cell goes yes → no
                                          // immediately. The rust-side
                                          // event will refetch and confirm
                                          // (or correct) shortly.
                                          setCachedNowOverride((prev) => {
                                            const next = new Map(prev);
                                            next.set(c.call_id, false);
                                            return next;
                                          });
                                          invoke("bust_llm_cache_entry", {
                                            stage: c.stage || "_unknown",
                                            cacheKey: c.cache_key,
                                          }).catch((err) => {
                                            console.error(
                                              "bust_llm_cache_entry failed:",
                                              err,
                                            );
                                            // On error, restore.
                                            setCachedNowOverride((prev) => {
                                              const next = new Map(prev);
                                              next.set(c.call_id, !!c.cached_now);
                                              return next;
                                            });
                                          });
                                        }}
                                      >
                                        ✕
                                      </button>
                                    </>
                                  ) : (
                                    // Empty cache: render a dash placeholder
                                    // instead of "no" — less visual noise on
                                    // the table when most rows are misses.
                                    <span className="in-cache-label">–</span>
                                  )}
                                </span>
                              </td>
                              <td className="run-details-detail-toggle-cell">
                                <button
                                  type="button"
                                  className="run-details-detail-toggle"
                                  aria-expanded={detailsOpen}
                                  aria-label={detailsOpen ? "Collapse details" : "Expand details"}
                                  title={detailsOpen ? "Hide call details" : "Show call details"}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleDetails(c.call_id);
                                  }}
                                >
                                  …
                                </button>
                              </td>
                            </tr>,
                            detailsOpen ? (
                              <tr
                                key={`${c.call_id}-details`}
                                className="run-details-call-detail-row"
                              >
                                <td colSpan={13}>
                                  <CallDetailPanel
                                    call={c}
                                    chainTry={
                                      expandable
                                        ? (tryByCallId.get(c.call_id)
                                           ?? null)
                                        : null
                                    }
                                    chainLength={
                                      expandable ? chain.length : null
                                    }
                                  />
                                </td>
                              </tr>
                            ) : null,
                          ].filter(Boolean);
                        });
                        return [...pre, ...__leafRows];
                        });
                      })()}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
            <FullPromptLogs calls={calls} />
          </>
        )}
        <div className="modal-actions">
          <button type="button" className="btn-primary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

// Per-row detail panel — colspan'd row beneath each per-call row in
// the dense table. Surfaces the long-tail fields that don't deserve
// their own column (most are present-but-quiet for normal runs):
//
//   request_extras: temperature, reasoning, max_tokens_reserved
//   budget:         stage_cap, max_output, scaffolding
//   timing-tokens:  ttft_ms, ttfr_ms, last_token_ms (streaming worker)
//   token tail:     reasoning_tokens, content_tokens (streaming worker)
//   provider tail:  finish_reason
//   error trace:    error.message + error.traceback (timeout / failure rows)
//   chain raw:      attempt + retry_of_call_id (vs the chain-collapse view)
//   template_hash:  moved from its prior column — same value across
//                   rows for the same template, more useful in detail
//
// Renders as a 2-column key/value grid. Fields with no value (most are
// missing on default runs) collapse to a "—" stub so the panel shape
// stays readable.
function CallDetailPanel({ call, chainTry, chainLength }) {
  const c = call;
  const ex = c.request_extras || {};
  const b = c.budget || {};
  const err = c.error || null;
  // "attempt" surfaced in the panel = chain position when this row
  // is part of a chain (matches the "(try N)" label in the row's
  // call_id column). Pre-fix it showed `c.attempt` which is the
  // wrapper-INTERNAL counter — every fresh wrapper invocation
  // (e.g. a stage-helper sample-N call) starts at 1, so a sample
  // attempt mid-chain showed "attempt 1" while the chain UI
  // labeled it "try 8" (run pd5w call 0065).
  const attemptLabel = chainTry !== null && chainTry !== undefined
    ? (chainLength != null ? `${chainTry} of ${chainLength}` : String(chainTry))
    : (c.attempt != null ? String(c.attempt) : null);
  const rows = [
    ["started_at", c.started_at_iso || null],
    // Backend provider this call ran against (tinfoil / ollama / mlx).
    // Stamped at begin-time so cache-hit and failure paths carry it
    // too; null on legacy records that pre-date the begin-time stamp.
    // Finer-grained than the user-facing `mode` (which folds MLX +
    // Ollama into "local").
    ["provider", c.provider || null],
    ["retry_delay",
      c.retry_delay_ms != null ? `${c.retry_delay_ms}ms` : null],
    ["temperature", ex.temperature],
    ["reasoning", typeof ex.reasoning === "boolean" ? String(ex.reasoning) : null],
    ["max_tokens_reserved", c.max_tokens_reserved],
    ["stage_cap", b.stage_cap],
    ["max_output", b.max_output],
    ["scaffolding", b.scaffolding],
    ["ttft_ms", c.ttft_ms],
    ["ttfr_ms", c.ttfr_ms],
    ["last_token_ms", c.last_token_ms],
    ["finish_reason", c.finish_reason],
    ["reasoning_tokens", c.reasoning_tokens],
    ["reasoning_tokens_source", c.reasoning_tokens_source],
    ["content_tokens", c.content_tokens],
    ["attempt", attemptLabel],
    ["retry_of_call_id", c.retry_of_call_id],
    ["template_hash", c.template_hash],
  ];
  return (
    <div className="run-details-detail-panel">
      <dl className="run-details-detail-grid">
        {rows.map(([k, v]) => (
          <div key={k} className="run-details-detail-pair">
            <dt>{k}</dt>
            <dd>
              {v === null || v === undefined || v === ""
                ? <span className="run-details-detail-empty">—</span>
                : <code>{String(v)}</code>}
            </dd>
          </div>
        ))}
      </dl>
      {err ? (
        <div className="run-details-detail-error">
          <h4>error</h4>
          {err.class ? <div><code>{err.class}</code></div> : null}
          {err.message ? <pre className="run-details-detail-error-msg">{err.message}</pre> : null}
          <TracebackDetails
            trace={err.traceback}
            preClassName="run-details-detail-error-tb"
          />
        </div>
      ) : (c.success === false && c.llm_status) ? (
        // A no-exception kernel failure: the `from_status` path finalizes
        // success=false WITHOUT raising, so `error` is null and there's no
        // traceback (injected or real LOAD / OTHER / TIMEOUT_WITH_TOKENS /
        // PARSE_ERROR). The kernel still categorized the outcome in
        // `llm_status`, so surface it in the same red block rather than
        // rendering an empty panel — the row's outcome column carries the
        // load/sizing label, this names the kernel status behind it (#963).
        <div className="run-details-detail-error">
          <h4>error</h4>
          <div><code>{c.llm_status}</code></div>
          <pre className="run-details-detail-error-msg">
            kernel-classified failure — no exception raised
          </pre>
        </div>
      ) : null}
    </div>
  );
}

// Section under the per-call table that surfaces the full prompt +
// response for any call where the runner stamped them (Settings →
// Development → Full prompt logging toggle for the stage was ON when
// the run fired). Renders as a collapsed list — each entry expands on
// click to show the messages list and the response text. No-op when
// no calls in the loaded run carry the fields. Disk impact is real;
// the modal lazy-renders content only for calls the user clicks open.
// Multimodal-aware renderer for one captured message's `content`.
// Vision-stage calls stamp content as an OpenAI-multimodal array
// (text parts + image_url parts with base64 data URLs); every other
// stage stamps a plain string. Rendering an array of part-objects
// inside a single <pre> crashes React with "objects are not valid as
// a child."
function FullPromptMessageContent({ content }) {
  if (typeof content === "string") {
    return <pre className="full-prompt-content">{content}</pre>;
  }
  if (Array.isArray(content)) {
    return (
      <>
        {content.map((part, j) => {
          if (part?.type === "text") {
            return (
              <pre key={j} className="full-prompt-content">
                {part.text || ""}
              </pre>
            );
          }
          if (part?.type === "image_url") {
            const url = part?.image_url?.url || "";
            return url ? (
              <img key={j} className="full-prompt-image" src={url} alt="" />
            ) : (
              <pre key={j} className="full-prompt-content">[empty image_url]</pre>
            );
          }
          return (
            <pre key={j} className="full-prompt-content">
              {JSON.stringify(part, null, 2)}
            </pre>
          );
        })}
      </>
    );
  }
  if (content == null) {
    return <pre className="full-prompt-content"></pre>;
  }
  return (
    <pre className="full-prompt-content">
      {JSON.stringify(content, null, 2)}
    </pre>
  );
}


// Copy a cache entry to the system clipboard. The naive form —
// `invoke(...).then(copyToClipboard)` — quietly fails in WKWebView
// (Tauri on macOS): awaiting the invoke promise drops out of the
// click handler's synchronous user-activation window, and
// `navigator.clipboard.writeText` then rejects with a focus/gesture
// error. The documented WebKit escape hatch is
// `navigator.clipboard.write([new ClipboardItem({ "text/plain": <promise> })])`:
// the ClipboardItem accepts a Promise<Blob>, the browser holds the
// activation context across its resolution, and the write lands.
// ClipboardItem is unavailable in jsdom (test env), so the fallback
// path runs there — vitest mocks `navigator.clipboard.writeText` and
// asserts directly against it.
function copyCacheEntryToClipboard(stage, cacheKey) {
  if (typeof ClipboardItem !== "undefined" && navigator.clipboard?.write) {
    const blobPromise = invoke("read_llm_cache_entry", { stage, cacheKey })
      .then((text) => new Blob([text ?? ""], { type: "text/plain" }));
    navigator.clipboard
      .write([new ClipboardItem({ "text/plain": blobPromise })])
      .catch((err) => console.error("clipboard.write cache entry:", err));
    return;
  }
  invoke("read_llm_cache_entry", { stage, cacheKey })
    .then((text) => navigator.clipboard.writeText(text ?? ""))
    .catch((err) => console.error("read_llm_cache_entry / clipboard:", err));
}

// Flatten one logged call into a single plain-text block for clipboard
// paste. Mirrors the on-screen layout (Prompt then Response, role
// headers, message bodies) so what the user copies matches what they
// see — no provider-specific JSON noise. Multimodal vision parts are
// inlined as text (text parts verbatim, image parts as a `[image: url]`
// placeholder) since clipboard text can't carry the bytes.
function fullPromptEntryClipboardText(c) {
  const parts = [];
  if (Array.isArray(c.full_prompt)) {
    parts.push("=== PROMPT ===");
    for (const m of c.full_prompt) {
      const role = (m?.role || "?").toUpperCase();
      parts.push(`--- ${role} ---`);
      if (typeof m?.content === "string") {
        parts.push(m.content);
      } else if (Array.isArray(m?.content)) {
        for (const part of m.content) {
          if (part?.type === "text") {
            parts.push(part.text || "");
          } else if (part?.type === "image_url") {
            parts.push(`[image: ${part?.image_url?.url || ""}]`);
          } else {
            parts.push(JSON.stringify(part, null, 2));
          }
        }
      } else if (m?.content != null) {
        parts.push(JSON.stringify(m.content, null, 2));
      }
    }
  }
  if (typeof c.full_response === "string") {
    parts.push("=== RESPONSE ===");
    parts.push(c.full_response);
  }
  return parts.join("\n\n");
}

function FullPromptLogs({ calls }) {
  const logged = useMemo(
    () => (calls || []).filter(
      (c) => Array.isArray(c.full_prompt) || typeof c.full_response === "string"
    ),
    [calls],
  );
  const [openIds, setOpenIds] = useState(() => new Set());
  if (logged.length === 0) return null;
  const toggle = (id) => {
    setOpenIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  return (
    <section className="run-details-section run-details-full-prompt-logs">
      <h3>Full prompt + response logs ({logged.length})</h3>
      <p className="field-hint">
        Captured because Settings → Development → Full prompt logging
        was on for the stage. Disk-heavy by design.
      </p>
      {logged.map((c) => {
        const open = openIds.has(c.call_id);
        return (
          <details
            key={c.call_id}
            open={open}
            className="full-prompt-entry"
            onToggle={(e) => {
              if (e.currentTarget.open !== open) toggle(c.call_id);
            }}
          >
            <summary>
              <code>{c.call_id}</code>{" "}
              <span className="full-prompt-entry-meta">
                {c.stage || "—"} · {c.category || "—"}
              </span>
            </summary>
            {open ? (
              <div className="full-prompt-entry-body">
                {/* Upper-right CopyButton — same affordance the
                    attestation modal uses. Copies prompt + response
                    together as one plain-text block so a debug paste
                    into a chat / scratchpad has both sides in context.
                    The bare CopyButton positions itself absolutely; the
                    parent div is position:relative via CSS. */}
                <CopyButton
                  onClick={() => copyToClipboard(
                    fullPromptEntryClipboardText(c)
                  )}
                  testId={`full-prompt-copy-${c.call_id}`}
                  label="Copy prompt + response"
                />
                {Array.isArray(c.full_prompt) ? (
                  <div className="full-prompt-block">
                    <h4>Prompt</h4>
                    {c.full_prompt.map((m, i) => (
                      <div key={i} className="full-prompt-message">
                        <div className="full-prompt-role">{m.role || "?"}</div>
                        <FullPromptMessageContent content={m.content} />
                      </div>
                    ))}
                  </div>
                ) : null}
                {typeof c.full_response === "string" ? (
                  <div className="full-prompt-block">
                    <h4>Response</h4>
                    <pre className="full-prompt-content">{c.full_response}</pre>
                  </div>
                ) : null}
              </div>
            ) : null}
          </details>
        );
      })}
    </section>
  );
}

// Modal that surfaces files filtered out at staging time (issue #156).
// Informational only — accepted files are already in the staging list
// when this opens. Grouped by reason so a flood of `.DS_Store` doesn't
// drown the one `too large` row that the user actually wants to see.
export function IgnoredFilesModal({ data, onClose }) {
  const { items = [], imported = 0, folder = false } = data || {};
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

  // Group by reason for scannability. Reasons are stable strings from
  // validateInputs (`system file (.DS_Store)`, `too large (45.0 MB > 40
  // MB limit)`, …); insertion-order iteration follows first-occurrence
  // order, which roughly mirrors the input order.
  const byReason = new Map();
  for (const r of items) {
    if (!byReason.has(r.reason)) byReason.set(r.reason, []);
    byReason.get(r.reason).push(r);
  }
  // Title reports up to three counts: successfully imported now (only
  // meaningful for a bulk folder load), already-staged duplicates, and
  // validation rejections. Empty clauses drop out, so a pure rejection
  // batch still reads "N files ignored" — unchanged from before
  // duplicates were surfaced here.
  const fileWord = (k) => `${k} file${k === 1 ? "" : "s"}`;
  const dupeCount = items.filter((r) => r.reason === "already imported").length;
  const ignoredCount = items.length - dupeCount;
  const title = [
    folder && imported && `${fileWord(imported)} imported`,
    dupeCount && `${fileWord(dupeCount)} already imported`,
    ignoredCount && `${fileWord(ignoredCount)} ignored`,
  ]
    .filter(Boolean)
    .join(", ");

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal ignored-files-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2>{title}</h2>
          <button
            type="button"
            className="modal-close"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <p className="modal-lead">
          These files weren&apos;t added to the staging list. Hover a row
          to see the full path.
        </p>
        <div className="ignored-files-body">
          {[...byReason.entries()].map(([reason, items]) => (
            <section key={reason} className="ignored-files-group">
              <h3>
                {reason} <span className="ignored-files-count">({items.length})</span>
              </h3>
              <ul>
                {items.map(({ path }) => (
                  <li key={path} title={path}>{basename(path) || path}</li>
                ))}
              </ul>
            </section>
          ))}
        </div>
        <div className="modal-actions">
          <button type="button" className="btn-primary" onClick={onClose}>
            OK
          </button>
        </div>
      </div>
    </div>
  );
}
