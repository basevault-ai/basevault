use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use tauri::menu::{AboutMetadataBuilder, MenuBuilder, MenuItemBuilder, SubmenuBuilder};
use tauri::{Emitter, Manager, State};
use tauri_plugin_log::{RotationStrategy, Target, TargetKind};
use tauri_plugin_updater::UpdaterExt;

// ── DEV TRACING ──────────────────────────────────────────────────────────
// Opt-in timing instrumentation gated by the `dev_tracing` config flag
// (Settings → Development). Off by default; flipping the toggle in
// Settings is the only knob. Output goes through `info!()` so markers
// land in ~/.basevault/logs/app/app.log (alongside per-run
// <run-id>/run.log at the logs/ top level) on a normal Finder launch
// — no CLI relaunch needed. The first emit per layer becomes that
// layer's t=0; `wall=<unix_seconds.ms>` on every line lets the
// frontend / Rust / Python timelines be merged after the run via:
//
//     grep '\[LAUNCH_TRACE\]' ~/.basevault/logs/app/app.log | sort -k4
//
// Marker prefix is `[LAUNCH_TRACE]` regardless of which kind of trace
// (current set covers click → run-row-visible); future timing markers
// reuse the same gate and prefix so one grep recovers everything.
fn dev_tracing_on() -> bool {
    get_config()
        .get("dev_tracing")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

// Opt-in Tinfoil HTTP wire-capture gated by the `dev_wire_capture`
// config flag (Settings → Development). Rust spawns the Python pipeline
// with BASEVAULT_DEV_WIRE_CAPTURE=1 in the env iff the toggle is ON;
// user-set shell values are scrubbed Rust-side so the config flag is
// the only knob. Python's llm.py reads the env var at module init and
// attaches httpx event hooks on the TinfoilAI singleton when set.
// Output goes to ~/.basevault/logs/tinfoil-wire-<pid>.jsonl.
fn dev_wire_capture_on() -> bool {
    get_config()
        .get("dev_wire_capture")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}
static DEV_TRACING_T0: OnceLock<Instant> = OnceLock::new();
fn ltrace_rust(step: &str) {
    if !dev_tracing_on() { return; }
    let t0 = *DEV_TRACING_T0.get_or_init(Instant::now);
    let t = t0.elapsed().as_secs_f64();
    let wall = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    info!("[LAUNCH_TRACE] {} t={:.3} wall={:.3}", step, t, wall);
}

#[tauri::command]
fn dev_tracing_enabled() -> bool { dev_tracing_on() }

/// High-frequency frontend markers that fire on every poll/coalesce
/// tick and carry no state transition: the per-event driver heartbeat,
/// the coalescer's already-armed no-op, and the redundant per-tick
/// refresh/received pair. On a long run these alone were ~83% of the
/// `[LAUNCH_TRACE]` volume. They are relayed at `debug!` so a
/// default-config log (plugin filter is `LevelFilter::Info`) keeps
/// only the state-transition + launch-path skeleton, while raising the
/// log level to Debug brings them back — demote, not delete. The
/// state-transition markers (`scheduleruns_armed`/`fired`, run
/// lifecycle, `click`, listener registration) stay at INFO.
const LTRACE_DEBUG_MARKERS: &[&str] = &[
    "pipeline_progress_event",
    "scheduleruns_skipped_pending",
    "refreshruns_fired",
    "runs_received",
];

/// True when a relayed `[LAUNCH_TRACE]` line's marker token is a
/// per-tick heartbeat that should relay at `debug!` rather than
/// `info!`. A non-prefixed or unrecognized line is never demoted.
fn ltrace_is_demoted(line: &str) -> bool {
    line.strip_prefix("[LAUNCH_TRACE] ")
        .and_then(|rest| rest.split_whitespace().next())
        .is_some_and(|marker| LTRACE_DEBUG_MARKERS.contains(&marker))
}

/// Frontend ships a fully-formed `[LAUNCH_TRACE] <marker> …` line; we
/// relay to the same log sink so the .app log captures it next to Rust
/// + Python markers. The marker token (first word after the prefix)
/// selects the level: per-tick heartbeats go to `debug!`, everything
/// else stays `info!` (see `LTRACE_DEBUG_MARKERS`).
#[tauri::command]
fn record_dev_trace(line: String) {
    if !dev_tracing_on() { return; }
    if ltrace_is_demoted(&line) {
        debug!("{}", line);
    } else {
        info!("{}", line);
    }
}
// ── /DEV TRACING ─────────────────────────────────────────────────────────

// ── Session bootstrap ────────────────────────────────────────────────────────

/// UTC timestamp formatted as `YYYY-MM-DDTHH-MM-SSZ` (colons replaced with
/// dashes so it's safe as a directory name). Shells out to `date -u` to
/// avoid pulling in chrono as a dependency.
fn iso_z() -> String {
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H-%M-%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown-time".into())
}

/// Same dir-safe `YYYY-MM-DDTHH-MM-SSZ` format as [`iso_z`], but for an
/// explicit Unix-seconds instant rather than "now". The single source
/// for the run-dir/conversation-dir timestamp shape is the format
/// string here and in `iso_z` — callers must never hand-roll it. Used
/// by `create_convo` to advance to the next free second when a
/// same-second conversation id collision would otherwise occur.
fn iso_z_at(secs: u64) -> String {
    std::process::Command::new("date")
        .args(["-u", "-r", &secs.to_string(), "+%Y-%m-%dT%H-%M-%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown-time".into())
}

/// 4-char alphanumeric run id for human reference (logs, vault dir,
/// UI). Matches the alphabet used in the Python side so any tooling
/// that filters runs by short_id can do so consistently. Char set
/// is lowercase letters + digits, minus visually ambiguous 0/1/i/l/o
/// (~9.4M combinations, collision risk negligible at user volumes).
fn short_id() -> String {
    const ALPHABET: &[u8] = b"abcdefghjkmnpqrstuvwxyz23456789";
    use std::time::{SystemTime, UNIX_EPOCH};
    // Cheap entropy: hash the wall-clock nanos. Good enough for a 4-char
    // human-friendly identifier — we're not relying on it for security.
    let mut seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 ^ (d.as_secs() << 32))
        .unwrap_or(0xdead_beef);
    let mut out = String::with_capacity(4);
    for _ in 0..4 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let idx = ((seed >> 32) as usize) % ALPHABET.len();
        out.push(ALPHABET[idx] as char);
    }
    out
}

/// Extract the 4-letter perma-id from a run dir name shaped like
/// `<iso-z>-<short_id>` (the segment after the last `-`). The dir name
/// is the immutable record of last resort: the suffix IS the id the
/// shells minted, so even a run whose `config.json` was never written
/// (killed before provider/model pick) or a legacy `run.json` run with
/// no `short_id` field still surfaces its perma-id. Returns `None`
/// when the suffix isn't exactly 4 chars from the mint alphabet
/// (mirrors Python's `_short_id_from_run_name`); same alphabet as
/// `short_id()`.
fn short_id_from_run_name(name: &str) -> Option<String> {
    const ALPHABET: &[u8] = b"abcdefghjkmnpqrstuvwxyz23456789";
    let suffix = name.rsplit('-').next().unwrap_or("");
    if suffix.len() == 4 && suffix.bytes().all(|b| ALPHABET.contains(&b)) {
        Some(suffix.to_string())
    } else {
        None
    }
}

/// UTC timestamp with colons intact — for ISO-8601 timestamp fields inside JSON.
fn iso_z_full() -> String {
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown-time".into())
}

/// `~/.basevault` — Unix-convention app state dir. Logs, run sidecars,
/// migration marker, and any future internal bookkeeping live here.
fn state_root() -> PathBuf {
    std::env::var("HOME")
        .map(|h| PathBuf::from(h).join(".basevault"))
        .unwrap_or_else(|_| PathBuf::from("/tmp/BaseVault-state"))
}

/// `~/.basevault/logs` — app pipeline-run logs only (the GUI is the
/// only writer to this root). Dev runs (scripts, smoke, sweeps, tests,
/// ad-hoc CLI) live under `~/.basevault/logs-dev/` and are not scanned
/// by the Tauri shell.
fn logs_root() -> PathBuf {
    state_root().join("logs")
}

/// `~/.basevault/sessions/` — per-app-launch session dirs. One dir is
/// minted per Tauri-app-launch and reused for the lifetime of the
/// process (the Tauri shell + every Python subprocess it spawns).
/// Holds session-scoped artifacts: `app.log` (the Rust shell's
/// rotating log), `tinfoil-wire.jsonl` for SDK bootstrap + UI
/// attestation calls, and future session-scoped diagnostics.
/// Run-scoped logs (`llm-calls.jsonl`, `run.log`, …) stay under
/// `~/.basevault/logs/<run-id>/`; conversation-scoped logs stay in
/// the convo dir.
fn sessions_root() -> PathBuf {
    state_root().join("sessions")
}

/// Per-app-launch session dir, minted on first call and cached for
/// the lifetime of the process. Path shape:
/// `~/.basevault/sessions/<iso-z>-<short_id>/`. Passed to every
/// Python subprocess via `BASEVAULT_SESSION_DIR` so the shell + all
/// subprocesses share one session dir per launch.
static SESSION_DIR: OnceLock<PathBuf> = OnceLock::new();
fn session_dir() -> PathBuf {
    SESSION_DIR.get_or_init(|| {
        let name = format!("{}-{}", iso_z(), short_id());
        let dir = sessions_root().join(&name);
        let _ = std::fs::create_dir_all(&dir);
        dir
    }).clone()
}

/// Hard default, unaware of user config — only used inside migration
/// (which runs before any config can be trusted) and as the fallback
/// below. Consumer code should call `vault_root()`.
fn default_vault_root() -> PathBuf {
    std::env::var("HOME")
        .map(|h| PathBuf::from(h).join("Documents").join("BaseVault"))
        .unwrap_or_else(|_| PathBuf::from("/tmp/BaseVault"))
}

/// User-visible, Obsidian-browseable vault root. Reads
/// `obsidian_vault_path` from `~/.basevault/config.json`; falls back to
/// `~/Documents/BaseVault/` when not set. Contains a flat list of per-run
/// vault dirs (one dir per pipeline run). Honored by both Rust commands
/// and the Python runner (via the `BASEVAULT_VAULT_ROOT` env var passed
/// from `spawn_pipeline`).
fn vault_root() -> PathBuf {
    get_config()
        .get("obsidian_vault_path")
        .and_then(|v| v.as_str())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(default_vault_root)
}

/// Walk under `<logs_root>` and return paths to every APP run dir that
/// has a recognised state sidecar (`config.json` since issue #165, or
/// the legacy `run.json` for runs that pre-date the split). Returns
/// the SIDECAR path (not the dir) so existing callers that read the
/// file directly keep working — `parent()` recovers the run dir when
/// needed.
///
/// Layout-tolerant: accepts BOTH the legacy nested
/// `<session>/<eval>/<run>/` (depth 3) and the flat `<run>/` (depth 1)
/// shapes. The sweep harness now writes to `~/.basevault/logs-dev/sweeps/`
/// — a separate root we never scan — but a legacy `sweeps/` subdir may
/// still sit under `<logs_root>` from before the split, so the skip
/// below remains as a guard against indexing those as app runs.
fn walk_run_jsons(logs: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(logs) else { return out };
    for entry in entries.flatten() {
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        // sweeps/ is the eval-namespace; never index it as app runs.
        if p.file_name().and_then(|n| n.to_str()) == Some("sweeps") {
            continue;
        }
        if let Some(sidecar) = run_state_sidecar(&p) {
            out.push(sidecar);
            continue;
        }
        walk_for_sidecars(&p, 2, &mut out);
    }
    out
}

fn walk_for_sidecars(dir: &Path, max_depth: u32, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        // If this dir is itself a run dir, push its sidecar and DON'T
        // descend further — `stages/<n>/` etc. is internal layout we
        // never want to interpret as a child run.
        if let Some(sidecar) = run_state_sidecar(&p) {
            out.push(sidecar);
            continue;
        }
        if max_depth > 0 {
            walk_for_sidecars(&p, max_depth - 1, out);
        }
    }
}

/// Pick the canonical state sidecar inside a run dir: `config.json`
/// (preferred since issue #165) or `run.json` (legacy fallback).
/// Returns `None` if neither exists.
fn run_state_sidecar(run_dir: &Path) -> Option<PathBuf> {
    let cfg = run_dir.join("config.json");
    if cfg.exists() {
        return Some(cfg);
    }
    let legacy = run_dir.join("run.json");
    if legacy.exists() {
        return Some(legacy);
    }
    None
}

/// Find the state sidecar for a given run_id by scanning logs_root.
/// Returns the path to `config.json` when present; falls back to
/// `run.json` for runs created before the issue-#165 split.
fn find_run_json(run_id: &str) -> Option<PathBuf> {
    let logs = logs_root();
    if !logs.exists() {
        return None;
    }
    for p in walk_run_jsons(&logs) {
        if p.parent()
            .and_then(|d| d.file_name())
            .and_then(|n| n.to_str())
            == Some(run_id)
        {
            return Some(p);
        }
    }
    None
}

/// Find the run DIR for a given run_id. Equivalent to
/// `find_run_json(run_id).and_then(|p| p.parent().map(Path::to_path_buf))`
/// but doesn't require either sidecar to exist — returns the dir as
/// long as ANY of `config.json` / `run.json` is present (the find walk
/// already filters on that).
fn find_run_dir(run_id: &str) -> Option<PathBuf> {
    find_run_json(run_id).and_then(|p| p.parent().map(Path::to_path_buf))
}

/// Load the start-of-run snapshot for a run dir. Reads `config.json`
/// when present (canonical post-#165), otherwise falls back to
/// `run.json` for legacy runs. The returned value carries every field
/// the sidecar holds; callers that only want static fields should
/// project down themselves — the read path stays simple.
fn read_run_state(run_dir: &Path) -> Option<serde_json::Value> {
    let sidecar = run_state_sidecar(run_dir)?;
    let text = std::fs::read_to_string(&sidecar).ok()?;
    serde_json::from_str(&text).ok()
}

// ── Run state derivation from llm-calls.jsonl ──────────────────────────────
//
// Issue #165 retires the dynamic fields from run.json. Status / progress /
// duration / error all derive from the append-only event log + filesystem
// markers:
//   - cycle_error event → "failed"
//   - cycle_cancelled event → "cancelled"
//   - last-stage marker (`stages/06-embeddings/phase_1_marker.json`) → "completed".
//     Per `vault/os/dev/technical.md` § "Pipeline Stages", this file is the
//     ONLY signal that the entire pipeline finished. cycle_end alone is
//     not sufficient — a SIGTERM-paused run flushes cycle_end via the
//     runner's signal handler, but its filesystem state stops at whatever
//     phase was in flight (issue #206).
//   - paused.flag file in run dir → "paused" (Rust-written runtime marker
//     — pause is a runtime control signal that fires WITHOUT the runner
//     present to emit an event)
//   - cycle_start present, marker missing, runner pid alive → "running"
//   - cycle_start present, marker missing, runner pid not alive → "paused"
//     (covers SIGTERM-paused runs whose paused.flag write may have been
//     blocked by the prior cycle_end-as-completed bug, plus orphans from
//     prior app sessions; both are resumable from the latest phase
//     marker per the spec)
//
// Progress (completed / in_flight / eta_seconds / bar_position):
//   - completed: count of LEAF `end` events. A leaf is a call_id that no
//     other begin's `retry_of_call_id` points back at — i.e. the terminal
//     attempt of a retry chain. Counting every end overshoots the
//     denominator on retry-heavy runs (load retries, halve cascades,
//     sample retries, reasoning-off retries all add intermediate ends
//     that are NOT productive leaves; the tracker's denominator only
//     grows on structural fan-out via `_bump_stage_est`). Counting leaves makes
//     `X / Y calls` track productive work on both sides. Failed leaves
//     go into `progress.failed`, not `progress.completed` — a chain
//     that exhausted retries is done but didn't produce useful output.
//   - retries: count of intermediate ends — total ends minus leaf ends.
//     Surfaces retry storms (`+N retries` in the UI) without polluting
//     the headline. A 100-end / 42-leaf run reads `42/42 (+58 retries)`
//     instead of hiding the storm or lying with `100/42`.
//   - in_flight_calls: count of PENDING unmatched begins only — a
//     begin without a matching end whose cycle is the latest AND that
//     cycle has not wound down. Begins from a superseded earlier cycle
//     (resume) or any begin once the latest cycle ended (paused /
//     cancelled / errored / completed) are ABORTED, not in flight, and
//     are excluded — counting them showed phantom "N calls in progress"
//     on paused/wound-down runs (issue #581). Mirrors the per-call
//     pending/aborted split in `materialize_calls_*`.
//   - eta_seconds + total + bar_position: read from the latest
//     progress_tick event the runner emits. Drives the elapsed /
//     remaining line in the UI; on single-call stages (insights /
//     actions) where the next event is the call's `end`, the JSX
//     interpolates bar_position via `bar_position_at` so the bar
//     doesn't freeze for the 30-60s the call takes.
//
// Cache: derivation result keyed on jsonl mtime. `list_runs` may be
// called repeatedly across renders; re-walking the jsonl every time
// would scale linearly with the number of runs × calls.

#[derive(Clone, Debug, Default, Serialize)]
struct DerivedProgress {
    stage: Option<String>,
    /// Successful leaves (terminal calls that succeeded). The bar's
    /// numerator. A chain 0001 (timeout) → 0002 (retry, success)
    /// counts as 1. A chain 0001 (timeout) → 0002 (timeout) — both
    /// failed, no retry left — counts as 0 here, +1 in `failed`.
    completed: u64,
    /// Failed terminal leaves (retries exhausted, gave up). These
    /// should NOT count toward the bar's "X / Y completed" because
    /// they didn't produce useful output. Surfaced separately so
    /// the UI can show e.g. "5 done, 2 failed".
    failed: u64,
    /// Non-leaf attempts that were superseded by a retry. Used for
    /// the "+N retries" indicator without conflating them with
    /// productive completion.
    retries: u64,
    in_flight_calls: u64,
    total: Option<u64>,
    eta_seconds: Option<f64>,
    bar_position: Option<f64>,
    /// Wall-clock timestamp of the latest `progress_tick` event (ISO-Z).
    /// Surfaced to the JSX so it can interpolate `bar_position` over
    /// elapsed time between ticks — important on single-call stages
    /// (insights / actions) where the next event is the call's `end`,
    /// which can be 30-60s away. Without interpolation the bar would
    /// freeze at the last tick's value for the duration of the call.
    bar_position_at: Option<String>,
    stage_eta_seconds: Option<f64>,
    elapsed_in_stage: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct DerivedRunState {
    /// One of: running / completed / cancelled / failed / paused / unknown.
    /// "unknown" only when no cycle_start was ever emitted and no
    /// markers landed — usually means the runner started, crashed
    /// before emitting cycle_start, and no marker landed. Caller may
    /// decide to treat as "failed" or hide.
    status: String,
    progress: DerivedProgress,
    /// Wall-clock duration in milliseconds. For terminal runs
    /// (completed/cancelled/failed): cycle_start.ts → terminator.ts.
    /// For running runs as returned by `derive_run_state`: closed-
    /// cycle accumulator extended by `(now - open_cycle_start_ts)`.
    ///
    /// The extension is applied OUTSIDE the cache so the displayed
    /// elapsed ticks between jsonl writes — the cache stores only
    /// the event-derived closed accumulator (mtime-keyed correctly),
    /// and `derive_run_state` re-anchors to `now` on every call.
    /// Cached entries from `derive_run_state_uncached` hold the
    /// closed-only value; the wrapper does the extension.
    duration_ms: Option<u64>,
    /// ISO-Z timestamp of the most-recent cycle_start whose
    /// terminator hasn't landed yet. None on terminal runs. Used by
    /// `derive_run_state` to extend `duration_ms` by `(now - this)`
    /// at call time, outside the mtime-keyed cache. Internal-only —
    /// skipped from the run payload (the frontend reads the
    /// already-extended `duration_ms`).
    #[serde(skip_serializing)]
    open_cycle_start_ts: Option<String>,
    /// jsonl mtime in ISO-8601 — surfaces "when was the last event".
    updated_at: Option<String>,
    /// First non-empty error message from a cycle_error event.
    error: Option<String>,
    /// pid recorded in the latest cycle_start event (if present). Used
    /// by orphan-recovery + delete_run for the kill path.
    pid: Option<u32>,
    /// Subject-resolution + bundle-mode payloads emitted by the
    /// entities stage as `entities_decision` events. None until the
    /// entities stage finishes. Surfaced into run.run_config by the
    /// list_runs merge so the modal's RunDetails section can render
    /// them on completed AND running runs.
    subject_resolution: Option<serde_json::Value>,
    bundle_mode: Option<bool>,
}

/// Walk a run's `llm-calls.jsonl` + check filesystem markers, producing
/// the derived state record. Pure function — no caching here so
/// callers can decide their own cache key.
fn derive_run_state_uncached(run_dir: &Path) -> DerivedRunState {
    let jsonl = run_dir.join("llm-calls.jsonl");
    let mut state = DerivedRunState::default();

    // jsonl mtime → updated_at. Cheap stat call.
    if let Ok(meta) = std::fs::metadata(&jsonl) {
        if let Ok(modified) = meta.modified() {
            if let Ok(dur) = modified.duration_since(std::time::UNIX_EPOCH) {
                state.updated_at = Some(format_iso_z_full_from_secs(dur.as_secs()));
            }
        }
    }

    // Parse jsonl. Track begins by call_id so we can compute in_flight,
    // and collect cycle markers.
    let mut cycle_start_ts: Option<String> = None;
    let mut cycle_start_pid: Option<u32> = None;
    // Active-runtime accumulator (#617). Walks (cycle_start, terminator)
    // pairs and sums each pair's duration. `open_cycle_start_ts` is the
    // timestamp of the most-recent cycle_start whose terminator hasn't
    // landed yet; the terminator closes it by adding (ts - open_start)
    // to `accumulated_ms`. While the run is in-flight running, we
    // extend by (now - latest open_start). Pause windows lie OUTSIDE
    // any open cycle, so they're excluded by construction. Orphan
    // cycles (cycle_start without matching terminator, then another
    // cycle_start) drop their runtime — the new cycle_start overwrites
    // open_cycle_start_ts.
    let mut accumulated_ms: i64 = 0;
    let mut open_cycle_start_ts: Option<String> = None;
    let mut cycle_start_stage: Option<String> = None;
    let mut latest_stage_from_begin: Option<String> = None;
    let mut cycle_cancelled_ts: Option<String> = None;
    let mut cycle_error_ts: Option<String> = None;
    let mut cycle_error_msg: Option<String> = None;
    // call_id → the cycle_seq the begin fired in. Used to split
    // unmatched begins into PENDING (latest cycle, still alive) vs
    // ABORTED (superseded earlier cycle, or latest cycle wound down)
    // for the in_flight count — same discriminant as the per-call
    // materializer (issue #581).
    let mut begins: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();
    // Running cycle counter. Mirrors `materialize_*`: prefer the
    // cycle_start event's `cycle_seq`; pre-cycle_seq runs / fixtures
    // fall back to a monotonic bump so resume ordering still holds.
    let mut current_cycle_seq: u64 = 0;
    // Set of call_ids that are retry-parents (some other begin's
    // retry_of_call_id pointed to them). The leaf computation at the
    // end of derive_run_state subtracts these from ended_call_ids so
    // a chain like 0001 (failed) → 0002 (retry, success) counts as
    // ONE completed call (the leaf 0002) — not two.
    let mut parents: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    // Every (call_id, is_success, cache_key) tuple with a matching
    // `end` event. Used together with `parents` to compute leaves.
    // Bar's `completed` counts only SUCCESSFUL leaves so a failed
    // terminal timeout doesn't bump the numerator (the user sees
    // X / Y read as "X work units done", not "X attempts ended").
    // `cache_key` is captured so the leaf count can dedupe cross-
    // cycle redos: when cycle 2 redoes cycle 1's already-successful
    // work (same prompt → same cache_key, cached=true), both ends
    // land in the jsonl but they represent ONE unit of pipeline
    // work, not two. Counting both would inflate `completed` past
    // `total` every restart by the size of the prior cycle's
    // successful set.
    let mut ended_call_ids: Vec<(String, bool, Option<String>)> = Vec::new();

    if let Ok(text) = std::fs::read_to_string(&jsonl) {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(ev) = serde_json::from_str::<serde_json::Value>(line) else { continue };
            let event = ev.get("event").and_then(|v| v.as_str()).unwrap_or("");
            match event {
                "cycle_start" => {
                    // Latest cycle_start wins (resume = new cycle); track
                    // the freshest pid + ts so orphan recovery targets
                    // the live process, not a defunct one.
                    let ts = ev.get("ts").and_then(|v| v.as_str()).map(String::from);
                    cycle_start_ts = ts.clone();
                    // Open a new accumulator slot. If a previous open_start
                    // was still set (orphan cycle — Python died via SIGKILL
                    // without flushing cycle_end), it's silently dropped:
                    // we have no terminator timestamp so we can't account
                    // for the orphan's runtime.
                    open_cycle_start_ts = ts;
                    if let Some(p) = ev.get("pid").and_then(|v| v.as_u64()) {
                        cycle_start_pid = u32::try_from(p).ok();
                    }
                    cycle_start_stage = Some("init".to_string());
                    if let Some(seq) = ev.get("cycle_seq").and_then(|v| v.as_u64()) {
                        current_cycle_seq = seq;
                    } else {
                        current_cycle_seq = current_cycle_seq.saturating_add(1);
                    }
                    // A new cycle resets terminators — we're in flight again.
                    cycle_cancelled_ts = None;
                    cycle_error_ts = None;
                    cycle_error_msg = None;
                }
                "cycle_end" => {
                    let ts = ev.get("ts").and_then(|v| v.as_str()).map(String::from);
                    if let (Some(start), Some(end)) =
                        (open_cycle_start_ts.as_deref(), ts.as_deref())
                    {
                        if let Some(d) = iso_delta_ms(start, end) {
                            accumulated_ms = accumulated_ms.saturating_add(d.max(0));
                        }
                        open_cycle_start_ts = None;
                    }
                }
                "cycle_cancelled" => {
                    let ts = ev.get("ts").and_then(|v| v.as_str()).map(String::from);
                    cycle_cancelled_ts = ts.clone();
                    if let (Some(start), Some(end)) =
                        (open_cycle_start_ts.as_deref(), ts.as_deref())
                    {
                        if let Some(d) = iso_delta_ms(start, end) {
                            accumulated_ms = accumulated_ms.saturating_add(d.max(0));
                        }
                        open_cycle_start_ts = None;
                    }
                }
                "cycle_error" => {
                    let ts = ev.get("ts").and_then(|v| v.as_str()).map(String::from);
                    cycle_error_ts = ts.clone();
                    cycle_error_msg = ev
                        .get("message")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    if let (Some(start), Some(end)) =
                        (open_cycle_start_ts.as_deref(), ts.as_deref())
                    {
                        if let Some(d) = iso_delta_ms(start, end) {
                            accumulated_ms = accumulated_ms.saturating_add(d.max(0));
                        }
                        open_cycle_start_ts = None;
                    }
                }
                "entities_decision" => {
                    if let Some(sr) = ev.get("subject_resolution") {
                        if !sr.is_null() {
                            state.subject_resolution = Some(sr.clone());
                        }
                    }
                    if let Some(bm) = ev.get("bundle_mode").and_then(|v| v.as_bool()) {
                        state.bundle_mode = Some(bm);
                    }
                }
                "begin" => {
                    if let Some(cid) = ev.get("call_id").and_then(|v| v.as_str()) {
                        begins.insert(cid.to_string(), current_cycle_seq);
                    }
                    // Track retry parents so the leaf computation
                    // below can subtract them from completed. A
                    // chain like 0001 (timeout) → 0002 (retry of
                    // 0001, succeeds) ends with TWO end events —
                    // but only ONE represents user-visible work
                    // completed (the leaf, 0002). Without this,
                    // failed retries inflate the progress bar's
                    // numerator.
                    if let Some(rof) = ev.get("retry_of_call_id").and_then(|v| v.as_str()) {
                        parents.insert(rof.to_string());
                    }
                    // Begin events carry `stage` for the per-call
                    // by_stage rollup (vision rows bucket under
                    // "vision", chat rows bucket under "extract" /
                    // "patterns" / etc.). They DO NOT drive the
                    // UI's progress-bar stage label — that comes
                    // from progress_tick events only, which emit
                    // runner._stage. Without this carve-out, vision
                    // calls during ingest would flip the UI label
                    // between "Ingesting inputs" and "Transcribing
                    // images" (vision is a sub-component of ingest,
                    // not its own user-visible stage).
                }
                "end" => {
                    if let Some(cid) = ev.get("call_id").and_then(|v| v.as_str()) {
                        begins.remove(cid);
                        let is_success = ev
                            .get("success")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        let cache_key = ev
                            .get("cache_key")
                            .and_then(|v| v.as_str())
                            .map(String::from);
                        ended_call_ids.push((cid.to_string(), is_success, cache_key));
                    }
                }
                "progress_tick" => {
                    // Runner emits one per `_emit()` call (LLM-call
                    // boundaries + stage transitions). Latest tick wins;
                    // we surface its progress fields on the derived
                    // record so the runs-list bar advances even on
                    // single-call stages where no begin/end events
                    // arrive between the start and the call's end.
                    if let Some(s) = ev.get("stage").and_then(|v| v.as_str()) {
                        latest_stage_from_begin = Some(s.to_string());
                    }
                    if let Some(t) = ev.get("total").and_then(|v| v.as_u64()) {
                        state.progress.total = Some(t);
                    }
                    if let Some(b) = ev.get("bar_position").and_then(|v| v.as_f64()) {
                        state.progress.bar_position = Some(b);
                    }
                    if let Some(eta) = ev.get("eta_seconds").and_then(|v| v.as_f64()) {
                        state.progress.eta_seconds = Some(eta);
                    }
                    if let Some(seta) = ev.get("stage_eta_seconds").and_then(|v| v.as_f64()) {
                        state.progress.stage_eta_seconds = Some(seta);
                    }
                    if let Some(eis) = ev.get("elapsed_in_stage").and_then(|v| v.as_f64()) {
                        state.progress.elapsed_in_stage = Some(eis);
                    }
                    if let Some(ts) = ev.get("ts").and_then(|v| v.as_str()) {
                        state.progress.bar_position_at = Some(ts.to_string());
                    }
                }
                _ => {} // counts, future events — ignore
            }
        }
    }

    // Leaves = ended call_ids that aren't retry-parents (terminal
    // attempts in their chain). `completed` counts only SUCCESSFUL
    // leaves — a failed terminal timeout shouldn't bump the bar's
    // numerator. `failed_leaves` counts terminal failures (work the
    // pipeline gave up on, not retried). `retries` counts non-leaf
    // ends (failed attempts that WERE retried).
    //
    // Successful leaves are ALSO deduped by `cache_key`: when cycle 2
    // resumes mid-stage it re-emits begin/end pairs for cycle 1's
    // already-successful calls (the rerun short-circuits via the LLM
    // cache; cached=true on the second end). Two ends with the same
    // cache_key represent ONE pipeline work unit; counting both
    // inflates `completed` past `total` every restart by the size of
    // the prior cycle's successful set, which clamps the bar. Each
    // distinct cache_key counts at most once. Leaves without a
    // cache_key (e.g. older runs that pre-date the field on `end`
    // events) count individually — back-compat.
    let leaves: Vec<&(String, bool, Option<String>)> = ended_call_ids
        .iter()
        .filter(|(cid, _, _)| !parents.contains(cid))
        .collect();
    let mut seen_success_cache_keys: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut successful_leaves: u64 = 0;
    let mut failed_leaves: u64 = 0;
    for (_cid, ok, ck) in leaves.iter() {
        if *ok {
            match ck {
                Some(k) if !k.is_empty() => {
                    if seen_success_cache_keys.insert(k.clone()) {
                        successful_leaves += 1;
                    }
                }
                _ => {
                    successful_leaves += 1;
                }
            }
        } else {
            failed_leaves += 1;
        }
    }
    state.progress.completed = successful_leaves;
    state.progress.failed = failed_leaves;
    state.progress.retries = (ended_call_ids.len() as u64)
        .saturating_sub(leaves.len() as u64);
    // in_flight_calls is assigned AFTER the status block below — a
    // begin without an end only counts as "in progress" when the run
    // is actually running its latest cycle. On a paused / cancelled /
    // failed / completed run, or for begins stranded by a resume's
    // superseding cycle, the call is ABORTED, not in flight. Counting
    // those produced phantom "N calls in progress" (issue #581).
    state.progress.stage = latest_stage_from_begin.or(cycle_start_stage);
    state.pid = cycle_start_pid;

    // Filesystem marker for paused. Pause is a Rust-driven runtime
    // signal that fires WITHOUT the runner present to emit a jsonl
    // event, so the only place to record it is the filesystem.
    let paused_flag = run_dir.join("paused.flag");

    // Pipeline-complete signal: the last stage's terminal phase marker
    // (per `vault/os/dev/technical.md` § "Pipeline Stages"). Existence
    // of this file is the canonical "the entire pipeline finished" bit.
    // cycle_end ≠ completed: the runner's SIGTERM handler emits cycle_end
    // when a pause flushes the rollup, even though the pipeline didn't
    // reach the terminal stage (issue #206).
    let last_stage_marker = run_dir
        .join("stages")
        .join("06-embeddings")
        .join("phase_1_marker.json");
    let pipeline_complete = last_stage_marker.exists();

    // Status precedence:
    //   1. cycle_error      → failed     (explicit terminal)
    //   2. cycle_cancelled  → cancelled  (explicit terminal — note this
    //      is the RUN-level status, distinct from the per-call
    //      `aborted` outcome bucket: the run was explicitly
    //      cancelled by the user, while a single call can be
    //      `aborted` in any run that wound down. Don't conflate.)
    //   3. last-stage marker → completed (pipeline finished)
    //   4. paused.flag      → paused     (Rust-written runtime control)
    //   5. cycle_start with live runner pid → running
    //   6. cycle_start with dead runner pid → paused
    //      (resume restarts from the latest phase marker)
    //   7. nothing → unknown
    state.status = if cycle_error_ts.is_some() {
        "failed".into()
    } else if cycle_cancelled_ts.is_some() {
        "cancelled".into()
    } else if pipeline_complete {
        "completed".into()
    } else if paused_flag.exists() {
        "paused".into()
    } else if cycle_start_ts.is_some() {
        // pid aliveness disambiguates running vs paused. Missing pid
        // (very old runs that didn't stamp it on cycle_start) defaults
        // to running — same fallback the pre-#206 logic always used.
        let alive = cycle_start_pid.map(is_pid_alive).unwrap_or(true);
        if alive { "running".into() } else { "paused".into() }
    } else {
        "unknown".into()
    };
    state.error = cycle_error_msg;

    // Pending-only in_flight (issue #581). A non-running run has, by
    // definition, no LLM call executing — every dangling begin is an
    // aborted artifact of the wind-down, so the count is 0. While
    // running, a begin is in flight only if it fired in the latest
    // cycle; begins from an earlier, superseded cycle (resume redid
    // the work) are aborted. This is the same pending-vs-aborted
    // discriminant the per-call materializer applies.
    state.progress.in_flight_calls = if state.status == "running" {
        begins
            .values()
            .filter(|&&seq| seq == current_cycle_seq)
            .count() as u64
    } else {
        0
    };

    // duration_ms: closed-cycle accumulator only. The accumulator
    // holds the sum of all closed (cycle_start, terminator) pairs we
    // walked — purely event-derived and cache-stable. The wallclock
    // extension `(now - open_cycle_start_ts)` lives outside the cache:
    // `derive_run_state` adds it after the mtime-keyed lookup so the
    // displayed elapsed ticks between jsonl writes. Pause windows are
    // not counted — the gap between a terminator and the next
    // cycle_start sits outside any open slot.
    if cycle_start_ts.is_some() {
        state.duration_ms = Some(accumulated_ms.max(0) as u64);
        state.open_cycle_start_ts = open_cycle_start_ts;
    }

    state
}

/// Cheap UTC ISO-8601 stamp (with colons) from a Unix-seconds value.
/// Avoids pulling in chrono — uses libc gmtime via `date -u`.
fn format_iso_z_full_from_secs(secs: u64) -> String {
    use std::process::Command;
    Command::new("date")
        .args(["-u", "-r", &secs.to_string(), "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown-time".into())
}

/// Best-effort ISO-Z delta in milliseconds. Returns None when either
/// stamp is unparseable. Mirrors the Python-side _iso_delta_ms — keeps
/// the formats compatible across the boundary.
fn iso_delta_ms(start: &str, end: &str) -> Option<i64> {
    fn parse(s: &str) -> Option<i64> {
        // Accept "YYYY-MM-DDTHH:MM:SSZ" or with millisecond precision
        // ("YYYY-MM-DDTHH:MM:SS.sssZ"). Also accept the filename-safe
        // dashes form "YYYY-MM-DDTHH-MM-SSZ" that runner.py's `_iso_z()`
        // stamps onto cycle_start / cycle_end / cycle_cancelled /
        // cycle_error events — the same convention as the run-dir name.
        // Both formats coexist on disk; without the dashes branch
        // duration_ms goes missing for every post-#169 run.
        let trimmed = s.trim_end_matches('Z');
        let (date_part, time_part) = trimmed.split_once('T')?;
        let mut date_iter = date_part.split('-');
        let y: i64 = date_iter.next()?.parse().ok()?;
        let mo: i64 = date_iter.next()?.parse().ok()?;
        let d: i64 = date_iter.next()?.parse().ok()?;
        let (hms, frac_ms) = match time_part.split_once('.') {
            Some((h, f)) => {
                // Accept any sub-second precision; pad / truncate to 3 digits.
                let mut s = String::from(f);
                while s.len() < 3 { s.push('0'); }
                let ms: i64 = s[..3].parse().ok()?;
                (h, ms)
            }
            None => (time_part, 0),
        };
        let hms_normalized = hms.replace('-', ":");
        let mut t_iter = hms_normalized.split(':');
        let h: i64 = t_iter.next()?.parse().ok()?;
        let mi: i64 = t_iter.next()?.parse().ok()?;
        let se: i64 = t_iter.next()?.parse().ok()?;
        // Convert to days since epoch via a lazy civil-from-Gregorian
        // calc (Howard Hinnant's chrono algorithm — public domain).
        let yy = if mo <= 2 { y - 1 } else { y };
        let era = if yy >= 0 { yy / 400 } else { (yy - 399) / 400 };
        let yoe = yy - era * 400;
        let doy = (153 * (mo + (if mo > 2 { -3 } else { 9 })) + 2) / 5
            + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        let days = era * 146097 + doe - 719468;
        let secs = days * 86_400 + h * 3600 + mi * 60 + se;
        Some(secs * 1000 + frac_ms)
    }
    let s = parse(start)?;
    let e = parse(end)?;
    Some(e - s)
}

// Cache for derive_run_state — keyed on (run_dir, jsonl_mtime). Avoids
// re-walking the jsonl on every list_runs call. The cache is small
// (one entry per known run); runs are evicted when their dir disappears
// is left to natural process churn — a stale entry just yields the
// last-snapshot derived state until eviction.
//
// Cache key carries the jsonl mtime, the paused.flag mtime, AND the
// last-stage marker mtime (each Option, since any of the three may be
// absent). Any write to any of them — runner appending an event, Rust
// writing/clearing paused.flag, runner materializing the stage-5 marker
// on completion — bumps a key component and the next call re-walks.
// Without the paused.flag mtime in the key, a `set_run_status "paused"`
// would be invisible until the next jsonl event landed (which on a
// paused-ergo-dead runner is "never"). Without the marker mtime in the
// key, a stage-5 completion that lands silently (no concurrent jsonl
// write) would not flip the cached "running" → "completed".
type DeriveCacheKey = (
    Option<std::time::SystemTime>,
    Option<std::time::SystemTime>,
    Option<std::time::SystemTime>,
);
static DERIVE_CACHE: std::sync::OnceLock<
    Mutex<std::collections::HashMap<PathBuf, (DeriveCacheKey, DerivedRunState)>>,
> = std::sync::OnceLock::new();

fn derive_cache() -> &'static Mutex<
    std::collections::HashMap<PathBuf, (DeriveCacheKey, DerivedRunState)>,
> {
    DERIVE_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Cached derivation: walks jsonl + markers only when one of the
/// inputs has changed since the last computation. The cache key is
/// (jsonl mtime, paused.flag mtime, last-stage marker mtime) — any
/// write to any of them re-walks; if none changed, the cached value
/// returns in O(1).
///
/// The cached payload holds the CLOSED-cycle accumulator (event-
/// derived, mtime-stable). After the lookup, the wallclock extension
/// `(now - open_cycle_start_ts)` is added in-place for running runs
/// so the displayed elapsed ticks between jsonl writes. A terminal
/// run has no open anchor and the extension is a no-op.
fn derive_run_state(run_dir: &Path) -> DerivedRunState {
    let jsonl = run_dir.join("llm-calls.jsonl");
    let paused = run_dir.join("paused.flag");
    let last_stage_marker = run_dir
        .join("stages")
        .join("06-embeddings")
        .join("phase_1_marker.json");
    let key: DeriveCacheKey = (
        std::fs::metadata(&jsonl).and_then(|m| m.modified()).ok(),
        std::fs::metadata(&paused).and_then(|m| m.modified()).ok(),
        std::fs::metadata(&last_stage_marker).and_then(|m| m.modified()).ok(),
    );
    let cached_hit = {
        let cache = derive_cache().lock().unwrap();
        cache
            .get(run_dir)
            .filter(|(cached_key, _)| cached_key == &key)
            .map(|(_, cached)| cached.clone())
    };
    let mut state = match cached_hit {
        Some(s) => s,
        None => {
            let fresh = derive_run_state_uncached(run_dir);
            let mut cache = derive_cache().lock().unwrap();
            cache.insert(run_dir.to_path_buf(), (key, fresh.clone()));
            fresh
        }
    };
    extend_open_cycle_to_now(&mut state);
    state
}

/// Add the wallclock extension `(now - open_cycle_start_ts)` to a
/// state pulled from the derive cache. The cache stores only the
/// event-derived closed accumulator (mtime-keyed correctly); the
/// extension lives here so a cache HIT still produces a now-anchored
/// duration on every call. No-op on terminal runs (no open anchor)
/// and on unparseable timestamps.
fn extend_open_cycle_to_now(state: &mut DerivedRunState) {
    if let Some(open_start) = state.open_cycle_start_ts.as_deref() {
        let now_ts = iso_z_full();
        if let Some(d) = iso_delta_ms(open_start, &now_ts) {
            let base = state.duration_ms.unwrap_or(0);
            state.duration_ms = Some(base.saturating_add(d.max(0) as u64));
        }
    }
}

#[cfg(test)]
fn clear_derive_cache_for_tests() {
    let mut cache = derive_cache().lock().unwrap();
    cache.clear();
}

/// In-flight materializer (issue #165): walk `llm-calls.jsonl` and
/// produce a rollup payload close enough in shape to llm-stats.json
/// that the JSX Run Details modal renders meaningfully on a running
/// run. The post-run Python rollup is richer (per-stage statistics
/// distribution, outcome heuristics for blanked/timeout/etc.); the
/// modal refetches as pipeline-progress events fire and eventually
/// picks up the full rollup at end-of-run, so the in-flight version
/// only needs the columns the modal actually displays.
///
/// Returns `Value::Null` when the jsonl is absent or empty.
fn materialize_llm_stats_from_jsonl(run_dir: &Path) -> serde_json::Value {
    use serde_json::{json, Value};
    let jsonl = run_dir.join("llm-calls.jsonl");
    let Ok(text) = std::fs::read_to_string(&jsonl) else { return Value::Null };
    if text.trim().is_empty() {
        return Value::Null;
    }

    // ── Per-call collection ────────────────────────────────────────────
    use std::collections::BTreeMap;
    // BTreeMap to preserve first-seen order via separate vec; the map
    // is just for O(1) lookup.
    let mut by_id: BTreeMap<String, serde_json::Map<String, Value>> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut ended_at_iso: Option<String> = None;
    // Walking-state cycle bookkeeping. Each begin stamps the
    // `current_cycle_seq` it fired in onto the call record, so the
    // post-walk classifier can distinguish three states for unmatched
    // begins:
    //   - begin.cycle_seq < latest cycle  → INCOMPLETE
    //     (the call's cycle was superseded by a later cycle_start;
    //     pause-and-resume case — work was redone in the new cycle)
    //   - begin.cycle_seq == latest cycle, latest cycle ended →
    //     INCOMPLETE (run wound down with this call mid-flight)
    //   - begin.cycle_seq == latest cycle, latest cycle alive →
    //     PENDING (the call is still streaming; end is coming)
    // Without per-call cycle_seq, the resume case mis-tagged cycle-1
    // orphans as "pending" once cycle 2 reset ended_at_iso (s9e3).
    let mut current_cycle_seq: u64 = 0;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(ev) = serde_json::from_str::<Value>(line) else { continue };
        let event = ev.get("event").and_then(|v| v.as_str()).unwrap_or("");
        // Cycle markers: track current cycle's seq for begin stamping
        // and the freshest terminator ts for duration anchoring. A
        // new cycle_start RESETS ended_at_iso so cycle 2's in-flight
        // begins don't see cycle 1's terminator and get mis-tagged
        // (mirrors derive_run_state's cycle_start handling).
        match event {
            "cycle_start" => {
                ended_at_iso = None;
                if let Some(seq) = ev.get("cycle_seq").and_then(|v| v.as_u64()) {
                    current_cycle_seq = seq;
                } else {
                    // Pre-cycle_seq runs (or fixtures without it):
                    // increment monotonically so ordering still works.
                    current_cycle_seq = current_cycle_seq.saturating_add(1);
                }
            }
            "cycle_end" | "cycle_cancelled" | "cycle_error" => {
                if let Some(ts) = ev.get("ts").and_then(|v| v.as_str()) {
                    ended_at_iso = Some(ts.to_string());
                }
            }
            _ => {}
        }
        let cid = match ev.get("call_id").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        match event {
            "begin" => {
                if by_id.contains_key(&cid) {
                    continue; // ignore duplicate begin
                }
                let mut rec = serde_json::Map::new();
                rec.insert("call_id".into(), Value::String(cid.clone()));
                for f in ["stage", "category", "model", "started_at_iso",
                          "template_hash"] {
                    if let Some(v) = ev.get(f) {
                        rec.insert(f.into(), v.clone());
                    } else {
                        rec.insert(f.into(), Value::Null);
                    }
                }
                if let Some(b) = ev.get("budget") {
                    rec.insert("budget".into(), b.clone());
                }
                if let Some(extras) = ev.get("request_extras") {
                    rec.insert("request_extras".into(), extras.clone());
                }
                if let Some(a) = ev.get("attempt") {
                    rec.insert("attempt".into(), a.clone());
                }
                if let Some(r) = ev.get("retry_of_call_id") {
                    rec.insert("retry_of_call_id".into(), r.clone());
                }
                if let Some(d) = ev.get("retry_delay_ms") {
                    rec.insert("retry_delay_ms".into(), d.clone());
                }
                // Defaults that the post-run materializer also writes;
                // keeps the JSX shape uniform between the two paths.
                // `mode` is legacy (filled by Python's _record_usage on
                // wire-success only; null on cache-hit / failure paths).
                // `provider` is the post-fold-in finer-grained backend
                // tag stamped at begin time, so cache-hit + failure
                // paths carry it too.
                rec.insert("mode".into(), Value::Null);
                rec.insert(
                    "provider".into(),
                    ev.get("provider").cloned().unwrap_or(Value::Null),
                );
                rec.insert("duration_ms".into(), Value::Null);
                rec.insert("success".into(), Value::Null);
                rec.insert("error".into(), Value::Null);
                // Issue #225: seed prompt_tokens from the begin-event
                // pre-flight estimate so the live in-flight pane shows
                // a real number for pending requests (not "—") and so
                // wire-cut completions whose end event reports 0 still
                // surface the estimate. Mirrors the Python materializer
                // at runner.py::_materialize_calls_from_jsonl. Pre-this-
                // PR begin events without `prompt_tokens_est` read as
                // Null — same as before.
                if let Some(est) = ev.get("prompt_tokens_est") {
                    rec.insert("prompt_tokens".into(), est.clone());
                } else {
                    rec.insert("prompt_tokens".into(), Value::Null);
                }
                rec.insert("completion_tokens".into(), Value::Null);
                rec.insert("reasoning_tokens".into(), Value::Null);
                rec.insert("reasoning_tokens_source".into(), Value::Null);
                rec.insert("content_tokens".into(), Value::Null);
                rec.insert("finish_reason".into(), Value::Null);
                rec.insert("ttft_ms".into(), Value::Null);
                rec.insert("ttfr_ms".into(), Value::Null);
                rec.insert("last_token_ms".into(), Value::Null);
                // Pre-call max_tokens reservation: the runner wrapper
                // pre-computes it and stamps the begin event so the
                // in-flight pane shows what the call reserved without
                // waiting for end-of-call. Mirrors the prompt_tokens_est
                // pattern above. Pre-this-PR begin events without
                // `max_tokens_reserved` read as Null — same default as
                // before, end event still fills it in.
                if let Some(mtr) = ev.get("max_tokens_reserved") {
                    rec.insert("max_tokens_reserved".into(), mtr.clone());
                } else {
                    rec.insert("max_tokens_reserved".into(), Value::Null);
                }
                rec.insert("input".into(), Value::Null);
                rec.insert("output".into(), Value::Null);
                rec.insert("aborted".into(), Value::Bool(false));
                // Issue #333: per-call user-skip. Marker-dir scan
                // below stamps `true` for matching call_ids; the
                // outcome classifier then produces "skipped" with
                // priority over both pending and aborted.
                rec.insert("skipped".into(), Value::Bool(false));
                rec.insert("cached".into(), Value::Bool(false));
                // Cycle the begin fired in. Used post-walk to
                // distinguish abandoned cycle-1 begins from live
                // cycle-2 begins after a resume.
                rec.insert(
                    "cycle_seq".into(),
                    Value::Number(current_cycle_seq.into()),
                );
                rec.insert("cache_key".into(), Value::Null);
                rec.insert("parse_error".into(), Value::Bool(false));
                rec.insert("empty_response".into(), Value::Bool(false));
                rec.insert("interrupted".into(), Value::Bool(false));
                by_id.insert(cid.clone(), rec);
                order.push(cid);
            }
            "end" => {
                let Some(rec) = by_id.get_mut(&cid) else { continue };
                // Issue #225: end's prompt_tokens overwrites the begin-
                // time estimate ONLY when truthy. A 0 (wire-cut, usage
                // chunk never arrived) leaves the estimate intact so
                // the per-call detail UI never shows `0 in` for a call
                // that genuinely sent thousands of input tokens.
                // Handled separately from the bulk loop below because
                // its predicate is "not null AND not zero," not just
                // "not null."
                if let Some(v) = ev.get("prompt_tokens") {
                    let truthy = match v {
                        Value::Null => false,
                        Value::Number(n) => n.as_i64().map(|i| i != 0)
                            .or_else(|| n.as_f64().map(|f| f != 0.0))
                            .unwrap_or(false),
                        _ => true,
                    };
                    if truthy {
                        rec.insert("prompt_tokens".into(), v.clone());
                    }
                }
                for f in [
                    "duration_ms", "success", "error",
                    "completion_tokens",
                    "reasoning_tokens", "reasoning_tokens_source",
                    "content_tokens", "finish_reason",
                    "ttft_ms", "ttfr_ms", "last_token_ms",
                    "max_tokens_reserved",
                    // #963: the kernel's categorized outcome, carried on
                    // the end event. The failure classifier below reads it
                    // as the authoritative load/sizing/other bucket; without
                    // copying it onto the rec a no-exception LOAD collapses
                    // to `failed (other)`.
                    "llm_status",
                ] {
                    if let Some(v) = ev.get(f) {
                        if !v.is_null() {
                            rec.insert(f.into(), v.clone());
                        }
                    }
                }
                if let Some(m) = ev.get("model") {
                    if !m.is_null() {
                        rec.insert("model".into(), m.clone());
                    }
                }
                if let Some(m) = ev.get("mode") {
                    if !m.is_null() {
                        rec.insert("mode".into(), m.clone());
                    }
                }
                if let Some(p) = ev.get("provider") {
                    if !p.is_null() {
                        rec.insert("provider".into(), p.clone());
                    }
                }
                if let Some(c) = ev.get("cached") {
                    if c.as_bool() == Some(true) {
                        rec.insert("cached".into(), Value::Bool(true));
                    }
                }
                if let Some(k) = ev.get("cache_key") {
                    if !k.is_null() {
                        rec.insert("cache_key".into(), k.clone());
                    }
                }
            }
            "counts" => {
                let Some(rec) = by_id.get_mut(&cid) else { continue };
                if let Some(i) = ev.get("input") {
                    if !i.is_null() {
                        rec.insert("input".into(), i.clone());
                    }
                }
                if let Some(o) = ev.get("output") {
                    if !o.is_null() {
                        rec.insert("output".into(), o.clone());
                    }
                }
                if ev.get("parse_error").and_then(|v| v.as_bool()) == Some(true) {
                    rec.insert("parse_error".into(), Value::Bool(true));
                }
                // Issue #225: empty-on-wire flag from counts event.
                if ev.get("empty_response").and_then(|v| v.as_bool()) == Some(true) {
                    rec.insert("empty_response".into(), Value::Bool(true));
                }
                // Issue #232: interrupted-on-wire flag from counts event.
                if ev.get("interrupted").and_then(|v| v.as_bool()) == Some(true) {
                    rec.insert("interrupted".into(), Value::Bool(true));
                }
            }
            "stream_progress" => {
                // One-shot event from `_consume_chat_stream` at the
                // first-token-received moment, carrying TTFT (and TTFR
                // for reasoning models). The `end` event overwrites
                // later (post-finalize ground truth); ordering in the
                // jsonl is begin → stream_progress → end → counts, so
                // the post-end overwrite happens naturally inside the
                // existing `end` branch above.
                //
                // Defensive: every field is `if let Some` — historical
                // logs carried per-chunk token counts on this event,
                // current logs do not, and pre-#237 logs have no
                // stream_progress events at all. All three cases
                // tolerated.
                let Some(rec) = by_id.get_mut(&cid) else { continue };
                if let Some(v) = ev.get("completion_tokens") {
                    if !v.is_null() {
                        rec.insert("completion_tokens".into(), v.clone());
                    }
                }
                if let Some(v) = ev.get("reasoning_tokens") {
                    if !v.is_null() {
                        rec.insert("reasoning_tokens".into(), v.clone());
                    }
                }
                if let Some(v) = ev.get("content_tokens") {
                    if !v.is_null() {
                        rec.insert("content_tokens".into(), v.clone());
                    }
                }
                for f in ["ttft_ms", "ttfr_ms", "last_token_ms"] {
                    if let Some(v) = ev.get(f) {
                        if !v.is_null() {
                            rec.insert(f.into(), v.clone());
                        }
                    }
                }
            }
            "full_io" => {
                // Legacy fallback (issue #195): pre-split runs stored
                // full_io inline in llm-calls.jsonl. Newer runs put
                // them in the sibling llm-payloads.jsonl read below.
                let Some(rec) = by_id.get_mut(&cid) else { continue };
                if let Some(p) = ev.get("full_prompt") {
                    rec.insert("full_prompt".into(), p.clone());
                }
                if let Some(r) = ev.get("full_response") {
                    rec.insert("full_response".into(), r.clone());
                }
            }
            _ => {} // unknown event — skip
        }
    }

    // ── Sibling payload stream (issue #195) ───────────────────────────
    // llm-payloads.jsonl carries the dev-tab full prompt + response
    // records, split out from llm-calls.jsonl so the metadata file
    // stays small. Off-by-default toggle: missing file is the common
    // case. Each record keys on call_id; threads onto the matching
    // already-built record so the Run Details modal sees the prompt
    // + response in expandable sub-sections.
    let payloads = run_dir.join("llm-payloads.jsonl");
    if let Ok(ptext) = std::fs::read_to_string(&payloads) {
        for line in ptext.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(ev) = serde_json::from_str::<Value>(line) else { continue };
            let cid = match ev.get("call_id").and_then(|v| v.as_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let Some(rec) = by_id.get_mut(&cid) else { continue };
            if let Some(p) = ev.get("full_prompt") {
                rec.insert("full_prompt".into(), p.clone());
            }
            if let Some(r) = ev.get("full_response") {
                rec.insert("full_response".into(), r.clone());
            }
        }
    }

    // Decide how to interpret unmatched begins (begin events without a
    // matching end). Three states, discriminated per-call:
    //
    //   - PENDING: the call's cycle is still alive. begin.cycle_seq ==
    //     latest_cycle_seq AND the latest cycle has no terminator yet.
    //     The end event is coming; render neutral.
    //   - INCOMPLETE (superseded): begin.cycle_seq < latest_cycle_seq.
    //     A later cycle_start fired, so this call's cycle was paused
    //     and the work was redone (or is being redone) in the new
    //     cycle. The call's record is just an aborted artifact of
    //     the previous cycle.
    //   - INCOMPLETE (terminated): begin.cycle_seq == latest_cycle_seq
    //     AND the latest cycle has a terminator (cycle_end /
    //     cycle_cancelled / cycle_error) OR paused.flag is on disk.
    //     The run wound down with this call mid-flight; nothing will
    //     finish it.
    //
    // The reason the run wound down (paused / cancelled / errored)
    // lives on the run-level status, not on the call. From the call's
    // POV "aborted" covers all three of those.
    //
    // s9e3 traced both bugs at this site: pre-#206 the synthetic
    // Cancelled error rendered as "failed"; the cycle_start-resets-
    // ended_at_iso fix then over-corrected to mark cycle-1 orphans as
    // pending until the run completed. Per-call cycle_seq tracking
    // gives both states their own classification.
    let latest_cycle_seq = current_cycle_seq;
    let latest_cycle_ended = ended_at_iso.is_some()
        || run_dir.join("paused.flag").exists();
    let now_iso = iso_z_full();
    let end_anchor = ended_at_iso.as_deref().unwrap_or(&now_iso);

    // Issue #333: per-call user-skip marker scan. Tauri's `skip_call`
    // writes `<run_dir>/skipped_calls/<call_id>` atomically; this
    // pass stamps `skipped=true` on every matching rec and nulls the
    // error payload so the classifier branch below produces "skipped"
    // distinct from any failure bucket. Mirrors Python's same-pass
    // logic in `_materialize_calls_from_jsonl`. Done BEFORE the
    // pending/aborted assignment loop so a skip wins over both
    // (the user signal takes priority over the run-wind-down side
    // effect).
    let skipped_dir = run_dir.join("skipped_calls");
    if let Ok(entries) = std::fs::read_dir(&skipped_dir) {
        for entry in entries.flatten() {
            let cid = entry.file_name().to_string_lossy().to_string();
            if let Some(rec) = by_id.get_mut(&cid) {
                rec.insert("skipped".into(), Value::Bool(true));
                rec.insert("error".into(), Value::Null);
                // If the wrapper finalized the call before the skip
                // landed (rare race), success may be False — leave it
                // so the rec stays internally consistent, but the
                // outcome classifier reads `skipped` and produces
                // "skipped" regardless.
            }
        }
    }
    for cid in &order {
        let Some(rec) = by_id.get_mut(cid) else { continue };
        if rec.get("success").map(|v| v.is_null()).unwrap_or(true) {
            // Always stamp duration_ms from the begin's started_at to
            // the anchor (terminator ts or now) — the modal's per-call
            // table reads this so the user can see how long the call
            // has been waiting / had been waiting before its cycle
            // wound down.
            let started = rec
                .get("started_at_iso")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let dur = iso_delta_ms(started, end_anchor).unwrap_or(0).max(0) as u64;
            rec.insert("duration_ms".into(), Value::Number(dur.into()));
            // Issue #333: if the marker-dir scan already stamped this
            // rec as skipped, the user-skip signal beats the wind-down
            // side effect — skip the aborted/pending assignment.
            let already_skipped = rec
                .get("skipped")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !already_skipped {
                let begin_cycle_seq = rec
                    .get("cycle_seq")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(latest_cycle_seq);
                let superseded = begin_cycle_seq < latest_cycle_seq;
                if superseded || latest_cycle_ended {
                    // INCOMPLETE: leave success=null and error=null so
                    // the JSX renders a neutral pill. The call itself
                    // didn't fail — synthesizing an error would lie
                    // about its outcome.
                    rec.insert("aborted".into(), Value::Bool(true));
                } else {
                    // PENDING: call is in the latest cycle and that
                    // cycle is still alive. End event is on its way.
                    rec.insert("pending".into(), Value::Bool(true));
                }
            }
        }
        // Outcome classification — mirrors Python's
        // `runner.py::_classify_outcome` + `_failure_class_for_label`
        // exactly. Live view MUST match the post-rollup label so the
        // user sees the same bucket in the in-flight pane and the
        // end-of-run rollup. Vocabulary follows the retry-policy
        // taxonomy: failure outcomes carry a parenthetical class tag
        // (`(load)` / `(sizing)` / `(other)`); specific names
        // (cap_hit / timeout / parse_error / empty_response /
        // interrupted) replace `failed` but keep the parenthetical.
        //
        // Two Rust-only outcomes (Python's post-run materializer
        // doesn't produce them — by the time it runs every call has
        // finalized):
        //   pending  → live in-flight call.
        //   aborted  → begin landed but end never did and the run
        //              wound down (paused / cancelled / errored /
        //              superseded by resume). Distinct from any
        //              failure bucket — the call itself didn't error.
        let pending = rec.get("pending").and_then(|v| v.as_bool()).unwrap_or(false);
        let aborted = rec
            .get("aborted")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        // Issue #333: skipped flag stamped above by the marker-dir
        // scan. Takes priority over pending AND aborted — a user
        // skip is an explicit human signal, not a wind-down side
        // effect.
        let skipped = rec
            .get("skipped")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let success = rec.get("success").and_then(|v| v.as_bool()).unwrap_or(false);
        let parse_error = rec
            .get("parse_error")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let empty_response = rec
            .get("empty_response")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let interrupted = rec
            .get("interrupted")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let finish_reason = rec
            .get("finish_reason")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let ttft_set = rec
            .get("ttft_ms")
            .map(|v| !v.is_null())
            .unwrap_or(false);

        // Substring-based shape detectors on the error dict — mirror
        // Python's `_is_timeout_error` and the midstream-class set
        // in `_failure_class_for_label`.
        let err_obj = rec.get("error").and_then(|v| v.as_object());
        let err_cls = err_obj
            .and_then(|e| e.get("class"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let err_msg = err_obj
            .and_then(|e| e.get("message"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let err_cls_lc = err_cls.to_lowercase();
        let err_msg_lc = err_msg.to_lowercase();
        let is_timeout_err = !err_cls.is_empty() && (
            err_cls_lc.contains("timeout")
            || err_msg_lc.contains("timeout")
            || err_msg_lc.contains("timed out")
        );
        // Issue #360: HTTP transport stream-cuts always route to Load,
        // never through the TTFT discriminator. Substring-match on the
        // class name so the qualified `httpx.RemoteProtocolError` shape
        // works. `ConnectTimeout` is included here despite its
        // "timeout" substring overlap because it's a connect-side
        // failure (no tokens ever flowed), not a stream-with-tokens
        // timeout. `ReadTimeout` is NOT in this set — it keeps the
        // TTFT discriminator per the spec's surviving "timeout with
        // tokens" carve-out.
        let is_stream_cut_err = !err_cls.is_empty() && (
            err_cls.contains("RemoteProtocolError")
            || err_cls.contains("ReadError")
            || err_cls.contains("ConnectError")
            || err_cls.contains("ConnectTimeout")
        );

        // TTFT-fraction rule: no tokens → load; tokens with
        // TTFT < 50% of duration → sizing; tokens with TTFT ≥ 50% of
        // duration → load. Missing / nonsensical duration falls back
        // to sizing-when-tokens-flowed. Mirrors Python's
        // `_ttft_fraction_class_for_rec` in runner.py.
        let duration_ms_i = rec
            .get("duration_ms")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let ttft_ms_i = rec.get("ttft_ms").and_then(|v| v.as_i64());
        let ttft_fraction_class: &str = if !ttft_set
            || ttft_ms_i.map(|t| t < 0).unwrap_or(true)
        {
            "load"
        } else if duration_ms_i <= 0 {
            "sizing"
        } else {
            let ttft = ttft_ms_i.unwrap_or(0) as f64;
            let dur = duration_ms_i as f64;
            if dur < ttft {
                "sizing"  // broken measurement; defensive
            } else if ttft < dur * 0.5 {
                "sizing"
            } else {
                "load"
            }
        };

        // failure_class_for_label — single source of truth for the
        // parenthetical strategy tag on failure outcomes. Mirrors
        // Python's `_failure_class_for_label` (load / sizing /
        // other). Sizing-vs-load on tokens-flowed failures uses the
        // TTFT-fraction rule above.
        // The kernel stamps each record with its OWN categorized outcome
        // (`llm_status` == LlmStatus.name), persisted onto the end event
        // (#963). When present it is authoritative: read the load/sizing/
        // other bucket straight off it — mirroring Python
        // `_failure_class_for_label` — instead of re-deriving from the
        // exception class. This is the path that lets a no-exception
        // failure (kernel `from_status`: success=false, error=None — e.g.
        // an injected or real LOAD that never raised) bucket as `load`
        // rather than collapsing to the `other` catch-all at the bottom of
        // the heuristic chain. Pre-kernel archived runs carry no
        // `llm_status` and fall through to the error-shape heuristics below.
        let kernel_status = rec
            .get("llm_status")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let failure_class: &str = if !kernel_status.is_empty() {
            // Mirrors Python `_KERNEL_BUCKET.get(status, "other")`.
            match kernel_status {
                "LOAD" => "load",
                "CAP_HIT" => "sizing",
                "PARSE_ERROR" => "sizing",
                "TIMEOUT_WITH_TOKENS" => "sizing",
                _ => "other",
            }
        } else if err_cls.ends_with("_CapHitResponse") {
            "sizing"
        } else if err_cls.ends_with("_ParseError") {
            "sizing"
        } else if err_cls.ends_with("_EmptyResponse") {
            "load"
        } else if err_cls.ends_with("_InterruptedResponse") {
            // Interrupted is application-level "stream cut with
            // tokens" — same spec category as transport-level
            // RemoteProtocolError; routes to Load unconditionally,
            // no TTFT discriminator.
            "load"
        } else if err_cls.ends_with("_SuccessEmpty") {
            // Parsed-empty: stream completed cleanly, JSON parsed, but
            // the structured output had zero entries. #616 routes
            // through Other for one retry (reasoning forced off if
            // the parent had it on).
            "other"
        } else if err_cls.ends_with("_ErraticResponse") {
            // Compat shim for runs archived before the umbrella
            // exception was split into the three classes above.
            // The kind is recovered from the rec flag / message.
            if parse_error || err_msg.contains("parse_error") {
                "sizing"
            } else if empty_response || err_msg.contains("empty_response") {
                "load"
            } else if interrupted || err_msg.contains("interrupted") {
                "load"
            } else {
                "sizing"  // defensive default
            }
        } else if err_cls.ends_with("_WorkloadNeeded") {
            // Compat shim for runs archived before the sizing-failure
            // synthesis folded into `_CapHitResponse`. Always sizing.
            "sizing"
        } else if is_stream_cut_err {
            // HTTP transport stream-cuts always Load. Listed before
            // the timeout branch because `ConnectTimeout` matches
            // both this set AND `is_timeout_err` via the lowercase-
            // "timeout" substring; ordering first ensures it routes
            // to Load via the stream-cut path instead of the TTFT
            // discriminator.
            "load"
        } else if is_timeout_err {
            ttft_fraction_class
        } else if err_cls.contains("RateLimitError")
            || err_cls.contains("InternalServerError")
            || err_cls.contains("APIConnectionError")
            || err_cls.contains("AttestationFlake")
            || err_cls.contains("TinfoilRouterUnavailable")
            // Bare `builtins.*` network errors are retriable→load in Python's
            // `retry._RETRIABLE_EXC_NAMES`; mirror them here so the live label
            // matches the retry classification instead of falling through to
            // "other". (`builtins.TimeoutError` already routes via the
            // "timeout" substring in `is_timeout_err` above.)
            || err_cls.contains("ConnectionError")
            || err_cls.contains("ConnectionResetError")
            || err_cls.contains("ConnectionAbortedError")
        {
            // Retriable HTTP shapes: 429 / 5xx / network / attestation
            // flake. Plus Tinfoil's ATC-discovery-empty wrap. All
            // load — provider is overloaded.
            "load"
        } else if err_cls.contains("APIStatusError") {
            // Status code isn't on the rec error dict; default to
            // other (non-5xx is the common bubble-up case).
            "other"
        } else {
            "other"
        };

        // Zero-size check (success-path bucketing). Mirrors Python
        // `llm._output_is_zero_size`: the producing stage records its
        // primary entry count as the FIRST integer leaf in `output`;
        // subsequent integers (capacity limits like `total_cap` /
        // `max_actions_cap`, kind tallies) are auxiliary. Summing every
        // integer would inflate the count past zero on stages whose
        // output carries cap fields beside the entry count (insights,
        // actions) and misroute a clean empty result through `success`
        // instead of `success_empty`. Booleans are skipped — serde_json
        // surfaces them via `as_i64()` as None already, but a JSON
        // `true`/`false` is never a primary count. Relies on the
        // `preserve_order` serde_json feature so the first-seen key
        // survives the jsonl round-trip.
        let output_is_zero_size: bool = {
            let out = rec.get("output").and_then(|v| v.as_object());
            let mut zero = false;
            if let Some(o) = out {
                for v in o.values() {
                    if v.is_boolean() {
                        continue;
                    }
                    if let Some(n) = v.as_i64() {
                        zero = n == 0;
                        break;
                    }
                }
            }
            zero
        };
        let has_err = err_obj.is_some();

        let outcome_owned: String = if skipped {
            // Issue #333: explicit user-skip. Priority over pending
            // and aborted — the user signal beats the run-wind-down
            // side effect. The marker-dir scan above also nulled the
            // error payload, so there's nothing to classify here.
            "skipped".to_string()
        } else if pending {
            "pending".to_string()
        } else if aborted {
            // Begin without end on a wound-down run — the call itself
            // didn't error (no error class to classify). Distinct
            // bucket so it doesn't pollute the failure tallies.
            "aborted".to_string()
        } else if has_err || !success {
            // Specific names take priority over the generic
            // `failed (X)`. Order matches Python:
            //   1. cap_hit (finish_reason=length OR _CapHitResponse)
            //   2. parse_error (rec flag OR err.message marker)
            //   3. empty_response (rec flag OR err.message marker)
            //   4. interrupted (rec flag OR err.message marker)
            //   5. timeout (sizing | load via TTFT-fraction rule)
            //   6. failed (load | sizing | other)
            if err_cls.ends_with("_CapHitResponse")
                || finish_reason == "length"
            {
                "cap_hit (sizing)".to_string()
            } else if parse_error || err_msg.contains("parse_error") {
                "parse_error (sizing)".to_string()
            } else if empty_response
                || err_msg.contains("empty_response")
            {
                "empty_response (load)".to_string()
            } else if interrupted
                || err_msg.contains("interrupted")
            {
                if failure_class == "sizing" {
                    "interrupted (sizing)".to_string()
                } else {
                    "interrupted (load)".to_string()
                }
            } else if err_cls.ends_with("_SuccessEmpty")
                || err_msg.contains("success_empty")
            {
                // Parsed-empty failure shape (#616). Surfaces as the
                // dedicated `success_empty` bucket — never the generic
                // `failed (other)`. Matches Python `_classify_outcome`
                // order: success_empty check sits between interrupted
                // and timeout so the routing reads top-down.
                "success_empty".to_string()
            } else if is_timeout_err {
                if failure_class == "sizing" {
                    "timeout (sizing)".to_string()
                } else {
                    "timeout (load)".to_string()
                }
            } else {
                format!("failed ({})", failure_class)
            }
        } else if finish_reason == "length" {
            // Defensive: cap-hit-on-success can't occur post-refactor
            // (wrapper raises _CapHitResponse), but mirror Python's
            // belt-and-braces.
            "cap_hit (sizing)".to_string()
        } else if parse_error {
            "parse_error (sizing)".to_string()
        } else if empty_response {
            "empty_response (load)".to_string()
        } else if output_is_zero_size {
            // Clean valid empty (`[]` or equivalent) — a non-empty
            // output dict whose primary count is 0. An absent or empty
            // output dict leaves `output_is_zero_size` false and falls
            // through to plain success. The zero-content-token failure
            // shapes (empty_response / interrupted / parse_error) have
            // their own labels; a parsed empty payload is a legitimate
            // answer regardless of reasoning_tokens spent arriving at it.
            "success_empty".to_string()
        } else {
            "success".to_string()
        };
        rec.insert("outcome".into(), Value::String(outcome_owned));
    }

    // ── Per-stage roll-up ──────────────────────────────────────────────
    // Match the post-run rollup's per_stage shape: the JSX reads
    // calls_total, calls_failed, calls_aborted, calls_cached,
    // outcomes, models_used, duration_ms.median, ttft_ms.median,
    // reasoning_tokens.total.
    let mut per_stage: BTreeMap<String, serde_json::Map<String, Value>> = BTreeMap::new();
    let mut stage_dur: BTreeMap<String, Vec<i64>> = BTreeMap::new();
    let mut stage_ttft: BTreeMap<String, Vec<i64>> = BTreeMap::new();
    let mut stage_rt_sum: BTreeMap<String, i64> = BTreeMap::new();
    let mut stage_models: BTreeMap<String, Vec<String>> = BTreeMap::new();

    // Outcome buckets the per-stage rollup pre-initializes to 0.
    // Mirrors Python's `_outcome_keys` in runner.py exactly so the
    // live in-flight view's per-stage column counts match the
    // end-of-run rollup. `pending` is Rust-only (Python's post-run
    // materializer doesn't produce it). `success_sampled` and
    // `success_reasoning_off` come from the chain-aware re-bucket
    // walk below. Vocabulary follows the retry-policy taxonomy
    // (load / sizing / other parenthetical on every failure outcome).
    let outcome_keys = [
        "success", "pending", "aborted", "skipped", "success_empty",
        "cap_hit (sizing)",
        "timeout (sizing)", "timeout (load)",
        "parse_error (sizing)", "empty_response (load)",
        "interrupted (sizing)", "interrupted (load)",
        "failed (load)", "failed (sizing)", "failed (other)",
        "success_sampled", "success_reasoning_off",
    ];

    // ── Chain-aware re-bucket: success leaves whose chain has any
    // /sample-N call → "sampled". Mirrors Python's
    // _apply_chain_aware_outcomes. Walked AFTER per-record outcome
    // classification (already in `rec.outcome` from the loop above)
    // and BEFORE the per-stage tally below — same order as the
    // Python rollup. A leaf is a record whose call_id isn't in any
    // other record's retry_of_call_id.
    let mut child_of: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for cid in &order {
        if let Some(rec) = by_id.get(cid) {
            if let Some(rof) = rec
                .get("retry_of_call_id")
                .and_then(|v| v.as_str())
            {
                child_of.insert(rof.to_string());
            }
        }
    }
    // Walk each leaf's chain via retry_of_call_id back to the root.
    // If any record on the path (including the leaf itself) has a
    // category containing "/sample-", re-bucket the leaf's outcome
    // from "success" to "sampled". Defensive seen-set guards
    // against pathological cycles.
    let leaf_ids: Vec<String> = order
        .iter()
        .filter(|cid| !child_of.contains(*cid))
        .cloned()
        .collect();
    for leaf_id in &leaf_ids {
        // Snapshot leaf outcome.
        let cur_outcome = by_id
            .get(leaf_id)
            .and_then(|r| r.get("outcome"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if cur_outcome != "success" {
            continue;
        }
        // Walk ancestors collecting categories. Mirrors Python's
        // `_apply_chain_aware_outcomes`: three sizing-chain paths
        // get re-bucketed when the chain went through one:
        //   • `/sample-N - retry/sizing`     → success_sampled
        //   • `/reasoning-off - retry/sizing` → success_reasoning_off
        //   • `/half-N - retry/sizing` only   → stays clean success
        // sample-N outranks reasoning-off (the bigger caveat —
        // input was reduced, not just compute path).
        let mut had_sample = false;
        let mut had_reasoning_off = false;
        let mut seen: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut cur_id = leaf_id.clone();
        for _ in 0..64 {  // depth cap (defensive)
            if !seen.insert(cur_id.clone()) {
                break;
            }
            if let Some(rec) = by_id.get(&cur_id) {
                if let Some(cat) = rec
                    .get("category")
                    .and_then(|v| v.as_str())
                {
                    if cat.contains("/sample-") {
                        had_sample = true;
                    } else if cat.contains("/reasoning-off") {
                        had_reasoning_off = true;
                    }
                }
                let next = rec
                    .get("retry_of_call_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                match next {
                    Some(p) => cur_id = p,
                    None => break,
                }
            } else {
                break;
            }
        }
        let new_outcome = if had_sample {
            Some("success_sampled")
        } else if had_reasoning_off {
            Some("success_reasoning_off")
        } else {
            None
        };
        if let Some(label) = new_outcome {
            if let Some(rec) = by_id.get_mut(leaf_id) {
                rec.insert(
                    "outcome".into(),
                    Value::String(label.into()),
                );
            }
        }
    }

    let total_calls = order.len();
    let mut total_cached = 0u64;
    let mut total_failed = 0u64;
    let mut total_aborted = 0u64;
    let mut total_pending = 0u64;
    // Issue #333: per-call user-skip counter, surfaced as
    // `totals.skipped` alongside `aborted` / `pending` / `failed`.
    let mut total_skipped = 0u64;

    for cid in &order {
        let Some(rec) = by_id.get(cid) else { continue };
        let stage = rec
            .get("stage")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let bucket = per_stage.entry(stage.clone()).or_insert_with(|| {
            let mut m = serde_json::Map::new();
            m.insert("name".into(), Value::String(stage.clone()));
            m.insert("calls_total".into(), Value::Number(0u64.into()));
            m.insert("calls_failed".into(), Value::Number(0u64.into()));
            m.insert("calls_aborted".into(), Value::Number(0u64.into()));
            m.insert("calls_skipped".into(), Value::Number(0u64.into()));
            m.insert("calls_pending".into(), Value::Number(0u64.into()));
            m.insert("calls_cached".into(), Value::Number(0u64.into()));
            let outcomes = serde_json::Map::from_iter(
                outcome_keys.iter().map(|k| {
                    (k.to_string(), Value::Number(0u64.into()))
                }),
            );
            m.insert("outcomes".into(), Value::Object(outcomes));
            m
        });
        let inc = |bucket: &mut serde_json::Map<String, Value>, key: &str| {
            let cur = bucket
                .get(key)
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            bucket.insert(key.into(), Value::Number((cur + 1).into()));
        };
        inc(bucket, "calls_total");
        let pending = rec.get("pending").and_then(|v| v.as_bool()).unwrap_or(false);
        let aborted = rec
            .get("aborted")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let skipped = rec
            .get("skipped")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let success = rec.get("success").and_then(|v| v.as_bool()).unwrap_or(false);
        let cached = rec.get("cached").and_then(|v| v.as_bool()).unwrap_or(false);
        if skipped {
            // Issue #333: skip wins over every other bucket — the
            // user's explicit ✕ click is the truth about this row.
            inc(bucket, "calls_skipped");
            total_skipped += 1;
        } else if pending {
            inc(bucket, "calls_pending");
            total_pending += 1;
        } else if aborted {
            inc(bucket, "calls_aborted");
            total_aborted += 1;
        } else if !success {
            inc(bucket, "calls_failed");
            total_failed += 1;
        }
        if cached {
            inc(bucket, "calls_cached");
            total_cached += 1;
        }
        // Outcome bucketing — per-stage. Defensive fallback uses the
        // post-#190 generic-other label (same as Python's classifier
        // when the failure-shape discriminator can't pin a more
        // specific bucket). Pre-#190 string was "other_failure" but
        // the new vocabulary is `failed (other)`.
        let outcome = rec
            .get("outcome")
            .and_then(|v| v.as_str())
            .unwrap_or("failed (other)");
        if let Some(ob) = bucket.get_mut("outcomes").and_then(|v| v.as_object_mut()) {
            let cur = ob.get(outcome).and_then(|v| v.as_u64()).unwrap_or(0);
            ob.insert(outcome.into(), Value::Number((cur + 1).into()));
        }
        // Distribution accumulators (medians computed below).
        if let Some(d) = rec.get("duration_ms").and_then(|v| v.as_i64()) {
            stage_dur.entry(stage.clone()).or_default().push(d);
        }
        if let Some(t) = rec.get("ttft_ms").and_then(|v| v.as_i64()) {
            stage_ttft.entry(stage.clone()).or_default().push(t);
        }
        if let Some(r) = rec.get("reasoning_tokens").and_then(|v| v.as_i64()) {
            *stage_rt_sum.entry(stage.clone()).or_insert(0) += r;
        }
        if let Some(m) = rec.get("model").and_then(|v| v.as_str()) {
            let v = stage_models.entry(stage.clone()).or_default();
            if !v.iter().any(|x| x == m) {
                v.push(m.to_string());
            }
        }
    }

    fn median_i64(mut xs: Vec<i64>) -> Option<i64> {
        if xs.is_empty() {
            return None;
        }
        xs.sort_unstable();
        let n = xs.len();
        Some(if n % 2 == 1 {
            xs[n / 2]
        } else {
            (xs[n / 2 - 1] + xs[n / 2]) / 2
        })
    }

    for (stage, bucket) in per_stage.iter_mut() {
        let dur_med = median_i64(stage_dur.remove(stage).unwrap_or_default());
        let ttft_med = median_i64(stage_ttft.remove(stage).unwrap_or_default());
        let rt_total = stage_rt_sum.remove(stage).unwrap_or(0);
        let mut dur_obj = serde_json::Map::new();
        dur_obj.insert(
            "median".into(),
            dur_med.map(|n| Value::Number(n.into())).unwrap_or(Value::Null),
        );
        bucket.insert("duration_ms".into(), Value::Object(dur_obj));
        let mut ttft_obj = serde_json::Map::new();
        ttft_obj.insert(
            "median".into(),
            ttft_med.map(|n| Value::Number(n.into())).unwrap_or(Value::Null),
        );
        bucket.insert("ttft_ms".into(), Value::Object(ttft_obj));
        let mut rt_obj = serde_json::Map::new();
        rt_obj.insert(
            "total".into(),
            if rt_total > 0 {
                Value::Number(rt_total.into())
            } else {
                Value::Null
            },
        );
        bucket.insert("reasoning_tokens".into(), Value::Object(rt_obj));
        let models = stage_models.remove(stage).unwrap_or_default();
        bucket.insert(
            "models_used".into(),
            Value::Array(models.into_iter().map(Value::String).collect()),
        );
    }

    // Issue #105 v3 follow-up: leaf-only warnings rollup, mirrors
    // Python's _merge_run_warnings. The user sees one number on the
    // run-row banner ("X calls flagged"); it MUST be leaf-only or
    // recovered chains inflate the count. Pre-fix the live view
    // showed all-attempts counts during the run and leaf-only after
    // the rollup — exactly the inconsistency the user was angry
    // about. Now both paths produce the same numbers.
    // Vocabulary follows the retry-policy taxonomy — match by
    // prefix so e.g. both `timeout (sizing)` and `timeout (load)`
    // count toward `timeouts`. Mirrors Python's
    // `_leaf_aware_warnings` (`_sum_prefix`).
    let mut leaf_timeouts = 0u64;
    let mut leaf_success_empty = 0u64;
    let mut leaf_parse_errors = 0u64;
    let mut leaf_empty_responses = 0u64;
    let mut leaf_interrupted = 0u64;
    let mut leaf_sampled = 0u64;
    let mut leaf_reasoning_off = 0u64;
    let mut leaf_cap_hits = 0u64;
    let mut leaf_failed = 0u64;
    for leaf_id in &leaf_ids {
        let Some(rec) = by_id.get(leaf_id) else { continue };
        let oc = rec
            .get("outcome")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if oc.starts_with("timeout") {
            leaf_timeouts += 1;
        } else if oc.starts_with("cap_hit") {
            leaf_cap_hits += 1;
        } else if oc.starts_with("parse_error") {
            leaf_parse_errors += 1;
        } else if oc.starts_with("empty_response") {
            leaf_empty_responses += 1;
        } else if oc.starts_with("interrupted") {
            leaf_interrupted += 1;
        } else if oc.starts_with("failed") {
            leaf_failed += 1;
        } else if oc == "success_empty" {
            leaf_success_empty += 1;
        } else if oc == "success_sampled" {
            leaf_sampled += 1;
        } else if oc == "success_reasoning_off" {
            leaf_reasoning_off += 1;
        }
    }

    let calls_arr: Vec<Value> = order
        .into_iter()
        .map(|cid| Value::Object(by_id.remove(&cid).unwrap()))
        .collect();
    let per_stage_obj: serde_json::Map<String, Value> = per_stage
        .into_iter()
        .map(|(k, v)| (k, Value::Object(v)))
        .collect();

    let mut payload = serde_json::Map::new();
    payload.insert(
        "schema".into(),
        Value::String("llm-stats/v1-inflight".into()),
    );
    payload.insert(
        "in_flight".into(),
        Value::Bool(true),
    );
    payload.insert(
        "totals".into(),
        json!({
            "calls": total_calls,
            // `successful` excludes pending, aborted, AND skipped —
            // none of those resolved cleanly to a success bucket.
            "successful": total_calls.saturating_sub(
                (total_failed + total_aborted + total_pending
                 + total_skipped) as usize),
            "failed": total_failed,
            "aborted": total_aborted,
            "skipped": total_skipped,
            "pending": total_pending,
        }),
    );
    payload.insert("per_stage".into(), Value::Object(per_stage_obj));
    payload.insert("calls".into(), Value::Array(calls_arr));
    payload.insert("totals_cached".into(), Value::Number(total_cached.into()));
    // Warnings block — mirrors the Python rollup's `warnings` field
    // 1:1 (issue #189). All buckets count LEAVES ONLY so chains that
    // recovered via halving don't inflate the badge. The banner UI
    // sums all keys and renders one "X calls flagged" total.
    //
    // input_overflows stays at 0: it's an in-memory `_call_warnings`
    // counter in Python (not persisted to jsonl). The Rust
    // materializer can't recover it; the Python rollup at end-of-run
    // can only because it runs in the same process. Post-#189 even
    // the Python on-demand readers (eval/judge) emit 0 — accepted
    // limitation, see issue #189 for context.
    //
    // empty_responses + interrupted both count from the per-call
    // outcome buckets now, no _call_warnings dependency.
    payload.insert(
        "warnings".into(),
        json!({
            "cap_hits": leaf_cap_hits,
            "empty_responses": leaf_empty_responses,
            "interrupted": leaf_interrupted,
            "input_overflows": 0,
            "timeouts": leaf_timeouts,
            "success_empty": leaf_success_empty,
            "parse_errors": leaf_parse_errors,
            "sampled": leaf_sampled,
            "reasoning_off": leaf_reasoning_off,
            "failed": leaf_failed,
        }),
    );

    Value::Object(payload)
}

// Cache for the on-demand materializer — keyed on jsonl mtime. Reads
// on an unchanged jsonl return the cached payload in O(1); a write
// invalidates by bumping the mtime. Issue #189 — llm-stats.json no
// longer exists on disk, so list_runs / read_run_llm_stats /
// overlay_rollup_warnings_and_cache all funnel through here on every
// call. Without the cache, a 1k-call jsonl gets re-parsed for every
// 500ms list_runs poll.
type MaterializeCacheKey = Option<std::time::SystemTime>;
static MATERIALIZE_CACHE: std::sync::OnceLock<
    Mutex<std::collections::HashMap<PathBuf, (MaterializeCacheKey, serde_json::Value)>>,
> = std::sync::OnceLock::new();

fn materialize_cache() -> &'static Mutex<
    std::collections::HashMap<PathBuf, (MaterializeCacheKey, serde_json::Value)>,
> {
    MATERIALIZE_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// mtime-cached wrapper around `materialize_llm_stats_from_jsonl`.
/// Invalidates when the jsonl is rewritten or appended to. Issue #189.
fn materialize_llm_stats_cached(run_dir: &Path) -> serde_json::Value {
    let jsonl = run_dir.join("llm-calls.jsonl");
    let key: MaterializeCacheKey = std::fs::metadata(&jsonl)
        .and_then(|m| m.modified()).ok();
    {
        let cache = materialize_cache().lock().unwrap();
        if let Some((cached_key, cached)) = cache.get(run_dir) {
            if &key == cached_key {
                return cached.clone();
            }
        }
    }
    let val = materialize_llm_stats_from_jsonl(run_dir);
    let mut cache = materialize_cache().lock().unwrap();
    cache.insert(run_dir.to_path_buf(), (key, val.clone()));
    val
}

#[cfg(test)]
fn clear_materialize_cache_for_tests() {
    let mut cache = materialize_cache().lock().unwrap();
    cache.clear();
}

// ── Live token overlay ──────────────────────────────────────────────────────
//
// Streaming LLM calls write a one-shot `stream_progress` JSONL event at
// first-token (TTFT marker, durable) and additionally print a
// rate-limited `live_tokens` heartbeat to STDOUT every ~1s per call.
// The heartbeat carries running chars/3 estimates of completion_tokens
// / reasoning_tokens / content_tokens / last_token_ms.
//
// jsonl deliberately doesn't get these heartbeats — they're ephemeral
// "this call is still streaming, here's its current count" signal, not
// part of the durable record. spawn_pipeline's stdout reader catches
// each heartbeat into this in-memory map. read_run_llm_stats overlays
// the live state on top of the (jsonl-materialized) in-flight call rec
// so the per-call detail UI shows moving counts between begin and end
// without paying a jsonl write per second per call.
//
// Eviction: per-call eviction does NOT happen on the stdout-event path.
// Python's `finalize_stat_record` writes the `end` event to
// `llm-calls.jsonl` only — there's no corresponding stdout print, so
// the stdout reader never sees an end signal carrying a call_id. The
// correctness guarantee instead comes from `overlay_live_token_state`
// skipping any rec whose `success` is non-null (the materializer sets
// it from the jsonl `end` event), so stale live state for a
// now-terminated call is harmlessly ignored. `clear_live_state_for_run`
// drops the run's whole entry when `spawn_pipeline` exits. Net memory
// is O(total LLM calls per run), cleared on run exit — bounded and
// trivial for the call counts the pipeline actually produces.
//
// Cross-session reads (Tauri restart, completed runs viewed days
// later, sweep runs not spawned by this Tauri instance) miss the live
// overlay entirely — the materializer's jsonl walk is the
// authoritative read path for those.
#[derive(Clone, Debug, Default)]
struct LiveTokenState {
    completion_tokens: Option<i64>,
    reasoning_tokens: Option<i64>,
    content_tokens: Option<i64>,
    last_token_ms: Option<i64>,
}

static LIVE_TOKEN_STATE: std::sync::OnceLock<
    Mutex<std::collections::HashMap<String, std::collections::HashMap<String, LiveTokenState>>>,
> = std::sync::OnceLock::new();

fn live_token_state() -> &'static Mutex<
    std::collections::HashMap<String, std::collections::HashMap<String, LiveTokenState>>,
> {
    LIVE_TOKEN_STATE.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Apply one stdout event line to the in-memory live state for `run_id`.
/// No-ops on anything that isn't a `live_tokens` heartbeat (the only
/// stdout event carrying a `call_id` today; `progress_tick` lines are
/// stage-scoped, jsonl-only events like `end` / `counts` are not
/// printed to stdout). Returns silently on parse failure; the caller
/// (`spawn_pipeline`'s stdout loop) only invokes this after the line
/// has already parsed as a JSON object, so a re-parse failure here is
/// not expected — the guard just keeps the helper independently safe.
/// Non-object / non-JSON stdout never reaches this path: it is skipped
/// before both the overlay update and the pipeline-progress emit.
fn apply_stdout_event_to_live_state(run_id: &str, line: &str) {
    let Ok(ev) = serde_json::from_str::<serde_json::Value>(line) else { return };
    let event = ev.get("event").and_then(|v| v.as_str()).unwrap_or("");
    if event != "live_tokens" {
        return;
    }
    let Some(call_id) = ev.get("call_id").and_then(|v| v.as_str()) else { return };
    let mut map = live_token_state().lock().unwrap();
    let per_run = map.entry(run_id.to_string()).or_default();
    let entry = per_run.entry(call_id.to_string()).or_default();
    if let Some(v) = ev.get("completion_tokens").and_then(|v| v.as_i64()) {
        entry.completion_tokens = Some(v);
    }
    if let Some(v) = ev.get("reasoning_tokens").and_then(|v| v.as_i64()) {
        entry.reasoning_tokens = Some(v);
    }
    if let Some(v) = ev.get("content_tokens").and_then(|v| v.as_i64()) {
        entry.content_tokens = Some(v);
    }
    if let Some(v) = ev.get("last_token_ms").and_then(|v| v.as_i64()) {
        entry.last_token_ms = Some(v);
    }
}

/// Clear all live overlay state for a run. Called when spawn_pipeline
/// exits (normal completion / crash / cancel) — the runner is no
/// longer writing, so anything still in the map is by definition
/// stale.
fn clear_live_state_for_run(run_id: &str) {
    let mut map = live_token_state().lock().unwrap();
    map.remove(run_id);
}

/// Mutate `calls` in `materialized` to overlay live state onto
/// in-flight records. A record is "in-flight" when its `success` is
/// null — the materializer's `end` branch sets `success` to the bool
/// from the end event, so null means no end event has landed yet.
///
/// Two overlays:
/// 1. **Token counts** from the in-memory live map (fed by stdout
///    heartbeats during streaming). Visible as moving completion /
///    reasoning / content_tokens in the per-call detail UI.
/// 2. **duration_ms** computed at read-time as
///    `now_iso - started_at_iso`. Server-clock based; no Python
///    needs to send it on heartbeats. Each modal refetch re-derives
///    against the current wall clock, so the per-call elapsed
///    ticks at refetch cadence (which the heartbeat drives at ~1Hz).
///
/// Once the end event lands the materializer writes the canonical
/// duration_ms from the wrapper's `time.time() - started_at`
/// measurement; `success` flips to non-null and this overlay
/// stops touching the rec. Idempotent and cheap.
fn overlay_live_token_state(materialized: &mut serde_json::Value, run_id: &str) {
    let now_iso = iso_z_full();
    let map = live_token_state().lock().unwrap();
    let per_run = map.get(run_id);
    let has_live_tokens = per_run.is_some_and(|r| !r.is_empty());
    let Some(calls) = materialized.get_mut("calls").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for c in calls.iter_mut() {
        // Only overlay in-flight records. Once `success` is set
        // (true/false from end event), the jsonl-derived counts are
        // ground truth and the live overlay is stale.
        let in_flight = c.get("success").is_none_or(|v| v.is_null());
        if !in_flight {
            continue;
        }
        // duration_ms = now - started_at_iso for any in-flight rec
        // with a parseable start timestamp. Runs unconditionally
        // (not gated on the live-token map) so calls in the pre-
        // first-token TTFT phase still tick their elapsed.
        let started = c
            .get("started_at_iso")
            .and_then(|v| v.as_str())
            .map(String::from);
        if let Some(start) = started {
            if let Some(ms) = iso_delta_ms(&start, &now_iso) {
                if ms >= 0 {
                    if let Some(obj) = c.as_object_mut() {
                        obj.insert(
                            "duration_ms".into(),
                            serde_json::Value::from(ms as u64),
                        );
                    }
                }
            }
        }
        // Token-count overlay from the in-memory heartbeat map.
        // Skipped when the run has no live state (cross-session
        // reads, sweep runs, runs Tauri didn't spawn) — the rec
        // keeps the materializer's null fields, which the JSX
        // renders as "—".
        if !has_live_tokens {
            continue;
        }
        let Some(cid) = c.get("call_id").and_then(|v| v.as_str()).map(String::from) else {
            continue;
        };
        let Some(live) = per_run.and_then(|r| r.get(&cid)) else { continue };
        let Some(obj) = c.as_object_mut() else { continue };
        if let Some(v) = live.completion_tokens {
            obj.insert("completion_tokens".into(), serde_json::Value::from(v));
        }
        if let Some(v) = live.reasoning_tokens {
            obj.insert("reasoning_tokens".into(), serde_json::Value::from(v));
        }
        if let Some(v) = live.content_tokens {
            obj.insert("content_tokens".into(), serde_json::Value::from(v));
        }
        if let Some(v) = live.last_token_ms {
            obj.insert("last_token_ms".into(), serde_json::Value::from(v));
        }
    }
}


/// Append one cycle event to a run's `llm-calls.jsonl`. Mirrors the
/// Python emitter shape (`{"event": ..., "ts": ..., "schema": "llm-calls/v1", ...}`)
/// so consumers don't need to fork on origin. Used by Rust-side
/// terminate flows (cancel / delete) — pause writes a marker file
/// instead since the runner is already dead.
///
/// Best-effort: a write failure is logged but doesn't propagate. The
/// derivation still works as a no-op for runs whose jsonl has the
/// natural cycle_end (completed) or whose pid is dead (orphan
/// cleanup).
fn append_cycle_event_to_jsonl(
    run_dir: &Path,
    event: &str,
    extras: serde_json::Map<String, serde_json::Value>,
) {
    let jsonl = run_dir.join("llm-calls.jsonl");
    // Don't create the jsonl if the runner never wrote one — the
    // run dir might exist with only run.json (legacy). In that case
    // the cycle event has nowhere to go and the read path falls back
    // to run.json's status field anyway.
    if !jsonl.exists() {
        return;
    }
    let mut payload = serde_json::Map::new();
    payload.insert("event".into(), serde_json::Value::String(event.to_string()));
    payload.insert("ts".into(), serde_json::Value::String(iso_z_full()));
    for (k, v) in extras {
        payload.insert(k, v);
    }
    payload.insert(
        "schema".into(),
        serde_json::Value::String("llm-calls/v1".into()),
    );
    let line = serde_json::to_string(&serde_json::Value::Object(payload))
        .unwrap_or_else(|_| "{}".into());
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(false)
        .append(true)
        .open(&jsonl)
    {
        if let Err(e) = writeln!(f, "{}", line) {
            warn!(
                "append_cycle_event_to_jsonl: write failed for {}: {}",
                jsonl.display(),
                e
            );
        }
    } else {
        warn!(
            "append_cycle_event_to_jsonl: could not open {} for append",
            jsonl.display()
        );
    }
}

/// Terminal statuses are sticky — once set they can't be demoted to a
/// non-terminal one. Protects against the race:
///   pipeline naturally finishes → writes "completed"
///   user clicks pause 10ms later → would flip to "paused"
///   user clicks resume → runner detects everything done, flips to
///       "completed" again. User perceives this as "paused then
///       unpaused on its own."
fn is_terminal_status(s: &str) -> bool {
    matches!(s, "completed" | "failed" | "cancelled")
}

/// Record a Rust-driven status change for a run. Replaces the legacy
/// `run.json` mutation with two complementary mechanisms:
///
///   - "paused": writes `paused.flag` in the run dir (a runtime control
///     marker — pause fires WITHOUT the runner present to emit a jsonl
///     event, so a marker file is the only place to record it).
///   - "cancelled": appends a `cycle_cancelled` event to llm-calls.jsonl.
///   - "failed":    appends a `cycle_error` event to llm-calls.jsonl
///     (with the optional error message in the event payload).
///   - "running" / other: noop (only the runner emits cycle_start, and
///     resume re-runs the runner which handles it).
///
/// Sticky-terminal rule still applies: a request to demote a derived
/// terminal status (completed / failed / cancelled) is silently
/// ignored. Implementation reads the derived status first.
fn set_run_status(
    run_id: &str,
    status: &str,
    error: Option<String>,
) -> Result<(), String> {
    let run_dir = find_run_dir(run_id)
        .ok_or_else(|| format!("run dir not found for run_id {}", run_id))?;

    // Sticky-terminal: read derived status; refuse to demote.
    let derived = derive_run_state(&run_dir);
    if is_terminal_status(&derived.status) && !is_terminal_status(status) {
        info!(
            "set_run_status: run={} keeping terminal {} (ignoring incoming {})",
            run_id, derived.status, status
        );
        return Ok(());
    }
    info!(
        "set_run_status: run={} {} → {}",
        run_id, derived.status, status
    );

    match status {
        "paused" => write_paused_flag(&run_dir, error.as_deref()),
        "cancelled" => {
            // Clear any stale paused.flag — a cancel after pause is a
            // newer terminal that supersedes the marker.
            let _ = std::fs::remove_file(run_dir.join("paused.flag"));
            let mut extras = serde_json::Map::new();
            extras.insert("reason".into(), serde_json::Value::String("user".into()));
            if let Some(e) = error {
                extras.insert("message".into(), serde_json::Value::String(e));
            }
            append_cycle_event_to_jsonl(&run_dir, "cycle_cancelled", extras);
            Ok(())
        }
        "failed" => {
            let _ = std::fs::remove_file(run_dir.join("paused.flag"));
            let mut extras = serde_json::Map::new();
            if let Some(e) = error {
                extras.insert("message".into(), serde_json::Value::String(e));
            }
            append_cycle_event_to_jsonl(&run_dir, "cycle_error", extras);
            Ok(())
        }
        "completed" => {
            // A natural completion comes from the runner emitting
            // cycle_end; Rust shouldn't synthesize one. Treat as no-op.
            Ok(())
        }
        "running" => {
            // Only the runner emits cycle_start. Clearing paused.flag
            // here lets resume_run flip the derived status without
            // racing the runner's first jsonl write.
            let _ = std::fs::remove_file(run_dir.join("paused.flag"));
            Ok(())
        }
        other => {
            warn!("set_run_status: unknown status {:?} for run {} — ignoring", other, run_id);
            Ok(())
        }
    }
}

/// Write `paused.flag` in the run dir to record a paused state. Atomic
/// via tmp+rename so a SIGKILL mid-write can't leave a torn file.
/// Body carries an optional human-readable reason for post-mortem.
fn write_paused_flag(run_dir: &Path, reason: Option<&str>) -> Result<(), String> {
    let p = run_dir.join("paused.flag");
    let tmp = run_dir.join("paused.flag.tmp");
    let body = reason.unwrap_or("paused").to_string() + "\n";
    std::fs::write(&tmp, body)
        .map_err(|e| format!("write tmp paused.flag: {}", e))?;
    std::fs::rename(&tmp, &p)
        .map_err(|e| format!("rename paused.flag: {}", e))?;
    // Touching the jsonl bumps its mtime so the derive-cache invalidates
    // — without this, list_runs would keep reporting the prior derived
    // status until the next jsonl event lands. Best-effort: a stat
    // failure means the run never had a jsonl; the derived status will
    // still pick up paused via the marker check.
    let jsonl = run_dir.join("llm-calls.jsonl");
    if jsonl.exists() {
        // Open + close with append flag bumps mtime atomically without
        // changing content. Alternatively `filetime::set_file_mtime`
        // would be cleaner but adds a dep.
        if let Ok(f) = std::fs::OpenOptions::new().append(true).open(&jsonl) {
            // Write zero bytes — touches the inode but not the file body.
            let _ = f.sync_data();
        }
    }
    Ok(())
}

/// kill(pid, 0) — no-op signal that tests whether the process exists and
/// we have permission to signal it. Returns 0 if alive, -1 otherwise.
/// Not 100% race-free (PID reuse after a process dies can report alive
/// for an unrelated process), but good enough for "is our former child
/// still around" checks at startup.
fn is_pid_alive(pid: u32) -> bool {
    // SAFETY: kill with signal 0 is a pure probe — doesn't modify any
    // process state, just returns an errno.
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    rc == 0
}

/// On startup, any run still in "running" status from a prior app
/// session needs attention. Status is derived from llm-calls.jsonl
/// (cycle_start without a terminator) — pid comes from the cycle_start
/// event payload (post-#165) or the legacy run.json.pid for old runs.
///
/// - If that pid is STILL ALIVE (the app crashed/quit without killing
///   its children, and launchd reparented them), SIGKILL it and mark
///   the run "paused" (filesystem marker). Survivors were the cause
///   of multiple simultaneous Private-Cloud runs after app restart.
/// - If the pid is dead (or wasn't recorded — pre-#165 legacy runs
///   without cycle markers), mark "failed". No resume possible; the
///   checkpoint on disk is what it is.
///
/// Runs with already-terminal status (completed / failed / cancelled /
/// paused) are left alone. "paused" specifically is fine to persist
/// across restarts — that's the whole point of pause.
///
/// `app` is `Some` in production so each recovered orphan drops a run-
/// diagnostics shareable (a crash is a wind-down worth a diagnostic);
/// unit tests pass `None` to skip the spawn.
fn cleanup_orphaned_runs(app: Option<&tauri::AppHandle>) {
    let logs = logs_root();
    if !logs.exists() {
        return;
    }
    for sidecar in walk_run_jsons(&logs) {
        let Some(run_dir) = sidecar.parent() else { continue };
        // Status comes from the derived state walk (jsonl + markers) —
        // post-#165 this is the single source of truth. For pre-cycle-
        // marker runs (very old) derivation returns "unknown" and we
        // fall back to whatever the legacy run.json carries, if any.
        let derived = derive_run_state(run_dir);
        let derived_status: String = if derived.status != "unknown" {
            derived.status.clone()
        } else {
            // Legacy fallback — no cycle markers on disk. Read run.json
            // directly to preserve the pre-#165 cleanup semantics.
            let rj = run_dir.join("run.json");
            if !rj.exists() {
                continue;
            }
            std::fs::read_to_string(&rj)
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                .and_then(|v| {
                    v.get("status")
                        .and_then(|s| s.as_str())
                        .map(|s| s.to_string())
                })
                .unwrap_or_default()
        };
        if derived_status != "running" {
            continue;
        }
        // pid: prefer derive_run_state.pid (extracted from cycle_start
        // event payload, post-#165), fall back to run.json.pid for
        // legacy runs.
        let pid = derived.pid.or_else(|| {
            let rj = run_dir.join("run.json");
            std::fs::read_to_string(&rj)
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                .and_then(|v| {
                    v.get("pid")
                        .and_then(|p| p.as_u64())
                        .and_then(|p| u32::try_from(p).ok())
                })
        });

        let (new_status, new_error) = match pid {
            Some(pid) if is_pid_alive(pid) => {
                // Surviving orphan. Kill it so it can't keep chewing
                // through LLM credits or collide with a fresh run in the
                // same mode.
                unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
                info!(
                    "cleanup_orphaned_runs: SIGKILL pid={} orphan for {}",
                    pid,
                    run_dir.display()
                );
                (
                    "paused",
                    "paused because the app was closed while this run was active — resume to continue",
                )
            }
            _ => {
                info!(
                    "cleanup_orphaned_runs: marking {} failed (process not alive)",
                    run_dir.display()
                );
                ("failed", "app restarted while run was active")
            }
        };

        // Post-#165: derivation reads cycle terminators from the jsonl,
        // not run.json status. Surface the orphan-recovery decision the
        // way derivation will see it on the next walk:
        //   "paused"  → write paused.flag (runtime control marker)
        //   "failed"  → append cycle_error to jsonl
        // This was the missing half of the orphan-recovery path post-
        // #165: the prior implementation only mutated run.json (which
        // derivation now ignores), so orphan runs stayed "running"
        // forever and blocked new same-mode launches. Issue surfaced
        // immediately post-merge: 5 pre-existing orphan jsonls held
        // every mode in the per-mode-gate's grip.
        match new_status {
            "paused" => {
                let _ = write_paused_flag(run_dir, Some(new_error));
            }
            "failed" => {
                let mut extras = serde_json::Map::new();
                extras.insert(
                    "message".into(),
                    serde_json::Value::String(new_error.to_string()),
                );
                append_cycle_event_to_jsonl(run_dir, "cycle_error", extras);
            }
            _ => {}
        }
        // Bust the derive-cache for this run so the next list_runs
        // walk picks up the freshly-appended event without having to
        // wait for an mtime tick (jsonl append IS a write so mtime
        // bumps, but a paused.flag write — which doesn't touch the
        // jsonl — would otherwise rely on the marker-mtime entry in
        // the cache key. Belt-and-braces invalidation is cheap.)
        derive_cache().lock().unwrap().remove(run_dir);

        // Orphan recovery is a wind-down (the run ended when the app
        // died) — drop the run-diagnostics shareable so a crashed run is
        // still debuggable. Skipped in unit tests, which pass `None`.
        if let Some(app) = app {
            if let Some(rid) = run_dir.file_name().and_then(|n| n.to_str()) {
                spawn_run_diagnostic_emit(app, rid);
            }
        }

        // For runs that still carry a legacy run.json (most pre-#165
        // runs, and any new run during the writes-both transition),
        // also keep its status field in sync so a stale read-fallback
        // path doesn't surface the prior "running" status.
        let rj = run_dir.join("run.json");
        if rj.exists() {
            if let Ok(text) = std::fs::read_to_string(&rj) {
                if let Ok(mut val) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(obj) = val.as_object_mut() {
                        obj.insert(
                            "status".into(),
                            serde_json::Value::String(new_status.to_string()),
                        );
                        obj.insert(
                            "error".into(),
                            serde_json::Value::String(new_error.to_string()),
                        );
                        obj.insert(
                            "updated_at".into(),
                            serde_json::Value::String(iso_z_full()),
                        );
                    }
                    let pretty = serde_json::to_string_pretty(&val).unwrap_or_default();
                    let tmp = rj.with_extension("json.tmp");
                    if std::fs::write(&tmp, pretty + "\n").is_ok() {
                        let _ = std::fs::rename(&tmp, &rj);
                    }
                }
            }
        }
    }
}

/// Guards the exit cleanup body: a single real quit can fire both
/// `ExitRequested` and `Exit`, so we run the SIGKILL + mark-paused +
/// diagnostic-emit loop at most once per process (a second pass would
/// double-spawn the emit and race the latest-wins overwrite).
static EXIT_CLEANUP_DONE: AtomicBool = AtomicBool::new(false);

/// Called when the app is about to exit (Cmd+Q, red button on last
/// window, System-triggered quit). Iterates active_runs and for each:
/// SIGKILL the Python subprocess, mark "paused" via paused.flag so the
/// user sees "resume" the next time they launch, and drop the run-
/// diagnostics shareable (app-close is a wind-down worth a diagnostic).
/// This is the graceful path; cleanup_orphaned_runs is the safety net
/// for crashes where this hook didn't run.
fn cleanup_active_runs_on_exit(app: &tauri::AppHandle, state: &AppState) {
    if EXIT_CLEANUP_DONE.swap(true, Ordering::SeqCst) {
        return;
    }
    let entries: Vec<(String, u32)> = {
        let mut guard = state.active_runs.lock().unwrap();
        let snapshot: Vec<(String, u32)> = guard
            .values()
            .map(|a| (a.run_id.clone(), a.pid))
            .collect();
        // Flag each with a terminal override so the spawn-thread's
        // post-wait path doesn't race us to "failed".
        for a in guard.values_mut() {
            a.terminal_override = Some("paused".to_string());
        }
        snapshot
    };

    for (run_id, pid) in entries {
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
        info!("cleanup_active_runs_on_exit: SIGKILL pid={} run={}", pid, run_id);
        let _ = set_run_status(
            &run_id,
            "paused",
            Some(
                "paused because the app was closed mid-run — resume to continue".into(),
            ),
        );
        // App-close is a wind-down — capture the diagnostic. The child is
        // spawned synchronously and outlives this exiting process, so it
        // completes even as the app dies.
        spawn_run_diagnostic_emit(app, &run_id);
    }
}

/// Find the Python interpreter that carries the pipeline + MLX deps.
/// - Release: the bundled sidecar at Resources/python/bin/python3.
/// - Dev: PIPELINE_PYTHON env var wins; otherwise the dev-time sidecar at
///   src-tauri/binaries/python/bin/python3 (populated by
///   `scripts/setup-bundled-python.sh`) so `tauri dev` has the same
///   interpreter + deps as release.
///
/// There is deliberately no system-`python3` fallback. The pipeline and
/// the local MLX inference path both require the bundled 3.14 tree with
/// its pinned deps; a clean Mac's `/usr/bin/python3` (Apple's 3.9.6)
/// lacks every one of them. A build with no bundled runtime is always a
/// broken build, never a degraded-but-usable config — so fail loudly
/// with a reinstall instruction instead of limping into an opaque
/// ImportError traceback downstream.
fn python_bin(app: &tauri::AppHandle) -> Result<String, String> {
    if !cfg!(debug_assertions) {
        if let Ok(resource_dir) = app.path().resource_dir() {
            let bundled = resource_dir.join("python").join("bin").join("python3");
            if bundled.exists() {
                return Ok(bundled.to_string_lossy().into_owned());
            }
        }
        return Err(
            "BaseVault's bundled Python runtime is missing from the app \
             bundle. This is a broken installation — reinstall BaseVault \
             from the original download."
                .to_string(),
        );
    }
    if let Ok(p) = std::env::var("PIPELINE_PYTHON") {
        return Ok(p);
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dev_bundled = manifest_dir
        .join("binaries")
        .join("python")
        .join("bin")
        .join("python3");
    if dev_bundled.exists() {
        return Ok(dev_bundled.to_string_lossy().into_owned());
    }
    Err(format!(
        "No bundled Python runtime found at {} and PIPELINE_PYTHON is \
         unset. Build it with `scripts/setup-bundled-python.sh` \
         before running `tauri dev`.",
        dev_bundled.display()
    ))
}

/// Resolve the single Python package root (`python/`). Every Python spawn
/// runs from here so `engine.*` / `kernel.*` / `testing.*` resolve as
/// fully-qualified imports — runner/chatbot launch as `python -m engine.<mod>`,
/// script entrypoints as `python python/engine/<mod>.py` with this on
/// PYTHONPATH.
/// - Dev:  relative to Cargo manifest dir → src-tauri/../python
/// - Prod: bundled into resource_dir()/python/
fn python_root_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    if cfg!(debug_assertions) {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("python");
        let dir = dir.canonicalize().map_err(|e| {
            format!("Cannot resolve python root {:?}: {}", dir, e)
        })?;
        Ok(dir)
    } else {
        app.path()
            .resource_dir()
            .map(|d| d.join("python"))
            .map_err(|e| e.to_string())
    }
}

#[derive(Default)]
struct AppState {
    output: Mutex<Option<String>>,
    /// At most one pipeline subprocess per mode is active at a time (so
    /// Local + TEE + Test can run side-by-side for observability, but two
    /// runs in the same mode fight over resources — Ollama slot, rate
    /// limits, etc.). Keyed by run_id so pause_run / cancel_run can look
    /// up directly.
    active_runs: Arc<Mutex<std::collections::HashMap<String, ActiveRun>>>,
    /// Cancel token for the live Chatbot sidecar, if any. Stop must
    /// actually terminate generation (not just disable the UI), so the
    /// sidecar's pid is tracked here for `chatbot_cancel` to SIGTERM/KILL.
    /// One at a time by UX contract (Send becomes Stop while streaming).
    /// Kept exactly as the pre-persistent cancel mechanism: the
    /// load-bearing #457 kill path is unchanged; only the normal
    /// turn-complete path no longer kills (the process stays warm).
    chatbot_inflight: Arc<Mutex<Option<ChatbotInflight>>>,
    /// The persistent Chatbot sidecar's I/O handle. The sidecar is
    /// spawned once per session and kept alive across turns so the
    /// ~2 s attested-client (TUF + ATC + enclave handshake)
    /// construction is paid once per process, not once per message.
    /// `chatbot` writes one newline-framed request here per turn over
    /// the kept-open stdin; a single long-lived reader thread drains
    /// stdout for the process's whole lifetime. Cleared on user-Stop
    /// (then eagerly re-spawned) or on an unexpected process death
    /// (then transparently re-spawned on the next message).
    chatbot_proc: Arc<Mutex<Option<ChatbotProc>>>,
    /// The run_id the user explicitly picked in the chat run selector
    /// (#507), or `None` for "use the most-recent-non-empty default".
    /// Persists across the session's re-spawns; changing it re-spawns
    /// the sidecar bound to the new run (a fresh pick = a fresh
    /// session). A selection that goes stale (run deleted, store
    /// emptied, OR aged out of the top-10 active list the dropdown
    /// renders) silently degrades to the default at resolve time —
    /// `resolve_chatbot_binding` anchors on the same list
    /// `chatbot_list_runs` returns, so the UI's bound highlight always
    /// matches the real spawn binding.
    chatbot_selected_run: Arc<Mutex<Option<String>>>,
    /// The active conversation's IMMUTABLE ISO-Z id (e.g.
    /// `2026-05-02T13-30-11Z`), not the dir name — so it survives a
    /// rename; spawn resolves it to the current dir under
    /// `chats_root()` via `find_convo_dir`. `None` before the first
    /// list. Drives which conversation dir the sidecar writes its
    /// `llm-*.jsonl` telemetry into: changing it re-spawns the sidecar
    /// (same machinery as a run rebind) so telemetry is scoped per
    /// conversation. NOT persisted — launch always opens the
    /// most-recently-active conversation (no active pointer on disk;
    /// the dir listing IS the source of truth).
    chatbot_active_convo: Arc<Mutex<Option<String>>>,
    /// Process-global monotonic turn counter. Each `chatbot` request
    /// gets the next value as its `turn_id`, echoed on every sidecar
    /// event. Lives in AppState (NOT per-`ChatbotProc`) so it does NOT
    /// reset when the sidecar re-spawns — a re-spawn (Stop, run-rebind,
    /// or the resume path's cancel→re-issue) keeps the counter strictly
    /// increasing, so a stale event from a pre-respawn generation can
    /// never share a `turn_id` with a post-respawn one. The client
    /// fences on the returned id and drops mismatches; without the
    /// global counter the first turn of every fresh process would be
    /// id 1 and collide with a still-draining prior generation.
    chatbot_turn_seq: Arc<AtomicU64>,
}

/// I/O handle for the persistent Chatbot sidecar. The cancel token
/// (pid + cancelled flag) stays in `chatbot_inflight` so the proven
/// #457 SIGTERM→SIGKILL path is byte-for-byte unchanged.
struct ChatbotProc {
    /// The process this handle drives — used to detect a stale handle
    /// (a re-spawn replaced it) before writing the next turn.
    pid: u32,
    /// Kept-open stdin; each turn writes one newline-framed JSON
    /// request line. Never closed between turns (closing it is the
    /// shutdown signal: the sidecar's request loop sees EOF and exits).
    stdin: std::process::ChildStdin,
}

struct ChatbotInflight {
    pid: u32,
    /// Set by `chatbot_cancel` before the kill so the spawn thread's
    /// post-`wait()` handler treats the non-zero exit as a
    /// user-initiated stop, not a sidecar crash (no spurious
    /// `chatbot_error` to the UI). Same role as `ActiveRun.terminal_override`.
    cancelled: Arc<AtomicBool>,
}

struct ActiveRun {
    run_id: String,
    pid: u32,
    mode: String,
    /// Set by pause_run / cancel_run before SIGKILL. When the spawn thread's
    /// child.wait() returns with a non-success status, it checks this flag
    /// to distinguish user-initiated termination (don't mark failed) from a
    /// genuine subprocess crash (mark failed). Without this, the "mark
    /// failed if currently running" check races with the user-initiated
    /// set_run_status("paused") and the UI flashes red briefly.
    terminal_override: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct StepResult {
    error: Option<String>,
}

/// Recursively collect all files under a directory path.
fn collect_files(dir: &std::path::Path, out: &mut Vec<String>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| e.path());
        for entry in entries {
            let path = entry.path();
            if path.is_file() {
                if let Some(s) = path.to_str() {
                    out.push(s.to_string());
                }
            } else if path.is_dir() {
                collect_files(&path, out);
            }
        }
    }
}

/// Expand a mixed list of file and directory paths into a flat, sorted list of files.
#[tauri::command]
fn expand_paths(paths: Vec<String>) -> Vec<String> {
    info!("expand_paths: {} inputs", paths.len());
    let mut files = Vec::new();
    for path_str in paths {
        let path = std::path::Path::new(&path_str);
        if path.is_file() {
            files.push(path_str);
        } else if path.is_dir() {
            collect_files(path, &mut files);
        }
    }
    info!("expand_paths: expanded to {} files", files.len());
    files
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct PathSize {
    path: String,
    size_bytes: u64,
}

/// Stat a list of files, returning `{path, size_bytes}` per input. A
/// missing or unreadable path collapses to `size_bytes: 0` rather than
/// erroring — the staging UI calls this for every path the user picks
/// and a single bad entry shouldn't blank the whole list. Directories
/// also map to 0 here; callers should pre-expand via `expand_paths`.
#[tauri::command]
fn stat_paths(paths: Vec<String>) -> Vec<PathSize> {
    paths
        .into_iter()
        .map(|p| {
            let size = std::fs::metadata(&p)
                .ok()
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .unwrap_or(0);
            PathSize { path: p, size_bytes: size }
        })
        .collect()
}

// ── Work estimation ──────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
struct WorkEstimate {
    total_bytes: u64,
    file_count: usize,
    est_llm_calls: u64,
    stages: Vec<StageEstimate>,
}

#[derive(Serialize, Deserialize, Clone)]
struct StageEstimate {
    name: String,
    calls: u64,
}

/// Estimate total LLM calls from file sizes.
///
/// Per file (B bytes):
///   segmenter:         ceil(B / 6000)           — one call per 6KB chunk
///   section_segmenter: ceil(B / 48000)           — windowed, ~16 candidates per call
///   summarizer:        1 + ceil(B / 3000)        — 1 doc + 1 per ~3KB section
///   metadata:          2                          — extraction + review
///   content_extractor: ceil(B / 768)             — one call per chunk (stride=768)
#[tauri::command]
fn estimate_work(paths: Vec<String>) -> WorkEstimate {
    info!("estimate_work: {} paths", paths.len());
    let mut total_bytes: u64 = 0;
    let file_count = paths.len();

    let mut seg_calls: u64 = 0;
    let mut sec_calls: u64 = 0;
    let mut sum_calls: u64 = 0;
    let mut meta_calls: u64 = 0;
    let mut ext_calls: u64 = 0;

    const IMAGE_EXTS: &[&str] = &[
        "jpg", "jpeg", "png", "heic", "heif", "bmp", "tiff", "webp", "gif",
    ];
    const LARGE_PDF_THRESHOLD: u64 = 1024 * 1024; // 1 MB

    for path_str in &paths {
        let b = std::fs::metadata(path_str)
            .map(|m| m.len())
            .unwrap_or(0);
        total_bytes += b;

        let ext = std::path::Path::new(path_str)
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        // Images: file bytes don't map to text bytes — vision transcribes into a
        // small blob. Flat estimate per image.
        if IMAGE_EXTS.contains(&ext.as_str()) {
            seg_calls += 1;
            sec_calls += 1;
            sum_calls += 2;
            meta_calls += 2;
            ext_calls += 1;
            continue;
        }

        // Large PDFs: likely scanned/image-heavy, minimal extractable text.
        // Flat small estimate; the re-estimate step will correct this after parsing.
        if ext == "pdf" && b >= LARGE_PDF_THRESHOLD {
            seg_calls += 1;
            sec_calls += 1;
            sum_calls += 2;
            meta_calls += 2;
            ext_calls += 1;
            continue;
        }

        // Zips: rough 3x decompression ratio for text content. The estimator
        // doesn't actually unzip — option 3 (trust re-estimate) is deferred.
        let effective_bytes = if ext == "zip" { b * 3 } else { b };

        // Text-based files (and small PDFs): size-proportional estimates
        // segmenter: 1 LLM call per 16KB of content
        seg_calls += (effective_bytes / 16000).max(1);
        // section_segmenter: windowed, ~16 candidates per call, ~3KB per candidate
        sec_calls += (effective_bytes / 48000).max(1);
        // summarizer: 1 doc summary + 1 per section (~3KB each)
        let est_sections = (effective_bytes / 3000).max(1);
        sum_calls += 1 + est_sections;
        // metadata: 2 calls per document (extraction + review)
        meta_calls += 2;
        // content_extractor: 1 call per chunk, stride = 3072 bytes
        ext_calls += (effective_bytes / 3072).max(1);
    }

    let stages = vec![
        StageEstimate { name: "segment".into(), calls: seg_calls },
        StageEstimate { name: "section_segment".into(), calls: sec_calls },
        StageEstimate { name: "summarize".into(), calls: sum_calls },
        StageEstimate { name: "metadata".into(), calls: meta_calls },
        StageEstimate { name: "extract".into(), calls: ext_calls },
    ];

    let est_llm_calls = seg_calls + sec_calls + sum_calls + meta_calls + ext_calls;

    info!(
        "estimate_work: {} files, {} bytes, ~{} LLM calls",
        file_count, total_bytes, est_llm_calls
    );
    WorkEstimate {
        total_bytes,
        file_count,
        est_llm_calls,
        stages,
    }
}

// ── Pipeline execution ───────────────────────────────────────────────────────

/// Spawn runner.py with the given args, track the child PID in `active_run`,
/// stream stdout to "pipeline-progress" events, and clear `active_run` on exit.
///
/// `run_id` is returned so the frontend can track this specific run.
/// If `resume_run_id` is Some, passes --resume-run-id instead of fresh args.
///
/// Enforces 1-at-a-time: errors if another run is already active.
fn spawn_pipeline(
    app: tauri::AppHandle,
    state: &AppState,
    run_id: String,
    paths: Vec<String>,
    mode: String,
    resume_run_id: Option<String>,
) -> Result<String, String> {
    // Per-mode gate: refuse if another run in the SAME mode is active.
    // Different modes can coexist (Local + TEE + Test side-by-side is the
    // use case for observing latency/quality differences).
    {
        let guard = state.active_runs.lock().unwrap();
        if let Some(existing) = guard.values().find(|a| a.mode == mode) {
            return Err(format!(
                "a {} run is already active ({})",
                mode, existing.run_id
            ));
        }
    }

    let py_bin = python_bin(&app)?;
    let py_root = python_root_dir(&app)?;

    let logs_root_s = logs_root().to_string_lossy().into_owned();
    let vault_root_s = vault_root().to_string_lossy().into_owned();

    let run_id_clone = run_id.clone();
    let mode_clone = mode.clone();
    let app_clone = app.clone();
    let active_runs_arc = Arc::clone(&state.active_runs);
    // #780: notify a live chatbot sidecar when this run finishes so a
    // session that started before any run existed can bind to the
    // just-completed corpus without restart or dropdown pick.
    let chatbot_proc_arc = Arc::clone(&state.chatbot_proc);

    ltrace_rust("spawn_pipeline_thread_spawned");  // dev_tracing instrumentation
    std::thread::spawn(move || {
        let active_runs_mutex: &Mutex<std::collections::HashMap<String, ActiveRun>> =
            &active_runs_arc;
        use std::io::{BufRead, BufReader};
        use std::process::{Command, Stdio};

        let mut cmd = Command::new(&py_bin);
        cmd.arg("-m").arg("engine.runner");
        cmd.arg("--mode").arg(&mode);
        if let Some(rid) = &resume_run_id {
            cmd.arg("--resume-run-id").arg(rid);
        }
        if !paths.is_empty() {
            cmd.arg("--paths");
            for p in &paths {
                cmd.arg(p);
            }
        }
        cmd.current_dir(&py_root)
            .env("BASEVAULT_RUN_NAME", &run_id_clone)
            .env("BASEVAULT_LOGS_ROOT", &logs_root_s)
            .env("BASEVAULT_VAULT_ROOT", &vault_root_s)
            // Surface the .app's version so the runner can record it in
            // config.json's run_config snapshot — useful when triaging a
            // pre-vs-post-update behavior change. Pure cosmetic for the
            // runner; missing env var → run_config.app_version=null.
            .env("BASEVAULT_APP_VERSION", env!("CARGO_PKG_VERSION"))
            // Defensively unset the legacy session/eval/sweep env vars
            // in case the user (or a parent shell) had any of them
            // exported. Post-flatten the runner writes to
            // <logs_root>/<run_id>/ unconditionally; stray env values
            // would only confuse downstream tools.
            .env_remove("BASEVAULT_SESSION")
            .env_remove("BASEVAULT_EVAL_ID")
            .env_remove("BASEVAULT_SWEEP_ID")
            // Stamp agent=app explicitly. The runner defaults to
            // agent=experiment (routes to ~/.basevault/logs-dev/) so
            // scripts, smoke runs, tests, and ad-hoc CLI invocations
            // stay out of the user's GUI runs list. The GUI is the
            // only producer of agent=app — and therefore the only
            // writer to ~/.basevault/logs/.
            .env("BASEVAULT_AGENT", "app")
            // dev_tracing handoff: scrub any inherited value first so a
            // shell-exported BASEVAULT_DEV_TRACING can't force tracing
            // ON when the Settings toggle is OFF — the config flag is
            // the only knob. Set the var conditionally from the config
            // read; Python's runner.py keys off it at module init.
            .env_remove("BASEVAULT_DEV_TRACING")
            // dev_wire_capture handoff: same shape as dev_tracing.
            // Scrub any shell-exported value so the Settings toggle is
            // the only knob; Python's llm.py reads it at module init
            // and attaches Tinfoil-client httpx hooks iff "1".
            .env_remove("BASEVAULT_DEV_WIRE_CAPTURE")
            // Pin every Python subprocess to the Tauri shell's
            // session dir so chat + pipeline + bootstrap events all
            // land in the same `app.log` under one dir per launch.
            .env("BASEVAULT_SESSION_DIR", session_dir())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if dev_tracing_on() {
            cmd.env("BASEVAULT_DEV_TRACING", "1");
        }
        if dev_wire_capture_on() {
            cmd.env("BASEVAULT_DEV_WIRE_CAPTURE", "1");
        }

        // Bundled-key fallback (#609). Python's load_dotenv runs with
        // override=False, so we set the env var ONLY when the user has
        // no key — otherwise the dotenv value would be shadowed.
        if user_tinfoil_key(&app_clone).is_none() {
            if let Some(k) = bundled_tinfoil_key() {
                cmd.env("TINFOIL_API_KEY", k);
            }
        }

        // Subject + TEE provider/model are read by Python directly from
        // config.json (the canonical "everything-non-secret" file). No
        // env bridge — the rule is: dotenv = secrets only, config.json =
        // everything else, both files read directly by the consumer.

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                error!("spawn_pipeline: failed to start python: {}", e);
                // Best-effort: mark the run as failed so the UI notices.
                let _ = set_run_status(
                    &run_id_clone,
                    "failed",
                    Some(format!("failed to start python: {}", e)),
                );
                return;
            }
        };
        ltrace_rust("subprocess_spawn_returned");  // dev_tracing instrumentation
        let pid = child.id();

        // Register as active.
        {
            let mut guard = active_runs_mutex.lock().unwrap();
            guard.insert(
                run_id_clone.clone(),
                ActiveRun {
                    run_id: run_id_clone.clone(),
                    pid,
                    mode: mode_clone.clone(),
                    terminal_override: None,
                },
            );
        }
        info!("spawn_pipeline: started pid={} run_id={}", pid, run_id_clone);

        // Drain stderr concurrently.
        let stderr_handle = child.stderr.take().map(|stderr| {
            std::thread::spawn(move || {
                let mut buf = String::new();
                BufReader::new(stderr).lines().flatten().for_each(|line| {
                    buf.push_str(&line);
                    buf.push('\n');
                });
                buf
            })
        });

        if let Some(stdout) = child.stdout.take() {
            // dev_tracing: first stdout line marks "Python is talking
            // back"; relay any [LAUNCH_TRACE] python lines through the
            // info! sink so they land in app.log alongside Rust +
            // frontend markers.
            let mut dt_first = true;
            for line in BufReader::new(stdout).lines().flatten() {
                if dt_first {
                    ltrace_rust("bufreader_first_line");
                    dt_first = false;
                }
                if dev_tracing_on() && line.starts_with("[LAUNCH_TRACE]") {
                    info!("{}", line);
                }
                // Only well-formed JSON-object lines drive the live
                // overlay + a pipeline-progress refresh. Every real
                // runner progress / heartbeat / stage / error line is
                // `print(json.dumps({...}))`; everything else on stdout
                // ([LAUNCH_TRACE] diagnostics when dev_tracing is on,
                // stray library prints, multi-line traceback bodies)
                // carries no UI state. Emitting pipeline-progress per
                // raw stdout line made a chatty (or tracing-on) run
                // fan every line out to an IPC event + React refresh —
                // the per-stdout-line storm. Gate both the overlay and
                // the emit on a parsed object so noise lines are inert.
                let Ok(ev) = serde_json::from_str::<serde_json::Value>(&line)
                else {
                    continue;
                };
                if !ev.is_object() {
                    continue;
                }
                // Update the in-memory live-token overlay BEFORE the
                // pipeline-progress emit so any React listener that
                // refetches on the event sees a consistent overlay.
                apply_stdout_event_to_live_state(&run_id_clone, &line);
                app_clone.emit("pipeline-progress", &line).ok();
            }
        }

        let stderr_output = stderr_handle
            .map(|h| h.join().unwrap_or_default())
            .unwrap_or_default();

        let wait_result = child.wait();

        // Snapshot terminal_override (set by pause_run / cancel_run before
        // SIGKILL), then remove from the active_runs map.
        let override_was_set = {
            let mut guard = active_runs_mutex.lock().unwrap();
            let was_set = guard
                .get(&run_id_clone)
                .map_or(false, |a| a.terminal_override.is_some());
            guard.remove(&run_id_clone);
            was_set
        };

        // Drop the live-token overlay for this run. Whatever in-flight
        // counts remained in memory are now stale — the runner is no
        // longer streaming. The materializer's jsonl walk is the
        // only read path for terminated runs.
        clear_live_state_for_run(&run_id_clone);

        match wait_result {
            Ok(status) if status.success() && !override_was_set => {
                info!("spawn_pipeline: pid={} completed", pid);
                // #780: tell a live chatbot sidecar this run is now
                // available so a started-before-any-run session binds
                // to it. No-op when no sidecar is up (the next spawn
                // resolves the binding via `resolve_chatbot_binding`
                // at start) or the sidecar is already bound (no
                // silent corpus swap mid-conversation).
                //
                // ``!override_was_set`` gates this on a NATURAL
                // completion: a SIGTERM-paused / cancelled run can
                // exit status 0 with a partial / empty vectors.db,
                // and pushing a binding to a half-written store
                // would surface garbage to the chat. The override
                // flag is set by `pause_run` / `cancel_run` before
                // the kill (same role its presence plays in the
                // failed-status arm below: terminal-state authority
                // belongs to the pause/cancel path).
                push_run_available_to_chatbot(&chatbot_proc_arc, &run_id_clone);
                // Refresh the chat panel's run-selector list (#780).
                // The stdin push above only reaches a LIVE sidecar; it
                // doesn't reach React directly, and React only refreshes
                // the dropdown on panel-open / dropdown-open / a
                // ``chatbot_bound`` event from the sidecar. So a user
                // who opened chat BEFORE any ingest finished and never
                // sent a message wouldn't see the new run in the
                // selector until they typed (which lazy-spawns the
                // sidecar). Emitting ``runs-changed`` here closes that
                // window: React listens directly, calls ``refreshRuns``,
                // dropdown updates the moment ingest completes whether
                // or not a sidecar is alive.
                let _ = app_clone.emit("runs-changed", &run_id_clone);
            }
            Ok(status) if status.success() => {
                // User-paused / cancelled: exit-0 shape but the
                // vectors.db may be partial. Skip the chatbot push
                // for the exact reason the failed arm below skips
                // a status overwrite — the pause/cancel path has
                // terminal-state authority. Reached only when the
                // previous arm's guard (`!override_was_set`) failed,
                // i.e. the pause/cancel handler set the flag.
                info!(
                    "spawn_pipeline: pid={} exit={:?} (user-paused/cancelled — skipping chatbot push)",
                    pid, status.code()
                );
            }
            Ok(status) => {
                if override_was_set {
                    // User-initiated termination — pause_run / cancel_run
                    // has authority over the derived terminal status
                    // (paused.flag / cycle_cancelled). Don't overwrite.
                    info!(
                        "spawn_pipeline: pid={} exit={:?} (user-initiated, status left to pause/cancel)",
                        pid, status.code()
                    );
                } else {
                    error!(
                        "spawn_pipeline: pid={} exit={:?} stderr_tail=<<<{}>>>",
                        pid,
                        status.code(),
                        tail_lines(&stderr_output, 20)
                    );
                    // Unexpected exit. If Python wrote a terminal state
                    // already (e.g. the FATAL preflight path), leave it.
                    if let Ok(Some(current)) = read_run_status(&run_id_clone) {
                        if current == "running" {
                            let _ = set_run_status(
                                &run_id_clone,
                                "failed",
                                Some(format!(
                                    "subprocess exit {:?}: {}",
                                    status.code(),
                                    tail_lines(&stderr_output, 5)
                                )),
                            );
                            // A crash is a wind-down — capture the diagnostic.
                            spawn_run_diagnostic_emit(&app_clone, &run_id_clone);
                        }
                    }
                }
            }
            Err(e) => {
                error!("spawn_pipeline: wait failed: {}", e);
            }
        }
    });

    Ok(run_id)
}

fn read_run_status(run_id: &str) -> Result<Option<String>, String> {
    // Status comes from the derived state walk (jsonl + markers). For
    // legacy runs that pre-date cycle markers, fall back to run.json's
    // status field so the orphan-recovery / spawn-thread error paths
    // keep working.
    let Some(run_dir) = find_run_dir(run_id) else { return Ok(None) };
    let derived = derive_run_state(&run_dir);
    if derived.status != "unknown" {
        return Ok(Some(derived.status));
    }
    let path = run_dir.join("run.json");
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("read {:?}: {}", path, e))?;
    let val: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("parse {:?}: {}", path, e))?;
    Ok(val
        .get("status")
        .and_then(|s| s.as_str())
        .map(|s| s.to_string()))
}

/// Run the full pipeline via the Python runner subprocess.
///
/// Generates a fresh run_id (`<iso-z>-<short_id>`) per click. The 4-char
/// short_id replaces the historical "manual" suffix — humans can quote
/// it to find the run in logs/vault. Refuses if another run is active.
/// Returns the run_id so the UI can track this specific run in the
/// runs list.
#[tauri::command]
fn run_pipeline(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    paths: Vec<String>,
    mode: String,
) -> Result<String, String> {
    ltrace_rust("run_pipeline_entry");  // dev_tracing instrumentation
    let run_id = format!("{}-{}", iso_z(), short_id());
    info!(
        "run_pipeline: run_id={} mode={} paths={}",
        run_id,
        mode,
        paths.len()
    );
    spawn_pipeline(app, state.inner(), run_id, paths, mode, None)
}

/// Resume a paused or failed run from its last checkpoint.
#[tauri::command]
fn resume_run(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    run_id: String,
) -> Result<String, String> {
    // mode + inputs are STATIC — recover them via the config.json /
    // run.json fallback path. read_run_state hides the source choice.
    let run_dir = find_run_dir(&run_id)
        .ok_or_else(|| format!("run dir not found for run_id {}", run_id))?;
    let val = read_run_state(&run_dir)
        .ok_or_else(|| format!("no state sidecar for run_id {}", run_id))?;
    let mode = val
        .get("mode")
        .and_then(|s| s.as_str())
        .unwrap_or("local")
        .to_string();
    let paths: Vec<String> = val
        .get("inputs")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    info!(
        "resume_run: run_id={} mode={} paths={}",
        run_id,
        mode,
        paths.len()
    );
    // Clear any paused.flag eagerly so the UI's derived status flips
    // off "paused" as soon as the runner emits its first event. The
    // runner's first jsonl write also bumps mtime → cache busts;
    // without removing the marker first, the next list_runs would
    // still synthesize "paused" until the new cycle_start lands.
    let _ = std::fs::remove_file(run_dir.join("paused.flag"));
    spawn_pipeline(app, state.inner(), run_id.clone(), paths, mode, Some(run_id))
}

/// SIGKILL the active subprocess (if run_id matches) and mark the run as
/// `paused`. No-op if no process is active; still flips disk state to
/// paused so the UI reflects intent. Atomic write. Like every wind-down,
/// a pause drops the run-diagnostics shareable once settled (latest-wins,
/// so a later resume→complete overwrites this partial snapshot).
#[tauri::command]
fn pause_run(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    run_id: String,
) -> Result<(), String> {
    let res = terminate_run(state.inner(), &run_id, "paused", None);
    if res.is_ok() {
        spawn_run_diagnostic_emit(&app, &run_id);
    }
    res
}

/// SIGKILL + mark `cancelled`. No resume possible after this. Like every
/// run wind-down, a cancel drops the content-free run-diagnostics
/// shareable once the run has settled — see `spawn_run_diagnostic_emit`.
#[tauri::command]
fn cancel_run(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    run_id: String,
) -> Result<(), String> {
    let res = terminate_run(state.inner(), &run_id, "cancelled", Some("cancelled by user".into()));
    if res.is_ok() {
        spawn_run_diagnostic_emit(&app, &run_id);
    }
    res
}

/// Best-effort: drop the content-free run-diagnostics shareable for a run
/// that just wound down (completed / paused / cancelled / failed /
/// app-close), by spawning `runner.py --emit-run-diagnostic <run_dir>`.
/// Fired AFTER the run has settled and its sidecar is gone, off the
/// SIGKILL-grace window an in-process emit would race. The RUN shareable
/// is latest-wins (overwrite), so emitting on every wind-down — including
/// a resumable pause — is safe: a later, more-complete state replaces an
/// earlier partial one (a pause snapshot is superseded by the eventual
/// resume→complete).
///
/// The child is spawned synchronously (so the app-exit path launches it
/// before the process dies — a spawned child outlives its parent) and
/// reaped on a detached thread (so we neither block the caller nor leak a
/// zombie in the long-running app). Every failure is logged and swallowed
/// — the diagnostic is an aid, never a gate on the path that called us.
fn spawn_run_diagnostic_emit(app: &tauri::AppHandle, run_id: &str) {
    use std::process::{Command, Stdio};
    let Some(run_dir) = find_run_dir(run_id) else {
        info!("emit-run-diagnostic: no run dir for {}, skipping", run_id);
        return;
    };
    let py_bin = match python_bin(app) {
        Ok(p) => p,
        Err(e) => { error!("emit-run-diagnostic: python_bin: {}", e); return; }
    };
    let py_root = match python_root_dir(app) {
        Ok(p) => p,
        Err(e) => { error!("emit-run-diagnostic: python_root_dir: {}", e); return; }
    };
    let logs_root_s = logs_root().to_string_lossy().into_owned();
    let vault_root_s = vault_root().to_string_lossy().into_owned();
    let run_dir_s = run_dir.to_string_lossy().into_owned();
    let run_id_owned = run_id.to_string();
    // Launch synchronously — a spawned child survives the parent, so the
    // app-exit path still gets the emit even as the app process dies.
    let child = Command::new(&py_bin)
        .arg("-m")
        .arg("engine.runner")
        .arg("--emit-run-diagnostic")
        .arg(&run_dir_s)
        .current_dir(&py_root)
        // Match the env the in-run emit sees so the shareable lands in the
        // same place (sibling of the user's logs root). run_dir is passed
        // absolute, so these are belt-and-suspenders.
        .env("BASEVAULT_LOGS_ROOT", &logs_root_s)
        .env("BASEVAULT_VAULT_ROOT", &vault_root_s)
        .env("BASEVAULT_AGENT", "app")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    match child {
        // Reap on a detached thread: no block, no zombie. If the app exits
        // first the thread dies, but the child (a separate process)
        // finishes the emit regardless.
        Ok(mut child) => {
            std::thread::spawn(move || match child.wait() {
                Ok(s) => info!(
                    "emit-run-diagnostic: run={} exit={:?}", run_id_owned, s.code()
                ),
                Err(e) => error!(
                    "emit-run-diagnostic: run={} wait failed: {}", run_id_owned, e
                ),
            });
        }
        Err(e) => error!(
            "emit-run-diagnostic: run={} spawn failed: {}", run_id_owned, e
        ),
    }
}

/// Issue #333: user-skip for a single in-flight LLM call. Atomic
/// touch under `<run_dir>/skipped_calls/<call_id>` — mirrors the
/// paused.flag pattern (tmp + rename) so a SIGKILL mid-write can't
/// leave a torn file. The Python runner's marker-dir poller picks up
/// the file within ~500ms and registers the id in
/// `llm._skipped_call_ids`; the next chunk read by
/// `_consume_chat_stream` for that call raises `_SkippedByUser`,
/// which short-circuits the wrapper through `decide_next_action` →
/// `give_up`. The scheduler is not charged (orthogonal to the
/// load/sizing/other taxonomy — user intervention, not a provider
/// shape).
///
/// No process signal — pure marker write. The poller + the per-chunk
/// stream check do the work on the Python side. Slightly slower for
/// a call hanging pre-TTFT (worst case is next provider keepalive)
/// but no cross-thread stream-state risk.
#[tauri::command]
fn skip_call(run_id: String, call_id: String) -> Result<(), String> {
    // Basic input sanitation: refuse anything that could escape the
    // run dir. call_id is a 4-digit zero-padded counter from the
    // Python wrapper today, but tighten defensively to alphanumeric
    // + dash + underscore so a malicious / mistyped id can't traverse
    // into a sibling directory.
    if call_id.is_empty()
        || call_id.len() > 64
        || !call_id.chars().all(|c| {
            c.is_ascii_alphanumeric() || c == '-' || c == '_'
        })
    {
        return Err(format!("invalid call_id: {:?}", call_id));
    }
    let run_dir = find_run_dir(&run_id)
        .ok_or_else(|| format!("run dir not found for run_id {}", run_id))?;
    let skip_dir = run_dir.join("skipped_calls");
    std::fs::create_dir_all(&skip_dir)
        .map_err(|e| format!("create skipped_calls dir: {}", e))?;
    let marker = skip_dir.join(&call_id);
    let tmp = skip_dir.join(format!("{}.tmp", call_id));
    // Empty body — presence is the signal. The file is never read,
    // only stat'd by the materializer + the poller's iterdir.
    std::fs::write(&tmp, "")
        .map_err(|e| format!("write tmp skip marker: {}", e))?;
    std::fs::rename(&tmp, &marker)
        .map_err(|e| format!("rename skip marker: {}", e))?;
    info!("skip_call: run={} call={}", run_id, call_id);
    Ok(())
}

fn terminate_run(
    state: &AppState,
    run_id: &str,
    new_status: &str,
    error: Option<String>,
) -> Result<(), String> {
    // Set terminal_override so the spawn thread knows the termination was
    // user-initiated and won't race to mark the run "failed" after SIGKILL.
    // Keep the ActiveRun entry in the map so the spawn thread can read
    // the flag when child.wait() returns; the thread removes it itself.
    let pid = {
        let mut guard = state.active_runs.lock().unwrap();
        if let Some(a) = guard.get_mut(run_id) {
            a.terminal_override = Some(new_status.to_string());
            Some(a.pid)
        } else {
            None
        }
    };

    if let Some(pid) = pid {
        // SIGTERM first so the Python sidecar's signal handler can flush
        // atexit-registered observability artifacts (llm-calls.jsonl
        // rollup, llm-stats.txt). SIGKILL bypasses Python entirely and
        // leaves the rollup unwritten — the rbw8 cancellation symptom.
        //
        // Grace period: 5 seconds. Round 4 — p63u showed that even with
        // an inline rollup write in the Python signal handler, the
        // first 1-2s post-SIGTERM can be eaten by other shutdown work
        // (any pending log flushes, atexit-registered cleanup). 5s
        // leaves headroom for the rollup write (50-200ms typical,
        // ~1-2s on a few-thousand-call run with the new per-stage
        // stats) without making cancel feel sluggish.
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGTERM);
        }
        info!("terminate_run: SIGTERM pid={} run_id={}", pid, run_id);

        let pid_t = pid as libc::pid_t;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let exited = loop {
            // kill(pid, 0) returns 0 if the process exists, -1 (ESRCH) if not.
            let alive = unsafe { libc::kill(pid_t, 0) } == 0;
            if !alive {
                break true;
            }
            if std::time::Instant::now() >= deadline {
                break false;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        };

        if !exited {
            unsafe {
                libc::kill(pid_t, libc::SIGKILL);
            }
            info!(
                "terminate_run: SIGTERM grace expired, SIGKILL pid={} run_id={}",
                pid, run_id
            );
        }
    } else {
        info!(
            "terminate_run: no active process for run_id {} — updating disk state only",
            run_id
        );
    }

    set_run_status(run_id, new_status, error)
}

/// Merge legacy run.json fields into a config.json payload. Static
/// fields already in `static_val` win (config.json is canonical for
/// the static set). Dynamic fields fall through from run.json so
/// pre-#165 runs and any in-progress fallback paths keep working.
fn merge_legacy_run_json_fields(
    static_val: &mut serde_json::Value,
    run_dir: &Path,
) {
    let legacy = run_dir.join("run.json");
    if !legacy.exists() {
        return;
    }
    let Ok(text) = std::fs::read_to_string(&legacy) else { return };
    let Ok(legacy_val) = serde_json::from_str::<serde_json::Value>(&text) else { return };
    let Some(legacy_obj) = legacy_val.as_object() else { return };
    let Some(target_obj) = static_val.as_object_mut() else { return };
    for (k, v) in legacy_obj {
        target_obj.entry(k.clone()).or_insert_with(|| v.clone());
    }
}

/// Overlay derived dynamic state from llm-calls.jsonl onto a run record.
///
/// Replaces the runner's run.json `status` / `progress` / `duration_ms`
/// / `updated_at` with values walked from the append-only event log.
/// Always overrides — derivation is the source of truth post-#165.
/// For runs whose jsonl pre-dates cycle markers (very old), derivation
/// returns "unknown"; the merge in that case leaves the record's
/// existing status alone.
fn overlay_derived_dynamic_fields(
    target: &mut serde_json::Value,
    run_dir: &Path,
) {
    let derived = derive_run_state(run_dir);
    let Some(obj) = target.as_object_mut() else { return };

    if derived.status != "unknown" {
        obj.insert(
            "status".into(),
            serde_json::Value::String(derived.status.clone()),
        );
    }
    // Progress: synthesize an object compatible with the JSX shape so
    // the existing run-row reads of progress.completed / .in_flight_calls
    // / .stage continue to work without changes.
    let mut prog = serde_json::Map::new();
    if let Some(stage) = &derived.progress.stage {
        prog.insert("stage".into(), serde_json::Value::String(stage.clone()));
    }
    prog.insert(
        "completed".into(),
        serde_json::Value::Number(derived.progress.completed.into()),
    );
    prog.insert(
        "in_flight_calls".into(),
        serde_json::Value::Number(derived.progress.in_flight_calls.into()),
    );
    if let Some(t) = derived.progress.total {
        prog.insert("total".into(), serde_json::Value::Number(t.into()));
    }
    if let Some(eta) = derived.progress.eta_seconds {
        if let Some(n) = serde_json::Number::from_f64(eta) {
            prog.insert("eta_seconds".into(), serde_json::Value::Number(n));
        }
    }
    if let Some(bp) = derived.progress.bar_position {
        if let Some(n) = serde_json::Number::from_f64(bp) {
            prog.insert("bar_position".into(), serde_json::Value::Number(n));
        }
    }
    // Surface the latest tick's wall-clock + per-stage timing fields
    // so the JSX can interpolate `bar_position` over elapsed time
    // between ticks. Single-call stages don't fan out begin/end events
    // — the bar would freeze on the last tick's value for the entire
    // call without the `bar_position_at` anchor + `stage_eta_seconds`
    // for client-side time extrapolation.
    if let Some(ts) = &derived.progress.bar_position_at {
        prog.insert(
            "bar_position_at".into(),
            serde_json::Value::String(ts.clone()),
        );
    }
    if let Some(seta) = derived.progress.stage_eta_seconds {
        if let Some(n) = serde_json::Number::from_f64(seta) {
            prog.insert("stage_eta_seconds".into(), serde_json::Value::Number(n));
        }
    }
    if let Some(eis) = derived.progress.elapsed_in_stage {
        if let Some(n) = serde_json::Number::from_f64(eis) {
            prog.insert("elapsed_in_stage".into(), serde_json::Value::Number(n));
        }
    }
    // Carry over any progress fields the runner stamped on run.json
    // that derivation doesn't know about. Only fill missing keys;
    // derivation wins where both have data.
    if let Some(existing_prog) = obj.get("progress").and_then(|v| v.as_object()) {
        for (k, v) in existing_prog {
            prog.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
    obj.insert("progress".into(), serde_json::Value::Object(prog));

    if let Some(d) = derived.duration_ms {
        obj.insert(
            "duration_ms".into(),
            serde_json::Value::Number(d.into()),
        );
    }
    if let Some(u) = derived.updated_at {
        obj.insert("updated_at".into(), serde_json::Value::String(u));
    }
    if let Some(e) = derived.error {
        obj.insert("error".into(), serde_json::Value::String(e));
    }
    if let Some(p) = derived.pid {
        obj.insert("pid".into(), serde_json::Value::Number(p.into()));
    }
    // Surface entities-stage decisions the runner emits as
    // `entities_decision` events. Land them on run_config so the
    // JSX RunDetails component reads `run.run_config.subject_resolution`
    // / `.bundle_mode` without forking on origin.
    if derived.subject_resolution.is_some() || derived.bundle_mode.is_some() {
        let cfg_existing = obj.get("run_config").cloned();
        let mut cfg_obj = match cfg_existing {
            Some(serde_json::Value::Object(m)) => m,
            _ => serde_json::Map::new(),
        };
        if let Some(sr) = derived.subject_resolution {
            cfg_obj.insert("subject_resolution".into(), sr);
        }
        if let Some(bm) = derived.bundle_mode {
            cfg_obj.insert("bundle_mode".into(), serde_json::Value::Bool(bm));
        }
        obj.insert("run_config".into(), serde_json::Value::Object(cfg_obj));
    }
}

/// Overlay `warnings` + `llm_cache` blocks onto the run record from
/// the materialized rollup. Issue #189 — llm-stats.json is no longer
/// written, so this always materializes from llm-calls.jsonl +
/// config.json. The legacy llm-stats.json file is read only for very
/// old runs whose jsonl is missing/empty (covers the file-fallback
/// case in the issue's acceptance criteria).
///
/// Same code path running and finished — the live banner sees the
/// same numbers the post-rollup view did before #189 (leaf-aware
/// outcomes, chain-aware sampled bucketing, etc.).
fn overlay_rollup_warnings_and_cache(
    target: &mut serde_json::Value,
    run_dir: &Path,
) {
    // Materialize first — the canonical path. Returns Null when the
    // jsonl is missing/empty.
    let materialized = materialize_llm_stats_cached(run_dir);
    let rollup: serde_json::Value = if !materialized.is_null() {
        materialized
    } else {
        // Legacy fallback: very old runs may have llm-stats.json on
        // disk without an llm-calls.jsonl (pre-#165 dump format).
        let path = run_dir.join("llm-stats.json");
        if !path.exists() { return; }
        let Ok(text) = std::fs::read_to_string(&path) else { return };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else { return };
        v
    };
    let Some(obj) = target.as_object_mut() else { return };
    if let Some(w) = rollup.get("warnings") {
        obj.entry("warnings".to_string()).or_insert_with(|| w.clone());
    }
    if let Some(c) = rollup.get("llm_cache") {
        obj.entry("llm_cache".to_string()).or_insert_with(|| c.clone());
    }
}

/// List all app-owned runs on disk, newest-first. Dev runs (scripts,
/// smoke, sweeps, tests, ad-hoc CLI) write to `~/.basevault/logs-dev/`
/// — a separate root we never scan from here. The `agent == "app"`
/// filter below is a belt-and-suspenders guard against legacy
/// `agent="experiment"` runs that pre-date the split and may still
/// sit at the top level of `~/.basevault/logs/`.
///
/// Each entry is built from:
///   1. config.json (preferred) or run.json (legacy fallback) — STATIC fields.
///   2. Dynamic state derived from llm-calls.jsonl + filesystem markers
///      (status, progress, duration_ms, updated_at, error, pid).
/// Derivation is mtime-cached so this command stays cheap on the
/// 500ms refresh cadence.
#[tauri::command]
fn list_runs() -> Vec<serde_json::Value> {
    let logs = logs_root();
    let mut out: Vec<serde_json::Value> = Vec::new();
    if !logs.exists() {
        return out;
    }
    for sidecar in walk_run_jsons(&logs) {
        let Ok(text) = std::fs::read_to_string(&sidecar) else { continue };
        let Ok(mut val) = serde_json::from_str::<serde_json::Value>(&text) else { continue };
        let agent = val.get("agent").and_then(|a| a.as_str()).unwrap_or("");
        if agent != "app" {
            continue;
        }
        let Some(run_dir) = sidecar.parent() else { continue };
        // When the sidecar is config.json, also fold in any legacy
        // run.json fields the new flow doesn't emit — keeps
        // backwards-compat read paths simple. Static fields in
        // config.json win; legacy dynamic fields get overlaid below
        // by the jsonl derivation.
        if sidecar.file_name().and_then(|n| n.to_str()) == Some("config.json") {
            merge_legacy_run_json_fields(&mut val, run_dir);
        }
        overlay_derived_dynamic_fields(&mut val, run_dir);
        overlay_rollup_warnings_and_cache(&mut val, run_dir);
        // Augment with vault_exists so the UI can distinguish "vault ready"
        // from "vault missing (deleted in Finder)" and offer a regen action
        // on completed-but-vaultless runs.
        let vault_exists = val
            .get("vault_dir")
            .and_then(|v| v.as_str())
            .map(|s| std::path::Path::new(s).exists())
            .unwrap_or(false);
        // Perma-id fallback (Fix B): the run dir name is `<iso-z>-<id>`
        // and is the immutable record. If the sidecar JSON has no
        // usable `short_id` (legacy `run.json`, or a `config.json` that
        // was never written because the run was killed before
        // provider/model pick), derive both `run_id` and `short_id`
        // from the dir leaf so the perma-id is never lost from the UI.
        let dir_leaf = run_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if let Some(obj) = val.as_object_mut() {
            obj.insert("vault_exists".into(), serde_json::Value::Bool(vault_exists));
            let has_run_id = obj
                .get("run_id")
                .and_then(|v| v.as_str())
                .map(|s| !s.is_empty())
                .unwrap_or(false);
            if !has_run_id && !dir_leaf.is_empty() {
                obj.insert(
                    "run_id".into(),
                    serde_json::Value::String(dir_leaf.clone()),
                );
            }
            let has_short_id = obj
                .get("short_id")
                .and_then(|v| v.as_str())
                .map(|s| !s.is_empty())
                .unwrap_or(false);
            if !has_short_id {
                if let Some(sid) = short_id_from_run_name(&dir_leaf) {
                    obj.insert("short_id".into(), serde_json::Value::String(sid));
                }
            }
        }
        out.push(val);
    }
    out.sort_by(|a, b| {
        let ak = a.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
        let bk = b.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
        bk.cmp(ak)
    });
    out
}

/// Read the preprocessed input markdown files for a run as a
/// `{ file_id: text }` map (no `.md` suffix on the keys). The Obsidian
/// renderer stamps these with footnote markers per fact citation; the
/// in-app FactsView renders the same source via `read_run_file`.
///
/// Returns an empty map when the run dir is missing or the
/// preprocessed/ stage hasn't run yet — the renderer's input-stamping
/// pass is a no-op in that case.
#[tauri::command]
fn read_run_preprocessed_inputs(run_id: String) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    let Some(log_dir) = run_log_dir(&run_id) else {
        return serde_json::Value::Object(out);
    };
    let docs = log_dir.join("stages").join("00-ingestion").join("documents");
    if !docs.is_dir() {
        return serde_json::Value::Object(out);
    }
    fn walk_md(dir: &Path, root: &Path, out: &mut serde_json::Map<String, serde_json::Value>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk_md(&p, root, out);
                continue;
            }
            if p.extension().and_then(|s| s.to_str()) != Some("md") { continue; }
            let Ok(rel) = p.strip_prefix(root) else { continue };
            let Some(rel_s) = rel.to_str() else { continue };
            let file_id = rel_s.trim_end_matches(".md").to_string();
            let Ok(text) = std::fs::read_to_string(&p) else { continue };
            out.insert(file_id, serde_json::Value::String(text));
        }
    }
    walk_md(&docs, &docs, &mut out);
    serde_json::Value::Object(out)
}

/// Persist a JS-rendered Obsidian vault to `<vault_root>/<run_id>/`.
/// Mirrors `vault_exporter.py`'s wipe-and-rewrite semantics: subdirs
/// (`1-facts/`, `2-entities/`, `3-patterns/`, `0-inputs/`) are wiped
/// before write so files dropped in the new render don't linger; the
/// stage roots that are single files (`4-insights.md`, `5-actions.md`,
/// `index.md`) just overwrite. `vault_root/README.md` is also updated
/// to keep the cross-run listing fresh.
///
/// `files` is the manifest the JS `exportRun` returns: relative paths
/// rooted at the run's vault dir → file contents. All paths are
/// validated to stay inside the run dir.
#[tauri::command]
fn write_run_vault(
    run_id: String,
    files: std::collections::HashMap<String, String>,
) -> Result<(), String> {
    let vault_dir = vault_root().join(&run_id);
    std::fs::create_dir_all(&vault_dir)
        .map_err(|e| format!("create {:?}: {}", vault_dir, e))?;

    // Wipe-and-rewrite subdirs the renderer always (re)materializes in
    // full, so files removed between runs (deleted topic, dropped
    // entity) don't linger.
    for sub in ["0-inputs", "1-facts", "2-entities", "3-patterns"] {
        let p = vault_dir.join(sub);
        if p.exists() {
            std::fs::remove_dir_all(&p)
                .map_err(|e| format!("remove {:?}: {}", p, e))?;
        }
    }
    // Single-file pages. Wipe so a stale render with a different
    // line count doesn't bleed through.
    for f in ["4-insights.md", "5-actions.md", "index.md"] {
        let p = vault_dir.join(f);
        if p.exists() {
            let _ = std::fs::remove_file(&p);
        }
    }

    for (rel, content) in &files {
        let rel_path = PathBuf::from(rel);
        if rel_path.is_absolute() {
            return Err(format!("rel path must be relative: {}", rel));
        }
        for comp in rel_path.components() {
            use std::path::Component;
            match comp {
                Component::Normal(_) => {}
                _ => return Err(format!("rel path may not contain {:?}: {}", comp, rel)),
            }
        }
        let target = vault_dir.join(&rel_path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("mkdir {:?}: {}", parent, e))?;
        }
        std::fs::write(&target, content.as_bytes())
            .map_err(|e| format!("write {:?}: {}", target, e))?;
    }
    info!(
        "write_run_vault: run_id={} files={} → {}",
        run_id, files.len(), vault_dir.display()
    );

    // Update the cross-run README. Cheap (one dir scan + sort) and
    // keeps the vault root navigable in Obsidian.
    if let Err(e) = write_vault_root_readme() {
        warn!("write_vault_root_readme: {}", e);
    }
    Ok(())
}

/// Refresh `<vault_root>/README.md` with a reverse-chrono listing of
/// run dirs. Mirrors `vault_exporter.write_vault_readme`. Idempotent;
/// safe to call after every export.
fn write_vault_root_readme() -> Result<(), String> {
    let vroot = vault_root();
    if !vroot.exists() {
        return Ok(());
    }
    let mut runs: Vec<String> = Vec::new();
    let mut has_insights: std::collections::HashMap<String, bool> = std::collections::HashMap::new();
    let entries = std::fs::read_dir(&vroot)
        .map_err(|e| format!("read_dir {:?}: {}", vroot, e))?;
    for e in entries.flatten() {
        let p = e.path();
        if !p.is_dir() { continue; }
        let Some(name) = p.file_name().and_then(|s| s.to_str()) else { continue };
        if name.starts_with('.') { continue; }
        runs.push(name.to_string());
        has_insights.insert(name.to_string(), p.join("4-insights.md").exists());
    }
    runs.sort_by(|a, b| b.cmp(a));  // reverse-chrono
    let mut lines: Vec<String> = vec![
        "# BaseVault runs\n".into(),
        format!("Total runs: {}\n", runs.len()),
        "## Most recent\n".into(),
    ];
    for r in runs.iter().take(20) {
        let target = if *has_insights.get(r).unwrap_or(&false) {
            format!("{}/4-insights", r)
        } else {
            r.clone()
        };
        lines.push(format!("- [[{}|{}]]", target, r));
    }
    if runs.len() > 20 {
        lines.push(format!("\n*(+{} older runs)*", runs.len() - 20));
    }
    let body = lines.join("\n") + "\n";
    let target = vroot.join("README.md");
    std::fs::write(&target, body)
        .map_err(|e| format!("write {:?}: {}", target, e))?;
    Ok(())
}

/// Delete both the log dir and the vault dir for a given run. Requires
/// an explicit confirm dialog in the UI before calling (destructive).
///
/// For in-flight runs (status `running` or `paused`) the pipeline
/// subprocess MUST be stopped before `rm -rf` — otherwise the runner
/// keeps writing into the deleted dir, and several of its writers
/// (e.g. `_append_fact_jsonl`) call `path.parent.mkdir(parents=True,
/// exist_ok=True)` which silently re-creates the freshly-deleted run
/// dir. The user perceives this as "I deleted it but it came back."
#[tauri::command]
fn delete_run(state: State<'_, AppState>, run_id: String) -> Result<(), String> {
    // Log dir: parent of run.json.
    let log_dir = find_run_json(&run_id).and_then(|rj| rj.parent().map(Path::to_path_buf));

    // If the run is in-flight, kill the pipeline subprocess (graceful
    // SIGTERM → wait 5s → SIGKILL) BEFORE rm -rf'ing the dir. Reads pid
    // from run.json so this works for runs the current app session didn't
    // start (theoretical orphan survivors).
    if let Some(ref dir) = log_dir {
        stop_inflight_pipeline(state.inner(), &run_id, dir);
    }

    if let Some(dir) = log_dir {
        if dir.exists() {
            std::fs::remove_dir_all(&dir)
                .map_err(|e| format!("remove log dir {:?}: {}", dir, e))?;
            info!("delete_run: removed log dir {}", dir.display());
        }
    }
    // Vault dir: <vault_root>/<run_id>/
    let vault_dir = vault_root().join(&run_id);
    if vault_dir.exists() {
        std::fs::remove_dir_all(&vault_dir)
            .map_err(|e| format!("remove vault dir {:?}: {}", vault_dir, e))?;
        info!("delete_run: removed vault dir {}", vault_dir.display());
    }
    Ok(())
}

/// If `run_id`'s run.json shows an in-flight status (`running` / `paused`)
/// and records a live pid, stop the subprocess so a follow-up `rm -rf`
/// of the run dir won't race the runner's writers.
///
/// SIGTERM first so the Python sidecar's signal handler can flush its
/// rollup (matching pause_run / cancel_run); SIGKILL after a 5s grace.
/// Best-effort — silently skips when the run.json is missing,
/// unparseable, terminal, or has no recorded pid.
fn stop_inflight_pipeline(state: &AppState, run_id: &str, run_dir: &Path) {
    // Status + pid come from the derived state (jsonl walk + markers).
    // For very old runs that pre-date cycle markers we fall back to
    // run.json so the pre-#165 cleanup semantics survive.
    let derived = derive_run_state(run_dir);
    let (status, pid) = if derived.status != "unknown" {
        (derived.status.clone(), derived.pid)
    } else {
        let rj = run_dir.join("run.json");
        let Ok(text) = std::fs::read_to_string(&rj) else { return };
        let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) else { return };
        let s = val.get("status").and_then(|s| s.as_str()).unwrap_or("").to_string();
        let p = val
            .get("pid")
            .and_then(|p| p.as_u64())
            .and_then(|p| u32::try_from(p).ok());
        (s, p)
    };
    if !matches!(status.as_str(), "running" | "paused") {
        return;
    }
    let Some(pid) = pid else {
        info!(
            "stop_inflight_pipeline: run={} is in-flight but has no recorded pid — skipping",
            run_id
        );
        return;
    };

    // If the run is in this session's active_runs map, mark
    // terminal_override so the spawn thread's wait-result handler
    // doesn't try to flip the run to "failed" after we've removed the
    // dir (the write would fail anyway, but it spams the log).
    {
        let mut guard = state.active_runs.lock().unwrap();
        if let Some(a) = guard.get_mut(run_id) {
            a.terminal_override = Some("cancelled".to_string());
        }
    }

    if !is_pid_alive(pid) {
        return;
    }
    info!("stop_inflight_pipeline: stopping pid={} run_id={}", pid, run_id);
    stop_pid_with_grace(pid, std::time::Duration::from_secs(5));
}

/// SIGTERM `pid`, poll for exit until `grace` elapses, then SIGKILL.
/// Pure libc — no AppState coupling so this is unit-testable against
/// a real (forked) child.
fn stop_pid_with_grace(pid: u32, grace: std::time::Duration) {
    let pid_t = pid as libc::pid_t;
    // SAFETY: kill is a thread-safe libc call; signum is a constant.
    unsafe { libc::kill(pid_t, libc::SIGTERM) };

    let deadline = std::time::Instant::now() + grace;
    while std::time::Instant::now() < deadline {
        if !is_pid_alive(pid) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    // Grace expired — hard kill. After SIGKILL the kernel destroys the
    // process; userspace writers cannot run another instruction. Safe to
    // rm -rf the run dir on return.
    unsafe { libc::kill(pid_t, libc::SIGKILL) };
    info!(
        "stop_pid_with_grace: SIGTERM grace expired, SIGKILL pid={}",
        pid
    );
}

/// Open the vault index file for a run. Uses Obsidian's URL scheme if the
/// user has entered an Obsidian vault name in Settings (configured via
/// `~/.basevault/config.json`'s `obsidian_vault_name`); otherwise falls
/// back to opening the run's vault dir in Finder.
///
/// We rely on the user-entered name rather than detecting `.obsidian/`
/// because `.obsidian/` alone doesn't guarantee Obsidian knows the vault
/// (Obsidian maintains its own list of registered vaults indexed by
/// name, not by path; `obsidian://open?path=…` isn't honored).
#[tauri::command]
fn open_vault_for_run(run_id: String) -> Result<(), String> {
    // Prefer the vault_dir recorded in run.json (authoritative; survives
    // migration). Fall back to vault_root()/run_id for runs that predate
    // the sidecar.
    let run_dir: PathBuf = find_run_json(&run_id)
        .and_then(|rj| {
            let text = std::fs::read_to_string(&rj).ok()?;
            let v: serde_json::Value = serde_json::from_str(&text).ok()?;
            let s = v.get("vault_dir").and_then(|x| x.as_str())?;
            Some(PathBuf::from(s))
        })
        .unwrap_or_else(|| vault_root().join(&run_id));

    if !run_dir.exists() {
        return Err(format!(
            "vault dir not found: {} (run {} may have been deleted or not yet written)",
            run_dir.display(),
            run_id
        ));
    }

    let vault = vault_root();
    let obsidian_vault_name = get_config()
        .get("obsidian_vault_name")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    if let Some(name) = obsidian_vault_name {
        // Obsidian resolves the file path from the vault ROOT it has
        // registered for this name — so we pass a path RELATIVE to
        // vault_root. Derived from run_dir so migrated runs whose on-disk
        // name differs from run_id still resolve.
        let rel = run_dir
            .strip_prefix(&vault)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| run_id.clone());
        let file_path = format!("{}/4-insights", rel);
        let url = format!(
            "obsidian://open?vault={}&file={}",
            url_encode(&name),
            url_encode(&file_path)
        );
        info!("open_vault_for_run: {}", url);
        let _ = std::process::Command::new("open").arg(&url).spawn();
    } else {
        info!(
            "open_vault_for_run: obsidian_vault_name not set — opening {} in Finder",
            run_dir.display()
        );
        let _ = std::process::Command::new("open").arg(&run_dir).spawn();
    }
    Ok(())
}

/// Reveal a path in the OS file manager. Used by the export success
/// modal's "Show in Finder" — `path` is the destination directory the
/// export was written into. macOS opens it in Finder; Linux uses
/// xdg-open (no file-selection equivalent, so we open the dir itself).
#[tauri::command]
fn reveal_in_finder(path: String) -> Result<(), String> {
    let p = PathBuf::from(&path);
    if !p.exists() {
        return Err(format!("path does not exist: {}", path));
    }
    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(not(target_os = "macos"))]
    let cmd = "xdg-open";
    std::process::Command::new(cmd)
        .arg(&p)
        .spawn()
        .map_err(|e| format!("{} {:?}: {}", cmd, p, e))?;
    Ok(())
}

/// Reveal the chat-conversation data root (`~/.basevault/chats`) in the
/// OS file browser. Backs the Settings → cache section "Open Chats"
/// button. Creates the directory first when absent: a fresh install has
/// no conversations yet, and the underlying [`reveal_in_finder`] errors
/// on a missing path.
#[tauri::command]
fn reveal_chats_dir() -> Result<(), String> {
    let p = chats_root();
    std::fs::create_dir_all(&p)
        .map_err(|e| format!("create_dir_all {:?}: {}", p, e))?;
    reveal_in_finder(p.to_string_lossy().into_owned())
}

// Obsidian setup is user-driven: the app can't create an Obsidian vault
// programmatically (the `obsidian://open?path=...` URL scheme only works
// for paths Obsidian already knows about). Users are expected to open
// Obsidian, add a vault pointing at `~/Documents/BaseVault/`, give it a
// name, and then enter that name in Settings. The `open_vault_for_run`
// command uses `obsidian://open?vault=<name>` which IS supported.

/// Resolve a run_id to its on-disk vault dir. Prefers the `vault_dir`
/// recorded in run.json (authoritative; survives migration); falls back
/// to `<vault_root>/<run_id>/` for runs that predate the sidecar.
fn run_vault_dir(run_id: &str) -> Option<PathBuf> {
    let recorded = find_run_json(run_id).and_then(|rj| {
        let text = std::fs::read_to_string(&rj).ok()?;
        let v: serde_json::Value = serde_json::from_str(&text).ok()?;
        v.get("vault_dir")
            .and_then(|x| x.as_str())
            .map(PathBuf::from)
    });
    let dir = recorded.unwrap_or_else(|| vault_root().join(run_id));
    if dir.exists() { Some(dir) } else { None }
}

#[derive(Serialize)]
struct TreeNode {
    name: String,
    rel_path: String,
    is_dir: bool,
    children: Vec<TreeNode>,
    /// Byte size on disk. Only populated for `0-inputs/*` files
    /// (ingested document sources where size matters at a glance).
    /// Other tree leaves carry their own count badge (facts, entities,
    /// patterns, …) and don't need byte sizes alongside.
    #[serde(skip_serializing_if = "Option::is_none")]
    size_bytes: Option<u64>,
}

/// Tree of files under a run's vault dir. Used by the middle pane in
/// run-view mode. Returns an empty list when the run hasn't been
/// written yet (e.g. mid-pipeline before the first artifact lands).
#[tauri::command]
fn list_run_tree(run_id: String) -> Vec<TreeNode> {
    let Some(log_dir) = run_log_dir(&run_id) else { return Vec::new() };
    let stages = log_dir.join("stages");
    let mut out: Vec<TreeNode> = Vec::new();

    // ── 0-inputs ─────────────────────────────────────────────────────────────
    // One entry per file under stages/00-ingestion/documents/. Each
    // leaf carries its on-disk byte size (the React tree renders it as
    // a kb/mb badge); the parent dir keeps the count in its label.
    let docs = stages.join("00-ingestion").join("documents");
    let mut input_children: Vec<TreeNode> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&docs) {
        let mut paths: Vec<PathBuf> = entries.flatten()
            .map(|e| e.path()).filter(|p| p.is_file()).collect();
        paths.sort();
        for p in paths {
            if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                if name.starts_with('.') {
                    continue;
                }
                let display_name = name.rsplit_once('.').map(|(stem, _)| stem).unwrap_or(name);
                let size = std::fs::metadata(&p).ok().map(|m| m.len());
                input_children.push(TreeNode {
                    name: title_case(display_name),
                    rel_path: format!("0-inputs/{}", name),
                    is_dir: false,
                    children: Vec::new(),
                    size_bytes: size,
                });
            }
        }
    }
    if !input_children.is_empty() {
        let n = input_children.len();
        out.push(TreeNode {
            name: format!("Inputs ({})", n),
            rel_path: "0-inputs".to_string(),
            is_dir: true,
            children: input_children,
            size_bytes: None,
        });
    }

    // ── 1-facts ──────────────────────────────────────────────────────────────
    // One entry per per-topic JSONL bucket. Count = lines in that JSONL.
    // Children sorted by count desc.
    let facts_dir = stages.join("01-extraction").join("facts");
    let mut fact_children: Vec<(usize, TreeNode)> = Vec::new();
    let mut total_facts = 0usize;
    if let Ok(entries) = std::fs::read_dir(&facts_dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            let Some(stem) = p.file_stem().and_then(|s| s.to_str()) else { continue };
            let n = std::fs::read_to_string(&p)
                .map(|t| t.lines().filter(|l| !l.trim().is_empty()).count())
                .unwrap_or(0);
            total_facts += n;
            fact_children.push((n, TreeNode {
                name: format!("{} ({})", title_case(stem), n),
                rel_path: format!("1-facts/{}.md", stem),
                is_dir: false,
                children: Vec::new(),
                size_bytes: None,
            }));
        }
    }
    if !fact_children.is_empty() {
        fact_children.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));
        out.push(TreeNode {
            name: format!("Facts ({})", total_facts),
            rel_path: "1-facts".to_string(),
            is_dir: true,
            children: fact_children.into_iter().map(|(_, n)| n).collect(),
            size_bytes: None,
        });
    }

    // ── 2-entities ───────────────────────────────────────────────────────────
    // One leaf per per-entity JSONL, grouped under a per-`entity_type`
    // node — the same shape facts get under per-topic nodes above
    // (group dir → `<bucket> (n)` → leaves), reusing the same
    // `(count, TreeNode)` + count-desc-then-name sort + `"{label} ({n})"`
    // badge idiom. The only structural delta vs facts: a fact's topic
    // *is* its file, while an entity's type is a field read from the
    // file's first line, so the type bucketing happens here in Rust.
    //
    // Two on-disk shapes share this dir:
    //
    //   (a) Stage 1 Phase 2 in-flight: multi-line mention stream — one
    //       line per (entity, fact) appended as extract LLM calls
    //       resolve. Each line carries `name` (LLM-emitted proper
    //       name) + `entity_type`; count = number of mention lines.
    //   (b) Post-Stage-2 Phase 1: single-line consolidated canonical
    //       record carrying `canonical_name` + `entity_type` +
    //       `evidence_fact_refs`; count = len(evidence_fact_refs).
    //
    // We discriminate by content (`canonical_name` field present) rather
    // than line count — a mention stream with exactly one mention line
    // would otherwise be mistaken for a canonical record and reported
    // with count=0 (no `evidence_fact_refs` field on a mention).
    // `entity_type` is present on the first line of both shapes; it
    // defaults to "other" (matching the pipeline's own default) when
    // absent or blank, so every entity lands in exactly one bucket.
    //
    // Entity leaf `rel_path` stays `2-entities/<stem>.md` regardless of
    // the type nesting: the React entity router regex and
    // `read_run_entity`'s `<id>.jsonl` join key off the bare stem, so
    // the type must NOT be encoded into the path. The per-type group
    // node carries a synthetic dir `rel_path` used only for
    // expand/collapse state; group dirs are never navigable.
    let ents_dir = stages.join("02-entities").join("entities");
    // (entity_type, Vec<(count, leaf)>) — linear find-or-insert; the
    // type set is tiny (person/place/org/concept/other) so a map is
    // overkill, and a Vec keeps insertion order out of the sort's way.
    let mut type_buckets: Vec<(String, Vec<(usize, TreeNode)>)> = Vec::new();
    let mut n_total = 0usize;
    if let Ok(entries) = std::fs::read_dir(&ents_dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            let Some(stem) = p.file_stem().and_then(|s| s.to_str()) else { continue };
            let text = std::fs::read_to_string(&p).unwrap_or_default();
            let lines: Vec<&str> = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .collect();
            let first = lines.first()
                .and_then(|l| serde_json::from_str::<serde_json::Value>(l).ok());
            let is_canonical = first.as_ref()
                .map(|v| v.get("canonical_name").is_some())
                .unwrap_or(false);
            let (display, count) = if is_canonical {
                // Canonical post-Stage-2 record. Title-case
                // canonical_name for display so common-noun entities
                // ("her", "daughter", "author") that the LLM returns
                // lowercase still read as proper names in the tree.
                // The on-disk canonical_id stays untouched.
                let v = first.as_ref().unwrap();
                let display = v.get("canonical_name")
                    .and_then(|x| x.as_str())
                    .map(title_case)
                    .unwrap_or_else(|| title_case(stem));
                let n = v.get("evidence_fact_refs")
                    .and_then(|x| x.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);
                (display, n)
            } else {
                // Stage 1 mention stream: one line per (entity, fact).
                // Display name from the first line's `name`; count =
                // number of mention lines on disk.
                let display = first.as_ref()
                    .and_then(|v| v.get("name")
                        .and_then(|x| x.as_str())
                        .map(title_case))
                    .unwrap_or_else(|| title_case(stem));
                (display, lines.len())
            };
            // Type from the first line of either shape; blank/absent →
            // "other" (the pipeline's own EntityRecord default), so the
            // bucket set never has an empty-string label.
            let ty = first.as_ref()
                .and_then(|v| v.get("entity_type").and_then(|x| x.as_str()))
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .unwrap_or("other")
                .to_string();
            let leaf = (count, TreeNode {
                name: format!("{} ({})", display, count),
                rel_path: format!("2-entities/{}.md", stem),
                is_dir: false,
                children: Vec::new(),
                size_bytes: None,
            });
            n_total += 1;
            match type_buckets.iter_mut().find(|(t, _)| *t == ty) {
                Some((_, v)) => v.push(leaf),
                None => type_buckets.push((ty, vec![leaf])),
            }
        }
    }
    if n_total > 0 {
        // Build one group dir per type; per-type count = number of
        // entities of that type (children.len()), consistent with
        // every other tree node's badge. Leaves within a type sort
        // count-desc-then-name; the type groups sort the same way.
        let mut type_children: Vec<(usize, TreeNode)> = type_buckets
            .into_iter()
            .map(|(ty, mut leaves)| {
                leaves.sort_by(|a, b| {
                    b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name))
                });
                let n = leaves.len();
                (n, TreeNode {
                    name: format!("{} ({})", title_case(&ty), n),
                    rel_path: format!("2-entities/__type__/{}", ty),
                    is_dir: true,
                    children: leaves.into_iter().map(|(_, l)| l).collect(),
                    size_bytes: None,
                })
            })
            .collect();
        type_children.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));
        out.push(TreeNode {
            name: format!("Entities ({})", n_total),
            rel_path: "2-entities".to_string(),
            is_dir: true,
            children: type_children.into_iter().map(|(_, n)| n).collect(),
            size_bytes: None,
        });
    }

    // ── 3-patterns ───────────────────────────────────────────────────────────
    // One entry per per-topic JSON under stages/03-patterns/patterns/.
    // Count = len(patterns_array).
    let pats_dir = stages.join("03-patterns").join("patterns");
    let mut pat_children: Vec<(usize, TreeNode)> = Vec::new();
    let mut total_patterns = 0usize;
    if let Ok(entries) = std::fs::read_dir(&pats_dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let Some(stem) = p.file_stem().and_then(|s| s.to_str()) else { continue };
            let n = std::fs::read_to_string(&p)
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                .and_then(|v| v.as_array().map(|a| a.len()))
                .unwrap_or(0);
            total_patterns += n;
            pat_children.push((n, TreeNode {
                name: format!("{} ({})", title_case(stem), n),
                rel_path: format!("3-patterns/{}.md", stem),
                is_dir: false,
                children: Vec::new(),
                size_bytes: None,
            }));
        }
    }
    if !pat_children.is_empty() {
        pat_children.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));
        out.push(TreeNode {
            name: format!("Patterns ({})", total_patterns),
            rel_path: "3-patterns".to_string(),
            is_dir: true,
            children: pat_children.into_iter().map(|(_, n)| n).collect(),
            size_bytes: None,
        });
    }

    // ── 4-insights / 5-actions ───────────────────────────────────────────────
    // Single virtual files exposed iff their phase markers exist.
    let count_payload = |path: &Path, key: &str| -> usize {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| v.get(key).and_then(|x| x.as_array()).map(|a| a.len()))
            .unwrap_or(0)
    };
    let insights_path = stages.join("04-insights").join("phase_1_marker.json");
    if insights_path.exists() {
        let n = count_payload(&insights_path, "cross_domain")
            + count_payload(&insights_path, "critical");
        out.push(TreeNode {
            name: format!("Insights ({})", n),
            rel_path: "4-insights.md".to_string(),
            is_dir: false,
            children: Vec::new(),
            size_bytes: None,
        });
    }
    let actions_path = stages.join("05-actions").join("phase_1_marker.json");
    if actions_path.exists() {
        let n = count_payload(&actions_path, "actions");
        out.push(TreeNode {
            name: format!("Actions ({})", n),
            rel_path: "5-actions.md".to_string(),
            is_dir: false,
            children: Vec::new(),
            size_bytes: None,
        });
    }

    out
}

/// Title-case a slug-style identifier ("daily-insights", "work_log") into
/// "Daily Insights", "Work Log". Idempotent on already-cased input.
fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut at_word_start = true;
    for c in s.chars() {
        if c == '-' || c == '_' {
            out.push(' ');
            at_word_start = true;
        } else if at_word_start {
            for upper in c.to_uppercase() {
                out.push(upper);
            }
            at_word_start = false;
        } else {
            out.push(c);
        }
    }
    out
}

/// Read a text file inside a run's vault dir. Path is relative to the
/// vault dir (`rel_path` from `list_run_tree`). Sandboxed: rejects
/// absolute paths and any segment that escapes the run's vault dir
/// (e.g. `..`). Caps the read at 4 MiB so a pathological file can't
/// blow up memory or hang the UI thread; truncates with a trailing
/// notice when the cap is hit. The cap is enforced via a bounded
/// reader, NOT a post-read string trim — the previous shape allocated
/// the whole file before trimming, which defeated the point.
///
/// Read a `0-inputs/<file>` source from the run dir's preprocessed
/// documents. The only kind of textual content the in-app path opens
/// from disk now — every other view (facts / entities / patterns /
/// insights / actions) is rendered in React from structured data
/// returned by the per-stage commands below (`read_run_facts`,
/// `read_run_entities`, `read_run_patterns`, …).
#[tauri::command]
fn read_run_file(run_id: String, rel_path: String) -> Result<String, String> {
    use std::io::Read;
    const MAX_BYTES: u64 = 4 * 1024 * 1024;
    let rel = PathBuf::from(&rel_path);
    if rel.is_absolute() {
        return Err(format!("rel_path must be relative, got {:?}", rel));
    }
    for comp in rel.components() {
        use std::path::Component;
        match comp {
            Component::Normal(_) => {}
            _ => return Err(format!("rel_path may not contain {:?}", comp)),
        }
    }
    let log_dir = run_log_dir(&run_id)
        .ok_or_else(|| format!("run dir not found for run {}", run_id))?;

    let Some(rest) = rel_path.strip_prefix("0-inputs/") else {
        return Err(format!(
            "read_run_file only serves 0-inputs/* paths now; rendered \
            views go through the per-stage structured-data commands. \
            Got: {}",
            rel_path,
        ));
    };
    let preproc_root = log_dir.join("stages").join("00-ingestion").join("documents");
    let preproc = preproc_root.join(rest);
    let canon_root = preproc_root
        .canonicalize()
        .map_err(|e| format!("canonicalize {:?}: {}", preproc_root, e))?;
    // Distinguish NotFound from other canonicalize failures so the
    // UI can show a friendly "file not available" rather than the
    // raw `canonicalize "/full/abs/path": No such file or directory
    // (os error 2)` shape, which leaks the absolute path into the
    // pane and reads as a crash. Anything else (permissions, IO)
    // keeps the diagnostic form.
    let canon_file = preproc.canonicalize().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            format!("file not available: {}", rest)
        } else {
            format!("canonicalize {:?}: {}", preproc, e)
        }
    })?;
    if !canon_file.starts_with(&canon_root) {
        return Err(format!("rel_path escapes its root: {:?}", canon_file));
    }
    let meta = std::fs::metadata(&canon_file)
        .map_err(|e| format!("stat {:?}: {}", canon_file, e))?;
    if !meta.is_file() {
        return Err(format!("{:?} is not a file", canon_file));
    }
    let truncated = meta.len() > MAX_BYTES;
    let f = std::fs::File::open(&canon_file)
        .map_err(|e| format!("open {:?}: {}", canon_file, e))?;
    let mut buf = Vec::with_capacity(std::cmp::min(meta.len(), MAX_BYTES) as usize);
    f.take(MAX_BYTES).read_to_end(&mut buf)
        .map_err(|e| format!("read {:?}: {}", canon_file, e))?;
    let text = String::from_utf8_lossy(&buf).into_owned();
    if truncated {
        return Ok(format!(
            "{}\n\n_… file truncated at {} MiB; open in an external editor for the full contents._",
            text,
            MAX_BYTES / (1024 * 1024)
        ));
    }
    Ok(text)
}

/// Locate a run's intermediate/ directory (parent of run.json).
fn run_log_dir(run_id: &str) -> Option<PathBuf> {
    find_run_json(run_id).and_then(|rj| rj.parent().map(|p| p.to_path_buf()))
}

#[derive(Serialize)]
struct EvidenceSpan {
    file_path: String,
    file_offset: u64,
    file_length: u64,
}

#[derive(Serialize)]
struct FactRecord {
    /// Stable per-topic-per-item-type id matching the anchor IDs the
    /// vault writer emits (e.g. "fact-1", "event-3"). The wikilink
    /// target `1-facts/<topic>.md#^fact-1` resolves to a heading with
    /// id="fact-1".
    id: String,
    topic: String,
    item_type: String,
    summary: String,
    /// Extraction confidence (0..1) carried straight from
    /// facts_by_topic.json. The frontend multiplies this by each
    /// pattern→fact match score to render a composite confidence on
    /// provenance rows. Defaults to 1.0 when missing so unweighted
    /// rows collapse to match-score-only.
    confidence: f64,
    evidence: Vec<EvidenceSpan>,
}

/// Read per-topic facts JSONL bucket files under
/// stages/01-extraction/facts/ for a run and return a flat list with
/// derived per-topic-per-item-type incremental IDs that match the
/// vault's heading anchors.
///
/// The same code path serves both the in-flight UI (during extraction,
/// each LLM call appends a line as it returns) and the canonical view
/// (post-Phase 3 sort). Returns an empty list when no JSONL files are
/// on disk — this is the "no marks to display" state, not an error.
#[tauri::command]
fn read_run_facts(run_id: String) -> Vec<FactRecord> {
    let Some(log_dir) = run_log_dir(&run_id) else { return Vec::new() };
    let facts_dir = log_dir.join("stages").join("01-extraction").join("facts");
    if !facts_dir.is_dir() {
        return Vec::new();
    }
    // Iterate topics in sorted (filename) order so per-topic counters
    // are deterministic across calls.
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(&facts_dir) {
        Ok(it) => it.flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("jsonl"))
            .collect(),
        Err(_) => return Vec::new(),
    };
    paths.sort();
    let mut out: Vec<FactRecord> = Vec::new();
    for path in paths {
        let topic = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let mut counters: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let item: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue, // skip torn / mid-write line; resume retries
            };
            // Per-topic JSONL uses the per-parent shorter schema:
            // "type" / "ref". Accept the legacy facts_by_topic.json
            // schema ("item_type" / "source_ref") too.
            let item_type = item
                .get("type")
                .or_else(|| item.get("item_type"))
                .and_then(|v| v.as_str())
                .unwrap_or("fact")
                .to_string();
            let counter = counters.entry(item_type.clone()).or_insert(0);
            *counter += 1;
            let id = format!("{}-{}", item_type, *counter);
            let summary = item
                .get("summary")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let confidence = item
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            let mut evidence: Vec<EvidenceSpan> = Vec::new();
            if let Some(ev_arr) = item.get("evidence").and_then(|v| v.as_array()) {
                for ev in ev_arr {
                    let file_path = ev
                        .get("file_path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let file_offset = ev
                        .get("file_offset")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let file_length = ev
                        .get("file_length")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    if file_path.is_empty() || file_length == 0 {
                        continue;
                    }
                    evidence.push(EvidenceSpan {
                        file_path,
                        file_offset,
                        file_length,
                    });
                }
            }
            if evidence.is_empty() {
                continue;
            }
            out.push(FactRecord {
                id: id.clone(),
                topic: topic.clone(),
                item_type,
                summary,
                confidence,
                evidence,
            });
        }
    }
    out
}

// ── Structured-data readers per stage ───────────────────────────────────────
//
// The in-app browse path renders rich UI (wikilinks, anchors, hover,
// citation chains, cross-stage navigation) directly from these
// structured payloads — no markdown intermediate. The vault export
// path (`obsidianRenderer.regenVault` JS-side) consumes the same
// payloads to produce portable Obsidian-flavored markdown for export.

/// Return the per-topic JSONL bucket file's facts as a list of dicts.
/// Empty list when the topic doesn't exist or hasn't been written to
/// yet (in-flight pre-extraction). Each fact dict has the on-disk
/// schema: type, summary, evidence[], entities[], topics[], tags[],
/// occurred_at, occurred_at_text, confidence, relation_candidate.
#[tauri::command]
fn read_run_facts_for_topic(run_id: String, topic: String) -> Vec<serde_json::Value> {
    let Some(log_dir) = run_log_dir(&run_id) else { return Vec::new() };
    let path = log_dir.join("stages").join("01-extraction").join("facts")
        .join(format!("{}.jsonl", topic));
    let Ok(text) = std::fs::read_to_string(&path) else { return Vec::new() };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// Return all per-topic facts as a {topic → list[fact dict]} map. Used
/// by views that need cross-topic awareness (entities cited-in-facts).
#[tauri::command]
fn read_run_facts_all(run_id: String) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    let Some(log_dir) = run_log_dir(&run_id) else { return serde_json::Value::Object(out) };
    let facts_dir = log_dir.join("stages").join("01-extraction").join("facts");
    if let Ok(entries) = std::fs::read_dir(&facts_dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("jsonl") { continue; }
            let Some(stem) = p.file_stem().and_then(|s| s.to_str()) else { continue };
            let Ok(text) = std::fs::read_to_string(&p) else { continue };
            let items: Vec<serde_json::Value> = text.lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect();
            out.insert(stem.to_string(), serde_json::Value::Array(items));
        }
    }
    serde_json::Value::Object(out)
}

/// Return the full entities output: { subject, entities[], relations[] }.
/// Reads the canonical aggregate at stages/02-entities/phase_3_marker.json.
/// Returns null when the entities stage hasn't finished yet.
#[tauri::command]
fn read_run_entities(run_id: String) -> serde_json::Value {
    let Some(log_dir) = run_log_dir(&run_id) else { return serde_json::Value::Null };
    let path = log_dir.join("stages").join("02-entities").join("phase_3_marker.json");
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(serde_json::Value::Null)
}

/// Return one entity's record from stages/02-entities/entities/<id>.jsonl.
///
/// The on-disk shape evolves with the pipeline: during Stage 1 Phase 2
/// the file is a multi-line per-mention stream (one line per
/// (entity, fact) pairing), and from Stage 2 Phase 1 onwards it's a
/// single-line consolidated canonical record. We discriminate by the
/// `canonical_name` field — the same key `list_run_tree` uses to pick
/// between mention-line counting and `evidence_fact_refs` counting,
/// so the run tree's count badge and the panel's `mention_count`
/// agree on every shape.
///
/// - Canonical (single line, has `canonical_name`): return verbatim.
/// - Mention stream (multi-line, no `canonical_name`): synthesize a
///   `_state: "consolidating"` record carrying the live mention list +
///   line-count `mention_count`. The panel renders an in-flight view
///   instead of misreading a lone mention dict as a canonical record
///   (which would leave `mention_count` undefined and the body empty
///   while the run tree's count badge already shows N).
#[tauri::command]
fn read_run_entity(run_id: String, entity_id: String) -> serde_json::Value {
    let Some(log_dir) = run_log_dir(&run_id) else { return serde_json::Value::Null };
    let path = log_dir.join("stages").join("02-entities").join("entities")
        .join(format!("{}.jsonl", entity_id));
    let Ok(text) = std::fs::read_to_string(&path) else {
        return serde_json::Value::Null;
    };
    let lines: Vec<serde_json::Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let Some(first) = lines.first() else { return serde_json::Value::Null };
    if first.get("canonical_name").is_some() {
        // Canonical post-Stage-2 record. The on-disk write is a single
        // atomic replace via .tmp+rename, so a well-formed file has
        // exactly one canonical line; LAST is defensive against any
        // future writer that appends instead of rewriting.
        return lines.into_iter().last().unwrap_or(serde_json::Value::Null);
    }
    let display_name = first
        .get("name")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let entity_type = first
        .get("entity_type")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let mut topics_seen: Vec<String> = Vec::new();
    for ln in &lines {
        if let Some(arr) = ln.get("topics").and_then(|x| x.as_array()) {
            for t in arr {
                if let Some(s) = t.as_str() {
                    if !topics_seen.iter().any(|x| x == s) {
                        topics_seen.push(s.to_string());
                    }
                }
            }
        } else if let Some(s) = ln.get("topic").and_then(|x| x.as_str()) {
            if !topics_seen.iter().any(|x| x == s) {
                topics_seen.push(s.to_string());
            }
        }
    }
    serde_json::json!({
        "_state": "consolidating",
        "canonical_name": display_name,
        "entity_type": entity_type,
        "mention_count": lines.len(),
        "topics": topics_seen,
        "mentions": lines,
        // Empty arrays so the panel's section gates (refs.length > 0,
        // aliases.length > 0) skip rendering canonical-only sections
        // without crashing on undefined.
        "evidence_fact_refs": [],
        "aliases": [],
    })
}

/// Return patterns for a single topic. Empty list when the topic file
/// doesn't exist (patterns stage in progress or no patterns produced).
/// Per-topic files live under stages/03-patterns/patterns/<TOPIC>.json;
/// the stage marker (phase_1_marker.json) sits at the stage root and
/// is naturally excluded by the path.
#[tauri::command]
fn read_run_patterns_for_topic(run_id: String, topic: String) -> Vec<serde_json::Value> {
    let Some(log_dir) = run_log_dir(&run_id) else { return Vec::new() };
    let path = log_dir
        .join("stages")
        .join("03-patterns")
        .join("patterns")
        .join(format!("{}.json", topic));
    let Ok(text) = std::fs::read_to_string(&path) else { return Vec::new() };
    serde_json::from_str::<Vec<serde_json::Value>>(&text).unwrap_or_default()
}

/// Return all per-topic patterns as a {topic → list[pattern dict]} map.
#[tauri::command]
fn read_run_patterns_all(run_id: String) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    let Some(log_dir) = run_log_dir(&run_id) else { return serde_json::Value::Object(out) };
    let pats_dir = log_dir.join("stages").join("03-patterns").join("patterns");
    if let Ok(entries) = std::fs::read_dir(&pats_dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("json") { continue; }
            let Some(stem) = p.file_stem().and_then(|s| s.to_str()) else { continue };
            let Ok(text) = std::fs::read_to_string(&p) else { continue };
            if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(&text) {
                out.insert(stem.to_string(), serde_json::Value::Array(arr));
            }
        }
    }
    serde_json::Value::Object(out)
}

/// Return the insights stage payload: { cross_domain[], critical[] }.
#[tauri::command]
fn read_run_insights(run_id: String) -> serde_json::Value {
    let Some(log_dir) = run_log_dir(&run_id) else { return serde_json::Value::Null };
    let path = log_dir.join("stages").join("04-insights").join("phase_1_marker.json");
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(serde_json::Value::Null)
}

/// Return the actions stage payload: { actions[] }.
#[tauri::command]
fn read_run_actions(run_id: String) -> serde_json::Value {
    let Some(log_dir) = run_log_dir(&run_id) else { return serde_json::Value::Null };
    let path = log_dir.join("stages").join("05-actions").join("phase_1_marker.json");
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(serde_json::Value::Null)
}

/// Materialize the rollup payload for a run on demand — feeds the
/// Details modal (issue #104) which renders per-stage outcome buckets,
/// retry chains, and per-call rows. Returns null when neither the
/// event log nor a legacy llm-stats.json have any data (deleted runs,
/// runs that died before any LLM call landed).
///
/// Issue #189 — llm-stats.json is no longer written. The materialized
/// payload is the canonical shape; same code path running and
/// finished, so the modal renders identical content for in-flight vs.
/// completed runs by construction.
///
/// Legacy fallback: very old runs may have llm-stats.json on disk
/// without an llm-calls.jsonl (pre-#165 dump format). That file gets
/// read only when materialization yields nothing.
///
/// On top of either shape, each call gets a `cached_now` boolean
/// stamped from a live filesystem check at the cache path — the
/// rollup's `cached` flag captures "served from cache at run time"
/// (historical), while `cached_now` answers "is the response still in
/// the cache right now?" so the UI's per-call cache-bust button can
/// show only for entries that still exist on disk.
#[tauri::command]
fn read_run_llm_stats(run_id: String) -> serde_json::Value {
    let Some(log_dir) = run_log_dir(&run_id) else { return serde_json::Value::Null };
    let materialized = materialize_llm_stats_cached(&log_dir);
    let mut data: serde_json::Value = if !materialized.is_null() {
        materialized
    } else {
        // Legacy fallback: pre-#165 runs may carry llm-stats.json
        // without an event log.
        let path = log_dir.join("llm-stats.json");
        match std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
        {
            Some(v) => v,
            None => return serde_json::Value::Null,
        }
    };
    // Overlay live token state for in-flight calls. The materializer
    // walks jsonl which only contains the one-shot stream_progress
    // event per call (TTFT marker); the running chars/3 estimates
    // flow through the stdout heartbeat into an in-memory map kept
    // by spawn_pipeline. Without this overlay the modal would show
    // null token counts on every in-flight row between begin and end.
    overlay_live_token_state(&mut data, &run_id);
    // Stamp cached_now per call. Read once: cache_root() reads $HOME
    // and per-call check is a metadata().is_ok() — so even at hundreds
    // of calls per run this is sub-ms.
    let croot = cache_root();
    if let Some(calls) = data.get_mut("calls").and_then(|v| v.as_array_mut()) {
        for c in calls.iter_mut() {
            let stage = c
                .get("stage")
                .and_then(|s| s.as_str())
                .unwrap_or("_unknown")
                .to_string();
            let key = c
                .get("cache_key")
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());
            let exists = match key {
                Some(k) if !k.is_empty() => {
                    let safe_bucket: String = stage
                        .chars()
                        .map(|ch| if ch.is_alphanumeric() || ch == '-' || ch == '_' { ch } else { '_' })
                        .collect();
                    croot.join(safe_bucket).join(format!("{}.json", k)).is_file()
                }
                _ => false,
            };
            if let Some(obj) = c.as_object_mut() {
                obj.insert(
                    "cached_now".to_string(),
                    serde_json::Value::Bool(exists),
                );
            }
        }
    }
    data
}

/// Delete one entry from the LLM prompt-hash cache. Used by the
/// per-call "remove from cache" button in the Details modal. Idempotent:
/// removing a key that doesn't exist returns Ok(false). Returns Ok(true)
/// when a file was actually deleted. The caller refreshes the modal's
/// `cached_now` for that row from the boolean.
///
/// `stage` is the per-call stage name written by the runner (matches
/// the cache_root subdir; non-alphanumeric chars sanitized to `_`).
/// `cache_key` is the sha256 hex digest from `llm_cache.compute_cache_key`,
/// stamped on the per-call stat record at lookup time.
/// Inner body of `bust_llm_cache_entry` — no AppHandle dep, callable
/// from tests. The Tauri-command wrapper below adds the event emit.
fn bust_llm_cache_entry_inner(stage: String, cache_key: String) -> Result<bool, String> {
    if cache_key.is_empty() {
        return Err("cache_key is empty".to_string());
    }
    // Mirror llm_cache._cache_path's sanitization: only [A-Za-z0-9_-]
    // survive in the bucket name. Stage is internal (set by
    // begin_stat_record) but defensive sanitization keeps the path
    // shell-safe and matches the Python writer byte-for-byte.
    let safe_bucket: String = stage
        .chars()
        .map(|ch| if ch.is_alphanumeric() || ch == '-' || ch == '_' { ch } else { '_' })
        .collect();
    // Defensive: refuse anything that could escape the cache root —
    // even though cache_key is sha256-hex in normal use, a tampered
    // run.json could push '..' here. Cheap to gate.
    if cache_key.contains('/') || cache_key.contains("..") {
        return Err("invalid cache_key".to_string());
    }
    let path = cache_root().join(safe_bucket).join(format!("{}.json", cache_key));
    if !path.is_file() {
        return Ok(false);
    }
    std::fs::remove_file(&path)
        .map_err(|e| format!("remove {}: {}", path.display(), e))?;
    info!("bust_llm_cache_entry: removed {}", path.display());
    Ok(true)
}

#[tauri::command]
fn bust_llm_cache_entry(
    app: tauri::AppHandle,
    stage: String,
    cache_key: String,
) -> Result<bool, String> {
    let removed = bust_llm_cache_entry_inner(stage, cache_key)?;
    // Same channel as wipe_llm_cache — listeners (Run-details modal)
    // refetch and pick up the new cached_now state without polling.
    let _ = app.emit("llm-cache-changed", ());
    Ok(removed)
}

/// Read a cache entry from disk and return its contents as a
/// pretty-printed JSON envelope (response, model, request_params,
/// computed_at, prompt_first_200_chars, …).
///
/// `llm_cache.store` writes the response as a JSON-stringified JSON
/// — so the raw file's `"response"` value is one long escaped line
/// (`"{\"items\":[...]}"`). For human-readable clipboard paste we
/// parse the envelope here, re-parse `response` as JSON when it's
/// valid JSON (the common case for structured-output stages), and
/// pretty-print the whole thing. Responses that aren't JSON (rare —
/// free-form text from an unstructured prompt) fall through as the
/// original string.
///
/// The per-call cache column's copy button uses this to put the entry
/// on the clipboard for ad-hoc debugging — replaces a hand-roll of
/// `cat ~/.basevault/cache/<stage>/<hash>.json | pbcopy | jq` plus a
/// second jq pass to un-escape the response.
///
/// Mirrors `bust_llm_cache_entry`'s arg shape so a single component
/// can wire either action against the same `(stage, cache_key)` row.
#[tauri::command]
fn read_llm_cache_entry(stage: String, cache_key: String) -> Result<String, String> {
    if cache_key.is_empty() {
        return Err("cache_key is empty".to_string());
    }
    // Same sanitization as bust_llm_cache_entry_inner — see that fn for
    // the why (matches llm_cache._cache_path byte-for-byte; refuses
    // traversal even though sha256-hex normally can't contain '/').
    let safe_bucket: String = stage
        .chars()
        .map(|ch| if ch.is_alphanumeric() || ch == '-' || ch == '_' { ch } else { '_' })
        .collect();
    if cache_key.contains('/') || cache_key.contains("..") {
        return Err("invalid cache_key".to_string());
    }
    let path = cache_root().join(safe_bucket).join(format!("{}.json", cache_key));
    if !path.is_file() {
        return Err(format!("cache entry not found: {}", path.display()));
    }
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("read {}: {}", path.display(), e))?;
    let mut envelope: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        // Cache file isn't JSON for some reason (corrupted, partial
        // write). Return the raw bytes — better than failing the copy.
        Err(_) => return Ok(raw),
    };
    if let Some(resp) = envelope.get_mut("response") {
        if let Some(s) = resp.as_str() {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                *resp = parsed;
            }
        }
    }
    serde_json::to_string_pretty(&envelope)
        .map_err(|e| format!("serialize {}: {}", path.display(), e))
}

/// Reset the main window to the ship-default size (1440×900).
/// Lives Rust-side so the JS layer doesn't need a separate
/// `core:window:allow-set-size` capability — the default capability
/// set already permits invoking arbitrary Tauri commands.
#[tauri::command]
fn reset_window_size(app: tauri::AppHandle) -> Result<(), String> {
    use tauri::LogicalSize;
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window not found".to_string())?;
    window
        .set_size(LogicalSize::new(1440.0, 900.0))
        .map_err(|e| format!("set_size: {}", e))?;
    info!("reset_window_size: set to 1440x900");
    Ok(())
}

/// Handle to the File ▸ Export Selected menu item, managed in app state
/// so the toggle command can reach it directly. `Menu::get` only walks
/// top-level items and won't descend into the File submenu, so a held
/// handle is the reliable way to mutate a nested item.
struct ExportMenuItem(tauri::menu::MenuItem<tauri::Wry>);

/// Enable/disable the File ▸ Export Selected menu item to mirror the
/// frontend run selection (export is meaningful only when at least one
/// selected run has finished). Called from a React effect.
#[tauri::command]
fn set_export_menu_enabled(item: State<'_, ExportMenuItem>, enabled: bool) {
    let _ = item.0.set_enabled(enabled);
}

/// Folder picker for the Export-selected-runs flow with the action
/// button labelled "Export" instead of the system default ("Open").
///
/// `tauri-plugin-dialog` wraps `rfd`, neither of which exposes
/// `NSOpenPanel.setPrompt:`, so JS-side `open({directory: true, ...})`
/// always shows "Open." We drive NSOpenPanel directly via objc2 to
/// label the button. Macros + features pulled in are macOS-only; on
/// other platforms the command is a stub returning an error so the
/// build still links.
#[tauri::command]
fn pick_export_dir(
    app: tauri::AppHandle,
    default_path: Option<String>,
) -> Result<Option<String>, String> {
    #[cfg(target_os = "macos")]
    {
        use objc2::rc::Retained;
        use objc2::MainThreadMarker;
        use objc2_app_kit::{NSModalResponseOK, NSOpenPanel};
        use objc2_foundation::{NSString, NSURL};
        use std::sync::mpsc;

        let (tx, rx) = mpsc::channel();
        app.run_on_main_thread(move || {
            let result: Result<Option<String>, String> = (|| {
                let mtm = MainThreadMarker::new()
                    .ok_or_else(|| "pick_export_dir: not on main thread".to_string())?;
                let panel: Retained<NSOpenPanel> = NSOpenPanel::openPanel(mtm);
                panel.setCanChooseDirectories(true);
                panel.setCanChooseFiles(false);
                panel.setAllowsMultipleSelection(false);
                panel.setCanCreateDirectories(true);
                panel.setPrompt(Some(&NSString::from_str("Export")));
                panel.setMessage(Some(&NSString::from_str("Export selected runs")));
                if let Some(p) = default_path.as_deref() {
                    let url = NSURL::fileURLWithPath_isDirectory(&NSString::from_str(p), true);
                    panel.setDirectoryURL(Some(&url));
                }
                let response = panel.runModal();
                if response == NSModalResponseOK {
                    Ok(panel.URL().and_then(|u| u.path()).map(|s| s.to_string()))
                } else {
                    Ok(None)
                }
            })();
            let _ = tx.send(result);
        })
        .map_err(|e| format!("pick_export_dir: schedule on main thread: {}", e))?;
        rx.recv()
            .map_err(|e| format!("pick_export_dir: channel recv: {}", e))?
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (app, default_path);
        Err("pick_export_dir: only implemented on macOS".to_string())
    }
}

/// Default destination for the Export-selected-runs flow. Lives at
/// `~/Documents/BaseVault/` (same root we use for vault output) but
/// not the same path — the picker dialog opens here on first click,
/// the user can navigate elsewhere. Created lazily if absent so the
/// dialog never opens at a missing path.
#[tauri::command]
fn export_default_dir() -> Result<String, String> {
    let dir = default_vault_root();
    if !dir.exists() {
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("create_dir_all {:?}: {}", dir, e))?;
        info!("export_default_dir: created {}", dir.display());
    }
    Ok(dir.to_string_lossy().into_owned())
}

/// Recursive directory copy with optional wiki-link rewrite. Walks
/// `src` and mirrors every file + subdirectory into `dst`. Skips
/// dotfiles to mirror the on-disk view the user sees in the file
/// tree (no .DS_Store, no .obsidian).
///
/// When `link_rewrite` is `Some((from, to))` and `from != to`, every
/// `.md` file is read, every occurrence of `[[<from>` is replaced by
/// `[[<to>` (covering the `[[run_id/...]]`, `[[run_id|...]]`, and
/// `[[run_id]]` shapes), and the rewritten text is written to the
/// target. The pipeline emits wiki-links prefixed with the run_id
/// (`vault_dir.name`); without this rewrite, exports renamed to an
/// alias would have all internal links broken in Obsidian.
fn copy_dir_recursive(
    src: &Path,
    dst: &Path,
    link_rewrite: Option<(&str, &str)>,
) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    let rewrite = link_rewrite.filter(|(from, to)| from != to);
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        if name.starts_with('.') {
            continue;
        }
        let target = dst.join(&name);
        if path.is_dir() {
            copy_dir_recursive(&path, &target, link_rewrite)?;
        } else if path.is_file() {
            if let Some((from, to)) = rewrite {
                if name.ends_with(".md") {
                    let text = std::fs::read_to_string(&path)?;
                    let needle = format!("[[{}", from);
                    let replacement = format!("[[{}", to);
                    let rewritten = text.replace(&needle, &replacement);
                    std::fs::write(&target, rewritten)?;
                    continue;
                }
            }
            std::fs::copy(&path, &target)?;
        }
    }
    Ok(())
}

/// Export a run's vault dir to a user-chosen destination. Used by
/// the bottom-bar Export action in the runs pane: the JS side runs
/// the Tauri dialog open({directory: true, defaultPath: ...}) to
/// pick a destination, then calls this command per selected run_id.
/// `label` is the cosmetic name (alias if set, else short_id) the
/// JS side passes through so the exported subdirectory carries a
/// human-readable name instead of the bare run_id.
/// Resolve the export target's leaf directory name. Same sanitization
/// path as `export_run`, factored out so `export_target_path` can
/// pre-check existence and ask the user before clobbering. Sanitize:
/// replace path separators + leading dots so a malicious alias like
/// "../../etc" can't escape `dest_root`.
fn export_leaf(run_id: &str, label: Option<&str>) -> String {
    let leaf = label
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| run_id.to_string());
    let sanitized: String = leaf
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' => '_',
            _ => c,
        })
        .collect::<String>()
        .trim_start_matches('.')
        .to_string();
    if sanitized.is_empty() { run_id.to_string() } else { sanitized }
}

/// Returns the absolute path the `Export` action would write to AND
/// whether something already lives there. Lets the JS side prompt the
/// user before the (potentially destructive) overwrite. The actual
/// `export_run` call still re-checks server-side so a concurrent
/// filesystem change between the prompt and the copy can't silently
/// clobber.
#[tauri::command]
fn export_target_path(
    run_id: String,
    dest: String,
    label: Option<String>,
) -> Result<(String, bool), String> {
    let dest_root = PathBuf::from(&dest);
    if !dest_root.is_dir() {
        return Err(format!("destination not a directory: {}", dest));
    }
    let final_leaf = export_leaf(&run_id, label.as_deref());
    let target = dest_root.join(&final_leaf);
    let exists = target.exists();
    Ok((target.display().to_string(), exists))
}

#[tauri::command]
fn export_run(
    run_id: String,
    dest: String,
    label: Option<String>,
    overwrite: bool,
) -> Result<(), String> {
    let src = run_vault_dir(&run_id)
        .ok_or_else(|| format!("vault dir not found for run {}", run_id))?;
    let dest_root = PathBuf::from(&dest);
    if !dest_root.is_dir() {
        return Err(format!("destination not a directory: {}", dest));
    }
    let final_leaf = export_leaf(&run_id, label.as_deref());
    let target = dest_root.join(&final_leaf);
    // Same-path case: when the user picks the vault root as the
    // export destination, regenVault already materialized the run dir
    // exactly where export_run would copy it. The pre-flight EXISTS
    // prompt fired BEFORE regen so it didn't catch this; without the
    // check below, export_run would return EXISTS (overwrite=false)
    // or wipe-and-fail (overwrite=true → remove_dir_all eats the src).
    // No-op success: the data is already where the user wants it.
    if let (Ok(s), Ok(t)) = (src.canonicalize(), target.canonicalize()) {
        if s == t {
            info!(
                "export_run: src == target ({}) — no-op (vault dir already at dest)",
                s.display()
            );
            return Ok(());
        }
    }
    if target.exists() {
        if !overwrite {
            info!(
                "export_run: target {} exists, overwrite=false, returning EXISTS",
                target.display()
            );
            return Err(format!("EXISTS:{}", target.display()));
        }
        // Replace semantics: wipe the old subdir before copying. Without
        // this, copy_dir_recursive merges new files over old ones and
        // leaves stale files behind that exist in the prior export but
        // not in this run's vault.
        std::fs::remove_dir_all(&target)
            .map_err(|e| format!("remove existing {:?}: {}", target, e))?;
    }
    info!("export_run: {} → {}", src.display(), target.display());
    copy_dir_recursive(&src, &target, Some((&run_id, &final_leaf)))
        .map_err(|e| format!("copy {:?} → {:?}: {}", src, target, e))?;
    Ok(())
}

/// Check that an Obsidian vault has been set up at the given path by
/// looking for a `.obsidian/` subdirectory (Obsidian creates this when
/// it first opens a folder as a vault). If `path` is empty/unprovided,
/// uses the default vault root.
#[tauri::command]
fn check_obsidian_vault(path: Option<String>) -> bool {
    let base = path
        .and_then(|s| {
            let trimmed = s.trim().to_string();
            if trimmed.is_empty() { None } else { Some(PathBuf::from(trimmed)) }
        })
        .unwrap_or_else(default_vault_root);
    let marker = base.join(".obsidian");
    let exists = marker.is_dir();
    info!("check_obsidian_vault: {} → {}", marker.display(), exists);
    exists
}

/// Validate a Tinfoil API key by making a raw authenticated HTTPS GET
/// to a router enclave's `/v1/models` endpoint. Returns Ok if the call
/// returns 200 with a `data` array; Err otherwise.
///
/// This is "does the key authenticate?" only — explicitly NOT the trust
/// chain. The Tinfoil SDK gates inference on a Sigstore + measurement
/// verification chain that takes seconds; constructing `TinfoilAI()`
/// runs the full chain even before any chat call. For Settings + Wizard
/// save flows we want only the auth answer, so we bypass the SDK and
/// talk to the router's OpenAI-compat surface directly via stdlib.
/// The upper-right Attest button owns the trust-chain answer.
///
/// `key` is optional: when caller passes a typed key it overrides
/// dotenv for this call (Settings/Wizard pre-save verify of a new key);
/// when empty or omitted we read `TINFOIL_API_KEY` from the user
/// dotenv (re-verify of the on-file key).
#[tauri::command]
async fn verify_tinfoil_key(app: tauri::AppHandle, key: Option<String>) -> Result<(), String> {
    let typed = key.unwrap_or_default().trim().to_string();
    // No typed key → re-verify whatever the run would actually use:
    // user key first, bundled key (#609) as fallback.
    let key = if typed.is_empty() {
        user_tinfoil_key(&app).or_else(bundled_tinfoil_key).unwrap_or_default()
    } else {
        typed
    };
    if key.is_empty() {
        return Err("key is empty".into());
    }
    let py_bin = python_bin(&app)?;
    let py_root = python_root_dir(&app)?;

    let key_clone = key.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        // Stdlib-only: fetch one router enclave address, then POST a
        // shape-invalid chat completion (empty messages) so the auth
        // gate fires without spending a token. 401 → bad key; any
        // other response → auth gate passed.
        //
        // `/v1/models` looks tempting but the routers don't gate it
        // behind auth — it returns 200 even with no/bad bearer, so
        // it can't answer "does this key authenticate". The chat
        // endpoint's auth check runs before payload validation, so
        // we can short-circuit with an empty messages list.
        let script = r#"
import os, sys, json, random, traceback, urllib.request, urllib.error
key = os.environ["TINFOIL_API_KEY"]
try:
    with urllib.request.urlopen("https://atc.tinfoil.sh/routers", timeout=10) as r:
        routers = json.loads(r.read())
    if not isinstance(routers, list) or not routers:
        print("ERROR: no Tinfoil routers available", file=sys.stderr)
        sys.exit(1)
    enclave = random.choice(routers)
    body = json.dumps({"model": "gemma4-31b", "messages": []}).encode()
    req = urllib.request.Request(
        f"https://{enclave}/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        # 2xx — auth gate passed and our intentionally-empty body
        # somehow worked. Treat as ok.
        print("ok")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("ERROR: 401 unauthorized", file=sys.stderr)
            sys.exit(1)
        # 4xx post-auth (e.g. 400 "messages list empty") means the
        # auth gate already accepted the key. The request was wrong
        # on purpose; the answer we care about is "did auth pass".
        print("ok")
except Exception as e:
    # ERROR line first so Rust's substring classifier still sees the
    # short summary on stderr[0]; full traceback follows so the UI
    # <details> expander can render the call chain. Both are tailed
    # together as the error message.
    print(f"ERROR: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"#;

        info!("verify_tinfoil_key: spawning python verify…");
        let output = std::process::Command::new(&py_bin)
            .arg("-c")
            .arg(script)
            .current_dir(&py_root)
            .env("TINFOIL_API_KEY", &key_clone)
            .output()
            .map_err(|e| format!("spawn python: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let stderr_trim = stderr.trim();
            error!("verify_tinfoil_key: failed: {}", stderr_trim);
            let lower = stderr_trim.to_lowercase();
            let summary = if lower.contains("401") || lower.contains("unauthorized") {
                "Invalid API key — check the key at tinfoil.sh and try again.".to_string()
            } else if lower.contains("connection")
                || lower.contains("resolve")
                || lower.contains("timeout")
                || lower.contains("name or service")
            {
                "Couldn't reach Tinfoil — check your network.".to_string()
            } else {
                // Fall back to the script's "ERROR: ..." first line.
                stderr_trim
                    .lines()
                    .next()
                    .unwrap_or(stderr_trim)
                    .trim_start_matches("ERROR: ")
                    .to_string()
            };
            // Ship summary + full sanitized stderr (traceback or
            // multi-line classifier context) separated by a blank line.
            // The frontend splits on the first "\n\n" to render the
            // summary on its own and the rest in a <details> block.
            let sanitized = sanitize_home_paths(stderr_trim);
            let has_trace = sanitized.lines().count() > 1;
            return Err(if has_trace {
                format!("{}\n\n{}", summary, sanitized)
            } else {
                summary
            });
        }

        info!("verify_tinfoil_key: ok");
        Ok(())
    })
    .await
    .map_err(|e| format!("verify task panicked: {}", e))?
}

/// Run remote attestation for the currently-configured TEE provider+model.
/// Returns an `AttestationResult`-shaped JSON object built by
/// `engine.attestation_view` from the kernel provider's attestation output.
///
/// This command is the single entrypoint for sanctioned attestation
/// call sites #1 and #2: the frontend invokes it for #1 (app-startup
/// verify, on mount + after wizard finish — see App.jsx
/// `refreshAttestation`) and #2 (the Settings re-check button). The
/// hourly background re-attest (sanctioned site #3, in `run()`'s
/// setup) reuses this exact path. Attestation runs from exactly these
/// three places and nowhere else; the pipeline / sidecar / inference
/// paths do not attest (per-request attestation is intrinsic to the
/// Tinfoil SDK client they build).
///
/// Reads `tee_provider` + `tee_model` from `~/.basevault/config.json` (the
/// app config). API keys come from the user dotenv via `parse_dotenv`.
/// Spawns Python (matches `verify_tinfoil_key`'s shape) to run the
/// same code path the pipeline uses at inference time.
///
/// The Python script returns a JSON object `{provider, model, ok,
/// fingerprint, error, ts, constituents}`. On any spawn / parse failure
/// we synthesize an `ok: false` result so the UI can render the failure
/// state without crashing.
///
/// `constituents` carries one entry per TEE model the current config routes
/// the pipeline + chat to (resolved by `engine.attestation_view` from the
/// kernel `ExecutionEnv`'s registered specs): each is the per-model
/// attestation result. Top-level `ok` is the AND across constituents. This
/// result is a non-blocking visibility surface — nothing gates cloud mode or
/// chat on it (the real per-connection guarantee lives in the kernel provider).
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
struct EnclaveAttestation {
    host: String,
    #[serde(default)]
    predicate: Option<String>,
    #[serde(default)]
    live_measurement: Option<String>,
    // TDX RTMR2 (the second runtime measurement register); None for SEV-SNP,
    // which has a single measurement.
    #[serde(default)]
    live_measurement2: Option<String>,
    #[serde(default)]
    published_measurement: Option<String>,
    #[serde(default)]
    published_measurement2: Option<String>,
    #[serde(default)]
    tls_key_fp: Option<String>,
    #[serde(default)]
    hpke_key: Option<String>,
    #[serde(default)]
    raw_quote_b64gz: Option<String>,
    #[serde(default)]
    raw_quote_hex: Option<String>,
    #[serde(default)]
    release_repo: Option<String>,
    #[serde(default)]
    release_tag: Option<String>,
    #[serde(default)]
    live_url: Option<String>,
    #[serde(default)]
    release_url: Option<String>,
    #[serde(default)]
    r#match: bool,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
struct ConstituentAttestation {
    provider: String,
    model: String,
    ok: bool,
    fingerprint: Option<String>,
    error: Option<String>,
    ts: f64,
    // Recoverable TUF-race blip vs a real verification failure — see
    // attestation.AttestationResult.transient. Drives the soft
    // "re-checking…" UI state instead of the red failure banner.
    #[serde(default)]
    transient: bool,
    // Forensic context for failure paths — None on success. Trace is
    // a sanitized Python stack from the SDK construct / auth probe;
    // doc_steps is the per-step status snapshot when the SDK gate
    // failed without raising (security_verified=False). Both flow
    // through to the UI's <details> traceback block.
    #[serde(default)]
    traceback: Option<String>,
    #[serde(default)]
    doc_steps: Option<serde_json::Value>,
    #[serde(default)]
    deployment_tag: Option<String>,
    #[serde(default)]
    model_repo: Option<String>,
    #[serde(default)]
    enclaves: Vec<EnclaveAttestation>,
    // Pipeline role(s) this model serves (extract / patterns /
    // embeddings / …), from the active per-stage routing. Empty for a
    // registered-but-unused backend. Drives the health view's role
    // label so a model name reads meaningfully.
    #[serde(default)]
    roles: Vec<String>,
    // Failure kind: "enclave_down" / "attestation_mismatch" /
    // "router_down" / "auth". None when ok. Lets the UI tell apart
    // causes with different remedies.
    #[serde(default)]
    failure_class: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct AttestationResult {
    provider: String,
    model: String,
    ok: bool,
    fingerprint: Option<String>,
    error: Option<String>,
    ts: f64,
    // True only for the recoverable sigstore TUF symlink-race class
    // (retry budget exhausted). The UI renders this soft + auto-
    // rechecks; ``ok`` is still false so gating is unchanged.
    #[serde(default)]
    transient: bool,
    // Top-level forensic mirrors of the first-failed constituent's
    // fields. Per-constituent values still ride on `constituents[]`
    // for batches; the UI prefers the top-level pair so it has one
    // failure to surface without picking through the vec.
    #[serde(default)]
    traceback: Option<String>,
    #[serde(default)]
    doc_steps: Option<serde_json::Value>,
    #[serde(default)]
    constituents: Vec<ConstituentAttestation>,
    // Router enclave (same across constituents in one verify).
    #[serde(default)]
    router: Option<EnclaveAttestation>,
    // Top-level per-model fields populated only for single-constituent
    // results. Multi-constituent results (more than one registered
    // model) leave these None — UI iterates constituents.
    #[serde(default)]
    deployment_tag: Option<String>,
    #[serde(default)]
    model_repo: Option<String>,
    #[serde(default)]
    enclaves: Vec<EnclaveAttestation>,
    // Top-level mirror of the first-failed constituent's failure class
    // for the summary banner. Per-constituent classes still ride on
    // `constituents[]` for the health view.
    #[serde(default)]
    failure_class: Option<String>,
}

#[tauri::command]
async fn verify_attestation(
    app: tauri::AppHandle,
    provider: Option<String>,
    model: Option<String>,
    api_key: Option<String>,
) -> Result<AttestationResult, String> {
    let py_bin = python_bin(&app)?;
    let py_root = python_root_dir(&app)?;

    // Provider + model resolution: explicit args from the caller win (so
    // Settings can attest the staged radio choice before save). When
    // omitted (App.jsx mount, post-wizard, post-save callsites), fall
    // back to config.json — that's the persisted truth and matches what
    // the pipeline will actually use on the next run.
    let cfg = read_config_json();
    let provider = provider
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            cfg.get("tee_provider")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "tinfoil".to_string());
    let model = model
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            cfg.get("tee_model")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "gpt-oss-120b".to_string());

    // Baseline: user key first, bundled key (#609) as fallback. An
    // explicit api_key from the caller (Settings' verify-typed-key
    // flow) overrides for the selected provider so the user can attest
    // a freshly-typed key before saving. Empty/missing → Python returns
    // "API key not set".
    let mut tinfoil_key = user_tinfoil_key(&app).or_else(bundled_tinfoil_key).unwrap_or_default();
    if let Some(k) = api_key.filter(|s| !s.trim().is_empty()) {
        if provider == "tinfoil" {
            tinfoil_key = k;
        }
    }

    let provider_clone = provider.clone();
    let model_clone = model.clone();

    tauri::async_runtime::spawn_blocking(move || -> Result<AttestationResult, String> {
        // Inline Python: ask the kernel's attested provider (via the
        // transport-free engine view) to attest every TEE backend the current
        // config routes to, and print the AttestationResult JSON. The engine
        // does NO transport itself — the kernel provider owns the whole trust
        // path — so this is a read-only visibility surface. Stdlib-only on the
        // script side.
        let script = r#"
import json, os, time
provider = os.environ["BASEVAULT_TEE_PROVIDER"]
model = os.environ["BASEVAULT_TEE_MODEL"]
if provider == "tinfoil":
    from engine.attestation_view import attest_tinfoil_pipeline
    print(json.dumps(attest_tinfoil_pipeline()))
else:
    print(json.dumps({
        "provider": provider, "model": model, "ok": False,
        "fingerprint": None,
        "error": f"unsupported provider for attestation: {provider}",
        "ts": time.time(), "transient": False, "constituents": [],
    }))
"#;

        info!(
            "verify_attestation: spawning python (provider={}, model={})",
            provider_clone, model_clone
        );
        let mut cmd = std::process::Command::new(&py_bin);
        cmd.arg("-c")
            .arg(script)
            .current_dir(&py_root)
            // `-c` doesn't put cwd on sys.path, so name the package root
            // explicitly for the `from engine.* import` lines above.
            .env("PYTHONPATH", py_root.to_string_lossy().into_owned())
            .env("BASEVAULT_TEE_PROVIDER", &provider_clone)
            .env("BASEVAULT_TEE_MODEL", &model_clone)
            .env("TINFOIL_API_KEY", &tinfoil_key)
            // Session dir + dev-toggle handoff (same scrub/set pattern
            // as pipeline + chatbot spawns). Scrub guarantees a
            // shell-exported `BASEVAULT_DEV_WIRE_CAPTURE` cannot force
            // capture ON for the attestation subprocess when the
            // Settings toggle is OFF — attestation traffic itself
            // doesn't ride the hooked httpx client today (sigstore /
            // TUF uses urllib), but keeping the scrub-by-default
            // invariant uniform across every Python spawn site
            // simplifies the security review.
            .env("BASEVAULT_SESSION_DIR", session_dir())
            .env_remove("BASEVAULT_DEV_WIRE_CAPTURE");
        if dev_wire_capture_on() {
            cmd.env("BASEVAULT_DEV_WIRE_CAPTURE", "1");
        }
        let output = cmd
            .output()
            .map_err(|e| format!("spawn python: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if !output.status.success() {
            // Python crashed before printing anything. Synthesize a
            // failed result with the stderr tail so the UI shows the
            // real cause (e.g. ImportError on cryptography).
            error!(
                "verify_attestation: python exited non-zero. stderr={}",
                stderr.trim()
            );
            // Python crashed before printing JSON — its own
            // unhandled-exception traceback is already on stderr. Keep
            // the friendly first line in `error` (so the UI summary
            // stays short) and ship the full sanitized stderr as the
            // traceback for the <details> expander. Sanitize home
            // paths so the surfaced trace is safe to paste/share, same
            // posture as the python-side _exception_dict.
            let stderr_sanitized = sanitize_home_paths(stderr.trim());
            let summary = tail_lines(&stderr_sanitized, 1);
            let trace = if stderr_sanitized.lines().count() > 1 {
                Some(stderr_sanitized.clone())
            } else {
                None
            };
            return Ok(AttestationResult {
                provider: provider_clone,
                model: model_clone,
                ok: false,
                fingerprint: None,
                error: Some(format!("python exit: {}", summary)),
                ts: 0.0,
                // A hard python crash (ImportError, OOM, segfault) is
                // not the known TUF-race transient — surface it red.
                transient: false,
                traceback: trace,
                doc_steps: None,
                constituents: Vec::new(),
                router: None,
                deployment_tag: None,
                model_repo: None,
                enclaves: Vec::new(),
                // A hard python crash (ImportError, OOM) means the whole
                // verification path is unusable, not a specific enclave.
                failure_class: Some("router_down".to_string()),
            });
        }
        // Python succeeded but if stdout doesn't parse, surface that.
        let result: AttestationResult = serde_json::from_str(stdout.trim())
            .map_err(|e| format!("parse python stdout as JSON: {} (stdout={:?})", e, stdout))?;
        if result.ok {
            info!(
                "verify_attestation: ok provider={} model={} fp={:?}",
                result.provider, result.model, result.fingerprint
            );
        } else {
            // Don't error-log a normal "no key set" — happens on first
            // run before the wizard. Info-log is enough; UI displays it.
            info!(
                "verify_attestation: failed provider={} model={} error={:?}",
                result.provider, result.model, result.error
            );
        }
        Ok(result)
    })
    .await
    .map_err(|e| format!("verify task panicked: {}", e))?
}

// ── User-facing app config (inputs list, UI state) ──────────────────────────
//
// Persisted at ~/.basevault/config.json so the app restarts into the same
// state the user left it in (last-loaded inputs, eventually more). Opaque
// JSON: the shape is owned by the UI, Rust just round-trips.

fn config_path() -> PathBuf {
    state_root().join("config.json")
}

/// Load the persisted config. Returns `{}` when the file is missing or
/// unparseable (best-effort — we don't want a corrupt config to brick the
/// app; the UI re-saves on the next change).
#[tauri::command]
fn get_config() -> serde_json::Value {
    let p = config_path();
    match std::fs::read_to_string(&p) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
            warn!("get_config: {} unparseable ({}); returning empty", p.display(), e);
            serde_json::json!({})
        }),
        Err(_) => serde_json::json!({}),
    }
}

fn write_config_atomic(config: &serde_json::Value) -> Result<(), String> {
    let p = config_path();
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("mkdir {:?}: {}", parent, e))?;
    }
    let pretty = serde_json::to_string_pretty(config)
        .map_err(|e| format!("serialize config: {}", e))?;
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, pretty + "\n").map_err(|e| format!("write tmp: {}", e))?;
    std::fs::rename(&tmp, &p).map_err(|e| format!("rename config: {}", e))?;
    Ok(())
}

/// Save the config. Atomic write (tmp + rename). Whole-object replace —
/// callers that only want to change a subset of keys must use
/// `update_config` instead, or they race other persisters (see below).
#[tauri::command]
fn set_config(config: serde_json::Value) -> Result<(), String> {
    write_config_atomic(&config)
}

/// Serializes the read-merge-write cycle in `update_config` so two
/// concurrent partial writers (export prefs, import dir, inputs list,
/// run aliases) can't lost-update each other.
static CONFIG_WRITE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// Atomically merge `patch`'s top-level keys into the persisted config,
/// following JSON Merge Patch (RFC 7386) at the top level: a `null`
/// patch value DELETES that key rather than storing a literal null.
///
/// Every UI persister changes only a few keys but used to express that
/// as a client-side get_config → {...existing, k} → set_config cycle.
/// Two such cycles interleaving drops the key written by whichever read
/// its snapshot first. Doing the read, shallow-merge, and atomic write
/// here under a process-wide lock makes each partial write see every
/// prior one, so independent persisters compose instead of clobbering.
///
/// Invariant the null-deletes rule depends on: no config key
/// distinguishes an explicit null from an absent key — every reader
/// (Rust, JS, pipeline) treats null and missing identically — so
/// collapsing "set to null" into "delete" is observationally
/// equivalent and lets a partial writer drop a key it owns (e.g.
/// resetting categories to the seed default) through the same path.
#[tauri::command]
fn update_config(patch: serde_json::Value) -> Result<(), String> {
    let patch_obj = patch
        .as_object()
        .ok_or_else(|| "update_config: patch must be a JSON object".to_string())?;
    let _guard = CONFIG_WRITE_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|e| format!("update_config: config lock poisoned: {}", e))?;
    let p = config_path();
    let mut cur = std::fs::read_to_string(&p)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::json!({}));
    let obj = cur.as_object_mut().expect("filtered to object above");
    for (k, v) in patch_obj {
        if v.is_null() {
            obj.remove(k);
        } else {
            obj.insert(k.clone(), v.clone());
        }
    }
    write_config_atomic(&cur)
}

// ── Chatbot UI surface (issue #422) ────────────────────────────────────────────
//
// The Chatbot sidecar is a **persistent** process: spawned once per
// session and kept alive across turns so the ~2 s attested-client
// (TUF + ATC + enclave handshake) construction is paid once per
// process, not once per message. `chatbot` writes one newline-framed
// JSON request per turn over the kept-open stdin; a single long-lived
// reader thread drains stdout for the process's whole lifetime.
//
// Event flow (unchanged, still incremental — the streaming fix this
// builds on must survive the persistent framing):
//   sidecar stdout (JSON line)  →  Rust line reader  →  app.emit("chatbot-event", payload)  →  React listener
//
// Lifecycle:
//   - Lazy spawn on the first message of a session; `_warm_attested_client`
//     runs at sidecar startup so a freshly-spawned process is warm by
//     the time the first request arrives.
//   - Normal turn finish (`chatbot_done`/`chatbot_error`): the process
//     is NOT killed — it stays warm and loops to the next request.
//     This is where the entire P1 win comes from.
//   - User Stop: `chatbot_cancel` SIGTERM→SIGKILLs the pid via the
//     unchanged #457 mechanism (in-flight TEE request abandoned —
//     generation actually stops), clears the handle, and EAGERLY
//     re-spawns so the ~2 s re-warm overlaps the user's next compose.
//   - Unexpected death (segfault / OOM): the reader thread sees EOF,
//     synthesizes a `chatbot_error` for the active turn, clears the
//     handle; the next message transparently re-spawns + re-warms.
//
// Turn fencing: each request carries a monotonic `turn_id` echoed on
// every event, so a late event from a finished/cancelled turn cannot
// bleed into the next turn's UI — defense-in-depth alongside the React
// panel's own monotonic counter (see ChatbotHelper.jsx).

// Spawn the persistent sidecar, register its cancel token + I/O handle,
// and start the long-lived stdout reader + stderr drainer. The reader
// runs for the whole process lifetime: on EOF it reaps the child and,
// unless this was a user-Stop, synthesizes a `chatbot_error` for the
// active turn and clears the handle so the next message re-spawns.
/// Push a `run_available` frame to a live chatbot sidecar, if any, so
/// a session that started before any run existed can bind to this
/// just-finished run without restart or dropdown interaction (#780).
/// Called from `spawn_pipeline`'s wait thread on a clean run completion.
///
/// Two preconditions, both checked before any write:
///   - The completed run produced a non-empty vectors.db
///     (`chatbot_run_store` enforces the same non-empty predicate the
///     binding resolver uses). An aborted/empty-input run that finished
///     without embeddings is skipped — pushing a binding to nothing
///     would just be noise.
///   - A chatbot sidecar is currently alive. If none exists, the next
///     `spawn_chatbot_sidecar` will pick up the run via
///     `resolve_chatbot_binding` at spawn-time anyway.
///
/// Sidecar binding logic on receipt (in `_handle_run_available`):
///   - Unbound → bind + emit `chatbot_bound` (React's reducer refreshes
///     the selector so the new run is marked current).
///   - Already bound → no-op on binding (NO silent swap to a newer
///     run; preserves the user's settled binding). The new run still
///     appears in the dropdown via the existing `chatbot_list_runs`
///     fs scan; the user picks it manually if they want to switch.
///
/// Writes are serialised through the same `chatbot_proc.stdin` mutex
/// the `chatbot` command uses, so push frames cannot interleave with
/// turn-request bytes. The sidecar's stdin read loop is single-
/// threaded, so a push line buffered behind an in-flight turn is
/// processed after that turn completes — no interruption mid-stream.
fn push_run_available_to_chatbot(
    proc_slot: &Arc<Mutex<Option<ChatbotProc>>>,
    run_id: &str,
) {
    use std::io::Write;

    let Some(store) = chatbot_run_store(run_id) else {
        // Run finished without a usable vectors.db (empty input,
        // aborted before embeddings, etc). Skip silently — no UX
        // value in pushing a binding to nothing.
        return;
    };
    let mut g = proc_slot.lock().unwrap();
    let Some(proc) = g.as_mut() else {
        // No live sidecar; the next spawn picks up this run via
        // `resolve_chatbot_binding` at start.
        return;
    };
    if !is_pid_alive(proc.pid) {
        // Stale handle (process died but the reader thread hasn't
        // cleared the slot yet). Skip and let the next `chatbot`
        // command re-spawn fresh — that fresh sidecar will resolve
        // the same run at start.
        return;
    }
    let mut line = serde_json::json!({
        "kind": "run_available",
        "run_id": run_id,
        "store_path": store.to_string_lossy(),
        // Only the explicit dropdown pick (`chatbot_select_run`)
        // produces a "user" binding — that path respawns the
        // sidecar with a fresh env. A Rust auto-push is always the
        // default (most-recent-non-empty) source.
        "selection": "default",
    })
    .to_string();
    line.push('\n');
    let pid = proc.pid;
    if let Err(e) = proc
        .stdin
        .write_all(line.as_bytes())
        .and_then(|_| proc.stdin.flush())
    {
        warn!(
            "push_run_available_to_chatbot: write failed pid={} run_id={}: {}",
            pid, run_id, e
        );
    } else {
        info!(
            "push_run_available_to_chatbot: pushed pid={} run_id={}",
            pid, run_id
        );
    }
}

fn spawn_chatbot_sidecar(
    app: &tauri::AppHandle,
    state: &AppState,
) -> Result<(), String> {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    let py_bin = python_bin(app)?;
    let py_root = python_root_dir(app)?;
    let logs_root_s = logs_root().to_string_lossy().into_owned();
    let chats_root_s = chats_root().to_string_lossy().into_owned();
    let vault_root_s = vault_root().to_string_lossy().into_owned();

    let mut cmd = Command::new(&py_bin);
    cmd.arg("-m").arg("engine.chatbot_sidecar")
        .current_dir(&py_root)
        // Same env contract as `spawn_pipeline`: the sidecar reads
        // BASEVAULT_LOGS_ROOT to find the latest run's vectors.db.
        // BASEVAULT_AGENT=app keeps the resolution rule consistent
        // with the runner (logs/ vs logs-dev/). BASEVAULT_CHATS_ROOT
        // is the parallel for chat conversation data — chats live
        // OUTSIDE the logs tree; the sidecar's `_chats_root()` honours
        // this override exactly as `_logs_root()` honours
        // BASEVAULT_LOGS_ROOT.
        .env("BASEVAULT_LOGS_ROOT", &logs_root_s)
        .env("BASEVAULT_CHATS_ROOT", &chats_root_s)
        .env("BASEVAULT_VAULT_ROOT", &vault_root_s)
        .env("BASEVAULT_AGENT", "app")
        // Shared session dir + dev-toggle handoff (same pattern as
        // the pipeline subprocess). Scrub then conditionally set so a
        // shell-exported value can't force capture ON when the
        // Settings toggle is OFF.
        .env("BASEVAULT_SESSION_DIR", session_dir())
        .env_remove("BASEVAULT_DEV_WIRE_CAPTURE")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if dev_wire_capture_on() {
        cmd.env("BASEVAULT_DEV_WIRE_CAPTURE", "1");
    }

    // #565 per-conversation telemetry: point the sidecar's
    // llm-{calls,payloads}.jsonl at the ACTIVE conversation's own dir
    // under chats/. Unset (pre-list / the ad-hoc test path) → the
    // sidecar falls back to `_chats_root()`.
    if let Some(id) = state.chatbot_active_convo.lock().unwrap().clone() {
        // active_convo holds the IMMUTABLE ISO id; resolve it to the
        // current dir name (a rename changes the name, never the id).
        if let Some(name) = find_convo_dir(&id) {
            let cdir = convos_root().join(&name);
            let _ = std::fs::create_dir_all(&cdir);
            cmd.env(
                "BASEVAULT_CHATBOT_CONVO_DIR",
                cdir.to_string_lossy().as_ref(),
            );
        }
    }

    // #507 run selector: resolve the corpus binding here (the user's
    // explicit pick if still valid, else the most-recent-non-empty
    // default — one predicate, one ordering) and pass it in. The
    // sidecar re-validates non-empty and falls back to its own
    // `_latest_store_path` only when no binding env is set (the
    // ad-hoc/test path). No qualifying run → no env, unbound session
    // exactly as before the feature.
    if let Some((store, run_id, source)) = resolve_chatbot_binding(state) {
        cmd.env("BASEVAULT_CHATBOT_STORE_PATH", store.to_string_lossy().as_ref())
            .env("BASEVAULT_CHATBOT_RUN_ID", &run_id)
            .env("BASEVAULT_CHATBOT_BIND_SOURCE", source);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to start sidecar: {}", e))?;
    let pid = child.id();
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "sidecar stdin pipe missing".to_string())?;

    let cancelled = Arc::new(AtomicBool::new(false));
    {
        // The cancel token stays in `chatbot_inflight` so the proven
        // #457 SIGTERM→SIGKILL path (cancel_chatbot_inflight) is
        // byte-for-byte unchanged.
        let mut g = state.chatbot_inflight.lock().unwrap();
        *g = Some(ChatbotInflight {
            pid,
            cancelled: cancelled.clone(),
        });
    }
    {
        let mut g = state.chatbot_proc.lock().unwrap();
        *g = Some(ChatbotProc { pid, stdin });
    }

    // Stderr drainer — a tracebacked sidecar must not block on a full
    // stderr pipe. Logged at warn! so app.log captures the Python
    // trace; the user only sees the in-band `chatbot_error`.
    if let Some(stderr) = child.stderr.take() {
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().flatten() {
                warn!("chatbot sidecar stderr: {}", line);
            }
        });
    }

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "sidecar stdout pipe missing".to_string())?;
    let app_clone = app.clone();
    let inflight_slot = state.chatbot_inflight.clone();
    let proc_slot = state.chatbot_proc.clone();
    let turn_seq_slot = state.chatbot_turn_seq.clone();
    let cancelled_thread = cancelled.clone();

    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().flatten() {
            // One JSON object per line. Forward the parsed Value so
            // React can switch on `payload.event` without re-parsing.
            // Non-JSON lines (import warnings, telemetry overlay) are
            // dropped — the sidecar's stdout contract is JSON-only.
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                let _ = app_clone.emit("chatbot-event", v);
            }
        }

        // stdout hit EOF → the process is exiting/exited. Reap it so it
        // doesn't linger as a zombie.
        let _ = child.wait();
        let was_cancelled = cancelled_thread.load(Ordering::SeqCst);

        if was_cancelled {
            // User-initiated Stop. `chatbot_cancel` already told the UI
            // (`chatbot_stopped`), cleared the handle, and re-spawned a
            // fresh warm process — do not touch the slots (a clear here
            // would race the re-spawn) and do not synthesize an error.
            info!("chatbot sidecar stopped by user (pid {})", pid);
            return;
        }

        // Unexpected death. Synthesize a `chatbot_error` for whatever
        // turn was active so the UI doesn't hang waiting for
        // `chatbot_done`, then clear the handle so the next message
        // transparently re-spawns + re-warms. Only clear if the slot
        // still points at THIS pid (an eager re-spawn may already have
        // replaced it).
        // Fence the synth error to the last-issued turn (the global
        // counter's current value) — but only if this dead pid is
        // still the active proc (an eager re-spawn may already have
        // replaced it, in which case its turn is a newer id and must
        // not be error-fenced by this stale death).
        let active_turn = {
            let g = proc_slot.lock().unwrap();
            g.as_ref()
                .filter(|p| p.pid == pid)
                .map(|_| turn_seq_slot.load(Ordering::SeqCst))
        };
        error!("chatbot sidecar died unexpectedly (pid {})", pid);
        let mut payload = serde_json::json!({
            "event": "chatbot_error",
            "message": "the chat process stopped unexpectedly — \
                        your next message will restart it",
        });
        if let Some(t) = active_turn {
            payload["turn_id"] = serde_json::json!(t);
        }
        let _ = app_clone.emit("chatbot-event", payload);
        {
            let mut g = proc_slot.lock().unwrap();
            if g.as_ref().map(|p| p.pid) == Some(pid) {
                *g = None;
            }
        }
        {
            let mut g = inflight_slot.lock().unwrap();
            if g.as_ref().map(|a| a.pid) == Some(pid) {
                *g = None;
            }
        }
    });

    Ok(())
}

#[tauri::command]
fn chatbot(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    query: String,
    history: Vec<serde_json::Value>,
) -> Result<u64, String> {
    use std::io::Write;

    // Spawn lazily if there's no live persistent process (first message
    // of a session, or after an unexpected death the reader cleared).
    let need_spawn = {
        let g = state.chatbot_proc.lock().unwrap();
        g.as_ref().map_or(true, |p| !is_pid_alive(p.pid))
    };
    if need_spawn {
        spawn_chatbot_sidecar(&app, &state)?;
    }

    // Frame one turn: assign the next monotonic turn_id and write a
    // single newline-delimited JSON request over the kept-open stdin.
    // stdin is NOT closed — closing it is the shutdown signal.
    let write_one = |state: &AppState| -> std::io::Result<u64> {
        let mut g = state.chatbot_proc.lock().unwrap();
        let p = g.as_mut().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotConnected, "no sidecar")
        })?;
        // Process-global monotonic id — survives sidecar re-spawns so a
        // re-issued turn after a cancel→respawn (the resume path) gets a
        // strictly higher id than the generation it replaces. The
        // client fences on the returned id; a stale pre-respawn event
        // can never collide.
        let turn_id = state.chatbot_turn_seq.fetch_add(1, Ordering::SeqCst) + 1;
        // `kind: "turn"` is explicit alongside the #780 `run_available`
        // push frame the same pipe now carries. The sidecar defaults
        // an absent `kind` to `"turn"` for one-shipped-pair backward
        // compat with older sidecars; an explicit tag here is the
        // forward-compatible shape and makes the dispatch contract
        // symmetric across the two writers (this command + the
        // push helper).
        let mut line = serde_json::json!({
            "kind": "turn",
            "query": &query,
            "history": &history,
            "turn_id": turn_id,
        })
        .to_string();
        line.push('\n');
        p.stdin.write_all(line.as_bytes())?;
        p.stdin.flush()?;
        Ok(turn_id)
    };

    match write_one(&state) {
        Ok(turn_id) => Ok(turn_id),
        Err(_) => {
            // The pipe broke between the liveness check and the write
            // (the process died in that window). Clear the stale handle
            // and re-spawn once; a second failure is surfaced to the UI.
            {
                let mut g = state.chatbot_proc.lock().unwrap();
                *g = None;
            }
            spawn_chatbot_sidecar(&app, &state)?;
            write_one(&state).map_err(|e| {
                let _ = app.emit(
                    "chatbot-event",
                    serde_json::json!({
                        "event": "chatbot_error",
                        "message": format!("could not reach the chat process: {}", e),
                    }),
                );
                format!("sidecar write failed: {}", e)
            })
        }
    }
}

/// Terminate the in-flight Chatbot sidecar, if any. Returns whether one
/// was running. Pure state + libc — no AppHandle coupling so this is
/// unit-testable against a real spawned child.
fn cancel_chatbot_inflight(state: &AppState) -> bool {
    let pid = {
        let mut g = state.chatbot_inflight.lock().unwrap();
        match g.take() {
            Some(a) => {
                a.cancelled.store(true, Ordering::SeqCst);
                a.pid
            }
            None => return false,
        }
    };
    if is_pid_alive(pid) {
        // SIGTERM then SIGKILL — the sidecar's in-flight TEE request
        // is abandoned (connection drop); generation actually stops.
        stop_pid_with_grace(pid, std::time::Duration::from_secs(3));
    }
    true
}

#[tauri::command]
fn chatbot_cancel(app: tauri::AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    // `cancel_chatbot_inflight` is the unchanged #457 mechanism: it
    // SIGTERM→SIGKILLs the pid (the in-flight TEE request is abandoned —
    // generation actually stops) and clears the cancel token. With a
    // persistent process the kill also destroys the warm client, so we
    // additionally clear the I/O handle and EAGERLY re-spawn a fresh
    // sidecar: its ~2 s re-warm overlaps the user's next compose, so
    // the message after a Stop is still fast.
    if cancel_chatbot_inflight(&state) {
        {
            let mut g = state.chatbot_proc.lock().unwrap();
            *g = None;
        }
        // Tell the UI the turn stopped so it leaves the streaming
        // state while keeping whatever partial answer arrived.
        let _ = app.emit(
            "chatbot-event",
            serde_json::json!({ "event": "chatbot_stopped" }),
        );
        // Eager re-warm. Best-effort: if it fails, the next message
        // lazy-spawns instead (just without the overlap benefit).
        if let Err(e) = spawn_chatbot_sidecar(&app, &state) {
            warn!("chatbot: eager re-warm after cancel failed: {}", e);
        }
    }
    Ok(())
}

// ── Chatbot run selector (#507) ────────────────────────────────────────────
//
// The user explicitly picks which processed run's vectors.db a chat
// session answers from, fixing the corpus-blind binding. The old
// resolver bound whichever vectors.db FILE was touched last (an older
// run's db, written last, could shadow a newer run) and accepted 0-byte
// in-flight stores (its docstring claimed "non-empty" but only checked
// is_file). The selector lists the 10 most-recent runs that have a
// non-empty store, labelled by their ingested input + creation date
// (not the opaque run-dir slug — the load-bearing UX point). Picking a
// run re-spawns the persistent sidecar bound to it: a fresh pick is a
// fresh session, reusing the exact clear-handle + re-spawn machinery
// the Stop path already uses — the sidecar's persistent-lifecycle
// internals (#506) are not touched, only its single resolution input.
//
// ONE predicate, ONE ordering, used for BOTH the list and the
// no-explicit-selection default — the default is literally the most
// recent list entry — so list and default can never disagree (the
// "consistent by construction" gate-1 commitment). The Python
// `_latest_store_path` mirrors the same rule for the ad-hoc/test path.

/// `<run_dir>/stages/06-embeddings/vectors.db` iff it is a regular file
/// with content. A 0-byte file is an in-flight / aborted store
/// (embeddings never finished writing) and is excluded from the list
/// AND any default (#507 defect #2 — the non-empty guarantee callers
/// long claimed but never enforced).
fn chatbot_store_path(run_dir: &Path) -> Option<PathBuf> {
    let p = run_dir
        .join("stages")
        .join("06-embeddings")
        .join("vectors.db");
    let md = std::fs::metadata(&p).ok()?;
    (md.is_file() && md.len() > 0).then_some(p)
}

/// First 14 digits (`YYYYMMDDHHMMSS`) of an ISO-8601-UTC timestamp,
/// punctuation-agnostic so both the run-dir form (`2026-05-16T03-14-54Z`)
/// and the `created_at` form (`2026-05-16T03:14:54Z`) normalise to the
/// same comparable key. `None` if fewer than 14 digits (not a stamp).
fn iso_digits(s: &str) -> Option<String> {
    let d: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    (d.len() >= 14).then(|| d[..14].to_string())
}

/// A chronologically-sortable key for when the run was CREATED, not
/// when its db file was last touched. File-mtime is the direct cause of
/// the shadowing defect #507 fixes, so it is deliberately never used.
/// Source order: the run-dir name's ISO-8601-UTC prefix (authoritative
/// for run order); else `config.json`'s `created_at`; else `None` — a
/// run with no determinable creation time is excluded entirely rather
/// than allowed to sort arbitrarily and silently win the default.
fn chatbot_run_time_key(run_dir: &Path) -> Option<String> {
    let name = run_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let b = name.as_bytes();
    // Fixed-width `YYYY-MM-DDTHH-MM-SSZ` = 20 bytes. Validate the shape
    // so a slug that merely starts with digits can't masquerade as one.
    let looks_iso = b.len() >= 20
        && b[0..4].iter().all(u8::is_ascii_digit)
        && b[4] == b'-'
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[7] == b'-'
        && b[8..10].iter().all(u8::is_ascii_digit)
        && b[10] == b'T'
        && b[11..13].iter().all(u8::is_ascii_digit)
        && b[13] == b'-'
        && b[14..16].iter().all(u8::is_ascii_digit)
        && b[16] == b'-'
        && b[17..19].iter().all(u8::is_ascii_digit)
        && b[19] == b'Z';
    if looks_iso {
        return iso_digits(&name[0..20]);
    }
    // Malformed / legacy dir name → fall to the recorded created_at.
    let created = read_run_state(run_dir)?
        .get("created_at")?
        .as_str()?
        .to_string();
    iso_digits(&created)
}

/// Basename of a path-ish string (the part the user recognises),
/// falling back to the whole trimmed string.
fn path_basename(s: &str) -> String {
    Path::new(s.trim())
        .file_name()
        .and_then(|n| n.to_str())
        .map(str::to_string)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| s.trim().to_string())
}

/// The corpus the run ingested, as the user thinks of it: the input
/// file/folder name. Folder/vault ingestion → the folder name; one or
/// more input files → the first file's basename (+`+N more`). `None`
/// when `config.json` is missing/unreadable or records no inputs — the
/// caller falls back to the run-dir slug so a run never vanishes from
/// the selector just because its label can't be enriched.
fn chatbot_run_subject(run_dir: &Path) -> Option<String> {
    let cfg = read_run_state(run_dir)?;
    if let Some(v) = cfg.get("vault_dir").and_then(|x| x.as_str()) {
        if !v.trim().is_empty() {
            return Some(path_basename(v));
        }
    }
    let paths: Vec<String> = cfg
        .get("inputs")
        .and_then(|i| i.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|e| {
                    e.as_str()
                        .map(str::to_string)
                        .or_else(|| {
                            e.get("path")
                                .and_then(|p| p.as_str())
                                .map(str::to_string)
                        })
                })
                .collect()
        })
        .unwrap_or_default();
    let first = paths.first()?;
    let base = path_basename(first);
    Some(if paths.len() > 1 {
        format!("{base} +{} more", paths.len() - 1)
    } else {
        base
    })
}

/// Human-meaningful selector label: the ingested corpus subject (never
/// the opaque `-f66s`/`-xttq` slug, unless the corpus can't be recovered
/// at all, in which case the slug still beats dropping the run).
///
/// The run's time is deliberately NOT embedded here. A server-formatted
/// wall-clock date would be UTC (the run-dir prefix is UTC and Rust has
/// no knowledge of the user's tz), so it rendered ahead of every
/// client-side timestamp for any non-UTC user. Every surface now renders
/// the time client-side from `created_at` in the user's local tz via the
/// shared `prettyDateTime`, so the label carries only the corpus subject.
fn chatbot_run_label(run_dir: &Path) -> String {
    match chatbot_run_subject(run_dir) {
        Some(subject) => subject,
        None => run_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("run")
            .to_string(),
    }
}

/// `YYYYMMDDHHMMSS` (the run time-key) → RFC3339 `YYYY-MM-DDTHH:MM:SSZ`,
/// a string `new Date()` parses on the frontend so the shared
/// `prettyDate`/`prettyDateTime` formatter renders it in the user's
/// local tz like every other timestamp surface. The run-dir prefix is
/// UTC, so the trailing `Z` labels the instant correctly and the client
/// converts to local on render. Empty when the key isn't the 14-digit
/// shape (caller omits the date line rather than printing a malformed one).
fn run_time_key_to_rfc3339(key: &str) -> String {
    if key.len() < 14 || !key.chars().all(|c| c.is_ascii_digit()) {
        return String::new();
    }
    format!(
        "{}-{}-{}T{}:{}:{}Z",
        &key[0..4], &key[4..6], &key[6..8],
        &key[8..10], &key[10..12], &key[12..14],
    )
}

#[derive(serde::Serialize)]
struct ChatbotRunOption {
    /// The run-dir name (also the run_id sent back to `chatbot_select_run`
    /// and matched against `chatbot_bound`'s `run`).
    run_id: String,
    /// Input-source corpus subject, e.g. `personal-os.txt`. The run's
    /// time is NOT here — surfaces render it client-side from
    /// `created_at` in local tz (a server-side date would be UTC-skewed).
    label: String,
    /// The non-empty vectors.db this run binds.
    store_path: String,
    /// Whether this is the run the next message will answer from (the
    /// explicit pick if still valid, else the default).
    bound: bool,
    /// The run's creation timestamp as parseable RFC3339 (UTC), so the
    /// shared 2-line dropdown can format line 2 with the SAME
    /// `prettyDateTime` every other surface uses — NOT string-parsed
    /// back out of `label`. Empty when the dir name carries no usable
    /// time-key (the row then shows just the title + `#short_id`).
    created_at: String,
    /// The run's 4-letter perma-id (#574/#576) — the immutable identity
    /// the dir-name suffix encodes. A real field so the dropdown's
    /// `#<id>` subline reads it directly instead of re-deriving it.
    short_id: String,
}

/// The 10 most-recent runs with a non-empty store, newest first, by the
/// shared non-empty predicate + creation-time ordering. The
/// timestamp-key sort runs over the cheap dir-name prefix (config is
/// read only for the legacy-name fallback); `config.json` is read for a
/// human label only for the surviving top 10, so a logs dir with
/// hundreds of runs costs ten config reads, not hundreds.
fn enumerate_chatbot_runs() -> Vec<ChatbotRunOption> {
    let logs = logs_root();
    let mut cands: Vec<(String, PathBuf, PathBuf)> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&logs) {
        for e in entries.flatten() {
            let dir = e.path();
            if !dir.is_dir() {
                continue;
            }
            let Some(store) = chatbot_store_path(&dir) else {
                continue;
            };
            let Some(key) = chatbot_run_time_key(&dir) else {
                continue;
            };
            cands.push((key, dir, store));
        }
    }
    cands.sort_by(|a, b| b.0.cmp(&a.0));
    cands.truncate(10);
    cands
        .into_iter()
        .map(|(key, dir, store)| {
            let run_id = dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            ChatbotRunOption {
                label: chatbot_run_label(&dir),
                store_path: store.to_string_lossy().into_owned(),
                bound: false,
                created_at: run_time_key_to_rfc3339(&key),
                short_id: short_id_from_run_name(&run_id).unwrap_or_default(),
                run_id,
            }
        })
        .collect()
}

/// The non-empty store for a specific run_id, if it still exists — the
/// validity check that lets a stale explicit selection degrade to the
/// default instead of binding a vanished/emptied store. Resolved via
/// the SAME flat `<logs_root>/<run_id>` layout + non-empty predicate
/// `enumerate_chatbot_runs` uses (not the config-sidecar walk), so a
/// pick can never disagree with the list: a run is bindable iff it has
/// a non-empty store, config.json or not.
fn chatbot_run_store(run_id: &str) -> Option<PathBuf> {
    if run_id.is_empty() || run_id.contains('/') || run_id.contains("..") {
        return None; // not a flat run-dir name — never traverse out
    }
    chatbot_store_path(&logs_root().join(run_id))
}

/// Resolve the binding for a freshly-spawned sidecar:
///   * a still-valid explicit user pick  → that run, source `"user"`;
///   * otherwise (no pick, or it went stale) → the most-recent
///     non-empty run, source `"default"`;
///   * no qualifying run at all → `None` (sidecar gets no binding env
///     and reports an unbound session, exactly as before this feature).
///
/// An explicit pick is honoured even if newer runs have since pushed it
/// out of the most-recent-10 *list* — silently rebinding a run the user
/// deliberately chose is the very bug class this fixes.
fn resolve_chatbot_binding(
    state: &AppState,
) -> Option<(PathBuf, String, &'static str)> {
    // Resolve membership against the SAME top-10 list `chatbot_list_runs`
    // uses for the dropdown, not a bare "store exists on disk" check.
    // Pre-fix the two functions diverged: the dropdown capped at 10 and
    // visually defaulted to the most-recent when the pick was older than
    // the 10th, but the spawn binding only checked the store file
    // existed — so a pick aged out of the top-10 silently kept binding
    // (sidecar bound to invisible-in-UI run X while the dropdown
    // highlighted Y as bound). The user's mental model is the dropdown:
    // if their pick isn't in the list they see, it must not be the
    // binding either. Anchoring both functions on the same list keeps
    // them honest.
    let rows = enumerate_chatbot_runs();
    let selected = state.chatbot_selected_run.lock().unwrap().clone();
    if let Some(id) = selected {
        if let Some(row) = rows.iter().find(|r| r.run_id == id) {
            return Some((PathBuf::from(&row.store_path), id, "user"));
        }
        // Stale pick (run deleted, store emptied, OR aged out of the
        // top-10 active list) — fall through to the default below.
    }
    rows.into_iter()
        .next()
        .map(|r| (PathBuf::from(r.store_path), r.run_id, "default"))
}

/// The 10 most-recent non-empty runs for the selector, with `bound`
/// marking the one the next message will answer from (the explicit pick
/// if still valid, else the default = the first entry). The marked run
/// is always exactly the one `resolve_chatbot_binding` would pick, so
/// the UI's "current" highlight can't drift from the real binding.
#[tauri::command]
fn chatbot_list_runs(state: State<'_, AppState>) -> Vec<ChatbotRunOption> {
    let mut rows = enumerate_chatbot_runs();
    let selected = state.chatbot_selected_run.lock().unwrap().clone();
    let effective = match &selected {
        Some(id) if rows.iter().any(|r| &r.run_id == id) => Some(id.clone()),
        // No pick, or the pick dropped out of the top-10 list: the
        // selector highlights the default (first entry).
        // `resolve_chatbot_binding` anchors on the same list, so a
        // pick that aged out also stops binding — the UI's highlight
        // and the real spawn binding agree by construction.
        _ => rows.first().map(|r| r.run_id.clone()),
    };
    for r in &mut rows {
        r.bound = effective.as_deref() == Some(r.run_id.as_str());
    }
    rows
}

/// Pick (or clear, with `None`) the run the chat answers from. Rebinding
/// is a fresh session: reuse the Stop path's exact clear-handle +
/// eager-re-spawn machinery so the new binding is live and warm for the
/// next message — #506's persistent-lifecycle internals are untouched,
/// only the resolution input changes. The fresh sidecar emits
/// `chatbot_bound` at start, refreshing the selector's current mark.
#[tauri::command]
fn chatbot_select_run(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    run_id: Option<String>,
) -> Result<(), String> {
    {
        let mut g = state.chatbot_selected_run.lock().unwrap();
        *g = run_id.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    }
    // Cancel any in-flight turn + drop the proc handle (the unchanged
    // #457 path), then eager re-spawn bound to the new selection.
    let was_inflight = cancel_chatbot_inflight(&state);
    {
        let mut g = state.chatbot_proc.lock().unwrap();
        *g = None;
    }
    if was_inflight {
        // A turn was streaming when the user switched corpus — let the
        // UI leave the streaming state (keeps whatever partial arrived),
        // same contract as Stop.
        let _ = app.emit(
            "chatbot-event",
            serde_json::json!({ "event": "chatbot_stopped" }),
        );
    }
    spawn_chatbot_sidecar(&app, &state)
}

// ── Chat transcript persistence ─────────────────────────────────────────────
//
// The chatbot transcript lived only in ChatbotHelper's in-memory React
// state. Closing the window (Cmd/Ctrl+W is the File ▸ Close accelerator;
// the handler intentionally lets the WKWebView be destroyed and keeps
// the process alive headless so background pipeline runs survive)
// followed by a reopen rebuilds the window fresh — remounting React
// from scratch and wiping the conversation. Data loss on a routine
// keystroke.
//
// We persist the transcript to a single JSON file under the state dir
// and rehydrate it on mount, so the conversation survives a
// window-destroy/reopen AND a full reload. Under `~/.basevault/chats/`,
// NOT `~/.basevault/cache/` (the cache is wiped routinely; the
// conversation must not be).
//
// Deliberately ONE persisted transcript — not a multi-chat history
// store. The scope here is "the conversation survives a close",
// nothing more; richer history + the pipeline-feedback loop are the
// remaining scope of the parent chat-persistence work.

/// Chat conversation data root — `~/.basevault/chats`. Chats are USER
/// conversation data, not run logs, so they deliberately live OUTSIDE
/// the logs tree (`logs_root()`). Rust uses the canonical app path and
/// passes it to the sidecar as `BASEVAULT_CHATS_ROOT`; the agent/dev
/// split (app → `chats`, non-app/dev → `chats-dev`) lives in the
/// sidecar's `_chats_root()`, mirroring the `logs_root()`/`_logs_root()`
/// split exactly (Rust side flat, sidecar side env-driven).
fn chats_root() -> PathBuf {
    state_root().join("chats")
}

/// Back-compat alias for the legacy #453 single-transcript location
/// (`chats/transcript.json`). The per-conversation dirs (#565) now live
/// directly under the same root; `transcript.json` is a plain file so
/// `list_convos()` ignores it (not a dir, doesn't parse).
fn chats_dir() -> PathBuf {
    chats_root()
}

/// The conversation-directory root (#565): one `<ISO-Z>-<sid>/` dir
/// per thread, directly under `chats_root()`. Moved here from
/// `logs/chatbot/` — chats are user data, not run logs (director
/// call on #568).
fn convos_root() -> PathBuf {
    chats_root()
}

fn chat_transcript_path() -> PathBuf {
    chats_dir().join("transcript.json")
}

/// Load the persisted chatbot transcript. Returns `[]` when the file is
/// missing or unparseable — best-effort, mirroring `get_config`: a
/// corrupt transcript must not brick the chat, the next save
/// overwrites it.
#[tauri::command]
fn chatbot_load_transcript() -> serde_json::Value {
    let p = chat_transcript_path();
    match std::fs::read_to_string(&p) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
            warn!(
                "chatbot_load_transcript: {} unparseable ({}); returning empty",
                p.display(),
                e
            );
            serde_json::json!([])
        }),
        Err(_) => serde_json::json!([]),
    }
}

/// Persist the chatbot transcript state — `{ open, turns }`. Atomic write
/// (tmp + rename), same pattern as `write_config_atomic`. Whole-object
/// replace: the client owns the state and re-sends it whenever it
/// settles. Shape-agnostic (a generic JSON value) so the client can
/// evolve the payload without a Rust change.
#[tauri::command]
fn chatbot_save_transcript(state: serde_json::Value) -> Result<(), String> {
    let p = chat_transcript_path();
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("mkdir {:?}: {}", parent, e))?;
    }
    let pretty = serde_json::to_string_pretty(&state)
        .map_err(|e| format!("serialize transcript: {}", e))?;
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, pretty + "\n").map_err(|e| format!("write tmp: {}", e))?;
    std::fs::rename(&tmp, &p).map_err(|e| format!("rename transcript: {}", e))?;
    Ok(())
}

// ── Per-conversation chat threads (#565) ────────────────────────────────────
//
// Multi-conversation chat. Each conversation is a DIRECTORY directly
// under `chats_root()` (`~/.basevault/chats/`), named exactly like a
// run dir's ISO prefix plus a 4-letter perma-id (`sid`):
// `2026-05-02T13-30-11Z-frst`. Inside it:
//   - `transcript.json` — the #453 persisted `{open,turns}` shape, now
//     one per conversation (each turn keeps its own #559 pinned run).
//   - `llm-calls.jsonl` / `llm-payloads.jsonl` — that conversation's
//     OWN sidecar telemetry (chatbot_sidecar.py retargets here when
//     BASEVAULT_CHATBOT_CONVO_DIR is set). This delivers part of #561's
//     goal (one conversation's payloads, not a shared firehose); #561's
//     content-free marker is NOT reimplemented here — the dir is shaped
//     so that marker can later sit alongside these files.
//
// Chats live OUTSIDE the logs tree (director call on #568): they are
// user conversation data, not run logs. No legacy migration — pre-#559
// chat history is being purged; any old `chats/transcript.json` or
// `logs/chatbot/` is inert and simply ignored (a loose file is not a
// dir and doesn't parse).
//
// NO index/manifest: the set of conversations IS the directory
// listing. The immutable ISO-8601-UTC creation prefix is the IDENTITY
// (unique by construction — see `create_convo`); `-<label>` after it
// is free text the user renames in place (only the dir's label
// segment moves; the prefix never does, so nothing keyed on a
// conversation breaks). Ordering is most-recently-active first, by
// each conversation's last turn `ts`; no-`ts` (legacy/empty) ones
// fall back to ISO-creation order, last. Launch opens the top of that
// ordering; there is deliberately no persisted "active conversation"
// pointer on disk.

/// Validate the fixed-width `YYYY-MM-DDTHH-MM-SSZ` (20-char) ISO-Z
/// prefix shape. This prefix is a conversation's IMMUTABLE IDENTITY:
/// rename changes only the label that follows it, never these 20
/// chars, so every reference (active-thread, by-id command, the
/// frontend's cached id) stays valid across a rename.
fn is_iso_z_id(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 20
        && b[0..4].iter().all(u8::is_ascii_digit)
        && b[4] == b'-'
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[7] == b'-'
        && b[8..10].iter().all(u8::is_ascii_digit)
        && b[10] == b'T'
        && b[11..13].iter().all(u8::is_ascii_digit)
        && b[13] == b'-'
        && b[14..16].iter().all(u8::is_ascii_digit)
        && b[16] == b'-'
        && b[17..19].iter().all(u8::is_ascii_digit)
        && b[19] == b'Z'
}

/// Parse a conversation dir name `<ISO-Z>-<label>`. Returns
/// `(iso_id, label)`: the 20-char immutable ISO-Z identity prefix and
/// the human label after it. The label is now FREE TEXT (rename is in
/// scope) — anything non-empty and `/`-free; only the ISO prefix is
/// validated/keyed-on. A name that isn't `<20-char ISO>-<non-empty>`
/// is not a conversation dir and is ignored (a stray file, the
/// `.migrating-…` staging dir — byte 0 `.` ≠ digit — a future #561
/// sibling). Also the path-traversal guard for by-id commands.
fn parse_convo_name(name: &str) -> Option<(String, String)> {
    if name.len() < 22 || !is_iso_z_id(&name[0..20]) || name.as_bytes()[20] != b'-' {
        return None;
    }
    let label = &name[21..];
    if label.is_empty() || label.contains('/') {
        return None;
    }
    Some((name[0..20].to_string(), label.to_string()))
}

/// The `ts` (epoch-millis, #567's canonical per-turn field set at turn
/// creation) of a conversation's LAST turn — the "last activity"
/// signal the picker orders on. Best-effort: missing transcript / no
/// turns / a legacy turn with no `ts` → `None` (those conversations
/// degrade to ISO-creation order, sorted after the active ones).
/// #567 owns the field; we only consume the tail, no schema coupling.
fn convo_last_ts(name: &str) -> Option<i64> {
    let p = convos_root().join(name).join("transcript.json");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()?;
    let turns = v.get("turns")?.as_array()?;
    turns.last()?.get("ts")?.as_i64()
}

/// A single `YYYYMMDDHHMMSS` sort key for a conversation: the last
/// turn's `ts` (#567's per-turn field) when present, ELSE the
/// immutable ISO creation prefix. One unified ordering — a brand-new
/// conversation has no messages, so its key is its creation time
/// (= "now"), which is exactly why it must sort to the TOP, not the
/// bottom (the user's "if no message, the date is the creation date"
/// rule). Reuses `iso_z_at`/`iso_digits` so last-`ts` and creation
/// time are the same fixed-width comparable form (no epoch math, no
/// dir mtime — that's unreliable).
fn convo_sort_key(iso: &str, last_ts: Option<i64>) -> String {
    if let Some(ms) = last_ts {
        let secs = (ms / 1000).max(0) as u64;
        if let Some(k) = iso_digits(&iso_z_at(secs)) {
            return k;
        }
    }
    iso_digits(iso).unwrap_or_default()
}

/// Conversation directories, MOST-RECENTLY-ACTIVE first (#568 director
/// call). The dir listing is the whole source of truth (no manifest).
/// Final order is `convo_sort_key` descending — last message time, or
/// creation time when there are no messages — so a just-created
/// conversation sorts FIRST.
///
/// Returns `(iso_id, sid, display_label, dir_name)` per row. `sid`
/// is the 4-letter perma-id (from `transcript.json`, the dir tail
/// for post-revert dirs); `display_label` is the derived picker
/// label `Conversation <N> · <Mon Day>` where N is the convo's
/// position in iso-ASCENDING order (stable, deterministic, no
/// persistence). Aliases are NOT folded in here — the frontend
/// renders `alias || display_label`.
fn list_convos() -> Vec<(String, String, String, String)> {
    let mut rows: Vec<(String, String, String, String)> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(convos_root()) {
        for e in entries.flatten() {
            if !e.path().is_dir() {
                continue;
            }
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some((iso, _)) = parse_convo_name(&name) {
                let key = convo_sort_key(&iso, convo_last_ts(&name));
                let sid = convo_short_id(&name).unwrap_or_default();
                rows.push((key, iso, sid, name));
            }
        }
    }
    // Positional N: rank by iso ascending (1-based), so `Conversation
    // 1` is the oldest creation. Recomputed every call — no persisted
    // counter — so the display stays consistent across renames/
    // deletes/out-of-order seeds without storing extra state.
    let mut by_iso: Vec<usize> = (0..rows.len()).collect();
    by_iso.sort_by(|&a, &b| rows[a].1.cmp(&rows[b].1));
    let mut n_for: Vec<usize> = vec![0; rows.len()];
    for (rank, &i) in by_iso.iter().enumerate() {
        n_for[i] = rank + 1;
    }
    // Final order: newest activity first; the immutable ISO id breaks
    // exact ties deterministically.
    let mut enriched: Vec<(String, String, String, String, String)> = rows
        .into_iter()
        .enumerate()
        .map(|(i, (key, iso, sid, name))| {
            let display = derive_display_label(n_for[i]);
            (key, iso, sid, display, name)
        })
        .collect();
    enriched.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
    enriched
        .into_iter()
        .map(|(_, iso, sid, display, name)| (iso, sid, display, name))
        .collect()
}

#[derive(serde::Serialize)]
struct ConvMeta {
    /// The IMMUTABLE identity = the 20-char ISO-Z creation prefix.
    /// Everything keys on this (by-id commands, active-thread, the
    /// frontend's cached id) so a rename — which changes only `label`
    /// / `name` — never breaks a reference. Citations are unaffected
    /// regardless: they key on each turn's own `runId` (a RUN id).
    id: String,
    /// Same as `id` (the immutable ISO-Z creation prefix). The
    /// creation date stays in the on-disk name; the picker does NOT
    /// show it.
    created: String,
    /// The raw human label after the ISO prefix — free text,
    /// user-editable via rename. The default IS the 4-letter perma-id
    /// (`short_id`): the dir is `<iso>-<sid>`. Nothing keys on it.
    /// This is the on-disk substring as-is; the UI renders
    /// `display_label` instead.
    label: String,
    /// The picker-facing label: `Conversation <N>` derived at list-time
    /// from the convo's positional rank (1-based, iso ascending).
    /// Recomputed every `list_convos` call, never persisted. The date is
    /// NOT here — the picker renders it client-side in local tz (a
    /// server-formatted date would be UTC-skewed). The frontend uses
    /// `alias || display_label` for the picker title; raw `label` is for
    /// data callers, not the UI.
    display_label: String,
    /// Current on-disk dir name (`<iso>-<sid>`). Resolve it from
    /// `id` via `find_convo_dir`, never cache it as identity.
    name: String,
    /// Epoch-millis of the LAST turn's `ts` — the conversation's last
    /// activity, what the picker shows beside the label (the frontend
    /// formats it; falls back to the creation date when `null`, i.e.
    /// the conversation has no messages yet).
    last_ts: Option<i64>,
    /// The 4-letter perma-id (#574) = the conversation's NAME (the dir
    /// is `<iso>-<short_id>`), also persisted in `transcript.json`,
    /// minted once and permanent — the run scheme extended to chats.
    /// The picker shows `#<short_id>` on the subline always, and uses
    /// it as the title when there's no alias.
    short_id: String,
    /// Cosmetic, user-set rename (run-style). Empty = none → the
    /// picker titles the row with `#<short_id>`. Persisted in
    /// `transcript.json`; the dir/id NEVER move on rename, so it's
    /// rename-proof by construction.
    alias: String,
}

fn convo_meta(name: &str) -> ConvMeta {
    let (id, label) = parse_convo_name(name)
        .unwrap_or_else(|| (name.to_string(), String::new()));
    // Singleton lookup: read `display_label` from the same list_convos
    // pass the picker sees, so a fresh-mint / post-delete convo's
    // ConvMeta carries the up-to-date positional N. Fallback to a
    // standalone derivation (N=1) when the dir isn't enumerable yet
    // (a race during create, or a non-existent name).
    let display_label = list_convos()
        .into_iter()
        .find(|(iso, _, _, _)| iso == &id)
        .map(|(_, _, d, _)| d)
        .unwrap_or_else(|| derive_display_label(1));
    ConvMeta {
        created: id.clone(),
        id,
        label,
        display_label,
        last_ts: convo_last_ts(name),
        short_id: convo_short_id(name).unwrap_or_default(),
        alias: convo_alias(name),
        name: name.to_string(),
    }
}

/// Resolve an immutable ISO-Z id to its CURRENT on-disk dir name. The
/// id never changes; the dir name does (rename), so every by-id
/// command goes through here. The creation guarantee that ISO
/// prefixes are unique (see `create_convo`) makes this unambiguous.
fn find_convo_dir(id: &str) -> Option<String> {
    if !is_iso_z_id(id) {
        return None;
    }
    list_convos()
        .into_iter()
        .find(|(iso, _, _, _)| iso == id)
        .map(|(_, _, _, name)| name)
}


/// One raw string field from a conversation's `transcript.json`
/// (best-effort: missing/corrupt/absent → `None`).
fn convo_transcript_str(name: &str, key: &str) -> Option<String> {
    let p = convos_root().join(name).join("transcript.json");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()?;
    v.get(key)?.as_str().map(str::to_string)
}

/// A conversation's persisted 4-letter perma-id (= its NAME — the dir
/// is `<iso>-<short_id>`, the run scheme extended to chats per #574).
/// Also stamped into `transcript.json` so it survives even a manual
/// dir-label edit and is content-greppable. `None` only for a dir
/// that pre-dates this and was never written; the migration / next
/// write mints one and it's then permanent.
fn convo_short_id(name: &str) -> Option<String> {
    short_id_from_run_name(&convo_transcript_str(name, "short_id")?)
}

/// A conversation's cosmetic alias (run-style rename): a user-facing
/// label persisted in `transcript.json`. Empty/absent → no alias, the
/// picker shows the 4-letter id. The id (the dir name) NEVER changes
/// on rename — only this alias does, so it's rename-proof by
/// construction.
fn convo_alias(name: &str) -> String {
    convo_transcript_str(name, "alias").unwrap_or_default()
}

/// Atomic per-conversation transcript write (tmp + rename, the #453
/// pattern). The client owns the `{open,turns}` shape (schema-agnostic
/// here); we additionally stamp the immutable identity (`cid` = the ISO
/// prefix = stable id, `created`, `label`) INTO the file, plus the
/// 4-letter perma-id (`short_id`) — the chat extension of the run-id
/// scheme (#574). The perma-id is MINTED ONCE: if `transcript.json`
/// already carries a valid one it's preserved verbatim (this fn runs
/// on every save, so reading-then-carrying is what makes it
/// never-regenerated — the same write-once guarantee a run's
/// `config.json` has). Rename only moves the dir label, never touches
/// `transcript.json`, so the perma-id is rename-proof.
fn write_convo_transcript(name: &str, state: &serde_json::Value) -> Result<(), String> {
    let dir = convos_root().join(name);
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir {:?}: {}", dir, e))?;
    let created = name.get(0..20).unwrap_or(name).to_string();
    let label = name.get(21..).unwrap_or("").to_string();
    let mut obj = match state {
        serde_json::Value::Object(m) => m.clone(),
        _ => serde_json::Map::new(),
    };
    // Mint-once perma-id. Precedence: an existing on-disk short_id
    // ALWAYS wins (never regenerated); else a valid one explicitly
    // seeded in the incoming state (create_convo passes the id it put
    // in the dir name so both agree); else a fresh mint (a legacy dir
    // written before #574). This fn runs on every save, so the
    // read-then-carry is what makes it permanent.
    let short_id = convo_short_id(name)
        .or_else(|| {
            obj.get("short_id")
                .and_then(|v| v.as_str())
                .and_then(short_id_from_run_name)
        })
        .unwrap_or_else(short_id);
    // Cosmetic alias (run-style rename), presence-based: if the
    // incoming state carries an `alias` key the caller is setting it
    // (the rename path — `""` deliberately CLEARS it); if the key is
    // absent the caller is a plain save ({open,turns}) and the prior
    // alias is preserved verbatim.
    let alias = match obj.get("alias").and_then(|v| v.as_str()) {
        Some(a) => a.to_string(),
        None => convo_alias(name),
    };
    obj.insert("cid".into(), serde_json::Value::String(created.clone()));
    obj.insert("created".into(), serde_json::Value::String(created));
    obj.insert("label".into(), serde_json::Value::String(label));
    obj.insert("short_id".into(), serde_json::Value::String(short_id));
    obj.insert("alias".into(), serde_json::Value::String(alias));
    let p = dir.join("transcript.json");
    let pretty = serde_json::to_string_pretty(&serde_json::Value::Object(obj))
        .map_err(|e| format!("serialize transcript: {}", e))?;
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, pretty + "\n").map_err(|e| format!("write tmp: {}", e))?;
    std::fs::rename(&tmp, &p).map_err(|e| format!("rename transcript: {}", e))?;
    Ok(())
}

/// Sanitize a user-supplied rename into a filesystem-safe dir-label:
/// trim, strip path separators / NUL / leading-or-trailing dots
/// (a leading `.` would make a hidden dir that `list_convos` skips),
/// cap length. `None` if nothing usable remains (caller rejects — a
/// label must be non-empty so the dir name still parses as
/// `<iso>-<label>`).
fn sanitize_label(raw: &str) -> Option<String> {
    let cleaned: String = raw
        .trim()
        .chars()
        .map(|c| if c == '/' || c == '\\' || c == '\0' { '-' } else { c })
        .collect();
    let cleaned = cleaned.trim().trim_matches('.').trim();
    if cleaned.is_empty() {
        return None;
    }
    Some(cleaned.chars().take(80).collect::<String>().trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Derived picker label for an unrenamed conversation: `Conversation
/// <N>` where N is the convo's positional rank by ISO ascending
/// (1-based) across the chats dir. Recomputed every `list_convos` call
/// — nothing persisted; an alias (when set) takes precedence at render
/// time.
///
/// The creation date is deliberately NOT baked in here. Slicing the
/// month/day straight out of the UTC ISO prefix server-side gave every
/// non-UTC user a date in the wrong tz — a convo created 00:00–04:00 UTC
/// showed tomorrow's date for a NYC (UTC−4) user, disagreeing with the
/// client-rendered local `convoLastDate`. The picker renders the
/// conversation's date client-side in local tz via the shared
/// `prettyDateTime`/`prettyDate`, so the label carries only the ordinal.
fn derive_display_label(n: usize) -> String {
    format!("Conversation {n}")
}

/// Mint a fresh empty conversation dir, returning its name. The
/// default human-visible label is the 4-letter perma-id (`sid`),
/// matching the run-dir convention: the dir is `<iso-z>-<sid>` and
/// the picker shows `#<sid>` when there's no alias. The same id is
/// also seeded into the transcript (`short_id`) — that's the
/// IMMUTABLE identity retrieval + #561 diagnostics resolve on
/// (#574/#576), independent of the dir name, so a rename never moves
/// it. The ISO-Z prefix is the unique key everything resolves on, so
/// it must not collide: `iso_z()` is second-resolution, so advance to
/// the next free second until the prefix is unused (preserves the
/// locked `YYYY-MM-DDTHH-MM-SSZ` format, keeps chronological order,
/// no index).
fn create_convo() -> String {
    let mut secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let taken: std::collections::HashSet<String> =
        list_convos().into_iter().map(|(iso, _, _, _)| iso).collect();
    let mut iso = iso_z_at(secs);
    while taken.contains(&iso) {
        secs += 1;
        iso = iso_z_at(secs);
    }
    let sid = short_id();
    let name = format!("{iso}-{sid}");
    let _ = write_convo_transcript(
        &name,
        &serde_json::json!({ "open": true, "turns": [], "short_id": sid }),
    );
    name
}

/// Make `name` the active conversation so the sidecar scopes its
/// telemetry to that dir (and, if `run_id` is given, bind that corpus —
/// the gate-1 Q1 default: a switched-into thread points "talking about"
/// at its most-recent turn's pinned run, freely changeable after;
/// `None`/empty leaves the binding).
///
/// Only re-spawns when a sidecar is already LIVE — the same cancel +
/// drop-handle + eager-respawn machinery as `chatbot_select_run`, so a
/// warm session retargets immediately. With no live process (a cold
/// session — e.g. set on mount before the panel is ever used) just the
/// state is updated: the lazy spawn on the first message reads it, so
/// telemetry is scoped from turn one WITHOUT forcing a background
/// process at every app launch (preserving the #506 lazy lifecycle).
fn set_active_convo_and_respawn(
    app: &tauri::AppHandle,
    state: &AppState,
    id: &str,
    run_id: Option<String>,
) -> Result<(), String> {
    {
        // Store the IMMUTABLE ISO id, never the dir name — so the
        // active thread survives a rename. spawn resolves it to the
        // current dir via find_convo_dir.
        let mut g = state.chatbot_active_convo.lock().unwrap();
        *g = Some(id.to_string());
    }
    if let Some(rid) = run_id {
        let v = rid.trim().to_string();
        if !v.is_empty() {
            let mut g = state.chatbot_selected_run.lock().unwrap();
            *g = Some(v);
        }
    }
    let proc_alive = {
        let g = state.chatbot_proc.lock().unwrap();
        g.as_ref().map_or(false, |p| is_pid_alive(p.pid))
    };
    if !proc_alive {
        // Cold: the next message's lazy spawn will read the new state.
        return Ok(());
    }
    let was_inflight = cancel_chatbot_inflight(state);
    {
        let mut g = state.chatbot_proc.lock().unwrap();
        *g = None;
    }
    if was_inflight {
        let _ = app.emit(
            "chatbot-event",
            serde_json::json!({ "event": "chatbot_stopped" }),
        );
    }
    spawn_chatbot_sidecar(app, state)
}

/// The conversation list, most-recently-active first. Guarantees at
/// least one conversation exists (a fresh install / empty `chats/`
/// gets a `<iso>-<sid>` dir minted lazily) so the chat surface is
/// never threadless. Each meta's `id` is the immutable ISO-Z prefix; the
/// frontend keys on that and pretty-prints `created`. No legacy
/// migration: pre-thread chat data is not relocated (director call —
/// pre-#559 history is being purged; any old `chats/transcript.json`
/// or `logs/chatbot/` is inert and ignored).
#[tauri::command]
fn chatbot_list_conversations() -> Vec<ConvMeta> {
    let mut convos = list_convos();
    if convos.is_empty() {
        let name = create_convo();
        if let Some((iso, _)) = parse_convo_name(&name) {
            let sid = convo_short_id(&name).unwrap_or_default();
            let display = derive_display_label(1);
            convos.push((iso, sid, display, name));
        }
    }
    convos.iter().map(|(_, _, _, name)| convo_meta(name)).collect()
}

/// Load one conversation's transcript by its immutable ISO-Z id.
/// Best-effort, mirroring `chatbot_load_transcript`: an unknown id /
/// missing / corrupt file reads back as `[]` so the chat can't be
/// bricked; the next save overwrites it.
#[tauri::command]
fn chatbot_load_conversation(id: String) -> serde_json::Value {
    let Some(name) = find_convo_dir(&id) else {
        return serde_json::json!([]);
    };
    let p = convos_root().join(&name).join("transcript.json");
    match std::fs::read_to_string(&p) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
            warn!(
                "chatbot_load_conversation: {} unparseable ({}); returning empty",
                p.display(),
                e
            );
            serde_json::json!([])
        }),
        Err(_) => serde_json::json!([]),
    }
}

/// Persist one conversation's `{open,turns}` by its immutable ISO-Z
/// id (resolved to the current dir — rename-safe; the id is also the
/// path-traversal guard, only a valid ISO-Z prefix resolves).
#[tauri::command]
fn chatbot_save_conversation(
    id: String,
    state: serde_json::Value,
) -> Result<(), String> {
    let name = find_convo_dir(&id)
        .ok_or_else(|| format!("unknown conversation id: {id}"))?;
    write_convo_transcript(&name, &state)
}

/// Start a fresh empty conversation, make it active (telemetry
/// retargets to its dir), and return its meta. The prior conversation
/// is untouched and stays selectable.
#[tauri::command]
fn chatbot_new_conversation(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<ConvMeta, String> {
    let name = create_convo();
    let meta = convo_meta(&name);
    set_active_convo_and_respawn(&app, &state, &meta.id, None)?;
    Ok(meta)
}

/// Delete a conversation's whole dir (transcript + its telemetry) by
/// immutable id. Falls back to the most-recently-active remaining
/// conversation; if none remain a fresh empty one is created so the
/// chat is never threadless. Returns the now-active conversation's
/// meta. Gated behind the app's single existing confirm modal on the
/// client (no new modal here).
#[tauri::command]
fn chatbot_delete_conversation(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    id: String,
) -> Result<ConvMeta, String> {
    if let Some(name) = find_convo_dir(&id) {
        let _ = std::fs::remove_dir_all(convos_root().join(&name));
    }
    let convos = list_convos();
    let active_name = convos
        .first()
        .map(|(_, _, _, name)| name.clone())
        .unwrap_or_else(create_convo);
    let meta = convo_meta(&active_name);
    set_active_convo_and_respawn(&app, &state, &meta.id, None)?;
    Ok(meta)
}

/// Switch the active conversation by immutable id (telemetry retargets
/// to its dir). `run_id` is the gate-1 Q1 default — point "talking
/// about" at the thread's most-recent turn's pinned run; `None` leaves
/// the binding.
#[tauri::command]
fn chatbot_set_active_conversation(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    id: String,
    run_id: Option<String>,
) -> Result<(), String> {
    if !is_iso_z_id(&id) {
        return Err(format!("not a conversation id: {id}"));
    }
    set_active_convo_and_respawn(&app, &state, &id, run_id)
}

/// Rename a conversation — a COSMETIC ALIAS only (run-style). The
/// conversation is named by its 4-letter perma-id and the dir
/// (`<iso>-<short_id>`) NEVER moves; rename just writes a user-facing
/// `alias` into `transcript.json`. So the id, every by-id ref,
/// active-thread, citations (keyed on each turn's own `runId`) and
/// the perma-id itself are untouched by construction — rename-proof.
/// An empty / whitespace-only label CLEARS the alias (the picker then
/// shows `#<short_id>` again). Returns the updated meta.
#[tauri::command]
fn chatbot_rename_conversation(
    id: String,
    label: String,
) -> Result<ConvMeta, String> {
    let name = find_convo_dir(&id)
        .ok_or_else(|| format!("unknown conversation id: {id}"))?;
    // Empty → clear the alias (no error — clearing is a valid action;
    // the id is always a usable name, so a conversation is never
    // nameless).
    let alias = sanitize_label(&label).unwrap_or_default();
    // Preserve {open,turns} — load the current transcript, inject the
    // alias, re-persist via the single writer (which keeps short_id /
    // identity stamped and atomic).
    let p = convos_root().join(&name).join("transcript.json");
    let mut state: serde_json::Value = std::fs::read_to_string(&p)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::json!({ "open": true, "turns": [] }));
    if let Some(o) = state.as_object_mut() {
        o.insert("alias".into(), serde_json::Value::String(alias));
    }
    write_convo_transcript(&name, &state)?;
    Ok(convo_meta(&name))
}

// ── LLM prompt-hash cache controls ──────────────────────────────────────────
//
// The pipeline writes per-(stage, prompt-hash) cache entries to
// ~/.basevault/cache/. The Settings UI exposes:
//   1. A checkbox bound to `cfg.llm_cache_enabled` (read/written via
//      get_config / set_config; the runner translates `false` into the
//      BASEVAULT_LLM_CACHE_BYPASS=1 env var for that run).
//   2. A "Wipe cache" button that calls `wipe_llm_cache()` below.
//
// The wipe path is destructive (recursive rmdir) so the UI MUST gate
// it behind a confirmation dialog. The command itself is a thin shell
// — it returns the (entries, bytes) it nuked so the UI can show a
// post-wipe toast, but it does not prompt or confirm.

fn cache_root() -> PathBuf {
    state_root().join("cache")
}

#[derive(serde::Serialize)]
struct CacheStats {
    /// Number of cache entries (`<stage>/<hash>.json` files) on disk.
    entries: u64,
    /// Total disk usage in bytes across all entries.
    bytes: u64,
}

fn walk_cache_dir(root: &Path) -> CacheStats {
    let mut entries = 0u64;
    let mut bytes = 0u64;
    let Ok(stages) = std::fs::read_dir(root) else {
        return CacheStats { entries, bytes };
    };
    for stage in stages.flatten() {
        let sp = stage.path();
        if !sp.is_dir() {
            continue;
        }
        let Ok(files) = std::fs::read_dir(&sp) else { continue };
        for f in files.flatten() {
            if let Ok(meta) = f.metadata() {
                if meta.is_file() {
                    entries += 1;
                    bytes += meta.len();
                }
            }
        }
    }
    CacheStats { entries, bytes }
}

/// Inspect the LLM prompt-hash cache without modifying it. The
/// Settings UI calls this BEFORE confirming a wipe so the dialog can
/// say "Are you sure? This will delete N entries (X MB)" with real
/// numbers. Returns {entries: 0, bytes: 0} if the directory doesn't
/// exist, never errors.
#[tauri::command]
fn get_llm_cache_stats() -> CacheStats {
    walk_cache_dir(&cache_root())
}

/// Recursively delete `~/.basevault/cache/`. Returns the
/// (entries, bytes) count snapshotted just before deletion so the UI
/// can render a post-wipe toast without re-walking the (now empty)
/// directory. Idempotent: wiping an already-empty cache is a no-op
/// returning {0, 0}.
///
/// Caller MUST confirm with the user first — this is destructive and
/// the on-disk responses are not recoverable. Cache misses on the
/// next run trigger fresh LLM calls (cost + wall-clock).
/// Inner body of `wipe_llm_cache` — no AppHandle dep, callable from
/// tests. The Tauri-command wrapper below adds the event emit.
fn wipe_llm_cache_inner() -> Result<CacheStats, String> {
    let root = cache_root();
    let stats_before = walk_cache_dir(&root);
    if !root.exists() {
        return Ok(stats_before); // {0, 0}
    }
    std::fs::remove_dir_all(&root)
        .map_err(|e| format!("remove {}: {}", root.display(), e))?;
    info!(
        "wipe_llm_cache: removed {} entries ({} bytes) at {}",
        stats_before.entries,
        stats_before.bytes,
        root.display(),
    );
    Ok(stats_before)
}

#[tauri::command]
fn wipe_llm_cache(app: tauri::AppHandle) -> Result<CacheStats, String> {
    let stats = wipe_llm_cache_inner()?;
    // Notify any open Run-details modal so its per-call "in cache"
    // column flips yes → no without the user reopening the modal.
    // Emitted unconditionally — even on a no-op wipe (root didn't
    // exist) the modal may be holding stale `cached_now=true` from a
    // prior-session record whose backing file was wiped externally.
    let _ = app.emit("llm-cache-changed", ());
    Ok(stats)
}

/// Minimal URL-encoding — only the characters that actually matter for
/// Obsidian URL: space, &, #, %, +, ?. Not a full RFC3986 implementation.
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b' ' => out.push_str("%20"),
            b'&' => out.push_str("%26"),
            b'#' => out.push_str("%23"),
            b'%' => out.push_str("%25"),
            b'+' => out.push_str("%2B"),
            b'?' => out.push_str("%3F"),
            _ => out.push(b as char),
        }
    }
    out
}

/// Return the last `n` lines of `s`, joined by newlines.
fn tail_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

/// Rewrite `/Users/<name>/` and `/home/<name>/` prefixes to `~/` so
/// surfaced tracebacks (UI <details>, audit log) don't leak usernames.
/// Mirrors the Python-side `_USER_HOME_PATH_RE` in engine/retry.py
/// so traces produced on either side of the bridge sanitize the same
/// way. Stdlib-only — no regex dep — by scanning for the two known
/// prefixes and consuming up to the next path separator.
fn sanitize_home_paths(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let rest = &s[i..];
        let prefix_len = if rest.starts_with("/Users/") {
            Some(7)
        } else if rest.starts_with("/home/") {
            Some(6)
        } else {
            None
        };
        if let Some(plen) = prefix_len {
            let after = &rest[plen..];
            if let Some(slash) = after.find('/') {
                let name = &after[..slash];
                // Username can't be empty or contain whitespace.
                if !name.is_empty() && !name.chars().any(char::is_whitespace) {
                    out.push_str("~/");
                    i += plen + slash + 1;
                    continue;
                }
            }
        }
        // Push one char and advance by its UTF-8 byte length.
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

// ── Existing commands ────────────────────────────────────────────────────────

/// Stub — kept for backwards compat during transition.
#[tauri::command]
fn run_pipeline_step(
    step: String,
    inputs: Vec<String>,
    mode: String,
    state: State<AppState>,
) -> StepResult {
    let _ = (step, inputs, mode, state);
    StepResult { error: None }
}

#[tauri::command]
fn get_output(state: State<AppState>) -> String {
    state
        .output
        .lock()
        .unwrap()
        .clone()
        .unwrap_or_default()
}

#[tauri::command]
fn open_output_folder(path: String) {
    info!("open_output_folder: {}", path);
    let _ = std::process::Command::new("open").arg(&path).spawn();
}

/// Open the global attestation log (`~/.basevault/attestations.jsonl`)
/// in the system default app. Powers the "View attestation log"
/// button in the AttestationPanel — power users can grep it, audit
/// fingerprints, replay timing, etc.
///
/// `touch -a` ensures the file exists before opening; otherwise
/// `open` would fail on a fresh install where no attestation has
/// been logged yet. The file gets created naturally on the first
/// attest call, but the button shouldn't be a dud before then.
#[tauri::command]
fn open_attestation_log() -> Result<(), String> {
    let path = state_root().join("attestations.jsonl");
    if !path.exists() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path);
    }
    info!("open_attestation_log: {}", path.display());
    std::process::Command::new("open")
        .arg(&path)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("spawn open: {}", e))
}

// ── User secrets (API keys) ──────────────────────────────────────────────────
//
// The pipeline reads the TINFOIL_API_KEY from a dotenv at
// `~/Library/Application Support/BaseVault/.env` (see `engine/_dotenv.py`).
// Everything non-secret (subject, tee_provider, tee_model, obsidian_*,
// local_setup_mode) lives in `~/.basevault/config.json` and is read
// directly by Python — no env-bridge.
//
// Tauri's `app_config_dir()` would resolve to `<config_dir>/ai.basevault`
// because it appends the bundle identifier. _dotenv.py uses the literal "BaseVault"
// directory name, so we use `config_dir()` (the root) + "BaseVault" to match.

/// Resolve the BaseVault user config dir: `~/Library/Application Support/BaseVault`.
fn user_config_dir(app: &tauri::AppHandle) -> PathBuf {
    app.path()
        .config_dir()
        .map(|d| d.join("BaseVault"))
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home).join("Library/Application Support/BaseVault")
        })
}

/// Parse `KEY=value` lines out of a dotenv file. Ignores comments + blanks.
/// Returns the first occurrence of each key (matches python-dotenv).
fn parse_dotenv(path: &std::path::Path) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else { return out };
    for line in text.lines() {
        let line = line.trim_start();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else { continue };
        let k = k.trim();
        if k.is_empty() || out.contains_key(k) {
            continue;
        }
        // Strip surrounding quotes so masking / display matches what the
        // Python runtime sees. Don't attempt full shell-style unescaping.
        let v = v.trim();
        let v = v
            .strip_prefix('"').and_then(|s| s.strip_suffix('"'))
            .or_else(|| v.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
            .unwrap_or(v);
        out.insert(k.to_string(), v.to_string());
    }
    out
}

/// Friends-and-family bundled Tinfoil key. Baked in at COMPILE TIME from
/// the `BUNDLED_TINFOIL_KEY` env var that `build.rs` resolves (release CI
/// secret > repo-root `.env` `TINFOIL_API_KEY` for local dev > empty), so a
/// dev rebuild, a release build, and a fresh clone are all keyed with no
/// per-checkout setup. NOT a tracked file — the key never enters the public
/// source tree.
///
/// WARNING: this keeps the key out of the SOURCE, not the BINARY. The value
/// is still a plaintext string in the shipped `.app` (`strings … | grep
/// '^tk_'` recovers it). Treat it as a shared, low-trust, rate-limited
/// free-tier credential — never a secret. A user-entered key always wins at
/// runtime; an empty value makes the build keyless (the Easy Wizard menu
/// entry then doesn't compile in).
const BUNDLED_TINFOIL_KEY: &str = env!("BUNDLED_TINFOIL_KEY");

fn bundled_tinfoil_key() -> Option<String> {
    let key = BUNDLED_TINFOIL_KEY.trim();
    if key.is_empty() {
        None
    } else {
        Some(key.to_string())
    }
}

/// The non-empty Tinfoil key the user entered themselves (wizard /
/// Settings), read from the user dotenv. `None` when they never set one.
fn user_tinfoil_key(app: &tauri::AppHandle) -> Option<String> {
    let env_file = user_config_dir(app).join(".env");
    if !env_file.exists() {
        return None;
    }
    parse_dotenv(&env_file)
        .get("TINFOIL_API_KEY")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Mask a secret so only the last 4 chars are visible: `tf_•••••wxyz`.
/// Empty or very-short values are reported as `•••` to avoid leaking
/// the whole value on short test keys.
fn mask_key(val: &str) -> String {
    if val.is_empty() {
        return String::new();
    }
    let chars: Vec<char> = val.chars().collect();
    if chars.len() <= 4 {
        return "•".repeat(chars.len());
    }
    let tail: String = chars[chars.len() - 4..].iter().collect();
    format!("{}{}", "•".repeat(chars.len().saturating_sub(4).min(8)), tail)
}

#[derive(Serialize)]
struct Settings {
    tinfoil_key_masked: Option<String>,
    tinfoil_key_set: bool,
    /// True iff this build carries a bundled Tinfoil key. The frontend
    /// reads this to pick the name-only Easy Wizard as the first-run
    /// default on keyed builds — a fresh user on a keyless build still
    /// gets the regular wizard. Mirrors the menu-entry gate
    /// (`bundled_tinfoil_key().is_some()`).
    bundled_key_set: bool,
}

/// Returns the secrets state from the user dotenv. Non-secret prefs
/// (subject, obsidian, tee_provider, tee_model, …) live in
/// `config.json` — the rule is: dotenv = API keys only, config.json =
/// everything else. Frontend reads both: this for what's-on-file flags,
/// `get_config` for the rest.
#[tauri::command]
fn get_settings(app: tauri::AppHandle) -> Settings {
    let file = user_config_dir(&app).join(".env");
    let parsed = if file.exists() {
        parse_dotenv(&file)
    } else {
        std::collections::HashMap::new()
    };
    let tinfoil_raw = parsed
        .get("TINFOIL_API_KEY")
        .filter(|s| !s.is_empty())
        .cloned();
    let tinfoil_key_set = tinfoil_raw.is_some();
    let tinfoil_key_masked = tinfoil_raw.as_deref().map(mask_key);

    Settings {
        tinfoil_key_masked,
        tinfoil_key_set,
        bundled_key_set: bundled_tinfoil_key().is_some(),
    }
}

/// Wizard fires iff the user has never set their subject (name). The check
/// looks at config.json, which is where subject lives now. The migration
/// step in `setup` (see `migrate_subject_to_config`) copies subject out of
/// any pre-existing dotenv into config.json before this runs, so returning
/// users from the previous layout still get past the wizard gate.
#[tauri::command]
fn needs_wizard(_app: tauri::AppHandle) -> bool {
    let cfg = read_config_json();
    cfg.get("subject")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().is_empty())
        .unwrap_or(true)
}

/// Upsert API keys into the user dotenv without touching other lines.
/// Creates the dir if missing, chmods the file to 600. Subject + UI
/// prefs live in config.json now (see `set_config`); this command
/// handles secrets only.
///
/// `tinfoil_key` uses the Some/None/Some("") semantics:
///   - `None`     → leave the line untouched.
///   - `Some("")` → remove the line entirely.
///   - `Some(v)`  → upsert `KEY=v`.
#[tauri::command]
fn save_settings(
    app: tauri::AppHandle,
    tinfoil_key: Option<String>,
) -> Result<(), String> {
    let dir = user_config_dir(&app);
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("create {:?}: {}", dir, e))?;
    let file = dir.join(".env");

    let existing = if file.exists() {
        std::fs::read_to_string(&file).map_err(|e| format!("read {:?}: {}", file, e))?
    } else {
        String::new()
    };

    let mut updates: Vec<(&str, Option<String>)> = Vec::new();
    if let Some(v) = tinfoil_key {
        let v = v.trim().to_string();
        if v.is_empty() {
            updates.push(("TINFOIL_API_KEY", None));
        } else {
            updates.push(("TINFOIL_API_KEY", Some(v)));
        }
    }

    let new_text = upsert_dotenv(&existing, &updates);
    std::fs::write(&file, &new_text).map_err(|e| format!("write {:?}: {}", file, e))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(&file) {
            let mut perms = meta.permissions();
            perms.set_mode(0o600);
            let _ = std::fs::set_permissions(&file, perms);
        }
    }

    info!("save_settings: wrote {:?} ({} bytes)", file, new_text.len());
    Ok(())
}

/// Read config.json directly without going through the Tauri command
/// (callable from contexts where there's no AppHandle, like
/// `needs_wizard`). Returns an empty Map on any error.
fn read_config_json() -> serde_json::Map<String, serde_json::Value> {
    let p = config_path();
    let Ok(text) = std::fs::read_to_string(&p) else {
        return serde_json::Map::new();
    };
    let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) else {
        return serde_json::Map::new();
    };
    val.as_object().cloned().unwrap_or_default()
}

/// One-shot migration from "subject in dotenv" to "subject in config.json".
/// Idempotent: if config.json already has a non-empty `subject`, do nothing.
/// Otherwise read BASEVAULT_SUBJECT from the dotenv and write it to config,
/// then strip the line from the dotenv (dotenv = secrets only is the new rule).
///
/// Called once at startup from the Tauri `setup` hook, before the wizard
/// gate fires. Failures are logged but never fatal — worst case the user
/// sees the wizard again and re-enters their name.
fn migrate_subject_to_config(app: &tauri::AppHandle) {
    let dotenv_path = user_config_dir(app).join(".env");
    if !dotenv_path.exists() {
        return;
    }

    let mut cfg = read_config_json();
    let already_in_config = cfg
        .get("subject")
        .and_then(|v| v.as_str())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);

    let dotenv_parsed = parse_dotenv(&dotenv_path);
    let dotenv_subject = dotenv_parsed
        .get("BASEVAULT_SUBJECT")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    if already_in_config {
        // Already migrated; just make sure the stale dotenv line is gone.
        if dotenv_subject.is_some() {
            if let Ok(existing) = std::fs::read_to_string(&dotenv_path) {
                let new_text = upsert_dotenv(&existing, &[("BASEVAULT_SUBJECT", None)]);
                let _ = std::fs::write(&dotenv_path, new_text);
                info!("migrate_subject_to_config: stripped stale BASEVAULT_SUBJECT from dotenv");
            }
        }
        return;
    }

    let Some(subj) = dotenv_subject else {
        return; // nothing to migrate
    };

    cfg.insert("subject".to_string(), serde_json::Value::String(subj.clone()));
    let pretty = match serde_json::to_string_pretty(&serde_json::Value::Object(cfg)) {
        Ok(s) => s,
        Err(e) => {
            warn!("migrate_subject_to_config: serialize failed: {}", e);
            return;
        }
    };
    let p = config_path();
    if let Some(parent) = p.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = p.with_extension("json.tmp");
    if std::fs::write(&tmp, pretty + "\n").is_err() {
        return;
    }
    if std::fs::rename(&tmp, &p).is_err() {
        return;
    }

    // Now strip BASEVAULT_SUBJECT from dotenv.
    if let Ok(existing) = std::fs::read_to_string(&dotenv_path) {
        let new_text = upsert_dotenv(&existing, &[("BASEVAULT_SUBJECT", None)]);
        let _ = std::fs::write(&dotenv_path, new_text);
    }
    info!("migrate_subject_to_config: copied BASEVAULT_SUBJECT to config.json + stripped from dotenv");
}

/// One-shot startup migration that retires legacy non-TEE/LOCAL routing
/// values out of `config.json`. The production binary routes user data
/// only through Mode.TEE or Mode.LOCAL; any other `mode` value would
/// hit a KeyError at the runner's mode-string lookup. We rewrite stale
/// configs to the surviving defaults so a returning user with a prior
/// build's `mode=test` (or similar) lands on TEE without an error.
///
/// Idempotent: every call is safe; no state is mutated when nothing
/// matches. Failures are logged and never fatal — worst case the user
/// sees a default mode tab on first run.
fn migrate_legacy_modes(_app: &tauri::AppHandle) {
    let mut cfg = read_config_json();
    let mut changed = false;
    let mut migrated: Vec<&str> = Vec::new();

    let mode = cfg
        .get("mode")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string());
    if let Some(m) = mode {
        let lower = m.to_lowercase();
        if lower != "local" && lower != "tee" && !m.is_empty() {
            cfg.insert("mode".into(), serde_json::Value::String("tee".into()));
            changed = true;
            migrated.push("mode→tee");
        }
    }

    let provider = cfg
        .get("tee_provider")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string());
    if let Some(p) = provider {
        let lower = p.to_lowercase();
        if lower != "tinfoil" && !p.is_empty() {
            cfg.insert(
                "tee_provider".into(),
                serde_json::Value::String("tinfoil".into()),
            );
            changed = true;
            migrated.push("tee_provider→tinfoil");
        }
    }

    if !changed {
        return;
    }

    let pretty = match serde_json::to_string_pretty(&serde_json::Value::Object(cfg)) {
        Ok(s) => s,
        Err(e) => {
            warn!("migrate_legacy_modes: serialize failed: {}", e);
            return;
        }
    };
    let p = config_path();
    if let Some(parent) = p.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = p.with_extension("json.tmp");
    if std::fs::write(&tmp, pretty + "\n").is_err() {
        return;
    }
    if std::fs::rename(&tmp, &p).is_err() {
        return;
    }
    info!(
        "migrate_legacy_modes: rewrote config.json ({})",
        migrated.join(", "),
    );
}

/// Apply a list of upserts to dotenv text. Preserves ordering, comments, and
/// untouched keys. `None` value means "delete any existing line for this key."
fn upsert_dotenv(existing: &str, updates: &[(&str, Option<String>)]) -> String {
    let mut touched: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut lines: Vec<String> = Vec::new();

    for line in existing.lines() {
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            lines.push(line.to_string());
            continue;
        }
        if let Some((k, _)) = trimmed.split_once('=') {
            let k = k.trim();
            if let Some((_, val)) = updates.iter().find(|(uk, _)| *uk == k) {
                match val {
                    Some(v) => {
                        lines.push(format!("{}={}", k, v));
                        touched.insert(k.to_string());
                    }
                    None => {
                        touched.insert(k.to_string());
                        // drop this line
                    }
                }
                continue;
            }
        }
        lines.push(line.to_string());
    }

    for (k, val) in updates {
        if touched.contains(*k) {
            continue;
        }
        if let Some(v) = val {
            lines.push(format!("{}={}", k, v));
        }
    }

    let mut out = lines.join("\n");
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// Run the local-mode setup verifier.
///
/// Setup is detect-only: the script never installs, pulls, or runs
/// brew. It verifies the selected local backend's precondition (MLX
/// model downloaded, or Ollama daemon up + model present) and emits a
/// structured diagnostic on failure. `mode` is accepted for caller
/// back-compat but no longer changes behavior.
#[tauri::command]
async fn setup_local(app: tauri::AppHandle, mode: Option<String>) -> Result<(), String> {
    let args: Vec<String> = match mode.as_deref() {
        Some("verify") => vec!["--verify-only".into()],
        _ => vec![],
    };
    stream_python_progress(&app, "engine.setup_local", args, "setup-progress", "setup_local").await
}

/// `~/.basevault/models/<repo-id>` — on-disk home for a downloaded MLX
/// model snapshot. Mirrors `llm.mlx_model_dir` (Python is the source of
/// truth for the convention); keep the two joins in sync.
fn mlx_model_dir(model_id: &str) -> PathBuf {
    state_root().join("models").join(model_id)
}

/// The bundled MLX model id when the user hasn't overridden it. Mirrors
/// `llm.DEFAULT_MLX_MODEL`; keep in sync.
const DEFAULT_MLX_MODEL: &str = "mlx-community/Qwen3.5-9B-4bit";

/// Parse a "MAJOR.MINOR[.PATCH]" macOS version into (major, minor).
#[cfg(target_os = "macos")]
fn parse_macos_version(s: &str) -> Option<(isize, isize)> {
    let mut parts = s.split('.');
    let major = parts.next()?.trim().parse().ok()?;
    let minor = parts.next().unwrap_or("0").trim().parse().ok()?;
    Some((major, minor))
}

/// Whether the bundled MLX can run on this machine. MLX is
/// Apple-Silicon-macOS only and its `libmlx.dylib` is compiled for the
/// bundle's declared minimum macOS (`bundle.macOS.minimumSystemVersion` in
/// tauri.conf.json — the single source of truth shared with the
/// wheel-pinning build step); older macOS lacks the libc++ symbol it
/// references and crashes on `import mlx.core`. Gates the Local mode picker
/// so the user can't select a dead-end.
#[cfg(target_os = "macos")]
fn mlx_os_supported(app: &tauri::AppHandle) -> bool {
    let floor = app
        .config()
        .bundle
        .macos
        .minimum_system_version
        .as_deref()
        .and_then(parse_macos_version);
    // No declared floor (not a bundled build) → don't gate on OS; the
    // model-download check still applies and there's no floor to enforce.
    let Some(floor) = floor else { return true };
    let v = objc2_foundation::NSProcessInfo::processInfo().operatingSystemVersion();
    (v.majorVersion, v.minorVersion) >= floor
}

#[cfg(not(target_os = "macos"))]
fn mlx_os_supported(_app: &tauri::AppHandle) -> bool {
    false
}

/// Configured MLX model id: config.json `local_mlx_model` else default.
fn configured_mlx_model() -> String {
    get_config()
        .get("local_mlx_model")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| DEFAULT_MLX_MODEL.to_string())
}

fn dir_size_bytes(p: &std::path::Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(p) else {
        return 0;
    };
    let mut total = 0u64;
    for entry in rd.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            total += dir_size_bytes(&entry.path());
        } else if meta.is_file() {
            total += meta.len();
        }
    }
    total
}

/// Spawn a bundled-Python script and stream its JSON progress lines to
/// the frontend on `event`. On non-zero exit, emits a structured error
/// on the same channel (so the UI stops spinning) and returns Err with
/// the stderr tail. Shared by every "long Python task with live
/// progress" command (setup verify, model download).
async fn stream_python_progress(
    app: &tauri::AppHandle,
    module: &'static str,
    args: Vec<String>,
    event: &'static str,
    label: &'static str,
) -> Result<(), String> {
    let py_bin = python_bin(app)?;
    let py_root = python_root_dir(app)?;
    info!("{}: started, python={}", label, py_bin);
    let app_clone = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        use std::io::{BufRead, BufReader};
        use std::process::{Command, Stdio};

        // Launch as a module from the package root so the entrypoint's
        // fully-qualified `from engine.* import …` resolves.
        let mut cmd = Command::new(&py_bin);
        cmd.arg("-m").arg(module).current_dir(&py_root);
        for a in &args {
            cmd.arg(a);
        }
        let mut child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                error!("{}: failed to spawn python: {}", label, e);
                format!("Failed to start Python: {}", e)
            })?;

        // Drain stderr on a background thread so a full pipe buffer never
        // deadlocks the stdout reader.
        let stderr_handle = child.stderr.take().map(|stderr| {
            std::thread::spawn(move || {
                let mut buf = String::new();
                BufReader::new(stderr)
                    .lines()
                    .map_while(Result::ok)
                    .for_each(|line| {
                        buf.push_str(&line);
                        buf.push('\n');
                    });
                buf
            })
        });

        if let Some(stdout) = child.stdout.take() {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                info!("{}: {}", label, line);
                app_clone.emit(event, &line).ok();
            }
        }

        let stderr_output = stderr_handle
            .map(|h| h.join().unwrap_or_default())
            .unwrap_or_default();

        let status = child.wait().map_err(|e| e.to_string())?;
        if !status.success() {
            let tail = tail_lines(&stderr_output, 20);
            error!("{}: exit={:?} stderr=<<<{}>>>", label, status.code(), tail);
            // Surface the failure so the UI stops spinning even when the
            // script died before emitting its own structured error
            // (e.g. an import crash from a broken bundled runtime).
            let msg = serde_json::json!({
                "status": "error",
                "step": "launch",
                "message": format!(
                    "{} exited with {:?}\n{}",
                    label, status.code(), tail
                ),
            })
            .to_string();
            app_clone.emit(event, &msg).ok();
            return Err(format!("{} failed (exit {:?})", label, status.code()));
        }

        info!("{}: completed successfully", label);
        Ok::<(), String>(())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Download the configured (or given) MLX model snapshot from the HF
/// Hub into `~/.basevault/models/<id>/`, streaming progress on
/// `model-download-progress`. User-triggered only — never auto-runs.
#[tauri::command]
async fn download_mlx_model(
    app: tauri::AppHandle,
    model: Option<String>,
) -> Result<(), String> {
    let model = model
        .map(|m| m.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(configured_mlx_model);
    stream_python_progress(
        &app,
        "engine.download_model",
        vec!["--model".into(), model],
        "model-download-progress",
        "download_mlx_model",
    )
    .await
}

/// Status of the configured MLX model: whether it's downloaded and how
/// big it is. Cheap enough for the Settings panel to poll.
#[tauri::command]
fn local_model_status(app: tauri::AppHandle) -> serde_json::Value {
    let model = configured_mlx_model();
    let dir = mlx_model_dir(&model);
    let downloaded = dir.is_dir() && dir.read_dir().map(|mut d| d.next().is_some()).unwrap_or(false);
    serde_json::json!({
        "model": model,
        "downloaded": downloaded,
        "os_supported": mlx_os_supported(&app),
        "path": dir.to_string_lossy(),
        "size_bytes": if downloaded { dir_size_bytes(&dir) } else { 0 },
    })
}

/// Delete a downloaded MLX model snapshot. The UI gates this behind a
/// confirm dialog (multi-GB, accidental deletes are expensive); this
/// command does the filesystem work only.
#[tauri::command]
fn delete_mlx_model(model: Option<String>) -> Result<(), String> {
    let model = model
        .map(|m| m.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(configured_mlx_model);
    let dir = mlx_model_dir(&model);
    if !dir.exists() {
        return Ok(());
    }
    let models_root = state_root().join("models");
    let canonical = dir.canonicalize().map_err(|e| e.to_string())?;
    // Refuse anything that isn't strictly inside the models root —
    // guards a config-supplied id like "../.." from escaping.
    if !canonical.starts_with(models_root.canonicalize().map_err(|e| e.to_string())?) {
        return Err(format!("refusing to delete outside models dir: {:?}", canonical));
    }
    std::fs::remove_dir_all(&canonical).map_err(|e| format!("delete model: {}", e))?;
    // Prune a now-empty org dir (e.g. models/mlx-community/).
    if let Some(parent) = dir.parent() {
        if parent != models_root && parent.read_dir().map(|mut d| d.next().is_none()).unwrap_or(false) {
            let _ = std::fs::remove_dir(parent);
        }
    }
    info!("delete_mlx_model: removed {:?}", canonical);
    Ok(())
}

/// "Remove all data": wipe `~/.basevault/` entirely — models, logs,
/// vault, config, cache. The UI gates this behind a confirm dialog.
/// macOS doesn't run uninstall hooks, so this is the in-app cleanup
/// surface (matches Ollama / LM Studio).
#[tauri::command]
fn reset_basevault() -> Result<(), String> {
    let root = state_root();
    if !root.exists() {
        return Ok(());
    }
    std::fs::remove_dir_all(&root).map_err(|e| format!("reset: {}", e))?;
    info!("reset_basevault: wiped {:?}", root);
    Ok(())
}

#[tauri::command]
fn app_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[derive(Serialize)]
struct UpdateInfo {
    current_version: String,
    available_version: String,
    body: Option<String>,
}

/// Stable channel: only stable releases ever land here. RC channel: the
/// newest build of either kind (a later stable supersedes its own rc by
/// semver precedence, which the updater compares natively — no tag-string
/// matching). The release pipeline writes both manifests on a stable cut
/// and only the rc manifest on an rc cut.
const STABLE_MANIFEST_URL: &str = "https://basevault-releases.s3.amazonaws.com/latest.json";
const RC_MANIFEST_URL: &str = "https://basevault-releases.s3.amazonaws.com/latest-rc.json";

fn include_release_candidates() -> bool {
    get_config()
        .get("include_release_candidates")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Updater pointed at the rc or stable manifest per the developer toggle,
/// read fresh on every call so flipping it takes effect without a restart.
/// Both the check and the install paths go through here so they can never
/// resolve different channels for the same action.
fn channel_updater(app: &tauri::AppHandle) -> Result<tauri_plugin_updater::Updater, String> {
    let url = if include_release_candidates() {
        RC_MANIFEST_URL
    } else {
        STABLE_MANIFEST_URL
    };
    let endpoint = tauri::Url::parse(url).map_err(|e| e.to_string())?;
    app.updater_builder()
        .endpoints(vec![endpoint])
        .map_err(|e| e.to_string())?
        .build()
        .map_err(|e| e.to_string())
}

/// Probe the updater endpoint. Returns `Some(UpdateInfo)` if the remote
/// version is newer than the running build, `None` if up-to-date. Errors
/// surface as `String` so the frontend can show them verbatim.
#[tauri::command]
async fn check_update(app: tauri::AppHandle) -> Result<Option<UpdateInfo>, String> {
    info!("check_update: started");
    let updater = channel_updater(&app).map_err(|e| {
        error!("check_update: updater init failed: {}", e);
        e
    })?;
    match updater.check().await {
        Ok(Some(update)) => {
            info!("check_update: update available: {}", update.version);
            Ok(Some(UpdateInfo {
                current_version: update.current_version.clone(),
                available_version: update.version.clone(),
                body: update.body.clone(),
            }))
        }
        Ok(None) => {
            info!("check_update: no update");
            Ok(None)
        }
        Err(e) => {
            warn!("check_update: failed: {}", e);
            Err(e.to_string())
        }
    }
}

/// Walk up from `current_exe()` looking for the `.app` bundle root on macOS.
/// Returns `Some(/path/to/Foo.app)` for an installed bundle, `None` for a dev
/// binary launched directly (e.g. `target/debug/basevault` from `tauri dev`).
fn macos_app_bundle_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let macos = exe.parent()?;
    if macos.file_name()? != "MacOS" {
        return None;
    }
    let contents = macos.parent()?;
    if contents.file_name()? != "Contents" {
        return None;
    }
    let bundle = contents.parent()?;
    if bundle.extension()? != "app" {
        return None;
    }
    Some(bundle.to_path_buf())
}

/// Download + install the update bundle, then relaunch via `open -n`.
/// The plugin verifies the minisign signature; bad signatures error out
/// before any install happens. Progress is broadcast on `update-progress`:
///   { phase: "download", downloaded, total }   — bytes received so far
///   { phase: "downloaded" }                    — download complete, install starting
///   { phase: "installing" }                    — bundle replacement in progress
///   { phase: "restarting" }                    — about to spawn `open -n`
///
/// We refuse to install when running outside an installed `.app` bundle
/// (e.g. `target/debug/basevault` via `restart.sh` / `tauri dev`). The
/// updater would otherwise log a misleading "install complete" while the
/// running dev binary isn't replaced and `app.restart()` errors with
/// ENOENT against a bundle path it never had.
///
/// Relaunch goes through `/usr/bin/open -n <bundle>` instead of the
/// builtin `app.restart()`. `restart()` re-execs `current_exe()`, but on
/// macOS the path inside the freshly-replaced `.app` may not match the
/// path the running process was launched with (especially when the
/// updater swaps the bundle by `mv`-ing into place). LaunchServices via
/// `open` resolves the bundle through the canonical path and starts a
/// new instance cleanly.
#[tauri::command]
async fn download_and_install_update(app: tauri::AppHandle) -> Result<(), String> {
    info!("download_and_install_update: started");

    let bundle = match macos_app_bundle_path() {
        Some(p) => {
            info!(
                "download_and_install_update: running from .app bundle {}",
                p.display()
            );
            p
        }
        None => {
            let exe = std::env::current_exe().ok();
            warn!(
                "download_and_install_update: not running from a .app bundle (current_exe={:?}); refusing install",
                exe
            );
            return Err(format!(
                "Auto-update only applies to installed BaseVault.app, not dev binaries \
                 (currently running from {}). Quit and relaunch from the installed \
                 BaseVault.app to test the update flow.",
                exe.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "?".into())
            ));
        }
    };

    let updater = channel_updater(&app).map_err(|e| {
        error!("download_and_install_update: updater init failed: {}", e);
        e
    })?;
    let update = match updater.check().await {
        Ok(Some(u)) => u,
        Ok(None) => {
            info!("download_and_install_update: no update available");
            return Err("No update available".to_string());
        }
        Err(e) => {
            warn!("download_and_install_update: check failed: {}", e);
            return Err(e.to_string());
        }
    };

    let app_for_progress = app.clone();
    let app_for_finish = app.clone();
    let mut downloaded: u64 = 0;
    update
        .download_and_install(
            move |chunk_length, content_length| {
                downloaded = downloaded.saturating_add(chunk_length as u64);
                let payload = serde_json::json!({
                    "phase": "download",
                    "downloaded": downloaded,
                    "total": content_length,
                });
                let _ = app_for_progress.emit("update-progress", payload);
            },
            move || {
                info!("download_and_install_update: download finished, installing");
                let _ = app_for_finish.emit(
                    "update-progress",
                    serde_json::json!({ "phase": "downloaded" }),
                );
                let _ = app_for_finish.emit(
                    "update-progress",
                    serde_json::json!({ "phase": "installing" }),
                );
            },
        )
        .await
        .map_err(|e| {
            error!("download_and_install_update: install failed: {}", e);
            e.to_string()
        })?;

    info!(
        "download_and_install_update: install complete, relaunching via `open -n {}`",
        bundle.display()
    );
    let _ = app.emit(
        "update-progress",
        serde_json::json!({ "phase": "restarting" }),
    );

    match std::process::Command::new("/usr/bin/open")
        .arg("-n")
        .arg(&bundle)
        .spawn()
    {
        Ok(_) => {
            // Give LaunchServices a beat to start the new instance before
            // the parent process exits.
            std::thread::sleep(std::time::Duration::from_millis(500));
            info!("download_and_install_update: relaunched, exiting current process");
            app.exit(0);
            Ok(())
        }
        Err(e) => {
            error!("download_and_install_update: failed to spawn `open`: {}", e);
            Err(format!(
                "Update installed but couldn't relaunch automatically. \
                 Quit BaseVault and reopen it. ({})",
                e
            ))
        }
    }
}

/// Set by the window-close handler so the run-loop's `ExitRequested`
/// arm can tell a user window-close (Cmd+W / red button / File ▸ Close
/// Window) apart from a real quit (Cmd+Q, the updater's `app.exit`, or
/// a system logout/shutdown `terminate:`). All of those surface as the
/// same `ExitRequested` event with no reliable built-in discriminator
/// — `code` is `None` for both menu-Quit and window-close — so a
/// window-close raises this flag first and the run loop consumes it.
/// A window-close keeps the process + pipeline subprocesses alive in
/// the background; everything else falls through to the run-pausing
/// cleanup.
static WINDOW_CLOSE_TRIGGERED: AtomicBool = AtomicBool::new(false);

/// Decide whether an `ExitRequested` should keep the process (and its
/// in-flight pipeline runs) alive instead of running the run-pausing
/// cleanup. Only a user window-close qualifies: `code` is `None` (user
/// interaction, not a programmatic `app.exit`) AND a `CloseRequested`
/// raised the flag first. A programmatic exit (`code` = Some: the
/// updater relaunch) or a quit with no preceding window-close (Cmd+Q /
/// menu Quit / system logout `terminate:`) falls through to cleanup.
fn exit_should_keep_alive(code: Option<i32>, from_window_close: bool) -> bool {
    code.is_none() && from_window_close
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Per-app-launch session dir holds the Rust shell's rotating
    // `app.log` and any session-scoped Python artifacts (Tinfoil
    // wire-capture fallback, etc.). One dir per launch, shared with
    // every Python subprocess via `BASEVAULT_SESSION_DIR`. Per-run
    // and per-conversation logs continue to live under their own
    // dirs (`logs/<run-id>/`, `chats/<convo>/`).
    let log_dir = session_dir();

    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::new()
                .level(log::LevelFilter::Info)
                // 5 MB max + KeepAll preserves the full log trail across
                // sessions; rotated files kept as app_<timestamp>.log.
                .max_file_size(5 * 1024 * 1024)
                .rotation_strategy(RotationStrategy::KeepAll)
                .targets([
                    Target::new(TargetKind::Stdout),
                    Target::new(TargetKind::Folder {
                        path: log_dir,
                        file_name: Some("app".to_string()),
                    }),
                ])
                .build(),
        )
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        // Persists window size + position across launches. Without this,
        // every launch resets to the tauri.conf.json default (1200x800).
        // Plugin writes to the OS-standard app config dir; the .conf.json
        // values act as the first-launch default + the floor (minWidth /
        // minHeight prevent persisting a too-small size).
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .manage(AppState::default())
        .setup(|app| {
            // Ensure ~/Documents/BaseVault/ exists eagerly so users can
            // point Obsidian at it from "Open folder as vault" even
            // before they've run their first pipeline.
            let _ = std::fs::create_dir_all(vault_root());
            // Same for the state dir, in case anything below wants to
            // read config.json before the first write.
            let _ = std::fs::create_dir_all(state_root());

            // Move BASEVAULT_SUBJECT from dotenv into config.json (the new
            // rule: dotenv = API keys only, everything else = config.json).
            // Idempotent and silent on no-op.
            migrate_subject_to_config(&app.handle());
            // Rewrite any legacy `mode` / `tee_provider` values that point
            // at retired providers — the production binary routes only
            // through TEE (Tinfoil) or LOCAL. Idempotent.
            migrate_legacy_modes(&app.handle());
            // Orphaned runs (status=running but no process) get flipped to
            // failed so the UI doesn't show them as live.
            cleanup_orphaned_runs(Some(&app.handle()));

            // macOS menu bar: custom BaseVault submenu with About (populated
            // with a description), Settings (Cmd+,), and standard hide/quit.
            // Services is deliberately omitted — friends don't need it, and
            // the menu is cleaner without it.
            let handle = app.handle();
            // macOS About panel: don't set `version` or `short_version`
            // explicitly. AppKit reads them from Info.plist
            // (CFBundleShortVersionString → leading "Version X",
            // CFBundleVersion → parens "(N)"). Tauri's bundler populates
            // both from tauri.conf.json: short version comes from the
            // package version (Cargo.toml [package].version after the
            // release pipeline's sed), build number comes from
            // `bundle.macOS.bundleVersion` (also sed-bumped by the
            // release pipeline to the cumulative `v*` git tag count).
            // Single source of truth on disk; no version strings live in
            // Rust code.
            let about_metadata = AboutMetadataBuilder::new()
                .name(Some("BaseVault"))
                .comments(Some(
                    "BaseVault turns your journals, notes, and chat archives \
                     into a private knowledge base. The pipeline runs on your \
                     machine or inside a trusted execution environment — raw \
                     data never leaves your device unless you explicitly opt \
                     in to a cloud mode."
                ))
                .authors(Some(vec!["BaseVault".to_string()]))
                .website(Some("https://github.com/basevault-ai/basevault"))
                .build();

            let settings_item = MenuItemBuilder::new("Settings…")
                .id("open-settings")
                .accelerator("CmdOrCtrl+,")
                .build(handle)?;

            let wizard_item = MenuItemBuilder::new("Onboarding Wizard…")
                .id("open-wizard")
                .build(handle)?;

            // Name-only "Easy Wizard" (#609). Only added when this
            // build carries the bundled key — otherwise the entry
            // would be a dead end.
            let easy_wizard_item = if bundled_tinfoil_key().is_some() {
                Some(
                    MenuItemBuilder::new("Easy Wizard…")
                        .id("open-easy-wizard")
                        .build(handle)?,
                )
            } else {
                None
            };

            let add_file_item = MenuItemBuilder::new("Add File…")
                .id("menu-add-file")
                .build(handle)?;

            let add_folder_item = MenuItemBuilder::new("Add Folder…")
                .id("menu-add-folder")
                .build(handle)?;

            // Disabled until the frontend reports a non-empty run
            // selection (via `set_export_menu_enabled`) — exporting
            // nothing is a no-op, so the item shouldn't invite the click.
            let export_selected_item = MenuItemBuilder::new("Export Selected…")
                .id("menu-export-selected")
                .enabled(false)
                .build(handle)?;

            let close_item = MenuItemBuilder::new("Close")
                .id("menu-close")
                .accelerator("CmdOrCtrl+W")
                .build(handle)?;

            // about_with_text forces "About BaseVault" — the predefined
            // .about() lets macOS derive the label from the executable
            // name (`basevault`, lowercase) on dev builds.
            let mut app_submenu_b = SubmenuBuilder::new(handle, "BaseVault")
                .about_with_text("About BaseVault", Some(about_metadata))
                .separator()
                .item(&settings_item)
                .item(&wizard_item);
            if let Some(easy) = &easy_wizard_item {
                app_submenu_b = app_submenu_b.item(easy);
            }
            let app_submenu = app_submenu_b
                .separator()
                .hide()
                .hide_others()
                .show_all()
                .separator()
                .quit()
                .build()?;

            // Close lives here (macOS convention puts Close in File, not
            // Window). Labeled just "Close" — the macOS File-menu
            // standard. Cmd+W closes the window, which fires
            // `CloseRequested`; the window-event handler below turns
            // that into a background so the process + in-flight runs
            // survive. This item only invokes that close path; the
            // window-vs-app lifecycle is owned entirely by the handler.
            let file_submenu = SubmenuBuilder::new(handle, "File")
                .item(&add_file_item)
                .item(&add_folder_item)
                .separator()
                .item(&export_selected_item)
                .separator()
                .item(&close_item)
                .build()?;

            let edit_submenu = SubmenuBuilder::new(handle, "Edit")
                .undo()
                .redo()
                .separator()
                .cut()
                .copy()
                .paste()
                .select_all()
                .build()?;

            let window_submenu = SubmenuBuilder::new(handle, "Window")
                .minimize()
                .build()?;

            let menu = MenuBuilder::new(handle)
                .items(&[&app_submenu, &file_submenu, &edit_submenu, &window_submenu])
                .build()?;

            app.set_menu(menu)?;
            app.manage(ExportMenuItem(export_selected_item));

            // Sanctioned attestation call site #3: the periodic
            // background re-attest. Attestation runs from exactly three
            // places — app startup verify, the Settings re-check
            // control, and this hourly timer — and nowhere else; the
            // pipeline / sidecar / inference paths do not attest (per-
            // request attestation is intrinsic to the kernel's attested
            // provider they build). This is a trigger only: it reuses
            // the exact `verify_attestation` path (which spawns the
            // engine's kernel-backed `attestation_view`), reimplementing
            // no attestation logic in Rust. The thread is owned by the
            // process — it has no shutdown channel because app exit
            // tears the process down, which is precisely "stops when
            // the app closes". Provider/model resolve from config.json
            // and the key from the user dotenv, identical to the
            // startup verify.
            //
            // The fresh result is EMITTED to the webview (#926): the
            // always-on attestation indicator must reflect the hourly
            // re-check. Without the emit the UI's attestation state would
            // freeze at whatever the startup verify produced and read as
            // "stale". attestations.jsonl records every run as the audit
            // log; attestation is non-blocking, so this only refreshes
            // what the user SEES, it gates nothing.
            let attest_timer_handle = app.handle().clone();
            std::thread::spawn(move || {
                let interval = std::time::Duration::from_secs(3600);
                loop {
                    std::thread::sleep(interval);
                    match tauri::async_runtime::block_on(verify_attestation(
                        attest_timer_handle.clone(),
                        None,
                        None,
                        None,
                    )) {
                        Ok(r) => {
                            info!(
                                "hourly re-attest: ok={} provider={} model={}",
                                r.ok, r.provider, r.model
                            );
                            if let Err(e) =
                                attest_timer_handle.emit("attestation-updated", r)
                            {
                                warn!("hourly re-attest: emit failed: {}", e);
                            }
                        }
                        Err(e) => warn!("hourly re-attest: spawn error: {}", e),
                    }
                }
            });

            Ok(())
        })
        .on_menu_event(|app, event| {
            match event.id().as_ref() {
                "open-settings" => {
                    let _ = app.emit("open-settings", ());
                }
                "open-wizard" => {
                    let _ = app.emit("open-wizard", ());
                }
                "open-easy-wizard" => {
                    let _ = app.emit("open-easy-wizard", ());
                }
                "menu-add-file" => {
                    let _ = app.emit("menu-add-file", ());
                }
                "menu-add-folder" => {
                    let _ = app.emit("menu-add-folder", ());
                }
                "menu-export-selected" => {
                    let _ = app.emit("menu-export-selected", ());
                }
                "menu-close" => {
                    if let Some(w) = app.get_webview_window("main") {
                        let _ = w.close();
                    }
                }
                _ => {}
            }
        })
        .on_window_event(|_window, event| {
            // Don't call `api.prevent_close()`: we let the window be
            // destroyed (no lingering idle webview) and instead keep the
            // process alive in the run loop's `ExitRequested` arm. Raising
            // the flag here — before the window tears down and the
            // last-window `ExitRequested` fires — is what tells that arm
            // this exit is a backgrounding, not a quit.
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                WINDOW_CLOSE_TRIGGERED.store(true, Ordering::SeqCst);
            }
        })
        .invoke_handler(tauri::generate_handler![
            expand_paths,
            stat_paths,
            estimate_work,
            run_pipeline,
            run_pipeline_step,
            get_output,
            open_output_folder,
            setup_local,
            download_mlx_model,
            local_model_status,
            delete_mlx_model,
            reset_basevault,
            app_version,
            check_update,
            download_and_install_update,
            get_settings,
            save_settings,
            needs_wizard,
            list_runs,
            pause_run,
            cancel_run,
            skip_call,
            resume_run,
            delete_run,
            read_run_preprocessed_inputs,
            write_run_vault,
            open_vault_for_run,
            reveal_in_finder,
            reveal_chats_dir,
            check_obsidian_vault,
            list_run_tree,
            read_run_file,
            export_default_dir,
            export_run,
            export_target_path,
            pick_export_dir,
            read_run_facts,
            read_run_facts_for_topic,
            read_run_facts_all,
            read_run_entities,
            read_run_entity,
            read_run_patterns_for_topic,
            read_run_patterns_all,
            read_run_insights,
            read_run_actions,
            read_run_llm_stats,
            reset_window_size,
            set_export_menu_enabled,
            verify_tinfoil_key,
            verify_attestation,
            open_attestation_log,
            get_config,
            set_config,
            update_config,
            get_llm_cache_stats,
            wipe_llm_cache,
            bust_llm_cache_entry,
            read_llm_cache_entry,
            // dev_tracing — Settings → Development toggle
            dev_tracing_enabled,
            record_dev_trace,
            // Chatbot UI surface (issue #422)
            chatbot,
            chatbot_cancel,
            // Chatbot run selector (issue #507)
            chatbot_list_runs,
            chatbot_select_run,
            // Chat transcript persistence (#453 — kept for back-compat;
            // migration reads the legacy path directly)
            chatbot_load_transcript,
            chatbot_save_transcript,
            // Conversation/thread picker (#565)
            chatbot_list_conversations,
            chatbot_load_conversation,
            chatbot_save_conversation,
            chatbot_new_conversation,
            chatbot_delete_conversation,
            chatbot_set_active_conversation,
            chatbot_rename_conversation,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            match event {
                tauri::RunEvent::ExitRequested { code, api, .. } => {
                    // A user window-close keeps the app + in-flight
                    // pipeline runs alive in the background. The runs are
                    // backend-driven (subprocess + Rust stdout-reader
                    // thread from `spawn_pipeline`); the webview only
                    // polls for display, so destroying it does not stall
                    // progression. The Dock icon stays (default Regular
                    // activation policy, no windows required) and a Dock
                    // click recreates the window via `RunEvent::Reopen`.
                    //
                    // Everything else is a real quit: Cmd+Q / menu Quit,
                    // the updater's `app.exit` relaunch (`code` = Some),
                    // or a system logout/shutdown `terminate:`. Those
                    // SIGKILL active Python subprocesses and mark their
                    // runs "paused" so the user can resume after
                    // re-launch — without this, the subprocesses reparent
                    // to launchd and surface as ghost "cancelled" /
                    // "failed" states or duplicate same-mode runs.
                    let from_window_close =
                        WINDOW_CLOSE_TRIGGERED.swap(false, Ordering::SeqCst);
                    if exit_should_keep_alive(code, from_window_close) {
                        api.prevent_exit();
                    } else {
                        let state: tauri::State<AppState> = app.state();
                        cleanup_active_runs_on_exit(app, &state);
                    }
                }
                tauri::RunEvent::Exit => {
                    let state: tauri::State<AppState> = app.state();
                    cleanup_active_runs_on_exit(app, &state);
                }
                // Dock click on the windowless background app: rebuild
                // the main window from its config so size / min-size /
                // title stay the single source of truth in
                // tauri.conf.json (no hardcoded duplication here). The
                // window-state plugin restores the last geometry. The
                // guard skips the rebuild when a window is already up
                // (e.g. a Dock click while visible).
                #[cfg(target_os = "macos")]
                tauri::RunEvent::Reopen { .. } if app.webview_windows().is_empty() => {
                    match app.config().app.windows.first().cloned() {
                        Some(cfg) => {
                            if let Err(e) =
                                tauri::WebviewWindowBuilder::from_config(app, &cfg)
                                    .and_then(|b| b.build())
                            {
                                error!("Reopen: failed to recreate window: {}", e);
                            }
                        }
                        None => error!("Reopen: no window config to rebuild from"),
                    }
                }
                _ => {}
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Tests that mutate the `HOME` env var must hold this lock — Cargo's
    /// default parallel runner would otherwise stomp HOME mid-test, and
    /// state_root()/logs_root()/vault_root() all read $HOME at call time.
    /// Acquired at the top of every HOME-mutating test; released on drop.
    static HOME_ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_home_dir<F: FnOnce(&std::path::Path)>(f: F) {
        let guard = HOME_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().expect("tempdir");
        let prev = std::env::var("HOME").ok();
        // SAFETY: tests are serialized on HOME_ENV_LOCK so no other test reads
        // $HOME concurrently.
        unsafe { std::env::set_var("HOME", tmp.path()) };
        f(tmp.path());
        match prev {
            Some(v) => unsafe { std::env::set_var("HOME", v) },
            None => unsafe { std::env::remove_var("HOME") },
        }
        drop(guard);
    }

    // ── Pure helpers ────────────────────────────────────────────────────────

    #[test]
    fn ltrace_demotes_only_the_four_per_tick_heartbeats() {
        // The bounded WS-2 demote set — exactly these four relay at
        // debug!; everything else (state-transition skeleton +
        // launch-path) stays info!.
        for m in [
            "pipeline_progress_event",
            "scheduleruns_skipped_pending",
            "refreshruns_fired",
            "runs_received",
        ] {
            assert!(
                ltrace_is_demoted(&format!("[LAUNCH_TRACE] {m} t=1.2 wall=3.4")),
                "{m} should demote",
            );
        }
        for m in [
            "scheduleruns_armed",
            "scheduleruns_fired",
            "scheduleruns_fired_hovering",
            "click",
            "row_clicked",
            "row_first_status_completed",
            "pipeline_progress_listener_registering",
            "run_entry",
            "bufreader_first_line",
        ] {
            assert!(
                !ltrace_is_demoted(&format!("[LAUNCH_TRACE] {m} t=1.2 wall=3.4")),
                "{m} must stay at info",
            );
        }
        // Non-prefixed / malformed lines are never demoted.
        assert!(!ltrace_is_demoted("refreshruns_fired (no prefix)"));
        assert!(!ltrace_is_demoted("[LAUNCH_TRACE] "));
        assert!(!ltrace_is_demoted(""));
    }

    #[test]
    fn iso_z_has_safe_dir_shape() {
        let s = iso_z();
        // YYYY-MM-DDTHH-MM-SSZ — 20 chars, dashes-only separators.
        assert_eq!(s.len(), 20, "unexpected len for {:?}", s);
        assert!(s.ends_with('Z'));
        assert!(!s.contains(':'), "iso_z must not contain colons (dir-name safe)");
    }

    #[test]
    fn iso_z_full_keeps_colons() {
        let s = iso_z_full();
        assert!(s.ends_with('Z'));
        assert!(s.contains(':'), "iso_z_full keeps colons for ISO-8601 fields");
    }

    #[test]
    fn short_id_is_four_chars_from_alphabet() {
        const ALPHABET: &str = "abcdefghjkmnpqrstuvwxyz23456789";
        for _ in 0..50 {
            let id = short_id();
            assert_eq!(id.len(), 4, "{:?}", id);
            for c in id.chars() {
                assert!(ALPHABET.contains(c), "char {:?} not in alphabet", c);
            }
        }
    }

    #[test]
    fn tail_lines_returns_last_n() {
        assert_eq!(tail_lines("a\nb\nc\nd\ne", 3), "c\nd\ne");
        assert_eq!(tail_lines("only-one", 5), "only-one");
        assert_eq!(tail_lines("", 5), "");
    }

    #[test]
    fn sanitize_home_paths_rewrites_users_and_home_prefixes() {
        // Plain Mac home → "~/".
        assert_eq!(
            sanitize_home_paths("at /Users/alice/code/x.py line 12"),
            "at ~/code/x.py line 12",
        );
        // Linux home → "~/".
        assert_eq!(
            sanitize_home_paths("File \"/home/bob/src/m.py\""),
            "File \"~/src/m.py\"",
        );
        // Multiple occurrences across a multi-line traceback.
        let tb = "  File \"/Users/alice/a.py\", line 1\n  File \"/Users/alice/b.py\", line 2";
        assert_eq!(
            sanitize_home_paths(tb),
            "  File \"~/a.py\", line 1\n  File \"~/b.py\", line 2",
        );
        // Non-home paths are left alone.
        assert_eq!(
            sanitize_home_paths("/opt/homebrew/lib/python3.11/site-packages/x.py"),
            "/opt/homebrew/lib/python3.11/site-packages/x.py",
        );
        // Empty input is a no-op.
        assert_eq!(sanitize_home_paths(""), "");
    }

    #[test]
    fn url_encode_handles_special_chars() {
        assert_eq!(url_encode("My Vault"), "My%20Vault");
        assert_eq!(url_encode("a&b#c%d+e?f"), "a%26b%23c%25d%2Be%3Ff");
        assert_eq!(url_encode("plain"), "plain");
    }

    #[test]
    fn is_terminal_status_recognizes_terminal_set() {
        assert!(is_terminal_status("completed"));
        assert!(is_terminal_status("failed"));
        assert!(is_terminal_status("cancelled"));
        // Pause is intentionally NOT terminal — it can transition back to running.
        assert!(!is_terminal_status("paused"));
        assert!(!is_terminal_status("running"));
        assert!(!is_terminal_status(""));
    }

    #[test]
    fn mask_key_hides_all_but_tail() {
        // Empty / very short keys are reported with no tail (defensive — would
        // otherwise leak the entire short value).
        assert_eq!(mask_key(""), "");
        assert_eq!(mask_key("ab"), "••");
        assert_eq!(mask_key("abcd"), "••••");
        // Longer keys: last 4 chars shown, leading bullets capped at 8.
        let masked = mask_key("tinfoil_key_abcd_wxyz");
        assert!(masked.ends_with("wxyz"));
        assert!(masked.starts_with('•'));
        assert!(!masked.contains("abcd_w"), "masked leaked too many chars");
    }

    #[test]
    fn is_pid_alive_recognizes_self_and_rejects_unlikely_pid() {
        // Our own pid is unambiguously alive — kill(self, 0) returns 0.
        assert!(is_pid_alive(std::process::id()));
        // A very high pid that's overwhelmingly unlikely to be in use:
        // kill returns -1 (ESRCH or EPERM); is_pid_alive returns false.
        // Note: kill(1, 0) is NOT a reliable test on macOS — non-root
        // users get EPERM probing launchd, which trips the same -1 path.
        assert!(!is_pid_alive(999_999_999));
    }

    // ── Dotenv parsing + upsert ────────────────────────────────────────────

    #[test]
    fn parse_dotenv_handles_quotes_comments_and_blanks() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join(".env");
        std::fs::write(
            &p,
            "# comment line\n\nFOO=bar\nQUOTED=\"with spaces\"\nSINGLE='single'\nDUP=first\nDUP=second\n",
        )
        .unwrap();
        let parsed = parse_dotenv(&p);
        assert_eq!(parsed.get("FOO"), Some(&"bar".to_string()));
        assert_eq!(parsed.get("QUOTED"), Some(&"with spaces".to_string()));
        assert_eq!(parsed.get("SINGLE"), Some(&"single".to_string()));
        // First-occurrence wins, matching python-dotenv.
        assert_eq!(parsed.get("DUP"), Some(&"first".to_string()));
    }

    #[test]
    fn parse_dotenv_returns_empty_for_missing_file() {
        let parsed = parse_dotenv(std::path::Path::new("/nonexistent/path/.env"));
        assert!(parsed.is_empty());
    }

    #[test]
    fn upsert_dotenv_inserts_when_key_absent() {
        let out = upsert_dotenv("EXISTING=v\n", &[("NEW_KEY", Some("hello".into()))]);
        assert!(out.contains("EXISTING=v"));
        assert!(out.contains("NEW_KEY=hello"));
    }

    #[test]
    fn upsert_dotenv_replaces_existing_key_in_place() {
        let out = upsert_dotenv(
            "FOO=old\nBAR=keep\n",
            &[("FOO", Some("new".into()))],
        );
        assert!(out.contains("FOO=new"));
        assert!(!out.contains("FOO=old"));
        assert!(out.contains("BAR=keep"));
    }

    #[test]
    fn upsert_dotenv_deletes_with_none_value() {
        let out = upsert_dotenv(
            "GONE=val\nKEPT=v2\n",
            &[("GONE", None)],
        );
        assert!(!out.contains("GONE"));
        assert!(out.contains("KEPT=v2"));
    }

    #[test]
    fn upsert_dotenv_preserves_comments_and_order() {
        let existing = "# top comment\nA=1\n# inline comment\nB=2\n";
        let out = upsert_dotenv(existing, &[("B", Some("changed".into()))]);
        // Comments survive
        assert!(out.contains("# top comment"));
        assert!(out.contains("# inline comment"));
        // Order: A still comes before B
        let a_idx = out.find("A=1").unwrap();
        let b_idx = out.find("B=changed").unwrap();
        assert!(a_idx < b_idx);
    }

    #[test]
    fn upsert_dotenv_ends_with_newline() {
        let out = upsert_dotenv("", &[("K", Some("v".into()))]);
        assert!(out.ends_with('\n'));
    }

    // ── Disk-touching commands (HOME-isolated) ──────────────────────────────

    #[test]
    fn config_path_lives_under_state_root() {
        with_home_dir(|home| {
            let p = config_path();
            assert!(p.starts_with(home), "{:?}", p);
            assert!(p.ends_with("config.json"));
        });
    }

    #[test]
    fn set_then_get_config_round_trips() {
        with_home_dir(|_home| {
            let payload = serde_json::json!({
                "subject": "Alice",
                "tee_provider": "tinfoil",
            });
            set_config(payload.clone()).expect("set_config");
            let fetched = get_config();
            assert_eq!(fetched, payload);
        });
    }

    #[test]
    fn chat_transcript_path_lives_under_state_root_not_cache() {
        with_home_dir(|home| {
            let p = chat_transcript_path();
            assert!(p.starts_with(home), "{:?}", p);
            assert!(p.ends_with("transcript.json"));
            // Must NOT live under the cache dir — the cache is wiped
            // routinely; the conversation must survive that.
            assert!(
                !p.starts_with(cache_root()),
                "transcript must not live under the cache root: {:?}",
                p
            );
            assert!(p.starts_with(chats_dir()), "{:?}", p);
        });
    }

    #[test]
    fn save_then_load_transcript_round_trips() {
        with_home_dir(|_home| {
            // Missing file → empty array (a fresh install, or a never-
            // used chat). Must not error. The client treats a bare []
            // as { open:false, turns:[] }.
            assert_eq!(chatbot_load_transcript(), serde_json::json!([]));

            // Real payload is the whole { open, turns } state,
            // including an in-flight (streaming, partial answer) turn —
            // the store is shape-agnostic and round-trips it verbatim.
            let state = serde_json::json!({
                "open": true,
                "turns": [
                    { "id": 1, "q": "my name is alex", "a": "Hi Alex.",
                      "status": "done", "resources": null },
                    { "id": 2, "q": "what's my name", "a": "You're Al",
                      "status": "streaming", "resources": null },
                ],
            });
            chatbot_save_transcript(state.clone()).expect("save");
            assert_eq!(chatbot_load_transcript(), state);
        });
    }

    #[test]
    fn load_transcript_tolerates_a_corrupt_file() {
        with_home_dir(|_home| {
            let p = chat_transcript_path();
            std::fs::create_dir_all(p.parent().unwrap()).expect("mkdir");
            std::fs::write(&p, "{ not json").expect("write garbage");
            // A corrupt transcript must not brick the chat — it reads
            // back as empty, and the next save overwrites it.
            assert_eq!(chatbot_load_transcript(), serde_json::json!([]));
        });
    }

    #[test]
    fn parse_convo_name_iso_prefix_identity_with_free_text_label() {
        // <20-char ISO-Z>-<label>; label is FREE TEXT (rename in
        // scope) — only the ISO prefix is the identity.
        assert_eq!(
            parse_convo_name("2026-05-02T13-30-11Z-conversation-1"),
            Some((
                "2026-05-02T13-30-11Z".to_string(),
                "conversation-1".to_string()
            )),
        );
        assert_eq!(
            parse_convo_name("2026-05-02T13-30-11Z-Tax notes 2026"),
            Some((
                "2026-05-02T13-30-11Z".to_string(),
                "Tax notes 2026".to_string()
            )),
        );
        // Renamed labels (non-`conversation-N`) still parse — the id
        // is the prefix, not the label.
        assert_eq!(
            parse_convo_name("2026-05-02T13-30-11Z-conversation-x")
                .map(|(i, _)| i),
            Some("2026-05-02T13-30-11Z".to_string()),
        );
        // Not conversations: path traversal, a stray #453 transcript
        // file, any dot-prefixed dir, missing/empty label, colon
        // (non-dir-safe) ISO.
        for bad in [
            "..",
            "transcript.json",
            ".hidden-dir",
            "2026-05-02T13-30-11Z",            // no label
            "2026-05-02T13-30-11Z-",           // empty label
            "2026-05-02T13:30:11Z-conversation-1", // colons
            "../2026-05-02T13-30-11Z-x",
        ] {
            assert!(parse_convo_name(bad).is_none(), "should reject: {bad}");
        }
    }

    #[test]
    fn convo_sort_key_uses_last_ts_else_creation() {
        // No messages → key is the creation ISO digits.
        assert_eq!(
            convo_sort_key("2026-05-02T13-30-11Z", None),
            "20260502133011"
        );
        // A last message → key is that instant (UTC, fixed-width),
        // independent of the (here much older) creation prefix.
        let k = convo_sort_key("2020-01-01T00-00-00Z", Some(2_000_000_000_000));
        assert!(k.starts_with("2033"), "key was {k}");
        assert!(
            k > convo_sort_key("2020-01-01T00-00-00Z", None),
            "a later last-message must outrank the creation key"
        );
    }

    #[test]
    fn list_convos_unified_order_new_empty_convo_is_not_dumped_last() {
        with_home_dir(|_home| {
            let base = convos_root();
            let mk = |name: &str, ts: Option<i64>| {
                let dir = base.join(name);
                std::fs::create_dir_all(&dir).unwrap();
                let turns = match ts {
                    Some(t) => serde_json::json!([{ "id": 1, "ts": t }]),
                    None => serde_json::json!([]), // brand-new, no messages
                };
                std::fs::write(
                    dir.join("transcript.json"),
                    serde_json::to_string(
                        &serde_json::json!({ "turns": turns }),
                    )
                    .unwrap(),
                )
                .unwrap();
            };
            // c1: ancient creation but a very recent last message.
            mk("2020-01-01T00-00-00Z-conversation-1", Some(2_000_000_000_000));
            // c2: freshly created, NO messages — its key is its
            // creation time, so it must sort by that (the user bug:
            // a new conversation must NOT fall to the bottom).
            mk("2031-01-01T00-00-00Z-conversation-2", None);
            // c3: created recently-ish but its last message is ancient.
            mk("2030-01-01T00-00-00Z-conversation-3", Some(100));
            // Stray #453 file + non-convo dir: ignored.
            std::fs::write(base.join("transcript.json"), "{}").unwrap();
            std::fs::create_dir_all(base.join("not-a-convo")).unwrap();
            let got: Vec<String> =
                list_convos().into_iter().map(|(_, _, _, n)| n).collect();
            assert_eq!(
                got,
                vec![
                    // c1 last-msg ≈2033 (recent activity wins)…
                    "2020-01-01T00-00-00Z-conversation-1",
                    // c2 no messages → ordered by creation (2031),
                    // ABOVE c3 whose last message is ancient (1970).
                    "2031-01-01T00-00-00Z-conversation-2",
                    "2030-01-01T00-00-00Z-conversation-3",
                ],
            );
        });
    }

    #[test]
    fn save_then_load_by_iso_id_round_trips_and_stamps_identity() {
        with_home_dir(|_home| {
            // create_convo makes the dir; everything is addressed by
            // the immutable ISO prefix, never the dir name.
            let name = create_convo();
            let id = parse_convo_name(&name).unwrap().0;
            let state = serde_json::json!({
                "open": true,
                "turns": [
                    { "id": 1, "q": "hi", "a": "hello", "ts": 42,
                      "status": "done", "resources": null,
                      "runId": "2026-05-01T00-00-00Z-ab3k" },
                ],
            });
            chatbot_save_conversation(id.clone(), state.clone())
                .expect("save");
            let back = chatbot_load_conversation(id.clone());
            assert_eq!(back["open"], serde_json::json!(true));
            // #559 per-turn pinned run round-trips verbatim.
            assert_eq!(
                back["turns"][0]["runId"],
                serde_json::json!("2026-05-01T00-00-00Z-ab3k")
            );
            // Immutable identity stamped in; cid == the ISO prefix.
            assert_eq!(back["cid"], serde_json::json!(id));
            // The label IS the 4-letter perma-id (the dir is
            // `<iso>-<sid>`); the same id is the persisted transcript
            // `short_id` — minted once, no second random mint.
            let lbl = back["label"].as_str().unwrap();
            let sid = back["short_id"].as_str().unwrap();
            assert_eq!(sid.len(), 4, "perma-id is the 4-letter id: {sid}");
            assert_eq!(lbl, sid, "default label IS the perma-id");
            assert_eq!(name, format!("{id}-{sid}"));
            // Unknown id → [] on load, Err on save (path-traversal /
            // junk can't resolve a dir).
            assert_eq!(
                chatbot_load_conversation("../evil".into()),
                serde_json::json!([])
            );
            assert!(chatbot_save_conversation(
                "../evil".into(),
                serde_json::json!({})
            )
            .is_err());
        });
    }

    #[test]
    fn rename_is_a_cosmetic_alias_dir_and_id_never_move() {
        // The dir is named by its 4-letter perma-id (`<iso>-<sid>`);
        // the same id is also persisted in transcript.json. A rename
        // is a cosmetic alias — the dir/id never move, refs hold.
        with_home_dir(|_home| {
            let name = create_convo();
            let id = parse_convo_name(&name).unwrap().0;
            let sid = convo_short_id(&name).unwrap();
            assert_eq!(
                name,
                format!("{id}-{sid}"),
                "dir is `<iso>-<sid>`"
            );
            assert_eq!(sid.len(), 4, "perma-id still minted into transcript");
            chatbot_save_conversation(
                id.clone(),
                serde_json::json!({ "open": true,
                    "turns": [{ "id": 1, "runId": "r-1" }] }),
            )
            .unwrap();
            let meta = chatbot_rename_conversation(
                id.clone(),
                "  Tax/notes.  ".into(), // exercises sanitization
            )
            .expect("rename");
            assert_eq!(meta.id, id, "ISO id immutable across rename");
            assert_eq!(meta.short_id, sid, "perma-id immutable across rename");
            assert_eq!(meta.alias, "Tax-notes"); // `/` → `-`, dots trimmed
            assert_eq!(meta.name, name, "dir name NEVER moves on rename");
            assert!(find_convo_dir(&id).is_some());
            // Refs + turns survive (load by the unchanged id).
            let back = chatbot_load_conversation(id.clone());
            assert_eq!(back["turns"][0]["runId"], serde_json::json!("r-1"));
            // Empty / whitespace-only CLEARS the alias (not an error —
            // the id is always a usable name).
            let cleared = chatbot_rename_conversation(id.clone(), "   ".into())
                .expect("empty clears, not errors");
            assert_eq!(cleared.alias, "");
            assert_eq!(cleared.short_id, sid);
            assert_eq!(cleared.name, name);
        });
    }

    #[test]
    fn convo_perma_id_minted_once_is_stable() {
        // #574/#576: the 4-letter perma-id is minted once at create,
        // used as the dir label (`<iso>-<sid>`), also persisted in
        // transcript.json, surfaces in ConvMeta, and is preserved
        // across a save and an alias rename. No double-mint.
        with_home_dir(|_home| {
            let name = create_convo();
            let id = parse_convo_name(&name).unwrap().0;
            let minted = convo_short_id(&name).expect("minted at create");
            assert_eq!(minted.len(), 4, "4-letter perma-id");
            assert!(
                minted.bytes().all(|b| b"abcdefghjkmnpqrstuvwxyz23456789"
                    .contains(&b)),
                "mint alphabet"
            );
            // The dir label IS the perma-id; create_convo must not
            // double-mint (transcript short_id == label).
            assert_eq!(name, format!("{id}-{minted}"));
            assert_eq!(convo_meta(&name).short_id, minted);
            // A save (whole {open,turns}, no short_id) PRESERVES it.
            chatbot_save_conversation(
                id.clone(),
                serde_json::json!({ "open": true,
                    "turns": [{ "id": 1, "ts": 1700000000000_i64 }] }),
            )
            .unwrap();
            assert_eq!(
                convo_short_id(&find_convo_dir(&id).unwrap()).as_deref(),
                Some(minted.as_str()),
                "perma-id stable across save"
            );
            // An alias rename leaves the perma-id (and dir) untouched.
            let meta = chatbot_rename_conversation(id.clone(), "My Taxes".into())
                .expect("rename");
            assert_eq!(meta.short_id, minted, "perma-id stable across rename");
            assert_eq!(meta.alias, "My Taxes");
            assert_eq!(meta.name, name, "dir unchanged by alias rename");
        });
    }

    #[test]
    fn create_convo_guarantees_a_unique_iso_prefix() {
        with_home_dir(|_home| {
            // Two conversations minted back-to-back (same wall second)
            // must still get DISTINCT ISO prefixes — the prefix is the
            // identity, so a collision would be a data-safety bug.
            let a = create_convo();
            let b = create_convo();
            let ia = parse_convo_name(&a).unwrap().0;
            let ib = parse_convo_name(&b).unwrap().0;
            assert_ne!(ia, ib, "ISO prefixes must be unique: {ia} vs {ib}");
            assert_eq!(list_convos().len(), 2);
        });
    }

    #[test]
    fn default_dir_label_is_the_4_letter_perma_id() {
        // DISK invariant: the dir is `<iso>-<sid>` and the same 4-letter
        // perma-id is in `transcript.json` as `short_id`. The picker
        // uses `display_label` for rendering, NOT the dir tail — that
        // split lives in `display_label_is_derived_at_list_time`.
        with_home_dir(|_home| {
            let name = create_convo();
            let (_iso, label) = parse_convo_name(&name).unwrap();
            assert_eq!(label.len(), 4, "label is the 4-letter perma-id: {label}");
            assert!(
                label.bytes().all(|b| b"abcdefghjkmnpqrstuvwxyz23456789"
                    .contains(&b)),
                "label uses the perma-id mint alphabet: {label}"
            );
            let sid = convo_short_id(&name).expect("perma-id in transcript");
            assert_eq!(sid, label, "transcript short_id == dir label");
            assert_eq!(convo_meta(&name).alias, "", "no alias by default");
        });
    }

    #[test]
    fn display_label_is_derived_at_list_time() {
        // UI invariant: the picker-facing `display_label` is
        // `Conversation <N>`, derived from the positional rank (NOT the
        // dir tail). The dir name stays `<iso>-<sid>` regardless — the
        // disk/display split. The date is intentionally absent: the
        // picker renders it client-side in local tz (a server-formatted
        // date would be UTC, skewing every non-UTC user — same bug class
        // as the run-label fix).
        with_home_dir(|_home| {
            let name = create_convo();
            let meta = convo_meta(&name);
            // Sole convo: N=1, no embedded date.
            assert_eq!(
                meta.display_label, "Conversation 1",
                "single convo gets `Conversation 1`, no baked date"
            );
            assert!(
                !meta.display_label.contains('·'),
                "no server-formatted date segment in the label"
            );
            // The dir tail is still the 4-letter perma-id, untouched
            // by the display derivation.
            assert_eq!(meta.label.len(), 4, "dir label remains the perma-id");
        });
    }

    #[test]
    fn display_label_positional_n_is_iso_ascending() {
        // Seed three convos out-of-order on disk; the display_label N
        // assignment must rank by iso ASCENDING (1-based), regardless
        // of last-activity order, so `Conversation 1` is always the
        // OLDEST creation. Mixed dir-tail shapes (legacy descriptive +
        // new perma-id) parse the same way.
        with_home_dir(|_home| {
            let base = convos_root();
            let mk = |name: &str, last_ts: Option<i64>| {
                let dir = base.join(name);
                std::fs::create_dir_all(&dir).unwrap();
                let turns = match last_ts {
                    Some(t) => serde_json::json!([{ "id": 1, "ts": t }]),
                    None => serde_json::json!([]),
                };
                std::fs::write(
                    dir.join("transcript.json"),
                    serde_json::to_string(
                        &serde_json::json!({ "turns": turns,
                                             "short_id": "abcd" }),
                    )
                    .unwrap(),
                )
                .unwrap();
            };
            // c-old: ancient creation, recent last-activity (sorts FIRST
            // by activity) but N=1 by iso ascending. Legacy descriptive
            // dir tail — still ranks by iso, not by tail.
            mk("2024-01-15T00-00-00Z-conversation-7-jan-15",
               Some(2_000_000_000_000));
            // c-mid: middle iso, no messages.
            mk("2025-06-30T00-00-00Z-mvfe", None);
            // c-new: newest iso, ancient last-activity (sorts LAST by
            // activity) but N=3 by iso ascending.
            mk("2026-05-21T13-39-32Z-44vu", Some(100));
            let got = list_convos();
            // Final order: c-old (recent activity) → c-mid (creation
            // ≥ c-new's ancient activity) → c-new.
            let names: Vec<&str> = got.iter().map(|(_, _, _, n)| n.as_str()).collect();
            assert_eq!(
                names,
                vec![
                    "2024-01-15T00-00-00Z-conversation-7-jan-15",
                    "2025-06-30T00-00-00Z-mvfe",
                    "2026-05-21T13-39-32Z-44vu",
                ],
            );
            // display_label N tracks ISO ascending, NOT this final
            // order. c-old's iso is the smallest → N=1. No baked date
            // (rendered client-side in local tz).
            let labels: Vec<&str> = got.iter().map(|(_, _, d, _)| d.as_str()).collect();
            assert_eq!(
                labels,
                vec![
                    "Conversation 1", // c-old, smallest iso
                    "Conversation 2", // c-mid
                    "Conversation 3", // c-new, largest iso
                ],
            );
        });
    }

    // #485: a file import then an export must not lose import_dir. The
    // old client-side get_config → {...existing} → set_config cycle drops
    // whichever write read its snapshot first; update_config merges
    // server-side so disjoint partial writers compose.
    #[test]
    fn update_config_partial_writers_do_not_clobber_each_other() {
        with_home_dir(|_home| {
            update_config(serde_json::json!({ "import_dir": "/notes" }))
                .expect("import write");
            update_config(serde_json::json!({ "export_dir": "/desktop" }))
                .expect("export write");
            let v = get_config();
            assert_eq!(v["import_dir"], "/notes");
            assert_eq!(v["export_dir"], "/desktop");
        });
    }

    // The lock must serialize the read-merge-write cycle: concurrent
    // disjoint-key writers both survive (no lost update under contention).
    #[test]
    fn update_config_serializes_concurrent_writers() {
        with_home_dir(|_home| {
            let n = 16;
            std::thread::scope(|s| {
                for i in 0..n {
                    s.spawn(move || {
                        update_config(serde_json::json!({ format!("k{i}"): i }))
                            .expect("concurrent update_config");
                    });
                }
            });
            let v = get_config();
            for i in 0..n {
                assert_eq!(v[format!("k{i}")], i, "lost update for k{i}");
            }
        });
    }

    #[test]
    fn update_config_rejects_non_object_patch() {
        with_home_dir(|_home| {
            assert!(update_config(serde_json::json!("nope")).is_err());
        });
    }

    // #489: Settings resets `categories` to the seed default by dropping
    // the key. Expressed as a `null` patch value, it must delete only
    // that key (RFC 7386) and leave a sibling key written by another
    // partial path (e.g. runtime import_dir) untouched — the same
    // no-clobber guarantee as the insert path, for deletions.
    #[test]
    fn update_config_null_value_deletes_only_that_key() {
        with_home_dir(|_home| {
            update_config(serde_json::json!({ "categories": ["x"] }))
                .expect("seed categories");
            update_config(serde_json::json!({ "import_dir": "/notes" }))
                .expect("runtime import_dir");
            update_config(serde_json::json!({ "categories": null }))
                .expect("categories reset = delete");
            let v = get_config();
            assert!(
                v.get("categories").is_none(),
                "null patch must remove the key, not store a literal null"
            );
            assert_eq!(v["import_dir"], "/notes", "sibling key clobbered");
        });
    }

    // #489 repro at the command layer, demonstrating the bug class
    // transition. A runtime Add File persists import_dir. A Settings
    // Save then writes only the keys it owns. The old whole-object
    // set_config (a client-side {...staleSnapshot, ...fields} replace)
    // drops import_dir because the snapshot it replaced from never saw
    // the runtime write; the narrow update_config patch leaves it
    // intact. Same command surface, real config.json on disk.
    #[test]
    fn settings_save_set_config_clobbers_but_update_config_preserves_import_dir() {
        with_home_dir(|_home| {
            // Add File from directory A (runtime path, #485).
            update_config(serde_json::json!({ "import_dir": "/dir/A" }))
                .expect("runtime import_dir");

            // OLD Settings Save: whole-object replace from a snapshot
            // taken before the runtime write existed.
            let stale_snapshot = serde_json::json!({ "subject": "User" });
            set_config(serde_json::json!({
                "subject": stale_snapshot["subject"],
                "sentiment_bias": "uplifting",
            }))
            .expect("legacy whole-object save");
            assert!(
                get_config().get("import_dir").is_none(),
                "the #489 bug: whole-object set_config drops import_dir"
            );

            // NEW Settings Save on the same surface: narrow patch only.
            update_config(serde_json::json!({ "import_dir": "/dir/A" }))
                .expect("re-establish runtime import_dir");
            update_config(serde_json::json!({
                "subject": "User",
                "sentiment_bias": "uplifting",
            }))
            .expect("narrow Settings patch");
            let v = get_config();
            assert_eq!(
                v["import_dir"], "/dir/A",
                "fixed: narrow update_config patch leaves import_dir intact"
            );
            assert_eq!(v["sentiment_bias"], "uplifting", "the edit still lands");
        });
    }

    #[test]
    fn get_config_returns_empty_on_missing_file() {
        with_home_dir(|_home| {
            let v = get_config();
            assert_eq!(v, serde_json::json!({}));
        });
    }

    #[test]
    fn get_config_returns_empty_when_file_unparseable() {
        with_home_dir(|_home| {
            let p = config_path();
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, "<<<not json>>>").unwrap();
            // Defensive: must NOT panic on corrupt config; UI re-saves on next change.
            let v = get_config();
            assert_eq!(v, serde_json::json!({}));
        });
    }

    // ── LLM prompt-hash cache commands ──────────────────────────────────────

    #[test]
    fn cache_root_lives_under_state_root() {
        with_home_dir(|home| {
            let p = cache_root();
            assert!(p.starts_with(home), "{:?}", p);
            assert!(p.ends_with("cache"));
        });
    }

    #[test]
    fn get_llm_cache_stats_is_zero_on_empty() {
        with_home_dir(|_home| {
            let s = get_llm_cache_stats();
            assert_eq!(s.entries, 0);
            assert_eq!(s.bytes, 0);
        });
    }

    #[test]
    fn get_llm_cache_stats_counts_files_and_bytes_across_stages() {
        with_home_dir(|_home| {
            let root = cache_root();
            let extract = root.join("extract");
            let entities = root.join("entities");
            std::fs::create_dir_all(&extract).unwrap();
            std::fs::create_dir_all(&entities).unwrap();
            std::fs::write(extract.join("aaa.json"), "x".repeat(100)).unwrap();
            std::fs::write(extract.join("bbb.json"), "x".repeat(50)).unwrap();
            std::fs::write(entities.join("ccc.json"), "x".repeat(200)).unwrap();

            let s = get_llm_cache_stats();
            assert_eq!(s.entries, 3);
            assert_eq!(s.bytes, 350);
        });
    }

    #[test]
    fn wipe_llm_cache_removes_directory_and_returns_pre_wipe_stats() {
        with_home_dir(|_home| {
            let root = cache_root();
            let bucket = root.join("patterns");
            std::fs::create_dir_all(&bucket).unwrap();
            std::fs::write(bucket.join("hash1.json"), "abcd").unwrap();
            std::fs::write(bucket.join("hash2.json"), "efghij").unwrap();

            let stats = wipe_llm_cache_inner().expect("wipe");
            // Returned counts reflect what was on disk just before deletion.
            assert_eq!(stats.entries, 2);
            assert_eq!(stats.bytes, 10);
            assert!(!root.exists(), "cache root must be gone after wipe");
            // Idempotent: a second wipe on an empty cache returns {0, 0}
            // without erroring.
            let stats2 = wipe_llm_cache_inner().expect("wipe-empty");
            assert_eq!(stats2.entries, 0);
            assert_eq!(stats2.bytes, 0);
        });
    }

    #[test]
    fn wipe_llm_cache_on_missing_root_is_a_no_op() {
        with_home_dir(|_home| {
            assert!(!cache_root().exists());
            let stats = wipe_llm_cache_inner().expect("wipe-missing");
            assert_eq!(stats.entries, 0);
            assert_eq!(stats.bytes, 0);
        });
    }

    #[test]
    fn bust_llm_cache_entry_removes_file_when_present() {
        with_home_dir(|_home| {
            let root = cache_root();
            let bucket = root.join("patterns");
            std::fs::create_dir_all(&bucket).unwrap();
            let key = "abc123def456";
            let path = bucket.join(format!("{}.json", key));
            std::fs::write(&path, "cached body").unwrap();
            assert!(path.is_file(), "fixture must be present pre-bust");

            let removed = bust_llm_cache_entry_inner("patterns".into(), key.into())
                .expect("bust");
            assert!(removed, "should report true when a file was deleted");
            assert!(!path.exists(), "cache entry must be gone after bust");
            // Sibling entry untouched.
            let sibling = bucket.join("xyz.json");
            std::fs::write(&sibling, "untouched").unwrap();
            // Re-busting the same (now-missing) key returns Ok(false), idempotent.
            let again = bust_llm_cache_entry_inner("patterns".into(), key.into())
                .expect("bust-missing");
            assert!(!again);
            assert!(sibling.is_file(), "sibling must survive idempotent re-bust");
        });
    }

    #[test]
    fn bust_llm_cache_entry_returns_false_when_unknown() {
        with_home_dir(|_home| {
            // No cache root, no file. Idempotent → Ok(false).
            let removed =
                bust_llm_cache_entry_inner("patterns".into(), "deadbeef".into())
                    .expect("bust-missing");
            assert!(!removed);
        });
    }

    #[test]
    fn bust_llm_cache_entry_rejects_traversal_keys() {
        with_home_dir(|_home| {
            // Path-traversal in cache_key: refused before touching the FS.
            // sha256-hex never contains '..' or '/' but a tampered run.json
            // could feed one through. Belt-and-braces.
            let err = bust_llm_cache_entry_inner("patterns".into(), "../../etc".into())
                .expect_err("must reject traversal");
            assert!(err.contains("invalid"), "got: {err}");
        });
    }

    #[test]
    fn bust_llm_cache_entry_rejects_empty_key() {
        with_home_dir(|_home| {
            let err = bust_llm_cache_entry_inner("patterns".into(), "".into())
                .expect_err("must reject empty key");
            assert!(err.contains("empty"), "got: {err}");
        });
    }

    #[test]
    fn read_llm_cache_entry_returns_pretty_envelope_for_free_form_response() {
        with_home_dir(|_home| {
            let root = cache_root();
            let bucket = root.join("extract");
            std::fs::create_dir_all(&bucket).unwrap();
            let key = "deadbeef";
            let body = "{\"response\":\"hello\",\"model\":\"foo\"}";
            std::fs::write(bucket.join(format!("{}.json", key)), body).unwrap();

            let got = read_llm_cache_entry("extract".into(), key.into())
                .expect("read");
            // Pretty-printed, response stays a string (not parseable
            // as JSON), envelope shape preserved.
            let parsed: serde_json::Value = serde_json::from_str(&got).unwrap();
            assert_eq!(parsed["response"], "hello");
            assert_eq!(parsed["model"], "foo");
            assert!(got.contains('\n'), "must be pretty-printed");
        });
    }

    #[test]
    fn read_llm_cache_entry_expands_json_string_response() {
        with_home_dir(|_home| {
            // The interesting case: response field is a JSON-stringified
            // JSON (the shape llm_cache.store writes for structured-output
            // stages). Expansion un-escapes it so the clipboard paste is
            // human-readable instead of one long "\"items\":[...]\\n" line.
            let root = cache_root();
            let bucket = root.join("extract");
            std::fs::create_dir_all(&bucket).unwrap();
            let key = "abc123";
            let inner = "{\"items\":[{\"x\":1}]}";
            let body = serde_json::json!({
                "response": inner,
                "model": "gpt-oss-120b",
            }).to_string();
            std::fs::write(bucket.join(format!("{}.json", key)), body).unwrap();

            let got = read_llm_cache_entry("extract".into(), key.into())
                .expect("read");
            let parsed: serde_json::Value = serde_json::from_str(&got).unwrap();
            // response is now an OBJECT, not a string.
            assert!(parsed["response"].is_object(), "response must be expanded to JSON");
            assert_eq!(parsed["response"]["items"][0]["x"], 1);
            // Original escapes (\n / \") gone from the clipboard text.
            assert!(!got.contains("\\\""), "no escaped quotes in pretty output");
        });
    }

    #[test]
    fn read_llm_cache_entry_errors_when_missing() {
        with_home_dir(|_home| {
            let err = read_llm_cache_entry("extract".into(), "nope".into())
                .expect_err("must error on missing");
            assert!(err.contains("not found"), "got: {err}");
        });
    }

    #[test]
    fn read_llm_cache_entry_rejects_traversal() {
        with_home_dir(|_home| {
            let err = read_llm_cache_entry("extract".into(), "../../etc/passwd".into())
                .expect_err("must reject traversal");
            assert!(err.contains("invalid"), "got: {err}");
        });
    }

    #[test]
    fn read_llm_cache_entry_rejects_empty_key() {
        with_home_dir(|_home| {
            let err = read_llm_cache_entry("extract".into(), "".into())
                .expect_err("must reject empty key");
            assert!(err.contains("empty"), "got: {err}");
        });
    }

    fn write_run_json(
        logs_dir: &std::path::Path,
        session: &str,
        eval: &str,
        run_id: &str,
        body: serde_json::Value,
    ) -> std::path::PathBuf {
        let dir = logs_dir.join(session).join(eval).join(run_id);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("run.json");
        std::fs::write(&p, serde_json::to_string_pretty(&body).unwrap()).unwrap();
        p
    }

    #[test]
    fn find_run_json_locates_nested_run() {
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            let written = write_run_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-abcd",
                serde_json::json!({"status": "running", "agent": "app"}),
            );
            let found = find_run_json("run-abcd").expect("should find run");
            assert_eq!(found, written);
        });
    }

    /// Helper: drop a config.json sibling in the run dir.
    fn write_config_json(
        logs_dir: &std::path::Path,
        session: &str,
        eval: &str,
        run_id: &str,
        body: serde_json::Value,
    ) -> std::path::PathBuf {
        let dir = logs_dir.join(session).join(eval).join(run_id);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("config.json");
        std::fs::write(&p, serde_json::to_string_pretty(&body).unwrap()).unwrap();
        p
    }

    #[test]
    fn find_run_json_prefers_config_json_when_both_present() {
        // Post-#165: writers write config.json (static) AND run.json
        // (dynamic, until commit 2). The find-walk should pick the new
        // canonical sidecar so reads of static fields (mode, inputs)
        // come from the immutable file rather than the live one.
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            write_run_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-both",
                serde_json::json!({"status": "running", "agent": "app"}),
            );
            let cfg_path = write_config_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-both",
                serde_json::json!({"agent": "app", "mode": "tee"}),
            );
            let found = find_run_json("run-both").expect("should find");
            assert_eq!(
                found, cfg_path,
                "config.json must win over run.json when both exist"
            );
        });
    }

    #[test]
    fn find_run_json_falls_back_to_run_json_for_legacy_runs() {
        // Old runs predate the split — only run.json on disk. The
        // fallback keeps them visible in the runs list.
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            let rj = write_run_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-legacy",
                serde_json::json!({"status": "completed", "agent": "app"}),
            );
            let found = find_run_json("run-legacy").expect("legacy lookup");
            assert_eq!(found, rj);
        });
    }

    /// Helper: drop a config.json directly under `<logs>/<run_id>/` —
    /// the flat post-flatten layout.
    fn write_flat_config_json(
        logs_dir: &std::path::Path,
        run_id: &str,
        body: serde_json::Value,
    ) -> std::path::PathBuf {
        let dir = logs_dir.join(run_id);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("config.json");
        std::fs::write(&p, serde_json::to_string_pretty(&body).unwrap()).unwrap();
        p
    }

    #[test]
    fn walk_finds_flat_run_dirs() {
        // Post-flatten app runs live directly under <logs>/<run_id>/.
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            let cfg = write_flat_config_json(
                &logs,
                "2026-05-08T10-00-00Z-flat",
                serde_json::json!({"agent": "app", "mode": "local"}),
            );
            let sidecars = walk_run_jsons(&logs);
            assert!(
                sidecars.iter().any(|p| p == &cfg),
                "flat run not found: walk returned {:?}",
                sidecars
            );
        });
    }

    #[test]
    fn walk_finds_legacy_nested_run_dirs() {
        // Legacy <logs>/<session>/<eval>/<run>/ shape must keep working
        // so corpora on disk from before the flatten remain readable.
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            let cfg = write_config_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-nested",
                serde_json::json!({"agent": "app", "mode": "tee"}),
            );
            let sidecars = walk_run_jsons(&logs);
            assert!(
                sidecars.iter().any(|p| p == &cfg),
                "nested run not found: walk returned {:?}",
                sidecars
            );
        });
    }

    #[test]
    fn walk_skips_sweeps_namespace() {
        // The sweep harness lands runs at <logs>/sweeps/<sweep_id>/<run>/.
        // walk_run_jsons must NOT index them — they're agent="experiment"
        // and served to the eval tooling, not the app's runs list.
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            let dir = logs
                .join("sweeps")
                .join("2026-05-08T10-00-00Z-experiment-test")
                .join("01_case_a-tee");
            std::fs::create_dir_all(&dir).unwrap();
            let cfg = dir.join("config.json");
            std::fs::write(
                &cfg,
                serde_json::to_string_pretty(
                    &serde_json::json!({"agent": "experiment", "mode": "tee"}),
                )
                .unwrap(),
            )
            .unwrap();
            // Drop a flat app run alongside so the assertion isn't
            // vacuously true on an empty walk.
            let app_cfg = write_flat_config_json(
                &logs,
                "2026-05-08T10-30-00Z-app",
                serde_json::json!({"agent": "app"}),
            );
            let sidecars = walk_run_jsons(&logs);
            assert!(
                sidecars.iter().any(|p| p == &app_cfg),
                "app run not found"
            );
            assert!(
                !sidecars.iter().any(|p| p == &cfg),
                "sweep run leaked into app walk: {:?}",
                sidecars
            );
        });
    }

    #[test]
    fn walk_finds_mixed_layouts_simultaneously() {
        // Migration window: flat new runs coexist with legacy nested
        // runs in the same logs root. Both shapes must be picked up by
        // a single walk so the runs list stays complete during the
        // transition.
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            let flat_cfg = write_flat_config_json(
                &logs,
                "2026-05-08T11-00-00Z-flat",
                serde_json::json!({"agent": "app"}),
            );
            let legacy_cfg = write_config_json(
                &logs,
                "2026-04-20T12-00-00Z-app",
                "eval-1",
                "run-legacy",
                serde_json::json!({"agent": "app"}),
            );
            let sidecars = walk_run_jsons(&logs);
            assert!(sidecars.iter().any(|p| p == &flat_cfg));
            assert!(sidecars.iter().any(|p| p == &legacy_cfg));
        });
    }

    #[test]
    fn walk_does_not_descend_into_run_internals() {
        // A run dir contains stages/<n>/, which themselves contain
        // documents/ and per-topic dirs. None of those carry a sidecar,
        // but the walk MUST NOT descend into them — partly perf, mostly
        // because if a future change drops a config.json into a stage
        // dir for some reason, we'd misclassify it as a run.
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            let cfg = write_flat_config_json(
                &logs,
                "2026-05-08T12-00-00Z-int",
                serde_json::json!({"agent": "app"}),
            );
            // Stage subdirs nested several levels deep, no sidecars.
            let stage_dir = cfg
                .parent()
                .unwrap()
                .join("stages")
                .join("01-extraction")
                .join("facts");
            std::fs::create_dir_all(&stage_dir).unwrap();
            std::fs::write(stage_dir.join("topic.jsonl"), "{}\n").unwrap();
            // Walk should still return exactly the run's sidecar.
            let sidecars = walk_run_jsons(&logs);
            assert_eq!(sidecars.len(), 1);
            assert_eq!(sidecars[0], cfg);
        });
    }

    #[test]
    fn find_run_json_locates_flat_run() {
        // The find walk must accept flat layout the same way the bulk
        // walk does — used by stop_inflight_pipeline, set_run_status,
        // open_run_dir, and the resume path.
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            let cfg = write_flat_config_json(
                &logs,
                "2026-05-08T13-00-00Z-find",
                serde_json::json!({"agent": "app"}),
            );
            let found = find_run_json("2026-05-08T13-00-00Z-find")
                .expect("flat lookup");
            assert_eq!(found, cfg);
        });
    }

    #[test]
    fn read_run_state_prefers_config_json() {
        // Static fields must come from config.json when it exists, even
        // if run.json carries a different value (e.g. a stale leftover
        // mode from a malformed write).
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            write_run_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-prefer",
                serde_json::json!({"agent": "app", "mode": "stale-from-runjson"}),
            );
            let cfg = write_config_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-prefer",
                serde_json::json!({"agent": "app", "mode": "tee"}),
            );
            let dir = cfg.parent().unwrap();
            let v = read_run_state(dir).expect("read");
            assert_eq!(v["mode"], "tee");
        });
    }

    #[test]
    fn read_run_state_falls_back_to_run_json() {
        // No config.json on disk → take whatever run.json carries.
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            let rj = write_run_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-fallback",
                serde_json::json!({"agent": "app", "mode": "local"}),
            );
            let dir = rj.parent().unwrap();
            let v = read_run_state(dir).expect("read");
            assert_eq!(v["mode"], "local");
        });
    }

    #[test]
    fn list_runs_merges_static_config_with_legacy_dynamic_fields() {
        // During the Commit 1 → Commit 2 transition, new runs land
        // both files: config.json (static) and run.json (dynamic). The
        // list must surface both so the JSX continues to read status /
        // progress without forking the Rust shape.
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            // run.json carries the dynamic state.
            write_run_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-merge",
                serde_json::json!({
                    "status": "running",
                    "progress": {"stage": "extract", "completed": 5},
                    "agent": "app",
                    "mode": "tee",
                    "created_at": "2026-04-29T12:00:00Z",
                }),
            );
            // config.json carries the immutable static snapshot.
            write_config_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-merge",
                serde_json::json!({
                    "agent": "app",
                    "mode": "tee",
                    "created_at": "2026-04-29T12:00:00Z",
                    "run_id": "run-merge",
                    "inputs": ["/tmp/a.txt"],
                }),
            );
            let runs = list_runs();
            assert_eq!(runs.len(), 1);
            // Static field survives from config.json…
            assert_eq!(runs[0]["inputs"][0], "/tmp/a.txt");
            // …and dynamic fields are merged in from the legacy sidecar.
            assert_eq!(runs[0]["status"], "running");
            assert_eq!(runs[0]["progress"]["stage"], "extract");
        });
    }

    #[test]
    fn find_run_json_returns_none_when_missing() {
        with_home_dir(|_home| {
            assert!(find_run_json("nonexistent-run").is_none());
        });
    }

    #[test]
    fn short_id_from_run_name_extracts_only_a_valid_4_letter_suffix() {
        // Canonical `<iso-z>-<short_id>` → the trailing 4-char id.
        assert_eq!(
            short_id_from_run_name("2026-05-17T09-00-00Z-ab3k").as_deref(),
            Some("ab3k")
        );
        // Mint alphabet excludes 0/1/i/l/o — a suffix using one of
        // those isn't a minted id, so don't fabricate one.
        assert_eq!(short_id_from_run_name("2026-05-17T09-00-00Z-ab1k"), None);
        // Wrong length / non-canonical names yield nothing.
        assert_eq!(short_id_from_run_name("2026-05-17T09-00-00Z-abc"), None);
        assert_eq!(short_id_from_run_name("run-merge"), None);
    }

    #[test]
    fn list_runs_derives_perma_id_from_dir_when_config_lacks_short_id() {
        // Fix B: a config.json written before provider/model pick (or a
        // legacy run.json) has no `short_id`. The dir leaf
        // `<iso-z>-<id>` is the immutable record, so the list must
        // still surface the 4-letter perma-id (and run_id) from it —
        // the id is never lost from the UI.
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            write_config_json(
                &logs,
                "2026-05-17T09-00-00Z-sess",
                "eval-1",
                "2026-05-17T09-00-00Z-ab3k",
                serde_json::json!({
                    "agent": "app",
                    "mode": "local",
                    "created_at": "2026-05-17T09:00:00Z",
                }),
            );
            let runs = list_runs();
            assert_eq!(runs.len(), 1);
            assert_eq!(runs[0]["short_id"], "ab3k");
            assert_eq!(runs[0]["run_id"], "2026-05-17T09-00-00Z-ab3k");
        });
    }

    #[test]
    fn list_runs_surfaces_duration_ms_for_finished_run_with_dashes_ts() {
        // Cross-layer regression pin (#261): the runs-list card's
        // total-duration value comes from `duration_ms` on the record
        // list_runs returns, which is stamped by overlay_derived_dynamic_fields
        // from the iso_delta_ms read of the on-disk cycle_start /
        // cycle_end ts. Both events use the dashes form `_iso_z()`
        // emits — so this test pins (a) that the parser still accepts
        // dashes and (b) that the materializer still surfaces the
        // derived value to the UI shape.
        clear_derive_cache_for_tests();
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            write_run_json(
                &logs,
                "2026-05-08T20-47-24Z-app",
                "eval-1",
                "run-dur",
                serde_json::json!({
                    "status": "running",
                    "progress": {"stage": "extract", "completed": 0},
                    "agent": "app",
                    "mode": "tee",
                    "created_at": "2026-05-08T20:47:24Z",
                }),
            );
            write_config_json(
                &logs,
                "2026-05-08T20-47-24Z-app",
                "eval-1",
                "run-dur",
                serde_json::json!({
                    "agent": "app",
                    "mode": "tee",
                    "created_at": "2026-05-08T20:47:24Z",
                    "run_id": "run-dur",
                    "inputs": ["/tmp/a.txt"],
                }),
            );
            // On-disk cycle markers in the dashes form runner.py
            // actually writes (`_iso_z()` — filename-safe, no colons).
            let run_dir = write_run_jsonl(
                &logs,
                "2026-05-08T20-47-24Z-app",
                "eval-1",
                "run-dur",
                &[
                    serde_json::json!({"event": "cycle_start", "ts": "2026-05-08T20-47-24Z"}),
                    serde_json::json!({"event": "cycle_end",   "ts": "2026-05-08T20-49-24Z"}),
                ],
            );
            // Stage-5 marker is the canonical completed signal post-#206;
            // cycle_end alone derives as paused (the runner's SIGTERM
            // handler emits cycle_end too).
            write_last_stage_marker(&run_dir);
            let runs = list_runs();
            assert_eq!(runs.len(), 1);
            // Status flipped to completed by the stage-5 marker.
            assert_eq!(runs[0]["status"], "completed");
            // 2 minutes between the dashes-form cycle_start and cycle_end.
            assert_eq!(runs[0]["duration_ms"], 120_000);
        });
    }

    // ── Run state derivation from llm-calls.jsonl ────────────────────────────

    /// Helper: write a jsonl event log under a run dir.
    fn write_run_jsonl(
        logs_dir: &std::path::Path,
        session: &str,
        eval: &str,
        run_id: &str,
        events: &[serde_json::Value],
    ) -> std::path::PathBuf {
        let dir = logs_dir.join(session).join(eval).join(run_id);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("llm-calls.jsonl");
        let body = events
            .iter()
            .map(|e| serde_json::to_string(e).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        std::fs::write(&p, body).unwrap();
        dir
    }

    /// Stage-5 terminal marker — the canonical "the entire pipeline
    /// finished" signal per `vault/os/dev/technical.md` § "Pipeline
    /// Stages". Tests that need to exercise the completed branch
    /// materialize this file.
    fn write_last_stage_marker(run_dir: &std::path::Path) {
        let p = run_dir
            .join("stages")
            .join("06-embeddings")
            .join("phase_1_marker.json");
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, "{}").unwrap();
    }

    #[test]
    fn derive_status_running_when_cycle_start_only() {
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        // Use the test process's own pid — guaranteed alive while the
        // test runs, so the running/paused disambiguation in derive
        // returns "running" for this fixture.
        let alive_pid = std::process::id();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[serde_json::json!({
                "event": "cycle_start",
                "ts": "2026-05-07T12:00:00Z",
                "pid": alive_pid,
            })],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.status, "running");
        assert_eq!(s.pid, Some(alive_pid));
    }

    #[test]
    fn derive_status_completed_when_last_stage_marker_present() {
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        // Production runner.py emits cycle_* `ts` fields via `_iso_z()`
        // (filename-safe dashes form `HH-MM-SS`), not the colon form.
        // The test must exercise the on-disk bytes the parser actually
        // sees — earlier colon-format fixtures masked a parser bug that
        // dropped duration_ms for every post-#169 run on disk.
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12-00-00Z"}),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12-01-00Z"}),
            ],
        );
        // The marker is what makes a run "completed" — cycle_end alone
        // is also emitted on SIGTERM-pause (issue #206).
        write_last_stage_marker(&dir);
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.status, "completed");
        // 60_000 ms = 1 minute between cycle_start and cycle_end.
        assert_eq!(s.duration_ms, Some(60_000));
    }

    #[test]
    fn derive_status_paused_when_cycle_end_present_but_marker_missing() {
        // Issue #206 repro: SIGTERM-paused run emits cycle_end via the
        // signal handler's rollup-flush, but no stage-5 marker. The
        // pre-fix derivation classified this as "completed" and the
        // resume button never surfaced. Post-fix: marker is the only
        // signal for "completed"; the dead runner pid + missing marker
        // resolves to "paused".
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({
                    "event": "cycle_start",
                    "ts": "2026-05-09T18-32-03Z",
                    "pid": 999_999_999u64,
                }),
                serde_json::json!({
                    "event": "cycle_end",
                    "ts": "2026-05-09T18-32-37Z",
                    "reason": "sigterm",
                }),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.status, "paused");
    }

    #[test]
    fn derive_status_paused_when_pid_dead_and_marker_missing() {
        // Bare "runner died, no terminator emitted" case: cycle_start
        // landed, no cycle_end / cycle_error / cycle_cancelled, no
        // marker, no paused.flag. The dead pid is what flips this from
        // "running" → "paused" so the user can resume from the latest
        // phase marker (which the runner picks up on resume).
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[serde_json::json!({
                "event": "cycle_start",
                "ts": "2026-05-09T18-32-03Z",
                "pid": 999_999_999u64,
            })],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.status, "paused");
    }

    #[test]
    fn iso_delta_ms_accepts_dashes_and_colons() {
        // Cross-format pairing: cycle_start / cycle_end use dashes
        // (runner.py `_iso_z()`), while begin/end + progress_tick stamps
        // for in-flight runs use colons (Rust `iso_z_full()` and Python
        // `_iso_z_full()`). Both must parse, and a mixed pair must
        // resolve too — the runs-list duration is derived from the
        // dashes pair, but other code paths feed in colon stamps.
        assert_eq!(
            iso_delta_ms("2026-05-07T12-00-00Z", "2026-05-07T12-01-00Z"),
            Some(60_000)
        );
        assert_eq!(
            iso_delta_ms("2026-05-07T12:00:00Z", "2026-05-07T12:01:00Z"),
            Some(60_000)
        );
        assert_eq!(
            iso_delta_ms("2026-05-07T12-00-00Z", "2026-05-07T12:01:00Z"),
            Some(60_000)
        );
    }

    #[test]
    fn derive_status_cancelled_after_cycle_cancelled_event() {
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "cycle_cancelled", "ts": "2026-05-07T12:00:30Z"}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.status, "cancelled");
    }

    #[test]
    fn derive_status_failed_after_cycle_error_event() {
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({
                    "event": "cycle_error",
                    "ts": "2026-05-07T12:00:30Z",
                    "message": "subprocess crashed",
                }),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.status, "failed");
        assert_eq!(s.error.as_deref(), Some("subprocess crashed"));
    }

    #[test]
    fn derive_status_paused_when_marker_present() {
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"})],
        );
        // paused.flag is the runtime control marker (Rust writes it on
        // pause_run since the runner is dead and can't emit a jsonl).
        std::fs::write(dir.join("paused.flag"), "user paused").unwrap();
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.status, "paused");
    }

    #[test]
    fn derive_terminator_precedence_error_beats_end() {
        // Sticky terminal: a cycle_error after cycle_end takes precedence
        // — captures crash-on-shutdown paths where both events landed.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:00:30Z"}),
                serde_json::json!({
                    "event": "cycle_error",
                    "ts": "2026-05-07T12:00:31Z",
                    "message": "boom",
                }),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.status, "failed");
    }

    // ── duration_ms: active-runtime accumulator (#617) ────────────────

    #[test]
    fn derive_duration_ms_single_cycle_completed_sums_one_pair() {
        // cycle_start → cycle_end. Single closed pair, no in-flight
        // extension. duration_ms = (cycle_end − cycle_start) = 30s.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:00:30Z"}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.duration_ms, Some(30_000));
    }

    #[test]
    fn derive_duration_ms_single_cycle_cancelled_freezes_at_cycle_cancelled() {
        // cycle_start → cycle_cancelled. Closes the slot like cycle_end.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "cycle_cancelled", "ts": "2026-05-07T12:00:42Z"}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.status, "cancelled");
        assert_eq!(s.duration_ms, Some(42_000));
    }

    #[test]
    fn derive_duration_ms_single_cycle_failed_freezes_at_cycle_error() {
        // cycle_start → cycle_error. Closes the slot like cycle_end.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({
                    "event": "cycle_error",
                    "ts": "2026-05-07T12:00:15Z",
                    "message": "boom",
                }),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.status, "failed");
        assert_eq!(s.duration_ms, Some(15_000));
    }

    #[test]
    fn derive_duration_ms_multi_cycle_pause_window_excluded() {
        // Two cycles separated by a pause window. duration_ms is the
        // sum of (cycle_start, cycle_end) pairs only — the pause gap
        // between cycle 1's cycle_end and cycle 2's cycle_start is
        // NOT counted. 5m + 30s = 5m30s.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                // Cycle 1: 12:00:00 → 12:05:00 (5m).
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:05:00Z", "reason": "sigterm"}),
                // 5m pause window (12:05:00 → 12:10:00) — excluded.
                // Cycle 2: 12:10:00 → 12:10:30 (30s).
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:10:00Z", "is_resume": true}),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:10:30Z", "reason": "sigterm"}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.duration_ms, Some(5 * 60_000 + 30_000));
    }

    #[test]
    fn derive_duration_ms_pause_resume_cancel_sums_all_closed_cycles() {
        // pause → resume → cancel: duration_ms = (cycle_1 closed) +
        // (cycle_2 closed by cycle_cancelled). Pause window excluded.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:00:10Z", "reason": "sigterm"}),
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:05:00Z", "is_resume": true}),
                serde_json::json!({"event": "cycle_cancelled", "ts": "2026-05-07T12:05:05Z"}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.status, "cancelled");
        // 10s + 5s = 15s. Pause window 12:00:10 → 12:05:00 (~5min) excluded.
        assert_eq!(s.duration_ms, Some(15_000));
    }

    #[test]
    fn derive_duration_ms_duplicate_cycle_end_does_not_double_count() {
        // Python's SIGTERM handler can emit a cycle_end after another
        // upstream caller already wrote one. The accumulator closes
        // the slot on the FIRST terminator and skips subsequent
        // terminators that see no open slot. Without this guarantee
        // duration_ms would be double-counted (or worse, negative).
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:00:30Z", "reason": "first"}),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:00:31Z", "reason": "sigterm"}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        // First cycle_end closes at +30s. Second cycle_end has no
        // open slot to close — duration stays at 30s.
        assert_eq!(s.duration_ms, Some(30_000));
    }

    #[test]
    fn derive_duration_ms_orphan_cycle_dropped_by_next_cycle_start() {
        // A cycle_start with no matching terminator, followed by
        // ANOTHER cycle_start (e.g. cleanup_orphaned_runs SIGKILL'd
        // the previous Python without giving its signal handler a
        // chance to emit cycle_end). The latest cycle_start
        // overwrites open_cycle_start_ts — the orphan's runtime is
        // silently dropped. The new (currently in-flight) cycle is
        // the only contribution.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        // Use the test process's own pid so derive's pid-alive check
        // returns true → status = "running".
        let alive_pid = std::process::id();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                // Orphan cycle: cycle_start without cycle_end.
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z", "pid": alive_pid}),
                // New cycle_start fires (resume after orphan recovery).
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:01:00Z", "is_resume": true, "pid": alive_pid}),
                // Latest cycle closes at +30s.
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:01:30Z"}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        // The orphan's 60s is dropped; only the closed second cycle
        // contributes its 30s.
        assert_eq!(s.duration_ms, Some(30_000));
    }

    #[test]
    fn derive_duration_ms_running_extends_open_slot_to_now() {
        // While running (no terminator after the latest cycle_start),
        // the accumulator extends by (now - latest_open_start). The
        // extension lives in `derive_run_state`, not the uncached
        // walker — uncached returns only the closed accumulator
        // plus the open-cycle anchor. Use 1970-01-01 as cycle_start
        // to guarantee `now - start` is huge regardless of when the
        // test runs.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let alive_pid = std::process::id();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[serde_json::json!({
                "event": "cycle_start",
                "ts": "1970-01-01T00:00:00Z",
                "pid": alive_pid,
            })],
        );
        let s = derive_run_state(&dir);
        assert_eq!(s.status, "running");
        // duration_ms should be ~now-since-epoch, i.e. tens of years
        // of ms. Floor at 30 years to verify the extension fires.
        let thirty_years_ms: u64 = 30 * 365 * 24 * 3600 * 1000;
        assert!(
            s.duration_ms.unwrap_or(0) > thirty_years_ms,
            "running duration should extend to now; got {:?}",
            s.duration_ms
        );
    }

    #[test]
    fn derive_run_state_running_duration_ticks_between_walks_when_jsonl_unchanged() {
        // The derive cache is keyed on jsonl / paused.flag /
        // last-stage-marker mtimes — all event-derived. If the
        // wallclock extension `(now - open_start)` were baked into
        // the cached duration_ms, a cache HIT would return a frozen
        // value between jsonl writes — and the displayed run-row
        // elapsed would only tick on LLM-call events. Two calls to
        // `derive_run_state` (the cached entry point) at different
        // wallclock instants must produce a strictly greater
        // duration on a running run even though the jsonl mtime is
        // identical across both calls.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[serde_json::json!({"event": "cycle_start", "ts": "1970-01-01T00:00:00Z"})],
        );
        let first = derive_run_state(&dir);
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let second = derive_run_state(&dir);
        assert_eq!(first.status, "running");
        assert_eq!(second.status, "running");
        let d1 = first.duration_ms.unwrap_or(0);
        let d2 = second.duration_ms.unwrap_or(0);
        assert!(
            d2 >= d1 + 1000,
            "running duration must advance with wallclock between cache hits; \
             d1={} d2={} delta={}",
            d1,
            d2,
            d2.saturating_sub(d1),
        );
    }

    #[test]
    fn derive_duration_ms_completed_cycle_does_not_change_after_walk() {
        // Repeatability: a completed cycle_start/cycle_end pair always
        // produces the same duration_ms. The accumulator must not be
        // affected by `iso_z_full()`'s now-value across re-derivations.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:00:30Z"}),
            ],
        );
        let first = derive_run_state_uncached(&dir);
        std::thread::sleep(std::time::Duration::from_millis(50));
        let second = derive_run_state_uncached(&dir);
        assert_eq!(first.duration_ms, second.duration_ms);
        assert_eq!(first.duration_ms, Some(30_000));
    }

    #[test]
    fn derive_status_unknown_when_jsonl_empty() {
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("sess").join("eval-1").join("run-x");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("llm-calls.jsonl"), "").unwrap();
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.status, "unknown");
    }

    #[test]
    fn derive_progress_completed_counts_leaves_no_retries() {
        // Baseline: 3 calls, 2 ended, 0 retries → completed = 2
        // (each end is a leaf since no retry_of_call_id chains).
        // The stage label comes from progress_tick events, not begin
        // events (begin events drive by_stage rollup but NOT the
        // UI label — see the carve-out in the begin handler).
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "progress_tick",
                                   "ts": "2026-05-07T12:00:01Z",
                                   "stage": "extract", "total": 3}),
                serde_json::json!({"event": "begin", "call_id": "0001", "stage": "extract"}),
                serde_json::json!({"event": "end", "call_id": "0001", "success": true}),
                serde_json::json!({"event": "begin", "call_id": "0002", "stage": "extract"}),
                serde_json::json!({"event": "end", "call_id": "0002", "success": true}),
                serde_json::json!({"event": "begin", "call_id": "0003", "stage": "extract"}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.progress.completed, 2);
        assert_eq!(s.progress.retries, 0);
        assert_eq!(s.progress.in_flight_calls, 1);
        assert_eq!(s.progress.stage.as_deref(), Some("extract"));
    }

    #[test]
    fn derive_progress_completed_excludes_failed_retries() {
        // The retry case the user reported: a call times out, gets
        // retried, and the retry succeeds. Pre-fix, both ends were
        // counted → completed = 2. Post-fix, only the leaf (the
        // retry that succeeded) counts → completed = 1, retries = 1.
        // Without this distinction, a flaky run with 5 calls each
        // retried once would show "10 / N calls" instead of "5 / N".
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "begin", "call_id": "0001", "stage": "vision"}),
                serde_json::json!({"event": "end", "call_id": "0001", "success": false,
                                   "error": {"class": "openai.APITimeoutError"}}),
                serde_json::json!({"event": "begin", "call_id": "0002", "stage": "vision",
                                   "retry_of_call_id": "0001", "attempt": 2}),
                serde_json::json!({"event": "end", "call_id": "0002", "success": true}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        // Leaf-only count: 0001 is a parent (0002 retries it), only
        // 0002 is a leaf.
        assert_eq!(s.progress.completed, 1);
        assert_eq!(s.progress.retries, 1);
        assert_eq!(s.progress.in_flight_calls, 0);
    }

    #[test]
    fn derive_progress_completed_handles_halve_fanout_chain() {
        // Halving cascade: 0001 fails → fan out to 0002 (half-1)
        // and 0003 (half-2). Each half succeeds. Bar should show 2
        // (the leaves) — NOT 3 (which would conflate the parent's
        // failed end with the children's successes).
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "begin", "call_id": "0001", "stage": "extract"}),
                serde_json::json!({"event": "end", "call_id": "0001", "success": false}),
                serde_json::json!({"event": "begin", "call_id": "0002", "stage": "extract",
                                   "retry_of_call_id": "0001"}),
                serde_json::json!({"event": "end", "call_id": "0002", "success": true}),
                serde_json::json!({"event": "begin", "call_id": "0003", "stage": "extract",
                                   "retry_of_call_id": "0001"}),
                serde_json::json!({"event": "end", "call_id": "0003", "success": true}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.progress.completed, 2);  // leaves: 0002, 0003
        assert_eq!(s.progress.retries, 1);    // parent: 0001
    }

    #[test]
    fn derive_progress_failed_terminal_leaf_does_not_count_as_completed() {
        // The user-reported bug: a chain that exhausts retries
        // ends with a FAILED terminal leaf. Pre-fix, leaf-counting
        // counted it as 1 completed (the bar inflated). Post-fix,
        // failed leaves go into `progress.failed`, not `completed`.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                // Successful leaf — counts as completed=1.
                serde_json::json!({"event": "begin", "call_id": "0001", "stage": "extract"}),
                serde_json::json!({"event": "end", "call_id": "0001", "success": true}),
                // Chain that exhausts retries: 0002 (timeout) →
                // 0003 (timeout, retry of 0002) → 0004 (timeout,
                // retry of 0003) → no more retries. Terminal leaf
                // is 0004 (failed). Should NOT bump completed.
                serde_json::json!({"event": "begin", "call_id": "0002", "stage": "extract"}),
                serde_json::json!({"event": "end", "call_id": "0002", "success": false,
                                   "error": {"class": "openai.APITimeoutError"}}),
                serde_json::json!({"event": "begin", "call_id": "0003", "stage": "extract",
                                   "retry_of_call_id": "0002"}),
                serde_json::json!({"event": "end", "call_id": "0003", "success": false,
                                   "error": {"class": "openai.APITimeoutError"}}),
                serde_json::json!({"event": "begin", "call_id": "0004", "stage": "extract",
                                   "retry_of_call_id": "0003"}),
                serde_json::json!({"event": "end", "call_id": "0004", "success": false,
                                   "error": {"class": "openai.APITimeoutError"}}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.progress.completed, 1, "only 0001 succeeded");
        assert_eq!(s.progress.failed, 1, "0004 is the terminal failure");
        assert_eq!(s.progress.retries, 2, "0002, 0003 are retried-past parents");
    }

    #[test]
    fn derive_progress_leaf_count_for_sample_loop_succeeds_terminal_only() {
        // Patterns sample loop: each iteration retries the same logical
        // productive call with reduced facts. Iter 1 succeeds with cap-
        // hit → iter 2 (retry_of=iter1) succeeds clean. Both attempts
        // succeed but only iter 2 is the leaf — the productive work is
        // ONE pattern call, not two, regardless of how many iterations
        // ran. Total denominator stays at 1 (no _bump_stage_est on
        // sampling); leaf-count must mirror that or the bar lies.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "begin", "call_id": "S0",
                                   "stage": "patterns"}),
                serde_json::json!({"event": "end", "call_id": "S0",
                                   "success": true}),
                serde_json::json!({"event": "begin", "call_id": "S1",
                                   "stage": "patterns",
                                   "retry_of_call_id": "S0"}),
                serde_json::json!({"event": "end", "call_id": "S1",
                                   "success": true}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.progress.completed, 1, "only S1 is the chain leaf");
        assert_eq!(s.progress.retries, 1);
    }

    #[test]
    fn derive_progress_late_retry_demotes_earlier_leaf() {
        // Race: `end` for call A lands on disk, then later a retry B
        // fires (retry_of=A). At the time A's end was written, no one
        // had retried it yet — A LOOKED like a leaf. Final-pass leaf
        // classification (after the whole jsonl is read) demotes A to
        // a parent because B's begin retroactively links back. Without
        // the deferred classification we'd transiently overcount.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "begin", "call_id": "A",
                                   "stage": "extract"}),
                serde_json::json!({"event": "end", "call_id": "A",
                                   "success": false}),
                // ... time passes, other events ...
                serde_json::json!({"event": "begin", "call_id": "X",
                                   "stage": "extract"}),
                serde_json::json!({"event": "end", "call_id": "X",
                                   "success": true}),
                // ... more time passes, retry of A finally fires ...
                serde_json::json!({"event": "begin", "call_id": "B",
                                   "stage": "extract",
                                   "retry_of_call_id": "A"}),
                serde_json::json!({"event": "end", "call_id": "B",
                                   "success": true}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        // Leaves at end of run: X, B (A was retried by B → not a leaf).
        assert_eq!(s.progress.completed, 2);
        assert_eq!(s.progress.retries, 1);
    }

    #[test]
    fn derive_outcome_timeout_carries_kind_suffix() {
        // The user-reported color bug: outcome strings without a
        // (sizing)/(load) suffix don't match the CSS classes
        // (.outcome-timeout_sizing / .outcome-timeout_load), so the
        // UI renders timeout pills as default grey instead of
        // orange/red. Live derivation must produce the suffixed
        // form, matching Python's runner.py OUTCOME_TIMEOUT_*
        // constants.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                // Load timeout: TTFT null (provider never emitted).
                serde_json::json!({"event": "begin", "call_id": "0001", "stage": "extract"}),
                serde_json::json!({"event": "end", "call_id": "0001", "success": false,
                                   "ttft_ms": null,
                                   "error": {"class": "openai.APITimeoutError",
                                             "message": "timed out"}}),
                // Sizing timeout: TTFT set (mid-stream cut).
                serde_json::json!({"event": "begin", "call_id": "0002", "stage": "extract"}),
                serde_json::json!({"event": "end", "call_id": "0002", "success": false,
                                   "ttft_ms": 1234,
                                   "error": {"class": "openai.APITimeoutError",
                                             "message": "read timed out"}}),
            ],
        );
        let val = materialize_llm_stats_from_jsonl(&dir);
        let calls = val.get("calls").and_then(|v| v.as_array()).unwrap();
        assert_eq!(calls[0]["outcome"], "timeout (load)");
        assert_eq!(calls[1]["outcome"], "timeout (sizing)");
    }

    #[test]
    fn derive_progress_in_flight_handles_unmatched_begins() {
        // A SIGKILL leaves begins with no matching end — these are the
        // "in flight" slots the modal renders as cancelled.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "begin", "call_id": "0001", "stage": "extract"}),
                serde_json::json!({"event": "begin", "call_id": "0002", "stage": "extract"}),
                serde_json::json!({"event": "begin", "call_id": "0003", "stage": "extract"}),
                serde_json::json!({"event": "end", "call_id": "0001", "success": true}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.progress.completed, 1);
        assert_eq!(s.progress.in_flight_calls, 2);
    }

    #[test]
    fn derive_picks_up_entities_decision_subject_and_bundle() {
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({
                    "event": "entities_decision",
                    "ts": "2026-05-07T12:00:30Z",
                    "subject_resolution": {
                        "canonical_id": "alice",
                        "display": "Alice",
                        "source": "alias_match",
                    },
                    "bundle_mode": true,
                }),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.subject_resolution.unwrap()["canonical_id"], "alice");
        assert_eq!(s.bundle_mode, Some(true));
    }

    #[test]
    fn derive_resume_resets_terminator_on_new_cycle_start() {
        // Resume = new cycle_start AFTER a previous cycle_end. The new
        // cycle_start should re-set the run as "running" (the prior
        // cycle's end is no longer the active terminator).
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:01:00Z", "reason": "complete"}),
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:02:00Z", "is_resume": true}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.status, "running");
    }

    #[test]
    fn derive_picks_up_progress_tick_fields() {
        // The runner emits progress_tick events on every LLM-call
        // boundary. Derivation reads the LATEST and surfaces
        // bar_position / eta_seconds / total / stage to the runs-list,
        // so single-call stages (insights / actions) don't show 0%
        // while waiting for the call to return.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                // First tick — early in the pipeline.
                serde_json::json!({
                    "event": "progress_tick",
                    "ts": "2026-05-07T12:00:10Z",
                    "stage": "extract",
                    "completed": 5, "total": 50,
                    "bar_position": 0.10,
                    "eta_seconds": 90.0,
                    "stage_eta_seconds": 30.0,
                    "elapsed_in_stage": 10.0,
                }),
                // Later tick — derivation should pick THIS up, not the first.
                serde_json::json!({
                    "event": "progress_tick",
                    "ts": "2026-05-07T12:01:00Z",
                    "stage": "insights",
                    "completed": 49, "total": 50,
                    "bar_position": 0.85,
                    "eta_seconds": 12.0,
                    "stage_eta_seconds": 12.0,
                    "elapsed_in_stage": 0.0,
                }),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.progress.stage.as_deref(), Some("insights"));
        assert_eq!(s.progress.total, Some(50));
        assert!((s.progress.bar_position.unwrap_or(0.0) - 0.85).abs() < 1e-6);
        assert!((s.progress.eta_seconds.unwrap_or(0.0) - 12.0).abs() < 1e-6);
        assert!((s.progress.stage_eta_seconds.unwrap_or(0.0) - 12.0).abs() < 1e-6);
        assert_eq!(
            s.progress.bar_position_at.as_deref(),
            Some("2026-05-07T12:01:00Z"),
        );
    }

    #[test]
    fn derive_progress_tick_overrides_begin_event_stage() {
        // Begin events also carry the `stage` field; the latest
        // progress_tick wins (it's emitted at stage transitions
        // BETWEEN call begins on stages with no calls yet — e.g. the
        // moment insights starts but its single LLM call is still
        // pending its begin event).
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                }),
                serde_json::json!({"event": "end", "call_id": "0001", "success": true}),
                serde_json::json!({
                    "event": "progress_tick",
                    "ts": "2026-05-07T12:00:30Z",
                    "stage": "insights",
                    "bar_position": 0.85,
                }),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.progress.stage.as_deref(), Some("insights"));
        assert!((s.progress.bar_position.unwrap_or(0.0) - 0.85).abs() < 1e-6);
    }

    #[test]
    fn derive_cache_pause_resume_progresses_through_stages() {
        // Pre-warm the cache at the pause snapshot, then incrementally
        // append cycle 2 events (resume → entities → patterns →
        // insights). Each derive_run_state call must reflect the
        // freshest appended content — the cache must invalidate on
        // each jsonl mtime bump and the re-walk must override the
        // pre-pause stage.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z", "cycle_seq": 1, "pid": std::process::id()}),
                serde_json::json!({"event": "begin", "call_id": "0001", "stage": "entities"}),
                serde_json::json!({"event": "end", "call_id": "0001", "success": true}),
                serde_json::json!({"event": "progress_tick", "ts": "2026-05-07T12:00:10Z", "stage": "entities", "total": 24}),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:01:00Z", "reason": "sigterm"}),
            ],
        );
        // Simulate pause → paused.flag marker.
        let paused_flag = dir.join("paused.flag");
        std::fs::write(&paused_flag, "paused").unwrap();
        let paused_state = derive_run_state(&dir);
        assert_eq!(paused_state.status, "paused");
        assert_eq!(paused_state.progress.stage.as_deref(), Some("entities"));
        assert_eq!(paused_state.progress.total, Some(24));

        // Resume — drop paused.flag, sleep past fs mtime resolution,
        // then append cycle 2 events one stage at a time. derive_run_state
        // must observe the fresh stage on each call.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::remove_file(&paused_flag).unwrap();
        let resume_pretick = derive_run_state(&dir);
        assert_eq!(resume_pretick.status, "running",
                   "paused.flag removed + cycle_start_ts present → running");

        let jsonl = dir.join("llm-calls.jsonl");
        let append = |events: &[serde_json::Value]| {
            use std::io::Write;
            let body: String = events.iter()
                .map(|e| serde_json::to_string(e).unwrap() + "\n")
                .collect();
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&jsonl)
                .unwrap();
            f.write_all(body.as_bytes()).unwrap();
        };

        // Cycle 2 cycle_start.
        std::thread::sleep(std::time::Duration::from_millis(20));
        append(&[serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:02:00Z", "cycle_seq": 2, "is_resume": true, "pid": std::process::id()})]);
        let s = derive_run_state(&dir);
        assert_eq!(s.status, "running");
        // No cycle 2 progress_tick yet → latest_stage_from_begin still
        // resolves to cycle 1's last entities tick.
        assert_eq!(s.progress.stage.as_deref(), Some("entities"));

        // Cycle 2: entities tick.
        std::thread::sleep(std::time::Duration::from_millis(20));
        append(&[serde_json::json!({"event": "progress_tick", "ts": "2026-05-07T12:02:01Z", "stage": "entities", "total": 24})]);
        let s = derive_run_state(&dir);
        assert_eq!(s.progress.stage.as_deref(), Some("entities"));
        assert_eq!(s.progress.total, Some(24));

        // Cycle 2: patterns tick.
        std::thread::sleep(std::time::Duration::from_millis(20));
        append(&[serde_json::json!({"event": "progress_tick", "ts": "2026-05-07T12:02:10Z", "stage": "patterns", "total": 36})]);
        let s = derive_run_state(&dir);
        assert_eq!(s.progress.stage.as_deref(), Some("patterns"),
                   "cycle 2's patterns tick must override the cached entities snapshot");
        assert_eq!(s.progress.total, Some(36));

        // Cycle 2: insights tick.
        std::thread::sleep(std::time::Duration::from_millis(20));
        append(&[serde_json::json!({"event": "progress_tick", "ts": "2026-05-07T12:02:20Z", "stage": "insights", "total": 37, "in_flight_calls": 1})]);
        let s = derive_run_state(&dir);
        assert_eq!(s.progress.stage.as_deref(), Some("insights"),
                   "cycle 2's insights tick must advance the bar past the cached entities snapshot; got {:?}", s.progress.stage);
        assert_eq!(s.progress.total, Some(37));
    }

    #[test]
    fn derive_completed_dedupes_cross_cycle_redo_by_cache_key() {
        // When cycle 2 resumes mid-stage, the rerun re-emits begin/end
        // pairs for cycle 1's already-successful calls (rerun short-
        // circuits via the LLM cache — second end has the same
        // cache_key, cached=true). Both ends land in the jsonl but
        // they represent ONE pipeline work unit, not two. `completed`
        // must dedupe by cache_key; otherwise it inflates past `total`
        // every restart by the size of the prior cycle's successful
        // set, which clamps the bar at 'N/N' until the run ends.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess", "eval-1", "run-x",
            &[
                // Cycle 1: 2 successful calls with distinct cache_keys.
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z", "cycle_seq": 1, "pid": std::process::id()}),
                serde_json::json!({"event": "begin", "call_id": "0001", "stage": "extract"}),
                serde_json::json!({"event": "end",   "call_id": "0001", "success": true, "cache_key": "KEY-A"}),
                serde_json::json!({"event": "begin", "call_id": "0002", "stage": "extract"}),
                serde_json::json!({"event": "end",   "call_id": "0002", "success": true, "cache_key": "KEY-B"}),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:01:00Z", "reason": "sigterm"}),
                // Cycle 2: redoes the 2 calls (same cache_keys, cache
                // hits) + makes 1 fresh call (new cache_key).
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:02:00Z", "cycle_seq": 2, "is_resume": true, "pid": std::process::id()}),
                serde_json::json!({"event": "begin", "call_id": "0003", "stage": "extract"}),
                serde_json::json!({"event": "end",   "call_id": "0003", "success": true, "cache_key": "KEY-A", "cached": true}),
                serde_json::json!({"event": "begin", "call_id": "0004", "stage": "extract"}),
                serde_json::json!({"event": "end",   "call_id": "0004", "success": true, "cache_key": "KEY-B", "cached": true}),
                serde_json::json!({"event": "begin", "call_id": "0005", "stage": "entities"}),
                serde_json::json!({"event": "end",   "call_id": "0005", "success": true, "cache_key": "KEY-C"}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        // 4 successful ends total in the jsonl; 3 distinct cache_keys
        // (A, B, C) representing 3 unique work units. `completed` must
        // count 3, not 4.
        assert_eq!(s.progress.completed, 3,
                   "completed should dedupe cross-cycle redo by cache_key (3 distinct keys A/B/C); got {}",
                   s.progress.completed);
    }

    #[test]
    fn derive_completed_no_cache_key_counts_individually() {
        // Older runs (or future event shapes) may omit `cache_key`
        // from the end event. Treat each such leaf as unique so the
        // count stays correct on the back-compat path.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess", "eval-1", "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z", "cycle_seq": 1, "pid": std::process::id()}),
                serde_json::json!({"event": "begin", "call_id": "0001", "stage": "extract"}),
                serde_json::json!({"event": "end",   "call_id": "0001", "success": true}),
                serde_json::json!({"event": "begin", "call_id": "0002", "stage": "extract"}),
                serde_json::json!({"event": "end",   "call_id": "0002", "success": true}),
            ],
        );
        let s = derive_run_state_uncached(&dir);
        assert_eq!(s.progress.completed, 2,
                   "no-cache_key leaves count individually; got {}", s.progress.completed);
    }

    #[test]
    fn derive_pause_resume_picks_up_post_resume_stage() {
        // Pause-then-resume sequence: cycle 1 reaches entities and is
        // paused (SIGTERM cycle_end); cycle 2 (cycle_seq=2, is_resume)
        // replays entities then advances through patterns / insights.
        // The latest progress_tick across BOTH cycles must drive
        // `state.progress` — i.e. cycle 2's stage label / total must
        // override cycle 1's pre-pause snapshot.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                // Cycle 1 — entities, pause.
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z", "cycle_seq": 1, "pid": std::process::id()}),
                serde_json::json!({"event": "begin", "call_id": "0001", "stage": "entities"}),
                serde_json::json!({"event": "end", "call_id": "0001", "success": true}),
                serde_json::json!({"event": "progress_tick", "ts": "2026-05-07T12:00:10Z", "stage": "entities", "total": 24}),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:01:00Z", "reason": "sigterm"}),
                // Cycle 2 (resume) — entities → patterns → insights.
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:02:00Z", "cycle_seq": 2, "is_resume": true, "pid": std::process::id()}),
                serde_json::json!({"event": "progress_tick", "ts": "2026-05-07T12:02:01Z", "stage": "entities", "total": 24}),
                serde_json::json!({"event": "begin", "call_id": "0023", "stage": "entities"}),
                serde_json::json!({"event": "end", "call_id": "0023", "success": true}),
                serde_json::json!({"event": "progress_tick", "ts": "2026-05-07T12:02:10Z", "stage": "patterns", "total": 36}),
                serde_json::json!({"event": "begin", "call_id": "0029", "stage": "patterns"}),
                serde_json::json!({"event": "end", "call_id": "0029", "success": true}),
                serde_json::json!({"event": "progress_tick", "ts": "2026-05-07T12:02:20Z", "stage": "insights", "total": 37, "in_flight_calls": 1}),
            ],
        );
        let s = derive_run_state(&dir);
        assert_eq!(s.status, "running", "no terminator after cycle 2 resume → running");
        assert_eq!(s.progress.stage.as_deref(), Some("insights"),
                   "latest progress_tick stage must win across cycles; got {:?}", s.progress.stage);
        assert_eq!(s.progress.total, Some(37),
                   "latest progress_tick total must win across cycles; got {:?}", s.progress.total);
        // Cycle 1 + cycle 2 leaves: 0001 + 0023 + 0029 = 3 successes.
        assert_eq!(s.progress.completed, 3, "leaves across cycles");
    }

    #[test]
    fn derive_cache_concurrent_walkers_no_stale_poisoning() {
        // Under concurrent list_runs calls + a writer thread appending
        // progress_ticks, a slow walker can complete AFTER a fast
        // walker and overwrite the cache with an older (key, derived)
        // pair. Subsequent calls must still observe the freshest jsonl
        // content (i.e. mtime-key staleness must not persist beyond
        // one call).
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z", "cycle_seq": 1}),
                serde_json::json!({"event": "progress_tick", "ts": "2026-05-07T12:00:01Z", "stage": "extract", "total": 10}),
            ],
        );
        let jsonl_path = dir.join("llm-calls.jsonl");
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        // Writer thread: appends progress_ticks every few ms, advancing
        // stage entities→patterns→insights→embeddings + bumping total.
        let writer_jsonl = jsonl_path.clone();
        let writer_stop = stop.clone();
        let writer = std::thread::spawn(move || {
            let stages = ["entities", "patterns", "insights", "embeddings"];
            let mut total = 11u64;
            let mut idx = 0;
            while !writer_stop.load(std::sync::atomic::Ordering::Relaxed) {
                let stage = stages[idx % stages.len()];
                idx += 1;
                total += 1;
                let line = format!(
                    "{}\n",
                    serde_json::json!({
                        "event": "progress_tick",
                        "ts": format!("2026-05-07T12:00:{:02}Z", (idx % 60) + 2),
                        "stage": stage,
                        "total": total,
                    })
                );
                let mut f = std::fs::OpenOptions::new()
                    .append(true)
                    .open(&writer_jsonl)
                    .unwrap();
                use std::io::Write;
                f.write_all(line.as_bytes()).unwrap();
                drop(f);
                std::thread::sleep(std::time::Duration::from_micros(500));
            }
            total
        });
        // Reader threads: hammer derive_run_state for 200ms, looking
        // for ANY return whose `total` lags behind the writer thread's
        // most recent insertion at observation time.
        let mut readers = Vec::new();
        for _ in 0..4 {
            let r_dir = dir.clone();
            readers.push(std::thread::spawn(move || {
                let mut max_seen: u64 = 0;
                let deadline = std::time::Instant::now()
                    + std::time::Duration::from_millis(200);
                while std::time::Instant::now() < deadline {
                    let d = derive_run_state(&r_dir);
                    if let Some(t) = d.progress.total {
                        if t > max_seen { max_seen = t; }
                    }
                }
                max_seen
            }));
        }
        std::thread::sleep(std::time::Duration::from_millis(150));
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let final_total = writer.join().unwrap();
        let max_seen: u64 = readers.into_iter().map(|h| h.join().unwrap()).max().unwrap_or(0);
        // Final read after the writer stopped — should always see the
        // freshest content (no concurrent invalidator left).
        let after = derive_run_state(&dir);
        assert_eq!(
            after.progress.total, Some(final_total),
            "after writer stops, derive must see the freshest total (= {}); got {:?}. \
             Readers' max during concurrent run = {}",
            final_total, after.progress.total, max_seen,
        );
    }

    #[test]
    fn derive_cache_returns_cached_when_jsonl_unchanged() {
        // First call walks the file; second call hits the cache and
        // returns the SAME shape without re-walking. Force this by
        // walking, mutating one in-memory field, calling again, and
        // verifying the cache-served value still matches.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "begin", "call_id": "0001", "stage": "extract"}),
                serde_json::json!({"event": "end", "call_id": "0001", "success": true}),
            ],
        );
        let first = derive_run_state(&dir);
        assert_eq!(first.progress.completed, 1);
        // Append directly without flushing mtime — read again. Naive
        // re-walk would pick up the new event, cache must NOT (mtime
        // hasn't progressed within the same nanosecond).
        let second = derive_run_state(&dir);
        assert_eq!(second.progress.completed, first.progress.completed);
        assert_eq!(second.progress.in_flight_calls, first.progress.in_flight_calls);
    }

    #[test]
    fn derive_cache_invalidates_when_paused_flag_appears() {
        // paused.flag is a runtime control marker the runner doesn't
        // emit. Without it in the cache key, a `set_run_status
        // "paused"` would be invisible until the next jsonl event
        // landed (which on a paused-ergo-dead runner is never).
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"})],
        );
        // First call — running.
        assert_eq!(derive_run_state(&dir).status, "running");
        // Drop paused.flag without touching the jsonl. Without the
        // marker mtime in the cache key, the second call would still
        // return "running" from the stale cache entry.
        std::fs::write(dir.join("paused.flag"), "user paused").unwrap();
        assert_eq!(derive_run_state(&dir).status, "paused");
    }

    #[test]
    fn derive_cache_invalidates_when_paused_flag_removed() {
        // Inverse: removing paused.flag (resume_run does this before
        // the runner re-emits cycle_start) must drop the cached
        // "paused" so the very next list_runs sees "running".
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"})],
        );
        std::fs::write(dir.join("paused.flag"), "u").unwrap();
        assert_eq!(derive_run_state(&dir).status, "paused");
        // Resume path: drop the marker.
        std::fs::remove_file(dir.join("paused.flag")).unwrap();
        assert_eq!(derive_run_state(&dir).status, "running");
    }

    #[test]
    fn derive_cache_invalidates_when_jsonl_mtime_changes() {
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({"event": "begin", "call_id": "0001", "stage": "extract"}),
                serde_json::json!({"event": "end", "call_id": "0001", "success": true}),
            ],
        );
        let first = derive_run_state(&dir);
        assert_eq!(first.progress.completed, 1);

        // Sleep just past the FS mtime resolution so the rewrite below
        // bumps the mtime visibly. macOS HFS+/APFS = 1ns; ext4 = 1ns;
        // some FUSE / network mounts = 1s. 50ms covers all.
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Append a new completed call; rewrite the file so mtime changes.
        let new_events = vec![
            serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
            serde_json::json!({"event": "begin", "call_id": "0001", "stage": "extract"}),
            serde_json::json!({"event": "end", "call_id": "0001", "success": true}),
            serde_json::json!({"event": "begin", "call_id": "0002", "stage": "extract"}),
            serde_json::json!({"event": "end", "call_id": "0002", "success": true}),
        ];
        let body = new_events
            .iter()
            .map(|e| serde_json::to_string(e).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        std::fs::write(dir.join("llm-calls.jsonl"), body).unwrap();
        let second = derive_run_state(&dir);
        assert_eq!(second.progress.completed, 2, "mtime change must bust the cache");
    }

    #[test]
    fn derive_cache_invalidates_when_last_stage_marker_appears() {
        // Stage-5 marker is the only "completed" signal. The runner
        // materializes it deterministically when the actions stage
        // finishes; in some configurations the surrounding writes may
        // not bump the jsonl in the same nanosecond. Without the
        // marker mtime in the cache key, the cached "running" entry
        // would shadow the freshly-completed run on the next walk.
        clear_derive_cache_for_tests();
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[serde_json::json!({
                "event": "cycle_start",
                "ts": "2026-05-07T12:00:00Z",
                "pid": std::process::id(),
            })],
        );
        // First call — running (alive pid, no marker yet).
        assert_eq!(derive_run_state(&dir).status, "running");
        // Stage 5 finishes — marker materializes without touching the
        // jsonl. Second call must re-walk and flip to completed.
        write_last_stage_marker(&dir);
        assert_eq!(derive_run_state(&dir).status, "completed");
    }

    #[test]
    fn append_cycle_event_appends_one_line_with_event_field() {
        let tmp = tempfile::tempdir().unwrap();
        let jsonl = tmp.path().join("llm-calls.jsonl");
        std::fs::write(
            &jsonl,
            "{\"event\":\"cycle_start\",\"ts\":\"2026-05-07T12:00:00Z\"}\n",
        )
        .unwrap();
        let mut extras = serde_json::Map::new();
        extras.insert(
            "message".into(),
            serde_json::Value::String("user cancelled".into()),
        );
        append_cycle_event_to_jsonl(tmp.path(), "cycle_cancelled", extras);
        let body = std::fs::read_to_string(&jsonl).unwrap();
        let lines: Vec<&str> = body.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 2, "second event appended on its own line");
        let last: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(last["event"], "cycle_cancelled");
        assert_eq!(last["message"], "user cancelled");
        assert_eq!(last["schema"], "llm-calls/v1");
        // Every line in llm-calls.jsonl carries an ISO-Z `ts` (#353).
        // The Python emitter stamps `ts` in `_append_event_jsonl`;
        // this Rust emitter is a separate sibling that the runner's
        // cancel/error paths flow through. Pinning the invariant on
        // both sides catches drift if one is updated without the other.
        let ts = last["ts"].as_str().expect("ts present on every line");
        // ISO-Z shape: `YYYY-MM-DDTHH:MM:SSZ` — 20 chars, T at 10, Z at 19.
        assert_eq!(ts.len(), 20, "ts wrong length for ISO-Z: {ts:?}");
        assert!(ts.ends_with('Z'), "ts must end with Z: {ts:?}");
        assert_eq!(&ts[10..11], "T", "ts must have T at pos 10: {ts:?}");
        let (_, time_part) = ts[..ts.len() - 1].split_once('T').unwrap();
        assert_eq!(
            time_part.matches(':').count(),
            2,
            "ts time-portion must use colons (ISO-Z, not filename-safe dashes): {ts:?}",
        );
    }

    #[test]
    fn append_cycle_event_no_op_when_jsonl_missing() {
        // Legacy run dirs may have no jsonl at all (pre-cycle-marker
        // runs). Don't create one — the read fallback path uses
        // run.json's status field for those.
        let tmp = tempfile::tempdir().unwrap();
        append_cycle_event_to_jsonl(tmp.path(), "cycle_cancelled", Default::default());
        assert!(!tmp.path().join("llm-calls.jsonl").exists());
    }

    // ── In-flight materializer (read_run_llm_stats fallback) ────────────────

    #[test]
    fn materialize_inflight_returns_null_when_jsonl_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let v = materialize_llm_stats_from_jsonl(tmp.path());
        assert!(v.is_null());
    }

    #[test]
    fn materialize_inflight_builds_calls_and_per_stage() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "model": "kimi-k2-6", "started_at_iso": "2026-05-07T12:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001", "success": true,
                    "duration_ms": 1200, "prompt_tokens": 100,
                    "completion_tokens": 50, "ttft_ms": 250,
                    "reasoning_tokens": 10, "cached": false,
                }),
                serde_json::json!({
                    "event": "begin", "call_id": "0002", "stage": "extract",
                    "model": "kimi-k2-6", "started_at_iso": "2026-05-07T12:00:02.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0002", "success": true,
                    "duration_ms": 800, "prompt_tokens": 80,
                    "completion_tokens": 40, "ttft_ms": 200,
                    "cached": true, "cache_key": "abc",
                }),
                // In-flight third call — never gets an end event.
                serde_json::json!({
                    "event": "begin", "call_id": "0003", "stage": "patterns",
                    "model": "qwen3", "started_at_iso": "2026-05-07T12:00:03.000Z",
                }),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        assert!(!v.is_null());
        assert_eq!(v["in_flight"], serde_json::Value::Bool(true));
        let calls = v["calls"].as_array().expect("calls array");
        assert_eq!(calls.len(), 3);
        // Order preserved (insertion order from begin events).
        assert_eq!(calls[0]["call_id"], "0001");
        assert_eq!(calls[2]["call_id"], "0003");
        // Unmatched begin on an IN-FLIGHT run → pending, NOT aborted.
        // The discriminator in materialize_llm_stats_from_jsonl uses
        // run_terminated (cycle terminator on disk OR paused.flag) to
        // pick between pending (live) and aborted (run wound down).
        assert_eq!(calls[2]["pending"], true);
        assert_eq!(calls[2]["aborted"], false);
        assert!(calls[2]["success"].is_null(), "pending leaves success=null");
        assert!(calls[2]["error"].is_null(), "pending must NOT carry a synthetic error");
        assert_eq!(calls[2]["outcome"], "pending");
        // Cached call from end event preserved.
        assert_eq!(calls[1]["cached"], true);
        assert_eq!(calls[1]["cache_key"], "abc");
        // Per-stage rollup: extract has 2 calls (1 cached, both
        // success); patterns has 1 pending (NOT aborted, NOT failed).
        let extract = &v["per_stage"]["extract"];
        assert_eq!(extract["calls_total"], 2);
        assert_eq!(extract["calls_cached"], 1);
        assert_eq!(extract["calls_failed"], 0);
        assert_eq!(extract["calls_aborted"], 0);
        assert_eq!(extract["calls_pending"], 0);
        assert_eq!(extract["outcomes"]["success"], 2);
        // Median duration of [800, 1200] = 1000.
        assert_eq!(extract["duration_ms"]["median"], 1000);
        assert_eq!(extract["ttft_ms"]["median"], 225);
        assert_eq!(extract["reasoning_tokens"]["total"], 10);
        let patterns = &v["per_stage"]["patterns"];
        assert_eq!(patterns["calls_total"], 1);
        assert_eq!(patterns["calls_pending"], 1);
        assert_eq!(patterns["calls_aborted"], 0);
        assert_eq!(patterns["calls_failed"], 0);
        assert_eq!(patterns["outcomes"]["pending"], 1);
        // Top-level totals reflect pending too: successful excludes pending.
        let totals = &v["totals"];
        assert_eq!(totals["calls"], 3);
        assert_eq!(totals["pending"], 1);
        assert_eq!(totals["aborted"], 0);
        assert_eq!(totals["successful"], 2);
        // Models used.
        let models = extract["models_used"].as_array().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0], "kimi-k2-6");
    }

    #[test]
    fn materialize_inflight_unmatched_begin_on_terminated_run_is_aborted() {
        // Discriminator side B: when the jsonl carries a terminator
        // (cycle_end / cycle_cancelled / cycle_error) OR a paused.flag
        // is on disk, unmatched begins are INCOMPLETE — the call's
        // begin landed but no end ever did, and never will from this
        // jsonl. Pre-#206 the call was tagged "cancelled" with a
        // synthetic Cancelled error; that polluted outcome stats and
        // confused paused-and-resumed-completed runs whose cycle-1
        // in-flight calls reported as failed.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "model": "kimi-k2-6", "started_at_iso": "2026-05-07T12:00:01.000Z",
                }),
                // No end event for 0001 — call was in flight when the
                // user clicked Cancel.
                serde_json::json!({"event": "cycle_cancelled", "ts": "2026-05-07T12:00:05Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["aborted"], true);
        assert!(calls[0]["success"].is_null(), "aborted leaves success=null");
        assert!(calls[0]["error"].is_null(), "aborted must NOT synthesize an error — the call didn't fail, the run wound down");
        assert_eq!(calls[0]["outcome"], "aborted");
        // pending field absent (or false) — the discriminator picked
        // the aborted branch.
        let pending = calls[0].get("pending").and_then(|v| v.as_bool()).unwrap_or(false);
        assert!(!pending, "terminated runs do NOT mark unmatched begins pending");
    }

    #[test]
    fn materialize_skip_marker_overrides_error_and_aborted() {
        // Issue #333: per-call user-skip marker scan. The Tauri
        // `skip_call` command writes `skipped_calls/<call_id>` while
        // a call is in flight; the materializer reads the marker and
        // stamps `skipped=true`, nulls the error payload, and clears
        // the aborted flag so the outcome classifier produces
        // `skipped` (priority over pending and aborted).
        //
        // This test covers two configurations under one run:
        //   0001 — wrapper finalized the call with an error payload
        //          (the user clicked ✕ mid-stream, _SkippedByUser
        //          raised, the wrapper recorded the exception in the
        //          end event), then the marker wins.
        //   0002 — call has only a begin event (no end ever fired)
        //          AND a marker. The aborted default would normally
        //          take over here; the marker overrides.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-12T10:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "model": "kimi-k2-6", "started_at_iso": "2026-05-12T10:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "duration_ms": 1200, "success": false,
                    "error": {
                        "class": "retry._SkippedByUser",
                        "message": "skipped by user (call_id=0001)",
                    },
                    "model": "kimi-k2-6", "mode": "tinfoil",
                }),
                serde_json::json!({
                    "event": "begin", "call_id": "0002", "stage": "extract",
                    "model": "kimi-k2-6", "started_at_iso": "2026-05-12T10:00:02.000Z",
                }),
                // No end for 0002; would default to aborted.
                serde_json::json!({"event": "cycle_cancelled", "ts": "2026-05-12T10:00:05Z"}),
            ],
        );
        // Drop markers for both calls.
        let skip_dir = dir.join("skipped_calls");
        std::fs::create_dir_all(&skip_dir).unwrap();
        std::fs::write(skip_dir.join("0001"), "").unwrap();
        std::fs::write(skip_dir.join("0002"), "").unwrap();

        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        let by_id: std::collections::HashMap<&str, &serde_json::Value> = calls
            .iter()
            .map(|c| (c["call_id"].as_str().unwrap(), c))
            .collect();

        // 0001 — wrapper finalized, marker overrode.
        assert_eq!(by_id["0001"]["skipped"], true);
        assert_eq!(by_id["0001"]["aborted"], false);
        assert!(by_id["0001"]["error"].is_null(),
                "skip marker nulls error — user signal, not a failure");
        assert_eq!(by_id["0001"]["outcome"], "skipped");

        // 0002 — would have been aborted; marker wins.
        assert_eq!(by_id["0002"]["skipped"], true);
        assert_eq!(by_id["0002"]["aborted"], false);
        assert_eq!(by_id["0002"]["outcome"], "skipped");

        // totals.skipped surfaces alongside aborted / failed.
        assert_eq!(v["totals"]["skipped"], 2);
        assert_eq!(v["totals"]["aborted"], 0);

        // per_stage carries calls_skipped.
        let extract = v["per_stage"]["extract"].as_object().unwrap();
        assert_eq!(extract["calls_skipped"], 2);
        assert_eq!(extract["calls_aborted"], 0);
    }

    #[test]
    fn materialize_inflight_paused_flag_treats_run_as_terminated() {
        // paused.flag is the runtime control marker — when present the
        // runner is dead and unmatched begins are aborted (not
        // pending). The user can resume; the runner restarts from the
        // latest phase marker and redoes whatever work was in flight,
        // so calling these "failed" lies about the call's outcome.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "model": "kimi-k2-6", "started_at_iso": "2026-05-07T12:00:01.000Z",
                }),
            ],
        );
        std::fs::write(dir.join("paused.flag"), "user paused").unwrap();
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["aborted"], true);
        assert_eq!(calls[0]["outcome"], "aborted");
        let pending = calls[0].get("pending").and_then(|v| v.as_bool()).unwrap_or(false);
        assert!(!pending);
    }

    #[test]
    fn materialize_paused_then_resumed_running_run_splits_orphans_from_in_flight() {
        // Issue #206 follow-up (s9e3 mid-resume): a paused-and-resumed
        // run that is currently RUNNING in cycle 2 has TWO classes of
        // unmatched begins on disk:
        //   - Cycle-1 orphans: SIGTERM cut their stream; the work is
        //     being redone in cycle 2. → INCOMPLETE (superseded).
        //   - Cycle-2 in-flight: response stream still arriving.
        //     → PENDING.
        //
        // Discriminator: the begin's cycle_seq vs the latest cycle's
        // seq. Pre-fix iterations of this test got both wrong:
        //   v1 (pre-#206): cycle_end sigterm tagged everything
        //                  cancelled→failed.
        //   v2 (cycle_start resets ended_at_iso): cycle-1 orphans
        //                  flipped to pending alongside cycle-2.
        //   v3 (per-call cycle_seq): cycle-1 → aborted,
        //                  cycle-2 → pending. This is v3.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                // Cycle 1: pause-orphaned begin, cycle_end (sigterm).
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-09T23-09-26Z", "cycle_seq": 1}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "entities",
                    "model": "gpt-oss-120b", "started_at_iso": "2026-05-09T23:09:30.000Z",
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-09T23-09-57Z", "reason": "sigterm"}),
                // Cycle 2: resume, fresh in-flight begin, NO terminator
                // yet (run is currently running).
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-09T23-10-12Z", "cycle_seq": 2, "is_resume": true}),
                serde_json::json!({
                    "event": "begin", "call_id": "0002", "stage": "entities",
                    "model": "gpt-oss-120b", "started_at_iso": "2026-05-09T23:10:15.000Z",
                }),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        // Cycle-1 orphan: superseded by cycle 2's cycle_start →
        // aborted. Work was redone in cycle 2 (or is being
        // redone); this record is just an artifact.
        assert_eq!(calls[0]["call_id"], "0001");
        assert_eq!(calls[0]["aborted"], true);
        assert_eq!(calls[0]["outcome"], "aborted");
        assert_eq!(calls[0]["cycle_seq"], 1);
        // Cycle-2 in-flight: latest cycle, no terminator yet → pending.
        assert_eq!(calls[1]["call_id"], "0002");
        assert_eq!(calls[1]["pending"], true);
        assert_eq!(calls[1]["outcome"], "pending");
        assert_eq!(calls[1]["cycle_seq"], 2);
        // Top-level: run is in_flight, 1 pending, 1 aborted.
        assert_eq!(v["in_flight"], serde_json::Value::Bool(true));
        let totals = &v["totals"];
        assert_eq!(totals["pending"], 1);
        assert_eq!(totals["aborted"], 1);
        assert_eq!(totals["successful"], 0);
    }

    #[test]
    fn materialize_paused_then_resumed_completed_run_marks_orphan_call_aborted() {
        // Issue #206 follow-up: a paused-and-resumed-completed run had
        // its cycle-1 in-flight calls (begin landed, SIGTERM hit
        // before end) showing as "failed" in the per-call view of a
        // completed run. The work was redone in cycle 2 (LLM cache
        // accelerates the redo per the spec); these calls aren't
        // failures, they're just aborted records of superseded
        // work. Mirrors k955's on-disk shape (28 begins, 27 ends, 2
        // cycle_starts, 2 cycle_ends).
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                // Cycle 1: begin 0001, no end (paused mid-flight),
                // cycle_end (sigterm).
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-09T22-40-14Z", "cycle_seq": 1}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "model": "kimi-k2-6", "started_at_iso": "2026-05-09T22:40:14.891Z",
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-09T22-40-35Z", "reason": "sigterm"}),
                // Cycle 2: begin 0002, end 0002, cycle_end (atexit).
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-09T22-40-44Z", "cycle_seq": 2, "is_resume": true}),
                serde_json::json!({
                    "event": "begin", "call_id": "0002", "stage": "extract",
                    "model": "kimi-k2-6", "started_at_iso": "2026-05-09T22:40:44.730Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0002",
                    "duration_ms": 17000, "success": true,
                    "prompt_tokens": 100, "completion_tokens": 50, "ttft_ms": 200,
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-09T22-42-21Z", "reason": "atexit"}),
            ],
        );
        // Last-stage marker so the run derives as completed at the
        // run level (the user sees the run as Completed in the UI;
        // the call-level per-row status is what we're testing here).
        let p = dir.join("stages").join("06-embeddings").join("phase_1_marker.json");
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, "{}").unwrap();

        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        // Cycle-1 orphan call is INCOMPLETE — not failed, not
        // cancelled, not pending. The work was redone in cycle 2.
        assert_eq!(calls[0]["call_id"], "0001");
        assert_eq!(calls[0]["aborted"], true);
        assert_eq!(calls[0]["outcome"], "aborted");
        assert!(calls[0]["error"].is_null(), "aborted calls must not synthesize an error");
        // Cycle-2 redo completed cleanly.
        assert_eq!(calls[1]["call_id"], "0002");
        assert_eq!(calls[1]["outcome"], "success");
        // Top-level totals: 1 aborted, 1 successful, 0 failed.
        let totals = &v["totals"];
        assert_eq!(totals["calls"], 2);
        assert_eq!(totals["successful"], 1);
        assert_eq!(totals["aborted"], 1);
        assert_eq!(totals["failed"], 0);
        assert_eq!(totals["pending"], 0);
        // Per-stage tally matches.
        let extract = &v["per_stage"]["extract"];
        assert_eq!(extract["calls_total"], 2);
        assert_eq!(extract["calls_aborted"], 1);
        assert_eq!(extract["calls_failed"], 0);
        assert_eq!(extract["outcomes"]["aborted"], 1);
        assert_eq!(extract["outcomes"]["success"], 1);
    }

    // ── Live classifier MUST mirror Python's `_classify_outcome` ─────────
    // Issue #105 v3 follow-up: pre-fix the Rust live materializer and
    // Python rollup classified outcomes differently. Vocabulary now
    // follows the retry taxonomy: failure outcomes carry a
    // parenthetical class tag (`(load)` / `(sizing)` / `(other)`);
    // specific names (cap_hit / timeout / parse_error /
    // empty_response / interrupted) replace `failed` but keep the
    // tag. These tests pin the Python-Rust contract.

    #[test]
    fn live_classifier_buckets_apitimeouterror_as_timeout_load() {
        // No ttft on the rec → timeout classifies as load (the
        // TTFT discriminator: tokens never flowed).
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "model": "kimi-k2-6", "started_at_iso": "2026-05-07T12:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": false, "duration_ms": 1800,
                    "error": {"class": "openai.APITimeoutError", "message": "timeout"},
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:00:30Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["outcome"], "timeout (load)",
            "APITimeoutError with no ttft must classify as 'timeout (load)', not 'failed (X)'");
    }

    #[test]
    fn live_classifier_kernel_load_status_buckets_as_failed_load() {
        // #963: a no-exception kernel failure (`from_status`: success=false,
        // error=None) carries the kernel's categorized `llm_status` on the
        // end event. The live classifier must read it as authoritative —
        // LOAD -> `failed (load)`. Pre-fix the classifier never read
        // `llm_status` and, with no error class to match, fell through to
        // the `other` catch-all, so an injected/real LOAD mislabeled as
        // `failed (other)` (the run-details acceptance-shot bug).
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-06-29T00:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "model": "gpt-oss-120b", "started_at_iso": "2026-06-29T00:00:01.000Z",
                    "attempt": 1,
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": false, "duration_ms": 7000,
                    // No `error` object (the from_status path raised nothing)
                    // and no error class to heuristically bucket on.
                    "llm_status": "LOAD",
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-06-29T00:00:30Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["outcome"], "failed (load)",
            "no-exception LOAD must read llm_status and bucket as 'failed (load)', not 'failed (other)'");
    }

    #[test]
    fn live_classifier_no_llm_status_failure_stays_other() {
        // Regression guard for the fallback: a failure with neither an
        // `llm_status` nor an error class (e.g. a pre-kernel archived run)
        // still buckets to the `other` catch-all — the heuristic chain is
        // unchanged when `llm_status` is absent.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-06-29T00:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "model": "m", "started_at_iso": "2026-06-29T00:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": false, "duration_ms": 500,
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-06-29T00:00:30Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["outcome"], "failed (other)");
    }

    #[test]
    fn live_classifier_cap_hit_on_finish_reason_length() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "actions",
                    "model": "m", "started_at_iso": "2026-05-07T12:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": true, "duration_ms": 800,
                    "finish_reason": "length",
                    "prompt_tokens": 1000, "completion_tokens": 200,
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:00:05Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["outcome"], "cap_hit (sizing)");
    }

    #[test]
    fn live_classifier_chain_aware_sampled_rebucket() {
        // The load-bearing chain-aware re-bucket: a successful leaf
        // whose chain has any /sample-N ancestor must classify as
        // "sampled" (not "success"). Pre-fix the live view showed
        // sample retries as green "ok"; post-rollup they showed
        // yellow "sampled". User: "I want consistency."
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                // Parent (work-reducing trigger).
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "patterns",
                    "category": "work", "model": "m",
                    "started_at_iso": "2026-05-07T12:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": false, "duration_ms": 1000,
                    "error": {"class": "openai.APITimeoutError", "message": "t"},
                }),
                // Sample-1 retry succeeded with reduced facts.
                // Category mirrors the production format from
                // `category_for_retry` for the sizing sample cascade
                // (patterns / insights / actions / entities_dedupe):
                // `<stage>/sample-N - retry/sizing`.
                serde_json::json!({
                    "event": "begin", "call_id": "0002", "stage": "patterns",
                    "category": "patterns/sample-1 - retry/sizing", "model": "m",
                    "retry_of_call_id": "0001",
                    "started_at_iso": "2026-05-07T12:00:02.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0002",
                    "success": true, "duration_ms": 800,
                    "prompt_tokens": 500, "completion_tokens": 100,
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:00:10Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        let leaf = calls.iter()
            .find(|c| c["call_id"] == "0002").unwrap();
        assert_eq!(leaf["outcome"], "success_sampled",
            "leaf with /sample- in chain must re-bucket success → success_sampled");
        // Parent unchanged — chain re-bucket only applies to leaves.
        let parent = calls.iter()
            .find(|c| c["call_id"] == "0001").unwrap();
        assert_eq!(parent["outcome"], "timeout (load)");
    }

    #[test]
    fn live_classifier_halve_chain_does_NOT_become_sampled() {
        // Halve children have categories like "split_00/half-1" —
        // NOT "/sample-". Don't accidentally re-bucket them as
        // "sampled". Defensive against a sloppy substring match.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "category": "split_00", "model": "m",
                    "started_at_iso": "2026-05-07T12:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": false, "duration_ms": 1000,
                    "error": {"class": "openai.APITimeoutError", "message": "t"},
                }),
                serde_json::json!({
                    "event": "begin", "call_id": "0002", "stage": "extract",
                    "category": "split_00/half-1", "model": "m",
                    "retry_of_call_id": "0001",
                    "started_at_iso": "2026-05-07T12:00:02.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0002",
                    "success": true, "duration_ms": 500,
                    "prompt_tokens": 500, "completion_tokens": 100,
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:00:10Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        let leaf = calls.iter()
            .find(|c| c["call_id"] == "0002").unwrap();
        // half-1 ≠ sample-N → stays plain success.
        assert_eq!(leaf["outcome"], "success",
            "halve child must stay 'success', not re-bucket to 'sampled'");
    }

    #[test]
    fn live_warnings_block_counts_LEAVES_ONLY() {
        // The run-row banner reads the inflight payload's `warnings`
        // block. Counts MUST be leaf-only — a chain that timed out
        // then succeeded on retry shouldn't surface the intermediate
        // timeout in the banner. Pre-fix the live counter inflated
        // because every attempt contributed (msgj run had 15
        // intermediate timeouts but 0 leaf timeouts; banner showed
        // 17 flagged calls).
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                // Chain: timeout (parent) → success (leaf).
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "category": "split_00", "model": "m",
                    "started_at_iso": "2026-05-07T12:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": false, "duration_ms": 1000,
                    "error": {"class": "openai.APITimeoutError", "message": "t"},
                }),
                serde_json::json!({
                    "event": "begin", "call_id": "0002", "stage": "extract",
                    "category": "split_00", "model": "m",
                    "retry_of_call_id": "0001",
                    "started_at_iso": "2026-05-07T12:00:02.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0002",
                    "success": true, "duration_ms": 500,
                    "prompt_tokens": 500, "completion_tokens": 100,
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:00:10Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let warnings = &v["warnings"];
        // Leaf 0002 is success → zero leaf timeouts.
        assert_eq!(warnings["timeouts"], 0,
            "intermediate timeout must NOT inflate leaf-only count");
    }

    #[test]
    fn live_warnings_block_carries_python_rollup_field_shape() {
        // Issue #189: the Rust live materializer's `warnings` block
        // must mirror Python's leaf-aware rollup field shape so the
        // run-row banner sums consistently across in-flight + post-run
        // states. Vocabulary follows the post-#190 + #225 + #232 spec.
        // input_overflows stays 0 (in-memory-only `_call_warnings`
        // counter in Python; can't be derived from jsonl alone).
        // Issues #225 / #232: `empty_responses` and `interrupted` ARE
        // derivable now (counts events carry the flag), so they
        // surface non-zero from the leaf-bucket walk above —
        // dropped the pre-#225 0-only contract.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-w",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-08T12:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "category": "doc-1", "model": "m",
                    "started_at_iso": "2026-05-08T12:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": true, "duration_ms": 100,
                    "finish_reason": "length",
                    "prompt_tokens": 100, "completion_tokens": 4096,
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-08T12:00:02Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let w = v["warnings"].as_object().expect("warnings is an object");
        // All required keys present. cap_hit / timeout already cover
        // those failure shapes via their sizing / load suffix.
        for key in [
            "cap_hits", "empty_responses", "interrupted",
            "input_overflows", "timeouts", "success_empty", "parse_errors",
            "sampled", "reasoning_off", "failed",
        ] {
            assert!(w.contains_key(key),
                "warnings missing required key {}: {:?}", key, w);
        }
        // The cap_hit leaf is counted under cap_hits (plural).
        assert_eq!(w["cap_hits"], 1,
            "leaf finish_reason=length must surface as cap_hits=1");
        // empty_responses + interrupted both 0 in this fixture —
        // only the cap_hit leaf is present. Pin this so a regression
        // that double-counts surfaces immediately.
        assert_eq!(w["empty_responses"], 0);
        assert_eq!(w["interrupted"], 0);
        // input_overflows stays 0 — in-memory-only counter, not in jsonl.
        assert_eq!(w["input_overflows"], 0);
    }

    // ── Post-stream-failure buckets ──────────────────────────────────
    // The Rust outcome classifier mirrors Python's `_classify_outcome`;
    // these tests pin the parse_error / empty_response / interrupted
    // buckets.

    #[test]
    fn live_classifier_empty_response_from_counts_event() {
        // Wire-empty extract call: complete() raises
        // `_EmptyResponse`, wrapper finalizes failure and emits a
        // counts event with empty_response=true. The materializer
        // reads the flag from counts and the classifier routes to
        // OUTCOME_EMPTY_RESPONSE.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-empty",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-10T04:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "category": "doc-1", "model": "kimi-k2-6",
                    "started_at_iso": "2026-05-10T04:00:01.000Z",
                    "prompt_tokens_est": 14000,
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": false, "duration_ms": 1412000,
                    "error": {"class": "retry._EmptyResponse",
                              "message": "empty_response (stage=extract)"},
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "finish_reason": serde_json::Value::Null,
                }),
                serde_json::json!({
                    "event": "counts", "call_id": "0001",
                    "input": serde_json::Value::Null,
                    "output": serde_json::Value::Null,
                    "empty_response": true,
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-10T04:24:06Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["outcome"], "empty_response (load)");
        // The wire-cut response's 0 must NOT clobber the begin-time
        // prompt-token estimate.
        assert_eq!(calls[0]["prompt_tokens"], 14000);
    }

    #[test]
    fn live_classifier_interrupted_from_counts_event() {
        // Partial-stream cut: bytes flowed, no terminating
        // finish_reason chunk; complete() raises
        // `_InterruptedResponse`. Counts event carries the flag;
        // interrupted routes to Load unconditionally (application-
        // level "stream cut with tokens" — same spec category as
        // transport-level RemoteProtocolError).
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-intr",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-10T04:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "category": "doc-1", "model": "kimi-k2-6",
                    "started_at_iso": "2026-05-10T04:00:01.000Z",
                    "prompt_tokens_est": 6700,
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": false, "duration_ms": 30000,
                    "error": {"class": "retry._InterruptedResponse",
                              "message": "interrupted (stage=extract)"},
                    "prompt_tokens": 0,
                    "completion_tokens": 42,  // streamed-text estimate
                    "finish_reason": serde_json::Value::Null,
                    "ttft_ms": 5000, "last_token_ms": 28000,
                }),
                serde_json::json!({
                    "event": "counts", "call_id": "0001",
                    "input": serde_json::Value::Null,
                    "output": serde_json::Value::Null,
                    "interrupted": true,
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-10T04:00:31Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["outcome"], "interrupted (load)");
        // Estimated completion_tokens from streamed text passes
        // through the materializer's truthy-overwrite check.
        assert_eq!(calls[0]["completion_tokens"], 42);
    }

    #[test]
    fn legacy_umbrella_class_name_still_classifies() {
        // Runs archived before the umbrella exception was split
        // carry the single legacy class name plus the kind in the
        // rec flag. The compat shim must still route them.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-legacy",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-10T04:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "category": "doc-1", "model": "kimi-k2-6",
                    "started_at_iso": "2026-05-10T04:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": false, "duration_ms": 1000,
                    "error": {"class": "retry._ErraticResponse",
                              "message": "erratic response: parse_error (stage=extract)"},
                }),
                serde_json::json!({
                    "event": "counts", "call_id": "0001",
                    "input": serde_json::Value::Null,
                    "output": serde_json::Value::Null,
                    "parse_error": true,
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-10T04:00:10Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["outcome"], "parse_error (sizing)");
    }

    #[test]
    fn live_classifier_kept_zero_with_reasoning_stays_success_empty() {
        // The post-stream-failure labels (empty_response /
        // interrupted / parse_error) are reserved for zero-content-
        // token failure shapes. Model emitted a clean valid empty
        // payload after burning reasoning tokens — that's a
        // legitimate "no findings applied" answer, not a failure;
        // the empty_response bucket must not false-positive here.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-eras",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-10T04:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "patterns",
                    "category": "topic", "model": "kimi-k2-6",
                    "started_at_iso": "2026-05-10T04:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": true, "duration_ms": 14000,
                    "prompt_tokens": 400, "completion_tokens": 21000,
                    "reasoning_tokens": 20995,
                    "finish_reason": "stop", "ttft_ms": 100,
                }),
                serde_json::json!({
                    "event": "counts", "call_id": "0001",
                    "input": serde_json::Value::Null,
                    "output": {"patterns": 0},
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-10T04:00:14Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["outcome"], "success_empty");
    }

    #[test]
    fn live_classifier_kept_zero_no_reasoning_stays_success_empty() {
        // Discriminator regression guard: explicit reasoning_tokens=0
        // stays as success_empty (the legitimate "no findings"
        // shape). Mirrors the matching Python test.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-empt",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-10T04:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "category": "doc-1", "model": "kimi-k2-6",
                    "started_at_iso": "2026-05-10T04:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": true, "duration_ms": 1000,
                    "prompt_tokens": 200, "completion_tokens": 5,
                    "reasoning_tokens": 0,
                    "finish_reason": "stop", "ttft_ms": 100,
                }),
                serde_json::json!({
                    "event": "counts", "call_id": "0001",
                    "input": serde_json::Value::Null,
                    "output": {"facts": 0},
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-10T04:00:02Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["outcome"], "success_empty");
    }

    #[test]
    fn live_classifier_insights_empty_with_caps_is_success_empty() {
        // The insights stage records its primary entry counts FIRST
        // in `output`, then auxiliary cap fields. A clean empty result
        // (0 entries across the three buckets) must classify as
        // success_empty even though the cap ints are non-zero. Pre-fix
        // the live materializer summed every int and surfaced this as
        // plain `success`. Anchored on the live m7pp insights call
        // shape (0 entries, caps 10/6/4). Mirrors the Python test.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-ins",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-10T04:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "insights",
                    "category": "insights", "model": "kimi-k2-6",
                    "started_at_iso": "2026-05-10T04:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": true, "duration_ms": 45000,
                    "prompt_tokens": 14523, "completion_tokens": 2422,
                    "reasoning_tokens": 2408,
                    "finish_reason": "stop", "ttft_ms": 44858,
                }),
                serde_json::json!({
                    "event": "counts", "call_id": "0001",
                    "input": {"patterns": 150, "topics_with_patterns": 9},
                    "output": {
                        "insights": 0, "cross_domain": 0, "critical": 0,
                        "kinds": {}, "total_cap": 10, "cross_cap": 6,
                        "critical_cap": 4,
                    },
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-10T04:00:46Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["outcome"], "success_empty");
    }

    #[test]
    fn live_classifier_actions_empty_with_cap_is_success_empty() {
        // Same convention for actions: primary count first, cap field
        // afterward. Clean empty result must classify as success_empty.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-act",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-10T04:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "actions",
                    "category": "actions", "model": "kimi-k2-6",
                    "started_at_iso": "2026-05-10T04:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": true, "duration_ms": 8000,
                    "prompt_tokens": 3000, "completion_tokens": 50,
                    "reasoning_tokens": 0,
                    "finish_reason": "stop", "ttft_ms": 100,
                }),
                serde_json::json!({
                    "event": "counts", "call_id": "0001",
                    "input": {"insights": 7},
                    "output": {
                        "actions": 0, "kinds": {},
                        "max_actions_cap": 8, "recommendation_lengths": {},
                    },
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-10T04:00:09Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["outcome"], "success_empty");
    }

    #[test]
    fn live_classifier_insights_nonzero_with_caps_stays_success() {
        // Counter-check: a non-zero primary count classifies as clean
        // success even with cap fields beside it. Guards against an
        // over-eager fix that ignores the primary count too.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-insn",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-10T04:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "insights",
                    "category": "insights", "model": "kimi-k2-6",
                    "started_at_iso": "2026-05-10T04:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": true, "duration_ms": 45000,
                    "prompt_tokens": 14523, "completion_tokens": 2422,
                    "reasoning_tokens": 2408,
                    "finish_reason": "stop", "ttft_ms": 44858,
                }),
                serde_json::json!({
                    "event": "counts", "call_id": "0001",
                    "input": {"patterns": 150, "topics_with_patterns": 9},
                    "output": {
                        "insights": 7, "cross_domain": 4, "critical": 3,
                        "kinds": {"opportunity": 4, "risk": 3},
                        "total_cap": 50, "cross_cap": 30, "critical_cap": 20,
                    },
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-10T04:00:46Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["outcome"], "success");
    }

    #[test]
    fn live_classifier_timeout_with_ttft_set_routes_to_sizing() {
        // TTFT discriminator (matches Python's
        // `_failure_class_for_label`): a timeout AFTER tokens started
        // flowing classifies as sizing (mid-stream cut), not load
        // (provider never emitted).
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-tw",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-10T04:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "category": "doc-1", "model": "kimi-k2-6",
                    "started_at_iso": "2026-05-10T04:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": false, "duration_ms": 1800000,
                    "error": {"class": "openai.APITimeoutError", "message": "timeout"},
                    // ttft_ms set ⇒ sizing (vs load when null —
                    // covered by live_classifier_buckets_apitimeouterror_as_timeout_load
                    // up above).
                    "ttft_ms": 5000,
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-10T04:30:01Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["outcome"], "timeout (sizing)");
    }

    #[test]
    fn live_materializer_picks_up_stream_progress() {
        // Issue #237: a stream_progress event between begin and end
        // updates the rec's completion_tokens / reasoning_tokens /
        // last_token_ms so the in-flight pane shows moving counts.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-prog",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-10T11:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "category": "doc-1", "model": "kimi-k2-6",
                    "started_at_iso": "2026-05-10T11:00:01.000Z",
                    "prompt_tokens_est": 14000,
                }),
                serde_json::json!({
                    "event": "stream_progress", "call_id": "0001",
                    "completion_tokens": 1200, "reasoning_tokens": 800,
                    "ttft_ms": 5000, "ttfr_ms": 1500, "last_token_ms": 28000,
                }),
                // No end event — call still in-flight.
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["completion_tokens"], 1200);
        assert_eq!(calls[0]["reasoning_tokens"], 800);
        assert_eq!(calls[0]["last_token_ms"], 28000);
        // prompt_tokens_est still surfaces from begin (#225 Bug 2).
        assert_eq!(calls[0]["prompt_tokens"], 14000);
    }

    #[test]
    fn live_tokens_heartbeat_updates_in_memory_state() {
        // Streaming LLM call writes a `live_tokens` JSON line to stdout
        // every ~1s. spawn_pipeline's stdout reader feeds each line to
        // apply_stdout_event_to_live_state, which mutates the run's
        // per-call entry without any disk I/O.
        let run_id = "run-live-update";
        apply_stdout_event_to_live_state(
            run_id,
            r#"{"event":"live_tokens","call_id":"0001","completion_tokens":42,"reasoning_tokens":10,"content_tokens":32,"last_token_ms":4500}"#,
        );
        let map = live_token_state().lock().unwrap();
        let entry = map.get(run_id).unwrap().get("0001").unwrap();
        assert_eq!(entry.completion_tokens, Some(42));
        assert_eq!(entry.reasoning_tokens, Some(10));
        assert_eq!(entry.content_tokens, Some(32));
        assert_eq!(entry.last_token_ms, Some(4500));
    }

    #[test]
    fn live_tokens_heartbeat_overwrites_prior_state() {
        // Successive heartbeats for the same call_id update the same
        // entry in place — the LATEST is what the overlay surfaces.
        // chars/3 is monotonically non-decreasing within one stream so
        // later values dominate. Asserting on all four fields keeps
        // the contract honest if any one of them ever drops out of
        // the apply path.
        let run_id = "run-live-overwrite";
        apply_stdout_event_to_live_state(
            run_id,
            r#"{"event":"live_tokens","call_id":"0001","completion_tokens":10,"reasoning_tokens":2,"content_tokens":8,"last_token_ms":1000}"#,
        );
        apply_stdout_event_to_live_state(
            run_id,
            r#"{"event":"live_tokens","call_id":"0001","completion_tokens":99,"reasoning_tokens":15,"content_tokens":80,"last_token_ms":9000}"#,
        );
        let map = live_token_state().lock().unwrap();
        let entry = map.get(run_id).unwrap().get("0001").unwrap();
        assert_eq!(entry.completion_tokens, Some(99));
        assert_eq!(entry.reasoning_tokens, Some(15));
        assert_eq!(entry.content_tokens, Some(80));
        assert_eq!(entry.last_token_ms, Some(9000));
    }

    #[test]
    fn apply_stdout_event_ignores_non_live_tokens_events() {
        // The stdout reader sees every line — non-JSON garbage, the
        // `_emit()` progress_tick / stage tags (which carry no
        // call_id), and event-shaped lines that aren't `live_tokens`.
        // Only `live_tokens` should mutate the live state. Notably
        // `end` / `counts` events are jsonl-only and would never
        // reach this function from real runner output, but we
        // exercise an end-shaped line here as a regression guard:
        // adding an `end` branch later would silently drop work
        // unless the test forces an explicit choice.
        let run_id = "run-live-noop";
        apply_stdout_event_to_live_state(run_id, "not-json-at-all");
        apply_stdout_event_to_live_state(
            run_id,
            r#"{"stage":"extract","total":32}"#,
        );
        apply_stdout_event_to_live_state(
            run_id,
            r#"{"event":"begin","call_id":"0001","stage":"extract"}"#,
        );
        apply_stdout_event_to_live_state(
            run_id,
            r#"{"event":"end","call_id":"0001","success":true,"duration_ms":3000}"#,
        );
        let map = live_token_state().lock().unwrap();
        assert!(map.get(run_id).is_none_or(|r| r.is_empty()));
    }

    #[test]
    fn clear_live_state_for_run_drops_whole_entry() {
        // spawn_pipeline calls clear_live_state_for_run when the
        // runner exits — anything left in the map at that point is
        // stale by definition (no longer being updated).
        let run_id = "run-live-clear";
        apply_stdout_event_to_live_state(
            run_id,
            r#"{"event":"live_tokens","call_id":"0001","completion_tokens":42}"#,
        );
        apply_stdout_event_to_live_state(
            "other-run",
            r#"{"event":"live_tokens","call_id":"0001","completion_tokens":99}"#,
        );
        clear_live_state_for_run(run_id);
        let map = live_token_state().lock().unwrap();
        assert!(!map.contains_key(run_id));
        // Other runs untouched.
        assert!(map.contains_key("other-run"));
    }

    #[test]
    fn overlay_live_state_writes_inflight_records() {
        // The materializer leaves in-flight records with success=null
        // and null token-count fields. overlay_live_token_state
        // fills those in from the in-memory map.
        let run_id = "run-live-overlay";
        apply_stdout_event_to_live_state(
            run_id,
            r#"{"event":"live_tokens","call_id":"0001","completion_tokens":1200,"reasoning_tokens":800,"content_tokens":400,"last_token_ms":28000}"#,
        );
        let mut materialized = serde_json::json!({
            "calls": [
                {
                    "call_id": "0001",
                    "stage": "extract",
                    "success": serde_json::Value::Null,
                    "completion_tokens": serde_json::Value::Null,
                    "reasoning_tokens": serde_json::Value::Null,
                    "content_tokens": serde_json::Value::Null,
                    "last_token_ms": serde_json::Value::Null,
                },
            ],
        });
        overlay_live_token_state(&mut materialized, run_id);
        let c = &materialized["calls"][0];
        assert_eq!(c["completion_tokens"], 1200);
        assert_eq!(c["reasoning_tokens"], 800);
        assert_eq!(c["content_tokens"], 400);
        assert_eq!(c["last_token_ms"], 28000);
    }

    #[test]
    fn overlay_live_state_skips_terminated_records() {
        // Once success is set (true or false) the end event has
        // landed and its counts are ground truth. The overlay must
        // not clobber those even if the live state still has an
        // entry for the call_id (eviction races, cross-session
        // restart with stale map, etc.).
        let run_id = "run-live-skip-terminal";
        apply_stdout_event_to_live_state(
            run_id,
            r#"{"event":"live_tokens","call_id":"0001","completion_tokens":1200}"#,
        );
        let mut materialized = serde_json::json!({
            "calls": [
                {
                    "call_id": "0001",
                    "stage": "extract",
                    "success": true,
                    "completion_tokens": 1180,  // end-event ground truth
                },
            ],
        });
        overlay_live_token_state(&mut materialized, run_id);
        assert_eq!(materialized["calls"][0]["completion_tokens"], 1180);
    }

    #[test]
    fn overlay_live_state_writes_duration_ms_for_inflight_rec() {
        // duration_ms = now - started_at_iso for any in-flight rec
        // with a parseable timestamp. Surfaces the per-call elapsed
        // in the modal table without Python needing to compute or
        // emit it. Runs even when there's no live-token state for
        // the run (TTFT phase before first chunk).
        // started_at_iso a known interval in the past via a date
        // far enough back that any "now" the test sees is past it.
        let mut materialized = serde_json::json!({
            "calls": [
                {
                    "call_id": "0001",
                    "stage": "extract",
                    "success": serde_json::Value::Null,
                    "started_at_iso": "2020-01-01T00:00:00.000Z",
                    "duration_ms": serde_json::Value::Null,
                },
            ],
        });
        overlay_live_token_state(&mut materialized, "run-live-duration");
        let d = materialized["calls"][0]["duration_ms"]
            .as_u64()
            .expect("duration_ms must be set as a positive number");
        // 2020-01-01 → any clock reading after that point. Generous
        // sanity floor (years' worth of ms) verifies the formula
        // without pinning a specific now.
        assert!(d > 100_000_000, "duration_ms should be huge for 2020 start; got {d}");
    }

    #[test]
    fn overlay_live_state_duration_ms_skips_terminated_records() {
        // Once success is set the materializer's duration_ms is the
        // canonical wrapper-measured value. The overlay must not
        // re-compute against the wall clock and overwrite it.
        let mut materialized = serde_json::json!({
            "calls": [
                {
                    "call_id": "0001",
                    "success": true,
                    "started_at_iso": "2020-01-01T00:00:00.000Z",
                    "duration_ms": 5432,
                },
            ],
        });
        overlay_live_token_state(&mut materialized, "run-terminated-dur");
        assert_eq!(materialized["calls"][0]["duration_ms"], 5432);
    }

    #[test]
    fn overlay_live_state_duration_ms_handles_missing_started_at() {
        // Defensive: if for some reason the begin event lacked
        // started_at_iso (legacy log, malformed) the overlay
        // skips duration_ms without panicking. The token overlay
        // still applies if its state is present.
        apply_stdout_event_to_live_state(
            "run-no-start",
            r#"{"event":"live_tokens","call_id":"0001","completion_tokens":42}"#,
        );
        let mut materialized = serde_json::json!({
            "calls": [
                {
                    "call_id": "0001",
                    "success": serde_json::Value::Null,
                    "completion_tokens": serde_json::Value::Null,
                    "duration_ms": serde_json::Value::Null,
                },
            ],
        });
        overlay_live_token_state(&mut materialized, "run-no-start");
        // Token overlay still applies.
        assert_eq!(materialized["calls"][0]["completion_tokens"], 42);
        // duration_ms stays null — no start timestamp to compute from.
        assert!(materialized["calls"][0]["duration_ms"].is_null());
    }

    #[test]
    fn overlay_live_state_noop_when_run_absent() {
        // Cross-session reads (Tauri restart, completed runs) and
        // sweep-mode runs (agent != "app", filtered out of list_runs)
        // have nothing in the live map. The overlay must return the
        // materialized payload untouched.
        let mut materialized = serde_json::json!({
            "calls": [
                {
                    "call_id": "0001",
                    "success": serde_json::Value::Null,
                    "completion_tokens": serde_json::Value::Null,
                },
            ],
        });
        overlay_live_token_state(&mut materialized, "no-such-run");
        // Nothing inserted; the field stays null.
        assert!(materialized["calls"][0]["completion_tokens"].is_null());
    }

    #[test]
    fn live_materializer_end_overwrites_stream_progress() {
        // End event's provider-reported usage is ground truth; must
        // overwrite anything stream_progress estimated.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-pend",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-10T11:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "category": "doc-1", "model": "kimi-k2-6",
                    "started_at_iso": "2026-05-10T11:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "stream_progress", "call_id": "0001",
                    "completion_tokens": 1200, "reasoning_tokens": 800,
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": true, "duration_ms": 30000,
                    "prompt_tokens": 14000, "completion_tokens": 1180,
                    "reasoning_tokens": 790,
                    "finish_reason": "stop",
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-10T11:00:31Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        assert_eq!(calls[0]["completion_tokens"], 1180);  // end wins
        assert_eq!(calls[0]["reasoning_tokens"], 790);
    }

    #[test]
    fn live_classifier_chain_aware_reasoning_off_rebucket() {
        // A leaf whose chain went through the sizing cascade's
        // reasoning-off step (`<chain>/reasoning-off - retry/sizing`)
        // re-buckets success → success_reasoning_off. Pin that case.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-wro",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-10T04:00:00Z"}),
                // Parent times out (work-reducing trigger).
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "patterns",
                    "category": "topic", "model": "m",
                    "started_at_iso": "2026-05-10T04:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": false, "duration_ms": 1000,
                    "error": {"class": "openai.APITimeoutError", "message": "t"},
                    "ttft_ms": 5000,
                }),
                // Reasoning-off retry succeeds with full input but
                // degraded compute path. Uses the production category.
                serde_json::json!({
                    "event": "begin", "call_id": "0002", "stage": "patterns",
                    "category": "patterns/reasoning-off - retry/sizing",
                    "model": "m",
                    "retry_of_call_id": "0001",
                    "started_at_iso": "2026-05-10T04:00:02.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0002",
                    "success": true, "duration_ms": 800,
                    "prompt_tokens": 500, "completion_tokens": 100,
                    "finish_reason": "stop", "ttft_ms": 100,
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-10T04:00:10Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let calls = v["calls"].as_array().unwrap();
        let leaf = calls.iter().find(|c| c["call_id"] == "0002").unwrap();
        assert_eq!(leaf["outcome"], "success_reasoning_off",
            "leaf with /reasoning-off in chain must re-bucket success → success_reasoning_off");
    }

    #[test]
    fn materialize_cache_invalidates_on_jsonl_mtime_change() {
        // Issue #189: the on-demand materializer is mtime-cached so
        // list_runs / read_run_llm_stats / overlay don't re-parse the
        // jsonl on every poll. A write must invalidate the cached
        // payload — without invalidation, an in-flight run's call
        // count stays frozen at the last refresh.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-mtime",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-08T12:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "category": "doc-1", "model": "m",
                    "started_at_iso": "2026-05-08T12:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": true, "duration_ms": 100,
                    "prompt_tokens": 50, "completion_tokens": 20,
                }),
            ],
        );
        clear_materialize_cache_for_tests();
        let v1 = materialize_llm_stats_cached(&dir);
        assert_eq!(v1["totals"]["calls"], 1);

        // Append a second call — bumps the jsonl mtime, the cache
        // entry must invalidate. Sleep a beat to defeat 1-second
        // mtime resolution on some filesystems (tempdirs commonly
        // sit on tmpfs or ext4 with second-precision mtimes).
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let jsonl = dir.join("llm-calls.jsonl");
        let mut content = std::fs::read_to_string(&jsonl).unwrap();
        content.push_str(&serde_json::json!({
            "event": "begin", "call_id": "0002", "stage": "extract",
            "category": "doc-2", "model": "m",
            "started_at_iso": "2026-05-08T12:00:02.000Z",
        }).to_string());
        content.push('\n');
        content.push_str(&serde_json::json!({
            "event": "end", "call_id": "0002",
            "success": true, "duration_ms": 100,
            "prompt_tokens": 60, "completion_tokens": 30,
        }).to_string());
        content.push('\n');
        std::fs::write(&jsonl, content).unwrap();

        let v2 = materialize_llm_stats_cached(&dir);
        assert_eq!(v2["totals"]["calls"], 2,
            "cache must invalidate after jsonl mtime change");
    }

    #[test]
    fn read_run_llm_stats_falls_back_to_legacy_file_when_jsonl_missing() {
        // Issue #189: legacy fallback for very old runs whose
        // llm-calls.jsonl is missing/empty. The materializer returns
        // null; read_run_llm_stats then reads the on-disk
        // llm-stats.json. The fresh derivation path is preferred when
        // the jsonl exists.
        with_home_dir(|home| {
            clear_materialize_cache_for_tests();
            seed_vault_run(home, "run-legacy", &[]);
            let run_dir = home
                .join(".basevault").join("logs")
                .join("2026-04-29T12-00-00Z-app").join("eval-1").join("run-legacy");
            // No llm-calls.jsonl in this run dir — only the legacy
            // rollup file. The reader must fall through to it.
            let payload = serde_json::json!({
                "schema": "llm-stats/v1",
                "totals": {"calls": 7},
                "calls": [],
            });
            std::fs::write(run_dir.join("llm-stats.json"), payload.to_string()).unwrap();
            let v = read_run_llm_stats("run-legacy".into());
            assert_eq!(v["totals"]["calls"], 7,
                "legacy llm-stats.json must be read when jsonl is absent");
        });
    }

    #[test]
    fn live_warnings_persistent_chain_failure_counts_once() {
        // Chain: timeout → timeout → timeout (no recovery). Final
        // leaf is the LAST timeout — counts ONCE, not three times.
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_run_jsonl(
            tmp.path(),
            "sess",
            "eval-1",
            "run-x",
            &[
                serde_json::json!({"event": "cycle_start", "ts": "2026-05-07T12:00:00Z"}),
                serde_json::json!({
                    "event": "begin", "call_id": "0001", "stage": "extract",
                    "category": "split_00", "model": "m",
                    "started_at_iso": "2026-05-07T12:00:01.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0001",
                    "success": false, "duration_ms": 1000,
                    "error": {"class": "openai.APITimeoutError", "message": "t"},
                }),
                serde_json::json!({
                    "event": "begin", "call_id": "0002", "stage": "extract",
                    "category": "split_00", "model": "m",
                    "retry_of_call_id": "0001",
                    "started_at_iso": "2026-05-07T12:00:02.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0002",
                    "success": false, "duration_ms": 1000,
                    "error": {"class": "openai.APITimeoutError", "message": "t"},
                }),
                serde_json::json!({
                    "event": "begin", "call_id": "0003", "stage": "extract",
                    "category": "split_00", "model": "m",
                    "retry_of_call_id": "0002",
                    "started_at_iso": "2026-05-07T12:00:03.000Z",
                }),
                serde_json::json!({
                    "event": "end", "call_id": "0003",
                    "success": false, "duration_ms": 1000,
                    "error": {"class": "openai.APITimeoutError", "message": "t"},
                }),
                serde_json::json!({"event": "cycle_end", "ts": "2026-05-07T12:00:10Z"}),
            ],
        );
        let v = materialize_llm_stats_from_jsonl(&dir);
        let warnings = &v["warnings"];
        assert_eq!(warnings["timeouts"], 1,
            "persistent failure chain leaf counts ONCE, not per attempt");
    }

    #[test]
    fn set_run_status_paused_writes_marker_file() {
        // Post-#165: pause is a runtime control signal recorded as a
        // filesystem marker (not a jsonl event — the runner is dead at
        // this point). Derived status reads the marker → "paused".
        with_home_dir(|home| {
            clear_derive_cache_for_tests();
            let logs = home.join(".basevault").join("logs");
            // Mid-flight cycle_start; no terminator yet.
            let _ = write_run_jsonl(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-xyz",
                &[serde_json::json!({"event": "cycle_start", "ts": "2026-04-29T12:00:00Z"})],
            );
            // config.json so find_run_dir resolves the run.
            write_config_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-xyz",
                serde_json::json!({"agent": "app", "mode": "tee"}),
            );
            set_run_status("run-xyz", "paused", Some("user paused".into()))
                .expect("set_run_status");
            let dir = find_run_dir("run-xyz").unwrap();
            let flag = dir.join("paused.flag");
            assert!(flag.exists(), "paused.flag must be written");
            // Body carries the reason for post-mortem.
            assert!(std::fs::read_to_string(&flag).unwrap().contains("user paused"));
            // Static config.json is untouched (write-once invariant).
            let cfg = std::fs::read_to_string(dir.join("config.json")).unwrap();
            assert!(!cfg.contains("paused"), "config.json must not be mutated");
        });
    }

    #[test]
    fn set_run_status_cancelled_appends_cycle_event() {
        // Post-#165: cancel writes `cycle_cancelled` to the jsonl so
        // the derivation + the post-run rollup both see it. config.json
        // is never mutated.
        with_home_dir(|home| {
            clear_derive_cache_for_tests();
            let logs = home.join(".basevault").join("logs");
            let _ = write_run_jsonl(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-can",
                &[serde_json::json!({"event": "cycle_start", "ts": "2026-04-29T12:00:00Z"})],
            );
            write_config_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-can",
                serde_json::json!({"agent": "app"}),
            );
            set_run_status("run-can", "cancelled", Some("by user".into())).expect("set");
            let dir = find_run_dir("run-can").unwrap();
            let body = std::fs::read_to_string(dir.join("llm-calls.jsonl")).unwrap();
            assert!(body.contains("cycle_cancelled"), "{body}");
            assert!(body.contains("by user"), "{body}");
        });
    }

    #[test]
    fn set_run_status_failed_appends_cycle_error_event() {
        with_home_dir(|home| {
            clear_derive_cache_for_tests();
            let logs = home.join(".basevault").join("logs");
            let _ = write_run_jsonl(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-fail",
                &[serde_json::json!({"event": "cycle_start", "ts": "2026-04-29T12:00:00Z"})],
            );
            write_config_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-fail",
                serde_json::json!({"agent": "app"}),
            );
            set_run_status("run-fail", "failed", Some("subprocess crashed".into()))
                .expect("set");
            let dir = find_run_dir("run-fail").unwrap();
            let body = std::fs::read_to_string(dir.join("llm-calls.jsonl")).unwrap();
            assert!(body.contains("cycle_error"));
            assert!(body.contains("subprocess crashed"));
        });
    }

    #[test]
    fn set_run_status_refuses_to_demote_derived_terminal() {
        // Sticky-terminal: a derived status of `completed` (stage-5
        // marker on disk) should NOT be flipped to `paused` even if
        // the user races a pause click against natural completion.
        with_home_dir(|home| {
            clear_derive_cache_for_tests();
            let logs = home.join(".basevault").join("logs");
            let dir = write_run_jsonl(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-done",
                &[
                    serde_json::json!({"event": "cycle_start", "ts": "2026-04-29T12:00:00Z"}),
                    serde_json::json!({"event": "cycle_end", "ts": "2026-04-29T12:01:00Z"}),
                ],
            );
            // The stage-5 marker is the canonical "completed" signal
            // post-#206 — without it, cycle_end alone is treated as a
            // SIGTERM-paused state, not a terminal.
            write_last_stage_marker(&dir);
            write_config_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-done",
                serde_json::json!({"agent": "app"}),
            );
            set_run_status("run-done", "paused", None).expect("ok no-op");
            // No paused.flag was written.
            assert!(!dir.join("paused.flag").exists());
        });
    }

    #[test]
    fn read_run_status_returns_derived_status() {
        with_home_dir(|home| {
            clear_derive_cache_for_tests();
            let logs = home.join(".basevault").join("logs");
            let _ = write_run_jsonl(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-rs",
                &[serde_json::json!({"event": "cycle_start", "ts": "2026-04-29T12:00:00Z"})],
            );
            write_config_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-rs",
                serde_json::json!({"agent": "app"}),
            );
            assert_eq!(
                read_run_status("run-rs").unwrap(),
                Some("running".to_string())
            );
        });
    }

    #[test]
    fn read_run_status_falls_back_to_run_json_for_legacy_runs() {
        // Pre-#165 runs may have only run.json and no jsonl. The status
        // field on run.json should still surface.
        with_home_dir(|home| {
            clear_derive_cache_for_tests();
            let logs = home.join(".basevault").join("logs");
            write_run_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-legacy",
                serde_json::json!({"status": "completed", "agent": "app"}),
            );
            assert_eq!(
                read_run_status("run-legacy").unwrap(),
                Some("completed".to_string())
            );
        });
    }

    #[test]
    fn cleanup_orphaned_runs_leaves_dead_pid_post_165_orphans_paused() {
        // Issue #206: a post-#165 run dir with cycle_start in jsonl,
        // no terminator, and a dead pid is now derived as "paused"
        // directly (marker missing AND runner pid not alive). The user
        // can resume from the latest phase marker — there's no need
        // for cleanup_orphaned_runs to mark it failed.
        //
        // cleanup_orphaned_runs only acts on derive=="running" runs,
        // so it's a no-op for these orphans now: no cycle_error
        // appended, status stays "paused".
        with_home_dir(|home| {
            clear_derive_cache_for_tests();
            let logs = home.join(".basevault").join("logs");
            // jsonl with cycle_start, no terminator, dead pid.
            let dir = write_run_jsonl(
                &logs,
                "2026-05-07T12-00-00Z-app",
                "eval-1",
                "run-orphan",
                &[serde_json::json!({
                    "event": "cycle_start",
                    "ts": "2026-05-07T12:00:00Z",
                    "pid": 999999999u64,
                })],
            );
            // config.json so the run is visible to walk_run_jsons.
            std::fs::write(
                dir.join("config.json"),
                serde_json::to_string(&serde_json::json!({
                    "agent": "app",
                    "mode": "tee",
                })).unwrap(),
            ).unwrap();

            assert_eq!(derive_run_state(&dir).status, "paused");
            cleanup_orphaned_runs(None);
            // Status unchanged — cleanup skipped this run because its
            // derive is "paused", not "running".
            assert_eq!(derive_run_state(&dir).status, "paused");
            // The jsonl must NOT carry cycle_error: marking a resumable
            // run as failed would strand the user's mid-run progress.
            let body = std::fs::read_to_string(dir.join("llm-calls.jsonl")).unwrap();
            assert!(
                !body.contains("cycle_error"),
                "cleanup must not flip a dead-pid orphan to failed; it's resumable"
            );
        });
    }

    #[test]
    fn cleanup_orphaned_runs_marks_dead_running_runs_failed() {
        // Pre-#165 fallback path: a legacy run.json with status="running"
        // and a dead pid still gets marked failed in run.json (the run
        // has no jsonl with cycle markers, so derivation can't help).
        with_home_dir(|home| {
            clear_derive_cache_for_tests();
            let logs = home.join(".basevault").join("logs");
            write_run_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-orph",
                serde_json::json!({
                    "status": "running",
                    "agent": "app",
                    "pid": 999999999u64,
                }),
            );
            cleanup_orphaned_runs(None);
            let path = find_run_json("run-orph").unwrap();
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            assert_eq!(v["status"], "failed");
            assert!(v["error"].as_str().unwrap().contains("app restarted"));
        });
    }

    #[test]
    fn cleanup_orphaned_runs_leaves_terminal_runs_alone() {
        with_home_dir(|home| {
            clear_derive_cache_for_tests();
            let logs = home.join(".basevault").join("logs");
            write_run_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-done",
                serde_json::json!({"status": "completed", "agent": "app"}),
            );
            cleanup_orphaned_runs(None);
            let path = find_run_json("run-done").unwrap();
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            assert_eq!(v["status"], "completed");
        });
    }

    #[test]
    fn list_runs_filters_to_app_agent_and_sorts_newest_first() {
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            write_run_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-old",
                serde_json::json!({
                    "status": "completed",
                    "agent": "app",
                    "created_at": "2026-04-29T12:00:00Z",
                    "vault_dir": "/nonexistent/path",
                }),
            );
            write_run_json(
                &logs,
                "2026-04-29T13-00-00Z-app",
                "eval-1",
                "run-new",
                serde_json::json!({
                    "status": "completed",
                    "agent": "app",
                    "created_at": "2026-04-29T13:00:00Z",
                }),
            );
            // Experiment runs should be filtered out.
            write_run_json(
                &logs,
                "2026-04-29T14-00-00Z-experiment-foo",
                "eval-1",
                "run-experiment",
                serde_json::json!({
                    "status": "completed",
                    "agent": "experiment",
                    "created_at": "2026-04-29T14:00:00Z",
                }),
            );
            let runs = list_runs();
            // Two app-agent runs, in newest-first order.
            assert_eq!(runs.len(), 2);
            assert_eq!(runs[0]["created_at"], "2026-04-29T13:00:00Z");
            assert_eq!(runs[1]["created_at"], "2026-04-29T12:00:00Z");
            // vault_exists augmentation
            assert_eq!(runs[1]["vault_exists"], false);
        });
    }

    #[test]
    fn expand_paths_recurses_into_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("a.txt"), "a").unwrap();
        std::fs::write(dir.join("b.txt"), "b").unwrap();
        let nested = dir.join("nested");
        std::fs::create_dir(&nested).unwrap();
        std::fs::write(nested.join("c.txt"), "c").unwrap();
        let expanded = expand_paths(vec![dir.to_string_lossy().into_owned()]);
        assert_eq!(expanded.len(), 3);
        assert!(expanded.iter().any(|p| p.ends_with("a.txt")));
        assert!(expanded.iter().any(|p| p.ends_with("b.txt")));
        assert!(expanded.iter().any(|p| p.ends_with("c.txt")));
    }

    #[test]
    fn expand_paths_passes_through_files_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("only.txt");
        std::fs::write(&f, "x").unwrap();
        let expanded = expand_paths(vec![f.to_string_lossy().into_owned()]);
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0], f.to_string_lossy());
    }

    #[test]
    fn stat_paths_returns_byte_lengths_for_existing_files() {
        let tmp = tempfile::tempdir().unwrap();
        let small = tmp.path().join("small.txt");
        let bigger = tmp.path().join("bigger.bin");
        std::fs::write(&small, b"hello").unwrap();
        std::fs::write(&bigger, vec![0u8; 4096]).unwrap();
        let out = stat_paths(vec![
            small.to_string_lossy().into_owned(),
            bigger.to_string_lossy().into_owned(),
        ]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].size_bytes, 5);
        assert_eq!(out[1].size_bytes, 4096);
        // Path strings round-trip in input order — the staging UI maps
        // result-by-index back to its inputs list.
        assert_eq!(out[0].path, small.to_string_lossy());
        assert_eq!(out[1].path, bigger.to_string_lossy());
    }

    #[test]
    fn stat_paths_returns_zero_for_missing_paths_without_erroring() {
        let tmp = tempfile::tempdir().unwrap();
        let ghost = tmp.path().join("does-not-exist.txt");
        let out = stat_paths(vec![ghost.to_string_lossy().into_owned()]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].size_bytes, 0);
        assert_eq!(out[0].path, ghost.to_string_lossy());
    }

    #[test]
    fn stat_paths_handles_mix_of_present_and_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let real = tmp.path().join("real.txt");
        let ghost = tmp.path().join("nope.txt");
        std::fs::write(&real, vec![b'x'; 100]).unwrap();
        let out = stat_paths(vec![
            real.to_string_lossy().into_owned(),
            ghost.to_string_lossy().into_owned(),
        ]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].size_bytes, 100);
        assert_eq!(out[1].size_bytes, 0);
    }

    #[test]
    fn stat_paths_returns_zero_for_directory_entries() {
        // Directories aren't valid input files; the picker shouldn't
        // surface them, but the helper must not return the dir entry's
        // size as if it were a file size.
        let tmp = tempfile::tempdir().unwrap();
        let out = stat_paths(vec![tmp.path().to_string_lossy().into_owned()]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].size_bytes, 0);
    }

    #[test]
    fn estimate_work_handles_image_files_with_flat_estimate() {
        let tmp = tempfile::tempdir().unwrap();
        let img = tmp.path().join("photo.jpg");
        // Image file content size is irrelevant — flat 5-call estimate.
        std::fs::write(&img, vec![0u8; 1_000_000]).unwrap();
        let est = estimate_work(vec![img.to_string_lossy().into_owned()]);
        assert_eq!(est.file_count, 1);
        // Sum: 1+1+2+2+1 = 7 calls per image.
        assert_eq!(est.est_llm_calls, 7);
    }

    #[test]
    fn estimate_work_scales_with_text_file_size() {
        let tmp = tempfile::tempdir().unwrap();
        let small = tmp.path().join("small.txt");
        let big = tmp.path().join("big.txt");
        std::fs::write(&small, vec![b'a'; 1_000]).unwrap();
        std::fs::write(&big, vec![b'a'; 100_000]).unwrap();
        let est_small = estimate_work(vec![small.to_string_lossy().into_owned()]);
        let est_big = estimate_work(vec![big.to_string_lossy().into_owned()]);
        assert!(
            est_big.est_llm_calls > est_small.est_llm_calls,
            "bigger file should estimate more calls: small={} big={}",
            est_small.est_llm_calls, est_big.est_llm_calls
        );
    }

    // ── terminate_run: disk-state-only path (no active subprocess) ───────────

    #[test]
    fn terminate_run_disk_state_only_when_no_active_pid() {
        // terminate_run is dual-purpose: with an active pid it sends
        // SIGTERM-then-SIGKILL; with no entry in active_runs it just
        // records the terminal state. Post-#165, that record is a
        // `cycle_cancelled` event in llm-calls.jsonl (cancel) — the
        // derivation surfaces it as status="cancelled".
        with_home_dir(|home| {
            clear_derive_cache_for_tests();
            let logs = home.join(".basevault").join("logs");
            // Mid-flight cycle_start so the derived status starts at "running".
            let _ = write_run_jsonl(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-noactive",
                &[serde_json::json!({"event": "cycle_start", "ts": "2026-04-29T12:00:00Z"})],
            );
            write_config_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-noactive",
                serde_json::json!({"agent": "app"}),
            );
            let state = AppState::default();
            terminate_run(&state, "run-noactive", "cancelled", Some("by user".into()))
                .expect("terminate_run");
            // Bust the derive-cache so the post-event read picks up
            // the freshly-appended `cycle_cancelled`.
            clear_derive_cache_for_tests();
            assert_eq!(
                read_run_status("run-noactive").unwrap(),
                Some("cancelled".to_string())
            );
            // The cancellation message landed in the jsonl event payload.
            let dir = find_run_dir("run-noactive").unwrap();
            let body = std::fs::read_to_string(dir.join("llm-calls.jsonl")).unwrap();
            assert!(body.contains("by user"));
        });
    }

    #[test]
    fn pause_then_set_terminal_override_blocks_failed_race() {
        // Documented behavior of `terminal_override`: when set on the
        // ActiveRun before SIGKILL, the spawn thread's wait-result handler
        // sees override_was_set=true and skips the "mark failed" branch.
        // This is purely a struct-level test (no real subprocess) — we
        // assert that setting the override is the in-memory effect of
        // calling terminate_run with an active entry.
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            write_run_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-override",
                serde_json::json!({"status": "running", "agent": "app"}),
            );
            // Simulate an active run by inserting one. Use our own pid so
            // the kill(0) probe in terminate_run sees the process alive
            // (and would normally SIGTERM-then-SIGKILL — but kill(self, 0)
            // is harmless and self-SIGTERM with default disposition would
            // kill the test process, so we pick a pid that's almost
            // certainly NOT alive instead). The test then asserts the
            // disk-state side-effect, which doesn't depend on the kill
            // landing.
            let state = AppState::default();
            {
                let mut guard = state.active_runs.lock().unwrap();
                guard.insert(
                    "run-override".to_string(),
                    ActiveRun {
                        run_id: "run-override".to_string(),
                        pid: 999999998, // very unlikely to be alive
                        mode: "tee".to_string(),
                        terminal_override: None,
                    },
                );
            }
            terminate_run(&state, "run-override", "paused", None).expect("ok");
            // After terminate_run returns, the entry remains in active_runs
            // (the spawn thread is responsible for removing it on wait()
            // return). Confirm terminal_override was set.
            let guard = state.active_runs.lock().unwrap();
            let entry = guard.get("run-override").expect("entry still present");
            assert_eq!(
                entry.terminal_override.as_deref(),
                Some("paused"),
                "terminal_override is the user-initiated signal that prevents the spawn thread from racing to 'failed'"
            );
        });
    }

    // ── delete_run: stops the in-flight subprocess before rm -rf ─────────────

    /// Spawn `sh -c <body>` and start a background reaper thread that
    /// calls `child.wait()` as soon as the child exits. Returns the pid
    /// + the JoinHandle of the reaper.
    ///
    /// The reaper matters for `is_pid_alive` correctness: until the
    /// parent calls wait(), an exited-child remains a zombie and
    /// `kill(pid, 0)` keeps returning 0 (alive). Production reaps the
    /// pipeline subprocess from the spawn_pipeline thread; this helper
    /// gives tests the same shape so `stop_pid_with_grace`'s polling
    /// loop sees ESRCH promptly after the child obeys SIGTERM.
    ///
    /// Bodies that should exit promptly on SIGTERM should `exec` the
    /// leaf process so the PID belongs to that process directly (not a
    /// sh wrapper that may swallow forwarded signals on macOS).
    fn spawn_sh_child_with_reaper(
        body: &str,
    ) -> (u32, std::thread::JoinHandle<std::process::ExitStatus>) {
        let mut child = std::process::Command::new("sh")
            .arg("-c")
            .arg(body)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn sh child");
        let pid = child.id();
        let reaper = std::thread::spawn(move || child.wait().expect("wait"));
        (pid, reaper)
    }

    #[test]
    fn stop_pid_with_grace_sigterm_kills_obedient_child_within_grace() {
        // `exec sleep 30` makes the spawned process BE sleep (not a sh
        // wrapper). sleep exits on default-disposition SIGTERM. The
        // reaper thread calls wait() in the background so kill(0)
        // observes ESRCH promptly (matches production: spawn_pipeline's
        // own thread reaps the child).
        let (pid, reaper) = spawn_sh_child_with_reaper("exec sleep 30");
        std::thread::sleep(std::time::Duration::from_millis(80));
        assert!(is_pid_alive(pid), "child should be alive before SIGTERM");

        let started = std::time::Instant::now();
        stop_pid_with_grace(pid, std::time::Duration::from_secs(5));
        let elapsed = started.elapsed();

        let status = reaper.join().expect("reaper thread");
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "obedient child should exit on SIGTERM well before the 5s grace; took {:?}",
            elapsed
        );
        assert!(
            !status.success(),
            "killed-by-signal exit should not report success"
        );
    }

    #[test]
    fn stop_pid_with_grace_sigkill_when_child_ignores_sigterm() {
        // `trap "" TERM` makes the shell IGNORE SIGTERM, so the only way to
        // stop it is SIGKILL after the grace expires. Use a short grace so
        // the test runs in well under a second.
        let (pid, reaper) =
            spawn_sh_child_with_reaper(r#"trap "" TERM; sleep 30"#);
        std::thread::sleep(std::time::Duration::from_millis(80));
        assert!(is_pid_alive(pid), "child should be alive before SIGTERM");

        let started = std::time::Instant::now();
        stop_pid_with_grace(pid, std::time::Duration::from_millis(300));
        let elapsed = started.elapsed();

        let status = reaper.join().expect("reaper thread");
        // Grace must have actually elapsed (we waited it out before SIGKILL).
        assert!(
            elapsed >= std::time::Duration::from_millis(300),
            "must have waited the full grace before SIGKILL; elapsed={:?}",
            elapsed
        );
        // And we shouldn't be hanging around forever — SIGKILL is prompt.
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "SIGKILL should land promptly after grace; elapsed={:?}",
            elapsed
        );
        assert!(
            !status.success(),
            "SIGKILL'd child should not report success"
        );
    }

    #[test]
    fn cancel_chatbot_inflight_terminates_the_sidecar_process() {
        // The director's hard requirement: Stop must ACTUALLY stop —
        // terminate generation, not just disable the UI. Prove the
        // real process is dead after cancel, not merely deregistered.
        let (pid, reaper) = spawn_sh_child_with_reaper("exec sleep 30");
        std::thread::sleep(std::time::Duration::from_millis(80));
        assert!(is_pid_alive(pid), "sidecar stand-in should be alive");

        let state = AppState::default();
        let cancelled = Arc::new(AtomicBool::new(false));
        *state.chatbot_inflight.lock().unwrap() = Some(ChatbotInflight {
            pid,
            cancelled: cancelled.clone(),
        });

        assert!(cancel_chatbot_inflight(&state), "should report one was running");
        let status = reaper.join().expect("reaper thread");

        assert!(
            !is_pid_alive(pid),
            "the sidecar process must be terminated by cancel, not just hidden"
        );
        assert!(
            !status.success(),
            "killed-by-signal exit should not report success"
        );
        assert!(
            cancelled.load(Ordering::SeqCst),
            "cancelled flag must be set so the spawn thread suppresses chatbot_error"
        );
        assert!(
            state.chatbot_inflight.lock().unwrap().is_none(),
            "slot must be cleared after cancel"
        );
    }

    #[test]
    fn cancel_chatbot_inflight_noop_when_idle() {
        let state = AppState::default();
        assert!(
            !cancel_chatbot_inflight(&state),
            "no in-flight sidecar → cancel reports nothing was running"
        );
    }

    #[test]
    fn delete_run_stops_inflight_subprocess_before_removing_dir() {
        // End-to-end for the issue #112 fix: an in-flight run with a live
        // pid recorded in run.json must have its subprocess killed BEFORE
        // the dir is removed. Otherwise the runner's mkdir-p writers
        // re-create the freshly-deleted dir.
        with_home_dir(|home| {
            // Spawn a real long-lived child to act as the pipeline subprocess.
            let (pid, reaper) = spawn_sh_child_with_reaper("exec sleep 30");
            std::thread::sleep(std::time::Duration::from_millis(80));
            assert!(is_pid_alive(pid));

            let logs = home.join(".basevault").join("logs");
            let rj_path = write_run_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-inflight",
                serde_json::json!({
                    "status": "running",
                    "agent": "app",
                    "mode": "tee",
                    "pid": pid as u64,
                }),
            );
            let run_dir = rj_path.parent().unwrap().to_path_buf();
            assert!(run_dir.exists());

            let state = AppState::default();
            // Simulate the run being tracked by this app session so we can
            // assert the override side-effect too.
            {
                let mut guard = state.active_runs.lock().unwrap();
                guard.insert(
                    "run-inflight".to_string(),
                    ActiveRun {
                        run_id: "run-inflight".to_string(),
                        pid,
                        mode: "tee".to_string(),
                        terminal_override: None,
                    },
                );
            }

            // The Tauri command takes State<'_, AppState>; the underlying
            // logic is inline so we drive it via the helpers directly,
            // matching what delete_run does in production.
            stop_inflight_pipeline(&state, "run-inflight", &run_dir);
            assert!(
                !is_pid_alive(pid),
                "subprocess must be dead before the dir is removed"
            );
            // Override is set so the (would-be) spawn thread doesn't race
            // to write status="failed" after we delete.
            {
                let guard = state.active_runs.lock().unwrap();
                let entry = guard.get("run-inflight").expect("entry");
                assert_eq!(entry.terminal_override.as_deref(), Some("cancelled"));
            }

            // Now safe to remove the dir.
            std::fs::remove_dir_all(&run_dir).expect("remove dir");
            assert!(!run_dir.exists());

            // Drain the reaper thread to keep the test process tidy.
            let _ = reaper.join().expect("reaper");
        });
    }

    #[test]
    fn stop_inflight_pipeline_skips_terminal_runs() {
        // For terminal runs (completed/failed/cancelled) there's no
        // subprocess to coordinate with — the helper must NOT signal
        // anything (the recorded pid may belong to an unrelated process
        // due to PID reuse). Use our own pid as the bait: if the helper
        // mistakenly SIGTERM'd it the test process would die.
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            let rj_path = write_run_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-done",
                serde_json::json!({
                    "status": "completed",
                    "agent": "app",
                    "mode": "tee",
                    "pid": std::process::id() as u64,
                }),
            );
            let run_dir = rj_path.parent().unwrap().to_path_buf();
            let state = AppState::default();
            stop_inflight_pipeline(&state, "run-done", &run_dir);
            // We're still alive — that's the assertion.
            assert!(is_pid_alive(std::process::id()));
        });
    }

    // ── default_vault_root sanity ────────────────────────────────────────────

    #[test]
    fn default_vault_root_lands_under_home_documents() {
        with_home_dir(|home| {
            let v = default_vault_root();
            assert_eq!(v, home.join("Documents").join("BaseVault"));
        });
    }

    // ── read_run_file sandbox ────────────────────────────────────────────────

    fn seed_vault_run(home: &std::path::Path, run_id: &str, files: &[(&str, &str)]) {
        let logs = home.join(".basevault").join("logs");
        let vault_run = home.join("Documents").join("BaseVault").join(run_id);
        std::fs::create_dir_all(&vault_run).unwrap();
        for (rel, body) in files {
            let p = vault_run.join(rel);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(p, body).unwrap();
        }
        // Also drop a shadow file at the home root so an escape attempt
        // has a real target to compare against.
        std::fs::write(home.join("escape-target.md"), "secret").unwrap();
        write_run_json(
            &logs,
            "2026-04-29T12-00-00Z-app",
            "eval-1",
            run_id,
            serde_json::json!({
                "agent": "app",
                "vault_dir": vault_run.to_string_lossy(),
            }),
        );
    }

    #[test]
    fn read_run_file_reads_a_real_file() {
        with_home_dir(|home| {
            seed_vault_run(home, "run-ok", &[]);
            // 0-inputs/<file> reads from stages/00-ingestion/documents/.
            let docs = home
                .join(".basevault").join("logs")
                .join("2026-04-29T12-00-00Z-app").join("eval-1").join("run-ok")
                .join("stages").join("00-ingestion").join("documents");
            std::fs::create_dir_all(&docs).unwrap();
            std::fs::write(docs.join("note.md"), "hello").unwrap();
            let body = read_run_file("run-ok".into(), "0-inputs/note.md".into())
                .expect("read should succeed");
            assert_eq!(body, "hello");
        });
    }

    #[test]
    fn read_run_file_rejects_absolute_paths() {
        with_home_dir(|home| {
            seed_vault_run(home, "run-abs", &[("0-inputs/note.md", "hello")]);
            let absolute = home.join("escape-target.md").to_string_lossy().into_owned();
            let err = read_run_file("run-abs".into(), absolute).expect_err("should reject");
            assert!(err.contains("relative"), "got: {err}");
        });
    }

    #[test]
    fn read_run_file_rejects_dotdot_components() {
        with_home_dir(|home| {
            seed_vault_run(home, "run-up", &[("0-inputs/note.md", "hello")]);
            let err = read_run_file(
                "run-up".into(),
                "../../escape-target.md".into(),
            )
            .expect_err("should reject");
            assert!(err.contains("ParentDir") || err.contains(".."), "got: {err}");
        });
    }

    #[test]
    fn read_run_file_rejects_symlink_escape_via_canonicalize() {
        // Symlink inside the docs/ dir pointing at $HOME/escape-target.md
        // — `0-inputs/escape.md` looks innocent, but canonicalize()
        // resolves it outside docs/. The starts_with(canon_root) check
        // is what catches this.
        with_home_dir(|home| {
            seed_vault_run(home, "run-sym", &[]);
            let docs = home
                .join(".basevault").join("logs")
                .join("2026-04-29T12-00-00Z-app").join("eval-1").join("run-sym")
                .join("stages").join("00-ingestion").join("documents");
            std::fs::create_dir_all(&docs).unwrap();
            let target = home.join("escape-target.md");
            let link = docs.join("escape.md");
            #[cfg(unix)]
            std::os::unix::fs::symlink(&target, &link).unwrap();
            #[cfg(not(unix))]
            return; // skip on non-Unix platforms
            let err = read_run_file("run-sym".into(), "0-inputs/escape.md".into())
                .expect_err("symlink escape should be rejected");
            assert!(err.contains("escapes its root"), "got: {err}");
        });
    }

    #[test]
    fn read_run_file_returns_err_when_run_dir_missing() {
        with_home_dir(|_home| {
            let err = read_run_file("nonexistent-run".into(), "any.md".into())
                .expect_err("should error when no vault dir");
            assert!(err.contains("not found"), "got: {err}");
        });
    }

    #[test]
    fn read_run_file_returns_friendly_error_when_target_file_missing() {
        // The fact-source link click-through used to surface the raw
        // `canonicalize "/Users/.../documents/README.md": No such file
        // or directory (os error 2)` shape — leaks the absolute path
        // and reads as a crash. The friendly form is what the React
        // pane renders verbatim after `Couldn't read file: {error}`.
        with_home_dir(|home| {
            seed_vault_run(home, "run-miss", &[]);
            let docs = home
                .join(".basevault").join("logs")
                .join("2026-04-29T12-00-00Z-app").join("eval-1").join("run-miss")
                .join("stages").join("00-ingestion").join("documents");
            std::fs::create_dir_all(&docs).unwrap();
            // documents/ exists (so the root canonicalize succeeds)
            // but README.md is NOT staged. Pre-fix this returned
            // `canonicalize <abs path>: No such file or directory`.
            let err = read_run_file("run-miss".into(), "0-inputs/README.md".into())
                .expect_err("missing target should error");
            assert!(
                err.starts_with("file not available: "),
                "expected friendly form, got: {err}",
            );
            assert!(err.contains("README.md"), "should name the file: {err}");
            // No raw canonicalize / debug-formatted path leak.
            assert!(!err.contains("canonicalize"), "should not leak canonicalize: {err}");
            assert!(!err.contains("os error"), "should not leak os error: {err}");
        });
    }

    #[test]
    fn read_run_file_only_serves_0_inputs() {
        // read_run_file is now a thin pass-through for 0-inputs/* —
        // every other view is rendered in React from structured data
        // returned by the per-stage commands (read_run_facts_for_topic
        // etc.). Non-0-inputs paths return an explanatory error.
        with_home_dir(|home| {
            seed_vault_run(home, "run-r", &[]);
            let err = read_run_file("run-r".into(), "1-facts/work.md".into())
                .expect_err("non-0-inputs paths should be rejected");
            assert!(err.contains("0-inputs/*"), "got: {err}");
        });
    }

    #[test]
    fn read_run_facts_for_topic_returns_dicts_from_jsonl() {
        with_home_dir(|home| {
            seed_vault_run(home, "run-facts", &[]);
            let facts_dir = home
                .join(".basevault").join("logs")
                .join("2026-04-29T12-00-00Z-app").join("eval-1").join("run-facts")
                .join("stages").join("01-extraction").join("facts");
            std::fs::create_dir_all(&facts_dir).unwrap();
            let line1 = serde_json::json!({
                "type": "fact",
                "summary": "Alice signed",
                "topics": ["work"],
                "confidence": 0.9,
            });
            let line2 = serde_json::json!({
                "type": "event",
                "summary": "Alice met Bob",
                "topics": ["work"],
                "confidence": 0.8,
            });
            std::fs::write(
                facts_dir.join("work.jsonl"),
                format!("{}\n{}\n", line1, line2),
            ).unwrap();
            let out = read_run_facts_for_topic("run-facts".into(), "work".into());
            assert_eq!(out.len(), 2);
            assert_eq!(out[0]["summary"], "Alice signed");
            assert_eq!(out[1]["type"], "event");
        });
    }

    #[test]
    fn read_run_facts_for_topic_empty_when_topic_missing() {
        with_home_dir(|home| {
            seed_vault_run(home, "run-empty", &[]);
            let out = read_run_facts_for_topic("run-empty".into(), "health".into());
            assert!(out.is_empty());
        });
    }

    #[test]
    fn read_run_llm_stats_returns_null_when_missing() {
        with_home_dir(|home| {
            seed_vault_run(home, "run-stats-missing", &[]);
            let v = read_run_llm_stats("run-stats-missing".into());
            assert!(v.is_null(), "no llm-stats.json on disk should yield null");
        });
    }

    #[test]
    fn read_run_llm_stats_returns_null_for_unknown_run() {
        with_home_dir(|home| {
            // No vault seed → run_log_dir() returns None.
            let _ = home;
            let v = read_run_llm_stats("never-existed".into());
            assert!(v.is_null());
        });
    }

    #[test]
    fn read_run_llm_stats_returns_payload_when_present() {
        with_home_dir(|home| {
            seed_vault_run(home, "run-stats-ok", &[]);
            // Drop a synthetic llm-stats.json next to the run.json (the
            // materializer writes there; we mimic that on-disk shape).
            let run_dir = home
                .join(".basevault").join("logs")
                .join("2026-04-29T12-00-00Z-app").join("eval-1").join("run-stats-ok");
            std::fs::create_dir_all(&run_dir).unwrap();
            let payload = serde_json::json!({
                "schema": "llm-stats/v1",
                "run_id": "run-stats-ok",
                "totals": {
                    "calls": 3,
                    "outcomes": {
                        "success": 1, "success_empty": 2, "blanked": 0,
                        "parse_error": 0, "timeout": 0, "other_failure": 0,
                    },
                },
                "per_stage": {
                    "patterns": {
                        "calls_total": 3,
                        "calls_cached": 0,
                        "outcomes": {
                            "success": 1, "success_empty": 2, "blanked": 0,
                            "parse_error": 0, "timeout": 0, "other_failure": 0,
                        },
                    },
                },
                "calls": [
                    // Two calls: 0001 has a cache file present, 0002 doesn't.
                    // The cached_now stamp on the read should be true / false
                    // respectively — the live filesystem check, not the
                    // historical `cached` field.
                    {"call_id": "0001", "stage": "patterns", "outcome": "success",
                     "template_hash": "abc123abcdef",
                     "cache_key": "key-aaa", "cached": true},
                    {"call_id": "0002", "stage": "patterns", "outcome": "success",
                     "template_hash": "abc123abcdef",
                     "cache_key": "key-bbb", "cached": false},
                ],
            });
            std::fs::write(run_dir.join("llm-stats.json"), payload.to_string()).unwrap();
            // Seed the cache layer: only key-aaa has a file on disk.
            let cache_bucket = cache_root().join("patterns");
            std::fs::create_dir_all(&cache_bucket).unwrap();
            std::fs::write(cache_bucket.join("key-aaa.json"), "cached body").unwrap();

            let v = read_run_llm_stats("run-stats-ok".into());
            assert!(!v.is_null(), "llm-stats.json present should yield non-null");
            assert_eq!(v["schema"], "llm-stats/v1");
            assert_eq!(v["totals"]["calls"], 3);
            assert_eq!(v["per_stage"]["patterns"]["outcomes"]["success_empty"], 2);
            assert_eq!(v["calls"][0]["template_hash"], "abc123abcdef");
            assert_eq!(v["calls"][0]["outcome"], "success");
            // cached_now: live filesystem check stamps each call.
            assert_eq!(v["calls"][0]["cached_now"], true,
                "0001's cache file is on disk → cached_now must be true");
            assert_eq!(v["calls"][1]["cached_now"], false,
                "0002's cache file is missing → cached_now must be false");
        });
    }

    #[test]
    fn read_run_entity_returns_canonical_record_verbatim() {
        // Post-Stage-2 single-line canonical: returned as-is so the
        // panel renders description / role / aliases / evidence_fact_refs
        // unchanged.
        with_home_dir(|home| {
            seed_vault_run(home, "run-ent-canon", &[]);
            let ents = home
                .join(".basevault").join("logs")
                .join("2026-04-29T12-00-00Z-app").join("eval-1").join("run-ent-canon")
                .join("stages").join("02-entities").join("entities");
            std::fs::create_dir_all(&ents).unwrap();
            let canonical = serde_json::json!({
                "canonical_id": "alice",
                "canonical_name": "Alice",
                "entity_type": "person",
                "role": "subject",
                "description": "Author of the journal.",
                "aliases": ["A.", "Al"],
                "mention_count": 12,
                "evidence_fact_refs": [["work", 0], ["work", 3]],
            });
            std::fs::write(
                ents.join("alice.jsonl"),
                canonical.to_string() + "\n",
            ).unwrap();
            let v = read_run_entity("run-ent-canon".into(), "alice".into());
            assert_eq!(v["canonical_id"], "alice");
            assert_eq!(v["canonical_name"], "Alice");
            assert_eq!(v["mention_count"], 12);
            assert_eq!(v["aliases"][0], "A.");
            assert_eq!(v["evidence_fact_refs"].as_array().unwrap().len(), 2);
            assert!(v.get("_state").is_none(),
                "canonical record must not carry the consolidating discriminator");
        });
    }

    #[test]
    fn read_run_entity_synthesizes_consolidating_view_for_mention_stream() {
        // Mid-Stage-1 multi-line mention stream — the bug from #146.
        // Pre-fix, returning only the LAST line gave the panel a single
        // mention dict missing canonical_name / mention_count / refs,
        // and the heading rendered "0 mentions" while the run tree's
        // count badge already showed N. Pin the contract: the synthesized
        // shape carries mention_count = line count (matching the tree)
        // and mentions[] = the live stream.
        with_home_dir(|home| {
            seed_vault_run(home, "run-ent-stream", &[]);
            let ents = home
                .join(".basevault").join("logs")
                .join("2026-04-29T12-00-00Z-app").join("eval-1").join("run-ent-stream")
                .join("stages").join("02-entities").join("entities");
            std::fs::create_dir_all(&ents).unwrap();
            let mention1 = serde_json::json!({
                "name": "Author",
                "entity_type": "person",
                "role": "subject",
                "topic": "work",
                "topics": ["work"],
                "fact_summary": "Author drafted chapter 3.",
            });
            let mention2 = serde_json::json!({
                "name": "Author",
                "entity_type": "person",
                "role": "subject",
                "topic": "health",
                "topics": ["health"],
                "fact_summary": "Author skipped breakfast.",
            });
            let mention3 = serde_json::json!({
                "name": "Author",
                "entity_type": "person",
                "role": "subject",
                "topic": "work",
                "topics": ["work"],
                "fact_summary": "Author met with editor.",
            });
            std::fs::write(
                ents.join("author.jsonl"),
                format!("{}\n{}\n{}\n", mention1, mention2, mention3),
            ).unwrap();

            let v = read_run_entity("run-ent-stream".into(), "author".into());
            assert_eq!(v["_state"], "consolidating",
                "mention stream must surface the consolidating discriminator");
            assert_eq!(v["canonical_name"], "Author");
            assert_eq!(v["entity_type"], "person");
            assert_eq!(v["mention_count"], 3,
                "mention_count must match the on-disk line count");
            assert_eq!(v["mentions"].as_array().unwrap().len(), 3);
            assert_eq!(v["mentions"][0]["fact_summary"], "Author drafted chapter 3.");
            // Topics deduplicated across mentions, order preserved.
            let topics: Vec<&str> = v["topics"].as_array().unwrap()
                .iter().filter_map(|x| x.as_str()).collect();
            assert_eq!(topics, vec!["work", "health"]);
            // Empty arrays for canonical-only fields → panel's section
            // gates skip rendering canonical-only sections instead of
            // crashing on undefined.
            assert!(v["evidence_fact_refs"].as_array().unwrap().is_empty());
            assert!(v["aliases"].as_array().unwrap().is_empty());
        });
    }

    #[test]
    fn read_run_entity_handles_single_line_mention_stream() {
        // Edge case: an entity tagged in exactly one fact during Stage 1.
        // The first line still lacks `canonical_name` so we synthesize
        // a consolidating view with mention_count = 1 (matches the tree
        // node's "(1)" badge for the same file).
        with_home_dir(|home| {
            seed_vault_run(home, "run-ent-one", &[]);
            let ents = home
                .join(".basevault").join("logs")
                .join("2026-04-29T12-00-00Z-app").join("eval-1").join("run-ent-one")
                .join("stages").join("02-entities").join("entities");
            std::fs::create_dir_all(&ents).unwrap();
            let mention = serde_json::json!({
                "name": "Hand",
                "entity_type": "concept",
                "topic": "work",
            });
            std::fs::write(
                ents.join("hand.jsonl"),
                mention.to_string() + "\n",
            ).unwrap();
            let v = read_run_entity("run-ent-one".into(), "hand".into());
            assert_eq!(v["_state"], "consolidating");
            assert_eq!(v["mention_count"], 1);
            assert_eq!(v["canonical_name"], "Hand");
        });
    }

    #[test]
    fn read_run_entity_returns_null_for_missing_or_empty() {
        with_home_dir(|home| {
            seed_vault_run(home, "run-ent-miss", &[]);
            // No file on disk → null.
            let v = read_run_entity("run-ent-miss".into(), "ghost".into());
            assert!(v.is_null());
            // File exists but has no non-empty parseable lines → null
            // (no first record to derive shape from).
            let ents = home
                .join(".basevault").join("logs")
                .join("2026-04-29T12-00-00Z-app").join("eval-1").join("run-ent-miss")
                .join("stages").join("02-entities").join("entities");
            std::fs::create_dir_all(&ents).unwrap();
            std::fs::write(ents.join("empty.jsonl"), "\n  \n").unwrap();
            let v = read_run_entity("run-ent-miss".into(), "empty".into());
            assert!(v.is_null());
        });
    }

    #[test]
    fn tree_entity_count_matches_read_run_entity_mention_count() {
        // Visible-vs-readable invariant from #146: the count badge in
        // the run tree (Entities → "Author (1174)") and the panel's
        // mention_count surface MUST agree for the same on-disk file,
        // regardless of phase. Both numbers derive from the same source
        // post-fix; this test pins that contract for both shapes.
        with_home_dir(|home| {
            let stages = seed_stages(home, "run-pin");
            let ents = stages.join("02-entities").join("entities");
            std::fs::create_dir_all(&ents).unwrap();
            // Mention-stream file: 4 lines.
            let m = serde_json::json!({"name": "Author", "entity_type": "person"})
                .to_string();
            std::fs::write(
                ents.join("author.jsonl"),
                format!("{}\n{}\n{}\n{}\n", m, m, m, m),
            ).unwrap();
            // Canonical file: 7 evidence refs.
            std::fs::write(
                ents.join("alice.jsonl"),
                serde_json::json!({
                    "canonical_id": "alice",
                    "canonical_name": "Alice",
                    "mention_count": 7,
                    "evidence_fact_refs": [
                        ["work", 0], ["work", 1], ["work", 2],
                        ["work", 3], ["health", 0], ["health", 1],
                        ["health", 2],
                    ],
                }).to_string() + "\n",
            ).unwrap();

            let tree = list_run_tree("run-pin".into());
            let entities_node = tree.iter().find(|n| n.name.starts_with("Entities"))
                .expect("Entities node");
            let leaf_names: std::collections::HashSet<&str> =
                entity_leaves(entities_node).iter()
                    .map(|n| n.name.as_str())
                    .collect();
            // Tree side (leaves now nest under per-type group dirs).
            assert!(leaf_names.contains("Author (4)"),
                "got tree leaves: {:?}", leaf_names);
            assert!(leaf_names.contains("Alice (7)"),
                "got tree leaves: {:?}", leaf_names);

            // Panel side. Both must agree on the count visible in the
            // tree badge — that's the whole point of the fix.
            let author = read_run_entity("run-pin".into(), "author".into());
            assert_eq!(author["mention_count"], 4,
                "mention-stream panel count must match tree's '(4)' badge");
            let alice = read_run_entity("run-pin".into(), "alice".into());
            assert_eq!(alice["mention_count"], 7,
                "canonical panel mention_count must match tree's '(7)' badge");
        });
    }

    #[test]
    fn read_run_entities_returns_payload_or_null() {
        with_home_dir(|home| {
            seed_vault_run(home, "run-ent", &[]);
            // Missing → null.
            let v = read_run_entities("run-ent".into());
            assert!(v.is_null());
            // Present → payload returned verbatim.
            let stages = home
                .join(".basevault").join("logs")
                .join("2026-04-29T12-00-00Z-app").join("eval-1").join("run-ent")
                .join("stages").join("02-entities");
            std::fs::create_dir_all(&stages).unwrap();
            let payload = serde_json::json!({
                "subject": {"canonical_id": "alice", "display": "Alice"},
                "entities": [{"canonical_id": "alice", "canonical_name": "Alice"}],
                "relations": [],
            });
            std::fs::write(stages.join("phase_3_marker.json"), payload.to_string()).unwrap();
            let v = read_run_entities("run-ent".into());
            assert_eq!(v["subject"]["canonical_id"], "alice");
            assert_eq!(v["entities"][0]["canonical_name"], "Alice");
        });
    }

    // ── list_run_tree shape ──────────────────────────────────────────────────

    fn seed_stages(home: &std::path::Path, run_id: &str) -> std::path::PathBuf {
        seed_vault_run(home, run_id, &[]);
        home.join(".basevault").join("logs")
            .join("2026-04-29T12-00-00Z-app").join("eval-1").join(run_id)
            .join("stages")
    }

    /// Flatten the entity *leaf* nodes out of the `Entities (...)` node,
    /// which now nests them one level under per-`entity_type` group
    /// dirs. Tests that only care about the leaf badge/rel_path (not
    /// the grouping) use this so they stay shape-agnostic.
    fn entity_leaves<'a>(entities_node: &'a TreeNode) -> Vec<&'a TreeNode> {
        entities_node
            .children
            .iter()
            .flat_map(|type_group| type_group.children.iter())
            .collect()
    }

    #[test]
    fn list_run_tree_synthesizes_from_stages_with_counts() {
        with_home_dir(|home| {
            let stages = seed_stages(home, "run-tree");
            // 0-inputs/note.md
            let docs = stages.join("00-ingestion").join("documents");
            std::fs::create_dir_all(&docs).unwrap();
            std::fs::write(docs.join("note.md"), "n").unwrap();
            // 1-facts/work.jsonl with 3 facts
            let facts = stages.join("01-extraction").join("facts");
            std::fs::create_dir_all(&facts).unwrap();
            std::fs::write(facts.join("work.jsonl"),
                "{\"summary\":\"a\"}\n{\"summary\":\"b\"}\n{\"summary\":\"c\"}\n").unwrap();
            std::fs::write(facts.join("health.jsonl"),
                "{\"summary\":\"x\"}\n").unwrap();
            // 2-entities/alice.jsonl with 5 evidence_fact_refs (canonical
            // post-Stage-2 shape — single line).
            let ents = stages.join("02-entities").join("entities");
            std::fs::create_dir_all(&ents).unwrap();
            std::fs::write(
                ents.join("alice.jsonl"),
                serde_json::json!({
                    "canonical_name": "Alice",
                    "entity_type": "person",
                    "evidence_fact_refs": [["work", 0], ["work", 1], ["work", 2],
                                            ["health", 0], ["health", 1]],
                }).to_string() + "\n",
            ).unwrap();
            // 4-insights with payload of 2 cross + 1 critical
            let ins = stages.join("04-insights");
            std::fs::create_dir_all(&ins).unwrap();
            std::fs::write(
                ins.join("phase_1_marker.json"),
                serde_json::json!({
                    "cross_domain": [{}, {}],
                    "critical": [{}],
                }).to_string(),
            ).unwrap();

            let tree = list_run_tree("run-tree".into());
            let names: Vec<&str> = tree.iter().map(|n| n.name.as_str()).collect();
            // Top-level: counts wired in, .md / prefix dropped, no index.md.
            assert_eq!(
                names,
                vec!["Inputs (1)", "Facts (4)", "Entities (1)", "Insights (3)"],
            );
            // Inputs: file display drops extension + title-cases.
            assert_eq!(tree[0].children[0].name, "Note");
            assert_eq!(tree[0].children[0].rel_path, "0-inputs/note.md");
            // Facts: children sorted by count desc (work=3 then health=1).
            assert_eq!(tree[1].children[0].name, "Work (3)");
            assert_eq!(tree[1].children[0].rel_path, "1-facts/work.md");
            assert_eq!(tree[1].children[1].name, "Health (1)");
            // Entities: one per-`entity_type` group dir, then the
            // canonical leaf under it (canonical_name from JSON,
            // citations as count, leaf rel_path unchanged).
            assert_eq!(tree[2].children[0].name, "Person (1)");
            assert!(tree[2].children[0].is_dir);
            assert_eq!(tree[2].children[0].rel_path, "2-entities/__type__/person");
            assert_eq!(tree[2].children[0].children[0].name, "Alice (5)");
            assert_eq!(tree[2].children[0].children[0].rel_path, "2-entities/alice.md");
        });
    }

    #[test]
    fn list_run_tree_skips_input_dotfiles() {
        with_home_dir(|home| {
            let stages = seed_stages(home, "run-dot");
            let docs = stages.join("00-ingestion").join("documents");
            std::fs::create_dir_all(&docs).unwrap();
            std::fs::write(docs.join("note.md"), "n").unwrap();
            std::fs::write(docs.join(".DS_Store"), "").unwrap();
            let tree = list_run_tree("run-dot".into());
            // 0-inputs/ should surface only note.md, not .DS_Store.
            let inputs = tree.iter().find(|n| n.name == "Inputs (1)").unwrap();
            assert_eq!(inputs.children.len(), 1);
            assert_eq!(inputs.children[0].name, "Note");
        });
    }

    #[test]
    fn list_run_tree_attaches_size_bytes_only_to_input_leaves() {
        with_home_dir(|home| {
            let stages = seed_stages(home, "run-sizes");
            // 0-inputs/big.md — non-trivial size to assert against.
            let docs = stages.join("00-ingestion").join("documents");
            std::fs::create_dir_all(&docs).unwrap();
            std::fs::write(docs.join("big.md"), vec![b'x'; 4096]).unwrap();
            // 1-facts/work.jsonl — should NOT carry size_bytes (its
            // count badge is the relevant signal, not byte size).
            let facts = stages.join("01-extraction").join("facts");
            std::fs::create_dir_all(&facts).unwrap();
            std::fs::write(facts.join("work.jsonl"), "{\"summary\":\"a\"}\n").unwrap();
            let tree = list_run_tree("run-sizes".into());
            let inputs = tree.iter().find(|n| n.name.starts_with("Inputs")).unwrap();
            assert_eq!(inputs.size_bytes, None, "parent dir carries no size");
            assert_eq!(inputs.children[0].size_bytes, Some(4096));
            let facts_node = tree.iter().find(|n| n.name.starts_with("Facts")).unwrap();
            assert_eq!(facts_node.size_bytes, None);
            assert_eq!(facts_node.children[0].size_bytes, None);
        });
    }

    #[test]
    fn list_run_tree_entities_mention_stream_counts_lines() {
        // Stage 1 Phase 2 streams per-fact mentions into
        // 02-entities/entities/<slug>.jsonl. Tree discriminates by the
        // `canonical_name` field (only canonical post-Stage-2 records
        // have it). A single-line mention file would otherwise be
        // misread as a canonical with empty evidence_fact_refs and
        // shown as (0). Pin both shapes.
        with_home_dir(|home| {
            let stages = seed_stages(home, "run-stream");
            let ents = stages.join("02-entities").join("entities");
            std::fs::create_dir_all(&ents).unwrap();
            // Multi-line mention stream: 3 mentions of "London".
            let mention = serde_json::json!({
                "name": "London",
                "entity_type": "place",
                "fact_summary": "Trip to London.",
            }).to_string();
            std::fs::write(
                ents.join("london.jsonl"),
                format!("{}\n{}\n{}\n", mention, mention, mention),
            ).unwrap();
            // Single-line mention stream (entity tagged once during
            // Stage 1) — pre-fix this rendered as (0).
            std::fs::write(
                ents.join("hand.jsonl"),
                serde_json::json!({"name": "Hand", "entity_type": "concept"})
                    .to_string() + "\n",
            ).unwrap();
            // Canonical post-Stage-2 record with 5 evidence refs.
            std::fs::write(
                ents.join("alice.jsonl"),
                serde_json::json!({
                    "canonical_name": "Alice",
                    "evidence_fact_refs": [["t", 0], ["t", 1], ["t", 2],
                                            ["t", 3], ["t", 4]],
                }).to_string() + "\n",
            ).unwrap();

            let tree = list_run_tree("run-stream".into());
            let entities = tree.iter().find(|n| n.name.starts_with("Entities"))
                .expect("Entities node");
            let names: std::collections::HashSet<&str> =
                entity_leaves(entities).iter()
                    .map(|n| n.name.as_str())
                    .collect();
            assert!(names.contains("Alice (5)"), "got {:?}", names);
            assert!(names.contains("London (3)"), "got {:?}", names);
            assert!(names.contains("Hand (1)"), "got {:?}", names);
        });
    }

    #[test]
    fn list_run_tree_entities_grouped_by_type() {
        // #603: entities nest under a per-`entity_type` group dir, the
        // same shape facts get under per-topic dirs — group dir →
        // "<Type> (n)" → leaves. Pins: bucket from `entity_type` on
        // either on-disk shape; missing/blank → "other"; per-type count
        // = #entities of that type; groups and leaves both sort
        // count-desc-then-name; leaf rel_path stays `2-entities/<id>.md`
        // so the React entity router is untouched; group dir rel_path is
        // the synthetic non-navigable key.
        with_home_dir(|home| {
            let stages = seed_stages(home, "run-by-type");
            let ents = stages.join("02-entities").join("entities");
            std::fs::create_dir_all(&ents).unwrap();
            // person ×2: Alice (canonical, 3 refs) + Bob (mention
            // stream, 5 lines).
            std::fs::write(
                ents.join("alice.jsonl"),
                serde_json::json!({
                    "canonical_name": "Alice",
                    "entity_type": "person",
                    "evidence_fact_refs": [["t", 0], ["t", 1], ["t", 2]],
                }).to_string() + "\n",
            ).unwrap();
            let bob = serde_json::json!({"name": "Bob", "entity_type": "person"})
                .to_string();
            std::fs::write(
                ents.join("bob.jsonl"),
                format!("{}\n{}\n{}\n{}\n{}\n", bob, bob, bob, bob, bob),
            ).unwrap();
            // place ×1: London (mention stream, 2 lines).
            let london = serde_json::json!({"name": "London", "entity_type": "place"})
                .to_string();
            std::fs::write(
                ents.join("london.jsonl"),
                format!("{}\n{}\n", london, london),
            ).unwrap();
            // concept ×1: Freedom (canonical, 1 ref).
            std::fs::write(
                ents.join("freedom.jsonl"),
                serde_json::json!({
                    "canonical_name": "Freedom",
                    "entity_type": "concept",
                    "evidence_fact_refs": [["t", 0]],
                }).to_string() + "\n",
            ).unwrap();
            // No entity_type → defaults to the "other" bucket
            // (canonical, 4 refs).
            std::fs::write(
                ents.join("ghost.jsonl"),
                serde_json::json!({
                    "canonical_name": "Ghost",
                    "evidence_fact_refs": [["t", 0], ["t", 1], ["t", 2], ["t", 3]],
                }).to_string() + "\n",
            ).unwrap();

            let tree = list_run_tree("run-by-type".into());
            let entities = tree.iter().find(|n| n.name.starts_with("Entities"))
                .expect("Entities node");
            // Parent: total entity count, unchanged dir rel_path.
            assert_eq!(entities.name, "Entities (5)");
            assert!(entities.is_dir);
            assert_eq!(entities.rel_path, "2-entities");
            // Group dirs sort count-desc then name: person=2 first,
            // then the count-1 trio alphabetical by full label.
            let group_names: Vec<&str> =
                entities.children.iter().map(|n| n.name.as_str()).collect();
            assert_eq!(
                group_names,
                vec!["Person (2)", "Concept (1)", "Other (1)", "Place (1)"],
            );
            let person = &entities.children[0];
            assert!(person.is_dir, "type group is a non-navigable dir");
            assert_eq!(person.rel_path, "2-entities/__type__/person");
            // Leaves within the type sort count-desc then name: Bob(5)
            // before Alice(3); leaf rel_path keeps the bare-id form.
            let person_leaves: Vec<(&str, &str)> = person.children.iter()
                .map(|n| (n.name.as_str(), n.rel_path.as_str()))
                .collect();
            assert_eq!(
                person_leaves,
                vec![
                    ("Bob (5)", "2-entities/bob.md"),
                    ("Alice (3)", "2-entities/alice.md"),
                ],
            );
            // Untyped entity landed in the "other" bucket, leaf intact.
            let other = entities.children.iter()
                .find(|n| n.name == "Other (1)").expect("Other group");
            assert_eq!(other.rel_path, "2-entities/__type__/other");
            assert_eq!(other.children[0].name, "Ghost (4)");
            assert_eq!(other.children[0].rel_path, "2-entities/ghost.md");
        });
    }

    #[test]
    fn list_run_tree_empty_run_returns_empty_tree() {
        with_home_dir(|home| {
            seed_stages(home, "run-empty");
            // Empty stages/ → empty tree (no virtual entries to surface).
            // index.md was deleted in this commit.
            let tree = list_run_tree("run-empty".into());
            assert!(tree.is_empty(), "expected empty tree, got {} nodes", tree.len());
        });
    }

    #[test]
    fn title_case_helper() {
        assert_eq!(title_case("work"), "Work");
        assert_eq!(title_case("daily-insights"), "Daily Insights");
        assert_eq!(title_case("work_log"), "Work Log");
        // Already-cased input stays cased (no force-lowercase).
        assert_eq!(title_case("Pepys-diary"), "Pepys Diary");
    }

    #[test]
    fn export_run_no_op_when_dest_equals_vault_root() {
        // Repro for the EXISTS bug: when the user picks the vault
        // root as the export destination, regenVault has already
        // materialized the run dir at vault_root/<run_id> — the same
        // path export_run would copy to. Pre-fix: returned EXISTS
        // (overwrite=false) or wiped src (overwrite=true). Post-fix:
        // src.canonicalize() == target.canonicalize() short-circuits
        // to Ok(()) without touching the filesystem.
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            // run.json with vault_dir pointing where the vault really lives.
            let vault_root = home.join("Documents").join("BaseVault");
            let run_id = "run-samepath";
            let vault_dir = vault_root.join(run_id);
            std::fs::create_dir_all(&vault_dir).unwrap();
            // Seed some content so we can verify it survives a no-op export.
            std::fs::write(vault_dir.join("README.md"), "vault content").unwrap();
            write_run_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                run_id,
                serde_json::json!({
                    "status": "completed",
                    "agent": "app",
                    "vault_dir": vault_dir.to_string_lossy(),
                }),
            );
            // Dest = vault_root → target == src.
            let res = export_run(
                run_id.into(),
                vault_root.to_string_lossy().into_owned(),
                None,
                false, // overwrite=false: pre-fix would return EXISTS here
            );
            assert!(res.is_ok(), "expected Ok, got {:?}", res);
            // Vault content untouched.
            let content = std::fs::read_to_string(vault_dir.join("README.md")).unwrap();
            assert_eq!(content, "vault content", "no-op must not touch the vault");
            // overwrite=true: same — no-op success, src not wiped.
            let res2 = export_run(
                run_id.into(),
                vault_root.to_string_lossy().into_owned(),
                None,
                true,
            );
            assert!(res2.is_ok(), "expected Ok, got {:?}", res2);
            let content2 = std::fs::read_to_string(vault_dir.join("README.md")).unwrap();
            assert_eq!(content2, "vault content", "no-op (overwrite=true) must not wipe the vault");
        });
    }

    #[test]
    fn read_run_preprocessed_inputs_returns_md_files_keyed_by_file_id() {
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            write_run_json(
                &logs,
                "2026-04-29T12-00-00Z-app",
                "eval-1",
                "run-pre",
                serde_json::json!({"status": "completed", "agent": "app"}),
            );
            let docs = logs
                .join("2026-04-29T12-00-00Z-app")
                .join("eval-1")
                .join("run-pre")
                .join("stages")
                .join("00-ingestion")
                .join("documents");
            std::fs::create_dir_all(&docs).unwrap();
            std::fs::write(docs.join("note.md"), "# hello").unwrap();
            // Subdir + non-md file: subdir traversal works, non-md ignored.
            std::fs::create_dir_all(docs.join("nested")).unwrap();
            std::fs::write(docs.join("nested").join("inner.md"), "inner body").unwrap();
            std::fs::write(docs.join("skip.txt"), "ignored").unwrap();

            let v = read_run_preprocessed_inputs("run-pre".into());
            let m = v.as_object().expect("object");
            assert_eq!(m.get("note").and_then(|x| x.as_str()), Some("# hello"));
            assert_eq!(
                m.get("nested/inner").and_then(|x| x.as_str()),
                Some("inner body"),
            );
            assert!(m.get("skip").is_none());
            assert_eq!(m.len(), 2);
        });
    }

    #[test]
    fn read_run_preprocessed_inputs_empty_when_run_missing() {
        with_home_dir(|_home| {
            let v = read_run_preprocessed_inputs("nonexistent".into());
            let m = v.as_object().expect("object");
            assert!(m.is_empty());
        });
    }

    #[test]
    fn write_run_vault_writes_files_and_wipes_subdirs() {
        with_home_dir(|home| {
            let vault_root = home.join("Documents").join("BaseVault");
            let run_id = "run-write";
            let vault_dir = vault_root.join(run_id);
            std::fs::create_dir_all(&vault_dir).unwrap();
            // Stale per-topic file that should be removed by the wipe.
            std::fs::create_dir_all(vault_dir.join("1-facts")).unwrap();
            std::fs::write(
                vault_dir.join("1-facts").join("stale-topic.md"),
                "should disappear",
            ).unwrap();
            // Single-file root that should be overwritten.
            std::fs::write(vault_dir.join("4-insights.md"), "old insights body").unwrap();

            let mut files = std::collections::HashMap::new();
            files.insert("1-facts/work.md".into(), "facts body".into());
            files.insert("4-insights.md".into(), "new insights body".into());
            files.insert("index.md".into(), "index body".into());

            write_run_vault(run_id.into(), files).expect("write_run_vault");

            // New file written.
            let work = std::fs::read_to_string(vault_dir.join("1-facts").join("work.md")).unwrap();
            assert_eq!(work, "facts body");
            // Stale file removed by the wipe-and-rewrite.
            assert!(!vault_dir.join("1-facts").join("stale-topic.md").exists());
            // Single-file root replaced.
            let ins = std::fs::read_to_string(vault_dir.join("4-insights.md")).unwrap();
            assert_eq!(ins, "new insights body");
            // README.md at the vault root was refreshed.
            let readme = std::fs::read_to_string(vault_root.join("README.md")).unwrap();
            assert!(readme.contains("# BaseVault runs"));
            assert!(readme.contains(run_id));
        });
    }

    #[test]
    fn write_run_vault_rejects_path_traversal() {
        with_home_dir(|_home| {
            let mut files = std::collections::HashMap::new();
            files.insert("../escape.md".into(), "no".into());
            let res = write_run_vault("run-evil".into(), files);
            assert!(res.is_err(), "expected Err for ..");
            let mut files2 = std::collections::HashMap::new();
            files2.insert("/etc/passwd".into(), "no".into());
            let res2 = write_run_vault("run-evil".into(), files2);
            assert!(res2.is_err(), "expected Err for absolute path");
        });
    }

    #[test]
    fn window_close_keeps_app_and_runs_alive() {
        // Cmd+W / red button / File ▸ Close Window: CloseRequested
        // raised the flag, exit is user interaction (code=None).
        assert!(exit_should_keep_alive(None, true));
    }

    #[test]
    fn quit_and_programmatic_exits_run_cleanup() {
        // Cmd+Q / menu Quit / system terminate: user-interaction exit
        // but no preceding window-close.
        assert!(!exit_should_keep_alive(None, false));
        // Updater relaunch via app.exit(0): programmatic (code=Some),
        // must pause runs so they resume after the new instance starts.
        assert!(!exit_should_keep_alive(Some(0), false));
        // Defensive: a programmatic exit always quits even if a
        // window-close flag is somehow still set — never strand a
        // requested quit.
        assert!(!exit_should_keep_alive(Some(0), true));
    }

    // ── Chatbot run selector (#507) ─────────────────────────────────────────

    fn make_run(logs: &Path, name: &str, size: usize) {
        let store = logs
            .join(name)
            .join("stages")
            .join("06-embeddings")
            .join("vectors.db");
        std::fs::create_dir_all(store.parent().unwrap()).unwrap();
        std::fs::write(&store, vec![b'x'; size]).unwrap();
    }

    #[test]
    fn store_path_excludes_zero_byte_defect_2() {
        let tmp = tempfile::tempdir().unwrap();
        make_run(tmp.path(), "good", 16);
        make_run(tmp.path(), "inflight", 0);
        assert!(chatbot_store_path(&tmp.path().join("good")).is_some());
        // 0-byte store is in-flight/aborted — never bindable.
        assert!(chatbot_store_path(&tmp.path().join("inflight")).is_none());
        assert!(chatbot_store_path(&tmp.path().join("missing")).is_none());
    }

    #[test]
    fn run_time_key_dir_prefix_then_config_then_none() {
        let tmp = tempfile::tempdir().unwrap();
        let iso = tmp.path().join("2026-05-16T03-14-54Z-xttq");
        std::fs::create_dir_all(&iso).unwrap();
        assert_eq!(
            chatbot_run_time_key(&iso).as_deref(),
            Some("20260516031454")
        );
        // A slug that merely starts with digits is NOT a stamp.
        let legacy = tmp.path().join("99-not-a-stamp");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(
            legacy.join("config.json"),
            r#"{"created_at":"2026-05-10T01:07:11Z"}"#,
        )
        .unwrap();
        assert_eq!(
            chatbot_run_time_key(&legacy).as_deref(),
            Some("20260510010711")
        );
        let bare = tmp.path().join("no-prefix-no-config");
        std::fs::create_dir_all(&bare).unwrap();
        assert!(chatbot_run_time_key(&bare).is_none());
    }

    #[test]
    fn run_label_is_input_subject_not_slug_and_carries_no_date() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("2026-05-16T03-14-54Z-xttq");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"inputs":["/data/test/personal-os.txt"]}"#,
        )
        .unwrap();
        // Subject only — the time lives in `created_at`, rendered
        // client-side in local tz. No embedded wall-clock date (it would
        // be UTC, skewing every non-UTC user ahead of the run view).
        let label = chatbot_run_label(&dir);
        assert_eq!(label, "personal-os.txt");
        assert!(!label.contains("xttq"));
        assert!(!label.contains('—'));

        // Multi-input → first +N more, still no date.
        std::fs::write(
            dir.join("config.json"),
            r#"{"inputs":["/a/one.txt","/a/two.txt","/a/three.txt"]}"#,
        )
        .unwrap();
        assert_eq!(chatbot_run_label(&dir), "one.txt +2 more");

        // Unreadable config → bare slug, never a vanished run, no date.
        let bare = tmp.path().join("2026-05-16T01-14-42Z-f66s");
        std::fs::create_dir_all(&bare).unwrap();
        assert_eq!(chatbot_run_label(&bare), "2026-05-16T01-14-42Z-f66s");
    }

    #[test]
    fn run_time_key_to_rfc3339_is_parseable_utc_or_empty() {
        // #577 Part B: the run-option `created_at` must be a string
        // `new Date()` parses, so the shared dropdown formats it with
        // the same `prettyDateTime` as every other surface.
        assert_eq!(
            run_time_key_to_rfc3339("20260516031454"),
            "2026-05-16T03:14:54Z"
        );
        // Trailing garbage past the 14 digits is ignored (the key is
        // fixed-width); too-short / non-digit yields empty so the
        // caller drops the date line instead of printing a lie.
        assert_eq!(
            run_time_key_to_rfc3339("20260516031454999"),
            "2026-05-16T03:14:54Z"
        );
        assert_eq!(run_time_key_to_rfc3339("2026051603145"), "");
        assert_eq!(run_time_key_to_rfc3339("not-a-key"), "");
        assert_eq!(run_time_key_to_rfc3339(""), "");
    }

    #[test]
    fn enumerate_carries_created_at_and_short_id() {
        // #577 Part B: each run option exposes a parseable RFC3339
        // `created_at` and the dir-derived 4-letter perma-id as REAL
        // fields (not string-parsed back out of `label`).
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            std::fs::create_dir_all(&logs).unwrap();
            make_run(&logs, "2026-05-16T03-14-54Z-xttq", 16);
            let rows = enumerate_chatbot_runs();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].run_id, "2026-05-16T03-14-54Z-xttq");
            assert_eq!(rows[0].created_at, "2026-05-16T03:14:54Z");
            assert_eq!(rows[0].short_id, "xttq");
        });
    }

    #[test]
    fn enumerate_orders_by_creation_not_mtime_and_caps_10() {
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            std::fs::create_dir_all(&logs).unwrap();
            // Older run, but touch its db LAST (the exact mtime trap).
            make_run(&logs, "2026-05-16T01-14-42Z-f66s", 16);
            make_run(&logs, "2026-05-16T03-14-54Z-xttq", 16);
            make_run(&logs, "2026-05-16T09-00-00Z-zzzz", 0); // in-flight
            // Set the OLDER run's db mtime AFTER the newer run's — the
            // exact file-mtime trap that caused the shadowing (#507 #1).
            let db = |n: &str| {
                logs.join(n).join("stages/06-embeddings/vectors.db")
            };
            let set_mtime = |p: PathBuf, t: std::time::SystemTime| {
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(&p)
                    .unwrap()
                    .set_modified(t)
                    .unwrap();
            };
            let now = std::time::SystemTime::now();
            let old = now - std::time::Duration::from_secs(9000);
            set_mtime(db("2026-05-16T03-14-54Z-xttq"), old);
            set_mtime(db("2026-05-16T01-14-42Z-f66s"), now);

            let rows = enumerate_chatbot_runs();
            // Newest NON-EMPTY run wins regardless of file-mtime; the
            // 0-byte run is absent entirely (defect #1 + #2).
            assert_eq!(rows[0].run_id, "2026-05-16T03-14-54Z-xttq");
            assert!(rows
                .iter()
                .all(|r| r.run_id != "2026-05-16T09-00-00Z-zzzz"));
            assert_eq!(rows.len(), 2);
        });
    }

    #[test]
    fn resolve_binding_user_pick_then_stale_fallback_to_default() {
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            std::fs::create_dir_all(&logs).unwrap();
            make_run(&logs, "2026-05-16T01-14-42Z-f66s", 16);
            make_run(&logs, "2026-05-16T03-14-54Z-xttq", 16);
            let state = AppState::default();

            // No pick → most-recent non-empty, source "default".
            let (_, run, src) = resolve_chatbot_binding(&state).unwrap();
            assert_eq!(run, "2026-05-16T03-14-54Z-xttq");
            assert_eq!(src, "default");

            // Explicit pick honoured even though it's the older run.
            *state.chatbot_selected_run.lock().unwrap() =
                Some("2026-05-16T01-14-42Z-f66s".into());
            let (_, run, src) = resolve_chatbot_binding(&state).unwrap();
            assert_eq!(run, "2026-05-16T01-14-42Z-f66s");
            assert_eq!(src, "user");

            // Stale pick (run gone) → silently degrade to the default,
            // never bind nothing / a vanished store.
            *state.chatbot_selected_run.lock().unwrap() =
                Some("deleted-run".into());
            let (_, run, src) = resolve_chatbot_binding(&state).unwrap();
            assert_eq!(run, "2026-05-16T03-14-54Z-xttq");
            assert_eq!(src, "default");
        });
    }

    /// Regression: a pick that ages out of the top-10 dropdown list
    /// must stop binding too, not just stop displaying. Pre-fix the
    /// dropdown capped at 10 and visually defaulted to the most-recent
    /// when the pick was older than the 10th, but
    /// `resolve_chatbot_binding` only checked the store-on-disk —
    /// so a sidecar respawn kept binding the aged-out pick even
    /// though the user couldn't see it in the dropdown anymore. The
    /// fix anchors both functions on the same top-10 list.
    #[test]
    fn resolve_binding_aged_out_pick_falls_through_to_default() {
        with_home_dir(|home| {
            let logs = home.join(".basevault").join("logs");
            std::fs::create_dir_all(&logs).unwrap();
            // 11 non-empty runs, oldest first. The dropdown caps at
            // the 10 most-recent → `aged-out` is excluded from the list.
            let aged_out_id = "2026-05-01T00-00-00Z-aged";
            make_run(&logs, aged_out_id, 16);
            for i in 0..10 {
                make_run(
                    &logs,
                    &format!("2026-05-16T{:02}-00-00Z-r{:03}", i, i),
                    16,
                );
            }
            let state = AppState::default();

            // Sanity: aged-out IS still on disk (the bare existence
            // check the pre-fix code used would honour it).
            assert!(chatbot_run_store(aged_out_id).is_some());

            // But the top-10 list `enumerate_chatbot_runs` returns
            // doesn't include it.
            let rows = enumerate_chatbot_runs();
            assert_eq!(rows.len(), 10);
            assert!(!rows.iter().any(|r| r.run_id == aged_out_id));

            // Pick the aged-out run. Its vectors.db is still on disk,
            // but it's not in the top-10 — fix degrades to default.
            *state.chatbot_selected_run.lock().unwrap() =
                Some(aged_out_id.into());
            let (_, run, src) = resolve_chatbot_binding(&state).unwrap();
            assert_ne!(run, aged_out_id);  // did NOT bind aged-out pick
            assert_eq!(src, "default");
            // The default = first row of the list = whichever row the
            // dropdown highlights as bound. UI and binding agree.
            assert_eq!(run, rows[0].run_id);
        });
    }

    // Issue #581: in_flight_calls must count only PENDING unmatched
    // begins. A wound-down run (cancelled / paused / etc.) has zero
    // calls executing — every dangling begin is an aborted artifact,
    // not "in progress". And after a resume, begins stranded in the
    // superseded earlier cycle are aborted, not in flight.
    #[test]
    fn in_flight_excludes_aborted_and_superseded_begins() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path();

        // 1) Live latest cycle: two begins still open, one begin/end
        //    pair closed → 2 pending in flight.
        std::fs::write(
            run_dir.join("llm-calls.jsonl"),
            concat!(
                r#"{"event":"cycle_start","ts":"2026-05-17T00-00-00Z","cycle_seq":1}"#, "\n",
                r#"{"event":"begin","call_id":"a","stage":"extract"}"#, "\n",
                r#"{"event":"begin","call_id":"b","stage":"extract"}"#, "\n",
                r#"{"event":"begin","call_id":"c","stage":"extract"}"#, "\n",
                r#"{"event":"end","call_id":"c","success":true}"#, "\n",
            ),
        )
        .unwrap();
        let st = derive_run_state_uncached(run_dir);
        assert_eq!(st.status, "running");
        assert_eq!(st.progress.in_flight_calls, 2);

        // 2) Same dangling begins, but the run was cancelled → the
        //    begins are aborted, NOT in flight. Pre-fix this reported 2.
        std::fs::write(
            run_dir.join("llm-calls.jsonl"),
            concat!(
                r#"{"event":"cycle_start","ts":"2026-05-17T00-00-00Z","cycle_seq":1}"#, "\n",
                r#"{"event":"begin","call_id":"a","stage":"extract"}"#, "\n",
                r#"{"event":"begin","call_id":"b","stage":"extract"}"#, "\n",
                r#"{"event":"cycle_cancelled","ts":"2026-05-17T00-01-00Z"}"#, "\n",
            ),
        )
        .unwrap();
        let st = derive_run_state_uncached(run_dir);
        assert_eq!(st.status, "cancelled");
        assert_eq!(st.progress.in_flight_calls, 0);

        // 3) Resume: cycle 1's begin was stranded, cycle 2 is live with
        //    one open begin → only the cycle-2 begin is in flight.
        std::fs::write(
            run_dir.join("llm-calls.jsonl"),
            concat!(
                r#"{"event":"cycle_start","ts":"2026-05-17T00-00-00Z","cycle_seq":1}"#, "\n",
                r#"{"event":"begin","call_id":"old","stage":"extract"}"#, "\n",
                r#"{"event":"cycle_start","ts":"2026-05-17T00-05-00Z","cycle_seq":2}"#, "\n",
                r#"{"event":"begin","call_id":"new","stage":"patterns"}"#, "\n",
            ),
        )
        .unwrap();
        let st = derive_run_state_uncached(run_dir);
        assert_eq!(st.status, "running");
        assert_eq!(st.progress.in_flight_calls, 1);
    }
}
