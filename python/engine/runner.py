"""
Pipeline runner — orchestrates the full pipeline end-to-end and emits
JSON progress lines to stdout so the Tauri shell can track LLM calls.

Each LLM call prints: {"stage": "<name>", "completed": <N>}
Final line:           {"stage": "done", "completed": <N>, "items": <count>}
Errors:               {"stage": "error", "message": "..."}

Output tree (root split by caller — see `_LOGS_ROOT` below):

    ~/.basevault/logs/<run-name>/                            # app runs only (Tauri sets agent=app)
        run.log
        config.json                                          # static start-of-run snapshot, write-once (#165)
        llm-calls.jsonl                                      # append-only event log; status / progress / cycle
                                                             #   markers all live here. Tauri-side
                                                             #   `derive_run_state` walks this for the runs list.
        paused.flag                                          # Tauri marker — present iff paused (#165)
        llm-stats.txt                                        # human-readable rollup sibling — derived
                                                             #   on demand via materialize_run_stats() (#189)
        run.json                                             # legacy sidecar — only present on pre-#165 runs;
                                                             #   read-only fallback for static fields when
                                                             #   config.json is absent
        stages/
            00-ingestion/
                documents/<DOCUMENT>                         # normalized text per source
                phase_1_marker.json                          # payload: list of generated docs
            01-extraction/
                facts/<TOPIC>.jsonl                          # per-topic, append-only during phase 2
                phase_1_marker.json                          # splitter output (segments + report)
                phase_2_marker.json                          # per-call extract metadata
                phase_3_marker.json                          # per-topic counts (post-sort)
            02-entities/
                entities/<ENTITY>.jsonl                      # in-flight mention stream during Stage 1 Phase 2;
                                                             # consolidated canonical record post-Stage-2 Phase 1
                                                             # (single line, evolved across Phase 1 → 2 → 3)
                phase_1_marker.json                          # group metadata (deterministic)
                phase_2_marker.json                          # entity summaries (LLM)
                phase_3_marker.json                          # canonical aggregate + alias list
            03-patterns/
                patterns/<TOPIC>.json                        # per-topic patterns (LLM)
                phase_1_marker.json                          # per-topic counts
            04-insights/
                phase_1_marker.json                          # full insights payload (no separate insights.json)
            05-actions/
                phase_1_marker.json                          # full actions payload (no separate actions.json)
        vault/                                               # experiment --emit-vault only

    ~/.basevault/logs-dev/<run-name>/                        # everything else: scripts, smoke,
                                                             # tests, ad-hoc CLI (agent="experiment",
                                                             # the default when BASEVAULT_AGENT is unset).
                                                             # Not scanned by the GUI's runs list.

    ~/.basevault/logs-dev/sweeps/<sweep-id>/                 # eval-namespace; sweep harness writes here
        eval.json                                            # sweep manifest (cases, modes, judge_model, model_tags)
        report/judge.md                                      # cross-case judge output
        <case>-<mode>/                                       # one run per (case, mode) — same shape as above

    ~/Documents/BaseVault/<run-name>/                        # app sessions only, Obsidian-visible, flat
        0-inputs/ 1-facts/ 2-entities/ 3-patterns/ 4-insights.md 5-actions.md

Layout-tolerant reads: legacy nested `<session>/<eval>/<run>/` corpora
on disk from before the flatten remain readable. Resume picks them up
via `_find_run_dir_by_id` (multi-pattern glob); the Rust list_runs
walk also accepts both shapes. No file moves required.

Environment contract:
    BASEVAULT_LOGS_ROOT   — explicit override for the logs root. Tests + sweep
                            harness (sweep callers point this at
                            <base>/sweeps/<sweep_id> so the runner writes flat
                            into the sweep dir without needing to know about
                            sweeps). When unset, the root is inferred from
                            BASEVAULT_AGENT: 'app' → ~/.basevault/logs/,
                            anything else → ~/.basevault/logs-dev/.
    BASEVAULT_VAULT_ROOT  — override for ~/Documents/BaseVault. Tests.
    BASEVAULT_RUN_NAME    — run dir name. Expected shape is
                            '<iso-z>-<4-char-id>' (e.g. '2026-04-24T23-32-06Z-f5pf').
                            The Rust shell always generates this format;
                            CLI users and the sweep harness pass their own.
    BASEVAULT_AGENT       — 'app' (Tauri shell sets it explicitly when
                            spawning runner.py) or 'experiment' (default).
                            Drives both the logs-root split and
                            `list_runs`'s app-vs-experiment filter via the
                            `agent` field stamped into config.json.

Usage:
    cd engine && python3 runner.py --paths file1.txt --mode local
    cd engine && python3 runner.py --resume-run-dir <abs path>
    cd engine && python3 runner.py --resume-run-id <run_name>
"""
from __future__ import annotations

# ── Reproducibility: pin hash randomization ──────────────────────────────
# The LLM disk cache keys on the exact prompt bytes. Python's per-process
# string-hash randomization makes set/dict iteration order vary run to
# run; if that order leaks into prompt assembly the cache key changes on
# identical inputs, so a cached-input run no longer replays — it silently
# recomputes with a fresh (non-deterministic) LLM call. Prompt builders
# are also seed-independent by construction (total ordering on every
# set→sorted path); this pin is the broad net for any other set/dict
# leak, present or future. Re-exec once with the seed pinned so EVERY
# entrypoint is covered uniformly — the packaged .app sidecar, the dev
# CLI, scripts — rather than relying on each spawner to set the env (the
# .app path would otherwise miss it). os.execv keeps the same PID, so the
# Tauri shell's process tracking and pause/cancel signal delivery are
# unaffected. Guarded to __main__ so `import runner` (tests, sibling
# imports) never re-execs the test runner out from under itself.
if __name__ == "__main__":
    import os as _os_seed
    import sys as _sys_seed
    if _os_seed.environ.get("PYTHONHASHSEED") != "0" and _sys_seed.executable:
        _os_seed.environ["PYTHONHASHSEED"] = "0"
        # Re-exec via sys.orig_argv (NOT sys.argv) so the EXACT launch form is
        # preserved. We're spawned as `python -m engine.runner …`; sys.argv
        # drops the `-m engine.runner` and leaves the bare runner.py path, so
        # re-execing it would run in script mode and lose the `engine` package
        # root from sys.path (ModuleNotFoundError: No module named 'engine').
        # orig_argv keeps the `-m engine.runner` invocation intact.
        _os_seed.execv(_sys_seed.executable, _sys_seed.orig_argv)

# ── DEV TRACING ──────────────────────────────────────────────────────────
# Opt-in timing instrumentation gated by the `dev_tracing` config flag
# (Settings → Development). Rust spawns this process with
# BASEVAULT_DEV_TRACING=1 in the env iff the toggle is ON; user-set
# shell values are scrubbed Rust-side so the config flag is the only
# knob. First call sets the Python-layer t=0; `wall` is unix epoch
# seconds for cross-layer correlation. Output goes to stdout — Tauri's
# BufReader forwards [LAUNCH_TRACE] lines through the Rust info! sink
# so they land in app.log alongside Rust + frontend markers.
import os as _os_dt
import time as _time_dt
_DEV_TRACING = _os_dt.environ.get("BASEVAULT_DEV_TRACING") == "1"
_DEV_TRACING_T0 = _time_dt.monotonic()
def _ltrace(step: str) -> None:
    if not _DEV_TRACING:
        return
    try:
        t = _time_dt.monotonic() - _DEV_TRACING_T0
        wall = _time_dt.time()
        print(f"[LAUNCH_TRACE] {step} t={t:.3f} wall={wall:.3f}", flush=True)
    except Exception:
        pass
_ltrace("runner_module_loaded")
# ── /DEV TRACING ─────────────────────────────────────────────────────────

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from engine._dotenv import load as _load_dotenv, candidates as _dotenv_candidates
_DOTENV_LOADED_FROM = _load_dotenv()
_DOTENV_CANDIDATES = _dotenv_candidates()


# CRITICAL — module aliasing for `python -m engine.runner` invocation.
# Tauri spawns `python -m engine.runner …`, which loads this file as
# `__main__`. Any sibling module that does `from engine.runner import X`
# causes Python to re-import runner.py as a SEPARATE module named
# `engine.runner`, so `__main__.X` and `engine.runner.X` become two
# different objects. A class raised from one and caught (via `except` /
# `isinstance`) against the other would not match — the handler
# falls through to `except Exception`, which silently drops the
# call. Aliasing `__main__` under the name `engine.runner` BEFORE any
# sibling import makes every subsequent `from engine.runner import …`
# return the SAME module object, so all symbols are identity-equal.
import sys as _sys_alias
if __name__ == "__main__":
    _sys_alias.modules.setdefault("engine.runner", _sys_alias.modules[__name__])


# Bump the soft NOFILE limit before any thread/socket/file work starts.
# The pipeline can fan out to ~16 concurrent LLM workers (see
# llm.max_workers); each holds an httpx connection-pool socket plus
# transient file handles for partial writes. Default macOS soft limit
# is 256, which a Day One JSON of thousands of tiny entries can exhaust
# (one fd per concurrent in-flight request × multiple stages).
# Best-effort: silently keep the default if the OS refuses.
try:
    import resource as _resource
    _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    _target = min(max(_soft, 8192), _hard if _hard != _resource.RLIM_INFINITY else 65536)
    if _target > _soft:
        _resource.setrlimit(_resource.RLIMIT_NOFILE, (_target, _hard))
except (ImportError, ValueError, OSError):
    pass


# ── Paths ─────────────────────────────────────────────────────────────────────

# Logs split by caller:
#   ~/.basevault/logs/      — real app runs (Tauri shell sets BASEVAULT_AGENT=app)
#   ~/.basevault/logs-dev/  — everything else (scripts, smoke, sweeps, tests,
#                             ad-hoc CLI). Default when BASEVAULT_AGENT is unset.
# The Obsidian vault lives under ~/Documents/BaseVault/ — a flat list of run dirs.
# `BASEVAULT_LOGS_ROOT` overrides the inferred root (sweep harness points it at
# its sweep dir; tests use it for tmp isolation).
_LOGS_ROOT = Path(
    os.environ.get("BASEVAULT_LOGS_ROOT")
    or (
        Path.home() / ".basevault"
        / ("logs" if os.environ.get("BASEVAULT_AGENT", "").strip() == "app" else "logs-dev")
    )
)
_VAULT_ROOT = Path(
    os.environ.get("BASEVAULT_VAULT_ROOT")
    or (Path.home() / "Documents" / "BaseVault")
)

# Legacy aliases retained so callers that still reference them don't break;
# nothing reads from these anymore.
_RUN_ROOT = _LOGS_ROOT
_OUTPUT_ROOT = _LOGS_ROOT

# Resolved per-run state — populated by _resolve_paths().
_run_name: str | None = None
_run_dir: Path | None = None
_vault_dir: Path | None = None

_log_file = None
_log_lock = threading.Lock()


def _truncate(text: str, n: int = 160) -> str:
    """Shorten a string to at most n chars, collapsing whitespace."""
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def _iso_z() -> str:
    """ISO-8601 UTC with colons replaced by dashes (safe as a dir name)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _iso_z_full() -> str:
    """ISO-8601 UTC with colons intact (for timestamp fields inside JSON)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Short-id helpers (alphabet, validator, parser, minter) all live in
# `common/utils.py` — single source of truth, byte-identical to the
# Rust short_id generator in `src-tauri/src/lib.rs`. The thin
# wrapper here preserves the historical "" (not None) return on no
# match so call sites don't have to learn a new sentinel.
from engine.common.utils import short_id_from_name as _short_id_from_name  # noqa: E402


def _short_id_from_run_name(run_name: str | None) -> str:
    return _short_id_from_name(run_name) or ""


def _agent() -> str:
    """`"app"` (Tauri shell sets it explicitly) or `"experiment"` (default —
    scripts, smoke, sweeps, tests, ad-hoc CLI). Drives both the logs-root
    split (`logs/` vs `logs-dev/`) and `list_runs`'s app-vs-experiment
    filter on the Tauri side so dev runs don't show up in the user's
    runs list.
    """
    val = os.environ.get("BASEVAULT_AGENT", "").strip()
    return val if val in ("app", "experiment") else "experiment"


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
            timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()[:12]
    except Exception:
        pass
    return None


def _git_branch() -> str | None:
    """Current git branch — populated when running from a source checkout
    (worker tmux, sweep harness, dev `tauri dev`), null in production .app
    where the bundled pipeline has no parent .git."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
            timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or None
    except Exception:
        pass
    return None


def _find_run_dir_by_id(run_id: str) -> Path | None:
    """Locate a run dir under _LOGS_ROOT whose leaf name matches `run_id`.

    App run names are timestamped and globally unique, so at most one hit
    is expected. If multiple hits (shouldn't happen but possible in tests),
    returns the one with the most recent mtime.

    Layout-tolerant: matches both flat `<logs_root>/<run_id>/` and the
    legacy nested `<logs_root>/<session>/<eval>/<run_id>/` (and the
    sweep variant at the same depth, e.g.
    `<logs_root>/sweeps/<sweep_id>/<run_id>/`).
    """
    logs_root = Path(globals().get("_LOGS_ROOT") or globals().get("_RUN_ROOT"))
    if not logs_root.exists():
        return None
    matches: list[Path] = []
    for pattern in (run_id, f"*/{run_id}", f"*/*/{run_id}"):
        matches.extend(p for p in logs_root.glob(pattern) if p.is_dir())
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _resolve_paths(
    resume_run_dir: Path | None = None,
    resume_run_id: str | None = None,
) -> None:
    """Populate module-level run/vault paths.

    Layout: `<logs_root>/<run_name>/`. The sweep harness lands its runs
    inside `<logs_root>/sweeps/<sweep_id>/<run_name>/` by setting
    `BASEVAULT_LOGS_ROOT` to its sweep dir before spawning the runner —
    this file stays unaware of sweeps.

    Reads `BASEVAULT_RUN_NAME` (run dir name) and `BASEVAULT_AGENT`
    (`"experiment"` default; `"app"` set explicitly by the Tauri shell —
    used to feed `list_runs`'s app-vs-experiment filter).

    Resume: `resume_run_id` is resolved layout-tolerantly via
    `_find_run_dir_by_id`, which accepts both flat and the legacy
    `<session>/<eval>/<run>/` shapes so corpora on disk from before
    the flatten remain resumable. The resolved dir is used directly;
    no parent-dir parsing.
    """
    global _run_name, _run_dir, _vault_dir

    # Resume by id resolves to a dir; from there we fall through to the
    # resume-by-dir branch so both paths share one code path.
    if resume_run_id is not None and resume_run_dir is None:
        found = _find_run_dir_by_id(resume_run_id)
        if found is None:
            raise FileNotFoundError(
                f"No run dir found under {globals().get('_LOGS_ROOT')} "
                f"matching run-id {resume_run_id!r}"
            )
        resume_run_dir = found

    if resume_run_dir is not None:
        resume_run_dir = Path(resume_run_dir).resolve()
        if not resume_run_dir.exists():
            raise FileNotFoundError(f"Resume run dir not found: {resume_run_dir}")
        _run_dir = resume_run_dir
        _run_name = _run_dir.name
        # Reflect the resolved run name into os.environ so any subprocess
        # that re-reads BASEVAULT_RUN_NAME at runtime sees the same value
        # the runner is using. Without this, a stale shell-exported value
        # would silently override the resume's run name for downstream
        # reads.
        os.environ["BASEVAULT_RUN_NAME"] = _run_name
    else:
        # If caller didn't pre-set the run name, mint one in the
        # canonical '<iso-z>-<short_id>' shape. The Rust shell always
        # passes BASEVAULT_RUN_NAME; this branch covers CLI runs.
        from engine.common.utils import new_id
        _run_name = (
            os.environ.get("BASEVAULT_RUN_NAME")
            or new_id()[1]
        )
        os.environ["BASEVAULT_RUN_NAME"] = _run_name

        # Re-read _LOGS_ROOT dynamically so test monkeypatching of either
        # _LOGS_ROOT or _RUN_ROOT (legacy name) still redirects output.
        logs_root = globals().get("_LOGS_ROOT") or globals().get("_RUN_ROOT")
        _run_dir = Path(logs_root) / _run_name

    _run_dir.mkdir(parents=True, exist_ok=True)
    # Per-stage layout: each stage gets its own subdir under stages/.
    # Sub-dirs are created lazily by individual stage code; we just ensure
    # the parent exists so writes don't race on first-mkdir.
    (_run_dir / "stages").mkdir(parents=True, exist_ok=True)

    # The runner never auto-creates a vault dir. Vault export is a
    # post-completion action triggered explicitly by the export button
    # in the UI, which renders the markdown JS-side via the
    # `regenVault()` orchestrator and writes through the
    # `write_run_vault` Tauri command.
    _vault_dir = None


# ── In-memory run state ─────────────────────────────────────────────────────
#
# Issue #165 retired the per-tick `run.json` write. The runner keeps an
# in-memory `_run_state` dict so legacy code paths that read it (warning
# stamps, the rollup materializer's input fields) keep working, but the
# file on disk is split into:
#
#   config.json       — written ONCE at run start by `_write_run_config_once`.
#                       Contains only static fields (mode, models, inputs,
#                       routing, version stamps). Never touched again.
#   llm-calls.jsonl   — append-only event log. cycle_start / cycle_end /
#                       cycle_cancelled / cycle_error events surface
#                       status; begin/end events surface progress.
#                       Tauri-side `derive_run_state` walks this to produce
#                       the live status / progress / duration fields the UI
#                       used to read from run.json.
#                       The cycle_start event payload also carries the
#                       runner pid — orphan recovery + cancel paths read
#                       it via `derive_run_state.pid` (replaces
#                       run.json.pid in the post-#165 layout).
#   paused.flag       — Tauri-written marker for paused state.
#
# Old runs that pre-date the split keep run.json on disk; the Tauri-side
# read path falls back to it for the static fields, and the derivation
# falls back to its status field when the jsonl has no cycle markers.

_run_state: dict | None = None
_run_state_lock = threading.Lock()
_run_started_monotonic: float | None = None


def _init_run_json(
    paths: list[str],
    mode_str: str,
    resume: bool,
    run_config: dict | None = None,
) -> None:
    """Initialize the in-memory run-state dict. On resume, load the
    prior run's config.json (or run.json fallback) so static fields
    survive across cycles. Despite the historical name, this no longer
    writes to run.json — the disk write of static fields happens via
    `_write_run_config_once()` once provider+model are picked."""
    global _run_state, _run_started_monotonic
    _run_started_monotonic = time.monotonic()

    existing: dict | None = None
    if resume and _run_dir is not None:
        for fname in ("config.json", "run.json"):
            p = _run_dir / fname
            if not p.exists():
                continue
            try:
                existing = json.loads(p.read_text(encoding="utf-8"))
                break
            except (json.JSONDecodeError, ValueError):
                continue

    with _run_state_lock:
        if existing:
            _run_state = dict(existing)
            _run_state["status"] = "running"
            _run_state["error"] = None
            _run_state["resumed_at"] = _iso_z_full()
            _run_state["pid"] = os.getpid()
        else:
            _run_state = {
                "run_id": _run_name,
                "short_id": _short_id_from_run_name(_run_name),
                "agent": _agent(),
                "created_at": _iso_z_full(),
                "updated_at": _iso_z_full(),
                "inputs": list(paths),
                "input_count": len(paths),
                "mode": mode_str,
                "status": "running",
                "progress": {"stage": "init", "completed": 0, "total": 0},
                "vault_dir": str(_vault_dir) if _vault_dir else None,
                "run_dir": str(_run_dir) if _run_dir else None,
                "duration_ms": None,
                "error": None,
                "pid": os.getpid(),
            }
            if run_config is not None:
                _run_state["run_config"] = run_config
    # pid lives only on the cycle_start jsonl event (emitted from
    # run() once `set_calls_jsonl_path` has the run dir). Rust's
    # orphan-recovery + stop_inflight_pipeline read it via
    # `derive_run_state.pid`. There used to be a parallel pid.txt
    # sidecar here as a "backup" but nothing read it — duplicate
    # state confused readers more than it helped.


def _set_run_subject_resolution(record: dict | None) -> None:
    """Record the entities-stage subject resolution. Pre-#165 this was
    stamped onto run.json's run_config; post-#165 it's emitted as an
    `entities_decision` jsonl event so the Tauri-side derivation can
    surface it on running runs (the modal renders it before the run
    finishes). The in-memory dict keeps the field too so the legacy
    rollup writer's per-run snapshot still includes it."""
    if _run_state is not None:
        with _run_state_lock:
            cfg = _run_state.setdefault("run_config", {})
            cfg["subject_resolution"] = record
    try:
        from engine.llm import emit_cycle_event
        emit_cycle_event("entities_decision", {
            "subject_resolution": record,
        })
    except Exception:
        pass


def _set_run_bundle_mode(is_bundle: bool) -> None:
    """Record the entities-stage bundle-mode decision. Same dual write
    as `_set_run_subject_resolution`: in-memory + `entities_decision`
    jsonl event."""
    if _run_state is not None:
        with _run_state_lock:
            cfg = _run_state.setdefault("run_config", {})
            cfg["bundle_mode"] = bool(is_bundle)
    try:
        from engine.llm import emit_cycle_event
        emit_cycle_event("entities_decision", {
            "bundle_mode": bool(is_bundle),
        })
    except Exception:
        pass


def _flush_run_json() -> None:
    """No-op since issue #165. Mid-run state lives in llm-calls.jsonl;
    the static config.json is opened ONCE and never re-written.
    Retained as a no-op shim so any straggling caller doesn't crash —
    callsites are progressively removed as they're audited."""
    return


# ── config.json — static snapshot, written once at run start ────────────────
#
# config.json holds every field that is FIXED at run-start time: routing,
# inputs, version stamps, identity. Status / progress / duration / error
# are derivable from llm-calls.jsonl and live there exclusively (consumers
# walk the jsonl + filesystem markers; see Rust-side `derive_run_state`).
# Splitting this from run.json means a partial mid-run write can never
# corrupt the start-of-run record — config.json is opened ONCE and never
# touched again.
#
# Field set is the contract; bumping it is a schema break for old readers.
# Anything dynamic (status, progress, etc.) MUST stay out — that's the
# whole point of the split.
_CONFIG_JSON_FIELDS: frozenset[str] = frozenset({
    "run_id", "short_id", "agent",
    "created_at", "inputs", "input_count", "mode",
    "vault_dir", "run_dir", "run_config", "provider", "model",
})


def _build_run_config_snapshot_for_disk() -> dict:
    """Project `_run_state` down to the static `_CONFIG_JSON_FIELDS`.
    Centralized so the field set stays the contract — adding a field
    requires updating the frozenset above, not finding every writer."""
    if _run_state is None:
        return {}
    return {
        k: _run_state[k]
        for k in _CONFIG_JSON_FIELDS
        if k in _run_state
    }


def _write_run_config_once() -> None:
    """Atomically write `<run_dir>/config.json` with the static snapshot.

    Idempotent: if config.json already exists (resume case — the file
    was written on the first cycle), skip. This preserves the original
    time-zero capture even across cycles. Old runs that pre-date this
    file get a fresh config.json on their next resume cycle.

    Atomic via tmp+rename so a SIGKILL mid-write can't leave a torn
    file. `_run_state_lock` held to serialize against any concurrent
    legacy `_flush_run_json` call (defensive — the static set doesn't
    change but the lock keeps the snapshot dict coherent)."""
    if _run_state is None or _run_dir is None:
        return
    with _run_state_lock:
        path = _run_dir / "config.json"
        if path.exists():
            return
        snapshot = _build_run_config_snapshot_for_disk()
        tmp = _run_dir / "config.json.tmp"
        tmp.write_text(
            json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)


def _update_run_progress(
    stage: str,
    total: int,
    *,
    in_flight_calls: int | None = None,
    eta_seconds: float | None = None,
    stage_eta_seconds: float | None = None,
    elapsed_in_stage: float | None = None,
) -> None:
    """Tick update — called from _emit() after each LLM call.

    `total` is the pipeline-cumulative est_calls sum across all
    registered stages. `in_flight_calls` is the currently-in-flight
    count (begin without end), surfaced in the run header as
    "X calls in progress". ETA fields drive the elapsed/remaining
    line; absent values are dropped from the payload so older runs
    (or test harnesses without the tracker) don't surface stale
    fields.

    Issue #209 cleanup: `completed` and `bar_position` are NO LONGER
    written here. The UI's "completed" comes from the rust derive's
    leaf count (one numerator, one source of truth); `bar_position`
    is dead — JSX computes pct from displayDone/total directly. Don't
    add them back without revisiting the entire bar/chip pipeline."""
    if _run_state is None:
        return
    progress: dict = {
        "stage": stage,
        "total": total,
    }
    if in_flight_calls is not None:
        progress["in_flight_calls"] = in_flight_calls
    if eta_seconds is not None:
        progress["eta_seconds"] = eta_seconds
    if stage_eta_seconds is not None:
        progress["stage_eta_seconds"] = stage_eta_seconds
    if elapsed_in_stage is not None:
        progress["elapsed_in_stage"] = elapsed_in_stage
    with _run_state_lock:
        _run_state["progress"] = progress
    _flush_run_json()


def _set_run_warnings(warnings: dict) -> None:
    """Attach a warnings dict to the in-memory run state. Post-#165 the
    rollup writer (`_do_write_llm_stats`) reads `_run_state["warnings"]`
    and includes it in llm-stats.json so the UI's runs list still
    surfaces flagged calls. Pre-#165 this also flushed to run.json."""
    if _run_state is None:
        return
    with _run_state_lock:
        _run_state["warnings"] = warnings
    _flush_run_json()


def _merge_run_warnings(extras: dict) -> None:
    """Merge `extras` into the in-memory warnings dict — preserves any
    fields the live-streaming `_set_run_warnings` already wrote (cap
    hits, empty responses, input overflows). Used by the rollup
    materializer to add outcome-derived fields (timeouts, blanked,
    success_empty) without clobbering the live-stream warnings."""
    if _run_state is None:
        return
    with _run_state_lock:
        cur = _run_state.get("warnings") or {}
        if not isinstance(cur, dict):
            cur = {}
        cur.update(extras)
        _run_state["warnings"] = cur
    _flush_run_json()


def _set_run_cache_stats(stats: dict) -> None:
    """Attach the LLM-cache hit/miss totals to the in-memory run state.
    Post-#165 these flow through to llm-stats.json (the rollup) where
    list_runs reads them. Shape is whatever `llm_cache.get_cache_stats()`
    returns: {hits, misses, stores, by_stage: {...}, bypass}."""
    if _run_state is None:
        return
    with _run_state_lock:
        _run_state["llm_cache"] = stats
    _flush_run_json()


def _emit_shareable_run_marker(run_dir: Path) -> None:
    """Best-effort: drop this run's content-free run/corpus diagnostics
    into ``~/.basevault/shareable/run-diagnostics/`` so a completed run
    is captured RUN-driven, independent of whether any chat ever binds
    to it. Goes through the single guarded ``shareable.emit`` — the
    same typed Marker + pre-serialization content-free guard the chat
    path uses, and idempotent for the RUN stream (if a chat session
    bound to this run already wrote the file this is a no-op; whoever
    reaches it first wins). A failure here must never fail an
    otherwise-complete run; nothing is written on a content-free
    violation (the guard crashes before any byte), so a swallowed
    error is safe, not a leak."""
    try:
        from engine import shareable
        from engine import shareable_markers
        from engine.rag_vector_store import open_store

        perma_id = shareable.resolve_perma_id(run_dir)
        if perma_id is None:
            return
        store_path = run_dir / "stages" / "06-embeddings" / "vectors.db"
        store_cm = open_store(store_path) if store_path.is_file() else None
        try:
            store = store_cm.__enter__() if store_cm is not None else None
            marker = shareable_markers.build_run_marker(
                store=store, run_dir=run_dir,
            )
        finally:
            if store_cm is not None:
                store_cm.__exit__(None, None, None)
        shareable.emit(shareable.Stream.RUN, perma_id, marker)
    except Exception as _e:  # noqa: BLE001 — diagnostics never fail a run
        try:
            _log_write(
                f"  shareable run-marker skipped: {type(_e).__name__}"
            )
        except Exception:
            pass


def _mark_run_terminal(status: str, error: str | None = None) -> None:
    """Record a terminal status. Post-#165 the on-disk record lives in
    llm-calls.jsonl as a cycle event:
      "completed" → cycle_end (already emitted by `_emit_cycle_end_once`,
                    so this is a no-op for the on-disk record; we just
                    update the in-memory mirror).
      "failed"    → cycle_error{message}
      "paused"    → no jsonl event (Rust writes paused.flag); update
                    the in-memory mirror only.
    The in-memory `_run_state` mirror keeps these fields so legacy
    callsites (the rollup's `_log_write` summary, atexit-stamped
    fields) keep working until they're individually retired."""
    if _run_state is not None:
        with _run_state_lock:
            _run_state["status"] = status
            _run_state["error"] = error
            if _run_started_monotonic is not None:
                _run_state["duration_ms"] = int(
                    (time.monotonic() - _run_started_monotonic) * 1000
                )
    if status == "failed":
        try:
            from engine.llm import emit_cycle_event
            emit_cycle_event("cycle_error", {
                "message": error or "",
            })
        except Exception:
            pass
    # Run-driven shareable diagnostics: drop the content-free run/corpus
    # file on every terminal/wind-down outcome the in-process runner
    # reaches — completed, failed, and the no-checkpoint paused. Every run
    # ending is worth a diagnostic; the RUN shareable is now latest-wins
    # (overwrite), so a partial paused snapshot is simply replaced by the
    # eventual resume→complete one. The GUI-driven pause/cancel and the
    # app-close paths terminate the sidecar from Rust (SIGTERM/SIGKILL)
    # without reaching here, so those fire the same emit from the Rust
    # side.
    if status in ("completed", "failed", "paused") and _run_dir is not None:
        _emit_shareable_run_marker(_run_dir)


def _init_log(resume: bool = False) -> Path:
    """Open the run log. Append on resume, overwrite otherwise."""
    global _log_file
    path = _run_dir / "run.log"
    _log_file = open(path, "a" if resume else "w", encoding="utf-8")
    _log_write(f"Pipeline run started — log: {path}")
    _log_write(f"Run:     {_run_name}")
    _log_write(f"Agent:   {_agent()}")
    _short_id = (_run_state or {}).get("short_id")
    if _short_id:
        _log_write(f"ID:      {_short_id}")
    if resume:
        _log_write(f"Resume mode: run dir {_run_dir}")
    # Cross-reference into the session's app.log so a reader scanning
    # session events can jump to this run's per-run logs.
    try:
        from engine.common.session import session_log
        session_log(
            f"pipeline run {'resumed' if resume else 'started'}: "
            f"{_run_name}"
        )
    except Exception:
        pass
    return path


def _log_write(msg: str):
    if _log_file is None or _log_file.closed:
        # Closed-file path matters for the atexit-fired stats dump:
        # `run()`'s normal exit closes the log file, but atexit can
        # still fire later (e.g. after pytest's process winds down,
        # or after a SIGTERM that bypasses run()'s cleanup). Without
        # this guard, the atexit handler raises ValueError on a
        # write to a closed handle and Python prints
        # "Exception ignored in atexit callback" to stderr, which
        # looks like a regression but is actually best-effort
        # logging hitting a closed file.
        return
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}\n"
    with _log_lock:
        _log_file.write(line)
        # Flush every write. The batching-every-N was a premature
        # optimization — LLM-call cadence makes fflush essentially free
        # here, and buffering made `tail -f run.log` unusable for
        # diagnosing slow stages (user sees nothing until 20 writes
        # have accumulated, which can take the whole stage).
        _log_file.flush()


        _log_file.flush()


def _log_stage(stage: str):
    _log_write(f"── STAGE: {stage} ──────────────────────────────────")
    # Tag any LLM calls that happen in this stage so tokens.json groups
    # usage by stage. Safe because stages are sequential within a run.
    try:
        from engine.llm import set_stage
        set_stage(stage)
    except ImportError:
        pass  # llm not importable in some test contexts
    # Wall-clock anchor for the ETA tracker. Skips non-LLM stages
    # (ingest/split/vault_export) because they aren't registered with
    # the tracker — _record_stage_started is a no-op on unregistered
    # stages other than closing any prior in-flight stage.
    _record_stage_started(stage)
    # Emit on stage transitions so the bar updates between LLM-call
    # boundaries. Without this, the inter-stage gap (writing entities.json,
    # loading patterns from disk, vault export, etc.) leaves the UI on
    # the prior stage's last emission until the new stage's first LLM
    # call lands — making the bar appear to "jump" at the next emission
    # despite the math being smooth across the gap.
    _emit()


# ── Progress tracking via monkey-patch ────────────────────────────────────────

_lock = threading.Lock()
_completed = 0
_stage = "init"
_announced_total = 0  # most recent total emitted to the UI
# In-flight call count = begins minus ends. Used by the UI's "X calls
# in progress" header — the round-3 brief explicitly requested
# CURRENTLY-IN-FLIGHT (not cumulative). Incremented in the wrapper at
# attempt start, decremented at attempt end (success OR final failure).
_in_flight_calls = 0

# Pipeline-wide ETA tracker. Created in run() so each invocation starts
# with a fresh historical-coefficient snapshot loaded from prior runs'
# llm-calls.jsonl files. None outside a run (tests, ad-hoc scripts).
from engine.progress import ProgressTracker  # noqa: E402
_progress_tracker: "ProgressTracker | None" = None
# Resolved real backend model id per stage for the active run. Filled
# alongside register_stage so duration-recording code can key on the
# same (stage, model) pair the historical lookup uses. Mode-pinned for
# non-mix runs; per-stage for TEE-mix.
_stage_model_map: dict[str, str] = {}


def _emit_total(total: int):
    """Emit an `estimate_updated` event so the UI's denominator widens
    when the tracker grows (halve cascades, mid-run re-estimates).

    Issue #209 cleanup: payload no longer carries `bar_position` or
    `completed`. The UI's `completed` is rust's leaf count from the
    jsonl; the bar's pct is `displayDone / total` in the JSX. ETA is
    still useful (drives the "X remaining" line), so it stays."""
    global _announced_total
    _announced_total = total
    payload = {"stage": "estimate_updated", "total": total}
    if _progress_tracker is not None:
        snap = _progress_tracker.snapshot()
        payload["eta_seconds"] = snap["eta_seconds"]
    try:
        print(json.dumps(payload), flush=True)
    except BrokenPipeError:
        pass


def _emit():
    """Emit a progress event after an LLM call (or at a stage boundary).

    Payload shape (post-#209):

        stage              user-visible stage label
        total              pipeline-cumulative est_calls sum (denominator)
        in_flight_calls    begin minus end across the whole pipeline
        eta_seconds        total pipeline remaining seconds (UI elapsed line)
        stage_eta_seconds  current stage remaining seconds
        elapsed_in_stage   wall-clock since stage started

    NOT emitted (deliberate):
      - `completed`: the UI reads rust's leaf count from the jsonl. The
        python tracker's `completed_calls` is an internal ETA detail
        (counts successful wrapper attempts, not leaves) — emitting it
        as a UI field would re-create the duplicate-numerator bug
        (#209) where the chip and bar showed different fractions.
      - `bar_position`: dead. JSX computes pct from displayDone/total.
        See `progressChipPct` in App.jsx for the single source of the
        bar's percentage formula.

    Falls back to a minimal stage-label-only emission when the tracker
    isn't initialized (tests, ad-hoc scripts that skip run()).
    """
    global _announced_total
    if _progress_tracker is None:
        # Legacy path — keep the stage label moving when the tracker
        # hasn't been wired (test harnesses, direct `_emit()` calls).
        # No completed/total to surface; just emit the stage so the UI
        # at least labels things correctly.
        try:
            print(json.dumps({"stage": _stage}), flush=True)
        except BrokenPipeError:
            pass
        _update_run_progress(_stage, _announced_total)
        return

    snap = _progress_tracker.snapshot()
    total = snap["total_calls_pipeline"]
    in_flight = _in_flight_calls
    # Keep _announced_total in sync — anything that grew past the last
    # announced total triggers an estimate_updated emission so the
    # frontend's denominator widens with the work.
    if total > _announced_total:
        _emit_total(total)
    # snapshot()'s `current` picks the latest in-flight stage, which
    # for entities_dedupe-inside-entities is the SUB-stage, not the
    # user-visible one. The runner's `_stage`
    # global is the source of truth for the visible label, so derive
    # the per-stage elapsed + ETA from `_stage` directly.
    visible_stage = _stage
    visible_stage_state = _progress_tracker._stages.get(visible_stage)
    if visible_stage_state is not None:
        visible_stage_elapsed = visible_stage_state.elapsed()
    else:
        visible_stage_elapsed = 0.0
    visible_stage_eta = (
        _progress_tracker.estimate_stage_seconds(visible_stage)
        if visible_stage_state is not None else 0.0
    )
    payload = {
        "stage": visible_stage,
        "total": total,
        "in_flight_calls": in_flight,
        "eta_seconds": snap["eta_seconds"],
        "stage_eta_seconds": visible_stage_eta,
        "elapsed_in_stage": visible_stage_elapsed,
    }
    try:
        print(json.dumps(payload), flush=True)
    except BrokenPipeError:
        pass
    # Mirror the same payload as a `progress_tick` event in
    # llm-calls.jsonl. Pre-#165 the runs-list polled run.json for these
    # fields; post-#165 derivation only sees the jsonl, so without this
    # event the bar's denominator + ETA freezes between LLM-call
    # boundaries on single-call stages (insights / actions).
    try:
        from engine.llm import emit_cycle_event
        emit_cycle_event("progress_tick", payload)
    except Exception:
        pass
    # Update in-memory progress mirror. Post-#165 the disk record is
    # the begin/end events streamed to llm-calls.jsonl (Tauri-side
    # derive_run_state walks it); the in-memory mirror is kept for
    # legacy callsites that still read _run_state["progress"].
    _update_run_progress(
        _stage, total,
        in_flight_calls=in_flight,
        eta_seconds=snap["eta_seconds"],
        stage_eta_seconds=visible_stage_eta,
        elapsed_in_stage=visible_stage_elapsed,
    )


def _record_stage_started(stage: str) -> None:
    """Tracker-aware: mark prior in-flight stage finished and the new
    one started. Safe to call when `_progress_tracker` is None."""
    if _progress_tracker is None:
        return
    # Close any other stage that was still open. There's only ever one
    # in-flight stage at a time in the linear pipeline; this loop is
    # a defensive idempotent close.
    for s, st in list(_progress_tracker._stages.items()):
        if s != stage and st.started_at is not None and st.finished_at is None:
            _progress_tracker.mark_stage_finished(s)
    _progress_tracker.mark_stage_started(stage)


def _record_call_duration(
    stage: str,
    duration_seconds: float,
    completion_tokens: int = 0,
) -> None:
    """Legacy hook — used by callers that don't track begin/end.
    Feeds live (duration, completion_tokens) to the tracker so both
    sec/call and sec/token coefficients refine."""
    if _progress_tracker is None:
        return
    _progress_tracker.record_call(stage, duration_seconds, completion_tokens)


def _record_call_begin(stage: str):
    """Wrapper-side hook called at the START of an outer LLM call
    (across the retry chain, not per-attempt). Returns a token the
    wrapper passes to `_record_call_end` on success or final failure
    so the in-flight set stays accurate.

    Returns None when there's no tracker — the caller can pass that
    None back to `_record_call_end` and it's a no-op."""
    if _progress_tracker is None:
        return None
    return _progress_tracker.record_call_begin(stage)


def _record_call_end(
    stage: str,
    token,
    duration_seconds: float,
    completion_tokens: int = 0,
    success: bool = True,
) -> None:
    """Pair `_record_call_begin`'s token with the call's outcome.
    `success=False` removes the in-flight entry without crediting a
    completed call (matches runner's `_completed` semantics)."""
    if _progress_tracker is None:
        return
    _progress_tracker.record_call_end(
        stage, token, duration_seconds, completion_tokens,
        success=success)


class _KernelLiveProgressHook:
    """Bridges the kernel call lifecycle to the progress tracker.

    The KernelTelemetryHook writes per-call ``llm-calls.jsonl`` records, but
    the progress-event stream the runs-list bar reads is fed here. Two things:

    * the run-details LIVE panel + per-call wait timer read ``in_flight_calls``
      off that stream — without a bump the panel sits frozen at 0 while a
      kernel stage runs ("not updating"); and

    * the tracker's per-stage ``completed_calls`` — the count
      ``mark_stage_finished`` snaps each finished stage's ``est_calls`` down to
      so the cumulative denominator equals "real work done so far + estimate
      for what's left". The legacy ``complete()`` wrapper bumped this in real
      time on every success; the kernel path bypasses ``complete()``, so when
      stage execution moved onto the kernel this credit was dropped. Every
      finished stage then snapped to ``completed_calls == 0`` and fell out of
      the denominator, so ``total`` decayed to bare remaining work — the chip
      read "99% (N / N)" and reset each stage (22/22 → 11/11). Restoring the
      credit here keeps the denominator a true cumulative total. The legacy
      wrapper is dead, so there is a single crediting path and no double-count.
    """

    def hook_llm_queued(self, call, execution_env) -> None:
        pass

    def hook_llm_started(self, call, execution_env) -> None:
        global _in_flight_calls
        with _lock:
            _in_flight_calls += 1
        _emit()

    def hook_llm_completed(
        self, call, execution_env, response, retry, from_cache, should_cache
    ) -> None:
        # Credit a successful leaf to the stage's completed_calls (incl. cache
        # hits — a served leaf is completed work; failures don't count). This
        # is what makes mark_stage_finished freeze a finished stage at its real
        # call count instead of 0. Mirrors the legacy wrapper's success-only
        # bump; the failure set is the telemetry hook's, kept as one source.
        if _progress_tracker is not None:
            from engine.phases.telemetry_hook import (
                _FAILURE_STATUSES,
                stage_label,
            )
            if (
                response.exception is None
                and response.status not in _FAILURE_STATUSES
            ):
                _progress_tracker.record_call(
                    stage_label(execution_env.phase.name()),
                    response.duration or 0.0,
                    response.completion_tokens or 0,
                )
        # Cache hits skip the scheduler, so hook_llm_started never fired and
        # in_flight was never bumped — don't decrement the live panel for them.
        if from_cache:
            return
        global _in_flight_calls
        with _lock:
            _in_flight_calls = max(0, _in_flight_calls - 1)
        _emit()


_KERNEL_LIVE_HOOK = _KernelLiveProgressHook()


def _bump_stage_est(stage: str, delta: int) -> None:
    """Issue #105 v3 follow-up: extract halving fans 1 parent into 2
    sub-calls. Each halving event should bump the stage's est_calls
    by +1 so the progress denominator tracks reality. Without this,
    halve-heavy runs overshoot 100% on the bar (`completed > total`).

    No-op when the tracker is None (tests, scripts) or the stage
    isn't registered. Re-emits the cumulative total after the bump
    so the front-end picks up the new denominator immediately."""
    if _progress_tracker is None:
        return
    _progress_tracker.bump_est_calls(stage, delta)
    # Re-emit the pipeline total so the bar's denominator updates
    # live without waiting for the next progress tick.
    new_total = _progress_tracker.compute_pipeline_total_calls()
    _emit_total(new_total)


def _pick_run_model_id(
    *,
    mode,
    unique_models: "list[str]",
    spec_model_id: str,
) -> str:
    """Pick the run-row label that honestly describes what will run.

    Priority:
      1. TEE map that collapses to one model → that single model id.
      2. TEE map with multiple distinct models (per-stage routing) →
         the "per-stage" sentinel, rendered as "per-stage models TEE"
         by the UI.
      3. Empty stage map / non-TEE mode → the mode-anchor spec.
    """
    from engine.llm import Mode as _Mode
    if mode == _Mode.TEE:
        if len(unique_models) == 1:
            return unique_models[0]
        if len(unique_models) > 1:
            return "per-stage"
    return spec_model_id


def _resolve_stage_model_map_for_run(mode, spec, raw: bool = False) -> dict[str, str]:
    """Per-stage model id for the current run.

    `raw=False` (default) returns the resolved real-backend id (a
    multi-model sentinel collapses to its first constituent) — what the
    tracker/ETA (stage, model) history keys on. `raw=True` returns the
    configured id verbatim (sentinel preserved) for display/record
    snapshots so Run details shows e.g. "kimi+glm", not "kimi-k2-6".
    The verbatim configured map is honored only on TEE (the one mode
    whose `_STAGE_MODEL_MAP` routing is live); LOCAL/TEST snapshots fall
    through to the mode-pinned spec so Run details records the local /
    test model that actually ran, not the stale configured cloud ids.

    Mirrors `llm._resolve_stage_override` for every pipeline stage so
    the tracker's historical lookup keys on the same (stage, model)
    pairs the wrapper actually records under. Mode-pinned for plain
    runs; per-stage for TEE-mix.

    Returns `{stage: model_id}` for every stage in
    `progress.PIPELINE_STAGES`. Vision is the one stage whose model
    isn't routed through `_STAGE_MODEL_MAP` — it has its own per-mode
    table (`vision._VISION_MODEL`) — so we read it directly. Falling back
    to the chat anchor would give the historical-median lookup a key
    the wrapper never writes against (vision records hit the configured
    vision model, e.g. `(vision, kimi-k2-6)`, not the chat anchor).
    """
    from engine.llm import (
        Mode as _Mode,
        _resolve_stage_override,
        _STAGE_MODEL_MAP,
        stage_model_id as _stage_model_id,
    )
    from engine.vision import _VISION_MODEL
    from engine.progress import PIPELINE_STAGES
    out: dict[str, str] = {}
    for stage in PIPELINE_STAGES:
        if stage == "vision":
            # Vision priority: user's per-stage Settings pick (config's
            # stage_models.vision) on TEE, falling back to the mode-
            # pinned ship default. Mirror chat-stage routing's TEE-only
            # gate (`_resolve_stage_override` only honors stage_models
            # on Mode.TEE today). TEST still uses _VISION_MODEL.
            cfg = (_STAGE_MODEL_MAP or {}).get("vision") or {}
            cfg_model = (cfg.get("model") or "").strip()
            if mode == _Mode.TEE and cfg_model:
                out[stage] = cfg_model
            elif raw and mode == _Mode.LOCAL:
                # Snapshot/record path in local mode: pin vision to the
                # local primary so Run details shows the local backend
                # for EVERY stage. _VISION_MODEL[LOCAL] is a stale, non-
                # local id no local run actually serves; recording it
                # would surface a cloud-looking model on a run that
                # touched only the local backend (a provenance lie).
                # raw=False keeps _VISION_MODEL so the tracker's
                # (stage, model) ETA lookup still keys on the id the
                # wrapper records vision calls under.
                out[stage] = spec.model_id
            else:
                out[stage] = _VISION_MODEL.get(mode, spec.model_id)
            continue
        if raw and mode == _Mode.TEE:
            # Display/record snapshot: keep the configured model id
            # verbatim, preserving a multi-model sentinel (e.g.
            # "kimi+glm") so Run details + run.log + config.json reflect
            # the parallel-dispatch config instead of collapsing to the
            # first constituent. Falls through to the resolved id when a
            # stage isn't explicitly configured. TEE-only gate: the
            # configured stage_models map is honored solely on TEE
            # (mirrors `_resolve_stage_override`), so LOCAL/TEST fall
            # through to the mode-pinned spec below — otherwise a local
            # run would record the configured cloud per-stage ids it
            # never dispatched.
            raw_id = _stage_model_id(stage)
            if raw_id:
                out[stage] = raw_id
                continue
        try:
            _eff_spec, override = _resolve_stage_override(mode, stage)
            out[stage] = override or _eff_spec.model_id
        except Exception:
            out[stage] = spec.model_id
    return out


def _resolve_stage_reasoning_map_for_run(mode) -> dict[str, bool]:
    """Per-stage reasoning state for the current run, computed from the
    same gates `llm.complete()` applies at call time:
    `_REASONING_WHITELIST` ∧ user toggle in `_STAGE_MODEL_MAP`.

    Snapshotting at run start lets config.json record what the run will
    actually do without scraping kwargs out of every call.
    """
    from engine.llm import _resolve_stage_override, _reasoning_enabled_for
    from engine.progress import PIPELINE_STAGES
    out: dict[str, bool] = {}
    for stage in PIPELINE_STAGES:
        try:
            spec, _override = _resolve_stage_override(mode, stage)
            out[stage] = bool(_reasoning_enabled_for(spec, stage))
        except Exception:
            out[stage] = False
    return out


def _file_metadata(paths: list[str]) -> list[dict]:
    """Compact per-input descriptor for run_config.inputs.

    Each entry: {path, size_bytes, sha256}. Hashing is done once at run
    start; for typical inputs (<10 MB) this adds <100ms total. Failures
    (file missing, unreadable) yield a partial entry with the failure
    note instead of raising — the snapshot is best-effort metadata,
    not a gating check.
    """
    import hashlib
    out: list[dict] = []
    for raw in paths:
        entry: dict = {"path": raw}
        try:
            p = Path(raw)
            stat = p.stat()
            entry["size_bytes"] = int(stat.st_size)
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            entry["sha256"] = h.hexdigest()
        except OSError as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        out.append(entry)
    return out


def _build_run_config_snapshot(
    *,
    mode,
    spec,
    paths: list[str],
    sentiment: str,
) -> dict:
    """Build the full config snapshot persisted to config.json at run start.

    Captures every user-controllable knob that materially shapes the
    run's outputs so post-hoc analysis can answer "what was set when
    this fired" without consulting Settings (which may have changed)
    or the user's memory.
    """
    from engine.llm import _STAGE_MODEL_MAP, _read_app_config
    from engine.content_extractor import _topics_for_run
    stage_models = _resolve_stage_model_map_for_run(mode, spec, raw=True)
    stage_reasoning = _resolve_stage_reasoning_map_for_run(mode)
    cfg = _read_app_config()
    llm_cache_enabled = cfg.get("llm_cache_enabled")
    if llm_cache_enabled is None:
        llm_cache_enabled = True
    llm_cache_bypass_env = bool(os.environ.get("BASEVAULT_LLM_CACHE_BYPASS"))
    snapshot = {
        "mode": mode.value if hasattr(mode, "value") else str(mode),
        "provider": spec.provider.value,
        "primary_model": spec.model_id,
        "stage_models": stage_models,
        "stage_reasoning": stage_reasoning,
        # Pipeline always runs at temperature=0 today. Recorded as a
        # single value (not per-stage) since there is no per-stage knob
        # — but kept under a name that won't surprise a future-where
        # one is added.
        "temperature": 0.0,
        "sentiment": sentiment,
        "llm_cache_enabled": bool(llm_cache_enabled) and not llm_cache_bypass_env,
        "stage_models_source": (
            "stage_models" if isinstance(cfg.get("stage_models"), dict)
            else ("tee_model" if cfg.get("tee_model") else "default")
        ),
        "tee_model_setting": cfg.get("tee_model"),
        "inputs": _file_metadata(paths),
        "pipeline_git_sha": _git_sha(),
        "branch": _git_branch(),
        "app_version": os.environ.get("BASEVAULT_APP_VERSION") or None,
        # Topic taxonomy as of run start. The snapshot itself is
        # frozen here so historical runs stay interpretable, but the
        # runtime read in content_extractor (`_topics_for_run`) is
        # live — edits to Settings → Categories mid-run will affect
        # extracts that fire after the edit. Same shape as every
        # other config field consumed live by the pipeline; edits-
        # during-run are rare. If a future use case needs strict
        # per-run pinning, stash the snapshot list in `_run_state`
        # and have `_topics_for_run` prefer it during an active run.
        "categories": list(_topics_for_run()),
    }
    # Surface any bare `_STAGE_MODEL_MAP` reasoning toggles too — useful
    # when a user has reasoning=True on a non-whitelisted (model, stage)
    # tuple. The effective `stage_reasoning` masks that off, but
    # surfacing the user's intent helps explain why the toggle did not
    # take effect.
    user_toggles: dict[str, bool] = {}
    for stage, entry in (_STAGE_MODEL_MAP or {}).items():
        if isinstance(entry, dict) and "reasoning" in entry:
            user_toggles[stage] = bool(entry["reasoning"])
    if user_toggles:
        snapshot["stage_reasoning_user_toggle"] = user_toggles
    return snapshot


# ── Per-call wrapper ─────────────────────────────────────────────────────
#
# Single-attempt + bookkeeping. Classification, retry policy, and
# scheduler-mediated dispatch all live OUTSIDE this wrapper — in
# `retry._classify_failure` (the load / sizing / other classifier)
# and the scheduler-thunk helpers `scheduler.run_sample_cascade` /
# `scheduler.build_call_thunk`. The wrapper here owns only the bits
# that need access to runner-internal globals: begin/finalize stat
# record, in-flight counter, progress tracker, payload log on
# failure, post-stream-failure stat stamping. Everything else is
# the caller's thunk.


def _budget_snapshot_for(stage: str | None) -> dict:
    """Capture stage-cap / max-output / scaffolding / has_llm_calls for
    the active mode + stage. Used as the static `budget` field on every
    stat record so a debug bundle reader can sanity-check 'why was
    this prompt over-cap' without poking the live config.

    Schema (stage_cap, max_output, scaffolding) is unchanged from
    pre-#205 records; downstream consumers (llm-stats rollup, runs-list
    UI) keep working. `has_llm_calls` is additive — False for parent
    stages (ingest) where the numeric fields are 0 sentinels rather
    than real budgets."""
    fallback = {
        "stage_cap": None, "max_output": None, "scaffolding": None,
        "has_llm_calls": None,
    }
    try:
        from engine.llm import compute_budget
    except Exception:
        return fallback
    mode = _resolve_active_mode()
    if mode is None:
        return fallback
    try:
        b = compute_budget(mode, stage)
        return {
            "stage_cap": b.max_input,
            "max_output": b.max_output,
            "scaffolding": b.scaffolding,
            "has_llm_calls": b.has_llm_calls,
        }
    except Exception:
        return fallback


def _resolve_active_mode():
    """Best-effort lookup of the runner's active Mode. Module-global
    set by main(). Returns None if the runner hasn't been initialized
    (tests, scripts) — budget then defaults to all-None."""
    return _active_mode


# Set by main() / runner entry; read by _budget_snapshot_for.
_active_mode = None


# Module-level guard so `_write_llm_stats` is safe to call from
# multiple paths (success-end-of-run, atexit, future early exits).
# First call writes; subsequent calls no-op.
#
# Reset by `_reset_llm_stats_dump_state()` at the top of each run() so
# multiple `run()` calls within one Python process (only realistic in
# tests; production spawns a fresh subprocess per run) each write
# their own llm-stats.json.
_llm_stats_written: bool = False

# Cycle-end idempotency guard (issue #52). The cycle_end event must
# fire exactly once per cycle, but multiple termination paths could
# race to emit it (normal completion, atexit, SIGTERM). Guarded the
# same way as `_llm_stats_written`; reset alongside it at run entry.
_cycle_end_emitted: bool = False


def _reset_llm_stats_dump_state() -> None:
    """Reset the idempotent-write guard at run() entry. Called BEFORE
    atexit registration so the new run's atexit can write even after
    the previous run's explicit-end-of-run write tripped the guard."""
    global _llm_stats_written, _cycle_end_emitted
    _llm_stats_written = False
    _cycle_end_emitted = False


def _emit_cycle_end_once(reason: str) -> None:
    """Idempotent cycle_end emit. Safe to call from any termination
    path (normal completion, atexit handler, SIGTERM signal handler).
    First call emits the event with the given reason; subsequent calls
    no-op. The reason field surfaces WHICH path closed the cycle —
    useful when reading the jsonl post-mortem.

    Wraps emit_cycle_event() in a try/except so a jsonl write failure
    during shutdown can never block the termination path."""
    global _cycle_end_emitted
    if _cycle_end_emitted:
        return
    _cycle_end_emitted = True
    try:
        from engine.llm import emit_cycle_event
        emit_cycle_event("cycle_end", {
            "reason": reason,
        })
    except Exception:
        # Termination paths must never raise — the rollup or the
        # process exit is more important than a missing marker.
        pass


def _bootstrap_per_stage_from_jsonl(
    jsonl_path: "Path",
) -> dict[str, dict]:
    """Per-stage seed for the ProgressTracker on resume.

    Returns `{stage: {"count": int, "samples": list[(dur_s, tokens)],
    "model": str | None}}`. Counts only successful end events
    (matching the wrapper's increment rule); samples carry both
    duration and completion_tokens so the resumed tracker has full
    sec/call + sec/token live data from the prior session.
    Empty dict for a fresh run.

    Dedupes by `cache_key` (per-stage): when cycle 2 resumes mid-
    stage it re-emits begin/end pairs for cycle 1's already-
    successful calls (rerun short-circuits via the LLM cache — second
    end shares the cache_key, cached=true). Two ends with the same
    cache_key represent ONE pipeline work unit; counting both
    inflates the per-stage seed by the size of the prior cycle's
    successful set every restart, which double-counts via
    `register_stage`'s `max(est, completed_calls)` clamp and clamps
    the bar at 'N/N' until the run ends. Each distinct cache_key
    contributes one sample. Empty / missing cache_key counts
    individually — back-compat with older runs.
    """
    if not jsonl_path.exists():
        return {}
    try:
        text = jsonl_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    begins: dict[str, dict] = {}
    out: dict[str, dict] = {}
    seen_cache_keys_per_stage: dict[str, set] = {}
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except Exception:
            continue
        evt = ev.get("event")
        cid = ev.get("call_id")
        if not isinstance(cid, str):
            continue
        if evt == "begin":
            begins[cid] = ev
        elif evt == "end":
            b = begins.pop(cid, None)
            if b is None or not ev.get("success"):
                continue
            stage = b.get("stage")
            model = b.get("model")
            dur_ms = ev.get("duration_ms")
            ct = ev.get("completion_tokens") or 0
            if not isinstance(ct, (int, float)) or ct < 0:
                ct = 0
            if not stage:
                continue
            cache_key = ev.get("cache_key")
            seen = seen_cache_keys_per_stage.setdefault(stage, set())
            if isinstance(cache_key, str) and cache_key:
                if cache_key in seen:
                    continue
                seen.add(cache_key)
            slot = out.setdefault(
                stage, {"count": 0, "samples": [], "model": model})
            slot["count"] += 1
            if isinstance(dur_ms, (int, float)) and dur_ms > 0:
                slot["samples"].append((dur_ms / 1000.0, int(ct)))
            if model and not slot.get("model"):
                slot["model"] = model
    return out


def _bootstrap_completed_from_jsonl(jsonl_path: "Path") -> int:
    """Count successful end events on disk from prior session(s).

    On resume, the wrapper's `_completed` counter (incremented on each
    successful LLM call) starts at 0 in the new Python process. The
    progress bar's total estimate, however, accounts for the entire
    pipeline. Without this seed, a resumed run renders the bar from
    0/total even though work is already on disk — visually undershoots
    until the bar catches up via remaining-work ticks.

    Reading the same llm-calls.jsonl that survives across sessions and
    counting `event:"end" success:true` events gives a stable seed:
    every retry that ultimately succeeded counts once, every aborted
    or failed attempt counts zero, exactly matching the wrapper's
    increment rule on a fresh run. Returns 0 when the file is missing
    or empty (non-resume runs)."""
    if not jsonl_path.exists():
        return 0
    n = 0
    try:
        text = jsonl_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("event") == "end" and ev.get("success") is True:
            n += 1
    return n


def _materialize_calls_from_jsonl(
    jsonl_path: "Path", ended_at_iso: str
) -> list[dict]:
    """Collapse the begin/end/counts event log into per-call records.

    A `begin` opens a record. An `end` populates duration / success /
    error / tokens. A `counts` attaches input/output. A begin without
    a matching end is the smoking gun for "this call was in flight
    when the run wound down" — the materialized record gets
    `aborted: true`, `success: False`, and an estimated `duration_ms`
    (now − started_at). No synthetic error is stamped: the call itself
    didn't fail, the run wound down. Downstream `_classify_outcome`
    surfaces this distinct from any `failed (X)` bucket as
    `OUTCOME_ABORTED`, matching the Rust live materializer.

    The .jsonl is canonical: this function does NOT consult
    `_stats_records` (the in-memory list) so the rollup is
    reproducible from disk alone (e.g. running this offline against
    a recovered .jsonl from a SIGKILL'd run)."""
    if not jsonl_path.exists():
        return []
    by_id: dict[str, dict] = {}
    order: list[str] = []
    for raw in jsonl_path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except Exception:
            continue
        cid = ev.get("call_id")
        if cid is None:
            continue
        kind = ev.get("event")
        if kind == "begin":
            if cid in by_id:
                continue  # ignore duplicate begin
            by_id[cid] = {
                "call_id": cid,
                "stage": ev.get("stage"),
                "category": ev.get("category"),
                "model": ev.get("model"),
                "mode": None,
                "started_at_iso": ev.get("started_at_iso"),
                "started_at_human": None,
                "duration_ms": None,
                "success": None,
                "error": None,
                # Seed `prompt_tokens` from the begin-event's pre-flight
                # estimate (issue #225). The end event's value overwrites
                # ONLY when truthy — a 0 from a wire-cut response leaves
                # the estimate intact so the per-call detail UI surfaces
                # a real number for the failed attempt. Pre-this-fix
                # jsonl files without the field read None here, identical
                # to the old behavior.
                "prompt_tokens": ev.get("prompt_tokens_est"),
                "completion_tokens": None,
                # Streaming observability fields (issue #104 part 1).
                # All None until the end event populates them; absent on
                # pre-this-PR jsonl files so consumers reading older
                # logs get a uniform shape with explicit "no data".
                "reasoning_tokens": None,
                # `reasoning_tokens_source` is "api" / "streamed" /
                # "estimated" / None — distinguishes provider-reported
                # counts from delta-math estimates so consumers can
                # surface confidence (Tinfoil today is always
                # "estimated" because it hides reasoning server-side).
                "reasoning_tokens_source": None,
                "content_tokens": None,
                "finish_reason": None,
                "ttft_ms": None,
                "ttfr_ms": None,
                "last_token_ms": None,
                # Actual `max_tokens` reservation passed to the API for
                # this call (Ollama: num_predict). Distinct from the
                # static `budget.max_output` per-stage ceiling: budget
                # is the cap, max_tokens_reserved is what the call
                # actually asked for after `dynamic_max_tokens()`. The
                # runner wrapper stamps the value on the begin event so
                # in-flight calls surface it; the end event overwrites
                # with the same value at completion. Cache hits leave it
                # None ("not applicable" — no reservation went on the
                # wire); pre-this-PR jsonl files leave None too. Either
                # way, `_agg_dist` filters None so the field's
                # distribution reflects only real reservations.
                "max_tokens_reserved": ev.get("max_tokens_reserved"),
                "budget": ev.get("budget") or {
                    "stage_cap": None, "max_output": None, "scaffolding": None,
                    "has_llm_calls": None,
                },
                "input": None,
                "output": None,
                "attempt": ev.get("attempt"),
                "retry_of_call_id": ev.get("retry_of_call_id"),
                # `llm_status` is the kernel's categorized outcome
                # (LlmStatus.name), populated from the end event below.
                # None until end lands (or on pre-this-fix jsonl files /
                # aborted begins) — `_failure_class_for_label` reads it
                # via `.get()` and buckets absent as "other".
                "llm_status": None,
                "aborted": False,
                # `skipped` (issue #333): True when the run dir's
                # `skipped_calls/<call_id>` marker exists. Stamped at
                # the end of `_materialize_calls_from_jsonl` once all
                # begin/end events are reconciled, so the flag survives
                # any error-finalize that hit before the marker was
                # observed.
                "skipped": False,
                # `cached` is set on the end-event payload by
                # finalize_stat_record when complete() short-circuited
                # on the prompt-hash cache. Initialize False so an
                # aborted call (begin without end) doesn't silently
                # carry a missing key — debug-bundle consumers can
                # rely on every record having the field.
                "cached": False,
                # `cache_key` is set on the end event by
                # `finalize_stat_record` from the live stat record. None
                # for calls that never hit the cache layer (failures
                # before the lookup, monkeypatched-complete() tests).
                "cache_key": None,
                # Materially-relevant call kwargs (reasoning,
                # temperature, etc.). None on pre-this-PR jsonl files
                # so consumers can distinguish "field absent" from
                # "field present with reasoning=False".
                "request_extras": ev.get("request_extras"),
                # `template_hash` is the stable scaffold fingerprint
                # passed by the stage caller via `_stat_template_hash`.
                # None for stages that don't yet plumb it through.
                "template_hash": ev.get("template_hash"),
                # `parse_error` is set by
                # record_stage_counts(failure_kind="parse_error")
                # when the stage's parser couldn't decode the response.
                # Distinct from `error` (provider/network failure):
                # parse_error means the provider returned text but it
                # wasn't valid JSON / didn't match the expected schema.
                "parse_error": False,
                # `empty_response` is set by
                # record_stage_counts(failure_kind="empty_response")
                # when the model returned literal "" / whitespace on
                # the wire — distinct from `success_empty` (model
                # returned valid `[]` / 0 entries). Outcome bucket
                # renders red.
                "empty_response": False,
                # `interrupted` is set by
                # record_stage_counts(failure_kind="interrupted")
                # when the stream delivered some bytes but closed
                # cleanly without a terminating finish_reason chunk
                # and below max_tokens_reserved — typically a clean
                # SSE close mid-flow that the SDK didn't surface as
                # an exception. Distinct from cap-hit
                # (finish_reason="length") and empty_response (no
                # bytes flowed at all). Outcome
                # bucket renders red.
                "interrupted": False,
            }
            order.append(cid)
        elif kind == "end":
            rec = by_id.get(cid)
            if rec is None:
                continue
            rec["duration_ms"] = ev.get("duration_ms")
            rec["success"] = ev.get("success")
            rec["error"] = ev.get("error")
            # Issue #225: end's prompt_tokens overwrites the begin-time
            # estimate ONLY when truthy. A 0 (wire-cut, provider's usage
            # chunk never arrived) leaves the estimate intact so the UI
            # never shows `0 in` for a call that genuinely sent thousands
            # of input tokens.
            _end_pt = ev.get("prompt_tokens")
            if _end_pt:
                rec["prompt_tokens"] = _end_pt
            rec["completion_tokens"] = ev.get("completion_tokens")
            if ev.get("model") is not None:
                rec["model"] = ev.get("model")
            if ev.get("mode") is not None:
                rec["mode"] = ev.get("mode")
            # Pass through streaming observability fields (issue #104
            # part 1). `.get()` so pre-this-PR jsonl files (no field on
            # the line) read as None instead of KeyError.
            for _f in (
                "reasoning_tokens", "reasoning_tokens_source",
                "content_tokens", "finish_reason",
                "ttft_ms", "ttfr_ms", "last_token_ms",
                "max_tokens_reserved",
            ):
                if ev.get(_f) is not None:
                    rec[_f] = ev.get(_f)
            # Pass through cache-hit flag. Old event logs (pre-this-PR
            # without `cached` in the payload) read as False — same
            # default as the begin-init above.
            if ev.get("cached") is not None:
                rec["cached"] = bool(ev.get("cached"))
            # `cache_key` is the sha256 cache identifier llm.complete
            # stamps on the live stat record at lookup time. Threading
            # it onto the materialized record lets the per-call detail
            # UI offer "remove from cache" without needing the raw
            # messages (which the rollup deliberately doesn't store).
            if ev.get("cache_key") is not None:
                rec["cache_key"] = ev.get("cache_key")
            # Kernel's categorized outcome — drives the failure-label
            # bucket. `.get()` so pre-this-fix end lines (no field) read
            # as None instead of clobbering the begin-init.
            if ev.get("llm_status") is not None:
                rec["llm_status"] = ev.get("llm_status")
        elif kind == "counts":
            rec = by_id.get(cid)
            if rec is None:
                continue
            if ev.get("input") is not None:
                rec["input"] = ev.get("input")
            if ev.get("output") is not None:
                rec["output"] = ev.get("output")
            if ev.get("parse_error"):
                rec["parse_error"] = True
            if ev.get("empty_response"):
                rec["empty_response"] = True
            if ev.get("interrupted"):
                rec["interrupted"] = True
        elif kind == "stream_progress":
            # One-shot event emitted by `_consume_chat_stream` at the
            # first-token-received moment, carrying TTFT (and TTFR for
            # reasoning models). Surfaces a durable "stream started"
            # signal on the in-flight rec before the end event lands.
            # The end event still wins at finalize for token counts;
            # token-count fields are tolerated for back-compat with
            # historical multi-event logs but no longer populated by
            # the current emitter.
            rec = by_id.get(cid)
            if rec is None:
                continue
            _ct = ev.get("completion_tokens")
            if _ct is not None:
                rec["completion_tokens"] = _ct
            _rt = ev.get("reasoning_tokens")
            if _rt is not None:
                rec["reasoning_tokens"] = _rt
            _cot = ev.get("content_tokens")
            if _cot is not None:
                rec["content_tokens"] = _cot
            for _tf in ("ttft_ms", "ttfr_ms", "last_token_ms"):
                _v = ev.get(_tf)
                if _v is not None:
                    rec[_tf] = _v
        elif kind == "full_io":
            # Legacy: pre-#195 runs stored full_io records inline in
            # llm-calls.jsonl. Newer runs split them out to a sibling
            # llm-payloads.jsonl (handled below); this branch keeps
            # old runs viewable. Off-by-default toggle — for normal
            # runs this never fires regardless of file layout.
            rec = by_id.get(cid)
            if rec is None:
                continue
            if ev.get("full_prompt") is not None:
                rec["full_prompt"] = ev.get("full_prompt")
            if ev.get("full_response") is not None:
                rec["full_response"] = ev.get("full_response")

    # ── Sibling payload stream (issue #195) ───────────────────────────
    # Full prompt + response payloads live in `llm-payloads.jsonl`
    # rather than inline in llm-calls.jsonl. Walk it after the main
    # event log so by_id is fully populated; payload records key on
    # call_id and thread `full_prompt` / `full_response` onto the
    # matching record. Missing file = no payloads logged this run.
    payloads_path = jsonl_path.parent / "llm-payloads.jsonl"
    if payloads_path.exists():
        for raw in payloads_path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            cid = ev.get("call_id")
            if cid is None:
                continue
            rec = by_id.get(cid)
            if rec is None:
                continue
            if ev.get("full_prompt") is not None:
                rec["full_prompt"] = ev.get("full_prompt")
            if ev.get("full_response") is not None:
                rec["full_response"] = ev.get("full_response")

    # Mark unmatched begins as aborted. The call itself didn't fail —
    # the run wound down (paused / cancelled / errored) with this call
    # mid-flight, so `error` stays None and `_classify_outcome` reads
    # `aborted=True` to produce the distinct `OUTCOME_ABORTED` bucket.
    # Mirrors the Rust live materializer at lib.rs:1101-1131.
    for cid in order:
        rec = by_id[cid]
        if rec["success"] is None:
            rec["aborted"] = True
            rec["success"] = False
            # Clamp to >= 0 — the begin's started_at_iso has millisecond
            # precision while ended_at_iso may be whole-second only
            # (`_iso_z_full`), so within the same second a begin can
            # appear "after" ended_at by a few hundred ms.
            d = _iso_delta_ms(rec.get("started_at_iso"), ended_at_iso) or 0
            rec["duration_ms"] = max(0, d)

    # Issue #333: per-call skip markers. The Tauri ✕ control writes
    # `<run_dir>/skipped_calls/<call_id>` on click; the marker is the
    # durable source of truth (the in-process `_skipped_call_ids` set
    # is the fast path for the stream consumer). Stamp `skipped=True`
    # on every matching rec and null out any error payload the wrapper
    # may have written before short-circuiting — a skip is a human
    # signal, not a provider failure, so the rec should not carry an
    # `_SkippedByUser` error class through to consumers.
    run_dir = jsonl_path.parent
    skipped_dir = run_dir / "skipped_calls"
    if skipped_dir.is_dir():
        try:
            skipped_ids = {p.name for p in skipped_dir.iterdir() if p.is_file()}
        except OSError:
            skipped_ids = set()
        for cid in skipped_ids:
            rec = by_id.get(cid)
            if rec is None:
                continue
            rec["skipped"] = True
            rec["error"] = None
            # `aborted` and `skipped` are mutually exclusive — a
            # skipped call's begin had a matching end (the wrapper
            # finalized when `_SkippedByUser` bubbled up), so this
            # is defensive against a race where the wrapper never
            # finalized and the rec was tagged aborted first.
            rec["aborted"] = False
    return [by_id[cid] for cid in order]


# Outcome enum surfaced on every materialized call record. Issue #104
# replaces the boolean `success` field with a richer categorization so
# the UI can distinguish "model timed out" from "model returned but
# everything was filtered" — both look like `success=False` to the
# legacy consumer. `success` stays on the record for back-compat;
# `outcome` is additive.
# Outcome label strings — re-exported from ``common.status`` (the
# single source of truth). Imports here so legacy ``runner.OUTCOME_*``
# references keep working without a separate definition that could
# drift from the canonical ``Outcome`` enum + the shareable
# diagnostic. See ``common/status.py`` for the full taxonomy + the
# load/sizing/other parenthetical convention.
from engine.common.status import (  # noqa: E402
    OUTCOME_ABORTED,
    OUTCOME_CAP_HIT,
    OUTCOME_EMPTY_RESPONSE,
    OUTCOME_FAILED_LOAD,
    OUTCOME_FAILED_OTHER,
    OUTCOME_FAILED_SIZING,
    OUTCOME_INTERRUPTED_LOAD,
    OUTCOME_INTERRUPTED_SIZING,
    OUTCOME_PARSE_ERROR,
    OUTCOME_SKIPPED,
    OUTCOME_SUCCESS,
    OUTCOME_SUCCESS_EMPTY,
    OUTCOME_SUCCESS_REASONING_OFF,
    OUTCOME_SUCCESS_SAMPLED,
    OUTCOME_TIMEOUT_LOAD,
    OUTCOME_TIMEOUT_SIZING,
)

def _is_timeout_error(err: dict | None) -> bool:
    """True when the error class/message looks timeout-shaped. Used by
    `_classify_outcome` to label timeouts specifically and bucket the
    rest as generic 'failed (CLASS)'."""
    if not err:
        return False
    cls = (err.get("class") or "").lower()
    msg = (err.get("message") or "").lower()
    if "timeout" in cls or "timeout" in msg:
        return True
    if "timed out" in msg:
        return True
    return False


def _failure_class_for_label(rec: dict, err: dict) -> str:
    """Map a failed record to one of `"load"` / `"sizing"` / `"other"` —
    the parenthetical suffix in user-visible labels like `'failed (load)'`
    / `'cap_hit (sizing)'`.

    The kernel stamps each record with `llm_status` — its OWN categorized
    outcome (`LlmStatus` already encodes load / sizing / other), written by
    the KernelTelemetryHook. Read the bucket straight off it. Pre-kernel
    records (no `llm_status`) predate the cutover and the legacy
    class-name classifier (`retry.classify_bucket`) is gone, so they bucket
    to `"other"`."""
    status = rec.get("llm_status")
    if not status:
        return "other"
    from kernel.enums import LlmStatus
    _KERNEL_BUCKET = {
        LlmStatus.LOAD.name: "load",
        LlmStatus.CAP_HIT.name: "sizing",
        LlmStatus.PARSE_ERROR.name: "sizing",
        LlmStatus.TIMEOUT_WITH_TOKENS.name: "sizing",
        LlmStatus.OTHER.name: "other",
    }
    return _KERNEL_BUCKET.get(status, "other")


def _classify_outcome(rec: dict) -> str:
    """Single source of truth for a per-call record's user-visible
    label. Reads the same fields the UI eventually renders, so
    derivation is stable across re-materialization (offline rollup
    vs end-of-run).

    Failure labels carry a parenthetical suffix marking which retry
    strategy classified them: `failed (load)`, `failed (sizing)`,
    `failed (other)`. When a more specific name is available —
    `cap_hit`, `timeout`, `parse_error`, `empty_response`,
    `interrupted` — that name replaces 'failed' but keeps the suffix:
    `cap_hit (sizing)`, `timeout (load)`, `interrupted (sizing)`, etc.

    Order of checks matters: timeouts win over generic failures;
    parse_error (parser-level rejection) wins over success_empty
    (parser succeeded, model returned []); the TTFT-fraction rule
    decides sizing vs load on tokens-flowed failures (timeout,
    midstream, interrupted). `skipped` and `aborted` are checked
    first (skipped before aborted) so a user-skip that races a
    pause/cancel still surfaces as the explicit human signal, and
    a begin-without-end rec on a wound-down run doesn't fall
    through to a failure bucket — the call didn't fail."""
    if rec.get("skipped"):
        return OUTCOME_SKIPPED
    if rec.get("aborted"):
        return OUTCOME_ABORTED
    err = rec.get("error")
    success_flag = rec.get("success")
    if err is not None or success_flag is not True:
        cls = (err or {}).get("class") or ""
        # Read the kernel's `llm_status` bucket even when there's no
        # exception (synthetic / from_status failures: success=False,
        # error=None). `_failure_class_for_label` already returns
        # "other" when `llm_status` is absent, so dropping the old
        # `if err` gate is safe and lets a no-exception LOAD label as
        # load instead of collapsing to other.
        fclass = _failure_class_for_label(rec, err or {})
        # Specific names take priority over the generic 'failed (X)'.
        if cls.endswith("_CapHitResponse") or rec.get("finish_reason") == "length":
            return OUTCOME_CAP_HIT  # cap_hit always classifies as sizing
        if rec.get("parse_error") or "parse_error" in (err or {}).get("message", ""):
            return OUTCOME_PARSE_ERROR  # parse_error always sizing
        if rec.get("empty_response") or "empty_response" in (err or {}).get("message", ""):
            return OUTCOME_EMPTY_RESPONSE  # empty_response always load
        if rec.get("interrupted") or "interrupted" in (err or {}).get("message", ""):
            # interrupted: TTFT-fraction discriminates sizing vs load.
            return (OUTCOME_INTERRUPTED_SIZING if fclass == "sizing"
                    else OUTCOME_INTERRUPTED_LOAD)
        if (cls.endswith("_SuccessEmpty")
                or rec.get("success_empty")
                or "success_empty" in (err or {}).get("message", "")):
            # Parsed-empty: the call streamed successfully but the
            # structured output had zero entries. Surfaces in run
            # records as the dedicated success_empty bucket — never
            # plain green even though the bytes-on-wire were clean.
            return OUTCOME_SUCCESS_EMPTY
        if _is_timeout_error(err):
            return (OUTCOME_TIMEOUT_SIZING if fclass == "sizing"
                    else OUTCOME_TIMEOUT_LOAD)
        # Generic failure with the strategy classification only.
        return {
            "load": OUTCOME_FAILED_LOAD,
            "sizing": OUTCOME_FAILED_SIZING,
            "other": OUTCOME_FAILED_OTHER,
        }[fclass]
    # Success path. Cap-hit on success can no longer occur in the
    # post-refactor architecture (wrapper raises _CapHitResponse and
    # routes through the failure branch above), but check defensively
    # in case a stage path bypasses the wrapper.
    if rec.get("finish_reason") == "length":
        return OUTCOME_CAP_HIT
    if rec.get("parse_error"):
        return OUTCOME_PARSE_ERROR
    if rec.get("empty_response"):
        return OUTCOME_EMPTY_RESPONSE
    # Defer the "is the primary entry count zero?" decision to the
    # shared rule `llm._output_is_zero_size` — same convention the
    # auto-payload-log trigger uses (first integer leaf in the output
    # dict is the primary count; subsequent ints like capacity limits
    # are auxiliary). Summing every int here would inflate `kept` past
    # zero on stages whose output dict carries cap fields alongside
    # the entry counts (insights, actions) and misroute a clean
    # empty result through `success` instead of `success_empty`.
    #
    # Clean valid empty: model returned `[]` (or equivalent) as its
    # parsed answer. The post-stream failure shapes (empty_response /
    # interrupted / parse_error) are reserved for zero-content-token
    # failures. Reasoning_tokens > 0 with a valid empty payload is
    # "model thought, then decided nothing applied" — a legitimate
    # answer, not a failure.
    from engine.llm import _output_is_zero_size as _is_zero
    output = rec.get("output")
    if _is_zero(output):
        return OUTCOME_SUCCESS_EMPTY
    return OUTCOME_SUCCESS


def _apply_chain_aware_outcomes(records: list[dict]) -> None:
    """Issue #105 v3 follow-up: re-bucket chain LEAVES whose chain
    contains a /sample-N call from `success` to `sampled`.

    A leaf is a record whose call_id doesn't appear in any other
    record's `retry_of_call_id`. For each successful leaf, walk back
    via retry_of_call_id; if any record on the path (including the
    leaf itself) has a category containing "/sample-", the leaf's
    output came from a reduced fact set — the patterns work-reducing
    loop succeeded only after dropping facts. The user wants this
    surfaced as a warning rather than clean success (b889 / #171
    follow-up #4).

    Operates IN PLACE on `records`. Earlier (non-leaf) records keep
    their per-call outcome so the chain-expand view still shows what
    actually happened at each attempt."""
    if not records:
        return
    by_id: dict[str, dict] = {}
    children_of: set[str] = set()
    for r in records:
        cid = r.get("call_id")
        if cid is None:
            continue
        by_id[cid] = r
        parent = r.get("retry_of_call_id")
        if parent is not None:
            children_of.add(parent)

    def _walk_chain(leaf: dict) -> list[dict]:
        chain = [leaf]
        seen = {leaf.get("call_id")}
        cur = leaf
        for _ in range(64):  # defensive cap on cycle / pathological depth
            parent_id = cur.get("retry_of_call_id")
            if parent_id is None or parent_id in seen:
                break
            parent = by_id.get(parent_id)
            if parent is None:
                break
            chain.append(parent)
            seen.add(parent_id)
            cur = parent
        return chain

    for r in records:
        cid = r.get("call_id")
        if cid is None or cid in children_of:
            # Not a leaf — chain-aware bucketing only applies to leaves.
            continue
        if r.get("outcome") != OUTCOME_SUCCESS:
            # Pre-fix only `success` leaves under sample chains
            # convert. Failed leaves keep their failure outcome —
            # the work-reducer didn't recover, so it's not "sampled
            # but ok," it's just a failure.
            continue
        chain = _walk_chain(r)
        # Three sizing-chain paths, three outcomes:
        #   • `/sample-N - retry/sizing` → success_sampled (reduced
        #     input set; full coverage NOT preserved).
        #   • `/reasoning-off - retry/sizing` (no sample) →
        #     success_reasoning_off (full coverage but degraded
        #     compute path; warn).
        #   • `/half-N - retry/sizing` only → clean success
        #     (both halves processed; nothing degraded).
        # `sample-N` outranks `reasoning-off` since the chain
        # might have BOTH (depth=1 reasoning-off step then
        # depth>=2 sampling). Sampling is the bigger caveat.
        had_sampled = False
        had_reasoning_off = False
        for c in chain:
            cat = c.get("category") or ""
            if "/sample-" in cat:
                had_sampled = True
            elif "/reasoning-off" in cat:
                had_reasoning_off = True
        if had_sampled:
            r["outcome"] = OUTCOME_SUCCESS_SAMPLED
        elif had_reasoning_off:
            r["outcome"] = OUTCOME_SUCCESS_REASONING_OFF


def _leaf_aware_warnings(
    leaf_outcomes: dict[str, int],
    leaf_input_overflows: int = 0,
) -> dict[str, int]:
    """Build the run-row warnings dict from leaf outcomes.

    Single source of truth for the warning buckets the UI's run-row
    badge sums into "X calls flagged". Every value reflects LEAVES
    only — chains that recovered via halving / sample-fewer-facts
    don't inflate counts. Per-call detail in llm-stats.calls / per-
    stage outcomes_total still shows every attempt for debugging.

    Bucket mapping:
      - cap_hits: leaves with any `cap_hit (...)` label.
      - timeouts: leaves with any `timeout (...)` label.
      - parse_errors / empty_responses / interrupted: leaves with
        the matching post-stream-failure prefix label.
      - input_overflows: leaf-filtered count of `kind=input_overflow`
        warnings. Request-shape signal (no leaf outcome bucket), so
        the rollup filters warnings by call_id ∈ leaves and passes
        the count via `leaf_input_overflows`.
      - success_empty / sampled: direct leaf outcome bucket counts."""
    # Outcome labels now carry a parenthetical strategy tag — match
    # by prefix so e.g. both `timeout (sizing)` and `timeout (load)`
    # count toward `timeouts`. Same for cap_hit / parse_error /
    # empty_response / interrupted.
    def _sum_prefix(prefix: str) -> int:
        return sum(
            n for label, n in leaf_outcomes.items()
            if isinstance(label, str) and label.startswith(prefix)
        )
    return {
        "cap_hits": _sum_prefix("cap_hit"),
        "empty_responses": _sum_prefix("empty_response"),
        "interrupted": _sum_prefix("interrupted"),
        "input_overflows": leaf_input_overflows,
        "timeouts": _sum_prefix("timeout"),
        "success_empty": leaf_outcomes.get(OUTCOME_SUCCESS_EMPTY, 0),
        "parse_errors": _sum_prefix("parse_error"),
        # Generic `failed (X)` leaves — catch-all for failures
        # without a specific name. One bucket across all four
        # strategy classifications.
        "failed": _sum_prefix("failed"),
        # Chain-leaf-only success-with-caveat outcomes (yellow
        # band). `sampled` keeps the ⚠ marker (input was reduced);
        # `success_empty` and `reasoning_off` are data-complete
        # and don't fire ⚠ in the per-row view.
        "sampled": leaf_outcomes.get(OUTCOME_SUCCESS_SAMPLED, 0),
        "reasoning_off": leaf_outcomes.get(
            OUTCOME_SUCCESS_REASONING_OFF, 0),
    }


def _agg_dist(values: list[int]) -> dict:
    """Distribution stats over a list of non-None ints. Returns total
    + avg / median / p95 / min / max. Empty input → all None / 0 total
    so the JSON shape is uniform across stages with zero data.

    `p95` is computed as the value at the rank-0.95 index of the sorted
    list (linear interpolation skipped — fine for ms / token counts and
    keeps the shape integer-friendly). For n=1 it equals the single
    sample; for n=2 it equals max."""
    if not values:
        return {
            "total": 0, "avg": None, "median": None, "p95": None,
            "min": None, "max": None,
        }
    n = len(values)
    sv = sorted(values)
    total = sum(values)
    avg = total / n
    if n % 2 == 1:
        median = float(sv[n // 2])
    else:
        median = (sv[n // 2 - 1] + sv[n // 2]) / 2
    if n == 1:
        p95 = float(sv[0])
    else:
        idx = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
        p95 = float(sv[idx])
    return {
        "total": total,
        "avg": round(avg, 2),
        "median": round(median, 2),
        "p95": round(p95, 2),
        "min": sv[0],
        "max": sv[-1],
    }


def materialize_run_stats(
    jsonl_path: "Path | str",
    config_path: "Path | str | None" = None,
    *,
    ended_at_iso: str | None = None,
    llm_cache_stats: dict | None = None,
    call_warnings: "list[dict] | None" = None,
    live_warnings_baseline: dict | None = None,
    primary_model: str | None = None,
    started_at_iso: str | None = None,
) -> dict:
    """Pure rollup function — single source of truth for the
    llm-stats/v1 payload. Same logic running and finished, by
    construction (issue #189).

    Reads `llm-calls.jsonl` (the canonical event log) and an optional
    `config.json` (run metadata) and returns a dict matching the
    llm-stats/v1 schema. No side effects: no file writes, no global
    state mutation, no logging.

    The end-of-run writer (`_write_llm_stats`) calls this to compute
    the .txt summary; on-demand readers (eval/judge.py, future Tauri
    Python bridge, ad-hoc scripts) call it to derive stats from disk
    without re-running the pipeline.

    Args:
      jsonl_path: path to llm-calls.jsonl. Missing/empty → records=[].
      config_path: optional config.json — read for run_id / short_id /
          agent / mode / created_at. Caller can override any field via
          the keyword args below.
      ended_at_iso: when the run "ended" — used to estimate
          duration_ms for unmatched-begin records (aborted in flight)
          and total_wall_clock_ms. Defaults to now.
      llm_cache_stats: in-process cache stats from
          `llm_cache.get_cache_stats()`. End-of-run callsite passes
          this; on-demand readers don't have it (rendered as {}).
      call_warnings: in-process call-warnings list from
          `llm.get_call_warnings()`. End-of-run callsite passes this;
          without it, `input_overflows = 0` (warning emit is
          in-memory only — not on-disk-derivable today).
      live_warnings_baseline: extras (cap_hits / empty_responses /
          input_overflows live counters) to merge in BEFORE the
          leaf-aware override. End-of-run callsite passes
          `_run_state["warnings"]`; on-demand readers omit.
      primary_model / started_at_iso: caller overrides for
          config.json's `model` / `created_at`. Used by the end-of-run
          callsite to keep behavior identical when a config field is
          missing or stale.
    """
    # ── Resolve metadata from config.json (with caller overrides) ──────
    config: dict = {}
    if config_path is not None:
        try:
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}

    run_id = config.get("run_id")
    short_id = config.get("short_id") or (
        _short_id_from_run_name(run_id) if run_id else None
    )
    agent = config.get("agent") or _agent()
    mode_str = config.get("mode") or "unknown"
    primary_model_resolved = primary_model or config.get("model")
    started_at = started_at_iso or config.get("created_at")

    ended_at = ended_at_iso or _iso_z_full()

    # ── Read + materialize the event log ──────────────────────────────
    jsonl_path = Path(jsonl_path)
    records_raw = _materialize_calls_from_jsonl(jsonl_path, ended_at)
    # Per-call outcome classification (issue #104). `outcome` is
    # additive — `success` boolean stays on every record so legacy
    # consumers keep working; the UI reads `outcome` for richer
    # bucketing.
    records = []
    for r in records_raw:
        rec = dict(r)
        rec["outcome"] = _classify_outcome(rec)
        records.append(rec)
    # Chain-leaf-only re-bucketing (issue #105 v3): `success` leaves
    # whose chain includes a /sample-N attempt convert to `sampled` —
    # the leaf output came from reduced facts, not the full set.
    _apply_chain_aware_outcomes(records)

    aborted_count = sum(1 for r in records if r.get("aborted"))
    skipped_count = sum(1 for r in records if r.get("skipped"))
    # `failed` excludes aborted AND skipped — both are neutral (run
    # wound down / user intervened), not provider failures.
    failed_count = sum(
        1 for r in records
        if (r.get("error") is not None
            and not r.get("aborted")
            and not r.get("skipped")))
    successful_count = sum(1 for r in records if r.get("success") is True)

    # Outcome totals (issue #104). Buckets mirror per_stage.outcomes.
    _outcome_keys_top = (
        OUTCOME_SUCCESS, OUTCOME_ABORTED, OUTCOME_SKIPPED,
        OUTCOME_CAP_HIT, OUTCOME_SUCCESS_EMPTY,
        OUTCOME_EMPTY_RESPONSE,
        OUTCOME_INTERRUPTED_SIZING, OUTCOME_INTERRUPTED_LOAD,
        OUTCOME_PARSE_ERROR,
        OUTCOME_TIMEOUT_SIZING, OUTCOME_TIMEOUT_LOAD,
        OUTCOME_FAILED_LOAD, OUTCOME_FAILED_SIZING,
        OUTCOME_FAILED_OTHER,
        OUTCOME_SUCCESS_SAMPLED,
        OUTCOME_SUCCESS_REASONING_OFF,
    )
    outcomes_total = {k: 0 for k in _outcome_keys_top}
    for r in records:
        oc = r.get("outcome")
        if oc in outcomes_total:
            outcomes_total[oc] += 1

    # Leaf-only counters for the run-row banner (issue #105 v3
    # follow-up). A chain like split_00 (timeout) → split_00/half-1
    # (success) inflated `warnings.timeouts=1` despite the chain
    # recovering cleanly. Leaves are records whose call_id is NOT in
    # any other record's retry_of_call_id.
    _child_of: set = set()
    for r in records:
        rof = r.get("retry_of_call_id")
        if rof is not None:
            _child_of.add(rof)
    leaf_outcomes = {k: 0 for k in _outcome_keys_top}
    _leaf_ids: set = set()
    for r in records:
        cid = r.get("call_id")
        if cid is None or cid in _child_of:
            continue
        _leaf_ids.add(cid)
        oc = r.get("outcome")
        if oc in leaf_outcomes:
            leaf_outcomes[oc] += 1
    # Leaf-filter input_overflow warnings by stamped call_id.
    # `_record_call_warning` writes call_id at emit time; parent
    # attempts that overflowed and were halved produce non-leaf
    # warnings that don't count toward the badge — only overflows
    # surviving to the chain leaf are user-visible failures.
    _warnings_for_filter = call_warnings if call_warnings is not None else []
    _leaf_input_overflows = sum(
        1 for w in _warnings_for_filter
        if w.get("kind") == "input_overflow"
        and w.get("call_id") in _leaf_ids
    )

    totals = {
        "calls": len(records),
        "successful": successful_count,
        "failed": failed_count,
        "aborted": aborted_count,
        "skipped": skipped_count,
        # categories_lost rolled up from by_stage below.
        "categories_lost": 0,
        "outcomes": outcomes_total,
        "prompt_tokens": sum((r.get("prompt_tokens") or 0) for r in records),
        "completion_tokens": sum(
            (r.get("completion_tokens") or 0) for r in records),
    }
    by_stage: dict[str, dict] = {}
    for r in records:
        s = r.get("stage") or "unknown"
        bucket = by_stage.setdefault(s, {
            "calls": 0, "successful": 0, "failed": 0,
            "aborted": 0, "skipped": 0,
            "categories_lost": 0,  # filled below
            "prompt_tokens": 0, "completion_tokens": 0,
            "wall_clock_ms": 0,
        })
        bucket["calls"] += 1
        if r.get("skipped"):
            bucket["skipped"] += 1
        elif r.get("aborted"):
            bucket["aborted"] += 1
        elif r.get("success") is True:
            bucket["successful"] += 1
        else:
            bucket["failed"] += 1
        bucket["prompt_tokens"] += (r.get("prompt_tokens") or 0)
        bucket["completion_tokens"] += (r.get("completion_tokens") or 0)
        bucket["wall_clock_ms"] += (r.get("duration_ms") or 0)

    # categories_lost = count of (stage, category) tuples where every
    # attempt across the retry chain failed (no successful record).
    # Distinct from `failed` (which counts attempts not topics): a
    # single topic that recovers on retry contributes 1 to `failed`
    # AND 0 to `categories_lost`; a topic that fails both attempts
    # contributes 2 to `failed` AND 1 to `categories_lost`. Surfaces
    # the actual user-facing data loss separately from the retry
    # cost. Run fagv showed totals.failed=7 but only 1 topic
    # actually lost — without this rollup, "7 failed calls" reads as
    # "7 lost topics" to a casual viewer.
    cat_outcomes: dict[tuple[str, str], list[bool]] = {}
    for r in records:
        s = r.get("stage") or "unknown"
        cat = r.get("category")
        if cat is None:
            continue
        cat_outcomes.setdefault((s, cat), []).append(r.get("success") is True)
    lost_per_stage: dict[str, int] = {}
    for (s, _cat), outcomes in cat_outcomes.items():
        if not any(outcomes):
            lost_per_stage[s] = lost_per_stage.get(s, 0) + 1
    for s, n in lost_per_stage.items():
        if s in by_stage:
            by_stage[s]["categories_lost"] = n
    totals["categories_lost"] = sum(lost_per_stage.values())
    by_model: dict[str, dict] = {}
    for r in records:
        m = r.get("model") or "unknown"
        bucket = by_model.setdefault(m, {
            "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        })
        bucket["calls"] += 1
        bucket["prompt_tokens"] += (r.get("prompt_tokens") or 0)
        bucket["completion_tokens"] += (r.get("completion_tokens") or 0)

    # ── Per-stage statistics (Round 3 — distribution, models_used) ─────
    # Richer view than `by_stage`: for each stage, capture the full
    # distribution of prompt_tokens / completion_tokens / duration_ms
    # (avg / median / min / max + total) plus the SET of models used
    # (not just one — if the user changed settings mid-session, or
    # per-stage routing sends different stages to different models,
    # multiple models can appear).
    # The legacy `by_stage` shape stays intact for back-compat with
    # eval/judge.py; this is additive.
    per_stage_stats: dict[str, dict] = {}
    stages_in_order: list[str] = []
    _outcome_keys = (
        OUTCOME_SUCCESS, OUTCOME_ABORTED, OUTCOME_SKIPPED,
        OUTCOME_CAP_HIT, OUTCOME_SUCCESS_EMPTY,
        OUTCOME_EMPTY_RESPONSE,
        OUTCOME_INTERRUPTED_SIZING, OUTCOME_INTERRUPTED_LOAD,
        OUTCOME_PARSE_ERROR,
        OUTCOME_TIMEOUT_SIZING, OUTCOME_TIMEOUT_LOAD,
        OUTCOME_FAILED_LOAD, OUTCOME_FAILED_SIZING,
        OUTCOME_FAILED_OTHER,
        OUTCOME_SUCCESS_SAMPLED,
        OUTCOME_SUCCESS_REASONING_OFF,
    )
    for r in records:
        s = r.get("stage") or "unknown"
        if s not in per_stage_stats:
            stages_in_order.append(s)
            per_stage_stats[s] = {
                "_pt": [], "_ct": [], "_dur": [], "_models": [],
                # Streaming observability buckets (issue #104 part 1).
                # Each call contributes at most one int per list; None
                # entries are dropped so cache-only records (ttft=0)
                # vs un-instrumented records (ttft=None) don't both
                # roll up at "0" and silently skew percentiles.
                "_rt": [], "_cont": [], "_ttft": [], "_ttfr": [], "_last": [],
                "_finish_reasons": {},
                "calls_total": 0, "calls_failed": 0,
                "calls_aborted": 0, "calls_skipped": 0,
                "_reasoning_seen": set(),
                "_extras_seen": [],
                "calls_cached": 0,
                "outcomes": {k: 0 for k in _outcome_keys},
            }
        bucket = per_stage_stats[s]
        bucket["calls_total"] += 1
        if r.get("skipped"):
            bucket["calls_skipped"] += 1
        elif r.get("aborted"):
            bucket["calls_aborted"] += 1
        elif r.get("success") is False:
            bucket["calls_failed"] += 1
        if r.get("cached"):
            bucket["calls_cached"] += 1
        oc = r.get("outcome")
        if oc in bucket["outcomes"]:
            bucket["outcomes"][oc] += 1
        if r.get("prompt_tokens") is not None:
            bucket["_pt"].append(int(r["prompt_tokens"]))
        if r.get("completion_tokens") is not None:
            bucket["_ct"].append(int(r["completion_tokens"]))
        if r.get("duration_ms") is not None:
            bucket["_dur"].append(int(r["duration_ms"]))
        # Streaming observability rollup. Skip None so providers /
        # call paths that didn't emit a metric (ollama has no usage
        # chunk for vision; cache hits emit ttfr=None) don't pull the
        # distribution toward zero.
        rt = r.get("reasoning_tokens")
        if rt is not None:
            bucket["_rt"].append(int(rt))
        ct_only = r.get("content_tokens")
        if ct_only is not None:
            bucket["_cont"].append(int(ct_only))
        ttft = r.get("ttft_ms")
        if ttft is not None:
            bucket["_ttft"].append(int(ttft))
        ttfr = r.get("ttfr_ms")
        if ttfr is not None:
            bucket["_ttfr"].append(int(ttfr))
        last_t = r.get("last_token_ms")
        if last_t is not None:
            bucket["_last"].append(int(last_t))
        fr = r.get("finish_reason")
        if fr:
            bucket["_finish_reasons"][fr] = (
                bucket["_finish_reasons"].get(fr, 0) + 1
            )
        m = r.get("model")
        if m and m not in bucket["_models"]:
            bucket["_models"].append(m)
        extras = r.get("request_extras") or {}
        if isinstance(extras, dict):
            bucket["_reasoning_seen"].add(bool(extras.get("reasoning", False)))
            # Capture the first non-empty extras dict per stage so the
            # rollup carries a representative example without
            # ballooning into per-call payloads.
            if extras and not bucket["_extras_seen"]:
                bucket["_extras_seen"].append(dict(extras))
    per_stage: dict[str, dict] = {}
    for s in stages_in_order:
        b = per_stage_stats[s]
        # reasoning_enabled is True when at least one call in this
        # stage carried reasoning=True. Defaults to False for stages
        # whose calls all pre-dated this PR's request_extras logging.
        reasoning_enabled = True in b["_reasoning_seen"]
        # mixed=True signals "some calls reasoning ON, some OFF" — only
        # surfaced when both states appear (e.g. mid-run config
        # change or a bug). Most runs are single-state.
        reasoning_mixed = (True in b["_reasoning_seen"] and
                            False in b["_reasoning_seen"])
        per_stage[s] = {
            "name": s,
            "calls_total": b["calls_total"],
            "calls_failed": b["calls_failed"],
            "calls_aborted": b["calls_aborted"],
            "calls_skipped": b["calls_skipped"],
            "calls_cached": b["calls_cached"],
            "outcomes": dict(b["outcomes"]),
            "models_used": list(b["_models"]),
            "prompt_tokens": _agg_dist(b["_pt"]),
            "completion_tokens": _agg_dist(b["_ct"]),
            "duration_ms": _agg_dist(b["_dur"]),
            # Streaming observability (issue #104 part 1). reasoning +
            # content_tokens roll up alongside completion_tokens — sum
            # is the cost lever, p50/p95 ttft is the admission /
            # latency signal retry-policy v2 reads. finish_reasons is
            # a {reason: count} dict (e.g. {"stop": 47, "length": 1}).
            "reasoning_tokens": _agg_dist(b["_rt"]),
            "content_tokens": _agg_dist(b["_cont"]),
            "ttft_ms": _agg_dist(b["_ttft"]),
            "ttfr_ms": _agg_dist(b["_ttfr"]),
            "last_token_ms": _agg_dist(b["_last"]),
            "finish_reasons": dict(b["_finish_reasons"]),
            "reasoning_enabled": reasoning_enabled,
            "reasoning_mixed": reasoning_mixed,
            "request_extras_sample": (
                b["_extras_seen"][0] if b["_extras_seen"] else None
            ),
        }

    # Top-level distribution stats across ALL calls regardless of stage.
    all_pt = [int(r["prompt_tokens"]) for r in records
              if r.get("prompt_tokens") is not None]
    all_ct = [int(r["completion_tokens"]) for r in records
              if r.get("completion_tokens") is not None]
    all_dur = [int(r["duration_ms"]) for r in records
               if r.get("duration_ms") is not None]
    statistics = {
        "calls_total": len(records),
        "calls_failed": failed_count,
        "calls_aborted": aborted_count,
        "calls_skipped": skipped_count,
        "models_used": sorted({r.get("model") or "unknown" for r in records}),
        "prompt_tokens": _agg_dist(all_pt),
        "completion_tokens": _agg_dist(all_ct),
        "duration_ms": _agg_dist(all_dur),
    }

    aborted_calls = [
        {
            "call_id": r["call_id"],
            "stage": r.get("stage"),
            "category": r.get("category"),
            "model": r.get("model"),
            "started_at_iso": r.get("started_at_iso"),
            "duration_ms": r.get("duration_ms"),
        }
        for r in records if r.get("aborted")
    ]
    skipped_calls = [
        {
            "call_id": r["call_id"],
            "stage": r.get("stage"),
            "category": r.get("category"),
            "model": r.get("model"),
            "started_at_iso": r.get("started_at_iso"),
            "duration_ms": r.get("duration_ms"),
        }
        for r in records if r.get("skipped")
    ]

    # Coarse total wall-clock from run.started → now. Per-call
    # wall_clock is exact; this is the human-friendly run length.
    total_ms = _iso_delta_ms(started_at, ended_at)

    # cycles_count (issue #52): how many `runner.run()` cycles
    # contributed to this jsonl. Counted from `cycle_start` event
    # records on disk. `cycles_count > 1` is a visible signal that
    # the jsonl spans multiple resume cycles — the per-stage / totals
    # numbers reflect the cumulative event log (which is what the
    # materializer reads) so cross-cycle scope is consistent here.
    from engine.llm import count_cycle_starts_in_jsonl
    cycles_count = max(1, count_cycle_starts_in_jsonl(jsonl_path))

    # Warnings + cache stats live in the rollup so the UI's runs list
    # can read them post-#165 (run.json no longer carries the mid-run
    # mirror; the rollup is the canonical end-of-run record). See
    # `_leaf_aware_warnings` for why every bucket maps to leaf outcomes
    # — no attempt-based counters survive.
    rollup_warnings = dict(live_warnings_baseline or {})
    rollup_warnings.update(_leaf_aware_warnings(
        leaf_outcomes,
        leaf_input_overflows=_leaf_input_overflows,
    ))
    rollup_llm_cache = dict(llm_cache_stats or {})

    return {
        "schema": "llm-stats/v1",
        "run_id": run_id,
        "short_id": short_id,
        "agent": agent,
        "mode": mode_str,
        "primary_model": primary_model_resolved,
        "started_at_iso": started_at,
        "ended_at_iso": ended_at,
        "total_wall_clock_ms": total_ms,
        "cycles_count": cycles_count,
        "totals": totals,
        "by_stage": by_stage,
        "by_model": by_model,
        "per_stage": per_stage,
        "statistics": statistics,
        "aborted_calls": aborted_calls,
        "skipped_calls": skipped_calls,
        "calls": records,
        "warnings": rollup_warnings,
        "llm_cache": rollup_llm_cache,
    }


def _write_llm_stats(
    out_dir: "Path", mode, primary_model: str,
    run_started_at_iso: str,
) -> None:
    """End-of-run side-effects for the rollup. Calls
    `materialize_run_stats` to build the dict, mirrors leaf-aware
    warnings to in-memory state (so other in-process consumers see
    them), and writes the human-readable .txt summary sibling.

    Three files form the observability trio per run:

      llm-calls.jsonl   append-only event log (begin/end/counts).
                        Written by llm.begin_stat_record /
                        finalize_stat_record / record_stage_counts.
                        Survives SIGKILL — the on-disk truth.
      config.json       static start-of-run snapshot (run_id,
                        agent, mode, model, created_at, etc.)
      llm-stats.txt     human-readable monospace summary,
                        materialized at end-of-run.

    The rollup dict (llm-stats/v1 schema) is no longer cached as
    llm-stats.json — consumers (UI, eval/judge.py, debug bundles)
    derive it on-demand from llm-calls.jsonl + config.json via
    `materialize_run_stats(...)`. See issue #189.

    Idempotent — guarded by `_llm_stats_written` so a normal-end-of-run
    call followed by an atexit call writes once.

    In-flight calls at SIGTERM-cancel/pause time get `aborted: true`
    records with `error=None` (the call didn't fail — the run wound
    down) and a duration estimate derived from begin's started_at.
    `_classify_outcome` surfaces these as `OUTCOME_ABORTED`."""
    global _llm_stats_written
    if _llm_stats_written:
        return
    _llm_stats_written = True
    try:
        _do_write_llm_stats(out_dir, mode, primary_model,
                            run_started_at_iso)
    except BaseException as e:
        # atexit silently swallows exceptions. Record them.
        import traceback as _tb
        try:
            err_path = out_dir / "atexit-error.txt"
            err_path.write_text(
                f"_write_llm_stats raised:\n"
                f"  class: {type(e).__module__}.{type(e).__name__}\n"
                f"  message: {e}\n\n"
                f"{_tb.format_exc()}"
            )
        except Exception:
            pass
        try:
            _log_write(
                f"_write_llm_stats raised "
                f"{type(e).__name__}: {e} — see atexit-error.txt"
            )
        except Exception:
            pass
        # Re-raise so non-atexit callers see the failure (the
        # signal-handler inline call wraps this in another try
        # so it doesn't block exit).
        raise


def _do_write_llm_stats(
    out_dir: "Path", mode, primary_model: str,
    run_started_at_iso: str,
) -> None:
    """Inner body of _write_llm_stats. Split out so the caller can
    wrap in a try/except that records failures to disk; atexit
    silently swallows exceptions otherwise.

    Builds the rollup dict via `materialize_run_stats`, mirrors
    leaf-aware warnings to in-memory state, and writes the .txt
    summary. The dict itself is no longer persisted (issue #189) —
    on-demand readers re-derive it from disk."""
    _log_write("rollup: starting (read jsonl → materialize → write txt)")
    ended_at = _iso_z_full()
    jsonl_path = out_dir / "llm-calls.jsonl"
    config_path = out_dir / "config.json"

    # In-process state the pure function can't read from disk:
    # warnings + cache stats live in module globals, populated as
    # the run progressed. On-demand callers (eval/judge, Tauri) pass
    # None for these and accept the limitation that input_overflows
    # / cache stats reflect "post-rollup" view only on the writing
    # process.
    from engine.llm import get_call_warnings
    live_warnings_baseline = (
        dict(_run_state.get("warnings") or {}) if _run_state else {}
    )
    live_llm_cache = (
        dict(_run_state.get("llm_cache") or {}) if _run_state else {}
    )
    payload = materialize_run_stats(
        jsonl_path,
        config_path,
        ended_at_iso=ended_at,
        llm_cache_stats=live_llm_cache,
        call_warnings=get_call_warnings(),
        live_warnings_baseline=live_warnings_baseline,
        primary_model=primary_model,
        started_at_iso=run_started_at_iso,
    )
    _log_write(
        f"rollup: built payload ({len(payload.get('calls') or [])} "
        f"records from {jsonl_path.name})"
    )

    # Recompute leaf-aware warnings from the payload's calls list so
    # the in-memory mirror matches what consumers re-deriving from
    # disk would see. Costs ~one pass over `calls`; trivial vs. the
    # full materialize that just ran.
    leaf_outcomes_now: dict = {}
    leaf_ids_now: set = set()
    _child_now: set = set()
    for r in (payload.get("calls") or []):
        rof = r.get("retry_of_call_id")
        if rof is not None:
            _child_now.add(rof)
    for r in (payload.get("calls") or []):
        cid = r.get("call_id")
        if cid is None or cid in _child_now:
            continue
        leaf_ids_now.add(cid)
        oc = r.get("outcome")
        leaf_outcomes_now[oc] = leaf_outcomes_now.get(oc, 0) + 1
    _leaf_input_overflows_now = sum(
        1 for w in get_call_warnings()
        if w.get("kind") == "input_overflow"
        and w.get("call_id") in leaf_ids_now
    )
    _merge_run_warnings(_leaf_aware_warnings(
        leaf_outcomes_now,
        leaf_input_overflows=_leaf_input_overflows_now,
    ))

    # Sibling human-readable summary. The .txt is the only on-disk
    # rollup artifact post-#189 — it's part of debug bundles and
    # doubles as a quick `cat`-able summary on a run dir.
    txt_target = out_dir / "llm-stats.txt"
    txt_target.write_text(_render_llm_stats_text(payload))
    _log_write(f"rollup: wrote {txt_target.name}")

    totals = payload.get("totals") or {}
    _log_write(
        f"LLM stats: {totals.get('calls', 0)} calls "
        f"({totals.get('successful', 0)} ok, {totals.get('failed', 0)} failed, "
        f"{totals.get('aborted', 0)} aborted, "
        f"{totals.get('skipped', 0)} skipped) — "
        f"{(totals.get('prompt_tokens') or 0):,} in / "
        f"{(totals.get('completion_tokens') or 0):,} out — "
        f"wrote {txt_target.name}"
    )


def _fmt_int(n: int | None) -> str:
    """Comma-grouped integer; None → '—'."""
    return "—" if n is None else f"{int(n):,}"


def _fmt_num(n) -> str:
    """Human-friendly numeric (avg/median may be float). None → '—'.
    Floats that are integral collapse to no-decimal form."""
    if n is None:
        return "—"
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if isinstance(n, int):
        return f"{n:,}"
    return f"{n:,.2f}"


def _fmt_duration_ms(ms: int | None) -> str:
    """Render a millisecond count as h?m?s/ms for the .txt summary."""
    if ms is None:
        return "—"
    s = ms / 1000.0
    if s < 1:
        return f"{int(ms)}ms"
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        m, sec = divmod(int(s), 60)
        return f"{m}m {sec}s"
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m}m {sec}s"


def _fmt_iso_local(iso: str | None) -> str:
    """ISO-Z → 'YYYY-MM-DD HH:MM:SS TZ' in local time. Best-effort —
    falls back to the raw string if parse fails."""
    if not iso:
        return "—"
    try:
        import datetime as _dt
        s = iso.replace("Z", "+00:00")
        utc = _dt.datetime.fromisoformat(s)
        local = utc.astimezone()
        tz = local.tzname() or ""
        return local.strftime("%Y-%m-%d %H:%M:%S") + (f" {tz}" if tz else "")
    except Exception:
        return iso


def _render_llm_stats_text(payload: dict) -> str:
    """Plain-text monospace summary of a run's llm-stats payload.

    Format: header (run id, status, duration, mode), per-stage table,
    models-used count, in-flight aborted list. Targeting a 120-char
    terminal — column widths are deliberate, not auto-fit, so the
    output is stable across runs (snapshot-friendly).

    The snapshot test in test_llm_stats_text.py pins this format; if
    you change column widths or labels, update the expected output."""
    lines: list[str] = []
    short_id = payload.get("short_id") or payload.get("run_id") or "—"
    status_dur = payload.get("total_wall_clock_ms")
    # The header reports the run mode only. The "Models used" table
    # below enumerates what actually ran per call — when the user
    # overrides per-stage models, a preset id in the header would
    # claim a config that wasn't honored.
    mode = payload.get("mode") or "—"

    totals = payload.get("totals") or {}
    # Heuristic: any aborted call ⇒ the run wound down before all
    # work finished, so the .txt header surfaces it as "cancelled"
    # status (run-level vocabulary). Aborted is the per-call axis;
    # the run-level "cancelled" status string is unchanged.
    aborted = totals.get("aborted", 0)
    failed = totals.get("failed", 0)
    if aborted > 0:
        status = "cancelled"
    elif failed > 0:
        status = "completed (with failures)"
    else:
        status = "completed"

    lines.append("BaseVault run summary")
    lines.append("=" * 21)
    lines.append(f"Run id:    {short_id}")
    lines.append(f"Status:    {status}")
    lines.append(f"Started:   {_fmt_iso_local(payload.get('started_at_iso'))}")
    lines.append(f"Ended:     {_fmt_iso_local(payload.get('ended_at_iso'))}")
    lines.append(f"Duration:  {_fmt_duration_ms(status_dur)}")
    lines.append(f"Mode:      {mode}")
    cycles_count = payload.get("cycles_count", 1)
    if cycles_count > 1:
        # Multi-cycle warning (issue #52): the numbers below describe
        # ONLY the latest cycle, while llm-calls.jsonl on disk has
        # events from every cycle. Splitting by `cycle_start` events
        # is the supported way to scope-match per cycle.
        lines.append(
            f"Cycles:    {cycles_count}  (jsonl spans every cycle; "
            f"stats below describe the latest cycle only)"
        )
    lines.append("")

    # Per-stage table.
    per_stage = payload.get("per_stage") or {}
    if per_stage:
        lines.append("Per-stage statistics")
        lines.append("-" * 20)
        # Stage column sized to the longest stage name + 1 so longer
        # names (e.g. `entities_dedupe`, 15 chars) don't push the
        # numeric columns rightward and break alignment.
        stage_w = max(len("stage"), max(len(s) for s in per_stage.keys())) + 1
        # `total_completion` is the new label for completion_tokens
        # (issue #104 part 1) — it's now the sum of content + reasoning
        # tokens per call, so the rename clarifies what it means when
        # the reasoning column is non-zero next to it.
        # Numbers right-aligned for column scanning.
        header = (
            f"{'stage':<{stage_w}}"
            f"{'calls':>6}"
            f"{'ok':>5}"
            f"{'fail':>5}"
            f"{'abrt':>5}"
            f"{'skip':>5}"
            f"  "
            f"{'prompt_tokens (avg / med / min / max)':<46}"
            f"{'total_completion (avg / med / min / max)':<46}"
            f"{'reasoning (sum)':>16}"
            f"  "
            f"{'duration (avg / med / min / max)':<36}"
            f"{'ttft p50':>10}"
        )
        lines.append(header)
        for s, b in per_stage.items():
            ok = (b["calls_total"] - b["calls_failed"]
                  - b["calls_aborted"] - b.get("calls_skipped", 0))
            pt = b["prompt_tokens"]
            ct = b["completion_tokens"]
            dur = b["duration_ms"]
            rt = b.get("reasoning_tokens") or {}
            ttft = b.get("ttft_ms") or {}
            pt_s = (
                f"{_fmt_num(pt['avg'])} / {_fmt_num(pt['median'])} "
                f"/ {_fmt_int(pt['min'])} / {_fmt_int(pt['max'])}"
            )
            ct_s = (
                f"{_fmt_num(ct['avg'])} / {_fmt_num(ct['median'])} "
                f"/ {_fmt_int(ct['min'])} / {_fmt_int(ct['max'])}"
            )
            rt_total = rt.get("total") if isinstance(rt, dict) else None
            rt_s = _fmt_int(rt_total) if rt_total else "—"
            dur_s = (
                f"{_fmt_duration_ms(int(dur['avg'])) if dur['avg'] is not None else '—'}"
                f" / {_fmt_duration_ms(int(dur['median'])) if dur['median'] is not None else '—'}"
                f" / {_fmt_duration_ms(dur['min'])} / {_fmt_duration_ms(dur['max'])}"
            )
            ttft_med = ttft.get("median") if isinstance(ttft, dict) else None
            ttft_s = _fmt_duration_ms(int(ttft_med)) if ttft_med is not None else "—"
            lines.append(
                f"{s:<{stage_w}}"
                f"{b['calls_total']:>6}"
                f"{ok:>5}"
                f"{b['calls_failed']:>5}"
                f"{b['calls_aborted']:>5}"
                f"{b.get('calls_skipped', 0):>5}"
                f"  "
                f"{pt_s:<46}"
                f"{ct_s:<46}"
                f"{rt_s:>16}"
                f"  "
                f"{dur_s:<36}"
                f"{ttft_s:>10}"
            )
        lines.append("")

    # Models used (count of calls per model). Annotated with a
    # reasoning marker when at least one stage that called this model
    # had reasoning=True — material to cost/quality and the user's
    # primary lever for runs feeling "slow / expensive".
    by_model = payload.get("by_model") or {}
    if by_model:
        per_stage = payload.get("per_stage") or {}
        models_with_reasoning: set[str] = set()
        reasoning_stages_by_model: dict[str, list[str]] = {}
        for s, b in per_stage.items():
            if not b.get("reasoning_enabled"):
                continue
            for m in b.get("models_used") or []:
                models_with_reasoning.add(m)
                reasoning_stages_by_model.setdefault(m, []).append(s)
        lines.append("Models used")
        lines.append("-" * 11)
        # Model column sized to the longest model id + 1 so vendor-
        # qualified ids (e.g. `mistralai/...`) don't push the call
        # counts rightward.
        model_w = max(25, max(len(m) for m in by_model.keys()) + 1)
        for m, b in by_model.items():
            tag = ""
            if m in models_with_reasoning:
                stages = reasoning_stages_by_model.get(m) or []
                tag = f"   ← reasoning ON ({', '.join(stages)})"
            lines.append(f"{m:<{model_w}} {b.get('calls', 0):>5} calls{tag}")
        lines.append("")

    # In-flight aborted list — the smoking gun for "what was running
    # when the run wound down".
    aborted_calls = payload.get("aborted_calls") or []
    if aborted_calls:
        lines.append("Calls in flight when run wound down")
        lines.append("-" * 35)
        # Stage / model columns sized to the longest values present, so
        # `entities_dedupe` or vendor-qualified model ids don't push
        # later fields rightward and break alignment between rows.
        c_stage_w = max(10, max(len(c.get("stage") or "—") for c in aborted_calls))
        c_model_w = max(20, max(len(c.get("model") or "—") for c in aborted_calls))
        for c in aborted_calls:
            lines.append(
                f"call {c['call_id']:>4}  stage={c.get('stage') or '—':<{c_stage_w}}  "
                f"model={c.get('model') or '—':<{c_model_w}}  "
                f"started={_fmt_iso_local(c.get('started_at_iso'))}  "
                f"in-flight={_fmt_duration_ms(c.get('duration_ms'))}"
            )
        lines.append("")

    # Calls the user explicitly skipped via the run details modal ✕
    # control (issue #333). Distinct section from aborted because the
    # cause is different — human intervention vs. run wind-down — and
    # the operator may want to scan the list separately to confirm
    # what was pruned.
    skipped_calls = payload.get("skipped_calls") or []
    if skipped_calls:
        lines.append("Calls skipped by user")
        lines.append("-" * 21)
        c_stage_w = max(10, max(len(c.get("stage") or "—") for c in skipped_calls))
        c_model_w = max(20, max(len(c.get("model") or "—") for c in skipped_calls))
        for c in skipped_calls:
            lines.append(
                f"call {c['call_id']:>4}  stage={c.get('stage') or '—':<{c_stage_w}}  "
                f"model={c.get('model') or '—':<{c_model_w}}  "
                f"started={_fmt_iso_local(c.get('started_at_iso'))}  "
                f"in-flight={_fmt_duration_ms(c.get('duration_ms'))}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _iso_delta_ms(start_iso: str | None, end_iso: str | None) -> int | None:
    """Best-effort millisecond delta between two ISO-Z timestamps.
    Returns ``None`` if either is missing or unparseable (defensive —
    ``total_wall_clock_ms`` isn't load-bearing). Thin int-cast wrapper
    around ``common.dates.iso_delta_ms``."""
    from engine.common.dates import iso_delta_ms as _common_delta
    delta = _common_delta(start_iso, end_iso)
    return int(delta) if delta is not None else None


class NoResumableCheckpoint(ValueError):
    """Raised by _detect_resume_point when a resume was attempted on a run
    dir that contains no intermediate artifacts at all (SIGKILL hit before
    even the first per-split partial was written).

    main() catches this specifically and reverts the run to `paused`
    rather than marking it terminally `failed`. Subclassing ValueError
    keeps backward compatibility with any caller that did
    `except ValueError`. The Rust shell's resume_run accepts both
    paused and failed, but `paused` carries the right intent — "no
    progress yet, restarting will redo this stage from scratch" — so
    the UI's pause/resume affordance makes sense to the user.
    """


# Constant message surfaced as the cycle_error event's message field
# (post-#165) so the UI can render a friendly explanation. Kept as a
# module-level string so tests can pin against it without duplicating
# the wording.
_NO_CHECKPOINT_USER_MSG = (
    "No resumable progress yet — restart will redo this stage from scratch."
)


def _store_chunk_count(db_path: Path) -> int:
    """Number of chunk-kind records in a `vectors.db`, or 0 if the
    store / its `records` table is absent. Plain sqlite COUNT — keeps
    the resume-detection path off the sqlite-vec extension load that
    `VectorStore.open` does (resume detection is a read-only existence
    probe and runs before the embedding deps are needed)."""
    if not db_path.exists():
        return 0
    import sqlite3
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM records WHERE kind = 'chunk'"
            ).fetchone()
        finally:
            conn.close()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def _ingestion_marker_documents(run_dir: Path) -> list[dict]:
    """The `documents` manifest the ingestion stage persisted to
    stages/00-ingestion/phase_1_marker.json, or [] when the marker is
    absent / unreadable.

    Written on every fresh run before extraction and never rewritten
    by a resume, so it is the resume-DURABLE record of what was
    ingested — unlike the documents/<file_id>.md snapshots, which only
    exist if `preprocess()` ran this run dir's original fresh pass.
    Both the resume-point chunk-coverage gate and the snapshotless
    chunk-source reload key off this so a legacy / snapshotless /
    split-multi-source run dir is caught, not whitelisted."""
    marker = run_dir / "stages" / "00-ingestion" / "phase_1_marker.json"
    if not marker.is_file():
        return []
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    docs = payload.get("documents") if isinstance(payload, dict) else None
    return docs if isinstance(docs, list) else []


def _chunks_expected(run_dir: Path) -> bool:
    """Whether the embeddings stage should produce chunk-kind records
    for this run dir.

    Union of two signals so the gate neither regresses nor keeps the
    #562 blind spot:
      * the resume-DURABLE ingestion marker lists ≥1 content-bearing
        document — catches legacy / snapshotless / split-multi-source
        dirs (no `documents/*.md`), the #562 fix; and
      * a `documents/*.md` snapshot exists — the original #520 signal,
        preserved so a dir that has the snapshot but (e.g.) a
        truncated / absent ingestion marker is still not whitelisted.
    Either signal ⇒ a chunkless store is incomplete, not "done"."""
    if any(int(d.get("content_len", 0)) > 0
           for d in _ingestion_marker_documents(run_dir)):
        return True
    docs_dir = run_dir / "stages" / "00-ingestion" / "documents"
    return docs_dir.is_dir() and any(docs_dir.glob("*.md"))


def _detect_resume_point(run_dir: Path) -> str:
    """Inspect an existing run dir and return the first stage that needs to run.

    Resume granularity is the stage boundary. Returns one of: "start",
    "entities", "patterns", "insights", "actions", "embeddings", "done".

    "start" means restart the pipeline from ingestion. The LLM cache
    accelerates re-execution for any prior call whose inputs are
    deterministic. We don't try to skip mid-extract — if extract didn't
    finish, we re-run ingest+split (fast) + extract (cache-hot).

    The embeddings marker's presence alone does NOT prove a complete
    store: an embeddings pass over an empty document set (the shape a
    resume produced before docs were reloaded on the resume path)
    writes the marker with zero chunk records. So when ingestion
    persisted source documents (chunks are expected) but the store has
    no chunk records, treat embeddings as unfinished and re-run it —
    this also self-heals run dirs already poisoned that way.

    Raises NoResumableCheckpoint when the run dir has no phase markers
    at all — preventing accidental "resume" of an empty dir.
    """
    stages = run_dir / "stages"

    if not stages.exists():
        raise NoResumableCheckpoint(
            f"No stages dir in {run_dir}; nothing to resume from."
        )

    embeddings = stages / "06-embeddings"
    actions = stages / "05-actions"
    insights = stages / "04-insights"
    patterns = stages / "03-patterns"
    entities = stages / "02-entities"
    extract = stages / "01-extraction"

    if (embeddings / "phase_1_marker.json").exists():
        # Chunk-coverage gate. `_chunks_expected` keys off the
        # resume-DURABLE ingestion marker (∪ the legacy documents/*.md
        # signal), not the fresh-run-only snapshot alone. The prior
        # snapshot-only predicate whitelisted exactly the legacy /
        # snapshotless / split-multi-source dirs that embed 0 chunks:
        # no snapshot ⇒ "expected" went False ⇒ a chunkless store was
        # accepted as "done" forever. The union predicate catches them
        # and forces an embeddings re-run, which now reconstructs chunk
        # source from the durable artifact (see `_load_documents`).
        if _chunks_expected(run_dir) and _store_chunk_count(embeddings / "vectors.db") == 0:
            return "embeddings"
        return "done"
    if (actions / "phase_1_marker.json").exists():
        return "embeddings"
    if (insights / "phase_1_marker.json").exists():
        return "actions"
    if (patterns / "phase_1_marker.json").exists():
        return "insights"
    if (entities / "phase_3_marker.json").exists():
        return "patterns"
    if (extract / "phase_3_marker.json").exists():
        return "entities"

    # Nothing past extract: restart from ingest + cache makes it fast.
    if any(stages.glob("*/phase_*_marker.json")):
        return "start"

    raise NoResumableCheckpoint(
        f"No resumable checkpoint in {run_dir}. "
        f"No phase markers under stages/."
    )


# Per-topic JSONL writes are append-only and concurrent (one writer
# thread per parent in the extract pool). Each topic's lock serializes
# appends; we hold one master lock just to vivify the per-topic lock on
# first encounter. POSIX small writes are atomic but JSON lines for
# fat items can exceed PIPE_BUF, so don't trust naked O_APPEND.
_TOPIC_LOCKS_LOCK = threading.Lock()
_TOPIC_LOCKS: dict[str, threading.Lock] = {}


def _topic_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _TOPIC_LOCKS_LOCK:
        lock = _TOPIC_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _TOPIC_LOCKS[key] = lock
        return lock


def _append_fact_jsonl(path: Path, item_dict: dict) -> None:
    """Thread-safe append of one fact (as a single JSON line) to a
    per-topic JSONL bucket file under stages/01-extraction/facts/."""
    line = json.dumps(item_dict, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _topic_lock(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _append_entity_jsonl(path: Path, mention_dict: dict) -> None:
    """Thread-safe append of one per-fact entity mention to a per-entity
    JSONL bucket file under stages/02-entities/entities/. Reuses the
    per-path topic-lock pool — concurrent writers across extract threads
    serialize on the file's lock, mirroring the facts dump."""
    line = json.dumps(mention_dict, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _topic_lock(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _serialize_entity_mention_dict(it, ref) -> dict:
    """One per-fact entity mention as it appears on a line in
    `02-entities/entities/<slug>.jsonl` during Stage 1 Phase 2.
    Carries the entity's identity (name + type + role within this
    fact), the fact's citation surface (topic + summary + occurred_at
    + evidence), the relation_candidate if any, and a co_entities
    sketch so the UI can show context without re-joining against the
    facts file. Stage 2 Phase 1 consumes only the canonical
    post-Phase-3 facts view and rewrites the same .jsonl files with
    one consolidated record per entity; this multi-line stream is
    the in-flight signal that lights up the run tree mid-Stage-1."""
    return {
        "name": ref.entity.name,
        "entity_type": ref.entity.entity_type,
        "role": ref.role,
        "topic": (it.topics[0] if it.topics else ""),
        "topics": list(it.topics or []),
        "fact_summary": it.summary,
        "occurred_at": it.occurred_at,
        # Per-fact extraction confidence is already in the canonical
        # facts file; surface it here too so the consolidating entity
        # view can color the live-mention dot.
        "confidence": it.confidence,
        "evidence": [
            {"text": e.text,
             "ref": e.source_ref,
             "start_char": e.start_char,
             "end_char": e.end_char,
             "file_path": e.file_path,
             "file_offset": e.file_offset,
             "file_length": e.file_length,
             "approximate": e.approximate}
            for e in it.evidence
        ],
        "candidate_relation": it.relation_candidate,
        "co_entities": [
            {"name": r.entity.name,
             "entity_type": r.entity.entity_type,
             "role": r.role}
            for r in it.entities if r is not ref
        ],
    }


def _serialize_fact_dict(it) -> dict:
    """Schema for one ExtractedItem on disk (one line in TOPIC.jsonl,
    one element in phase_3_marker.json's payload). Mirrors the legacy
    per-parent shape so downstream readers don't fork."""
    return {
        "type": it.item_type,
        "summary": it.summary,
        "occurred_at": it.occurred_at,
        "occurred_at_text": it.occurred_at_text,
        "entities": [
            {"name": r.entity.name, "type": r.entity.entity_type, "role": r.role}
            for r in it.entities
        ],
        "evidence": [
            {"text": e.text,
             "ref": e.source_ref,
             "start_char": e.start_char,
             "end_char": e.end_char,
             "file_path": e.file_path,
             "file_offset": e.file_offset,
             "file_length": e.file_length,
             "approximate": e.approximate}
            for e in it.evidence
        ],
        "topics": it.topics,
        "affect": it.affect,
        "tags": it.tags,
        "confidence": it.confidence,
        "relation_candidate": it.relation_candidate,
    }


def _fact_sort_key(d: dict) -> tuple:
    """Sort key for a fact dict (file_path, file_offset, summary).
    Mirrors the in-memory _sort_key used during the canonical write."""
    ev = d.get("evidence") or []
    first = ev[0] if ev else {}
    return (
        first.get("file_path") or "",
        first.get("file_offset") or 0,
        d.get("summary", ""),
    )


def _parse_jsonl_facts_tolerant(jsonl_path: Path) -> list[dict]:
    """Parse a per-topic facts JSONL, skipping malformed lines.

    A process kill mid-write of `_append_fact_jsonl` can leave a
    truncated line in an otherwise well-formed file. The strict
    `[json.loads(l) for l ...]` form aborts the run on the first such
    line, throwing away potentially hours of LLM work upstream. This
    variant catches `json.JSONDecodeError`, logs the byte offset + a
    head excerpt of the bad line via `_log_write`, and returns the
    surviving records."""
    raw = jsonl_path.read_text(encoding="utf-8")
    out: list[dict] = []
    offset = 0
    for line in raw.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped.strip():
            try:
                out.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                excerpt = stripped[:80].replace("\n", "\\n")
                _log_write(
                    f"extract phase 3: skipping malformed JSONL line in "
                    f"{jsonl_path.name} at byte offset {offset} "
                    f"({e.msg}): {excerpt!r}"
                )
        offset += len(line.encode("utf-8"))
    return out


def _repair_partial_jsonl_tails(facts_dir: Path) -> None:
    """Idempotent sanity sweep run before Phase 2 starts appending.

    A process kill mid-write of `_append_fact_jsonl` can leave a
    per-topic JSONL whose last line is missing its trailing newline
    (and possibly truncated mid-token). The next append would
    concatenate into that half-line, producing a corrupt record that
    parses as JSON but carries garbage. Truncating to the last `\\n`
    (or to 0 if none is present) restores the well-formed-line
    invariant. No-op on well-formed files (last byte is `\\n`)."""
    if not facts_dir.exists():
        return
    for p in sorted(facts_dir.glob("*.jsonl")):
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size == 0:
            continue
        with open(p, "rb") as f:
            f.seek(-1, 2)
            last = f.read(1)
        if last == b"\n":
            continue
        with open(p, "rb") as f:
            data = f.read()
        idx = data.rfind(b"\n")
        new_size = idx + 1 if idx >= 0 else 0
        with open(p, "r+b") as f:
            f.truncate(new_size)
        dropped = size - new_size
        _log_write(
            f"extract phase 2 entry: repaired truncated JSONL tail in "
            f"{p.name} (dropped {dropped} bytes; new size {new_size})"
        )


def _load_facts_by_topic(run_dir: Path):
    """Reconstruct facts_by_topic from the per-topic JSONL bucket files
    under stages/01-extraction/facts/. Returns dict[str, list[ExtractedItem]].
    Used on resume from stages downstream of extraction."""
    from engine.content_extractor import ExtractedItem, EvidenceSpan, Entity, EntityRef

    facts_dir = run_dir / "stages" / "01-extraction" / "facts"
    result: dict[str, list] = {}
    if not facts_dir.exists():
        return result
    for jsonl_path in sorted(facts_dir.glob("*.jsonl")):
        topic = jsonl_path.stem
        items = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                it = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            # Per-topic JSONL uses the per-parent shorter schema
            # ("type", "ref"). Accept the legacy facts_by_topic.json
            # field names ("item_type", "source_ref") too so older
            # runs still load.
            evidence = [
                EvidenceSpan(
                    text=e.get("text", ""),
                    source_ref=e.get("ref") or e.get("source_ref") or "",
                    start_char=e.get("start_char"),
                    end_char=e.get("end_char"),
                    file_path=e.get("file_path"),
                    file_offset=e.get("file_offset"),
                    file_length=e.get("file_length"),
                    approximate=e.get("approximate", False),
                )
                for e in it.get("evidence", [])
            ]
            entities = [
                EntityRef(
                    entity=Entity(name=r["name"], entity_type=r["type"]),
                    role=r.get("role", "mentioned"),
                )
                for r in it.get("entities", [])
            ]
            items.append(ExtractedItem(
                item_type=it.get("type") or it.get("item_type"),
                summary=it["summary"],
                evidence=evidence,
                occurred_at=it.get("occurred_at"),
                occurred_at_text=it.get("occurred_at_text"),
                entities=entities,
                topics=it.get("topics", []),
                affect=it.get("affect", []),
                tags=it.get("tags", []),
                confidence=it.get("confidence", 1.0),
                relation_candidate=it.get("relation_candidate"),
            ))
        result[topic] = items
    return result


def _load_documents(run_dir: Path):
    """Reconstruct ingested Documents for the resume-path chunk plan.
    Returns list[Document].

    Resume into any post-ingestion stage skips the ingest path that
    populates `all_docs`; without reconstruction the embeddings plan
    sees zero source documents and silently embeds a chunkless store.

    Preferred source is the preprocessed markdown snapshot under
    stages/00-ingestion/documents/: documents/<file_id>.md IS the
    canonical text the chunker saw (per `preprocessor`), and a
    multi-Document source (e.g. a Day One bundle) was merged into one
    .md, so chunking it reproduces the offset frame the rest of the
    resume path (fact-evidence anchors) already uses.

    That snapshot only exists if THIS run dir's original fresh pass
    ran `preprocess()` — dirs from builds predating the snapshot
    writer, and any dir whose snapshot was lost, have none. Keying the
    resume solely on it is the root cause of the 0-chunks store: no
    snapshot ⇒ [] ⇒ chunkless store marked complete. So when the
    snapshot is absent we reconstruct from the resume-DURABLE
    ingestion marker by re-ingesting each recorded `source_path`
    (deterministic — re-runs the same parse + Day One year-split a
    fresh run would). Re-ingest yields per-source offsets rather than
    the merged-snapshot frame; that is correct for chunk retrieval and
    strictly better than the chunkless store the snapshotless path
    produced before. (Re-ingest uses the default vision mode: image
    sub-docs come back as placeholders, an accepted degradation for
    snapshotless legacy dirs — the common journal/text case is exact;
    a chunkless store was the alternative.)"""
    from engine.ingestor import Document, SourceType

    docs_dir = run_dir / "stages" / "00-ingestion" / "documents"
    out: list = []
    if docs_dir.is_dir():
        for md_path in sorted(docs_dir.glob("*.md")):
            file_id = md_path.stem
            out.append(Document(
                id=file_id,
                source_path=str(md_path),
                source_type=SourceType.MD_FILE,
                content=md_path.read_text(encoding="utf-8"),
                title=file_id,
                file_id=file_id,
                origin_char=0,
            ))
    if out:
        return out

    # Snapshotless fallback: re-ingest from the durable ingestion
    # manifest's source paths. Distinct paths only — a Day One bundle
    # lists one source_path per entry, but re-ingesting that file once
    # reproduces every entry (and the year-split) itself.
    from engine.ingestor import ingest as _ingest

    seen: set[str] = set()
    for d in _ingestion_marker_documents(run_dir):
        sp = str(d.get("source_path") or "")
        if not sp or sp in seen:
            continue
        seen.add(sp)
        if not Path(sp).exists():
            _log_write(
                f"  resume chunk-source: original input no longer on "
                f"disk, cannot reconstruct: {sp}")
            continue
        try:
            out.extend(_ingest(sp))
        except Exception as e:  # noqa: BLE001 — degrade, never crash resume
            _log_write(
                f"  resume chunk-source: re-ingest failed for {sp}: "
                f"{type(e).__name__}: {e}")
    if out:
        _log_write(
            f"  resume chunk-source: snapshot absent — reconstructed "
            f"{len(out)} document(s) by re-ingesting "
            f"{len(seen)} source file(s)")
    return out


# ── Phase-boundary normalization (issue #339) ──────────────────────────────
#
# Every cross-phase data shape goes through a normalizer before being
# consumed. The fresh in-memory build path and the resume-from-disk
# load path produce structurally equivalent payloads with slightly
# different default orderings (insertion order vs alphabetical glob
# order vs marker-write order). Without a single canonical ordering at
# each boundary, downstream prompt construction (which iterates these
# shapes verbatim) emits prompts whose bytes drift across a
# pause-and-resume even though the facts on disk are unchanged — and
# the prompt-hash cache misses.
#
# Each helper is idempotent: passing already-normalized input returns
# the same shape. They produce NEW containers rather than mutating in
# place, so callers can hold references to the un-normalized form
# without surprise. Sort keys are picked for stability + readability
# (canonical_id where present, otherwise canonical_name); ties broken
# by additional fields so the order is fully total.


def _normalize_facts_by_topic(facts_by_topic: dict) -> dict:
    """Topic keys alphabetized; items within each topic sorted by
    (file_path, file_offset, summary) — same key the canonical extract
    write uses on disk. Used at entry to entities / patterns / actions
    so the topic-iteration order downstream stages observe is identical
    on fresh and on resume."""
    def _item_key(it):
        ev = it.evidence[0] if it.evidence else None
        return (
            (ev.file_path or "") if ev else "",
            (ev.file_offset if ev and ev.file_offset is not None else 0),
            it.summary,
        )
    return {
        topic: sorted(facts_by_topic[topic], key=_item_key)
        for topic in sorted(facts_by_topic.keys())
    }


def _normalize_entities_output(eo):
    """Entities sorted by canonical_id; relations sorted by
    (from_id, to_id, relation); evidence_fact_refs lists sorted
    within each record. Subject untouched (it's a single ref).
    Used at entry to patterns / insights / actions so the order of
    iteration over entities + relations is identical regardless of
    whether `eo` came from the in-memory `detect_entities` return or
    from `_load_entities`'s marker read."""
    if eo is None:
        return None
    from dataclasses import replace
    sorted_entities = [
        replace(
            e,
            aliases=sorted(e.aliases),
            topics=sorted(e.topics),
            evidence_fact_refs=sorted(e.evidence_fact_refs),
        )
        for e in sorted(eo.entities, key=lambda e: e.canonical_id)
    ]
    sorted_relations = [
        replace(
            r,
            evidence_fact_refs=sorted(r.evidence_fact_refs),
        )
        for r in sorted(
            eo.relations,
            key=lambda r: (r.from_id, r.to_id, r.relation),
        )
    ]
    return replace(eo, entities=sorted_entities, relations=sorted_relations)


def _sort_patterns_within_topic(pats: list) -> list:
    """Most-grounded patterns first: fact count descending, name tiebreak.
    Drives both the on-disk per-topic JSON and the in-memory dict
    consumed by insights / actions / vault render.

    Keyed off `len(source_facts)` rather than the LLM-emitted `count`
    field: source_facts is the authoritative downstream weight after
    citation resolution, the two can drift when refs are hallucinated
    or deduped, and the bug this sort fixes is about backed-by-facts
    grounding, not the LLM's self-estimate."""
    return sorted(pats, key=lambda p: (-len(p.source_facts), p.name))


def _normalize_patterns_by_topic(patterns_by_topic: dict) -> dict:
    """Topic keys alphabetized; patterns within each topic sorted via
    `_sort_patterns_within_topic` so prompt construction in insights /
    actions sees a stable, most-grounded-first order across fresh and
    resume."""
    return {
        topic: _sort_patterns_within_topic(patterns_by_topic[topic])
        for topic in sorted(patterns_by_topic.keys())
    }


def _normalize_insight_output(io):
    """Order-preserving by design: we intentionally keep the insights
    LLM's emission order rather than imposing a sort, because we treat
    that emission order as already ranked by importance (most
    significant insight first). This is the one phase-boundary
    normalizer that is a deliberate no-op on ordering — facts /
    entities / patterns / actions each sort by a content-derived
    "it matters" key (provenance, id, grounding count, score), but for
    insights the model's own sequence IS that signal, so the old
    by-name canonicalization only destroyed it (alphabetical order
    carries no importance).

    Kept as a named hook rather than deleted so the phase-boundary
    normalization story stays symmetric across stages and this
    decision is documented at the seam. Emission order round-trips
    faithfully through the insights marker (JSON array order), so fresh
    and resume still feed the actions stage the identical sequence —
    the prompt-hash and the (scope, idx) index space stay stable across
    resume exactly as before."""
    if io is None:
        return None
    return io


def _normalize_action_list(actions):
    """Action list sorted by score descending — the same deterministic
    ordering the fresh parser (`actions._parse_output`) enforces before
    the marker is written. Routing both the fresh and resume-loaded
    lists through this collapses them onto one canonical shape so the
    downstream embeddings stage sees identical input + cache keys
    regardless of fresh-vs-resume (Issue #339 phase-boundary parity).
    Stable: ties keep input order, so this is idempotent on the
    already-sorted fresh list and yields the same order on resume
    (the marker preserves the fresh sort order)."""
    if actions is None:
        return None
    return sorted(actions, key=lambda a: -a.score)


def _load_entities(run_dir: Path):
    """Load EntitiesOutput from stages/02-entities/phase_3_marker.json.
    Returns None when the marker doesn't exist (older runs / resume
    before entities stage finished)."""
    from engine.entities import EntitiesOutput, EntityRecord, RelationEdge, SubjectRef

    path = run_dir / "stages" / "02-entities" / "phase_3_marker.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)

    subject = None
    subj_raw = data.get("subject") or None
    if isinstance(subj_raw, dict) and subj_raw.get("canonical_id"):
        subject = SubjectRef(
            canonical_id=str(subj_raw["canonical_id"]),
            display=str(subj_raw.get("display", subj_raw["canonical_id"])),
            source=str(subj_raw.get("source", "unknown")),
        )

    entities = []
    for e in data.get("entities", []):
        entities.append(EntityRecord(
            canonical_id=e["canonical_id"],
            canonical_name=e.get("canonical_name", e["canonical_id"]),
            entity_type=e.get("entity_type", "other"),
            aliases=list(e.get("aliases", [])),
            role=e.get("role", ""),
            description=e.get("description", ""),
            mention_count=int(e.get("mention_count", 0)),
            topics=list(e.get("topics", [])),
            evidence_fact_refs=[tuple(r) for r in e.get("evidence_fact_refs", [])],
        ))

    relations = []
    for r in data.get("relations", []):
        relations.append(RelationEdge(
            from_id=r.get("from") or r.get("from_id", ""),
            to_id=r.get("to") or r.get("to_id", ""),
            relation=r.get("relation", ""),
            confidence=float(r.get("confidence", 1.0)),
            evidence_fact_refs=[tuple(x) for x in r.get("evidence_fact_refs", [])],
        ))

    return EntitiesOutput(subject=subject, entities=entities, relations=relations)


def _load_patterns(out_dir: Path):
    """Load patterns_by_topic from per-topic JSON files under
    stages/03-patterns/patterns/<TOPIC>.json. The stage marker
    (phase_1_marker.json) lives at the stage root and is skipped
    here by virtue of looking only inside the patterns/ subdir."""
    from engine.patterns import Pattern

    patterns_dir = out_dir / "stages" / "03-patterns" / "patterns"
    result: dict[str, list[Pattern]] = {}
    if not patterns_dir.is_dir():
        return result
    for p in sorted(patterns_dir.glob("*.json")):
        topic = p.stem
        with open(p) as f:
            data = json.load(f)
        result[topic] = [
            Pattern(
                name=x["name"],
                description=x.get("description", ""),
                domain=topic,
                kind=x.get("kind"),
                count=int(x.get("count", 1)),
                source_facts=[tuple(s) for s in x.get("source_facts", [])],
                hallucinated_ref_count=x.get("hallucinated_ref_count", 0),
            )
            for x in data
        ]
    return result


def _load_insights(out_dir: Path):
    """Load InsightOutput from the insights stage's phase_1_marker.json
    (which carries the full payload per the doc — no separate
    insights.json file)."""
    from engine.insights import Insight, InsightOutput

    path = out_dir / "stages" / "04-insights" / "phase_1_marker.json"
    with open(path) as f:
        data = json.load(f)

    def _reconstruct(entries, kind):
        return [
            Insight(
                name=x["name"],
                description=x.get("description", ""),
                mechanism=x.get("mechanism", ""),
                implication=x.get("implication", ""),
                domains=x.get("domains", []),
                kind=kind,
                proposed_actions=x.get("proposed_actions", []),
                source_patterns=[tuple(s) for s in x.get("source_patterns", [])],
                hallucinated_ref_count=x.get("hallucinated_ref_count", 0),
            )
            for x in entries
        ]

    return InsightOutput(
        cross_domain=_reconstruct(data.get("cross_domain", []), "cross_domain"),
        critical=_reconstruct(data.get("critical", []), "critical"),
    )


def _load_actions(out_dir: Path):
    """Load the action list from the actions stage's phase_1_marker.json
    (which carries the full payload per the doc — no separate
    actions.json file). Reconstructs the type-identical dataclass list
    `generate_actions` returns: `source_insights` is re-tupled the same
    way the sibling insight loader handles its ref field, and `score`
    is a derived property (not a constructor arg) so it round-trips for
    free from the scoring dims."""
    from engine.actions import Action

    path = out_dir / "stages" / "05-actions" / "phase_1_marker.json"
    with open(path) as f:
        data = json.load(f)

    return [
        Action(
            recommendation=x["recommendation"],
            objective=x.get("objective", ""),
            why=x.get("why", ""),
            immediate_action=x.get("immediate_action", ""),
            habit=x.get("habit", ""),
            success_metric=x.get("success_metric", ""),
            horizon=x.get("horizon", ""),
            review_date=x.get("review_date", ""),
            kind=x.get("kind", ""),
            regret_reduction=float(x.get("regret_reduction", 0.0)),
            leverage=float(x.get("leverage", 0.0)),
            consequence=float(x.get("consequence", 0.0)),
            generativity=float(x.get("generativity", 0.0)),
            decisiveness=float(x.get("decisiveness", 0.0)),
            time_to_feedback=float(x.get("time_to_feedback", 0.0)),
            constraint_fit=float(x.get("constraint_fit", 0.0)),
            confidence=float(x.get("confidence", 0.0)),
            source_insights=[tuple(s) for s in x.get("source_insights", [])],
            hallucinated_ref_count=x.get("hallucinated_ref_count", 0),
        )
        for x in data.get("actions", [])
    ]


# ── Pipeline orchestration ────────────────────────────────────────────────────

_VALID_SENTIMENTS = (
    "brutally-honest", "critical", "neutral", "uplifting", "bubbly",
)
_DEFAULT_SENTIMENT = "neutral"


def run(
    paths: list[str],
    mode_str: str,
    subject: str,
    resume_run_dir: Path | None = None,
    resume_run_id: str | None = None,
    sentiment: str = _DEFAULT_SENTIMENT,
):
    """Run the pipeline. `subject` is required — pass the name of the
    narrator / author of the corpus so subject-discipline prompts can
    anchor on a concrete identity. Callers that don't know the subject
    must pass "the author" (or similar) explicitly; there is no
    implicit default.

    `sentiment` ∈ _VALID_SENTIMENTS controls insight + action
    framing tone. Default is `neutral`. Patterns are descriptive and
    are NOT tone-tunable — sentiment touches insights and actions
    only."""
    _ltrace("run_entry")
    # Eval-only extension specs/modes are registered by the eval subprocess
    # entry point (``testing.eval._eval_runner``) BEFORE this runs, on the
    # testing side. The engine deliberately performs no dynamic import here:
    # production runs use the core registries populated by ``llm.py`` at
    # module load.
    if subject is None or not str(subject).strip():
        raise ValueError("run() requires a non-empty `subject` string")
    if sentiment not in _VALID_SENTIMENTS:
        # Fall back to neutral on unknown values rather than raising —
        # keeps older config.json files (no sentiment_bias field) and
        # any typos working without breaking the pipeline.
        sentiment = _DEFAULT_SENTIMENT
    global _stage, _completed
    _run_started_at_iso_local = _iso_z_full()

    _resolve_paths(
        resume_run_dir=resume_run_dir,
        resume_run_id=resume_run_id,
    )

    is_resume = resume_run_dir is not None or resume_run_id is not None
    # On resume, recover the original inputs list from the start-of-run
    # snapshot if the caller didn't pass --paths. config.json is the
    # canonical static record; old runs predate the split and only have
    # run.json — fall back to it (extracting only the static fields we
    # need). The extract stage needs the paths if we're still mid-
    # extract; later stages rebuild from disk without them.
    if is_resume and not paths:
        for fname in ("config.json", "run.json"):
            sidecar = _run_dir / fname
            if not sidecar.exists():
                continue
            try:
                prior = json.loads(sidecar.read_text(encoding="utf-8"))
                paths = list(prior.get("inputs") or [])
                if paths:
                    break
            except (json.JSONDecodeError, ValueError):
                continue

    # Build the run_config snapshot, stamp _run_state, and materialize
    # config.json BEFORE the diagnostic banner / heavy stage imports.
    # The runs-list query (Rust `list_runs`)
    # filters by `agent: "app"` from config.json, so until config.json
    # exists on disk every refreshRuns triggered by stdout activity
    # silently skips this run dir. Writing config.json here — instead
    # of after `_pick_run_model_id` ~140 lines below — lets the
    # preflight emit ALSO act as a row-appearance trigger, instead of
    # firing into the void and waiting for the first wrapper-routed
    # LLM call (which on a fully-cached extract is many seconds away).
    #
    # Single write, no partial state: the snapshot is final the moment
    # we have spec + model, and the model pick (single model vs per-
    # stage) only needs module-level llm symbols already loaded by the
    # import below. Provider + model are stamped into _run_state before
    # `_write_run_config_once` so the snapshot's `primary_model` field
    # reflects the honest run-row label rather than the bare spec.
    from engine.llm import (
        Mode as _Mode_for_cfg,
        get_mode_spec as _get_spec_for_cfg,
        unique_models_in_stage_map,
        # Pulled forward (used to be imported a few hundred lines down,
        # right before cycle_start). Needed up here so the cycle_start
        # event can land in jsonl immediately after config.json hits
        # disk — see the early-cycle-start block below.
        set_calls_jsonl_path,
        emit_cycle_event,
        count_cycle_starts_in_jsonl,
    )
    _ltrace("first_llm_import_done")
    _mode_for_cfg = {
        "local": _Mode_for_cfg.LOCAL,
        "tee":   _Mode_for_cfg.TEE,
    }.get(mode_str, _Mode_for_cfg.LOCAL)
    _spec_for_cfg = _get_spec_for_cfg(_mode_for_cfg)
    _run_model_id = _pick_run_model_id(
        mode=_mode_for_cfg,
        unique_models=unique_models_in_stage_map(),
        spec_model_id=_spec_for_cfg.model_id,
    )
    _run_config_snapshot = _build_run_config_snapshot(
        mode=_mode_for_cfg,
        spec=_spec_for_cfg,
        paths=paths,
        sentiment=sentiment,
    )
    # Override primary_model with the pick — `_build_run_config_snapshot`
    # writes `spec.model_id` (the bare anchor), but the run row should
    # show the single-model id / "per-stage" label as appropriate.
    _run_config_snapshot["primary_model"] = _run_model_id
    _ltrace("snapshot_built")

    _init_run_json(
        paths=paths,
        mode_str=mode_str,
        resume=is_resume,
        run_config=_run_config_snapshot,
    )
    if _run_state is not None:
        with _run_state_lock:
            _run_state["provider"] = _spec_for_cfg.provider.value
            _run_state["model"] = _run_model_id
        _write_run_config_once()
    _ltrace("config_json_written")

    # ── Early cycle_start + preflight tick ──────────────────────────
    # Wire the jsonl writer and emit `cycle_start` + a preflight-stage
    # `progress_tick` BEFORE the heavy import block, attestation
    # prewarm, and progress-tracker init below.
    #
    # Reason: derive_run_state surfaces `status: "running"` only after
    # `cycle_start` lands in `llm-calls.jsonl` (see lib.rs §
    # `derive_run_state_uncached`); RunRow gates the entire
    # `.run-progress` block on that status. Until cycle_start exists
    # the row shows a literal "unknown" badge with no bar / stage
    # label / counts.
    #
    # Historically cycle_start was emitted ~350 lines down, after the
    # heavyweight `from llm/ingestor/splitter/content_extractor
    # import` block (~250–500ms) and the run-head TEE setup. That gap
    # was the "row but no bar" window (LOCAL ~700ms, TEE cold several
    # seconds while the first Tinfoil client constructs + verifies).
    #
    # Emitting cycle_start here collapses that window to <200ms in
    # all modes — the bar renders with stage="preflight" → the user-
    # visible label "Verifying inference environment" while the rest
    # of the pre-LLM-call setup runs underneath.
    set_calls_jsonl_path(_run_dir / "llm-calls.jsonl")
    _cycle_seq = count_cycle_starts_in_jsonl() + 1
    emit_cycle_event("cycle_start", {
        "run_id": _run_dir.name,
        "is_resume": bool(is_resume),
        "cycle_seq": _cycle_seq,
        # pid lands in cycle_start so post-#165 orphan-recovery + cancel
        # paths read it from the jsonl walk (derive_run_state.pid)
        # rather than run.json. Pre-#165 runs fall back to run.json.pid.
        "pid": os.getpid(),
    })
    _stage = "preflight"
    # ── ONE pre-ingest estimate ──────────────────────────────────────
    # Cheap pre-scan of input files (stat + zip namelist; no parse) →
    # ONE call to `estimate_pipeline`, used at the preflight tick and
    # again at the ingest tracker registration (same `calls_per_stage`
    # dict, no recomputation). Post-ingest, `_estimate_per_stage` re-
    # estimates with REAL parsed Documents and supersedes the pre-scan
    # via `register_stage`'s `max(new_est, completed_calls)`. Once the
    # tracker has live samples, `_emit()` takes over with the
    # historical-median-blended ETA.
    #
    # `historical_durations` + `stage_model_map` are loaded HERE
    # (before the preflight emit) so `estimate_pipeline` can use the
    # same token-aware per-call computation the live tracker applies
    # moments later. Without this, preflight uses FALLBACK constants
    # (token-blind) and the tracker uses historical-blended (token-
    # aware); for models whose historicals diverge from the FALLBACK
    # assumption, the two produce wildly different eta values (issue
    # #394 — TEE-mix kimi-k2-6 run showed 173s preflight → 1225s
    # ingest, 7× jump with the call-count denominator stable). The
    # tracker init block below reuses both values so the historicals
    # don't get walked twice.
    from engine.ingestor import pipeline_stats_from_paths
    from engine.progress import (
        estimate_pipeline,
        load_historical_durations as _load_hist_pre,
    )
    _pre_hist_durations = _load_hist_pre(
        skip_path=_run_dir / "llm-calls.jsonl")
    _pre_stage_model_map = _resolve_stage_model_map_for_run(
        _mode_for_cfg, _spec_for_cfg)
    _pre_stats = pipeline_stats_from_paths(paths)
    _pre = estimate_pipeline(
        _pre_stats,
        is_local=(mode_str == "local"),
        historical_durations=_pre_hist_durations,
        stage_model_map=_pre_stage_model_map,
    )
    _pre_estimate = _pre.calls_per_stage
    _preflight_total_est = _pre.total_calls
    # Companion `progress_tick` for the preflight stage. Without this,
    # derive_run_state's `latest_stage_from_begin` is None and the
    # bar falls back to `cycle_start_stage = "init"` → label
    # "Initializing pipeline". With it, the bar reads the more honest
    # "Verifying inference environment" while attestation + imports
    # run. Total + ETA come straight off `_pre` so the chip's
    # denominator AND its remaining-time label are both grounded in
    # the same single estimate from the moment the run row appears.
    emit_cycle_event("progress_tick", {
        "stage": _stage,
        "total": _preflight_total_est,
        "in_flight_calls": 0,
        "eta_seconds": _pre.total_seconds,
        "stage_eta_seconds": 0.0,
        "elapsed_in_stage": 0.0,
    })

    # Preflight emit — config.json is on disk now, so this print is
    # what surfaces the run row in the UI. Fires for every mode (the
    # old LOCAL carve-out predated the issue-#165 split and assumed
    # config.json was already on disk via run.json; post-split, the
    # row appearance depends on the emit + sidecar pair landing
    # together). Cost: a single visible "preflight" frame in the UI
    # before extract starts, regardless of mode.
    _ltrace("first_emit_called")
    _emit()

    resume_from = "start"
    if is_resume:
        resume_from = _detect_resume_point(_run_dir)

    log_path = _init_log(resume=(resume_run_dir is not None))
    _log_write(f"Paths: {paths}")
    _log_write(f"Mode: {mode_str}")
    if resume_run_dir is not None:
        _log_write(f"Resuming from stage: {resume_from}")

    # ── Diagnostic banner ────────────────────────────────────────────────
    # Surface enough state at startup that "0 items extracted" failures can
    # be diagnosed from the run.log alone (which interpreter, which deps
    # missing, which env vars set, where .env loaded from).
    _log_write("── Runtime diagnostics ──")
    _log_write(f"  python:     {sys.executable}")
    _log_write(f"  version:    {sys.version.split(chr(10))[0]}")
    _log_write(f"  sys.path[0]: {sys.path[0] if sys.path else '<empty>'}")
    _log_write(f"  cwd:        {os.getcwd()}")
    _log_write(f"  .env from:  {_DOTENV_LOADED_FROM or 'NONE — checked: ' + ', '.join(str(p) for p in _DOTENV_CANDIDATES)}")
    _log_write(f"  env: TINFOIL_API_KEY={'set' if os.environ.get('TINFOIL_API_KEY') else 'UNSET'}, "
               f"PIPELINE_PYTHON={os.environ.get('PIPELINE_PYTHON') or 'UNSET'}")
    _log_write("─────────────────────────")

    from engine.llm import (
        Mode, reset_usage_log, reset_call_warnings, reset_stat_records,
        chunk_cap_for_stage,
        _MAX_RATIO_BY_STAGE, _TYPICAL_RATIO_BY_STAGE,
        _mode_ctx,
        get_mode_spec,
    )
    from engine.ingestor import ingest
    from engine.splitter import (
        split_documents,
        carve_first_batch,
        report as splitter_report,
    )

    # Reset per-call token log + cap-hit list + stat records so this
    # run is captured in isolation.
    reset_usage_log()
    reset_call_warnings()
    reset_stat_records()
    # Clear LLM-cache hit/miss counters before this run so the totals
    # in llm-stats.json reflect just this run, not accumulation since
    # the process started (which would matter for sweep harnesses
    # sharing one Python process).
    from engine.llm_cache import reset_cache_stats as _reset_cache_stats
    _reset_cache_stats()
    # Honor the Settings checkbox `cfg.llm_cache_enabled` (default
    # true). When the user disables caching, translate that into the
    # env-var bypass so llm_cache.lookup/store no-op for the run.
    # Explicit BASEVAULT_LLM_CACHE_BYPASS overrides config — lets a
    # power user toggle for one run without flipping Settings.
    if "BASEVAULT_LLM_CACHE_BYPASS" not in os.environ:
        try:
            _cfg_path = Path.home() / ".basevault" / "config.json"
            if _cfg_path.exists():
                _cfg = json.loads(_cfg_path.read_text(encoding="utf-8"))
                if isinstance(_cfg, dict) and _cfg.get("llm_cache_enabled") is False:
                    os.environ["BASEVAULT_LLM_CACHE_BYPASS"] = "1"
                    _log_write("LLM cache disabled by Settings (cfg.llm_cache_enabled=false)")
        except (json.JSONDecodeError, ValueError, OSError):
            pass  # config unreadable → leave cache on (default behavior)

    _core_modes = {
        "local": Mode.LOCAL,
        "tee":   Mode.TEE,
    }
    mode = _core_modes.get(mode_str)
    if mode is None:
        # Extension mode: an entry registered into MODE_SPEC by the eval
        # subprocess entry point (``testing.eval._eval_runner``) before
        # run(). Pass the bare string through; complete() handles
        # Mode | str. An unknown string raises — silent-fallback to
        # Mode.LOCAL would mask typo + config drift bugs (a run that
        # meant TEE but typed `--mode tea` should fail loud, not
        # silently re-route data through LOCAL).
        from engine.llm import MODE_SPEC as _mode_spec_table
        if mode_str in _mode_spec_table:
            mode = mode_str
        else:
            valid = sorted(_core_modes) + sorted(
                k for k in _mode_spec_table
                if isinstance(k, str) and k not in _core_modes
            )
            raise ValueError(
                f"unknown --mode {mode_str!r}; valid: {valid}"
            )
    # Publish to module-global so _budget_snapshot_for() can resolve
    # stage caps / max_output / scaffolding when the wrapper opens
    # each stat record.
    global _active_mode
    _active_mode = mode

    # Per-stage input/output budgets derived from each stage's measured
    # max ratio (see llm._MAX_RATIO_BY_STAGE) against the mode's
    # context window. Splitter targets the extract row; other stages
    # enforce their own caps at call time. _TYPICAL_RATIO_BY_STAGE is
    # used elsewhere for estimates (progress bar, expected output).
    _ctx_tokens = _mode_ctx(mode)
    _chunk_budget = chunk_cap_for_stage(mode, "extract")
    _spec = get_mode_spec(mode)
    # `mode` and `_spec.provider` are both `… | str` here: extension
    # modes registered by the eval tree (the str-keyed "test" path)
    # arrive as bare strings rather than the production enums, so a raw
    # `.value` access AttributeErrors. Route through the llm stringify
    # helpers — the same widening the line-1321 snapshot guard relies on.
    from engine.llm import _mode_str, _provider_str
    # _run_model_id, provider/model stamping, and config.json write all
    # happen up at the snapshot-build site so the row appears at the
    # preflight emit instead of waiting on this block. The values land
    # in scope here via the early `_run_model_id` local.
    _log_write("── Budget ──")
    _log_write(f"  mode:             {_mode_str(mode)}")
    _log_write(f"  provider:         {_provider_str(_spec.provider)}")
    _log_write(f"  model:            {_run_model_id}")
    _log_write(f"  context window:   {_ctx_tokens:,} tokens")
    _log_write(f"  streaming:        {_spec.require_streaming}")
    # Unified per-stage routing block — model + reasoning + temperature
    # inline so a debug-bundle reader can answer "what was set for which
    # stage" without cross-referencing run.json. Source of truth is the
    # run_config snapshot; we read it here so the log header and
    # run.json never disagree.
    _stage_models_snap = _run_config_snapshot.get("stage_models", {})
    _stage_reasoning_snap = _run_config_snapshot.get("stage_reasoning", {})
    _temp_snap = _run_config_snapshot.get("temperature", 0.0)
    if _stage_models_snap:
        _log_write("  per-stage routing (model · reasoning · temperature):")
        from engine.progress import PIPELINE_STAGES as _PIPELINE_STAGES_FOR_LOG
        for _stage_name in _PIPELINE_STAGES_FOR_LOG:
            _model = _stage_models_snap.get(_stage_name, "?")
            _reason = bool(_stage_reasoning_snap.get(_stage_name, False))
            _reason_tag = "ON " if _reason else "off"
            _annot = "  ← reasoning ON" if _reason else ""
            _log_write(
                f"    {_stage_name:<17} {_model:<22} reasoning={_reason_tag}  "
                f"temp={_temp_snap}{_annot}"
            )
    # Reasoning + any other per-call kwargs are logged at each LLM call
    # below; they're computed dynamically from _REASONING_WHITELIST + the
    # model's branch in _reasoning_kwargs in llm.py.
    _log_write("  per-stage limits  (max:1 / typical → scaffolding → chunk input cap → max output):")
    # Stage table includes ingest + vision so the user-visible
    # accounting matches the LLM-stats / progress tracker (#115).
    # vision is a first-class LLM stage (post-#205): it has ratio
    # entries and renders through the same path as the chat stages.
    # ingest is a parent stage that does NOT issue LLM calls at this
    # level; compute_budget reports has_llm_calls=False and the row
    # holds dashes.
    from engine.llm import compute_budget as _compute_budget
    _DASH_T = "       —"  # right-aligned 8-wide for the token columns
    _DASH_RATIO = " — "
    for _stage_name in ("ingest", "vision",
                        "extract", "entities", "entities_dedupe",
                        "patterns", "insights", "actions"):
        _budget = _compute_budget(mode, _stage_name)
        if not _budget.has_llm_calls:
            _r_str = _DASH_RATIO
            _tr_str = _DASH_RATIO
            _sc_str = _DASH_T
            _ci_str = _DASH_T
            _mo_str = _DASH_T
            _tag = ""
        else:
            _r = _MAX_RATIO_BY_STAGE.get(_stage_name)
            _tr = _TYPICAL_RATIO_BY_STAGE.get(_stage_name)
            _r_str = f"{_r:>3.1f}" if _r is not None else " — "
            _tr_str = f"{_tr:>3.1f}" if _tr is not None else " — "
            _sc_str = f"{_budget.scaffolding:>5,}t"
            _ci_str = f"{_budget.max_input:>7,}t"
            _mo_str = f"{_budget.max_output:>7,}t"
            _tag = " ← splitter budget" if _stage_name == "extract" else ""
        _log_write(
            f"    {_stage_name:15s}  max {_r_str} / typ {_tr_str}   "
            f"scaf={_sc_str}   {_ci_str}   {_mo_str}{_tag}"
        )
    _log_write("────────────")

    # Pre-flight: required API key for cloud modes.
    # API key check: derive from the actual provider pinned to this
    # mode. Mode.LOCAL has no API key; Mode.TEE → Tinfoil.
    from engine.llm import Provider
    _provider_to_env_key = {
        Provider.TINFOIL: "TINFOIL_API_KEY",
        Provider.OLLAMA:  None,
        Provider.MLX:     None,
    }
    _required_key = _provider_to_env_key.get(_spec.provider)
    if _required_key and not os.environ.get(_required_key):
        msg = (
            f"FATAL: {_required_key} is not set in the environment, "
            f"but mode={mode_str} (provider={_spec.provider.value}) requires it.\n"
            f"  .env loaded from: {_DOTENV_LOADED_FROM or 'NONE — checked ' + ', '.join(str(p) for p in _DOTENV_CANDIDATES)}\n"
            f"  Fix: open the app, pick Settings… from the BaseVault menu,\n"
            f"  and set your {_spec.provider.value} key (or add {_required_key} manually to\n"
            f"  ~/Library/Application Support/BaseVault/.env)."
        )
        print(msg, file=sys.stderr, flush=True)
        _log_write(msg)
        _mark_run_terminal("failed", error=msg)
        sys.exit(2)


    # No attestation pre-warm here by design. Per-request Tinfoil
    # attestation is intrinsic to the SDK client (the kernel
    # `TinfoilProvider`: the `TinfoilAI()` ctor crypto-verifies the
    # enclave and fails the
    # first call closed if it cannot, then TLS-pins every request). The
    # cold-cache parallel-TUF race a sequential pre-warm used to guard
    # against cannot occur — nothing attests at inference time. The
    # supplementary cross-check + audit log run only from the three
    # sanctioned app-side call sites (startup verify, Settings
    # re-check, hourly timer), never per pipeline run.

    out_dir = _run_dir  # dump paths are relative to run_dir
    vault_dir = _vault_dir

    # ── Observability: streaming + atexit dump-on-failure ────────────────
    #
    # llm-calls.jsonl is the append-only event log; one line per
    # begin/end/counts event. Wrapper streams begin at begin_stat_record
    # and end at finalize; this survives SIGKILL / segfault / any kill
    # the kernel doesn't route through Python.
    #
    # llm-stats.txt (the human-readable rollup summary) is written
    # at end-of-run AND from an atexit handler. The rollup dict
    # itself is no longer cached on disk (issue #189) — consumers
    # derive it via `materialize_run_stats(jsonl, config)`. Atexit
    # fires on:
    #   - normal sys.exit() of any code,
    #   - unhandled exception that propagates to the process top,
    #   - SIGTERM, but ONLY because we install a Python-level handler
    #     below that converts SIGTERM → sys.exit. Default Python on
    #     SIGTERM is C-level termination (atexit skipped).
    # The Tauri Cancel button sends SIGTERM (with a 3s grace before
    # SIGKILL). On SIGKILL or os._exit() atexit is skipped — the
    # event log is the contingency for those paths.
    #
    # Both `_write_llm_stats` and `set_calls_jsonl_path` are
    # idempotent — calling them from multiple exit paths is safe.
    # `set_calls_jsonl_path` was already wired at the early
    # cycle_start block above (right after config.json hits disk);
    # only `set_payloads_jsonl_path` and the bootstrap need to fire
    # here.
    from engine.llm import (
        set_payloads_jsonl_path,
        bootstrap_call_id_counter_from_jsonl,
    )
    # Sibling payload stream (issue #195). The dev-tab full prompt +
    # response logger writes here when its per-stage toggle is on; off
    # otherwise. Always wired so the toggle takes effect mid-run
    # without a runner restart.
    set_payloads_jsonl_path(out_dir / "llm-payloads.jsonl")
    # On resume the prior session's llm-calls.jsonl is already on disk.
    # Two pieces of state need to pick up where session 1 left off so
    # the rollup stays coherent across sessions:
    #   1. _call_id_counter (in llm.py) — otherwise session 2 emits
    #      0001, 0002, ... colliding with session 1's ids in the SAME
    #      append-only file. The materializer keys on call_id alone
    #      and would mis-pair begin↔end events from different
    #      sessions, producing garbage records (wrong stage, wrong
    #      duration, wrong model). The bootstrap helper reads the max
    #      seen begin call_id and advances the counter past it; first
    #      call in session 2 lands at e.g. 0051 if session 1 ended at
    #      0050.
    #   2. `_completed` (the runner's progress-bar counter) — the
    #      wrapper increments it on every successful LLM call, so a
    #      resume that bootstraps it from prior-session successes lets
    #      the progress bar render cumulative completion across both
    #      sessions instead of resetting to 0/total.
    # No-op for non-resume runs (file doesn't exist or is empty).
    _bootstrap_max_id = bootstrap_call_id_counter_from_jsonl()
    global _completed, _announced_total
    _completed = _bootstrap_completed_from_jsonl(out_dir / "llm-calls.jsonl")
    _announced_total = 0
    if is_resume and (_bootstrap_max_id > 0 or _completed > 0):
        _log_write(
            f"Resume bootstrap: call_id counter advanced past "
            f"{_bootstrap_max_id} (next call lands at "
            f"{_bootstrap_max_id + 1:04d}); _completed seeded from "
            f"{_completed} successful prior-session call(s)."
        )

    # ── Pipeline-wide ETA tracker ────────────────────────────────────
    # Reuse the historicals + stage_model_map already loaded above for
    # the preflight estimate. Same files, same window — loading twice
    # would just walk the jsonl tree again with the same skip_path.
    # Falls back to a fresh load if preflight took the except path and
    # left them as None.
    global _progress_tracker, _stage_model_map
    from engine.progress import (
        ProgressTracker, load_historical_durations,
        PARALLELISM_PER_STAGE, PARALLELISM_LOCAL_OVERRIDE,
    )
    if _pre_hist_durations is not None:
        _hist_durations = _pre_hist_durations
    else:
        _hist_durations = load_historical_durations(
            skip_path=out_dir / "llm-calls.jsonl",
        )
    # LOCAL mode runs Ollama with a single GPU — fan-out is 1 across
    # the board. Cloud modes use the cloud-side defaults
    # (PARALLELISM_PER_STAGE).
    _stage_parallelism = (
        {s: PARALLELISM_LOCAL_OVERRIDE for s in PARALLELISM_PER_STAGE}
        if mode == Mode.LOCAL else dict(PARALLELISM_PER_STAGE)
    )
    _progress_tracker = ProgressTracker(
        historical_durations=_hist_durations,
        parallelism_per_stage=_stage_parallelism,
    )
    _stage_model_map = (
        _pre_stage_model_map
        if _pre_stage_model_map is not None
        else _resolve_stage_model_map_for_run(mode, _spec)
    )
    _log_write(
        f"Progress: loaded historical durations for "
        f"{len(_hist_durations)} (stage, model) keys; "
        f"per-stage models: {_stage_model_map}"
    )
    # On resume, seed per-stage state from the prior session's jsonl
    # so the tracker's completed-call counts and live-duration arrays
    # pick up where session 1 left off. Without this, a resumed run
    # would compute ETA assuming zero work done in earlier stages
    # even though their outputs are on disk and `_completed` is
    # already non-zero.
    _per_stage_seed = _bootstrap_per_stage_from_jsonl(
        out_dir / "llm-calls.jsonl"
    )
    if _per_stage_seed:
        _log_write(
            f"Progress: seeding tracker from prior-session per-stage "
            f"state: { {k: v['count'] for k, v in _per_stage_seed.items()} }"
        )
        for _seed_stage, _seed in _per_stage_seed.items():
            _seed_model = (
                _stage_model_map.get(_seed_stage)
                or _seed.get("model") or _spec.model_id
            )
            # Register with at least the count seen on disk; later
            # stage-entry callsites widen est_calls with their
            # measured estimate.
            _progress_tracker.register_stage(
                _seed_stage, _seed_model, max(1, _seed["count"]))
            _stg = _progress_tracker._stages[_seed_stage]
            _stg.completed_calls = _seed["count"]
            _stg.live_samples = list(_seed["samples"])
            # Stages whose work landed on disk fully are marked
            # finished retroactively; durations contributed total
            # elapsed even though we don't have exact wall-clock
            # for them. _record_stage_started will take care of
            # the in-flight stage at its callsite.
            _seed_total_dur = sum(d for d, _t in _seed["samples"]) or 0.001
            _stg.mark_started(0.0)
            _stg.mark_finished(_seed_total_dur)

    # Cycle marker — issue #52. cycle_start is now emitted up at the
    # early-cycle-start block, right after config.json lands, so the
    # run row's status flips to "running" within ~200ms instead of
    # waiting for prewarm + the heavy import block down here. The
    # matching `cycle_end` fires from `_emit_cycle_end_once`
    # (idempotent — tracks via `_cycle_end_emitted` to dedupe normal-
    # completion + atexit + SIGTERM paths).

    _reset_llm_stats_dump_state()
    _atexit_kwargs = {
        "primary_model": _spec.model_id,
        "run_started_at_iso": _run_started_at_iso_local,
    }
    import atexit
    # Register cycle_end BEFORE _write_llm_stats so atexit fires it
    # FIRST (atexit is LIFO). That way the rollup that runs after
    # observes the cycle_end record on disk and can include it in any
    # future per-cycle materialization.
    atexit.register(_emit_cycle_end_once, "atexit")
    atexit.register(_write_llm_stats, out_dir, mode, **_atexit_kwargs)

    # Issue #333: marker-dir poller for per-call user-skip. Tauri's
    # `skip_call` command writes `<run_dir>/skipped_calls/<call_id>`
    # atomically (tmp + rename, mirroring `paused.flag`). The poller
    # scans the dir every 500ms and registers any new ids into
    # `llm._skipped_call_ids`; `_consume_chat_stream` then short-
    # circuits the matching call on its next chunk via the
    # `_SkippedByUser` exception path.
    #
    # Polling cadence: 500ms. Faster would shave latency on the rare
    # skip event but cost more inode stats in steady state; slower
    # makes the ✕ feel sluggish. The user-visible latency upper bound
    # is poll-interval + next-chunk-arrival, which already dominates
    # at TTFT-hang time (the proposal accepted that trade-off — no
    # watcher-thread force-close of the stream).
    from engine.llm import clear_skipped, register_skip
    from engine.phases.telemetry_hook import skip_kernel_call as _skip_kernel_call
    clear_skipped()
    _skip_marker_dir = out_dir / "skipped_calls"

    def _skip_marker_poller() -> None:
        """Scan the marker dir at a fixed cadence and register any new
        call_ids. Idempotent — `register_skip` adds to a set and shuts
        down any registered live stream's socket, both of which tolerate
        repeat calls. Errors during scan (dir doesn't exist yet,
        permission issues) silently retry next tick; the polling thread
        must not crash the run.

        Cadence is tight (100ms) because `register_skip` is the single
        bottleneck for how fast a skip click takes effect: the click
        writes the marker via Tauri instantly, `register_skip` shuts
        down the in-flight socket, and the consumer's exception wrapper
        finalizes the call. Anything slower here adds directly to the
        observed 'skip → call still running' window. Scan cost is a
        small `iterdir()` over the per-run marker directory — the `seen`
        set memoizes already-registered ids, so steady-state scanning is
        essentially free."""
        import time as _time
        seen: set[str] = set()
        while True:
            try:
                if _skip_marker_dir.is_dir():
                    for entry in _skip_marker_dir.iterdir():
                        try:
                            if entry.is_file() and entry.name not in seen:
                                seen.add(entry.name)
                                register_skip(entry.name)        # legacy path
                                _skip_kernel_call(entry.name)     # kernel path
                        except OSError:
                            continue
            except OSError:
                # Dir was rmtree'd / unreadable. Try again next tick.
                pass
            _time.sleep(0.1)

    _poll_thread = threading.Thread(
        target=_skip_marker_poller,
        name="skip-marker-poller",
        daemon=True,
    )
    _poll_thread.start()

    # Install a Python-level SIGTERM handler so the Cancel button
    # (Tauri sidecar sends SIGTERM) flushes the rollup before the
    # process exits.
    #
    # Round 4 — p63u investigation: previously this handler did
    # `sys.exit(0)`. That raises SystemExit, which propagates up
    # through the main thread. If the main thread is inside a
    # `with ThreadPoolExecutor(...)` block (which it is, mid-run),
    # the `__exit__` calls `executor.shutdown(wait=True)` which
    # blocks waiting for in-flight LLM calls to complete. When
    # those calls are 429-stuck or attestation-stuck, shutdown can
    # take 30-60s — well past the Tauri-side 3-5s SIGTERM grace,
    # so SIGKILL fires before atexit runs and the rollup is lost.
    #
    # Fix: write the rollup INLINE in the handler (before any
    # executor.shutdown can block), then os._exit(0). os._exit
    # bypasses the rest of atexit but the rollup is already on
    # disk so we don't lose it. The threadpool's pending futures
    # die with the process — they were going to die in 0-2s anyway
    # at the Rust SIGKILL fallback.
    #
    # Both `_write_llm_stats` and `set_calls_jsonl_path` remain
    # idempotent — atexit + signal-handler-inline call is safe.
    import signal
    def _sigterm_flush_and_exit(_signum, _frame):
        try:
            _log_write(
                "Received SIGTERM — flushing rollup inline before exit "
                "(executor.shutdown(wait=True) would otherwise block "
                "atexit past the Tauri SIGKILL fallback)."
            )
        except Exception:
            pass
        # Cycle-end marker before the rollup so the rollup observes
        # the marker on disk (matters when consumers materialize
        # per-cycle slices later from the jsonl).
        _emit_cycle_end_once("sigterm")
        try:
            _write_llm_stats(out_dir, mode, **_atexit_kwargs)
        except BaseException as e:
            # Never let an exception in the rollup write block exit.
            # The body of _write_llm_stats has its own try/except
            # that records errors to atexit-error.txt; this is a
            # belt-and-suspenders.
            try:
                _log_write(
                    f"SIGTERM rollup write raised "
                    f"{type(e).__name__}: {e}"
                )
            except Exception:
                pass
        # os._exit bypasses the rest of atexit and any stuck
        # threadpool shutdown. The rollup is already on disk.
        os._exit(0)
    try:
        signal.signal(signal.SIGTERM, _sigterm_flush_and_exit)
    except (ValueError, OSError):
        # signal.signal raises ValueError outside the main thread
        # and OSError on some platforms. Best-effort — without the
        # handler, SIGTERM-cancelled runs lose the rollup but keep
        # the .jsonl, which is the whole point of the sidecar.
        pass

    def _dump(filename: str, obj):
        """Write a JSON-serializable object under the run dir.

        Atomic: serialize into a sibling .tmp then rename. SIGTERM /
        SIGKILL mid-write would otherwise leave a truncated JSON on
        disk; the per-parent loader (08_items.json)
        wraps its reads in try/except and recovers, but the
        stage-boundary loaders (`_load_facts_by_topic`,
        `_load_entities`, `_load_insights`, `_load_patterns`) do raw
        `json.load` and would crash a resume on a corrupted half-write
        because `_detect_resume_point` checks file existence only.
        Atomic-rename closes that window."""
        p = out_dir / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        tmp.replace(p)
        _log_write(f"  >> wrote {p}")

    def _dump_jsonl_record(filename: str, obj):
        """Write a single JSON record to a `.jsonl` path on one line,
        with a trailing newline so subsequent reads via `splitlines()`
        treat it as one well-formed line. Atomic via .tmp + rename like
        `_dump`. Used for the per-entity canonical files at
        `02-entities/entities/<canonical_id>.jsonl`, where
        `_dump`'s `indent=2` would emit multi-line JSON and break
        line-oriented readers (Rust `read_run_entity` parses the LAST
        non-empty line; an indented dump's last line is just `}`)."""
        p = out_dir / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=str))
            f.write("\n")
        tmp.replace(p)
        _log_write(f"  >> wrote {p}")

    # Early exit for already-complete resume
    if resume_from == "done":
        _log_write("Already complete — nothing to resume")
        _mark_run_terminal("completed")
        print(json.dumps({
            "stage": "done",
            "completed": 0,
            "facts": 0,
            "entities": 0,
            "relations": 0,
            "patterns": 0,
            "insights": 0,
            "actions": 0,
            "output_dir": str(out_dir),
        }), flush=True)
        if _log_file:
            _log_file.close()
        return

    # Placeholders populated by whichever branch runs
    all_docs = []
    docs = []
    all_items: list = []
    facts_by_topic: dict[str, list] = {}
    patterns_by_topic: dict = {}

    if resume_from == "start":
        # 1. Ingest ────────────────────────────────────────────────────────────
        # Pre-scan + register stages BEFORE _log_stage("ingest") so
        # the ingest tick already carries the correct total. Without
        # this order, _log_stage emits with total=0 (tracker has no
        # stages yet) and the bar renders 0/0 → ?% briefly until the
        # first vision call fires another _emit() with the now-
        # registered total.
        # Pre-scan walks the inputs (cheap stat + zip namelist) for
        # rough per-stage call estimates. Without this step, vision
        # finishes before extract registers → tracker total = vision
        # only → bar trips progressChipPct's `displayDone >= total`
        # 99% clamp on the moment vision finishes. Image / PDF /
        # DOCX bytes use a flat 10kB-per-file estimate (file size
        # is bloated by formatting, not text content); md / txt /
        # json use real on-disk size.
        # Post-ingest, `_register_initial_estimates` re-registers
        # every stage with REAL document-derived counts (parsed
        # tokens, real chunk counts via the splitter, real topic
        # count). `register_stage` uses `max(new_est,
        # completed_calls)` — so the real numbers REPLACE the
        # pre-scan estimates (clamped only above already-completed).
        _stage = "ingest"
        # Register stages with the ONE pre-ingest estimate computed
        # at the top of run() (line ~4153). No recomputation here —
        # _pre_estimate is the same dict that drove the preflight
        # tick. Real numbers land post-ingest via _register_initial_
        # estimates() with parsed Document content.
        if _progress_tracker is not None:
            from engine.progress import PIPELINE_STAGES as _PIPELINE_STAGES
            for _stage_name in _PIPELINE_STAGES:
                _est = _pre_estimate.get(_stage_name, 1)
                if _est <= 0:
                    continue
                _progress_tracker.register_stage(
                    _stage_name,
                    _stage_model_map.get(_stage_name, "?"),
                    _est,
                )
            _emit_total(
                _progress_tracker.snapshot()["total_calls_pipeline"])
        # Now log the stage transition + emit the first ingest tick
        # — by this point the tracker has all stages registered, so
        # the tick carries the right `total`.
        _log_stage("ingest")
        _emit()
        # Parallel fan-out across the top-level paths so a picker
        # selection of N image files doesn't process them one-at-a-
        # time (one vision call per file = serial wall-clock). When
        # any of the paths is a directory, ingest() recurses through
        # `_ingest_dir`'s own ThreadPoolExecutor so concurrency
        # nests naturally. Vision mode MUST match pipeline mode —
        # never leak across trust boundaries.
        from concurrent.futures import (
            ThreadPoolExecutor as _TPE,
            as_completed as _as_completed,
        )
        from engine.llm import max_workers as _mw
        if len(paths) == 1:
            all_docs.extend(ingest(paths[0], vision_mode=mode))
        else:
            n_ingest_workers = min(len(paths), _mw(mode))
            with _TPE(max_workers=n_ingest_workers) as _ex:
                _futs = [
                    _ex.submit(ingest, p, vision_mode=mode)
                    for p in paths
                ]
                for _f in _as_completed(_futs):
                    all_docs.extend(_f.result())
        _log_write(f"Ingested {len(all_docs)} document(s)")

        # Vision dispatch — separate sub-phase after the parser-level
        # ingest. `_parse_image` returns placeholder Documents marked
        # `metadata["_pending_vision"] = True`; this stage actually
        # transcribes them via the scheduler-driven `describe_images_all`
        # so the gate paces the vision cohort (one call per
        # interval_s). Pre-refactor vision ran inside the ingestor's
        # per-file TPE, which made `describe_image` a concurrent-
        # dispatch site incompatible with the scheduler's exclusive
        # `_stage_active` lock. Lifting it here is the same shape every
        # other stage uses.
        _pending_vision_count = sum(
            1 for d in all_docs
            if d.metadata.get("_pending_vision")
        )
        if _pending_vision_count > 0:
            _log_write(
                f"Vision: transcribing {_pending_vision_count} image(s) "
                f"via the scheduler"
            )
            from engine.ingestor import SourceType as _SourceType
            from kernel.enums import PhaseName as _PhaseName
            from engine.phases.ingestion import describe_images
            from engine.phases.model_specs import build_stage_env
            _imgs = [
                d for d in all_docs
                if d.source_type == _SourceType.IMAGE
                and d.metadata.get("_pending_vision")
            ]
            describe_images(
                _imgs, mode,
                execution_env=build_stage_env(_PhaseName.INGESTION, mode, extra_hooks=[_KERNEL_LIVE_HOOK]),
            )
            # Vision-failed docs are filtered here.
            # Filter out docs whose vision call failed / yielded no
            # text — extract has nothing useful to do with an empty
            # image document.
            _failed = [
                d for d in all_docs
                if d.metadata.get("_vision_failed")
            ]
            for d in _failed:
                reason = (
                    d.metadata.get("_vision_error")
                    or d.metadata.get("_vision_skip_reason")
                    or "unknown"
                )
                _log_write(
                    f"  Image vision failed for {d.source_path}: "
                    f"{reason}"
                )
            all_docs = [
                d for d in all_docs
                if not d.metadata.get("_vision_failed")
            ]
            _log_write(
                f"Vision: kept {_pending_vision_count - len(_failed)} of "
                f"{_pending_vision_count} image transcriptions"
            )

        if not all_docs:
            if paths:
                msg = (
                    f"All {len(paths)} input(s) were skipped — see warnings above "
                    f"for per-file reasons. Pipeline cannot proceed with zero documents."
                )
            else:
                msg = "No documents found"
            print(json.dumps({"stage": "error", "message": msg}), flush=True)
            _log_write(f"ERROR: {msg}")
            _mark_run_terminal("failed", error=msg)
            return

        # 1b. Preprocess — snapshot canonical text per file_id to disk.
        from engine.preprocessor import preprocess
        preprocessed_files = preprocess(all_docs, out_dir / "stages" / "00-ingestion")
        _log_write(
            f"Preprocessed {len(preprocessed_files)} file(s) to "
            f"stages/00-ingestion/documents/"
        )

        # Phase 1 marker — payload is the manifest of generated documents.
        _dump("stages/00-ingestion/phase_1_marker.json", {
            "documents": [
                {"id": d.id, "source_path": d.source_path,
                 "source_type": d.source_type.value, "content_len": len(d.content),
                 "title": d.title, "date": d.date, "file_id": d.file_id}
                for d in all_docs
            ],
        })

        # 1c. Files manifest — append-only insertion-order log of every
        # input file ever seen, keyed by file_id. New files append at
        # the end; existing files keep their position. Drives
        # cache-stable entity-batch packing: when a user adds a new
        # file to their inputs, only the new file's stages bust cache
        # because existing groups stay in the same manifest slot.
        # Hash the post-preprocess content (stable, normalized) rather
        # than the source bytes so reformat-only edits don't invalidate.
        from engine.files_manifest import (
            update_manifest as _update_manifest,
            position_index as _manifest_position_index,
        )
        import hashlib as _hashlib
        _discovered = []
        for pf in sorted(preprocessed_files, key=lambda p: p.file_id):
            try:
                content_bytes = pf.path.read_bytes()
            except OSError:
                continue
            ch = _hashlib.sha256(content_bytes).hexdigest()
            _discovered.append((pf.file_id, ch))
        _manifest_entries = _update_manifest(_discovered)
        manifest_pos = _manifest_position_index(_manifest_entries)
        _log_write(
            f"Files manifest: {len(_discovered)} file(s) reconciled, "
            f"{len(_manifest_entries)} total entries"
        )

        # Stage 1: Extraction — `_log_stage("extract")` prints the
        # `── STAGE: extract ──` banner, transitions the bar, and
        # marks the tracker stage started. Splitter is Phase 1 of
        # this stage per the pipeline-stages spec; the LLM-call phase
        # is Phase 2. We enter the stage HERE (before the splitter)
        # so the user-visible stage label is "extract" through both
        # phases — pre-fix the splitter ran with _stage = "ingest"
        # and then jumped to "extract" at the LLM phase.
        _stage = "extract"
        _log_stage("extract")
        # Phase 1: deterministic splitter.
        _log_write("── Phase 1: splitter (deterministic) ──")
        split_rep = splitter_report(all_docs, budget_tokens=_chunk_budget)
        oversized = [r for r in split_rep if not r["fits"]]
        _log_write(f"Splitter: {len(split_rep)} doc(s); {len(oversized)} oversized "
                   f"(budget={_chunk_budget}t)")
        for r in oversized:
            _log_write(f"  oversize: {r['id']} {r['tokens']}t → "
                       f"{r['n_splits']} parts: {r.get('split_sizes_tokens')}")

        # Default ON. Set BASEVAULT_NO_COMBINE=1 to disable batching of
        # small consecutive entries — used by the pre/post measurement and
        # as an emergency kill-switch if batching ever causes a regression.
        _combine_small = os.environ.get("BASEVAULT_NO_COMBINE", "") not in ("1", "true", "True")
        docs = split_documents(
            all_docs, budget_tokens=_chunk_budget, combine_small=_combine_small,
        )
        # Early first-extraction batch: shrink the first emitted
        # unit so the extract stage's first LLM call (already gate-exempt
        # as the stage's first dispatch) returns fast and real extracted
        # facts surface within seconds-to-low-minutes instead of waiting
        # on a full-size first call. Model-agnostic; the cut reuses the
        # existing splitter backoff and only sub-divides the first unit,
        # so the final fact union is unchanged. Default ON; set
        # BASEVAULT_NO_EARLY_BATCH=1 as an emergency kill-switch (mirrors
        # the BASEVAULT_NO_COMBINE precedent).
        if os.environ.get("BASEVAULT_NO_EARLY_BATCH", "") not in ("1", "true", "True"):
            docs = carve_first_batch(docs, rest_budget=_chunk_budget)
        _n_batched = sum(1 for d in docs if d.metadata.get("combined_entries"))
        _n_inner = sum(
            len(d.metadata.get("combined_entries") or []) for d in docs
        )
        _log_write(
            f"Split → {len(docs)} doc(s) total"
            + (f" (batched: {_n_batched} unit(s) covering {_n_inner} entries)"
               if _n_batched else "")
        )
        # Phase 1 marker (extraction): payload covers the splitter's
        # per-input report (one row per source doc — fits/oversize +
        # how it would be cut) and the post-split segments (one row
        # per chunk that becomes a Phase 2 LLM call).
        _dump("stages/01-extraction/phase_1_marker.json", {
            "report": split_rep,
            "segments": [
                {"id": d.id, "source_path": d.source_path,
                 "source_type": d.source_type.value, "content_len": len(d.content),
                 "title": d.title, "date": d.date,
                 "origin_char": d.origin_char,
                 "split_of": d.metadata.get("split_of"),
                 "split_first50": d.metadata.get("split_first50"),
                 "split_last50": d.metadata.get("split_last50"),
                 "combined_count": d.metadata.get("combined_count"),
                 "combined_entry_ids": [
                     e["id"] for e in (d.metadata.get("combined_entries") or [])
                 ] or None}
                for d in docs
            ],
        })

    def _estimate_per_stage(docs_list) -> dict[str, int]:
        """Post-ingest per-stage call estimate. Builds PipelineStats
        from REAL parsed Documents + splitter output, then calls the
        SAME `estimate_pipeline` the preflight tick uses. ONE
        estimator, two stats producers."""
        from engine.ingestor import pipeline_stats_from_docs
        from engine.progress import estimate_pipeline
        from engine.llm import chunk_cap_for_stage
        from engine.content_extractor import _topics_for_run
        from collections import defaultdict as _dd
        # Splitter output count (one chunk per doc id; chunks share
        # the parent file_id when the splitter divided one input
        # into multiple Documents).
        parents = _dd(int)
        for d in docs_list:
            parents[d.file_id or d.id] += 1
        n_splits = sum(parents.values())
        stats = pipeline_stats_from_docs(
            docs_list,
            n_splits=n_splits,
            chunk_cap=max(1, chunk_cap_for_stage(mode, "extract")),
            n_topics=max(1, len(_topics_for_run())),
        )
        est = estimate_pipeline(stats, is_local=(mode_str == "local"))
        out = dict(est.calls_per_stage)
        # Strip vision row when no images — issue #115 acceptance
        # criterion ("no `vision` row in `by_stage` if no images
        # were ingested"). Symmetric guarantee on the est-calls
        # rollup table.
        if out.get("vision", 0) <= 0:
            out.pop("vision", None)
        return out


    def _register_initial_estimates(per_stage: dict[str, int]) -> None:
        """Push `_estimate_per_stage` output into the ProgressTracker so
        the cumulative pipeline total reflects every upcoming stage."""
        if _progress_tracker is None:
            return
        for stage, count in per_stage.items():
            model = _stage_model_map.get(stage, "?")
            _progress_tracker.register_stage(stage, model, count)

    facts_dir = out_dir / "stages" / "01-extraction" / "facts"
    # Per-fact entity mention stream lands directly under the Stage 2
    # entities surface so the run tree's existing Entities node
    # populates as facts roll in mid-Stage-1 — no separate Stage 1
    # location to wire into the UI. Stage 2 Phase 1 wipes-and-rewrites
    # this same dir with canonical post-grouping records.
    stream_entities_dir = out_dir / "stages" / "02-entities" / "entities"

    # Per-LLM-call records collected as splits resolve. Persisted to
    # `phase_2_marker.json` at end of phase. One entry per LLM call
    # (one per split); aggregate timing / token data lives in
    # `llm-calls.jsonl`, but the marker carries enough to reason
    # about WHAT each call produced (topics + types + counts) without
    # cross-referencing.
    import threading as _threading
    _extract_calls: list[dict] = []
    _extract_calls_lock = _threading.Lock()

    def _persist_split(doc, items, summaries=None):
        """Per-split callback fired by extract_items as soon as one split's
        LLM call returns and parses. Each item is appended (as a single
        JSON line) to every TOPIC.jsonl bucket it claims, AND each
        entity mention in the item is appended to its per-entity JSONL
        bucket under `stages/02-entities/entities/`. Concurrent appends
        across parent threads are serialized by per-file locks.

        The facts buckets are the in-flight UI feed AND the resume input
        — Phase 3 sorts them in place to produce the canonical view.
        The entities buckets share the same on-disk location the Stage
        2 Phase 1 consolidation will rewrite — that lets the run tree's
        existing Entities node populate as facts stream in mid-Stage-1
        without a separate listing surface. The canonical post-Phase-3
        facts file remains the source of truth for Stage 2's grouping
        logic; the in-flight per-mention lines get overwritten by the
        consolidation pass.

        Also records the per-call manifest entry — chunk id, source
        file, byte offsets, item counts, AND the per-split summaries
        the extract LLM produced (one entry per split id; for batched
        docs that's one per inner entry). The summaries land in
        `phase_2_marker.json` so downstream consumers (RAG / chunk
        enrichment) can pick them up via the same chunk_id join key."""
        from engine.entities import _slugify as _entity_slug
        for it in items:
            d = _serialize_fact_dict(it)
            for topic in it.topics:
                _append_fact_jsonl(facts_dir / f"{topic}.jsonl", d)
            for ref in it.entities:
                name = (ref.entity.name or "").strip()
                if not name:
                    continue
                slug = _entity_slug(name) or "entity"
                mention = _serialize_entity_mention_dict(it, ref)
                _append_entity_jsonl(
                    stream_entities_dir / f"{slug}.jsonl", mention,
                )
        # One LLM call ↔ one split/pack from Phase 1. Record what the
        # call saw + produced. `chunk_id` is the per-split id the
        # splitter assigned (e.g. "<file>::split_02" or a pack id);
        # `source_file` is the original input doc it came from.
        # `first50` / `last50` peek at the chunk's text edges for
        # human cross-reference against the source. `origin_char` is
        # the byte offset within the source where this chunk starts.
        topics_seen = sorted({t for it in items for t in (it.topics or [])})
        types_seen = sorted({(it.item_type or "fact") for it in items})
        summaries_list = [
            {"id": sid, "summary": text}
            for sid, text in sorted((summaries or {}).items())
        ]
        record = {
            "chunk_id": doc.id,
            "source_file": doc.file_id or doc.id,
            "origin_char": doc.origin_char,
            "first50": doc.metadata.get("split_first50"),
            "last50": doc.metadata.get("split_last50"),
            "n_items": len(items),
            "topics": topics_seen,
            "item_types": types_seen,
            "split_summaries": summaries_list,
        }
        with _extract_calls_lock:
            _extract_calls.append(record)

    from collections import defaultdict

    if resume_from == "start":
        per_stage = _estimate_per_stage(docs)
        _register_initial_estimates(per_stage)
        # Vision is done by this point; mark the stage finished so it
        # doesn't sit in-flight and so any over-estimate (n_vision >
        # number of successful records, e.g. some calls failed silently)
        # tightens to the actual completed count.
        #
        # Pre-#216 (where vision routed through `describe_image` and
        # bypassed the chat wrapper), this block also retroactively
        # credited each successful stat record into the tracker via
        # `record_call("vision", ...)` and bumped the runner's
        # `_completed` global by the same count. Post-#216 vision
        # routes through `complete()` like every other chat stage —
        # the wrapper already calls `record_call_begin/end` and bumps
        # `_completed` on success in real time. The retroactive sweep
        # is now redundant AND double-counts (issue #244): the chip
        # showed `total=25` for 20 actual calls because vision's
        # `completed_calls` was 5 (real-time) + 5 (sweep) = 10, then
        # `mark_stage_finished` froze est at 10.
        if per_stage.get("vision", 0) > 0:
            _progress_tracker.mark_stage_finished("vision")
        revised_total = sum(per_stage.values())
        _emit_total(revised_total)
        _log_write(
            f"Revised estimate: {revised_total} LLM calls expected "
            f"(per stage: {per_stage})"
        )

        parents_map: dict[str, list] = defaultdict(list)
        for d in docs:
            parents_map[d.file_id or d.id].append(d)

        if not parents_map:
            _log_write("No documents to process after split; skipping extraction.")
            _mark_run_terminal("completed")
            if _log_file:
                _log_file.close()
            print(json.dumps({
                "stage": "done",
                "completed": _completed,
                "facts": 0, "patterns": 0, "insights": 0, "actions": 0,
                "output_dir": str(out_dir),
            }), flush=True)
            return

        # Phase 2 entry sanity sweep: repair any per-topic JSONL whose
        # last line was truncated mid-write by a process kill. Idempotent
        # on well-formed files; redundant under the start-path wipe
        # below but kept here as a safety net for any future entry path
        # that hits the extract loop without wiping first.
        _repair_partial_jsonl_tails(facts_dir)

        # Phase 2 (Extract LLM): wipe any stale per-topic JSONL bucket
        # files from a prior interrupted run BEFORE the threadpool starts
        # appending. The LLM cache makes re-execution fast; the bucket
        # files have to start clean so duplicates don't pile up. Wipe the
        # parallel per-entity in-flight buckets under
        # `02-entities/entities/` for the same reason — a re-run would
        # otherwise pile new mention lines onto stale ones (and Stage 2
        # Phase 1's wipe-then-rewrite happens later, leaving a window of
        # mixed state during early extraction).
        if facts_dir.exists():
            for p in facts_dir.glob("*.jsonl"):
                p.unlink()
        else:
            facts_dir.mkdir(parents=True, exist_ok=True)
        if stream_entities_dir.exists():
            for p in stream_entities_dir.glob("*.jsonl"):
                p.unlink()
        else:
            stream_entities_dir.mkdir(parents=True, exist_ok=True)

        # Phase 2: Extraction (LLM) — _stage transitioned to "extract"
        # at the splitter entry above; just log the phase boundary
        # here without re-emitting a stage transition (the bar is
        # already on "extract").
        _log_write("── Phase 2: extract (LLM) — Stage 1 (Extraction) ──")

        # Per-parent breakdown manifest. Splits across parents are
        # interleaved by the unified pool below — this manifest
        # documents the input shape, NOT processing order.
        for fid, parent_splits in parents_map.items():
            _log_write(
                f"── {fid}: {len(parent_splits)} split(s), "
                f"total {sum(len(d.content) for d in parent_splits)} chars ──"
            )

        # ONE POOL for the extract stage. extract_items owns the only
        # ThreadPoolExecutor that fires LLM calls in this stage; the
        # runner used to wrap it in an outer per-parent pool, making
        # effective concurrency multiplicative (parents × splits)
        # instead of capped at max_workers(mode). Hand all splits across
        # all parents in one call so the inner pool's
        # `min(len(docs), max_workers(mode))` is the true cap.
        # The kernel path fires on_split_complete per chunk after run_all
        # drains — so _extract_calls / the split-ids marker populate the same,
        # just not mid-stage incrementally.
        from kernel.enums import PhaseName as _PhaseName
        from engine.phases.extraction_llm import run_extraction_all
        from engine.phases.model_specs import build_stage_env
        all_items.extend(run_extraction_all(
            docs,
            build_stage_env(_PhaseName.EXTRACTION_LLM, mode, extra_hooks=[_KERNEL_LIVE_HOOK]),
            on_split_complete=_persist_split,
        ))

        # Per-parent rollup AFTER the unified pool drains — sourced
        # from the per-call records `_persist_split` accumulated as
        # splits resolved (each carries source_file + n_items).
        _per_parent_items: dict[str, int] = defaultdict(int)
        with _extract_calls_lock:
            for r in _extract_calls:
                _per_parent_items[r["source_file"] or ""] += int(r["n_items"])
        for fid in parents_map.keys():
            _log_write(
                f"  [{fid}] {_per_parent_items.get(fid, 0)} item(s) extracted"
            )

        # Phase 2 marker — one record per LLM call (one per split/pack
        # that came out of Phase 1), sorted by source_file + origin_char
        # so the same input order from disk yields the same marker.
        # Token / latency data lives in llm-calls.jsonl (the system-wide
        # LLM-call log); the marker carries the extraction-specific
        # outcomes (chunks → topics + item_types + counts) so it's
        # self-contained for inspection.
        with _extract_calls_lock:
            calls_sorted = sorted(
                _extract_calls,
                key=lambda r: (r["source_file"] or "",
                               r["origin_char"] if r["origin_char"] is not None else 0),
            )
        _dump("stages/01-extraction/phase_2_marker.json", {
            "calls": calls_sorted,
            "n_calls": len(calls_sorted),
            "n_items": len(all_items),
            "n_source_files": len(parents_map),
        })

        # Phase 3 (Completion, deterministic): sort each per-topic JSONL
        # in place by (file_path, file_offset, summary) — same key the
        # in-memory aggregator uses below — so on-disk JSONL and
        # in-memory canonical agree on ordering and IDs.
        topic_counts: dict[str, int] = {}
        for jsonl_path in sorted(facts_dir.glob("*.jsonl")):
            dicts = _parse_jsonl_facts_tolerant(jsonl_path)
            dicts.sort(key=_fact_sort_key)
            tmp = jsonl_path.with_suffix(".jsonl.tmp")
            tmp.write_text(
                "\n".join(json.dumps(d, ensure_ascii=False) for d in dicts) + "\n",
                encoding="utf-8",
            )
            tmp.replace(jsonl_path)
            topic_counts[jsonl_path.stem] = len(dicts)

        _dump("stages/01-extraction/phase_3_marker.json", {
            "topics": topic_counts,
            "total_facts": sum(topic_counts.values()),
        })

        # In-memory facts_by_topic for downstream stages — sorted by the
        # same key, so IDs assigned at this layer match the JSONL line
        # order.
        _facts_by_topic = defaultdict(list)
        for it in all_items:
            for topic in it.topics:
                _facts_by_topic[topic].append(it)

        def _sort_key(it):
            ev = it.evidence[0] if it.evidence else None
            return (
                (ev.file_path or "") if ev else "",
                (ev.file_offset if ev and ev.file_offset is not None else 0),
                it.summary,
            )

        facts_by_topic = {
            topic: sorted(items, key=_sort_key)
            for topic, items in _facts_by_topic.items()
        }

    elif resume_from in ("entities", "patterns", "insights", "actions", "embeddings"):
        _log_write("Loading facts_by_topic from disk...")
        facts_by_topic = _load_facts_by_topic(_run_dir)
        _log_write(f"Loaded {sum(len(v) for v in facts_by_topic.values())} facts across {len(facts_by_topic)} topics")
        # Resume skipped ingest, so `all_docs` is still empty. The
        # embeddings plan derives every chunk-kind record from it;
        # without this reload the resumed run embeds zero chunks and
        # marks itself complete on a chunkless store.
        all_docs = _load_documents(_run_dir)
        _log_write(f"Loaded {len(all_docs)} preprocessed document(s) for the embeddings chunk plan")
        # Manifest wasn't built this run (resume skipped ingest).
        # Re-load the global manifest from disk so entity batching has
        # a stable position lookup and any cache hits from prior runs
        # carry through.
        from engine.files_manifest import (
            load_manifest as _load_manifest,
            position_index as _manifest_position_index,
        )
        manifest_pos = _manifest_position_index(_load_manifest())
        # Register all upcoming pipeline stages so the tracker's
        # total_calls_pipeline reflects the FULL pipeline from cycle 2
        # onward, not just the past-stage seed. Fresh runs do this at
        # the resume_from=="start" branch above; on resume the only
        # registration the runner used to perform was the per-stage
        # seed of prior cycles' completed counts — future stages
        # registered lazily as each one entered its body. Net effect:
        # emitted `total` capped at the seed sum while Rust derive's
        # `completed` keeps accumulating across cycles; once completed
        # catches up the JSX bar clamps `displayDone = min(completed,
        # total) = total` and the chip pins at "X/X" for the rest of
        # the run.
        if _progress_tracker is not None:
            _register_initial_estimates(_estimate_per_stage(all_docs))
            _emit_total(_progress_tracker.compute_pipeline_total_calls())
            _log_write(
                f"Resume: pre-registered upcoming pipeline stages; "
                f"tracker total = "
                f"{_progress_tracker.compute_pipeline_total_calls()}"
            )

    # Issue #339: collapse fresh-vs-resume ordering drift at the
    # phase boundary. The fresh path above sorts items within each
    # topic but iterates topics in insertion order (whichever topic
    # an extract-stage fact mentioned first); the resume path walks
    # `sorted(facts_dir.glob("*.jsonl"))` alphabetically. Downstream
    # `entities._group_entities` iterates `facts_by_topic.items()`
    # verbatim, so the per-group `g.facts` / `g.evidence_fact_refs`
    # accumulate in different orders across the two paths, and the
    # rendered prompts hash to different cache keys on resume. One
    # canonical sort here erases both drifts at once.
    facts_by_topic = _normalize_facts_by_topic(facts_by_topic)

    # The runner doesn't write to a vault dir during the pipeline.
    # Vault export is on-demand via the Export button — JS-side
    # `regenVault()` reads each stage's payload through the existing
    # `read_run_*` Tauri commands and writes the rendered markdown
    # through `write_run_vault`.

    def _reemit_total(remaining: int):
        """Refine the CURRENT stage's call count: `remaining` is
        what's left from now until end of this stage. Update the
        tracker's stage est_calls = completed_in_stage + remaining,
        then re-emit the pipeline-cumulative total.

        Pre-PR semantic: `_completed + remaining` was emitted as the
        "total" — but `remaining` was per-stage, so the bar denominator
        only ever reached end-of-current-stage and shrank back when
        the next stage updated its estimate. The exact bug surfaced
        as 66/68 (97%) at "Resolving entities" while patterns +
        insights + actions still needed to run.

        Now: tracker accumulates est_calls across all stages, and the
        emitted total is the sum.
        """
        if _progress_tracker is None:
            revised = _completed + remaining
            _emit_total(revised)
            _log_write(
                f"Revised estimate: {revised} ({_completed} done + "
                f"{remaining} remaining)")
            return
        cur_stage = _stage
        cur_model = _stage_model_map.get(cur_stage, "?")
        with _progress_tracker._lock:
            existing = _progress_tracker._stages.get(cur_stage)
            completed_in_stage = existing.completed_calls if existing else 0
        _progress_tracker.register_stage(
            cur_stage, cur_model, completed_in_stage + max(0, remaining))
        new_total = _progress_tracker.compute_pipeline_total_calls()
        _emit_total(new_total)
        _log_write(
            f"Revised estimate (stage={cur_stage}): pipeline total "
            f"{new_total}; stage='{cur_stage}' est now "
            f"{completed_in_stage + remaining} "
            f"(completed_in_stage={completed_in_stage}, "
            f"remaining={remaining})")

    # ── Entities (resolve canonical entities + relations + subject) ──────────
    entities_output = None
    if resume_from in ("start", "entities"):
        _stage = "entities"
        _log_stage("entities")
        # Per-entity rewrite: ~sqrt(N_groups) per-entity Phase-C calls
        # in this stage; Phase-D dedupe is its own (entities_dedupe)
        # stage with its own est_calls. Estimate N_groups from fact
        # count using ~5 facts/group typical ratio (Pepys empirics).
        _n_facts_so_far = sum(len(v) for v in facts_by_topic.values())
        _n_groups_est = max(1, _n_facts_so_far // 5)
        _n_entities_est = max(1, int(_n_groups_est ** 0.5))
        _reemit_total(_n_entities_est)
        _emit()

        # Per-phase markers + per-entity files. detect_entities calls
        # back at the end of each sub-phase (deterministic prep,
        # per-entity LLM batches). Marker payloads carry the full
        # phase view; per-entity files at `stages/02-entities/entities/`
        # are the user-facing surface and populate as soon as Phase 1
        # finishes (deterministic grouping) so the in-app run tree's
        # Entities section can render mid-stage. Phase 2 enriches each
        # file in place with the LLM-produced description / role /
        # finalized relations; Phase 3 wipes-and-rewrites with the
        # post-dedupe canonical view.
        def _entities_safe_id(cid: str) -> str:
            return "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in cid
            )

        def _dump_entity_ui_file(safe: str, record: dict) -> None:
            """Serialize one per-entity (UI-derived) file, collapsing the
            cross-category fact fan-out so the entity's "Cited in facts"
            list shows one row per fact. Applied here — the single write
            path for these files — so it holds at every phase (1 grouping,
            2 enrichment, 3 post-dedupe rewrite) by construction. The
            canonical phase_3 marker keeps the full mention map separately,
            as the authoritative fact->entity map for patterns context and
            the RAG graph."""
            from engine.entities import _collapse_cross_category_fact_refs
            out = dict(record)
            out["evidence_fact_refs"] = [
                list(r) for r in _collapse_cross_category_fact_refs(
                    [tuple(r) for r in out.get("evidence_fact_refs", [])],
                    facts_by_topic,
                )
            ]
            _dump_jsonl_record(
                f"stages/02-entities/entities/{safe}.jsonl", out,
            )

        def _entities_write_phase1(payload: dict) -> None:
            entities_subdir = out_dir / "stages" / "02-entities" / "entities"
            entities_subdir.mkdir(parents=True, exist_ok=True)
            # Wipe the in-flight per-mention streams that Stage 1 Phase 2
            # accumulated under the same .jsonl paths. Two reasons:
            # (1) we're about to overwrite each surviving canonical's
            # file with a single-line consolidated record — leaving the
            # multi-line stream in place would mix mention-level and
            # canonical-level shapes; (2) groups whose canonical_name
            # got promoted during deterministic grouping land at a
            # different slug than the raw mention name, so the
            # original mention-stream file would otherwise linger as
            # an orphan until Phase 3's post-dedupe wipe.
            #
            # We sequence as write-new-then-delete-orphans rather than
            # delete-all-then-write so the run tree's Entities count
            # never flickers to zero during Stage 2 Phase 1: at any
            # moment, every surviving entity has either its old (mention
            # stream) or new (canonical) file on disk. Orphans (slugs
            # that won't be in the post-grouping set) are unlinked AFTER
            # the new writes land.
            pre_existing = {
                p.stem for p in entities_subdir.glob("*.jsonl")
            }
            written: dict[str, dict] = {}
            for g in payload.get("groups", []):
                cid = g.get("canonical_id") or _entities_safe_id(
                    g.get("canonical_name", "entity")
                )
                safe = _entities_safe_id(cid)
                if safe in written:
                    # Rare canonical_id collision: two distinct _Group
                    # buckets slugified to the same id. Merge their
                    # evidence + candidate_relations into one file; Phase
                    # D's LLM dedupe will further consolidate as needed.
                    existing = written[safe]
                    existing_refs = {tuple(r) for r in existing["evidence_fact_refs"]}
                    for r in g.get("evidence_fact_refs", []):
                        t = tuple(r)
                        if t not in existing_refs:
                            existing["evidence_fact_refs"].append(list(r))
                            existing_refs.add(t)
                    existing["candidate_relations"].extend(
                        g.get("candidate_relations", [])
                    )
                    existing["mention_count"] = (
                        existing.get("mention_count", 0)
                        + g.get("mention_count", 0)
                    )
                    for top in g.get("topics", []):
                        if top not in existing["topics"]:
                            existing["topics"].append(top)
                    _log_write(
                        f"  [entities] phase_1 canonical_id collision: "
                        f"{safe!r} merged from second group "
                        f"{g.get('canonical_name')!r}"
                    )
                    _dump_entity_ui_file(safe, existing)
                    continue
                record = {
                    "canonical_id": cid,
                    "canonical_name": g.get("canonical_name", ""),
                    "entity_type": g.get("entity_type", ""),
                    "mention_count": g.get("mention_count", 0),
                    "topics": list(g.get("topics", [])),
                    "evidence_fact_refs": [
                        list(r) for r in g.get("evidence_fact_refs", [])
                    ],
                    "candidate_relations": [
                        dict(c) for c in g.get("candidate_relations", [])
                    ],
                }
                written[safe] = record
                _dump_entity_ui_file(safe, record)
            # Orphans: pre-existing slugs that didn't survive the
            # mention-name → canonical-name promotion (e.g. mention
            # "Alice" + "Alice Smith" → canonical "Alice Smith"; the
            # Alice file is now stale). Drop them now that the new
            # files are on disk.
            for stem in pre_existing - set(written.keys()):
                stale = entities_subdir / f"{stem}.jsonl"
                try:
                    stale.unlink()
                except OSError as e:
                    _log_write(
                        f"  [entities] phase 1 orphan cleanup could not "
                        f"unlink {stale.name}: {type(e).__name__}: {e}"
                    )

        def _entities_enrich_phase2(payload: dict) -> None:
            annotations = payload.get("annotations", {}) or {}
            groups_meta = payload.get("groups_by_gid", {}) or {}
            entities_subdir = out_dir / "stages" / "02-entities" / "entities"

            # Iterate clones in deterministic order so the per-file
            # description/role pick (first non-empty wins) is stable
            # across runs.
            for gid in sorted(annotations.keys()):
                llm_resp = annotations[gid] or {}
                meta = groups_meta.get(gid, {})
                parent_gid = meta.get("parent_gid", gid)
                parent_meta = groups_meta.get(parent_gid, meta)
                canonical_id = (
                    parent_meta.get("canonical_id")
                    or meta.get("canonical_id")
                )
                if not canonical_id:
                    continue
                safe = _entities_safe_id(canonical_id)
                path = entities_subdir / f"{safe}.jsonl"
                if not path.exists():
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        rec = json.load(f)
                except Exception:
                    continue
                desc = (llm_resp.get("description") or "").strip()
                if desc and not (rec.get("description") or "").strip():
                    rec["description"] = desc
                role = (llm_resp.get("role") or "").strip()
                if role and not (rec.get("role") or "").strip():
                    rec["role"] = role
                existing_rels = rec.setdefault("relations", [])
                seen = {(r.get("to"), r.get("relation")) for r in existing_rels}
                for rel in llm_resp.get("relations", []):
                    target_gid = rel.get("to_id", "")
                    target_meta = groups_meta.get(target_gid, {})
                    target_parent_gid = target_meta.get("parent_gid", target_gid)
                    target_parent_meta = groups_meta.get(
                        target_parent_gid, target_meta
                    )
                    target_id = target_parent_meta.get("canonical_id")
                    if not target_id or target_id == canonical_id:
                        continue
                    verb = (
                        str(rel.get("verb", "")).strip().lower().replace(" ", "_")
                    )
                    key = (target_id, verb)
                    if key in seen:
                        continue
                    seen.add(key)
                    existing_rels.append({
                        "to": target_id,
                        "relation": verb,
                        "confidence": float(rel.get("confidence", 1.0)),
                    })
                _dump_entity_ui_file(safe, rec)

        def _entities_phase_done(phase: str, payload: dict) -> None:
            _dump(f"stages/02-entities/{phase}_marker.json", payload)
            if phase == "phase_1":
                _entities_write_phase1(payload)
            elif phase == "phase_2":
                _entities_enrich_phase2(payload)

        # The kernel path fires the SAME on_phase_done callback (phase_1 after
        # grouping, phase_2 after the summarize batches) so the per-entity
        # grouping/enrichment files + markers stream mid-stage.
        from engine.phases.entities_job import (
            build_entities_env,
            detect_entities,
        )
        _ent_env = build_entities_env(mode, extra_hooks=[_KERNEL_LIVE_HOOK])
        entities_output = detect_entities(
            facts_by_topic, mode, model=None, subject=subject,
            manifest_pos=manifest_pos, execution_env=_ent_env,
            on_phase_done=_entities_phase_done,
        )
        n_entities = len(entities_output.entities)
        n_relations = len(entities_output.relations)
        subj_id = entities_output.subject.canonical_id if entities_output.subject else None
        _log_write(f"  {n_entities} canonical entities, {n_relations} relations, subject={subj_id}")

        # Stamp the entities-stage routing decisions on run.json's
        # run_config so post-hoc analysis can answer "which subject
        # branch did this run take, and was the input treated as a
        # bundle". Both feed into how downstream patterns/insights/
        # actions stages frame their prompts.
        if entities_output.subject is not None:
            _set_run_subject_resolution({
                "canonical_id": entities_output.subject.canonical_id,
                "display": entities_output.subject.display,
                "source": entities_output.subject.source,
            })
        else:
            _set_run_subject_resolution(None)
        try:
            from engine.entities import _is_bundle as _is_bundle_fn
            _set_run_bundle_mode(_is_bundle_fn(facts_by_topic))
        except Exception:
            pass

        # Per-entity files under stages/02-entities/entities/<ENTITY>.jsonl
        # — one file per canonical entity. Aspirational doc layout; lets
        # the UI display per-entity without parsing one big aggregate.
        # Phase D's deterministic + LLM dedupe may have collapsed /
        # renamed groups whose Phase-1/2 files are still on disk; we
        # write each new canonical's file FIRST (atomic .tmp+rename
        # via _dump_jsonl_record) and THEN drop any pre-existing files
        # whose stem isn't in the new set. Sequencing the writes ahead
        # of the orphan deletes keeps the run tree's Entities count
        # from flickering to zero between phases.
        entities_subdir = out_dir / "stages" / "02-entities" / "entities"
        entities_subdir.mkdir(parents=True, exist_ok=True)
        pre_existing_p3 = {
            p.stem for p in entities_subdir.glob("*.jsonl")
        }
        kept: set[str] = set()
        for e in entities_output.entities:
            safe_name = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in e.canonical_id
            )
            kept.add(safe_name)
            _dump_entity_ui_file(safe_name, {
                "canonical_id": e.canonical_id,
                "canonical_name": e.canonical_name,
                "entity_type": e.entity_type,
                "aliases": e.aliases,
                "role": e.role,
                "description": e.description,
                "mention_count": e.mention_count,
                "topics": e.topics,
                "evidence_fact_refs": [list(r) for r in e.evidence_fact_refs],
                "relations": [
                    {"to": r.to_id, "relation": r.relation,
                     "confidence": r.confidence,
                     "evidence_fact_refs": [list(x) for x in r.evidence_fact_refs]}
                    for r in entities_output.relations if r.from_id == e.canonical_id
                ],
            })
        for stem in pre_existing_p3 - kept:
            stale = entities_subdir / f"{stem}.jsonl"
            try:
                stale.unlink()
            except OSError as ex:
                _log_write(
                    f"  [entities] phase 3 orphan cleanup could not "
                    f"unlink {stale.name}: {type(ex).__name__}: {ex}"
                )

        # Stage canonical: phase 3 marker carries the full aggregate
        # (subject + entities + relations). This is what _load_entities
        # reads; per-entity files are a derived view for UI use.
        _dump("stages/02-entities/phase_3_marker.json", {
            "subject": (
                {"canonical_id": entities_output.subject.canonical_id,
                 "display": entities_output.subject.display,
                 "source": entities_output.subject.source}
                if entities_output.subject else None
            ),
            "entities": [
                {
                    "canonical_id": e.canonical_id,
                    "canonical_name": e.canonical_name,
                    "entity_type": e.entity_type,
                    "aliases": e.aliases,
                    "role": e.role,
                    "description": e.description,
                    "mention_count": e.mention_count,
                    "topics": e.topics,
                    "evidence_fact_refs": [list(r) for r in e.evidence_fact_refs],
                }
                for e in entities_output.entities
            ],
            "relations": [
                {
                    "from": r.from_id,
                    "to": r.to_id,
                    "relation": r.relation,
                    "confidence": r.confidence,
                    "evidence_fact_refs": [list(x) for x in r.evidence_fact_refs],
                }
                for r in entities_output.relations
            ],
        })

    elif resume_from in ("patterns", "insights", "actions", "embeddings"):
        _log_write("Loading entities_output from disk...")
        entities_output = _load_entities(_run_dir)
        if entities_output is not None:
            _log_write(
                f"Loaded {len(entities_output.entities)} entities, "
                f"{len(entities_output.relations)} relations"
            )

    # Issue #339: same phase-boundary normalization rule as
    # facts_by_topic above. `detect_entities` returns lists in
    # internal iteration order; the marker write preserves that;
    # `_load_entities` reads back in marker order. Both paths feed
    # the same downstream consumers (patterns / insights / actions
    # build prompts that embed entity descriptions + relations in
    # iteration order) so a canonical sort here is the cache-stable
    # contract regardless of where the data came from.
    entities_output = _normalize_entities_output(entities_output)

    # Resolve subject string to the canonical display name for downstream
    # prompts. Falls back to the original subject arg if resolution failed.
    #
    # Override guard: when the CLI subject string is a generic placeholder
    # AND the entities stage only found a subject via the mention-count
    # fallback tier, we do NOT replace the CLI string. That fallback picks
    # the most-mentioned person, which is frequently a peripheral figure
    # rather than the narrator — threading that display name into the
    # patterns/insights/actions prompts causes false attribution. Keeping
    # the CLI string intact lets the stages' _SUBJECT_DISCIPLINE prompt
    # blocks operate on the caller's original phrasing, same as before
    # this PR.
    _GENERIC_SUBJECTS = {"the author", "subject", "me", ""}
    resolved_subject = subject
    entities_context: str | None = None
    entities_context_by_topic: dict[str, str] = {}
    if entities_output is not None:
        from dataclasses import replace
        from engine.entities import build_context_block
        subj = entities_output.subject
        is_generic_cli = subject.strip().lower() in _GENERIC_SUBJECTS
        untrustworthy = subj is not None and subj.source == "mention_count_fallback"
        if subj is not None and is_generic_cli and untrustworthy:
            # Skip override: keep CLI string as resolved_subject, AND scrub
            # the untrusted subject out of the context block so downstream
            # prompts aren't told "subject: Guy in White Hoodie" while the
            # _SUBJECT_DISCIPLINE block says "subject is the author".
            # on-disk entities.json still records what the stage produced —
            # only the prompt-prefix view is filtered.
            _log_write(
                f"  subject override skipped: CLI='{subject}' (generic) + "
                f"entities.subject.source={subj.source!r}; keeping "
                f"resolved_subject='{subject}' and scrubbing untrusted "
                f"subject from prompt context"
            )
            ctx_output = replace(entities_output, subject=None)
        else:
            ctx_output = entities_output
            if subj is not None:
                resolved_subject = subj.display

        # Run-scoped block consumed by insights + actions. Cap scales
        # with the total entity count: `max(100, x/10)` covers ~80%+
        # of mention mass on long-tail corpora (m7pp: 3,373 entities,
        # top 10% ≈ 84% of mentions) without unbounded prompt growth.
        # Small corpora hit the 100-entity floor.
        x_run = len(ctx_output.entities)
        run_cap = max(100, x_run // 10)
        entities_context = build_context_block(
            ctx_output, max_entities=run_cap,
        ) or None

        # Per-topic blocks consumed by patterns. Each topic's block
        # lists only entities referenced in that topic's fact set
        # (derived from the entity → (topic, fact_idx) evidence map),
        # sized by `max(50, x_topic/10)`. Topics with no resolved
        # entities are absent from the dict — detect_patterns_all
        # treats a missing key as "no context block" and proceeds
        # without one (matches the prior global-None behaviour).
        topic_to_entity_ids: dict[str, set[str]] = {}
        for e in ctx_output.entities:
            for ref_topic, _idx in e.evidence_fact_refs:
                topic_to_entity_ids.setdefault(ref_topic, set()).add(
                    e.canonical_id
                )
        for topic in facts_by_topic.keys():
            subset_ids = topic_to_entity_ids.get(topic, set())
            if not subset_ids:
                continue
            x_topic = len(subset_ids)
            topic_cap = max(50, x_topic // 10)
            block = build_context_block(
                ctx_output,
                max_entities=topic_cap,
                entity_subset_ids=subset_ids,
            )
            if block:
                entities_context_by_topic[topic] = block

    # ── Patterns (merged compression + within-theme synthesis) ───────────────
    if resume_from in ("start", "entities", "patterns"):
        _stage = "patterns"
        _log_stage("patterns")

        # Intra-stage checkpoint: load per-topic files already on disk from
        # a prior aborted run; only synthesize topics that are missing.
        existing_patterns = _load_patterns(_run_dir)
        pending = {
            t: facts for t, facts in facts_by_topic.items()
            if t not in existing_patterns
        }
        if existing_patterns:
            _log_write(
                f"  resuming patterns: {len(existing_patterns)} topic(s) "
                f"cached, {len(pending)} pending"
            )

        pattern_calls = sum(1 for facts in pending.values() if len(facts) >= 3)
        _reemit_total(pattern_calls)
        _emit()

        _log_write(f"  synthesizing {len(pending)} topic(s): {sorted(pending.keys())}")
        for topic, facts in pending.items():
            _log_write(f"    {topic}: {len(facts)} facts → patterns...")

        def _persist_topic_patterns(topic: str, pats: list) -> None:
            """Per-topic write fired from inside detect_patterns_all's
            worker pool as each topic's call resolves — drives the
            run-detail tree's incremental fill-in for the patterns
            stage. Skip empty payloads: keeps the patterns dir tidy
            (nothing downstream wants a file full of `[]`); on resume,
            a topic without a file is treated as "pending" and the
            stage re-runs it — cheap, since detect_patterns_all
            short-circuits topics with <3 facts and a genuine-zero
            re-call is one quick cache-hot LLM request."""
            _log_write(f"    {topic}: {len(facts_by_topic[topic])} facts → {len(pats)} patterns")
            if not pats:
                return
            ordered = _sort_patterns_within_topic(pats)
            _dump(f"stages/03-patterns/patterns/{topic}.json", [
                {"name": p.name, "description": p.description,
                 "kind": p.kind, "count": p.count,
                 "source_facts": [[i, c] for i, c in p.source_facts],
                 "hallucinated_ref_count": p.hallucinated_ref_count}
                for p in ordered
            ])

        # The kernel path streams the per-topic patterns files via the SAME
        # on_topic_done callback (fired per topic as its call resolves, in
        # phases/patterns.py) so the run-tree's Patterns node fills in live.
        if not pending:
            new_patterns = {}
        else:
            from kernel.enums import PhaseName as _PhaseName
            from engine.phases.patterns import detect_patterns
            from engine.phases.model_specs import build_stage_env
            new_patterns = detect_patterns(
                pending, mode,
                execution_env=build_stage_env(_PhaseName.PATTERNS, mode, extra_hooks=[_KERNEL_LIVE_HOOK]),
                subject=resolved_subject,
                entities_context_by_topic=entities_context_by_topic,
                on_topic_done=_persist_topic_patterns,
            )

        patterns_by_topic = {**existing_patterns, **new_patterns}
        _dump("stages/03-patterns/phase_1_marker.json", {
            "topics": {t: len(p) for t, p in patterns_by_topic.items()},
            "total_patterns": sum(len(p) for p in patterns_by_topic.values()),
        })

    elif resume_from in ("insights", "actions", "embeddings"):
        _log_write("Loading patterns_by_topic from disk...")
        patterns_by_topic = _load_patterns(_run_dir)
        _log_write(f"Loaded {sum(len(v) for v in patterns_by_topic.values())} patterns across {len(patterns_by_topic)} topics")

    # Issue #339: same phase-boundary normalization rule. The fresh
    # path builds `patterns_by_topic` via dict-union of two sources
    # (existing on-disk topics + newly-synthesized topics from
    # `detect_patterns_all`), so the topic key order is "alphabetical
    # then synthesis-order" — splices the two together in a way that
    # doesn't replicate on resume (where `_load_patterns` walks
    # `sorted(patterns_dir.glob("*.json"))` alphabetically). Sort the
    # outer + inner once at the boundary.
    if "patterns_by_topic" in locals():
        patterns_by_topic = _normalize_patterns_by_topic(patterns_by_topic)

    # ── Insights (cross-topic synthesis) ─────────────────────────────────────
    if resume_from in ("start", "entities", "patterns", "insights"):
        _stage = "insights"
        _log_stage("insights")
        # Insights stage issues exactly one LLM call (cross_domain +
        # critical synthesis are produced from a single complete()
        # invocation). Hardcode 1.
        _reemit_total(1)
        _emit()

        _total_facts_for_synth = sum(
            len(items) for items in facts_by_topic.values()
        )
        # The kernel env carries the KernelTelemetryHook so the per-call
        # llm-calls.jsonl records are produced; the markers / _dump / normalize
        # below are unchanged (they read insight_output).
        from kernel.enums import PhaseName as _PhaseName
        from engine.phases.insights import detect_insights
        from engine.phases.model_specs import build_stage_env
        insight_output = detect_insights(
            patterns_by_topic, mode,
            execution_env=build_stage_env(_PhaseName.INSIGHTS, mode, extra_hooks=[_KERNEL_LIVE_HOOK]),
            subject=resolved_subject,
            entities_context=entities_context,
            total_facts=_total_facts_for_synth,
            sentiment=sentiment,
        )
        # Normalize BEFORE the marker dump, not after, so the marker
        # (what the UI numbers an insight by) stays in the same order
        # the actions source-index space + embeddings record IDs +
        # chatbot grounding all index against; normalizing after the
        # dump would leave the marker in a different order than the
        # records that reference it (the #733 off-by-permutation).
        # The normalizer now PRESERVES the LLM's emission order (see
        # _normalize_insight_output) — we treat that order as already
        # importance-ranked, so there is no by-name re-sort to undo it.
        insight_output = _normalize_insight_output(insight_output)
        _log_write(f"  {len(insight_output.cross_domain)} cross-domain, {len(insight_output.critical)} critical")

        # Phase 1 marker carries the full payload — per the doc, the
        # marker IS the canonical artifact for insights (no separate
        # insights.json file).
        _dump("stages/04-insights/phase_1_marker.json", {
            "cross_domain": [
                {"name": i.name, "description": i.description,
                 "mechanism": i.mechanism, "implication": i.implication,
                 "domains": i.domains, "proposed_actions": i.proposed_actions,
                 "source_patterns": [[t, idx, c] for t, idx, c in i.source_patterns],
                 "hallucinated_ref_count": i.hallucinated_ref_count}
                for i in insight_output.cross_domain
            ],
            "critical": [
                {"name": i.name, "description": i.description,
                 "mechanism": i.mechanism, "implication": i.implication,
                 "domains": i.domains, "proposed_actions": i.proposed_actions,
                 "source_patterns": [[t, idx, c] for t, idx, c in i.source_patterns],
                 "hallucinated_ref_count": i.hallucinated_ref_count}
                for i in insight_output.critical
            ],
        })

    elif resume_from in ("actions", "embeddings"):
        _log_write("Loading insight_output from disk...")
        insight_output = _load_insights(_run_dir)
        # Same canonical order as the fresh path applies before the marker
        # dump — keeps the actions prompt-hash stable and the resume-path
        # ordering identical to fresh.
        insight_output = _normalize_insight_output(insight_output)
        _log_write(f"Loaded {len(insight_output.cross_domain)} cross-domain, "
                   f"{len(insight_output.critical)} critical insights")

    # ── Actions (prioritize and plan) ────────────────────────────────────────
    if resume_from in ("start", "entities", "patterns", "insights", "actions"):
        _stage = "actions"
        _log_stage("actions")
        has_insights = bool(insight_output.cross_domain or insight_output.critical)
        _reemit_total(1 if has_insights else 0)
        _emit()

        _total_facts_for_actions = sum(
            len(items) for items in facts_by_topic.values()
        )
        # Pin `today` once at run start (snapshot) and pass it explicitly
        # to actions so the prompt-assembly path is fully deterministic
        # given (inputs, settings, today). Without an explicit value
        # generate_actions would call date.today() during prompt assembly,
        # which makes the cache key non-deterministic mid-run if the run
        # spans midnight. Same-day re-runs cache-hit; cross-day re-runs
        # bust only the actions stage.
        #
        # `BASEVAULT_TODAY=YYYY-MM-DD` overrides the live snapshot — used
        # by the e2e cached-pipeline test to pin the date to a known value
        # so the actions cache replays cleanly across calendar days. Not
        # documented as a user-facing knob; production unset = today's date.
        from datetime import date as _date
        _today_override = os.environ.get("BASEVAULT_TODAY", "").strip()
        if _today_override:
            try:
                _today_for_actions = _date.fromisoformat(_today_override)
            except ValueError:
                _log_write(
                    f"  WARNING: BASEVAULT_TODAY={_today_override!r} is not a "
                    f"valid YYYY-MM-DD date; falling back to date.today()"
                )
                _today_for_actions = _date.today()
        else:
            _today_for_actions = _date.today()
        # Patterns reference for the actions prompt — per the doc, actions
        # consumes patterns + insights + entities. Insights summarize
        # patterns but the underlying patterns add grounding the LLM uses
        # to refine recommendation specificity. The actions stage re-
        # renders the context block on each WR-sample escalation, so we
        # forward the per-topic Pattern dict rather than a pre-rendered
        # string.
        from kernel.enums import PhaseName as _PhaseName
        from engine.phases.actions import generate_actions
        from engine.phases.model_specs import build_stage_env
        action_list = generate_actions(
            insight_output, mode, _today_for_actions,
            execution_env=build_stage_env(_PhaseName.ACTIONS, mode, extra_hooks=[_KERNEL_LIVE_HOOK]),
            subject=resolved_subject,
            entities_context=entities_context,
            patterns_by_topic=patterns_by_topic,
            total_facts=_total_facts_for_actions,
            sentiment=sentiment,
        )
        _log_write(f"  {len(action_list)} actions generated")

        # Phase 1 marker carries the full payload — per the doc, the marker
        # IS the canonical artifact for actions (no separate actions.json
        # file). Format: {"actions": [...]} so a future addition (e.g.
        # cross-action diagnostics) doesn't break the schema.
        _dump("stages/05-actions/phase_1_marker.json", {
            "actions": [
                {
                    "recommendation": a.recommendation,
                    "kind": a.kind,
                    "objective": a.objective,
                    "why": a.why,
                    "immediate_action": a.immediate_action,
                    "habit": a.habit,
                    "success_metric": a.success_metric,
                    "horizon": a.horizon,
                    "review_date": a.review_date,
                    "regret_reduction": a.regret_reduction,
                    "leverage": a.leverage,
                    "consequence": a.consequence,
                    "generativity": a.generativity,
                    "decisiveness": a.decisiveness,
                    "time_to_feedback": a.time_to_feedback,
                    "constraint_fit": a.constraint_fit,
                    "confidence": a.confidence,
                    "score": a.score,
                    "source_insights": [[k, i, c] for k, i, c in a.source_insights],
                    "hallucinated_ref_count": a.hallucinated_ref_count,
                }
                for a in action_list
            ],
        })

    elif resume_from == "embeddings":
        _log_write("Loading action_list from disk...")
        action_list = _load_actions(_run_dir)
        _log_write(f"Loaded {len(action_list)} actions")

    # Issue #339: phase-boundary normalization for actions. The
    # embeddings stage consumes the action list, so list order is
    # load-bearing for the embed-call batching + cache keys. Fresh
    # path: order set by `actions._parse_output`'s score-descending
    # sort. Resume path: order set by the marker write (the same
    # score-descending order). Route both through the canonicalizer
    # to collapse them onto one shape regardless of fresh-vs-resume.
    action_list = _normalize_action_list(action_list)

    # ── Embeddings ───────────────────────────────────────────────────────────
    # Re-chunks every ingested document at retrieval scale (RAG chunker,
    # parallel to the LLM-call splitter), gathers per-record bare text
    # for facts / entities / patterns / insights / actions, fires
    # batched embed() calls through the shared cascade core, and
    # persists vectors + metadata to a sqlite-vec-backed store under
    # the run's stages tree. Per-record graph enrichment (upstream +
    # downstream context) lands in PR 4.2.
    from engine.embeddings import (
        build_embeddings_plan as _build_embeddings_plan,
        STAGE_NAME as _EMBEDDINGS_STAGE,
        TINFOIL_NOMIC_MODEL_ID as _EMBEDDING_MODEL_ID,
    )
    _stage = _EMBEDDINGS_STAGE
    _log_stage(_EMBEDDINGS_STAGE)
    # `_extract_calls` is the per-LLM-call accumulator the extract
    # phase populated; each entry carries the splitter chunk_id +
    # split_summaries the LLM produced. The enricher uses it to map
    # the splitter chunk ids on fact evidence back to summaries so
    # the chunk-kind embed text can carry the extraction-time gist.
    # Resume-from-checkpoint runs (where extract didn't fire this
    # session) pass an empty list; the prefix builder degrades
    # gracefully (no Split-summaries section in the chunk prefix).
    with _extract_calls_lock:
        _extract_calls_snapshot = list(_extract_calls)
    _embed_plan = _build_embeddings_plan(
        docs=all_docs,
        facts_by_topic=facts_by_topic,
        entities_output=entities_output,
        patterns_by_topic=patterns_by_topic,
        insight_output=insight_output,
        action_list=action_list,
        extract_calls=_extract_calls_snapshot,
    )
    _log_write(
        f"  embedding plan: {len(_embed_plan.records)} records "
        f"({_embed_plan.counts_by_kind()}) "
        f"→ {_embed_plan.num_calls} batched embed call(s)"
    )
    # Hard, surfaced backstop. If ingestion recorded content-bearing
    # documents (chunks ARE expected) but the plan has zero chunk
    # records, refuse to write the store: a chunkless vectors.db is
    # the exact silent failure that shipped — the chatbot binds it and
    # retrieves nothing from the raw input while the run looks
    # "complete". With the resume chunk-source reconstruction above
    # this should not happen; when it does (e.g. every source file
    # deleted off disk) it is a terminal, user-visible error, never a
    # complete marker on an empty store. Keyed off the resume-durable
    # ingestion marker, NOT the documents/*.md snapshot.
    _chunk_records = _embed_plan.counts_by_kind().get("chunk", 0)
    if _chunks_expected(_run_dir) and _chunk_records == 0:
        msg = (
            "Embeddings: source documents were ingested but the chunk "
            "plan is empty — refusing to write a chunkless store (the "
            "chatbot would retrieve nothing from the raw input). The "
            "resume chunk-source could not be reconstructed (original "
            "input file(s) missing or unreadable); re-run the pipeline "
            "on the original input."
        )
        print(json.dumps({"stage": "error", "message": msg}), flush=True)
        _log_write(f"ERROR: {msg}")
        _mark_run_terminal("failed", error=msg)
        return
    # Register the embeddings stage as ONE collective progress unit
    # (issue #581) — NOT its real batch-call count. The stage issues
    # ceil(records / batch_size) sub-second wire calls (~7 for a
    # ~200-record run); counting each as a full pipeline call balloons
    # the bar's denominator, holding the bar back through the long
    # stages then lurching when the ~2s embed burst lands. One unit
    # keeps the bar proportionate while running. The model id is still
    # passed so the tracker keys per-call durations against the
    # embedding model. The REAL count is restored on completion: the
    # end-of-run `mark_stage_finished` sweep snaps embeddings' est_calls
    # down to its actual `completed_calls`, and the done-state chip
    # reads the real leaf count from the jsonl. Display-only — embeddings
    # throughput / batching is untouched. (Chat-stage model resolution
    # doesn't cover embedding models — that path is a follow-up.)
    _EMBED_PROGRESS_UNITS = 1
    if _progress_tracker is not None:
        _progress_tracker.register_stage(
            _EMBEDDINGS_STAGE, _EMBEDDING_MODEL_ID, _EMBED_PROGRESS_UNITS)
        _emit_total(
            _progress_tracker.snapshot()["total_calls_pipeline"])
    _reemit_total(_EMBED_PROGRESS_UNITS)
    _emit()
    _embed_store_path = out_dir / "stages" / "06-embeddings" / "vectors.db"
    # The stage writes the COMPLETE plan and `VectorStore.add` is
    # INSERT-only (no upsert). A vectors.db left by an earlier,
    # incomplete embeddings pass — e.g. the chunkless store a
    # pre-fix resume produced — would otherwise gain duplicate rows
    # on top of its stale ones. Drop it so each pass is idempotent.
    if _embed_store_path.exists():
        _embed_store_path.unlink()
        _log_write(
            "  cleared stale vectors.db — re-embedding the full plan")

    # Embeddings batch on the kernel — the kernel batch data model carries N
    # texts per call and N vectors per LlmResponse, so the phase fires one call
    # per DEFAULT_BATCH_SIZE-record batch (the ~20-32× wire-call reduction).
    # Live progress comes from the KernelTelemetryHook + _KERNEL_LIVE_HOOK.
    from kernel.enums import PhaseName as _PhaseName
    from engine.phases.embeddings import run_embeddings_stage
    from engine.phases.model_specs import build_stage_env
    _embed_result = run_embeddings_stage(
        _embed_plan,
        store_path=_embed_store_path,
        mode=mode,
        execution_env=build_stage_env(
            _PhaseName.EMBEDDINGS, mode, extra_hooks=[_KERNEL_LIVE_HOOK]
        ),
    )
    if _embed_result.calls > 0:
        _completed += _embed_result.calls
    _log_write(
        f"  {_embed_result.calls} embedding call(s) completed → "
        f"{_embed_store_path.relative_to(out_dir)}"
    )
    _emit()
    # Phase 1 marker — counts + call total + model id + dim + store
    # path. Path is recorded relative to the run dir so a copied /
    # moved run remains openable from a sibling tool.
    _dump("stages/06-embeddings/phase_1_marker.json", {
        "calls": _embed_result.calls,
        "counts": _embed_result.counts,
        "model": _embed_result.model,
        "dim": _embed_result.dim,
        "store_path": str(_embed_store_path.relative_to(out_dir)),
        "batch_size": _embed_result.batch_size,
    })

    total_facts = sum(len(items) for items in facts_by_topic.values())
    total_patterns = sum(len(p) for p in patterns_by_topic.values())
    total_insights = len(insight_output.cross_domain) + len(insight_output.critical)
    total_actions = len(action_list)

    # No vault writes here — vault export is post-completion, on demand
    # via the export button (JS `regenVault` + `write_run_vault`).

    # Dump per-call token usage, aggregated per stage + overall. Cost
    # (Tinfoil $/1M token rates) is derived downstream.
    from engine.llm import get_usage_log, get_call_warnings
    _usage = get_usage_log()
    _warnings_list = get_call_warnings()
    if _warnings_list:
        _cap_count = sum(1 for w in _warnings_list if w["kind"] == "max_tokens")
        _empty_count = sum(1 for w in _warnings_list if w["kind"] == "empty")
        _overflow_count = sum(1 for w in _warnings_list if w["kind"] == "input_overflow")
        _summary = []
        if _cap_count: _summary.append(f"{_cap_count} hit output cap")
        if _empty_count: _summary.append(f"{_empty_count} returned empty")
        if _overflow_count: _summary.append(f"{_overflow_count} input overflowed")
        _log_write(
            f"⚠ WARNING: {len(_warnings_list)} LLM call(s) flagged "
            f"({', '.join(_summary)}) — results may be incomplete."
        )
        for w in _warnings_list:
            _log_write(
                f"    [{w['stage']}] {w['kind']} — {w['model']} — "
                f"prompt={w['prompt_tokens']}t, completion={w['completion_tokens']}t "
                f"(cap={w['max_tokens']}t){' — ' + w['note'] if w.get('note') else ''}"
            )
        try:
            print(json.dumps({
                "stage": "warnings",
                "cap_hits": _cap_count,
                "empty_responses": _empty_count,
                "input_overflows": _overflow_count,
            }), flush=True)
        except BrokenPipeError:
            pass
        _set_run_warnings({
            "cap_hits": _cap_count,
            "empty_responses": _empty_count,
            "input_overflows": _overflow_count,
        })
    # LLM-cache observability. Counters live in llm_cache; we surface
    # them in run.json so the UI / debug bundles can answer "how much
    # of this run actually hit the prompt cache". Bypass mode is
    # included so runs that explicitly disabled caching are
    # distinguishable from runs that just had no hits.
    from engine.llm_cache import get_cache_stats as _get_cache_stats
    _cache_stats = _get_cache_stats()
    _set_run_cache_stats(_cache_stats)
    _log_write(
        f"  LLM cache: {_cache_stats['hits']} hit / "
        f"{_cache_stats['misses']} miss "
        f"(stored {_cache_stats['stores']}"
        + (", bypass=on" if _cache_stats.get('bypass') else "")
        + ")"
    )

    # End-of-run rollup side-effects: write llm-stats.txt summary,
    # mirror leaf-aware warnings to in-memory state. The rollup dict
    # itself is no longer cached on disk (issue #189) — consumers
    # derive it on demand via materialize_run_stats(jsonl, config).
    _write_llm_stats(
        out_dir, mode, primary_model=_spec.model_id,
        run_started_at_iso=_run_started_at_iso_local,
    )

    _n_ent = len(entities_output.entities) if entities_output else 0
    _n_rel = len(entities_output.relations) if entities_output else 0
    _log_write(f"── DONE: {_completed} LLM calls, {total_facts} facts, "
               f"{_n_ent} entities ({_n_rel} relations), "
               f"{total_patterns} patterns, {total_insights} insights, "
               f"{total_actions} actions ──")
    _log_write(f"Log: {log_path}")
    if vault_dir is not None:
        _log_write(f"Vault: {vault_dir}")

    # End-of-run close: tighten EVERY unfinished stage to its actual
    # completed count. Issue #244: the prior gate of `started_at is
    # not None` skipped stages that were registered with a non-zero
    # estimate but never entered (e.g. an early-exit run, or a stage
    # whose work dropped to zero items). Their pre-estimate then
    # leaked into the chip's denominator forever, leaving the bar
    # short of 100%. Iterating ALL registered stages — including
    # never-started ones — collapses unentered stages to 0 (their
    # `completed_calls` is 0) and tightens started-but-not-finished
    # stages to actuals. The condition that prior stages were closed
    # at every transition (via `_record_stage_started`'s loop) keeps
    # mid-run behavior unchanged.
    if _progress_tracker is not None:
        for _s, _st in list(_progress_tracker._stages.items()):
            if _st.finished_at is None:
                _progress_tracker.mark_stage_finished(_s)

    _mark_run_terminal("completed")

    if _log_file:
        _log_file.close()

    # Done event. `completed` here is the python wrapper-attempt count
    # — kept for backward compat with shell readers that grep this
    # line, but the UI's `completed` is already at parity with the
    # leaf count via rust derive when status flips to "completed". No
    # `bar_position` in the payload — JSX hard-clamps pct to 100 on
    # status==completed (issue #209).
    print(json.dumps({
        "stage": "done",
        "completed": _completed,
        "facts": total_facts,
        "entities": _n_ent,
        "relations": _n_rel,
        "patterns": total_patterns,
        "insights": total_insights,
        "actions": total_actions,
        "output_dir": str(out_dir),
        "run_name": _run_name,
        "eta_seconds": 0.0,
    }), flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    # One-shot diagnostic emit: drop the run-diagnostics shareable for an
    # already-terminal run dir, then exit. The Rust cancel-settle path
    # invokes this AFTER the canceled run's sidecar is gone — off the
    # SIGKILL-grace window an in-process emit would race — so a canceled
    # run gets the same content-free debug artifact a completed run does
    # (a cancel is high-signal; the diagnostic is exactly what you want to
    # keep). Handled before the run argparse so the pipeline-only required
    # flags (--subject) don't reject it. Best-effort: the emit helper
    # swallows its own errors and the cancel already succeeded regardless.
    if "--emit-run-diagnostic" in sys.argv:
        idx = sys.argv.index("--emit-run-diagnostic")
        try:
            _emit_shareable_run_marker(Path(sys.argv[idx + 1]))
        except Exception:
            pass
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="*", default=[])
    # Mode choices are open-ended: core "local" / "tee" plus any
    # extension mode the eval entry point (``testing.eval._eval_runner``)
    # has registered into MODE_SPEC by argparse time. The
    # run() entry validates the resolved Mode against MODE_SPEC; an
    # unknown string falls back to Mode.LOCAL with no crash.
    parser.add_argument("--mode", default="local")
    parser.add_argument("--resume-run-dir", default=None,
                        help="Absolute path to an existing run dir to resume "
                             "from its last checkpoint. Accepts both flat "
                             "(<logs_root>/<run>/) and the legacy nested "
                             "(<logs_root>/<session>/<eval>/<run>/) shape.")
    parser.add_argument("--resume-run-id", default=None,
                        help="Run id (leaf dir name, e.g. "
                             "'2026-04-22T20-48-02Z-f5pf') to resume from. "
                             "Locates the run dir by scanning $BASEVAULT_LOGS_ROOT "
                             "(or ~/.basevault/logs). Used by the app shell "
                             "for its pause/resume flow. Alternative to "
                             "--resume-run-dir.")
    # Subject + sentiment resolution: CLI flag overrides; otherwise
    # read ~/.basevault/config.json (where the wizard / Settings UI
    # writes them). No env var path — pipeline reads from the two
    # canonical files (dotenv = secrets, config.json = everything
    # else).
    _default_subject = None
    _default_sentiment = _DEFAULT_SENTIMENT
    try:
        import json as _json
        _cfg_path = Path.home() / ".basevault" / "config.json"
        if _cfg_path.exists():
            _cfg = _json.loads(_cfg_path.read_text())
            if isinstance(_cfg, dict):
                _candidate = _cfg.get("subject")
                if isinstance(_candidate, str) and _candidate.strip():
                    _default_subject = _candidate.strip()
                _sent_candidate = _cfg.get("sentiment_bias")
                if (
                    isinstance(_sent_candidate, str)
                    and _sent_candidate.strip() in _VALID_SENTIMENTS
                ):
                    _default_sentiment = _sent_candidate.strip()
    except Exception:
        pass  # config.json unreadable → fall through to argparse required-flag error
    parser.add_argument("--subject", default=_default_subject,
                        required=(_default_subject is None),
                        help="Name of the subject of interest. Required — "
                             "pass the narrator / author's name explicitly so "
                             "subject-discipline prompts anchor on a concrete "
                             "identity. Reads from ~/.basevault/config.json "
                             "`subject` field by default (set via the wizard); "
                             "pass --subject to override.")
    parser.add_argument("--sentiment", default=_default_sentiment,
                        choices=_VALID_SENTIMENTS,
                        help="Tone bias for insights + actions. Default "
                             "neutral. Reads from ~/.basevault/config.json "
                             "`sentiment_bias` field if set. Sentiment "
                             "shifts emphasis and word choice; it never "
                             "overrides source fidelity, the dual "
                             "optimization target (actions), or the kind "
                             "taxonomies.")
    args = parser.parse_args()

    if not args.paths and not args.resume_run_dir and not args.resume_run_id:
        parser.error("--paths is required unless --resume-run-dir or --resume-run-id is given")

    try:
        run(
            args.paths, args.mode,
            resume_run_dir=Path(args.resume_run_dir) if args.resume_run_dir else None,
            resume_run_id=args.resume_run_id,
            subject=args.subject,
            sentiment=args.sentiment,
        )
    except NoResumableCheckpoint as e:
        # Resume was attempted on a run dir with zero intermediate
        # artifacts. Don't mark the run terminally failed — the user
        # should still be able to click Resume (which will restart the
        # stage from scratch in the same dir). Revert to `paused` with
        # a friendly message so the UI explains what happened.
        _log_write(f"No resumable progress: {e}")
        _log_write(_NO_CHECKPOINT_USER_MSG)
        if _log_file:
            _log_file.close()
        try:
            _mark_run_terminal("paused", error=_NO_CHECKPOINT_USER_MSG)
        except Exception:
            pass
        # Emit a normal stage-progress message rather than 'error' so
        # the Rust shell doesn't surface a red banner. The UI reads
        # run.json.status separately for the paused/failed distinction.
        print(json.dumps({
            "stage": "paused",
            "message": _NO_CHECKPOINT_USER_MSG,
        }), flush=True)
        sys.exit(0)
    except Exception as e:
        _log_write(f"FATAL: {e}")
        if _log_file:
            import traceback
            _log_file.write(traceback.format_exc())
            _log_file.close()
        try:
            _mark_run_terminal("failed", error=str(e))
        except Exception:
            pass
        print(json.dumps({"stage": "error", "message": str(e)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
