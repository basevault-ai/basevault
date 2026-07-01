"""Replay a real `llm-calls.jsonl` through the new ProgressTracker
to validate that the ETA reading at every stage transition stays
within ±50% of the actual remaining time.

Usage:
    python -m scripts.replay_progress \
        ~/.basevault/logs-dev/<run-id>/llm-calls.jsonl
        [--target-jsonl <other_run>]
        [--csv <out.csv>]

The replay isolates THIS run from itself when computing the historical
baseline: history is loaded from every other llm-calls.jsonl in the
default logs root (`~/.basevault/logs-dev/` when `BASEVAULT_AGENT` is
unset; overridable via `--logs-root`), and the target jsonl is excluded
so we don't "cheat" by feeding the run's own data into its own coefficient.

Outputs:
  - per-call CSV (call_idx, stage, completed, total, eta_seconds,
    actual_remaining_seconds, bar_position)
  - human-readable summary table per stage with median/p10/p90 of
    relative ETA error AT EVERY STAGE TRANSITION.

The brief required: ETA at every stage transition stays within ±50%
of actual remaining time, NO "97% with 10min left" frames. This script
is the harness that enforces both. It is NOT used at runtime.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


from engine.progress import (
    PIPELINE_STAGES,
    ProgressTracker,
    load_historical_durations,
)


def _read_events(jsonl: Path, first_cycle_only: bool = True) -> list[dict]:
    """Read events from a jsonl. When `first_cycle_only` is True (the
    default), drop events that arrive after the first actions→entities
    transition — multi-cycle resumes (a single jsonl that captured 3
    full pipeline runs from the cancel-restore experiment) confuse the
    replay's per-stage timing because each cycle restarts the
    runner's stage clocks. The replay's purpose is per-stage ETA
    accuracy on a clean linear run; multi-cycle runs need to be
    measured per cycle independently."""
    raw: list[dict] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw.append(json.loads(line))
        except Exception:
            continue
    if not first_cycle_only:
        return raw
    # Find the first actions→entities transition (a resume marker) and
    # drop events at-or-after the entities transition's started_at.
    cutoff_iso: str | None = None
    last_stage: str | None = None
    for ev in raw:
        if ev.get("event") != "begin":
            continue
        stage = ev.get("stage")
        if last_stage == "actions" and stage == "entities":
            cutoff_iso = ev.get("started_at_iso")
            break
        last_stage = stage
    if cutoff_iso is None:
        return raw
    out: list[dict] = []
    for ev in raw:
        ts = ev.get("started_at_iso") or ""
        if ts and ts >= cutoff_iso:
            continue
        out.append(ev)
    return out


def _ts_to_monotonic_offset(events: list[dict]) -> dict[str, float]:
    """Map call_id -> (started_at_offset, end_offset) seconds, computed
    from the first begin event's started_at_iso. Lets the replay use
    monotonic-equivalent timestamps without actually waiting."""
    from datetime import datetime
    iso0 = None
    for ev in events:
        if ev.get("event") == "begin":
            iso0 = ev.get("started_at_iso")
            if iso0:
                break
    if iso0 is None:
        return {}
    t0 = datetime.fromisoformat(iso0.replace("Z", "+00:00")).timestamp()
    starts: dict[str, float] = {}
    durs: dict[str, float] = {}
    for ev in events:
        cid = ev.get("call_id")
        if not cid:
            continue
        if ev.get("event") == "begin":
            iso = ev.get("started_at_iso")
            if iso:
                ts = datetime.fromisoformat(
                    iso.replace("Z", "+00:00")).timestamp()
                starts[cid] = ts - t0
        elif ev.get("event") == "end":
            d = ev.get("duration_ms")
            if isinstance(d, (int, float)):
                durs[cid] = d / 1000.0
    return {"starts": starts, "durations": durs}


def _per_stage_estimates_from_events(
    events: list[dict],
) -> tuple[dict[str, int], dict[str, str]]:
    """Count how many successful calls each stage actually made + the
    model used. The replay treats this as "the perfect estimate".
    A real run only has the per-stage estimate function's prediction;
    this gives us the ground truth so we can ask: how close was the
    ETA at every emit boundary to the actual remaining time?"""
    counts: dict[str, int] = defaultdict(int)
    models: dict[str, str] = {}
    begins_by_id: dict[str, dict] = {}
    for ev in events:
        cid = ev.get("call_id")
        if not cid:
            continue
        if ev.get("event") == "begin":
            begins_by_id[cid] = ev
        elif ev.get("event") == "end" and ev.get("success"):
            b = begins_by_id.get(cid)
            if b is None:
                continue
            stage = b.get("stage")
            if not stage:
                continue
            counts[stage] += 1
            if stage not in models:
                models[stage] = b.get("model") or "?"
    return dict(counts), models


def replay(
    target_jsonl: Path,
    logs_root: Path | None = None,
    csv_out: Path | None = None,
) -> dict:
    """Replay all successful end events through a fresh ProgressTracker.
    Returns a dict of summary statistics."""
    events = _read_events(target_jsonl)
    if not events:
        raise SystemExit(f"no events in {target_jsonl}")

    actual_per_stage, stage_models = _per_stage_estimates_from_events(events)

    # Historical durations: every OTHER llm-calls.jsonl in the root.
    hist = load_historical_durations(
        logs_root=logs_root,
        skip_path=target_jsonl,
    )

    tracker = ProgressTracker(historical_durations=hist)
    # Register at the actual final per-stage count (perfect estimate).
    for stage in PIPELINE_STAGES:
        n = actual_per_stage.get(stage, 0)
        if n > 0:
            tracker.register_stage(
                stage, stage_models.get(stage, "?"), n)

    # Pair begin/end into ordered call records (sorted by end timestamp).
    by_id: dict[str, dict] = {}
    pairs: list[tuple[str, dict, dict]] = []
    for ev in events:
        cid = ev.get("call_id")
        if not cid:
            continue
        if ev.get("event") == "begin":
            by_id[cid] = {"begin": ev}
        elif ev.get("event") == "end":
            slot = by_id.get(cid)
            if slot is None:
                continue
            slot["end"] = ev
            if ev.get("success"):
                pairs.append((cid, slot["begin"], slot["end"]))

    offsets = _ts_to_monotonic_offset(events)
    starts = offsets.get("starts", {})
    # Total wall-clock = max end-time across all calls (last call's
    # start + its duration).
    total_runtime = 0.0
    for cid, b, e in pairs:
        s = starts.get(cid, 0.0)
        total_runtime = max(total_runtime, s + e["duration_ms"] / 1000.0)

    # Drive the tracker by monotonic offsets using an injected `now`
    # via a private setter on each stage. To keep the production code
    # untouched, we use the public mark_started + mark_finished APIs
    # with explicit `now=` and lazily call record_call.
    rows: list[dict] = []
    seen_stages: list[str] = []

    # Order pairs by COMPLETION TIME (start + duration). The runner
    # emits a progress event after each call finalizes, so the replay
    # has to walk calls in completion order, not start order — at
    # high fan-out, two calls that started together can finish 60s
    # apart, and the bar should advance at the second's completion.
    pairs.sort(key=lambda t: starts.get(t[0], 0.0)
               + (t[2].get("duration_ms") or 0) / 1000.0)

    last_now = 0.0
    for cid, b, e in pairs:
        stage = b.get("stage")
        if not stage:
            continue
        s_offset = starts.get(cid, last_now)
        dur_s = e["duration_ms"] / 1000.0
        end_offset = s_offset + dur_s
        # Stage transition: mark prior in-flight stage finished, mark
        # this one started — replay uses absolute offsets as monotonic.
        if stage not in seen_stages:
            for prev in list(seen_stages):
                ps = tracker._stages.get(prev)
                if ps and ps.started_at is not None and ps.finished_at is None:
                    tracker.mark_stage_finished(prev)
                    ps.finished_at = s_offset
            ss = tracker._stages.get(stage)
            if ss is None:
                continue
            ss.mark_started(s_offset)
            seen_stages.append(stage)
        # Record this call AT END-TIME (mirrors the wrapper's order:
        # finalize_stat_record then _record_call_duration). Token count
        # comes from the end event's completion_tokens field.
        tokens = e.get("completion_tokens") or 0
        tracker.record_call(stage, dur_s, completion_tokens=int(tokens))
        # Snapshot the tracker AT END-TIME of this call.
        snap = tracker.snapshot(now=end_offset)
        bar = tracker.compute_bar_position(now=end_offset)
        actual_remaining = max(0.0, total_runtime - end_offset)
        eta = snap["eta_seconds"]
        # Avoid divide-by-zero at the very last call.
        rel_err = (eta - actual_remaining) / max(1.0, actual_remaining)
        # Bar invariant (issue #68): bar = completed_pipeline /
        # total_pipeline, capped at [0, 0.99]. No in-flight term —
        # the bar must equal the call ratio the user reads in the
        # UI. ETA still tracks in-flight calls; bar does not.
        completed_pipeline = sum(
            st.completed_calls for st in tracker._stages.values())
        total_pipeline = sum(
            st.est_calls for st in tracker._stages.values())
        if total_pipeline > 0:
            expected_bar = min(0.99, max(0.0,
                completed_pipeline / total_pipeline))
        else:
            expected_bar = 0.0
        rows.append({
            "call_idx": len(rows) + 1,
            "stage": stage,
            "model": b.get("model"),
            "completed_pipeline": snap["completed_calls_pipeline"],
            "total_pipeline": snap["total_calls_pipeline"],
            "eta_seconds": round(eta, 1),
            "actual_remaining_seconds": round(actual_remaining, 1),
            "rel_err": round(rel_err, 3),
            "bar_position": round(bar, 3),
            "expected_bar": round(expected_bar, 3),
            "bar_eq_expected": abs(bar - expected_bar) < 0.001,
            "elapsed_offset": round(end_offset, 1),
        })
        last_now = end_offset

    if csv_out is not None:
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {csv_out} ({len(rows)} rows)")

    # Summary: per-stage error stats AT END-OF-STAGE (last row whose
    # stage matches before the next stage begins).
    summary: dict[str, dict] = {}
    transitions = []  # rows at stage boundary
    last_stage = None
    for i, r in enumerate(rows):
        if last_stage is not None and r["stage"] != last_stage:
            transitions.append(rows[i - 1])
        last_stage = r["stage"]
    if rows:
        transitions.append(rows[-1])

    rel_errs = [r["rel_err"] for r in rows]
    bar_at_transition = [r["bar_position"] for r in transitions]

    summary["total_calls"] = len(rows)
    summary["total_runtime_seconds"] = round(total_runtime, 1)
    summary["rel_err_p10"] = (
        statistics.quantiles(rel_errs, n=10)[0] if len(rel_errs) >= 10 else min(rel_errs))
    summary["rel_err_median"] = statistics.median(rel_errs)
    summary["rel_err_p90"] = (
        statistics.quantiles(rel_errs, n=10)[8] if len(rel_errs) >= 10 else max(rel_errs))
    summary["rel_err_max"] = max(rel_errs, key=abs)
    summary["bar_at_transition"] = bar_at_transition
    summary["transitions"] = [
        {"stage": t["stage"], "bar": t["bar_position"],
         "eta_s": t["eta_seconds"], "actual_remaining_s": t["actual_remaining_seconds"],
         "rel_err": t["rel_err"]}
        for t in transitions
    ]

    # The brief's pass criterion: every stage transition's ETA must
    # be within ±50% of actual remaining.
    failures = [
        t for t in transitions
        if abs(t["rel_err"]) > 0.5
    ]
    summary["transitions_outside_band"] = len(failures)
    summary["pass"] = len(failures) == 0

    # Round 3 invariant: bar = elapsed / (elapsed + eta) at every call,
    # no ratchet. Should always hold mathematically (it's how the bar
    # is computed). Sanity-check across all rows; any False here is a
    # bug, not a tolerance miss.
    bar_consistency_failures = [
        r for r in rows if not r["bar_eq_expected"]
    ]
    summary["bar_eq_expected_failures"] = len(bar_consistency_failures)
    if bar_consistency_failures:
        summary["pass"] = False
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target_jsonl", type=Path)
    ap.add_argument("--logs-root", type=Path, default=None,
                    help="Override ~/.basevault/logs (test fixtures).")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Per-call replay CSV output path.")
    args = ap.parse_args()
    summary = replay(
        args.target_jsonl,
        logs_root=args.logs_root,
        csv_out=args.csv,
    )
    print(json.dumps(summary, indent=2, default=str))
    sys.exit(0 if summary["pass"] else 2)


if __name__ == "__main__":
    main()
