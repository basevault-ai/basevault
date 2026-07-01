"""
Time-based progress estimation for the pipeline runner.

The runner emits a JSON line after every LLM call so the Tauri shell
(App.jsx) can render a progress bar. Pre-PR, those lines were per
stage:

    {"stage": "entities", "completed": 67}
    {"stage": "estimate_updated", "total": 68}

A 67/68 (97%) reading while four stages were still pending was
misleading — users assumed the whole pipeline was almost done.

This module replaces that with two changes:

  - Cumulative pipeline-total LLM-call count (Part 1). The denominator
    becomes the sum across entities + patterns + insights + actions +
    extract + metadata, not the active stage's calls alone. As later
    stages refine their own estimates from inputs they only learn
    mid-run, the cumulative total is re-emitted via the existing
    `estimate_updated` event.

  - Time-based ETA grounded in historical (stage, model) durations
    from `llm-calls.jsonl` across past runs (Part 2). The bar moves by
    elapsed wall-clock vs. estimated wall-clock, NOT by call counts —
    a 50/100 call count can mean 60% of pipeline time done OR 5%
    depending on which stages are slow. Per-call durations spread 5×
    across stages on the same model.

Coefficients are MEDIANS (not means). One 200s outlier on a
30s-typical call would otherwise drag the projection. p10/p90 are
captured by the validator to characterize the historical
distribution but are NOT used in the live estimator — only the
median, clamped per call to a sanity band.

Bar ratchet: position is monotonic non-decreasing. If the ETA grows
mid-stage (a slow call refines the median upward), the bar pauses
rather than snapping backward. The user reads "stuck" as "still
working" — preferable to "going backward".
"""
from __future__ import annotations

import json
import os
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Stages that issue LLM calls. Keep in sync with runner.py's `_stage`
# assignments. Order is the wall-clock order the runner traverses on a
# fresh run; resumed runs may skip earlier entries.
PIPELINE_STAGES: tuple[str, ...] = (
    "vision",
    "extract",
    "entities",
    "entities_dedupe",
    "patterns",
    "insights",
    "actions",
    "embeddings",
)

# Sanity bands for live per-call duration. A single call's contribution
# to the running median is clamped to [floor × hist_median, ceil ×
# hist_median]. Loose enough to allow real variation, tight enough that
# one stuck attestation handshake (15min) doesn't push the ETA into
# absurd territory.
PER_CALL_FLOOR = 0.3
PER_CALL_CEIL = 3.0

# How many past runs (newest by mtime) to consult when building the
# historical-median table. Older runs are stale (a model swap changes
# per-call duration). Tunable; small to bound the cold-path cost of
# scanning all .jsonl files.
HIST_WINDOW_RUNS = 30

# Minimum live-sample count before we trust the live median over the
# historical median. Below this we blend the two.
LIVE_TRUST_MIN_SAMPLES = 3

# Hard fallback per-call duration when neither historical nor live data
# exists. First-run-from-fresh-install case, OR a freshly-introduced
# (stage, model) pair with no history yet. Once historical data exists
# for the (stage, model), fallback doesn't fire — so this only matters
# at cold start.
#
# How these are computed:
#   1. Run `python -m scripts.refresh_fallback_constants` from
#      engine/. The script walks `~/.basevault/logs/**/llm-calls.jsonl`,
#      pairs begin/end events per call_id, drops failures + in-flight
#      pairs, and computes the per-stage AGGREGATE median (across all
#      models) of `duration_ms / 1000`.
#   2. Update the dict below to match. Round to whole seconds — sub-
#      second precision is noise here; this is a cold-start fallback,
#      not a live coefficient.
#
# Snapshot 2026-04-30 (1169-sample pool of tinfoil/gpt-oss-120b runs +
# legacy cross-model history before the production gut to TEE-only):
#                          aggregate_median   prior_const   delta
#   metadata                12.9s              13.0s        ~0
#   extract                 67.8s              68.0s        ~0
#   entities                21.7s              22.0s        ~0
#   entities_dedupe         32.2s              65.0s       -32.8s  ← was way off
#   patterns                21.2s              24.0s        -2.8s
#   insights                19.8s              23.0s        -3.2s
#   actions                 25.1s              33.0s        -7.9s
#
# entities_dedupe shifted dramatically because the mix of (model, dedupe)
# pairs in the pool changed as gemma4-31b and gpt-oss became the dominant
# routings.
FALLBACK_SECONDS_PER_CALL: dict[str, float] = {
    "vision":           15.0,
    "extract":          68.0,
    "entities":         22.0,
    "entities_dedupe":  32.0,
    "patterns":         21.0,
    "insights":         20.0,
    "actions":          25.0,
    # Embeddings is the one stage where sub-second per-call latency is
    # the norm — Tinfoil-nomic round-trips at ~130-450 ms over the
    # enclave-pinned transport, an order of magnitude faster than the
    # chat stages. Whole-second rounding (the rule documented above for
    # chat stages) would inflate the cold-start floor 3-4× and distort
    # the bar's projected ETA noticeably at higher fact counts. The
    # coefficient sits at the median of the observed range; the
    # tracker's live-sample blend takes over from the per-run
    # calibration embedding call onwards.
    "embeddings":        0.3,
}

# Max worker concurrency per stage. The runner fans out into a
# threadpool for stages that batch by parent / topic / entity-group,
# so wall-clock for `remaining_calls × per_call_seconds` overshoots
# real time by ~stage_parallelism. Without this divider the ETA at
# extract's entry overshoots: 54 calls × 70s/call = 63min, but with
# 16 concurrent workers the actual wall-clock is closer to ~12-15min.
#
# Values mirror llm.max_workers (16 for cloud) for stages that fan
# out, and 1 for stages that issue a single call. LOCAL mode forces
# 1 across the board via PARALLELISM_LOCAL_OVERRIDE — stage parallelism
# can never exceed the LLM threadpool's max_workers.
PARALLELISM_PER_STAGE: dict[str, int] = {
    "vision":           16,
    "extract":          16,
    "entities":         16,
    "entities_dedupe":   1,
    "patterns":         16,
    "insights":          1,
    "actions":           1,
    # Stub stage fires its dummy calls sequentially. PR 3.3 introduces
    # the real per-chunk fan-out alongside the chunking + vector store
    # work; until then there's no batching surface to parallelize.
    "embeddings":        1,
}
PARALLELISM_LOCAL_OVERRIDE = 1

# Batch tail factor: when N calls run concurrently in one batch, the
# wall-clock is the SLOWEST of the N (max-of-N), not the median.
# Per-call median × n_batches systematically underestimates wall-clock
# at fan-out > 1 because it averages instead of maxing. A flat 2.0×
# multiplier on the per-call median is a pragmatic stand-in for the
# max-of-N ratio observed across our (stage, model) historicals
# (p3ax/8m9n/spu2 patterns runs measured p90/median ≈ 2.0×). Stages
# with parallelism=1 (insights, actions, dedupe) get tail_factor=1
# because there's no batching tail.
BATCH_TAIL_FACTOR = 2.0


# ── Pipeline-scope estimator ────────────────────────────────────────────────
#
# Lives here (not in ingestor) because it's pure workload projection: takes
# a `PipelineStats` dict produced by ingestor's file walkers, returns per-
# stage call counts + cold-cache seconds via the same FALLBACK_SECONDS_PER_CALL
# / PARALLELISM_PER_STAGE / BATCH_TAIL_FACTOR constants above. Was originally
# adjacent to its inputs in ingestor.py; moved here so the "progress / ETA"
# surface is one module.

@dataclass(frozen=True)
class PipelineEstimate:
    """One picture of pipeline scope: per-stage call counts AND
    per-stage cold-cache wall-clock seconds, plus pipeline totals
    and a `percentage(completed_calls)` helper.

    Used at every site that needs to project pipeline scope ahead of
    live data:
      - preflight emit (before `ProgressTracker` exists)
      - ingest registration (registers stages on the tracker)
      - post-ingest re-estimate (refines with parsed-Document stats)

    `seconds_per_stage` uses the SAME formula
    `ProgressTracker._stage_wall_clock_seconds` applies on cold cache
    (no live samples, no historical median):
      (calls / parallelism) × FALLBACK_SECONDS_PER_CALL[stage] × tail

    Once the tracker initializes and stages start producing live
    samples, the tracker's blended ETA takes over via `_emit()`. This
    object is the floor — the picture before any live data lands.
    """
    calls_per_stage: dict
    seconds_per_stage: dict
    total_calls: int
    total_seconds: float

def _default_n_topics() -> int:
    """Live fallback for `n_topics` when callers don't pass one. Reads
    `len(content_extractor._DEFAULT_TOPICS)` so the pre-ingest bar's
    patterns denominator tracks the actual taxonomy size (currently
    13) instead of a stale magic number. Lazy-imported because
    `content_extractor` imports `ingestor.Document` and `ingestor`
    re-exports this from progress in some paths — top-level
    progress→content_extractor at module load would risk loops as
    the import graph grows."""
    from engine.content_extractor import _DEFAULT_TOPICS
    return len(_DEFAULT_TOPICS)


def estimate_pipeline(
    stats: dict,
    is_local: bool = False,
    historical_durations: dict | None = None,
    stage_model_map: dict[str, str] | None = None,
) -> PipelineEstimate:
    """SINGLE pipeline estimator. The ONLY place that decides
    per-stage est_calls AND the cold-cache per-stage ETA.
    Pre-ingest, ingest registration, and post-ingest all call this
    with a `PipelineStats` dict from one of `ingestor`'s two stats
    producers (`pipeline_stats_from_paths`, `pipeline_stats_from_docs`).

    ── Design choice: ACCURATE > STABLE ────────────────────────────
    The progress bar's denominator REFINES at phase boundaries:
      - preflight: stats from file stat() walk (cheap, no LLM
        imports). text_bytes is approximated (10kB per binary file,
        real for text), n_splits is None.
      - ingest registration: same stats dict registered with the
        ProgressTracker.
      - post-ingest: stats from parsed Documents + splitter output.
        n_splits is now the real chunk count → drives extract
        est_calls directly. text_bytes is real Document.content
        sizes. chunk_cap is the real `chunk_cap_for_stage` value.
      - mid-run halve cascades: extract est_calls bumps via
        `_bump_stage_est` for each fan-out (one parent → two
        children).
    The trade-off: bar denominator MOVES during the run as estimates
    refine. Bar adjusts ~3-5 times. Accepted for accuracy. The
    alternative (register once, never refine) would be smoother but
    persistently inaccurate.

    Call-count heuristic:
      - vision: n_images
      - extract: n_splits if known (post-ingest), else
        max(n_files, ceil(tokens / chunk_cap)) (pre-ingest, floored
        at n_files so small inputs read sensibly regardless of
        splitter packing).
      - entities: ceil(sqrt(n_facts / FACTS_PER_GROUP)), floored at 2.
      - entities_dedupe / insights / actions: 1 each.
      - patterns: n_topics.
      - embeddings: 1. The stage issues many small batched wire
        calls (ceil(records / batch_size) — ~7 for a ~200-record
        run), but each is sub-second work, not comparable to an
        extract / pattern / insight call (tens of seconds). Counting
        the real batch fan-out in the bar's denominator balloons it:
        the bar gets held back by phantom embedding units through
        the long stages, then lurches when the ~2s burst lands
        (issue #581). For PROGRESS purposes the whole stage counts
        as ONE collective unit — same weight as the other terminal
        single-call stages (insights / actions). The REAL call count
        is restored on completion: `mark_stage_finished` snaps
        embeddings' est_calls down to its actual `completed_calls`
        at end-of-run, and the done-state chip reads the real leaf
        count. Display-only; embeddings throughput is untouched.

    Per-stage seconds: cold-cache wall-clock formula. `is_local`
    picks the LOCAL-mode parallelism override (1 across the board);
    cloud modes use `PARALLELISM_PER_STAGE`.
    """
    from math import ceil
    n_images = stats.get("n_images", 0)
    n_text_files = stats.get("n_text_files", 0)
    n_input_files = n_images + n_text_files
    text_bytes = stats.get("text_bytes", 0)
    input_tokens = text_bytes // 3
    n_splits = stats.get("n_splits")
    if n_splits is not None and n_splits > 0:
        n_extract = n_splits
    else:
        chunk_cap = max(1, stats.get("chunk_cap", 30_000))
        n_extract = max(
            1,
            n_input_files,
            ceil(input_tokens / chunk_cap),
        )
    facts_size = int(input_tokens * 0.5)
    AVG_TOK_PER_FACT = 80
    FACTS_PER_GROUP = 5
    n_facts = max(1, facts_size // AVG_TOK_PER_FACT)
    n_groups = max(1, n_facts // FACTS_PER_GROUP)
    n_entities = max(2, ceil(n_groups ** 0.5))
    calls = {
        "vision": n_images,
        "extract": n_extract,
        "entities": n_entities,
        "entities_dedupe": 1,
        "patterns": max(
            1,
            stats["n_topics"] if stats.get("n_topics") is not None
            else _default_n_topics(),
        ),
        "insights": 1,
        "actions": 1,
        # One collective unit for the whole stage — see the
        # call-count heuristic in the docstring (issue #581). The
        # real batch fan-out is restored at end-of-run via
        # `mark_stage_finished`.
        "embeddings": 1,
    }
    parallelism = (
        {s: PARALLELISM_LOCAL_OVERRIDE for s in PARALLELISM_PER_STAGE}
        if is_local else PARALLELISM_PER_STAGE
    )
    seconds = {}
    for stage, n in calls.items():
        if n <= 0:
            seconds[stage] = 0.0
            continue
        # Per-call seconds match the live tracker's preflight-time
        # computation (token-aware historical decomposition with
        # FALLBACK fallback) when historical_durations + stage_model_map
        # are provided. Without them, falls back to FALLBACK constants
        # (token-blind) — same as the legacy behavior.
        if historical_durations is not None:
            model = (stage_model_map or {}).get(stage, "?")
            per_call = per_call_seconds_at_preflight(
                stage, model, historical_durations)
        else:
            per_call = FALLBACK_SECONDS_PER_CALL.get(stage, 30.0)
        p = max(1, parallelism.get(stage, 1))
        tail = BATCH_TAIL_FACTOR if p > 1 else 1.0
        seconds[stage] = (n / p) * per_call * tail
    return PipelineEstimate(
        calls_per_stage=calls,
        seconds_per_stage=seconds,
        total_calls=sum(calls.values()),
        total_seconds=sum(seconds.values()),
    )


def _logs_root() -> Path:
    """Honor BASEVAULT_LOGS_ROOT; otherwise infer from BASEVAULT_AGENT
    (app → ~/.basevault/logs, anything else → ~/.basevault/logs-dev).
    Matches runner.py's path resolution so historical-duration lookups
    pull from the same root the runner writes to."""
    raw = os.environ.get("BASEVAULT_LOGS_ROOT")
    if raw:
        return Path(raw).expanduser()
    agent = os.environ.get("BASEVAULT_AGENT", "").strip()
    leaf = "logs" if agent == "app" else "logs-dev"
    return Path.home() / ".basevault" / leaf


def load_historical_durations(
    logs_root: Path | None = None,
    window_runs: int = HIST_WINDOW_RUNS,
    skip_path: Path | None = None,
) -> dict[tuple[str, str], list[tuple[float, int]]]:
    """Scan past `llm-calls.jsonl` files and return per-call
    `(duration_seconds, completion_tokens)` tuples keyed by
    (stage, model).

    `window_runs` is a soft cap — newest .jsonl files first by mtime.
    `skip_path` is the current run's own .jsonl, excluded so live
    progress doesn't echo through historical lookups during the same
    run.

    `completion_tokens` is taken from the `end` event's
    `completion_tokens` field (PR #36 wired it into the streaming
    log). Calls where the field is missing or zero get `0` — the
    sec/token coefficient skips zero-token calls when computing
    the median.

    Failed calls and `begin`-without-`end` events are skipped — they
    distort the median by either being a fast 5xx (10ms auth bail) or
    unbounded (still in flight).
    """
    root = logs_root or _logs_root()
    if not root.exists():
        return {}
    files = list(root.rglob("llm-calls.jsonl"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if skip_path is not None:
        try:
            skip_resolved = skip_path.resolve()
        except OSError:
            skip_resolved = None
        if skip_resolved is not None:
            files = [f for f in files if f.resolve() != skip_resolved]
    files = files[:window_runs]

    by_key: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for jp in files:
        try:
            text = jp.read_text(encoding="utf-8")
        except OSError:
            continue
        begins: dict[str, dict] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
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
                if b is None:
                    continue
                if not ev.get("success"):
                    continue
                stage = b.get("stage")
                model = b.get("model")
                dur_ms = ev.get("duration_ms")
                if not stage or not model:
                    continue
                if not isinstance(dur_ms, (int, float)) or dur_ms <= 0:
                    continue
                ct = ev.get("completion_tokens") or 0
                if not isinstance(ct, (int, float)) or ct < 0:
                    ct = 0
                by_key[(stage, model)].append((dur_ms / 1000.0, int(ct)))
    return dict(by_key)


def _median_or_none(xs: Iterable[float]) -> float | None:
    arr = [x for x in xs if x is not None]
    if not arr:
        return None
    return statistics.median(arr)


def per_call_seconds_at_preflight(
    stage: str,
    model: str,
    historical_durations: dict[tuple[str, str], list[tuple[float, int]]] | None,
) -> float:
    """Per-call wall-clock seconds for `(stage, model)` at preflight time,
    using the same `hist_call_fixed + hist_per_token × hist_tokens`
    decomposition the live tracker applies in `_per_call_seconds_locked`
    when no live samples exist for the stage.

    Pre-PR, preflight read `FALLBACK_SECONDS_PER_CALL[stage]` directly —
    a token-blind constant. For models whose historical completion-token
    distribution sits far from the FALLBACK's implicit assumption (e.g.
    kimi-k2-6 emitting ~10-20k tokens per dense call while FALLBACK
    encodes a generic average), the two computations diverged by 5-10×.
    A TEE-mix run on kimi-k2-6 produced a 7× preflight→ingest ETA jump
    (173s → 1225s, same 28-call denominator) — issue #394.

    Returns the FALLBACK constant when `historical_durations` is None or
    when the (stage, model) lookup yields no samples, matching the
    legacy preflight behavior for fresh installs.
    """
    fallback = FALLBACK_SECONDS_PER_CALL.get(stage, 30.0)
    if not historical_durations:
        return fallback
    hist = _hist_lookup(historical_durations, stage, model)
    if not hist:
        return fallback
    sec_call_fixed, sec_per_token, hist_tokens = _decompose_coefficients(
        hist, fallback)
    return sec_call_fixed + sec_per_token * max(0, hist_tokens)


def _hist_lookup(
    hist: dict[tuple[str, str], list[tuple[float, int]]],
    stage: str,
    model: str,
) -> list[tuple[float, int]]:
    """Exact (stage, model) match, with a fallback to all models for
    the stage when the exact key is empty (e.g. a new model variant).
    Returns a list of (duration_seconds, completion_tokens) tuples —
    never None.
    """
    direct = hist.get((stage, model), [])
    if direct:
        return direct
    aggregated: list[tuple[float, int]] = []
    for (s, _m), entries in hist.items():
        if s == stage:
            aggregated.extend(entries)
    return aggregated


def _decompose_coefficients(
    samples: list[tuple[float, int]],
    fallback_seconds: float,
) -> tuple[float, float, int]:
    """Return `(sec_per_call_fixed, sec_per_token_median, median_tokens)`
    from a list of (duration_s, completion_tokens) samples.

    Decomposition model: each call's wall-clock ≈ fixed_overhead +
    per_token × completion_tokens. The brief frames sec/call as the
    fixed overhead (network, attestation, model warm-up) and sec/token
    as the variable generation cost. Computing both as raw medians of
    different ratios over the same population would double-count
    (medianTime + medianRate × tokens ≈ 2× medianTime for a typical
    call). Instead:

      sec_per_token_median = median(duration / tokens) over calls
                              with tokens > 0
      sec_per_call_fixed   = median(max(0, duration - sec_per_token *
                                         tokens)) — i.e. the median
                              residual after subtracting the variable
                              component. Clamped non-negative; ~0 in
                              practice for stages where token rate
                              dominates.

    For the typical call (median tokens), the sum
    sec_per_call_fixed + sec_per_token × median_tokens ≈ median(duration),
    so the new estimator agrees with the old one at the median while
    scaling correctly with token count for variable-payload stages
    like extract.

    Falls back to (fallback_seconds, 0, 0) when no token-bearing
    samples exist — preserves the old sec/call-only behavior.
    """
    with_tokens = [(d, t) for d, t in samples if t > 0]
    if not with_tokens:
        # No token data — old sec/call-only path. Fixed overhead =
        # median(duration), token rate = 0.
        durations = [d for d, _ in samples]
        if not durations:
            return (fallback_seconds, 0.0, 0)
        return (statistics.median(durations), 0.0, 0)
    rates = [d / t for d, t in with_tokens]
    sec_per_token = statistics.median(rates)
    residuals = [max(0.0, d - sec_per_token * t) for d, t in with_tokens]
    sec_per_call_fixed = statistics.median(residuals)
    median_tokens = int(statistics.median([t for _, t in with_tokens]))
    return (sec_per_call_fixed, sec_per_token, median_tokens)


class StageTimings:
    """Live state for one stage in one pipeline run."""

    __slots__ = (
        "stage", "model", "est_calls", "est_tokens_per_call",
        "completed_calls",
        "live_samples",  # list of (duration_s, completion_tokens) tuples
        "started_at", "finished_at",
    )

    def __init__(
        self,
        stage: str,
        model: str,
        est_calls: int,
        est_tokens_per_call: int | None = None,
    ) -> None:
        self.stage = stage
        self.model = model
        self.est_calls = max(1, int(est_calls))
        # None = use historical median tokens for this (stage, model).
        # Pass an int when the caller has a tighter per-call estimate
        # (e.g. extract chunks, where chunk size is known at split time).
        self.est_tokens_per_call: int | None = est_tokens_per_call
        self.completed_calls = 0
        self.live_samples: list[tuple[float, int]] = []
        self.started_at: float | None = None
        self.finished_at: float | None = None

    def mark_started(self, now: float | None = None) -> None:
        if self.started_at is None:
            self.started_at = now if now is not None else time.monotonic()

    def mark_finished(self, now: float | None = None) -> None:
        if self.finished_at is None:
            self.finished_at = now if now is not None else time.monotonic()

    def record_call(
        self,
        duration_seconds: float,
        completion_tokens: int = 0,
    ) -> None:
        if duration_seconds > 0:
            self.live_samples.append(
                (duration_seconds, max(0, int(completion_tokens))))
        self.completed_calls += 1

    def elapsed(self, now: float | None = None) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else (
            now if now is not None else time.monotonic()
        )
        return max(0.0, end - self.started_at)


class ProgressTracker:
    """Pipeline-wide progress estimator. One instance per run.

    Thread-safe: every public method holds `self._lock` for any read
    or write of stage state. The runner pokes at it from the
    threadpool worker that just finished an LLM call (`record_call`)
    and from the main thread at stage boundaries
    (`register_stage` / `mark_stage_started` / `mark_stage_finished`).
    """

    def __init__(
        self,
        historical_durations:
        dict[tuple[str, str], list[tuple[float, int]]] | None = None,
        parallelism_per_stage: dict[str, int] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        # Historical samples list-per-(stage, model): [(dur_s, tokens), ...].
        self._hist = historical_durations or {}
        self._stages: dict[str, StageTimings] = {}
        # Insertion order = stage execution order.
        self._order: list[str] = []
        # Per-stage worker fan-out. Stages with fan-out > 1 see their
        # remaining-seconds estimate divided by min(remaining_calls,
        # parallelism). Defaults to PARALLELISM_PER_STAGE (cloud mode);
        # callers (e.g. LOCAL mode) override.
        self._parallelism = (
            dict(parallelism_per_stage)
            if parallelism_per_stage is not None
            else dict(PARALLELISM_PER_STAGE)
        )
        # Per-stage in-flight call tracking. Keyed by stage name; each
        # entry is a list of (started_at_monotonic, expected_duration_s)
        # tuples. Used by the ETA estimator to deduct elapsed-in-call
        # wall-clock from each currently-running call:
        #   ETA contribution: max remaining = max(0, expected - elapsed) per call
        # Without it, a parallel batch of 64 calls in flight would freeze
        # the ETA at one batch's expected duration until the slowest
        # call lands. The BAR does NOT read this set — bar is strict
        # completed/total per issue #68.
        self._in_flight_per_stage: dict[str, list[tuple[float, float]]] = {}

    # ── Stage lifecycle ───────────────────────────────────────────────

    def register_stage(
        self,
        stage: str,
        model: str,
        est_calls: int,
        est_tokens_per_call: int | None = None,
    ) -> None:
        """Add or update a stage's call estimate. Called at run start
        with rough estimates for every stage, then re-called at each
        stage's entry point with a refined estimate.

        `est_tokens_per_call` is optional — when provided, the
        estimator's variable component uses this number directly.
        When None, the historical median completion-token count for
        (stage, model) is used. The runner can pass a tighter number
        for stages where it knows the per-call payload size (e.g.
        extract chunks).
        """
        with self._lock:
            existing = self._stages.get(stage)
            if existing is None:
                self._stages[stage] = StageTimings(
                    stage, model, est_calls, est_tokens_per_call)
                self._order.append(stage)
            else:
                # Don't shrink est below already-completed.
                existing.est_calls = max(int(est_calls), existing.completed_calls)
                # Model can change on resume / TEE-mix routing
                # finalization. Trust the latest caller.
                existing.model = model
                if est_tokens_per_call is not None:
                    existing.est_tokens_per_call = est_tokens_per_call

    def bump_est_calls(self, stage: str, delta: int) -> None:
        """Add `delta` to `stage`'s est_calls. Use when a fan-out
        retry strategy increases the call count beyond the original
        estimate — e.g. extract halving spawns 2 sub-calls in place
        of 1 parent, so each halving event needs to bump the
        denominator by +1 (issue #105 v3 follow-up: pre-fix the
        progress bar overshot 100% on halve-heavy runs because the
        denominator stayed at the pre-halving estimate).

        No-op for unregistered stages."""
        if delta == 0:
            return
        with self._lock:
            s = self._stages.get(stage)
            if s is None:
                return
            new = max(s.completed_calls, s.est_calls + delta)
            s.est_calls = new

    def mark_stage_started(self, stage: str) -> None:
        with self._lock:
            s = self._stages.get(stage)
            if s is not None:
                s.mark_started()

    def mark_stage_finished(self, stage: str) -> None:
        """Mark `stage` finished and tighten est_calls down to the
        actual completed count.

        Without the tightening, an over-estimated stage (e.g. entities'
        `sqrt(facts/5)` heuristic predicts 4 batches but actuals only
        run 2) keeps inflating the cumulative denominator forever.
        User instruction: "always compute remaining calls based on
        current and future stages, old ones should no longer be
        relevant" — past stages must contribute 0 to remaining. After
        this snap, `s.est_calls == s.completed_calls`, so
        `s.est_calls - s.completed_calls = 0`.
        """
        with self._lock:
            s = self._stages.get(stage)
            if s is None:
                return
            s.mark_finished()
            s.est_calls = s.completed_calls

    def record_call(
        self,
        stage: str,
        duration_seconds: float,
        completion_tokens: int = 0,
    ) -> None:
        """Record a successful call. Auto-marks the stage started if
        it wasn't already — handles metadata (fires inside the extract
        threadpool) and entities_dedupe (fires inside the entities
        stage code), neither of which gets its own _log_stage call in
        the runner.

        This is the legacy single-call API kept for tests. The runner
        uses `record_call_begin` / `record_call_end` to feed the
        in-flight set that powers the time-deducting ETA.
        """
        with self._lock:
            s = self._stages.get(stage)
            if s is None:
                return
            if s.started_at is None:
                s.mark_started()
            s.record_call(duration_seconds, completion_tokens)

    def record_call_begin(self, stage: str) -> tuple[float, float] | None:
        """Mark an LLM call as started for `stage`. Returns a token
        the caller passes back to `record_call_end` to remove this
        call from the in-flight set on completion.

        Token shape: `(started_at_monotonic, expected_duration_s)` —
        opaque to callers but used by the ETA estimator to deduct
        elapsed wall-clock from each in-flight call within a
        parallel batch.

        Returns None for unregistered stages — callers should treat
        None as a no-op and not pass it to `record_call_end`.
        """
        with self._lock:
            s = self._stages.get(stage)
            if s is None:
                return None
            now = time.monotonic()
            # Expected duration uses CURRENT coefficient blend
            # (historical ± live). Captured at begin-time so the
            # ETA's per-call deduction denominator stays stable for
            # THIS call even if later calls refine the median.
            per_call = self._per_call_seconds_locked(
                stage, s.model, list(s.live_samples), s.est_tokens_per_call)
            self._in_flight_per_stage.setdefault(stage, []).append(
                (now, per_call))
            if s.started_at is None:
                s.mark_started(now)
            return (now, per_call)

    def record_call_end(
        self,
        stage: str,
        token: tuple[float, float] | None,
        duration_seconds: float,
        completion_tokens: int = 0,
        success: bool = True,
    ) -> None:
        """Pair `record_call_begin`'s token with the call's outcome.
        Removes the in-flight entry; on success also appends the live
        sample and increments completed_calls.

        `token` may be None (call wasn't tracked at begin) — caller
        gets a no-op for the in-flight removal but the success-path
        bookkeeping still applies.

        On `success=False` the call is removed from the in-flight set
        but completed_calls is NOT bumped (the runner's `_completed`
        global similarly only increments on success). Without this
        gating, a final-failure would credit a phantom completion.
        """
        with self._lock:
            s = self._stages.get(stage)
            if s is None:
                return
            lst = self._in_flight_per_stage.get(stage)
            if lst is not None and token is not None:
                try:
                    lst.remove(token)
                except ValueError:
                    pass
            if not success:
                return
            if s.started_at is None:
                s.mark_started()
            s.record_call(duration_seconds, completion_tokens)

    # ── Pure estimators ───────────────────────────────────────────────

    def _coefficients_locked(
        self,
        stage: str,
        model: str,
        live: list[tuple[float, int]],
    ) -> tuple[float, float, int]:
        """Return `(sec_per_call_fixed, sec_per_token, default_tokens_per_call)`
        for `(stage, model)`.

        Coefficient blend = completion-ratio-weighted average of
        historical and clamped-live medians. As the stage progresses
        through its est_calls, weight on live grows linearly:
            weight_live = completed_calls / est_calls    (clamped [0, 1])
            sec_per_call = (1 - weight) × hist_med + weight × live_med

        Same for sec_per_token. At 0% complete the blend is 100%
        historical; at 100% complete it's 100% live (but the stage's
        remaining=0 by then so the per-call value is moot for that
        stage's ETA).

        Round-4 follow-up PIVOT — replaced the prior
        `LIVE_TRUST_MIN_SAMPLES=3` step function. The hard switch
        caused ETA to JUMP at sample #3 when the first live samples
        happened to be slower than historical (observed live in a
        real run as bar 5%→20% + ETA 5m10s→7m simultaneously).
        Completion-ratio weighting is smooth and principled — no
        magic sample-count target.

        Independent sanity bands per coefficient: each per-call live
        sample is clamped to [PER_CALL_FLOOR, PER_CALL_CEIL] × the
        corresponding historical median before going into the live
        median. Bounds outliers without forcing live to ignore real
        drift.

        `default_tokens_per_call` is the historical median of
        completion_tokens — used when the caller doesn't supply an
        explicit `est_tokens_per_call` for the stage.

        Caller holds self._lock.
        """
        hist = _hist_lookup(self._hist, stage, model)
        fallback_total = FALLBACK_SECONDS_PER_CALL.get(stage, 30.0)
        hist_call_fixed, hist_per_token, hist_tokens = _decompose_coefficients(
            hist, fallback_total)

        # Per-coefficient sanity bands — each live sample is clamped
        # against the corresponding historical median.
        call_floor = PER_CALL_FLOOR * max(hist_call_fixed, 1e-3)
        call_ceil = PER_CALL_CEIL * max(hist_call_fixed, 1e-3)
        token_floor = PER_CALL_FLOOR * hist_per_token
        token_ceil = PER_CALL_CEIL * hist_per_token

        # Live decomposition — same residual approach as
        # _decompose_coefficients but on the live sample list with
        # per-coefficient clamping.
        live_with_tokens = [(d, t) for d, t in live if t > 0]
        live_token_rates = [d / t for d, t in live_with_tokens]
        clamped_live_token_rates = [
            max(token_floor, min(token_ceil, r))
            for r in live_token_rates
        ] if hist_per_token > 0 else []
        live_token_med = (
            float(statistics.median(clamped_live_token_rates))
            if clamped_live_token_rates else hist_per_token
        )
        # Use whichever sec_per_token estimate is freshest for the
        # residual computation. Doesn't matter much; live_token_med
        # falls back to hist_per_token when no live samples exist.
        live_fixed_residuals = [
            max(0.0, d - live_token_med * t) for d, t in live
        ]
        clamped_live_fixed = [
            max(call_floor, min(call_ceil, x)) if x > 0 else 0.0
            for x in live_fixed_residuals
        ]
        live_call_med = (
            float(statistics.median(clamped_live_fixed))
            if clamped_live_fixed else hist_call_fixed
        )

        # Completion-ratio blend weight.
        s = self._stages.get(stage)
        if s is None or s.est_calls <= 0:
            weight_live = 0.0
        else:
            weight_live = max(0.0, min(1.0,
                s.completed_calls / s.est_calls))
        # No live samples = stay on historical regardless of weight
        # (the stage hasn't actually produced any data yet).
        if not live:
            weight_live = 0.0

        sec_per_call_est = (
            (1.0 - weight_live) * hist_call_fixed
            + weight_live * live_call_med
        )
        sec_per_token_est = (
            (1.0 - weight_live) * hist_per_token
            + weight_live * live_token_med
        )
        return (sec_per_call_est, sec_per_token_est, hist_tokens)

    def _per_call_seconds_locked(
        self,
        stage: str,
        model: str,
        live: list[tuple[float, int]],
        est_tokens_per_call: int | None = None,
    ) -> float:
        """Wall-clock seconds for one call of `(stage, model)`.

        eta_per_call = sec_per_call_fixed + sec_per_token × tokens

        `tokens` is `est_tokens_per_call` when provided, else the
        historical median completion-token count.

        Caller holds self._lock.
        """
        sec_per_call, sec_per_token, default_tokens = self._coefficients_locked(
            stage, model, live)
        tokens = (
            int(est_tokens_per_call)
            if est_tokens_per_call is not None
            else default_tokens
        )
        return sec_per_call + sec_per_token * max(0, tokens)

    def _stage_wall_clock_seconds(
        self, stage: str, remaining_calls: int, per_call: float,
    ) -> float:
        """Wall-clock seconds for `stage`'s remaining work.

        Formula: `(remaining_calls / parallelism) × per_call × tail`.

        Real division — NOT `ceil`. Round 4 dropped the previous
        `ceil(remaining/parallelism) × per_call × tail` because it
        froze the ETA inside a single batch: with 10 calls remaining
        on a parallelism-16 stage, `n_batches = ceil(10/16) = 1` for
        ALL of the first 10 completions, so the ETA was stuck at
        `1 × per_call × tail` until the LAST call landed. Real
        division gives smooth proportional countdown:
        10 calls → ~0.625 batches × per_call × tail
         5 calls → ~0.313 batches × per_call × tail
         1 call  → ~0.063 batches × per_call × tail

        The `tail = 2.0` factor for parallel stages still approximates
        max-of-N batch wall-clock vs median per-call.

        Caller holds self._lock.
        """
        if remaining_calls <= 0:
            return 0.0
        parallelism = max(1, self._parallelism.get(stage, 1))
        tail = BATCH_TAIL_FACTOR if parallelism > 1 else 1.0
        return (remaining_calls / parallelism) * per_call * tail

    def estimate_stage_seconds(
        self,
        stage: str,
        model: str | None = None,
        est_calls: int | None = None,
        completed_calls: int | None = None,
        live_samples: list[tuple[float, int]] | None = None,
        est_tokens_per_call: int | None = None,
    ) -> float:
        """Remaining wall-clock seconds for `stage`. With all overrides
        None, reads the registered StageTimings; with overrides,
        computes ad-hoc (used by the validator + tests).

        Wall-clock — not serial-sum. Stages with fan-out (extract,
        entities, patterns) divide the per-call total by parallelism.

        The per-call estimate uses both coefficients:
            sec/call_fixed + sec/token × est_tokens_per_call
        See `_per_call_seconds_locked` for details.
        """
        with self._lock:
            s = self._stages.get(stage)
            if s is not None:
                model = model or s.model
                est_calls = est_calls if est_calls is not None else s.est_calls
                completed_calls = (
                    completed_calls if completed_calls is not None
                    else s.completed_calls
                )
                live_samples = (
                    live_samples if live_samples is not None
                    else list(s.live_samples)
                )
                if est_tokens_per_call is None:
                    est_tokens_per_call = s.est_tokens_per_call
            else:
                model = model or "?"
                est_calls = est_calls if est_calls is not None else 1
                completed_calls = completed_calls or 0
                live_samples = live_samples or []
            remaining = max(0, int(est_calls) - int(completed_calls))
            if remaining == 0:
                return 0.0
            per_call = self._per_call_seconds_locked(
                stage, model, live_samples, est_tokens_per_call)
            return self._stage_wall_clock_seconds(stage, remaining, per_call)

    def compute_total_eta_seconds(self) -> float:
        with self._lock:
            keys = list(self._order)
        return sum(self.estimate_stage_seconds(k) for k in keys)

    def compute_pipeline_total_calls(self) -> int:
        with self._lock:
            return sum(s.est_calls for s in self._stages.values())

    def compute_bar_position(self, now: float | None = None) -> float:
        """Tracker's internal completed/total ratio in [0, 0.99].

        Formula: `completed_calls / est_calls` summed across stages,
        capped at 0.99.

        ⚠️  NOT surfaced to the UI (issue #209). The chip's percentage
        is computed in JSX as `displayDone / total` where `displayDone`
        is rust-derived LEAVES (terminal calls in each retry chain) and
        `total` is the tracker's `est_calls` sum. The python tracker's
        `completed_calls` counts SUCCESSFUL WRAPPER ATTEMPTS — not
        leaves — and using it as a UI percentage produced "50%
        completed (22 / 42 calls)" on retry-storm runs because the
        chip and bar pulled from different numerators.

        Kept as an internal helper for tests + tracker introspection;
        do NOT pipe back into the UI without re-deriving from leaves.

        `now` is accepted for API compatibility with `snapshot` but
        is not read.
        """
        with self._lock:
            stages = self._stages.values()
            completed = sum(s.completed_calls for s in stages)
            total = sum(s.est_calls for s in stages)
            if total <= 0:
                return 0.0
            return min(0.99, max(0.0, completed / total))

    def snapshot(self, now: float | None = None) -> dict:
        """One consistent picture of tracker state. `current` is the
        LATEST in-flight stage — e.g. when metadata + extract overlap,
        snapshot reports extract (declared via _log_stage), so the UI
        label tracks the user-visible stage name rather than the
        sub-stage.

        Issue #209: the runner's `_emit()` now uses ONLY a subset of
        these fields for the UI payload (stage, total, eta, stage_eta,
        elapsed_in_stage). `completed_calls_pipeline` and
        `bar_position` are kept here for tests + tracker introspection
        but are NOT surfaced as UI fields — see the docstrings on each
        for why.
        """
        with self._lock:
            stage_keys = list(self._order)
            in_flight = [
                k for k in stage_keys
                if self._stages[k].started_at is not None
                and self._stages[k].finished_at is None
            ]
            # Pick the latest-started in-flight stage so a sub-stage
            # like metadata / entities_dedupe doesn't shadow the
            # extract / entities label the user expects to see.
            current = None
            if in_flight:
                current = max(
                    in_flight,
                    key=lambda k: self._stages[k].started_at or 0.0,
                )
            current_elapsed = 0.0
            current_eta = 0.0
            if current is not None:
                cs = self._stages[current]
                current_elapsed = cs.elapsed(now)
                remaining = max(0, cs.est_calls - cs.completed_calls)
                if remaining > 0:
                    per_call = self._per_call_seconds_locked(
                        current, cs.model, list(cs.live_samples),
                        cs.est_tokens_per_call)
                    current_eta = self._stage_wall_clock_seconds(
                        current, remaining, per_call)
            total_completed = sum(s.completed_calls for s in self._stages.values())
            total_est = sum(s.est_calls for s in self._stages.values())
        # compute_bar_position + compute_total_eta_seconds re-acquire
        # the lock; called outside the with-block above so we don't
        # deadlock.
        return {
            "stage": current,
            "completed_calls_pipeline": total_completed,
            "total_calls_pipeline": total_est,
            "eta_seconds": self.compute_total_eta_seconds(),
            "stage_eta_seconds": current_eta,
            "elapsed_in_stage": current_elapsed,
            "bar_position": self.compute_bar_position(now),
        }
