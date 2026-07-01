"""Recompute `progress.FALLBACK_SECONDS_PER_CALL` from the current
historical pool of `llm-calls.jsonl` files.

Walks `~/.basevault/logs-dev/**/llm-calls.jsonl` (default — the root
inherited from `progress._logs_root()` when `BASEVAULT_AGENT` is unset,
where dev / sweep / smoke runs land). Pairs begin/end events per
call_id, drops failures + in-flight pairs, and prints the per-stage
aggregate median (across all models) of `duration_ms / 1000`. The
aggregate is the right baseline for FALLBACK because fallback only
fires when there's no historical data for a (stage, model) — i.e.
a freshly-introduced model variant. The cross-model median is what
"a typical call for this stage" looks like across every model we've
seen so far.

Usage (from engine/):

    python -m scripts.refresh_fallback_constants

Prints a recommended dict to paste into progress.py. Round to whole
seconds — sub-second precision is noise; this is cold-start, not a
live coefficient.
"""
from __future__ import annotations

from collections import defaultdict


from engine.progress import (  # noqa: E402
    FALLBACK_SECONDS_PER_CALL,
    PIPELINE_STAGES,
    load_historical_durations,
)


def main() -> None:
    hist = load_historical_durations()
    by_stage: dict[str, list[float]] = defaultdict(list)
    for (stage, _model), samples in hist.items():
        for dur, _tokens in samples:
            by_stage[stage].append(dur)

    print(f"Pool: {len(hist)} (stage, model) keys, "
          f"{sum(len(v) for v in by_stage.values())} successful calls")
    print()
    print(f"{'stage':<18} {'n':>5} {'median':>9} {'prior':>9} {'delta':>9} "
          f"{'recommended':>13}")
    print("-" * 68)
    for stage in PIPELINE_STAGES:
        durs = sorted(by_stage.get(stage, []))
        n = len(durs)
        if n == 0:
            continue
        med = durs[n // 2]
        prior = FALLBACK_SECONDS_PER_CALL.get(stage, 30.0)
        rec = round(med)
        delta = med - prior
        flag = "  ← drift" if abs(delta) >= 5 else ""
        print(f"{stage:<18} {n:>5} {med:>8.1f}s {prior:>8.1f}s "
              f"{delta:>+8.1f}s {rec:>12.0f}s{flag}")
    print()
    print("Paste into FALLBACK_SECONDS_PER_CALL after rounding "
          "and updating the snapshot date / pool size in the docstring.")


if __name__ == "__main__":
    main()
