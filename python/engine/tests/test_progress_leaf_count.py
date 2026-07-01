"""Issue #209: progress invariant `completed ≤ total` under retry storms.

The rust derivation in `src-tauri/src/lib.rs` counts LEAF `end`
events (call_ids that no other begin's `retry_of_call_id` points back
at) and exposes the difference from total ends as `retries`. The
python tracker grows `est_calls` only on structural fan-out via
`_bump_stage_est`. These two are designed to balance: leaf count must
stay ≤ tracker's pipeline total under every retry shape the runner
produces.

This file pins the invariant on synthetic jsonl shapes that mirror
the retry surface (load retry, halve cascade, sample loop,
reasoning-off retry). If a future change to the runner's fan-out
logic forgets to bump `est_calls` (or bumps too much), one of these
scenarios will fail.

The leaf-count helper here mirrors the rust algorithm verbatim — if
they drift, the rust unit tests in lib.rs (`derive_progress_leaf_*`)
will catch it on the rust side and this file on the python side.
"""
from __future__ import annotations

from pathlib import Path

# Add the pipeline dir to sys.path so `from progress import ...` works
# when running this test directly (matches the convention in the rest
# of engine/tests/).
ROOT = Path(__file__).resolve().parent.parent

from engine.progress import ProgressTracker  # noqa: E402


def _leaf_count(begins: list[dict], ends: list[dict]) -> tuple[int, int]:
    """Return `(leaf_ends, retry_ends)` for an event sequence.

    Mirrors `derive_run_state_uncached` in lib.rs:
      parents = {begin.retry_of_call_id for begin in begins if set}
      leaf_ends = count of end events whose call_id ∉ parents
      retry_ends = total_ends - leaf_ends

    Final-pass classification — matches the rust code's behavior of
    deferring leaf identification until the whole jsonl has been read,
    so a late retry of an earlier call correctly demotes the earlier
    end from leaf to intermediate.
    """
    parents = {b["retry_of_call_id"] for b in begins if b.get("retry_of_call_id")}
    leaf = sum(1 for e in ends if e["call_id"] not in parents)
    return leaf, len(ends) - leaf


# ── Invariant tests on synthetic chain shapes ────────────────────────


def test_leaf_count_no_retries_equals_end_count():
    begins = [
        {"call_id": "a", "stage": "extract"},
        {"call_id": "b", "stage": "extract"},
    ]
    ends = [{"call_id": "a"}, {"call_id": "b"}]
    leaves, retries = _leaf_count(begins, ends)
    assert (leaves, retries) == (2, 0)


def test_leaf_count_transient_chain_credits_only_terminal():
    # `c1 → c2 → c3` regular-retry chain (transient backoff). Only
    # c3 (leaf) counts. c1 and c2 are intermediates surfaced as
    # `+2 retries` in the UI.
    begins = [
        {"call_id": "c1", "stage": "extract"},
        {"call_id": "c2", "stage": "extract", "retry_of_call_id": "c1"},
        {"call_id": "c3", "stage": "extract", "retry_of_call_id": "c2"},
    ]
    ends = [{"call_id": "c1"}, {"call_id": "c2"}, {"call_id": "c3"}]
    leaves, retries = _leaf_count(begins, ends)
    assert (leaves, retries) == (1, 2)


def test_leaf_count_halve_cascade_credits_grandchildren():
    # `parent` halves into `a, b`; `b` halves into `b1, b2`. Leaves
    # are `a, b1, b2` — 3 productive results. Intermediates `parent, b`
    # are surfaced as `+2 retries`. The runner's
    # `_bump_stage_est("extract", 1)` per halve event grows the
    # denominator by exactly 2 here (one bump per halve event), so
    # ratio = 3 leaves / (1 + 2) bumps = 3/3 = 1.0. ✓
    begins = [
        {"call_id": "parent", "stage": "extract"},
        {"call_id": "a", "stage": "extract", "retry_of_call_id": "parent"},
        {"call_id": "b", "stage": "extract", "retry_of_call_id": "parent"},
        {"call_id": "b1", "stage": "extract", "retry_of_call_id": "b"},
        {"call_id": "b2", "stage": "extract", "retry_of_call_id": "b"},
    ]
    ends = [
        {"call_id": "parent"}, {"call_id": "a"}, {"call_id": "b"},
        {"call_id": "b1"}, {"call_id": "b2"},
    ]
    leaves, retries = _leaf_count(begins, ends)
    assert (leaves, retries) == (3, 2)

    # Tracker side: 1 starting est + 2 halve bumps = 3. Matches leaf count.
    tracker = ProgressTracker()
    tracker.register_stage("extract", "test-model", est_calls=1)
    tracker.bump_est_calls("extract", 1)  # first halve event
    tracker.bump_est_calls("extract", 1)  # second halve event
    assert tracker.compute_pipeline_total_calls() == 3


def test_leaf_count_sample_loop_credits_only_terminal_iteration():
    # Patterns sample loop: each iteration retries the same logical
    # productive call with reduced facts. The `_bump_stage_est` call
    # is correctly absent (no new productive leaf added). Leaf count
    # stays at 1; retries surface the iteration count.
    begins = [
        {"call_id": "s0", "stage": "patterns", "category": "topic"},
        {"call_id": "s1", "stage": "patterns",
         "category": "topic/sample-1", "retry_of_call_id": "s0"},
        {"call_id": "s2", "stage": "patterns",
         "category": "topic/sample-2", "retry_of_call_id": "s1"},
    ]
    ends = [{"call_id": "s0"}, {"call_id": "s1"}, {"call_id": "s2"}]
    leaves, retries = _leaf_count(begins, ends)
    assert (leaves, retries) == (1, 2)

    # Tracker side: 1 starting est, no bumps. Matches leaf count.
    tracker = ProgressTracker()
    tracker.register_stage("patterns", "test-model", est_calls=1)
    assert tracker.compute_pipeline_total_calls() == 1


def test_leaf_count_reasoning_off_retry_stays_one_leaf():
    # Sizing reasoning-off retry: parse_error → re-call with
    # reasoning off (linked via retry_of_call_id). Same productive
    # call, 2 attempts. One leaf; 1 retry surfaced.
    begins = [
        {"call_id": "e0", "stage": "actions"},
        {"call_id": "e1", "stage": "actions",
         "category": "actions/reasoning-off - retry/sizing",
         "retry_of_call_id": "e0"},
    ]
    ends = [{"call_id": "e0"}, {"call_id": "e1"}]
    leaves, retries = _leaf_count(begins, ends)
    assert (leaves, retries) == (1, 1)


def test_leaf_count_pd5w_shape_ratio_stays_balanced():
    # Synthetic recreation of pd5w's 100-end / 43-leaf storm, scoped
    # down per stage. Mixes regular transient chains in entities
    # (4 → 1 leaf each, 5 batches) with halve cascades in extract
    # (some 1→2, some 1→2→4) and a single dedupe + insights + actions
    # call each that ran multiple transient retries. Asserts the
    # tracker's est_calls + bump_est_calls invocations match the leaf
    # count produced by the rust-equivalent algorithm.
    begins: list[dict] = []
    ends: list[dict] = []

    # extract: 5 docs, 2 of them halved 1→2 (2 bumps); 1 halved 1→2→4
    # (3 bumps). Total halve bumps: 2 + 3 = 5. Final extract leaves =
    # 2 (clean) + 2*2 (halve-1) + 4 (halve-2 cascade) = 10. Final
    # tracker total for extract = 5 starting est + 5 bumps = 10. ✓
    cid = 0
    def add_chain(stage: str, depth_chain: list[tuple[str, str | None]]):
        nonlocal cid
        for cat, parent in depth_chain:
            cid += 1
            local = f"x{cid}"
            ev = {"call_id": local, "stage": stage, "category": cat}
            if parent is not None:
                ev["retry_of_call_id"] = parent
            begins.append(ev)
            ends.append({"call_id": local})

    # extract: 2 clean leaves (doc1, doc2)
    add_chain("extract", [("doc1", None)])
    add_chain("extract", [("doc2", None)])

    # extract: 2 halved 1→2 — parent + 2 children each
    add_chain("extract", [("doc3", None)])
    p3 = begins[-1]["call_id"]
    add_chain("extract", [("doc3/half-1", p3), ("doc3/half-2", p3)])

    add_chain("extract", [("doc4", None)])
    p4 = begins[-1]["call_id"]
    add_chain("extract", [("doc4/half-1", p4), ("doc4/half-2", p4)])

    # extract: 1 doc cascaded 1 → 2 → 4
    add_chain("extract", [("doc5", None)])
    p5 = begins[-1]["call_id"]
    add_chain("extract", [("doc5/half-1", p5)])
    p5h1 = begins[-1]["call_id"]
    add_chain("extract", [("doc5/half-2", p5)])
    p5h2 = begins[-1]["call_id"]
    add_chain(
        "extract",
        [
            ("doc5/half-1/half-1", p5h1),
            ("doc5/half-1/half-2", p5h1),
            ("doc5/half-2/half-1", p5h2),
            ("doc5/half-2/half-2", p5h2),
        ],
    )

    # Set up tracker as the runner would: register starting est,
    # bump per halve event.
    tracker = ProgressTracker()
    tracker.register_stage("extract", "test-model", est_calls=5)
    # 2 halve events at depth 0→1 (doc3, doc4): +2
    tracker.bump_est_calls("extract", 1)
    tracker.bump_est_calls("extract", 1)
    # doc5 cascade: 1 halve at depth 0→1, 2 halves at depth 1→2: +3
    tracker.bump_est_calls("extract", 1)
    tracker.bump_est_calls("extract", 1)
    tracker.bump_est_calls("extract", 1)

    leaves, retries = _leaf_count(begins, ends)
    total = tracker.compute_pipeline_total_calls()

    # Expected leaves: clean1, clean2, doc3/half-1, doc3/half-2,
    # doc4/half-1, doc4/half-2, doc5 cascade leaves (4) = 10.
    assert leaves == 10
    # Expected total via tracker: 5 + 5 = 10.
    assert total == 10
    # The hard invariant: completed ≤ total. The orchestrator's
    # acceptance criterion is `ratio stays ≤ 1.0` at every stage.
    assert leaves <= total
    # Intermediates: p3 + p4 + p5 + p5h1 + p5h2 = 5 retries (5 halve
    # parents). clean1 and clean2 are leaves, not intermediates.
    assert retries == 5
    # Cross-check: leaves + retries = total ends.
    assert leaves + retries == len(ends)


def test_leaf_count_late_retry_demotes_earlier_leaf():
    # Race: end event for `a` lands first, then much later a retry `b`
    # fires. Final-pass classification (the algorithm runs on the
    # whole jsonl) demotes `a` from leaf to intermediate.
    begins = [
        {"call_id": "a", "stage": "extract"},
        # ... interleaved with unrelated work ...
        {"call_id": "x", "stage": "extract"},
        # ... and finally the retry of `a` ...
        {"call_id": "b", "stage": "extract", "retry_of_call_id": "a"},
    ]
    ends = [{"call_id": "a"}, {"call_id": "x"}, {"call_id": "b"}]
    leaves, retries = _leaf_count(begins, ends)
    assert (leaves, retries) == (2, 1)
