"""
Per-stage streaming rollup tests (issue #104 part 1).

The on-demand rollup (`materialize_run_stats(jsonl, config)` —
issue #189) aggregates the per-call observability fields into
per-stage distributions. This file pins:

  - `reasoning_tokens` and `content_tokens` show up under per_stage[]
    as `_agg_dist`-shaped distributions (sum/avg/median/p95/min/max).
  - `ttft_ms`, `ttfr_ms`, `last_token_ms` get the same treatment so
    the .txt summary can render p50 TTFT and the UI can read p95 for
    admission-signal SLOs.
  - `finish_reasons` is a `{reason: count}` dict per stage so e.g. the
    UI can flag "47 stop / 1 length" without recomputing.
  - `_agg_dist` includes p95 (added in this PR).

No live LLM calls. Run with:
    cd engine && pytest tests/test_per_stage_streaming_rollup.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest


from engine import llm  # noqa: E402
from engine.runner import (  # noqa: E402
    _agg_dist,
    _do_write_llm_stats,
    _reset_llm_stats_dump_state,
    materialize_run_stats,
)


# ── _agg_dist gains a p95 field ───────────────────────────────────────────────


class TestAggDistP95:
    def test_p95_present_on_empty(self):
        d = _agg_dist([])
        assert "p95" in d
        assert d["p95"] is None

    def test_p95_equals_max_for_two_values(self):
        d = _agg_dist([100, 200])
        assert d["p95"] == 200.0

    def test_p95_ranks_correctly_on_twenty_values(self):
        # Sorted: 1..20. rank-0.95 of n=20 → idx=round(0.95*19)=18 → value 19.
        d = _agg_dist(list(range(1, 21)))
        assert d["p95"] == 19.0

    def test_p95_handles_single_value(self):
        d = _agg_dist([42])
        assert d["p95"] == 42.0


# ── per_stage[*].reasoning_tokens / content_tokens / ttft_ms ──────────────────


def _seed_jsonl(tmp_path, calls: list[dict]) -> Path:
    """Write a synthetic llm-calls.jsonl file by calling the live
    streaming helpers — keeps the test honest about the real schema
    (vs. hand-crafting events that drift from the production format).

    `calls` items:
      {
        "stage": str, "category": str, "model": str,
        "prompt_tokens": int, "completion_tokens": int,
        "reasoning_tokens": int, "finish_reason": str,
        "ttft_ms": int | None, "ttfr_ms": int | None, "last_token_ms": int | None,
        "duration_ms": int,
      }
    """
    p = tmp_path / "llm-calls.jsonl"
    llm.set_calls_jsonl_path(p)
    llm.reset_stat_records()
    for c in calls:
        cid = llm.begin_stat_record(c["stage"], c["category"], c["model"])
        # Stamp the streaming-observability fields directly via
        # _record_usage so the helper exercises the same flow
        # complete() does on a real provider call. Post-#264
        # `_record_usage` takes call_id explicitly (no thread-scoped
        # back-channel) so pass `cid` through here.
        llm._record_usage(
            "tinfoil", c["model"],
            c["prompt_tokens"], c["completion_tokens"],
            call_id=cid,
            reasoning_tokens=c.get("reasoning_tokens", 0),
            finish_reason=c.get("finish_reason"),
            ttft_ms=c.get("ttft_ms"),
            ttfr_ms=c.get("ttfr_ms"),
            last_token_ms=c.get("last_token_ms"),
        )
        llm.finalize_stat_record(cid, success=True, duration_ms=c["duration_ms"])
    llm.set_calls_jsonl_path(None)
    return p


class TestPerStageRollup:
    """The materialized rollup payload must carry the per-stage
    distributions; UI + .txt rendering both read these."""

    def test_per_stage_carries_reasoning_and_ttft(self, tmp_path):
        # Seed three calls to the same stage with mixed reasoning sizes.
        _seed_jsonl(tmp_path, [
            {"stage": "patterns", "category": "topic-a", "model": "kimi-k2-6",
             "prompt_tokens": 500, "completion_tokens": 100,
             "reasoning_tokens": 30, "finish_reason": "stop",
             "ttft_ms": 1200, "ttfr_ms": 300, "last_token_ms": 5000,
             "duration_ms": 5500},
            {"stage": "patterns", "category": "topic-b", "model": "kimi-k2-6",
             "prompt_tokens": 800, "completion_tokens": 200,
             "reasoning_tokens": 80, "finish_reason": "stop",
             "ttft_ms": 1500, "ttfr_ms": 400, "last_token_ms": 7000,
             "duration_ms": 7200},
            {"stage": "patterns", "category": "topic-c", "model": "kimi-k2-6",
             "prompt_tokens": 600, "completion_tokens": 150,
             "reasoning_tokens": 0, "finish_reason": "stop",
             "ttft_ms": 900, "ttfr_ms": None, "last_token_ms": 4000,
             "duration_ms": 4100},
        ])
        payload = materialize_run_stats(tmp_path / "llm-calls.jsonl")
        ps = payload["per_stage"]["patterns"]

        # reasoning_tokens distribution: sum 110, p95 ≈ max value (80).
        rt = ps["reasoning_tokens"]
        assert rt["total"] == 110
        # 3 values: [0, 30, 80]; p95 idx = round(0.95*2)=2 → value 80.
        assert rt["p95"] == 80.0

        # content_tokens distribution: sum 100-30 + 200-80 + 150-0 = 70+120+150 = 340.
        ct = ps["content_tokens"]
        assert ct["total"] == 340

        # ttft_ms: values [1200, 1500, 900] → median 1200, max 1500.
        ttft = ps["ttft_ms"]
        assert ttft["median"] == 1200.0
        assert ttft["max"] == 1500

        # ttfr only populated on 2 of 3 calls — None entries get dropped
        # so the distribution reflects "calls that emitted reasoning"
        # rather than averaging "0 ttfr" into the bucket.
        ttfr = ps["ttfr_ms"]
        assert ttfr["total"] == 700  # 300 + 400; the None value didn't contribute
        # min comes from the populated subset
        assert ttfr["min"] == 300

        # finish_reasons: dict of {reason: count}.
        assert ps["finish_reasons"] == {"stop": 3}

    def test_finish_reasons_breaks_down_by_kind(self, tmp_path):
        _seed_jsonl(tmp_path, [
            {"stage": "extract", "category": "doc-1", "model": "gpt-oss-120b",
             "prompt_tokens": 100, "completion_tokens": 50,
             "reasoning_tokens": 0, "finish_reason": "stop",
             "ttft_ms": 800, "ttfr_ms": None, "last_token_ms": 1500,
             "duration_ms": 1600},
            {"stage": "extract", "category": "doc-2", "model": "gpt-oss-120b",
             "prompt_tokens": 100, "completion_tokens": 4096,
             "reasoning_tokens": 0, "finish_reason": "length",
             "ttft_ms": 600, "ttfr_ms": None, "last_token_ms": 90000,
             "duration_ms": 91000},
        ])
        payload = materialize_run_stats(tmp_path / "llm-calls.jsonl")
        ps = payload["per_stage"]["extract"]
        # Both finish_reasons categorized in the rollup so retry-policy
        # v2 can read "1 length" off the rollup without scanning calls.
        assert ps["finish_reasons"] == {"stop": 1, "length": 1}


# ── llm-stats.txt — the new reasoning + ttft columns ──────────────────────────


class TestRollupTextFormat:
    """The .txt summary gains a `reasoning (sum)` column and a
    `ttft p50` column — both read by humans inspecting failing runs."""

    def test_reasoning_column_present_in_txt(self, tmp_path, monkeypatch):
        _seed_jsonl(tmp_path, [
            {"stage": "patterns", "category": "topic-a", "model": "kimi-k2-6",
             "prompt_tokens": 500, "completion_tokens": 100,
             "reasoning_tokens": 60, "finish_reason": "stop",
             "ttft_ms": 1200, "ttfr_ms": 300, "last_token_ms": 5000,
             "duration_ms": 5500},
        ])
        from engine import runner
        monkeypatch.setattr(runner, "_log_write", lambda *a, **kw: None)
        _reset_llm_stats_dump_state()
        _do_write_llm_stats(
            tmp_path, llm.Mode.TEE, "kimi-k2-6",
            "2026-05-07T12:00:00Z",
        )
        text = (tmp_path / "llm-stats.txt").read_text()
        # Header carries the new column labels.
        assert "reasoning (sum)" in text
        assert "ttft p50" in text
        # And the renamed completion column.
        assert "total_completion" in text
        # The patterns row carries the reasoning sum (60).
        for line in text.splitlines():
            if line.startswith("patterns"):
                assert "60" in line
                # ttft 1200ms renders as "1.2s" via _fmt_duration_ms.
                assert "1.2s" in line
                break
        else:
            pytest.fail("patterns row not rendered in llm-stats.txt")
