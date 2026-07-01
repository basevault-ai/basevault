"""
llm-stats.txt format tests.

llm-stats.txt is the human-readable monospace summary materialized
at end-of-run. The format is OWNED by runner; consumers (human-only,
for now) read it left-to-right on a 120-char terminal. Post-#189
this is the only on-disk rollup artifact — the dict itself is
derived on demand via `materialize_run_stats(jsonl, config)`.

These tests pin the format with a small known payload so a
column-width or label change is caught immediately. If you change
the format intentionally, regenerate the expected text in this file.

No live LLM calls. Run with:
    cd engine && pytest tests/test_llm_stats_text.py -v
"""
from __future__ import annotations


import pytest


from engine.runner import _render_llm_stats_text


def _mk_payload() -> dict:
    """Build a minimal-but-realistic rollup payload — 2 stages, one
    aborted call, two models. All numbers chosen to produce stable
    formatted output (round numbers, no rounding-tie surprises)."""
    return {
        "schema": "llm-stats/v1",
        "run_id": "2026-04-29T17-07-29Z-rbw8",
        "short_id": "rbw8",
        "agent": "app",
        "mode": "tee",
        "primary_model": "gpt-oss-120b",
        "started_at_iso": "2026-04-29T17:07:35.000Z",
        "ended_at_iso": "2026-04-29T17:11:35.000Z",
        "total_wall_clock_ms": 240_000,
        "totals": {
            "calls": 4, "successful": 2, "failed": 1, "aborted": 1,
            "categories_lost": 0, "prompt_tokens": 6000,
            "completion_tokens": 600,
        },
        "by_stage": {},
        "by_model": {
            "gpt-oss-120b": {"calls": 3, "prompt_tokens": 3000, "completion_tokens": 300},
            "glm-5-2": {"calls": 1, "prompt_tokens": 3000, "completion_tokens": 300},
        },
        "per_stage": {
            "extract": {
                "name": "extract",
                "calls_total": 3,
                "calls_failed": 1,
                "calls_aborted": 1,
                "models_used": ["gpt-oss-120b"],
                "prompt_tokens": {
                    "total": 3000, "avg": 1500.0, "median": 1500.0,
                    "min": 1000, "max": 2000,
                },
                "completion_tokens": {
                    "total": 200, "avg": 100.0, "median": 100.0,
                    "min": 50, "max": 150,
                },
                "duration_ms": {
                    "total": 60_000, "avg": 30_000.0, "median": 30_000.0,
                    "min": 20_000, "max": 40_000,
                },
            },
            "patterns": {
                "name": "patterns",
                "calls_total": 1,
                "calls_failed": 0,
                "calls_aborted": 0,
                "models_used": ["glm-5-2"],
                "prompt_tokens": {
                    "total": 3000, "avg": 3000.0, "median": 3000.0,
                    "min": 3000, "max": 3000,
                },
                "completion_tokens": {
                    "total": 400, "avg": 400.0, "median": 400.0,
                    "min": 400, "max": 400,
                },
                "duration_ms": {
                    "total": 50_000, "avg": 50_000.0, "median": 50_000.0,
                    "min": 50_000, "max": 50_000,
                },
            },
        },
        "statistics": {
            "calls_total": 4,
            "calls_failed": 1,
            "calls_aborted": 1,
            "models_used": ["glm-5-2", "gpt-oss-120b"],
            "prompt_tokens": {
                "total": 6000, "avg": 2000.0, "median": 1500.0,
                "min": 1000, "max": 3000,
            },
            "completion_tokens": {
                "total": 600, "avg": 200.0, "median": 150.0,
                "min": 50, "max": 400,
            },
            "duration_ms": {
                "total": 110_000, "avg": 36_666.67, "median": 35_000.0,
                "min": 20_000, "max": 50_000,
            },
        },
        "aborted_calls": [
            {
                "call_id": "0042",
                "stage": "extract",
                "category": "split_07",
                "model": "gpt-oss-120b",
                "started_at_iso": "2026-04-29T17:09:45.000Z",
                "duration_ms": 24_000,
            },
        ],
        "calls": [],
    }


class TestRunStatsTextStructure:
    def test_header_lines_present(self):
        text = _render_llm_stats_text(_mk_payload())
        assert "BaseVault run summary" in text
        assert "Run id:    rbw8" in text
        # aborted count > 0 → .txt status header still reads "cancelled"
        # (run-level vocabulary; aborted is the per-call axis).
        assert "Status:    cancelled" in text
        # Mode line carries the run mode only. The preset id is not
        # appended — the "Models used" table is authoritative for what
        # actually ran (issue #93: header used to lie when the user
        # overrode per-stage models away from the preset).
        assert "Mode:      tee" in text
        assert "mixed-gpt-oss-glm" not in text
        assert "(gpt-oss-120b)" not in text
        # Duration: 240,000ms = 4m 0s.
        assert "4m 0s" in text

    def test_per_stage_section_present(self):
        text = _render_llm_stats_text(_mk_payload())
        assert "Per-stage statistics" in text
        # Both stages appear.
        assert "extract" in text
        assert "patterns" in text
        # Header columns.
        assert "stage" in text
        assert "calls" in text
        assert "fail" in text
        assert "abrt" in text
        # Numeric values for extract: 3 calls, 1 fail, 1 abrt.
        # The exact column-aligned line should be present.
        for line in text.splitlines():
            if line.startswith("extract"):
                # 3 (calls), 1 ok, 1 fail, 1 abrt.
                # Right-aligned numbers.
                assert "     3" in line  # calls right-aligned in width 6
                # Token avgs/medians render with comma grouping.
                assert "1,500" in line
                break
        else:
            pytest.fail("extract row not rendered")

    def test_models_used_section(self):
        text = _render_llm_stats_text(_mk_payload())
        assert "Models used" in text
        # Both models with their call counts.
        assert "gpt-oss-120b" in text
        assert "glm-5-2" in text

    def test_aborted_calls_section(self):
        text = _render_llm_stats_text(_mk_payload())
        assert "Calls in flight when run wound down" in text
        assert "call 0042" in text
        assert "stage=extract" in text
        # Duration 24s renders as "24.0s".
        assert "24.0s" in text

    def test_completed_status_when_no_aborted(self):
        p = _mk_payload()
        p["totals"]["aborted"] = 0
        p["totals"]["failed"] = 0
        p["per_stage"]["extract"]["calls_aborted"] = 0
        p["per_stage"]["extract"]["calls_failed"] = 0
        p["aborted_calls"] = []
        text = _render_llm_stats_text(p)
        assert "Status:    completed" in text
        # No "cancelled" in the status line.
        for line in text.splitlines():
            if line.startswith("Status:"):
                assert "cancelled" not in line.lower()

    def test_completed_with_failures_status(self):
        p = _mk_payload()
        p["totals"]["aborted"] = 0
        p["totals"]["failed"] = 1
        p["aborted_calls"] = []
        text = _render_llm_stats_text(p)
        assert "Status:    completed (with failures)" in text

    def test_text_ends_with_single_newline(self):
        # Avoids "no newline at end of file" annoyances when consumers
        # cat or pipe the file.
        text = _render_llm_stats_text(_mk_payload())
        assert text.endswith("\n")
        assert not text.endswith("\n\n\n")


class TestPerStageColumnAlignment:
    """Per-stage table must keep columns aligned even when a stage
    name exceeds the historical 10-char hardcoded width
    (`entities_dedupe` is 15 chars). Header and data rows are pinned
    by literal string compare — exact alignment is the contract."""

    def _payload_with_long_stage(self) -> dict:
        p = _mk_payload()
        p["per_stage"] = {
            "extract": {
                "name": "extract",
                "calls_total": 2,
                "calls_failed": 0,
                "calls_aborted": 0,
                "models_used": ["gpt-oss-120b"],
                "prompt_tokens": {
                    "total": 2000, "avg": 1000.0, "median": 1000.0,
                    "min": 1000, "max": 1000,
                },
                "completion_tokens": {
                    "total": 200, "avg": 100.0, "median": 100.0,
                    "min": 100, "max": 100,
                },
                "duration_ms": {
                    "total": 20_000, "avg": 10_000.0, "median": 10_000.0,
                    "min": 10_000, "max": 10_000,
                },
            },
            "entities_dedupe": {
                "name": "entities_dedupe",
                "calls_total": 1,
                "calls_failed": 0,
                "calls_aborted": 0,
                "models_used": ["gpt-oss-120b"],
                "prompt_tokens": {
                    "total": 3000, "avg": 3000.0, "median": 3000.0,
                    "min": 3000, "max": 3000,
                },
                "completion_tokens": {
                    "total": 300, "avg": 300.0, "median": 300.0,
                    "min": 300, "max": 300,
                },
                "duration_ms": {
                    "total": 15_000, "avg": 15_000.0, "median": 15_000.0,
                    "min": 15_000, "max": 15_000,
                },
            },
        }
        return p

    def test_long_stage_name_keeps_columns_aligned(self):
        text = _render_llm_stats_text(self._payload_with_long_stage())
        lines = text.splitlines()
        header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("stage"))
        header = lines[header_idx]
        rows = [lines[header_idx + 1], lines[header_idx + 2]]

        # Longest stage is `entities_dedupe` (15 chars) → column
        # width is 16 (15 + 1 trailing space). Header columns:
        #   stage:<16>  calls:>6  ok:>5  fail:>5  abrt:>5  skip:>5  "  "
        # giving prefix "stage" + 11 spaces + " calls   ok fail abrt skip  ".
        expected_header_prefix = (
            "stage" + " " * 11
            + " calls" + "   ok" + " fail" + " abrt" + " skip" + "  "
        )
        assert header.startswith(expected_header_prefix), (
            f"header prefix mismatch.\n"
            f"got:  {header[:len(expected_header_prefix)]!r}\n"
            f"want: {expected_header_prefix!r}"
        )

        # Literal pin for both data rows. Stage name padded to 16,
        # then calls/ok/fail/abrt/skip right-aligned in widths
        # 6/5/5/5/5.
        expected_extract_prefix = (
            "extract" + " " * 9
            + "     2" + "    2" + "    0" + "    0" + "    0" + "  "
        )
        expected_dedupe_prefix = (
            "entities_dedupe "
            + "     1" + "    1" + "    0" + "    0" + "    0" + "  "
        )
        extract_row = next(r for r in rows if r.startswith("extract"))
        dedupe_row = next(r for r in rows if r.startswith("entities_dedupe"))
        assert extract_row.startswith(expected_extract_prefix), (
            f"extract row prefix mismatch.\n"
            f"got:  {extract_row[:len(expected_extract_prefix)]!r}\n"
            f"want: {expected_extract_prefix!r}"
        )
        assert dedupe_row.startswith(expected_dedupe_prefix), (
            f"entities_dedupe row prefix mismatch.\n"
            f"got:  {dedupe_row[:len(expected_dedupe_prefix)]!r}\n"
            f"want: {expected_dedupe_prefix!r}"
        )

        # Cross-row column-position check: the trailing "  " separator
        # between the numeric block and the token block lives at the
        # same column index in the header and every data row.
        sep_col = header.index(" calls") + 6 + 5 + 5 + 5 + 5  # end of skip
        for ln in [header, extract_row, dedupe_row]:
            assert ln[sep_col:sep_col + 2] == "  ", (
                f"separator misaligned at col {sep_col} in line: {ln!r}"
            )

    def test_short_stages_only_keeps_layout(self):
        # When all stage names are short, column width is
        # max(len("stage"), max(stage names)) + 1.
        # Payload has `extract` (7) and `patterns` (8) → width = 9.
        text = _render_llm_stats_text(_mk_payload())
        lines = text.splitlines()
        header = next(ln for ln in lines if ln.startswith("stage"))
        expected_header_prefix = (
            "stage" + " " * 4
            + " calls" + "   ok" + " fail" + " abrt" + " skip" + "  "
        )
        assert header.startswith(expected_header_prefix), (
            f"header prefix mismatch.\n"
            f"got:  {header[:len(expected_header_prefix)]!r}\n"
            f"want: {expected_header_prefix!r}"
        )
