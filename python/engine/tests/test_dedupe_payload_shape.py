"""Tests for the dedupe prompt's payload-shape behavior (#256).

Pins:
  1. `_format_dedupe_row` drops the aliases segment entirely when the
     entity has no aliases — no `aliases: —` placeholder dead-token.
  2. `minimal=True` renders id + name + type only (no aliases / desc).
  3. `_render_dedupe_rows(sample_step=N)` keeps every record visible
     and renders the bottom (1 - 1/2**N) fraction minimally; sort key
     is (-mention_count, canonical_name) so the keep-set is stable
     across runs.

No LLM calls. Run with:
    pytest engine/tests/test_dedupe_payload_shape.py -v
"""
from __future__ import annotations



from engine.entities import (  # noqa: E402
    EntityRecord,
    _format_dedupe_row,
    _render_dedupe_rows,
)


def _rec(
    *, cid: str, name: str, type_: str = "person",
    aliases: list[str] | None = None,
    desc: str = "test desc",
    mention_count: int = 1,
) -> EntityRecord:
    return EntityRecord(
        canonical_id=cid,
        canonical_name=name,
        entity_type=type_,
        aliases=list(aliases or []),
        role="subject",
        description=desc,
        mention_count=mention_count,
        topics=[],
        evidence_fact_refs=[],
    )


class TestFormatDedupeRow:
    def test_with_aliases_keeps_aliases_segment(self):
        r = _rec(cid="a-1", name="Alice", aliases=["A.", "Ali"],
                 mention_count=7)
        out = _format_dedupe_row(r)
        assert "aliases: A., Ali" in out
        assert "desc: test desc" in out
        assert "mentions=7" in out
        assert out == "- a-1 | Alice (person) | mentions=7 | aliases: A., Ali | desc: test desc"

    def test_no_aliases_drops_segment_entirely(self):
        # Pre-#256 this rendered `| aliases: —` — dead tokens that
        # ate output budget for nothing. Now: segment omitted.
        r = _rec(cid="b-2", name="Bob", aliases=[], mention_count=3)
        out = _format_dedupe_row(r)
        assert "aliases:" not in out
        assert "—" not in out
        assert out == "- b-2 | Bob (person) | mentions=3 | desc: test desc"

    def test_minimal_drops_aliases_and_desc(self):
        # Minimal still drops aliases/desc, but mention count is
        # now load-bearing for the dedupe LLM (the high-count row is
        # the canonical naming) — surfaced in BOTH full and minimal.
        r = _rec(cid="c-3", name="Carol", aliases=["C."],
                 desc="Carol description", mention_count=5)
        out = _format_dedupe_row(r, minimal=True)
        assert "aliases:" not in out
        assert "desc:" not in out
        assert "Carol description" not in out
        assert out == "- c-3 | Carol (person) | mentions=5"

    def test_empty_desc_falls_back_to_placeholder(self):
        r = _rec(cid="d-4", name="Dave", desc="")
        out = _format_dedupe_row(r)
        assert "desc: (no description)" in out

    def test_aliases_capped_at_six(self):
        many = [f"a{i}" for i in range(10)]
        r = _rec(cid="e-5", name="Eve", aliases=many)
        out = _format_dedupe_row(r)
        # Only first 6 rendered.
        assert "aliases: a0, a1, a2, a3, a4, a5" in out
        assert "a6" not in out


class TestRenderDedupeRows:
    def test_sample_step_zero_renders_all_full(self):
        records = [
            _rec(cid="a-1", name="Alice", aliases=["A."], mention_count=10),
            _rec(cid="b-2", name="Bob", aliases=[], mention_count=2),
        ]
        out = _render_dedupe_rows(records, sample_step=0)
        # Full rendering means both rows carry desc; no record dropped.
        assert "Alice" in out
        assert "Bob" in out
        assert "desc: test desc" in out
        # Two lines.
        assert out.count("\n") == 1

    def test_sample_step_one_keeps_top_half_full_bottom_minimal(self):
        records = [
            _rec(cid=f"r-{i}", name=f"R{i}",
                 aliases=[f"alias-{i}"],
                 desc=f"description-{i}",
                 mention_count=10 - i)
            for i in range(8)
        ]
        out = _render_dedupe_rows(records, sample_step=1)
        lines = out.split("\n")
        # Every record stays visible (record COUNT preserved).
        assert len(lines) == 8
        # Top 4 (highest mention_count) keep desc.
        for i in range(4):
            assert f"description-{i}" in lines[i], (
                f"top half row {i} missing desc: {lines[i]}")
        # Bottom 4 are minimal — no desc, no aliases.
        for i in range(4, 8):
            assert "desc:" not in lines[i]
            assert "aliases:" not in lines[i]
            assert lines[i].startswith("- r-")

    def test_sample_step_two_keeps_top_quarter_full(self):
        records = [
            _rec(cid=f"r-{i}", name=f"R{i}",
                 desc=f"description-{i}",
                 mention_count=100 - i)
            for i in range(8)
        ]
        out = _render_dedupe_rows(records, sample_step=2)
        lines = out.split("\n")
        assert len(lines) == 8  # all records still visible
        # Top 2 (8 // 2**2 = 2) full.
        full_count = sum(1 for ln in lines if "desc:" in ln)
        assert full_count == 2

    def test_sample_step_floor_at_one_full_record(self):
        # Even at very deep sample steps, at least one row renders
        # full so the model has an anchor. n_full = max(1, n // 2**N).
        records = [_rec(cid=f"r-{i}", name=f"R{i}") for i in range(4)]
        out = _render_dedupe_rows(records, sample_step=10)
        full_count = sum(1 for ln in out.split("\n") if "desc:" in ln)
        assert full_count == 1

    def test_sort_key_stable(self):
        # Mention_count desc, then canonical_name asc — deterministic.
        records = [
            _rec(cid="a-1", name="Charlie", mention_count=5),
            _rec(cid="b-2", name="Alice", mention_count=5),  # same count, name wins
            _rec(cid="c-3", name="Bob", mention_count=10),
        ]
        out = _render_dedupe_rows(records, sample_step=1)
        # n=3, sample_step=1 → n_full = 3 // 2 = 1. Top one full, two minimal.
        # Top by sort: mention_count=10 → Bob.
        lines = out.split("\n")
        assert "Bob" in lines[0]
        assert "desc:" in lines[0]
        # The two minimal lines: Alice (count=5, name first) then Charlie.
        assert "Alice" in lines[1]
        assert "desc:" not in lines[1]
        assert "Charlie" in lines[2]

    def test_records_never_dropped(self):
        # The whole point of the new shape: record count is invariant
        # under sample_step. Pre-#256 the sample dropped records,
        # losing identity coverage; now identity tokens always survive.
        records = [_rec(cid=f"r-{i}", name=f"R{i}") for i in range(20)]
        for step in range(5):
            lines = _render_dedupe_rows(records, sample_step=step).split("\n")
            assert len(lines) == 20, (
                f"sample_step={step}: expected 20 records, got {len(lines)}"
            )

    def test_empty_records_returns_empty(self):
        assert _render_dedupe_rows([], sample_step=0) == ""
        assert _render_dedupe_rows([], sample_step=3) == ""
