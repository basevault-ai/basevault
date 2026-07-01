"""Resilience of extract Phase 2 entry + Phase 3 parser to truncated JSONLs.

A process kill mid-write of `_append_fact_jsonl` can leave a per-topic
JSONL with a half-line at the end (or, in the future, in the middle if
multiple writers ever interleave). The strict
`[json.loads(l) for l in lines]` form in the original Phase 3 aborted
the run on the first malformed line, throwing away potentially hours
of upstream LLM work.

Two unit tests on the tolerant parser, one on the tail-repair guard.
"""
from __future__ import annotations

import io
import json



from engine import runner  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────


def _capture_log(monkeypatch) -> io.StringIO:
    """Redirect runner._log_write at the module-global file handle so
    parser/guard log lines are observable from the test."""
    buf = io.StringIO()

    class _Handle:
        closed = False

        def write(self, s):
            buf.write(s)

        def flush(self):
            pass

    monkeypatch.setattr(runner, "_log_file", _Handle())
    return buf


def _good_fact_line(summary: str, file_offset: int = 0) -> str:
    return json.dumps({
        "type": "fact",
        "summary": summary,
        "occurred_at": "2026-04-15",
        "occurred_at_text": None,
        "entities": [],
        "evidence": [{
            "text": summary,
            "ref": "fixture",
            "file_path": "fixture.md",
            "file_offset": file_offset,
        }],
        "topics": ["work"],
        "tags": [],
        "confidence": 0.9,
        "relation_candidate": False,
    })


# ── Phase 3 parser ────────────────────────────────────────────────────────────


def test_phase3_parser_skips_truncated_trailing_line(tmp_path, monkeypatch):
    log = _capture_log(monkeypatch)
    p = tmp_path / "work.jsonl"
    good = [_good_fact_line(f"good-{i}", i) for i in range(3)]
    truncated = '{"type": "fact", "summary": "Alice signed an unt'  # no `}`
    p.write_text("\n".join(good) + "\n" + truncated, encoding="utf-8")

    dicts = runner._parse_jsonl_facts_tolerant(p)
    assert len(dicts) == 3
    assert [d["summary"] for d in dicts] == ["good-0", "good-1", "good-2"]

    # The truncated line is at byte offset = sum of (good lines + their \n).
    expected_offset = sum(len((g + "\n").encode("utf-8")) for g in good)
    log_text = log.getvalue()
    assert "skipping malformed JSONL line" in log_text
    assert f"byte offset {expected_offset}" in log_text
    assert "Alice signed an unt" in log_text  # head excerpt


def test_phase3_parser_skips_truncated_middle_line(tmp_path, monkeypatch):
    log = _capture_log(monkeypatch)
    p = tmp_path / "work.jsonl"
    good_pre = _good_fact_line("good-before", 0)
    bad = '{"type": "fact", "summary": "Alice'  # no `}`
    good_post = _good_fact_line("good-after", 100)
    p.write_text(
        good_pre + "\n" + bad + "\n" + good_post + "\n",
        encoding="utf-8",
    )

    dicts = runner._parse_jsonl_facts_tolerant(p)
    assert [d["summary"] for d in dicts] == ["good-before", "good-after"]

    expected_offset = len((good_pre + "\n").encode("utf-8"))
    log_text = log.getvalue()
    assert "skipping malformed JSONL line" in log_text
    assert f"byte offset {expected_offset}" in log_text


def test_phase3_parser_no_op_on_well_formed_file(tmp_path, monkeypatch):
    log = _capture_log(monkeypatch)
    p = tmp_path / "work.jsonl"
    good = [_good_fact_line(f"good-{i}", i) for i in range(5)]
    p.write_text("\n".join(good) + "\n", encoding="utf-8")

    dicts = runner._parse_jsonl_facts_tolerant(p)
    assert len(dicts) == 5
    assert "skipping malformed" not in log.getvalue()


# ── Tail-truncation guard ─────────────────────────────────────────────────────


def test_repair_partial_jsonl_tails_drops_half_line(tmp_path, monkeypatch):
    log = _capture_log(monkeypatch)
    facts_dir = tmp_path / "facts"
    facts_dir.mkdir()
    p = facts_dir / "work.jsonl"
    good = _good_fact_line("good-0", 0)
    half = '{"type": "fact", "summary": "Alice signed'
    # No trailing newline → half-line on disk.
    p.write_text(good + "\n" + half, encoding="utf-8")
    assert not p.read_bytes().endswith(b"\n")

    runner._repair_partial_jsonl_tails(facts_dir)

    repaired = p.read_bytes()
    assert repaired.endswith(b"\n")
    assert b"Alice signed" not in repaired
    # Surviving good line is intact and parses.
    surviving = repaired.decode("utf-8").splitlines()
    assert len(surviving) == 1
    assert json.loads(surviving[0])["summary"] == "good-0"

    assert "repaired truncated JSONL tail" in log.getvalue()


def test_repair_partial_jsonl_tails_no_op_on_well_formed(tmp_path, monkeypatch):
    log = _capture_log(monkeypatch)
    facts_dir = tmp_path / "facts"
    facts_dir.mkdir()
    p = facts_dir / "work.jsonl"
    good = "\n".join(_good_fact_line(f"good-{i}", i) for i in range(3)) + "\n"
    p.write_text(good, encoding="utf-8")
    before = p.read_bytes()

    runner._repair_partial_jsonl_tails(facts_dir)

    assert p.read_bytes() == before
    assert "repaired truncated JSONL tail" not in log.getvalue()


def test_repair_partial_jsonl_tails_handles_no_newline_anywhere(tmp_path, monkeypatch):
    """Edge case: a file that's entirely a single half-line (no newline
    at all) gets truncated to 0 bytes."""
    _capture_log(monkeypatch)
    facts_dir = tmp_path / "facts"
    facts_dir.mkdir()
    p = facts_dir / "work.jsonl"
    p.write_text('{"type": "fact", "summary": "Alice', encoding="utf-8")

    runner._repair_partial_jsonl_tails(facts_dir)

    assert p.read_bytes() == b""


def test_repair_partial_jsonl_tails_no_op_on_missing_dir(tmp_path):
    # Pre-Phase-2 sanity sweep must not blow up on a fresh run where
    # facts_dir hasn't been created yet.
    runner._repair_partial_jsonl_tails(tmp_path / "does-not-exist")
