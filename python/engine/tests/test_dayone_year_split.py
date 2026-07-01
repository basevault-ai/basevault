"""
Tests for per-year splitting of Day One JSON in ingestor._parse_dayone_json.

The original filename keeps the most-recent year of journaling + any undated
entries. Older years are pulled out into year-suffixed archive subfiles
(`<stem>-<YYYY>.<ext>`). The original is the "hot" file the user keeps
re-syncing into; older years are read-only archive.

`origin_file_id` metadata is set on docs that got split out (so a fact's
evidence anchor on a historical-year doc can surface "originally from <X>"),
and is NOT set on docs that stayed in the original.

Run from engine/:
    cd engine && pytest tests/test_dayone_year_split.py
"""
import json
from pathlib import Path


from engine.ingestor import SourceType, ingest


def _write_dayone(tmp_path: Path, name: str, entries: list[dict]) -> Path:
    f = tmp_path / name
    f.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return f


# ── Case 1: Multi-year (no undated) ───────────────────────────────────────────

class TestMultiYearNoUndated:
    def test_original_keeps_most_recent_year(self, tmp_path):
        f = _write_dayone(tmp_path, "Journal.json", [
            {"uuid": "A", "text": "entry 2018", "creationDate": "2018-06-15T10:00:00Z"},
            {"uuid": "B", "text": "entry 2020", "creationDate": "2020-01-01T00:00:00Z"},
            {"uuid": "C", "text": "entry 2024", "creationDate": "2024-09-01T12:00:00Z"},
        ])
        docs = ingest(f)
        by_file = {d.file_id: d.metadata["uuid"] for d in docs}
        assert by_file == {
            "Journal.json": "C",
            "Journal-2018.json": "A",
            "Journal-2020.json": "B",
        }

    def test_only_split_docs_get_origin_file_id(self, tmp_path):
        f = _write_dayone(tmp_path, "Journal.json", [
            {"uuid": "A", "text": "x", "creationDate": "2018-01-01T00:00:00Z"},
            {"uuid": "B", "text": "y", "creationDate": "2024-01-01T00:00:00Z"},
        ])
        docs = ingest(f)
        for d in docs:
            if d.file_id == "Journal.json":
                assert "origin_file_id" not in d.metadata
            else:
                assert d.metadata["origin_file_id"] == "Journal.json"

    def test_id_uniqueness_across_emitted_files(self, tmp_path):
        f = _write_dayone(tmp_path, "Journal.json", [
            {"uuid": "A", "text": "x", "creationDate": "2018-01-01T00:00:00Z"},
            {"uuid": "B", "text": "y", "creationDate": "2024-01-01T00:00:00Z"},
        ])
        docs = ingest(f)
        ids = [d.id for d in docs]
        assert len(ids) == len(set(ids))

    def test_year_suffix_preserves_extension(self, tmp_path):
        f = _write_dayone(tmp_path, "Journal.stripped.json", [
            {"uuid": "A", "text": "x", "creationDate": "2018-01-01T00:00:00Z"},
            {"uuid": "B", "text": "y", "creationDate": "2024-01-01T00:00:00Z"},
        ])
        docs = ingest(f)
        file_ids = {d.file_id for d in docs}
        assert file_ids == {"Journal.stripped.json", "Journal.stripped-2018.json"}

    def test_each_year_groups_its_entries(self, tmp_path):
        f = _write_dayone(tmp_path, "Journal.json", [
            {"uuid": "A1", "text": "alpha 2018", "creationDate": "2018-03-01T00:00:00Z"},
            {"uuid": "A2", "text": "beta 2018", "creationDate": "2018-04-01T00:00:00Z"},
            {"uuid": "B1", "text": "gamma 2024", "creationDate": "2024-05-01T00:00:00Z"},
        ])
        docs = ingest(f)
        by_year_file = {}
        for d in docs:
            by_year_file.setdefault(d.file_id, []).append(d.metadata["uuid"])
        assert sorted(by_year_file["Journal-2018.json"]) == ["A1", "A2"]
        assert by_year_file["Journal.json"] == ["B1"]


# ── Case 2: Multi-year + undated ──────────────────────────────────────────────

class TestMultiYearWithUndated:
    def test_original_keeps_most_recent_year_plus_undated(self, tmp_path):
        f = _write_dayone(tmp_path, "Journal.json", [
            {"uuid": "A", "text": "old 2018", "creationDate": "2018-06-15T10:00:00Z"},
            {"uuid": "B", "text": "old 2020", "creationDate": "2020-01-01T00:00:00Z"},
            {"uuid": "C", "text": "current 2024", "creationDate": "2024-09-01T12:00:00Z"},
            {"uuid": "D", "text": "no date"},
            {"uuid": "E", "text": "garbage date", "creationDate": "not-a-date"},
        ])
        docs = ingest(f)
        by_file = {}
        for d in docs:
            by_file.setdefault(d.file_id, set()).add(d.metadata["uuid"])
        assert by_file == {
            "Journal.json": {"C", "D", "E"},
            "Journal-2018.json": {"A"},
            "Journal-2020.json": {"B"},
        }

    def test_undated_entries_have_no_origin_file_id(self, tmp_path):
        f = _write_dayone(tmp_path, "Journal.json", [
            {"uuid": "A", "text": "old", "creationDate": "2018-01-01T00:00:00Z"},
            {"uuid": "B", "text": "cur", "creationDate": "2024-01-01T00:00:00Z"},
            {"uuid": "C", "text": "no date"},
        ])
        docs = ingest(f)
        undated = next(d for d in docs if d.metadata["uuid"] == "C")
        assert undated.file_id == "Journal.json"
        assert "origin_file_id" not in undated.metadata


# ── Case 3: Single year only ──────────────────────────────────────────────────

class TestSingleYearOnly:
    def test_unchanged(self, tmp_path):
        f = _write_dayone(tmp_path, "Journal.json", [
            {"uuid": "A", "text": "x", "creationDate": "2024-01-01T00:00:00Z"},
            {"uuid": "B", "text": "y", "creationDate": "2024-12-31T00:00:00Z"},
        ])
        docs = ingest(f)
        assert {d.file_id for d in docs} == {"Journal.json"}
        for d in docs:
            assert "origin_file_id" not in d.metadata


# ── Case 4: Single year + undated (NEW — previously would have split) ────────

class TestSingleYearWithUndated:
    def test_original_keeps_both_no_splits(self, tmp_path):
        f = _write_dayone(tmp_path, "Journal.json", [
            {"uuid": "A", "text": "real 2024", "creationDate": "2024-06-15T00:00:00Z"},
            {"uuid": "B", "text": "no date"},
            {"uuid": "C", "text": "garbage", "creationDate": "xyz"},
        ])
        docs = ingest(f)
        assert {d.file_id for d in docs} == {"Journal.json"}
        assert {d.metadata["uuid"] for d in docs} == {"A", "B", "C"}
        for d in docs:
            assert "origin_file_id" not in d.metadata


# ── Case 5: All undated ───────────────────────────────────────────────────────

class TestAllUndated:
    def test_unchanged(self, tmp_path):
        f = _write_dayone(tmp_path, "Journal.json", [
            {"uuid": "A", "text": "no date"},
            {"uuid": "B", "text": "garbage", "creationDate": "xyz"},
        ])
        docs = ingest(f)
        assert {d.file_id for d in docs} == {"Journal.json"}
        assert {d.metadata["uuid"] for d in docs} == {"A", "B"}
        for d in docs:
            assert "origin_file_id" not in d.metadata


# ── Case 6: Only one historical year + undated ────────────────────────────────
#
# Spec contains both a prose rule ("most recent year present is the home year")
# and a 7-case table. The two conflict here: the prose rule says 2018 IS the
# max, so it stays in the original alongside undated. The table line for this
# case says the year splits and undated stays.
#
# We implement the prose rule (cleaner, deterministic, no calendar-year
# threshold) — see PR body for the full divergence note.

class TestOnlyHistoricalYearWithUndated:
    def test_prose_rule_keeps_both_in_original(self, tmp_path):
        f = _write_dayone(tmp_path, "Journal.json", [
            {"uuid": "A", "text": "old 2018", "creationDate": "2018-06-15T00:00:00Z"},
            {"uuid": "B", "text": "no date"},
        ])
        docs = ingest(f)
        assert {d.file_id for d in docs} == {"Journal.json"}
        assert {d.metadata["uuid"] for d in docs} == {"A", "B"}
        for d in docs:
            assert "origin_file_id" not in d.metadata


# ── Case 7: Year-boundary timestamps ──────────────────────────────────────────

class TestYearBoundary:
    def test_max_int_year_picker_handles_jan_1_boundary(self, tmp_path):
        f = _write_dayone(tmp_path, "Journal.json", [
            {"uuid": "A", "text": "last sec of 2017", "creationDate": "2017-12-31T23:59:59Z"},
            {"uuid": "B", "text": "first sec of 2018", "creationDate": "2018-01-01T00:00:00Z"},
        ])
        docs = ingest(f)
        by_file = {d.file_id: d.metadata["uuid"] for d in docs}
        assert by_file == {
            "Journal.json": "B",
            "Journal-2017.json": "A",
        }


# ── Other source types in same dir untouched ──────────────────────────────────

class TestOtherSourceTypesUntouched:
    def test_md_and_txt_in_dir_unaffected_by_dayone_split(self, tmp_path):
        (tmp_path / "notes.md").write_text("# Hello", encoding="utf-8")
        (tmp_path / "log.txt").write_text("plain text", encoding="utf-8")
        _write_dayone(tmp_path, "Journal.json", [
            {"uuid": "A", "text": "x", "creationDate": "2018-01-01T00:00:00Z"},
            {"uuid": "B", "text": "y", "creationDate": "2024-01-01T00:00:00Z"},
        ])
        docs = ingest(tmp_path)
        md_docs = [d for d in docs if d.source_type == SourceType.MD_FILE]
        txt_docs = [d for d in docs if d.source_type == SourceType.TXT]
        dayone_docs = [d for d in docs if d.source_type == SourceType.DAYONE_JSON]
        assert len(md_docs) == 1 and md_docs[0].file_id == "notes.md"
        assert len(txt_docs) == 1 and txt_docs[0].file_id == "log.txt"
        dayone_files = {d.file_id for d in dayone_docs}
        assert dayone_files == {"Journal.json", "Journal-2018.json"}
