"""
Tests for ingestor.py

Run from engine/:
    cd engine && pytest tests/
"""
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

from engine.ingestor import (
    SourceType,
    ingest,
    pipeline_stats_from_paths,
)
from engine.progress import estimate_pipeline

# Note: TestDetectType was removed when the explicit `_detect_type` helper
# was replaced by extension-driven dispatch in ingestor._ingest_file. The
# behaviors it covered (image/pdf/docx detection, html exclusion, etc.) are
# now exercised by the per-format tests below and by `ingest()` itself.


# ── WhatsApp ──────────────────────────────────────────────────────────────────

class TestWhatsApp:
    def test_single_doc_per_file(self, whatsapp_txt):
        docs = ingest(whatsapp_txt)
        assert len(docs) == 1
        assert docs[0].source_type == SourceType.WHATSAPP
        assert "Alice" in docs[0].content
        assert "Bob" in docs[0].content

    def test_media_omitted_stripped(self, whatsapp_txt):
        docs = ingest(whatsapp_txt)
        assert "<Media omitted>" not in docs[0].content

    def test_title_is_stem(self, whatsapp_txt):
        docs = ingest(whatsapp_txt)
        assert docs[0].title == "WhatsApp Chat - Alice"

    def test_zip_produces_docs(self, whatsapp_zip):
        docs = ingest(whatsapp_zip)
        assert len(docs) == 1
        assert docs[0].source_type == SourceType.WHATSAPP
        assert "Alice" in docs[0].content


# ── Plain text ────────────────────────────────────────────────────────────────

class TestTXT:
    def test_single_doc(self, txt_file):
        docs = ingest(txt_file)
        assert len(docs) == 1
        assert docs[0].source_type == SourceType.TXT

    def test_content_preserved(self, txt_file):
        docs = ingest(txt_file)
        assert "plain text" in docs[0].content

    def test_title_is_stem(self, txt_file):
        docs = ingest(txt_file)
        assert docs[0].title == "notes"

    def test_content_stripped(self, txt_file):
        docs = ingest(txt_file)
        assert docs[0].content == docs[0].content.strip()


# ── Notion MD ─────────────────────────────────────────────────────────────────

class TestNotionMD:
    def test_single_doc(self, notion_md_file):
        docs = ingest(notion_md_file)
        assert len(docs) == 1

    def test_source_type(self, notion_md_file):
        docs = ingest(notion_md_file)
        assert docs[0].source_type == SourceType.MD_FILE

    def test_notion_zip(self, notion_zip):
        docs = ingest(notion_zip)
        assert len(docs) == 2
        assert all(d.source_type == SourceType.MD_FILE for d in docs)


# ── Day One JSON ──────────────────────────────────────────────────────────────

class TestDayOneJSON:
    def test_one_doc_per_entry(self, dayone_json_file):
        docs = ingest(dayone_json_file)
        assert len(docs) == 2

    def test_dates_extracted(self, dayone_json_file):
        docs = ingest(dayone_json_file)
        dates = {d.date for d in docs}
        assert "2024-03-01" in dates
        assert "2024-03-02" in dates

    def test_tags_in_metadata(self, dayone_json_file):
        docs = ingest(dayone_json_file)
        tagged = next(d for d in docs if d.metadata.get("uuid") == "ABC123")
        # Tags are only included when non-empty
        assert "personal" in tagged.metadata["tags"]

    def test_empty_tags_omitted_from_metadata(self, dayone_json_file):
        docs = ingest(dayone_json_file)
        no_tags = next(d for d in docs if d.metadata.get("uuid") == "DEF456")
        assert "tags" not in no_tags.metadata

    def test_backslash_escapes_removed(self, dayone_json_file):
        docs = ingest(dayone_json_file)
        first = next(d for d in docs if d.metadata.get("uuid") == "ABC123")
        assert "\\" not in first.content

    def test_source_type(self, dayone_json_file):
        docs = ingest(dayone_json_file)
        assert all(d.source_type == SourceType.DAYONE_JSON for d in docs)

    def test_trailing_comma_json(self, tmp_path):
        bad_json = '{"entries": [{"uuid": "X1", "text": "hi", "creationDate": "2024-01-01T00:00:00Z", "tags": [],},]}'
        f = tmp_path / "bad.json"
        f.write_text(bad_json, encoding="utf-8")
        docs = ingest(f)
        assert len(docs) == 1
        assert docs[0].content == "hi"

    def test_timezone_near_midnight_uses_local_date(self, tmp_path):
        # 2024-01-05 23:30 EDT (UTC-5) = 2024-01-06T04:30Z; must date to Jan 5, not Jan 6.
        import json
        data = {"entries": [
            {"uuid": "TZ1", "text": "late night entry",
             "creationDate": "2024-01-06T04:30:00Z",
             "timeZone": "America/New_York"},
        ]}
        f = tmp_path / "tz_test.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        docs = ingest(f)
        assert len(docs) == 1
        assert docs[0].date == "2024-01-05"

    def test_missing_timezone_falls_back_to_utc(self, tmp_path):
        # No timeZoneName — should not crash, falls back to UTC slice.
        import json
        data = {"entries": [
            {"uuid": "TZ2", "text": "no tz entry",
             "creationDate": "2024-06-15T12:00:00Z"},
        ]}
        f = tmp_path / "no_tz.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        docs = ingest(f)
        assert len(docs) == 1
        assert docs[0].date == "2024-06-15"


# ── Plain MD file ─────────────────────────────────────────────────────────────

class TestMDFile:
    def test_single_doc(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Hello\nWorld", encoding="utf-8")
        docs = ingest(f)
        assert len(docs) == 1
        assert docs[0].source_type == SourceType.MD_FILE
        assert docs[0].title == "readme"
        assert "Hello" in docs[0].content


# ── PDF ───────────────────────────────────────────────────────────────────────

def _mock_reader(text_per_page: list[str]):
    mock_pages = []
    for text in text_per_page:
        p = MagicMock()
        p.extract_text.return_value = text
        mock_pages.append(p)
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = mock_pages
    return MagicMock(return_value=mock_reader_instance)


# `_parse_pdf`/`_parse_docx` import their parser lazily (`from pypdf import
# PdfReader` / `from docx import Document`) inside the function, so the patch
# target is the source module, not an `ingestor.*` attribute (which doesn't
# exist — patching it AttributeError'd, and left a stray attr that leaked
# across tests).
class TestPDF:
    def test_pdf_parsed(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 fake")
        with patch("pypdf.PdfReader", _mock_reader(["Page content here."])):
            docs = ingest(f)
        assert len(docs) == 1
        assert docs[0].source_type == SourceType.PDF
        assert "Page content here." in docs[0].content

    def test_pdf_no_text_is_skipped(self, tmp_path):
        f = tmp_path / "scanned.pdf"
        f.write_bytes(b"%PDF-1.4 fake")
        with patch("pypdf.PdfReader", _mock_reader([""])):
            docs = ingest(f)
        assert docs == []

    def test_pdf_multiple_pages_joined(self, tmp_path):
        f = tmp_path / "long.pdf"
        f.write_bytes(b"%PDF-1.4 fake")
        with patch("pypdf.PdfReader", _mock_reader(["Page one.", "Page two."])):
            docs = ingest(f)
        assert "Page one." in docs[0].content
        assert "Page two." in docs[0].content


# Real text-PDF round-trip via the actual pypdf parser (no mocks).
# Guards against pypdf disappearing from the bundled venv (was a silent
# ingest skip in run m6xt). Fixture is a hand-built minimal PDF checked
# in under tests/fixtures/ — kept tiny and non-private so it can ship
# in the repo without tripping the *.pdf gitignore.
_PDF_FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


class TestPDFRealFile:
    def test_real_pdf_yields_nonempty_doc(self):
        docs = ingest(_PDF_FIXTURE)
        assert len(docs) == 1
        assert docs[0].source_type == SourceType.PDF
        assert "Sample PDF Fixture" in docs[0].content


# ── DOCX ──────────────────────────────────────────────────────────────────────

def _mock_docx_doc(paragraphs: list[str]):
    mock_paras = []
    for text in paragraphs:
        p = MagicMock()
        p.text = text
        mock_paras.append(p)
    mock_doc_instance = MagicMock()
    mock_doc_instance.paragraphs = mock_paras
    return MagicMock(return_value=mock_doc_instance)


class TestDOCX:
    def test_docx_parsed(self, tmp_path):
        f = tmp_path / "doc.docx"
        f.write_bytes(b"PK fake")
        with patch("docx.Document", _mock_docx_doc(["First paragraph.", "", "Third paragraph."])):
            docs = ingest(f)
        assert len(docs) == 1
        assert docs[0].source_type == SourceType.DOCX
        assert "First paragraph." in docs[0].content
        assert "Third paragraph." in docs[0].content
        assert "\n\n\n" not in docs[0].content

    def test_docx_title_is_stem(self, tmp_path):
        f = tmp_path / "my_doc.docx"
        f.write_bytes(b"PK fake")
        with patch("docx.Document", _mock_docx_doc(["Content."])):
            docs = ingest(f)
        assert docs[0].title == "my_doc"


# ── Image / vision ────────────────────────────────────────────────────────────
#
# `_parse_image` no longer transcribes synchronously. It emits a placeholder
# Document tagged `_pending_vision`; the runner's dedicated vision stage
# (`vision.describe_images_all`) does the actual transcription and drops
# vision-failed docs. So the ingestor's contract here is *the placeholder*,
# not extracted text — transcription + empty-skip coverage lives in
# test_describe_images_all.py and the runner vision-stage path.

class TestImage:
    def test_image_yields_pending_vision_placeholder(self, tmp_path):
        f = tmp_path / "screenshot.png"
        f.write_bytes(b"\x89PNG fake")
        docs = ingest(f)
        assert len(docs) == 1
        assert docs[0].source_type == SourceType.IMAGE
        # Content is empty at ingest; the vision stage fills it later.
        assert docs[0].content == ""
        assert docs[0].metadata.get("_pending_vision") is True
        assert "_vision_mode" in docs[0].metadata



# ── Unsupported ───────────────────────────────────────────────────────────────

class TestUnsupported:
    def test_html_skipped(self, html_file):
        docs = ingest(html_file)
        assert docs == []


# ── Directory ingestion ───────────────────────────────────────────────────────

class TestDirectoryIngest:
    def test_recurses_into_subdirs(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "notes.txt").write_text("note one", encoding="utf-8")
        (sub / "more.txt").write_text("note two", encoding="utf-8")
        docs = ingest(tmp_path)
        assert len(docs) == 2
        assert all(d.source_type == SourceType.TXT for d in docs)

    def test_mixed_formats(self, tmp_path):
        (tmp_path / "notes.md").write_text("# Hello", encoding="utf-8")
        (tmp_path / "export.txt").write_text("some text", encoding="utf-8")
        docs = ingest(tmp_path)
        types = {d.source_type for d in docs}
        assert SourceType.MD_FILE in types
        assert SourceType.TXT in types

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            ingest("/tmp/does_not_exist_xyz_abc_123")


# ── Document fields ───────────────────────────────────────────────────────────

class TestDocumentFields:
    def test_id_is_unique_across_json_entries(self, dayone_json_file):
        docs = ingest(dayone_json_file)
        ids = [d.id for d in docs]
        assert len(ids) == len(set(ids))

    def test_source_path_is_string(self, notion_md_file):
        docs = ingest(notion_md_file)
        assert isinstance(docs[0].source_path, str)

    def test_content_stripped(self, txt_file):
        docs = ingest(txt_file)
        for d in docs:
            assert d.content == d.content.strip()


# ── pipeline_stats_from_paths ───────────────────────────────────────────────────────────


class TestPreScanInputs:
    """`pipeline_stats_from_paths` walks paths cheaply (stat + zip namelist) and
    returns rough per-stage call estimates BEFORE ingest fires. The
    runner uses these to register the progress tracker's stages so
    the bar's denominator covers the whole pipeline from the first
    LLM call (no 99% cap-trip when vision finishes; no jump from
    `total=vision_count` to `total=full_pipeline` post-ingest)."""

    def test_counts_images_with_real_size(self, tmp_path):
        for ext in (".png", ".jpg", ".heic", ".webp", ".gif"):
            (tmp_path / f"img{ext}").write_bytes(b"\x00" * 100)
        r = pipeline_stats_from_paths([str(tmp_path)])
        assert r["n_images"] == 5
        # 10kB per image — flat estimate, real file size doesn't
        # reflect post-vision transcript size.
        assert r["text_bytes"] == 5 * 10_000

    def test_text_files_use_real_size(self, tmp_path):
        (tmp_path / "notes.md").write_text("hello " * 1000)  # 6kB
        (tmp_path / "log.txt").write_text("x" * 2000)        # 2kB
        r = pipeline_stats_from_paths([str(tmp_path)])
        assert r["n_text_files"] == 2
        # text_bytes = sum of real on-disk sizes for text formats.
        assert r["text_bytes"] == 6000 + 2000

    def test_pdf_docx_get_flat_10kb_estimate_not_real_size(
            self, tmp_path):
        """PDFs / DOCXes have on-disk size dominated by formatting,
        not text content. A 50MB PDF might extract to 10kB of text.
        Pre-scan uses the flat 10kB estimate to avoid blowing up the
        extract chunk-count estimate by 50×."""
        (tmp_path / "big.pdf").write_bytes(b"\x00" * 5_000_000)   # 5MB pdf
        (tmp_path / "doc.docx").write_bytes(b"\x00" * 1_000_000)  # 1MB docx
        r = pipeline_stats_from_paths([str(tmp_path)])
        # Both files counted as text-bearing (will produce Documents).
        assert r["n_text_files"] == 2
        # 10kB each, flat — not the 6MB on-disk size.
        assert r["text_bytes"] == 2 * 10_000

    def test_explicit_file_list_does_not_recurse_dirs(self, tmp_path):
        """When the user picks individual files via the picker, paths
        is a list of file paths. Pre-scan should walk just those, not
        their containing directory."""
        d = tmp_path / "subdir"
        d.mkdir()
        (d / "extra.txt").write_text("not picked")  # in dir but not selected
        f1 = tmp_path / "a.md"
        f1.write_text("aaa")
        f2 = tmp_path / "b.png"
        f2.write_bytes(b"\x00" * 100)
        r = pipeline_stats_from_paths([str(f1), str(f2)])
        assert r["n_images"] == 1
        assert r["n_text_files"] == 1
        # text_bytes = 3 (real "aaa" md content) + 10_000 (image flat).
        # The .txt file in the subdirectory is NOT counted (we passed
        # explicit file paths, not the parent dir).
        assert r["text_bytes"] == 3 + 10_000

    def test_zip_namelist_peek(self, tmp_path):
        """Zips: peek namelist + member sizes without extracting.
        Inner images count toward n_images; inner pdfs/docxes get
        the flat 10kB estimate; inner text uses member.file_size."""
        import zipfile
        zpath = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("photo.jpg", b"\x00" * 100)
            zf.writestr("notes.md", "hello world")  # 11 bytes
            zf.writestr("doc.pdf", b"\x00" * 100_000)  # 100kB pdf
        r = pipeline_stats_from_paths([str(zpath)])
        assert r["n_images"] == 1
        # 1 md + 1 pdf
        assert r["n_text_files"] == 2
        # text_bytes folds all three into one bucket: 11 (md) +
        # 10_000 (image flat) + 10_000 (pdf flat).
        assert r["text_bytes"] == 11 + 10_000 + 10_000

    def test_missing_path_skipped_gracefully(self):
        r = pipeline_stats_from_paths(["/nonexistent/does/not/exist"])
        assert r["n_images"] == 0
        assert r["n_text_files"] == 0
        assert r["text_bytes"] == 0
        assert r["n_splits"] is None

    def test_user_estimate_realistic_for_10_files(self, tmp_path):
        """User complaint: 10 files (5 images + 5 small text/pdf)
        showed `total=16` on the bar — vision (5) + extract (1) +
        sum of others = 16. They expected ≥15 just for vision and
        extract. Post-fix: extract is FLOORED at n_files=10, so the
        total reads ~25+ from the start."""
        for i in range(5):
            (tmp_path / f"photo{i}.png").write_bytes(b"\x00" * 100)
        (tmp_path / "n1.md").write_text("a" * 500)
        (tmp_path / "n2.md").write_text("b" * 500)
        (tmp_path / "n3.txt").write_text("c" * 500)
        (tmp_path / "doc.pdf").write_bytes(b"\x00" * 5_000_000)
        (tmp_path / "j.json").write_text("{}")
        pre = pipeline_stats_from_paths([str(tmp_path)])
        est = estimate_pipeline(pre).calls_per_stage
        assert est["vision"] == 5
        assert est["extract"] >= 10  # n_files floor: 5 images + 5 text
        # Total includes vision + extract + entities + dedupe + patterns + insights + actions
        total = sum(est.values())
        assert total >= 25, f"expected total >= 25, got {total}: {est}"

    def test_user_complaint_10_files_5_images_5_text(self, tmp_path):
        """The exact case the user reported: 10 files, 5 images +
        5 text/pdf. Pre-fix: extract estimate hit 531 (using raw
        file sizes including bloated PDFs) or 1 (when text was
        small). Post-fix: image/pdf bytes are flat 10kB, and the
        runner floors extract at n_files."""
        for i in range(5):
            (tmp_path / f"photo{i}.png").write_bytes(b"\x00" * 100)
        (tmp_path / "n1.md").write_text("hello")
        (tmp_path / "n2.md").write_text("world")
        (tmp_path / "n3.txt").write_text("foo")
        (tmp_path / "doc.pdf").write_bytes(b"\x00" * 5_000_000)  # 5MB pdf
        (tmp_path / "j.json").write_text("{}")
        r = pipeline_stats_from_paths([str(tmp_path)])
        assert r["n_images"] == 5
        assert r["n_text_files"] == 5
        # text_bytes = 5 images × 10kB + 1 pdf × 10kB + tiny text
        # files (~16 chars). Sane, not 5MB+ from raw PDF size.
        assert 60_000 <= r["text_bytes"] <= 60_100


class TestEstimatePipeline:
    """`estimate_pipeline` translates a `PipelineStats` dict into a
    `PipelineEstimate` (call counts + cold-cache seconds + totals +
    percentage helper). Same function used by:
      - the preflight tick (before ProgressTracker exists)
      - the ingest tracker registration (registers stages with the
        same `calls_per_stage` dict, no recomputation)
      - the post-ingest re-estimate (real Document stats via
        `pipeline_stats_from_docs`)
    ONE estimator. Two stats producers."""

    def test_extract_floored_at_n_files(self):
        # Tiny inputs: token-based extract estimate would be 1, but
        # n_files (10) is the floor. Without this, a 10-file run
        # reads "5 vision + 1 extract + ... = 16" mid-run instead
        # of "5 + 10 + ... = 25+".
        stats = {
            "n_images": 5, "n_text_files": 5,
            "text_bytes": 50_000,  # 5 images × 10kB
        }
        est = estimate_pipeline(stats).calls_per_stage
        assert est["vision"] == 5
        assert est["extract"] == 10
        assert sum(est.values()) >= 20

    def test_extract_grows_with_input_tokens_above_floor(self):
        # 5MB of text → ~1.6M tokens → 50+ chunks at 30k cap. The
        # n_files floor is irrelevant when tokens dominate.
        scan = {
            "n_images": 0, "n_text_files": 1,
            "text_bytes": 5_000_000,
            "chunk_cap": 30_000,
        }
        est = estimate_pipeline(scan).calls_per_stage
        assert est["extract"] >= 50

    def test_no_input_files_minimal_estimate(self):
        pre = {
            "n_images": 0, "n_text_files": 0,
            "text_bytes": 0,
        }
        est = estimate_pipeline(pre).calls_per_stage
        assert est["vision"] == 0
        assert est["extract"] == 1  # min floor when no files
        # Other stages have their own floors.
        assert est["entities"] >= 2
        assert est["entities_dedupe"] == 1

    def test_entities_floored_at_2(self):
        # Small input → naive sqrt(n_groups) = 1, but real runs
        # always fire ≥2 entity batches. Floor at 2.
        pre = {
            "n_images": 0, "n_text_files": 1,
            "text_bytes": 200,
        }
        est = estimate_pipeline(pre).calls_per_stage
        assert est["entities"] >= 2

    def test_n_topics_drives_patterns_estimate(self):
        scan = {
            "n_images": 1, "n_text_files": 1,
            "text_bytes": 11_000,  # 1kB md + 10kB image flat
            "n_topics": 12,
        }
        est = estimate_pipeline(scan).calls_per_stage
        assert est["patterns"] == 12

    def test_real_n_splits_wins_over_token_heuristic(self):
        """Post-ingest path: caller knows the real splitter chunk
        count. The estimator uses it directly, ignoring the
        token-based ceiling."""
        scan = {
            "n_images": 0, "n_text_files": 100,
            "text_bytes": 1_000_000,
            "chunk_cap": 30_000,
            "n_splits": 7,  # real splitter output
        }
        est = estimate_pipeline(scan).calls_per_stage
        assert est["extract"] == 7  # real wins, not token-based ceiling

    def test_historical_durations_drive_per_stage_seconds(self):
        """Issue #394 fix: when caller passes historical_durations +
        stage_model_map, per-stage seconds reflect the same token-aware
        decomposition the live tracker uses (not FALLBACK constants).
        Pin the directional invariant: a model with much faster
        historical pace than FALLBACK produces a smaller preflight
        eta; a model with much slower pace produces a larger one."""
        from engine.progress import FALLBACK_SECONDS_PER_CALL
        stats = {
            "n_images": 0, "n_text_files": 1,
            "text_bytes": 50_000,
            "n_splits": 1,
            "n_topics": 1,
        }
        # Slow-model historical for entities: 200s per call, 5000 tokens.
        slow_hist = {("entities", "slow-model"): [(200.0, 5000)] * 5}
        slow_map = {s: "slow-model" for s in (
            "vision","extract","entities","entities_dedupe",
            "patterns","insights","actions")}
        # Fast-model historical for entities: 5s per call, 5000 tokens.
        fast_hist = {("entities", "fast-model"): [(5.0, 5000)] * 5}
        fast_map = {s: "fast-model" for s in slow_map}

        baseline = estimate_pipeline(stats, is_local=True)  # FALLBACK
        slow = estimate_pipeline(
            stats, is_local=True,
            historical_durations=slow_hist, stage_model_map=slow_map)
        fast = estimate_pipeline(
            stats, is_local=True,
            historical_durations=fast_hist, stage_model_map=fast_map)

        # Slow-model historical (200s/call) inflates entities seconds
        # vs FALLBACK-based baseline (22s/call).
        assert slow.seconds_per_stage["entities"] > baseline.seconds_per_stage["entities"]
        # Fast-model historical (5s/call) shrinks entities seconds vs
        # FALLBACK-based baseline.
        assert fast.seconds_per_stage["entities"] < baseline.seconds_per_stage["entities"]
        # And stages without matching historicals still use FALLBACK
        # (legacy preflight behavior unchanged for fresh-install paths).
        assert slow.seconds_per_stage["extract"] == baseline.seconds_per_stage["extract"]
        # FALLBACK constant is per-call — silence the unused-import check.
        assert FALLBACK_SECONDS_PER_CALL["entities"] > 0

    def test_historical_param_optional_preserves_legacy_behavior(self):
        """Default call site (no historical_durations) gets the exact
        same numbers it always did. Callers that don't yet wire the
        historicals through stay on the FALLBACK-based estimate."""
        stats = {
            "n_images": 0, "n_text_files": 1,
            "text_bytes": 30_000,
            "n_splits": 2,
            "n_topics": 4,
        }
        legacy = estimate_pipeline(stats, is_local=False)
        explicit_none = estimate_pipeline(
            stats, is_local=False,
            historical_durations=None, stage_model_map=None)
        assert legacy.total_seconds == explicit_none.total_seconds
        assert legacy.calls_per_stage == explicit_none.calls_per_stage

    # ── Cold-cache ETA + totals + percentage on PipelineEstimate ─────

    def test_estimate_carries_per_stage_seconds_and_totals(self):
        # For 1-call-per-stage with parallelism=1 (local) and tail=1.0
        # the per-stage seconds reduces to FALLBACK_SECONDS_PER_CALL.
        from engine.progress import FALLBACK_SECONDS_PER_CALL
        stats = {
            "n_images": 1, "n_text_files": 1,
            "text_bytes": 11_000,
            "n_splits": 1,
            "n_topics": 1,
        }
        est = estimate_pipeline(stats, is_local=True)
        # Vision + extract + entities (≥2) + dedupe + patterns(=1) +
        # insights + actions. Local parallelism=1 → seconds_per_stage
        # equals FALLBACK × n_calls per stage.
        assert est.seconds_per_stage["vision"] == pytest.approx(
            FALLBACK_SECONDS_PER_CALL["vision"])
        assert est.seconds_per_stage["actions"] == pytest.approx(
            FALLBACK_SECONDS_PER_CALL["actions"])
        assert est.total_calls == sum(est.calls_per_stage.values())
        assert est.total_seconds == pytest.approx(
            sum(est.seconds_per_stage.values()))

    def test_cloud_parallelism_and_tail_shrink_seconds(self):
        # 16 extract calls at parallelism=16 with tail=2 → 1 batch ×
        # 68s × 2.0 = 136s. Local mode same calls would be 16 × 68 ×
        # 1.0 = 1088s. Cloud mode strictly shorter.
        stats = {
            "n_images": 0, "n_text_files": 0,
            "text_bytes": 0,
            "n_splits": 16,
        }
        cloud = estimate_pipeline(stats, is_local=False)
        local = estimate_pipeline(stats, is_local=True)
        assert cloud.seconds_per_stage["extract"] < (
            local.seconds_per_stage["extract"])

    def test_issue_238_repro_eta_clears_5s_floor(self):
        # Issue #238: 32 calls returned `<5s remaining` because the
        # preflight emit hardcoded eta_seconds=0. Cold-cache estimate
        # for the reported shape must be well above the UI's 5-second
        # bucket so the chip reports a sensible figure.
        stats = {
            "n_images": 5, "n_text_files": 5,
            "text_bytes": 60_000,
            "n_topics": 6,
        }
        est = estimate_pipeline(stats, is_local=False)
        assert est.total_calls >= 25
        assert est.total_seconds > 60.0, (
            f"preflight ETA collapsed to {est.total_seconds:.1f}s")


    def test_embeddings_est_is_one_collective_unit(self):
        """Issue #581: the embeddings stage's many sub-second batched
        wire calls count as ONE collective progress unit, NOT a
        fact-scaled fan-out. It must stay flat at 1 regardless of input
        size — a fact-scaled estimate balloons the bar's denominator,
        holding it back through the long stages then lurching when the
        ~2s embed burst lands. The REAL batch count is restored on
        completion via `mark_stage_finished` (covered in
        test_progress.py), not in the pre-run estimate."""
        small = estimate_pipeline(
            {"n_images": 0, "n_text_files": 1, "text_bytes": 1_000}
        ).calls_per_stage["embeddings"]
        big = estimate_pipeline(
            {"n_images": 0, "n_text_files": 1, "text_bytes": 1_000_000}
        ).calls_per_stage["embeddings"]
        assert small == 1
        assert big == 1  # does NOT widen with input size anymore

    def test_embeddings_est_one_for_empty_input(self):
        """No input → the embeddings stage still registers its one
        collective unit (issue #581) so it has a denominator slot on
        cold starts."""
        est = estimate_pipeline(
            {"n_images": 0, "n_text_files": 0, "text_bytes": 0}
        ).calls_per_stage
        assert est["embeddings"] == 1

    def test_pipeline_stats_from_docs_builds_dict_from_documents(self):
        """`pipeline_stats_from_docs` is the post-ingest companion of
        `pipeline_stats_from_paths`. It accepts parsed Documents and returns
        a scan dict in the same shape so `estimate_pipeline`
        can consume both pre- and post-ingest paths uniformly."""
        from engine.ingestor import pipeline_stats_from_docs, Document, SourceType
        docs = [
            Document(id="d1", source_path="img1.png",
                     source_type=SourceType.IMAGE, content="transcript text"),
            Document(id="d2", source_path="notes.md",
                     source_type=SourceType.MD_FILE, content="hello world"),
        ]
        scan = pipeline_stats_from_docs(docs, n_splits=3, chunk_cap=20_000, n_topics=4)
        assert scan["n_images"] == 1
        assert scan["n_text_files"] == 1
        assert scan["n_splits"] == 3
        assert scan["chunk_cap"] == 20_000
        assert scan["n_topics"] == 4
        # text_bytes = sum of ALL doc content lengths (image
        # transcript + text-file content).
        assert scan["text_bytes"] == len("transcript text") + len("hello world")
