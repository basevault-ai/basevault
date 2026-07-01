"""
Preprocessor — writes each ingested Document's content as plain markdown.

By the time a Document reaches this stage, its `content` field is already
the canonical text for that file (pypdf output for PDFs, docx paragraphs,
vision transcript for images, cleaned text for Day One JSON, etc.). The
preprocessor's job is to snapshot that text to a known path so that:

  1. Every downstream offset (EvidenceSpan.file_offset/file_length) refers
     to a byte position in a real on-disk file. Users can open the
     preprocessed markdown and jump to that offset directly.
  2. The vault exporter can copy the same text into 0-inputs/ and annotate
     it with footnote backlinks to extracted items.

The preprocessor does NOT run LLM calls. Image vision transcription
already happened in the ingestor. The preprocessor only handles the
final "save to disk as markdown" step.

Location: logs/<ts>/<run-name>/stages/00-ingestion/documents/<file_id>.md

Usage:
    from engine.preprocessor import preprocess
    preprocess(docs, stage_dir)  # stage_dir = run_dir/stages/00-ingestion
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from engine.ingestor import Document


@dataclass
class PreprocessedFile:
    """One preprocessed markdown file on disk."""
    file_id: str
    path: Path
    size: int


def preprocess(docs: list[Document], stage_dir: Path) -> list[PreprocessedFile]:
    """
    Write each unique file_id's preprocessed content to a markdown file
    under `stage_dir/documents/` (where stage_dir is the ingestion stage
    dir, typically `run_dir/stages/00-ingestion`).

    Multiple Documents can share a file_id (e.g. Day One JSON with many
    entries, each becoming its own Document with the same file_id plus a
    uuid suffix on id). For preprocessing we group by file_id and write
    one markdown file per source file.

    Returns a list of PreprocessedFile records for the vault exporter.
    """
    out_root = stage_dir / "documents"
    out_root.mkdir(parents=True, exist_ok=True)

    # Group documents by their file_id (the stable short identifier).
    by_file: dict[str, list[Document]] = {}
    for d in docs:
        fid = d.file_id or d.id
        by_file.setdefault(fid, []).append(d)

    results: list[PreprocessedFile] = []
    for file_id, group in by_file.items():
        content = _merge_documents(group)
        out_path = out_root / f"{file_id}.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        results.append(PreprocessedFile(
            file_id=file_id,
            path=out_path,
            size=len(content.encode("utf-8")),
        ))

    return results


def _merge_documents(group: list[Document]) -> str:
    """
    Produce the preprocessed markdown body for a group of Documents that
    share a file_id. Mutates each Document's `origin_char` in-place so
    downstream offset math (doc.origin_char + section.origin_char + ...)
    resolves to the right byte position in the merged file.

    For plain-text-like sources there is one Document per file and its
    origin_char stays at 0. For Day One JSON the ingestor emits one
    Document per entry and we concatenate them with date headings; each
    Document's origin_char becomes its start position in the merged file.
    """
    if len(group) == 1:
        body = group[0].content.rstrip() + "\n"
        group[0].origin_char = 0
        return body

    buf: list[str] = []
    cursor = 0
    for d in group:
        heading = d.title or d.id
        if d.date:
            heading = f"{heading} ({d.date})"
        prefix = f"## {heading}\n\n"
        buf.append(prefix)
        cursor += len(prefix)

        # The Document's content starts here in the merged file.
        d.origin_char = cursor

        suffix = d.content.rstrip() + "\n\n"
        buf.append(suffix)
        cursor += len(suffix)

    return "".join(buf).rstrip() + "\n"
