"""
Hierarchical retrieval chunker for the Embeddings stage.

Different consumer than `splitter.py`: the splitter sizes for LLM call
budgets (large windows, no overlap, optional batching of small atoms);
this chunker sizes for retrieval (smaller overlapping windows so a
question targeting one paragraph doesn't have to embed a whole file).

Strategy, per spec § RAG enhancements:

    Document → H1/H2/H3 markdown sections → sliding token window
                                            (CHUNK_TOKENS with
                                             OVERLAP_TOKENS overlap)

Sections that already fit `CHUNK_TOKENS` are emitted whole; sections
that don't are sliced into overlapping windows. Header ancestry is
recorded on each chunk (`section_path`) so PR 4.2's enrichment step
can prepend the breadcrumb without re-parsing the source. The embed
input for *this* PR is the bare chunk text — no breadcrumb, no graph.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from engine.ingestor import Document
from engine.tokens import CHARS_PER_TOKEN, count_tokens


CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64


@dataclass(frozen=True)
class RagChunk:
    text: str
    file_id: str
    source_path: str
    char_offset: int
    section_path: tuple[str, ...]
    # ISO YYYY-MM-DD if the source document carries one (Day One entry
    # date, file-frontmatter date, etc.); empty when unknown. Threaded
    # into the embeddings record's ``extra.file_date`` so the chatbot
    # dispatcher can sort facts/chunks by recency without re-reading
    # the source docs.
    date: str = ""


_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)


def _section_spans(text: str) -> list[tuple[int, int, tuple[str, ...]]]:
    """Walk H1/H2/H3 headers and return (start, end, section_path) for
    each section. Pre-header content (if any) is the first span with an
    empty `section_path`. Section paths track ancestor headers
    outermost-first so PR 4.2 can render breadcrumbs without re-parsing.
    """
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [(0, len(text), ())]

    spans: list[tuple[int, int, tuple[str, ...]]] = []
    if matches[0].start() > 0:
        spans.append((0, matches[0].start(), ()))

    stack: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        ancestors = tuple(t for _, t in stack)
        section_path = ancestors + (title,)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append((start, end, section_path))
        stack.append((level, title))
    return spans


def _slide(
    text: str,
    char_base: int,
    doc: Document,
    section_path: tuple[str, ...],
) -> list[RagChunk]:
    """Slide a CHUNK_TOKENS window with OVERLAP_TOKENS overlap across
    `text`. Stride is CHUNK_TOKENS - OVERLAP_TOKENS; tail-window emits
    once even if shorter. Sub-token math runs in char units (single
    global CHARS_PER_TOKEN) — same approximation the splitter uses.
    """
    size_chars = CHUNK_TOKENS * CHARS_PER_TOKEN
    stride_chars = (CHUNK_TOKENS - OVERLAP_TOKENS) * CHARS_PER_TOKEN

    chunks: list[RagChunk] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size_chars, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(RagChunk(
                text=piece,
                file_id=doc.file_id or doc.id,
                source_path=doc.source_path,
                char_offset=doc.origin_char + char_base + start,
                section_path=section_path,
                date=doc.date or "",
            ))
        if end >= n:
            break
        start += stride_chars
    return chunks


def chunk_document(doc: Document) -> list[RagChunk]:
    """Hierarchical chunker for a single ingested Document. Whole-doc
    when it fits CHUNK_TOKENS; otherwise H1/H2/H3 split, with each
    oversized section subdivided by sliding window."""
    text = doc.content
    if not text.strip():
        return []
    if count_tokens(text) <= CHUNK_TOKENS:
        return [RagChunk(
            text=text.strip(),
            file_id=doc.file_id or doc.id,
            source_path=doc.source_path,
            char_offset=doc.origin_char,
            section_path=(),
            date=doc.date or "",
        )]

    chunks: list[RagChunk] = []
    for start, end, section_path in _section_spans(text):
        body = text[start:end]
        if not body.strip():
            continue
        if count_tokens(body) <= CHUNK_TOKENS:
            chunks.append(RagChunk(
                text=body.strip(),
                file_id=doc.file_id or doc.id,
                source_path=doc.source_path,
                char_offset=doc.origin_char + start,
                section_path=section_path,
                date=doc.date or "",
            ))
        else:
            chunks.extend(_slide(body, start, doc, section_path))
    return chunks


def chunk_documents(docs: list[Document]) -> list[RagChunk]:
    out: list[RagChunk] = []
    for d in docs:
        out.extend(chunk_document(d))
    return out
