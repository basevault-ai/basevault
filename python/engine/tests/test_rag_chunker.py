"""Unit tests for the RAG retrieval chunker."""
from __future__ import annotations

from engine.ingestor import Document, SourceType
from engine.rag_chunker import (
    CHUNK_TOKENS,
    OVERLAP_TOKENS,
    chunk_document,
    chunk_documents,
)
from engine.tokens import CHARS_PER_TOKEN, count_tokens


def _doc(content: str, file_id: str = "f1", source_path: str = "/tmp/f1.md") -> Document:
    return Document(
        id=file_id,
        source_path=source_path,
        source_type=SourceType.MD_FILE,
        content=content,
        file_id=file_id,
    )


def test_short_doc_emits_single_chunk_with_empty_section_path():
    doc = _doc("Short body that fits in one chunk.")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Short body that fits in one chunk."
    assert chunks[0].section_path == ()
    assert chunks[0].file_id == "f1"
    assert chunks[0].char_offset == 0


def test_blank_doc_emits_no_chunks():
    doc = _doc("\n\n  \n")
    assert chunk_document(doc) == []


_PAD = "lorem ipsum dolor sit amet " * 80


def test_h1_h2_h3_splits_each_become_their_own_chunk():
    text = (
        "Preamble. " + _PAD + "\n\n"
        "# Top\n\nBody of top. " + _PAD + "\n\n"
        "## Mid\n\nBody of mid. " + _PAD + "\n\n"
        "### Leaf\n\nBody of leaf. " + _PAD + "\n\n"
        "## Another mid\n\nMore. " + _PAD + "\n"
    )
    doc = _doc(text)
    chunks = chunk_document(doc)
    section_paths = [c.section_path for c in chunks]
    assert () in section_paths
    assert ("Top",) in section_paths
    assert ("Top", "Mid") in section_paths
    assert ("Top", "Mid", "Leaf") in section_paths
    assert ("Top", "Another mid") in section_paths


def test_section_path_pops_at_same_or_shallower_level():
    text = (
        "# A\n\nbody a " + _PAD + "\n\n"
        "## A1\n\nbody a1 " + _PAD + "\n\n"
        "# B\n\nbody b " + _PAD + "\n\n"
        "## B1\n\nbody b1 " + _PAD + "\n\n"
        "### B1a\n\nbody b1a " + _PAD + "\n\n"
        "## B2\n\nbody b2 " + _PAD + "\n"
    )
    doc = _doc(text)
    section_paths = [c.section_path for c in chunk_document(doc)]
    assert ("A",) in section_paths
    assert ("A", "A1") in section_paths
    assert ("B",) in section_paths
    assert ("B", "B1") in section_paths
    assert ("B", "B1", "B1a") in section_paths
    assert ("B", "B2") in section_paths


def test_small_doc_with_headers_emits_single_whole_chunk():
    # File-level fits CHUNK_TOKENS → emit as one chunk regardless of
    # H1/H2/H3 boundaries inside. Section breakdown kicks in only when
    # the whole-file shortcut overflows.
    text = (
        "Pre.\n\n"
        "# Top\n\nshort top body.\n\n"
        "## Sub\n\nshort sub body.\n"
    )
    chunks = chunk_document(_doc(text))
    assert len(chunks) == 1
    assert chunks[0].section_path == ()


def test_oversized_section_slides_with_overlap():
    long_para = "x " * (CHUNK_TOKENS * CHARS_PER_TOKEN * 3 // 2)
    text = f"# Big\n\n{long_para}"
    doc = _doc(text)
    chunks = chunk_document(doc)
    big_chunks = [c for c in chunks if c.section_path == ("Big",)]
    assert len(big_chunks) >= 2
    for c in big_chunks:
        assert count_tokens(c.text) <= CHUNK_TOKENS + OVERLAP_TOKENS


def test_oversized_section_overlap_is_nonzero_between_consecutive_chunks():
    body = " ".join(f"word{i:05d}" for i in range(CHUNK_TOKENS * 2))
    text = f"# Big\n\n{body}"
    doc = _doc(text)
    chunks = [c for c in chunk_document(doc) if c.section_path == ("Big",)]
    assert len(chunks) >= 2
    # Stride is CHUNK_TOKENS - OVERLAP_TOKENS; consecutive char_offsets
    # should differ by exactly that × CHARS_PER_TOKEN.
    expected_stride = (CHUNK_TOKENS - OVERLAP_TOKENS) * CHARS_PER_TOKEN
    assert chunks[1].char_offset - chunks[0].char_offset == expected_stride


def test_char_offset_anchors_to_parent_origin_for_subdoc():
    doc = Document(
        id="sub", source_path="/tmp/parent.md",
        source_type=SourceType.MD_FILE,
        content="# H\n\nbody\n",
        file_id="parent",
        origin_char=1234,
    )
    chunks = chunk_document(doc)
    assert all(c.char_offset >= 1234 for c in chunks)
    assert all(c.file_id == "parent" for c in chunks)


def test_chunk_documents_concatenates_across_docs():
    docs = [_doc("alpha", "a"), _doc("beta", "b"), _doc("", "c")]
    chunks = chunk_documents(docs)
    file_ids = [c.file_id for c in chunks]
    assert file_ids == ["a", "b"]


def test_empty_input_list_returns_empty():
    assert chunk_documents([]) == []
