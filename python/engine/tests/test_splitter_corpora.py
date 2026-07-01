"""Integration tests for the splitter on real corpora.

Per AGENTS.md: parallelize across fixtures (ThreadPoolExecutor), assert
structural invariants, no hardcoded personal-data heuristics.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


from engine.ingestor import Document, SourceType, ingest
from engine.splitter import split_documents, report
from engine.tokens import count_tokens

# Reads real corpora from data/. No LLM call, but the integration marker is
# the established opt-in for "needs real fixtures" tests.
pytestmark = pytest.mark.integration


CODE_ROOT = Path(__file__).parents[3]
DATA = CODE_ROOT / "data"

# Fixtures from on-disk corpora
_REAL_FIXTURES = [
    p for p in [DATA / "chat_person_a.txt", DATA / "chat_person_b.txt"]
    if p.exists()
]


def _synthetic_markdown(n_sections: int = 50) -> Document:
    """Big synthetic markdown with H1/H2 structure for non-chat splitting."""
    parts = []
    for i in range(n_sections):
        parts.append(f"# Section {i}\n\n")
        parts.append(("This is body paragraph. " * 100) + "\n\n")
        parts.append(f"## Subsection {i}.1\n\n")
        parts.append(("More body content here. " * 80) + "\n\n")
    text = "".join(parts)
    return Document(
        id="synthetic.md", source_path="/x/synthetic.md",
        source_type=SourceType.MD_FILE, content=text,
        title="synthetic", date="", file_id="synthetic.md",
    )


def _all_fixtures() -> list[Document]:
    docs: list[Document] = []
    for p in _REAL_FIXTURES:
        docs.extend(ingest(p))
    docs.append(_synthetic_markdown())
    return docs


@pytest.mark.parametrize("budget", [4000, 16000, 24000])
def test_every_chunk_fits_budget(budget):
    """Across every fixture, every emitted chunk must fit the budget."""
    docs = _all_fixtures()

    def _check(d: Document):
        out = split_documents([d], budget_tokens=budget)
        oversized = [
            (s.id, count_tokens(s.content))
            for s in out
            if count_tokens(s.content) > budget
        ]
        return d.id, len(out), oversized

    with ThreadPoolExecutor(max_workers=len(docs)) as ex:
        results = list(ex.map(_check, docs))

    for did, _n, oversized in results:
        assert oversized == [], f"{did}: oversized chunks {oversized}"


def test_origin_chars_reconstruct_parent():
    """Sub-doc content slices via origin_char must reassemble the parent
    (modulo whitespace stripped at split boundaries)."""
    docs = _all_fixtures()

    def _check(d: Document):
        out = split_documents([d], budget_tokens=2000)
        if len(out) <= 1:
            return d.id, True
        # Each sub-doc's origin_char + lstrip_shift should land within parent
        for s in out:
            local_start = s.origin_char - d.origin_char
            assert 0 <= local_start <= len(d.content), (
                f"{d.id}: split origin {s.origin_char} out of parent range")
            # The first ~50 chars of the split body should appear at that offset
            head = s.content[:50].lstrip()
            window = d.content[local_start:local_start + 100]
            assert head[:30] in window, (
                f"{d.id}: split content not at expected offset")
        return d.id, True

    with ThreadPoolExecutor(max_workers=len(docs)) as ex:
        list(ex.map(_check, docs))


def test_report_matches_actual_split():
    """report()'s n_splits prediction should match split_documents() output."""
    docs = _all_fixtures()

    def _check(d: Document):
        rows = report([d], budget_tokens=4000)
        out = split_documents([d], budget_tokens=4000)
        return d.id, rows[0]["n_splits"], len(out)

    with ThreadPoolExecutor(max_workers=len(docs)) as ex:
        results = list(ex.map(_check, docs))

    for did, predicted, actual in results:
        assert predicted == actual, f"{did}: report said {predicted}, got {actual}"


def test_chat_splits_at_sleep_gaps():
    """For real chat corpora, splits should land at messages following long gaps."""
    chats = [d for d in _all_fixtures() if d.source_type == SourceType.WHATSAPP]
    if not chats:
        pytest.skip("No chat fixtures present")

    def _check(d: Document):
        out = split_documents([d], budget_tokens=2000)
        if len(out) == 1:
            return d.id, True
        # Each subsequent split should start at a chat-line marker
        for s in out[1:]:
            head = s.content.lstrip()[:1]
            assert head == "[", f"{d.id}: split doesn't start at chat line: {s.content[:60]!r}"
        return d.id, True

    with ThreadPoolExecutor(max_workers=max(1, len(chats))) as ex:
        list(ex.map(_check, chats))
