"""
Splitter — emits sub-Documents that each fit within a token budget AND packs
small consecutive Documents into batched units to avoid one-LLM-call-per-tiny-
entry waste.

No LLM calls, no overlap, no multi-level summaries. Two passes:

  1. Per-doc split: oversize Documents are cut into ranges that each fit
     budget_tokens. Whole-doc by default; only oversize docs get split.
       - IMAGE → pass through (assumed small)
       - WHATSAPP / chat → deterministic greedy time-gap split (sleep gaps
         first, then progressively smaller gaps, until every chunk fits)
       - Everything else → whole doc; if oversize, split by H1/H2 headings,
         then by blank-line paragraphs, then char-truncate as last resort.

  2. Combine pass: walk the resulting list and pack consecutive small same-
     source-type Documents into synthetic combined Documents under the same
     budget. Keeps a Day One year of 365 tiny entries from emitting 365
     extract calls when 7-15 batched calls suffice. Each combined Document
     carries `metadata["combined_entries"]` so content_extractor can route
     fact attribution back to the original entry's file_id / origin_char /
     id.

Output: list[Document]. Each emitted sub-Document carries:
  - .origin_char relative to the parent doc's origin (preserves offset chains)
  - .file_id unchanged (same source file) — except for batched units, whose
    file_id is the synthetic batch id (so the runner treats each batch as
    its own parent in parents_map)
  - .id suffixed with `::split_NN` if the parent was split, or
    `::batch_NNN` if multiple small docs were combined into one unit

This keeps every downstream stage (metadata, extractor, mapper, exporter)
working without changes — they just see more (or batched) Documents. The
content_extractor consults `combined_entries` to route evidence_ref
attribution per fact when present.
"""
from __future__ import annotations

import re
from dataclasses import replace

from engine.ingestor import Document, SourceType
from engine.tokens import count_tokens


# ── Config ────────────────────────────────────────────────────────────────────

# Default per-chunk input budget. Pick so that input + expected output fits in
# the model's num_ctx with room. Extraction can produce ~half the input in
# output tokens, so 16k input + 8k output + headroom ≈ 24-32k context.
DEFAULT_BUDGET_TOKENS = 16000

# Minimum gap (seconds) between chat messages to count as a candidate cut.
# 6h ≈ a sleep cycle; shorter gaps almost always sit mid-conversation.
_MIN_CHAT_GAP_S = 6 * 3600

# Source types whose Documents are small atomic entries that can be safely
# concatenated with sibling entries from the same parent. WhatsApp has its
# own time-gap splitter and individual messages aren't fungible. IMAGE
# transcriptions are usually short and combining them gives the LLM no
# useful context across image boundaries. PDF/DOCX are kept un-batched
# because each tends to be a structured document on its own.
_BATCHABLE_TYPES = frozenset({"dayone_json", "md_file", "txt"})

# Absolute token cap for "tiny enough to batch": entries above this size
# already produce dense per-call extractions when sent solo (one
# 14k-token entry can yield 30 facts in its own call). Combining them
# with neighbors compresses extraction density, and the call-count
# savings don't offset the lost facts.
#
# Tuned so a few-sentences-per-day journal entry (~50-200 tokens)
# always batches, a multi-paragraph entry (~500-5000 t) batches with
# neighbors, and a long-form entry (>8000 t) goes solo. This is the
# absolute brake — even when chunk_cap_for_stage is huge (a 200k ctx
# yields a ~65 k budget), we still refuse to pack long-form entries
# together.
#
# Earlier value (1500) bound at ~5% of stage_cap on dense corpora —
# median batch utilization on a 15-year journal run sat at 5%, leaving
# 95% of every call's budget unused. 8000 lifts the absolute brake
# enough that mid-size entries pack with neighbors while long-form
# pieces still go solo; the halving cascade catches outliers.
_BATCH_SIZE_TOKEN_CAP = 8000

# Relative cap as a sanity floor for very small per-stage chunk caps
# (e.g. LOCAL Ollama). Don't pack docs above this fraction of the
# splitter's budget — there's no room left for a meaningful sibling.
_BATCH_SIZE_FRACTION = 0.25

# Absolute cap on the per-extract-call input size — applied both to
# bundled batches (`combine_documents`) and to per-doc splits
# (`split_documents`). Independent of the stage chunk_cap passed in as
# `budget_tokens`; distinct from `_BATCH_SIZE_TOKEN_CAP` which gates
# per-entry eligibility for being bundled at all.
#
# When `budget_tokens` (= chunk_cap_for_stage(extract)) is huge — e.g.
# 60-200k on Tinfoil/large-context models — the unbounded packing /
# unbounded splitting produced 30-50k+ token calls that misbehaved:
#   - Slow drain: a 66k-token call took 31 minutes before timing out,
#     starving the orchestrator's parallel slots
#   - Cap-hit risk: bigger input → bigger expected output → higher
#     chance of running into the per-call output cap mid-stream
#   - Density compression: dense-extraction yield drops when many
#     entries share one call (one 14k entry solo can yield 30 facts;
#     bundled with neighbors the model compresses per-entry density)
#   - Annotation locality (#174 prereq): annotating one fact
#     invalidates its parent call's cache; smaller per-call inputs
#     bound the recompute footprint per annotation.
#
# 16k trades a small loss of per-call density on legitimately-many-
# small-entries corpora against bounded per-annotation re-ingest cost
# and lower cap-hit pressure on dense corpora. 64k+ is where slow-
# drain and output-cap failures showed up; the chosen anchor sits well
# below that. Re-tune as we collect more cap-hit / timeout data. When
# `budget_tokens` is smaller than this (e.g. LOCAL Ollama with a 16k
# chunk_cap), the budget itself is the tighter brake via
# `min(budget_tokens, _BATCH_TOTAL_TOKEN_CAP)`.
#
# Single-group exception: fragmenting a naturally-one-call surface
# buys no annotation locality (any annotation there invalidates the
# only call regardless) and the parallelism win disappears with only
# one call, so a surface that is already one call is not fragmented.
# This is an exception to *fragmentation*, never to the ceiling: the
# cap remains a true upper bound on per-call input. On the per-doc
# path it is enforced unconditionally — a doc within the cap is one
# call by construction (the splitter returns it untouched), a doc
# above it is split down to the cap. On the combine path a bundled
# run whose natural total ≤ `budget_tokens` is still emitted as one
# group; that group can exceed this cap when the stage chunk_cap is
# far larger, which is the same structural property fixed on the
# per-doc path and is left for a separate change to the combine path.
_BATCH_TOTAL_TOKEN_CAP = 16_000

# Separator inserted between batched entries. Visible to the LLM so it sees
# clear entry boundaries and can attribute occurred_at per entry; carries
# the original date, title, and doc id for explicit attribution.
_BATCH_SEPARATOR_FMT = "\n\n--- ENTRY: {date} | {title} [{doc_id}] ---\n\n"


# ── Anchor helpers ────────────────────────────────────────────────────────────

def _anchor(text: str, n: int = 50) -> str:
    """Compact anchor for logs/debugging: collapse whitespace, trim."""
    s = re.sub(r"\s+", " ", text.strip())
    return s if len(s) <= n else s[: n - 1] + "…"


# ── Generic doc splitting (non-chat) ──────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,2})\s+.+$", re.MULTILINE)
_BLANKLINE_RE = re.compile(r"\n\s*\n")
_NEWLINE_RE = re.compile(r"\n")
_SENTENCE_END_RE = re.compile(r"[.!?]\s")


def _split_at_offsets(text: str, cut_offsets: list[int]) -> list[tuple[int, int]]:
    """Given sorted cut offsets, return list of (start, end) covering text."""
    bounds: list[tuple[int, int]] = []
    prev = 0
    for c in cut_offsets:
        if c > prev:
            bounds.append((prev, c))
            prev = c
    if prev < len(text):
        bounds.append((prev, len(text)))
    return bounds


def _candidate_cuts_generic(text: str, budget_tokens: int = 4000) -> list[int]:
    """
    Return candidate split offsets for non-chat documents. Candidates are
    merged across structural levels (headings, paragraphs, single newlines,
    sentence endings) and sorted by position; the greedy splitter walks them
    left-to-right and backs off to the latest candidate before budget overflow.

    Dense fine-grained candidates (single newlines, sentence endings) keep the
    backoff from snapping to a coarse char-bucket boundary when the prose has
    no paragraph structure (e.g. an index or concordance section). The char-
    bucket level is appended last as an absolute fallback for text with zero
    natural break candidates.
    """
    levels: list[list[int]] = []

    # Level 1: H1/H2 boundaries
    h_offsets = sorted({m.start() for m in _HEADING_RE.finditer(text)} - {0})
    if h_offsets:
        levels.append(h_offsets)

    # Level 2: blank-line paragraph boundaries
    p_offsets = sorted({m.end() for m in _BLANKLINE_RE.finditer(text)})
    if p_offsets:
        levels.append(p_offsets)

    # Level 3: single-newline boundaries (line wraps, no paragraph break).
    # Covers list/index/concordance sections that lack `\n\n` structure but
    # still have one cut candidate per line.
    nl_offsets = [m.end() for m in _NEWLINE_RE.finditer(text)]
    if nl_offsets:
        levels.append(nl_offsets)

    # Level 4: sentence endings (`. `, `! `, `? `). Covers prose that runs
    # without internal newlines — the backoff lands at a sentence boundary
    # instead of a mid-sentence char-bucket cut.
    s_offsets = [m.end() for m in _SENTENCE_END_RE.finditer(text)]
    if s_offsets:
        levels.append(s_offsets)

    # Level 5: budget-proportional char-bucket fallback. ~3 chars/token for
    # English; min 200 chars so tiny budgets don't degenerate to empty cuts.
    step = max(200, budget_tokens * 3)
    bucket = list(range(step, len(text), step))
    if bucket:
        levels.append(bucket)

    # Greedy splitter walks candidates left-to-right and assumes monotonic
    # position. Dedupe across levels and sort by offset so it works regardless
    # of which level contributed each candidate.
    return sorted({c for level in levels for c in level})


def _greedy_split_generic(text: str, budget_tokens: int) -> list[tuple[int, int]]:
    """
    Whole-doc by default. If text exceeds budget, place cuts greedily:
    walk candidate offsets in document order, backing off to the latest
    candidate that fit when adding the next would blow the budget.

    Backoff priority within candidates:
      1. Paragraph-or-stronger (heading / blank-line paragraph break)
      2. Sentence end (`. `, `! `, `? `)
      3. Plain newline (line wrap inside a paragraph or list/index entry)
      4. Char-bucket fallback (last resort for text with no natural breaks)
      5. Force-cut at the overflow candidate (none of the above fit)

    Why these tiers exist:
      - Paragraph > sentence keeps tight semantic clusters together. A
        single thematic paragraph (e.g. a multi-sentence passage on one
        topic) is preserved as one chunk even when sentence-ends inside
        it sit closer to the budget boundary — sentence-ends would
        otherwise win as the denser tier-1 candidate and break the
        paragraph mid-thought.
      - Sentence > newline keeps line-wrapped prose from snapping
        cuts to mid-sentence line breaks. Line-wraps are dense (every
        wrapped line) and would always win the "latest candidate"
        race against sparser sentence-ends without an explicit tier.
      - Newline > bucket keeps structureless line-based text (index /
        concordance sections that have newlines but no sentence
        punctuation) cutting at line boundaries rather than at the
        coarse char-bucket fallback.

    Returns list of (start_char, end_char) ranges covering the full text.
    """
    if count_tokens(text) <= budget_tokens:
        return [(0, len(text))]

    candidates = _candidate_cuts_generic(text, budget_tokens)
    if not candidates:
        return [(0, len(text))]

    # Classify candidates by tier. A position lands in the highest tier
    # it qualifies for: a `\n\n` paragraph break ends with `\n` and would
    # also match `_NEWLINE_RE`, but it belongs to the paragraph tier.
    paragraph_set = (
        {m.start() for m in _HEADING_RE.finditer(text)}
        | {m.end() for m in _BLANKLINE_RE.finditer(text)}
    )
    sentence_set = {m.end() for m in _SENTENCE_END_RE.finditer(text)}
    step = max(200, budget_tokens * 3)
    bucket_set = set(range(step, len(text), step))

    bounds: list[tuple[int, int]] = []
    chunk_start = 0
    last_paragraph = chunk_start
    last_sentence = chunk_start
    last_newline = chunk_start
    last_bucket = chunk_start
    for c in candidates:
        if c <= chunk_start:
            continue
        if count_tokens(text[chunk_start:c]) > budget_tokens:
            if last_paragraph > chunk_start:
                cut = last_paragraph
            elif last_sentence > chunk_start:
                cut = last_sentence
            elif last_newline > chunk_start:
                cut = last_newline
            elif last_bucket > chunk_start:
                cut = last_bucket
            else:
                cut = c
            bounds.append((chunk_start, cut))
            chunk_start = cut
            last_paragraph = chunk_start
            last_sentence = chunk_start
            last_newline = chunk_start
            last_bucket = chunk_start
        elif c in paragraph_set:
            last_paragraph = c
        elif c in sentence_set:
            last_sentence = c
        elif c in bucket_set:
            last_bucket = c
        else:
            last_newline = c
    if chunk_start < len(text):
        bounds.append((chunk_start, len(text)))
    return bounds


# ── Chat splitting (WhatsApp) ─────────────────────────────────────────────────

# WhatsApp line pattern: `[d/m/yy, h:mm:ss AM/PM] Sender: text` or
# `d/m/yy, h:mm AM/PM - Sender: text`.
_CHAT_LINE = re.compile(
    r"""^
    \[?
    (?P<date>\d{1,2}/\d{1,2}/\d{2,4}),?\s+
    (?P<time>\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)
    \]?\s+
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _parse_chat_messages(text: str) -> list[tuple[int, int]]:
    """
    Return [(char_offset, epoch_seconds_or_None), ...] for each chat message
    start. epoch is None for messages we couldn't time-parse.
    """
    from datetime import datetime

    out: list[tuple[int, int]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        m = _CHAT_LINE.match(line)
        if m:
            ts = None
            for fmt in ("%d/%m/%y, %I:%M:%S %p", "%d/%m/%y, %H:%M:%S",
                        "%d/%m/%Y, %I:%M:%S %p", "%d/%m/%Y, %H:%M:%S",
                        "%m/%d/%y, %I:%M:%S %p", "%m/%d/%y, %H:%M:%S",
                        "%d/%m/%y, %I:%M %p", "%d/%m/%y, %H:%M",
                        "%m/%d/%y, %I:%M %p", "%m/%d/%y, %H:%M"):
                try:
                    ts = int(datetime.strptime(
                        f"{m.group('date')}, {m.group('time')}", fmt
                    ).timestamp())
                    break
                except ValueError:
                    continue
            out.append((pos, ts))
        pos += len(line)
    return out


def _greedy_split_chat(text: str, budget_tokens: int) -> list[tuple[int, int]]:
    """
    Greedy gap-based chat splitter: keep cutting at the largest time gap
    inside any oversized chunk until every chunk fits.

    Falls back to generic splitting if message parsing yields nothing useful.
    """
    if count_tokens(text) <= budget_tokens:
        return [(0, len(text))]

    msgs = [(off, ts) for off, ts in _parse_chat_messages(text) if ts is not None]
    if len(msgs) < 2:
        return _greedy_split_generic(text, budget_tokens)

    # Build (gap_seconds, cut_offset) for each between-message boundary.
    gaps: list[tuple[int, int]] = []
    for i in range(1, len(msgs)):
        gap = msgs[i][1] - msgs[i - 1][1]
        if gap >= _MIN_CHAT_GAP_S:
            gaps.append((gap, msgs[i][0]))

    # Start with a single chunk covering everything; greedily cut the largest
    # gap inside whichever current chunk overflows budget. Repeat until all fit.
    bounds: list[tuple[int, int]] = [(0, len(text))]
    gaps_sorted = sorted(gaps, key=lambda x: -x[0])
    used: set[int] = set()
    bypass: set[tuple[int, int]] = set()  # ranges we've given up trying to split
    while True:
        # Find the first oversized chunk we haven't already accepted as terminal.
        idx = None
        for i, (s, e) in enumerate(bounds):
            if (s, e) in bypass:
                continue
            if count_tokens(text[s:e]) > budget_tokens:
                idx = i
                break
        if idx is None:
            return bounds
        s, e = bounds[idx]
        # Largest unused gap that lies strictly inside (s, e).
        cut = None
        for _g, off in gaps_sorted:
            if off in used:
                continue
            if s < off < e:
                cut = off
                used.add(off)
                break
        if cut is None:
            # No more gap-based cuts inside this chunk — try generic
            # splitting for the same range.
            sub = _greedy_split_generic(text[s:e], budget_tokens)
            sub_global = [(s + a, s + b) for a, b in sub]
            # If generic also can't subdivide, accept the oversize chunk
            # rather than loop forever. (e.g. a single message that's huge.)
            if sub_global == [(s, e)]:
                bypass.add((s, e))
                continue
            bounds = bounds[:idx] + sub_global + bounds[idx + 1:]
            continue
        bounds = bounds[:idx] + [(s, cut), (cut, e)] + bounds[idx + 1:]


# ── Per-type dispatch ─────────────────────────────────────────────────────────

def _split_text(doc: Document, budget_tokens: int) -> list[tuple[int, int]]:
    if doc.source_type == SourceType.IMAGE:
        return [(0, len(doc.content))]
    if doc.source_type == SourceType.WHATSAPP:
        return _greedy_split_chat(doc.content, budget_tokens)
    return _greedy_split_generic(doc.content, budget_tokens)


def _emit_subdocs(doc: Document, ranges: list[tuple[int, int]]) -> list[Document]:
    """Build sub-Documents for each (start, end) range, preserving offsets."""
    if len(ranges) == 1 and ranges[0] == (0, len(doc.content)):
        return [doc]
    out: list[Document] = []
    for i, (s, e) in enumerate(ranges):
        chunk = doc.content[s:e]
        stripped = chunk.lstrip()
        lstrip_shift = len(chunk) - len(stripped)
        body = stripped.rstrip()
        if not body:
            continue
        out.append(replace(
            doc,
            id=f"{doc.id}::split_{i:02d}",
            content=body,
            origin_char=doc.origin_char + s + lstrip_shift,
            metadata={
                **doc.metadata,
                "split_of": doc.id,
                "split_index": i,
                "split_first50": _anchor(body[:80]),
                "split_last50": _anchor(body[-80:]),
            },
        ))
    return out


# ── Combining (batch small consecutive docs) ─────────────────────────────────

def _is_batchable(d: Document, budget_tokens: int) -> bool:
    """A doc is eligible for combining if it's a small atomic entry from a
    batchable source type AND not itself the output of an oversize-split
    (split_of metadata is set). Splits already sit close to budget so packing
    them with anything else would blow past it.

    "Small" is bounded by both an absolute token cap (so batching only
    targets the genuinely-tiny daily-entry pathology) AND a relative cap
    (so we don't try to batch in modes with very small chunk budgets)."""
    if d.source_type.value not in _BATCHABLE_TYPES:
        return False
    if d.metadata.get("split_of"):
        return False
    tokens = count_tokens(d.content)
    if tokens >= _BATCH_SIZE_TOKEN_CAP:
        return False
    if tokens >= int(_BATCH_SIZE_FRACTION * budget_tokens):
        return False
    return True


def _make_batch_document(
    group: list[Document],
    batch_seq: int,
) -> Document:
    """Build one synthetic Document from `group`. content is the inner
    entries' text joined by `_BATCH_SEPARATOR_FMT`; metadata
    `combined_entries` records each inner entry's [content_start,
    content_end) range plus the original file_id, origin_char, id, date,
    title — everything content_extractor needs to route a quoted span back
    to the right source."""
    parts: list[str] = []
    entries: list[dict] = []
    cursor = 0
    for idx, d in enumerate(group):
        if idx > 0:
            sep = _BATCH_SEPARATOR_FMT.format(
                date=d.date or "unknown",
                title=d.title or d.id,
                doc_id=d.id,
            )
            parts.append(sep)
            cursor += len(sep)
        start = cursor
        parts.append(d.content)
        cursor += len(d.content)
        end = cursor
        entries.append({
            "id": d.id,
            "file_id": d.file_id,
            "source_path": d.source_path,
            "origin_char": d.origin_char,
            "title": d.title,
            "date": d.date,
            "content_start": start,
            "content_end": end,
            "tags": list(d.metadata.get("tags", []) or []),
            "uuid": d.metadata.get("uuid"),
            "location": d.metadata.get("location"),
        })

    head = group[0]
    new_id = f"{head.file_id or head.id}::batch_{batch_seq:03d}"

    dates = sorted({e["date"] for e in entries if e["date"]})
    if len(dates) == 1:
        new_date = dates[0]
    elif dates:
        new_date = f"{dates[0]}..{dates[-1]}"
    else:
        new_date = ""

    titles = [d.title for d in group if d.title]
    if titles and len(set(titles)) == 1:
        new_title = titles[0]
    elif titles:
        new_title = f"{titles[0]} → {titles[-1]} ({len(group)} entries)"
    else:
        new_title = f"batch_{batch_seq:03d}"

    # file_id intentionally inherits the head's original file_id so the
    # runner's parents_map groups all batches from one source file under
    # the same parent. That keeps metadata extraction at one call per
    # source file (instead of one per batch — which on a 3000-entry Day
    # One JSON would emit ~50 redundant metadata calls and fan the
    # parent ThreadPoolExecutor out wide enough to hit the macOS file-
    # descriptor limit). The synthetic batch identity lives on `id`
    # (used for partial-file names and source_ref).
    return Document(
        id=new_id,
        source_path=head.source_path,
        source_type=head.source_type,
        content="".join(parts),
        title=new_title,
        date=new_date,
        file_id=head.file_id or head.id,
        origin_char=0,
        metadata={
            "combined_entries": entries,
            "combined_count": len(entries),
        },
    )


def combine_documents(
    docs: list[Document],
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> list[Document]:
    """Pack consecutive small same-source-type Documents into synthetic
    batched Documents that fit within `budget_tokens`. Non-batchable docs
    (oversized, wrong source type, or already a split) pass through
    unchanged. A run of length 1 also passes through — combining adds no
    value and would inject a misleading combined_entries metadata for a
    solo entry.

    Each batched Document carries metadata["combined_entries"]: a list of
    dicts mapping inner [content_start, content_end) ranges back to
    {id, file_id, source_path, origin_char, date, title, tags, uuid,
    location}. content_extractor consults this list to route fact
    attribution back to the right source.
    """
    out: list[Document] = []
    i = 0
    n = len(docs)
    batch_seq = 0

    def _sep_tokens(doc: Document) -> int:
        return count_tokens(_BATCH_SEPARATOR_FMT.format(
            date=doc.date or "unknown",
            title=doc.title or doc.id,
            doc_id=doc.id,
        ))

    def _emit(group: list[Document]) -> None:
        nonlocal batch_seq
        if len(group) == 1:
            out.append(group[0])
            return
        out.append(_make_batch_document(group, batch_seq))
        batch_seq += 1

    while i < n:
        d = docs[i]
        if not _is_batchable(d, budget_tokens):
            out.append(d)
            i += 1
            continue

        # Identify the full run of consecutive batchable same-source-type
        # docs and its natural (uncapped) total, so we can decide whether
        # the cap fires or the single-group exception applies — see the
        # comment on `_BATCH_TOTAL_TOKEN_CAP`.
        run_end = i + 1
        run_total = count_tokens(d.content)
        while run_end < n:
            nxt = docs[run_end]
            if not _is_batchable(nxt, budget_tokens):
                break
            if nxt.source_type != d.source_type:
                break
            run_total += count_tokens(nxt.content) + _sep_tokens(nxt)
            run_end += 1

        run = docs[i:run_end]
        i = run_end

        if run_total <= budget_tokens:
            # Single-group exception: the whole run fits in `budget_tokens`
            # as one group. Fragmenting would buy no annotation-locality
            # (the only call IS the whole run) so the cap is skipped.
            _emit(list(run))
            continue

        # Run exceeds `budget_tokens`; fragment, with the cap acting as
        # the tighter brake on per-group size.
        size_limit = min(budget_tokens, _BATCH_TOTAL_TOKEN_CAP)
        j = 0
        while j < len(run):
            group = [run[j]]
            group_tokens = count_tokens(run[j].content)
            k = j + 1
            while k < len(run):
                nxt = run[k]
                added = count_tokens(nxt.content) + _sep_tokens(nxt)
                if group_tokens + added > size_limit:
                    break
                group.append(nxt)
                group_tokens += added
                k += 1
            _emit(group)
            j = k
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def split_documents(
    docs: list[Document],
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    combine_small: bool = True,
) -> list[Document]:
    """
    Two-pass:
      1. Per-doc split — whole-document by default; cut only when a doc
         exceeds budget_tokens, using per-type rules.
      2. Combine — pack consecutive small same-source-type Documents into
         synthetic batched units under budget_tokens. Disabled when
         combine_small=False (used by the pre/post baseline measurement).

    Returns a list of Documents that each fit within budget. Order of
    original docs is preserved (combining only packs adjacent runs).
    """
    out: list[Document] = []
    for d in docs:
        if not d.content.strip():
            continue
        # The absolute per-call brake is a true ceiling on the per-doc
        # path, sourced from the one shared constant the combine path's
        # fragmentation loop also reads — no divergent per-path limit.
        # The single-group exception still holds, but only as "don't
        # fragment a one-call surface", never as "lift a call above the
        # brake": a doc within the brake emits whole (the splitter
        # returns it as one chunk untouched), while a doc above the
        # brake is split down to it. This matters when the stage chunk
        # cap is far larger than the brake (e.g. 60k+ on a large-context
        # TEE model): without the clamp a single continuous document —
        # the characteristic shape of a WhatsApp export — fits the cap
        # whole and bypasses the ceiling, producing one oversized, slow,
        # density-compressed extract call.
        split_budget = min(budget_tokens, _BATCH_TOTAL_TOKEN_CAP)
        ranges = _split_text(d, split_budget)
        out.extend(_emit_subdocs(d, ranges))
    if combine_small:
        out = combine_documents(out, budget_tokens=budget_tokens)
    return out


# Ceiling (NOT an exact target — less is fine) on the very first
# extraction chunk so the user sees real extracted facts within
# seconds-to-low-minutes instead of waiting on a full-size first call.
#
# Empirical basis (kimi extraction calls, ~/laptop-runs, uncached +
# success + ≥60s real calls): per-call duration ≈ ~350s fixed floor +
# ~40s per 1k prompt tokens. The fixed floor dominates, so a token cap
# can only remove the input-proportional term — but on the slowest
# model in the fleet that still cuts time-to-first-fact from ~25-30 min
# (today's median first call is ~53k tokens) to ~80-120s. 2000 is the
# smallest size at which real kimi extraction calls were observed
# (min 1,938 tokens), so it's a measured floor, not an extrapolation.
#
# Model-agnostic by construction: applied to every run regardless of
# the configured extraction model. Sizing against the slowest model is
# a conservative bound — a cap that returns acceptably under kimi
# returns at least as fast under any faster model. No per-model table
# (that would over-fit and drift); one fixed constant.
#
# This is a CEILING via the existing boundary backoff, not a new cut:
# `carve_first_batch` peels ONLY the leading slice of the first unit at
# this smaller budget (its end lands on the same tiered paragraph/
# sentence/newline boundary every other chunk uses; char-bucket only as
# the existing last resort), then splits the REMAINDER of that unit at
# the normal stage budget. Net effect is ONE extra cut in the first
# unit — [≤cap slice] + [remainder at the normal budget] — NOT a
# fragmentation of the whole unit into cap-sized pieces. Fragmenting the
# whole unit would multiply per-call fixed cost (catastrophic on a model
# with a large fixed per-call floor like kimi: one 16k call → eight 2k
# calls × the floor) and contradicts the approved "one extra cut point"
# design. Frequently the natural boundary lands well under 2000 — that's
# expected and only faster.
FIRST_BATCH_TOKEN_CAP = 2000


def carve_first_batch(
    docs: list[Document],
    cap: int = FIRST_BATCH_TOKEN_CAP,
    rest_budget: int = DEFAULT_BUDGET_TOKENS,
) -> list[Document]:
    """Make the FIRST emitted unit small so the extract stage's first
    LLM call returns fast and real extracted facts surface early.

    The extract stage's first LLM call is already gate-exempt: it is the
    first dispatch of the first scheduler stage, so the per-stage
    interval gate doesn't bind it (`_last_start_ts` is the `-inf`
    sentinel) and the pool starts at healthy size. The producer yields
    `docs` in order, so making `docs[0]` small is the entire mechanism —
    no scheduler/producer change, no model routing.

    Operates ONLY on `docs[0]`, and only when it is a plain
    (non-combined) Document above `cap`. Peels the leading ≤`cap` slice
    using the existing `split_documents` backoff, then re-splits the
    REMAINDER of that unit at `rest_budget` (the normal stage chunk
    budget) — so the first unit gains exactly ONE extra cut:
    `[≤cap slice] + [remainder at the normal budget]`. The remainder is
    NOT fragmented into cap-sized pieces (that would multiply per-call
    fixed cost and is net-slower on a high-fixed-floor model like kimi).
    `docs[1:]` is appended unchanged (by reference, in order). The fact
    union over the corpus is unchanged — downstream entities/dedupe
    already converge facts across adjacent chunks; validated by the
    eval + per-stage extract regression parity harness.

    Passes through untouched when:
      - `docs` is empty;
      - `docs[0]` is a combined batch (`combined_entries` set) —
        re-slicing it would cut across the synthetic entry separators
        and corrupt per-entry offset attribution; never do that;
      - `docs[0]` already fits `cap` — it is already a fast first batch
        (a small first document or a small batch); nothing to gain;
      - `docs[0]` cannot be cut below `cap` (one unbreakable blob) —
        nothing to peel, leave the run unchanged.
    """
    if not docs:
        return docs
    lead = docs[0]
    if lead.metadata.get("combined_entries"):
        return docs
    if count_tokens(lead.content) <= cap:
        return docs
    # Peel ONLY the leading slice at `cap` via the existing per-doc
    # backoff (single doc → nothing to combine). The first emitted piece
    # is the fast first batch.
    head_pieces = split_documents([lead], budget_tokens=cap, combine_small=False)
    if len(head_pieces) <= 1:
        # Unbreakable below cap — carving buys nothing; leave unchanged.
        return docs
    early = replace(head_pieces[0], id=f"{lead.id}::head")
    # Remainder = lead content from the next slice boundary onward,
    # re-split at the NORMAL stage budget so the rest of the first unit
    # keeps today's chunk size. `head_pieces[1].origin_char` is the
    # absolute start of the second slice; the inter-slice gap is edge
    # whitespace the splitter already drops everywhere.
    cut_at = head_pieces[1].origin_char - lead.origin_char
    remainder = replace(
        lead,
        id=f"{lead.id}::rest",
        content=lead.content[cut_at:],
        origin_char=lead.origin_char + cut_at,
        metadata={**lead.metadata, "split_of": lead.id},
    )
    rest_pieces = split_documents(
        [remainder], budget_tokens=rest_budget, combine_small=False,
    )
    return [early] + rest_pieces + docs[1:]


def report(
    docs: list[Document],
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> list[dict]:
    """
    Pre-split inspection: per-doc token count, whether it fits the
    budget, and (if it doesn't) how it would be split. No mutation;
    safe to call before split_documents.
    """
    rows: list[dict] = []
    for d in docs:
        toks = count_tokens(d.content)
        if toks <= budget_tokens:
            rows.append({
                "id": d.id, "tokens": toks, "fits": True,
                "n_splits": 1, "type": d.source_type.value,
            })
            continue
        ranges = _split_text(d, budget_tokens)
        rows.append({
            "id": d.id, "tokens": toks, "fits": False,
            "n_splits": len(ranges), "type": d.source_type.value,
            "split_sizes_tokens": [count_tokens(d.content[s:e]) for s, e in ranges],
        })
    return rows
