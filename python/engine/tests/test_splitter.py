"""Unit tests for splitter — the new whole-doc-by-default + chat time-gap chunker."""

from engine.ingestor import Document, SourceType
from engine.splitter import (
    _anchor,
    _candidate_cuts_generic,
    _greedy_split_generic,
    _greedy_split_chat,
    _parse_chat_messages,
    _emit_subdocs,
    _BATCH_SEPARATOR_FMT,
    _BATCH_TOTAL_TOKEN_CAP,
    combine_documents,
    split_documents,
    carve_first_batch,
    FIRST_BATCH_TOKEN_CAP,
    report,
)
from engine.tokens import count_tokens


# ── _anchor ───────────────────────────────────────────────────────────────────

def test_anchor_truncates_long():
    out = _anchor("a" * 100, n=20)
    assert len(out) == 20 and out.endswith("…")


def test_anchor_collapses_whitespace():
    assert _anchor("hello   \n\t world") == "hello world"


def test_anchor_short_passthrough():
    assert _anchor("short") == "short"


# ── _candidate_cuts_generic ───────────────────────────────────────────────────

def test_candidate_cuts_includes_headings():
    text = "intro para\n\nmiddle\n\n# Heading\nbody\n## Sub\nmore"
    cuts = _candidate_cuts_generic(text)
    h1_offset = text.index("# Heading")
    h2_offset = text.index("## Sub")
    # Both heading offsets must be present (cuts are now sorted by position,
    # not by priority level, since the greedy splitter assumes monotonic
    # iteration).
    assert h1_offset in cuts and h2_offset in cuts
    assert cuts == sorted(set(cuts))


def test_candidate_cuts_paragraph_fallback_when_no_headings():
    text = "para one\n\npara two\n\npara three"
    cuts = _candidate_cuts_generic(text)
    assert cuts  # should produce paragraph cuts
    # No headings means cuts come from blank-line offsets only
    assert all(text[c - 2:c] == "\n\n" or text[c - 1:c + 1] == "\n\n" for c in cuts)


def test_candidate_cuts_char_bucket_when_nothing_else():
    # Single line, no headings, no blank lines.
    # Step is budget-proportional (~3 chars/token), min 200.
    text = "x" * 12000
    cuts = _candidate_cuts_generic(text, budget_tokens=400)
    # step = max(200, 400*3) = 1200 → cuts at 1200, 2400, 3600, ...
    assert cuts and cuts[0] == 1200 and cuts[-1] < len(text)


# ── _greedy_split_generic ────────────────────────────────────────────────────

def test_generic_under_budget_returns_whole():
    text = "small content"
    out = _greedy_split_generic(text, budget_tokens=1000)
    assert out == [(0, len(text))]


def test_generic_oversize_splits_when_oversize():
    # Three sections of ~1100 tokens each (chars/3 estimator), separated
    # by H1 headings — total well over budget. Splitter should cut.
    chunk = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
             "Sed do eiusmod tempor incididunt ut labore et dolore. " * 30)
    text = chunk + "\n\n# Section A\n" + chunk + "\n\n# Section B\n" + chunk
    assert count_tokens(text) > 1100   # sanity: above budget
    out = _greedy_split_generic(text, budget_tokens=1100)
    # Multiple chunks + full coverage. The splitter merges all candidate
    # cut points (headings, blank lines, char-bucket fallback) into one
    # sorted list and picks last-OK before overflow — it does NOT
    # preferentially cut at headings, so don't assert that. Each chunk
    # fits within 1.25× budget (greedy's last-segment can overflow when
    # no candidate exists past chunk_start).
    assert len(out) >= 2
    assert out[0][0] == 0 and out[-1][1] == len(text)
    for s, e in out:
        assert count_tokens(text[s:e]) <= int(1100 * 1.25)


def test_generic_splits_index_style_without_overflow():
    # Index/concordance prose: many lines, no `\n\n` paragraph breaks.
    # Without fine-grained candidates the splitter would snap to the
    # char-bucket fallback (~budget*3 chars) and emit oversized chunks
    # whenever an entry happened to span a bucket boundary. With single-
    # newline + sentence-end candidates the cut backs off to the last
    # newline before the budget — no chunk exceeds the budget.
    line = "     Some index entry phrase that runs for a while across the page\n"
    text = line * 6000  # ~135k chars, ~45k tokens — no `\n\n` anywhere
    assert "\n\n" not in text  # sanity: no paragraph break candidates
    out = _greedy_split_generic(text, budget_tokens=1000)
    assert len(out) >= 2
    assert out[0][0] == 0 and out[-1][1] == len(text)
    for s, e in out:
        # Tight bound: every chunk lands at-or-under budget (no 1.25×
        # slack), because newline candidates are dense enough that the
        # backoff is always within ~one line of the budget.
        assert count_tokens(text[s:e]) <= 1000


def test_generic_splits_prose_without_newlines_at_sentence_ends():
    # Prose with no newlines anywhere — only sentence endings remain as
    # candidates. The splitter should still land at sentence boundaries
    # instead of the char-bucket fallback.
    sentence = "Lorem ipsum dolor sit amet. "  # ends with `. ` — Level 4
    text = sentence * 2000  # ~56k chars, ~19k tokens, no newlines
    assert "\n" not in text  # sanity: no newline candidates
    out = _greedy_split_generic(text, budget_tokens=1000)
    assert len(out) >= 2
    for s, e in out:
        # Each chunk ends right after a sentence-ending whitespace, so
        # the slice should end with the period+space pattern (or be the
        # final chunk).
        chunk = text[s:e]
        assert count_tokens(chunk) <= 1000
        if (s, e) != out[-1]:
            # Non-final chunk: cut at a sentence end → ends after `. `.
            assert chunk.rstrip()[-1] in ".!?" or chunk.endswith(". ")


def test_generic_prefers_paragraph_break_over_sentence_end():
    # A paragraph contains multiple sentences. Earlier paragraphs are
    # short enough that more than one fits in the budget; the splitter
    # should cut at the paragraph break before the next paragraph, NOT
    # at a sentence-end inside that next paragraph — even though the
    # mid-paragraph sentence-end would be denser-packed.
    para_a = "Short opening paragraph. Two sentences. " * 30  # ~360 t
    para_b = (
        "The wind rose and tapped the line against the flag-staff at the\n"
        "Coastguard Station. It roared through my hair and past my ears.\n"
        "On the way home we saw the wind darting hither and thither.\n"
        "The wind! Oh! the wind--I have an enormous faith in the\n"
        "curative properties of the wind. I feel better already.\n"
    )
    para_c = "Closing paragraph. " * 30
    text = para_a + "\n\n" + para_b + "\n" + para_c
    # Budget sized so para_a + para_b fits but para_a + para_b + para_c does not.
    budget = count_tokens(para_a) + count_tokens(para_b) + 20
    out = _greedy_split_generic(text, budget_tokens=budget)
    assert len(out) >= 2
    # First cut MUST be at a `\n\n` paragraph break — the wind paragraph
    # (para_b) must end up entirely on one side of the cut, not split
    # between "Coastguard Station." and "It roared…".
    first_cut = out[0][1]
    assert text[first_cut - 2:first_cut] == "\n\n", (
        f"First cut should land at paragraph break, got: "
        f"...{text[max(0, first_cut-30):first_cut]!r}"
    )
    # Coverage: ranges cover full text byte-for-byte.
    assert "".join(text[s:e] for s, e in out) == text


def test_generic_prefers_sentence_end_over_line_wrap_newline():
    # Line-wrapped prose (Barbellion-style): every line ends with `\n`
    # but sentences span multiple lines. Without the semantic-tier
    # preference, the greedy walker would pick the latest `\n` candidate
    # — which often sits mid-sentence — over the latest sentence end.
    # With the preference, every non-final cut lands at a sentence end,
    # so adjacent chunks join cleanly at sentence boundaries.
    line = "Quite a long sentence that wraps across two lines like this\n"
    # Make some line ends NOT be sentence ends: break the period across
    # a wrap by joining two `line`s back-to-back, then ending with one.
    paragraph = (line + line + line.rstrip() + ". ") * 80
    text = paragraph
    assert "\n" in text and ". " in text  # sanity
    out = _greedy_split_generic(text, budget_tokens=400)
    assert len(out) >= 2
    for s, e in out:
        chunk = text[s:e]
        assert count_tokens(chunk) <= 400
        if (s, e) != out[-1]:
            # Non-final cut must land at a sentence boundary (last char
            # before any trailing whitespace is `.`/`!`/`?`), NOT at a
            # mid-sentence line wrap.
            stripped = chunk.rstrip()
            assert stripped and stripped[-1] in ".!?", (
                f"Non-final chunk ends mid-sentence: ...{stripped[-50:]!r}"
            )
    # Coverage: rebuilt text equals original (no characters lost).
    assert "".join(text[s:e] for s, e in out) == text


def test_generic_split_covers_full_text():
    text = ("para\n\n" * 1000)
    out = _greedy_split_generic(text, budget_tokens=200)
    # Reconstructable
    rebuilt = "".join(text[s:e] for s, e in out)
    assert rebuilt == text


# ── _parse_chat_messages ─────────────────────────────────────────────────────

def test_parse_chat_bracketed():
    text = "[8/10/25, 1:24:55 PM] Alice: hi\n[8/10/25, 1:25:00 PM] Bob: yo\n"
    msgs = _parse_chat_messages(text)
    assert len(msgs) == 2
    assert all(ts is not None for _, ts in msgs)
    # 5s apart
    assert msgs[1][1] - msgs[0][1] == 5


def test_parse_chat_no_seconds():
    text = "[8/10/25, 1:24 PM] Alice: hi\n"
    msgs = _parse_chat_messages(text)
    assert len(msgs) == 1 and msgs[0][1] is not None


def test_parse_chat_unparseable_yields_none_ts():
    text = "no timestamps here\nor here either\n"
    msgs = _parse_chat_messages(text)
    assert msgs == []  # no _CHAT_LINE matches at all


# ── _greedy_split_chat ────────────────────────────────────────────────────────

def _chat_messages(spans):
    """Build a chat-formatted text from list of (yyyy_md_dot, hh_mm, body)."""
    out = []
    for date, time, body in spans:
        out.append(f"[{date}, {time}] User: {body}\n")
    return "".join(out)


def test_chat_under_budget_passthrough():
    text = _chat_messages([
        ("8/10/25", "1:00:00 PM", "hi"),
        ("8/10/25", "1:01:00 PM", "hello"),
    ])
    assert _greedy_split_chat(text, budget_tokens=1000) == [(0, len(text))]


def test_chat_cuts_at_largest_gap():
    # Three message bursts separated by big sleep gaps. Force splits.
    # Budget sized to ~one burst (610 tokens at chars/3) so each burst
    # naturally becomes its own chunk and the splitter cuts at the gaps.
    burst = lambda day, n: _chat_messages(
        [(day, f"{(i % 12) + 1}:00:00 PM", "x" * 200) for i in range(n)]
    )
    text = burst("8/10/25", 8) + burst("8/15/25", 8) + burst("8/20/25", 8)
    bounds = _greedy_split_chat(text, budget_tokens=600)
    assert len(bounds) >= 3
    # Chat splitter is best-effort under tight budgets — allow small overage
    # when the only cut points are dense bursts within a single day.
    for s, e in bounds:
        assert count_tokens(text[s:e]) <= int(600 * 1.25)


def test_chat_falls_back_to_generic_when_few_timed_msgs():
    # Single timed message, lots of text — no gaps available; falls back.
    text = "[8/10/25, 1:00:00 PM] Alice: " + ("x" * 8000)
    bounds = _greedy_split_chat(text, budget_tokens=200)
    # Did produce splits via generic fallback
    assert len(bounds) > 1


# ── _emit_subdocs ─────────────────────────────────────────────────────────────

def _doc(content, origin_char=0, source_type=SourceType.TXT):
    return Document(
        id="parent",
        source_path="/x/parent.txt",
        source_type=source_type,
        content=content,
        title="parent",
        date="",
        file_id="parent",
        origin_char=origin_char,
    )


def test_emit_passthrough_when_single_full_range():
    d = _doc("hello world")
    out = _emit_subdocs(d, [(0, len(d.content))])
    assert out == [d]


def test_emit_preserves_origin_char_chain():
    d = _doc("abc\n\ndef\n\nghi", origin_char=100)
    out = _emit_subdocs(d, [(0, 5), (5, 10), (10, 13)])
    assert [s.origin_char for s in out] == [100, 105, 110]
    assert all(s.id.startswith("parent::split_") for s in out)
    assert all("split_first50" in s.metadata for s in out)


def test_emit_skips_empty_bodies():
    d = _doc("aaa\n\n\n\n   \n\n\n\nbbb")
    out = _emit_subdocs(d, [(0, 4), (4, 12), (12, len(d.content))])
    bodies = [s.content for s in out]
    assert "aaa" in bodies and "bbb" in bodies
    # Whitespace-only middle range was dropped
    assert all(b.strip() for b in bodies)


def test_emit_lstrip_shift_adjusts_origin():
    d = _doc("  hello world", origin_char=10)
    out = _emit_subdocs(d, [(0, 5), (5, len(d.content))])
    # First chunk was "  hel" → lstripped "hel" → origin_char +=2
    assert out[0].origin_char == 12


# ── split_documents (public) ──────────────────────────────────────────────────

def test_split_documents_passes_small_through():
    d = _doc("tiny doc")
    out = split_documents([d], budget_tokens=1000)
    assert out == [d]


def test_split_documents_drops_empty():
    d = _doc("   \n\n\t  ")
    assert split_documents([d]) == []


def test_split_documents_chat_splits_when_oversize():
    # Build oversize chat
    msgs = []
    for day in range(1, 6):
        for hr in range(0, 10):
            msgs.append((f"8/{day:02d}/25", f"{(hr % 12) + 1}:00:00 PM", "x" * 200))
    text = _chat_messages(msgs)
    d = Document(
        id="chat", source_path="/x/c.txt", source_type=SourceType.WHATSAPP,
        content=text, title="chat", date="", file_id="chat",
    )
    out = split_documents([d], budget_tokens=1000)
    assert len(out) > 1
    for sub in out:
        assert count_tokens(sub.content) <= 1000


def test_split_documents_oversize_doc_bounded_by_total_cap():
    # On big-context models, chunk_cap_for_stage("extract") arrives huge
    # (60-200 k+). When a single doc dwarfs even that — e.g. a 50-year
    # diary at millions of tokens — the per-doc split must still bound
    # each split at _BATCH_TOTAL_TOKEN_CAP so annotation re-ingest cost
    # is bounded per chunk.
    body = ("para\n\n" * 100_000)  # ~200 k tokens at chars/3
    d = _doc(body)
    out = split_documents([d], budget_tokens=80_000, combine_small=False)
    cap_with_slack = _BATCH_TOTAL_TOKEN_CAP + 100
    assert len(out) > 1  # cap fragments where budget alone would not
    for sub in out:
        assert count_tokens(sub.content) <= cap_with_slack


def test_split_documents_per_doc_clamped_to_brake_under_huge_stage_cap():
    # The per-doc single-group exception waives *fragmentation*, never
    # the ceiling: a single doc above the brake is split down to it even
    # when the stage chunk cap is far larger and the doc fits the cap
    # whole. (Previously this ~22 k doc under a 30 k budget stayed as one
    # oversized chunk — the bug.)
    body = ("para\n\n" * 10_000)  # ~22 k tokens at chars/3
    assert count_tokens(body) > _BATCH_TOTAL_TOKEN_CAP  # sanity
    d = _doc(body)
    out = split_documents([d], budget_tokens=30_000, combine_small=False)
    assert len(out) > 1
    for sub in out:
        assert count_tokens(sub.content) <= _BATCH_TOTAL_TOKEN_CAP + 100


def test_split_documents_dense_gapless_whatsapp_chat_fragments_under_huge_cap():
    # #602 reproduction. A continuous WhatsApp chat with no large time
    # gaps (messages a minute apart) whose total exceeds the brake but
    # fits a large-context stage chunk cap whole. The time-gap splitter
    # finds no >=6h cut; the per-doc clamp must still bound it so it
    # fragments into multiple bounded extract calls instead of one
    # oversized (slow, density-compressed) call.
    msgs = []
    for i in range(600):
        day = i // 1440 + 1
        hour = (i // 60) % 24
        minute = i % 60
        msgs.append((f"8/{day:02d}/25", f"{hour:02d}:{minute:02d}:00", "x" * 300))
    text = _chat_messages(msgs)
    huge_stage_cap = 200_000  # mirrors Tinfoil/large-context extract cap
    assert _BATCH_TOTAL_TOKEN_CAP < count_tokens(text) < huge_stage_cap  # sanity
    d = Document(
        id="chat", source_path="/x/c.txt", source_type=SourceType.WHATSAPP,
        content=text, title="chat", date="", file_id="chat",
    )
    out = split_documents([d], budget_tokens=huge_stage_cap, combine_small=False)
    assert len(out) > 1
    for sub in out:
        assert count_tokens(sub.content) <= _BATCH_TOTAL_TOKEN_CAP + 100


# ── report ────────────────────────────────────────────────────────────────────

def test_report_reports_fits_and_oversize():
    small = _doc("small")
    big = _doc("para\n\n" * 5000)
    rows = report([small, big], budget_tokens=500)
    assert len(rows) == 2
    fits = {r["id"]: r["fits"] for r in rows}
    assert fits["parent"] is False or fits["parent"] is True  # both are id="parent"; check shape
    assert any(r["fits"] for r in rows)
    assert any(not r["fits"] and "split_sizes_tokens" in r for r in rows)


def test_report_does_not_mutate():
    d = _doc("para\n\n" * 5000, origin_char=42)
    report([d], budget_tokens=500)
    assert d.origin_char == 42 and d.id == "parent"


# ── combine_documents ────────────────────────────────────────────────────────


def _journal_entry(idx, content, date="2026-01-01", file_id="journal.json", origin_char=0):
    """Build a Day One-style entry — small, batchable."""
    return Document(
        id=f"{file_id}::entry_{idx:03d}",
        source_path=f"/x/{file_id}",
        source_type=SourceType.DAYONE_JSON,
        content=content,
        title=f"Journal {date}",
        date=date,
        file_id=file_id,
        origin_char=origin_char,
        metadata={"uuid": f"uuid-{idx}"},
    )


def test_combine_packs_consecutive_small_dayone_entries():
    entries = [_journal_entry(i, f"entry {i} body " * 5, date=f"2026-01-{i+1:02d}")
               for i in range(5)]
    out = combine_documents(entries, budget_tokens=10000)
    # All five should pack into one batched unit (each ~70 chars; budget is huge).
    assert len(out) == 1
    batch = out[0]
    assert batch.metadata["combined_count"] == 5
    assert len(batch.metadata["combined_entries"]) == 5
    assert "::batch_000" in batch.id
    # The batched Document inherits the head's original file_id (NOT
    # a synthetic one) so the runner's parents_map groups all batches
    # from one source file under the same parent — one metadata call
    # per source file and a sane parent-level fan-out.
    assert batch.file_id == entries[0].file_id == "journal.json"
    # Every original entry is recoverable by slicing the combined content
    # at the recorded ranges.
    for orig, rec in zip(entries, batch.metadata["combined_entries"]):
        sliced = batch.content[rec["content_start"]:rec["content_end"]]
        assert sliced == orig.content
        assert rec["id"] == orig.id
        assert rec["origin_char"] == orig.origin_char
        assert rec["file_id"] == orig.file_id


def test_combine_respects_budget():
    # Each entry at ~600 tokens (1800 chars at chars/3). Budget=2000 should
    # pack at most ~3 per batch; this test ensures the batch stays under
    # budget.
    body = "x" * 1800
    entries = [_journal_entry(i, body, date=f"2026-01-{i+1:02d}") for i in range(10)]
    out = combine_documents(entries, budget_tokens=2000)
    for d in out:
        assert count_tokens(d.content) <= 2000
    # And every original entry still appears in some batch's combined_entries.
    seen = set()
    for d in out:
        for rec in (d.metadata.get("combined_entries") or [{"id": d.id}]):
            seen.add(rec["id"])
    assert seen == {e.id for e in entries}


def test_combine_passes_solo_doc_unchanged():
    [d] = [_journal_entry(0, "hello")]
    out = combine_documents([d], budget_tokens=10000)
    assert out == [d]
    assert "combined_entries" not in out[0].metadata


def test_combine_skips_docs_above_token_cap():
    # The absolute token cap (currently ~8000 t) is the dominant gate:
    # long-form entries (above the cap) extract densely on their own
    # and shouldn't be packed with neighbors even in a budget that has
    # plenty of room.
    big_body = "y" * 30_000  # ~10000 tokens at chars/3 — above the absolute cap
    big = _journal_entry(0, big_body, date="2026-01-01")
    small = _journal_entry(1, "small body", date="2026-01-02")
    small2 = _journal_entry(2, "another small", date="2026-01-03")
    out = combine_documents([big, small, small2], budget_tokens=63000)
    # big is emitted as-is; small + small2 batch together.
    assert out[0] is big
    assert "combined_entries" not in out[0].metadata
    assert out[1].metadata["combined_count"] == 2


def test_combine_skips_when_relative_cap_dominates_in_small_budgets():
    # In modes with tiny chunk caps (e.g. LOCAL Ollama 20k), the relative
    # 0.25-of-budget guard should also reject mid-size entries even when
    # they're below the absolute cap. Build a 1100-token entry against a
    # 4000-token budget — relative gate (1000 t) excludes it.
    medium_body = "z" * 3300  # 1100 t at chars/3 — under absolute cap (8000)
    medium = _journal_entry(0, medium_body, date="2026-01-01")
    small = _journal_entry(1, "tiny", date="2026-01-02")
    out = combine_documents([medium, small], budget_tokens=4000)
    # Both pass through individually since the run never fully forms.
    assert out == [medium, small]


def test_combine_cap_fragments_oversize_run():
    # When the run's natural total exceeds budget_tokens, the cap is the
    # tighter brake on per-group size. 300 × 500 t ≈ 150 k of content +
    # separators is far above the 32 k budget here, so the single-group
    # exception doesn't apply and fragmentation kicks in.
    body = "x" * 1500  # 500 t at chars/3 — well under per-entry cap
    entries = [_journal_entry(i, body) for i in range(300)]
    out = combine_documents(entries, budget_tokens=32_000)
    # Cap is enforced on the running token sum (each piece's count_tokens
    # is floor(chars/3)). count_tokens on the assembled batch floors once
    # over the whole content, so per-piece rounding losses can add up to
    # ~80 tokens with this many entries+separators. Allow a small slack.
    cap_with_slack = _BATCH_TOTAL_TOKEN_CAP + 100
    for d in out:
        assert count_tokens(d.content) <= cap_with_slack
    # 300×500 t ≈ 150 k content; at 16 k cap we expect ~10 batches.
    assert len(out) >= 5
    # Every original entry is still routed through some batch.
    seen = set()
    for d in out:
        for rec in (d.metadata.get("combined_entries") or [{"id": d.id}]):
            seen.add(rec["id"])
    assert seen == {e.id for e in entries}


def test_combine_single_group_exception_skips_cap():
    # Single-group exception: when a run of consecutive batchable docs
    # would fit in budget_tokens as one group, the cap is skipped.
    # 100 × 500 t ≈ 50 k of content + separators fits within a 64 k
    # budget, so this is one extract call's worth of input regardless
    # of whether the cap fires. Annotation locality buys nothing from
    # fragmenting (the only call IS the whole run), so it's left alone.
    body = "x" * 1500
    entries = [_journal_entry(i, body) for i in range(100)]
    out = combine_documents(entries, budget_tokens=64_000)
    assert len(out) == 1
    batch = out[0]
    # The batch must exceed _BATCH_TOTAL_TOKEN_CAP — that's the whole
    # point of the exception. It must also fit within budget_tokens.
    tokens = count_tokens(batch.content)
    assert tokens > _BATCH_TOTAL_TOKEN_CAP
    assert tokens <= 64_000
    assert batch.metadata["combined_count"] == 100


def test_combine_small_budget_unaffected_by_total_cap():
    # When budget_tokens is at or below _BATCH_TOTAL_TOKEN_CAP, the
    # budget itself is the tighter brake and the single-group exception
    # cannot help on an oversize run.
    body = "x" * 1500
    entries = [_journal_entry(i, body) for i in range(100)]
    out = combine_documents(entries, budget_tokens=16_000)
    for d in out:
        assert count_tokens(d.content) <= 16_000
    # Run total of ~50 k forces multiple groups under a 16 k size limit.
    assert len(out) >= 3


def test_combine_exception_at_budget_boundary():
    # Run total ≤ budget_tokens triggers the exception (boundary is `≤`,
    # not `<`). Build a run whose total lands a touch under a chosen
    # budget so the boundary check fires cleanly.
    body = "x" * 1500
    entries = [_journal_entry(i, body) for i in range(40)]
    # 40 × 500 t = 20 000 t content + 39 separators ≈ 20.5 k. With
    # budget = 24_000 the run fits in one group; exception applies.
    out = combine_documents(entries, budget_tokens=24_000)
    assert len(out) == 1
    assert count_tokens(out[0].content) > _BATCH_TOTAL_TOKEN_CAP


def test_combine_exception_just_over_budget_fragments():
    # Run total just over budget_tokens disables the exception and the
    # cap fragments the run.
    body = "x" * 1500
    entries = [_journal_entry(i, body) for i in range(40)]
    # Same ~20.5 k total against a 20 k budget — over by a hair, so the
    # exception does not fire and the cap (min(20k, 16k) = 16 k) splits
    # the run into multiple groups.
    out = combine_documents(entries, budget_tokens=20_000)
    assert len(out) >= 2
    for d in out:
        assert count_tokens(d.content) <= 16_000 + 100


def test_combine_runs_around_oversize_doc_are_decided_independently():
    # An oversize (non-batchable) doc in the middle splits a sequence
    # into two independent runs. Each run gets its own exception/cap
    # decision: here both runs fit in budget, so both bypass the cap.
    big_body = "y" * 30_000  # 10 000 t — above _BATCH_SIZE_TOKEN_CAP
    big = _journal_entry(99, big_body, date="2026-01-99")
    pre = [_journal_entry(i, "x" * 1500, date=f"2026-01-{i+1:02d}")
           for i in range(10)]
    post = [_journal_entry(100 + i, "x" * 1500, date=f"2026-02-{i+1:02d}")
            for i in range(10)]
    out = combine_documents(pre + [big] + post, budget_tokens=64_000)
    # Expected: [batch_pre, big, batch_post] — 3 outputs.
    assert len(out) == 3
    assert out[1] is big
    assert out[0].metadata["combined_count"] == 10
    assert out[2].metadata["combined_count"] == 10


def test_combine_runs_split_by_source_type_decided_independently():
    # Source-type boundaries break runs. Two short same-source-type
    # subsequences flanking a txt doc form two independent runs; each
    # gets the exception independently.
    dayone_a = [_journal_entry(i, "x" * 1500, date=f"2026-01-{i+1:02d}")
                for i in range(5)]
    txt = Document(
        id="middle.txt", source_path="/x/middle.txt",
        source_type=SourceType.TXT, content="some text body",
        title="middle", date="2026-01-15", file_id="middle.txt",
    )
    dayone_b = [_journal_entry(100 + i, "x" * 1500, date=f"2026-02-{i+1:02d}")
                for i in range(5)]
    out = combine_documents(dayone_a + [txt] + dayone_b, budget_tokens=64_000)
    # Each dayone run becomes one batch; the txt sits alone (run of 1).
    assert len(out) == 3
    assert out[0].metadata.get("combined_count") == 5
    assert out[1] is txt
    assert out[2].metadata.get("combined_count") == 5


def test_combine_separator_tokens_counted_in_run_total():
    # Separator overhead (~12 t per separator on these short titles)
    # must count toward the run total used by the exception check.
    # Build a run whose entry-tokens alone fit a budget, but whose
    # entry-tokens + separator-tokens exceed it — that should fragment.
    body = "x" * 1500  # 500 t per entry
    # 40 entries × 500 t = 20 000 t of content alone; 39 separators
    # carry their own tokens. With budget=20_050 (slightly above content
    # tokens), the separator-bearing total still exceeds budget, so the
    # exception must not fire.
    entries = [_journal_entry(i, body) for i in range(40)]
    out = combine_documents(entries, budget_tokens=20_050)
    assert len(out) >= 2  # fragmented — separators tipped the balance


def test_combine_solo_batchable_run_passes_through():
    # A run of one batchable doc has run_total ≤ budget by construction
    # (_is_batchable already enforces per-entry size). The exception's
    # len==1 branch emits it as a solo passthrough (no synthetic batch).
    [d] = [_journal_entry(0, "x" * 600)]
    out = combine_documents([d], budget_tokens=64_000)
    assert out == [d]
    assert "combined_entries" not in out[0].metadata


def test_combine_does_not_cross_source_type():
    # Different source types break the run.
    a = _journal_entry(0, "dayone body 1", date="2026-01-01")
    b = Document(
        id="b.txt", source_path="/x/b.txt", source_type=SourceType.TXT,
        content="txt body", title="b", date="", file_id="b.txt",
    )
    c = _journal_entry(2, "dayone body 2", date="2026-01-02")
    out = combine_documents([a, b, c], budget_tokens=10000)
    # No two adjacent docs share source_type, so no batching: 3 in, 3 out.
    assert len(out) == 3
    assert all("combined_entries" not in d.metadata for d in out)


def test_combine_skips_split_outputs():
    # A split-output sub-doc (split_of present) shouldn't be batched even
    # if it's small.
    split_doc = Document(
        id="parent::split_00",
        source_path="/x/parent.txt",
        source_type=SourceType.TXT,
        content="split body",
        title="parent",
        date="",
        file_id="parent",
        metadata={"split_of": "parent"},
    )
    other = Document(
        id="other.txt", source_path="/x/other.txt", source_type=SourceType.TXT,
        content="other body", title="other", date="", file_id="other.txt",
    )
    out = combine_documents([split_doc, other], budget_tokens=10000)
    # split_doc passes through; `other` is alone in its batchable run → solo.
    assert out == [split_doc, other]


def test_combine_skips_chat_and_image_types():
    # WHATSAPP and IMAGE are NOT in _BATCHABLE_TYPES. They pass through.
    chat = Document(
        id="chat.txt", source_path="/x/chat.txt", source_type=SourceType.WHATSAPP,
        content="chat body", title="chat", date="", file_id="chat.txt",
    )
    img = Document(
        id="img.jpg", source_path="/x/img.jpg", source_type=SourceType.IMAGE,
        content="image transcription", title="img", date="", file_id="img.jpg",
    )
    out = combine_documents([chat, img], budget_tokens=10000)
    assert out == [chat, img]
    assert all("combined_entries" not in d.metadata for d in out)


def test_combine_separator_visible_in_content():
    a = _journal_entry(0, "alpha", date="2026-01-01")
    b = _journal_entry(1, "bravo", date="2026-01-02")
    [batch] = combine_documents([a, b], budget_tokens=10000)
    sep = _BATCH_SEPARATOR_FMT.format(date=b.date, title=b.title, doc_id=b.id)
    assert sep in batch.content
    # alpha appears before bravo in content, in the same order they were
    # passed in.
    assert batch.content.index("alpha") < batch.content.index("bravo")


def test_combine_preserves_origin_char_per_entry():
    # An entry with non-zero origin_char (e.g. came from a parent at offset 100).
    a = _journal_entry(0, "alpha", date="2026-01-01", origin_char=100)
    b = _journal_entry(1, "bravo", date="2026-01-02", origin_char=200)
    [batch] = combine_documents([a, b], budget_tokens=10000)
    entries = batch.metadata["combined_entries"]
    assert entries[0]["origin_char"] == 100
    assert entries[1]["origin_char"] == 200


def test_split_documents_combines_when_combine_small_true():
    entries = [_journal_entry(i, "body " * 5, date=f"2026-01-{i+1:02d}")
               for i in range(3)]
    out = split_documents(entries, budget_tokens=10000, combine_small=True)
    assert len(out) == 1
    assert out[0].metadata["combined_count"] == 3


def test_split_documents_skips_combine_when_combine_small_false():
    entries = [_journal_entry(i, "body " * 5, date=f"2026-01-{i+1:02d}")
               for i in range(3)]
    out = split_documents(entries, budget_tokens=10000, combine_small=False)
    assert len(out) == 3
    assert all("combined_entries" not in d.metadata for d in out)


# ── carve_first_batch (early first-extraction batch) ──────────────────────────

def _big_doc(n_paras=80):
    """A plain doc well over FIRST_BATCH_TOKEN_CAP with clean paragraph
    boundaries for the existing backoff to land on."""
    para = "This is a sentence about something. " * 6
    return _doc("\n\n".join(f"Para {i}. {para}" for i in range(n_paras)))


def test_carve_first_batch_empty_passthrough():
    assert carve_first_batch([]) == []


def test_carve_first_batch_small_lead_is_noop():
    d = _doc("short content")
    assert carve_first_batch([d], cap=FIRST_BATCH_TOKEN_CAP) == [d]


def test_carve_first_batch_combined_lead_passthrough_uncorrupted():
    # A combined batch as docs[0] must pass through untouched — re-slicing
    # would cut across the synthetic entry separators and corrupt
    # per-entry offset attribution.
    entries = [_journal_entry(i, "small entry body. " * 20,
                              date=f"2026-01-{i+1:02d}")
               for i in range(6)]
    combined = combine_documents(entries, budget_tokens=10000)
    assert combined[0].metadata.get("combined_entries")
    tail = _doc("tail")
    out = carve_first_batch([combined[0], tail])
    assert out == [combined[0], tail]
    assert (out[0].metadata["combined_entries"]
            == combined[0].metadata["combined_entries"])


def test_carve_first_batch_shrinks_oversize_lead_under_cap():
    big = _big_doc()
    assert count_tokens(big.content) > FIRST_BATCH_TOKEN_CAP
    tail = _doc("second doc")
    out = carve_first_batch([big, tail], cap=FIRST_BATCH_TOKEN_CAP)
    # First emitted unit is now a real, smaller carve at/under the cap.
    assert count_tokens(out[0].content) <= FIRST_BATCH_TOKEN_CAP
    assert len(out[0].content) < len(big.content)
    # Carve pieces are named ::head (fast slice) + ::rest (remainder),
    # not a confusing ::split_NN double-suffix.
    assert out[0].id == f"{big.id}::head"
    assert out[1].id.startswith(f"{big.id}::rest")
    # docs[1:] is byte-identical to today (tail object untouched).
    assert out[-1] is tail


def test_carve_first_batch_preserves_content_union_and_offsets():
    big = _big_doc()
    out = carve_first_batch([big], cap=FIRST_BATCH_TOKEN_CAP)
    assert len(out) > 1  # actually subdivided
    offs = [s.origin_char for s in out]
    assert offs == sorted(offs)          # monotonic
    assert offs[0] == big.origin_char    # chained off the parent
    # No content lost: concatenated bodies cover the parent text (modulo
    # the splitter's per-chunk lstrip/rstrip of edge whitespace).
    norm = lambda s: s.replace(" ", "").replace("\n", "")
    assert norm("".join(s.content for s in out)) == norm(big.content)


def test_carve_first_batch_cuts_at_natural_boundary_not_midword():
    # The leading slice ends at a boundary the existing backoff picks
    # (paragraph/sentence/newline) — never a hard mid-token cut. With
    # paragraph-structured input the rstrip'd body ends on sentence
    # punctuation.
    out = carve_first_batch([_big_doc()], cap=FIRST_BATCH_TOKEN_CAP)
    assert out[0].content.rstrip()[-1] in ".!?"


def test_carve_first_batch_does_not_fragment_remainder():
    # Regression: a lead far larger than rest_budget must yield ONE
    # extra cut — [≤cap slice] + [remainder at rest_budget] — NOT the
    # whole unit shattered into cap-sized pieces (the cmju over-
    # fragmentation: one 16k unit → nine 2k calls). The remainder
    # pieces must be sized at rest_budget, i.e. each > cap.
    huge = _big_doc(n_paras=600)               # ~45k tokens
    cap, rest = 2000, 16000
    assert count_tokens(huge.content) > 2 * rest
    out = carve_first_batch([huge], cap=cap, rest_budget=rest)
    # First piece is the small fast batch.
    assert count_tokens(out[0].content) <= cap
    # Remainder split at the BIG budget, not at cap: few pieces, each
    # well over cap. If carve wrongly used `cap` for the remainder this
    # would be dozens of ≤cap pieces.
    remainder = out[1:]
    assert len(remainder) <= (count_tokens(huge.content) // rest) + 2
    assert all(count_tokens(d.content) > cap for d in remainder)
    # Content union preserved across the one-extra-cut layout.
    norm = lambda s: s.replace(" ", "").replace("\n", "")
    assert norm("".join(d.content for d in out)) == norm(huge.content)
