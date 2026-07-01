"""
Content Extractor — produces structured ExtractedItems (Facts) from raw
Document text.

Each item is one of six atomic types:
  fact       — default; durable statement about reality
  decision   — explicit choice or commitment
  event      — thing that happened in time
  emotion    — explicit felt state
  signal     — specific observable behavior that may matter later (not a pattern)
  open_loop  — unresolved risk, question, or required follow-up

Each item also carries an `affect` dimension — the emotional register the
source EXPRESSED (joy, anxiety, grief, …), distinct from the `emotion` item
type. Affect is source-expressed tone, not inferred mood, and is empty when
the content reads as neutral; the model abstains rather than fabricate tone.

Extraction is source-grounded: every item traces back to an exact quote in
the document. No inference, interpretation, or cross-document synthesis.
Pattern-level reasoning (principles, fears, traits) belongs in the
Patterns stage, not here.

Documents fed to extract_items are expected to be at-or-below the model's
input budget (the splitter handles this). Each call processes one whole
Document; topics + people are emitted per item directly (no shared
document-level metadata stage).

Usage:
    from engine.content_extractor import extract_items, ExtractedItem
    items = extract_items(docs, mode=Mode.TEE)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# Prefer rapidfuzz when available (C-backed, 20-100x faster on the fuzzy
# fallback path in _resolve_span). Fall back to stdlib difflib so dev
# machines without the dep still run — the only cost is wall-clock on
# large chunks with many paraphrased quotes.
try:
    from rapidfuzz import fuzz as _rf_fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

from engine.ingestor import Document
from engine.llm import (
    strip_fences as _strip_fences,
)


def _log_exception_for_stage(stage: str, ctx: str, exc: BaseException) -> None:
    """Mirror of the per-stage `_log_exception` shape used by other
    pipeline modules (insights/actions/dedupe/patterns). Routes
    through the module's `_log_info` so the runner-side log hook
    catches it."""
    import traceback
    head = f"  [{stage}] LLM call raised ({ctx}): {type(exc).__name__}: {exc}"
    _log_info(head)
    for line in traceback.format_exc().rstrip().splitlines():
        _log_info(f"    {line}")


# ── IR types ──────────────────────────────────────────────────────────────────

ITEM_TYPES = [
    "fact", "decision", "event", "emotion", "signal", "open_loop",
]
ENTITY_TYPES = ["person", "place", "org", "concept", "other"]
ROLES = ["subject", "object", "mentioned"]

# Affect / expressed-sentiment taxonomy. A general, bounded set of common
# emotional registers source content can EXPRESS. Distinct from the
# `emotion` item type: emotion is a fact whose CONTENT is a felt state;
# affect is a dimension on ANY item, recording the register in which the
# source expressed it. Kept deliberately general — no corpus-specific
# registers. An empty affect list means no clear register was expressed
# (the neutral / abstain answer); the model is never forced to pick one.
AFFECT_REGISTERS: tuple[str, ...] = (
    "joy", "love", "hope", "pride", "gratitude", "relief",
    "anger", "frustration", "anxiety", "fear", "sadness", "grief",
    "shame", "disgust", "surprise",
)
_AFFECT_REGISTER_SET = frozenset(AFFECT_REGISTERS)
# Cap per item to bound hallucination — real content rarely expresses more
# than two distinct registers at once; beyond that is usually over-labeling.
# The prompt asks for 0-2 and the parser enforces the ceiling.
_MAX_AFFECT = 2

# Default topic taxonomy. Items emit topics from this list; topics drive
# the patterns stage's per-topic synthesis. Settings → General → Categories
# can override at runtime via `app_config.categories`; `_topics_for_run`
# resolves the effective list. The default seeds a fresh install and is
# the fallback when the config field is absent / empty / wrong-shape.
_DEFAULT_TOPICS: tuple[str, ...] = (
    "admin", "education", "family", "finance", "health",
    "housing", "legal", "logistics", "other", "relationships",
    "spirituality", "travel", "work",
)


def _topics_for_run() -> list[str]:
    """Return the active topic taxonomy for this run.

    Reads `app_config.categories` (string list) and falls back to
    `_DEFAULT_TOPICS` when the field is absent, empty, or not a list of
    non-empty strings. The pipeline calls this at extract-prompt-build
    time and at output-filter time so the same taxonomy gates input
    instructions and output validation.
    """
    from engine.llm import _read_app_config
    raw = _read_app_config().get("categories")
    if not isinstance(raw, list):
        return list(_DEFAULT_TOPICS)
    cleaned = [str(t).strip() for t in raw if isinstance(t, str) and str(t).strip()]
    return cleaned if cleaned else list(_DEFAULT_TOPICS)


@dataclass
class Entity:
    name: str
    entity_type: str   # person | place | org | concept | other


@dataclass
class EntityRef:
    entity: Entity
    role: str          # subject | object | mentioned


@dataclass
class EvidenceSpan:
    text: str
    source_ref: str    # chunk.id — logical path through the pipeline
    start_char: int | None = None  # offset within chunk content
    end_char: int | None = None    # offset within chunk content
    # Canonical ref into the original ingested file: path + (offset, length).
    # Populated at extraction time by summing the offset chain
    # (seg_doc → section → chunk → span).
    file_path: str | None = None
    file_offset: int | None = None
    file_length: int | None = None
    # True when the span came from the longest-common-substring fallback
    # rather than a direct or whitespace-normalized substring match. The
    # anchor still points to a reasonable region but the quote is not a
    # literal substring of the source — the vault renders a "≈" marker.
    approximate: bool = False


@dataclass
class ExtractedItem:
    item_type: str                   # one of ITEM_TYPES
    summary: str
    evidence: list[EvidenceSpan]
    occurred_at: str | None = None       # ISO YYYY-MM-DD
    occurred_at_text: str | None = None  # raw mention
    entities: list[EntityRef] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)   # from runtime topic taxonomy
    # Expressed-sentiment registers (subset of AFFECT_REGISTERS), capped at
    # _MAX_AFFECT. Empty = no register expressed (neutral / abstained). This is
    # the source's expressed tone on this item, NOT the `emotion` item type.
    affect: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    # AT MOST ONE explicitly-stated relation per fact. None when no
    # relation between two named entities is stated unambiguously in
    # this fact's evidence. Shape: {"from": str, "to": str, "verb": str,
    # "confidence": float}. The from/to strings are entity names that
    # the entities stage resolves to canonical_ids; unresolvable
    # candidates are dropped at grouping time.
    relation_candidate: dict | None = None


# ── Default models ────────────────────────────────────────────────────────────




# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You extract durable, pattern-ready information from arbitrary source content \
(chat, journal, notes, technical docs, medical reports, self-reflective \
essays, curated personal OS documents, etc.).

Hard rules:
- Extract everything that would still matter weeks or months from now.
- Calibrate density to the document: dense, already-curated content yields
  many items per page; coordination-heavy chat yields few.
- Noise (scheduling, greetings, ephemeral status, restatements of the obvious)
  is NOT extractable on its own.
- Durable content mentioned inside noisy text IS extractable — mine it out.
- Self-reflective narrative IS NOT noise. Self-observations about one's
  own state, behavior, or history are durable facts about the author.
  Record them as grounded facts/emotions/events — do NOT promote them
  into principles, fears, or other pattern types here. Pattern synthesis
  is a later stage and works on the grounded facts you produce.
- DO NOT synthesize patterns, traits, beliefs, principles, or fears
  across items or within a single item. Emit only grounded statements.
- Summaries must stay as close as possible to what is explicitly supported
  by evidence.
- If interpreting internal meaning, use hedged wording: "appears to",
  "describes feeling", "frames as".
- Never convert symbolism, desire, or emotional intensity into objective fact.
- DO NOT turn contextual guesses into facts. If a detail is not explicitly
  supported by evidence, stay generic or omit it.
- DO NOT output anything outside the JSON schema.
- DO NOT emit reasoning, chain-of-thought, plans, or commentary. The
  schema below is your entire response. The first non-whitespace
  character of your response must be `{` and the last must be `}`.
- Use ONLY the fields named in the schema below. Do not add any other
  fields. If you would naturally include extra metadata, omit it.\
"""

_TASK = """\
Extract only significant, durable items from the document below. Be selective
about transient noise, but do not miss durable content embedded inside it.

Document title: {title}
Document date: {date}

Item types — use ONLY these exact values. "fact" is the default when no
more specific type applies. Principles, beliefs, fears, and other
synthesizable patterns belong to a later stage and should NOT be emitted
here — record the grounded statements instead and the pattern stage will
surface the synthesis.
- fact       : default. Durable factual statement about a person, place,
               situation, constraint, or relationship that does not fit
               a more specific type below.
- decision   : significant choice or commitment with lasting impact
- event      : notable thing that happened — milestones, incidents, meetings with context
- emotion    : explicit felt state
- signal     : specific observable behavior that may matter later — must be concrete,
               NOT a trait or pattern
- open_loop  : unresolved risk, follow-up, or question with real stakes

Durable vs noise — the core distinction:
- Durable: anything that remains true, relevant, or meaningful beyond the
  immediate moment. Facts about who someone is, their family, health, work,
  constraints, commitments, preferences; stated principles, beliefs, wounds,
  values; emotions tied to significant events, recurring states, or
  identity-relevant insight; self-observations about recurring patterns.
- Noise: coordination and scheduling that vanishes once the moment passes,
  filler prose, greetings, confirmations, trivial status updates, obvious
  restatements, unimportant observations.
- Durable is NOT only objective facts. Subjective, emotional, relational, and
  textural content — how something felt, what it meant, the tone in which it
  was expressed, the quality of a moment or relationship — is durable
  substance in its own right, co-equal with logistical or factual statements.
  Do not down-rank a statement because it is felt rather than factual; the
  emotional and relational texture is often exactly what makes the record
  worth keeping. Under-extracting it is as much a miss as dropping a hard fact.

Document annotations (do not confuse with content):
- Bracketed dates like [YYYY-MM-DD] are metadata timestamps on entries —
  don't extract them as events; use them to populate occurred_at on the
  underlying durable statement.
- Wiki-links like [[some-name]] and hashtags like #essential are author
  annotations — they point at related content but aren't themselves items.
- Markdown headings and list bullets are structure. Extract from the substance
  inside them, not from the structural markers.

Mining density — calibrate to the input:
- A scheduling chat may yield 0-3 items per thousand tokens.
- A journal entry may yield 5-15.
- A distilled personal-OS / self-reflective document can yield 30+ —
  nearly every declarative line is a durable fact, event, or emotion.
- If the document is substantive (more than a few hundred tokens of real
  content) and you are returning an empty array, you are almost certainly
  filtering too hard. Re-read for grounded statements before emitting [].

Topic taxonomy — assign relevant topics per item, ONLY these exact values:
{topics}

Affect dimension — the emotional register the SOURCE expressed:
- Separate from the `emotion` item type. The `emotion` TYPE is for items whose
  content IS a felt state. `affect` is a DIMENSION on any item — fact,
  decision, event, signal, etc. — recording the emotional register in which
  the source expressed it.
- Use ONLY these registers, ONLY these exact values: {affect_registers}
- Assign 0-2 registers per item, drawn from the source's own wording — the
  register the author expressed, NOT your reaction to it and NOT an inferred
  mood.
- Neutral content takes an EMPTY affect list. Dry, factual, logistical, or
  unemotional statements express no register — emit []. Empty is the correct,
  expected answer; do NOT invent affect to fill the field.
- Abstain over fabricate: include a register only when the wording reasonably
  supports it. When unsure, leave it out — a missing affect is always better
  than a fabricated one.

Respond with ONLY a single JSON object with this exact shape:
{{
  "split_summaries": [
    {{
      "id": "{doc_id}",
      "summary": "<1-2 sentence durable summary of this split's content>"
    }}
  ],
  "items": [
    {{
      "type": "<item type>",
      "summary": "<atomic 1-sentence statement>",
      "evidence": [
        {{
          "text": "<exact verbatim quote from the document>",
          "source_ref": "{doc_id}"
        }}
      ],
      "occurred_at_text": "<raw date mention from text, or null>",
      "occurred_at": "<YYYY-MM-DD if inferrable from text or document date, or null>",
      "entities": [
        {{
          "name": "<canonical name>",
          "entity_type": "person|place|org|concept|other",
          "role": "subject|object|mentioned"
        }}
      ],
      "topics": ["<topic from taxonomy>"],
      "affect": ["<affect register from taxonomy>"],
      "tags": ["<tag>", ...],
      "confidence": <0.0-1.0>,
      "relation_candidate": {{
        "from": "<entity name>",
        "to": "<entity name>",
        "verb": "<short verb phrase paraphrasing the stated relation>",
        "confidence": <0.0-1.0>
      }} or null
    }}
  ]
}}

Split summary rules:
- Emit the split_summaries array FIRST in your response so it is preserved
  if the output is truncated.
- Capture what the split is ABOUT in 1-2 sentences: topics covered,
  key actors, decisions, or events. Stay durable — skip ephemeral
  coordination chatter. Do NOT restate facts already in items; the
  summary is a chunk-level gist for downstream retrieval, not a recap.
- Use the same hedged-language and no-synthesis discipline as items:
  paraphrase what's there, do not interpret patterns or invent
  meaning that isn't on the page.
- The `id` MUST match the document id given at the top of the
  schema (`{doc_id}`) for a single-document split. See the BATCH NOTE
  below if a batch separator appears in the document.

Item rules:
- Each item must be atomic (one thing per item)
- signal must anchor to a specific behavior, not a trait. Traits, beliefs,
  fears, principles, and other synthesizable patterns are NOT emitted at
  this stage — they emerge later from the grounded facts below.
- relation_candidate: emit ONLY when this fact's evidence EXPLICITLY states
  a relation between two named entities ("X is Y's spouse", "X works at Y",
  "X met with Y for lunch"). At most ONE relation per fact — pick the most
  salient if multiple are stated. Prefer `null` on any ambiguity. Do NOT
  emit a relation just because two entities co-occur in the same evidence;
  co-occurrence is not a relation. The from/to fields must be entity names
  that also appear in this item's `entities` list. Confidence reflects how
  explicit the source is.
- occurred_at: use document date to resolve partial mentions; null if genuinely unknown
- No extra keys, no prose, no markdown fences
- If primary value is what happened, use event. If primary value is what was
  felt, use emotion.
- Before emitting an item, ask: would this plausibly matter in 30+ days? If
  no, drop unless it anchors a major event.
- Do not emit duplicate items that restate the same underlying point. Prefer
  one strongest version.

Salience ordering (important):
- Emit items in DESCENDING order of salience: most consequential,
  pattern-bearing, identity-defining, or recurring items first; incidental
  details last.
- If the response gets cut off before you finish, the items at the end are
  the ones we can most afford to lose. Optimize for "what matters most got
  said first."

QUOTE RULE (critical — read carefully):
- evidence.text MUST be an EXACT contiguous substring of the document.
- Copy the characters verbatim: preserve punctuation, capitalization,
  whitespace, and line breaks exactly as they appear.
- DO NOT paraphrase, summarize, shorten, reorder, add words, or "clean up"
  the text. Copy, do not describe.
- DO NOT concatenate phrases from different parts of the document.
- If you cannot find a verbatim substring that supports the item, DROP
  THE ITEM — do not invent a quote.

Confidence rubric:
- 0.95-1.0 = directly explicit factual statement strongly supported by quote
- 0.75-0.9 = clear explicit meaning, decision, or emotion
- 0.5-0.7 = paraphrased interpretation strongly supported
- <0.5 = weak ambiguity; usually omit

Language discipline:
- Do NOT use these words unless explicitly quoted or repeatedly evidenced:
  always, never, deepest, defining, ultimate, pattern, proves, chosen one,
  core, everyone, nothing.

DOCUMENT [{doc_id}]:
{content}\
"""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_prompt(doc: Document) -> str:
    date = doc.date or "unknown"
    title = doc.title or doc.id
    content = doc.content

    prompt = _TASK.format(
        title=title,
        date=date,
        topics=", ".join(_topics_for_run()),
        affect_registers=", ".join(AFFECT_REGISTERS),
        doc_id=doc.id,
        content=content,
    )

    combined_entries = doc.metadata.get("combined_entries") if doc.metadata else None
    if combined_entries:
        prompt += (
            f"\n\nBATCH NOTE — read carefully:\n"
            f"This document is a batch of {len(combined_entries)} INDEPENDENT "
            f"entries separated by `--- ENTRY: <date> | <title> [<doc_id>] ---` "
            f"lines. Each entry is its own context, written on a different "
            f"day. Mine each entry's durable items at the SAME density you "
            f"would if they were sent one at a time:\n"
            f"- The dedupe rule (\"prefer one strongest version\") applies "
            f"WITHIN an entry, not across entries. Two facts on different "
            f"dates are two facts, not one — even if they share wording. A "
            f"recurring theme repeated across entries is durable evidence of "
            f"a pattern; emit each occurrence with its own occurred_at.\n"
            f"- Set `occurred_at` to the per-entry date of the entry the fact "
            f"came from, not a span over all entries.\n"
            f"- Do NOT extract the `--- ENTRY: ... ---` separator lines "
            f"themselves; they are scaffolding.\n"
            f"- split_summaries MUST contain ONE element per inner entry, "
            f"not one for the whole batch. Use the entry's `[<doc_id>]` from "
            f"the separator line as the summary's `id` field, and summarize "
            f"only that entry's content. The order should follow the entries "
            f"as they appear in the document."
        )
    return prompt


def _find_inner_entry(combined_entries: list[dict], start: int) -> dict | None:
    """Return the combined-entries record whose [content_start, content_end)
    range contains `start`. Returns None when the offset falls on a separator
    or is otherwise out of range. O(n) — n is small (10s of entries per batch
    in practice; we don't index)."""
    for e in combined_entries:
        if e["content_start"] <= start < e["content_end"]:
            return e
    return None


def _resolve_span(text: str, chunk_content: str) -> tuple[int | None, int | None, bool]:
    """
    Find a quoted span in chunk content. Returns (start, end, approximate)
    where approximate=True when the match came from the longest-common-
    substring fallback rather than a literal/fuzzy substring match. Returns
    (None, None, False) if no match is possible.

    LLMs frequently return quotes that differ from the source by whitespace,
    smart quotes, or trimmed punctuation — and sometimes by outright
    paraphrase. We try progressively looser matches:

    1. Exact match (fast path, most quotes)
    2. Whitespace-normalized match (LLM collapsed runs of whitespace)
    3. First-sentence / prefix match (LLM truncated or added trailing chars)
    4. Longest-common-substring fallback (LLM paraphrased; anchor to the
       best-matching region so the reader still lands near the source)
    """
    if not text:
        return None, None, False

    # 1. Exact
    idx = chunk_content.find(text)
    if idx != -1:
        return idx, idx + len(text), False

    # Normalize smart quotes / whitespace before comparing
    def _norm(s: str) -> str:
        s = s.replace("\u2018", "'").replace("\u2019", "'")
        s = s.replace("\u201c", '"').replace("\u201d", '"')
        return re.sub(r"\s+", " ", s).strip()

    # 2. Whitespace-normalized: find the normalized quote in a whitespace
    # collapsed chunk, then map back to a chunk offset using a running index.
    norm_text = _norm(text)
    if not norm_text:
        return None, None, False

    # Build a map from normalized-chunk positions → original-chunk positions.
    norm_chunk_chars: list[str] = []
    pos_map: list[int] = []  # pos_map[i] = original offset of the i-th normalized char
    prev_ws = False
    for i, ch in enumerate(chunk_content):
        if ch.isspace():
            if not prev_ws:
                norm_chunk_chars.append(" ")
                pos_map.append(i)
                prev_ws = True
            continue
        prev_ws = False
        # Replace smart quotes
        if ch in ("\u2018", "\u2019"):
            ch = "'"
        elif ch in ("\u201c", "\u201d"):
            ch = '"'
        norm_chunk_chars.append(ch)
        pos_map.append(i)

    norm_chunk_full = "".join(norm_chunk_chars)
    n_idx = norm_chunk_full.find(norm_text)
    if n_idx != -1 and n_idx + len(norm_text) <= len(pos_map):
        start = pos_map[n_idx]
        end_norm_idx = n_idx + len(norm_text) - 1
        if end_norm_idx < len(pos_map):
            end = pos_map[end_norm_idx] + 1
            if start < end <= len(chunk_content):
                return start, end, False

    # 3. Prefix match: try finding just the first 40 chars of the quote.
    # Useful when the LLM added a trailing ellipsis or paraphrased the end.
    prefix = text[:40].strip()
    if prefix:
        pidx = chunk_content.find(prefix)
        if pidx != -1:
            # Match extends as long as the original text — best effort.
            end = min(len(chunk_content), pidx + len(text))
            return pidx, end, False

    # 4. Fuzzy-locate fallback. The LLM paraphrased, but a distinctive
    # sub-phrase (a name, a number, a rare noun) may still appear
    # verbatim in the chunk. Require the shared region to be meaningful:
    # at least 12 chars AND at least a quarter of the quote length, so
    # short matches don't anchor to single common words like "the team".
    #
    # chunk_content is bounded by the splitter's chunk_in_cap (~31k tokens
    # on TEE/LOCAL ≈ 93k chars), so the pure-Python difflib path is O(n*m) ≈ 47M ops with
    # text=500 chars — still seconds per call, but those seconds multiply
    # across parallel chunks because _resolve_span runs under the GIL
    # (thread-based parallelism doesn't help pure-Python work). On a
    # journal JSON with ~45 extracted items × 3 evidence spans × 5
    # parallel chunks, this was producing ~11 minutes of CPU-bound work
    # AFTER all LLM calls had returned.
    #
    # rapidfuzz.fuzz.partial_ratio_alignment does the same find-best-
    # overlap job in C and scales much better; we keep the difflib path
    # as a fallback for environments where rapidfuzz isn't installed.
    min_size = max(12, len(text) // 4)
    if _HAS_RAPIDFUZZ:
        align = _rf_fuzz.partial_ratio_alignment(text, chunk_content)
        # align.score is 0-100. The default scorer's own normalized score
        # threshold of ~75 would reject many paraphrases we still want to
        # anchor; use the original min_size rule on the matched region
        # instead, which matches the difflib-path's meaning.
        match_len = align.dest_end - align.dest_start
        if match_len >= min_size:
            start = align.dest_start
            span_len = max(match_len, min(len(text), 200))
            end = min(len(chunk_content), start + span_len)
            return start, end, True
    else:
        matcher = SequenceMatcher(None, chunk_content, text, autojunk=False)
        match = matcher.find_longest_match(0, len(chunk_content), 0, len(text))
        if match.size >= min_size:
            start = match.a
            span_len = max(match.size, min(len(text), 200))
            end = min(len(chunk_content), start + span_len)
            return start, end, True

    return None, None, False


def _halve_doc_content(content: str) -> tuple[str, str, int] | None:
    """Split `content` near its midpoint, preferring a paragraph or
    sentence boundary within ±10% of the center. Returns
    (first_half, second_half, split_offset). Returns None when
    content is too short to halve usefully (< 200 chars — the LLM
    would produce comparable output on the whole thing as on either
    half).

    Used by `_extract_with_halving` on a work-reducing trigger. The
    boundary preference (paragraph > line > sentence > raw midpoint)
    keeps each half a coherent slice for the LLM to extract from
    rather than bisecting mid-token."""
    n = len(content)
    if n < 200:
        return None
    center = n // 2
    window = max(40, n // 10)
    lo = max(0, center - window)
    hi = min(n, center + window)
    for sep in ("\n\n", "\n", ". "):
        idx_r = content.find(sep, center, hi)
        idx_l = content.rfind(sep, lo, center)
        candidates = []
        if idx_r != -1:
            candidates.append(idx_r + len(sep))
        if idx_l != -1:
            candidates.append(idx_l + len(sep))
        if candidates:
            split = min(candidates, key=lambda c: abs(c - center))
            return content[:split], content[split:], split
    return content[:center], content[center:], center


# Set by runner._patch_llm_calls so extract timings land in run.log.
_runner_log = None


def _log_info(msg: str) -> None:
    """Runner-aware logger. Writes to run.log via the injected hook, or
    falls back to stdout when run outside the pipeline (tests, CLI)."""
    if _runner_log is not None:
        try:
            _runner_log(msg)
            return
        except Exception:
            pass
    _safe_print(msg)


def _safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        pass


# ── Parsing ───────────────────────────────────────────────────────────────────

def _salvage_truncated_json(raw: str) -> str | None:
    """
    Truncation salvage for either response shape: the envelope
    `{"split_summaries": [...], "items": [...]}` or a legacy bare array
    `[...]`. Trims to the last complete element and adds the closing
    punctuation so we keep the items that DID come through when the
    LLM response is cut off mid-JSON (max_tokens, streaming hiccup).

    Returns the salvaged string (parseable as JSON) or None if no
    salvage is possible.
    """
    stripped = raw.lstrip()
    if not stripped:
        return None
    if stripped.startswith("["):
        return _salvage_bare_array(raw)
    if stripped.startswith("{"):
        return _salvage_envelope(raw)
    return None


def _salvage_bare_array(raw: str) -> str | None:
    """Salvage a truncated bare-array response. Looks for the last `}`
    that sits at array nesting depth 1; truncates there and appends
    `]`. Kept for cases where the LLM emits the legacy shape against
    the envelope-required prompt — degraded (no summaries) but still
    yields the surviving items."""
    depth = 0
    in_string = False
    escape = False
    last_complete_close = -1

    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[" or ch == "{":
            depth += 1
        elif ch == "]" or ch == "}":
            depth -= 1
            if ch == "}" and depth == 1:
                last_complete_close = i

    if last_complete_close < 0:
        return None
    return raw[: last_complete_close + 1] + "]"


def _salvage_envelope(raw: str) -> str | None:
    """Salvage a truncated envelope response. Locates the `items` array's
    opening `[`, then trims items to the last complete element (the
    deepest `}` at the items-array's element-depth) and emits a
    well-formed envelope with the surviving items and whatever
    `split_summaries` content sits before the items array.

    Truncation almost never lands inside split_summaries because the
    prompt asks the model to emit summaries first and they are short.
    When it does, we lose summaries on that one call but still emit
    parseable JSON via the bracket-balance fallback at the tail."""
    items_idx = raw.find('"items"')
    if items_idx == -1:
        # No items array opened yet — try a generic balance close.
        return _balance_close(raw)

    i = items_idx + len('"items"')
    while i < len(raw) and raw[i] != '[':
        i += 1
    if i >= len(raw):
        return _balance_close(raw)
    items_bracket = i

    depth = 1  # depth relative to items array's opening `[`
    in_string = False
    escape = False
    last_element_close = -1

    for j in range(items_bracket + 1, len(raw)):
        ch = raw[j]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if ch == "}" and depth == 1:
                last_element_close = j
            elif ch == "]" and depth == 0:
                # items array closed cleanly; close the envelope.
                return raw[: j + 1] + "}"

    if last_element_close < 0:
        # Truncated before any item finished. Emit envelope with empty items.
        return raw[: items_bracket + 1] + "]}"
    return raw[: last_element_close + 1] + "]}"


def _balance_close(raw: str) -> str | None:
    """Last-resort balance close: walk raw tracking quotes + brackets,
    append closing punctuation to make the result parseable. The
    resulting JSON is best-effort and may have empty / missing fields,
    but it never raises in `json.loads`. Returns None when the input
    is genuinely empty."""
    stripped = raw.rstrip()
    if not stripped:
        return None
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in stripped:
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if stack:
                stack.pop()
    out = stripped
    if in_string:
        out += '"'
    while stack:
        opener = stack.pop()
        out += "]" if opener == "[" else "}"
    return out


def _make_doc_parser(doc: Document):
    """Build the extract stage's wrapper parser callable for one doc.

    The retry wrapper calls the returned callable on each attempt's
    raw response to decide pass / `_ParseError` / `_SuccessEmpty`
    BEFORE the structured shape extraction in `on_success`. It parses
    via the SAME `_parse_items` the success path uses, so the
    retry-side empty check and the display-side output count
    (`output={"facts": len(items)}`) read one number and can't
    disagree — the prior split between a context-free validator and
    the doc-bound parser is what let extract record a parsed-empty
    response as plain success while the classifier labelled it
    `success_empty`.

    Raises:
      - `_SuccessEmpty(stage="extract")` when the response cleanly
        parses to zero usable items: whitespace-only raw, bare `[]`,
        `{}`, or the schema's `{split_summaries, items: []}` envelope.
        Per the universal empty-after-parse rule, produce-nothing is
        never silent — it eats one Other retry (reasoning forced off)
        before settling as `success_empty`. True wire-empty is caught
        upstream by `_raise_if_empty_on_wire`.
      - `_ParseError(stage="extract")` when raw is non-empty but
        unparseable after `_parse_items`' truncation salvage (e.g. a
        fence opener with no JSON body) — routes to the sizing cascade.

    Returns `raw` unchanged otherwise; the offset / evidence-span /
    inner-entry shape work happens downstream in `_parse_items` (it
    needs the doc context carried here)."""
    from engine.retry import _ParseError, _SuccessEmpty

    def _parser(raw: str) -> str:
        if not (raw or "").strip():
            raise _SuccessEmpty(stage="extract")
        items, _summaries, parse_err = _parse_items(_strip_fences(raw), doc)
        if parse_err:
            raise _ParseError(stage="extract")
        if not items:
            raise _SuccessEmpty(stage="extract")
        return raw

    return _parser


def _parse_items(
    raw: str, doc: Document,
) -> tuple[list[ExtractedItem], dict[str, str], bool]:
    """Parse an extract response into (items, summaries, parse_error).

    The wire format is the envelope
    `{"split_summaries": [{"id", "summary"}, ...], "items": [...]}` —
    summaries first so they survive truncation salvage. Legacy
    bare-array responses (`[...]`) are accepted with summaries={} for
    robustness; the envelope shape is what the prompt asks for.

    `summaries` maps split id → summary text. For non-batch docs it's
    `{doc.id: "..."}` (1 entry) when the model complied. For batched
    docs (combined_entries metadata) it maps each inner-entry id to
    its summary. Unknown / mismatching ids that don't appear in the
    doc or its inner entries are still returned — the caller decides
    whether to keep them.

    `parse_error=True` when raw was non-empty but unparseable (and
    salvage didn't recover JSON)."""
    stripped = _strip_fences(raw)
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        # Response likely truncated mid-item. Salvage what we can and log
        # so we can tell how often this happens (signal for raising the
        # output budget or shrinking input chunks).
        salvaged = _salvage_truncated_json(stripped)
        if salvaged is None:
            _log_info(f"  [extractor] truncated response, no salvage possible "
                      f"(doc={doc.id}, chars={len(stripped)})")
            return [], {}, True
        try:
            data = json.loads(salvaged)
        except (json.JSONDecodeError, ValueError):
            _log_info(f"  [extractor] truncated response, salvage parse failed "
                      f"(doc={doc.id}, chars={len(stripped)})")
            return [], {}, True
        _log_info(f"  [extractor] TRUNCATED response salvaged "
                  f"(doc={doc.id}, raw_chars={len(stripped)}, "
                  f"salvaged_chars={len(salvaged)})")

    summaries: dict[str, str] = {}
    if isinstance(data, dict):
        raw_summaries = data.get("split_summaries", [])
        if isinstance(raw_summaries, list):
            for s in raw_summaries:
                if not isinstance(s, dict):
                    continue
                sid = str(s.get("id", "")).strip() or doc.id
                stext = str(s.get("summary", "")).strip()
                if stext:
                    # Last write wins on duplicates — rare; prompt asks for
                    # one per id. Halve-call merging happens upstream in
                    # extract_items, not here.
                    summaries[sid] = stext
        items_data = data.get("items", [])
        if not isinstance(items_data, list):
            _log_info(
                f"  [extractor] envelope `items` is not a list "
                f"(doc={doc.id}, type={type(items_data).__name__})"
            )
            items_data = []
    elif isinstance(data, list):
        # Prompt-non-compliant response: bare array of items, no
        # summaries. Accept gracefully to avoid a needless parse-error
        # retry.
        items_data = data
    else:
        _log_info(f"  [extractor] parsed JSON is not envelope or list "
                  f"(doc={doc.id}, type={type(data).__name__}, "
                  f"preview={str(data)[:200]!r})")
        return [], {}, True

    results: list[ExtractedItem] = []
    # Resolve the active topic taxonomy once per response — config read +
    # set build happens here, not inside the inner loop.
    allowed_topics = set(_topics_for_run())
    for item in items_data:
        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type", "")).strip().lower()
        if item_type not in ITEM_TYPES:
            continue

        summary = str(item.get("summary", "")).strip()
        if not summary:
            continue

        # Evidence spans — resolve doc-local offset then add doc.origin_char
        # to land at a position in the preprocessed source file. For batched
        # combined Documents, route attribution to the inner entry whose
        # range contains the resolved span (see splitter.combine_documents).
        combined_entries = doc.metadata.get("combined_entries") or []
        evidence: list[EvidenceSpan] = []
        for ev in item.get("evidence", []):
            if not isinstance(ev, dict):
                continue
            text = str(ev.get("text", "")).strip()
            if not text:
                continue
            source_ref = str(ev.get("source_ref", doc.id)).strip() or doc.id
            start, end, approximate = _resolve_span(text, doc.content)

            file_offset: int | None = None
            file_length: int | None = None
            file_path = doc.file_id or doc.source_path
            if start is not None and end is not None:
                file_length = end - start
                if combined_entries:
                    inner = _find_inner_entry(combined_entries, start)
                    if inner is not None:
                        file_path = inner["file_id"]
                        file_offset = inner["origin_char"] + (start - inner["content_start"])
                        # Override the LLM's source_ref with the inner
                        # entry id when we can attribute it; otherwise
                        # keep what the LLM emitted (the batch id).
                        source_ref = inner["id"]
                    else:
                        # Span landed on a separator (very rare). Drop the
                        # file_offset rather than report a misleading one.
                        file_offset = None
                else:
                    file_offset = doc.origin_char + start

            evidence.append(EvidenceSpan(
                text=text,
                source_ref=source_ref,
                start_char=start,
                end_char=end,
                file_path=file_path,
                file_offset=file_offset,
                file_length=file_length,
                approximate=approximate,
            ))

        # Entities
        entities: list[EntityRef] = []
        for ent in item.get("entities", []):
            if not isinstance(ent, dict):
                continue
            name = str(ent.get("name", "")).strip()
            if not name:
                continue
            entity_type = str(ent.get("entity_type", "other")).strip().lower()
            if entity_type not in ENTITY_TYPES:
                entity_type = "other"
            role = str(ent.get("role", "mentioned")).strip().lower()
            if role not in ROLES:
                role = "mentioned"
            entities.append(EntityRef(
                entity=Entity(name=name, entity_type=entity_type),
                role=role,
            ))

        occurred_at_text = item.get("occurred_at_text") or None
        occurred_at = item.get("occurred_at") or None
        if occurred_at:
            occurred_at = str(occurred_at).strip() or None

        topics = [str(t).strip() for t in item.get("topics", []) if str(t).strip() in allowed_topics]
        tags = [str(t).strip() for t in item.get("tags", []) if str(t).strip()]

        # Affect registers — validate against the fixed taxonomy, lowercase,
        # dedupe preserving order, cap at _MAX_AFFECT. A non-list value (model
        # emitted a bare string or null) collapses to empty rather than
        # iterating character-by-character. Empty is the expected neutral case.
        raw_affect = item.get("affect", [])
        affect: list[str] = []
        if isinstance(raw_affect, list):
            for a in raw_affect:
                reg = str(a).strip().lower()
                if reg in _AFFECT_REGISTER_SET and reg not in affect:
                    affect.append(reg)
        affect = affect[:_MAX_AFFECT]

        try:
            confidence = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        confidence = max(0.0, min(1.0, confidence))

        relation_candidate: dict | None = None
        rc_raw = item.get("relation_candidate")
        if isinstance(rc_raw, dict):
            rc_from = str(rc_raw.get("from", "")).strip()
            rc_to = str(rc_raw.get("to", "")).strip()
            rc_verb = str(rc_raw.get("verb", "")).strip()
            if rc_from and rc_to and rc_from != rc_to:
                try:
                    rc_conf = float(rc_raw.get("confidence", 1.0))
                except (TypeError, ValueError):
                    rc_conf = 1.0
                rc_conf = max(0.0, min(1.0, rc_conf))
                relation_candidate = {
                    "from": rc_from,
                    "to": rc_to,
                    "verb": rc_verb,
                    "confidence": rc_conf,
                }

        results.append(ExtractedItem(
            item_type=item_type,
            summary=summary,
            evidence=evidence,
            occurred_at=occurred_at,
            occurred_at_text=occurred_at_text,
            entities=entities,
            topics=topics,
            affect=affect,
            tags=tags,
            confidence=confidence,
            relation_candidate=relation_candidate,
        ))

    return results, summaries, False


# ── Public API ────────────────────────────────────────────────────────────────
