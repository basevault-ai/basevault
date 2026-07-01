"""
Patterns — merges grouping (was: compressor) and within-theme synthesis
(was: pattern_detector) into a single LLM pass per topic.

Takes Facts for one topic and returns Pattern objects. A Pattern names a
durable concept visible across multiple facts — a constraint, a fear, a
goal, a principle, a behavior, or an untyped theme. Patterns at this
stage carry NO mechanism / implication / action; those belong to the
Insights stage where cross-topic synthesis happens.

Pre-pass:
  Deterministic dedup by normalized summary — cheap, removes obvious
  repetition before the LLM sees anything.

Output:
  List of Pattern per topic with name, description, domain, kind, count,
  source_facts, hallucinated_ref_count.

Usage:
    from engine.patterns import detect_patterns, Pattern
    patterns = detect_patterns(facts, topic="health", mode=Mode.TEE)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from engine.content_extractor import ExtractedItem
from engine.llm import (
    chunk_cap_for_stage as _llm_chunk_cap_for_stage,  # noqa: F401 (re-export for phases.patterns)
    strip_fences as _strip_fences,
)
from engine.tokens import estimate_prompt_tokens as _llm_estimate_prompt_tokens


# Set by runner._patch_llm_calls so per-stage exception tracebacks land
# in run.log alongside the wrapper-level logging. Stays None when the
# module runs outside the runner (tests, CLI, scripts) — `_log_info`
# falls back to stdout.
_runner_log = None


def _log_info(msg: str) -> None:
    """Runner-aware logger. Mirrors content_extractor._log_info shape."""
    if _runner_log is not None:
        try:
            _runner_log(msg)
            return
        except Exception:
            pass
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        pass


def _log_exception(stage: str, ctx: str, exc: BaseException) -> None:
    """Log a stage-level exception with full traceback. Used at every
    `except Exception` site near a `complete()` call so root cause stays
    visible — the silent fallbacks here have been throwing exception
    data away for months."""
    import traceback
    head = f"  [{stage}] LLM call raised ({ctx}): {type(exc).__name__}: {exc}"
    _log_info(head)
    for line in traceback.format_exc().rstrip().splitlines():
        _log_info(f"    {line}")


# ── IR ────────────────────────────────────────────────────────────────────────

PATTERN_KINDS = [
    "principle", "fear", "behavior", "goal", "constraint",
]


@dataclass
class Pattern:
    name: str
    description: str
    domain: str                       # topic (e.g. "health", "work")
    kind: str | None = None           # one of PATTERN_KINDS, or None (untyped)
    count: int = 1                    # rough recurrence count across facts
    # Weighted refs into facts_by_topic[domain]. (index, confidence).
    # Sorted by confidence desc.
    source_facts: list[tuple[int, float]] = field(default_factory=list)
    # Number of fact IDs the LLM returned that were out of range.
    hallucinated_ref_count: int = 0


# ── Defaults ──────────────────────────────────────────────────────────────────



# ── Pre-pass: deterministic dedup ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _dedup_facts(facts: list[ExtractedItem]) -> tuple[list[ExtractedItem], list[list[int]]]:
    """Drop exact duplicates by normalized summary. Keeps first occurrence.
    Returns (deduped, orig_indices) where orig_indices[i] is the list of
    original indices that map to deduped[i]."""
    seen: dict[str, int] = {}
    out: list[ExtractedItem] = []
    orig: list[list[int]] = []
    for i, f in enumerate(facts):
        key = _normalize(f.summary)
        if not key:
            continue
        if key in seen:
            orig[seen[key]].append(i)
            continue
        seen[key] = len(out)
        out.append(f)
        orig.append([i])
    return out, orig


# ── Adaptive cap ──────────────────────────────────────────────────────────────

# Hard ceiling on patterns per topic regardless of corpus size. Above this,
# downstream stages start blurring rather than synthesizing — and a single
# topic with >75 patterns is almost always a sign the topic should have
# been split. Worth revisiting once we see how Pepys's densest topics
# behave under the new formula.
_PATTERN_CAP_CEILING = 75


def _cap(n_facts: int) -> int:
    """Sub-linear cap on pattern count. n^0.6 picked as the empirical
    middle between sqrt (too conservative on rich topics) and cube root
    (over-extracts). No floor — degenerate small topics legitimately
    have fewer patterns; the prompt already says "emit zero if no
    recurrence." Reference points: 100→16, 500→42, 1000→63, ceiling
    hit around 1500 facts."""
    if n_facts <= 0:
        return 0
    return min(_PATTERN_CAP_CEILING, round(n_facts ** 0.6))


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You synthesize durable patterns from a list of grounded facts about a \
single person within one life domain. Be aggressive in merging. Drop \
trivia. A pattern names a concept that is visible across multiple facts — \
a constraint, a fear, a goal, a principle, a behavior, or an untyped \
recurring theme.

Output format hard rules (CRITICAL — violations break downstream parsing):
- Respond with a single JSON array. Nothing else.
- DO NOT emit chain-of-thought, reasoning, analysis, or explanation.
- DO NOT prefix the output with phrases like "Let me", "I'll", "Here is", "Analyzing", etc.
- DO NOT narrate the domain before the JSON.
- The very first character of your response MUST be `[`.
- The very last character of your response MUST be `]`.
- Use ONLY the fields named in the schema below. Do not add any other
  fields. If you would naturally include extra metadata, omit it.

Content hard rules:
- DO NOT list atomic facts one by one.
- DO NOT summarize the domain itself.
- DO NOT restate facts in cleaner language.
- Name patterns the way a thoughtful friend or therapist would — plain
  language, common human tendencies. Avoid novel or abstract formulations
  that are not grounded in the source material.
- Source fidelity: if source facts name an operational threshold (cadence,
  time limit, count, frequency — "daily", "72h", "only 1 at a time"),
  carry it into the pattern EXACTLY. Do not soften thresholds.
- Source fidelity: if source facts name a structure (gate, prerequisite,
  sequence, dependency, bottleneck), preserve it in the pattern.

Two-subject conflation — negative example (DO NOT emit patterns shaped like
this):
  BAD: {"name": "Avoids confrontation in close relationships",
        "description": "John and Alex both withdraw when conflict escalates;
        he processes it alone before returning."}
Why bad: the description binds two distinct named parties ("John and Alex")
into one "he", then asserts a behavior as if the pronoun resolves to a single
subject. This is the exact failure mode — when two people appear in the facts,
the pattern must attribute behavior to ONE subject and treat the other as
context. Pronouns resolve to the subject only.\
"""


_SUBJECT_DISCIPLINE = """\
Subject discipline — the subject of these patterns is {subject}:
- Every pattern must describe {subject}'s behavior, decisions, emotional
  loops, constraints, or identity-level drivers — NOT other parties'
  traits, strategy, or intent.
- Other parties in the source may appear as context, events, or
  constraints that {subject} operates within. They may NOT appear as the
  driver of a pattern unless there is explicit repeated evidence across
  multiple distinct contexts/times that the behavior is genuinely theirs
  and materially shapes {subject}'s loops.
- A single message, one-line question, or short exchange from a non-subject
  party is a quoted fact — not a window into that party's psyche.

Core invariant: a claim may not be more stable, strategic, or identity-level
than the evidence supporting it. A one-off event is a fact. A short sequence
is an event. Only repetition across time or contexts earns a pattern.

Pre-emit pronoun check (required before finalizing the JSON):
  Scan each output item — name and description — for every pronoun
  (he / she / they / his / her / their / him / them / himself /
  herself / themselves). Each MUST resolve unambiguously to {subject}.
  If a pronoun could be read as referring to any other named person
  in the input facts, rewrite the sentence to name {subject}
  explicitly OR reframe so the pronoun disappears. If a pronoun
  genuinely has no plausible resolution to {subject}, the pattern is
  about the wrong party — drop it.\
"""


_EVENT_VS_PATTERN = """\
Event vs pattern — earn recurrence:
- A pattern requires ONE of:
    (a) recurrence across TIME — the same behavior/concept seen on multiple
        occasions separated in time
    (b) recurrence across CONTEXTS — the same behavior/concept seen in
        distinct situations, domains, or with distinct counterparties
    (c) repeated signal from DISTINCT FACT CLUSTERS — multiple separate
        groups of facts converging on the same underlying concept
- The following are NOT patterns and must NOT be promoted to a pattern
  even if the framing sounds compelling:
    · a single fact
    · a short sequence within one relationship, project, or counterparty
      context (two sequential events in one trajectory are a sequence,
      not recurrence)
    · one-off logistics or coordination exchanges
    · a single speaker describing a past life state once
- When deciding whether to emit a pattern, ask: "If I removed this specific
  episode, would the pattern still be visible in the remaining facts?"
  If no, it's an event, not a pattern.
- Fabricated recurrence is worse than empty output, BUT defaulting to
  empty when real recurrence IS visible is also a failure — not a
  neutral choice. If the domain's facts converge on a real behavior or
  concept via time, context, or distinct fact clusters, emit the
  best-grounded pattern you can defend. Emit zero ONLY when the domain
  holds only single events with no recurring signal — not as a safety
  default when unsure.\
"""


_TASK = """\
You are given a list of grounded facts from a single domain: {domain}

Each fact has a numeric ID in square brackets like [1], [2], etc. You
must reference these IDs in your output so we can trace which facts
support each pattern.

SYNTHESIZE patterns that are visible across multiple facts.

Goal:
- Name durable concepts (constraints, fears, goals, principles, behaviors,
  or untyped themes) that the facts converge on
- Preserve only pattern-relevant information
- Each pattern must be supported by at least 2 distinct facts; singletons
  stay at the Facts stage and do NOT become patterns

`kind` field — one of principle, fear, behavior, goal, constraint, OR null.
Use null when the pattern does not cleanly fit one of the five. Do NOT
invent a kind to satisfy the schema.

- principle  : an enduring rule, value, or self-claim the subject holds
- fear       : a recurring dread, avoidance loop, or vulnerability the
               subject experiences
- behavior   : a recurring action pattern — what the subject tends to do
- goal       : a durable aim the subject is pursuing
- constraint : a limit the subject operates within (time, attention,
               money, health, relational bandwidth)
- null       : the pattern is a durable recurring theme but does not fit
               any of the above cleanly (e.g. "Frequent NYC travel")

Output size:
- Hard cap: {hard_cap} patterns. Stop at the cap; do NOT aim for it.
  The cap is a ceiling, not a target. Emit every pattern that meets
  the recurrence rule below, in any order, until either the patterns
  run out or you hit the cap.
- Recurrence rule: a pattern requires ≥2 distinct supporting facts
  (see Event vs pattern above). Singletons stay at the Facts stage.
- Err toward inclusion when recurrence is visible: if ≥2 facts converge
  on the same behavior, constraint, or theme, emit the pattern — even
  if the name feels obvious or ordinary. Plain-language patterns that
  accurately name real recurrence are the goal, not terse omission.
- Emit zero patterns for this topic ONLY when every fact in the topic
  is a single unrelated event with no shared behavior or concept. Do
  NOT emit zero as a hedge when the facts clearly cluster.

Respond with ONLY a valid JSON array. Each element:
{{
  "name": "<short pattern name, 2-6 words>",
  "description": "<1-2 line description of the pattern>",
  "kind": "principle|fear|behavior|goal|constraint" or null,
  "count": <integer — rough count of facts supporting this pattern>,
  "sources": [
    {{"id": <integer ID from INPUT FACTS>, "confidence": <float 0.0-1.0>}},
    ...
  ]
}}

`sources` lists every input fact this pattern draws from:
- id: the [N] ID shown in INPUT FACTS (only use IDs you actually see)
- confidence: 1.0 = central pillar; 0.5 = corroborating detail; don't
  invent numbers — reflect real attribution
- at least 2 sources required per pattern

List highest-confidence sources first.

INPUT FACTS ({n_facts} total):
{facts_text}\
"""


def _build_prompt(facts: list[ExtractedItem], domain: str, hard_cap: int) -> str:
    lines: list[str] = []
    for i, f in enumerate(facts, start=1):
        date = f.occurred_at or "?"
        ents = ", ".join(r.entity.name for r in f.entities) if f.entities else ""
        ent_suffix = f" [{ents}]" if ents else ""
        lines.append(f"[{i}] ({f.item_type}, {date}){ent_suffix} {f.summary}")
    facts_text = "\n".join(lines)

    return _TASK.format(
        domain=domain,
        hard_cap=hard_cap,
        n_facts=len(facts),
        facts_text=facts_text,
    )


def _parse_patterns(
    raw: str,
    domain: str,
    dedup_to_orig: list[list[int]],
) -> tuple[list[Pattern], bool]:
    """Parse LLM output into Patterns, translating prompt-local 1-indexed
    source_ids back to original facts_by_topic[domain] indices.

    Returns (patterns, parse_error). `parse_error` is True if the
    response couldn't be JSON-parsed at all. Patterns whose entire
    citation list resolved to nothing (every cited ID hallucinated)
    are skipped silently — there is no observable pattern there to
    surface. Singletons and multi-source patterns alike survive;
    downstream stages weight by `len(source_facts)`."""
    try:
        data = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, ValueError):
        return [], True

    if not isinstance(data, list):
        return [], True

    n_deduped = len(dedup_to_orig)
    out: list[Pattern] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        description = str(entry.get("description", "")).strip()
        try:
            count = int(entry.get("count", 1))
        except (TypeError, ValueError):
            count = 1

        raw_kind = entry.get("kind")
        if raw_kind is None:
            kind: str | None = None
        else:
            k = str(raw_kind).strip().lower()
            kind = k if k in PATTERN_KINDS else None

        # Resolve sources — list of {id, confidence}. Translate prompt-local
        # 1-indexed IDs through dedup map back to original fact indices.
        raw_sources = entry.get("sources") or []
        if not isinstance(raw_sources, list):
            raw_sources = []
        resolved: dict[int, float] = {}
        hallucinated = 0
        for s in raw_sources:
            if isinstance(s, dict):
                sid = s.get("id")
                conf = s.get("confidence", 1.0)
            else:
                sid = s
                conf = 1.0
            try:
                idx = int(sid) - 1
                conf = float(conf)
            except (TypeError, ValueError):
                hallucinated += 1
                continue
            if not (0 <= idx < n_deduped):
                hallucinated += 1
                continue
            conf = max(0.0, min(1.0, conf))
            for orig in dedup_to_orig[idx]:
                if orig not in resolved or resolved[orig] < conf:
                    resolved[orig] = conf

        source_facts = sorted(
            resolved.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )

        # Drop patterns whose entire citation list resolved to nothing
        # (every cited ID was out-of-range / non-numeric — fully
        # hallucinated refs). Singletons survive; insights and actions
        # see `len(source_facts)` per pattern and weight accordingly.
        if not source_facts:
            continue

        out.append(Pattern(
            name=name,
            description=description,
            domain=domain,
            kind=kind,
            count=count,
            source_facts=source_facts,
            hallucinated_ref_count=hallucinated,
        ))

    return out, False


# ── Overflow handling ────────────────────────────────────────────────────────

def _build_messages(
    facts: list[ExtractedItem],
    topic: str,
    hard_cap: int,
    subject: str,
    entities_context: str | None,
) -> list[dict]:
    """Materialize the full chat-message payload (system + user) used both
    for the LLM call and for size estimation. Single source of truth so
    sampling math agrees with the actual call."""
    prompt = _build_prompt(facts, topic, hard_cap)
    if entities_context:
        prompt = entities_context + "\n\n" + prompt
    system_content = (
        _SYSTEM
        + "\n\n" + _SUBJECT_DISCIPLINE.format(subject=subject)
        + "\n\n" + _EVENT_VS_PATTERN
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]


def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    """Pick k indices from range(n) evenly spaced, with the last index
    (most recent fact in chronologically-ordered input) always included.
    Returns sorted ascending. Centered stride preserves diversity across
    long timelines instead of clustering at the start."""
    if k >= n:
        return list(range(n))
    if k <= 0:
        return []
    if k == 1:
        return [n - 1]
    idx = sorted({int((i + 0.5) * n / k) for i in range(k)})
    if idx and idx[-1] != n - 1:
        idx[-1] = n - 1
    return idx


def _sample_to_fit(
    deduped: list[ExtractedItem],
    topic: str,
    hard_cap: int,
    subject: str,
    entities_context: str | None,
    cap: int,
) -> list[int]:
    """Return indices into `deduped` whose prompt fits under `cap`. When
    the full set fits, returns range(len(deduped)) (no sampling). When it
    doesn't, binary-searches the largest K such that K facts (evenly
    spaced + most-recent included) fit. Returns sorted ascending so the
    sampled facts are presented to the LLM in input/chronological
    order."""
    def _est(idx: list[int]) -> int:
        sample = [deduped[i] for i in idx]
        msgs = _build_messages(sample, topic, hard_cap, subject, entities_context)
        return _llm_estimate_prompt_tokens(msgs)

    full = list(range(len(deduped)))
    if _est(full) <= cap:
        return full

    # At least 2 facts required by parser. Binary-search [2, n].
    lo, hi = 2, len(deduped)
    best = 2
    while lo <= hi:
        mid = (lo + hi) // 2
        idx = _evenly_spaced_indices(len(deduped), mid)
        if _est(idx) <= cap:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return _evenly_spaced_indices(len(deduped), best)


# ── Public API ────────────────────────────────────────────────────────────────



def _confidence_prioritized_halving(
    sampled: list[ExtractedItem],
    sampled_to_orig: list[list[int]],
) -> tuple[list[ExtractedItem], list[list[int]]]:
    """Issue #105 v3 patterns work-reducer: drop the lowest-confidence
    half of `sampled`, keeping the dedup_to_orig mapping in sync.

    Patterns has unbounded INPUT (thousands of facts per topic) but
    bounded output. On a work-reducing trigger we don't fan out (the
    output is bounded — 2 calls don't help; the bottleneck is the
    model holding too many facts in context). Instead we discard the
    cheapest-to-lose half and retry. Confidence prioritization means
    the high-signal facts survive the cut: a 0.9-confidence health
    insight beats a 0.4-confidence aside.

    Returns (new_sampled, new_sampled_to_orig). Both halve in step
    so the dedup_to_orig mapping the parser uses to translate
    prompt-local IDs back to original facts stays consistent."""
    if len(sampled) < 4:
        # Nothing useful to halve; the parser already requires ≥2
        # source-facts per pattern.
        return sampled, sampled_to_orig
    # Sort indices by descending confidence; keep top half.
    keep_count = max(2, len(sampled) // 2)
    indices = sorted(
        range(len(sampled)),
        key=lambda i: -float(getattr(sampled[i], "confidence", 1.0) or 1.0),
    )[:keep_count]
    indices.sort()  # restore chronological order so the prompt reads naturally
    new_sampled = [sampled[i] for i in indices]
    new_to_orig = [sampled_to_orig[i] for i in indices]
    return new_sampled, new_to_orig


def _validate_patterns_or_raise(raw: str) -> str:
    """Parser callable for the patterns wrapper. Raises
    `_EmptyResponse` on whitespace-only output, `_ParseError` on
    non-empty but unparseable JSON, and `_SuccessEmpty` on a parseable
    JSON array that contains zero patterns. Returns raw unchanged
    otherwise."""
    from engine.parse_signals import _EmptyResponse, _ParseError, _SuccessEmpty
    if not (raw or "").strip():
        raise _EmptyResponse(stage="patterns")
    try:
        data = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, ValueError):
        raise _ParseError(stage="patterns")
    if isinstance(data, list) and not data:
        raise _SuccessEmpty(stage="patterns")
    return raw






def build_context_block(
    patterns_by_topic: dict[str, list[Pattern]],
    detail_keys: set[tuple[str, str]] | None = None,
) -> str:
    """``Patterns reference`` block injected into the Actions prompt so
    action selection can ground in the underlying patterns, not just
    the insight summaries on top of them. Empty input → empty string.

    Per-topic grouping; every pattern is rendered (no per-topic cap,
    no description truncation). Per-pattern row: kind + name + count +
    facts (= number of distinct supporting facts) + description. The
    facts annotation is the salience signal — patterns with higher
    fact counts represent stronger recurrence; singletons may inform
    but shouldn't dominate.

    When `detail_keys` is set (the WR-sample cascade's escape valve),
    only patterns whose (topic, name) appears in the set render the
    description suffix — all other patterns still appear by name +
    count + facts so the model never loses sight of an entry, just
    its detail. `None` (default) means every pattern gets full
    detail.
    """
    if not patterns_by_topic:
        return ""

    lines: list[str] = ["Patterns reference:"]
    for topic in sorted(patterns_by_topic.keys()):
        pats = patterns_by_topic[topic] or []
        if not pats:
            continue
        ranked = sorted(pats, key=lambda p: (-p.count, p.name))
        lines.append(f"  {topic} ({len(pats)} pattern{'s' if len(pats) != 1 else ''}):")
        for p in ranked:
            kind_part = f"[{p.kind}] " if p.kind else ""
            facts_n = len(p.source_facts)
            head = (
                f"    - {kind_part}{p.name} "
                f"(count={p.count}, facts={facts_n})"
            )
            if detail_keys is not None and (topic, p.name) not in detail_keys:
                lines.append(head)
                continue
            desc = (p.description or "").strip().replace("\n", " ")
            lines.append(head + (f": {desc}" if desc else ""))
    if len(lines) <= 1:
        # No patterns rendered (every topic was empty) — return empty
        # so the caller skips the prepend.
        return ""
    lines.append(
        "Weight by recurrence: patterns with higher `facts` counts "
        "carry stronger recurrence signal. Singletons (facts=1) are "
        "observations that may inform but shouldn't dominate the "
        "synthesis."
    )
    return "\n".join(lines)


def select_top_pattern_keys(
    patterns_by_topic: dict[str, list[Pattern]],
    keep_count: int,
) -> set[tuple[str, str]]:
    """Return the (topic, name) keys for the top-`keep_count` patterns
    by count, ranked globally (-count, topic, name). Used by the WR
    sample cascade in stages that consume patterns as context (actions)
    or as input (insights) to decide which patterns retain description
    detail at each escalation depth."""
    flat: list[tuple[int, str, str]] = []
    for topic, pats in patterns_by_topic.items():
        for p in pats or []:
            flat.append((-int(getattr(p, "count", 1) or 1), topic, p.name))
    flat.sort()
    return {(topic, name) for _, topic, name in flat[:keep_count]}
