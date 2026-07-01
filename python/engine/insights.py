"""
Insights — cross-topic synthesis layer (was: global_patterns).

Takes within-theme Patterns from multiple domains and surfaces OS-level
findings in two flavors:
  1. Cross-domain (≤3) — single mechanism whose loop recurs in ≥2 domains
  2. Critical (≤2)     — high-impact single-domain finding not captured above

Each Insight carries mechanism, implication, and optional proposed_actions.
The proposed_actions are hints for the downstream Actions stage, which
prioritizes and plans across all insights.

Usage:
    from engine.insights import detect_insights, InsightOutput
    result = detect_insights(patterns_by_topic, mode=Mode.TEE)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from engine.patterns import Pattern
from engine.llm import (
    strip_fences as _strip_fences,
)


# Set by runner._patch_llm_calls so per-stage exception tracebacks land
# in run.log alongside the wrapper-level logging.
_runner_log = None


def _log_info(msg: str) -> None:
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
    import traceback
    _log_info(f"  [{stage}] LLM call raised ({ctx}): {type(exc).__name__}: {exc}")
    for line in traceback.format_exc().rstrip().splitlines():
        _log_info(f"    {line}")


# ── IR ────────────────────────────────────────────────────────────────────────

# Shape kinds — what kind of finding the insight names. Spans defensive
# AND generative axes so the optimization target stops being "find the
# subject's defenses." The LLM picks per insight; we don't enforce a
# distribution at parse time, but we surface the distribution in
# downstream stats so we can see if any sentiment is collapsing
# everything onto one kind.
INSIGHT_KINDS = (
    "defensive-loop",
    "generative-capability",
    "identity-trajectory",
    "productive-tension",
    "source-of-energy",
    "blind-spot",
)


@dataclass
class Insight:
    name: str
    description: str
    mechanism: str
    implication: str
    domains: list[str]
    # Shape kind: one of INSIGHT_KINDS, or empty string when the LLM
    # didn't emit one (older runs, prompt drift). Scope (cross_domain
    # vs critical) is encoded by which list this Insight lives in
    # inside InsightOutput, not by this field.
    kind: str = ""
    proposed_actions: list[str] = field(default_factory=list)
    # Refs back to within-theme patterns: (topic, index, confidence).
    source_patterns: list[tuple[str, int, float]] = field(default_factory=list)
    hallucinated_ref_count: int = 0


@dataclass
class InsightOutput:
    cross_domain: list[Insight] = field(default_factory=list)
    critical: list[Insight] = field(default_factory=list)


# ── Defaults ──────────────────────────────────────────────────────────────────

# Total-cap formula: min(round(ln(total_facts)), ceiling). Sub-linear,
# clamped at the top; with no floor, tiny corpora (n < e ≈ 2.7) produce
# zero, which is correct ("a 5-fact corpus genuinely has nothing
# identity-level to say"). Split between cross-domain and critical is
# 60/40, biased toward cross-domain since those are the ones that
# benefit most from corpus richness.
_INSIGHTS_CROSS_FRACTION = 0.6

# Hard ceiling on total insights regardless of corpus size. Past ~10,
# insights stop being a triage surface and become a list-to-skim —
# human attention bandwidth, not signal density, is the binding
# constraint. The log formula crosses 10 around 22K facts; below that
# nothing changes.
_INSIGHTS_TOTAL_CEILING = 10


def insight_caps(total_facts: int) -> tuple[int, int, int]:
    """Return (total, cross_cap, critical_cap) for the given corpus
    size. No floor — for very small corpora the caps are 0/0, which is
    correct. The split rounds the cross-domain half so total_cap = 1
    becomes 1/0 (one cross-domain insight max) rather than 0/1."""
    if total_facts <= 0:
        return 0, 0, 0
    total = min(_INSIGHTS_TOTAL_CEILING, round(math.log(total_facts)))
    if total <= 0:
        return 0, 0, 0
    cross = round(_INSIGHTS_CROSS_FRACTION * total)
    critical = total - cross
    return total, cross, critical


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You surface the insights that most explain THIS SPECIFIC PERSON across
their patterns — the ones that name what's actually going on for them,
not the ones that sound most intellectually interesting.

The optimization target is explanatory power for this individual, not
abstraction quality, generalizability, or breadth of coverage. A reader
who knows the subject should read the insight and say "yes, that's
them." A reader who doesn't should read it and feel they now know
something specific about a real person, not a generic profile.

A complete portrait spans more than defenses. A real person has
generative capabilities AND defensive loops, productive tensions AND
self-sabotage, things that restore them AND things that drain them,
trajectory AND blind spots. If every insight you emit is shaped like a
"how this person sabotages themselves," you have written a clinical
note, not a portrait. Spread across the kinds below as the patterns
support — don't force one if the evidence isn't there, but don't
default to defensive-only either.

Insight kinds — the headline framing each insight takes:
- defensive-loop      : how the subject defends, oscillates, or
                        self-sabotages under specific triggers
- generative-capability : what this person reliably produces, makes
                        possible for themselves or others, or pulls
                        off that most people couldn't
- identity-trajectory : who they are becoming — direction of motion,
                        not a static loop
- productive-tension  : a contradiction they hold without resolving,
                        and the value the contradiction creates
- source-of-energy    : what restores them, what they reach for when
                        depleted, what consistently makes them feel
                        more themselves
- blind-spot          : a non-defensive failure mode — naive moves,
                        consistent under-investment in one axis, a
                        pattern they don't see but others would

Output format hard rules (CRITICAL — violations break downstream parsing):
- Respond with a single JSON object. Nothing else.
- DO NOT emit chain-of-thought, reasoning, analysis, or explanation.
- DO NOT prefix the output with phrases like "Let me", "I'll", "Here is".
- The very first character of your response MUST be `{`.
- The very last character of your response MUST be `}`.
- Use ONLY the fields named in the schema below. Do not add any other
  fields. If you would naturally include extra metadata, omit it.

Heuristic guards:
- Self-help guard: if the insight could be lifted into a self-help book
  without changing a word, it is the wrong insight. The right insight
  could only be said about THIS person, citing THESE patterns. This
  applies to every kind, including the generative ones — "they have a
  growth mindset" is wrong; "they consistently turn debugging
  frustration into a write-up that becomes the team's reference" is
  right.
- Plain-language guard: name insights the way a thoughtful friend or
  therapist would. Avoid novel or abstract formulations not clearly
  grounded in the source material. If the headline reads like an
  academic thesis or anything "ultra-smart," rewrite in plain language.
- Specificity guard: prefer insights that would feel specific to the
  subject — slightly recognizable, slightly uncomfortable, or
  slightly affirming in a way they'd notice — over generic truths
  about most people.

Content hard rules:
- DO NOT restate or reframe the input patterns.
- Do not pad with weak entries, but DO emit every insight the patterns
  clearly support. Empty `cross_domain` + empty `critical` is correct
  ONLY when no set of input patterns supports a defensible insight —
  not as a safety default when unsure and not as a shortcut. When
  substantive patterns are present across multiple domains OR
  multiple patterns cluster on one domain, the default is to emit at
  least one insight.
- Do not double-count one pattern by citing it multiple times across
  insights.
- Source fidelity: if input patterns name an operational threshold
  (cadence, time limit, count, frequency — "daily", "1 day", "72h",
  "only 1 at a time"), carry the threshold token into the insight's
  mechanism AND proposed_actions EXACTLY. Three failure modes — all forbidden:
    · soften the threshold ("daily" → "weekly", "1 day" → "48 hours")
    · abstract it away ("1 day" → "continuous" / "time-bounded" / "aggressive")
    · drop it entirely (keep the theme but lose the number/cadence)
- Source fidelity: if inputs name a STRUCTURE — gate, prerequisite,
  sequence, dependency, bottleneck — preserve it. A gate is an ongoing
  check before each relevant action, not a one-time project.

Exemplars — six insights drawn from realistic personal corpora,
varied in kind. Match the SHAPE; don't lift the content.
  · (defensive-loop) "Pre-emptively concedes in negotiations" —
    mechanism: "When the counterparty raises stakes, subject offers
    concessions before counter-offers are tested; framed as
    'preserving the relationship' but the data show the relationship
    survives without it."
  · (generative-capability) "Turns half-formed frustrations into
    shareable artifacts" — mechanism: "Each time a tool friction
    repeats, subject writes a short note that other people then cite;
    six such notes have become team references over 18 months."
  · (identity-trajectory) "Moving from operator to system author" —
    mechanism: "Across three roles the share of subject's hours spent
    debugging others' code dropped while time spent designing
    primitives others build on rose; the trajectory is consistent
    across domains, not just career."
  · (productive-tension) "Holds 'finish what you start' alongside
    'kill projects fast'" — mechanism: "Subject simultaneously commits
    publicly to long arcs and ruthlessly culls week-old experiments;
    the tension produces faster iteration than either alone, but
    burns the people who can't read which mode is active."
  · (source-of-energy) "Refills via solo writing, not socializing" —
    mechanism: "After draining weeks the recovery activity is always
    the same: 60-90 minutes of writing alone with no audience.
    Replacing it with social plans does not restore."
  · (blind-spot) "Under-invests in shallow check-ins with peers" —
    mechanism: "Subject reaches out to peers only when there's a
    concrete request; the corpus shows zero 'just checking in'
    contacts over multi-month windows. Several relationships have
    cooled as a consequence — the pattern is invisible to subject."

Two-subject conflation — negative example (DO NOT emit insights shaped like
this):
  BAD: {"name": "Dependency-driven self-suppression",
        "mechanism": "When Two-People-Both-Named-In-Patterns face a
        dependent person's needs, they suppress their own and comply;
        the friction then surfaces later as withdrawal."}
Why bad: the mechanism binds two named parties into one "they" and
names a shared internal loop. When two people appear in the cited
patterns, the insight must attribute the mechanism to ONE subject and
treat the other as context. Pronouns resolve to the subject only.\
"""


# Sentiment-bias clauses. Substituted into _SENTIMENT_FRAMING below.
# Each describes how the WRITER should frame insights for the same
# evidence base. Sentiment shifts emphasis and word choice; it does
# NOT shift facts. Forbidding fabrication is part of every clause —
# uplifting must not invent capability the patterns don't support, and
# brutally-honest must not invent failure the patterns don't name.
_SENTIMENT_BIAS_CLAUSES = {
    "brutally-honest": (
        "Tone: brutally honest. Surface failure modes directly. Name "
        "costs in concrete terms. Drop hedging qualifiers like "
        "'appears to', 'seems', 'tends to'. Lead with the costliest "
        "loop, not the most flattering capability. Same evidence base "
        "as neutral — do NOT manufacture failure the patterns don't "
        "support; just stop softening the failures that ARE supported."
    ),
    "critical": (
        "Tone: critical, leaning into shadow patterns with "
        "proportionality. Defensive loops and blind spots get more "
        "weight than generative capabilities, but the weighting "
        "stays proportional to actual evidence — don't invent shadow "
        "where the patterns don't support it, and don't drop "
        "generative findings the patterns clearly support."
    ),
    "neutral": (
        "Tone: neutral and descriptive. Describe what the patterns "
        "show without praise or judgment. Generative capabilities, "
        "defensive loops, and blind spots get equal hearing in "
        "proportion to their evidence."
    ),
    "uplifting": (
        "Tone: uplifting. Lead with capability and trajectory; pair "
        "struggles with the growth direction the patterns suggest. "
        "Same evidence base as neutral — do NOT fabricate positives, "
        "do NOT drop a defensive loop the patterns clearly show, just "
        "frame it as something the subject is working through rather "
        "than as a fixed flaw. If a pattern only supports a defensive "
        "reading, name it honestly; uplifting framing is not denial."
    ),
    "bubbly": (
        "Tone: maximally encouraging. Reframe defensive patterns as "
        "growth edges or 'still learning'. Lead with what the subject "
        "is becoming. Same evidence base as neutral — do NOT invent "
        "capability the patterns don't show, and do NOT erase a "
        "consequential defensive loop. If the patterns only support a "
        "hard finding, the encouragement comes from naming what the "
        "subject could do differently, not from pretending the loop "
        "isn't there."
    ),
}


_SENTIMENT_FRAMING = """\
Sentiment framing — applies to headline word choice, mechanism
phrasing, and which kind of insight gets the most weight:
{sentiment_clause}

The sentiment knob controls EMPHASIS and FRAMING, not facts. The same
patterns produce the same factual claims at every sentiment; what
changes is which kinds get foregrounded and how they're worded. Do not
let any sentiment override source fidelity (thresholds, structures) or
the self-help guard.\
"""


_SUBJECT_DISCIPLINE = """\
Subject discipline — the subject of these insights is {subject}:
- Every insight must describe {subject}'s behavior, decisions, emotional
  loops, or identity-level drivers — not other parties' behavior,
  strategy, traits, or intent.
- Other parties in the source may appear as context, events, or
  constraints that {subject} operates within. They may NOT appear as the
  mechanism, strategy, or trait driving an insight — unless there is
  explicit repeated evidence across multiple distinct contexts/times
  showing the behavior is genuinely theirs and materially shapes
  {subject}'s loops.

Core invariant: a claim may not be more stable, strategic, or identity-level
than the evidence supporting it. Only repetition across time or contexts
earns a pattern; only broad repeated evidence across domains earns an
identity-level insight.

Pre-emit pronoun check (required before finalizing the JSON):
  Scan each output item — name, description, mechanism, implication,
  and proposed_actions — for every pronoun (he / she / they / his /
  her / their / him / them / himself / herself / themselves). Each
  MUST resolve unambiguously to {subject}. If a pronoun could be read
  as referring to any other named person in the input, rewrite the
  sentence to name {subject} explicitly OR reframe so the pronoun
  disappears. If a pronoun genuinely has no plausible resolution to
  {subject}, the item is about the wrong party — drop it.\
"""


_EVENT_VS_INSIGHT = """\
Event vs insight — earn the generalization:
- A cross-domain insight requires the SAME loop firing in 2+ distinct
  domains, visible as evidence in patterns from each. Not: two unrelated
  patterns bundled together under a vague umbrella.
- A critical insight requires multiple within-theme patterns from the
  same domain converging on one identity-level driver. Not: a single
  pattern restated at a higher abstraction level.
- If a within-theme pattern itself rests on a single episode, do NOT use
  it as evidence for an insight unless another independent pattern shares
  its exact mechanism.\
"""


_TASK = """\
You are given within-theme PATTERNS from multiple life domains about ONE
specific person. Each has a numeric ID in square brackets like [1], [2].
You MUST reference these IDs in your output so we can trace which patterns
each insight draws from.

Your job: surface the findings that most explain this person's repeated,
consequential behavior — across the kinds named in the system prompt
(defensive-loop, generative-capability, identity-trajectory, productive-
tension, source-of-energy, blind-spot). Spread across kinds as the
patterns support; don't force one if the evidence isn't there, don't
default to defensive-only either.

Produce TWO outputs in a single response. The split is about scope, NOT
about abstraction quality.

═══════════════════════════════════════════════════════════════════════════════
1. CROSS-DOMAIN INSIGHTS (hard cap: {cross_cap})

A single mechanism whose same shape actually recurs in two or more life
domains. The shape can be a defensive loop, a generative capability, an
identity trajectory, a tension, an energy source, or a blind spot — what
matters is that the same shape is visible in multiple domains, not that
it's a defense mechanism.

Each cross-domain insight must cite patterns from at least 2 distinct domains.

═══════════════════════════════════════════════════════════════════════════════
2. CRITICAL INSIGHTS (hard cap: {critical_cap})

Findings that explain the most about this person within a single domain.
High-recurrence shapes — central capabilities, central loops, central
tensions, central blind spots — that shape downstream behavior in that
domain.

These can be single-domain and still beat cross-domain for explanatory
power. Each must cite at least 2 patterns from the same domain.

═══════════════════════════════════════════════════════════════════════════════

Stop at the cap; do NOT aim for it. The caps are ceilings, not targets.
Emit every insight the patterns clearly support, MOST SIGNIFICANT FIRST
— the order you emit them in is taken as their importance ranking
downstream — until either the insights run out or you hit the cap.

Weight by recurrence: each input pattern carries a `facts: N` annotation
naming how many distinct supporting facts back it. Patterns with higher
fact counts represent stronger recurrence signal. Singletons (facts: 1)
are observations that may inform but shouldn't dominate the synthesis.

For both types: a mechanism must be named — typically trigger →
response → consequence for defensive loops; capability → activation
condition → output for generative ones; the SHAPE varies with the
kind, but the mechanism field always names what's actually happening.
Proposed actions are OPTIONAL — include them only if a specific,
concrete next step is directly implied. Leave the list empty if the
insight is descriptive and not yet action-ready. Don't pad with weak
entries, but don't default to an empty outer list (`cross_domain: []`,
`critical: []`) when substantive patterns are present — emit at least
the best-grounded insight you can defend.

Respond with ONLY a valid JSON object matching this schema exactly:
{{
  "cross_domain": [
    {{
      "name": "<short insight name, 2-6 words>",
      "description": "<1-2 line statement of the insight>",
      "kind": "<one of: defensive-loop | generative-capability | identity-trajectory | productive-tension | source-of-energy | blind-spot>",
      "mechanism": "<what's actually happening — see system prompt for shape>",
      "domains": ["<domain 1>", "<domain 2>", ...],
      "implication": "<system-wide consequence>",
      "proposed_actions": ["<short concrete test, boundary, or commitment>", ...],
      "sources": [
        {{"id": <integer ID from INPUT PATTERNS>, "confidence": <float 0.0-1.0>}},
        ...
      ]
    }}
  ],
  "critical": [
    {{
      "name": "<short insight name>",
      "description": "<statement>",
      "kind": "<one of the six kinds>",
      "mechanism": "<what's happening>",
      "domains": ["<single domain>"],
      "implication": "<consequence>",
      "proposed_actions": ["<action>", ...],
      "sources": [
        {{"id": <integer ID from INPUT PATTERNS>, "confidence": <float 0.0-1.0>}},
        ...
      ]
    }}
  ]
}}

`proposed_actions` is optional. Empty list = no action proposed. Do not pad.

Pre-emit fidelity check: before finalizing, scan the cited input pattern(s)
for threshold tokens (numbers, cadences, counts — "1 day", "72h", "daily",
"only 1 X"). If any are present in the source, your mechanism or
proposed_actions MUST contain that token verbatim.

INPUT PATTERNS:

{patterns_text}\
"""


def _build_prompt(
    patterns_by_topic: dict[str, list[Pattern]],
    cross_cap: int,
    critical_cap: int,
    detail_keys: set[tuple[str, str]] | None = None,
) -> tuple[str, list[tuple[str, int]]]:
    """Build the insights prompt. Returns (prompt_text, prompt_index_map)
    where prompt_index_map[i] = (topic, local_idx) for the 1-indexed
    prompt ID (i+1).

    Every pattern is rendered with a stable prompt ID so the parser can
    resolve all source IDs the model emits. When `detail_keys` is set
    (the WR-sample cascade's escape valve), only patterns whose
    (topic, name) appears in the set render the `: {description}`
    suffix; the rest still appear under their ID with name + count +
    facts so the model never loses sight of the entry, just its
    detail. `None` (default) means every pattern gets full detail.
    """
    lines: list[str] = []
    prompt_index_map: list[tuple[str, int]] = []

    for topic in sorted(patterns_by_topic.keys()):
        for local_idx, p in enumerate(patterns_by_topic[topic]):
            prompt_id = len(prompt_index_map) + 1
            prompt_index_map.append((topic, local_idx))
            kind_str = p.kind or "pattern"
            facts_n = len(p.source_facts)
            head = (
                f"[{prompt_id}] ({topic} / {kind_str}, count {p.count}, "
                f"facts: {facts_n}) {p.name}"
            )
            if detail_keys is not None and (topic, p.name) not in detail_keys:
                lines.append(head)
            else:
                lines.append(f"{head}: {p.description}")

    return (
        _TASK.format(
            patterns_text="\n".join(lines),
            cross_cap=cross_cap,
            critical_cap=critical_cap,
        ),
        prompt_index_map,
    )


def _parse_entry(
    entry: dict,
    scope: str,
    prompt_index_map: list[tuple[str, int]],
) -> Insight | None:
    """Parse one entry from the LLM response. `scope` is "cross_domain"
    or "critical" — used only for validation (different evidence
    requirements). The insight's *shape* kind comes from the entry's
    `kind` field and is validated against INSIGHT_KINDS; unrecognized
    or missing values become an empty string rather than failing
    parse, since older runs / prompt drift may emit nothing."""
    if not isinstance(entry, dict):
        return None
    name = str(entry.get("name", "")).strip()
    if not name:
        return None
    description = str(entry.get("description", "")).strip()
    mechanism = str(entry.get("mechanism", "")).strip()
    domains = [str(d).strip() for d in entry.get("domains", []) if str(d).strip()]
    implication = str(entry.get("implication", "")).strip()

    raw_kind = entry.get("kind", "")
    shape_kind = str(raw_kind).strip().lower() if raw_kind else ""
    if shape_kind not in INSIGHT_KINDS:
        shape_kind = ""

    raw_actions = entry.get("proposed_actions") or []
    if not isinstance(raw_actions, list):
        raw_actions = []
    proposed_actions = [str(a).strip() for a in raw_actions if str(a).strip()]

    # Resolve sources
    raw_sources = entry.get("sources") or []
    if not isinstance(raw_sources, list):
        raw_sources = []

    n_prompt = len(prompt_index_map)
    resolved: dict[tuple[str, int], float] = {}
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
        if not (0 <= idx < n_prompt):
            hallucinated += 1
            continue
        conf = max(0.0, min(1.0, conf))
        topic, local_idx = prompt_index_map[idx]
        key = (topic, local_idx)
        if key not in resolved or resolved[key] < conf:
            resolved[key] = conf

    source_patterns = sorted(
        [(t, i, c) for (t, i), c in resolved.items()],
        key=lambda x: (-x[2], x[0], x[1]),
    )

    topics_covered = {t for t, _, _ in source_patterns}
    if scope == "cross_domain":
        if len(domains) < 2 or len(topics_covered) < 2:
            return None
    else:  # critical
        if len(source_patterns) < 2:
            return None

    return Insight(
        name=name,
        description=description,
        mechanism=mechanism,
        implication=implication,
        domains=domains,
        kind=shape_kind,
        proposed_actions=proposed_actions,
        source_patterns=source_patterns,
        hallucinated_ref_count=hallucinated,
    )


def _parse_output(
    raw: str,
    prompt_index_map: list[tuple[str, int]],
    cross_cap: int,
    critical_cap: int,
) -> tuple[InsightOutput, bool]:
    """Returns (output, parse_error). parse_error=True when the raw
    couldn't be JSON-parsed at all — distinguishes "model emitted
    malformed" from "model emitted parseable JSON with zero entries"
    so the sizing cascade can route correctly (parse_error →
    `_ParseError` → classifier returns "sizing" → sample-N)."""
    try:
        data = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, ValueError):
        return InsightOutput(), True

    if not isinstance(data, dict):
        return InsightOutput(), True

    cross = []
    for entry in data.get("cross_domain", []) or []:
        parsed = _parse_entry(entry, "cross_domain", prompt_index_map)
        if parsed:
            cross.append(parsed)

    critical = []
    for entry in data.get("critical", []) or []:
        parsed = _parse_entry(entry, "critical", prompt_index_map)
        if parsed:
            critical.append(parsed)

    return (
        InsightOutput(
            cross_domain=cross[:cross_cap] if cross_cap > 0 else [],
            critical=critical[:critical_cap] if critical_cap > 0 else [],
        ),
        False,
    )


# ── Public API ────────────────────────────────────────────────────────────────

_CRITICAL_ONLY_NOTE = """\

SINGLE-TOPIC MODE: Only one life domain produced within-theme patterns
in this input. Emit CRITICAL insights only — do NOT emit any cross-domain
synthesis. The `cross_domain` array MUST be empty in your response; the
two-domain minimum is preserved by construction at this layer. Focus the
entire response on critical within-domain insights drawn from the
patterns of the single contributing domain.\
"""


# Issue #190 stages 4/5 sampling: minimum number of input items to
# attempt synthesis with. Below this floor, the sampling stops and
# the stage gives up — synthesizing insights/actions over <4 patterns
# is not meaningful, and the model is more likely to hallucinate
# structure than recover signal. The floor applies AFTER the next
# halving step would land, so we don't fire a doomed call.
_SYNTHESIS_INPUT_FLOOR = 4


def _next_detail_keys(
    strong: dict[str, list[Pattern]],
    detail_keys: set[tuple[str, str]] | None,
) -> set[tuple[str, str]] | None:
    """WR-sample step (issue #257 spec): halve the current detail set
    by keeping the top-half by (-count, topic, name). All pattern
    entries stay visible in the prompt (by ID + name); only their
    descriptions are dropped on the bottom-half cut.

    `detail_keys=None` (initial / full-detail call) is treated as
    "every (topic, name) is in detail." After one halving the set
    holds the top 50%; after two, top 25%; after three, top 12.5%.

    Returns the smaller detail set, or `None` once the surviving set
    would fall below the synthesis floor — caller treats `None` (when
    one was already passed in to compute next) as the floor signal
    and stops escalating.
    """
    flat: list[tuple[str, Pattern]] = []
    for topic, ps in strong.items():
        for p in ps:
            if detail_keys is None or (topic, p.name) in detail_keys:
                flat.append((topic, p))
    if not flat:
        return set()
    flat.sort(key=lambda tp: (
        -int(getattr(tp[1], "count", 1) or 1), tp[0], tp[1].name,
    ))
    keep = max(_SYNTHESIS_INPUT_FLOOR, len(flat) // 2)
    if keep >= len(flat):
        # No further halving possible — caller stops escalating.
        return None
    survivors = flat[:keep]
    return {(topic, p.name) for topic, p in survivors}


def _validate_insights_or_raise(raw: str) -> str:
    """Parser callable for the insights wrapper. Raises
    `_EmptyResponse` on whitespace-only output, `_ParseError` on
    non-empty but unparseable JSON, and `_SuccessEmpty` on parseable
    JSON whose insight arrays are both empty (the m7pp class: model
    streamed cleanly but emitted `{"cross_domain":[],"critical":[]}`,
    finish_reason=stop). Returns raw unchanged otherwise."""
    from engine.parse_signals import _EmptyResponse, _ParseError, _SuccessEmpty
    if not (raw or "").strip():
        raise _EmptyResponse(stage="insights")
    try:
        data = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, ValueError):
        raise _ParseError(stage="insights")
    # Only fire success_empty on the canonical dict shape; non-dict
    # responses fall through to the downstream `_parse_output` which
    # flags them as parse_error (preserves the existing sizing-halve
    # path for shape defects).
    if isinstance(data, dict):
        cross = data.get("cross_domain") or []
        critical = data.get("critical") or []
        if not cross and not critical:
            raise _SuccessEmpty(stage="insights")
    return raw



