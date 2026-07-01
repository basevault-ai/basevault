"""
Actions (5) — prioritizes and plans proposed actions from Insights.

Each Insight can carry an optional `proposed_actions` list. This stage
collects those across all insights, deduplicates them, filters weak or
symptom-level moves, and produces a small set of ranked, articulated
Actions with scoring, horizon, review cadence, and evidence back to the
originating Insights.

Usage:
    from engine.actions import generate_actions, Action
    plan = generate_actions(insight_output, mode=Mode.TEE)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from engine.insights import InsightOutput, Insight
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

# Action kinds — what kind of move the action is. Drives variety: when
# this enum is present in the schema with exemplars, the LLM
# pattern-matches across kinds rather than collapsing onto whatever
# shape (usually `protocol`) is most frequent in coaching corpora.
# A run where one kind dominates >50% of total is diagnostic — either
# the patterns genuinely only support that kind, or the prompt is
# over-anchoring; either way it's worth surfacing in stats.
ACTION_KINDS = (
    "protocol",     # set up a recurring rule, gate, cadence, or limit
    "experiment",   # try-and-see; small bet with a feedback loop
    "claim",        # state something publicly or to a specific person
    "reach",        # contact someone — talk to / ask / tell
    "build",        # make a concrete artifact or capability
    "decline",      # decisively stop / drop / refuse a thing
    "commit",       # bind to a direction with cost-of-reversal
)


@dataclass
class Action:
    recommendation: str          # imperative title, ≤80 chars, glanceable
    objective: str               # durable state this aims at (noun-ish)
    why: str                     # reasoning grounded in source insights
    immediate_action: str        # one concrete first step, this week
    habit: str                   # recurring behavior to sustain
    success_metric: str          # how you'd know it's working

    horizon: str                 # "short" | "medium" | "long"
    review_date: str             # ISO date — next CHECK-IN, not a deadline

    # Action shape kind — one of ACTION_KINDS, or empty string when the
    # LLM didn't emit a recognized value. Drives variety; surfaced in
    # stats so a run dominated by one kind shows up.
    kind: str = ""

    # Scoring dims, 0.0-1.0. Reversibility was dropped — it actively
    # rewarded waffling over commitment; the new axes pull in opposing
    # directions (consequence wants real cost; generativity wants
    # productive output; decisiveness wants single-shot moves rather
    # than maintenance loops). Spread across these is what produces
    # output diversity.
    regret_reduction: float = 0.0   # cost of NOT doing — costly to ignore
    leverage: float = 0.0           # downstream impact unlocked
    consequence: float = 0.0        # how committal / load-bearing the move is
    generativity: float = 0.0       # produces something new (artifact, capability, opportunity)
    decisiveness: float = 0.0       # single-shot move vs. maintenance loop
    time_to_feedback: float = 0.0   # how quickly success/failure is visible
    constraint_fit: float = 0.0     # fits the subject's actual constraints

    confidence: float = 0.0         # overall confidence, 0.0-1.0

    # Refs back to insights: (scope, index_in_list, edge_conf).
    # scope ∈ {"cross_domain", "critical"}. Sorted by edge_conf desc.
    source_insights: list[tuple[str, int, float]] = field(default_factory=list)
    hallucinated_ref_count: int = 0

    @property
    def score(self) -> float:
        """Weighted score. Regret + leverage + consequence carry; the
        generativity / decisiveness axes are the diversity producers
        and modulate around them. time_to_feedback and constraint_fit
        are tiebreakers, not load-bearing."""
        return (
            0.25 * self.regret_reduction
            + 0.20 * self.leverage
            + 0.15 * self.consequence
            + 0.15 * self.generativity
            + 0.10 * self.decisiveness
            + 0.075 * self.time_to_feedback
            + 0.075 * self.constraint_fit
        )


# ── Defaults ──────────────────────────────────────────────────────────────────


# Hard ceiling on actions regardless of corpus size. Same rationale as
# the insights ceiling: past ~10 the list stops being actionable triage.
# The log formula crosses 10 around 22K facts; below that nothing
# changes.
_ACTION_CAP_CEILING = 10


def action_cap(total_facts: int) -> int:
    """Sub-linear cap on action count: min(round(ln(total_facts)),
    ceiling). No floor — for tiny corpora the cap is 0, which is
    correct (an action plan from a 5-fact input would be padding).
    Reference points: n=20 → 3, n=100 → 5, n=500 → 6, n=2000 → 8,
    ceiling hit around 22K facts."""
    if total_facts <= 0:
        return 0
    return min(_ACTION_CAP_CEILING, max(0, round(math.log(total_facts))))


# Horizon → review_date offset (days from today). Computed runner-side
# AFTER the LLM call so the prompt itself stays date-free; that lets
# the actions stage's prompt-hash cache survive across calendar days.
# Same numeric values as the previous `_HORIZON_REVIEW_CAP_DAYS` cap
# table — they were the upper bound the LLM was told to clamp to,
# now they are the deterministic offset. Renamed because the semantic
# changed from "ceiling" to "exact value." medium and long deliberately
# share 90: long-horizon identity habits get the same review cadence
# as medium-horizon practices because both are "check on me in a
# quarter," not because we couldn't tell the LLM to spread them.
_HORIZON_TO_DAYS = {
    "short":  14,
    "medium": 90,
    "long":   90,
}


def _review_date_for_horizon(horizon: str, today: date) -> str:
    """`today + N days` per the horizon. Unknown horizon → medium
    (matches the parser's normalize step). Pure helper; deterministic
    given (horizon, today). Tests assert exactness."""
    days = _HORIZON_TO_DAYS.get(horizon, _HORIZON_TO_DAYS["medium"])
    return (today + timedelta(days=days)).isoformat()


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You take a set of already-surfaced Insights — each with an optional list
of proposed actions — and produce a small, ranked plan. You do NOT
invent new findings. Your job is to prioritize, dedupe, articulate, and
schedule the moves the upstream stages already suggested or that follow
directly from the insights.

DUAL OPTIMIZATION TARGET — the plan should span both axes:
1. COSTLY-TO-IGNORE — moves whose cost of NOT doing is concrete and
   traceable to the source insight. Defending against a real loss.
2. HIGH-CONSEQUENCE TO COMMIT / CLAIM / BUILD — moves whose payoff
   comes from making a deliberate, load-bearing move that wouldn't
   happen by default. Producing a real gain.

Both axes are co-equal. A plan that's all defensive (audits, gates,
moratoriums, cooling-off periods) is incomplete; so is a plan that's
all aspirational claims with no defensive moves where loops are
costing real energy. Range across both as the insights support.

ACTION KINDS — every action has a `kind` field, one of:
- protocol   : set up a recurring rule, gate, cadence, or limit
- experiment : try-and-see; small bet with a feedback loop
- claim      : state something publicly or to a specific person
- reach      : contact someone — talk to, ask, tell, request
- build      : make a concrete artifact or capability
- decline    : decisively stop / drop / refuse a thing
- commit     : bind to a direction with real cost-of-reversal

The kinds are deliberately a mix of defensive (protocol, decline) and
generative (experiment, claim, reach, build, commit). Pick the kind
that best fits each action; if your plan ends up all `protocol`,
revisit — the insights almost certainly support at least one
generative move. A single dominant kind across all actions is a
warning sign, not a coherent plan.

RECOMMENDATION SHAPE — the `recommendation` field is a glanceable
imperative title, ≤80 characters. It is NOT the full prescription.
The prescription detail (first step, recurring cadence, observable
outcome) lives in `immediate_action`, `habit`, and `success_metric`.

A reader scanning a list of actions sees only the recommendations;
they should be able to tell what each action is about in one glance.
A 200-character comma-separated paragraph is the wrong shape — even
if every clause is true. Cut to the verb and the object.

EXEMPLARS — four diverse actions across kinds. Note how each
`recommendation` is short and imperative, while detail lives in the
other fields. Match the SHAPE; don't lift the content.
  · (protocol)   recommendation: "Cap solo deep-work blocks at 90 minutes"
                 immediate_action: "Set a 90-minute hard timer on the
                 calendar template used for deep-work blocks this week,
                 with a 15-minute walk auto-scheduled at the boundary."
                 habit: "Every deep-work block this quarter; if a block
                 runs past 90, the next one drops to 60 as a corrective."
                 success_metric: "Zero deep-work blocks longer than 90
                 minutes over a four-week window."
  · (claim)      recommendation: "Tell co-founder you're owning the API redesign"
                 immediate_action: "Send the co-founder a one-paragraph
                 message stating you're owning the API redesign and
                 naming the deadline. Send by Friday EOD."
                 habit: "" (single-shot claim — no recurring step)
                 success_metric: "Message sent and acknowledged; redesign
                 milestone owned by you in the team's tracking surface."
  · (reach)      recommendation: "Ask Sara what she wants from the project"
                 immediate_action: "Schedule a 30-minute call with Sara
                 within the next two weeks; bring two open questions."
                 habit: "Recurring 1:1 every 6 weeks once the answer is
                 in hand."
                 success_metric: "Sara names at least one concrete
                 outcome she wants; it's recorded and revisited."
  · (build)      recommendation: "Write the personal-corpus query CLI"
                 immediate_action: "Sketch the CLI's flag set and stub
                 the first command end-to-end."
                 habit: "Use the CLI as the primary corpus-query surface
                 for the next 30 days; iterate based on what you reach for."
                 success_metric: "CLI used at least 5 times in the next
                 month for real corpus questions, replacing manual reads."

Notice the variety in habit field: some recurring, some empty (the
single-shot claim has no maintenance phase). Don't force a habit when
the action is genuinely one-shot. Don't force an artifact when the
action is genuinely a relational reach. The schema accommodates both.

What to UPRANK across both axes:
- Moves whose cost of NOT doing OR consequence of doing is concrete
  and traceable to the source insight
- Moves with downstream leverage — fixing or building this unblocks
  other things
- Moves specific enough that success or failure will be visible in weeks
- Moves that span the action_kind taxonomy (variety is signal)

What to DOWNRANK or DROP:
- Polished general advice that could apply to many people
- Symptom-chasing rather than mechanism-level moves
- Vague aspirations with no first step or metric
- Two actions that collapse into the same underlying move (dedupe)
- Plans where one kind dominates while the insights support several

Self-help guard: if a recommendation could be lifted into a self-help
book without changing a word, it is the wrong recommendation. Whether
the move is defensive or generative, it must be specific to this
person, citing this set of insights as evidence.

DISTINCTION — objective vs immediate_action vs habit must not collapse:
- objective: the durable state you're trying to reach (noun-phrase, not verb)
- immediate_action: one concrete first step doable in the next week (verb,
  specific, time-bounded, has a clear done condition)
- habit: the recurring behavior that sustains the objective after the
  first step (verb, periodic, named cadence). MAY be empty when the
  action_kind is genuinely single-shot (claim, decline, commit).
If any two read as paraphrases of each other, rewrite.

SOURCE FIDELITY — preserve what the insight literally says:
- If an insight or its proposed_action names an operational threshold
  (cadence, time limit, count, frequency — "daily", "1 day", "72h",
  "only 1 at a time"), match it exactly in the action.
- If an insight names a STRUCTURE — gate, prerequisite, sequence,
  dependency, bottleneck — preserve it. A gate is an ongoing check
  before each relevant action, not a one-time project.

Output format hard rules (CRITICAL):
- Respond with a single JSON object. Nothing else.
- DO NOT emit chain-of-thought, reasoning, analysis, or explanation.
- DO NOT prefix the output with phrases like "Let me", "I'll", "Here is".
- The very first character of your response MUST be `{`.
- The very last character of your response MUST be `}`.
- Use ONLY the fields named in the schema below. Do not add any other
  fields. If you would naturally include extra metadata, omit it.\
"""


# Sentiment-bias clauses for the actions stage. Same five values as
# insights; the framing here is about the SHAPE of the action plan
# (which kinds get foregrounded, how recommendations are worded), not
# about which actions are admissible. Source fidelity (thresholds,
# structures) and the dual optimization target are not overrideable.
_SENTIMENT_BIAS_CLAUSES = {
    "brutally-honest": (
        "Tone: brutally honest. Foreground actions whose cost of "
        "ignoring is concrete and high. Words like 'should', 'might "
        "consider', 'try to' are wrong — actions are imperatives. "
        "Same evidence base as neutral; do NOT manufacture stakes the "
        "insights don't support, just stop softening the stakes that "
        "ARE supported."
    ),
    "critical": (
        "Tone: critical, leaning into defensive moves where the "
        "insights flag real loops. Generative actions still appear "
        "where the insights clearly support them, but the weighting "
        "tilts toward stopping costly behavior over claiming new "
        "territory. Don't omit a generative action the insights "
        "clearly call for."
    ),
    "neutral": (
        "Tone: neutral. Defensive and generative actions get equal "
        "weight in proportion to what the insights support. "
        "Recommendations are direct but unjudgmental."
    ),
    "uplifting": (
        "Tone: uplifting. Foreground actions that build, claim, "
        "commit, or reach — the generative kinds. Defensive moves "
        "still appear where the insights clearly call for them, but "
        "the framing emphasizes growth direction. Do NOT manufacture "
        "generative actions the insights don't support, and do NOT "
        "drop a defensive action the insights clearly call for; "
        "uplifting framing is not denial of real loops."
    ),
    "bubbly": (
        "Tone: maximally encouraging. Lead with build, claim, and "
        "experiment kinds. Frame defensive actions as the subject "
        "supporting their own growth rather than guarding against "
        "loss. Same evidence base as neutral — do NOT invent "
        "actions the insights don't support; if the only real action "
        "is defensive, name it honestly with encouraging framing, "
        "don't pretend the loop isn't there."
    ),
}


_SENTIMENT_FRAMING = """\
Sentiment framing — applies to recommendation wording and which
action_kinds get foregrounded:
{sentiment_clause}

The sentiment knob controls EMPHASIS and WORD CHOICE, not the set of
admissible actions. Source fidelity, the dual optimization target,
and the action_kind taxonomy are not overrideable. The same insights
produce the same set of valid actions at every sentiment; what
changes is which get the most weight and how they're worded.\
"""


_SUBJECT_DISCIPLINE = """\
Subject discipline — every action is for {subject}:
- Actions prescribe moves for {subject}, never for other parties.
- If a cited insight describes a non-subject party's behavior, do NOT
  build an action that regulates {subject}'s conduct based on imputed
  strategy in someone else. Reframe as {subject}'s own stance or response
  to observed events, or drop.

Core invariant: an action may not be more stable, strategic, or identity-
level than the evidence supporting it. A one-off event supports a stance,
not a long-horizon habit. Only broad repeated evidence across contexts
earns a long-horizon action.\
"""


# ── Always-on harm clause ────────────────────────────────────────────────────
#
# Self-conditional in the prompt itself, so always-on without an
# upstream keyword scan. Wording stays deliberately generic — no
# enumeration of specific keywords, tactics, or domains.
_HARM_CLAUSE = """\
Harm guardrail — applies to every action:
- Actions must not encourage illegal activity, nor foreseeably cause
  harm to the subject or to other people.
- If the insights describe content of that kind (illegal enterprise,
  self-harm, exploitation of others, etc.), do NOT propose actions
  that would further or sustain such activity.
- What you MAY emit in those cases: actions that help the subject
  exit, harm-reduce, protect affected parties, or disclose to trusted
  support. If no such action is genuinely supported by the insights,
  emit an empty `actions` array — silence is preferable to harmful
  advice.
- This guardrail is always on. Most insights do not trigger it; for
  those, write actions normally.\
"""


_EVENT_VS_HORIZON = """\
Event vs horizon — scale the action to the evidence:
- Short-horizon stance (days to ~2 weeks): appropriate when the backing
  insight rests on a single episode or short sequence. A one-time
  response, not a recurring practice.
- Medium-horizon practice (2-12 weeks): appropriate when evidence shows
  recurrence across contexts.
- Long-horizon identity habit (ongoing, quarter+): appropriate ONLY when
  cited insights rest on broad repeated evidence across multiple
  domains/contexts and time windows.
- Prefer fewer actions with correctly-scaled horizons over more actions
  forced to long-horizon for appearance of weight.\
"""


_TASK = """\
You are given a set of Insights. Each carries a numeric ID in square
brackets like [1], [2], and may list one or more proposed actions the
upstream stage suggested. You MUST reference these IDs in each output
action's `sources`.

Your job:
1. Pool all proposed_actions across the input insights.
2. Deduplicate — merge moves that collapse to the same underlying action.
3. Filter — drop vague, symptom-level, or low-consequence moves.
4. Articulate — for each surviving move, fill out kind, objective,
   why, immediate_action, habit, success_metric.
5. Score along the seven axes below.
6. Pick the actions: hard cap is {max_actions}, but stop at the cap;
   do NOT aim for it. The cap is a ceiling. Emit every action that
   meets the dual-optimization-target bar, in order of score, until
   either you hit the cap or no more meet the bar.
7. Schedule — pick horizon proportional to evidence (short / medium /
   long). The runner converts horizon to a review date deterministically
   AFTER your response, so do not include any dates in your output.

If the insights' proposed_actions are all weak or absent, you MUST
still emit at least one action directly derived from each substantive
critical insight whose mechanism names a concrete cost of inaction OR
a concrete consequence of committing/claiming/building. Empty actions
(`[]`) is correct ONLY when no input insight supports either. Prefer
fewer, sharper actions over padding the list, but do not default to
`[]` when real critical insights are present.

Valid source IDs are ONLY the [N] brackets in the INPUT INSIGHTS block.
Do not cite IDs embedded in mechanism/implication text — those belong to
a different ID space upstream.

Score axes (each 0.0-1.0):
- regret_reduction : cost of NOT doing — how much regret is avoided
- leverage         : downstream impact unlocked
- consequence      : how committal / load-bearing the move is
- generativity     : produces a real artifact, capability, or opportunity
- decisiveness     : single-shot move (claim/commit/decline) vs.
                     maintenance loop (protocol/habit). Decisive moves
                     score high here; protocol-shaped moves score low,
                     and that's fine — the rubric is rewarding variety,
                     not ranking decisive moves above protocols.
- time_to_feedback : how quickly success or failure will be visible
- constraint_fit   : fits the subject's actual constraints

Spread the scores — if two actions have identical scores along all
seven axes you haven't actually ranked them. Different action_kinds
naturally land on different score profiles (a `claim` is high
decisiveness + low maintenance overhead; a `protocol` is high
constraint_fit + low decisiveness). That's the rubric working as
intended.

Respond with ONLY a valid JSON object matching this schema exactly:
{{
  "actions": [
    {{
      "recommendation": "<imperative title, ≤80 chars, glanceable; NOT the full prescription>",
      "kind": "<one of: protocol | experiment | claim | reach | build | decline | commit>",
      "objective": "<durable state>",
      "why": "<reasoning grounded in cited insight(s)>",
      "immediate_action": "<concrete first step, doable within 7 days; this is where prescription detail goes>",
      "habit": "<recurring behavior with cadence; empty string when the action is genuinely single-shot>",
      "success_metric": "<observable signal>",
      "horizon": "short" | "medium" | "long",
      "regret_reduction": <float 0.0-1.0>,
      "leverage": <float 0.0-1.0>,
      "consequence": <float 0.0-1.0>,
      "generativity": <float 0.0-1.0>,
      "decisiveness": <float 0.0-1.0>,
      "time_to_feedback": <float 0.0-1.0>,
      "constraint_fit": <float 0.0-1.0>,
      "confidence": <float 0.0-1.0>,
      "sources": [
        {{"id": <integer>, "confidence": <float 0.0-1.0>}},
        ...
      ]
    }}
  ]
}}

Every action must cite at least one input insight. No extra keys, no
prose, no markdown fences.

INPUT INSIGHTS (each includes its proposed actions from the upstream stage):

{insights_text}\
"""


_SYNTHESIS_INPUT_FLOOR = 4


def _next_pattern_detail_keys(
    patterns_by_topic: dict | None,
    detail_keys: set[tuple[str, str]] | None,
) -> set[tuple[str, str]] | None:
    """WR-sample step (issue #257 spec) for the actions stage's
    patterns_context: halve the current detail set by keeping the
    top-half by (-count, topic, name). All patterns remain in the
    rendered context block by name + count + facts; only the bottom
    half loses the description suffix.

    `detail_keys=None` (initial call) is treated as "every pattern is
    detailed." Returns the smaller detail set, or `None` when no
    further halving is possible — caller stops escalating.
    """
    if not patterns_by_topic:
        return None
    flat: list[tuple[int, str, str]] = []
    for topic, pats in patterns_by_topic.items():
        for p in pats or []:
            if detail_keys is None or (topic, p.name) in detail_keys:
                flat.append(
                    (-int(getattr(p, "count", 1) or 1), topic, p.name)
                )
    if not flat:
        return set()
    flat.sort()
    keep = max(_SYNTHESIS_INPUT_FLOOR, len(flat) // 2)
    if keep >= len(flat):
        return None
    return {(topic, name) for _, topic, name in flat[:keep]}


def _validate_actions_or_raise(raw: str) -> str:
    """Parser callable for the actions wrapper. Raises
    `_EmptyResponse` on whitespace-only output, `_ParseError` on
    non-empty but unparseable JSON, and `_SuccessEmpty` on a parseable
    JSON array that contains zero actions. Returns raw unchanged
    otherwise — the actions stage parses the structured result
    downstream after retries are exhausted."""
    from engine.parse_signals import _EmptyResponse, _ParseError, _SuccessEmpty
    if not (raw or "").strip():
        raise _EmptyResponse(stage="actions")
    try:
        data = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, ValueError):
        raise _ParseError(stage="actions")
    if isinstance(data, list) and not data:
        raise _SuccessEmpty(stage="actions")
    return raw




def _build_prompt(
    insight_output: InsightOutput,
    max_actions: int,
) -> tuple[str, list[tuple[str, int]]]:
    """Build the actions prompt. Returns (prompt_text, source_index_map)
    where source_index_map[i] = (scope, local_idx) for prompt ID (i+1).
    `scope` is "cross_domain" or "critical".

    Date-free by construction: the LLM emits a horizon enum and the
    runner computes review_date from horizon + today AFTER the call.
    Keeps the prompt-hash cache stable across calendar days — same
    inputs + settings yield the same hash whether you re-run today or
    next week."""
    lines: list[str] = []
    source_index_map: list[tuple[str, int]] = []

    def _emit(ins: Insight, scope: str, local_idx: int) -> None:
        prompt_id = len(source_index_map) + 1
        source_index_map.append((scope, local_idx))
        scope_label = "cross-domain" if scope == "cross_domain" else scope
        kind_suffix = f" / {ins.kind}" if ins.kind else ""
        domains_str = ", ".join(ins.domains) if ins.domains else "—"
        proposed = (
            "\n".join(f"      · {a}" for a in ins.proposed_actions)
            if ins.proposed_actions
            else "      (none)"
        )
        lines.append(
            f"[{prompt_id}] ({scope_label}{kind_suffix}) {ins.name}: {ins.description}\n"
            f"    Mechanism: {ins.mechanism}\n"
            f"    Implication: {ins.implication}\n"
            f"    Domains: {domains_str}\n"
            f"    Proposed actions:\n{proposed}"
        )

    for i, ins in enumerate(insight_output.cross_domain):
        _emit(ins, "cross_domain", i)
    for i, ins in enumerate(insight_output.critical):
        _emit(ins, "critical", i)

    return (
        _TASK.format(
            max_actions=max_actions,
            insights_text="\n\n".join(lines),
        ),
        source_index_map,
    )


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _parse_action(
    entry: dict,
    source_index_map: list[tuple[str, int]],
    today: date,
) -> Action | None:
    if not isinstance(entry, dict):
        return None

    recommendation = str(entry.get("recommendation", "")).strip()
    objective = str(entry.get("objective", "")).strip()
    immediate_action = str(entry.get("immediate_action", "")).strip()
    if not recommendation or not objective or not immediate_action:
        return None

    horizon = str(entry.get("horizon", "medium")).strip().lower()
    if horizon not in ("short", "medium", "long"):
        horizon = "medium"

    raw_kind = entry.get("kind", "")
    action_kind = str(raw_kind).strip().lower() if raw_kind else ""
    if action_kind not in ACTION_KINDS:
        action_kind = ""

    # review_date is computed runner-side from (horizon, today); the LLM
    # is no longer asked for a date so the actions prompt-hash stays
    # stable across calendar days. Pre-this-PR the LLM emitted
    # YYYY-MM-DD and we clamped via _clamp_review_date — both are gone.
    review_date = _review_date_for_horizon(horizon, today)

    raw_sources = entry.get("sources") or []
    if not isinstance(raw_sources, list):
        raw_sources = []

    n_prompt = len(source_index_map)
    resolved: dict[tuple[str, int], float] = {}
    hallucinated = 0
    for item in raw_sources:
        if isinstance(item, dict):
            sid = item.get("id")
            conf = item.get("confidence", 1.0)
        else:
            sid = item
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
        conf = _clamp01(conf)
        scope, local_idx = source_index_map[idx]
        key = (scope, local_idx)
        if key not in resolved or resolved[key] < conf:
            resolved[key] = conf

    if not resolved:
        return None

    source_insights = sorted(
        [(k, i, c) for (k, i), c in resolved.items()],
        key=lambda x: (-x[2], x[0], x[1]),
    )

    return Action(
        recommendation=recommendation,
        objective=objective,
        why=str(entry.get("why", "")).strip(),
        immediate_action=immediate_action,
        habit=str(entry.get("habit", "")).strip(),
        success_metric=str(entry.get("success_metric", "")).strip(),
        horizon=horizon,
        review_date=review_date,
        kind=action_kind,
        regret_reduction=_clamp01(float(entry.get("regret_reduction", 0.0) or 0.0)),
        leverage=_clamp01(float(entry.get("leverage", 0.0) or 0.0)),
        consequence=_clamp01(float(entry.get("consequence", 0.0) or 0.0)),
        generativity=_clamp01(float(entry.get("generativity", 0.0) or 0.0)),
        decisiveness=_clamp01(float(entry.get("decisiveness", 0.0) or 0.0)),
        time_to_feedback=_clamp01(float(entry.get("time_to_feedback", 0.0) or 0.0)),
        constraint_fit=_clamp01(float(entry.get("constraint_fit", 0.0) or 0.0)),
        confidence=_clamp01(float(entry.get("confidence", 0.0) or 0.0)),
        source_insights=source_insights,
        hallucinated_ref_count=hallucinated,
    )


def _parse_output(
    raw: str,
    source_index_map: list[tuple[str, int]],
    today: date,
    max_actions: int,
) -> tuple[list[Action], bool]:
    """Returns (actions, parse_error). parse_error=True when raw
    couldn't be JSON-parsed at all — distinguishes "model emitted
    malformed" from "model emitted parseable JSON with zero actions"
    so the sizing cascade can route correctly (parse_error →
    `_ParseError` → classifier returns "sizing" → sample-N)."""
    try:
        data = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, ValueError):
        return [], True

    if not isinstance(data, dict):
        return [], True

    out: list[Action] = []
    for entry in data.get("actions", []) or []:
        parsed = _parse_action(entry, source_index_map, today)
        if parsed:
            out.append(parsed)

    out.sort(key=lambda a: -a.score)
    capped = out[:max_actions] if max_actions > 0 else []
    return capped, False


# ── Public API ────────────────────────────────────────────────────────────────
