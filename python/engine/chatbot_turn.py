"""
Per-turn orchestration for the chatbot — the multi-hop ReAct loop.

One ``run()`` call drives one user message → one assistant reply, fanning
out across 1..N internal LLM calls as the model iteratively decides to
look up more data or finalize. The flow:

    call 1  (decision persona)         → tool call         [must look up]
            ↓ dispatch
    call 2  (grounded_decision)        → tool call or prose answer
            ↓ dispatch (if call)
    …
    call N  (grounded_decision)        → tool call or prose answer
            ↓ dispatch (if call)
    call N+1 (grounded_final)          → prose answer      [forced]

Capped at ``MAX_HOPS`` retrievals per turn; the (MAX_HOPS+1)th call
is ``grounded_final`` — no tool protocol, model must answer from the
accumulated CONTEXT (including an honest "I didn't find this" when
appropriate). The model may also finalize voluntarily on any
``grounded_decision`` call by emitting prose instead of a JSON tool call;
that prose IS the user-visible answer (no extra rewrite call).

Intentionally I/O-free: this module knows nothing about stdin/stdout or
the Tauri protocol. The sidecar layer wires in an ``emit`` callback for
events (``chatbot_thinking`` / ``chatbot_hop`` / ``chatbot_chunk`` / …)
and a ``tracked_complete`` callable that brackets each LLM call with the
stat-record + payload-capture plumbing. Returns a structured ``TurnResult``
the sidecar uses to emit the final ``chatbot_done`` + shareable markers.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from engine.chatbot import (
    CARRYOVER_CAP,
    build_chat_prompt,
    build_grounded_decision_prompt,
    build_grounded_prompt,
    neutralize_dead_brackets,
)
from engine.chatbot_dispatch import dispatch
from engine.chatbot_tools import (
    DEFAULT_K,
    LOCAL_DEFAULT_K,
    LOCAL_MAX_COUNT,
    LOCAL_MAX_LOOKUPS,
    MAX_COUNT,
    MAX_LOOKUPS,
    ToolCall,
    ToolCallError,
    describe,
    parse_tool_call,
    validate_tool_call,
)
from engine.llm import Mode, stage_scope
from engine.rag_vector_store import open_store
from engine.retrieval import RetrievedRecord
from engine.shareable import StoreStats
from engine.shareable_markers import build_store_stats


# Maximum number of retrieval lookups per turn. After this many lookups
# have been issued, the next (and last) LLM call swaps to the
# ``grounded_final`` persona — no tool protocol, model must answer from
# the accumulated CONTEXT. Total LLM calls per turn = 1 + N + (1 forced),
# capped at MAX_HOPS + 1.
MAX_HOPS = 4

# How many times a turn tolerates a malformed tool call — a reply that
# opens a ``{"tool": ...}`` object but isn't valid JSON, so
# ``parse_tool_call`` can't extract it — before giving up on tools and
# forcing a grounded answer from whatever context accumulated. Each
# malformed reply is fed back to the model (as a NOTES "previous
# attempts" line) and the hop retries, so a one-off JSON hiccup
# self-corrects; a model that's genuinely stuck bails to
# ``grounded_final`` fast instead of burning the whole hop budget — and,
# critically, instead of the pre-fix behavior where the raw unparseable
# JSON leaked straight to the user as the "answer". Parse failures are
# retriable here the same way the pipeline stages retry a parse_error.
MAX_PARSE_RETRIES = 2

# Fed back to the model (verbatim, as a NOTES "previous attempts" line)
# when its prior reply opened a tool call that didn't parse. Phrased as
# a correction the model can act on: re-emit ONE valid object, or answer
# in prose — not a description of the internal failure.
PARSE_FAILURE_NOTE = (
    '(your previous reply began a {"tool": ...} call but was NOT valid '
    "JSON and could not be parsed — re-emit exactly ONE valid JSON tool "
    "object, or answer the user in prose from the CONTEXT above)"
)


# Deterministic refusal text the loop emits when the model asks for a
# corpus lookup AND the session has no run bound (#780). Fixed and
# model-free by design: the bug this guards against was the grounded
# call inventing/refusing fluently against an empty retrieval set, with
# no UI signal that retrieval never ran. A deterministic chunk can't
# hallucinate; paired with the still-`boundRun=null` selector state the
# user has an unambiguous signal that the corpus is empty.
NO_CORPUS_REFUSAL_TEXT = (
    "I don't have a corpus to search yet — finish ingesting a folder, "
    "then start a new chat to use it."
)

# Shown when every hop this turn came back with no usable text after the
# kernel already retried to exhaustion — almost always a transient upstream
# error (e.g. a 502 from the model enclave, classified LlmStatus.LOAD and
# retried 5×). Without this the turn falls through to ``answer = ""`` and the
# UI renders a blank bubble with a "done" status, which reads as the model
# deliberately saying nothing rather than a temporary failure the user can
# simply retry.
EMPTY_REPLY_TEXT = (
    "I couldn't get a response from the model just now — this is usually a "
    "temporary hiccup. Please try again."
)

# Per-turn cap on the unioned record pool fed back into each grounded
# call's CONTEXT. Sized to absorb the protocol's theoretical max
# fan-out — ``MAX_LOOKUPS`` × ``MAX_COUNT`` × ``MAX_HOPS`` ≈ 240 on
# the cloud path — without any record evicting mid-turn, so a multi-
# rung trace's earlier rungs stay anchorable as ``has_neighbor``
# targets all the way to finalize.
ACCUMULATOR_CAP = 240

# LOCAL-mode cap is the matching theoretical max for the tighter
# LOCAL caps (2 lookups × 15 max × 4 hops = 120). Routed per-mode
# via ``_caps_for_mode`` so the smaller local chat model never sees
# a CONTEXT block sized for the cloud fan-out it doesn't have the
# context window to absorb.
LOCAL_ACCUMULATOR_CAP = 120


def _caps_for_mode(mode: Mode) -> tuple[int, int, int, int]:
    """Per-mode ``(default_k, max_count, max_lookups,
    accumulator_cap)``. LOCAL gets tighter discipline matching its
    weaker chat model + weaker embedder: smaller per-lookup result
    pool, smaller per-call lookup fan-out, and a smaller per-turn
    accumulator. Cloud keeps the historical defaults. All per-mode
    tuning lives here in one helper so a future mode can land its
    own row without scattering branches."""
    if mode == Mode.LOCAL:
        return (
            LOCAL_DEFAULT_K, LOCAL_MAX_COUNT,
            LOCAL_MAX_LOOKUPS, LOCAL_ACCUMULATOR_CAP,
        )
    return DEFAULT_K, MAX_COUNT, MAX_LOOKUPS, ACCUMULATOR_CAP


# Canonical tool-call onset pattern. The mid-stream leak detector in
# ``_StreamGate`` flips to suppression the moment accumulated stream
# text first matches this — by construction it's exactly the prefix
# ``parse_tool_call`` would recognise post-stream (whose JSON object
# scan tolerates whitespace between ``{`` and the first key), so the
# detector and extractor agree on every shape the model can emit:
# ``{"tool``, ``{ "tool``, ``{\n  "tool``, etc.
_TOOL_CALL_ONSET_RE = re.compile(r'\{\s*"tool')


def _is_tool_call_attempt(reply: str) -> bool:
    """True when ``reply`` structurally OPENS a ``{"tool": ...}`` object
    (optionally wrapped in a ```json fence) as its first content — i.e.
    the model was trying to call a tool, not writing a prose answer that
    merely mentions one.

    Paired with ``parse_tool_call(reply) is None``, this is the
    "malformed tool call" signal: the model opened a tool object but the
    JSON didn't parse, so the loop must NOT surface the raw bytes as the
    answer — it retries (see ``MAX_PARSE_RETRIES``). The anchor is the
    START of the reply (after optional whitespace / a code fence) so
    prose that merely quotes the protocol (``the format is {"tool": …``)
    does not trip it — that genuinely is a prose answer.
    """
    s = reply.lstrip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        s = s.lstrip()
    return _TOOL_CALL_ONSET_RE.match(s) is not None


class _StreamGate:
    """Per-call decision: forward each LLM delta as a ``chatbot_chunk``
    event, or suppress them when the reply turns out to be a tool call.

    Classification fires on the first non-whitespace char of the reply.
    ``{`` or `` ` `` (the bare-JSON / ```json fence shapes the dispatch
    layer parses) → suppress; any other char → flush accumulated head
    and stream subsequent deltas. The horizon is one char so the
    suppress decision happens before the UI sees the opening brace or
    fence delimiter.

    The same onset probe runs in two places. (1) On the would-be
    flushed head in the ``undecided`` → ``streaming`` transition, so a
    batched provider that bundles the prose preamble AND the JSON tool
    call into a single first delta (``Looking… {"tool":...}``) is
    suppressed without ever emitting a chunk — zero-leak path. (2) On
    the joined accumulator after each post-streaming delta, so an
    onset that lands in a later delta still flips the gate to a
    terminal ``buffered_after_leak`` state. Either way the loop pairs
    this with a post-parse ``chatbot_replace`` wipe in the dispatch
    path (gated on ``gate.streamed`` — the zero-leak undecided path
    correctly skips the wipe, since no leak ever reached the UI).

    Every delta is appended to ``parts`` regardless of the decision so
    the loop's content-anchored fallback (when the provider returns a
    non-streamed response and ``on_chunk`` never fires) still has the
    captured text to fall back on.

    Pass ``gated=False`` to open the gate immediately — the
    ``grounded_final`` persona is structurally forced to be an answer
    (no tool-call protocol) and streams from delta one.
    """

    __slots__ = (
        "_emit", "_state", "_pending", "parts",
        "_emitted_any", "_leak_detected", "_gated",
    )

    def __init__(self, emit: Callable[..., None], *, gated: bool) -> None:
        self._emit = emit
        self._state = "undecided" if gated else "streaming"
        self._pending: list[str] = []
        self.parts: list[str] = []
        self._emitted_any = False
        self._leak_detected = False
        # Remember whether the gate was created in tool-call-capable
        # mode. The mid-stream onset probe only runs when this is
        # True: the ``grounded_final`` persona (``gated=False``) has
        # no dispatch path, so suppressing on a literal ``{"tool``
        # substring would just truncate a legitimate grounded answer
        # that quotes the protocol with no recovery (the loop's
        # grounded_final branch skips the false-positive
        # ``chatbot_replace`` recovery, and the trailing fallback
        # skips on ``gate.streamed``).
        self._gated = gated

    @property
    def streamed(self) -> bool:
        """True iff at least one ``chatbot_chunk`` was emitted through
        this gate. The trailing single-emit fallback fires only when
        this is False (buffered branch, or provider returned a
        non-streamed response so ``on_chunk`` never advanced the gate).
        """
        return self._emitted_any

    @property
    def leak_detected(self) -> bool:
        """True iff the mid-stream onset probe fired this call. The
        loop pairs this signal with ``parse_tool_call``'s post-stream
        result to decide whether a ``chatbot_replace`` wipe should be
        emitted before dispatching the extracted call."""
        return self._leak_detected

    def on_chunk(self, delta: str) -> None:
        self.parts.append(delta)
        if self._state == "streaming":
            self._emit("chatbot_chunk", delta=delta)
            self._emitted_any = True
            # Onset probe over the joined accumulator (not just the
            # current delta) so a ``{...\s..."tool`` opening split
            # across two chunks is still caught at the boundary. The
            # regex tolerates whitespace between ``{`` and the first
            # key — same tolerance ``parse_tool_call``'s JSON-object
            # scan has — so a model emitting ``{ "tool": ...`` doesn't
            # bypass detection just because of one space. The probe
            # only runs on tool-call-capable gates (``self._gated``):
            # the grounded_final persona has no dispatch path, and
            # suppressing it on a literal substring would truncate a
            # legitimate answer.
            if self._gated and _TOOL_CALL_ONSET_RE.search(
                "".join(self.parts)
            ):
                self._state = "buffered_after_leak"
                self._leak_detected = True
            return
        if self._state == "buffered" or self._state == "buffered_after_leak":
            return
        self._pending.append(delta)
        head = "".join(self._pending)
        for ch in head:
            if ch.isspace():
                continue
            if ch == "{" or ch == "`":
                self._state = "buffered"
                self._pending = []
            else:
                self._pending = []
                # Onset probe on the would-be flushed head before any
                # emission. A batching provider may bundle the prose
                # preamble AND the full JSON tool call into a single
                # delta (``Looking… {"tool":...}``); flushing the head
                # would leak the whole JSON in one chunk before the
                # streaming-branch detector ever ran. Suppressing the
                # head instead gets zero leak with the same post-parse
                # dispatch — the wipe trigger keys on ``gate.streamed``
                # and stays correctly off in this branch.
                if _TOOL_CALL_ONSET_RE.search(head):
                    self._state = "buffered_after_leak"
                    self._leak_detected = True
                    return
                self._state = "streaming"
                self._emit("chatbot_chunk", delta=head)
                self._emitted_any = True
            return


class _TrackedComplete(Protocol):
    """The sidecar's per-call wrapper signature.

    Brackets each LLM call with begin/end stat records, call-id threading,
    and payload capture — the sidecar owns it because it owns the
    telemetry paths. The loop just needs a callable that returns the
    completion result.
    """

    def __call__(
        self,
        messages: list[dict],
        *,
        _chatbot_stage: str,
        _chatbot_category: str,
        **kwargs: Any,
    ) -> Any: ...


@dataclass(frozen=True)
class TurnContext:
    """Everything the loop needs to drive one turn, with zero I/O coupling.

    Built by the sidecar from its session state + the just-arrived turn
    request, passed verbatim to ``run()``. Frozen so the loop can't
    accidentally mutate session-level state.
    """

    query: str
    history: list[dict]
    turn_index: int
    session_id: str
    store_path: Path | None
    bound_run: str | None
    chatbot_config: dict   # {"model": str, "reasoning": bool}
    tracked_complete: _TrackedComplete
    emit: Callable[..., None]            # emit(event, **payload)
    # Per-mode model + provider kwargs the sidecar pre-resolves and the
    # loop splats into every ``tracked_complete`` call this turn —
    # ``{model, mode}`` on LOCAL, ``{model, mode, _force_model_id}`` on
    # the attested cloud path. Empty (the default) keeps the historical
    # cloud-shaped kwargs at the call sites so a ctx built without
    # resolving mode (older tests) behaves exactly as before.
    complete_kwargs: dict = field(default_factory=dict)
    # Plain provider mode threaded into the retrieval ``dispatch()``
    # call so the dense-KNN query embed (slice A's chokepoint) and the
    # local cross-encoder reranker (slice D, once it lands) run in the
    # session's mode instead of a hardcoded cloud default. Defaults to
    # ``Mode.TEE`` — the attested cloud path — the fail-safe baseline
    # for tests / callers that don't resolve mode themselves.
    mode: Mode = Mode.TEE
    on_tool_call: Callable[[ToolCall, int], None] = lambda *a, **k: None
    # Prior-turn cited records seeded into this turn's accumulator at
    # positions ``[1..K]`` (K = len(seed_records) ≤ ``CARRYOVER_CAP``).
    # Re-hydrated by the sidecar from the most-recent assistant turn's
    # ``cited_refs`` payload (walking back through history if the
    # immediately-prior turn had no groundings). Empty default keeps
    # tests + ad-hoc invocations behaving exactly like before.
    seed_records: list[RetrievedRecord] = field(default_factory=list)
    # Per-hop diagnostic sink — content-free numbers / bools / closed-
    # enum strings only. The loop appends one dict per LLM call this
    # turn fires; the sidecar reads ``hops_diag`` after ``run()`` and
    # threads it through the marker builder. Replaces the previous
    # ``diag_box: dict`` (#781): a single mutable dict could only carry
    # ONE dispatch's signal, so the last hop's update overwrote the
    # rest. The list-of-dicts shape preserves every hop's data.
    hops_diag: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class TurnResult:
    """Per-turn outcome the sidecar uses to emit the final
    ``chatbot_done`` + shareable markers.

    ``lookup_fired`` is True iff at least one tool call was dispatched
    this turn (matches slice-1's shareable-marker contract). ``retrieved``
    is the FINAL accumulated union pool — the records the answer could
    have cited; the sidecar uses it to render the resources block.

    ``store_stats`` is the bound corpus's per-turn structural snapshot
    (captured once on the first hop's open; the db doesn't mutate
    within a turn). All per-hop latencies + per-lookup detail live in
    ``ctx.hops_diag`` (one entry per LLM call), which the sidecar reads
    after ``run()`` returns.

    ``refused`` is True only for #780's no-corpus carve-out (model
    asked for a corpus lookup but the session has no run bound); the
    sidecar branches on this to emit ``chatbot_done resources=None,
    run=None`` (pure-conversation shape) instead of the empty ``[]``
    resources state that would imply "we searched and matched
    nothing."
    """

    answer: str
    retrieved: list[RetrievedRecord]
    lookup_fired: bool
    hops: int   # how many LLM calls fired this turn (1..MAX_HOPS+1)
    store_stats: StoreStats | None = None
    refused: bool = False


def _accumulate(
    acc: list[RetrievedRecord], new: list[RetrievedRecord], cap: int,
) -> list[RetrievedRecord]:
    """Union ``new`` into ``acc`` deduping by ``(kind, record_id)``,
    preserving the order in which records first appear, capped at
    ``cap``. The dispatcher's stage-3 union/ordering will replace this
    helper when that PR lands — the contract (input lists, deduped
    output, stable order, cap) is forward-compatible.
    """
    seen = {(r.record.kind, r.record.record_id) for r in acc}
    out = list(acc)
    for r in new:
        key = (r.record.kind, r.record.record_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= cap:
            break
    return out


# Integer-bracket shape: ``[N]`` only. The dispatcher accepts no
# other anchor form; non-bracket strings are dropped (treated as the
# model drifting to its canonical-id training prior).
_BRACKET_RE = re.compile(r"^\s*\[(\d+)\]\s*$")


def _resolve_bracket_anchors(
    raw_call: dict, acc: "list[RetrievedRecord]",
) -> list[dict]:
    """Translate ``has_neighbor: ["[N]", ...]`` brackets to canonical
    ``"kind/record_id"`` form using the current turn's accumulator,
    in place on ``raw_call``.

    The model sees only ``[N]`` brackets in CONTEXT (the parenthetical
    canonical ids were dropped); it naturally references those same
    brackets in ``has_neighbor``. ``acc`` is the per-turn union of
    retrieved records, indexed by 1-based bracket position. ``[1]``
    → ``acc[0]``, ``[2]`` → ``acc[1]``, etc.

    Mixed entries are fine — a list with some brackets and some
    canonical-id strings (``"action/4"``) passes through, with only
    the brackets rewritten. Out-of-range or non-numeric brackets are
    dropped silently from the model's perspective: the loop's
    existing ``ToolCallError → empty dispatch`` path handles the case
    of a has_neighbor list that translates to nothing, and a missing
    anchor is more honest than a fabricated one.

    Drops are reported via the returned list — one closed-vocab dict
    per dropped anchor with ``{"reason", "value", "acc_size"}`` — so
    the loop can surface them on ``hop_diag`` (and thus into
    ``llm-payloads.yaml`` + shareable diagnosis dumps) for dev-side
    debugging. The model-facing NOTES surface stays untouched.

    Walks the raw dict's ``lookups`` (the ``{"tool": "search", ...}``
    shape ``parse_tool_call`` returns); other tool shapes are no-ops.
    """
    drops: list[dict] = []
    if not isinstance(raw_call, dict):
        return drops
    lookups = raw_call.get("lookups")
    if not isinstance(lookups, list):
        return drops
    for lookup in lookups:
        if not isinstance(lookup, dict):
            continue
        anchors = lookup.get("has_neighbor")
        if not isinstance(anchors, list):
            continue
        rewritten: list[str] = []
        for a in anchors:
            if not isinstance(a, str):
                drops.append({
                    "reason": "non_string",
                    "value": repr(a)[:40],
                    "acc_size": len(acc),
                })
                continue
            m = _BRACKET_RE.match(a)
            if m is None:
                # Not a bracket — drop. The model only sees ``[N]``
                # labels in CONTEXT and never sees canonical ids, so
                # a canonical-form has_neighbor like ``"action/3"``
                # is the model drifting to its training prior;
                # rejecting it forces a clean retry on the next hop
                # instead of silently dispatching against a record
                # the model couldn't have known to pick. An entirely-
                # empty has_neighbor list lands at the validator as a
                # no-anchor lookup; the dispatcher handles that as an
                # unanchored search.
                drops.append({
                    "reason": "canonical_form",
                    "value": a[:40],
                    "acc_size": len(acc),
                })
                continue
            idx = int(m.group(1))
            if 1 <= idx <= len(acc):
                rec = acc[idx - 1].record
                rewritten.append(f"{rec.kind}/{rec.record_id}")
            else:
                # Out-of-range bracket: drop silently from the model's
                # next-hop CONTEXT, surface here for dev-side debugging.
                drops.append({
                    "reason": "out_of_range",
                    "value": a,
                    "acc_size": len(acc),
                })
        lookup["has_neighbor"] = rewritten
    return drops


def _persona_for_call(call_num: int) -> str:
    """Which persona drives the ``call_num``-th LLM call this turn.

    - 1 → ``decision`` (the persona instructs the model to emit a tool
      call; the loop body still handles a voluntary prose reply as a
      graceful answer if the model violates the contract).
    - 2 .. MAX_HOPS → ``grounded_decision`` (may emit tool call OR
      prose answer; carries the accumulated CONTEXT).
    - MAX_HOPS + 1 → ``grounded_final`` (forced answer; no tool
      protocol).
    """
    if call_num == 1:
        return "decision"
    if call_num <= MAX_HOPS:
        return "grounded_decision"
    return "grounded_final"


def _build_prompt(
    persona: str, ctx: TurnContext,
    acc: list[RetrievedRecord],
    lookups_remaining: int, previous_attempts: list[str],
) -> list[dict]:
    """Dispatch to the right prompt builder for the current call's
    persona. ``acc`` / ``previous_attempts`` are unused on the first
    ``decision`` call (no CONTEXT yet, no prior attempts); the budget
    signal is injected only on calls where the model can still issue
    a lookup. The model's strategic ``plan`` from prior hops rides
    along via the JSON snapshot of its previous attempts — no
    separate sidebar bullet needed.
    """
    if persona == "decision":
        return build_chat_prompt(ctx.query, ctx.history)
    if persona == "grounded_decision":
        return build_grounded_decision_prompt(
            ctx.query, acc, ctx.history,
            lookups_remaining=lookups_remaining,
            previous_attempts=previous_attempts or None,
        )
    # grounded_final
    return build_grounded_prompt(
        ctx.query, acc, ctx.history,
        previous_attempts=previous_attempts or None,
    )


def run(ctx: TurnContext) -> TurnResult:
    """Drive one user message → one assistant reply through the
    multi-hop loop. See module docstring for the call-shape diagram.

    Events emitted via ``ctx.emit`` (the sidecar relays them to stdout):

    - ``chatbot_thinking`` — once at turn start.
    - ``chatbot_hop`` — once per dispatched lookup, with
      ``{hop, total, tool, query}``.
    - ``chatbot_chunk`` — the user-visible answer text. Streamed
      token-by-token when the reply is prose; a tool-call-shaped reply
      (first non-whitespace char ``{`` or `` ` ``) is suppressed by
      the gate so the JSON never reaches the UI. A non-streaming
      provider (``on_chunk`` silent on TEE) falls back to a single
      emit of the full answer after the call.

    Returns ``TurnResult`` the sidecar uses to emit ``chatbot_done``.
    """
    emit = ctx.emit

    # Empty query short-circuit. Stays in the loop module (not sidecar)
    # so the I/O layer has one path: build ctx → run → done.
    if not ctx.query.strip():
        return TurnResult(answer="", retrieved=[], lookup_fired=False, hops=0)

    # "Thinking..." once at turn start — NOT per hop. Re-emitting it
    # before every LLM call stomps the per-hop ``chatbot_retrieving``
    # status the UI just set, so the user sees a static "Thinking..."
    # all turn instead of the changing per-hop search text. Slice-1
    # behavior: one thinking at turn start, then chatbot_retrieving's
    # changing query per hop drives the visible state between hops.
    emit("chatbot_thinking")

    # Carryover seed: prior turn's cited records placed at brackets
    # ``[1..K]`` so the user can follow up with "tell me more about
    # [3]" and the bracket resolves to the same record without a fresh
    # search. Capped at ``CARRYOVER_CAP``; the first hop's tool call's
    # ``has_neighbor: ["[3]"]`` lands against the seeded acc.
    acc: list[RetrievedRecord] = list(ctx.seed_records[:CARRYOVER_CAP])
    # ``store_stats`` is captured ONCE on the first hop's open (the db
    # is immutable within a turn) and lifts to the TurnResult so the
    # sidecar can stamp it onto the marker turn-level. Per-hop
    # latencies + dispatch's per-lookup detail live inside each entry
    # of ``ctx.hops_diag`` (one entry per LLM call, populated below).
    store_stats: StoreStats | None = None
    # Running list of the EXACT JSON tool calls the model has already
    # emitted this turn, threaded into each subsequent hop's NOTES
    # block so the model can see its own prior emissions verbatim and
    # doesn't waste budget repeating a structurally-identical lookup
    # under different rationalizations. Tool-call history is stripped
    # from the LLM conversation between hops (see chatbot.py — only
    # user+grounded-answer survive into the next turn), so without
    # this signal the model is hop-blind.
    previous_attempts: list[str] = []
    lookups_done = 0
    # Count of malformed tool calls this turn (opened a ``{"tool"`` object
    # that didn't parse). Bounded by ``MAX_PARSE_RETRIES``; on exhaustion
    # the persona is forced to ``grounded_final`` so the turn ends in a
    # real answer rather than another doomed tool attempt.
    parse_failures = 0
    answer = ""
    full_reply = ""
    # Captured across iterations so the trailing single-emit fallback
    # can read the FINAL call's gate to decide whether streaming
    # already covered the answer.
    gate: _StreamGate | None = None

    for call_num in range(1, MAX_HOPS + 2):  # 1..MAX_HOPS+1 inclusive
        persona = _persona_for_call(call_num)
        # A model that has already burned its parse-retry budget on
        # unparseable tool calls is stuck — stop offering it the tool
        # protocol and force a grounded answer from whatever context
        # accumulated. Done here (before the prompt + gate are built) so
        # the whole hop runs as ``grounded_final``: no tool protocol in
        # the prompt, gate open from delta one.
        if parse_failures >= MAX_PARSE_RETRIES and persona != "grounded_final":
            persona = "grounded_final"
        lookups_remaining = MAX_HOPS - lookups_done

        # Per-hop diag entry. Stamped with everything the marker
        # builder needs to produce a HopMarker for this LLM call.
        # ``lookups_remaining_in_budget`` is set only on the
        # grounded_decision persona (the only one where the prompt's
        # NOTES exposes the budget); decision + grounded_final pass
        # ``None`` so the field renders absent.
        hop_diag: dict = {
            "previous_attempts_count": len(previous_attempts),
            "lookups_remaining_in_budget": (
                lookups_remaining
                if persona == "grounded_decision"
                else None
            ),
        }

        prompt = _build_prompt(
            persona, ctx, acc, lookups_remaining, previous_attempts,
        )

        category = (
            "chatbot_converse" if persona == "decision"
            else "chatbot_grounded_decision" if persona == "grounded_decision"
            else "chatbot_answer"
        )

        # Streaming policy: stream by default; suppress only when the
        # reply's first non-whitespace char looks like a tool call
        # (``{`` or `` ` ``). The grounded_final persona has no tool
        # protocol so its gate opens at delta one; the other two
        # personas may emit either a tool call or a finalizing prose
        # reply, and the one-char gate keeps the prose case streaming
        # without leaking the bare-JSON or fenced-JSON cases.
        gate = _StreamGate(emit, gated=(persona != "grounded_final"))

        # Per-mode model/provider kwargs come from
        # ``ctx.complete_kwargs`` (sidecar resolves once per turn via
        # ``_chat_call_kwargs``). Empty → fall back to the historical
        # cloud-shaped default so a ctx built without mode resolution
        # (older tests) keeps its behavior. The sidecar always
        # populates it post-slice-B; either way every hop's complete()
        # call routes through the same provider/model the turn started
        # in — modes never split mid-turn.
        kwargs: dict = {
            "_chatbot_stage": "chatbot",
            "_chatbot_category": category,
            "on_chunk": gate.on_chunk,
            **(ctx.complete_kwargs or {
                "model": ctx.chatbot_config["model"],
                "mode": Mode.TEE,
                "_force_model_id": True,
            }),
        }

        with stage_scope("chatbot"):
            result = ctx.tracked_complete(prompt, **kwargs)
        full_reply = (result.content or "").strip()
        # ``streamed_to_user`` reflects what reached the UI this hop —
        # True only when ``_StreamGate`` actually emitted chunks.
        # Captured per-hop so the marker can render the
        # voluntary-finalize-without-streaming case (TEE provider
        # returned non-streamed; gate never advanced).
        hop_diag["streamed_to_user"] = bool(gate.streamed)

        if persona == "grounded_final":
            # Forced answer. ``result.content`` is authoritative
            # (deltas can fail to fire on TEE per the non-streamed
            # path); fall back to the captured deltas only when the
            # provider returned an empty content blob.
            answer = full_reply or "".join(gate.parts)
            hop_diag["hop_outcome"] = "prose_answer"
            ctx.hops_diag.append(hop_diag)
            break

        # Decision-style call: classify the reply.
        raw_call = parse_tool_call(full_reply)
        if raw_call is None and _is_tool_call_attempt(full_reply):
            # Malformed tool call: the model OPENED a ``{"tool": ...}``
            # object but the JSON didn't parse, so ``parse_tool_call``
            # returned None. This is NOT a prose answer — surfacing the
            # raw bytes would leak unparseable JSON to the user (the bug
            # this guards). Treat it like the pipeline's retriable
            # parse_error: wipe any leaked partial, feed the failure back
            # as a NOTES "previous attempts" line so the model sees what
            # to fix, and retry on the next hop. Bounded by
            # ``MAX_PARSE_RETRIES`` (the persona override above forces
            # ``grounded_final`` once exhausted), so a model that can't
            # emit clean JSON still terminates in a real grounded answer.
            parse_failures += 1
            if gate.streamed:
                emit("chatbot_replace", text="")
            previous_attempts.append(PARSE_FAILURE_NOTE)
            hop_diag["hop_outcome"] = "malformed_tool_call"
            ctx.hops_diag.append(hop_diag)
            continue

        if raw_call is None:
            # Voluntary prose finalize (slice-2 design open #4: the
            # ``grounded_decision`` persona inherits the grounded-answer
            # rules so the prose is already constrained — no separate
            # rewrite call needed).
            answer = full_reply
            # False-positive recovery: the mid-stream onset probe is
            # high-precision (regex matches what ``parse_tool_call``
            # also matches) but not perfectly tight — a model writing
            # prose that literally mentions ``{"tool`` (e.g. explaining
            # the tool-call protocol) will trip the probe, get its
            # tail suppressed, then have ``parse_tool_call`` return
            # None because no balanced JSON tool object exists. Without
            # this recovery the UI would carry only the leaked prefix
            # forever — the trailing single-emit fallback skips on
            # ``gate.streamed``, so nothing else fires. Emit a
            # ``chatbot_replace`` with the full prose so the bubble +
            # persisted transcript carry the answer the model actually
            # produced.
            if gate.leak_detected:
                emit("chatbot_replace", text=full_reply)
                # The chatbot_replace delivered the answer to the UI
                # in full. Mark the gate as "UI has the answer" so the
                # trailing single-emit fallback (keyed on
                # ``not gate.streamed``) doesn't ALSO fire — the
                # batched-first-delta false-positive path leaves
                # ``gate.streamed`` False (suppressed before any chunk
                # emitted), and an unguarded fallback would duplicate
                # the full prose into a ``chatbot_chunk`` after the
                # replace, leaving the bubble + transcript with two
                # copies. Treating the recovery emit as "covering the
                # UI" keeps the trailing-fallback semantic intact
                # (answer reached the UI exactly once) without a new
                # flag.
                gate._emitted_any = True
            hop_diag["hop_outcome"] = "prose_answer"
            ctx.hops_diag.append(hop_diag)
            break

        # Mixed-shape recovery: the model emitted a prose preamble in
        # front of (and possibly around) the JSON tool call. The gate
        # streamed the preamble — and however much of the JSON body
        # landed in the chunk that completed the onset substring —
        # before suppression kicked in. Emit a ``chatbot_replace`` with
        # empty text so the UI overwrites the leaked bubble content;
        # the next-hop ``chatbot_retrieving`` event then drives the
        # visible state. Trigger gates on BOTH ``raw_call is not None``
        # (the JSON actually parsed) and ``gate.streamed`` (we did leak
        # chunks) so the clean-tool-call case (gate buffered from char
        # 1) and the voluntary-prose case (no JSON to dispatch) both
        # skip the wipe.
        if gate.streamed:
            emit("chatbot_replace", text="")

        # Snapshot the model's verbatim emission BEFORE the bracket
        # resolver mutates ``raw_call``: the NOTES "Previous attempts"
        # bullet on the next hop renders from this string, so the
        # model sees what it actually wrote (``has_neighbor: ["[3]"]``)
        # instead of the post-resolution form (``"action/5"``). Keeping
        # canonical ids out of the model's own sidebar is what closes
        # the second leak channel the Option-A regression surfaced —
        # the resolver still rewrites ``raw_call`` in place for the
        # validator + dispatcher; only the attempts log is decoupled.
        raw_call_snapshot = json.dumps(
            raw_call, separators=(", ", ": "), ensure_ascii=False,
        )

        # Backend translation of ``[N]`` bracket refs in ``has_neighbor``.
        # The CONTEXT block shows the model only bracket indices (no
        # canonical id parenthetical), so the model's ``has_neighbor``
        # filter naturally uses ``[1]``, ``[5]`` shapes that index into
        # THIS turn's accumulator. Translate to the canonical
        # ``kind/record_id`` form the validator + dispatcher already
        # speak. Mutates ``raw_call`` in place; out-of-range or
        # unparseable brackets are dropped so the validator surfaces
        # a clean error if everything translated out. Drops surface
        # on ``hop_diag`` for the dev-side payload + diagnosis dumps
        # but are not fed back to the model — the persona stays the
        # only model-facing layer.
        anchor_drops = _resolve_bracket_anchors(raw_call, acc)
        if anchor_drops:
            hop_diag["anchor_drops"] = anchor_drops

        # Tool call. Validate; an invalid call falls through to an empty
        # dispatch (slice-1 behavior preserved) — the model gets nothing
        # new on the next hop, which the prompt handles conversationally.
        # ``_caps_for_mode`` returns every per-mode tuning knob — per-
        # lookup result cap, per-call lookup fan-out, per-turn
        # accumulator cap — so a LOCAL turn lands on the tightened
        # discipline matching its weaker chat model + weaker embedder.
        (
            default_k, max_count, max_lookups, accumulator_cap,
        ) = _caps_for_mode(ctx.mode)
        call: ToolCall | None
        try:
            call = validate_tool_call(
                raw_call, default_k=default_k, max_count=max_count,
                max_lookups=max_lookups,
            )
        except ToolCallError:
            # Malformed tool call: degrade gracefully — no dispatch, no
            # audit, no records added; the loop continues into the next
            # hop with the same accumulator. The model's next prompt
            # carries the same CONTEXT as before, which the grounded
            # rules handle conversationally ("I couldn't find that").
            # No stderr write here — keeps the loop body I/O-free.
            call = None

        # Stamp hop_outcome now that we've classified the reply.
        # ``tool_call`` covers any validated tool call (whether it
        # actually dispatched against a store or not); ``invalid_tool_call``
        # is the structurally-malformed case.
        hop_diag["hop_outcome"] = (
            "tool_call" if call is not None else "invalid_tool_call"
        )
        # Strategic ``plan`` from the validated call. Silent-defaults
        # to ``""`` when the model omits it; surfaced verbatim so the
        # shareable marker + payload dumps record the walk the model
        # declared this hop, and so the next hop's NOTES "Previous
        # attempts" replay carries it through verbatim in the call
        # JSON snapshot.
        if call is not None:
            hop_diag["plan"] = call.plan

        # #780 no-corpus carve-out: the model just asked for a corpus
        # lookup AND the session has no run bound (none had finished
        # ingesting at process start and no `run_available` push has
        # rebound the session since). Don't dispatch, don't emit
        # `chatbot_retrieving` / `chatbot_hop` (no search to surface),
        # don't run a grounded-answer call against an empty retrieval
        # set — that path was the silent invention this guards against.
        # Emit a fixed deterministic refusal chunk in place of the
        # model's answer and end the turn. The sidecar branches on
        # ``refused=True`` to emit ``chatbot_done resources=None,
        # run=None`` (the empty `[]` resources state would imply "we
        # searched and matched nothing" — wrong here, no tool ran).
        # The hop_diag entry is still appended so the marker reflects
        # the lookup intent + the no-binding skip-reason — the
        # store_open_latency / dispatch_latency / union_size stay None
        # because no dispatch fired.
        if call is not None and ctx.store_path is None:
            emit("chatbot_chunk", delta=NO_CORPUS_REFUSAL_TEXT)
            hop_diag["store_open_latency_ms"] = None
            hop_diag["dispatch_latency_ms"] = None
            hop_diag["union_size_after"] = None
            ctx.hops_diag.append(hop_diag)
            return TurnResult(
                answer=NO_CORPUS_REFUSAL_TEXT,
                retrieved=[],
                # True so the shareable marker shows the lookup intent
                # for yaml legibility (#781's hops trace renders the
                # ``no_bound_run`` skip-reason via the turn-level
                # resolver — see ``_resolve_skipped_reason_from_hops``).
                lookup_fired=True,
                hops=call_num,
                store_stats=store_stats,
                refused=True,
            )

        call_desc = describe(call) if call else ""
        # Emit BOTH events: the new ``chatbot_hop`` (richer structure for
        # the slice-2 step-trail UI) AND the legacy ``chatbot_retrieving``
        # (which is what the current UI reducer in ChatbotHelper.jsx
        # actually handles today). Without the legacy emit, the UI shows
        # only ``Thinking…`` between hops because it drops unknown
        # events — a real "no progress between hops" regression for the
        # user. Keeping both means slice-1 UI builds keep working AND
        # the future step-trail UI has the structured ``chatbot_hop`` to
        # build on.
        emit(
            "chatbot_hop",
            hop=call_num,
            total=MAX_HOPS,
            tool=(call.tool if call else ""),
            query=call_desc,
        )
        emit("chatbot_retrieving", query=call_desc)

        retrieved: list[RetrievedRecord] = []
        store_open_ms: float | None = None
        dispatch_ms: float | None = None
        if call is not None and ctx.store_path is not None:
            _t_open = time.monotonic()
            with open_store(ctx.store_path) as store:
                store_open_ms = (time.monotonic() - _t_open) * 1000.0
                # Snapshot the bound store ONCE — the db is immutable
                # within a turn, so subsequent hops would just re-read
                # the same counts. Best-effort: a telemetry count
                # failure leaves store_stats=None (the resolver then
                # falls through to ``sink_never_called`` rather than
                # misreporting ``empty_store``) without breaking the
                # turn — the load-bearing ``store.count() > 0`` gate
                # below is the authoritative signal.
                if store_stats is None:
                    try:
                        store_stats = build_store_stats(store)
                    except Exception:
                        store_stats = None
                if store.count() > 0:
                    _t_retrieve = time.monotonic()
                    retrieved = dispatch(
                        call,
                        store=store,
                        # Session's resolved mode (slice B); the query
                        # embed inside dispatch already routes by mode
                        # via slice A's embed chokepoint, and a future
                        # local cross-encoder reranker will key off
                        # this too — passing it now keeps the dispatch
                        # surface forward-correct instead of pinning
                        # cloud on every retrieval.
                        mode=ctx.mode,
                        # Per-hop sink: each hop fires its own sink so
                        # dispatch's per-call diag (including the new
                        # per_lookup_diag list) lands in THIS hop_diag,
                        # not clobbered by the next hop's dispatch.
                        diag_sink=hop_diag.update,
                    )
                    dispatch_ms = (
                        time.monotonic() - _t_retrieve
                    ) * 1000.0
        hop_diag["store_open_latency_ms"] = store_open_ms
        hop_diag["dispatch_latency_ms"] = dispatch_ms

        if call is not None:
            # Audit row for every validated call, regardless of whether
            # a store was available — slice-1 wrote a row with
            # ``result_count=0`` for the no-store fallback path too, and
            # dropping that silently loses replay/debug traces. Out of
            # the store-path gate.
            ctx.on_tool_call(call, len(retrieved))
        # Record this hop's raw JSON into the running attempts list so
        # the next hop's NOTES block carries the model's verbatim
        # emission. Recording even invalid calls (validate raised →
        # ``call is None``) keeps the model honest about what it
        # emitted, not just what passed validation — same model can't
        # then rationalize an invalid duplicate as a fresh attempt.
        # The snapshot is the pre-resolution form (bracket anchors
        # intact) so the model sees what it wrote, not the post-
        # dispatch canonical-id rewrite.
        if raw_call is not None:
            previous_attempts.append(raw_call_snapshot)

        acc = _accumulate(acc, retrieved, accumulator_cap)
        lookups_done += 1
        # Stamp the cumulative accumulator size AFTER this hop's
        # _accumulate merged in — the marker's per-hop running total.
        hop_diag["union_size_after"] = len(acc)
        ctx.hops_diag.append(hop_diag)

    # Single-emit fallback when the gate didn't already stream the
    # answer. Two scenarios hit this path: the buffered branch (a
    # tool-call-shaped reply that turned out to be prose after
    # ``parse_tool_call`` returned None), and the non-streaming
    # provider case (``on_chunk`` never fired on TEE so the gate never
    # advanced). In both, the captured ``full_reply`` becomes one
    # ``chatbot_chunk`` so the UI gets the answer.

    # Retry-exhausted empty: every hop returned no usable text (the kernel
    # already retried transient LOAD failures to exhaustion). Substitute a
    # clear retry prompt so the user never sees a blank "done" bubble.
    if not answer.strip():
        answer = EMPTY_REPLY_TEXT

    # Dead-citation cleanup on lookup-fired turns: a weak chat model can
    # finalize a grounded answer citing an index it never grounded (an
    # out-of-range ``[N]``). Those brackets resolve to no record, so the
    # UI renders them as inert text — a dead citation marker beside an
    # otherwise-real answer. Strip the unresolvable ones (in-range
    # citations stay clickable) so the user never sees a dead bracket.
    # Gated on a fired lookup: the no-lookup prose-finalize path has its
    # own handling (the persona's no-brackets-without-lookup rule) and is
    # out of scope here. When the answer streamed token-by-token, the
    # dead bracket already reached the bubble, so a ``chatbot_replace``
    # overwrites it with the cleaned text (same mechanism the mixed-shape
    # leak recovery uses); the non-streamed answer is cleaned before the
    # single-emit fallback below carries it.
    if lookups_done > 0:
        cleaned = neutralize_dead_brackets(answer, acc)
        if cleaned != answer:
            answer = cleaned
            if gate is not None and gate.streamed:
                emit("chatbot_replace", text=answer)

    if answer and (gate is None or not gate.streamed):
        emit("chatbot_chunk", delta=answer)

    return TurnResult(
        answer=answer,
        retrieved=acc,
        lookup_fired=lookups_done > 0,
        hops=call_num,
        store_stats=store_stats,
    )


__all__ = [
    "MAX_HOPS",
    "ACCUMULATOR_CAP",
    "LOCAL_ACCUMULATOR_CAP",
    "CARRYOVER_CAP",
    "NO_CORPUS_REFUSAL_TEXT",
    "EMPTY_REPLY_TEXT",
    "TurnContext",
    "TurnResult",
    "run",
]
