"""
Structured tool-call surface for the chatbot's vault retrieval.

The chatbot reaches the user's processed data through a small, fixed set
of named retrieval tools. On the decision turn the model either answers
from the conversation or emits a single JSON tool call; this module is
the language-facing half of that surface:

  1. ``TOOLS`` — the registry of callable tools and their parameters.
     The model is shown stable tool *signatures*, never table or column
     names, so the underlying store schema can evolve without
     re-prompting.
  2. ``parse_tool_call`` — detect a tool call the model emitted and pull
     out the raw ``{tool, args}`` (replacing the old free-form
     ``LOOKUP:`` line). Tolerant of a prose preamble and markdown
     fences, the same way the decision turn tolerated preamble before
     the directive.
  3. ``validate_tool_call`` — schema-validate the raw call against the
     registry: the tool must be known, every lookup non-empty, list
     fields coerced to their declared element type and capped, and
     ``count`` clamped to ``MAX_COUNT``. Raises ``ToolCallError`` on
     anything it cannot make safe.
  4. ``audit_record`` — the content-light structured record of a
     dispatched call, appended to the per-conversation tool-call audit
     log for replay.

The search tool accepts a **list of lookup objects**. Within one lookup,
all filters AND-compose; across the list, the results set-union (dedup
by ``(kind, record_id)``). Each lookup may carry any subset of:

  - ``query`` — semantic anchor; ranks the lookup's hits by cosine
    distance.
  - ``has_neighbor`` — adjacency anchors as ``"kind/record_id"``
    strings; restricts the lookup to records reachable in the persisted
    edge index from one of the anchors.
  - ``exact_match`` — case-insensitive literal substrings; ≥1 must
    match the record's text.
  - ``entry_type`` — restrict to one or more record kinds.
  - ``source`` — case-insensitive substrings of a source file's name;
    ≥1 must match the record's ``file_id``. Anchors a lookup on a named
    file when the user asks about one.
  - ``count`` — per-lookup result cap (1..``MAX_COUNT``).

A lookup must have at least one filter or a query. The model may emit
the slice-1 single-lookup shape (``{tool, query, kind, k}``) and the
validator treats it as a one-element ``lookups`` list.

The safety contract this surface exists to hold (the reason the model
never emits SQL): the model produces only a tool name and typed
parameters. The dispatcher (see ``chatbot_dispatch``) maps those to
fixed, parameterized, read-only queries with bound params — there is no
path from model output to an interpolated query string, so a string in
the user's own vault content cannot steer a destructive or malformed
query. This module is where the typed contract is enforced before a
call is ever dispatched.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from engine.rag_vector_store import RECORD_KINDS


# Per-lookup result cap. The validator clamps every ``count`` argument
# to this ceiling, so a model (or a string in the vault nudging it)
# cannot request an unbounded pull from a single lookup. Slice-2
# calibration drops the ceiling from 30 → 20 per the addendum spec;
# the accumulator cap (``MAX_TOTAL_RESULTS`` below, enforced in the
# dispatcher) dominates the total context size across hops regardless.
#
# The default values are the cloud caps. LOCAL mode uses tighter ones
# (see ``LOCAL_DEFAULT_K`` / ``LOCAL_MAX_COUNT`` and the per-mode
# resolver below) — the smaller local chat model has a narrower context
# window and benefits from a smaller per-hop result set.
DEFAULT_K = 15
MAX_COUNT = 20

# LOCAL-mode per-lookup caps. Per spec § Chatbot Architecture / Reranker:
# the cross-encoder reranker (qwen3-0.6b on llama.cpp-Metal) is held in
# reserve as an optional follow-up; lowering the caps captures most of
# the local-context-pressure win without paying the reranker latency.
LOCAL_DEFAULT_K = 10
LOCAL_MAX_COUNT = 15
# LOCAL also tightens the per-call lookup fan-out from the cloud cap
# of 3 down to 2 — the smaller local chat model handles a more
# focused dispatch better, and a 2-leg call still leaves room to pair
# a tight lookup with a fallback. Enforced via the same silent-
# truncate path the cloud cap uses; the persona text shows the cloud
# value (the existing convention for DEFAULT_K / MAX_COUNT).
LOCAL_MAX_LOOKUPS = 2

# Backwards-compatible alias for the slice-1 cap name. External
# integration tests pinned to ``chatbot_tools.MAX_K`` read this without
# the validator clamp changing values.
MAX_K = MAX_COUNT

# Upper bound on lookups per tool call. Three gives the model room to
# pair a tight lookup with a fallback and one cross-type peer while
# keeping each call's fan-out small enough that the model has to
# commit to a few clear angles rather than spraying. Lookups beyond
# this cap are silently truncated in ``validate_tool_call`` — the
# model is steered to the cap by the persona; a malformed over-
# fanout call still dispatches its first ``MAX_LOOKUPS`` legs
# instead of failing the whole turn.
MAX_LOOKUPS = 3

# Per-turn cap on returned entries after the array's lookups union and
# dedup. Enforced in the dispatcher; named here so the contract lives
# beside the rest of the result-shaping constants.
MAX_TOTAL_RESULTS = 50


# Per-tool-call forcing-function cap on the strategic ``plan`` field:
# the model articulates the multi-step walk it's executing and where
# it is in it, rewritten each hop to accommodate mid-walk discovery
# + pivot. 250 chars is enough for a 2-3 step plan with the current
# rung explicitly named, without growing into a scratchpad that
# competes with the actual answer.
MAX_PLAN_CHARS = 250


# Kinds a search lookup may restrict to via ``entry_type`` — the
# canonical record kinds the vector store holds. Kept in sync with the
# store's own tuple so a new kind is filterable the moment it is
# embedded.
SEARCHABLE_KINDS: frozenset[str] = frozenset(RECORD_KINDS)


class ToolCallError(ValueError):
    """A parsed tool call that cannot be made safe to dispatch: an
    unknown tool, a lookup with no query and no filters, a parameter
    that cannot be coerced to its declared type, or a malformed
    neighbor anchor. The dispatcher turns this into an honest "couldn't
    run that lookup" rather than guessing.
    """


@dataclass(frozen=True)
class Lookup:
    """A validated single lookup inside a tool call. All collection
    fields are tuples (immutable + hashable) so a ``Lookup`` can be
    used as a dict key during audit/dedup if a future caller needs it.
    Each field is either ``None`` / empty-tuple (filter not used) or a
    non-empty, type-coerced, capped value ready for the dispatcher.
    """

    query: str | None = None
    has_neighbor: tuple[tuple[str, str], ...] = ()
    exact_match: tuple[str, ...] = ()
    entry_type: tuple[str, ...] = ()
    source: tuple[str, ...] = ()
    count: int = DEFAULT_K

    def as_audit(self) -> dict:
        """Audit-friendly dict (lists, not tuples; ``has_neighbor`` as
        the original ``kind/id`` strings) for replay logs and YAML
        payloads."""
        out: dict = {"count": self.count}
        if self.query is not None:
            out["query"] = self.query
        if self.has_neighbor:
            out["has_neighbor"] = [f"{k}/{r}" for (k, r) in self.has_neighbor]
        if self.exact_match:
            out["exact_match"] = list(self.exact_match)
        if self.entry_type:
            out["entry_type"] = list(self.entry_type)
        if self.source:
            out["source"] = list(self.source)
        return out


@dataclass(frozen=True)
class ToolCall:
    """A validated, ready-to-dispatch tool call: a known tool name and a
    fully coerced/clamped argument map. The dispatcher consumes this and
    nothing else — by the time a ``ToolCall`` exists, every value has
    been type-checked and capped, so the dispatch path needs no further
    sanitisation of model output. ``args`` always carries a non-empty
    ``lookups`` tuple of ``Lookup`` objects.

    ``plan`` is the model's strategic execution plan: the multi-step
    walk it's executing this turn and where in the walk this call
    sits, capped at ``MAX_PLAN_CHARS``. Rewritten each hop (not
    sticky) so the model can pivot mid-walk on discovery; the loop
    accumulates the trajectory across hops into the next call's
    NOTES sidebar. A missing or malformed plan silent-defaults to
    ``""`` rather than failing — the persona asks for it but the
    contract stays soft.
    """

    tool: str
    args: dict
    plan: str = ""


# ── The tool registry ───────────────────────────────────────────────────────
#
# Each entry declares the human-facing description shown in the protocol
# prompt and the parameter contract `validate_tool_call` enforces. The
# initial set is a single `search` tool that accepts a list of lookup
# objects. Each lookup combines optional filters that AND together;
# across the array, the union of all lookups' hits is returned (deduped
# by entry id), so the model can pair a tight lookup with a more general
# fallback in one turn.
TOOLS: dict[str, dict] = {
    "search": {
        "summary": (
            "Look things up in the user's vault — their notes, "
            "journals, messages, and the facts, patterns, insights, "
            "and actions derived from them. Pass one lookup or a list "
            "of lookups; within a lookup all filters apply at once, "
            "and across the list the matching entries are merged into "
            "one numbered result."
        ),
        "params": {
            "lookups": (
                "A list of lookup objects (or a single object). Each "
                "may combine any subset of the per-lookup fields below; "
                "must have at least one. Up to "
                f"{MAX_LOOKUPS} lookups per call, up to "
                f"{MAX_TOTAL_RESULTS} entries returned across the list."
            ),
            "query": (
                "Per-lookup: a full, self-contained natural-language "
                "question. Ranks that lookup's hits by similarity to "
                "the question."
            ),
            "has_neighbor": (
                "Per-lookup: list of bracket-index strings anchoring "
                "the walk, like [\"[1]\"] or [\"[3]\", \"[5]\"]. Each "
                "anchor is the bracketed index of an entry from THIS "
                "turn's CONTEXT block — the backend resolves it to "
                "the underlying record. Restricts the lookup to "
                "entries reachable from one of those anchors in the "
                "vault's derivation/mention graph — for example an "
                "action's source patterns, a pattern's source facts, "
                "or a fact's original chunk. Anchors only resolve to "
                "entries you can SEE in this turn's CONTEXT — to "
                "walk from a record you saw on an earlier turn, "
                "search for it again first so it lands in the new "
                "CONTEXT."
            ),
            "exact_match": (
                "Per-lookup: list of literal substrings; at least one "
                "must appear (case-insensitive) in the entry's text."
            ),
            "entry_type": (
                "Per-lookup: list restricting to one or more entry "
                "kinds: "
                + ", ".join(sorted(SEARCHABLE_KINDS))
                + "."
            ),
            "source": (
                "Per-lookup: list of source file names (or fragments of "
                "them); an entry matches when one appears (case-"
                "insensitive) anywhere in its file path. Use this when "
                "the user names a specific file — e.g. "
                "[\"meditations.txt\"] — to anchor the lookup on that "
                "file. Matches the file-scoped kinds (document, chunk, "
                "fact). Pair with entry_type [\"document\"] for "
                "\"do I have file X?\" / \"tell me about file X\", "
                "entry_type [\"fact\"] for the facts drawn from a file, "
                "or leave entry_type open to pull that file's chunks. "
                "(Patterns/insights/actions span many files and carry no "
                "single source — reach them by filtering facts, then "
                "has_neighbor up.)"
            ),
            "count": (
                f"Per-lookup: max results (1-{MAX_COUNT}, default "
                f"{DEFAULT_K})."
            ),
        },
    },
}


def _iter_json_objects(text: str):
    """Yield each top-level ``{...}`` substring in ``text``, in order.

    A brace-depth scan that is aware of JSON string literals (and their
    escapes), so a ``{`` or ``}`` inside a query string never throws off
    the balance. Only depth-zero objects are yielded; nested objects ride
    along inside their parent's substring. This is what lets
    ``parse_tool_call`` find the call whether the model emitted it bare,
    inside a ```json fence, or after a sentence of preamble.
    """
    depth = 0
    start = -1
    in_str = False
    escaped = False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]
                    start = -1


def parse_tool_call(reply: str) -> dict | None:
    """The raw ``{"tool": ..., ...}`` object the model emitted, or
    ``None`` when the reply is ordinary conversation.

    The model is asked to make a vault lookup its entire reply — a single
    JSON object naming a tool — but, as with the old directive line, that
    is a soft instruction: it sometimes wraps the object in a ```json
    fence or prepends a sentence. So a tool call is the first top-level
    JSON object anywhere in the reply that parses and carries a string
    ``"tool"`` key. Prose that merely mentions a tool name, or JSON
    without a ``"tool"`` key, is left as conversation (returns ``None``);
    validation of the named tool + its params happens separately in
    ``validate_tool_call``.
    """
    if not reply:
        return None
    for blob in _iter_json_objects(reply):
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
            return obj
    return None


def _opt_str(raw: dict, key: str) -> str | None:
    val = raw.get(key)
    if val is None:
        return None
    if not isinstance(val, str) or not val.strip():
        return None
    return val.strip()


def _str_list(raw: dict, key: str) -> list[str]:
    """A list of non-empty stripped strings under ``key``, or ``[]`` when
    the key is absent / null. A scalar string is accepted as a
    one-element list (models sometimes drop the brackets when a list has
    one entry). Non-string elements are skipped silently — a hostile or
    malformed list shouldn't fail the whole call, but it also shouldn't
    smuggle non-strings through to a parameterized SQL binding."""
    val = raw.get(key)
    if val is None:
        return []
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    if not isinstance(val, list):
        return []
    out: list[str] = []
    for item in val:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out


def _clamp_count(
    raw: dict,
    key: str = "count",
    *,
    default_k: int = DEFAULT_K,
    max_count: int = MAX_COUNT,
) -> int:
    """Coerce the optional per-lookup result cap to an int in
    ``1..max_count`` (default ``default_k``). A non-integer or
    out-of-range value is clamped rather than rejected — a result count
    is a soft preference, not the kind of parameter where a wrong value
    produces a wrong-but-plausible answer. Accepts the slice-1 alias
    ``k`` when ``key`` is the default. ``OverflowError`` is caught for
    the ``Infinity`` literal (which ``json.loads`` accepts and
    ``int(float('inf'))`` raises on).

    ``default_k`` / ``max_count`` keyword-default to the cloud values so
    every existing caller (and ad-hoc usage in tests) keeps its
    behavior. The turn loop threads the per-mode caps through
    ``validate_tool_call`` for LOCAL turns; see ``caps_for_mode``.
    """
    val = raw.get(key)
    if val is None and key == "count":
        val = raw.get("k", default_k)
    try:
        n = int(val)
    except (TypeError, ValueError, OverflowError):
        return default_k
    return max(1, min(max_count, n))


def _parse_neighbor_anchor(raw: str, tool: str) -> tuple[str, str]:
    """Split a ``"kind/record_id"`` neighbor anchor into its tuple form.
    Record ids can themselves contain ``/`` (chunk ids are
    ``file_id@offset`` but other kinds might in future), so split on the
    *first* ``/`` only. The kind must be a known record kind."""
    s = raw.strip()
    if "/" not in s:
        raise ToolCallError(
            f"{tool}: has_neighbor anchor {raw!r} missing 'kind/' prefix"
        )
    kind, _, rid = s.partition("/")
    kind = kind.strip()
    rid = rid.strip()
    if kind not in SEARCHABLE_KINDS:
        raise ToolCallError(
            f"{tool}: has_neighbor anchor {raw!r} has unknown kind {kind!r}"
        )
    if not rid:
        raise ToolCallError(
            f"{tool}: has_neighbor anchor {raw!r} missing record_id"
        )
    return (kind, rid)


def _validate_search_lookup(
    raw: dict,
    tool: str,
    *,
    default_k: int = DEFAULT_K,
    max_count: int = MAX_COUNT,
) -> Lookup:
    """One lookup object → a validated ``Lookup``. Raises
    ``ToolCallError`` for an unparseable filter or a fully-empty lookup
    (no query and no filters — the dispatcher would have nothing to
    select on). ``default_k`` / ``max_count`` flow through to
    ``_clamp_count`` so per-mode caps reach the count field."""
    if not isinstance(raw, dict):
        raise ToolCallError(f"{tool}: lookup entry is not an object")
    query = _opt_str(raw, "query")
    has_neighbor = tuple(
        _parse_neighbor_anchor(s, tool) for s in _str_list(raw, "has_neighbor")
    )
    exact_match = tuple(_str_list(raw, "exact_match"))
    entry_type_raw = _str_list(raw, "entry_type")
    # Slice-1 shorthand: ``kind`` (singular string) treated as a
    # one-element ``entry_type`` list. New callers should use the list
    # form; the alias keeps slice-1's worked example a valid call.
    kind_alias = _opt_str(raw, "kind")
    if kind_alias and not entry_type_raw:
        entry_type_raw = [kind_alias]
    for k in entry_type_raw:
        if k not in SEARCHABLE_KINDS:
            raise ToolCallError(f"{tool}: unknown entry_type {k!r}")
    entry_type = tuple(entry_type_raw)
    # ``source`` (the file filter) accepts ``file_id`` / ``file`` as
    # aliases — models reach for any of the three when anchoring on a
    # filename. Whichever is present feeds the same case-insensitive
    # file_id substring match in the dispatcher.
    source = tuple(
        _str_list(raw, "source")
        or _str_list(raw, "file_id")
        or _str_list(raw, "file")
    )
    count = _clamp_count(raw, default_k=default_k, max_count=max_count)
    if (
        query is None
        and not has_neighbor
        and not exact_match
        and not entry_type
        and not source
    ):
        raise ToolCallError(
            f"{tool}: lookup must have at least one of "
            f"query / has_neighbor / exact_match / entry_type / source"
        )
    return Lookup(
        query=query,
        has_neighbor=has_neighbor,
        exact_match=exact_match,
        entry_type=entry_type,
        source=source,
        count=count,
    )


def _coerce_lookups(raw: dict, tool: str) -> list[dict]:
    """Pull the lookup array out of a raw search call, accepting either
    the canonical ``{"lookups": [...]}`` shape or the slice-1 shorthand
    where the lookup's fields ride on the tool call object itself. The
    shorthand path lets the model emit the simpler one-lookup form the
    protocol prompt shows as the everyday example."""
    if "lookups" in raw:
        lookups = raw["lookups"]
        if isinstance(lookups, dict):
            return [lookups]
        if not isinstance(lookups, list) or not lookups:
            raise ToolCallError(f"{tool}: lookups must be a non-empty list")
        return lookups
    # Slice-1 shorthand: the call object IS the single lookup, minus the
    # ``tool`` key. Strip ``tool`` so it doesn't trip the empty-lookup
    # check; everything else (query / has_neighbor / kind / k / ...) is
    # validated by ``_validate_search_lookup``.
    return [{k: v for k, v in raw.items() if k != "tool"}]


def _parse_plan(raw: dict) -> str:
    """Pull ``plan`` off the call as a single-line string ≤
    ``MAX_PLAN_CHARS``. Whitespace collapses (the model often emits
    multi-line plans the validator flattens); over-length plans
    truncate (the cap is a budget, not a fault). A missing or non-
    string ``plan`` returns ``""`` — the persona asks for it but the
    contract stays soft."""
    val = raw.get("plan")
    if not isinstance(val, str):
        return ""
    s = " ".join(val.split())
    if not s:
        return ""
    if len(s) > MAX_PLAN_CHARS:
        s = s[:MAX_PLAN_CHARS].rstrip()
    return s


def validate_tool_call(
    raw: dict,
    *,
    default_k: int = DEFAULT_K,
    max_count: int = MAX_COUNT,
    max_lookups: int = MAX_LOOKUPS,
) -> ToolCall:
    """Validate a raw tool call into a dispatch-ready ``ToolCall``.

    Enforces the registry contract: the tool must be known, the lookup
    array must be non-empty, each lookup must carry at least one filter
    (or a query), list-typed fields must coerce to strings, neighbor
    anchors must parse into ``(kind, record_id)`` pairs whose kind is
    known, and ``count`` is clamped to ``max_count``. Raises
    ``ToolCallError`` for the cases where guessing would be worse than
    an honest failure.

    Lookups beyond ``max_lookups`` are silently truncated rather than
    raising — the persona steers the model to the cloud cap; a model
    that over-fans-out (or a LOCAL turn whose mode-specific cap is
    tighter than the persona's stated number) still dispatches its
    first ``max_lookups`` legs instead of failing the whole turn.

    The ``plan`` field is parsed alongside; a missing or malformed
    plan silent-defaults to ``""``. The persona asks for it; the loop
    renders it back into the next hop's NOTES sidebar, but the
    contract stays soft.

    ``default_k`` / ``max_count`` / ``max_lookups`` keyword-default to
    the cloud caps; the turn loop overrides them per-mode for LOCAL
    (smaller chat-model context window + tighter fan-out — see
    ``LOCAL_DEFAULT_K`` / ``LOCAL_MAX_COUNT`` / ``LOCAL_MAX_LOOKUPS``).
    """
    if not isinstance(raw, dict):
        raise ToolCallError("tool call is not an object")
    tool = raw.get("tool")
    if tool not in TOOLS:
        raise ToolCallError(f"unknown tool {tool!r}")

    plan = _parse_plan(raw)

    if tool == "search":
        raw_lookups = _coerce_lookups(raw, tool)
        raw_lookups = raw_lookups[:max_lookups]
        lookups = tuple(
            _validate_search_lookup(
                lk, tool, default_k=default_k, max_count=max_count,
            )
            for lk in raw_lookups
        )
        return ToolCall(
            tool=tool, args={"lookups": lookups}, plan=plan,
        )

    # Unreachable: every key in TOOLS has a branch above. A new tool
    # added to the registry without a validation branch lands here loudly
    # rather than dispatching unvalidated.
    raise ToolCallError(f"no validator for tool {tool!r}")


def describe(call: ToolCall) -> str:
    """A user-facing description of what a tool call is doing, for the
    "searching your data…" status the UI renders while a lookup runs.

    Per lookup: ``<query> [<entry_type> * matching '<phrase>' *
    neighbors of N entries]`` — the brackets surface the filter
    signature so the user can see when the model restricted by kind /
    phrase / anchor rather than guessing the query alone explains the
    result. A filter-only lookup drops the query prefix and renders
    just the brackets. A multi-lookup call emits one line per lookup
    so each branch of the union is visible to the user.

    (The ReAct loop's ``Already searched this turn: …`` block records
    the raw call JSON via ``json.dumps(raw_call)`` for repetition
    prevention, not this string — so ``describe`` only owes the UI a
    readable label.)
    """
    if call.tool != "search":
        return call.tool
    lookups: tuple[Lookup, ...] = call.args.get("lookups") or ()
    if not lookups:
        return ""
    lines: list[str] = []
    for lk in lookups:
        bits: list[str] = []
        if lk.entry_type:
            bits.append("|".join(lk.entry_type))
        if lk.source:
            quoted = ", ".join(f"'{s}'" for s in lk.source)
            bits.append(f"in {quoted}")
        if lk.exact_match:
            quoted = ", ".join(f"'{s}'" for s in lk.exact_match)
            bits.append(f"matching {quoted}")
        if lk.has_neighbor:
            n = len(lk.has_neighbor)
            bits.append(
                f"neighbors of {n} {'entry' if n == 1 else 'entries'}"
            )
        brackets = f"[{' * '.join(bits)}]" if bits else ""
        if lk.query and brackets:
            lines.append(f"{lk.query} {brackets}")
        elif lk.query:
            lines.append(lk.query)
        elif brackets:
            lines.append(brackets)
    return "\n".join(lines) if lines else call.tool


def audit_record(call: ToolCall, *, result_count: int) -> dict:
    """The content-light structured record of a dispatched tool call for
    the replay/audit log: the tool, every lookup's validated args, and
    how many records came back across the union. Args are the model's
    already-validated typed parameters — the same shape the dispatcher
    ran — so the log is a faithful, replayable trace of what was asked
    without re-deriving it. No record bodies or vault content ride along
    here; the records themselves are surfaced (and audited for citation)
    through the resources block.
    """
    lookups: tuple[Lookup, ...] = call.args.get("lookups") or ()
    return {
        "tool": call.tool,
        "args": {"lookups": [lk.as_audit() for lk in lookups]},
        "result_count": int(result_count),
    }


__all__ = [
    "DEFAULT_K",
    "LOCAL_DEFAULT_K",
    "LOCAL_MAX_COUNT",
    "LOCAL_MAX_LOOKUPS",
    "MAX_COUNT",
    "MAX_K",
    "MAX_LOOKUPS",
    "MAX_PLAN_CHARS",
    "MAX_TOTAL_RESULTS",
    "SEARCHABLE_KINDS",
    "TOOLS",
    "Lookup",
    "ToolCall",
    "ToolCallError",
    "audit_record",
    "describe",
    "parse_tool_call",
    "validate_tool_call",
]
