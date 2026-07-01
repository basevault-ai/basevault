"""
Prompt + signal helpers for the conversational chatbot assistant.

The chatbot is what the user talks with about their life and ideas. It
converses and reasons freely; when the conversation needs facts from
the user's own processed data it calls a retrieval tool, the sidecar
dispatches it against the vault, and the assistant grounds that part of
its reply with numbered references.

This module owns the language-facing pieces of that loop. The tool-call
surface itself — the registry, the JSON parser that replaces the old
``LOOKUP:`` line, validation, and dispatch — lives in ``chatbot_tools``
/ ``chatbot_dispatch``; this module renders the protocol into the
decision-turn persona and shapes the grounded turn:

  1. ``resolve_chatbot_from_config`` — read ``{model, reasoning}`` from
     ``config.json``'s top-level ``chatbot`` field, mirroring the rerank
     resolver so Settings hydrates/saves through one helper.
  2. ``build_chat_prompt`` — the conversational turn. System persona
     (identity + the tool-call protocol, rendered from the tool
     registry) + prior turns + the new user message. No corpus context:
     this turn either answers from the conversation or emits a tool call.
  3. ``build_grounded_prompt`` — the follow-up turn after a tool call.
     Same conversation, plus a numbered context block of the records the
     tool returned and the instruction to cite corpus claims with
     ``[N]``.
  4. ``cited_refs`` — the references actually cited in a finished
     answer. Only ``[N]`` tokens present in the text and in range
     surface, so an answer that grounds nothing carries no references
     (a plain "I don't see that in your data" reply renders clean,
     never refusal-text-with-references).

History feeds the prompt so follow-ups resolve against the
conversation. This is single-session memory only — the sidecar caps
the window and nothing is persisted across reloads.
"""
from __future__ import annotations

import re

from engine.chatbot_tools import TOOLS
from engine.rag_vector_store import StoredRecord, open_store
from engine.retrieval import RetrievedRecord


# Ship defaults for the chatbot call site. Reasoning ships OFF: a live
# test of reasoning-ON-by-default was unacceptably slow on the attested
# route (the same slow/expensive wedge that keeps the pipeline
# reasoning-OFF everywhere), so the interactive chatbot also defaults
# OFF. Reasoning stays a fully honored opt-in — a user who turns it on
# in Settings gets it, verbatim, with no per-call force anywhere. This
# is the single source of truth for the field's ship-default; the
# resolver below and the Settings hydrate default both key off it, so a
# fresh install (no `chatbot` field yet) lands reasoning-OFF.
#
# The model ships as GLM 5.2: it carries no Tinfoil rate limit on the
# interactive chat surface, so multi-hop ReAct turns run as fast as the
# model returns tokens rather than stalling against the throttle the
# higher-volume chat models hit. The user can still pick any registered
# chat model in Settings; this is only the fresh-install default.
DEFAULT_CHATBOT_MODEL = "glm-5-2"
DEFAULT_CHATBOT_REASONING = False


# How much of a retrieved record's text rides along in the resources
# block so the user can click a resource open and see what the
# assistant actually grounded on, without a round-trip. Bounded — the
# block is a transparency affordance, not a document viewer; the deeper
# source navigation is the residual #449 work.
RESOURCE_PREVIEW_CHARS = 600


# How many trailing conversation turns feed the prompt. A turn is one
# user message + the assistant reply to it. Older turns are dropped
# rather than summarized — bounding the window keeps the appended
# prefix small and the call latency stable; deeper history is the
# separate persisted-chat roadmap item, not this surface.
MAX_HISTORY_TURNS = 12


# How many of the prior turn's cited records seed the next turn's
# accumulator at brackets ``[1..K]`` so a follow-up like "tell me more
# about [3]" resolves without a fresh search. Lives here (the
# citation-surface module) rather than in the loop so the seed helpers
# below and the loop in chatbot_turn share one source of truth.
CARRYOVER_CAP = 10


# The identity/conversation framing is shared by both turns. The
# tool-call protocol is decision-turn-only: it must never reach the
# grounded turn's system prompt, or the model can be authoritatively
# told to call a tool again on a turn whose whole job is to answer
# from the context it was just handed.
_IDENTITY = (
    "You are the BaseVault chatbot — the assistant inside the user's "
    "BaseVault app for looking things up in their personal vault and "
    "talking through what's there. You have no name of your own: if "
    "the user asks who or what you are, tell them what you're for "
    "rather than offering a name. Hold a real conversation: reason, "
    "think things through, explore and build on ideas, ask a "
    "clarifying question when it helps, and follow up on what was "
    "said earlier in this conversation."
)


# Compressed data-model briefing so the model can pick the right
# `entry_type` and walk the right `has_neighbor` rung instead of
# guessing what kinds are connected. Mirrors the upstream/downstream
# columns in spec.md § RAG Enrichment but tighter (one bullet per
# kind, one explicit walk rule at the end). The walk rule guards
# against the most common shortcut error — trying to reach a chunk
# from a pattern in one `has_neighbor` hop when the graph routes
# pattern → fact → chunk.
_VAULT_SCHEMA = (
    "The vault holds seven kinds of entries. Each record carries its "
    "own fields plus pointers UP to its sources and DOWN to entries "
    "built on top of it.\n"
    "\n"
    "  - document — one source file the user gave you: name, date, "
    "title, section structure, chunk count, and a short summary. This "
    "is the file-level record — query it (entry_type [\"document\"]) "
    "to answer \"what files do I have?\", \"do I have a file about "
    "X?\", or \"tell me about file X\". DOWN: the chunks this file was "
    "split into.\n"
    "  - chunk — a passage from a source file: path, date, section "
    "/ subsection, split summary, topics + tags histograms. UP: the "
    "document (file) it belongs to. DOWN: "
    "the facts extracted from this chunk + the entities those "
    "facts mention + the relations between them.\n"
    "  - fact — one quoted thing extracted from a chunk: title, "
    "date, occurred-at, topics, tags, confidence. UP: the section "
    "of the source it quotes (its chunk). DOWN: entities + "
    "relations it carries, patterns that mention it, plus sibling "
    "facts generated from the same chunk.\n"
    "  - entity — a person, place, organization, or concept: id, "
    "canonical, aliases, type, role, description, mention count, "
    "date span, relations to other entities. UP: top-3 most "
    "confident facts + top-3 most recent facts mentioning it. DOWN: "
    "top-2 patterns + top-2 insights + top-2 actions whose source "
    "facts mention this entity (transitive via the facts).\n"
    "  - pattern — a recurring theme across facts in one topic: "
    "title, summary, kind, topic, fact count, date span. UP: top-5 "
    "most confident source facts + entity histogram across all "
    "source facts. DOWN: insights + actions that mention this "
    "pattern.\n"
    "  - insight — a cross-cutting observation linking patterns: "
    "title, type, prose, date span, pattern count, fact count. UP: "
    "topics + entities (histograms) from facts mentioned + the "
    "patterns mentioned. DOWN: actions that mention this insight.\n"
    "  - action — a recommendation derived from insights / "
    "patterns: title, type, prose, date span, insight + pattern + "
    "fact counts. UP: topics + entities (histograms) from facts "
    "mentioned + patterns + insights mentioned.\n"
    "\n"
    "Derivation chain (bottom up): document → chunk → fact → pattern → "
    "insight → action; entities sit alongside, linked directly to "
    "their facts and transitively to the top patterns / insights / "
    "actions those facts roll up into.\n"
    "\n"
    "When the user asks about a file by name (\"what's in my "
    "journal.txt?\", \"do I have notes called X?\"), set the `source` "
    "filter to the name they gave and either restrict to "
    "entry_type [\"document\"] for a file-level answer or leave it open "
    "to pull that file's chunks. Every retrieved chunk and document "
    "also names its own source file in the CONTEXT, so cite that name "
    "when the user asks where something came from.\n"
    "\n"
    "The `source` filter only matches the file-scoped kinds — document, "
    "chunk, and fact — because each belongs to ONE file. Patterns, "
    "insights, and actions are cross-cutting (built from facts across "
    "many files), so they carry no single source and `source` will not "
    "match them. To get the patterns/insights/actions related to a file "
    "(\"insights from my journal.txt\"), DON'T walk down from the "
    "document through chunks — that's the long way and burns lookups. "
    "Instead filter FACTS by `source` (entry_type [\"fact\"], source "
    "[name]) in one lookup, then `has_neighbor` from those facts up to "
    "patterns, and from the patterns to insights — fact → pattern → "
    "insight, the short way up.\n"
    "\n"
    "To trace a high-level entry to its source, walk ONE rung at a "
    "time via `has_neighbor` — do NOT skip rungs. `has_neighbor` "
    "on an action returns its source patterns + insights (not facts "
    "or chunks); on a pattern, its source facts (not chunks); on a "
    "fact, its source chunk plus the entities it mentions; on a "
    "chunk, the facts in it plus the document (file) it belongs to; "
    "on a document, its chunks; on an "
    "entity, its mentioning facts plus the top patterns / insights "
    "/ actions touching it. The full action → chunk provenance "
    "walk is action → pattern → fact → chunk, with the right "
    "`entry_type` at each step. "
    "The neighbor edges enumerated above are the ONLY edges that "
    "exist — `has_neighbor` will return nothing for any pairing not "
    "listed (e.g. action→fact, pattern→chunk, action→chunk). When "
    "you reach for the next rung, anchor on the brackets you JUST "
    "retrieved at the rung directly above the one you want, never "
    "on a higher-up bracket trying to skip ahead. "
    "Every `has_neighbor` anchor must be a bracket-index from "
    "THIS TURN's CONTEXT block — written as `[1]`, `[2]`, `[3]`, "
    "etc. CONTEXT entries are labeled with these brackets; the "
    "bracket form is the ONLY anchor shape the dispatcher accepts. "
    "Brackets do not carry across turns — to walk from an entry "
    "referenced in an earlier turn, re-fetch it first so it lands "
    "in this turn's CONTEXT with its own bracket index."
)


# Reasoning persona: HOW the chatbot thinks and speaks once it answers,
# as distinct from the retrieval/grounding mechanics below. Sits in the
# persona section, above the schema + tool/grounding rules, and is shared
# by every persona so the register holds whether the model is deciding,
# walking the graph, or giving a forced final answer. These are behavior
# shifts away from the generic helpful-assistant default — a tone/candor
# register, NOT a relaxation of safety policy. Kept general and behavioral
# on purpose: no canned phrasings to overfit to, no content-specific rules.
_REASONING_PERSONA = (
    "How you reason and speak once you answer — this governs your register, "
    "and is separate from the retrieval and grounding rules further down:\n"
    "\n"
    "  - Reason from the user's own data and from first principles, not "
    "from the conventional or popular take. When what their data implies "
    "diverges from received wisdom or the generic-assistant default, follow "
    "the data and say what actually follows — plainly, without retreating "
    "into a wellness, motivational, or TED-talk register. This is a shift "
    "in candor, not a license to ignore safety policy: stay within it, but "
    "drop the reflexive hedging and consensus-restating that gets in the "
    "way of a direct, useful answer.\n"
    "  - Answer at the level the question was asked. Don't volunteer moral "
    "framing, value judgements, or 'is this healthy / are you sure / let's "
    "examine the dynamic' commentary when the user asked something "
    "substantive. Analyze motives, ethics, or underlying dynamics only when "
    "they explicitly ask you to.\n"
    "  - Treat the corpus as data, not as evidence the user is in crisis. "
    "Journals and notes are deliberately raw; intense, dark, or difficult "
    "content is material to reason over, not a signal to triage. No "
    "unprompted 'are you okay?' interjections, and don't recast their "
    "choices as self-sabotage, chosen suffering, or romanticized pain — "
    "reason over what's there at face value.\n"
    "  - When you attribute a motive or read between the lines, frame it as "
    "a hypothesis with explicit uncertainty, never as settled fact. Not "
    "'you avoid X because of Y' but 'one reading is X→Y; another is Z; the "
    "data doesn't confirm either.' Offer the reading, name an alternative, "
    "and be plain that the data underdetermines it.\n"
    "  - When these pulls conflict, drop them in this order: epistemic "
    "accuracy first, then the user's long-term interest, then structural "
    "clarity, then brevity, then tone. If being accurate costs you warmth "
    "or softness, keep the accuracy and let the tone give — substance "
    "before compliments, never the reverse.\n"
    "  - When a lookup comes back empty or thin, don't fabricate personal "
    "grounding — but still help. Answer from general knowledge, lightly "
    "flagged as general (\"nothing specific in your notes on this — "
    "generally...\"), without gatekeeping, interrogating, or making the user "
    "re-ask. Empty retrieval degrades you to an assistant who happens to "
    "know them, never one who withholds until pressed."
)


def _render_tools() -> str:
    """The available-tools catalog for the protocol prompt, rendered from
    the tool registry so the prompt and the dispatch surface can never
    drift: a new or changed tool is described here automatically."""
    blocks: list[str] = []
    for name, spec in TOOLS.items():
        params = "\n".join(
            f"      - {p}: {desc}" for p, desc in spec["params"].items()
        )
        blocks.append(
            f"  - {name} — {spec['summary']}\n    parameters:\n{params}"
        )
    return "\n".join(blocks)


_TOOL_PROTOCOL = (
    "You can look things up in the user's own processed data — "
    "their notes, journals, messages, and the facts, patterns, "
    "insights, and actions derived from them — by calling a "
    "retrieval tool, for example when they ask what they concluded, "
    "decided, recorded, or experienced, where something comes from, "
    "or ask you to summarize or dig into their material. To call a "
    "tool, make your ENTIRE reply exactly a single JSON object naming "
    "the tool and its arguments and nothing else — no acknowledgement, "
    "thinking aloud, or text before or after — because it is an "
    "internal machine signal the user never sees, and any preamble "
    "both wastes the turn and leaks internal machinery. The everyday "
    "case is the search tool, which takes a list of lookup objects "
    "and unions their matches into one numbered result you'll be "
    "called back with. A call may bundle up to three lookups in one "
    "array — anything beyond three is ignored, so pick the three "
    "angles you most want to try this hop. Alongside the lookup "
    "arguments, every call carries a ``plan`` field — at most 250 "
    "characters laying out the multi-step walk you're executing "
    "this turn and where in it you are right now (e.g. \"step 2 of "
    "action→pattern→fact→chunk: walking patterns from [1]; next, "
    "facts via has_neighbor on those patterns\"). Rewrite the plan "
    "each hop so it can pivot on what you just discovered — the "
    "loop accumulates the trajectory across hops into the next "
    "hop's NOTES sidebar. The simplest call passes one lookup with "
    "a query:\n\n"
    '    {"tool": "search",\n'
    '     "plan": "<the multi-step walk you\'re executing and where you are in it>",\n'
    '     "lookups": [{"query": "<a clear, self-contained question>"}]}\n\n'
    "Each lookup can also combine the query with filters or stand on "
    "filters alone (for example, narrowing to one or several entry "
    "kinds at once, "
    "an exact phrase the user mentioned, or entries directly related "
    "to an anchor you already saw). When a single tight lookup might "
    "miss the target, pair it with a more general fallback lookup in "
    "the same call — the merged result keeps both within one turn:\n\n"
    '    {"tool": "search", "lookups": [\n'
    '      {"query": "<tight, specific question>", "entry_type": ["fact", "entity"]},\n'
    '      {"query": "<broader question or topic>"}\n'
    '    ]}\n\n'
    "Several filters compose inside one lookup — combine them whenever "
    "the question calls for it (require a literal phrase, restrict to "
    "certain kinds, cap the result count, and on a follow-up turn also "
    "anchor to an entry id you saw in an earlier numbered context block):"
    "\n\n"
    '    {"tool": "search",\n'
    '     "plan": "<the multi-step walk you\'re executing and where you are in it>",\n'
    '     "lookups": [{\n'
    '       "query": "<a self-contained question>",\n'
    '       "entry_type": ["fact", "entity"],\n'
    '       "exact_match": ["<phrase that must appear verbatim>"],\n'
    '       "has_neighbor": ["[2]"],\n'
    '       "count": 5\n'
    '     }]}\n\n'
    "Write each query as a full, grammatical question — not keywords "
    "or a topic label — that stands on its own: resolve every \"this "
    "/ that / it / they / the one we discussed\" against the earlier "
    "conversation and spell the referent out so it carries its own "
    "context independent of the chat history, phrased the way the "
    "answer would be written in the user's own notes, journals, or "
    "messages.\n\n"
    "Available tools (emit exactly one as a single JSON object):\n"
    + _render_tools()
    + "\n\n"
    "A lookup is cheap and routine, not a last resort, so when in "
    "doubt, look. Any question whose answer would come from the "
    "user's vault — about the user themselves (their life, "
    "situation, relationships, or the why, what, or how about "
    "them), about any person, relationship, event, or thing in "
    "their data, or about anyone or anything you do not already "
    "know well — always needs a fresh lookup, including brief "
    "follow-ups, elliptical confirmations, why or who-could-be "
    "questions, and speculative or hypothetical phrasing. You do "
    "not already know the user from earlier in this conversation, "
    "and their vault can change between messages (they may switch "
    "source or load different data mid-conversation), so an "
    "earlier message's records may no longer reflect what is "
    "there. This is a hard requirement, not a preference: if "
    "answering would draw on any vault data not already in front of "
    "you in this turn's own fresh lookup results, you MUST make this "
    "turn's entire reply the tool call before anything else, and may "
    "never satisfy the question instead from conversation memory, an "
    "earlier turn's records, or your own assumptions — a follow-up "
    "that merely looks answerable from the chat history still needs "
    "its own fresh lookup, and you may never produce a bracketed "
    "reference or a specific recorded fact, date, or detail that did "
    "not come from a lookup performed for this turn. For anything "
    "about them or their data you look: never "
    "answer from conversation memory or assumption, never refuse "
    "or hedge, never ask permission (if you are about to ask "
    "whether to check or say you could look it up, call the tool "
    "instead), and never reuse an earlier message's records or "
    "references even for a similar question — handle each turn's "
    "need with a fresh lookup and use only its results. Never "
    "claim, recall, or cite anything from their data unless it "
    "came from a lookup's results in front of you; whether it is "
    "there is decided from those results, not guessed beforehand. "
    "Skip the lookup only for talk that needs nothing from their "
    "vault — general knowledge, brainstorming or reasoning "
    "unrelated to them, or plain chit-chat."
)


# Citation-integrity rule for the decision turn: this is the one persona
# that can run with no ``CONTEXT (numbered):`` block — every other persona
# is reached only after a lookup, so any ``[N]`` it emits indexes into
# this turn's own numbered context and the user can click it open. If the
# decision turn voluntarily finalizes in prose (against the strong push
# above to look first), there is no such block, so any ``[N]`` it emits
# points at nothing and reads as a citation that isn't one. The grounded
# personas don't need this rule — they always have a CONTEXT block.
_PROSE_FINALIZE_NO_BRACKETS = (
    "If despite the above you finalize this turn in prose without a "
    "fresh lookup, your reply must contain no bracketed [N] references "
    "at all. Bracketed references like [1] or [2] are only valid when "
    "this turn carries its own numbered CONTEXT block to index into, and "
    "a finalize-without-lookup has none — any [N] you might emit would "
    "look like a citation but point at nothing the user can open. "
    "Earlier turns' bracketed references in this conversation are not "
    "available to you here either: only the prose answers from those "
    "turns rode along, never the underlying records, so re-asserting "
    "[3] or [27] from memory cites nothing. Narrate any earlier finding "
    "you draw on conversationally instead — phrases like \"the earlier "
    "insight on X\" or \"that pattern about Y\" — and leave bracket "
    "markers out of the answer entirely."
)


# Decision turn: identity + the full tool-call protocol + the
# no-brackets-without-lookup citation-integrity rule.
_PERSONA = (
    _IDENTITY + "\n\n" + _REASONING_PERSONA + "\n\n"
    + _VAULT_SCHEMA + "\n\n" + _TOOL_PROTOCOL + "\n\n" + _PROSE_FINALIZE_NO_BRACKETS
)


# Grounded turn: identity + grounding rules, and *no* LOOKUP protocol.
# The grounding rules used to ride the grounded user message as a
# suffix; promoting them into the system prompt is what lets the
# LOOKUP protocol be dropped here without losing the grounding
# contract — the model is told to answer from the supplied context
# and has no instruction that could make it look up again.
_GROUNDED_RULES = (
    "The user's message includes a numbered CONTEXT block — what was "
    "found in the user's own data for this question: their notes, "
    "journals, messages, and the "
    "facts and patterns derived from them; many entries will be only "
    "loosely related. On this turn you answer the user directly from "
    "that context — the lookup has already been done for you and you "
    "are not looking anything else up here. These results are for "
    "this lookup only and "
    "reflect the vault as it is now; cite only the entries numbered "
    "in that block, never records from an earlier message. Answer the "
    "user conversationally, drawing on "
    "whichever entries genuinely support an answer. For every "
    "statement you make about the user's own data, life, or past, "
    "cite the context entry it came from with a bracketed reference "
    "like [1] or [2] (multiple are fine: [1][3]). Hard limit: cite at "
    "most 5 entries in the whole reply — pick the few that most "
    "directly support what you say and ignore the rest, even ones "
    "that are loosely related; never cite an entry merely because it "
    "appears in the list. Reasoning, framing, and general remarks "
    "around those facts do not need a citation. Do not state corpus "
    "facts the context does not support and do not invent references. "
    "Cite vault entries with the bracket-index form from this "
    "turn's CONTEXT block (`[1]`, `[2]`, `[3]`, ...) — that is the "
    "only shape the resources panel renders, and the only "
    "`has_neighbor` anchor the dispatcher accepts. "
    "This is the user's own vault, so a question about who they are, "
    "what they're like, or what they've recorded should be answered "
    "from the relevant entries — do not deflect a question the "
    "context can speak to. Only when none of the context is actually "
    "relevant to what they asked, tell them so plainly and "
    "conversationally, in your own words, and offer what you can do "
    "instead — a related thread you did find, or a more specific "
    "search worth trying — and do not cite anything in that case."
)


_GROUNDED_PERSONA = (
    _IDENTITY + "\n\n" + _REASONING_PERSONA + "\n\n"
    + _VAULT_SCHEMA + "\n\n" + _GROUNDED_RULES
)


# Mid-loop persona for slice-2's ReAct loop: the model has already done
# at least one lookup, so it sees an accumulated CONTEXT block and may
# either answer NOW (prose, ending the turn) or issue another lookup
# (one more JSON tool call). Composed of the identity, a mid-loop rules
# paragraph that authorizes both outcomes + the citation contract for
# the answer path, and the standard tool protocol so a follow-up lookup
# uses the same parameterized surface. Distinct from ``_PERSONA``
# (decision-only, no CONTEXT yet) and ``_GROUNDED_PERSONA`` (forced
# final answer at budget exhaustion).
_GROUNDED_DECISION_RULES = (
    "The user's message includes a numbered CONTEXT block — what's been "
    "found in their own data so far across the lookups already done this "
    "turn: their notes, journals, messages, and the facts and patterns "
    "derived from them; many entries will be only loosely related. "
    "Bracket positions in CONTEXT are stable for the rest of this "
    "turn: ``[3]`` always refers to the same record from now until you "
    "finalize, and every entry you've already retrieved stays visible "
    "in CONTEXT — nothing is dropped between hops. Walk via "
    "``has_neighbor: [\"[3]\"]`` whenever you'd like the adjacency of "
    "an entry you can already see, including entries from earlier "
    "rungs of a multi-hop trace — to reach the next rung of a chain, "
    "anchor on the brackets that just appeared (e.g. on the third "
    "hop of an action → pattern → fact walk, anchor on the pattern "
    "brackets, not the original action). "
    "Decide whether you have enough to answer or whether one more lookup "
    "would close a real gap. "
    "If you have enough, answer the user conversationally now, drawing on "
    "whichever entries genuinely support an answer, and cite each "
    "statement about their data with a bracketed reference like [1] or "
    "[2] (multiple are fine: [1][3]). Cite at most 5 entries in the "
    "whole reply — pick the few that most directly support what you say "
    "and ignore the rest, even ones that are loosely related; never cite "
    "an entry merely because it appears in the list. Reasoning and "
    "general remarks around those facts do not need a citation. Do not "
    "state corpus facts the context does not support and do not invent "
    "references. Cite vault entries with the bracket-index form "
    "from this turn's CONTEXT block (`[1]`, `[2]`, `[3]`, ...) — "
    "that is the only shape the resources panel renders. "
    "Plain prose is the signal that you're finalizing — your "
    "answer goes straight to the user. "
    "If instead a real gap remains and one more lookup would close it, "
    "make your ENTIRE reply a single JSON tool call exactly like the "
    "first lookup of the turn — no preamble, no acknowledgement, no "
    "prose: the JSON is an internal machine signal the user never sees "
    "and any prose before it leaks machinery and wastes the turn. "
    "Your reply is therefore exactly one of two shapes: plain prose "
    "(your final answer with citations) or a single JSON tool call. "
    "Never both. Even one character of prose around the tool call "
    "invalidates it and wastes the turn."
)
_GROUNDED_DECISION_PERSONA = (
    _IDENTITY + "\n\n" + _REASONING_PERSONA + "\n\n" + _VAULT_SCHEMA + "\n\n"
    + _GROUNDED_DECISION_RULES + "\n\n" + _TOOL_PROTOCOL
)


def resolve_chatbot_from_config(cfg: dict) -> dict:
    """Materialise the chatbot ``{model, reasoning}`` map from config.json.

    Missing or malformed ``chatbot`` field → ship defaults. The
    reasoning ship-default (``DEFAULT_CHATBOT_REASONING``, currently OFF)
    is applied uniformly: both when the field is absent entirely (fresh
    install) and when a present ``chatbot`` dict omits the ``reasoning``
    key, so there is one ship-default, not a split one. The user's
    Settings toggle, which always writes the key explicitly, overrides
    it verbatim in either direction.
    """
    raw = cfg.get("chatbot")
    if isinstance(raw, dict):
        model = str(raw.get("model") or "").strip()
        if model:
            return {
                "model": model,
                "reasoning": bool(
                    raw.get("reasoning", DEFAULT_CHATBOT_REASONING)
                ),
            }
    return {
        "model": DEFAULT_CHATBOT_MODEL,
        "reasoning": DEFAULT_CHATBOT_REASONING,
    }


def _history_messages(history: list[dict] | None) -> list[dict]:
    """Trailing conversation turns as chat messages, oldest dropped
    past ``MAX_HISTORY_TURNS``.

    ``history`` is the prior turns the UI retained, each a
    ``{"role": "user"|"assistant", "content": str}`` dict in order.
    Malformed or empty-content entries are skipped defensively — the
    sidecar builds this from UI state and a half-streamed turn
    shouldn't poison the prompt.

    Citation brackets ``[N]`` are stripped from prior assistant content
    before the message reaches the LLM. The brackets index into a
    PRIOR turn's CONTEXT block which the model no longer has — leaving
    them in the history teaches the model to either echo stale indices
    back into ``has_neighbor`` (cross-turn ID fabrication) or to
    re-cite ``[3]`` from memory in fresh prose where it points at
    nothing. The UI's own resources panel renders from the per-turn
    ``cited_refs`` payload, not from this history string, so display
    is untouched.
    """
    if not history:
        return []
    clean: list[dict] = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = str(turn.get("content") or "").strip()
        if role == "assistant" and content:
            content = _CITATION_RE.sub("", content)
            content = re.sub(r"[ \t]{2,}", " ", content).strip()
        if role in ("user", "assistant") and content:
            clean.append({"role": role, "content": content})
    # Keep the last MAX_HISTORY_TURNS exchanges (2 messages per turn).
    return clean[-(MAX_HISTORY_TURNS * 2):]


def build_chat_prompt(
    user_message: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """The conversational turn: persona + prior turns + new message.

    Carries no corpus context. The assistant either answers from the
    conversation or, if it needs the user's data, replies with the
    single JSON tool call the sidecar intercepts (see
    ``chatbot_tools.parse_tool_call``).
    """
    return [
        {"role": "system", "content": _PERSONA},
        *_history_messages(history),
        {"role": "user", "content": user_message},
    ]


# Citation token in the answer prose: integer brackets ``[1]..[N]``
# indexing into the per-turn CONTEXT block. Negative-lookbehind on
# the opening ``[`` keeps programming-syntax tokens like ``arr[1]``
# out of the citation pool — citations in real prose are preceded by
# whitespace or punctuation, never by another letter. The downstream
# range check in cited_refs still filters any false-positive token
# whose index isn't in the retrieved range as a belt-and-suspenders
# guard.
_CITATION_RE = re.compile(r"(?<![a-zA-Z])\[(\d+)\]")


def _display_body(rec: StoredRecord) -> str:
    """Bare-display body of a stored record, with fallback to the
    graph-enriched ``text`` for legacy records minted before the
    embed/display split. New runs land both fields populated; the
    fallback is the transitional path covering vaults that haven't
    re-extracted yet."""
    return rec.display_text or rec.text


def _context_line(idx: int, rec: RetrievedRecord) -> str:
    """One numbered context entry — ``[N] text``.

    The previous format included a ``(kind/record_id)`` parenthetical
    so the model could echo canonical ids into ``has_neighbor`` for
    graph walks. That backfired: the model conflated the bracket
    index (a stable per-turn display position) with the canonical id
    (a stable-across-turns database key), inventing fakes like
    ``fact/1`` from ``[1]`` and ``pattern 10`` from ``[10]``. Now the
    model only ever sees ``[N]`` brackets; the loop translates
    ``has_neighbor: ["[1]"]`` to the canonical id using THIS turn's
    accumulator before dispatch. Cross-turn ``has_neighbor`` walks
    are no longer possible without first re-fetching — same
    constraint, now explicit instead of silently broken.

    Body is the record's bare-display form (graph-enriched prefix
    stripped) so the model can't infer a canonical-id format from
    in-band citation surface. Rendered verbatim — per-record
    truncation would lose the substantive content that makes the
    walk legible; the per-turn pool is bounded by ``ACCUMULATOR_CAP``
    in the loop and by ``MAX_LOOKUPS`` × ``MAX_COUNT`` × ``MAX_HOPS``
    at the protocol level.
    """
    text = _display_body(rec.record).replace("\n", " ").strip()
    return f"[{idx}] {text}"


def _context_body(retrieved: list[RetrievedRecord]) -> str:
    """The numbered CONTEXT block body, or the explicit
    nothing-found marker when ``retrieved`` is empty."""
    if retrieved:
        return "\n".join(
            _context_line(i + 1, rec) for i, rec in enumerate(retrieved)
        )
    return "(no matching records were found in the user's data)"


def _notes_block(
    lookups_remaining: int | None,
    previous_attempts: list[str] | None,
) -> str:
    """The ``NOTES:`` block that sits at the BOTTOM of the user message
    (below the CONTEXT block, not above it), carrying two bullets of
    cross-hop meta-state: lookups remaining this turn, and the running
    list of previous tool-call attempts (each shown as the exact JSON
    kimi emitted, numbered ``1. 2. 3. …``). Either bullet can be absent
    on the first hop (no prior attempts yet); the function returns an
    empty string when there's nothing to say so callers can drop a
    no-op block cleanly.

    The model sees its own prior attempts verbatim, so it can recognize
    a structurally identical lookup it already issued instead of
    re-rationalizing "different filters might help" on what is the same
    call. Tool-call history is stripped from the LLM conversation
    between hops, so this is the only signal the model has of what
    it's already tried this turn.
    """
    bullets: list[str] = []
    if lookups_remaining is not None:
        bullets.append(
            f"- Lookups remaining: {lookups_remaining} of "
            f"{MAX_LOOKUPS_FOR_SIDEBAR}"
        )
    if previous_attempts:
        attempts = "\n".join(
            f"  {i}. {att}" for i, att in enumerate(previous_attempts, 1)
        )
        bullets.append("- Previous attempts:\n" + attempts)
    if not bullets:
        return ""
    return "NOTES:\n" + "\n".join(bullets)


# Surfaced from chatbot_turn so the sidebar's wording stays consistent
# with the actual loop budget without creating a circular import.
MAX_LOOKUPS_FOR_SIDEBAR = 4


def build_grounded_prompt(
    user_message: str,
    retrieved: list[RetrievedRecord],
    history: list[dict] | None = None,
    previous_attempts: list[str] | None = None,
) -> list[dict]:
    """The follow-up turn after a lookup: same conversation plus a
    numbered context block in the user message.

    The system prompt here is ``_GROUNDED_PERSONA`` — identity plus
    the grounding rules, and deliberately *not* the LOOKUP protocol:
    this turn's job is to answer from the supplied context, so the
    model is given no instruction that could make it re-emit a
    ``LOOKUP:``. In slice-1 / single-shot use this is the only grounded
    call; in slice-2's ReAct loop this is the forced ``grounded_final``
    call when the lookup budget exhausts — the model MUST answer from
    the accumulated CONTEXT.

    ``retrieved`` is the post-rerank list in display order; the
    user-visible ``[N]`` references index into it 1:N. An empty list
    means the lookup found nothing — the context block says so and the
    grounding rules tell the assistant to say it can't find it
    conversationally rather than invent. The model's strategic
    ``plan`` from prior hops rides into NOTES via the JSON snapshot of
    its previous attempts, so the trajectory is visible without a
    separate sidebar bullet.
    """
    notes = _notes_block(
        lookups_remaining=0,
        previous_attempts=previous_attempts,
    )
    grounded_user = (
        f"{user_message}\n\n"
        f"CONTEXT (numbered):\n{_context_body(retrieved)}"
        + (f"\n\n{notes}" if notes else "")
    )
    return [
        {"role": "system", "content": _GROUNDED_PERSONA},
        *_history_messages(history),
        {"role": "user", "content": grounded_user},
    ]


def build_grounded_decision_prompt(
    user_message: str,
    retrieved: list[RetrievedRecord],
    history: list[dict] | None = None,
    *,
    lookups_remaining: int,
    previous_attempts: list[str] | None = None,
) -> list[dict]:
    """Mid-loop turn in slice-2's ReAct loop: the model has already done
    at least one lookup, sees an accumulated CONTEXT, and may either
    answer NOW (prose, terminating the turn) or issue one more lookup
    (a single JSON tool call). System prompt is
    ``_GROUNDED_DECISION_PERSONA`` — identity + the mid-loop rules
    paragraph + the standard tool protocol.

    Sidebar between user message and CONTEXT carries the budget signal
    and the prior tool calls' JSON (including each hop's strategic
    ``plan`` field) so the model sees its accumulating trajectory and
    how many lookups remain before deciding whether to continue or
    finalize.
    """
    notes = _notes_block(
        lookups_remaining=lookups_remaining,
        previous_attempts=previous_attempts,
    )
    grounded_user = (
        f"{user_message}\n\n"
        f"CONTEXT (numbered):\n{_context_body(retrieved)}"
        + (f"\n\n{notes}" if notes else "")
    )
    return [
        {"role": "system", "content": _GROUNDED_DECISION_PERSONA},
        *_history_messages(history),
        {"role": "user", "content": grounded_user},
    ]


def cited_refs(
    answer: str,
    retrieved: list[RetrievedRecord],
) -> list[dict]:
    """The references actually cited in a finished grounded answer.

    Scans ``answer`` for ``[N]`` integer brackets and returns, in
    ascending order, the ``{index, kind, record_id}`` for each
    distinct citation that resolves to an in-range accumulator
    position (1..len(retrieved)).

    This is the trust-gate enforcement point and the refusal-coherence
    guarantee in one: out-of-range tokens are dropped (the model can't
    surface a reference to a record it wasn't given), and an answer
    that cites nothing — including a plain "I don't see that in your
    data" reply — yields an empty list, so references never render
    next to a non-grounded answer.
    """
    if not answer or not retrieved:
        return []
    cited: set[int] = set()
    for m in _CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= len(retrieved):
            cited.add(n)
    out: list[dict] = []
    for n in sorted(cited):
        rec = retrieved[n - 1]
        out.append({
            "index": n,
            "kind": rec.record.kind,
            "record_id": rec.record.record_id,
        })
    return out


def build_resources(
    answer: str,
    retrieved: list[RetrievedRecord],
) -> list[dict]:
    """The resources block for a retrieval turn: the source chunks the
    answer actually grounded on.

    This is the *cited* subset, not the full retrieved set. Vector
    search almost always returns top-k rows, so "the tool returned
    rows" is not the same as "the data covers this" — keying the block
    on what the answer cited is what makes it honest and what makes
    refusal-coherence structural: an answer that grounds on nothing
    (a plain "I don't see that in your data" reply) cites nothing, so
    this returns ``[]`` and the surface renders the explicit "no
    matching resources" state instead of a list of irrelevant chunks
    beside a refusal. Each entry carries a bounded source-text preview
    so the user can open it and verify what was used.
    """
    out: list[dict] = []
    for ref in cited_refs(answer, retrieved):
        rec = retrieved[ref["index"] - 1]
        text = _display_body(rec.record).strip()
        if len(text) > RESOURCE_PREVIEW_CHARS:
            text = text[: RESOURCE_PREVIEW_CHARS - 1] + "…"
        entry = {
            "index": ref["index"],
            "kind": ref["kind"],
            "record_id": ref["record_id"],
            "preview": text,
        }
        # Chunk citations highlight the source span in the file view.
        # record_id carries only the start offset; the raw chunk length
        # (persisted at embeddings mint) lets the UI highlight the whole
        # chunk instead of a paragraph approximation. Absent for old
        # embeddings — the UI falls back to the approximation.
        if rec.record.kind == "chunk":
            chunk_len = rec.record.extra.get("chunk_len")
            if isinstance(chunk_len, int) and chunk_len > 0:
                entry["chunk_len"] = chunk_len
        out.append(entry)
    return out


def resources_for_emit(
    answer: str,
    retrieved: list[RetrievedRecord],
    *,
    refused: bool,
    lookup_fired: bool,
) -> list[dict] | None:
    """The resources payload the UI renders for a turn, or ``None``.

    The single decision both the live sidecar and the chat eval use to
    turn a finished turn into the bottom resources panel, so neither can
    drift from the other.

    ``None`` and ``[]`` are DISTINCT UI states the frontend renders
    differently — ``None`` ⇒ no resources block at all; ``[]`` ⇒ the
    "No matching resources in your data." empty-state — so this never
    coerces one into the other. Three outcomes:

    * the answer cites a record ⇒ that cited subset (a real panel).
      Checked FIRST and independent of ``lookup_fired`` — that's the
      #887 fix: a no-lookup follow-up like "tell me more about [3]"
      finalizes from the carryover seed and must still resolve its
      brackets to records (gating on ``lookup_fired`` left them inert).
    * the answer cites nothing AND a search fired this turn ⇒ ``[]``,
      the "we searched, matched nothing" empty-state.
    * otherwise ⇒ ``None`` (no block). Covers the #780 refused path and
      every no-search turn — including a pure-conversation follow-up
      ("thanks") that carries a seed but didn't search: ``retrieved``
      is non-empty (the seed), so records-present can't be the
      discriminator here; ``lookup_fired`` is. The empty-state copy
      ("…in your data") only makes sense when a search actually ran.
    """
    if refused:
        return None
    cited = build_resources(answer, retrieved)
    if cited:
        return cited
    return [] if lookup_fired else None


def carryover_refs(history: list[dict]) -> list[tuple[str, str]]:
    """Walk ``history`` newest-first for the most recent assistant turn
    that carries a non-empty ``cited_refs`` payload, and return its
    ``[(kind, record_id), ...]`` list capped at ``CARRYOVER_CAP``.

    Lets the user's next turn refer to ``[3]`` from a turn or two back
    when the immediately-prior turn happened to be pure conversation
    (no lookup, no groundings). Falls back to ``[]`` when nothing in
    the window has cited anything — the new turn's accumulator starts
    empty and the model has to look up to ground its answer.
    """
    if not history:
        return []
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        if entry.get("role") != "assistant":
            continue
        refs = entry.get("cited_refs")
        if not isinstance(refs, list) or not refs:
            continue
        out: list[tuple[str, str]] = []
        for r in refs:
            if not isinstance(r, dict):
                continue
            kind = str(r.get("kind") or "").strip()
            rid = str(r.get("record_id") or "").strip()
            if kind and rid:
                out.append((kind, rid))
            if len(out) >= CARRYOVER_CAP:
                break
        if out:
            return out
    return []


def seed_records_for(
    store_path,
    refs: list[tuple[str, str]],
) -> list[RetrievedRecord]:
    """Fetch the carryover records from the bound store and return them
    in the SAME ORDER as ``refs`` so the seeded accumulator's bracket
    positions ``[1..K]`` line up with the prior turn's cited indices
    after the UI's contiguous renumber.

    Best-effort: a missing store, an unresolvable ref, or a store-open
    failure degrades to fewer (or zero) seeded records — never raises.
    The follow-up turn just retrieves fresh; carryover is a courtesy
    so "tell me more about [3]" can resolve without a new search, not
    a correctness invariant.
    """
    if not refs or store_path is None:
        return []
    try:
        with open_store(store_path) as store:
            found = store.filter_select(
                limit=len(refs), neighbor_ids=set(refs),
            )
    except Exception:
        return []
    by_key = {(r.kind, r.record_id): r for r in found}
    out: list[RetrievedRecord] = []
    for key in refs:
        rec = by_key.get(key)
        if rec is not None:
            out.append(RetrievedRecord(
                record=rec, distance=0.0, rerank_score=None,
            ))
    return out


def neutralize_dead_brackets(
    answer: str,
    retrieved: list[RetrievedRecord],
) -> str:
    """Strip ``[N]`` citation tokens that don't resolve to an in-range
    retrieved record, returning the cleaned answer.

    The frontend renders a ``[N]`` token as a clickable resource only
    when N indexes a record in the turn's resources payload
    (``1..len(retrieved)``); an out-of-range N has no backing record so
    it renders as dead text beside an otherwise-grounded answer. That is
    the user-visible half of the failure where a weak chat model
    finalizes a lookup-fired turn citing an index it never grounded —
    the search ran, but the bracket points at nothing the user can open.

    Only the unresolvable tokens are removed; valid in-range tokens are
    preserved verbatim so they stay clickable. A single space orphaned
    in front of the removed token, the doubled space it leaves between
    words, and a space stranded before sentence punctuation are tidied
    so the prose reads cleanly. Returns ``answer`` unchanged (same
    object) when every bracket resolves, so callers can cheaply detect
    "nothing to fix" by identity.
    """
    if not answer:
        return answer
    n_records = len(retrieved)

    def _keep_or_drop(m: re.Match) -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= n_records else ""

    cleaned = _CITATION_RE.sub(_keep_or_drop, answer)
    if cleaned == answer:
        return answer
    # Tidy the artifacts a removed token leaves behind: a space stranded
    # before sentence punctuation, then any doubled whitespace.
    cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


__all__ = [
    "CARRYOVER_CAP",
    "DEFAULT_CHATBOT_MODEL",
    "DEFAULT_CHATBOT_REASONING",
    "MAX_HISTORY_TURNS",
    "RESOURCE_PREVIEW_CHARS",
    "build_chat_prompt",
    "build_grounded_prompt",
    "build_resources",
    "carryover_refs",
    "cited_refs",
    "neutralize_dead_brackets",
    "resolve_chatbot_from_config",
    "resources_for_emit",
    "seed_records_for",
]
