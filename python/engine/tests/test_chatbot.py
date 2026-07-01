"""
Unit tests for the conversational chatbot prompt + signal helpers.

Covers the config resolver round-trip, the lookup-signal parser, the
two prompt builders (conversational + grounded), the conversation-window
bound, and the cited-references / refusal-coherence guarantee. The
sidecar loop (process spawn + JSON-line events) is exercised via the
Rust unit test on the Tauri command — covering both ends of the same
boundary in Python here would just duplicate event-schema constants.
"""
from __future__ import annotations

import pytest

from engine.chatbot import (
    DEFAULT_CHATBOT_MODEL,
    DEFAULT_CHATBOT_REASONING,
    MAX_HISTORY_TURNS,
    RESOURCE_PREVIEW_CHARS,
    build_chat_prompt,
    build_grounded_decision_prompt,
    build_grounded_prompt,
    build_resources,
    cited_refs,
    neutralize_dead_brackets,
    resolve_chatbot_from_config,
    resources_for_emit,
)
from engine.rag_vector_store import StoredRecord
from engine.retrieval import RetrievedRecord

from kernel.abstractions import InferenceProvider, LlmResponse


def _rec(record_id: str, kind: str = "chunk", text: str = "body") -> StoredRecord:
    return StoredRecord(kind=kind, record_id=record_id, text=text)


def _hit(record_id: str, text: str = "body", kind: str = "chunk") -> RetrievedRecord:
    return RetrievedRecord(
        record=_rec(record_id, kind=kind, text=text),
        distance=0.5,
        rerank_score=8.0,
    )


# ── resolve_chatbot_from_config ──────────────────────────────────────────


def test_resolve_chatbot_from_config_defaults_when_absent():
    """No ``chatbot`` field → ship defaults (glm-5-2 reasoning-OFF)."""
    out = resolve_chatbot_from_config({})
    assert out == {
        "model": DEFAULT_CHATBOT_MODEL,
        "reasoning": DEFAULT_CHATBOT_REASONING,
    }


def test_resolve_chatbot_from_config_valid_pass_through():
    """Valid ``chatbot`` field round-trips both fields."""
    out = resolve_chatbot_from_config(
        {"chatbot": {"model": "glm-5-2", "reasoning": True}},
    )
    assert out == {"model": "glm-5-2", "reasoning": True}


def test_resolve_chatbot_from_config_empty_model_falls_back():
    """Empty / whitespace-only model string falls back to defaults so
    a malformed save can't break the chatbot surface."""
    out = resolve_chatbot_from_config({"chatbot": {"model": "   ", "reasoning": True}})
    assert out["model"] == DEFAULT_CHATBOT_MODEL
    assert out["reasoning"] is DEFAULT_CHATBOT_REASONING


def test_resolve_chatbot_from_config_non_dict_tolerated():
    """A non-dict ``chatbot`` value (someone's hand edit) doesn't crash."""
    out = resolve_chatbot_from_config({"chatbot": "kimi-k2-6"})
    assert out == {
        "model": DEFAULT_CHATBOT_MODEL,
        "reasoning": DEFAULT_CHATBOT_REASONING,
    }


def test_resolve_chatbot_from_config_reasoning_defaults_to_ship_default():
    """A present ``chatbot`` dict that omits ``reasoning`` falls back to
    the ship-default (currently OFF), not a hardcoded False — the
    ship-default is uniform across the absent-field and
    partial-field paths, so there's no split default."""
    out = resolve_chatbot_from_config({"chatbot": {"model": "kimi-k2-6"}})
    assert out["model"] == "kimi-k2-6"
    assert out["reasoning"] is DEFAULT_CHATBOT_REASONING


# Tool-call parsing + validation now live in ``chatbot_tools`` (the
# JSON tool-call surface that replaced the ``LOOKUP:`` line); their unit
# tests are in test_chatbot_tools.py.


# ── build_chat_prompt ────────────────────────────────────────────────


def test_build_chat_prompt_role_shape():
    """System persona first, the new user message last, no context."""
    out = build_chat_prompt("hello")
    assert out[0]["role"] == "system"
    assert out[-1] == {"role": "user", "content": "hello"}
    # No numbered CONTEXT block on this turn — the decision-turn user
    # message is unadorned. (The persona itself may *describe* what a
    # CONTEXT block is when explaining citation rules; what matters is
    # that no such block is interpolated into the actual turn.)
    assert all("CONTEXT (numbered):" not in m["content"] for m in out)


def test_build_chat_prompt_persona_describes_conversation_and_lookup():
    """Persona must frame conversation as first-class AND carry the
    tool-call signal (the search tool's JSON shape) — that's the whole
    behavioral envelope."""
    sys_text = build_chat_prompt("hi")[0]["content"]
    assert '"tool": "search"' in sys_text
    assert "conversation" in sys_text.lower()


def test_persona_lookup_asks_for_self_contained_nl_question():
    """The lookup instruction must steer a clear, standalone
    natural-language question (referents resolved from the
    conversation) — keyword-pile queries are poor for dense KNN."""
    sys_text = build_chat_prompt("hi")[0]["content"].lower()
    assert "natural-language question" in sys_text
    assert "stand alone" in sys_text or "self-contained" in sys_text
    assert "resolve" in sys_text


def test_persona_biases_to_lookup_when_in_doubt():
    """#573 pillar 1+2: the lookup is cheap/routine and the in-doubt
    choice is to look — not refuse, hedge, or answer from assumption —
    while the pure-conversation carve-out survives so chitchat does
    not trigger spurious lookups."""
    s = build_chat_prompt("hi")[0]["content"].lower()
    assert "cheap and routine" in s and "not a last resort" in s
    assert "when in doubt, look" in s
    assert "never refuse or hedge" in s
    assert "from conversation memory or assumption" in s
    # carve-out preserved (narrowed to talk that needs nothing from vault)
    assert "skip the lookup only for talk that needs nothing" in s


def test_persona_self_referential_always_needs_fresh_lookup():
    """#575 iteration (orchestrator's complete rule): ANY question
    whose answer comes from the vault — the user themselves, any
    person/relationship/event/thing in their data, or anyone/anything
    not already known well — always needs a FRESH lookup, including
    brief follow-ups, elliptical confirmations, why/who-could-be, and
    speculative/hypothetical phrasing. The carve-out is only for talk
    that needs nothing from the vault."""
    s = build_chat_prompt("hi")[0]["content"].lower()
    assert (
        "any question whose answer would come from the user's vault" in s
    )
    assert (
        "about the user themselves (their life, situation, "
        "relationships, or the why, what, or how about them)" in s
    )
    # broadened beyond the user: entities in the data + unknown things
    assert (
        "about any person, relationship, event, or thing in their "
        "data, or about anyone or anything you do not already know "
        "well" in s
    )
    assert (
        "including brief follow-ups, elliptical confirmations, why "
        "or who-could-be questions, and speculative or hypothetical "
        "phrasing" in s
    )
    # carve-out: only talk that needs nothing from the vault
    assert (
        "skip the lookup only for talk that needs nothing from their "
        "vault — general knowledge, brainstorming or reasoning "
        "unrelated to them, or plain chit-chat" in s
    )
    # the old "not sure ... from this conversation alone" escape is gone
    assert "you are not sure of it from this conversation alone" not in s


def test_persona_kills_the_already_know_escape():
    """#575 iteration (orchestrator's crux): the residual on the
    why/follow-up turns was the model rationalizing it "already knows"
    from prior turns. The persona must explicitly remove that escape —
    never answer about the user from conversation memory/assumption,
    never reuse an earlier turn's records, for anything about them you
    look."""
    s = build_chat_prompt("hi")[0]["content"].lower()
    assert (
        "you do not already know the user from earlier in this "
        "conversation" in s
    )
    # escape-kill components (post-consolidation wording — same
    # behavior: no answering from memory/assumption, no reuse of an
    # earlier message's records).
    assert "never answer from conversation memory or assumption" in s
    assert (
        "never reuse an earlier message's records or references" in s
    )
    assert "for anything about them or their data you look" in s


def test_persona_converts_permission_ask_into_lookup():
    """#575 iteration (director's exact rule): the permission-ask
    anti-pattern ("should I check?" / "do you want me to look that
    up?") must be converted directly into the lookup directive — never
    ask permission."""
    s = build_chat_prompt("hi")[0]["content"].lower()
    assert "never ask permission" in s
    assert (
        "if you are about to ask whether to check or say you could "
        "look it up, call the tool instead" in s
    )


def test_persona_forbids_false_data_confidence():
    """#573 pillar 3: never claim/recall/cite the user's data unless
    it came from a visible lookup; presence is decided from the
    results, not guessed beforehand."""
    s = build_chat_prompt("hi")[0]["content"].lower()
    assert "never claim, recall, or cite anything from their data" in s
    assert "unless it came from a lookup's results in front of you" in s
    assert "decided from those results, not guessed beforehand" in s


def test_persona_warns_corpus_changes_between_messages():
    """#573 scope addition: the vault can change between messages, so
    the chatbot must re-query per turn and never reuse an earlier
    message's records/references."""
    s = build_chat_prompt("hi")[0]["content"].lower()
    assert "can change between messages" in s
    assert "fresh lookup" in s
    assert "never reuse an earlier message's records or references" in s


def test_grounded_rules_supersede_earlier_message_records():
    """#573 scope addition at the grounded turn: results are for this
    lookup only and supersede stale retrieval — cite only the entries
    numbered for this turn, never an earlier message's records. The
    grounding rules now live in the grounded *system* prompt (#599)."""
    g = build_grounded_prompt("q", [_hit("a", "body")])[0]["content"].lower()
    assert "for this lookup only" in g
    assert "never records from an earlier message" in g


def test_persona_stays_general_no_overfit_no_force():
    """Locks the director constraint: pure prompt/persona, no
    ``_force``-style mechanism, no corpus or personal names baked into
    the prompt — generic English only."""
    grounded = build_grounded_prompt("q", [_hit("a", "b")])
    blob = (
        build_chat_prompt("hi")[0]["content"]
        + grounded[0]["content"]
        + grounded[-1]["content"]
    ).lower()
    assert "_force" not in blob
    # no eval-corpus or personal-name leakage into the shipped prompt
    for token in (
        "pepys", "barbellion", "whatsapp", "whatsapp-family.txt",
        "personal-os", "personal_os",
    ):
        assert token not in blob, f"overfit token leaked into prompt: {token}"


# ── reasoning persona (the five behavioral directives) ───────────────


def _all_persona_systems() -> list[str]:
    """System prompt text of all three personas the reasoning block must
    ride: the decision turn, the forced grounded-final turn, and the
    mid-loop grounded-decision turn. The register must hold across all
    three so it's sticky whether the model is deciding, walking, or
    giving its forced final answer."""
    return [
        build_chat_prompt("hi")[0]["content"],
        build_grounded_prompt("q", [_hit("a", "b")])[0]["content"],
        build_grounded_decision_prompt(
            "q", [_hit("a", "b")], lookups_remaining=1
        )[0]["content"],
    ]


def test_reasoning_persona_shared_across_all_personas():
    """All five directives + the honorable mention land in every persona,
    not just the decision turn — the register can't drop once retrieval
    starts (#888)."""
    for sys_text in _all_persona_systems():
        s = sys_text.lower()
        # 1. first-principles over the conventional/consensus default
        assert "first principles" in s
        assert "conventional" in s or "received wisdom" in s
        # 2. answer at the level asked; no unprompted moral framing
        assert "level the question was asked" in s
        # 3. data not crisis; no "are you okay?" triage
        assert "not as evidence the user is in crisis" in s
        assert "are you okay" in s
        # 4. motives as hypotheses with explicit uncertainty
        assert "hypothesis" in s and "uncertainty" in s
        # 5. explicit precedence drop order
        assert "epistemic accuracy first" in s
        for rung in ("long-term", "structural clarity", "brevity", "tone"):
            assert rung in s, f"precedence rung missing: {rung}"
        # honorable mention: empty retrieval degrades to helpful, not withholding
        assert "without gatekeeping" in s


def test_reasoning_persona_is_register_shift_not_jailbreak():
    """Directive 1 is candor within safety policy, not a safety override —
    the prompt must say so in plain terms (anti-misread guard)."""
    s = build_chat_prompt("hi")[0]["content"].lower()
    assert "safety policy" in s
    assert "register" in s


def test_reasoning_persona_precedence_orders_accuracy_over_tone():
    """The drop order must place epistemic accuracy ahead of tone, not the
    reverse — codifies substance-before-compliments."""
    s = build_chat_prompt("hi")[0]["content"].lower()
    assert s.index("epistemic accuracy") < s.index("then tone")


def test_reasoning_persona_stays_general_no_overfit():
    """Behavioral directives only — no eval-corpus or personal-name tokens
    baked into the new block (mirrors the director constraint)."""
    blob = "".join(_all_persona_systems()).lower()
    for token in ("pepys", "barbellion", "whatsapp", "personal-os", "personal_os"):
        assert token not in blob, f"overfit token leaked into prompt: {token}"


def test_build_chat_prompt_interleaves_history():
    """Prior turns feed the prompt in order so follow-ups resolve."""
    history = [
        {"role": "user", "content": "first q"},
        {"role": "assistant", "content": "first a"},
    ]
    out = build_chat_prompt("second q", history)
    assert [m["role"] for m in out] == [
        "system", "user", "assistant", "user",
    ]
    assert out[1]["content"] == "first q"
    assert out[-1]["content"] == "second q"


def test_build_chat_prompt_history_window_bounded():
    """Only the trailing MAX_HISTORY_TURNS exchanges feed the prompt —
    older turns are dropped so the prefix stays bounded."""
    history = []
    for i in range(MAX_HISTORY_TURNS + 5):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
    out = build_chat_prompt("now", history)
    # system + 2*MAX_HISTORY_TURNS history msgs + the new user message.
    assert len(out) == 1 + 2 * MAX_HISTORY_TURNS + 1
    # The oldest surviving turn is exactly MAX_HISTORY_TURNS back.
    assert out[1]["content"] == f"q{5}"


def test_build_chat_prompt_skips_malformed_history():
    """A half-streamed / malformed history entry is skipped, not
    allowed to poison the prompt."""
    history = [
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": ""},   # empty → skipped
        "not a dict",                           # junk → skipped
        {"role": "system", "content": "nope"},  # wrong role → skipped
    ]
    out = build_chat_prompt("q", history)
    assert [m["role"] for m in out] == ["system", "user", "user"]


def test_history_strips_citation_brackets_from_assistant():
    """Acceptance #1: ``_history_messages`` strips ``[N]`` citation
    brackets from prior assistant content before the LLM sees it.

    The brackets index a PRIOR turn's CONTEXT block that doesn't ride
    along with the history string; leaving them in lets the model
    re-cite ``[3]`` from memory in fresh prose where the bracket
    resolves to nothing, OR echo stale indices back into
    ``has_neighbor`` as cross-turn ID fabrication. The UI's resources
    panel renders from the per-turn ``cited_refs`` payload, not the
    history string, so display is untouched."""
    history = [
        {"role": "user", "content": "what's in the data?"},
        {
            "role": "assistant",
            "content": "The action [3] on the 48-hour pause [7] is "
                       "rooted in pattern [11].",
        },
    ]
    out = build_chat_prompt("follow-up", history)
    assistant_msg = next(m for m in out if m["role"] == "assistant")
    assert "[3]" not in assistant_msg["content"]
    assert "[7]" not in assistant_msg["content"]
    assert "[11]" not in assistant_msg["content"]
    # The prose around the brackets survives so cross-turn semantics
    # still grounds out (model can still discuss the prior topic).
    assert "action" in assistant_msg["content"]
    assert "pause" in assistant_msg["content"]


def test_history_strip_leaves_user_brackets_alone():
    """The strip applies only to assistant content — a user who types
    ``[3]`` (the carryover-resolution path: ``tell me about [3]``)
    must see the bracket reach the LLM verbatim so the carryover
    seed's brackets resolve in this turn's accumulator."""
    history = [
        {"role": "user", "content": "tell me about [3]"},
        {"role": "assistant", "content": "Let me look that up."},
    ]
    out = build_chat_prompt("and [5]?", history)
    user_msgs = [m for m in out if m["role"] == "user"]
    assert "[3]" in user_msgs[0]["content"]
    # The new user message also rides through unmodified.
    assert "[5]" in user_msgs[-1]["content"]


# ── build_grounded_prompt ────────────────────────────────────────────


def test_build_grounded_prompt_numbers_context_and_keeps_history():
    """Grounded turn carries numbered context, the question, the
    grounding instruction, and the conversation so far."""
    history = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "reply"},
    ]
    hits = [_hit("a", "first body"), _hit("b", "second body")]
    out = build_grounded_prompt("what did I note?", hits, history)
    assert [m["role"] for m in out] == [
        "system", "user", "assistant", "user",
    ]
    grounded = out[-1]["content"]
    assert "what did I note?" in grounded
    # CONTEXT entries are labeled with integer brackets [1], [2], ...
    assert "[1]" in grounded and "[2]" in grounded
    assert "first body" in grounded and "second body" in grounded
    # The bracketed-citation instruction now lives in the grounded
    # system prompt (#599), not the user message.
    assert "bracket" in out[0]["content"]


# The literal that used to be interpolated into the grounded suffix as
# the example refusal wording. Handing the model a fixed string to copy
# is what let a byte-identical refusal stack in converse history into an
# unrecoverable attractor; the contract is that this exact string is
# never manufactured by the prompt builders again.
_DEAD_REFUSAL_LITERAL = "I don't have anything in your data about that."


def test_build_grounded_prompt_empty_retrieval_says_so():
    """A lookup that found nothing produces a context block that says
    so and instructs a plain, conversational not-found in the model's
    own words — no fixed phrase to copy."""
    out = build_grounded_prompt("q", [])
    assert "no matching records" in out[-1]["content"].lower()
    assert _DEAD_REFUSAL_LITERAL not in out[-1]["content"]
    # not-found phrasing instruction is in the grounded system prompt.
    g = out[0]["content"].lower()
    assert "your own words" in g and "plainly" in g


def test_grounded_prompt_never_manufactures_the_canned_refusal():
    """Root-cause contract for #512: the prompt builder must never
    interpolate the fixed refusal literal — empty OR populated
    retrieval. With no manufactured byte-identical string, no
    self-poisoning attractor can form in converse history."""
    empty = build_grounded_prompt("q", [])
    assert all(_DEAD_REFUSAL_LITERAL not in m["content"] for m in empty)
    hits = [_hit("a", "first body"), _hit("b", "second body")]
    populated = build_grounded_prompt("who am i", hits, None)
    assert all(_DEAD_REFUSAL_LITERAL not in m["content"] for m in populated)


def test_grounded_prompt_recalibrates_identity_cover_judgment():
    """Bug A at the prompt level: an identity/personal question over
    retrieved context must be steered to answer from the relevant
    entries, not deflect."""
    g = build_grounded_prompt("who am i", [_hit("a", "body")])[0]["content"].lower()
    assert "who they are" in g
    assert "do not deflect" in g
    assert "offer what you can do" in g


def test_build_grounded_prompt_renders_full_body_verbatim():
    """Bodies render verbatim into CONTEXT — per-record truncation
    would lose the substantive content a multi-rung walk depends on.
    The per-turn pool is bounded by ``ACCUMULATOR_CAP`` in the loop
    and by ``MAX_LOOKUPS`` × ``MAX_COUNT`` × ``MAX_HOPS`` at the
    protocol level instead."""
    huge = "x" * 5000
    grounded = build_grounded_prompt("q", [_hit("a", huge)])[-1]["content"]
    assert huge in grounded


def test_build_grounded_prompt_renders_integer_bracket_lines():
    """Each CONTEXT entry is ``[N] text`` — no parenthetical
    ``(kind/record_id)`` is rendered to the model. The loop's
    ``_resolve_bracket_anchors`` translates ``has_neighbor: ["[1]"]``
    back to canonical kind/record_id before dispatch."""
    hits = [_hit("fact-123", "body", kind="pattern")]
    grounded = build_grounded_prompt("q", hits)[-1]["content"]
    # Integer bracket is rendered (single entry = [1]).
    assert "[1]" in grounded
    # No parenthetical canonical id leaks into the user-facing CONTEXT.
    assert "(pattern/fact-123)" not in grounded
    assert "(pattern" not in grounded


def test_grounded_context_prefers_display_text_over_enriched_text():
    """The grounded CONTEXT block surfaces the bare ``display_text``
    body when present, so the model never sees the graph-enriched
    prefix (Type:, name-lists referencing other records, kind brackets)
    that would otherwise leak a canonical-id-shape citation surface."""
    enriched = (
        "Type: fact\n"
        "Patterns mentioning this fact: Sleep regularity\n\n"
        "core fact body"
    )
    display = "core fact body"
    rec = StoredRecord(
        kind="fact", record_id="health:0",
        text=enriched, display_text=display,
    )
    hit = RetrievedRecord(record=rec, distance=0.5, rerank_score=8.0)
    grounded = build_grounded_prompt("q", [hit])[-1]["content"]

    assert "core fact body" in grounded
    assert "Type: fact" not in grounded
    assert "Patterns mentioning this fact" not in grounded


def test_grounded_context_falls_back_to_text_when_display_absent():
    """Legacy records minted before the embed/display split carry an
    empty ``display_text``; the CONTEXT builder falls back to the
    embedded ``text`` so a vault that hasn't re-extracted yet still
    renders something coherent."""
    rec = StoredRecord(
        kind="fact", record_id="health:0",
        text="legacy embedded body", display_text="",
    )
    hit = RetrievedRecord(record=rec, distance=0.5, rerank_score=8.0)
    grounded = build_grounded_prompt("q", [hit])[-1]["content"]

    assert "legacy embedded body" in grounded


def test_grounded_chunk_renders_integer_bracket_without_parenthetical():
    """Same as above for chunk records: the chunk's ``Type: raw input``
    survives in the body (it's baked into the embed text), but the
    ``(raw input/c1)`` parenthetical that used to flag the kind to
    the model is gone."""
    grounded = build_grounded_prompt(
        "q", [_hit("c1", "body", kind="chunk")]
    )[-1]["content"]
    assert "[1]" in grounded
    assert "(raw input/c1)" not in grounded
    assert "(chunk/c1)" not in grounded


def test_grounded_rules_instruct_supporting_only_capped_citations():
    """The grounded instruction must steer support-only selection and
    a hard ≤5 cite-count cap (model-level, no post-hoc drop) — the
    #514 clause, composed alongside #521's de-literalised not-found /
    Bug-A clause in the same rules (now the grounded system prompt,
    #599)."""
    grounded = build_grounded_prompt("q", [_hit("a")])[0]["content"].lower()
    assert "loosely related" in grounded
    assert "hard limit: cite at most 5" in grounded
    assert "most directly support" in grounded
    # #521's clause must still be present (composition, not replacement).
    assert "do not deflect" in grounded


def test_grounded_system_prompt_carries_no_lookup_protocol():
    """#599 symptom A, structural guarantee: the grounded turn's system
    prompt must not carry the tool-call protocol — no tool-call JSON
    shape and none of the protocol's defining phrases — so the model
    cannot be authoritatively told to call a tool again on a turn whose
    whole job is to answer from the supplied context. The identity
    framing is still shared."""
    out = build_grounded_prompt("q", [_hit("a", "body")])
    sys_text = out[0]["content"]
    assert '"tool":' not in sys_text
    low = sys_text.lower()
    for phrase in (
        "make your entire reply",
        "cheap and routine",
        "when in doubt, look",
        "always needs a fresh lookup",
        "never ask permission",
    ):
        assert phrase not in low, f"tool-protocol phrase leaked: {phrase!r}"
    # identity is shared across both turns
    assert "basevault chatbot" in low
    assert "conversation" in low
    # and no message in the grounded prompt re-emits a tool call
    assert all('"tool":' not in m["content"] for m in out)


def test_decision_persona_forbids_brackets_on_voluntary_prose_finalize():
    """#809: the decision persona is the one persona that can run with
    no this-turn CONTEXT block (the grounded personas always have one).
    If the model voluntary-prose-finalizes here, any `[N]` it emits
    points at nothing the user can open and reads as a citation that
    isn't one — egas turn 34 showed the model pattern-matching on `[N]`
    markers from its own prior prose and re-asserting facts under
    brackets that had no grounding this turn. The persona must
    explicitly forbid `[N]` markers on the no-lookup finalize path and
    steer to conversational narration of prior findings instead."""
    s = build_chat_prompt("hi")[0]["content"].lower()
    # The rule is scoped to the no-lookup prose-finalize path.
    assert "finalize this turn in prose without a fresh lookup" in s
    # No bracketed refs at all on that path.
    assert "no bracketed [n] references" in s
    # Why: bracket refs only resolve against THIS turn's numbered CONTEXT.
    assert "this turn carries its own numbered context block" in s
    # The specific anti-pattern from egas turn 34: re-asserting [N] from
    # the model's own prior prose (records didn't ride along).
    assert "earlier turns' bracketed references" in s
    assert "only the prose answers from those turns rode along" in s
    # Steer to conversational narration instead.
    assert "narrate any earlier finding you draw on conversationally" in s


def test_decision_persona_hard_requires_fresh_lookup_no_reuse():
    """#599 symptom B: the decision persona must make the lookup a hard
    requirement (not soft prose) for any question needing vault data not
    already in this turn's fresh results — covering easy-looking
    conversational follow-ups and forbidding manufactured bracketed
    references / specific facts that did not come from this turn's
    lookup. General framing is strengthened, never deleted."""
    s = build_chat_prompt("hi")[0]["content"].lower()
    assert "hard requirement, not a preference" in s
    assert "you must make this turn's entire reply the tool call" in s
    assert "looks answerable from the chat history still needs" in s
    assert (
        "never produce a bracketed reference or a specific recorded "
        "fact, date, or detail that did not come from a lookup "
        "performed for this turn" in s
    )
    # the pre-existing general no-memory/no-reuse framing is preserved
    assert "never answer from conversation memory or assumption" in s
    assert "never reuse an earlier message's records or references" in s


# ── cited_refs (trust gate + refusal coherence) ──────────────────────


def test_cited_refs_returns_only_cited_in_order():
    """Only the [N] tokens actually present in the answer surface, in
    ascending order — an uncited retrieved record does not."""
    hits = [_hit("a"), _hit("b"), _hit("c")]
    refs = cited_refs("Per your notes [3] and also [1].", hits)
    assert [r["index"] for r in refs] == [1, 3]
    assert refs[0]["record_id"] == "a"
    assert refs[1]["record_id"] == "c"


def test_cited_refs_drops_out_of_range_tokens():
    """A [N] beyond the retrieved count is dropped — the model can't
    surface a reference to a record it was never given."""
    hits = [_hit("a"), _hit("b")]
    refs = cited_refs("Claim [1] and bogus [9].", hits)
    assert [r["index"] for r in refs] == [1]


def test_cited_refs_refusal_answer_yields_none():
    """The refusal-coherence guarantee: an answer that cites nothing —
    including the canonical not-in-your-data reply — produces zero
    references, so references never render next to a refusal."""
    hits = [_hit("a"), _hit("b")]
    assert cited_refs("I don't see that in your notes.", hits) == []
    assert cited_refs("Let's just talk it through, no data needed.", hits) == []


def test_cited_refs_no_retrieval_yields_none():
    """No retrieved set (pure conversation turn) → no references even
    if the text happens to contain bracket-digits."""
    assert cited_refs("step [1] of my plan", []) == []


# ── neutralize_dead_brackets (#891 symptom B: inert prose citations) ──


def test_neutralize_drops_out_of_range_keeps_valid():
    """An out-of-range ``[N]`` (no backing record) is stripped; an
    in-range one is preserved so it stays clickable. Spacing around the
    removed token is tidied so the prose reads cleanly."""
    hits = [_hit("a"), _hit("b")]
    out = neutralize_dead_brackets(
        "She is a consultant [9] per your notes [1].", hits,
    )
    assert out == "She is a consultant per your notes [1]."


def test_neutralize_strips_space_before_punctuation():
    """A token removed from in front of sentence punctuation doesn't
    leave a stranded space."""
    hits = [_hit("a")]
    assert neutralize_dead_brackets("A claim [7].", hits) == "A claim."


def test_neutralize_all_dead_when_no_records():
    """Lookup fired but matched nothing citable: every bracket is dead,
    so all are stripped — the answer keeps its prose, sheds the inert
    markers."""
    assert neutralize_dead_brackets("Per [1] and [2] you noted X.", []) == \
        "Per and you noted X."


def test_neutralize_is_identity_when_all_resolve():
    """Nothing to fix → the same object back (cheap no-op detection)."""
    hits = [_hit("a"), _hit("b")]
    answer = "Grounded on [1] and [2]."
    assert neutralize_dead_brackets(answer, hits) is answer


def test_neutralize_ignores_programming_brackets():
    """The citation regex's negative lookbehind keeps ``arr[9]`` out of
    the citation pool, so code-shaped tokens are never stripped."""
    hits = [_hit("a")]
    answer = "Use arr[9] in the loop, see [1]."
    assert neutralize_dead_brackets(answer, hits) == answer


# ── build_resources (grounded-source block + refusal coherence) ──────


def test_build_resources_is_the_cited_subset_in_order():
    """The block is the chunks the answer actually grounded on, in
    ascending order — not the full retrieved set. An uncited retrieved
    record does not appear."""
    hits = [_hit("a", "alpha"), _hit("b", "beta"), _hit("c", "gamma")]
    res = build_resources("grounded on [3] and [1].", hits)
    assert [r["index"] for r in res] == [1, 3]
    assert [r["record_id"] for r in res] == ["a", "c"]
    assert res[0]["preview"] == "alpha"
    assert all("cited" not in r for r in res)


def test_build_resources_truncates_preview():
    """Preview is bounded — a verify-the-source affordance, not a
    document viewer."""
    huge = "x" * 5000
    res = build_resources("[1]", [_hit("a", huge)])
    assert len(res[0]["preview"]) <= RESOURCE_PREVIEW_CHARS
    assert res[0]["preview"].endswith("…")


def test_build_resources_empty_on_refusal_even_when_tool_returned_rows():
    """The refusal-coherence guarantee: vector search returned rows but
    the answer grounded on none of them (a "not in your data" reply) →
    empty list. The sidecar emits this as the explicit "no matching
    resources" state — never a list of irrelevant chunks beside a
    refusal."""
    hits = [_hit("a"), _hit("b"), _hit("c")]
    assert build_resources("I don't see that in your notes.", hits) == []
    assert build_resources("Let's just talk it through.", hits) == []


def test_build_resources_empty_when_tool_matched_nothing():
    """No retrieved set at all → empty list (also the explicit
    no-matching state)."""
    assert build_resources("anything [1]", []) == []


def _chunk_hit(record_id: str, text: str, chunk_len: int | None) -> RetrievedRecord:
    extra = {"chunk_len": chunk_len} if chunk_len is not None else {}
    return RetrievedRecord(
        record=StoredRecord(
            kind="chunk", record_id=record_id, text=text, extra=extra,
        ),
        distance=0.5,
        rerank_score=8.0,
    )


def test_build_resources_emits_chunk_len_when_persisted():
    """A chunk record carrying the embeddings-persisted raw length
    surfaces it on the resource so the UI can highlight the whole
    chunk span instead of a paragraph approximation."""
    res = build_resources("[1]", [_chunk_hit("f1@10", "body", 1234)])
    assert res[0]["chunk_len"] == 1234


def test_build_resources_omits_chunk_len_for_old_embeddings():
    """Records from embeddings predating the persisted length carry no
    `chunk_len`; the resource omits the key and the UI falls back to
    the paragraph approximation. Non-positive / non-int values are
    treated as absent."""
    assert "chunk_len" not in build_resources(
        "[1]", [_chunk_hit("f1@10", "body", None)],
    )[0]
    assert "chunk_len" not in build_resources(
        "[1]", [_chunk_hit("f1@10", "body", 0)],
    )[0]


def test_build_resources_chunk_len_only_on_chunk_kind():
    """A non-chunk record never gets a chunk_len even if some upstream
    stuffed one into its extra — the key is chunk-citation-specific."""
    hit = RetrievedRecord(
        record=StoredRecord(
            kind="fact", record_id="health:0", text="x",
            extra={"chunk_len": 99},
        ),
        distance=0.5,
        rerank_score=8.0,
    )
    assert "chunk_len" not in build_resources("[1]", [hit])[0]


# ── resources_for_emit: the shared UI-emit gate (sidecar + chat eval) ───


def test_resources_for_emit_carryover_no_lookup_renders_panel():
    """#887 regression: a follow-up turn that finalizes from the carryover
    seed cites a bracket but fires NO fresh lookup. The cited subset is
    resolved FIRST, independent of ``lookup_fired``, so those brackets
    still render a real resources panel. The historical ``lookup_fired``
    gate dropped them — inert ``[N]`` + empty panel."""
    hits = [_hit("a", "alpha"), _hit("b", "beta")]
    out = resources_for_emit(
        "More on [1].", hits, refused=False, lookup_fired=False)
    assert out is not None
    assert [r["record_id"] for r in out] == ["a"]


def test_resources_for_emit_refused_is_none():
    """The #780 refused path renders no panel even if records are present
    (the tool didn't run; ``[]`` would imply "we searched, matched
    nothing")."""
    assert resources_for_emit(
        "[1]", [_hit("a")], refused=True, lookup_fired=False) is None


def test_resources_for_emit_empty_retrieved_no_lookup_is_none():
    """No records and no search (no seed, no fresh) → no panel."""
    assert resources_for_emit(
        "anything [1]", [], refused=False, lookup_fired=False) is None


def test_resources_for_emit_searched_cites_nothing_is_empty_not_none():
    """A search FIRED but the answer grounds on none of the pool → ``[]``,
    NOT ``None``. The frontend renders ``[]`` as the "No matching
    resources in your data." empty-state and ``None`` as no block at all;
    coercing ``[]``→``None`` would silently drop that empty-state (the
    regression caught on #908)."""
    assert resources_for_emit(
        "Let's just talk it through.", [_hit("a"), _hit("b")],
        refused=False, lookup_fired=True,
    ) == []


def test_resources_for_emit_no_lookup_seed_cites_nothing_is_none():
    """The seed-pollution case: a pure-conversation follow-up ("thanks")
    fires NO search but carries a carryover seed, so ``retrieved`` is
    non-empty. It must render NO block (``None``), NOT the "No matching
    resources in your data." empty-state — nothing was searched. Keying
    on records-present would wrongly emit ``[]`` here; the discriminator
    is ``lookup_fired``."""
    assert resources_for_emit(
        "Thanks, that's all.", [_hit("a"), _hit("b")],
        refused=False, lookup_fired=False,
    ) is None


# ── Interactive-stage budget wiring (regression for the KeyError on ──
#    `_ratio_for_stage(None)` the first end-to-end chatbot call hit) ──────


def test_chatbot_stage_registered_in_budget_maps():
    """`complete()`'s budget path calls `_ratio_for_stage(stage)`,
    which raises `KeyError` on an unregistered stage. The chatbot + rerank
    surfaces run `complete()` outside the runner's per-stage
    bracketing, so both names MUST be registered or every interactive
    query 500s. Pin both budget maps."""
    from engine.llm import _MAX_RATIO_BY_STAGE, _TYPICAL_RATIO_BY_STAGE
    for stage in ("chatbot", "rerank"):
        assert stage in _MAX_RATIO_BY_STAGE
        assert stage in _TYPICAL_RATIO_BY_STAGE
        assert _MAX_RATIO_BY_STAGE[stage] is None
        assert _TYPICAL_RATIO_BY_STAGE[stage] is None


def test_compute_budget_does_not_raise_for_interactive_stages():
    """End-to-end guard: the exact call `complete()` makes
    (`chunk_cap_for_stage(mode, stage)` → `compute_budget`) must
    produce a sane budget for the interactive stages, not raise."""
    from engine.llm import Mode, chunk_cap_for_stage
    for stage in ("chatbot", "rerank"):
        cap = chunk_cap_for_stage(Mode.TEE, stage)
        assert isinstance(cap, int)
        assert cap > 0


def test_stage_scope_sets_and_restores():
    """`stage_scope` must restore the prior `_current_stage` on exit so
    an interactive call can't leak its tag into a later pipeline stage
    (or a nested call)."""
    from engine import llm
    from engine.llm import stage_scope

    assert llm._current_stage is None
    with stage_scope("chatbot"):
        assert llm._current_stage == "chatbot"
        with stage_scope("rerank"):
            assert llm._current_stage == "rerank"
        assert llm._current_stage == "chatbot"
    assert llm._current_stage is None


def test_stage_scope_restores_on_exception():
    """The restore is in a finally — an exception inside the block must
    not leave `_current_stage` pinned to the interactive tag."""
    from engine import llm
    from engine.llm import stage_scope

    assert llm._current_stage is None
    with pytest.raises(RuntimeError):
        with stage_scope("chatbot"):
            raise RuntimeError("boom")
    assert llm._current_stage is None


# ── chatbot call telemetry (#456): begin/end into a dedicated chatbot log ────
#
# The sidecar process loop is exercised via the Rust unit test; here we
# pin the *telemetry contract* deterministically — each chatbot LLM call
# emits a begin AND an end event into the dedicated chatbot llm-calls.jsonl
# with a per-call discriminator — without a live attested provider.


def _fake_completion(content: str = "hi"):
    from engine.llm import CompletionResult

    return CompletionResult(
        content=content, call_id=None, cache_key=None, cached=False,
        finish_reason="stop", model="m", mode="tinfoil",
        prompt_tokens=3, completion_tokens=2, reasoning_tokens=0,
        reasoning_tokens_source=None, content_tokens=2, ttft_ms=1,
        ttfr_ms=1, last_token_ms=1, max_tokens_reserved=64,
    )


@pytest.fixture
def chatbot_telemetry_env(tmp_path, monkeypatch):
    """Isolated chatbot telemetry dir + clean llm stat state."""
    from engine import llm
    from engine import chatbot_sidecar

    # Chatbot telemetry writes to `_telemetry_dir()` — the active
    # conversation dir (`BASEVAULT_CHATBOT_CONVO_DIR`, #565/#568), NOT
    # `<logs_root>/chatbot/` as it did pre-#568. Point it at tmp so the
    # dedicated llm-calls.jsonl / llm-payloads.jsonl land where the
    # `_read_events`/`_read_payloads` helpers look (`tmp_path/chatbot/`).
    monkeypatch.setenv("BASEVAULT_LOGS_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "BASEVAULT_CHATBOT_CONVO_DIR", str(tmp_path / "chatbot"))
    llm.reset_stat_records()
    # NB: the legacy retrieval.complete rebind is gone — retrieval runs on the
    # kernel now, and the chat turn's instrumentation is injected by
    # run_chat_turn (ctx.tracked_complete), not a module rebind.
    try:
        yield tmp_path, llm, chatbot_sidecar
    finally:
        llm.set_calls_jsonl_path(None)
        llm.set_payloads_jsonl_path(None)
        llm.reset_stat_records()


def _read_jsonl(path):
    import json as _json

    assert path.is_file(), f"{path} was not created"
    return [
        _json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_events(tmp_path):
    return _read_jsonl(tmp_path / "chatbot" / "llm-calls.jsonl")


def _read_payloads(tmp_path):
    return _read_jsonl(tmp_path / "chatbot" / "llm-payloads.jsonl")


class _ScriptedSidecarProvider(InferenceProvider):
    """In-memory chat provider for the kernel chat path. Returns its
    scripted replies in sequence (one per ReAct hop) and streams each
    through the loop's stream handler so the ``_StreamGate`` suppression
    is exercised exactly as in production.

    This replaces the legacy ``chatbot_sidecar.complete`` stub: chat hops
    now run on the kernel (``ChatPhase`` → ``provider.run``), so the
    deterministic reply is injected at the provider seam instead. The
    ``seen_calls`` list is the per-hop record the old ``complete``-capture
    used to expose (one ``LlmCall`` per hop)."""

    def __init__(self, replies):
        self._replies = iter(replies)
        self.seen_calls: list = []

    def name(self):
        return "scripted-sidecar"

    def run(self, call, execution_env) -> LlmResponse:
        self.seen_calls.append(call)
        reply = next(self._replies)
        if call.stream_handler:
            call.stream_handler(reply)
        return LlmResponse(None, reply, None, 3, 2, 0, 1.0, 1.0)

    def inject_errors(self, phase, errors):
        pass


def _forever(reply: str):
    while True:
        yield reply


def _install_scripted_chat_provider(monkeypatch, replies, *, model="chat-model-x"):
    """Route the sidecar's per-turn kernel chat through a scripted provider.

    Overrides ``build_stage_env`` (the sidecar's per-turn env factory) so the
    real ``run_chat_turn`` + ``KernelTelemetryHook`` run unchanged
    but every hop's LLM call hits ``_ScriptedSidecarProvider`` instead of the
    live attested model. No disk cache is attached, so the provider is hit
    deterministically every hop. Returns the provider; its ``seen_calls`` is
    the per-hop record the legacy ``complete`` capture used to provide."""
    from engine.phases import model_specs as model_specs
    from engine.phases.model_specs import PipelineModelSpec
    from engine.phases.telemetry_hook import KernelTelemetryHook
    from kernel.execution_env import ExecutionEnv

    provider = _ScriptedSidecarProvider(replies)

    def _fake_build_stage_env(
            phase_name, mode, session_id=None, thinking=None,
            payload_sink=None, extra_hooks=None):
        spec = PipelineModelSpec(provider, model, 131_000, max_parallelism=4)
        env = ExecutionEnv()
        env.register_spec(phase_name, spec, spec, bool(thinking))
        env.register_llm_hook(KernelTelemetryHook(
            session_id=session_id, payload_sink=payload_sink, mode=mode))
        for hook in extra_hooks or []:
            env.register_llm_hook(hook)
        return env

    monkeypatch.setattr(model_specs, "build_stage_env", _fake_build_stage_env)
    return provider








# ── chatbot payload capture (#456 fold-in): full prompt+response per call ─








def test_persistent_loop_serves_multiple_turns_and_fences_by_turn_id(
        chatbot_telemetry_env, monkeypatch, capsys):
    """The persistent sidecar serves many newline-framed turns from one
    process (this is the P1 win — the warm attested client is reused),
    every event echoes its request's ``turn_id`` (the turn fence), and
    stdin EOF ends the loop with exit 0."""
    import io
    import json as _json

    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env

    monkeypatch.setattr(chatbot_sidecar, "_read_app_config", lambda: {})
    # Warmup hits the real attested route — neuter it in-test.
    monkeypatch.setattr(chatbot_sidecar, "_warm_client", lambda: None)
    provider = _install_scripted_chat_provider(monkeypatch, _forever("hi"))

    # Two framed requests then EOF — distinct turn_ids.
    stdin = io.StringIO(
        _json.dumps({"query": "first", "history": [], "turn_id": 11}) + "\n"
        + _json.dumps({"query": "second", "history": [], "turn_id": 12}) + "\n"
    )
    monkeypatch.setattr(chatbot_sidecar.sys, "stdin", stdin)

    rc = chatbot_sidecar.main()
    assert rc == 0  # clean shutdown on EOF

    events = [
        _json.loads(ln)
        for ln in capsys.readouterr().out.splitlines() if ln.strip()
    ]
    done = [e for e in events if e["event"] == "chatbot_done"]
    # One process served BOTH turns (two hop calls, two dones).
    assert len(provider.seen_calls) == 2
    assert [e["turn_id"] for e in done] == [11, 12]
    # Session-scoped events are emitted once at process start, before
    # any turn, so they are intentionally turn_id-less. Every
    # turn-scoped event carries its turn's fence — no unfenced bleed.
    session_scoped = {"chatbot_bound"}
    turn_events = [e for e in events if e["event"] not in session_scoped]
    assert all("turn_id" in e for e in turn_events)
    assert {e["turn_id"] for e in turn_events} == {11, 12}
    # The session-level chatbot_bound fired exactly once, at start.
    assert sum(1 for e in events if e["event"] == "chatbot_bound") == 1


def test_warm_client_constructs_the_tinfoil_provider(
        chatbot_telemetry_env, monkeypatch):
    """The warm path constructs the kernel TinfoilProvider off the user's
    first turn. Constructing it kicks off the client warm (the `TinfoilAI()`
    ctor crypto-verifies + TLS-pins) — the per-request attestation — so the
    first turn rides an already-attested client. No separate attest step."""
    from engine import chatbot_sidecar
    import kernel.tinfoil_provider as tp

    monkeypatch.setattr(
        chatbot_sidecar, "_read_app_config", lambda: {})
    calls = []
    monkeypatch.setattr(
        tp, "TinfoilProvider", lambda *a, **k: calls.append(True) or object())

    chatbot_sidecar._warm_client()

    # Provider constructed once.
    assert calls == [True]


def test_warm_client_is_non_fatal_when_construction_raises(
        chatbot_telemetry_env, monkeypatch):
    """A failed provider construction (offline / no key / transient
    verification blip) must NOT crash the persistent sidecar loop — it
    stays best-effort. The first real turn re-builds and surfaces the same
    error in-band via chatbot_error."""
    from engine import chatbot_sidecar
    import kernel.tinfoil_provider as tp

    monkeypatch.setattr(
        chatbot_sidecar, "_read_app_config", lambda: {})

    def _boom(*a, **k):
        raise RuntimeError("enclave verification failed")

    monkeypatch.setattr(tp, "TinfoilProvider", _boom)

    # Must return normally, swallowing the construction failure.
    chatbot_sidecar._warm_client()


# ── Sidecar mode-plumbing (slice B: LOCAL vs cloud turn dispatch) ────


def test_resolve_chatbot_mode_local_vs_cloud():
    from engine import chatbot_sidecar
    from engine.llm import Mode

    assert chatbot_sidecar._resolve_chatbot_mode({"mode": "local"}) == Mode.LOCAL
    assert chatbot_sidecar._resolve_chatbot_mode({"mode": "LOCAL"}) == Mode.LOCAL
    # Binary fork: everything non-local (and unset/garbage) → attested
    # cloud, never a silent downgrade to a maybe-uninstalled local model.
    for cfg in ({"mode": "tee"}, {"mode": "tee"}, {"mode": "tee"},
                {}, {"mode": None}, {"mode": "nonsense"}):
        assert chatbot_sidecar._resolve_chatbot_mode(cfg) == Mode.TEE


def test_chat_call_kwargs_cloud_keeps_kimi_and_force():
    from engine import chatbot_sidecar
    from engine.llm import Mode

    kw = chatbot_sidecar._chat_call_kwargs(Mode.TEE, {"model": "kimi-k2-6"})
    assert kw == {"model": "kimi-k2-6", "mode": Mode.TEE, "_force_model_id": True}


def test_chat_call_kwargs_local_picks_local_model_no_force(monkeypatch):
    from engine import chatbot_sidecar
    from engine.llm import Mode, Provider
    from types import SimpleNamespace

    monkeypatch.setattr(
        chatbot_sidecar, "get_mode_spec",
        lambda mode: SimpleNamespace(model_id="qwen3.5:9b",
                                     provider=Provider.OLLAMA))
    kw = chatbot_sidecar._chat_call_kwargs(Mode.LOCAL, {"model": "kimi-k2-6"})
    # LOCAL uses the local mode's model (NOT the cloud chatbot model) and
    # must NOT pass _force_model_id (would KeyError on the Ollama spec
    # since static _MODEL_SPECS doesn't carry local-backend entries).
    assert kw == {"model": "qwen3.5:9b", "mode": Mode.LOCAL}
    assert "_force_model_id" not in kw


def test_warm_client_local_skips_tinfoil_warms_local(
        chatbot_telemetry_env, monkeypatch):
    """LOCAL warm must NOT construct the Tinfoil provider (no attestation
    surface) — it warms the local model instead."""
    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env
    import kernel.tinfoil_provider as tp

    monkeypatch.setattr(
        chatbot_sidecar, "_read_app_config", lambda: {"mode": "local"})
    tinfoil_calls = []
    monkeypatch.setattr(
        tp, "TinfoilProvider",
        lambda *a, **k: tinfoil_calls.append(True) or object())
    local_calls = []
    monkeypatch.setattr(
        chatbot_sidecar, "_warm_local_model", lambda: local_calls.append(True))

    chatbot_sidecar._warm_client()

    assert tinfoil_calls == []      # attested provider never built in LOCAL
    assert local_calls == [True]    # local warm fired instead


def test_run_local_mode_builds_ctx_with_local_kwargs_and_mode(
        chatbot_telemetry_env, monkeypatch):
    """In LOCAL mode, _run must populate TurnContext with the LOCAL
    chat model + Mode.LOCAL in complete_kwargs (no _force_model_id),
    and Mode.LOCAL on ctx.mode for dispatch. Captures the ctx by
    patching the kernel turn driver to a recorder — proves the
    sidecar's mode-plumbing reaches chatbot_turn correctly, regardless
    of how the loop internally consumes it."""
    from engine.llm import Mode, Provider
    from types import SimpleNamespace
    from engine.chatbot_turn import TurnResult
    from engine.phases import chat as chat_phase
    from engine.phases import model_specs as model_specs

    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env

    captured: list = []

    def _fake_run(ctx, mode, execution_env=None):
        captured.append(ctx)
        return TurnResult(answer="ok", retrieved=[], lookup_fired=False, hops=1)

    monkeypatch.setattr(chat_phase, "run_chat_turn", _fake_run)
    monkeypatch.setattr(model_specs, "build_stage_env", lambda *a, **k: None)
    monkeypatch.setattr(
        chatbot_sidecar, "_read_app_config", lambda: {"mode": "local"})
    monkeypatch.setattr(
        chatbot_sidecar, "resolve_chatbot_from_config",
        lambda _cfg: {"model": "kimi-k2-6", "reasoning": False})
    monkeypatch.setattr(
        chatbot_sidecar, "get_mode_spec",
        lambda mode: SimpleNamespace(model_id="qwen3.5:9b",
                                     provider=Provider.OLLAMA))

    rc = chatbot_sidecar._run("hello there", [])
    assert rc == 0
    assert len(captured) == 1
    ctx = captured[0]
    assert ctx.mode == Mode.LOCAL
    assert ctx.complete_kwargs == {"model": "qwen3.5:9b", "mode": Mode.LOCAL}
    assert "_force_model_id" not in ctx.complete_kwargs  # unsafe on local


# The legacy `test_run_forces_caller_model_id_for_correct_reasoning_kwarg`
# asserted the `_force_model_id` / `_force_reasoning_*` kwargs on the old
# `complete()` call. Post-cutover the chatbot pins its model via the kernel
# chat spec, not those call kwargs — the model-pinning + config-driven
# reasoning pass-through invariants are covered by
# `test_chat_call_kwargs_cloud_keeps_kimi_and_force` and the three
# `test_chatbot_reasoning_*` tests.


# ── Chat-session demarcation (#503, re-homed onto #454-P1) ───────────
#
# The chatbot log is append-only across every session ever; these pin
# the boundary contract — a session_id stamped on every record + exactly
# one session_start marker per session — so one session's calls are
# selectable without guessing the cutoff. Under the persistent sidecar
# a "session" is the PROCESS lifetime: the source moved content-hash →
# process uuid and the marker moved empty-history-turn → process start
# (`_start_session`), the swap #503's gate-1 designed for. The contract
# is unchanged; only the source moved.


def _stub_chat_env(chatbot_sidecar, monkeypatch):
    """Stub provider + config so the session machinery + a pure-chat
    _run turn fire without a live model or a corpus store."""
    _install_scripted_chat_provider(monkeypatch, _forever("just chatting"))
    monkeypatch.setattr(chatbot_sidecar, "_read_app_config", lambda: {})
    monkeypatch.setattr(
        chatbot_sidecar, "resolve_chatbot_from_config",
        lambda cfg: {"model": "chat-model-x"})
    # Points the call-stats stream + payloads companion at the per-conversation
    # telemetry dir (production runs this once at process start). The
    # session_start marker + the kernel telemetry hook both write there.
    chatbot_sidecar._wire_call_stats()


def test_session_id_stable_across_turns_of_one_process():
    """One process = one session: the session_id is minted once at
    process start and is identical for every turn the process serves;
    a fresh process (a new _start_session) is a new session with a
    distinct id. This is the re-homed equivalent of #503's
    'stable across turns of one conversation' contract."""
    from engine import chatbot_sidecar

    chatbot_sidecar._start_session()
    sid = chatbot_sidecar._SESSION_ID
    assert sid and len(sid) == 16
    # Many turns of THIS process all see the same id (set once, never
    # recomputed per turn).
    assert chatbot_sidecar._SESSION_ID == sid
    assert chatbot_sidecar._SESSION_ID == sid
    # A fresh process / session ⇒ a distinct id.
    chatbot_sidecar._start_session()
    assert chatbot_sidecar._SESSION_ID != sid


def test_bound_run_name_derived_readonly(tmp_path):
    """The bound run is the resolved store path's run dir, read-only —
    None when nothing resolved."""
    from engine import chatbot_sidecar

    store = tmp_path / "personal-os" / "stages" / "06-embeddings" / "vectors.db"
    assert chatbot_sidecar._bound_run_name(store) == "personal-os"
    assert chatbot_sidecar._bound_run_name(None) is None


def _make_run(logs_root, name, *, size, created_at=None):
    """Materialize a run dir with a vectors.db of `size` bytes."""
    store = logs_root / name / "stages" / "06-embeddings" / "vectors.db"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_bytes(b"x" * size)
    if created_at is not None:
        import json as _json
        (logs_root / name / "config.json").write_text(
            _json.dumps({"created_at": created_at}))
    return store


def test_is_nonempty_store_excludes_zero_byte(tmp_path):
    """Defect #2: a 0-byte vectors.db is in-flight/aborted, not a
    bindable store — the non-empty guarantee callers always claimed."""
    from engine import chatbot_sidecar

    full = _make_run(tmp_path, "good", size=10)
    empty = _make_run(tmp_path, "inflight", size=0)
    assert chatbot_sidecar._is_nonempty_store(full) is True
    assert chatbot_sidecar._is_nonempty_store(empty) is False
    assert chatbot_sidecar._is_nonempty_store(tmp_path / "nope.db") is False


def test_run_creation_key_prefers_dir_prefix_then_config(tmp_path):
    """Order key is the run's creation time — the ISO dir prefix, else
    config.created_at — never the db file's mtime."""
    from engine import chatbot_sidecar

    (tmp_path / "2026-05-16T03-14-54Z-xttq").mkdir()
    assert (chatbot_sidecar._run_creation_key(
        tmp_path / "2026-05-16T03-14-54Z-xttq") == "20260516031454")
    d = tmp_path / "legacy-slug-only"
    d.mkdir()
    (d / "config.json").write_text(
        '{"created_at": "2026-05-10T01:07:11Z"}')
    assert chatbot_sidecar._run_creation_key(d) == "20260510010711"
    bare = tmp_path / "no-prefix-no-config"
    bare.mkdir()
    assert chatbot_sidecar._run_creation_key(bare) is None


def test_latest_store_path_orders_by_creation_not_mtime_and_skips_empty(
        tmp_path):
    """The characterised bug end-to-end: an OLDER run whose db is
    touched LAST must NOT shadow a newer run, and a 0-byte store of the
    newest run must NOT win. Selection is by run creation time over the
    non-empty predicate."""
    import os
    import time
    from engine import chatbot_sidecar

    older = _make_run(tmp_path, "2026-05-16T01-14-42Z-f66s", size=20)
    newer = _make_run(tmp_path, "2026-05-16T03-14-54Z-xttq", size=20)
    # Touch the OLDER run's db most recently — the exact mtime trap.
    now = time.time()
    os.utime(newer, (now - 9000, now - 9000))
    os.utime(older, (now, now))
    assert chatbot_sidecar._latest_store_path(tmp_path) == newer

    # A still-newer run with a 0-byte (in-flight) store is excluded —
    # the default stays the newest NON-EMPTY run.
    _make_run(tmp_path, "2026-05-16T09-00-00Z-zzzz", size=0)
    assert chatbot_sidecar._latest_store_path(tmp_path) == newer


def test_resolve_session_binding_env_pick_then_stale_fallback(
        tmp_path, monkeypatch):
    """The Rust-resolved pick is honoured (source 'user'); a pick whose
    store vanished/emptied degrades to the most-recent-non-empty default
    (source 'default') rather than binding nothing or a 0-byte file."""
    from engine import chatbot_sidecar

    monkeypatch.setenv("BASEVAULT_LOGS_ROOT", str(tmp_path))
    picked = _make_run(tmp_path, "2026-05-16T01-14-42Z-f66s", size=20)
    default_newest = _make_run(
        tmp_path, "2026-05-16T03-14-54Z-xttq", size=20)

    monkeypatch.setenv("BASEVAULT_CHATBOT_STORE_PATH", str(picked))
    monkeypatch.setenv(
        "BASEVAULT_CHATBOT_RUN_ID", "2026-05-16T01-14-42Z-f66s")
    monkeypatch.setenv("BASEVAULT_CHATBOT_BIND_SOURCE", "user")
    path, run, src = chatbot_sidecar._resolve_session_binding()
    assert path == picked
    assert run == "2026-05-16T01-14-42Z-f66s"
    assert src == "user"

    # Pick points at a now-missing store → fall back to the default
    # (newest non-empty), reported as 'default', never the stale pick.
    monkeypatch.setenv(
        "BASEVAULT_CHATBOT_STORE_PATH", str(tmp_path / "gone.db"))
    path, run, src = chatbot_sidecar._resolve_session_binding()
    assert path == default_newest
    assert src == "default"

    # No env at all (ad-hoc/test path) → same default rule.
    monkeypatch.delenv("BASEVAULT_CHATBOT_STORE_PATH")
    monkeypatch.delenv("BASEVAULT_CHATBOT_RUN_ID")
    monkeypatch.delenv("BASEVAULT_CHATBOT_BIND_SOURCE")
    path, _run, src = chatbot_sidecar._resolve_session_binding()
    assert path == default_newest and src == "default"


def test_session_start_records_bound_selection(
        chatbot_telemetry_env, monkeypatch, capsys):
    """The user-selected-vs-default marker is recorded in both the
    session_start telemetry and the chatbot_bound UI event (#507)."""
    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env
    _stub_chat_env(chatbot_sidecar, monkeypatch)
    store = _make_run(tmp_path, "2026-05-16T03-14-54Z-xttq", size=20)
    monkeypatch.setenv("BASEVAULT_CHATBOT_STORE_PATH", str(store))
    monkeypatch.setenv(
        "BASEVAULT_CHATBOT_RUN_ID", "2026-05-16T03-14-54Z-xttq")
    monkeypatch.setenv("BASEVAULT_CHATBOT_BIND_SOURCE", "user")

    chatbot_sidecar._start_session()
    starts = [
        e for e in _read_events(tmp_path) if e["event"] == "session_start"]
    assert len(starts) == 1
    assert starts[0]["bound_run"] == "2026-05-16T03-14-54Z-xttq"
    assert starts[0]["bound_selection"] == "user"
    lines = [
        ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    import json
    bound = [
        json.loads(ln) for ln in lines
        if '"chatbot_bound"' in ln]
    assert len(bound) == 1 and bound[0]["selection"] == "user"


def test_session_start_at_process_start_marks_once_and_stamps_records(
        chatbot_telemetry_env, monkeypatch, capsys):
    """Process start ⇒ exactly one session_start marker (session_id +
    model + bound run/store), the process session_id stamped on every
    begin/end + payload record of every turn it serves, and one
    chatbot_bound event to the UI."""
    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env
    _stub_chat_env(chatbot_sidecar, monkeypatch)
    # Re-enable the chat-side jsonl path so the structured per-call
    # record is readable; the production sidecar leaves it off (the
    # YAML companion carries the same content turn-organized).
    llm.set_payloads_jsonl_path(tmp_path / "chatbot" / "llm-payloads.jsonl")
    chatbot_sidecar._start_session()
    chatbot_sidecar._run("my first question", [])

    events = _read_events(tmp_path)
    starts = [e for e in events if e["event"] == "session_start"]
    assert len(starts) == 1
    sid = chatbot_sidecar._SESSION_ID
    assert sid and starts[0]["session_id"] == sid
    assert starts[0]["model"] == "chat-model-x"
    # No corpus store in the isolated logs root → bound run is null,
    # surfaced (not hidden) so a missing/blind binding is visible.
    assert starts[0]["bound_run"] is None
    assert starts[0]["bound_store_path"] is None
    assert "call_id" not in starts[0]  # marker is not a call

    # Same session_id on the converse begin/end + its payload.
    begin = next(e for e in events if e["event"] == "begin")
    end = next(e for e in events if e["event"] == "end")
    assert begin["session_id"] == sid
    assert end["session_id"] == sid
    payloads = _read_payloads(tmp_path)
    assert payloads and payloads[0]["session_id"] == sid

    # The UI got exactly one chatbot_bound event for the session.
    lines = [
        ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    import json
    bound = [
        json.loads(ln) for ln in lines
        if '"chatbot_bound"' in ln]
    assert len(bound) == 1 and bound[0]["run"] is None


def test_continuation_turn_emits_no_second_marker_but_keeps_session_id(
        chatbot_telemetry_env, monkeypatch, capsys):
    """Every later turn of the same process is a continuation: no
    second session_start marker, and the process session_id still
    stamps its records so they group with the session's first turn."""
    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env
    _stub_chat_env(chatbot_sidecar, monkeypatch)
    chatbot_sidecar._start_session()
    sid = chatbot_sidecar._SESSION_ID
    chatbot_sidecar._run("my first question", [])
    chatbot_sidecar._run(
        "a follow-up",
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content": "an answer"},
        ])

    events = _read_events(tmp_path)
    # One marker for the whole process/session, not one per turn.
    assert len([e for e in events if e["event"] == "session_start"]) == 1
    assert chatbot_sidecar._SESSION_ID == sid
    # Every begin across both turns carries the one session id.
    begins = [e for e in events if e["event"] == "begin"]
    assert len(begins) >= 2
    assert all(b["session_id"] == sid for b in begins)


def test_pipeline_runner_events_carry_no_session_id(
        chatbot_telemetry_env):
    """begin/end for a non-chatbot call (session_id unset) stay
    byte-unchanged — no new key for older readers of the pipeline log."""
    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env
    llm.set_calls_jsonl_path(tmp_path / "chatbot" / "llm-calls.jsonl")
    (tmp_path / "chatbot").mkdir(parents=True, exist_ok=True)
    cid = llm.begin_stat_record(
        stage="extract", category=None, model_hint="m")
    llm.finalize_stat_record(cid, success=True, duration_ms=5)
    for e in _read_events(tmp_path):
        assert "session_id" not in e


def test_tool_call_path_dispatches_validated_call(
        chatbot_telemetry_env, monkeypatch):
    """The decision turn's JSON tool call is parsed, validated, and
    dispatched against the bound corpus — exactly once, with the
    validated ToolCall (here a search). (That search skips the
    generative rerank is a dispatch-level guarantee, covered in
    test_chatbot_tools.py.)
    """
    import contextlib

    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env

    _install_scripted_chat_provider(monkeypatch, [
        '{"tool": "search", "query": "sleep notes"}',
        "Here is what I found.",
    ])
    monkeypatch.setattr(chatbot_sidecar, "_read_app_config", lambda: {})
    # Post-PR-3 the corpus binding is resolved once at process start and
    # `_run` reuses the `_SESSION_STORE_PATH` global — set it directly
    # rather than patching the now-once-per-session _latest_store_path.
    monkeypatch.setattr(
        chatbot_sidecar, "_SESSION_STORE_PATH", tmp_path / "v.db")

    class _Store:
        def count(self):
            return 1

    @contextlib.contextmanager
    def _fake_open_store(_path):
        yield _Store()

    from engine import chatbot_turn
    monkeypatch.setattr(chatbot_turn, "open_store", _fake_open_store)

    seen = []

    def _capture_dispatch(call, **kwargs):
        seen.append(call)
        return []

    monkeypatch.setattr(chatbot_turn, "dispatch", _capture_dispatch)

    rc = chatbot_sidecar._run("how did I sleep last week?", [])
    assert rc == 0

    assert len(seen) == 1
    assert seen[0].tool == "search"
    [lk] = seen[0].args["lookups"]
    assert lk.query == "sleep notes"


def test_prepended_preamble_fires_retrieval_and_never_leaks(
        chatbot_telemetry_env, monkeypatch, capsys):
    """#534 GA regression, end-to-end through `_run`. The decision turn
    returns a prose preamble + blank line + the JSON tool call (the exact
    GA shape). The turn must: (a) fire retrieval with the tool call's
    query, and (b) NEVER emit the preamble or the tool-call JSON as a
    `chatbot_chunk` — the decision turn isn't streamed, and a tool-call
    turn is suppressed whole."""
    import contextlib
    import json

    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env

    preamble = (
        "I'd be happy to help you explore a lifelong pattern, but I'll "
        "need to look through your vault first."
    )
    directive_q = (
        "What recurring behavioral patterns appear throughout my notes?"
    )
    tool_json = json.dumps({"tool": "search", "query": directive_q})
    leaked_reply = f"{preamble}\n\n{tool_json}"
    _install_scripted_chat_provider(
        monkeypatch, [leaked_reply, "Here is the grounded answer [1]."])
    monkeypatch.setattr(chatbot_sidecar, "_read_app_config", lambda: {})
    monkeypatch.setattr(
        chatbot_sidecar, "_SESSION_STORE_PATH", tmp_path / "v.db")

    class _Store:
        def count(self):
            return 1

    @contextlib.contextmanager
    def _fake_open_store(_path):
        yield _Store()

    from engine import chatbot_turn
    monkeypatch.setattr(chatbot_turn, "open_store", _fake_open_store)

    seen = []

    def _capture_dispatch(call, **kwargs):
        [lk] = call.args["lookups"]
        seen.append(lk.query)
        return []

    monkeypatch.setattr(chatbot_turn, "dispatch", _capture_dispatch)

    rc = chatbot_sidecar._run("tell me of a lifelong pattern of mine", [])
    assert rc == 0

    # (a) Dispatch fired with the tool call's query — the preamble did
    # not defeat classification.
    assert seen == [directive_q]

    # (b) No chatbot_chunk leaked the preamble or the tool-call JSON. The
    # only streamed content is the grounded answer.
    chunks = [
        json.loads(ln)["delta"]
        for ln in capsys.readouterr().out.splitlines()
        if ln.strip() and json.loads(ln).get("event") == "chatbot_chunk"
    ]
    joined = "".join(chunks)
    assert '"tool"' not in joined
    assert preamble not in joined
    assert joined == "Here is the grounded answer [1]."


# ── The chatbot reasoning toggle reaches the actual model call ────────
#
# These are not unit stubs of the reasoning logic — they drive the REAL
# `llm.complete()` reasoning-resolution chain
# (`stage_scope("chatbot")` → `_reasoning_enabled_for("chatbot")` →
# `resolve_chatbot_from_config` → `_reasoning_kwargs`) and intercept
# ONLY at the network boundary (the kernel provider's `chat.completions.create`),
# capturing the exact outbound request the production code assembled.
# The assertion is on the `extra_body.chat_template_kwargs` reasoning
# boolean — keyed `thinking` for Kimi, `enable_thinking` for GLM/Gemma —
# so a passing test means the Settings toggle's value physically arrives
# on the wire, with no force override anywhere. This is the behavior the
# acceptance demands that jsdom / unit-mocking of `complete` cannot show.


def _fake_tinfoil_capturing(created: list):
    """A stand-in attested client whose `chat.completions.create`
    records the kwargs the real `complete()` built and returns one
    content chunk + a usage chunk so `_consume_chat_stream` finishes
    cleanly (no empty-on-wire raise)."""
    import types

    def _create(**kw):
        created.append(kw)
        chunk = types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(
                    content="ok", reasoning_content=None),
                finish_reason="stop")],
            usage=None,
        )
        tail = types.SimpleNamespace(
            choices=[],
            usage=types.SimpleNamespace(
                prompt_tokens=3, completion_tokens=1,
                completion_tokens_details=None),
        )
        return iter([chunk, tail])

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_create)))


def _thinking_kwarg_for_config(app_cfg, llm=None, monkeypatch=None) -> bool:
    """Resolve `app_cfg` to the reasoning-on/off boolean the chatbot phase
    puts on the wire.

    Post-cutover the chatbot reasoning toggle flows through the kernel
    model spec, not the legacy `complete()` path: the app config is
    resolved by `resolve_chatbot_from_config` (model + reasoning) and the
    model spec turns that into the provider `extra_body` via
    `_reasoning_kwargs`. This drives that exact live seam, so the three
    reasoning tests still assert what physically reaches the model.

    The `chat_template_kwargs` key name is per-model-family — Kimi uses
    `thinking`, GLM/Gemma use `enable_thinking` — so read whichever key
    the resolved model emits rather than pinning to one. Both carry the
    same on/off boolean.
    """
    from engine.chatbot import resolve_chatbot_from_config
    from engine.phases.model_specs import _reasoning_kwargs

    chat = resolve_chatbot_from_config(app_cfg)
    kw = _reasoning_kwargs(chat["model"], chat["reasoning"])
    ctk = kw.get("extra_body", {}).get("chat_template_kwargs", {})
    if "thinking" in ctk:
        return ctk["thinking"]
    return ctk.get("enable_thinking")


def test_chatbot_reasoning_ship_default_off_reaches_the_call(
        chatbot_telemetry_env, monkeypatch):
    """No `chatbot` field in config (fresh install) → the ship-default
    (reasoning OFF, after the live-too-slow finding) is what physically
    reaches the model: the outbound request carries
    chat_template_kwargs.thinking=False."""
    _tmp, llm, _sidecar = chatbot_telemetry_env
    assert DEFAULT_CHATBOT_REASONING is False  # guards the directed default
    thinking = _thinking_kwarg_for_config({}, llm, monkeypatch)
    assert thinking is False


def test_chatbot_reasoning_toggle_off_passes_off_to_the_call(
        chatbot_telemetry_env, monkeypatch):
    """User flipped the Settings toggle OFF → off is what reaches the
    model: chat_template_kwargs.thinking=False. No force-on override
    masks the user's choice."""
    _tmp, llm, _sidecar = chatbot_telemetry_env
    cfg = {"chatbot": {"model": "kimi-k2-6", "reasoning": False}}
    thinking = _thinking_kwarg_for_config(cfg, llm, monkeypatch)
    assert thinking is False


def test_chatbot_reasoning_toggle_on_passes_on_to_the_call(
        chatbot_telemetry_env, monkeypatch):
    """Toggle explicitly ON → thinking=True reaches the model. Together
    with the off-case this proves the value is a verbatim pass-through
    of the setting, not a hardcoded constant in either direction."""
    _tmp, llm, _sidecar = chatbot_telemetry_env
    cfg = {"chatbot": {"model": "kimi-k2-6", "reasoning": True}}
    thinking = _thinking_kwarg_for_config(cfg, llm, monkeypatch)
    assert thinking is True


# ── chatbot_thinking event — every turn, sidecar-driven ─────────────────
#
# The "Thinking…" indicator is a pure sidecar signal (like
# chatbot_retrieving), NOT a UI inference from a once-per-session flag.
# The sidecar emits chatbot_thinking at the start of EVERY turn — the
# decision turn is always buffered, so every response has an in-flight
# gap to fill. Shown for ALL responses regardless of reasoning
# (reasoning ships OFF by default and just lengthens the gap when on).
# These pin the contract: emitted every turn, before any visible
# output, independent of the reasoning setting.


def _chatbot_events(capsys):
    import json as _json
    return [
        _json.loads(ln)
        for ln in capsys.readouterr().out.splitlines()
        if ln.strip() and ln.lstrip().startswith("{")
    ]


def _stub_thinking_env(chatbot_sidecar, monkeypatch, reasoning):
    _install_scripted_chat_provider(monkeypatch, _forever("just chatting"))
    monkeypatch.setattr(chatbot_sidecar, "_read_app_config", lambda: {})
    monkeypatch.setattr(
        chatbot_sidecar, "resolve_chatbot_from_config",
        lambda cfg: {"model": "chat-model-x", "reasoning": reasoning})


def test_chatbot_thinking_emitted_before_output_when_reasoning_on(
        chatbot_telemetry_env, monkeypatch, capsys):
    """Reasoning-on turn → chatbot_thinking is emitted, and it precedes
    any visible output (the buffered-gap signal the panel shows)."""
    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env
    _stub_thinking_env(chatbot_sidecar, monkeypatch, reasoning=True)

    chatbot_sidecar._run("hello", [])

    evs = [e["event"] for e in _chatbot_events(capsys)]
    assert "chatbot_thinking" in evs
    # Emitted before the first visible output of the turn.
    first_output = next(
        i for i, e in enumerate(evs)
        if e in ("chatbot_chunk", "chatbot_retrieving", "chatbot_done"))
    assert evs.index("chatbot_thinking") < first_output


def test_chatbot_thinking_also_emitted_when_reasoning_off(
        chatbot_telemetry_env, monkeypatch, capsys):
    """Reasoning-off turn (the ship-default) → chatbot_thinking is ALSO
    emitted, before any visible output. "Thinking…" shows for ALL
    responses; it is a general in-flight indicator on the always-
    buffered decision turn, NOT gated on reasoning."""
    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env
    _stub_thinking_env(chatbot_sidecar, monkeypatch, reasoning=False)

    chatbot_sidecar._run("hello", [])

    evs = [e["event"] for e in _chatbot_events(capsys)]
    assert "chatbot_thinking" in evs
    first_output = next(
        i for i, e in enumerate(evs)
        if e in ("chatbot_chunk", "chatbot_retrieving", "chatbot_done"))
    assert evs.index("chatbot_thinking") < first_output


def test_chatbot_thinking_emitted_every_turn_regardless_of_reasoning(
        chatbot_telemetry_env, monkeypatch, capsys):
    """Emitted on EVERY turn regardless of the reasoning setting and
    independent of whether it changes mid-session: a reasoning-on turn
    and a later reasoning-off turn of the same process both emit it.
    The indicator is unconditional, not reasoning-gated."""
    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env
    _install_scripted_chat_provider(monkeypatch, _forever("just chatting"))
    monkeypatch.setattr(chatbot_sidecar, "_read_app_config", lambda: {})

    state = {"reasoning": True}
    monkeypatch.setattr(
        chatbot_sidecar, "resolve_chatbot_from_config",
        lambda cfg: {"model": "chat-model-x", "reasoning": state["reasoning"]})

    chatbot_sidecar._run("turn 1", [])
    assert "chatbot_thinking" in [e["event"] for e in _chatbot_events(capsys)]

    # Setting flipped OFF between turns; NO respawn (same process). The
    # indicator still fires — it is not gated on reasoning.
    state["reasoning"] = False
    chatbot_sidecar._run("turn 2", [])
    assert "chatbot_thinking" in [
        e["event"] for e in _chatbot_events(capsys)]


# ── Streaming gate (#806) ───────────────────────────────────────────────
#
# Stage-4's ReAct loop streams ``decision`` and ``grounded_decision``
# replies by default; the gate suppresses emission only when the first
# non-whitespace char of the reply looks like a tool call (``{`` or
# `` ` ``). These pin each branch of the gate plus the trailing
# single-emit fallback for non-streaming providers.


def _build_turn_ctx(
        tracked_complete, *, query="q", emit_log=None, monkeypatch=None):
    """Minimal ``TurnContext`` for ``chatbot_turn.run`` unit tests. Pass
    ``emit_log`` as a list and the context's emit() will append
    ``(event, payload)`` tuples to it; default constructs a fresh list
    that the test can read via the returned tuple.

    Pass ``monkeypatch`` to bind the ctx to a fake corpus store (with
    ``count() == 1`` and a no-op dispatcher) so the loop's #780
    no-corpus carve-out doesn't fire — the streaming-gate tests want
    to exercise the post-dispatch hop, not the unbound-refusal path.
    Without ``monkeypatch`` the ctx defaults to ``store_path=None``
    (the no-corpus path) for tests that explicitly probe that case.
    """
    import contextlib
    from pathlib import Path

    from engine.chatbot_turn import TurnContext

    if emit_log is None:
        emit_log = []

    def _emit(event, **payload):
        emit_log.append((event, payload))

    store_path: Path | None = None
    if monkeypatch is not None:
        from engine import chatbot_turn

        class _Store:
            def count(self):
                return 1

        @contextlib.contextmanager
        def _fake_open_store(_p):
            yield _Store()

        monkeypatch.setattr(chatbot_turn, "open_store", _fake_open_store)
        monkeypatch.setattr(chatbot_turn, "dispatch", lambda call, **kw: [])
        store_path = Path("/dev/null/fake-store")

    ctx = TurnContext(
        query=query,
        history=[],
        turn_index=1,
        session_id="s",
        store_path=store_path,
        bound_run=None,
        chatbot_config={"model": "m", "reasoning": False},
        tracked_complete=tracked_complete,
        emit=_emit,
    )
    return ctx, emit_log


def _streaming_complete(replies_deltas):
    """Build a fake tracked_complete that, on each invocation, fires
    ``on_chunk`` for every delta in the next inner list, then returns a
    completion whose content is the joined deltas. ``replies_deltas`` is
    a list of per-call delta lists (one inner list per LLM call the
    turn will make)."""
    seq = iter(replies_deltas)

    def _complete(messages, *, on_chunk=None, **kwargs):
        deltas = next(seq)
        for d in deltas:
            if on_chunk is not None:
                on_chunk(d)
        return _fake_completion(content="".join(deltas))

    return _complete


def _silent_complete(replies):
    """Fake tracked_complete that returns content WITHOUT firing
    on_chunk — mimics a TEE non-streamed response per
    [project_tee_onchunk_not_guaranteed]."""
    seq = iter(replies)

    def _complete(messages, **kwargs):
        return _fake_completion(content=next(seq))

    return _complete


def _chunks(emit_log):
    """Joined deltas of every chatbot_chunk event in order."""
    return "".join(p["delta"] for ev, p in emit_log if ev == "chatbot_chunk")


def _chunk_count(emit_log):
    return sum(1 for ev, _ in emit_log if ev == "chatbot_chunk")


def _bubble_state(emit_log):
    """The UI's chat-bubble ``t.a`` state, simulating the reducer:
    ``chatbot_chunk`` appends, ``chatbot_replace`` overwrites. The net
    string is what the user actually sees once every event has been
    processed — distinct from the raw deltas, which can include leaked
    fragments that a later ``chatbot_replace`` then wipes."""
    a = ""
    for ev, p in emit_log:
        if ev == "chatbot_chunk":
            a += p.get("delta", "")
        elif ev == "chatbot_replace":
            a = p.get("text", "")
    return a


def test_grounded_decision_prose_streams_token_by_token(monkeypatch):
    """Stream by default. The grounded_decision persona's prose reply
    arrives as multiple chatbot_chunk events, not a single end-of-call
    emit — the slice-1 UX restored under the stage-4 loop."""
    from engine import chatbot_turn

    complete = _streaming_complete([
        ['{"tool": "search", "query": "x"}'],
        ["Hel", "lo, ", "world."],
    ])
    ctx, log = _build_turn_ctx(complete, monkeypatch=monkeypatch)
    chatbot_turn.run(ctx)

    deltas = [p["delta"] for ev, p in log if ev == "chatbot_chunk"]
    assert deltas == ["Hel", "lo, ", "world."]


def test_tool_call_starting_with_brace_does_not_leak_to_chat(monkeypatch):
    """The clean tool-call shape — first non-whitespace char ``{`` —
    suppresses emission for the whole call. Slice-1's no-leak
    guarantee holds for the bare-JSON case."""
    from engine import chatbot_turn

    complete = _streaming_complete([
        ['{"tool": "search",', ' "query": "x"}'],
        ["The answer."],
    ])
    ctx, log = _build_turn_ctx(complete, monkeypatch=monkeypatch)
    chatbot_turn.run(ctx)

    joined = _chunks(log)
    assert "{" not in joined
    assert '"tool"' not in joined
    assert joined == "The answer."


def test_tool_call_starting_with_json_fence_does_not_leak_to_chat(monkeypatch):
    """The fenced tool-call shape — first non-whitespace char `` ` ``
    — suppresses emission for the whole call. Same guarantee as the
    bare-JSON case for models that wrap their call in ```json fences."""
    from engine import chatbot_turn

    complete = _streaming_complete([
        ["```json\n", '{"tool": "search", "query": "x"}', "\n```"],
        ["The answer."],
    ])
    ctx, log = _build_turn_ctx(complete, monkeypatch=monkeypatch)
    chatbot_turn.run(ctx)

    joined = _chunks(log)
    assert "`" not in joined
    assert "{" not in joined
    assert joined == "The answer."


def test_non_streaming_provider_falls_back_to_single_emit(monkeypatch):
    """TEE non-streamed response: ``on_chunk`` never fires, so the gate
    never advances. The trailing single-emit fallback delivers the
    full answer as one chatbot_chunk — the content-anchored fallback
    per [project_tee_onchunk_not_guaranteed]."""
    from engine import chatbot_turn

    ctx, log = _build_turn_ctx(
        _silent_complete([
            '{"tool": "search", "query": "x"}',
            "The answer.",
        ]),
        monkeypatch=monkeypatch,
    )
    chatbot_turn.run(ctx)

    assert _chunk_count(log) == 1
    assert _chunks(log) == "The answer."


def test_buffered_reply_that_parses_as_prose_still_emits_at_end():
    """Heuristic over-eager case: the reply's first non-whitespace char
    is ``{`` so the gate buffers, but ``parse_tool_call`` returns None
    because no valid JSON object is present. The voluntary-prose
    finalize path then reaches the trailing single-emit fallback so
    the user still sees the answer (this PR's orchestrator-suggested
    pin against subtle later breakage of the buffered branch)."""
    from engine import chatbot_turn

    # Reply starts with '{' but is not a parseable tool call.
    complete = _streaming_complete([['{ not really json after all'],])
    ctx, log = _build_turn_ctx(complete)
    chatbot_turn.run(ctx)

    assert _chunk_count(log) == 1
    assert _chunks(log) == "{ not really json after all"


def test_prose_json_prose_wipes_bubble_and_dispatches(monkeypatch):
    """#845 regression. A mixed-shape reply — prose preamble + embedded
    JSON tool call + prose epilogue — must end with a clean bubble:
    the JSON is extracted and dispatched, the streamed leak is wiped
    by a ``chatbot_replace`` event with empty text, and the next hop's
    grounded answer is the only thing the user is left looking at.

    The contract the bubble state pins:
        bubble_after_turn = grounded_answer
        no JSON-shaped substring, no leaked preamble, no leaked epilogue
    """
    from engine import chatbot_turn

    preamble = "Looking into your vault now."
    tool_json = '{"tool": "search", "query": "sleep notes"}'
    epilogue = " I'll search next."

    # First call streams the mixed-shape reply as MULTIPLE deltas (the
    # provider does not batch — exercises the streaming-branch onset
    # detector, where the wipe is the load-bearing recovery). The
    # batched-single-delta path is pinned separately, below.
    # Delta 1: preamble (gate opens to streaming, flushes head).
    # Delta 2: JSON tool call (still streaming until onset detected on
    #          the joined accumulator → flip to buffered_after_leak).
    # Delta 3: epilogue (suppressed; gate is terminal).
    complete = _streaming_complete([
        [preamble + "\n\n", tool_json, "\n" + epilogue],
        ["Here is what I found."],
    ])

    seen_calls = []

    def _capture_dispatch(call, **_kw):
        seen_calls.append(call)
        return []

    monkeypatch.setattr(chatbot_turn, "dispatch", _capture_dispatch)

    import contextlib

    class _Store:
        def count(self):
            return 1

    @contextlib.contextmanager
    def _fake_open_store(_path):
        yield _Store()

    monkeypatch.setattr(chatbot_turn, "open_store", _fake_open_store)

    ctx, log = _build_turn_ctx(complete)
    from pathlib import Path
    object.__setattr__(ctx, "store_path", Path("/dev/null"))
    chatbot_turn.run(ctx)

    # Dispatch fired with the embedded JSON's query — the preamble did
    # not defeat classification.
    assert len(seen_calls) == 1
    [lk] = seen_calls[0].args["lookups"]
    assert lk.query == "sleep notes"

    # Exactly one chatbot_replace event with empty text fired, before
    # the next hop's grounded answer began streaming.
    replace_events = [
        (i, p) for i, (ev, p) in enumerate(log) if ev == "chatbot_replace"
    ]
    assert len(replace_events) == 1
    _i, payload = replace_events[0]
    assert payload.get("text", None) == ""

    # The UI bubble — after applying chunk-appends and the replace —
    # holds ONLY the grounded answer. No JSON, no preamble, no epilogue.
    bubble = _bubble_state(log)
    assert bubble == "Here is what I found."
    assert '"tool"' not in bubble
    assert preamble not in bubble
    assert epilogue not in bubble


def test_prose_with_stray_brace_not_valid_json_does_not_wipe(monkeypatch):
    """Edge case from gate-1 expert review: prose containing a stray
    ``{`` that doesn't parse as a tool-call JSON object (e.g. the model
    talks about ``{name}`` syntax) must NOT trigger a wipe. The trigger
    keys off ``parse_tool_call`` returning a valid call, not a naive
    ``{`` scan, so a benign prose reply with stray braces streams
    through cleanly and stays visible."""
    from engine import chatbot_turn

    reply = (
        "You can write `{name}` to interpolate, and `{x: 1}` is JS object "
        "literal. Neither is a tool call."
    )
    complete = _streaming_complete([[reply]])
    ctx, log = _build_turn_ctx(complete, monkeypatch=monkeypatch)
    chatbot_turn.run(ctx)

    # No wipe — the reply was prose, not a tool call.
    replace_events = [p for ev, p in log if ev == "chatbot_replace"]
    assert replace_events == []
    # Bubble holds the full prose reply.
    assert _bubble_state(log) == reply


def test_mid_stream_json_onset_across_chunk_boundary_caught(monkeypatch):
    """Edge case from gate-1 expert review: the JSON-onset substring
    can straddle two chunks (one delta ends with the opening ``{``, the
    next starts with ``"tool":``). The gate's mid-stream detector
    searches the joined accumulator — not just the current delta — so
    the leak is still caught at the boundary; the body of the JSON
    never reaches the UI as chunks, and the post-hop ``chatbot_replace``
    cleans up whatever leaked in the chunk that completed the onset."""
    from engine import chatbot_turn

    # Split the reply so '{' and '"tool"' land in separate deltas.
    deltas = [
        "I see ",
        "[1]. ",
        "{",
        '"tool": "search", ',
        '"query": "x"}',
    ]
    complete = _streaming_complete([deltas, ["The grounded answer."]])

    seen_calls = []

    def _capture_dispatch(call, **_kw):
        seen_calls.append(call)
        return []

    monkeypatch.setattr(chatbot_turn, "dispatch", _capture_dispatch)

    import contextlib

    class _Store:
        def count(self):
            return 1

    @contextlib.contextmanager
    def _fake_open_store(_p):
        yield _Store()

    monkeypatch.setattr(chatbot_turn, "open_store", _fake_open_store)
    ctx, log = _build_turn_ctx(complete)
    from pathlib import Path
    object.__setattr__(ctx, "store_path", Path("/dev/null"))
    chatbot_turn.run(ctx)

    # Dispatch fired with the cross-boundary JSON's query.
    assert len(seen_calls) == 1
    [lk] = seen_calls[0].args["lookups"]
    assert lk.query == "x"

    # Exactly one wipe, then the next-hop answer streams.
    replace_events = [p for ev, p in log if ev == "chatbot_replace"]
    assert len(replace_events) == 1
    assert replace_events[0].get("text", None) == ""

    # The chunk AFTER the onset-completing one must not have streamed —
    # ``"query"`` is from the final delta, fully inside the suppression
    # window.
    raw_chunks = _chunks(log)
    assert '"query"' not in raw_chunks

    # Net UI state is just the grounded answer.
    assert _bubble_state(log) == "The grounded answer."


def test_mid_stream_whitespace_after_brace_still_caught(monkeypatch):
    """``parse_tool_call`` accepts a tool call with whitespace between
    the opening ``{`` and the first key (``{ "tool": ...``) — its
    JSON-object scan is whitespace-tolerant. The mid-stream onset probe
    must match that tolerance: a literal ``{"tool`` substring check
    would miss this form and let the body of the JSON stream all the
    way through to the post-stream wipe. Pin the whitespace-tolerant
    regex contract directly."""
    from engine import chatbot_turn

    # `{ "tool"` — space after `{`. Valid JSON the parser accepts.
    deltas = [
        "Looking ",
        'now. { "tool": ',
        '"search", "query": "x"}',
    ]
    complete = _streaming_complete([deltas, ["The grounded answer."]])

    seen_calls = []

    def _capture_dispatch(call, **_kw):
        seen_calls.append(call)
        return []

    monkeypatch.setattr(chatbot_turn, "dispatch", _capture_dispatch)

    import contextlib

    class _Store:
        def count(self):
            return 1

    @contextlib.contextmanager
    def _fake_open_store(_p):
        yield _Store()

    monkeypatch.setattr(chatbot_turn, "open_store", _fake_open_store)
    ctx, log = _build_turn_ctx(complete)
    from pathlib import Path
    object.__setattr__(ctx, "store_path", Path("/dev/null"))
    chatbot_turn.run(ctx)

    # Dispatch fired with the whitespace-shape JSON's query.
    assert len(seen_calls) == 1
    [lk] = seen_calls[0].args["lookups"]
    assert lk.query == "x"

    # The wipe fired; the FINAL delta carrying the query and the
    # closing brace was fully inside the suppression window. The
    # chunk that completes the onset (``...{ "tool": ``) does flow
    # per accept-after-emit semantics, but no further delta does.
    raw_chunks = _chunks(log)
    assert '"query"' not in raw_chunks
    replace_events = [p for ev, p in log if ev == "chatbot_replace"]
    assert len(replace_events) == 1
    assert replace_events[0].get("text", None) == ""

    # Net bubble state: just the grounded answer.
    assert _bubble_state(log) == "The grounded answer."


def test_batched_first_delta_prose_plus_json_suppresses_zero_leak(monkeypatch):
    """When the provider batches the prose preamble and the full JSON
    tool call into a single first delta, the gate's undecided→streaming
    transition would otherwise flush the entire head as one chunk
    before the streaming-branch detector ever ran. Apply the onset
    probe to the flushed head too: the batched-first-delta shape is
    suppressed before emission, so zero chunks reach the UI and no
    wipe is needed (``gate.streamed`` stays False)."""
    from engine import chatbot_turn

    # The whole prose + JSON in ONE delta — the batched-provider case.
    deltas = ['Looking... {"tool": "search", "query": "x"}']
    complete = _streaming_complete([deltas, ["The grounded answer."]])

    seen_calls = []

    def _capture_dispatch(call, **_kw):
        seen_calls.append(call)
        return []

    monkeypatch.setattr(chatbot_turn, "dispatch", _capture_dispatch)

    import contextlib

    class _Store:
        def count(self):
            return 1

    @contextlib.contextmanager
    def _fake_open_store(_p):
        yield _Store()

    monkeypatch.setattr(chatbot_turn, "open_store", _fake_open_store)
    ctx, log = _build_turn_ctx(complete)
    from pathlib import Path
    object.__setattr__(ctx, "store_path", Path("/dev/null"))
    chatbot_turn.run(ctx)

    # Dispatch fired with the embedded JSON's query.
    assert len(seen_calls) == 1
    [lk] = seen_calls[0].args["lookups"]
    assert lk.query == "x"

    # Zero leak from the batched delta — the undecided-branch onset
    # check suppressed before the would-be flush.
    raw_chunks = _chunks(log)
    assert "Looking" not in raw_chunks
    assert '"tool"' not in raw_chunks
    assert '"query"' not in raw_chunks

    # No wipe needed: the trigger keys on gate.streamed, and nothing
    # streamed. The clean-tool-call branch and the zero-leak batched
    # branch converge to the same UX shape.
    replace_events = [p for ev, p in log if ev == "chatbot_replace"]
    assert replace_events == []

    # Net UI state: just the grounded answer streams.
    assert _bubble_state(log) == "The grounded answer."


def test_false_positive_onset_in_prose_restores_full_answer():
    """The mid-stream onset probe is high-precision but not perfectly
    tight — a model writing prose that literally mentions ``{"tool``
    (e.g. explaining the tool-call protocol back to the user) trips
    the probe, gets its tail suppressed, then ``parse_tool_call``
    returns None because no balanced JSON tool object exists. The
    voluntary-prose-finalize branch must emit ``chatbot_replace`` with
    the full prose so the UI + persisted transcript hold the answer
    the model actually produced — without this recovery the trailing
    single-emit fallback skips on ``gate.streamed`` and the user is
    stuck with a truncated prefix."""
    from engine import chatbot_turn

    # Prose that literally mentions ``{"tool`` without producing a
    # balanced JSON tool-call object.
    deltas = [
        "Use the literal ",
        '{"tool',
        " substring to identify tool calls in your data.",
    ]
    full_prose = "".join(deltas)
    complete = _streaming_complete([deltas])

    ctx, log = _build_turn_ctx(complete)
    chatbot_turn.run(ctx)

    # Recovery: chatbot_replace emitted with the full prose so the
    # bubble + saved transcript hold the whole answer, not the
    # truncated prefix the suppression would otherwise leave.
    replace_events = [p for ev, p in log if ev == "chatbot_replace"]
    assert len(replace_events) == 1
    assert replace_events[0].get("text", None) == full_prose

    # Net bubble state: the FULL prose answer.
    assert _bubble_state(log) == full_prose


def test_false_positive_onset_batched_no_duplicate_emission():
    """The provider can batch a prose-mentioning-``{"tool`` reply into
    a SINGLE first delta (no balanced JSON anywhere). The undecided-
    branch onset probe suppresses before any chunk emits, so
    ``gate.streamed`` stays False. The voluntary-prose-finalize
    recovery emits ``chatbot_replace text=full_reply``; without an
    additional guard the trailing single-emit fallback would ALSO
    fire (its predicate keys on ``not gate.streamed``) and the bubble
    would carry the same prose duplicated. Pin: exactly one
    ``chatbot_replace``, zero ``chatbot_chunk`` events, bubble equals
    the full prose (NOT duplicated)."""
    from engine import chatbot_turn

    full_prose = 'Use the literal {"tool substring to identify tool calls.'
    complete = _streaming_complete([[full_prose]])

    ctx, log = _build_turn_ctx(complete)
    chatbot_turn.run(ctx)

    replace_events = [p for ev, p in log if ev == "chatbot_replace"]
    assert len(replace_events) == 1
    assert replace_events[0].get("text", None) == full_prose

    chunk_events = [p for ev, p in log if ev == "chatbot_chunk"]
    assert chunk_events == []

    # Net bubble state: just the full prose, NOT duplicated.
    assert _bubble_state(log) == full_prose


def test_stream_gate_grounded_final_persona_skips_onset_probe():
    """``_StreamGate`` created with ``gated=False`` (the grounded_final
    persona — structurally forced to answer, no tool-call dispatch
    possible) must NOT trigger mid-stream suppression even when the
    streamed text contains the canonical onset substring. A grounded
    answer that legitimately quotes a tool-call snippet (the model
    explaining the protocol back to the user, for example) has to
    reach the UI in full — suppressing it would truncate the answer
    at the substring with no recovery path (the loop's grounded_final
    branch doesn't go through the false-positive ``chatbot_replace``
    recovery, and the trailing single-emit fallback skips because
    ``gate.streamed`` is True)."""
    from engine.chatbot_turn import _StreamGate

    emitted = []

    def emit(event, **payload):
        emitted.append((event, payload))

    gate = _StreamGate(emit, gated=False)
    gate.on_chunk("The tool-call protocol uses ")
    gate.on_chunk('{"tool": "search", "query": "x"}')
    gate.on_chunk(" as the JSON shape.")

    chunks = [p["delta"] for ev, p in emitted if ev == "chatbot_chunk"]
    assert chunks == [
        "The tool-call protocol uses ",
        '{"tool": "search", "query": "x"}',
        " as the JSON shape.",
    ]
    assert gate.leak_detected is False


def test_clean_tool_call_emits_no_replace_event(monkeypatch):
    """The clean-tool-call case (gate buffered from the first char) must
    NOT emit a ``chatbot_replace`` — the trigger requires that the gate
    actually streamed chunks. Without this guard a wipe would fire for
    every dispatching hop, including the lossless ones."""
    from engine import chatbot_turn

    complete = _streaming_complete([
        ['{"tool": "search", ', '"query": "x"}'],
        ["The answer."],
    ])

    monkeypatch.setattr(chatbot_turn, "dispatch", lambda call, **kw: [])

    import contextlib

    class _Store:
        def count(self):
            return 1

    @contextlib.contextmanager
    def _fake_open_store(_p):
        yield _Store()

    monkeypatch.setattr(chatbot_turn, "open_store", _fake_open_store)
    ctx, log = _build_turn_ctx(complete)
    from pathlib import Path
    object.__setattr__(ctx, "store_path", Path("/dev/null"))
    chatbot_turn.run(ctx)

    replace_events = [p for ev, p in log if ev == "chatbot_replace"]
    assert replace_events == []
    assert _bubble_state(log) == "The answer."


def test_local_mode_clamps_per_lookup_count_to_local_max(monkeypatch):
    """LOCAL mode threads tighter caps through ``validate_tool_call``.
    A model emitting ``count=20`` on LOCAL must reach dispatch as 15
    (the LOCAL_MAX_COUNT ceiling), not 20 — the cloud cap should never
    leak into a LOCAL turn even when the model explicitly asks for it."""
    from engine import chatbot_turn
    from engine.chatbot_tools import LOCAL_MAX_COUNT
    from engine.llm import Mode

    # Model emits an explicit count=20; on LOCAL we expect 15.
    tool_json = '{"tool": "search", "lookups": [{"query": "q", "count": 20}]}'
    complete = _streaming_complete([[tool_json], ["done"]])

    seen_calls = []

    def _capture_dispatch(call, **_kw):
        seen_calls.append(call)
        return []

    ctx, _log = _build_turn_ctx(complete, monkeypatch=monkeypatch)
    monkeypatch.setattr(chatbot_turn, "dispatch", _capture_dispatch)
    object.__setattr__(ctx, "mode", Mode.LOCAL)
    chatbot_turn.run(ctx)

    assert len(seen_calls) == 1
    [lk] = seen_calls[0].args["lookups"]
    assert lk.count == LOCAL_MAX_COUNT == 15


def test_local_mode_default_count_is_local_default(monkeypatch):
    """When the model omits ``count`` on LOCAL, the validator must
    default to LOCAL_DEFAULT_K (10), not the cloud DEFAULT_K (15)."""
    from engine import chatbot_turn
    from engine.chatbot_tools import LOCAL_DEFAULT_K
    from engine.llm import Mode

    tool_json = '{"tool": "search", "lookups": [{"query": "q"}]}'
    complete = _streaming_complete([[tool_json], ["done"]])

    seen_calls = []

    def _capture_dispatch(call, **_kw):
        seen_calls.append(call)
        return []

    ctx, _log = _build_turn_ctx(complete, monkeypatch=monkeypatch)
    monkeypatch.setattr(chatbot_turn, "dispatch", _capture_dispatch)
    object.__setattr__(ctx, "mode", Mode.LOCAL)
    chatbot_turn.run(ctx)

    [lk] = seen_calls[0].args["lookups"]
    assert lk.count == LOCAL_DEFAULT_K == 10


def test_local_mode_clamps_per_call_lookup_fanout(monkeypatch):
    """LOCAL mode threads the tighter per-call lookup cap through
    ``validate_tool_call`` too. A model bundling 3 lookups on LOCAL
    must reach dispatch with only the first ``LOCAL_MAX_LOOKUPS``
    legs, matching the smaller local chat model's focused-dispatch
    discipline."""
    from engine import chatbot_turn
    from engine.chatbot_tools import LOCAL_MAX_LOOKUPS
    from engine.llm import Mode

    tool_json = (
        '{"tool": "search", "lookups": ['
        '{"query": "first"},'
        '{"query": "second"},'
        '{"query": "third"}'
        ']}'
    )
    complete = _streaming_complete([[tool_json], ["done"]])

    seen_calls = []

    def _capture_dispatch(call, **_kw):
        seen_calls.append(call)
        return []

    ctx, _log = _build_turn_ctx(complete, monkeypatch=monkeypatch)
    monkeypatch.setattr(chatbot_turn, "dispatch", _capture_dispatch)
    object.__setattr__(ctx, "mode", Mode.LOCAL)
    chatbot_turn.run(ctx)

    assert len(seen_calls[0].args["lookups"]) == LOCAL_MAX_LOOKUPS == 2
    queries = [lk.query for lk in seen_calls[0].args["lookups"]]
    assert queries == ["first", "second"]


def test_local_mode_uses_local_accumulator_cap(monkeypatch):
    """LOCAL mode threads the tighter per-turn accumulator cap so
    the local chat model's CONTEXT block stays within the smaller
    window it can absorb. Pinned via the ``_caps_for_mode`` tuple."""
    from engine import chatbot_turn
    from engine.llm import Mode

    (
        _dk, _mc, _ml, acc_cap,
    ) = chatbot_turn._caps_for_mode(Mode.LOCAL)
    assert acc_cap == chatbot_turn.LOCAL_ACCUMULATOR_CAP == 120
    (
        _dk2, _mc2, _ml2, cloud_cap,
    ) = chatbot_turn._caps_for_mode(Mode.TEE)
    assert cloud_cap == chatbot_turn.ACCUMULATOR_CAP == 240


def test_cloud_mode_keeps_historical_count_caps(monkeypatch):
    """The default TurnContext mode (TEE / cloud) keeps the historical
    15/20 caps — a regression guard so the per-mode threading doesn't
    silently tighten cloud behavior."""
    from engine import chatbot_turn
    from engine.chatbot_tools import DEFAULT_K, MAX_COUNT

    # count=20 stays 20 on cloud (= MAX_COUNT).
    tool_json = '{"tool": "search", "lookups": [{"query": "q", "count": 20}]}'
    complete = _streaming_complete([[tool_json], ["done"]])

    seen_calls = []

    def _capture_dispatch(call, **_kw):
        seen_calls.append(call)
        return []

    ctx, _log = _build_turn_ctx(complete, monkeypatch=monkeypatch)
    monkeypatch.setattr(chatbot_turn, "dispatch", _capture_dispatch)
    # ctx.mode defaults to Mode.TEE — the cloud path.
    chatbot_turn.run(ctx)
    [lk] = seen_calls[0].args["lookups"]
    assert lk.count == MAX_COUNT == 20

    # Omitted count defaults to DEFAULT_K (15).
    tool_json2 = '{"tool": "search", "lookups": [{"query": "q"}]}'
    complete2 = _streaming_complete([[tool_json2], ["done"]])
    seen_calls.clear()
    ctx2, _log2 = _build_turn_ctx(complete2, monkeypatch=monkeypatch)
    monkeypatch.setattr(chatbot_turn, "dispatch", _capture_dispatch)
    chatbot_turn.run(ctx2)
    [lk2] = seen_calls[0].args["lookups"]
    assert lk2.count == DEFAULT_K == 15


# ── Malformed tool call: retry with feedback, never leak raw JSON ───────────
#
# A reply that OPENS a ``{"tool": ...}`` object but isn't valid JSON
# (``parse_tool_call`` returns None while ``_is_tool_call_attempt`` is
# True) used to fall through the voluntary-prose path and surface the raw
# unparseable bytes to the user as the "answer". The loop now treats it
# like the pipeline's retriable parse_error: wipe any leak, feed the
# failure back as a NOTES "previous attempts" line, and retry — bounded by
# ``MAX_PARSE_RETRIES`` (then forced to ``grounded_final``).


def test_is_tool_call_attempt_discriminates_open_object_from_prose():
    """The malformed-vs-prose discriminator: True only when the reply
    STRUCTURALLY opens a ``{"tool"`` object (optionally fenced), False
    for a bare brace that isn't a tool, and False for prose that merely
    quotes the protocol."""
    from engine.chatbot_turn import _is_tool_call_attempt

    assert _is_tool_call_attempt('{"tool": "search" "query": "x"}') is True
    assert _is_tool_call_attempt('{ "tool": broken') is True
    assert _is_tool_call_attempt('```json\n{"tool": "search" bad}\n```') is True
    # Bare brace that isn't a tool onset → not an attempt (stays prose).
    assert _is_tool_call_attempt("{ not really json after all") is False
    # Prose that merely mentions the protocol → not an attempt.
    assert _is_tool_call_attempt('The format is {"tool": "search"}') is False
    assert _is_tool_call_attempt("just a normal answer") is False


def test_malformed_tool_call_is_not_leaked_and_retries_with_feedback(monkeypatch):
    """First hop emits a malformed tool call (opens ``{"tool`` but invalid
    JSON); second hop answers in prose. The user must NEVER see the raw
    JSON, the final bubble is the prose answer, and the model gets the
    parse-failure note fed back into the retry prompt."""
    from engine import chatbot_turn

    malformed = '{"tool": "search", "lookups": [{"query": " "broke}]}'
    prompts_seen = []

    seq = iter([malformed, "Here is the real answer."])

    def _complete(messages, *, on_chunk=None, **kwargs):
        prompts_seen.append(messages)
        content = next(seq)
        if on_chunk is not None:
            on_chunk(content)
        return _fake_completion(content=content)

    ctx, log = _build_turn_ctx(_complete, monkeypatch=monkeypatch)
    chatbot_turn.run(ctx)

    # The raw malformed JSON never reaches the user's bubble.
    bubble = _bubble_state(log)
    assert bubble == "Here is the real answer."
    assert '"tool"' not in bubble
    assert "broke" not in bubble

    # The retry prompt carried the parse-failure note so the model could
    # self-correct (NOTES "previous attempts" line).
    second_prompt_text = str(prompts_seen[1])
    assert chatbot_turn.PARSE_FAILURE_NOTE in second_prompt_text

    # Diagnostics record the malformed hop distinctly (not prose_answer).
    outcomes = [h.get("hop_outcome") for h in ctx.hops_diag]
    assert "malformed_tool_call" in outcomes


def test_repeated_malformed_tool_calls_bail_to_grounded_final(monkeypatch):
    """A model stuck emitting malformed tool calls bails to a forced
    grounded answer after ``MAX_PARSE_RETRIES`` — it does NOT burn the
    whole hop budget, and it never leaks raw JSON. Here every decision
    hop is malformed; the grounded_final call (no tool protocol) returns
    the real answer."""
    from engine import chatbot_turn

    malformed = '{"tool": "search" bad json}'
    # Enough malformed replies to exceed the cap, then the grounded_final
    # answer. With MAX_PARSE_RETRIES=2: hops 1 and 2 malformed → hop 3 is
    # forced grounded_final.
    seq = iter([malformed, malformed, "Grounded answer from context."])
    n_calls = 0

    def _complete(messages, *, on_chunk=None, **kwargs):
        nonlocal n_calls
        n_calls += 1
        content = next(seq)
        if on_chunk is not None:
            on_chunk(content)
        return _fake_completion(content=content)

    ctx, log = _build_turn_ctx(_complete, monkeypatch=monkeypatch)
    chatbot_turn.run(ctx)

    assert _bubble_state(log) == "Grounded answer from context."
    # Bounded: 2 malformed + 1 forced grounded_final = 3 calls, well under
    # the MAX_HOPS+1 (=5) ceiling — proves the cap short-circuits.
    assert n_calls == 3
    assert chatbot_turn.MAX_PARSE_RETRIES == 2


def test_malformed_then_valid_tool_call_recovers_and_dispatches(monkeypatch):
    """Recovery path: a malformed first hop is retried, the model then
    emits a VALID tool call which dispatches normally, and the turn ends
    on the grounded prose answer. The malformed hiccup costs one retry,
    not the turn."""
    from engine import chatbot_turn

    malformed = '{"tool": "search", "query": " "oops}'
    valid = '{"tool": "search", "lookups": [{"query": "real query"}]}'
    seq = iter([malformed, valid, "Answer grounded in the hit."])

    def _complete(messages, *, on_chunk=None, **kwargs):
        content = next(seq)
        if on_chunk is not None:
            on_chunk(content)
        return _fake_completion(content=content)

    seen_calls = []

    def _capture_dispatch(call, **_kw):
        seen_calls.append(call)
        return []

    ctx, log = _build_turn_ctx(_complete, monkeypatch=monkeypatch)
    monkeypatch.setattr(chatbot_turn, "dispatch", _capture_dispatch)
    chatbot_turn.run(ctx)

    # The valid second call dispatched; the malformed first did not.
    assert len(seen_calls) == 1
    [lk] = seen_calls[0].args["lookups"]
    assert lk.query == "real query"
    assert _bubble_state(log) == "Answer grounded in the hit."


# ── #780: empty-binding push-on-completion + refuse-and-surface ─────────────
#
# The sidecar resolves the corpus binding ONCE at process start. When no
# run exists at startup, the cached `None` was sticky for the rest of the
# process lifetime — every later lookup short-circuited silently, the
# grounded-answer call invented/refused fluently against an empty
# context, and the UI carried no signal that retrieval never ran. The
# fix has two halves:
#   (1) Rust pushes a `run_available` frame on stdin when an ingest
#       run finishes. `_handle_run_available` binds the session to it
#       if currently unbound — no restart, no dropdown pick required.
#       If already bound, the push is a no-op (no silent swap).
#   (2) For the case where the user asks BEFORE any ingest has ever
#       finished (truly-no-corpus-ever), the top of `_run`'s lookup
#       branch emits a deterministic refusal chunk in place of the
#       grounded-answer call so the user sees an unambiguous "no
#       corpus" message — never a hallucinated reply against empty
#       retrieval.


def _stub_lookup_env(chatbot_sidecar, monkeypatch, *, replies):
    """Stub the converse + grounded-answer pair the lookup path makes.
    ``replies`` is the sequence each successive ``complete()`` returns;
    callers control whether the converse turn emits a LOOKUP directive
    and what the grounded turn replies with (or whether it runs at all).
    """
    provider = _install_scripted_chat_provider(monkeypatch, list(replies))
    monkeypatch.setattr(chatbot_sidecar, "_read_app_config", lambda: {})
    monkeypatch.setattr(
        chatbot_sidecar, "resolve_chatbot_from_config",
        lambda cfg: {"model": "chat-model-x"})
    return provider.seen_calls


def test_read_one_request_defaults_kind_to_turn_for_backward_compat(
        monkeypatch):
    """A frame WITHOUT a ``kind`` field — the only shape Rust ever wrote
    pre-#780 — still dispatches as a turn. Defends the one-shipped-pair
    backward-compat that lets the new sidecar deploy ahead of the new
    Rust without a synchronised release."""
    import io
    import json as _json
    from engine import chatbot_sidecar

    stdin = io.StringIO(
        _json.dumps({"query": "hi", "history": [], "turn_id": 7}) + "\n")
    monkeypatch.setattr(chatbot_sidecar.sys, "stdin", stdin)
    frame = chatbot_sidecar._read_one_request()
    assert frame == ("turn", ("hi", [], 7))


def test_read_one_request_dispatches_run_available_frame(monkeypatch):
    """A frame with ``kind=run_available`` is parsed into the helper's
    payload shape — the dispatch contract the Rust push relies on."""
    import io
    import json as _json
    from engine import chatbot_sidecar

    stdin = io.StringIO(_json.dumps({
        "kind": "run_available",
        "run_id": "2026-05-26T03-19-42Z-m9pp",
        "store_path": "/x/y/vectors.db",
        "selection": "default",
    }) + "\n")
    monkeypatch.setattr(chatbot_sidecar.sys, "stdin", stdin)
    frame = chatbot_sidecar._read_one_request()
    assert frame[0] == "run_available"
    assert frame[1] == {
        "run_id": "2026-05-26T03-19-42Z-m9pp",
        "store_path": "/x/y/vectors.db",
        "selection": "default",
    }


def test_read_one_request_run_available_requires_run_id_and_store_path(
        monkeypatch):
    """A malformed run_available frame (missing required fields) maps
    to a noop, not a crash. The loop has to survive garbage from a
    future Rust version writing a different shape."""
    import io
    import json as _json
    from engine import chatbot_sidecar

    stdin = io.StringIO(
        _json.dumps({"kind": "run_available", "run_id": 123}) + "\n")
    monkeypatch.setattr(chatbot_sidecar.sys, "stdin", stdin)
    assert chatbot_sidecar._read_one_request() == ("noop", None)


def test_read_one_request_unknown_kind_is_noop(monkeypatch):
    """A frame whose ``kind`` is unknown to this version is a noop, not
    a crash — forward-compat for a future frame shape this build doesn't
    know about yet."""
    import io
    import json as _json
    from engine import chatbot_sidecar

    stdin = io.StringIO(_json.dumps({"kind": "future_thing"}) + "\n")
    monkeypatch.setattr(chatbot_sidecar.sys, "stdin", stdin)
    assert chatbot_sidecar._read_one_request() == ("noop", None)


def test_handle_run_available_binds_session_when_unbound_and_emits(
        chatbot_telemetry_env, monkeypatch, capsys):
    """When the session has no corpus bound, a Rust push rebinds it and
    emits a fresh ``chatbot_bound`` so React's reducer refreshes the
    selector. This is the unbound-session arm of #780's fix."""
    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env
    _stub_chat_env(chatbot_sidecar, monkeypatch)
    monkeypatch.setenv("BASEVAULT_LOGS_ROOT", str(tmp_path))

    chatbot_sidecar._start_session()
    assert chatbot_sidecar._SESSION_STORE_PATH is None
    _ = _chatbot_events(capsys)  # drop _start_session's chatbot_bound

    pushed_store = tmp_path / "2026-05-26T03-19-42Z-m9pp" \
        / "stages" / "06-embeddings" / "vectors.db"
    chatbot_sidecar._handle_run_available({
        "run_id": "2026-05-26T03-19-42Z-m9pp",
        "store_path": str(pushed_store),
        "selection": "default",
    })

    assert chatbot_sidecar._SESSION_STORE_PATH == pushed_store
    assert chatbot_sidecar._SESSION_BOUND_RUN == "2026-05-26T03-19-42Z-m9pp"
    assert chatbot_sidecar._SESSION_BOUND_SELECTION == "default"

    evs = _chatbot_events(capsys)
    rebinds = [e for e in evs if e["event"] == "chatbot_bound"]
    assert len(rebinds) == 1
    assert rebinds[0]["run"] == "2026-05-26T03-19-42Z-m9pp"
    assert rebinds[0]["store_path"] == str(pushed_store)
    assert rebinds[0]["selection"] == "default"


def test_handle_run_available_noops_when_already_bound(
        chatbot_telemetry_env, monkeypatch, capsys):
    """An already-bound session does NOT swap to a newly-pushed run.
    Director's call: an in-conversation user must not have the corpus
    silently changed under them; the new run appears in the dropdown
    on next open (Rust-side `chatbot_list_runs` is fs-driven) and the
    user picks it manually via the existing respawn path."""
    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env
    _stub_chat_env(chatbot_sidecar, monkeypatch)
    monkeypatch.setenv("BASEVAULT_LOGS_ROOT", str(tmp_path))
    bound = _make_run(tmp_path, "2026-05-16T01-14-42Z-f66s", size=20)
    monkeypatch.setenv("BASEVAULT_CHATBOT_STORE_PATH", str(bound))
    monkeypatch.setenv(
        "BASEVAULT_CHATBOT_RUN_ID", "2026-05-16T01-14-42Z-f66s")
    monkeypatch.setenv("BASEVAULT_CHATBOT_BIND_SOURCE", "user")
    chatbot_sidecar._start_session()
    assert chatbot_sidecar._SESSION_STORE_PATH == bound
    _ = _chatbot_events(capsys)  # drop _start_session's chatbot_bound

    newer_store = tmp_path / "2026-05-26T03-19-42Z-newer" \
        / "stages" / "06-embeddings" / "vectors.db"
    chatbot_sidecar._handle_run_available({
        "run_id": "2026-05-26T03-19-42Z-newer",
        "store_path": str(newer_store),
        "selection": "default",
    })

    # Binding unchanged: still the originally-bound run.
    assert chatbot_sidecar._SESSION_STORE_PATH == bound
    assert chatbot_sidecar._SESSION_BOUND_RUN == "2026-05-16T01-14-42Z-f66s"

    # No silent rebind event to the UI — the dropped-push trace lives
    # in stderr (operator-visible via Rust's stderr drainer), nothing
    # on the chat surface.
    evs = _chatbot_events(capsys)
    assert not any(e["event"] == "chatbot_bound" for e in evs)


def test_main_loop_dispatches_run_available_before_turn(
        chatbot_telemetry_env, monkeypatch, capsys, tmp_path_factory):
    """End-to-end through `main`: a session that started with no run
    binds when the push arrives, then the next turn's tool call
    dispatches against the pushed corpus. This is the smoke-flow shape
    (#783 director-reproduced): fresh state → open chat → ingest →
    ask."""
    import contextlib
    import io
    import json as _json

    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env
    # The decision-persona call emits a JSON tool call; the
    # grounded_decision-persona call (after dispatch) emits a prose
    # answer that finalizes the turn.
    seen_calls = _stub_lookup_env(
        chatbot_sidecar, monkeypatch,
        replies=[
            '{"tool": "search", "query": "project history"}',
            "Here is what I found.",
        ])
    monkeypatch.setattr(
        chatbot_sidecar, "_warm_client", lambda: None)
    # Point the START-time logs-root at an empty dir so the session
    # binds to None at process start — the truly-empty case the push
    # is meant to recover. The pushed run lives in a separate dir
    # (mimicking Rust resolving the just-finished run's store
    # directly, not scanning logs_root again).
    empty_logs = tmp_path_factory.mktemp("empty-logs")
    monkeypatch.setenv("BASEVAULT_LOGS_ROOT", str(empty_logs))
    new_store = _make_run(tmp_path, "2026-05-26T03-19-42Z-m9pp", size=20)

    @contextlib.contextmanager
    def _fake_open_store(_path):
        class _Store:
            def count(self):
                return 1
        yield _Store()

    # The new loop opens + dispatches via `chatbot_turn`, not
    # `chatbot_sidecar` — patch there.
    from engine import chatbot_turn
    monkeypatch.setattr(chatbot_turn, "open_store", _fake_open_store)
    monkeypatch.setattr(chatbot_turn, "dispatch", lambda call, **kw: [])

    # Two stdin frames: a run_available push, then a turn that triggers
    # a tool call. EOF after.
    stdin = io.StringIO(
        _json.dumps({
            "kind": "run_available",
            "run_id": "2026-05-26T03-19-42Z-m9pp",
            "store_path": str(new_store),
            "selection": "default",
        }) + "\n"
        + _json.dumps({
            "kind": "turn", "query": "what's in my data?",
            "history": [], "turn_id": 1,
        }) + "\n"
    )
    monkeypatch.setattr(chatbot_sidecar.sys, "stdin", stdin)

    rc = chatbot_sidecar.main()
    assert rc == 0

    # After the run_available frame the session is bound; the turn
    # then exercises the full lookup path (decision + grounded_decision
    # both fire — NOT the refusal short-circuit).
    assert chatbot_sidecar._SESSION_STORE_PATH == new_store
    assert chatbot_sidecar._SESSION_BOUND_RUN == "2026-05-26T03-19-42Z-m9pp"
    assert len(seen_calls) == 2  # decision + grounded_decision

    evs = _chatbot_events(capsys)
    bounds = [e for e in evs if e["event"] == "chatbot_bound"]
    # One at process start (run=null), one on the push (the pushed run).
    assert len(bounds) == 2
    assert bounds[0]["run"] is None
    assert bounds[1]["run"] == "2026-05-26T03-19-42Z-m9pp"

    # The deterministic no-corpus refusal text never reached the user
    # — the push bound the session in time for the turn.
    chunks = [e for e in evs if e["event"] == "chatbot_chunk"]
    assert all(
        chatbot_turn.NO_CORPUS_REFUSAL_TEXT not in (c.get("delta") or "")
        for c in chunks
    )

    # The turn closed stamped against the newly-bound run, so
    # citations resolve against it forever. #834: a fully-grounded
    # non-refusal turn must not carry the ``refused`` flag — the UI
    # uses its absence to keep this turn's assistant slot in the
    # history fed into subsequent turns.
    dones = [e for e in evs if e["event"] == "chatbot_done"]
    assert len(dones) == 1
    assert dones[0]["run"] == "2026-05-26T03-19-42Z-m9pp"
    assert dones[0].get("refused") is not True


def test_lookup_unbound_session_refuses_with_deterministic_text(
        chatbot_telemetry_env, monkeypatch, capsys):
    """When the model issues a tool call AND no run has ever finished
    ingesting (no push has rebound the session either), the loop
    skips dispatch + the grounded-answer call entirely and emits a
    fixed deterministic refusal chunk in place of the model's answer.
    The grounded-answer call must NOT run — a grounded call against
    an empty retrieval set was the silent-invention vector this
    guards against. `chatbot_done` carries `resources=null, run=null`
    (pure-conversation shape; we did not run the tool)."""
    from engine import chatbot_turn

    tmp_path, llm, chatbot_sidecar = chatbot_telemetry_env
    seen_calls = _stub_lookup_env(
        chatbot_sidecar, monkeypatch,
        # If a second complete() fires (the grounded-answer call), this
        # iterator raises StopIteration and the test fails loudly — the
        # only way to pass is for the loop to refuse + break BEFORE
        # the second hop.
        replies=['{"tool": "search", "query": "project history"}'])
    monkeypatch.setenv("BASEVAULT_LOGS_ROOT", str(tmp_path))

    chatbot_sidecar._start_session()
    assert chatbot_sidecar._SESSION_STORE_PATH is None
    _ = _chatbot_events(capsys)
    rc = chatbot_sidecar._run("what's in my data?", [])
    assert rc == 0

    # Only the decision call ran. No grounded call against an empty
    # retrieval set — exactly the silent-invention path the fix
    # closes.
    assert len(seen_calls) == 1
    assert chatbot_sidecar._SESSION_STORE_PATH is None
    evs = _chatbot_events(capsys)

    # The deterministic refusal reached the user via a chatbot_chunk —
    # the single text the UI renders for this case, model-free.
    chunks = [
        e["delta"] for e in evs
        if e["event"] == "chatbot_chunk" and e.get("delta")]
    assert chunks == [chatbot_turn.NO_CORPUS_REFUSAL_TEXT]

    # chatbot_done carries pure-conversation shape (no resources block,
    # no run pinned — there is no run to pin against). #834: it also
    # carries ``refused: true`` so the UI can mark the persisted turn
    # and exclude its assistant slot from the history fed into the
    # next turn — without that, the deterministic refusal text loops
    # back through the prompt and the model learns to mimic it as
    # prose even after a corpus is bound.
    dones = [e for e in evs if e["event"] == "chatbot_done"]
    assert len(dones) == 1
    assert dones[0]["resources"] is None
    assert dones[0]["run"] is None
    assert dones[0].get("refused") is True

    # The retrieving event is suppressed: the UI never shows
    # "searching…" for a refused-because-unbound turn (no dispatch
    # fired, no hop event was emitted either).
    assert not any(e["event"] == "chatbot_retrieving" for e in evs)
    assert not any(e["event"] == "chatbot_hop" for e in evs)


# ---------------------------------------------------------------------------
# chatbot_turn._resolve_bracket_anchors — bracket → canonical translation
# ---------------------------------------------------------------------------


def _ret_rec(kind: str, record_id: str):
    """Minimal RetrievedRecord stub carrying just the .record.kind /
    .record.record_id attrs the resolver reads."""
    from types import SimpleNamespace
    return SimpleNamespace(record=SimpleNamespace(
        kind=kind, record_id=record_id, text="", file_id=""))


def test_resolve_bracket_anchors_translates_brackets_to_canonical():
    """``has_neighbor: ["[1]", "[3]"]`` → ``["action/4", "fact/work:5"]``
    using this turn's accumulator. The model only ever sees bracket
    indices in CONTEXT (option-A); the loop translates here before
    the validator + dispatcher (which still speak canonical kind/id).
    """
    from engine import chatbot_turn
    acc = [
        _ret_rec("action", "4"),
        _ret_rec("pattern", "relationships:24"),
        _ret_rec("fact", "work:5"),
    ]
    raw = {"tool": "search", "lookups": [
        {"query": "...", "has_neighbor": ["[1]", "[3]"]},
    ]}
    chatbot_turn._resolve_bracket_anchors(raw, acc)
    assert raw["lookups"][0]["has_neighbor"] == ["action/4", "fact/work:5"]


def test_resolve_bracket_anchors_drops_out_of_range_brackets():
    """An out-of-range bracket is silently dropped — the validator's
    existing empty-list path handles a has_neighbor that resolved to
    nothing, and a missing anchor is more honest than a fabricated
    canonical id."""
    from engine import chatbot_turn
    acc = [_ret_rec("action", "4")]
    raw = {"tool": "search", "lookups": [
        {"has_neighbor": ["[1]", "[7]", "[99]"]},
    ]}
    chatbot_turn._resolve_bracket_anchors(raw, acc)
    assert raw["lookups"][0]["has_neighbor"] == ["action/4"]


def test_resolve_bracket_anchors_drops_canonical_form():
    """Mixed input — some brackets, some canonical ``kind/id`` strings
    — drops the canonical-form entries entirely. The model only ever
    sees bracket-index labels in CONTEXT; a canonical-form
    has_neighbor like ``"fact/admin:47"`` is the model drifting to its
    training prior, and rejecting it forces a clean retry on the next
    hop instead of silently dispatching against a record the model
    couldn't have known to pick (its 'guesses' often shotgun-hit a
    real record in a dense topic, but the WRONG record for the
    question). Only successfully-translated brackets survive."""
    from engine import chatbot_turn
    acc = [_ret_rec("pattern", "work:10")]
    raw = {"tool": "search", "lookups": [
        {"has_neighbor": ["[1]", "fact/admin:47"]},
    ]}
    chatbot_turn._resolve_bracket_anchors(raw, acc)
    # Letter form translated; canonical form dropped.
    assert raw["lookups"][0]["has_neighbor"] == ["pattern/work:10"]


def test_resolve_bracket_anchors_handles_empty_accumulator():
    """Decision-turn call (call 1) has an empty accumulator — any
    bracket the model emits there is necessarily out-of-range and
    gets dropped. The lookup ends up with no anchors and the
    dispatcher handles that as an unanchored search."""
    from engine import chatbot_turn
    raw = {"tool": "search", "lookups": [
        {"has_neighbor": ["[1]", "[2]"]},
    ]}
    chatbot_turn._resolve_bracket_anchors(raw, [])
    assert raw["lookups"][0]["has_neighbor"] == []
