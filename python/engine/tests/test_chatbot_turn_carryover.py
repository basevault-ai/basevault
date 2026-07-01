"""
Contract pins for the chatbot citation-surface-protocol slice.

The slice ships five interlocking pieces against the chatbot ReAct
loop's citation surface:

  1. History bracket-strip — covered separately in test_chatbot.py.
  2. Carryover seed at turn start — the sidecar-side walk that
     re-hydrates the most-recent assistant turn's cited records into
     this turn's accumulator at brackets ``[1..K]``.
  3. Forcing-function ``keep`` + ``note`` on every tool call — parsed
     by ``validate_tool_call``; surfaced on per-hop diagnostics here.
  4. Silent put-back of ever-selected positions — the loop tracks
     ``ever_selected`` as the union of every ``keep`` declaration,
     surfaced on per-hop diag so a forward eviction policy has the
     hook ready.
  5. Pre-resolved JSON snapshot in NOTES "Previous attempts" —
     ``raw_call`` is snapshotted BEFORE the bracket resolver mutates
     it, so the model's own sidebar never sees the canonical-id form
     it never wrote.

Plus dev-side capture of silent anchor drops (out-of-range bracket,
canonical-form anchor) on the same per-hop diag entries — model-facing
NOTES surface stays untouched per the trimmed proposal.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

from engine import chatbot
from engine import chatbot_sidecar
from engine import chatbot_turn
from engine.chatbot_turn import (
    CARRYOVER_CAP,
    TurnContext,
    _resolve_bracket_anchors,
)
from engine.llm import CompletionResult
from engine.rag_vector_store import StoredRecord
from engine.retrieval import RetrievedRecord


def _rec(rid: str, kind: str = "fact", text: str = "body") -> StoredRecord:
    return StoredRecord(kind=kind, record_id=rid, text=text)


def _hit(rid: str, kind: str = "fact", text: str = "body") -> RetrievedRecord:
    return RetrievedRecord(
        record=_rec(rid, kind=kind, text=text),
        distance=0.5,
        rerank_score=None,
    )


def _completion(content: str) -> CompletionResult:
    return CompletionResult(
        content=content, call_id=None, cache_key=None, cached=False,
        finish_reason="stop", model="m", mode="tinfoil",
        prompt_tokens=3, completion_tokens=2, reasoning_tokens=0,
        reasoning_tokens_source=None, content_tokens=2, ttft_ms=1,
        ttfr_ms=1, last_token_ms=1, max_tokens_reserved=64,
    )


def _scripted_complete(replies: list[str]):
    """Sequentially returns the next reply on each call. No on_chunk
    fired — the gate's non-streamed fallback covers UI emission, which
    isn't what the slice's tests are measuring."""
    seq = iter(replies)

    def _complete(messages, **kwargs):
        return _completion(next(seq))

    return _complete


def _build_ctx(
    monkeypatch,
    tracked_complete,
    *,
    seed_records: list[RetrievedRecord] | None = None,
    dispatch_returns: list[list[RetrievedRecord]] | None = None,
):
    """Minimal ``TurnContext`` with a bound (fake) store + scripted
    dispatcher. Each entry of ``dispatch_returns`` is the result for the
    n-th tool call's dispatch.
    """
    emit_log: list[tuple] = []

    def _emit(event, **payload):
        emit_log.append((event, payload))

    class _Store:
        def count(self):
            return 1

    @contextlib.contextmanager
    def _fake_open_store(_p):
        yield _Store()

    monkeypatch.setattr(chatbot_turn, "open_store", _fake_open_store)

    seq = iter(dispatch_returns or [])

    def _dispatch(call, **kw):
        try:
            return next(seq)
        except StopIteration:
            return []

    monkeypatch.setattr(chatbot_turn, "dispatch", _dispatch)

    ctx = TurnContext(
        query="q",
        history=[],
        turn_index=1,
        session_id="s",
        store_path=Path("/dev/null/fake-store"),
        bound_run=None,
        chatbot_config={"model": "m", "reasoning": False},
        tracked_complete=tracked_complete,
        emit=_emit,
        seed_records=list(seed_records or []),
    )
    return ctx, emit_log


# ── Anchor drops surface on hop_diag (acceptance #3) ────────────────────


def test_resolve_brackets_records_out_of_range_drop():
    """Out-of-range brackets drop silently from the model's perspective
    but land on the returned drop list with ``reason=out_of_range`` so
    payload dumps can surface the omission to dev humans."""
    acc = [_hit("a"), _hit("b")]  # length 2
    raw_call = {
        "tool": "search",
        "lookups": [{"has_neighbor": ["[1]", "[99]"]}],
    }
    drops = _resolve_bracket_anchors(raw_call, acc)
    # Only the in-range anchor survives in the rewritten lookup.
    rewritten = raw_call["lookups"][0]["has_neighbor"]
    assert rewritten == ["fact/a"]
    # The out-of-range bracket is recorded with the acc size for
    # debugging context.
    reasons = [d["reason"] for d in drops]
    assert "out_of_range" in reasons
    out_of_range = next(d for d in drops if d["reason"] == "out_of_range")
    assert out_of_range["value"] == "[99]"
    assert out_of_range["acc_size"] == 2


def test_resolve_brackets_records_canonical_form_drop():
    """A canonical-form anchor (the Option-A regression Finding 2
    drift: ``"action/5"``, ``"insight/critical:1"``) drops silently
    and surfaces with ``reason=canonical_form`` so the dev-side dump
    captures the drift mechanism for diagnosis."""
    acc = [_hit("a")]
    raw_call = {
        "tool": "search",
        "lookups": [{"has_neighbor": ["[1]", "action/5"]}],
    }
    drops = _resolve_bracket_anchors(raw_call, acc)
    assert raw_call["lookups"][0]["has_neighbor"] == ["fact/a"]
    canonical = [d for d in drops if d["reason"] == "canonical_form"]
    assert len(canonical) == 1
    assert canonical[0]["value"] == "action/5"


def test_anchor_drops_land_on_hop_diag(monkeypatch):
    """Wired-through pin: a tool call carrying an out-of-range bracket
    fires the loop's ``hop_diag['anchor_drops']`` entry so the
    shareable marker + payload dumps include drop reasons. The next
    hop's NOTES are NOT corrected — model-facing surface stays
    silent per the trimmed proposal."""
    seed = [_hit("seed-1"), _hit("seed-2")]
    replies = [
        # Hop 1: out-of-range bracket against the seed (acc=2).
        '{"tool": "search", "plan": "n",'
        ' "lookups": [{"query": "x", "has_neighbor": ["[99]"]}]}',
        # Hop 2 finalize.
        "Done.",
    ]
    ctx, _log = _build_ctx(
        monkeypatch, _scripted_complete(replies),
        seed_records=seed, dispatch_returns=[[]],
    )
    chatbot_turn.run(ctx)
    hop1 = ctx.hops_diag[0]
    assert "anchor_drops" in hop1
    drops = hop1["anchor_drops"]
    assert any(d["reason"] == "out_of_range" for d in drops)


# ── Pre-resolved snapshot in NOTES Previous attempts (acceptance #4) ──


def test_previous_attempts_uses_pre_resolved_snapshot(monkeypatch):
    """The next hop's NOTES bullet renders the model's verbatim
    emission (``has_neighbor: ["[1]"]``), NOT the post-resolution
    canonical-id form. Closes the leak channel where the model's own
    sidebar would otherwise teach it the canonical-id surface it
    isn't supposed to see."""
    captured_prompts: list[list[dict]] = []

    def _complete(messages, **kwargs):
        captured_prompts.append(messages)
        if len(captured_prompts) == 1:
            # Hop 1: the model emits a bracket anchor against the
            # carryover seed. We seed [1]=fact/seed-1 below.
            return _completion(
                '{"tool": "search", "plan": "walking",'
                ' "lookups": [{"has_neighbor": ["[1]"]}]}'
            )
        # Hop 2: finalize so the loop exits.
        return _completion("ok")

    ctx, _log = _build_ctx(
        monkeypatch, _complete,
        seed_records=[_hit("seed-1")], dispatch_returns=[[]],
    )
    chatbot_turn.run(ctx)
    # Second prompt's user message carries the NOTES "Previous
    # attempts" bullet. It must contain the bracket form, not the
    # resolved canonical id.
    hop2_user = captured_prompts[1][-1]["content"]
    assert "Previous attempts" in hop2_user
    assert '["[1]"]' in hop2_user or '"[1]"' in hop2_user
    # The resolver rewrites raw_call IN PLACE to "fact/seed-1", but
    # that string must NOT reach the NOTES.
    assert "fact/seed-1" not in hop2_user


# ── Within-turn bracket immutability ────────────────────────────────────


def test_seeded_carryover_brackets_stay_stable_across_hops(monkeypatch):
    """A record seeded into ``acc[0]`` at turn start is still
    addressable as bracket position [1] after a subsequent hop's
    retrievals union into the accumulator. The acc is monotonic;
    positions never re-bind to a different record."""
    seed = [_hit("seed-1")]
    new_hit = _hit("new-1")
    replies = [
        '{"tool": "search", "plan": "n",'
        ' "lookups": [{"query": "fresh"}]}',
        '{"tool": "search", "plan": "n",'
        ' "lookups": [{"has_neighbor": ["[1]"]}]}',
        "ok",
    ]
    # Capture the resolved canonical ids hop-2 emits so we can verify
    # [1] still resolves to seed-1 (NOT to new-1 even though new-1
    # arrived in hop 1's dispatch).
    resolved: list[list[str]] = []

    def _dispatch(call, **kw):
        # Snapshot the call's first lookup's resolved anchors.
        lookups = call.args.get("lookups") or ()
        if lookups and lookups[0].has_neighbor:
            resolved.append([
                f"{k}/{r}" for (k, r) in lookups[0].has_neighbor
            ])
        # Hop 1's dispatch returns new-1; hop 2's dispatch returns [].
        return [new_hit] if not resolved or resolved[-1] == [] else []

    ctx, _log = _build_ctx(
        monkeypatch, _scripted_complete(replies),
        seed_records=seed,
    )
    # Override the helper's no-op dispatch with the call-aware variant.
    monkeypatch.setattr(chatbot_turn, "dispatch", _dispatch)
    chatbot_turn.run(ctx)
    # Hop 2's has_neighbor: ["[1]"] must resolve to the SEED record
    # (kind=fact, id=seed-1), not the post-hop-1 new arrival. The
    # carryover seed at acc[0] is bracket-immutable for the turn.
    assert any("fact/seed-1" in r for r in resolved if r), resolved


# ── keep + note silent-default to hop_diag (acceptance #5) ─────────────


def test_missing_plan_silent_defaults(monkeypatch):
    """A call that omits ``plan`` silent-defaults to ``""`` and lands
    on ``hop_diag['plan']`` as the empty string — no raise, no
    model-facing surface change."""
    replies = [
        '{"tool": "search", "lookups": [{"query": "x"}]}',
        "ok",
    ]
    ctx, _log = _build_ctx(
        monkeypatch, _scripted_complete(replies),
        dispatch_returns=[[]],
    )
    chatbot_turn.run(ctx)
    hop1 = ctx.hops_diag[0]
    assert hop1["plan"] == ""


# ── Carryover seed at turn start (acceptance #2) ────────────────────────


def test_carryover_refs_walks_history_backward_for_last_known():
    """Acceptance #2 (last-known fallback): when the immediately
    prior assistant turn carries no ``cited_refs`` (pure conversation,
    no lookup), the walk picks the next-most-recent assistant turn
    that grounded — so ``tell me more about [3]`` still resolves
    two turns later, capped at MAX_HISTORY_TURNS."""
    history = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": "with grounding",
            "cited_refs": [
                {"kind": "fact", "record_id": "f-a"},
                {"kind": "pattern", "record_id": "p-b"},
            ],
        },
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "chit-chat reply"},
    ]
    refs = chatbot.carryover_refs(history)
    assert refs == [("fact", "f-a"), ("pattern", "p-b")]


def test_carryover_refs_caps_at_carryover_cap():
    """Acceptance #2 (cap): the walk caps at ``CARRYOVER_CAP``, so a
    prior turn that cited 20 records seeds only the first 10."""
    refs_payload = [
        {"kind": "fact", "record_id": f"f-{i}"} for i in range(20)
    ]
    history = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "many refs",
            "cited_refs": refs_payload,
        },
    ]
    refs = chatbot.carryover_refs(history)
    assert len(refs) == CARRYOVER_CAP


def test_carryover_refs_returns_empty_when_nothing_grounded():
    """Empty when no assistant turn in the window cited anything —
    the new turn's accumulator starts empty and the model has to
    look up to ground."""
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "no refs"},
    ]
    assert chatbot.carryover_refs(history) == []
    assert chatbot.carryover_refs([]) == []


def test_empty_reply_falls_back_to_retry_message(monkeypatch):
    """Every hop returns empty (the kernel already retried transient LOAD
    failures to exhaustion) — the turn must surface EMPTY_REPLY_TEXT instead
    of a blank ``answer``, and emit it so the user actually sees the prompt
    to retry rather than a blank bubble with a "done" status."""
    from engine.chatbot_turn import EMPTY_REPLY_TEXT
    ctx, emit_log = _build_ctx(monkeypatch, _scripted_complete([""]))
    result = chatbot_turn.run(ctx)
    assert result.answer == EMPTY_REPLY_TEXT
    assert not result.lookup_fired
    assert any(
        ev == "chatbot_chunk" and p.get("delta") == EMPTY_REPLY_TEXT
        for ev, p in emit_log
    ), emit_log


def test_seed_records_re_hydrate_in_carryover_order(monkeypatch):
    """The seed-record fetch returns records in the SAME ORDER as the
    requested refs so the seeded accumulator's bracket positions
    ``[1..K]`` line up with the prior turn's cited indices after the
    UI's contiguous renumber."""
    refs = [("fact", "f-x"), ("pattern", "p-y"), ("chunk", "c-z")]

    class _StoreShim:
        def filter_select(self, *, limit, neighbor_ids, **kw):
            # Simulate the real store returning insertion order, NOT
            # request order — the helper must reorder.
            return [
                _rec("c-z", kind="chunk"),
                _rec("f-x", kind="fact"),
                _rec("p-y", kind="pattern"),
            ]

    @contextlib.contextmanager
    def _fake_open_store(_p):
        yield _StoreShim()

    monkeypatch.setattr(chatbot, "open_store", _fake_open_store)
    out = chatbot.seed_records_for(
        Path("/dev/null/store"), refs,
    )
    assert [(r.record.kind, r.record.record_id) for r in out] == refs


# ── Acceptance #8: cross-turn carryover resolution end-to-end ──────────


def test_sidecar_run_seeds_acc_from_history_cited_refs(monkeypatch):
    """Acceptance #8 (sidecar integration): the chat ``_run`` path
    extracts the prior assistant turn's ``cited_refs`` from the
    history payload, fetches the records from the bound store, and
    seeds the loop's accumulator. The first tool call's
    ``has_neighbor: ["[3]"]`` therefore resolves against the prior
    turn's third cited record — closing the "tell me more about [3]"
    cross-turn round-trip without a fresh retrieval."""
    history = [
        {"role": "user", "content": "tell me about the project"},
        {
            "role": "assistant",
            "content": "Found three groundings.",
            "cited_refs": [
                {"kind": "fact", "record_id": "f-alpha"},
                {"kind": "fact", "record_id": "f-beta"},
                {"kind": "pattern", "record_id": "p-gamma"},
            ],
        },
    ]

    fetched_refs: list[set[tuple[str, str]]] = []

    class _StoreShim:
        def count(self):
            return 1

        def filter_select(self, *, limit, neighbor_ids, **kw):
            fetched_refs.append(set(neighbor_ids))
            return [
                _rec("f-alpha", kind="fact"),
                _rec("f-beta", kind="fact"),
                _rec("p-gamma", kind="pattern"),
            ]

    @contextlib.contextmanager
    def _fake_open_store(_p):
        yield _StoreShim()

    # The seed fetch now flows through ``chatbot.seed_records_for`` →
    # ``chatbot.open_store``; the loop's own retrieval still uses
    # ``chatbot_turn.open_store``. Patch both so neither hits a real db.
    monkeypatch.setattr(chatbot, "open_store", _fake_open_store)
    monkeypatch.setattr(chatbot_turn, "open_store", _fake_open_store)
    monkeypatch.setattr(
        chatbot_sidecar, "_SESSION_STORE_PATH", Path("/dev/null/store"),
    )

    # The model on hop 1 walks straight from [3] — the
    # carryover-resolution path the user just typed.
    captured_anchors: list[list[str]] = []

    def _dispatch(call, **kw):
        lookups = call.args.get("lookups") or ()
        if lookups and lookups[0].has_neighbor:
            captured_anchors.append([
                f"{k}/{r}" for (k, r) in lookups[0].has_neighbor
            ])
        return []

    monkeypatch.setattr(chatbot_turn, "dispatch", _dispatch)

    # Stub the model. Chat runs on the kernel (``run_chat_turn`` injects its
    # own kernel-backed ``tracked_complete``), so patching the sidecar's
    # placeholder ``_tracked_complete`` no longer intercepts the call. Patch
    # ``run_chat_turn`` to drive the existing loop with a scripted model —
    # the same seam the other tests use via ``_build_ctx`` — exercising the
    # sidecar → loop carryover wiring without a live kernel dispatch.
    from engine.phases import chat as phases_chat

    def _fake_run_chat_turn(ctx, mode, execution_env=None):
        from dataclasses import replace
        return chatbot_turn.run(
            replace(ctx, tracked_complete=_scripted_complete([
                '{"tool": "search", "plan": "follow-up on [3]",'
                ' "lookups": [{"has_neighbor": ["[3]"]}]}',
                "Following up on the pattern.",
            ]))
        )

    monkeypatch.setattr(phases_chat, "run_chat_turn", _fake_run_chat_turn)
    monkeypatch.setattr(chatbot_sidecar, "_read_app_config", lambda: {})
    monkeypatch.setattr(
        chatbot_sidecar, "resolve_chatbot_from_config",
        lambda cfg: {"model": "m"},
    )
    monkeypatch.setattr(
        chatbot_sidecar, "_resolve_chatbot_mode", lambda cfg: chatbot_sidecar.Mode.TEE,
    )
    emit_log: list[dict] = []
    monkeypatch.setattr(
        chatbot_sidecar, "_emit",
        lambda event, **payload: emit_log.append({"event": event, **payload}),
    )
    # Silence telemetry side effects unrelated to the contract.
    monkeypatch.setattr(chatbot_sidecar, "begin_payloads_yaml_turn", lambda *a, **k: None)
    monkeypatch.setattr(chatbot_sidecar, "flush_payloads_yaml", lambda: None)
    monkeypatch.setattr(
        chatbot_sidecar.shareable_markers, "llm_calls_baseline", lambda p: 0,
    )
    monkeypatch.setattr(
        chatbot_sidecar, "_emit_shareable_markers", lambda **kw: None,
    )

    chatbot_sidecar._run("tell me more about [3]", history)

    # Store was opened during carryover-seed fetch with the three
    # prior cited refs as the neighbor filter.
    assert fetched_refs, "sidecar never opened store for carryover seed"
    assert {("fact", "f-alpha"), ("fact", "f-beta"), ("pattern", "p-gamma")} == fetched_refs[0]

    # The model's first hop's ``[3]`` resolved against the seeded
    # accumulator's position 3 — that's the pattern carryover record,
    # NOT a fresh retrieval.
    assert captured_anchors, "first hop dispatched no anchored lookup"
    assert "pattern/p-gamma" in captured_anchors[0]


def test_carryover_seeds_acc_for_first_tool_call(monkeypatch):
    """Acceptance #8 (cross-turn bracket stability, in-loop pin): the
    sidecar-built ctx seeds the loop's ``acc`` so the model's first
    tool call resolving ``has_neighbor: ["[3]"]`` lands against the
    third carryover record — the user-typed "tell me more about [3]"
    flow without a fresh lookup having fired yet."""
    seed = [
        _hit("turn1-cited-1", kind="fact"),
        _hit("turn1-cited-2", kind="fact"),
        _hit("turn1-cited-3", kind="pattern"),
    ]

    resolved: list[tuple[str, str]] = []

    def _dispatch(call, **kw):
        lookups = call.args.get("lookups") or ()
        if lookups and lookups[0].has_neighbor:
            resolved.extend(lookups[0].has_neighbor)
        return []

    ctx, _log = _build_ctx(
        monkeypatch,
        _scripted_complete([
            # Hop 1: walk from the seeded [3] without a query.
            '{"tool": "search", "plan": "follow-up on [3]",'
            ' "lookups": [{"has_neighbor": ["[3]"]}]}',
            "Done.",
        ]),
        seed_records=seed,
    )
    monkeypatch.setattr(chatbot_turn, "dispatch", _dispatch)
    chatbot_turn.run(ctx)
    # The bracket [3] in the model's emission resolved to the third
    # seeded record (kind=pattern, id=turn1-cited-3) via the carryover
    # — no fresh retrieval needed to make [3] addressable.
    assert ("pattern", "turn1-cited-3") in resolved


# ── #891 symptom B: dead-citation neutralization on lookup-fired turns ──


def _streaming_complete(replies: list[str]):
    """Like ``_scripted_complete`` but drives ``on_chunk`` char-by-char
    so the loop's ``_StreamGate`` advances exactly as a real streaming
    provider would — needed to exercise the streamed branch where a
    dead bracket already reached the bubble and a ``chatbot_replace``
    must correct it."""
    seq = iter(replies)

    def _complete(messages, **kwargs):
        content = next(seq)
        on_chunk = kwargs.get("on_chunk")
        if on_chunk:
            for ch in content:
                on_chunk(ch)
        return _completion(content)

    return _complete


def test_lookup_fired_strips_out_of_range_bracket_nonstreamed(monkeypatch):
    """A grounded finalize that cites an out-of-range ``[N]`` (the model
    grounded nothing at that index) ships with the dead bracket removed
    and the in-range one preserved. Non-streamed path: the cleaned
    answer rides the single-emit fallback into the UI."""
    replies = [
        # Hop 1 (decision): a real tool call → dispatch returns 2 records.
        '{"tool": "search", "plan": "n", "lookups": [{"query": "x"}]}',
        # Hop 2 (grounded finalize): [5] is dead (acc=2), [1] is valid.
        "She is a consultant [5] per your notes [1].",
    ]
    ctx, log = _build_ctx(
        monkeypatch, _scripted_complete(replies),
        dispatch_returns=[[_hit("a"), _hit("b")]],
    )
    result = chatbot_turn.run(ctx)
    assert result.lookup_fired is True
    assert result.answer == "She is a consultant per your notes [1]."
    # The UI got the cleaned text (no dead bracket ever reaches it).
    chunks = [p["delta"] for e, p in log if e == "chatbot_chunk"]
    assert chunks == ["She is a consultant per your notes [1]."]


def test_lookup_fired_replaces_bubble_when_streamed(monkeypatch):
    """Streamed path: the dead ``[5]`` already streamed into the bubble,
    so the loop emits a ``chatbot_replace`` carrying the cleaned answer
    to overwrite it. The in-range ``[1]`` stays clickable."""
    replies = [
        '{"tool": "search", "plan": "n", "lookups": [{"query": "x"}]}',
        "She is a consultant [5] per your notes [1].",
    ]
    ctx, log = _build_ctx(
        monkeypatch, _streaming_complete(replies),
        dispatch_returns=[[_hit("a"), _hit("b")]],
    )
    result = chatbot_turn.run(ctx)
    assert result.answer == "She is a consultant per your notes [1]."
    replaces = [p["text"] for e, p in log if e == "chatbot_replace"]
    assert "She is a consultant per your notes [1]." in replaces


def test_no_lookup_turn_leaves_brackets_untouched(monkeypatch):
    """The neutralization is gated on a fired lookup — a no-lookup
    prose finalize is the other PR's surface (and the persona's
    no-brackets-without-lookup rule), so this path must not strip."""
    replies = [
        # Hop 1 (decision) finalizes directly in prose, no tool call.
        "I think the answer is [9], roughly.",
    ]
    ctx, _log = _build_ctx(monkeypatch, _scripted_complete(replies))
    result = chatbot_turn.run(ctx)
    assert result.lookup_fired is False
    # Untouched — out of this PR's scope.
    assert result.answer == "I think the answer is [9], roughly."
