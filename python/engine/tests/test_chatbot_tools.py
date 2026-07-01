"""
Unit tests for the chatbot's structured tool-call surface.

Covers the two modules behind the structured retrieval contract:

  - ``chatbot_tools`` — parsing a JSON tool call out of a model reply,
    validating it (each lookup's filters, the per-call lookup cap, the
    per-lookup ``count`` clamp), and the audit/describe helpers.
  - ``chatbot_dispatch`` — composing each lookup's parameterized SELECT
    against the store, unioning across the array, applying the cosine
    junk floor and the salience-vs-distance ranking rules.

All offline + deterministic: the dispatch tests use a tiny in-memory
store + stubbed embedding so neither the live model nor the
``~/.basevault`` filesystem is touched.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from engine import chatbot_dispatch
from engine import chatbot_tools as t
from engine.rag_vector_store import (
    COSINE_JUNK_DISTANCE,
    StoredRecord,
    open_store,
)


# ── parse_tool_call ──────────────────────────────────────────────────────────


def test_parse_tool_call_bare_object():
    raw = t.parse_tool_call('{"tool": "search", "query": "my sleep notes"}')
    assert raw == {"tool": "search", "query": "my sleep notes"}


def test_parse_tool_call_conversation_returns_none():
    assert t.parse_tool_call("Sure, let's think that through together.") is None
    assert t.parse_tool_call("") is None


def test_parse_tool_call_tolerates_prose_preamble():
    """The model sometimes prepends a sentence before the call — the
    first top-level JSON object with a ``tool`` key is still found."""
    reply = (
        "I'd be happy to help, but I'll need to look first.\n"
        '{"tool": "search", "query": "budget decisions"}'
    )
    assert t.parse_tool_call(reply) == {
        "tool": "search", "query": "budget decisions"}


def test_parse_tool_call_tolerates_markdown_fence():
    reply = 'Let me check.\n```json\n{"tool": "search", "query": "x"}\n```'
    assert t.parse_tool_call(reply) == {"tool": "search", "query": "x"}


def test_parse_tool_call_braces_inside_query_dont_break_balance():
    """A ``{`` or ``}`` inside the query string must not throw off the
    brace scan."""
    reply = '{"tool": "search", "query": "what about {curly} braces?"}'
    assert t.parse_tool_call(reply) == {
        "tool": "search", "query": "what about {curly} braces?"}


def test_parse_tool_call_first_object_with_tool_key_wins():
    reply = (
        '{"note": "not a tool"}\n'
        '{"tool": "search", "query": "first real call"}\n'
        '{"tool": "search", "query": "second"}'
    )
    assert t.parse_tool_call(reply)["query"] == "first real call"


def test_parse_tool_call_non_tool_json_is_conversation():
    """JSON without a string ``tool`` key is not a tool call."""
    assert t.parse_tool_call('{"answer": "42"}') is None
    assert t.parse_tool_call('{"tool": 7}') is None


def test_parse_tool_call_malformed_json_is_conversation():
    assert t.parse_tool_call("here is a brace { but not json") is None


# ── validate_tool_call: slice-1 shorthand ────────────────────────────────────


def _lookups(call: t.ToolCall) -> tuple[t.Lookup, ...]:
    return call.args["lookups"]


def test_validate_search_minimal_shorthand():
    """The slice-1 single-lookup form (``{tool, query}``) still
    validates and lands as a one-element ``lookups`` list."""
    call = t.validate_tool_call({"tool": "search", "query": "q"})
    assert call.tool == "search"
    [lk] = _lookups(call)
    assert lk.query == "q"
    assert lk.count == t.DEFAULT_K
    assert lk.entry_type == ()
    assert lk.has_neighbor == ()
    assert lk.exact_match == ()


def test_validate_search_kind_alias_becomes_entry_type():
    """Slice-1's singular ``kind`` is accepted as a one-element
    ``entry_type`` list."""
    call = t.validate_tool_call(
        {"tool": "search", "query": "q", "kind": "fact"})
    [lk] = _lookups(call)
    assert lk.entry_type == ("fact",)


def test_validate_search_unknown_kind_raises():
    with pytest.raises(t.ToolCallError):
        t.validate_tool_call({"tool": "search", "query": "q", "kind": "bogus"})


def test_validate_search_source_filter_parsed():
    """The ``source`` (file) filter coerces to a string tuple and rides
    on the validated Lookup, scalar or list form."""
    call = t.validate_tool_call(
        {"tool": "search", "lookups": [{"source": "meditations.txt"}]})
    [lk] = _lookups(call)
    assert lk.source == ("meditations.txt",)
    call2 = t.validate_tool_call(
        {"tool": "search",
         "lookups": [{"source": ["a.txt", "b.txt"], "entry_type": ["document"]}]})
    [lk2] = _lookups(call2)
    assert lk2.source == ("a.txt", "b.txt")
    assert lk2.entry_type == ("document",)


def test_validate_search_source_is_a_standalone_filter():
    """A lookup carrying only ``source`` (no query / other filter) is
    valid — anchoring on a named file is a complete lookup."""
    call = t.validate_tool_call(
        {"tool": "search", "lookups": [{"source": "trip.txt"}]})
    [lk] = _lookups(call)
    assert lk.source == ("trip.txt",) and lk.query is None


def test_validate_search_file_id_alias_for_source():
    """Models reach for ``file_id`` / ``file`` as well as ``source``;
    all three feed the same filter."""
    call = t.validate_tool_call(
        {"tool": "search", "lookups": [{"file_id": "x.txt"}]})
    [lk] = _lookups(call)
    assert lk.source == ("x.txt",)


def test_document_is_a_searchable_kind():
    """`document` is filterable via entry_type the moment it's a record
    kind — SEARCHABLE_KINDS tracks the store's RECORD_KINDS."""
    assert "document" in t.SEARCHABLE_KINDS
    call = t.validate_tool_call(
        {"tool": "search", "query": "q", "entry_type": ["document"]})
    [lk] = _lookups(call)
    assert lk.entry_type == ("document",)


def test_validate_search_k_clamped_to_cap():
    call = t.validate_tool_call({"tool": "search", "query": "q", "k": 999})
    [lk] = _lookups(call)
    assert lk.count == t.MAX_COUNT == 20


def test_validate_search_count_clamped_to_cap():
    """The canonical ``count`` field clamps to the same cap as the
    slice-1 ``k`` alias."""
    call = t.validate_tool_call(
        {"tool": "search", "lookups": [{"query": "q", "count": 9999}]})
    [lk] = _lookups(call)
    assert lk.count == t.MAX_COUNT


def test_validate_search_count_handles_infinity():
    """``json.loads`` accepts the non-standard ``Infinity`` literal as
    ``float('inf')``; ``int(float('inf'))`` raises ``OverflowError``.
    The validator catches it and clamps to the default rather than
    propagating an unrecoverable exception to the sidecar."""
    import json
    raw = json.loads('{"tool": "search", "query": "q", "count": Infinity}')
    call = t.validate_tool_call(raw)
    [lk] = _lookups(call)
    assert lk.count == t.DEFAULT_K


def test_validate_search_k_non_int_falls_back_to_default():
    call = t.validate_tool_call({"tool": "search", "query": "q", "k": "lots"})
    [lk] = _lookups(call)
    assert lk.count == t.DEFAULT_K


def test_validate_search_k_infinity_falls_back_to_default():
    """``int(float('inf'))`` raises ``OverflowError`` — separate from
    the ``ValueError`` path — so a model emitting ``{"k": Infinity}``
    via the slice-1 alias must still land in the malformed-call
    recovery default, not blow up the dispatcher. Covers both
    ``+inf`` and ``-inf``."""
    call = t.validate_tool_call(
        {"tool": "search", "query": "q", "k": float("inf")},
    )
    [lk_pos] = _lookups(call)
    assert lk_pos.count == t.DEFAULT_K
    call_neg = t.validate_tool_call(
        {"tool": "search", "query": "q", "k": float("-inf")},
    )
    [lk_neg] = _lookups(call_neg)
    assert lk_neg.count == t.DEFAULT_K


def test_validate_search_local_caps_override_cloud_defaults():
    """LOCAL mode caps (10/15) override the cloud defaults (15/20) when
    threaded through ``validate_tool_call``. A model emitting ``count=20``
    on LOCAL clamps to 15; an omitted count defaults to 10 (not 15)."""
    call = t.validate_tool_call(
        {"tool": "search", "lookups": [{"query": "q", "count": 20}]},
        default_k=t.LOCAL_DEFAULT_K,
        max_count=t.LOCAL_MAX_COUNT,
    )
    [lk] = _lookups(call)
    assert lk.count == t.LOCAL_MAX_COUNT == 15

    # Omitted count → mode-default, not cloud-default.
    call = t.validate_tool_call(
        {"tool": "search", "lookups": [{"query": "q"}]},
        default_k=t.LOCAL_DEFAULT_K,
        max_count=t.LOCAL_MAX_COUNT,
    )
    [lk] = _lookups(call)
    assert lk.count == t.LOCAL_DEFAULT_K == 10


def test_validate_search_cloud_caps_unchanged():
    """Default caller (no keyword override) keeps the historical 15/20
    cloud behavior — backwards-compatible for every existing call site
    that hasn't been mode-threaded."""
    call = t.validate_tool_call(
        {"tool": "search", "lookups": [{"query": "q", "count": 999}]},
    )
    [lk] = _lookups(call)
    assert lk.count == t.MAX_COUNT == 20

    call = t.validate_tool_call({"tool": "search", "lookups": [{"query": "q"}]})
    [lk] = _lookups(call)
    assert lk.count == t.DEFAULT_K == 15


def test_validate_search_empty_lookup_raises():
    """A lookup with no query and no filters has nothing to select on
    — the dispatcher would return the whole records table. Refused at
    the validator."""
    with pytest.raises(t.ToolCallError):
        t.validate_tool_call({"tool": "search"})
    with pytest.raises(t.ToolCallError):
        t.validate_tool_call({"tool": "search", "query": "   "})
    with pytest.raises(t.ToolCallError):
        t.validate_tool_call({"tool": "search", "lookups": [{"count": 5}]})


def test_validate_unknown_tool_raises():
    with pytest.raises(t.ToolCallError):
        t.validate_tool_call({"tool": "drop_table", "query": "q"})


# ── validate_tool_call: plan protocol field ──────────────────────────────────


def test_validate_plan_passes_through_clean():
    """The model's strategic execution plan — the multi-step walk
    it's executing and where in it the current call sits — rides
    into the next hop's NOTES sidebar via the JSON snapshot, capped
    at ``MAX_PLAN_CHARS``."""
    plan = "step 2 of action→pattern→fact→chunk; walking patterns from [1]"
    call = t.validate_tool_call({
        "tool": "search",
        "plan": plan,
        "lookups": [{"query": "q"}],
    })
    assert call.plan == plan


def test_validate_plan_truncates_to_cap():
    """A plan over ``MAX_PLAN_CHARS`` truncates rather than failing —
    the cap is a budget, not a contract violation."""
    long_plan = "x" * (t.MAX_PLAN_CHARS + 50)
    call = t.validate_tool_call({
        "tool": "search",
        "plan": long_plan,
        "lookups": [{"query": "q"}],
    })
    assert len(call.plan) == t.MAX_PLAN_CHARS


def test_validate_plan_missing_silent_defaults():
    """Missing ``plan`` silent-defaults to ``""`` — the persona asks
    for it; the validator stays soft."""
    call = t.validate_tool_call({
        "tool": "search",
        "lookups": [{"query": "q"}],
    })
    assert call.plan == ""


def test_validate_plan_collapses_multiline():
    """The model sometimes emits multi-line plans (newlines from a
    multi-step write-up). The validator flattens whitespace so the
    plan rides as a single line in the next hop's NOTES sidebar."""
    call = t.validate_tool_call({
        "tool": "search",
        "plan": "step 1: find action\nstep 2: walk to patterns\nnow on step 2",
        "lookups": [{"query": "q"}],
    })
    assert "\n" not in call.plan
    assert call.plan == (
        "step 1: find action step 2: walk to patterns now on step 2"
    )


# ── validate_tool_call: new multi-lookup surface ─────────────────────────────


def test_validate_explicit_lookups_array():
    call = t.validate_tool_call({
        "tool": "search",
        "lookups": [
            {"query": "first"},
            {"query": "second", "entry_type": ["fact", "chunk"], "count": 5},
        ],
    })
    lookups = _lookups(call)
    assert len(lookups) == 2
    assert lookups[0].query == "first"
    assert lookups[1].entry_type == ("fact", "chunk")
    assert lookups[1].count == 5


def test_validate_lookups_single_object_accepted():
    """A lone object under ``lookups`` is normalised to a one-element
    list, so the model can drop the bracket pair when there is only one
    lookup."""
    call = t.validate_tool_call(
        {"tool": "search", "lookups": {"query": "q"}})
    assert len(_lookups(call)) == 1


def test_validate_lookups_exceeding_cap_truncates():
    """A model that bundles more than ``MAX_LOOKUPS`` legs into one
    call gets the first ``MAX_LOOKUPS`` dispatched silently — the
    cap is a soft steer enforced in the handler, not a fault. The
    persona instructs the model to pick its top angles; over-fanout
    drops the tail rather than aborting the whole turn."""
    raw = {
        "tool": "search",
        "lookups": [
            {"query": f"q{i}"} for i in range(t.MAX_LOOKUPS + 4)
        ],
    }
    call = t.validate_tool_call(raw)
    lookups = _lookups(call)
    assert len(lookups) == t.MAX_LOOKUPS
    # The first MAX_LOOKUPS legs survive — by-position truncation, not
    # priority reordering.
    assert [lk.query for lk in lookups] == [
        f"q{i}" for i in range(t.MAX_LOOKUPS)
    ]


def test_validate_filter_only_lookup_is_legal():
    """Filter-only lookups (no query) are part of the slice-2 design —
    the dispatcher handles them via a plain records SELECT, not vector
    KNN. At least one filter must be present."""
    call = t.validate_tool_call({
        "tool": "search",
        "lookups": [{"entry_type": ["action"]}],
    })
    [lk] = _lookups(call)
    assert lk.query is None
    assert lk.entry_type == ("action",)


def test_validate_has_neighbor_parses_kind_id():
    """``has_neighbor`` items are ``kind/record_id`` strings; the
    validator splits on the first ``/`` so a record_id containing more
    slashes (e.g. a chunk's ``file@offset``) survives intact."""
    call = t.validate_tool_call({
        "tool": "search",
        "lookups": [{
            "query": "q",
            "has_neighbor": ["action/3", "chunk/journal.md@4096"],
        }],
    })
    [lk] = _lookups(call)
    assert lk.has_neighbor == (
        ("action", "3"),
        ("chunk", "journal.md@4096"),
    )


def test_validate_has_neighbor_unknown_kind_raises():
    with pytest.raises(t.ToolCallError):
        t.validate_tool_call({
            "tool": "search",
            "lookups": [{"has_neighbor": ["mystery/123"]}],
        })


def test_validate_has_neighbor_missing_separator_raises():
    with pytest.raises(t.ToolCallError):
        t.validate_tool_call({
            "tool": "search",
            "lookups": [{"has_neighbor": ["123"]}],
        })


def test_validate_exact_match_keeps_substrings_verbatim():
    call = t.validate_tool_call({
        "tool": "search",
        "lookups": [{
            "exact_match": ["airport", "50%"],
        }],
    })
    [lk] = _lookups(call)
    assert lk.exact_match == ("airport", "50%")


def test_validate_entry_type_unknown_raises():
    with pytest.raises(t.ToolCallError):
        t.validate_tool_call({
            "tool": "search",
            "lookups": [{"entry_type": ["chunk", "bogus"]}],
        })


# ── describe / audit_record ──────────────────────────────────────────────────


def test_describe_search_is_the_first_query():
    call = t.validate_tool_call({"tool": "search", "query": "sleep"})
    assert t.describe(call) == "sleep"


def test_describe_search_filter_only_falls_back_to_filter_summary():
    call = t.validate_tool_call({
        "tool": "search",
        "lookups": [{"entry_type": ["action"], "exact_match": ["budget"]}],
    })
    desc = t.describe(call)
    assert "action" in desc and "budget" in desc


def test_describe_search_query_with_filters_shows_full_signature():
    """Codex-flagged UI gap: with a query present the prior describe
    returned just the query, hiding entry_type / exact_match /
    has_neighbor restrictions from the "searching your data…" status.
    The user couldn't tell when the model actually narrowed by kind.
    Surface every filter in the bracket suffix."""
    call = t.validate_tool_call({
        "tool": "search",
        "lookups": [{
            "query": "What was decided?",
            "entry_type": ["fact", "entity"],
            "exact_match": ["airport"],
            "has_neighbor": ["pattern/topic:0"],
        }],
    })
    desc = t.describe(call)
    assert desc.startswith("What was decided?")
    assert "fact|entity" in desc
    assert "matching 'airport'" in desc
    assert "neighbors of 1 entry" in desc


def test_describe_search_multi_lookup_one_line_per_lookup():
    """A multi-lookup call emits one line per lookup so the UI shows
    each branch of the union, not just the first."""
    call = t.validate_tool_call({
        "tool": "search",
        "lookups": [
            {"query": "tight"},
            {"query": "broad", "entry_type": ["chunk"]},
        ],
    })
    lines = t.describe(call).splitlines()
    assert len(lines) == 2
    assert lines[0] == "tight"
    assert "broad" in lines[1] and "chunk" in lines[1]


def test_audit_record_carries_each_lookup():
    call = t.validate_tool_call({
        "tool": "search",
        "lookups": [
            {"query": "q1"},
            {"has_neighbor": ["action/0"], "entry_type": ["pattern"]},
        ],
    })
    rec = t.audit_record(call, result_count=4)
    assert rec["tool"] == "search"
    assert rec["result_count"] == 4
    lookups = rec["args"]["lookups"]
    assert lookups[0]["query"] == "q1"
    assert lookups[1]["has_neighbor"] == ["action/0"]
    assert lookups[1]["entry_type"] == ["pattern"]


# ── Dispatch: backing store + embedder fakes ─────────────────────────────────


@dataclass
class _EmbedStub:
    """Replace ``embed_texts()`` in the dispatcher with a callable that maps
    strings to fixed pre-computed unit-norm vectors. Pre-computed so
    distance assertions are reproducible across runs. Accepts (and ignores)
    the ``mode`` arg the kernel embed seam now carries."""
    vectors: dict[str, list[float]]

    def __call__(self, queries, mode=None):
        return [self.vectors[q] for q in queries]


def _unit(*comps: float) -> list[float]:
    """Pad to the embedding dim with zeros and unit-normalise. Tiny
    helper so per-test fixtures stay readable."""
    import math
    from engine.rag_vector_store import EMBEDDING_DIM
    vec = list(comps) + [0.0] * (EMBEDDING_DIM - len(comps))
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _seed_store(store, records: list[tuple[StoredRecord, list[float]]]):
    """Insert ``records`` (record + matching vector) into ``store``."""
    rs = [r for r, _ in records]
    vs = [v for _, v in records]
    store.add(rs, vs)


def _seed_edges(store, edges: list[tuple[str, str, str, str, str]]):
    """Insert directed edges by 5-tuple. Stage-3 only reads; the
    dispatcher's #785-shaped consumer can be exercised end-to-end this
    way without waiting on the embed-stage producer."""
    store.conn.executemany(
        "INSERT OR IGNORE INTO edges(src_kind, src_id, dst_kind, dst_id, edge_kind) "
        "VALUES (?, ?, ?, ?, ?)",
        edges,
    )
    store.conn.commit()


def _make_record(kind: str, rid: str, text: str, **extra) -> StoredRecord:
    return StoredRecord(
        kind=kind, record_id=rid, text=text,
        file_id=extra.pop("file_id", ""),
        source_ref=extra.pop("source_ref", ""),
        section_path=extra.pop("section_path", ""),
        topic=extra.pop("topic", ""),
        char_offset=extra.pop("char_offset", 0),
        extra=extra,
    )


@pytest.fixture
def stub_embed(monkeypatch):
    """Default embed-stub that knows the query strings used across the
    dispatch tests. Tests that need an extra query register it via
    ``stub_embed.add``."""
    vectors = {
        "qA": _unit(1.0, 0.0),
        "qB": _unit(0.0, 1.0),
    }
    stub = _EmbedStub(vectors)
    monkeypatch.setattr(chatbot_dispatch, "embed_texts", stub)

    def _add(query: str, vec: list[float]) -> None:
        vectors[query] = vec

    stub.add = _add  # type: ignore[attr-defined]
    return stub


# ── Dispatch: single-lookup query (slice-1 parity) ───────────────────────────


def test_dispatch_search_query_only(tmp_path, stub_embed):
    """A one-lookup query call returns ascending-distance hits, junk
    floor honoured."""
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record("fact", "topic:0", "near A"), _unit(0.99, 0.01)),
            (_make_record("fact", "topic:1", "mid A"), _unit(0.7, 0.7)),
            # cosine_distance(_unit(1,0), _unit(-1,0)) = 2.0 > 0.5 → junk
            (_make_record("fact", "topic:2", "opposite"), _unit(-1.0, 0.0)),
        ])
        call = t.validate_tool_call({"tool": "search", "query": "qA"})
        out = chatbot_dispatch.dispatch(call, store=s)
    rids = [r.record.record_id for r in out]
    assert rids[0] == "topic:0"
    assert "topic:2" not in rids  # junk floor dropped it


# ── Dispatch: each axis solo ─────────────────────────────────────────────────


def test_dispatch_entry_type_only_solo(tmp_path, stub_embed):
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record("fact", "topic:0", "fact body"), _unit(1.0, 0.0)),
            (_make_record(
                "pattern", "topic:0", "pattern body", mention_count=3,
            ), _unit(0.0, 1.0)),
            (_make_record(
                "action", "0", "action body",
            ), _unit(0.5, 0.5)),
        ])
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [{"entry_type": ["pattern"]}],
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    assert [r.record.kind for r in out] == ["pattern"]


def test_dispatch_exact_match_solo_is_case_insensitive(tmp_path, stub_embed):
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record(
                "chunk", "f@0", "Notes about the AIRPORT delay",
            ), _unit(1.0, 0.0)),
            (_make_record(
                "chunk", "f@500", "Notes about coffee", file_id="f",
            ), _unit(0.0, 1.0)),
        ])
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [{"exact_match": ["airport"]}],
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    assert [r.record.record_id for r in out] == ["f@0"]


def test_dispatch_exact_match_escapes_wildcards(tmp_path, stub_embed):
    """A ``%`` in the model's substring must match literally, not as
    the SQL wildcard."""
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record(
                "chunk", "f@0", "We need 50% confidence on this",
            ), _unit(1.0, 0.0)),
            (_make_record(
                "chunk", "f@500", "Plain coffee",
            ), _unit(0.0, 1.0)),
        ])
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [{"exact_match": ["50%"]}],
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    assert [r.record.record_id for r in out] == ["f@0"]


def test_dispatch_has_neighbor_solo(tmp_path, stub_embed):
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record("pattern", "p:0", "pattern A"), _unit(1.0, 0.0)),
            (_make_record("pattern", "p:1", "pattern B"), _unit(0.0, 1.0)),
            (_make_record("fact", "f:0", "fact"), _unit(0.5, 0.5)),
        ])
        _seed_edges(s, [
            ("action", "0", "pattern", "p:0", "derivation"),
            ("action", "0", "pattern", "p:1", "derivation"),
        ])
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [{"has_neighbor": ["action/0"]}],
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    rids = sorted(r.record.record_id for r in out)
    assert rids == ["p:0", "p:1"]


# ── Dispatch: AND-composition within one lookup ──────────────────────────────


def test_dispatch_and_compose_query_plus_entry_type(tmp_path, stub_embed):
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record("fact", "topic:0", "near A"), _unit(0.99, 0.01)),
            (_make_record("pattern", "topic:0", "also near A"),
             _unit(0.99, 0.01)),
        ])
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [{"query": "qA", "entry_type": ["fact"]}],
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    assert [r.record.kind for r in out] == ["fact"]


def test_dispatch_and_compose_has_neighbor_plus_entry_type(
    tmp_path, stub_embed,
):
    """The slice-2 marquee composition: walk one hop with
    ``has_neighbor`` and restrict the result to one kind."""
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record("pattern", "p:0", "pattern"), _unit(1.0, 0.0)),
            (_make_record("fact", "f:0", "fact"), _unit(0.5, 0.5)),
        ])
        _seed_edges(s, [
            ("action", "0", "pattern", "p:0", "derivation"),
            ("action", "0", "fact", "f:0", "derivation"),
        ])
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [{
                "has_neighbor": ["action/0"],
                "entry_type": ["pattern"],
            }],
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    assert [r.record.record_id for r in out] == ["p:0"]


# ── Dispatch: multi-lookup union ─────────────────────────────────────────────


def test_dispatch_union_dedupes_by_kind_record_id(tmp_path, stub_embed):
    """An entry returned by two lookups appears once in the union."""
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record("fact", "topic:0", "both"), _unit(0.99, 0.01)),
            (_make_record("fact", "topic:1", "qA only"), _unit(0.95, 0.05)),
            (_make_record("fact", "topic:2", "qB only"), _unit(0.05, 0.95)),
        ])
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [{"query": "qA"}, {"query": "qB"}],
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    rids = [r.record.record_id for r in out]
    assert sorted(set(rids)) == sorted(rids)  # no dupes
    assert "topic:0" in rids and "topic:1" in rids and "topic:2" in rids


def test_dispatch_union_mixed_query_and_filter_only_tiers(
    tmp_path, stub_embed,
):
    """Mixed array: query lookup's hits land first by cosine distance;
    filter-only lookup's exclusive hits land in the salience tier
    after them."""
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record("fact", "qA:0", "near A"), _unit(0.99, 0.01)),
            (_make_record("action", "5", "filter-only action"),
             _unit(0.0, 0.0, 1.0)),
        ])
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [
                {"query": "qA", "entry_type": ["fact"]},
                {"entry_type": ["action"]},
            ],
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    kinds = [r.record.kind for r in out]
    # Query-bearing entries (fact via qA) come before filter-only
    # entries (action), independent of stage rank.
    assert kinds.index("fact") < kinds.index("action")


def test_dispatch_union_caps_at_max_total_results(tmp_path, stub_embed):
    """The post-union trim fires at ``MAX_TOTAL_RESULTS`` (50)
    regardless of how many lookups asked for."""
    with open_store(tmp_path / "vectors.db") as s:
        records = []
        for i in range(70):
            records.append((
                _make_record("fact", f"topic:{i}", f"row {i}"),
                _unit(1.0, 0.0),
            ))
        _seed_store(s, records)
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [{"query": "qA", "count": 20}] * 5,
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    assert len(out) <= t.MAX_TOTAL_RESULTS


# ── Dispatch: filter-only ranking (stage + salience) ─────────────────────────


def test_dispatch_filter_only_orders_by_stage_rank(tmp_path, stub_embed):
    """Without any query, the union is ordered by stage rank then
    salience (no cosine distance to lean on)."""
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record("chunk", "f@0", "chunk text", file_date="2024-01-01"),
             _unit(1.0, 0.0)),
            (_make_record("fact", "topic:0", "fact text", file_date="2024-01-01"),
             _unit(0.0, 1.0)),
            (_make_record(
                "pattern", "topic:0", "pattern", mention_count=5,
            ), _unit(0.5, 0.5)),
            (_make_record("action", "0", "action"), _unit(0.5, 0.5, 0.5)),
        ])
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [{"entry_type": ["action", "pattern", "fact", "chunk"]}],
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    kinds = [r.record.kind for r in out]
    # action < pattern < fact < chunk per the addendum's stage rank.
    assert kinds == ["action", "pattern", "fact", "chunk"]


def test_dispatch_filter_only_action_salience_by_index(tmp_path, stub_embed):
    """Two actions with no query → ordered by positional index ascending
    (action 0 outranks action 3)."""
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record("action", "3", "later"), _unit(1.0, 0.0)),
            (_make_record("action", "0", "earlier"), _unit(0.0, 1.0)),
        ])
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [{"entry_type": ["action"]}],
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    assert [r.record.record_id for r in out] == ["0", "3"]


def test_dispatch_filter_only_pattern_salience_by_mention_count(
    tmp_path, stub_embed,
):
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record(
                "pattern", "topic:a", "few", mention_count=1,
            ), _unit(1.0, 0.0)),
            (_make_record(
                "pattern", "topic:b", "many", mention_count=10,
            ), _unit(0.0, 1.0)),
        ])
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [{"entry_type": ["pattern"]}],
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    assert [r.record.record_id for r in out] == ["topic:b", "topic:a"]


def test_dispatch_filter_only_chunk_salience_by_recency(tmp_path, stub_embed):
    """Without a query, chunks rank by ``(file_date, char_offset)`` desc."""
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record(
                "chunk", "old@0", "older", file_date="2023-05-01",
                char_offset=0,
            ), _unit(1.0, 0.0)),
            (_make_record(
                "chunk", "new@0", "newer", file_date="2024-08-01",
                char_offset=0,
            ), _unit(0.0, 1.0)),
        ])
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [{"entry_type": ["chunk"]}],
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    assert [r.record.record_id for r in out] == ["new@0", "old@0"]


# ── Dispatch: cosine junk floor ──────────────────────────────────────────────


def test_dispatch_drops_above_cosine_junk_floor(tmp_path, stub_embed):
    """An entry whose distance > ``COSINE_JUNK_DISTANCE`` is dropped
    even if it was the only hit; no junk slips into the grounded turn."""
    assert COSINE_JUNK_DISTANCE == 0.5  # pin the constant we calibrated against
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record("fact", "topic:0", "near"), _unit(0.99, 0.01)),
            (_make_record("fact", "topic:1", "far"), _unit(-1.0, 0.0)),
        ])
        call = t.validate_tool_call({"tool": "search", "query": "qA"})
        out = chatbot_dispatch.dispatch(call, store=s)
    assert [r.record.record_id for r in out] == ["topic:0"]


# ── Slice-3 marquee chain: action → pattern → fact → chunk ───────────────────


def test_dispatch_action_to_chunk_chain_via_has_neighbor(tmp_path, stub_embed):
    """The slice-3 acceptance criterion realised at the filter layer:
    three chained ``has_neighbor`` + ``entry_type`` lookups walk
    action → pattern → fact → chunk against a synthetic corpus +
    edge set. The model is expected to issue each hop in stage 4's
    ReAct loop, but stage 3 must already dispatch every hop correctly
    in single-shot mode."""
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record("action", "0", "deliberate"), _unit(1.0, 0.0, 0.0)),
            (_make_record("pattern", "topic:0", "pattern body"),
             _unit(0.0, 1.0, 0.0)),
            (_make_record("fact", "topic:0", "fact body",
                          file_date="2024-01-01"),
             _unit(0.0, 0.0, 1.0)),
            (_make_record("chunk", "journal.md@2048", "chunk body",
                          file_id="journal.md", file_date="2024-01-01"),
             _unit(0.5, 0.5, 0.5)),
        ])
        _seed_edges(s, [
            ("action", "0", "pattern", "topic:0", "derivation"),
            ("pattern", "topic:0", "fact", "topic:0", "derivation"),
            ("fact", "topic:0", "chunk", "journal.md@2048", "evidence"),
        ])
        # Hop 1: action → patterns.
        call1 = t.validate_tool_call({
            "tool": "search",
            "lookups": [{"has_neighbor": ["action/0"], "entry_type": ["pattern"]}],
        })
        hop1 = chatbot_dispatch.dispatch(call1, store=s)
        assert [r.record.record_id for r in hop1] == ["topic:0"]

        # Hop 2: pattern → facts.
        call2 = t.validate_tool_call({
            "tool": "search",
            "lookups": [{
                "has_neighbor": ["pattern/topic:0"],
                "entry_type": ["fact"],
            }],
        })
        hop2 = chatbot_dispatch.dispatch(call2, store=s)
        assert [r.record.record_id for r in hop2] == ["topic:0"]

        # Hop 3: fact → original (OG) chunk. The marquee acceptance:
        # the ``has_neighbor`` walk lands exactly on the source chunk,
        # not by lexical similarity.
        call3 = t.validate_tool_call({
            "tool": "search",
            "lookups": [{
                "has_neighbor": ["fact/topic:0"],
                "entry_type": ["chunk"],
            }],
        })
        hop3 = chatbot_dispatch.dispatch(call3, store=s)
        assert [r.record.record_id for r in hop3] == ["journal.md@2048"]


# ── Tool registry surface ────────────────────────────────────────────────────


def test_tools_registry_advertises_every_filter_field():
    """The system prompt's tool block is rendered from this registry;
    the contract is one params entry per field the validator accepts."""
    params = t.TOOLS["search"]["params"]
    for field in (
        "lookups", "query", "has_neighbor",
        "exact_match", "entry_type", "count",
    ):
        assert field in params


# ── Empty / no-op corners ────────────────────────────────────────────────────


def test_dispatch_empty_store_short_circuits(tmp_path, stub_embed):
    with open_store(tmp_path / "vectors.db") as s:
        call = t.validate_tool_call({"tool": "search", "query": "qA"})
        out = chatbot_dispatch.dispatch(call, store=s)
    assert out == []


def test_dispatch_has_neighbor_with_no_edges_returns_empty(
    tmp_path, stub_embed,
):
    """An anchor that the edges table has never heard of returns no
    rows — never silently falls back to ranked-by-vector hits, which
    would be misleading."""
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record("pattern", "p:0", "pattern"), _unit(1.0, 0.0)),
        ])
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [{"has_neighbor": ["action/99"], "entry_type": ["pattern"]}],
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    assert out == []


# ── Codex P1 / P2 fixes: silent-truncation + degenerate-query guards ────────


def test_filter_select_neighbor_survives_past_limit(tmp_path):
    """The bug Codex P1 caught: ``filter_select`` must enforce the
    neighbor restriction in SQL before applying ``LIMIT``. Without
    that, a records-table scan whose first ``LIMIT`` rows are all
    outside the neighbor pool silently drops the valid neighbor that
    sits further along — looks like "no results" even when the
    answer exists.

    Fixture: 10 non-neighbor pattern rows followed by 1 neighbor
    pattern row in records-table insertion order; the neighbor pool
    of size 1. With LIMIT=5 applied pre-filter, the Python
    post-filter would drop all 5 fetched rows → empty result. The
    fixed code pushes the neighbor predicate into SQL and returns the
    one matching row.
    """
    from engine.rag_vector_store import open_store
    with open_store(tmp_path / "vectors.db") as s:
        records = []
        for i in range(10):
            records.append((
                _make_record("pattern", f"noise:{i}", "noise"),
                _unit(1.0, 0.0),
            ))
        # The single neighbor lands last in insertion order, well past
        # the validator-clamped ``count`` cap a small lookup would use.
        records.append((
            _make_record("pattern", "target:0", "the one"),
            _unit(0.0, 1.0),
        ))
        _seed_store(s, records)
        # An anchor whose only neighbor is the last-inserted row.
        _seed_edges(s, [
            ("action", "0", "pattern", "target:0", "derivation"),
        ])
        neighbor_ids = s.neighbors_of([("action", "0")])
        assert neighbor_ids == {("pattern", "target:0")}
        # ``limit=5`` — pre-fix would scan the first 5 noise rows,
        # post-filter them all out, and return []. Post-fix the SQL
        # has the row-value IN clause and the one neighbor lands.
        rows = s.filter_select(
            limit=5,
            kinds=("pattern",),
            neighbor_ids=neighbor_ids,
        )
    assert [r.record_id for r in rows] == ["target:0"]


def test_query_filtered_neighbor_survives_past_top_k(tmp_path, stub_embed):
    """Same class of bug on the query-bearing path: ``query_filtered``
    must restrict by ``rowid IN (...)`` in the vec0 MATCH clause when
    a neighbor pool is supplied. Pre-fix, the KNN's top-200 could
    contain zero neighbor rows even when one exists below — the
    Python post-filter would silently drop everything and return []."""
    from engine.rag_vector_store import open_store
    with open_store(tmp_path / "vectors.db") as s:
        records = []
        # 50 patterns close to qA in embedding space — would dominate
        # any KNN top-k that doesn't restrict by neighbor pool.
        for i in range(50):
            records.append((
                _make_record("pattern", f"noise:{i}", "noise"),
                _unit(0.99, 0.01),
            ))
        # The one neighbor is FAR from qA — distance ~1.0 vs ~0 for
        # the noise rows. Pre-fix, the noise rows would crowd the
        # top-k and the neighbor wouldn't survive any reasonable
        # over-fetch + post-filter.
        records.append((
            _make_record("pattern", "target:0", "the one"),
            _unit(0.5, 0.5),  # cosine distance from qA ~ 1 - 0.5/sqrt(0.5) ≈ 0.29
        ))
        _seed_store(s, records)
        _seed_edges(s, [
            ("action", "0", "pattern", "target:0", "derivation"),
        ])
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [{
                "query": "qA",
                "has_neighbor": ["action/0"],
                "entry_type": ["pattern"],
            }],
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    assert [r.record.record_id for r in out] == ["target:0"]


def test_dispatch_drops_degenerate_query_vector(tmp_path, stub_embed):
    """A zero / constant / non-finite query vector ties every KNN
    distance to a constant, so ``ORDER BY distance`` falls back to
    insertion order. The dispatcher fails closed (drops that lookup's
    contribution) rather than returning insertion-order rows dressed
    as relevance."""
    stub_embed.add("zero", _unit())  # _unit() on empty input → padded zeros
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record("fact", "topic:0", "row 0"), _unit(0.99, 0.01)),
            (_make_record("fact", "topic:1", "row 1"), _unit(0.99, 0.01)),
        ])
        call = t.validate_tool_call({"tool": "search", "query": "zero"})
        out = chatbot_dispatch.dispatch(call, store=s)
    assert out == []


def test_dispatch_drops_when_all_distances_tied(tmp_path, stub_embed):
    """A non-discriminating corpus (every stored vector equal) ties
    every top-k distance even on a healthy query. The result has no
    ranking signal — insertion order would be served as "most
    relevant." Fail closed instead."""
    # Every stored vector identical → every cosine distance to a
    # well-formed query is the same constant.
    constant = _unit(1.0, 0.0)
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record("fact", "topic:0", "row 0"), list(constant)),
            (_make_record("fact", "topic:1", "row 1"), list(constant)),
            (_make_record("fact", "topic:2", "row 2"), list(constant)),
        ])
        call = t.validate_tool_call({"tool": "search", "query": "qA"})
        out = chatbot_dispatch.dispatch(call, store=s)
    assert out == []


def test_dispatch_multi_kind_entry_type_returns_both(tmp_path, stub_embed):
    """Live-chat field case: user asks for "just facts and entities"
    in one turn. Pre-slice-2 the only available shape was a single
    ``kind`` — the model had to pick one (e.g. ``kind: "fact"``),
    returned 15 facts + 0 entities, and the grounded answer was
    incomplete. With the list-form ``entry_type`` the dispatcher's
    ``kind IN (?, ?)`` returns rows of every requested kind in one
    lookup. Pinned by name because this is the marquee win of the
    slice-2 lookup surface."""
    with open_store(tmp_path / "vectors.db") as s:
        _seed_store(s, [
            (_make_record("fact", "topic:0", "fact body"), _unit(0.99, 0.01)),
            (_make_record("entity", "ent_a", "entity body"), _unit(0.95, 0.05)),
            (_make_record("pattern", "topic:0", "pattern body"),
             _unit(0.9, 0.1)),
            (_make_record("chunk", "f@0", "chunk body"), _unit(0.85, 0.15)),
        ])
        call = t.validate_tool_call({
            "tool": "search",
            "lookups": [{
                "query": "qA",
                "entry_type": ["fact", "entity"],
            }],
        })
        out = chatbot_dispatch.dispatch(call, store=s)
    kinds = {r.record.kind for r in out}
    assert kinds == {"fact", "entity"}
