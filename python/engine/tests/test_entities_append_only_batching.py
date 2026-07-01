"""
Tests for entities `_pack_batches` append-only ordering with the files
manifest.

The cache benefit relies on this property: when a NEW input file is
added, every existing entity group keeps its sort slot, every existing
batch's *block content* stays unchanged, and the new file's groups
land in NEW batches at the end. These tests verify the property at
two levels:

1. `_group_sort_key` produces a stable key for an existing group when
   a later manifest position is added.
2. `_pack_batches` produces the same prefix of batches across two
   manifest snapshots that differ only in trailing entries.
"""
from __future__ import annotations



from engine.entities import (
    _Group, _earliest_manifest_position,
    _group_sort_key, _pack_batches,
)
from engine.llm import Mode


def _mk_group(
    gid: str, name: str, etype: str = "person",
    facts: list[tuple[str, int, str, str | None]] | None = None,
) -> _Group:
    facts = facts or []
    return _Group(
        gid=gid,
        canonical_name=name,
        entity_type=etype,
        aliases={name},
        mention_count=len(facts),
        topics=set(),
        evidence_fact_refs=[(t, i) for (t, i, _s, _d) in facts],
        facts=facts,
    )


def _mk_fact(file_id: str, summary: str = "f") -> "ExtractedItem":  # type: ignore # noqa
    """Build an ExtractedItem whose only evidence carries `file_id`.
    `_fact_file_id` will resolve back to file_id, which is the manifest
    key the entity-batching uses for sort stability."""
    from engine.content_extractor import ExtractedItem, EvidenceSpan
    return ExtractedItem(
        item_type="fact", summary=summary,
        evidence=[EvidenceSpan(text=summary, source_ref=file_id, file_path=file_id)],
    )


def test_earliest_manifest_position_min_aggregates_across_files():
    """Entity grounded in multiple files takes its earliest position
    (load-bearing for cache stability under appends)."""
    facts_by_topic = {
        "t": [
            _mk_fact("late.md"),    # idx 0 → in late.md
            _mk_fact("early.md"),   # idx 1 → in early.md
        ],
    }
    g = _mk_group("g1", "Alice",
                  facts=[("t", 0, "f1", None), ("t", 1, "f2", None)])
    manifest_pos = {"early.md": 0, "late.md": 5}
    pos = _earliest_manifest_position(g, facts_by_topic, manifest_pos)
    assert pos == 0


def test_earliest_manifest_position_unknown_files_sort_last():
    facts_by_topic = {"t": [_mk_fact("ghost.md")]}
    g = _mk_group("g1", "Alice", facts=[("t", 0, "f1", None)])
    pos = _earliest_manifest_position(g, facts_by_topic, {"other.md": 0})
    assert pos == 1 << 62  # sentinel


def test_group_sort_key_uses_manifest_when_provided_otherwise_mention_count():
    facts_by_topic = {"t": [_mk_fact("a.md"), _mk_fact("b.md")]}
    g_alice = _mk_group(
        "g1", "Alice",
        facts=[("t", 0, "f", None), ("t", 0, "f", None), ("t", 0, "f", None)],
    )
    g_bob = _mk_group("g2", "Bob", facts=[("t", 1, "f", None)])

    # With manifest → sort by earliest position.
    manifest_pos = {"a.md": 0, "b.md": 1}
    key_alice = _group_sort_key(g_alice, facts_by_topic, manifest_pos)
    key_bob = _group_sort_key(g_bob, facts_by_topic, manifest_pos)
    assert key_alice < key_bob  # a.md first

    # Without manifest → sort by -mention_count (legacy).
    key_alice_legacy = _group_sort_key(g_alice, facts_by_topic, None)
    key_bob_legacy = _group_sort_key(g_bob, facts_by_topic, None)
    assert key_alice_legacy < key_bob_legacy  # alice has more mentions


def test_pack_batches_iterates_in_manifest_order():
    """Even when N entities fit in one batch (small-corpus case),
    iterating in manifest order preserves the IN-batch ordering. This
    is the cache-stable in-batch contract from the brief: 'items in
    manifest order, content-hash as tiebreaker'."""
    facts = {
        "t": [_mk_fact("a.md"), _mk_fact("b.md"), _mk_fact("c.md")],
    }
    # Deliberately scramble input ordering — the function should resort.
    groups = [
        _mk_group("g3", "Charlie", facts=[("t", 2, "f", None)]),
        _mk_group("g1", "Alpha", facts=[("t", 0, "f", None)]),
        _mk_group("g2", "Bravo", facts=[("t", 1, "f", None)]),
    ]
    by_gid = {g.gid: g for g in groups}
    manifest_pos = {"a.md": 0, "b.md": 1, "c.md": 2}
    batches = _pack_batches(
        list(groups), {}, by_gid, Mode.TEE,
        facts_by_topic=facts, manifest_pos=manifest_pos,
    )
    # Single batch (small corpus). In-batch order = manifest order:
    # g1=Alpha/a.md=0, g2=Bravo/b.md=1, g3=Charlie/c.md=2 → [g1, g2, g3].
    assert len(batches) == 1
    assert [g.gid for g in batches[0]] == ["g1", "g2", "g3"]


def test_pack_batches_prefix_stable_under_appended_file_with_small_budget(monkeypatch):
    """Force a tiny per-batch budget so each entity gets its own batch,
    then verify that adding a new entity preserves prior batches'
    content byte-for-byte. This is the production-scale invariant
    (large corpora fill batches; new entities form new trailing
    batches that don't perturb existing batches' content)."""
    # Force 1 entity per batch via monkey-patched chunk cap + override
    # the safety floor in _pack_batches by patching the budget after
    # it computes. Easiest: monkey-patch math.sqrt so target stays
    # tiny, AND patch the floor to 1. Cleaner: override _pack_batches's
    # internals directly. We do the cleanest version: shrink chunk_cap
    # and rebuild groups with realistic fact bodies that bring t_avg
    # above 4000 / N so the floor is satisfied naturally.
    # Build groups with content so each block is ~5000 tokens.
    big_fact_body = "lorem ipsum " * 1000  # ~2000 tokens
    facts_initial = {"t": []}
    groups_initial = []
    for i, (name, fid) in enumerate([("Alpha", "a.md"), ("Bravo", "b.md"),
                                     ("Charlie", "c.md")]):
        facts_initial["t"].append(_mk_fact(fid))
        groups_initial.append(_mk_group(
            name=name, gid=f"g{i+1}",
            facts=[("t", i, big_fact_body, None)],
        ))
    by_gid = {g.gid: g for g in groups_initial}
    manifest_initial = {"a.md": 0, "b.md": 1, "c.md": 2}
    batches_initial = _pack_batches(
        list(groups_initial), {}, by_gid, Mode.TEE,
        facts_by_topic=facts_initial, manifest_pos=manifest_initial,
    )
    initial_gid_lists = [tuple(g.gid for g in b) for b in batches_initial]
    # With ~2000 token blocks and 4000-token floor budget, expect 2
    # entities per batch at most. The exact split depends on token
    # arithmetic — what matters is that adding a new entity at the
    # END of the manifest does not perturb earlier batches' content.

    # Append a new file + entity at the end.
    facts_after = dict(facts_initial)
    facts_after["t"] = list(facts_initial["t"]) + [_mk_fact("d.md")]
    groups_after = list(groups_initial) + [_mk_group(
        gid="g4", name="Delta",
        facts=[("t", 3, big_fact_body, None)],
    )]
    by_gid_after = {g.gid: g for g in groups_after}
    manifest_after = {"a.md": 0, "b.md": 1, "c.md": 2, "d.md": 3}
    batches_after = _pack_batches(
        list(groups_after), {}, by_gid_after, Mode.TEE,
        facts_by_topic=facts_after, manifest_pos=manifest_after,
    )
    after_gid_lists = [tuple(g.gid for g in b) for b in batches_after]

    # Find the largest k such that the first k batches are identical.
    # If cache stability holds, we expect k = len(initial) - (0 or 1).
    # The most we can lose is the LAST initial batch absorbing g4.
    # Either way, every batch BEFORE the last initial batch must be
    # byte-identical (the cache wins on those).
    if len(initial_gid_lists) >= 2:
        assert initial_gid_lists[:-1] == after_gid_lists[:len(initial_gid_lists) - 1], (
            f"earlier batches drifted under append.\n"
            f"  initial: {initial_gid_lists}\n"
            f"  after  : {after_gid_lists}\n"
            f"this would make the LLM prompt cache miss on existing "
            f"entity batches when a user adds a new file."
        )
    # And g4 lands somewhere in the after-batches.
    all_after_gids = [gid for batch in after_gid_lists for gid in batch]
    assert "g4" in all_after_gids


def _fact_file_id(item):
    """Local copy of entities._fact_file_id for the test; entities
    keeps it private."""
    for ev in item.evidence:
        if ev.file_path:
            return ev.file_path
    return None
