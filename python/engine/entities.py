"""
Entities — stage 3 of the pipeline. Sits between Facts and Patterns.

Architecture (per-entity rewrite, replaces the prior batched-groups design):

  PHASE A — DETERMINISTIC GROUPING
    `_group_entities` buckets all entity_refs by (normalized_name,
    entity_type) into `_Group` records. Per-file narrator synthesis
    handles bundle inputs. All facts are kept on each group — no
    sampling, no narrator-replication.

  PHASE B — RELATION CANDIDATE RESOLUTION
    Each fact may carry one `relation_candidate` from extract
    (explicitly-stated relation between two named entities). We resolve
    candidate from/to names to canonical_ids using `_resolve_candidates`.
    Unresolvable candidates are dropped with a recorded warning. The
    survivors index by canonical_id so per-entity calls can be told
    which candidates relate to them.

  PHASE C — PER-ENTITY CANONICALIZATION (parallel)
    Groups are greedy-packed into batches under a token budget of
    `T_avg × sqrt(N)` (clipped at chunk_cap_for_stage("entities")), so
    expected batch count is ~sqrt(N) — for ~180 entities, ~13 calls.
    A heavy entity whose own block exceeds the per-batch budget is
    SPLIT-BY-NAME into K clones each fitting under the budget; clones
    share canonical_name + entity_type and are reunified in Phase D
    via the deterministic-collapse floor.

    Each batch is one LLM call. For each canonical group X in the
    batch, the call returns: canonical_name, role, description, type,
    aliases, mention_count, is_subject_likelihood, relations[].
    Inputs to the call: X's full facts + a CATALOG (id+name+type only,
    no facts) of OTHER entities mentioned in X's facts + the resolved
    relation_candidates with X as either from or to. The LLM
    consolidates all candidates into AT MOST ONE relation per (X, other)
    pair — intra-call deduplication.

  PHASE D — CROSS-ENTITY DEDUPE (single LLM call + deterministic floor)
    1. Deterministic floor (load-bearing): collapse all entities with
       identical (canonical_name, entity_type) into one. Required for
       Phase C's heavy-entity split-by-name to round-trip cleanly.
    2. LLM quality layer: one call over (id, name, type, aliases,
       description) rows emits merge pairs with synthesized_description
       for fuzzy matches ("Mom" + "Jane Doe", "Penn" + "William
       Penn"). Applied on top of the deterministic floor.

    Relations are remapped through the merge map and bilateral-merged:
    A→B and B→A both kept when both directions exist.

  SUBJECT IDENTIFICATION
    Argmax `is_subject_likelihood` across canonical entities. Tie-break
    by mention_count desc, then alphabetical canonical_name. Bundle
    inputs scrub the subject to None.

Output dataclasses:
  - EntityRecord   — one per canonical entity
  - RelationEdge   — directed edge between entities
  - SubjectRef     — resolved subject (id + display name + source tier)
  - EntitiesOutput — container holding all three

Usage:
    from engine.entities import detect_entities
    out = detect_entities(facts_by_topic, mode=Mode.TEE, subject="John")
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

from engine.content_extractor import ExtractedItem
from engine.llm import (
    Mode,
    chunk_cap_for_stage as _llm_chunk_cap_for_stage,
    _record_call_warning as _llm_record_call_warning,
    _resolve_stage_override as _llm_resolve_stage_override,  # noqa: F401 (re-export for phases.entity_dedupe)
    strip_fences as _strip_fences,
)
from engine.splitter import _BATCH_TOTAL_TOKEN_CAP
from engine.tokens import (
    count_tokens as _count_tokens,
)


# Set by runner._patch_llm_calls so per-stage exception tracebacks land
# in run.log alongside the wrapper-level logging. Stays None when this
# module runs outside the runner (tests, CLI, scripts) — _log_info
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
    import traceback
    head = f"  [{stage}] LLM call raised ({ctx}): {type(exc).__name__}: {exc}"
    _log_info(head)
    for line in traceback.format_exc().rstrip().splitlines():
        _log_info(f"    {line}")


# ── IR ────────────────────────────────────────────────────────────────────────


@dataclass
class EntityRecord:
    canonical_id: str
    canonical_name: str
    entity_type: str                 # person | place | org | concept | other
    aliases: list[str] = field(default_factory=list)
    role: str = ""                   # "subject" | "friend" | "spouse" | "organization" | "concept" | ...
    description: str = ""            # 1 short sentence, role-bearing
    mention_count: int = 0
    topics: list[str] = field(default_factory=list)
    evidence_fact_refs: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class RelationEdge:
    from_id: str
    to_id: str
    relation: str                    # short snake_case label
    confidence: float = 1.0
    evidence_fact_refs: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class SubjectRef:
    canonical_id: str
    display: str
    # Tier of subject resolution:
    #   "argmax"                   — argmax is_subject_likelihood across entities
    #   "mention_count_fallback"   — every is_subject_likelihood was 0 / missing;
    #                                fall back to most-mentioned person
    # runner.py treats "mention_count_fallback" as untrustworthy when the
    # CLI subject string is generic.
    source: str = "unknown"


@dataclass
class EntitiesOutput:
    subject: SubjectRef | None = None
    entities: list[EntityRecord] = field(default_factory=list)
    relations: list[RelationEdge] = field(default_factory=list)

    def by_id(self) -> dict[str, EntityRecord]:
        return {e.canonical_id: e for e in self.entities}


# ── Deterministic name handling ──────────────────────────────────────────────


def _normalize_name(name: str) -> str:
    """Fold case, strip accents, collapse punctuation/whitespace.
    Used for grouping — never shown to users."""
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower().strip()
    n = re.sub(r"[^\w\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _slugify(name: str) -> str:
    """Kebab-case slug for canonical_id."""
    n = _normalize_name(name).replace(" ", "-")
    n = re.sub(r"-+", "-", n).strip("-")
    return n or "entity"


def build_phase1_marker_payload(
    groups, candidates_by_gid, groups_by_gid, n_resolved, n_dropped, n_batches
) -> dict:
    """The on_phase_done("phase_1") marker payload — the grouping view the
    runner writes per-entity (and the phase_1 marker). Module-level so the
    kernel grouping phase produces byte-identical markers to legacy
    detect_entities without re-deriving the shape."""
    def _group_candidate_relations(g) -> list[dict]:
        out: list[dict] = []
        for c in candidates_by_gid.get(g.gid, []):
            from_g = groups_by_gid.get(c.from_id)
            to_g = groups_by_gid.get(c.to_id)
            if from_g is None or to_g is None:
                continue
            out.append({
                "from": _slugify(from_g.canonical_name),
                "to": _slugify(to_g.canonical_name),
                "verb": c.verb,
                "confidence": c.confidence,
                "evidence_fact_ref": list(c.evidence_fact_ref),
            })
        return out

    return {
        "groups": [
            {
                "gid": g.gid,
                "canonical_name": g.canonical_name,
                "canonical_id": _slugify(g.canonical_name),
                "entity_type": g.entity_type,
                "aliases": sorted(g.aliases),
                "mention_count": g.mention_count,
                "topics": sorted(g.topics),
                "evidence_fact_refs": [list(r) for r in g.evidence_fact_refs],
                "candidate_relations": _group_candidate_relations(g),
            }
            for g in groups
        ],
        "relation_candidates_resolved": n_resolved,
        "relation_candidates_dropped": n_dropped,
        "n_batches": n_batches,
    }


def build_phase2_marker_payload(annotations, groups_by_gid, n_batches) -> dict:
    """The on_phase_done("phase_2") marker payload — per-entity enrichment
    (annotations + clone→parent resolution). Module-level twin of
    build_phase1_marker_payload."""
    return {
        "annotations": annotations,
        "n_batches": n_batches,
        "groups_by_gid": {
            gid: {
                "canonical_name": g.canonical_name,
                "canonical_id": _slugify(g.canonical_name),
                "entity_type": g.entity_type,
                "parent_gid": g.parent_gid or gid,
            }
            for gid, g in groups_by_gid.items()
        },
    }


# Determiners filtered from the noise-alias check below. Subset of the
# `_DETERMINER_TOKENS` set declared further down (we re-list here rather
# than forward-reference because the dedupe constants live near
# `_passes_name_overlap_gate` which is far below). Keep in sync.
_ALIAS_NOISE_DETERMINERS: frozenset[str] = frozenset({"the", "a", "an"})


def _normalize_name_no_determiners(name: str) -> str:
    """Like `_normalize_name`, then ALSO drop pure determiners.
    Used by `_apply_merges_to_records` to filter aliases that are
    just the canonical name plus a leading "the" / "a" / "an" — they
    add no information ("the King" when canonical is "King"). Distinct
    from `_normalize_name` because grouping (Phase A) depends on
    keeping determiners as part of the bucket key — bucketing on the
    determiner-stripped form would over-collapse e.g. "the Smith
    twins" with "Smith"."""
    norm = _normalize_name(name)
    if not norm:
        return ""
    return " ".join(t for t in norm.split() if t not in _ALIAS_NOISE_DETERMINERS)


# ── _Group: deterministic pre-pass bucket ────────────────────────────────────


@dataclass
class _Group:
    """One canonical bucket. Becomes one EntityRecord post Phase D."""
    gid: str                              # temp id ("g1", …)
    canonical_name: str
    entity_type: str
    aliases: set[str] = field(default_factory=set)
    mention_count: int = 0
    topics: set[str] = field(default_factory=set)
    evidence_fact_refs: list[tuple[str, int]] = field(default_factory=list)
    # (topic, fact_idx, summary, occurred_at). Kept ordered for prompt
    # display. Full set — no sampling cap.
    facts: list[tuple[str, int, str, str | None]] = field(default_factory=list)
    # Other-entity catalog for X: {other_canonical_name: other_entity_type}
    # populated by `_build_other_entity_catalog` after groups are sorted.
    other_catalog: dict[str, str] = field(default_factory=dict)
    # Set on clones produced by `_split_heavy_entity` to point at the
    # original group's gid. Originals leave this empty; consumers that
    # need a uniform "which canonical group does this gid belong to"
    # answer should use `parent_gid or gid`. Storing it explicitly (vs
    # parsing the gid string) keeps clone→parent resolution independent
    # of the gid naming scheme — anyone changing gid construction later
    # doesn't silently misroute clones.
    parent_gid: str = ""


# ── Bundle / narrator detection ──────────────────────────────────────────────


def _fact_file_id(item: ExtractedItem) -> str | None:
    for ev in item.evidence:
        if ev.file_path:
            return ev.file_path
    return None


def _fact_provenance_key(item: ExtractedItem) -> tuple | None:
    """Identity for a fact independent of which category file it is filed
    under. A fact extracted under several categories is fanned out to one
    identical record per category; those copies share the same evidence
    provenance (source file + byte span), item type, and date. Keying on
    that provenance — never on the summary text alone — lets the entity
    view collapse the fan-out into one row while keeping two same-text
    facts from DIFFERENT extractions (different provenance) visibly
    separate, so a genuine duplicate-extraction anomaly is never silently
    merged away. Returns None when the fact carries no evidence to key on;
    those facts are never collapsed."""
    spans = tuple(
        (ev.file_path, ev.file_offset, ev.file_length, ev.start_char, ev.end_char)
        for ev in item.evidence
        if ev.file_path is not None
    )
    if not spans:
        return None
    return (item.item_type, item.occurred_at, spans)


def _collapse_cross_category_fact_refs(
    refs: list[tuple[str, int]],
    facts_by_topic: dict[str, list[ExtractedItem]],
) -> list[tuple[str, int]]:
    """Collapse references pointing to the SAME fact fanned out across
    categories into one reference, anchored to the alphabetically-first
    category. References whose facts share no provenance key (or carry no
    provenance) are left untouched and stay visible.

    This builds the entity's UI-facing citation list only. It is applied
    when writing the per-entity (derived) view, NOT to the canonical
    aggregate: the full reference list stays the authoritative fact→entity
    mention map for patterns context and the RAG graph, and per-category
    fact arrays / (topic, index) addressing are never touched."""
    by_key: dict[object, list[tuple[str, int]]] = {}
    order: list[object] = []
    for ref in refs:
        topic, idx = ref
        items = facts_by_topic.get(topic) or []
        item = items[idx] if 0 <= idx < len(items) else None
        key = _fact_provenance_key(item) if item is not None else None
        if key is None:
            # No shared identity to key on — keep this ref as its own row.
            key = ("\x00unkeyed", topic, idx)
        if key not in by_key:
            by_key[key] = []
            order.append(key)
        by_key[key].append(ref)
    return [min(by_key[key], key=lambda r: r[0]) for key in order]


_FIRST_PERSON_RE = re.compile(
    r"(?:\bI\b|\bI'm\b|\bI've\b|\bI'd\b|\bI'll\b|\b[Mm]y\b|\b[Mm]e\b|\b[Mm]ine\b|\b[Mm]yself\b)"
)


def _has_first_person(text: str) -> bool:
    return bool(text) and bool(_FIRST_PERSON_RE.search(text))


def _is_bundle(facts_by_topic: dict[str, list[ExtractedItem]]) -> bool:
    """True when facts span more than one file_id."""
    files: set[str] = set()
    for items in facts_by_topic.values():
        for it in items:
            fid = _fact_file_id(it)
            if fid:
                files.add(fid)
            if len(files) > 1:
                return True
    return False


# ── Phase A: deterministic grouping ───────────────────────────────────────────


def _earliest_manifest_position(
    g: _Group,
    facts_by_topic: dict[str, list[ExtractedItem]],
    manifest_pos: dict[str, int],
    *,
    sentinel: int = 1 << 62,
) -> int:
    """Lowest manifest position among files this group is grounded in.

    Groups whose source files aren't in the manifest at all collapse to
    `sentinel` so they sort to the end, after every manifest-known
    group. The min-aggregation matters for cache stability: when a NEW
    file later mentions an EXISTING entity, the entity's earliest
    position stays the same (its first sighting), so its sort slot
    doesn't move and its prior batch's prompt is unchanged.
    """
    earliest = sentinel
    for topic, idx in g.evidence_fact_refs:
        items = facts_by_topic.get(topic, [])
        if 0 <= idx < len(items):
            fid = _fact_file_id(items[idx])
            if fid is not None:
                pos = manifest_pos.get(fid)
                if pos is not None and pos < earliest:
                    earliest = pos
    return earliest


def _group_sort_key(
    g: _Group,
    facts_by_topic: dict[str, list[ExtractedItem]],
    manifest_pos: dict[str, int] | None,
) -> tuple:
    """Single source of truth for entity-group ordering.

    With a populated manifest: (earliest manifest position, name) — the
    append-only-stable key the LLM prompt cache relies on. Without a
    manifest (tests / ad-hoc CLI): fall back to the legacy
    (-mention_count, name) order so existing call sites behave
    unchanged. Used for BOTH gid assignment in `_group_entities` and
    batch packing in `_pack_batches`; they MUST agree, otherwise
    batches reference gids that don't match the gid-assignment order
    and the prompt cache fragments.
    """
    name_key = g.canonical_name.lower()
    if manifest_pos:
        return (
            _earliest_manifest_position(g, facts_by_topic, manifest_pos),
            name_key,
        )
    return (-g.mention_count, name_key)


def _group_entities(
    facts_by_topic: dict[str, list[ExtractedItem]],
    manifest_pos: dict[str, int] | None = None,
) -> list[_Group]:
    """Bucket entity_refs by (normalized_name, entity_type) across all facts.

    Bundle behavior: per-file narrator synthesis when a file has ≥2
    first-person facts and >half lack a named subject — synthesizes a
    `Narrator of <file_id>` group per such file. Bundle scrub later
    drops any single subject pick; the synthesized groups still exist
    as separate entities.
    """
    buckets: dict[tuple[str, str], _Group] = {}
    fp_facts_by_file: dict[str, int] = {}
    fp_named_subjects_by_file: dict[str, int] = {}
    fp_facts_by_file_full: dict[str, list[tuple[str, int, str, str | None]]] = {}

    for topic, items in facts_by_topic.items():
        for idx, item in enumerate(items):
            has_fp = any(_has_first_person(ev.text or "") for ev in item.evidence)
            fid = _fact_file_id(item)
            has_named_subject = any(
                (r.role or "").strip().lower() == "subject"
                and (r.entity.name or "").strip()
                for r in item.entities
            )
            if has_fp and fid is not None:
                fp_facts_by_file[fid] = fp_facts_by_file.get(fid, 0) + 1
                if has_named_subject:
                    fp_named_subjects_by_file[fid] = (
                        fp_named_subjects_by_file.get(fid, 0) + 1
                    )
                fp_facts_by_file_full.setdefault(fid, []).append(
                    (topic, idx, item.summary, item.occurred_at)
                )
            for ref in item.entities:
                name = (ref.entity.name or "").strip()
                if not name:
                    continue
                etype = (ref.entity.entity_type or "other").strip().lower()
                key = (_normalize_name(name), etype)
                if not key[0]:
                    continue
                g = buckets.get(key)
                if g is None:
                    g = _Group(
                        gid="",
                        canonical_name=name,
                        entity_type=etype,
                    )
                    buckets[key] = g
                g.aliases.add(name)
                g.mention_count += 1
                g.topics.update(item.topics)
                ref_key = (topic, idx)
                if ref_key not in g.evidence_fact_refs:
                    g.evidence_fact_refs.append(ref_key)
                g.facts.append((topic, idx, item.summary, item.occurred_at))
                if len(name) > len(g.canonical_name):
                    g.canonical_name = name

    groups = list(buckets.values())

    # Per-file narrator synthesis on bundles. Keeps file-scoped narrators
    # distinct so downstream can render "Narrator of journal-X" / "Narrator
    # of journal-Y" instead of merging into one ambiguous author.
    unique_files = set(fp_facts_by_file.keys())
    if len(unique_files) > 1:
        for fid in sorted(unique_files):
            fp_total = fp_facts_by_file.get(fid, 0)
            fp_named = fp_named_subjects_by_file.get(fid, 0)
            unnamed_fp = fp_total - fp_named
            if fp_total >= 2 and unnamed_fp * 2 >= fp_total:
                label = fid
                for prefix in ("input_", "input-"):
                    if label.startswith(prefix):
                        label = label[len(prefix):]
                        break
                for suffix in (".md", ".txt", ".json"):
                    if label.endswith(suffix):
                        label = label[: -len(suffix)]
                        break
                canonical = f"Narrator of {label}"
                facts = fp_facts_by_file_full.get(fid, [])
                g = _Group(
                    gid="",
                    canonical_name=canonical,
                    entity_type="person",
                    aliases={canonical, "Author", "narrator"},
                    mention_count=fp_total,
                    topics=set(),
                    evidence_fact_refs=[(t, i) for (t, i, _s, _d) in facts],
                    facts=facts,
                )
                groups.append(g)

    groups.sort(key=lambda g: _group_sort_key(g, facts_by_topic, manifest_pos))
    for i, g in enumerate(groups, start=1):
        g.gid = f"g{i}"
    return groups


# ── Phase B: relation_candidate resolution ────────────────────────────────────


@dataclass
class _ResolvedCandidate:
    """One relation_candidate after canonical-id resolution.

    `from_id` and `to_id` index into the post-grouping canonical set.
    `evidence_fact_ref` is the (topic, fact_idx) the candidate came from.
    """
    from_id: str       # group gid
    to_id: str         # group gid
    verb: str
    confidence: float
    evidence_fact_ref: tuple[str, int]


def _build_name_index(groups: list[_Group]) -> dict[str, list[_Group]]:
    """Index groups by normalized name. Multiple types under one name
    return all candidates; the resolver picks by best (name, type) match
    against the fact's own entity list."""
    idx: dict[str, list[_Group]] = defaultdict(list)
    for g in groups:
        for alias in g.aliases:
            idx[_normalize_name(alias)].append(g)
        idx[_normalize_name(g.canonical_name)].append(g)
    # Dedup within each bucket
    for k, v in idx.items():
        seen = set()
        out = []
        for g in v:
            if g.gid not in seen:
                seen.add(g.gid)
                out.append(g)
        idx[k] = out
    return dict(idx)


def _resolve_one_candidate(
    name: str,
    fact: ExtractedItem,
    name_idx: dict[str, list[_Group]],
) -> _Group | None:
    """Best-effort resolution of one fact-relation-candidate name to a
    group. Prefers groups whose entity_type matches the type tagged
    on this fact's entity list for the same name; falls back to any
    name-match group."""
    norm = _normalize_name(name)
    if not norm:
        return None
    candidates = name_idx.get(norm)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Multiple groups share this normalized name (ambiguous) — try to
    # disambiguate by looking up the entity_type the LLM tagged this
    # name with in the same fact's entity list.
    fact_type: str | None = None
    for ref in fact.entities:
        if _normalize_name(ref.entity.name or "") == norm:
            fact_type = (ref.entity.entity_type or "").strip().lower()
            break
    if fact_type:
        for g in candidates:
            if g.entity_type == fact_type:
                return g
    # No type signal — return the most-mentioned candidate (stable choice).
    return max(candidates, key=lambda g: (g.mention_count, g.canonical_name.lower()))


def _resolve_candidates(
    facts_by_topic: dict[str, list[ExtractedItem]],
    groups: list[_Group],
) -> tuple[dict[str, list[_ResolvedCandidate]], int, int]:
    """Walk every fact's relation_candidate, resolve from/to to canonical
    groups, and index the survivors by group gid (each candidate appears
    under both endpoints' gids).

    Returns (candidates_by_gid, n_resolved, n_dropped). Dropped count is
    surfaced as a warning; per-fact failures don't fail the stage.
    """
    name_idx = _build_name_index(groups)
    out: dict[str, list[_ResolvedCandidate]] = defaultdict(list)
    n_resolved = 0
    n_dropped = 0
    for topic, items in facts_by_topic.items():
        for idx, item in enumerate(items):
            rc = item.relation_candidate
            if not isinstance(rc, dict):
                continue
            from_g = _resolve_one_candidate(rc.get("from", ""), item, name_idx)
            to_g = _resolve_one_candidate(rc.get("to", ""), item, name_idx)
            if from_g is None or to_g is None or from_g.gid == to_g.gid:
                n_dropped += 1
                continue
            rcand = _ResolvedCandidate(
                from_id=from_g.gid,
                to_id=to_g.gid,
                verb=str(rc.get("verb", "")).strip(),
                confidence=float(rc.get("confidence", 1.0) or 1.0),
                evidence_fact_ref=(topic, idx),
            )
            out[from_g.gid].append(rcand)
            out[to_g.gid].append(rcand)
            n_resolved += 1
    return dict(out), n_resolved, n_dropped


def _build_name_key_map(
    groups: list[_Group],
) -> dict[tuple[str, str], _Group]:
    """Map (normalized_name, etype) → canonical group, including aliases.
    Built once from the original groups so name→gid resolution is stable."""
    by_key: dict[tuple[str, str], _Group] = {}
    for g in groups:
        by_key[(_normalize_name(g.canonical_name), g.entity_type)] = g
        for alias in g.aliases:
            by_key.setdefault(
                (_normalize_name(alias), g.entity_type), g
            )
    return by_key


def _catalog_for_refs(
    refs: "list[tuple[str, int]] | set[tuple[str, int]]",
    self_gid: str,
    by_key: dict[tuple[str, str], _Group],
    facts_by_topic: dict[str, list[ExtractedItem]],
) -> dict[str, str]:
    """The OTHER-entities catalog for a set of facts: {gid: canonical_name}
    for every canonical entity co-mentioned in those facts, minus
    `self_gid`. Used per-entity by `_build_other_catalogs` and per-CLONE by
    `_split_heavy_entity` — a clone holds only a SLICE of its parent's
    facts, so its catalog must be rebuilt from that slice, not inherited
    whole, or fact-splitting can't shrink it."""
    catalog: dict[str, str] = {}
    for topic, idx in refs:
        items = facts_by_topic.get(topic, [])
        if idx >= len(items):
            continue
        for ref in items[idx].entities:
            name = (ref.entity.name or "").strip()
            if not name:
                continue
            etype = (ref.entity.entity_type or "other").strip().lower()
            other = by_key.get((_normalize_name(name), etype))
            if other is None or other.gid == self_gid:
                continue
            catalog[other.gid] = other.canonical_name
    return catalog


def _build_other_catalogs(
    facts_by_topic: dict[str, list[ExtractedItem]],
    groups: list[_Group],
) -> None:
    """Populate group.other_catalog: for each group X, the OTHER canonical
    entities mentioned alongside X in any of X's facts. Catalog rows carry
    only id+name+type — NEVER X's facts about them — so per-entity calls
    keep their input scoped to X's POV."""
    by_key = _build_name_key_map(groups)
    for g in groups:
        g.other_catalog = _catalog_for_refs(
            g.evidence_fact_refs, g.gid, by_key, facts_by_topic
        )


# ── Per-entity prompt rendering ───────────────────────────────────────────────


_SYSTEM = """\
You annotate entities extracted from a personal-content pipeline. Each
input is ONE canonical entity with all its facts, plus a catalog of OTHER
entities those facts mention, plus zero or more relation candidates that
extract proposed for this entity.

Your job for each entity:
  1. Pick a canonical name (consolidate alias variants).
  2. Pick a role tag (single word — subject, friend, spouse, parent,
     sibling, colleague, employer, organization, concept, place, …).
  3. Write ONE short role-bearing sentence describing this entity's
     role in the source material. Role-bearing means it states how
     this entity relates to the subject or what position it holds.
     Must NOT interpret personality, infer intent, or hedge.
  4. Score is_subject_likelihood (0.0–1.0): how likely is THIS entity
     the subject of the corpus (the author / the person whose journal
     or notes this is). 1.0 = clearly the narrator; 0.0 = clearly not.
  5. Emit ONE relation per (this_entity → other_entity) pair, even if
     extract proposed multiple candidates with overlapping verbs. CONSOLIDATE
     across candidates: pick the single best verb that captures the relation.
     Confidence reflects how explicit and how recurrent the relation is.

Output format hard rules (violations break downstream parsing):
- Respond with a single JSON object. Nothing else.
- DO NOT emit chain-of-thought, reasoning, or explanation.
- DO NOT prefix with "Let me", "Here is", "Analyzing", etc.
- The very first character of your response MUST be `{`.
- The very last character MUST be `}`.
- Use ONLY the fields named in the schema below. Do not add any other
  fields. If you would naturally include extra metadata, omit it.\
"""


_TASK = """\
Subject (CLI hint, may be generic): "{subject}"

You are given {n_groups} entity blocks below. Each is one canonical
entity with the entity's facts, the catalog of OTHER entities those
facts mention, and zero or more relation candidates that extract
proposed for this entity.

Respond with ONLY a valid JSON object matching this exact schema:

{{
  "entities": [
    {{
      "group_id": "<gN>",
      "canonical_name": "<most complete, properly-capitalized form>",
      "role": "<single-word or short tag>",
      "description": "<ONE short sentence stating this entity's role. Role-bearing, not narrative. No personality guesses.>",
      "is_subject_likelihood": <float 0.0-1.0>,
      "relations": [
        {{
          "to_id": "<gN of an entity in THIS entity's catalog>",
          "verb": "<short snake_case label — spouse, parent, child, sibling, close_friend, colleague, employer_of, employed_by, lives_in, member_of, met_with, mentioned_with, …>",
          "confidence": <float 0.0-1.0>
        }}
      ]
    }}
  ]
}}

Rules:
- Description is ONE sentence. Role-bearing, not narrative. Must not
  interpret personality.
- Every claim in a description must be supported by the entity's facts.
- relations.to_id MUST be one of the gids in this entity's `catalog`
  block. NEVER invent gids; NEVER reference an entity outside the
  entity's catalog.
- AT MOST ONE relation per (this_entity, other_entity) pair. If extract
  proposed five candidates from different facts that all describe the
  same relation, consolidate into ONE relation with the best verb.
- Skip relations entirely when nothing in the facts EXPLICITLY supports
  one. Co-occurrence is not a relation.
- is_subject_likelihood reflects the entity itself, not how often it
  appears. A side character mentioned 50× has likelihood near 0; the
  narrator (first-person voice, "I"/"my"/"me") is near 1.
- No extra keys, no prose, no markdown fences.

ENTITIES IN THIS BATCH ({n_groups} total):

{groups_text}\
"""


_BUNDLE_NOTE = """\


BUNDLE INPUT DETECTED: facts in this run come from multiple source
files (potentially different first-person narrators). Per-file
narrator entities (`Narrator of …`) appear among the blocks below —
keep them separate. Their is_subject_likelihood should reflect THAT
narrator's role within their file, not pin them as the corpus subject."""


def _format_facts_block(facts: list[tuple[str, int, str, str | None]]) -> list[str]:
    """Format the facts list for one entity. Each fact is one line:
    `[date] (topic/idx) summary`. Long summaries are truncated."""
    lines: list[str] = []
    for topic, fact_idx, summary, occurred_at in facts:
        clean = (summary or "").replace("\n", " ").strip()
        if len(clean) > 220:
            clean = clean[:217] + "…"
        date_part = f"[{occurred_at}] " if occurred_at else ""
        lines.append(f"      - {date_part}({topic}/{fact_idx}) {clean}")
    return lines


def _format_catalog_block(
    other_catalog: dict[str, str],
    groups_by_gid: dict[str, _Group],
) -> list[str]:
    """Format the OTHER-entities catalog for one entity. id+name+type only;
    no facts."""
    if not other_catalog:
        return ["      (no other named entities in this entity's facts)"]
    lines: list[str] = []
    for gid in sorted(other_catalog.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        og = groups_by_gid.get(gid)
        if og is None:
            continue
        lines.append(f"      - [{og.gid}] {og.canonical_name} ({og.entity_type})")
    return lines


def _format_candidates_block(
    candidates: list[_ResolvedCandidate],
    target_gid: str,
    groups_by_gid: dict[str, _Group],
) -> list[str]:
    """Format relation candidates touching this entity. Shown both
    directions ('X→Y by extract' / 'Y→X by extract'). LLM is expected
    to consolidate across these into one outgoing relation per (X, Y)
    pair."""
    if not candidates:
        return ["      (no relation candidates from extract)"]
    lines: list[str] = []
    for c in candidates:
        if c.from_id == target_gid:
            other = groups_by_gid.get(c.to_id)
            if other is None:
                continue
            arrow = "→"
            other_id = c.to_id
        else:
            other = groups_by_gid.get(c.from_id)
            if other is None:
                continue
            arrow = "←"
            other_id = c.from_id
        verb = c.verb or "(no verb)"
        lines.append(
            f"      - {arrow} [{other_id}] {other.canonical_name} "
            f"({other.entity_type}) — verb=\"{verb}\" "
            f"conf={c.confidence:.2f} "
            f"@({c.evidence_fact_ref[0]}/{c.evidence_fact_ref[1]})"
        )
    return lines


def _render_entity_block(
    g: _Group,
    candidates: list[_ResolvedCandidate],
    groups_by_gid: dict[str, _Group],
    facts_override: list[tuple[str, int, str, str | None]] | None = None,
    catalog_override: dict[str, str] | None = None,
) -> str:
    """Full prompt block for one entity X.

    `facts_override` is used by heavy-entity split-by-name to render a
    clone with a subset of X's facts. `catalog_override` lets that path
    measure a slice against the catalog scoped to the slice's own facts
    (not the parent's full catalog) — so removing facts actually shrinks
    the render. Clones share aliases with the parent; their catalog and
    candidates are scoped to their fact subset."""
    facts = facts_override if facts_override is not None else g.facts
    catalog = catalog_override if catalog_override is not None else g.other_catalog
    aliases = sorted(g.aliases, key=lambda a: (-len(a), a))[:8]
    alias_str = ", ".join(aliases)
    topics = sorted(g.topics)
    topics_str = ", ".join(topics) if topics else "—"
    lines: list[str] = [
        f"[{g.gid}] {g.canonical_name} ({g.entity_type}, mentions={g.mention_count})",
        f"    aliases: {alias_str}",
        f"    topics:  {topics_str}",
        "    facts:",
    ]
    lines.extend(_format_facts_block(facts))
    lines.append("    catalog (other entities mentioned in this entity's facts):")
    lines.extend(_format_catalog_block(catalog, groups_by_gid))
    lines.append("    relation candidates from extract (to consolidate):")
    lines.extend(_format_candidates_block(candidates, g.gid, groups_by_gid))
    return "\n".join(lines)


def _build_prompt(
    blocks: list[str], subject: str, n_groups: int, is_bundle: bool
) -> str:
    body = "\n\n".join(blocks)
    task = _TASK.format(subject=subject, n_groups=n_groups, groups_text=body)
    if is_bundle:
        task = task + _BUNDLE_NOTE
    return task


# ── Phase C: per-entity batching + LLM calls ──────────────────────────────────


def _split_heavy_entity(
    g: _Group,
    candidates_by_gid: dict[str, list[_ResolvedCandidate]],
    groups_by_gid: dict[str, _Group],
    budget_tokens: int,
    by_key: dict[tuple[str, str], _Group] | None = None,
    facts_by_topic: dict[str, list[ExtractedItem]] | None = None,
) -> list[_Group]:
    """When a single group's render exceeds the per-batch budget, split
    its facts into K date-ordered slices, each rendering under budget.

    Each clone holds only a SLICE of the parent's facts, so its catalog
    and candidates are rebuilt from that slice (via `_catalog_for_refs`,
    the same per-fact scoping `_build_other_catalogs` does per entity) —
    not inherited whole. That's what makes fact-splitting effective on a
    large corpus: a clone shrinks as facts are removed, so the bisection
    converges instead of running to one fact per clone. `by_key` +
    `facts_by_topic` are required to scope; when omitted, clones fall back
    to the parent's full catalog.

    Clones share canonical_name + entity_type. The deterministic
    collapse in Phase D (load-bearing) merges them back into one record.
    """
    cands = candidates_by_gid.get(g.gid, [])
    can_scope = by_key is not None and facts_by_topic is not None

    def slice_catalog(
        slice_facts: list[tuple[str, int, str, str | None]],
    ) -> dict[str, str]:
        if not can_scope:
            return dict(g.other_catalog)
        refs = [(t, i) for (t, i, _s, _d) in slice_facts]
        return _catalog_for_refs(refs, g.gid, by_key, facts_by_topic)

    def slice_cands(
        slice_facts: list[tuple[str, int, str, str | None]],
    ) -> list[_ResolvedCandidate]:
        refs = {(t, i) for (t, i, _s, _d) in slice_facts}
        return [c for c in cands if c.evidence_fact_ref in refs]

    def render_with_facts(facts: list[tuple[str, int, str, str | None]]) -> str:
        return _render_entity_block(
            g, slice_cands(facts), groups_by_gid,
            facts_override=facts, catalog_override=slice_catalog(facts),
        )

    # Sort by occurred_at (None last), then by (topic, idx) for determinism.
    facts_sorted = sorted(
        g.facts,
        key=lambda f: (f[3] is None, f[3] or "", f[0], f[1]),
    )

    # Bisect on K (number of clones). Start with K=2, double until each
    # slice fits under budget, capping K at len(facts_sorted) (one fact
    # per clone is the floor).
    n = len(facts_sorted)
    if n <= 1:
        # Singleton facts that still overflow ship as-is — surfaces as
        # an input_overflow warning at call time. Better than silently
        # losing the entity.
        return [g]
    K = 2
    while K <= n:
        slice_size = math.ceil(n / K)
        slices = [facts_sorted[i:i + slice_size] for i in range(0, n, slice_size)]
        worst = max(_count_tokens(render_with_facts(s)) for s in slices if s)
        if worst <= budget_tokens or K == n:
            break
        K *= 2
    K = min(K, n)
    slice_size = math.ceil(n / K)
    slices = [facts_sorted[i:i + slice_size] for i in range(0, n, slice_size)]

    clones: list[_Group] = []
    for ci, slice_facts in enumerate(slices, start=1):
        if not slice_facts:
            continue
        clone = _Group(
            gid=f"{g.gid}c{ci}",
            canonical_name=g.canonical_name,
            entity_type=g.entity_type,
            aliases=set(g.aliases),
            mention_count=len(slice_facts),
            topics=set(g.topics),
            evidence_fact_refs=[(t, i) for (t, i, _s, _d) in slice_facts],
            facts=list(slice_facts),
            other_catalog=slice_catalog(slice_facts),
            parent_gid=g.gid,
        )
        clones.append(clone)
    # Wire candidates: candidates that touched g.gid now reference clone
    # gids. Each clone receives ONLY the candidates whose evidence fact
    # is in this clone's facts subset — sending the parent's full
    # candidate list to every clone bloats the prompt with relations the
    # clone cannot verify from its own facts, and inflates per-clone
    # token cost. The clones' relations get merged in Phase D via the
    # deterministic collapse.
    for clone in clones:
        clone_refs = set(clone.evidence_fact_refs)
        # Rewire candidate from_id/to_id from g.gid → clone.gid for
        # this clone's view; the OTHER endpoint of each candidate keeps
        # whatever gid it had (which is fine; that other group isn't
        # cloned in this branch unless it itself is heavy).
        new_cands: list[_ResolvedCandidate] = []
        for c in cands:
            if c.evidence_fact_ref not in clone_refs:
                continue
            if c.from_id == g.gid:
                new_cands.append(_ResolvedCandidate(
                    from_id=clone.gid, to_id=c.to_id,
                    verb=c.verb, confidence=c.confidence,
                    evidence_fact_ref=c.evidence_fact_ref,
                ))
            elif c.to_id == g.gid:
                new_cands.append(_ResolvedCandidate(
                    from_id=c.from_id, to_id=clone.gid,
                    verb=c.verb, confidence=c.confidence,
                    evidence_fact_ref=c.evidence_fact_ref,
                ))
        candidates_by_gid[clone.gid] = new_cands
    return clones


def _pack_batches(
    groups: list[_Group],
    candidates_by_gid: dict[str, list[_ResolvedCandidate]],
    groups_by_gid: dict[str, _Group],
    mode: Mode,
    facts_by_topic: dict[str, list[ExtractedItem]] | None = None,
    manifest_pos: dict[str, int] | None = None,
    by_key: dict[tuple[str, str], _Group] | None = None,
) -> list[list[_Group]]:
    """Greedy-pack canonical groups into batches sized at T_avg×sqrt(N),
    clipped at chunk_cap.

    Iteration order over `groups` matters for the prompt cache. When a
    `manifest_pos` is supplied, groups are sorted by
    `_group_sort_key` (earliest manifest position, name) so adding a
    new file to inputs only appends new groups at the end; existing
    groups keep their slot, prior batches form identically across
    runs, and the LLM prompt cache hits on every unchanged batch. The
    one trade is that a brand-new file can leave the trailing
    light-batch under-filled — accepted as a small token loss for the
    much larger cache-hit win on existing batches.

    A "heavy" entity is one whose own render exceeds the per-batch
    packing budget — typically a narrator-class entity whose facts
    list, OTHER-entities catalog, and relation candidates collectively
    dominate. These cannot be packed alongside other groups under the
    small budget, but they DO fit in the model's much larger chunk_cap
    on their own. Sending them solo at chunk_cap costs one call per
    heavy entity instead of cloning into many tiny calls.

    Only "ultra-heavy" entities (own render > chunk_cap) trigger the
    split-by-name path; that's the case where even a solo batch
    overflows the model. For those we split with chunk_cap as the
    per-clone budget so each clone is the largest single call the
    model accepts. The deterministic-collapse step in Phase D
    re-unifies clones via shared (canonical_name, entity_type).

    Result for typical narrator-heavy corpora (e.g. Pepys, kzrq):
    light-group batches ≈ sqrt(N), heavy-entity solo batches = small
    constant (1-3). For non-narrator corpora the result reduces to
    ≈ sqrt(N) batches.

    Empty groups list short-circuits to [].
    """
    N = len(groups)
    if N == 0:
        return []

    # Re-sort to the same key as gid assignment in `_group_entities`.
    # The two MUST agree: gids are assigned in the post-sort order, so
    # iterating `groups` here in any other order would emit batches
    # with non-monotonic gids and silently fragment the cache.
    if facts_by_topic is not None:
        groups = sorted(
            groups,
            key=lambda g: _group_sort_key(g, facts_by_topic, manifest_pos),
        )

    # Name→gid map for scoping ultra-heavy clone catalogs to their own
    # facts. Built here from the (clone-free) `groups` when not supplied;
    # the production path (`detect_entities`) passes a prebuilt one.
    if by_key is None and facts_by_topic is not None:
        by_key = _build_name_key_map(groups)

    # Render each group's block once, measure tokens.
    block_tokens: dict[str, int] = {}
    for g in groups:
        cands = candidates_by_gid.get(g.gid, [])
        block = _render_entity_block(g, cands, groups_by_gid)
        block_tokens[g.gid] = _count_tokens(block)

    chunk_cap = _llm_chunk_cap_for_stage(mode, "entities")
    t_avg = sum(block_tokens.values()) / max(1, N)
    target = t_avg * math.sqrt(N)
    budget = int(min(target, chunk_cap))
    # Safety floor: never let budget drop below 2× t_avg or 4000 tokens —
    # guarantees a healthy 2+ groups per batch on tiny corpora.
    budget = max(budget, int(2 * t_avg), 4000)
    # Per-call cap on LIGHT-packing budget only (multi-group batches).
    # Single-group exception: heavy-solo and ultra-heavy-clone paths
    # below ship 1 group per call, so the cap doesn't apply there —
    # they keep the raw `chunk_cap` so a big entity isn't fragmented
    # further than it has to be. Same `_BATCH_TOTAL_TOKEN_CAP` knob
    # that bounds extract-stage bundling, applied here to bound the
    # per-call annotation-invalidation footprint at the entities
    # stage (#174 prereq).
    if sum(block_tokens.values()) > chunk_cap:
        budget = min(budget, _BATCH_TOTAL_TOKEN_CAP)

    # Sort groups: heavy first (so they get their own batches), light
    # last (greedy-packed under budget). The split is at `block > budget`.
    light_groups: list[_Group] = []
    heavy_batches: list[list[_Group]] = []
    n_heavy = 0
    n_ultra_heavy = 0
    for g in groups:
        bt = block_tokens[g.gid]
        if bt <= budget:
            light_groups.append(g)
            continue

        n_heavy += 1
        if bt <= chunk_cap:
            # Heavy but not ultra-heavy: solo batch at chunk_cap, no
            # clone split needed. One call covers the whole entity.
            heavy_batches.append([g])
            continue

        # Ultra-heavy: split-by-name with chunk_cap as the per-clone
        # budget. Clones still likely fill chunk_cap individually, so
        # each clone gets its own solo batch.
        n_ultra_heavy += 1
        clones = _split_heavy_entity(
            g, candidates_by_gid, groups_by_gid, chunk_cap,
            by_key=by_key, facts_by_topic=facts_by_topic,
        )
        for clone in clones:
            cands = candidates_by_gid.get(clone.gid, [])
            block = _render_entity_block(clone, cands, groups_by_gid)
            block_tokens[clone.gid] = _count_tokens(block)
            groups_by_gid[clone.gid] = clone
            heavy_batches.append([clone])

    # Greedy-pack the light groups under `budget`.
    light_batches: list[list[_Group]] = []
    cur: list[_Group] = []
    cur_tokens = 0
    for g in light_groups:
        bt = block_tokens.get(g.gid, 0)
        if cur and cur_tokens + bt > budget:
            light_batches.append(cur)
            cur = [g]
            cur_tokens = bt
        else:
            cur.append(g)
            cur_tokens += bt
    if cur:
        light_batches.append(cur)

    batches = light_batches + heavy_batches
    _log_info(
        f"  [entities] _pack_batches diagnostics: "
        f"N={N} t_avg={t_avg:.1f}t target={target:.0f}t "
        f"budget={budget}t chunk_cap={chunk_cap}t "
        f"heavy={n_heavy} ultra_heavy={n_ultra_heavy} "
        f"light_batches={len(light_batches)} heavy_batches={len(heavy_batches)} "
        f"total_batches={len(batches)}"
    )
    return batches


def _parse_per_entity_response(
    raw: str, batch: list[_Group], groups_by_gid: dict[str, _Group],
) -> tuple[dict[str, dict], bool]:
    """Parse the per-entity batch response. Returns {gid: annotation}.

    annotation keys: canonical_name, role, description,
    is_subject_likelihood, relations (list of {to_id, verb, confidence}).
    Missing/unparseable gids fall through with empty annotations.

    Returns (annotations, parse_error). `parse_error` is True when
    the model returned non-empty content that didn't survive JSON
    parsing — distinguishes "model emitted malformed JSON" from
    "model emitted parseable JSON with zero entities" so the
    sizing cascade can route correctly (parse_error →
    `_ParseError` → classifier returns "sizing" → halve)."""
    out: dict[str, dict] = {}
    if not raw or not raw.strip():
        return out, False
    try:
        data = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, ValueError):
        return out, True
    if not isinstance(data, dict):
        return out, True
    ents = data.get("entities") or []
    if not isinstance(ents, list):
        return out, True

    batch_gids = {g.gid for g in batch}
    for e in ents:
        if not isinstance(e, dict):
            continue
        gid = str(e.get("group_id", "")).strip()
        if gid not in batch_gids:
            continue
        # Dedup intra-call: at most ONE relation per (this_entity, other_id).
        seen_other: set[str] = set()
        rels: list[dict] = []
        for r in e.get("relations", []) or []:
            if not isinstance(r, dict):
                continue
            to_id = str(r.get("to_id", "")).strip()
            # to_id must point to an existing canonical group (NOT a clone gid)
            if to_id not in groups_by_gid or to_id == gid:
                continue
            if to_id in seen_other:
                continue
            seen_other.add(to_id)
            verb = str(r.get("verb", "")).strip().lower().replace(" ", "_")
            if not verb:
                continue
            try:
                conf = float(r.get("confidence", 1.0))
            except (TypeError, ValueError):
                conf = 1.0
            conf = max(0.0, min(1.0, conf))
            rels.append({"to_id": to_id, "verb": verb, "confidence": conf})
        try:
            isl = float(e.get("is_subject_likelihood", 0.0))
        except (TypeError, ValueError):
            isl = 0.0
        isl = max(0.0, min(1.0, isl))
        # First annotation wins per gid (clones can't return the same gid).
        if gid not in out:
            out[gid] = {
                "canonical_name": str(e.get("canonical_name", "")).strip(),
                "role": str(e.get("role", "")).strip().lower(),
                "description": str(e.get("description", "")).strip(),
                "is_subject_likelihood": isl,
                "relations": rels,
            }
    return out, False


def _validate_entities_or_raise(raw: str) -> str:
    """Parser callable for the entities-summarize wrapper. Raises
    `_EmptyResponse` on whitespace-only output, `_ParseError` on
    non-empty but unparseable JSON, and `_SuccessEmpty` on a parseable
    JSON object whose `entities` list is empty or missing. Returns raw
    unchanged otherwise — `_parse_per_entity_response` does the
    structured shape work downstream.

    The empty-check keys on the CONTENT (`entities` list), not the
    outer container: the schema is `{"entities": [...]}`, so the
    model's "nothing found" response is `{"entities": []}`, not a bare
    `{}`. An outer-only `not data` check let that schema-shaped empty
    pass as a plain success the display still labelled `success_empty`
    while the run never retried it. Now mirrors insights (both arrays
    empty) / patterns (empty list) / extract (empty `items`). Non-dict
    shapes fall through to `_parse_per_entity_response`'s parse_error →
    sizing path (preserves the shape-defect cascade)."""
    from engine.parse_signals import _EmptyResponse, _ParseError, _SuccessEmpty
    if not (raw or "").strip():
        raise _EmptyResponse(stage="entities")
    try:
        data = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, ValueError):
        raise _ParseError(stage="entities")
    if isinstance(data, dict) and not (data.get("entities") or []):
        raise _SuccessEmpty(stage="entities")
    return raw


# ── Phase D: cross-entity dedupe ──────────────────────────────────────────────

_DEDUPE_SYSTEM = """\
You deduplicate canonical entities. Inputs are entity rows that have
already been canonicalized once; your job is to find ALIAS DUPLICATES
— two rows that name the SAME real-world entity. Hallucinated merges
are far worse than missed merges: a missed merge produces a slight
duplication in the output; a hallucinated merge silently fuses two
distinct people / places / things. When in any doubt, DO NOT MERGE.

Output format hard rules (violations break downstream parsing):
- Respond with a single JSON object. Nothing else.
- DO NOT emit chain-of-thought, reasoning, or explanation.
- DO NOT prefix with "Let me", "Here is", etc.
- The very first character of your response MUST be `{`.
- The very last character MUST be `}`.
- Use ONLY the fields named in the schema below. Do not add any other
  fields. If you would naturally include extra metadata, omit it.\
"""

_DEDUPE_TASK = """\
Below are {n} canonical entity rows. Each has an id, canonical_name,
type, total mention count, aliases, and a one-sentence description.
Mention count is the number of source facts that reference this
entity — high-count rows are the canonical naming for that entity,
low-count rows are the candidates to merge INTO them when an alias
relationship is supported.

Find pairs that refer to the SAME real-world entity. Be conservative:
when uncertain, do not merge.

═══ Cross-type merges: name match dominates type mismatch ═══

Every row carries a `type` in parentheses: `(person)`, `(place)`,
`(org)`, `(concept)`, or `(other)`. Types come from upstream
extract and are noisy — the SAME real entity often appears with
different types across rows because different facts cued different
categorizations (an AI system tagged as `concept` in one fact and
`person` in another; a company tagged as `org` here and `other`
there).

When two rows have IDENTICAL or VERY SIMILAR canonical_names,
MERGE them even if types differ. The name match is stronger
evidence of identity than the type mismatch is evidence of
distinctness.
- `Acme (concept)` + `Acme (person)` + `Acme (other)` — all named
  "Acme", all describing the same company/system → MERGE all three.
- `Acme Corp (org)` + `Acme Corp (other)` → MERGE.
- `Acme Inc. (org)` + `Acme (concept)` (descriptions both
  about the company) → MERGE.

When two rows have DIFFERENT canonical_names AND different types,
the cases below remain forbidden — different name + different
type signals a RELATION, not identity:
- `Alice Smith (person)` + `Director of Engineering (org)`
  → DO NOT MERGE. Alice HOLDS the office; she is not the office.
- `Lord Bob (person)` + `Bob Estate (place)` → DO NOT
  MERGE. A person and the place named after them are different.
- `Queen Carol (person)` + `the Crown (concept)` → DO NOT
  MERGE. The person and the institution are different.
- `Building D (place)` + `Treasury (org)` → DO NOT MERGE.
  Different types AND different referents.
- `Alice's library (concept)` + `University E (org)` → DO
  NOT MERGE. The library was bequeathed to the university; they
  remain different things.

If you find yourself reaching across BOTH different names AND
different types because the entities are tightly linked in the
descriptions, STOP. That linkage is a relation, not an identity.
Relations live elsewhere; this stage only deduplicates aliases of
the same entity.

═══ END ═══

Respond with ONLY a valid JSON object:

{{
  "merges": [
    {{
      "a_id": "<id>",
      "b_id": "<id>",
      "confidence": <float 0.0-1.0>,
      "synthesized_description": "<ONE sentence consolidating both descriptions. Role-bearing.>"
    }}
  ]
}}

Rules:
- a_id and b_id must both appear in the rows below.
- Prefer NOT merging when uncertain. Hallucinated merges silently
  fuse distinct entities; missed merges only leave a slight
  duplication. Missed merges are the cheaper failure mode.

What COUNTS as a merge candidate (same entity; type may differ):
- Initials/abbreviation paired with the full name: "A. Smith"
  + "Alice Smith"  → MERGE (A. is the initial of Alice, surnames match).
- Honorific variant of the same proper name: "Smith" + "Dr. Smith"
  → MERGE.
- A relationship/role word paired with a specific name when the
  descriptions clearly point at the same person: "Mom" +
  "Alice Smith" (descriptions both mention the author's mother)
  → MERGE.
- Capitalization, punctuation, accent, or whitespace variants of
  the same string: "St. John" + "St John" → MERGE.
- Same canonical_name with DIFFERENT types when the descriptions
  describe the same real entity: "Acme (concept)" + "Acme (person)"
  both about the same company → MERGE. Type variance reflects
  upstream extract noise, not actual distinctness.

What does NOT count and MUST NOT MERGE (even within the same type):
- Two entities sharing only a title with DIFFERENT proper names:
  "A. Smith" + "A. Jones" → DO NOT MERGE (different surnames).
  "Lord Bob" + "Lord Carol" → DO NOT MERGE.
- Two role words that name DIFFERENT roles: "King" + "Queen" → DO
  NOT MERGE. "Father" + "Mother" → DO NOT MERGE.
- Two specific names with NO token overlap, even if both are
  mentioned in similar contexts: "Alice Smith" + "Bob Jones"
  → DO NOT MERGE.
- An entity name and a different entity's role label: "Alice
  Smith" + "the legal team" → DO NOT MERGE.
- Anyone named differently, even within the same family: "Bob
  Smith" + "Carol Smith-Jones" (siblings) → DO NOT MERGE.

Default to NOT merging. The downstream pipeline tolerates a few
extra entity rows; it cannot recover from two distinct people fused
into one.

CONFIDENCE — score each merge honestly. Use the full 0.0-1.0 range:
- 0.95-1.0: an unambiguous alias variant (initials + full name with
  matching surname, capitalization differences only, "Dr. Smith" +
  "Smith").
- 0.85-0.95: a strong fuzzy match where the descriptions
  corroborate ("Mom" + "Alice Smith" with matching family
  context).
- 0.7-0.85: a probable match with one or two pieces of evidence
  but residual uncertainty.
- Below 0.7: a weak guess.

You don't need to filter merges by confidence yourself — emit every
merge you'd defend at any confidence ≥ 0.7 and SCORE it honestly.
The downstream filter handles the threshold; your job is honest
emission, not threshold enforcement. But: do not pad the merges
list with low-confidence guesses just to seem complete. Each merge
should be one you'd argue for at the score you assigned.

- One synthesized_description per merge — ALWAYS supply it. This
  string REPLACES the merged entity's `description` field
  downstream, so it must obey the same contract as a per-entity
  description from the entities stage: ONE short role-bearing
  sentence describing the entity's role in the source material —
  role-bearing means it states how this entity relates to the
  subject or what position it holds; not narrative, no personality
  guesses, no intent inference, no hedging. Every claim must be
  supported by the source rows.

  CONSOLIDATE distinct information across the source descriptions
  into ONE sentence. When sources differ only by which facet they
  emphasize, the consolidated sentence must NAME the union of
  facets compactly, not re-state any one source. The output is a
  single sentence regardless of how many source rows the merge
  spans.

- SELF-CLEAN bloated single-row descriptions. If a row's `desc`
  field already contains the ` · ` separator (i.e. is more than
  one sentence — the residue of a prior fallback concat), emit a
  SELF-MERGE entry with `a_id == b_id == <that row's id>` and a
  `synthesized_description` that compresses the multi-sentence
  desc to one role-bearing sentence per the contract above. This
  is independent of any alias merge: a row gets a self-merge
  rewrite when its desc is bloated, even if no other row deduplicates
  with it. Same shape contract on the rewritten description (one
  short role-bearing sentence, no narrative, no personality
  guesses, every claim source-supported). Compresses the union
  of facets across the bloated sentences, doesn't re-state any
  single one.
- No extra keys, no prose, no markdown fences.

ROWS ({n}):

{rows}\
"""

# Honorifics / role-titles to strip before checking name-token overlap
# in `_passes_name_overlap_gate`. Lowercase. Single tokens only — the
# tokenizer drops these AFTER lowercasing, so case variants are
# handled implicitly. "of" / "the" appear because canonical names
# like "Duke of York" / "Lord of Sandwich" otherwise contribute
# spurious "of"/"the" overlap matches between unrelated entities.
_HONORIFIC_TOKENS: frozenset[str] = frozenset({
    "sir", "lord", "lady", "mr", "mrs", "ms", "miss", "dr", "doctor",
    "captain", "capt", "major", "colonel", "col", "general", "gen",
    "admiral", "rev", "father", "mother", "fr", "saint", "st",
    "king", "queen", "prince", "princess", "duke", "duchess",
    "earl", "count", "countess", "baron", "baroness",
    "the", "of", "a", "an",
})


# Pure determiners — meaningless on their own, contribute zero
# identity. Filtered from the honorific-only fallback so "the King"
# + "the Queen" don't accidentally merge via shared "the". Subset
# of `_HONORIFIC_TOKENS`. Honorifics with substantive role meaning
# (king, duke, lord, etc.) stay outside this set so they CAN
# anchor a fallback merge of "King" + "the King".
_DETERMINER_TOKENS: frozenset[str] = frozenset({"the", "of", "a", "an"})


# Relationship/role words that are allowed to merge with specific
# names via short-form leniency. Used in `_passes_name_overlap_gate`
# Path 3. The brief explicitly endorses "Mom" + "Jane Doe";
# this set is the closed list of single-token role words that can
# anchor such a merge. NOT a generic "≤ N chars" rule — that was
# tried and rejected because common short surnames like "Snow",
# "Cook", "Ford" would have falsely qualified.
_RELATIONSHIP_WORDS: frozenset[str] = frozenset({
    "mom", "dad", "wife", "husband",
    "son", "daughter", "brother", "sister",
    "uncle", "aunt", "cousin", "nephew", "niece",
    "kid", "child", "spouse", "partner",
    "narrator", "author",
})


def _name_substantive_tokens(name: str) -> set[str]:
    """Tokens of `name` after normalize + drop honorifics + drop
    single-letter remnants of stripped initials.

    "Sir W. Pen" → {"w", "pen"}
    "Sir William Penn" → {"william", "penn"}
    "Lord Sandwich" → {"sandwich"}
    "Mom" → {"mom"}
    """
    norm = _normalize_name(name)
    return {
        t for t in norm.split()
        if t and t not in _HONORIFIC_TOKENS
    }


def _le1_distance(a: str, b: str) -> bool:
    """True iff Damerau-Levenshtein(a, b) ≤ 1 — accepts one insertion,
    deletion, substitution, OR adjacent transposition as a single edit.

    Plain Levenshtein-1 misses common typos like Smith ↔ Smtih, Brian ↔
    Brain, Carol ↔ Cralo (each is two substitutions in the basic metric).
    Damerau treats adjacent transposition as one edit, catching them.

    Linear in min(len)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        # Walk and count diffs, fail-fast on second.
        diffs = 0
        first_diff = -1
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                diffs += 1
                if diffs > 2:
                    return False
                if first_diff == -1:
                    first_diff = i
        if diffs <= 1:
            return True
        # Exactly two diffs at adjacent positions where the chars are
        # swapped (a[i]=b[i+1] and a[i+1]=b[i]) → single transposition.
        if diffs == 2 and first_diff + 1 < la \
           and a[first_diff] == b[first_diff + 1] \
           and a[first_diff + 1] == b[first_diff]:
            # Confirm there are no diffs after the swap.
            return a[first_diff + 2:] == b[first_diff + 2:]
        return False
    # insertion: ensure `a` is the shorter
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    # find first differing position; the rest of `b` after the
    # inserted char must equal the rest of `a`.
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def _tokens_similar(ta: str, tb: str) -> bool:
    """True if two substantive tokens are similar enough to count as
    overlap. Three acceptance paths:

    1. Exact match: "smith" == "smith".
    2. Prefix match (both ≥ 3 chars, length difference ≤ 2):
       "pen" + "penn", "penn" + "pennington" (latter on prefix
       only). Cap at 2 prevents nicknames-as-prefixes like "brun"
       matching "bruncker".
    3. Levenshtein-1 typo on long tokens (both ≥ 5 chars): catches
       "bruncker" + "brunker" (insertion), "smith" + "smyth"
       (substitution). Restricted to ≥5 chars to avoid false
       positives on short names ("anna" + "anne").
    """
    if ta == tb:
        return True
    if len(ta) < 3 or len(tb) < 3:
        return False
    if abs(len(ta) - len(tb)) <= 2 and (ta.startswith(tb) or tb.startswith(ta)):
        return True
    if len(ta) >= 5 and len(tb) >= 5 and _le1_distance(ta, tb):
        return True
    return False


def _passes_name_overlap_gate(name_a: str, name_b: str) -> bool:
    """Reject merges where the two canonical names share NO
    substantive token after honorifics are stripped.

    Initial-only matching ("W." == "William") was tried and
    rejected — too lenient: "Sir W. Pen" passes against "Sir
    William Batten" because W matches the first letter of William,
    even though Pen ≠ Batten. The token-similarity rule below
    catches the legitimate cases via prefix-match ("W. Pen" → "Pen
    Penn") and typo-tolerance ("Bruncker" → "Brunker") without the
    false positives.

    Acceptance paths:

    1. Token-similarity overlap: at least one substantive token of
       one name is _tokens_similar to a substantive token of the
       other. Catches exact match, prefix variants, and ≤1-edit
       typos on long tokens.

    2. Honorific-only fallback: when one side has zero substantive
       tokens (e.g. "King" alone, or "Duke"), check unstripped
       overlap with the other side. "King" + "King Charles II"
       share "king" in the unstripped tokens — clearly the same
       category-named entity referenced two ways. The other side
       must still have substantive tokens for this to apply (so
       "Duke" + "Lord" doesn't pass — both honorific-only, no
       specific identity to anchor on).

    3. Short-form leniency: one side is a single short substantive
       token (≤ 4 chars) AND the other side has a substantive
       token of length ≥ 4. Mirrors the brief's explicitly-endorsed
       "Mom" + "Jane Doe" merge.
    """
    a_tokens = _name_substantive_tokens(name_a)
    b_tokens = _name_substantive_tokens(name_b)

    # Path 2 first: honorific-only fallback. When at least one side
    # has zero substantive tokens, fall back to UNSTRIPPED token
    # overlap, MINUS pure determiners ("the", "of", "a", "an") to
    # avoid spurious "the King" + "the Queen" matches via shared
    # "the". This catches:
    #   - "King" ({}) ↔ "King Charles II" ({charles, ii}) — share
    #     "king" unstripped.
    #   - "King" ↔ "the King" — both honorific-only but share
    #     "king" after stripping "the".
    #   - "Duke" ↔ "Duke of York" — share "duke" unstripped.
    # And rejects:
    #   - "the King" ↔ "the Queen" — only common token is "the",
    #     filtered as a determiner.
    #   - "Duke" ↔ "Lord" — no overlap at all.
    if not a_tokens or not b_tokens:
        a_un = set(_normalize_name(name_a).split()) - _DETERMINER_TOKENS
        b_un = set(_normalize_name(name_b).split()) - _DETERMINER_TOKENS
        if a_un and b_un and (a_un & b_un):
            return True
        return False

    # Path 1: token-similarity overlap.
    for ta in a_tokens:
        for tb in b_tokens:
            if _tokens_similar(ta, tb):
                return True

    # Path 3: short-form leniency for explicit relationship words.
    def _is_relationship_word_single(toks: set[str]) -> bool:
        return len(toks) == 1 and next(iter(toks)) in _RELATIONSHIP_WORDS

    def _has_length4_token(toks: set[str]) -> bool:
        return any(len(t) >= 4 for t in toks)

    if _is_relationship_word_single(a_tokens) and _has_length4_token(b_tokens):
        return True
    if _is_relationship_word_single(b_tokens) and _has_length4_token(a_tokens):
        return True
    return False


# Confidence floor for accepting an LLM merge. Sampling of post-fix
# Pepys merges showed bad merges at 0.7-0.85, good merges at 0.9+.
# 0.85 is a sensible rejection floor without over-pruning legitimate
# fuzzy merges. Tightened with the prompt + name-overlap gate above.
_DEDUPE_CONFIDENCE_FLOOR: float = 0.85

# At or above this confidence, the LLM's judgment overrides the
# deterministic name-overlap gate. The Pepys/bkwp run had the LLM
# propose `Elizabeth St. Michel ↔ "Pepys's wife"` at 0.95 — a
# correct merge that the gate rejected because tokens didn't
# overlap. Trusting high-confidence LLM merges catches these
# semantic-but-not-token-overlap cases. Type-mismatch is NEVER
# bypassed; the LLM has been observed proposing
# `Samuel Pepys (person) ↔ Secretary of the Admiralty (org)` at
# 0.95, which is a structural type error and must always reject.
_DEDUPE_TRUST_HIGH_CONF: float = 0.9


def _format_dedupe_row(
    rec: EntityRecord, *, minimal: bool = False,
) -> str:
    """Render one entity row for the dedupe prompt.

    `minimal=True` drops aliases + description, leaving only the
    identity head (`- {id} | {name} ({type})`). Used for the bottom
    fraction of rows under sample-step retries — the model still sees
    every entity's identity tokens (so it can propose merges across
    the full set), just without the per-row content that amplifies
    output length.

    When `minimal=False`, the aliases segment is omitted entirely if
    the entity has no aliases (no `—` placeholder dead-token)."""
    head = (
        f"- {rec.canonical_id} | {rec.canonical_name} "
        f"({rec.entity_type}) | mentions={int(rec.mention_count or 0)}"
    )
    if minimal:
        return head
    parts = [head]
    if rec.aliases:
        parts.append(f"aliases: {', '.join(rec.aliases[:6])}")
    desc = (rec.description or "").strip() or "(no description)"
    parts.append(f"desc: {desc}")
    return " | ".join(parts)


def _render_dedupe_rows(
    records: list["EntityRecord"], sample_step: int = 0,
) -> str:
    """Render the rows block for the dedupe prompt.

    `sample_step=0` (default) renders every row full-detail.
    `sample_step=N` (N>=1) renders the top `len // 2**N` rows full and
    the bottom rows minimally (identity head only). Sort key is
    `(-mention_count, canonical_name)` — same as the prior drop-half
    sampler — so the keep-set is stable across runs and the cache key
    survives.

    Records are never dropped: the model always sees every entity's
    identity tokens so cross-set merges remain possible. Sample
    retries shed per-row content (descriptions, aliases), not
    records — payload shrinks from ~30-40 tokens/row to ~8-12 for
    the trimmed fraction without losing identity coverage."""
    if not records:
        return ""
    if sample_step <= 0:
        return "\n".join(_format_dedupe_row(r) for r in records)
    sorted_records = sorted(
        records,
        key=lambda r: (-int(r.mention_count or 0), r.canonical_name),
    )
    n = len(sorted_records)
    n_full = max(1, n // (2 ** sample_step))
    lines = [_format_dedupe_row(r) for r in sorted_records[:n_full]]
    lines.extend(
        _format_dedupe_row(r, minimal=True)
        for r in sorted_records[n_full:]
    )
    return "\n".join(lines)






def derive_entities_context(
    entities_output,
    facts_by_topic: dict,
    subject: str,
) -> tuple[str, "str | None", dict[str, str]]:
    """Derive the grounding blocks patterns / insights / actions consume
    from the entities output: ``(resolved_subject, entities_context,
    entities_context_by_topic)``. Extracted from the runner VERBATIM so the
    flat PipelineJob grounds the synthesis stages identically to the legacy
    runner orchestration.

    - ``resolved_subject``: the entities-resolved subject display (unless the
      CLI subject is generic AND the resolved subject is the untrustworthy
      mention-count fallback, in which case the CLI string is kept and the
      untrusted subject is scrubbed from the context block).
    - ``entities_context``: run-wide block (cap ``max(100, x/10)``) for
      insights + actions.
    - ``entities_context_by_topic``: per-topic blocks (cap ``max(50, x/10)``,
      restricted to the topic's referenced entities) for patterns.
    """
    from dataclasses import replace

    _GENERIC_SUBJECTS = {"the author", "subject", "me", ""}
    resolved_subject = subject
    entities_context: str | None = None
    entities_context_by_topic: dict[str, str] = {}
    if entities_output is None:
        return resolved_subject, entities_context, entities_context_by_topic

    subj = entities_output.subject
    is_generic = subject.strip().lower() in _GENERIC_SUBJECTS
    untrustworthy = subj is not None and subj.source == "mention_count_fallback"
    if subj is not None and is_generic and untrustworthy:
        ctx_output = replace(entities_output, subject=None)
    else:
        ctx_output = entities_output
        if subj is not None:
            resolved_subject = subj.display

    x_run = len(ctx_output.entities)
    run_cap = max(100, x_run // 10)
    entities_context = build_context_block(ctx_output, max_entities=run_cap) or None

    topic_to_entity_ids: dict[str, set[str]] = {}
    for e in ctx_output.entities:
        for ref_topic, _idx in e.evidence_fact_refs:
            topic_to_entity_ids.setdefault(ref_topic, set()).add(e.canonical_id)
    for topic in facts_by_topic.keys():
        subset_ids = topic_to_entity_ids.get(topic, set())
        if not subset_ids:
            continue
        topic_cap = max(50, len(subset_ids) // 10)
        block = build_context_block(
            ctx_output, max_entities=topic_cap, entity_subset_ids=subset_ids
        )
        if block:
            entities_context_by_topic[topic] = block
    return resolved_subject, entities_context, entities_context_by_topic


def _filter_dedupe_merges(
    raw_merges: list,
    records: list[EntityRecord],
    mode: Mode,
    model: str,
    last_call_id: str | None = None,
) -> list[dict]:
    """Apply the dedupe merge filters (confidence floor + name-overlap gate +
    cross-type rule) to the LLM's raw merge proposals, returning the kept
    merges. Extracted from ``_llm_dedupe_impl`` so the kernel ENTITY_DEDUPE
    phase reuses the EXACT filter without the legacy cascade. (The legacy
    ``_llm_dedupe_impl`` keeps an inline copy that is deleted at cutover.)"""
    from engine.llm import record_stage_counts as _record_stage_counts

    records_by_id = {r.canonical_id: r for r in records}
    valid_ids = set(records_by_id.keys())
    merges: list[dict] = []
    n_proposed = 0
    n_dropped_conf = 0
    n_dropped_overlap = 0
    n_dropped_type = 0
    for m in raw_merges:
        if not isinstance(m, dict):
            continue
        a = str(m.get("a_id", "")).strip()
        b = str(m.get("b_id", "")).strip()
        if a not in valid_ids or b not in valid_ids or a == b:
            continue
        n_proposed += 1
        try:
            conf = float(m.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        rec_a = records_by_id[a]
        rec_b = records_by_id[b]
        # Confidence floor.
        if conf < _DEDUPE_CONFIDENCE_FLOOR:
            n_dropped_conf += 1
            continue
        # Name-overlap gate. Same-type may bypass at conf ≥ trust-high;
        # cross-type never bypasses (guards the person↔org false-merge class).
        cross_type = rec_a.entity_type != rec_b.entity_type
        overlap_ok = _passes_name_overlap_gate(
            rec_a.canonical_name, rec_b.canonical_name
        )
        if not overlap_ok and (cross_type or conf < _DEDUPE_TRUST_HIGH_CONF):
            if cross_type:
                n_dropped_type += 1
            else:
                n_dropped_overlap += 1
            continue
        merges.append({
            "a_id": a,
            "b_id": b,
            "confidence": conf,
            "synthesized_description": str(
                m.get("synthesized_description", "")
            ).strip(),
        })
    n_dropped = n_dropped_conf + n_dropped_overlap + n_dropped_type
    if n_dropped:
        _llm_record_call_warning(
            "entities_dedupe_merges_filtered",
            mode.value, model, 0, 0, 0,
            note=(f"LLM proposed {n_proposed} merges; kept {len(merges)} "
                  f"after filters (dropped {n_dropped_conf} low-conf, "
                  f"{n_dropped_overlap} same-type-no-overlap, "
                  f"{n_dropped_type} cross-type-no-overlap)"),
        )
    _record_stage_counts(
        last_call_id,
        input={"rows": len(records)},
        output={"merges": len(merges)},
    )
    return merges


def _apply_merges_to_records(
    records: list[EntityRecord],
    merges: list[dict],
    relations: list[RelationEdge],
    mode: Mode,
) -> tuple[list[EntityRecord], list[RelationEdge], dict[str, str]]:
    """Apply a sequence of merges via union-find. Consolidate descriptions
    via three precedence tiers: (1) LLM-supplied synthesized_description
    from a pair merge wins; (2) LLM-supplied self-merge description
    rewrite (a_id == b_id; signals "this row's desc is bloated from an
    earlier fallback concat, here is the compressed replacement") wins
    next; (3) fallback: concatenate distinct non-empty source descriptions
    with " · ". Returns (merged_records, remapped_relations, id_remap)."""
    parent = {r.canonical_id: r.canonical_id for r in records}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Make the higher-mention representative the root for
            # display stability; both refs land under one name.
            recs_by_id = {r.canonical_id: r for r in records}
            ra_rec, rb_rec = recs_by_id[ra], recs_by_id[rb]
            if rb_rec.mention_count > ra_rec.mention_count:
                parent[ra] = rb
            else:
                parent[rb] = ra

    # Track LLM-supplied descriptions so we can prefer them over concat.
    # syn_desc covers pair merges; desc_rewrites covers self-merges
    # (a_id == b_id), which signal a solo description-rewrite (no union)
    # — used to clean up bloated descriptions that came in from upstream
    # concat fallback (e.g. the dominant-narrator path through
    # _deterministic_collapse, which emits empty synth and forces the
    # concat fallback).
    syn_desc: dict[tuple[str, str], str] = {}
    desc_rewrites: dict[str, str] = {}
    for m in merges:
        a, b = m["a_id"], m["b_id"]
        if a == b:
            # Self-merge: description rewrite only, no union.
            if m.get("synthesized_description"):
                desc_rewrites[a] = m["synthesized_description"]
            continue
        union(a, b)
        if m.get("synthesized_description"):
            syn_desc[(a, b)] = m["synthesized_description"]
            syn_desc[(b, a)] = m["synthesized_description"]

    # Group records by representative.
    by_rep: dict[str, list[EntityRecord]] = defaultdict(list)
    for r in records:
        by_rep[find(r.canonical_id)].append(r)

    out: list[EntityRecord] = []
    id_remap: dict[str, str] = {}
    for rep_id, members in by_rep.items():
        if len(members) == 1:
            r = members[0]
            id_remap[r.canonical_id] = r.canonical_id
            if r.canonical_id in desc_rewrites:
                r.description = desc_rewrites[r.canonical_id]
            out.append(r)
            continue
        # Pick canonical: highest mention_count, longest name as tiebreak.
        primary = max(members, key=lambda r: (r.mention_count, len(r.canonical_name)))
        aliases: set[str] = set()
        topics: set[str] = set()
        evidence_refs: list[tuple[str, int]] = []
        seen_refs: set[tuple[str, int]] = set()
        descs: list[str] = []
        roles: list[str] = []
        mention_count = 0
        for m in members:
            id_remap[m.canonical_id] = primary.canonical_id
            aliases.update(m.aliases)
            aliases.add(m.canonical_name)
            topics.update(m.topics)
            for ref in m.evidence_fact_refs:
                if ref not in seen_refs:
                    seen_refs.add(ref)
                    evidence_refs.append(ref)
            if m.description.strip() and m.description not in descs:
                descs.append(m.description)
            if m.role.strip() and m.role not in roles:
                roles.append(m.role)
            mention_count += m.mention_count
        # Synthesized description from LLM wins when this rep had any.
        # Take the first matching synthesized desc among member pairs.
        synth: str | None = None
        for a in [m.canonical_id for m in members]:
            for b in [m.canonical_id for m in members]:
                if a != b and (a, b) in syn_desc:
                    synth = syn_desc[(a, b)]
                    break
            if synth:
                break
        if synth:
            description = synth
        else:
            # No pair-syn: fall back to a self-merge rewrite for any
            # member if one exists; the concat-fallback is the last
            # resort because it produces wall-of-text on near-duplicate
            # sources.
            self_rewrite = next(
                (desc_rewrites[m.canonical_id] for m in members
                 if m.canonical_id in desc_rewrites),
                None,
            )
            if self_rewrite:
                description = self_rewrite
            elif descs:
                description = " · ".join(descs)
            else:
                description = ""
        aliases.discard(primary.canonical_name)
        # Drop aliases that are just the canonical name plus a leading
        # determiner ("the King" when canonical is "King", "a Smith"
        # when canonical is "Smith"). After honorific-only-fallback
        # merges these are extremely common on royals / role-words and
        # add zero new information vs the canonical_name. Compare on
        # the determiner-stripped normalized form.
        primary_canon_norm = _normalize_name_no_determiners(primary.canonical_name)
        aliases = {
            a for a in aliases
            if _normalize_name_no_determiners(a) != primary_canon_norm
        }
        merged = EntityRecord(
            canonical_id=primary.canonical_id,
            canonical_name=primary.canonical_name,
            entity_type=primary.entity_type,
            # Total-order key (raw `a` last). See _materialize_record:
            # without it, aliases tying on (len, lowercased) keep `set`
            # iteration order → PYTHONHASHSEED-randomized → leaks into
            # the dedupe prompt and breaks the LLM cache key across runs.
            aliases=sorted(aliases, key=lambda a: (-len(a), a.lower(), a)),
            role=roles[0] if roles else "",
            description=description,
            mention_count=mention_count,
            topics=sorted(topics),
            evidence_fact_refs=sorted(evidence_refs),
        )
        out.append(merged)

    # Remap relations through id_remap; drop self-loops; bilateral keep
    # (A→B and B→A as separate edges) — that's the brief's spec.
    #
    # Relation-conflict resolution (spec 2026-05-02): when canonical and
    # alias both carry an A→B edge, we keep ONE edge per directed pair.
    # Winner = the (verb, confidence) backed by the most fact citations
    # (tiebreak by highest confidence, then verb alphabetical for
    # determinism). Loser's evidence_fact_refs are unioned under the
    # winner's edge — citations are evidence-preserving so no fact loses
    # its attribution when its source row got merged into an alias.
    bucketed: dict[tuple[str, str], list[RelationEdge]] = defaultdict(list)
    for rel in relations:
        new_from = id_remap.get(rel.from_id, rel.from_id)
        new_to = id_remap.get(rel.to_id, rel.to_id)
        if new_from == new_to:
            continue
        bucketed[(new_from, new_to)].append(RelationEdge(
            from_id=new_from,
            to_id=new_to,
            relation=rel.relation,
            confidence=rel.confidence,
            evidence_fact_refs=list(rel.evidence_fact_refs),
        ))

    remapped: list[RelationEdge] = []
    for (nf, nt), edges in bucketed.items():
        # Winner = most evidence, then highest confidence; ties on both
        # break to the alphabetically-first verb so the choice is
        # deterministic across runs. `min(... -conf, verb)` reads
        # cleaner than negating ord-tuples; `-len(refs)` flips the
        # primary key into min-form too.
        winner = min(
            edges,
            key=lambda e: (
                -len(e.evidence_fact_refs),
                -e.confidence,
                e.relation,
            ),
        )
        unioned: list[tuple[str, int]] = []
        seen_refs: set[tuple[str, int]] = set()
        for e in edges:
            for ref in e.evidence_fact_refs:
                t = tuple(ref) if not isinstance(ref, tuple) else ref
                if t in seen_refs:
                    continue
                seen_refs.add(t)
                unioned.append(t)
        unioned.sort()
        remapped.append(RelationEdge(
            from_id=nf,
            to_id=nt,
            relation=winner.relation,
            confidence=winner.confidence,
            evidence_fact_refs=unioned,
        ))
    return out, remapped, id_remap


def _deterministic_collapse(
    records: list[EntityRecord],
    relations: list[RelationEdge],
    mode: Mode,
) -> tuple[list[EntityRecord], list[RelationEdge]]:
    """Load-bearing deterministic floor: collapse all entities with
    identical (normalized canonical_name, entity_type) into one. Required
    for Phase C's heavy-entity split-by-name to round-trip cleanly —
    clones always share these two fields."""
    by_key: dict[tuple[str, str], list[EntityRecord]] = defaultdict(list)
    for r in records:
        key = (_normalize_name(r.canonical_name), r.entity_type)
        by_key[key].append(r)
    pseudo_merges: list[dict] = []
    for key, recs in by_key.items():
        if len(recs) <= 1:
            continue
        # Chain merges: each rec[i+1] merges into rec[0]; union-find
        # collapses the chain to one rep.
        for r in recs[1:]:
            pseudo_merges.append({
                "a_id": recs[0].canonical_id,
                "b_id": r.canonical_id,
                "confidence": 1.0,
                "synthesized_description": "",
            })
    if not pseudo_merges:
        return records, relations
    merged, remapped, _ = _apply_merges_to_records(
        records, pseudo_merges, relations, mode
    )
    return merged, remapped


# ── Subject argmax ────────────────────────────────────────────────────────────


def _resolve_subject(
    records: list[EntityRecord],
    likelihoods: dict[str, float],
    subject_display: str,
) -> SubjectRef | None:
    """Argmax is_subject_likelihood. Tie-break: mention_count desc,
    alphabetical canonical_name asc. Falls back to mention-count
    fallback (same as old code) when every likelihood is 0/missing.

    `likelihoods` is keyed by canonical_id."""
    if not records:
        return None

    best: EntityRecord | None = None
    best_l: float = -1.0
    for r in records:
        if r.entity_type != "person":
            continue
        l = likelihoods.get(r.canonical_id, 0.0)
        if l > best_l:
            best = r
            best_l = l
            continue
        if l == best_l and best is not None:
            # Tie-break: mention_count desc, alphabetical name asc.
            if (r.mention_count, -ord(r.canonical_name[:1].lower() or "z"[0])) > \
               (best.mention_count, -ord(best.canonical_name[:1].lower() or "z"[0])):
                best = r
    if best is not None and best_l > 0.0:
        return SubjectRef(
            canonical_id=best.canonical_id,
            display=best.canonical_name,
            source="argmax",
        )

    # Mention-count fallback.
    person_recs = [r for r in records if r.entity_type == "person"]
    if not person_recs:
        return None
    person_recs.sort(key=lambda r: (-r.mention_count, r.canonical_name.lower()))
    return SubjectRef(
        canonical_id=person_recs[0].canonical_id,
        display=person_recs[0].canonical_name,
        source="mention_count_fallback",
    )


def _scrub_bundle_subject(out: EntitiesOutput, is_bundle: bool) -> EntitiesOutput:
    """Bundle inputs cannot have a single canonical subject. Drop the
    pick post-hoc; per-file Narrator entities still exist as separate
    records."""
    if is_bundle and out.subject is not None:
        out.subject = None
        for r in out.entities:
            if r.role == "subject":
                r.role = ""
    return out


# ── Public API ────────────────────────────────────────────────────────────────


def _materialize_record(
    g: _Group,
    ann: dict,
    records: list[EntityRecord],
    likelihoods: dict[str, float],
    pre_relations: list[RelationEdge],
    groups_by_gid: dict[str, _Group],
) -> None:
    """Build one EntityRecord from a group + its LLM annotation. Pushes
    onto `records`, fills `likelihoods`, and emits any A→B relations
    into `pre_relations` (where to_id is also resolved to the eventual
    canonical_id of the target group)."""
    canonical_name = ann.get("canonical_name", "").strip() or g.canonical_name
    # Total-order key: (-len, lowercased, raw). The raw-string final
    # component is load-bearing — without it, aliases that tie on
    # (length, lowercased form) — e.g. "W.N.P. BARBELLION" vs
    # "W.N.P. Barbellion" — keep `set` iteration order, which is
    # PYTHONHASHSEED-randomized per process. That leaked into the
    # dedupe prompt bytes and broke the LLM cache key across runs.
    aliases = sorted(
        (a for a in g.aliases if a != canonical_name),
        key=lambda a: (-len(a), a.lower(), a),
    )
    rec = EntityRecord(
        canonical_id=_slugify(canonical_name),
        canonical_name=canonical_name,
        entity_type=g.entity_type,
        aliases=aliases,
        role=ann.get("role", ""),
        description=ann.get("description", ""),
        mention_count=g.mention_count,
        topics=sorted(g.topics),
        evidence_fact_refs=sorted(g.evidence_fact_refs),
    )
    records.append(rec)
    likelihoods[rec.canonical_id] = float(ann.get("is_subject_likelihood", 0.0))
    for rel in ann.get("relations", []):
        target_g = groups_by_gid.get(rel.get("to_id", ""))
        if target_g is None:
            continue
        target_id = _slugify(target_g.canonical_name)
        if target_id == rec.canonical_id:
            continue
        pre_relations.append(RelationEdge(
            from_id=rec.canonical_id,
            to_id=target_id,
            relation=str(rel.get("verb", "")).strip().lower().replace(" ", "_"),
            confidence=float(rel.get("confidence", 1.0)),
            evidence_fact_refs=[],   # filled later from relation_evidence map
        ))


def _disambiguate_ids(records: list[EntityRecord]) -> None:
    """Suffix duplicate canonical_ids with `-2`, `-3`, … in discovery order.

    Note: Phase D's deterministic collapse runs AFTER this and merges
    records that should share an id (same normalized name + type),
    so any -2/-3 suffix surviving past Phase D is genuinely two
    different entities (e.g. two different "Smith"s)."""
    seen: dict[str, int] = {}
    for r in records:
        base = r.canonical_id
        n = seen.get(base, 0) + 1
        seen[base] = n
        if n > 1:
            r.canonical_id = f"{base}-{n}"


# ── Context block for downstream stages ───────────────────────────────────────


def build_context_block(
    output: EntitiesOutput,
    max_entities: int = 20,
    entity_subset_ids: set[str] | None = None,
) -> str:
    """Compact ``Entities reference`` block injected into Patterns /
    Insights / Actions prompts. Empty output → empty string.

    `max_entities` is the cap on the rendered entity list. Callers
    compute it stage-aware: `max(100, x/10)` for the run-scoped block
    consumed by insights + actions; `max(50, x/10)` for per-topic
    blocks consumed by patterns. `x` is the population at the scope
    (run-wide entity count for insights/actions; distinct entities
    referenced in a topic's fact set for patterns).

    `entity_subset_ids` (when set) restricts the candidate set to
    entities whose canonical_id is in the subset — patterns uses this
    to render a per-topic slice of the global inventory. `None` means
    run-wide. Relations are filtered to the same subset so the per-
    topic block stays internally consistent (no dangling endpoints).

    Entities are sorted by mention count descending, then by
    canonical_id for stability, and capped at `max_entities`. The
    omitted-tail tally reports the remaining count after the subset
    filter is applied, so the LLM sees an accurate "more omitted"
    signal scoped to what was actually relevant."""
    if not output.entities:
        return ""

    candidates = output.entities
    if entity_subset_ids is not None:
        candidates = [e for e in candidates if e.canonical_id in entity_subset_ids]
    if not candidates:
        return ""

    candidates = sorted(
        candidates,
        key=lambda e: (-int(e.mention_count or 0), e.canonical_id),
    )

    lines: list[str] = ["Entities reference:"]
    if output.subject is not None:
        lines.append(f"  subject: {output.subject.display} ({output.subject.canonical_id})")
    lines.append("  entities:")

    shown = candidates[:max_entities]
    for e in shown:
        role_part = f"[{e.role}] " if e.role else ""
        desc = (e.description or "").strip() or "(no description)"
        aliases = ""
        if e.aliases:
            aliases = f" aliases: {', '.join(e.aliases[:4])}"
        lines.append(
            f"    - {e.canonical_id} {role_part}({e.entity_type}, "
            f"mentions={e.mention_count}){aliases} — {desc}"
        )
    if len(candidates) > max_entities:
        lines.append(f"    … ({len(candidates) - max_entities} more entities omitted)")

    if output.relations:
        if entity_subset_ids is not None:
            shown_rels = [
                r for r in output.relations
                if r.from_id in entity_subset_ids and r.to_id in entity_subset_ids
            ]
        else:
            shown_rels = list(output.relations)
        if shown_rels:
            lines.append("  relations:")
            for r in shown_rels[:40]:
                lines.append(
                    f"    - {r.from_id} --{r.relation}--> {r.to_id} "
                    f"(conf {r.confidence:.2f})"
                )
    return "\n".join(lines)
