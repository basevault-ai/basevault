"""The single content-free shareable-diagnostics emitter.

The trust contract is a LOCATION, not a per-file judgment. Everything
the app writes under ``~/.basevault/shareable/`` is anonymized BY
CONSTRUCTION, so a user (or the director) can hand over a file from
there with zero redaction review. The directory *is* the boundary.

This module is the only thing in the tree that may name a path under
``shareable/``. It enforces the guarantee by construction, not by the
folder name, on four independent legs:

1. **Single writer.** ``_shareable_root()`` is private and is the sole
   site in the repo that joins ``shareable``. ``emit()`` is the sole
   public write entry. The enforcement test pins both.
2. **Typed input.** ``emit()`` accepts only a frozen ``Marker`` whose
   every field is ``int | float | bool | None``, an ISO-Z timestamp
   string, a closed-vocabulary enum, or a list / nested frozen marker
   of those. Free text is *unrepresentable in the input type* — there
   is no ``dict`` / ``str`` blob / ``**kwargs`` channel.
3. **Runtime assert.** Before a byte is written, every leaf is checked
   against the closed content-free type. A violation raises; a leak is
   a crash, never a silent write.
4. **Never the raw record id.** The one content-bearing retrieval field
   (``StoredRecord.record_id`` — embeds ``file_id`` / entity canonical
   name / ``topic`` for several kinds, see ``embeddings.py``) is never
   carried; per-candidate rows use a marker-local positional ordinal.

There is exactly ONE file per perma-id (per conversation / per run):
``<iso-z>-<perma-id>-anonymized.yaml`` — a YAML multi-document stream.
A small header document (``schema_version``/``perma_id``/``stream``/
``created_at``) is written once; every later turn appends one more
``---``-separated YAML document. YAML multi-doc is both append-friendly
(a turn = concatenate another ``---`` document — a single O(1) atomic
``O_APPEND``, no read-modify-write, emit never reads any file back) AND
human-legible as-is — no closing bracket, no trick. A parser reads it
with ``yaml.safe_load_all``; a person just reads it. Date-first so the
dir sorts chronologically (same convention as the run / conversation
dirs); the timestamp is the file's creation time, minted once on first
emit and stable across turns. The sole retrieval path is
``grep <4 letters> shareable/`` -> the one file (the perma-id is
practically unique; no resolver). The
perma-id is read through ``resolve_perma_id()`` (the single seam onto
the perma-id model — consumed verbatim from the persisted ``short_id``,
never minted or regenerated here).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path

from engine.common.dates import ISO_Z_RE as _ISO_Z_RE, now_iso_z as _now_iso_z
from engine.common.status import Outcome, StageToken
# Re-exported: shareable_markers (and any future caller that drives the
# marker tree) imports ``stage_token`` from here so the closed-vocab
# entry point stays in one place — the ``as stage_token`` idiom marks
# the import as an intentional re-export so ruff doesn't flag F401.
from engine.common.status import stage_token as stage_token

SCHEMA_VERSION = 1

# The short-id alphabet (byte-identical to runner.py
# ``_SHORT_ID_CHARS`` / lib.rs ``ALPHABET``). A perma-id that is not
# exactly 4 of these is rejected so a bogus key never creates a junk
# file under the trust dir.
_PERMA_ID_RE = re.compile(r"^[abcdefghjkmnpqrstuvwxyz23456789]{4}$")

# Structural identity, never vault text. The runtime guard accepts a
# str leaf iff it is an ISO-Z timestamp (``_ISO_Z_RE`` re-exported
# from ``common.dates``), a closed-enum value, or a zero-padded
# 4-digit call id (the runner's standard short-id form;
# ``f"{counter:04d}"`` in llm.py — purely structural, monotonic, no
# vault content possible).
_CALL_ID_RE = re.compile(r"^\d{4}$")

# Sidecar process id — 16 lowercase hex chars, minted once per chatbot
# sidecar process at startup and stamped on every event/call from that
# process. The chatbot sidecar today uses ``uuid.uuid4().hex[:16]``
# (chatbot_sidecar.py:_start_session); ``os.urandom(8).hex()`` is the
# equivalent shape. Either way the contract is "16 lowercase hex
# digits" — structurally random, no vault-text bits encodable. Surfaced
# on the chat marker so a reader can distinguish "same conversation,
# new sidecar process" (turn_index resets per process) from a fresh
# convo; without this, the 16 sidecar respawns observed in a real
# session look indistinguishable from spurious turn-index discontinuity.
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{16}$")


# ── closed-vocabulary enums ───────────────────────────────────────────
#
# Every str leaf in a marker is either an ISO-Z timestamp or one of
# these values. ``_ENUM_VOCAB`` (built below) is the closed set the
# runtime guard checks membership against; nothing else may be a str.


class Stream(str, Enum):
    """The two shareable streams (also the subdir name)."""
    CHAT = "chat-diagnostics"
    RUN = "run-diagnostics"


class RecordKind(str, Enum):
    """Mirrors ``rag_vector_store.RECORD_KINDS`` — a closed set."""
    document = "document"
    chunk = "chunk"
    fact = "fact"
    entity = "entity"
    pattern = "pattern"
    insight = "insight"
    action = "action"


class Provider(str, Enum):
    tee = "tee"
    local = "local"
    other = "other"


class FinishReason(str, Enum):
    stop = "stop"
    length = "length"
    content_filter = "content_filter"
    tool_calls = "tool_calls"
    error = "error"
    other = "other"
    none = "none"


class RetryClass(str, Enum):
    """The ` - retry/<class>` tail of a retry call's category."""
    load = "load"
    sizing = "sizing"
    length = "length"
    interrupted = "interrupted"
    skipped = "skipped"
    other = "other"
    none = "none"


class RetryTransform(str, Enum):
    """Accumulated structural-transform suffixes (``/half-N`` /
    ``/sample-N`` / ``/reasoning-off``) — the KIND only, never the
    free topic/filename prefix of the category string."""
    half = "half"
    sample = "sample"
    reasoning_off = "reasoning_off"


class CategoryToken(str, Enum):
    """Closed call-category vocabulary."""
    chatbot_converse = "chatbot_converse"
    chatbot_rerank = "chatbot_rerank"
    chatbot_grounded_decision = "chatbot_grounded_decision"
    chatbot_answer = "chatbot_answer"
    embed = "embed"
    other = "other"
    none = "none"


class RetrieveSkippedReason(str, Enum):
    """Why a lookup turn's retrieval-diagnostic tier (the ``hops``
    block) is absent or empty. Stamped on every chat marker; ``none``
    covers both healthy retrieval and pure-conversational turns (no
    skip occurred). The other values distinguish failure modes that
    today render visually identical in the shareable yaml — the gap
    this enum closes."""
    none = "none"
    no_bound_run = "no_bound_run"
    empty_store = "empty_store"
    degenerate_query = "degenerate_query"
    sink_never_called = "sink_never_called"


class HopOutcome(str, Enum):
    """How one ReAct-loop LLM call resolved. ``chatbot_grounded_decision``
    calls can emit either a tool call (continue the loop) or prose (the
    answer — voluntary finalize); ``chatbot_answer`` (=grounded_final)
    is structurally forced prose. The ``category`` on the joined LlmCall
    tells the reader which PERSONA fired; this enum tells what came
    BACK. The combination disambiguates voluntary vs forced finalize."""
    tool_call = "tool_call"
    prose_answer = "prose_answer"
    # Model emitted JSON shaped like a tool call but ``validate_tool_call``
    # rejected it (unknown tool, missing field, bad type). The
    # dispatcher records the invalid attempt into NOTES so the next hop
    # can see what was rejected.
    invalid_tool_call = "invalid_tool_call"
    # Model OPENED a ``{"tool": ...}`` object but it was not valid JSON,
    # so ``parse_tool_call`` could not extract it at all. Distinct from
    # ``invalid_tool_call`` (parsed fine, failed validation): here the
    # bytes never parsed. The loop feeds the failure back and retries on
    # the next hop (bounded by ``MAX_PARSE_RETRIES``) instead of
    # surfacing the raw unparseable JSON to the user as the answer.
    malformed_tool_call = "malformed_tool_call"


_CATEGORY_NAMES = frozenset(
    m.value
    for m in CategoryToken
    if m not in (CategoryToken.other, CategoryToken.none)
)


def category_token(raw: str | None) -> CategoryToken:
    if not raw:
        return CategoryToken.none
    s = str(raw).lower()
    return CategoryToken(s) if s in _CATEGORY_NAMES else CategoryToken.other


class SizeBucket(str, Enum):
    """Fixed char-length buckets for the size histograms."""
    lt_200 = "<200"
    b_200_500 = "200-500"
    b_500_1k = "500-1k"
    b_1k_2k = "1k-2k"
    gte_2k = ">=2k"


# Known model-id family tokens. The live model id is mapped through
# ``model_token()`` to one of these (or ``other``) so the marker never
# carries a raw, free-form model string.
class ModelToken(str, Enum):
    kimi_k2 = "kimi-k2"
    minimax = "minimax"
    gpt_oss = "gpt-oss"
    gemma = "gemma"
    llama = "llama"
    other = "other"
    none = "none"


# Substring -> token. First match wins; order is most-specific-first
# only where families could overlap. Matching is case-insensitive on
# the lowercased model id, so this stays robust to provider-prefixed
# ids (``moonshotai/Kimi-K2.6-TEE`` -> ``kimi-k2``).
_MODEL_SUBSTR = (
    ("kimi", ModelToken.kimi_k2),
    ("minimax", ModelToken.minimax),
    ("gpt-oss", ModelToken.gpt_oss),
    ("gemma", ModelToken.gemma),
    ("llama", ModelToken.llama),
)


def model_token(raw: str | None) -> ModelToken:
    """Map a live model id to a closed family token. Unknown -> ``other``;
    empty/None -> ``none``. The raw id never enters a marker."""
    if not raw:
        return ModelToken.none
    low = str(raw).lower()
    for needle, tok in _MODEL_SUBSTR:
        if needle in low:
            return tok
    return ModelToken.other


def provider_token(raw: str | None) -> Provider:
    if not raw:
        return Provider.other
    low = str(raw).lower()
    if "tee" in low:
        return Provider.tee
    if "local" in low or "ollama" in low:
        return Provider.local
    return Provider.other


# The closed str vocabulary the runtime guard enforces. Every str leaf
# that reaches the marker is checked against this set (modulo ISO-Z
# timestamps + the 4-digit call-id form); no free string can ever pass.
# Each new enum added below must be appended here or it will be
# rejected as a content-free violation, which is the intent (the union
# is single-source).
_ENUM_VOCAB: frozenset[str] = frozenset(
    m.value
    for enum_cls in (
        Stream, RecordKind, Outcome, Provider, SizeBucket, ModelToken,
        StageToken, CategoryToken, FinishReason, RetryClass,
        RetryTransform, RetrieveSkippedReason, HopOutcome,
    )
    for m in enum_cls
)


# ── frozen marker dataclasses ─────────────────────────────────────────
#
# Free text is unrepresentable: every field is a content-free leaf, a
# list of them, or a nested frozen marker. There is deliberately no
# dict / free-str field anywhere in this tree.


@dataclass(frozen=True)
class LlmCall:
    """One ``llm-calls.jsonl`` record rendered content-free. Never the
    prompt, response, error text, or template hash.

    Field order is the YAML emit order. Identity fields lead so a
    reader can correlate a sampled call back to its row in the run's
    ``llm-calls.jsonl`` / ``llm-payloads.jsonl`` at a glance — call_id
    is the join key across all three streams."""
    # Identity — first in the YAML for at-a-glance correlation. The
    # zero-padded 4-digit form (\"0041\", not 41) matches the runner's
    # standard short-id format used in llm-calls.jsonl / llm-payloads.jsonl,
    # so a reader can grep for the same call across all three streams
    # without format gymnastics.
    call_id: str | None
    retry_of_call_id: str | None
    stage: StageToken
    category: CategoryToken
    model: ModelToken
    provider: Provider
    reasoning: bool
    outcome: Outcome
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    content_tokens: int | None
    total_tokens: int | None
    duration_ms: float | None
    ttft_ms: float | None
    max_tokens_reserved: int | None
    attempt: int
    is_retry: bool
    parse_error: bool
    started_at: str | None
    ended_at: str | None
    cached: bool | None
    finish_reason: FinishReason
    retry_class: RetryClass
    retry_transforms: tuple[RetryTransform, ...]


@dataclass(frozen=True)
class CandidateRow:
    """One retrieved-candidate row. ``position`` is a marker-local
    ordinal minted at emit time, NOT ``record_id`` (which carries vault
    text for chunk/entity/fact/pattern kinds)."""
    rank: int
    kind: RecordKind
    position: int
    distance: float
    rerank_score: float | None


@dataclass(frozen=True)
class OutcomeCount:
    """One closed-enum-keyed row of the per-stage outcome distribution."""
    outcome: Outcome
    count: int


@dataclass(frozen=True)
class RetryClassCount:
    """One closed-enum-keyed row of the per-stage retry-class
    distribution. ``none`` rows are omitted from the tuple to keep the
    YAML lean — the absence of a row means count 0."""
    retry_class: RetryClass
    count: int


@dataclass(frozen=True)
class StageStat:
    # Identity. ``stage`` is the closed-enum stage token (never a free
    # string); ``stage_index`` is reassigned post-sort so the printed
    # ordinal matches the emit position (canonical pipeline order).
    stage: StageToken
    stage_index: int
    # ``present`` means the stage actually RAN in this run — either it
    # fired at least one LLM call OR its ``phase_1_marker.json``
    # completion sentinel is on disk. A scaffolding dir with no calls
    # and no marker is NOT present (e.g. an actions stage skipped
    # because no patterns produced inputs).
    present: bool
    completed: bool
    # Derived: stage completed AND zero non-success calls. A stage
    # that fired zero calls is ``success`` iff it is also ``completed``
    # — empty work is not failure.
    success: bool
    # Wall-clock span across the stage's LLM calls, derived from the
    # per-call ISO-Z timestamps in the run's ``llm-calls.jsonl``. None
    # when the stage fired no calls (or every call's timestamp was
    # malformed and got dropped by the ISO-Z guard).
    started_at: str | None
    ended_at: str | None
    wall_ms: float | None
    # Per-call rollup, aggregated from the stage's slice of the run's
    # ``llm-calls.jsonl``. Token sums are 0 when a stage's calls don't
    # carry token counts (e.g. embeddings). ``success_count`` includes
    # any ``success*`` outcome (plain success, success_empty,
    # success_sampled, success_reasoning_off); the breakdown lives in
    # ``outcome_dist``.
    call_count: int
    success_count: int
    failure_count: int
    cache_hit_count: int
    retry_count: int
    prompt_tokens_sum: int
    completion_tokens_sum: int
    # The stage's produced-item count, read content-free from its
    # phase_1_marker.json (an integer only — never any name/text).
    item_count: int | None
    # Latency distribution across the stage's calls. None when the
    # stage fired no calls, or when no call carried the corresponding
    # field (ttft is null on non-streaming stages like embeddings).
    duration_ms_p50: float | None
    duration_ms_p95: float | None
    duration_ms_mean: float | None
    ttft_ms_p50: float | None
    ttft_ms_p95: float | None
    # Categorical distributions — closed-enum-keyed maps. Rows with
    # count 0 are omitted; an absent row means zero. ``outcome_dist``
    # is the canonical breakdown — every value mirrors a runner
    # ``OUTCOME_*`` label (see ``Outcome``).
    outcome_dist: tuple[OutcomeCount, ...]
    retry_class_dist: tuple[RetryClassCount, ...]
    # Per-stage LLM calls. For low-volume stages (vision /
    # entities_dedupe / patterns / insights / actions) this is every
    # call the stage fired. For high-volume stages (extraction /
    # entities / embeddings) it is the deterministic
    # first/median/last-by-started_at sample of successes (cap 3) plus
    # ALL failures — failures are never sampled. ``successful_calls_sampled``
    # disambiguates so a reader never mistakes a sample for a complete
    # list: when True, ``call_count`` (total) vs ``len(calls)`` (shown)
    # makes the elision unambiguous.
    calls: tuple[LlmCall, ...]
    successful_calls_sampled: bool


@dataclass(frozen=True)
class SizeHistBucket:
    bucket: SizeBucket
    count: int


@dataclass(frozen=True)
class KindCount:
    kind: RecordKind
    count: int


@dataclass(frozen=True)
class QueryStats:
    char_len: int
    token_count: int
    embed_dim: int
    embed_norm: float
    all_zero: bool
    non_finite: bool


@dataclass(frozen=True)
class RetrievalShape:
    kind_filter: RecordKind | None
    k_requested: int
    k_returned: int
    empty_retrieval: bool
    degenerate_query: bool
    all_tied: bool


@dataclass(frozen=True)
class DistanceDist:
    dist_min: float | None
    dist_max: float | None
    dist_mean: float | None
    all_equal: bool


@dataclass(frozen=True)
class RerankInfo:
    rerank_applied: bool
    rerank_model: ModelToken
    rerank_parse_ok: bool
    fell_back_to_embed_order: bool


@dataclass(frozen=True)
class EmbeddingStats:
    records_embedded: int | None
    embed_dim: int | None
    # ``None`` for the no-signal case (a canceled / unfinished run
    # produced no embedded records); ``_to_jsonable`` then omits the
    # field. Misclassifying no-signal as ``failed_other`` would skew
    # any downstream that treats ``failed*`` as a real failure.
    embed_outcome: Outcome | None
    embed_call_count: int | None
    batch_size: int | None


@dataclass(frozen=True)
class LlmCallsBlock:
    """A per-scope LLM-call rollup. For the chat stream this carries the
    turn's call list (one or two calls per turn — small, fully enumerated).
    For the run stream the run-level block keeps only the four-int
    aggregate; the per-call detail lives under each ``StageStat.calls``
    instead (see the run-marker docstring)."""
    call_count: int
    total_prompt_tokens: int
    total_completion_tokens: int
    wall_ms_total: float
    # Empty for the RUN scope by design — every call has a home under
    # its stage entry, sampled (extraction/entities/embeddings) or fully
    # enumerated (everything else). Populated for the CHAT scope (one
    # or two calls per turn, no scope for bloat).
    calls: tuple[LlmCall, ...]


@dataclass(frozen=True)
class StoreStats:
    """Per-turn snapshot of the bound corpus store. ``total_records=0``
    with a non-None ``bound_run`` is the empty-store case that today
    looks identical to a real-retrieval miss; the kind breakdown also
    surfaces a store growing across a session as ingests land."""
    total_records: int
    records_by_kind: tuple[KindCount, ...]


@dataclass(frozen=True)
class ScoreSamples:
    """Three-point summary of a per-lookup distance distribution:
    closest (best/smallest distance), farthest (worst kept), and a
    middle anchor. Together they pin the distribution's shape in three
    floats — tighter than min/max/mean and order-preserving (you can
    tell at a glance whether the lookup returned tightly-clustered hits
    or a spread). ``None`` for filter-only lookups (no distances) and
    for query-bearing lookups with ``k_returned == 0``."""
    closest: float
    middle: float
    farthest: float


@dataclass(frozen=True)
class LookupShape:
    """One entry of a search tool call's ``lookups`` array. Pairs the
    model's request (the knobs it chose) with the dispatch outcome (what
    the store actually returned) so a reader can attribute "which
    lookup contributed what" within a multi-lookup tool call. Every
    field is content-free: closed-enum kinds, numeric counts, content-
    free embed signals. Never the query text, never an exact-match
    substring, never an anchor record_id."""
    # The model's request — straight reads from the ``chatbot_tools.Lookup``
    # dataclass (validated upstream). Each captures one knob the model
    # explicitly chose, not an inferred property.
    entry_types: tuple[RecordKind, ...]
    k_requested: int
    has_neighbor_count: int
    # Kind-half of each ``(kind, record_id)`` anchor — content-free. The
    # record_id half is vault text (chunk id = ``file_id@offset``, entity
    # canonical names, etc.) and is dropped entirely; only the count
    # plus this closed-enum tuple of kinds surfaces.
    has_neighbor_kinds: tuple[RecordKind, ...]
    exact_match_count: int
    query_present: bool
    # Embed signal — populated only when ``query_present``. Same content-
    # free leaves as the previous flat ``QueryStats`` block, now per-
    # lookup so a reader can see which lookup's vector degenerated.
    query_char_len: int | None
    embed_dim: int | None
    embed_norm: float | None
    embed_all_zero: bool | None
    embed_non_finite: bool | None
    # Dispatch outcome — what ``_dispatch_search``'s iteration loop saw
    # for THIS lookup. Captured per-lookup (not as a global aggregate)
    # via a ``per_lookup_diag: list[dict]`` the dispatcher emits beside
    # its existing global tallies; see ``chatbot_dispatch.py``.
    degenerate: bool
    tied: bool
    k_returned: int
    junk_dropped: int
    kind_counts: tuple[KindCount, ...]
    score_samples: ScoreSamples | None


@dataclass(frozen=True)
class HopMarker:
    """One LLM call's full ReAct-loop entry. The chat surface fires
    1..MAX_HOPS+1 calls per turn under #635-slice-2: an initial
    ``chatbot_converse`` decision, 0..MAX_HOPS ``chatbot_grounded_decision``
    hops (each of which can dispatch a tool call OR finalize with prose),
    and at most one forced ``chatbot_answer`` (grounded_final). The
    ``category`` of each call lives on the joined ``LlmCall``; this
    marker carries the per-hop ReAct state and (when the hop dispatched)
    the per-lookup detail."""
    # Join key. The 4-digit zero-padded form matches the runner's
    # standard short-id format, so a reader correlates each hop to its
    # row in ``llm_calls.calls`` (and to llm-calls.jsonl /
    # llm-payloads.jsonl) by call_id without format gymnastics.
    call_id: str
    hop_outcome: HopOutcome
    # Whether the call's user-visible answer streamed through the gate
    # to the UI (prose answers) vs got suppressed at the tool-call
    # shape detection (``_StreamGate.streamed``). False for tool_call
    # outcomes (gate buffered the JSON) and for the rare voluntary-
    # finalize-but-non-streaming-provider case (content-anchored
    # fallback emits the answer in one chunk after the call).
    streamed_to_user: bool
    # ReAct budget signal that already appears in the prompt's NOTES
    # block (``Lookups remaining: N of 4``). ``MAX_HOPS - lookups_done``
    # at the moment this hop fires. ``None`` on the first decision call
    # (budget not yet exposed) and on the forced grounded_final hop (no
    # budget — that call IS the budget exhaustion).
    lookups_remaining_in_budget: int | None
    # How many tool-call attempts this turn had ALREADY emitted before
    # this hop fired — the value the prompt's NOTES "Previous attempts:
    # 1, 2, 3" derives from (``len(previous_attempts)`` in
    # ``chatbot_turn``). Surfaces patterns like "model re-emitted the
    # same JSON 3 times before finalizing".
    previous_attempts_count: int
    # Per-hop wall-clock. ``store_open_latency_ms`` is real per-hop
    # today: ``chatbot_turn.py``'s ReAct loop opens a fresh
    # ``open_store(ctx.store_path)`` context inside each hop iteration.
    # Hoisting the open to per-turn (the db is immutable within a turn)
    # is a separate optimization that would change these latencies'
    # semantics to "first-hop only / null after".
    store_open_latency_ms: float | None
    dispatch_latency_ms: float | None
    # Running cumulative size of the turn's accumulator (the ``acc`` in
    # ``chatbot_turn.run``, capped at ``ACCUMULATOR_CAP``) AFTER this
    # hop's ``_accumulate`` merged in. Distinct from each lookup's
    # k_returned and from the dispatcher's per-call ``union_size``: this
    # is the post-dedup-across-hops total the next hop's grounded prompt
    # will enumerate as ``[1]..[N]``. ``None`` when the hop didn't
    # dispatch (decision, prose-answer hops, grounded_final).
    union_size_after: int | None
    # Populated only when ``hop_outcome == tool_call`` AND the call
    # validated cleanly. Length 1..MAX_LOOKUPS (=5). One entry per
    # lookup in the model's array — even when the model emitted the
    # slice-1 single-lookup shape, the validator coerces it to a
    # one-element list, so this is uniformly the array view.
    lookups: tuple[LookupShape, ...]


@dataclass(frozen=True)
class Marker:
    """Typed base — the only thing the two stream markers share is the
    schema version. The streams are deliberately DISJOINT: a field
    lives in exactly one of them, never both (no duplicated info)."""
    schema_version: int


@dataclass(frozen=True)
class ChatMarker(Marker):
    """One conversation turn — everything chat, nothing run/corpus.
    Appended per turn to ``chat-diagnostics/<…>-<convo-id>-…yaml``.
    The corpus it queried is identified ONLY by its run perma-id (see
    ``RunMarker``); its structural stats are NOT duplicated here.

    Schema rev (#781 final): the previous flat ``query`` /
    ``shape`` / ``distances`` / ``rerank`` / ``candidates`` block
    collapsed a multi-hop turn's N dispatch results into one (the last
    hop overwrote the rest via ``ctx.diag_box.update``). The post-#799
    ReAct architecture fires up to MAX_HOPS+1 LLM calls per turn, each
    of which may dispatch its own search-lookups array, so the flat
    block was structurally incapable of representing what the turn
    actually did. Replaced with ``hops: tuple[HopMarker, ...]`` — one
    entry per LLM call, each carrying its own per-lookup detail when
    the call dispatched. The reader walks the hops list and
    reconstructs the ReAct trace; per-hop entries join to
    ``llm_calls.calls[N]`` by ``call_id``."""
    ts: str
    turn_index: int
    # Sidecar process id — 16-hex closed-form, content-free. A single
    # conversation can span many sidecar processes (the persistent
    # chatbot sidecar respawns on Stop / re-warm / crash / config
    # change); ``turn_index`` is per-process so it resets across
    # respawns. Surfacing the session id here lets a reader see "same
    # convo, fresh process" rather than mistaking a turn-index
    # discontinuity for missing data.
    session_id: str
    lookup_fired: bool
    llm_calls: LlmCallsBlock
    # The multi-hop trace — one entry per LLM call this turn fired,
    # in order. Empty tuple on pure-conversation turns (no lookup
    # fired) — the writer's "empty list is omitted" guarantee keeps
    # those turns lean.
    hops: tuple[HopMarker, ...]
    # Per-turn state that distinguishes the empty-binding / empty-store
    # cases from a real-retrieval miss (#781). Carried through verbatim
    # from the prior schema rev — these are turn-level (not per-hop)
    # and remain content-free.
    retrieve_skipped_reason: RetrieveSkippedReason = RetrieveSkippedReason.none
    bound_run: str | None = None
    store_stats: StoreStats | None = None
    history_turn_count: int = 0
    resources_emitted_count: int = 0


@dataclass(frozen=True)
class RunMarker(Marker):
    """One corpus run — everything run/corpus, nothing chat. Written
    ONCE per run perma-id to ``run-diagnostics/<…>-<run-id>-…yaml``
    (the corpus is immutable for a run, so emit is idempotent and never
    re-appended per chat turn — no duplicated info).

    ``llm_calls`` here is a slim run-level rollup of the RUN's OWN
    pipeline-stage calls (read from the run's ``llm-calls.jsonl``:
    extraction / entities / patterns / insights / actions / embeddings)
    — NOT the chatbot's. Only the four-int aggregate (count + token
    sums + wall_ms_total) is kept at the run scope; per-call detail
    lives under each ``StageStat.calls`` instead, so the reader skims
    one block per stage rather than a single flat tuple of every call
    in the run. High-volume stages
    (extraction / entities / embeddings) carry a deterministic sample
    of successes plus ALL failures; everything else carries every call.
    ``StageStat.successful_calls_sampled`` disambiguates which is which."""
    created_at: str
    embedding: EmbeddingStats | None
    record_counts_by_kind: tuple[KindCount, ...]
    chunk_size_hist: tuple[SizeHistBucket, ...]
    file_size_hist: tuple[SizeHistBucket, ...]
    stages: tuple[StageStat, ...]
    llm_calls: LlmCallsBlock | None


# ── runtime content-free guard ────────────────────────────────────────


class ContentFreeViolation(AssertionError):
    """Raised when a marker leaf is not content-free. By construction
    this should be unreachable from typed call sites; it is the
    crash-not-silent-write backstop."""


def _assert_content_free(value: object, path: str = "$") -> None:
    """Recursively assert every leaf is content-free. Allowed leaves:
    ``None``/``bool``/``int``/``float``; an ``Enum`` whose value is in
    the closed vocabulary; a ``str`` that is an ISO-Z timestamp or a
    closed-vocabulary value; a list/tuple of those; or a nested frozen
    dataclass marker. Anything else (notably any free ``str`` or any
    ``dict``) raises."""
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, Enum):
        if value.value not in _ENUM_VOCAB:
            raise ContentFreeViolation(
                f"{path}: enum value {value!r} not in closed vocabulary"
            )
        return
    if isinstance(value, str):
        # 4-digit call-id form is allowed ONLY on the two call-id
        # fields — narrowing the bypass so e.g. ``ChatMarker.ts="1234"``
        # still fails. The check looks at the path's trailing field
        # name; defense-in-depth on top of the typed schema (which
        # already constrains where strings can land). Same shape for
        # the 4-letter perma-id leaf on ``ChatMarker.bound_run``: only
        # accepted when the path ends in ``.bound_run``, so a
        # perma-id-shaped value smuggled into any other str field
        # still crashes (e.g. ``ChatMarker.ts="ab3k"`` would fail).
        # And for the 16-hex session_id on ``ChatMarker.session_id``,
        # same path-narrowed pattern.
        is_call_id_field = (
            path.endswith(".call_id")
            or path.endswith(".retry_of_call_id")
        )
        is_bound_run_field = path.endswith(".bound_run")
        is_session_id_field = path.endswith(".session_id")
        if _ISO_Z_RE.match(value) or value in _ENUM_VOCAB:
            return
        if is_call_id_field and _CALL_ID_RE.match(value):
            return
        if is_bound_run_field and _PERMA_ID_RE.match(value):
            return
        if is_session_id_field and _SESSION_ID_RE.match(value):
            return
        raise ContentFreeViolation(
            f"{path}: free string is not a content-free leaf "
            f"(only ISO-Z timestamps, closed-enum values, 4-digit "
            f"call ids on call_id / retry_of_call_id fields, "
            f"4-letter perma-ids on bound_run fields, or 16-hex "
            f"session ids on session_id fields allowed)"
        )
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _assert_content_free(item, f"{path}[{i}]")
        return
    if is_dataclass(value) and not isinstance(value, type):
        for f in fields(value):
            _assert_content_free(getattr(value, f.name), f"{path}.{f.name}")
        return
    raise ContentFreeViolation(
        f"{path}: leaf of type {type(value).__name__} is not content-free"
    )


def _to_jsonable(value: object) -> object:
    """Render for serialization, succinctly: a field that is ``None``
    or an empty list is OMITTED entirely (users skim these before
    sharing — no ``key: null`` / empty-block noise). This is purely
    cosmetic and runs AFTER the content-free guard, so it cannot affect
    the privacy guarantee."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        out: dict = {}
        for f in fields(value):
            v = getattr(value, f.name)
            if v is None or (isinstance(v, (list, tuple)) and not v):
                continue
            out[f.name] = _to_jsonable(v)
        return out
    return value


# ── the single guarded writer ─────────────────────────────────────────


def _state_root() -> Path:
    """``~/.basevault`` — mirrors the Rust ``state_root()``."""
    return Path(os.path.expanduser("~")) / ".basevault"


def _shareable_root() -> Path:
    """``~/.basevault/shareable`` — top-level, sibling of
    ``logs/``/``cache/``/``chats/``. THE ONLY place in the tree that
    joins ``shareable``. Never wiped by the daily ``cache/`` wipe (that
    wipe operates strictly on ``cache_root()``; a sibling dir is
    structurally out of its reach). The enforcement test pins this as
    the sole join site."""
    return _state_root() / "shareable"


def resolve_perma_id(state_dir: Path | str | None) -> str | None:
    """The single seam onto the perma-id model.

    Resolves the immutable 4-letter perma-id for a run dir or a
    conversation dir, mirroring the perma-id model's own resolution
    order WITHOUT minting or regenerating: the persisted ``short_id``
    field in the dir's sidecar json (run ``config.json`` / chat
    ``transcript.json``), else the canonical ``<iso-z>-<short_id>``
    dir-name suffix. Runs and conversations both carry the 4-letter in
    BOTH the dir name and the sidecar json, so both sides resolve; a
    legacy dir that predates the scheme (no sidecar field, no 4-letter
    suffix) resolves to ``None`` and the caller best-effort skips.
    Consume-verbatim is the contract — this never mints or regenerates,
    and is the single place that knows where the perma-id lives."""
    if not state_dir:
        return None
    d = Path(state_dir)
    for sidecar in ("config.json", "transcript.json"):
        try:
            obj = json.loads((d / sidecar).read_text())
        except (OSError, ValueError):
            continue
        sid = obj.get("short_id")
        if isinstance(sid, str) and _PERMA_ID_RE.match(sid):
            return sid
    # Dir-name suffix fallback (last '-'-segment), runs only.
    leaf = d.name.rsplit("-", 1)
    if len(leaf) == 2 and _PERMA_ID_RE.match(leaf[1]):
        return leaf[1]
    return None


def emit(stream: Stream, perma_id: str, marker: Marker) -> None:
    """The single public content-free write entry. Validates the key
    and EVERY leaf of ``marker``, then appends it as one more YAML
    document in the ONE file for this perma-id at
    ``shareable/<stream>/<iso-z>-<perma-id>-anonymized.yaml``.

    The file is a YAML multi-document stream: a small header document
    (``schema_version``/``perma_id``/``stream``/``created_at``) written
    once, then one ``---``-separated document per turn. YAML multi-doc
    is both append-friendly (a turn = concatenate another ``---``
    document) AND human-legible as-is — no closing bracket, no trick,
    no read-modify-write, and emit never reads any file back (so
    nothing it reads can influence what it writes). It is valid YAML to
    a parser (``yaml.safe_load_all``) and readable to a person.

    YAML is safe here BY CONSTRUCTION: ``_assert_content_free`` has
    already proven every leaf is a closed type (``int|float|bool|None``,
    an ISO-Z timestamp, or a closed-enum value) — there is no free
    string, so no YAML free-text ambiguity or injection is possible.

    Raises ``ContentFreeViolation`` (crash, never a silent write) on any
    non-content-free leaf, and ``ValueError`` on a bad key. Callers
    treat emission as best-effort telemetry and must not let a failure
    break the user-facing turn — but the failure is loud, not masked.
    """
    if not isinstance(stream, Stream):
        raise ValueError("stream must be a Stream enum")
    if not isinstance(marker, Marker):
        raise ValueError("marker must be a frozen Marker dataclass")
    if not isinstance(perma_id, str) or not _PERMA_ID_RE.match(perma_id):
        raise ValueError(f"perma_id must be a 4-letter perma-id; got {perma_id!r}")

    # Every leaf of the marker is re-validated before serialization —
    # the runtime default-deny backstop. The header keys (schema_version
    # int / validated 4-letter perma_id / closed-enum stream value /
    # machine-generated ISO-Z created_at) are content-free by
    # construction, so the whole file carries only content-free data.
    _assert_content_free(marker)
    record = _to_jsonable(marker)

    out_dir = _shareable_root() / stream.value
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exactly one file per perma-id. Locate it by the perma-id
    # (filename is `<iso-z>-<perma-id>-anonymized.yaml`, date-first); the
    # perma-id is practically unique so there is at most one match (no
    # resolver, per the locked retrieval model). A 0-byte file (crash
    # before the header landed) is treated as fresh.
    existing = [
        p for p in sorted(out_dir.glob(f"*-{perma_id}-anonymized.yaml"))
        if p.stat().st_size > 0
    ]
    if stream is Stream.RUN and existing:
        # Latest-state-wins: a run's diagnostic is (re)written on every
        # wind-down — completed / paused / cancelled / failed / app-close
        # — so a later, more-complete state replaces an earlier partial
        # one (a pause snapshot is overwritten by the eventual
        # resume→complete; a canceled run keeps whatever it reached).
        # Wipe the prior file(s) for this perma-id and fall through to
        # write a fresh single header+doc. Still exactly one file per
        # perma-id (the per-chat-turn RUN re-dump that the old write-once
        # guard prevented is no longer emitted — only the chat stream
        # appends per turn).
        for p in existing:
            try:
                p.unlink()
            except OSError:
                pass
        existing = []
    if existing:
        path = existing[0]
        chunk = _yaml_doc(record)
    else:
        created_at = _now_iso_z()
        path = out_dir / (
            f"{created_at.replace(':', '-')}-{perma_id}-anonymized.yaml"
        )
        header = {
            "schema_version": SCHEMA_VERSION,
            "perma_id": perma_id,
            "stream": stream.value,
            "created_at": created_at,
        }
        chunk = _yaml_doc(header) + _yaml_doc(record)

    # Single atomic O_APPEND of one chunk — POSIX guarantees a lone
    # appender's modest write is not interleaved, so the multi-doc
    # stream never tears. No file is ever read back here.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, chunk.encode("utf-8"))
    finally:
        os.close(fd)


def _yaml_scalar(v: object) -> str:
    """A leaf as a YAML scalar. By construction (the guard ran first)
    every str is an ISO-Z timestamp or a closed-enum value — never free
    text — so single-quoting (with the standard ``''`` escape, applied
    defensively though the closed vocabulary contains no quote) is
    unambiguous and injection-free. Numbers/bools/None map directly."""
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def _yaml_lines(obj: object, indent: int) -> list[str]:
    """Recursively emit block-style YAML for the closed
    dict/list/scalar shape ``_to_jsonable`` produces. Dependency-free
    on purpose: no third-party parser in the trust path, and it runs on
    any Python (the packaged app's interpreter has no PyYAML). Safe
    precisely because the leaves are already a closed type."""
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(obj, dict):
        for k, val in obj.items():
            if isinstance(val, (dict, list)) and val:
                lines.append(f"{pad}{k}:")
                lines += _yaml_lines(val, indent + 1)
            elif isinstance(val, dict):
                lines.append(f"{pad}{k}: {{}}")
            elif isinstance(val, list):
                lines.append(f"{pad}{k}: []")
            else:
                lines.append(f"{pad}{k}: {_yaml_scalar(val)}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)) and item:
                sub = _yaml_lines(item, indent + 1)
                lines.append(f"{pad}- {sub[0].lstrip()}")
                lines += sub[1:]
            elif isinstance(item, dict):
                lines.append(f"{pad}- {{}}")
            elif isinstance(item, list):
                lines.append(f"{pad}- []")
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
    return lines


def _yaml_doc(obj: object) -> str:
    """One ``---``-prefixed YAML document for a closed-type dict."""
    return "---\n" + "\n".join(_yaml_lines(obj, 0)) + "\n"
