"""Extraction LLM phase on the kernel (issue #912).

The production extraction stage as a kernel ``Phase``. REUSES the
production extraction logic verbatim — ``_build_prompt`` / ``_parse_items``
/ ``_halve_doc_content`` and the ``_SYSTEM`` system prompt from
``content_extractor`` — so the migration changes only the retry / halve
ladder PLUMBING (now owned by ``BoundExecutionEnv.run`` + ``RetryPolicy``),
not the prompt or the parser.

Extraction is non-degrading: a sizing failure (cap-hit / parse-error /
timeout-with-tokens) splits the *document* into halves and re-runs each,
never sampling. The phase recovers the document a call was built from via
the kernel's per-call ``LlmCall.context`` (no side-map), and the
system/user split carries the ``_SYSTEM`` rules as a distinct system turn
(both kernel capabilities added by the director in ``78366ca``).

This is the validated template the rest of the migration follows (spike
#910 / PR #911 proved L1 ladder-equivalence + L2 eval-invariant parity).
"""
from __future__ import annotations

from dataclasses import replace
from typing import override

from kernel.abstractions import Job, LlmCall, Phase, PhaseResult
from kernel.enums import JobName, LlmStatus, PhaseName
from kernel.execution_env import BoundExecutionEnv, ExecutionEnv

from engine.phases.telemetry_hook import record_call_counts, set_call_category

from engine.content_extractor import (
    ExtractedItem,
    _build_prompt,
    _halve_doc_content,
    _parse_items,
    _SYSTEM,
)
from engine.ingestor import Document
from engine.llm import dynamic_max_tokens, strip_fences as _strip_fences
from engine.tokens import count_tokens

# Fallback per-call output budget when the caller doesn't pin one. The
# per-call budget is a CALLER concern (payload-size + stage tuned), passed
# through the input ``PhaseResult``; the ModelSpec only needs
# ``context_window`` (spike gaps report #7).
_DEFAULT_MAX_TOKENS = 32_000


class ExtractionPhase(Phase):
    """The extraction LLM phase as a kernel ``Phase``.

    ``run_main`` builds the prompt for the whole document, fires one LLM
    call through ``execution_env.run`` (which owns the retry / halve
    ladder), and parses every leaf response into ``ExtractedItem``s.
    """

    def __init__(self, job: Job):
        super().__init__(job)
        self._docs: list[Document] = []
        # Explicit per-call output budget (eval runners pin one); None →
        # size each doc dynamically like the legacy stage.
        self._max_tokens_override: int | None = None
        self._mode = None
        self.items: list[ExtractedItem] = []

    @override
    def name(self) -> PhaseName:
        return PhaseName.EXTRACTION_LLM

    @override
    def init_from_memory(self, input: PhaseResult) -> bool:
        self._state = dict(input.data)
        self._docs = list(input.data["docs"])
        self._max_tokens_override = input.data.get("max_tokens")
        self._mode = input.data.get("mode")
        return True

    def _max_tokens_for(self, doc: Document) -> int:
        if self._max_tokens_override:
            return int(self._max_tokens_override)
        if self._mode is not None:
            return dynamic_max_tokens(
                count_tokens(doc.content), self._mode, stage="extract"
            )
        return _DEFAULT_MAX_TOKENS

    @override
    def init_from_disk(self) -> None:
        # Crash recovery is the LLM cache's job (spike gaps report #6): a
        # cancel→restart replays every completed call from cache. The
        # phase always inits from memory.
        raise NotImplementedError("ExtractionPhase always inits from memory")

    @override
    def validate(self, payload: str | list[float]) -> LlmStatus | None:
        """Phase-specific validation — the kernel's seam for the parser.

        Mirrors ``content_extractor._make_doc_parser``: empty →
        success-empty; non-empty-unparseable → parse-error;
        clean-parse-zero-items → success-empty; otherwise OK (``None``).

        The parse classification is DOC-INSENSITIVE — ``_parse_items``
        keeps every type+summary-valid item regardless of whether its
        evidence span resolves in the doc (``_resolve_span`` only sets
        offsets / ``approximate``, never drops the item). So validating
        against any doc (here ``self._docs[0]``) yields the same
        PARSE_ERROR / SUCCESS_EMPTY / OK decision for every doc's call,
        which is what lets the multi-doc fan-out share one ``validate``.
        Per-doc span resolution happens in ``run_main`` against the doc
        that produced each result.
        """
        if not isinstance(payload, str):
            return LlmStatus.PARSE_ERROR
        if not payload.strip():
            return LlmStatus.SUCCESS_EMPTY
        ref_doc = self._docs[0]
        items, _summaries, parse_err = _parse_items(_strip_fences(payload), ref_doc)
        if parse_err:
            return LlmStatus.PARSE_ERROR
        if not items:
            return LlmStatus.SUCCESS_EMPTY
        return None

    @override
    def halve_llm_call(self, call: LlmCall) -> list[LlmCall]:
        """Sizing-failure recovery: split the document, not the prompt.

        Recovers the document this call was built from via
        ``call.context`` (the kernel's per-call work-item handle), halves
        its CONTENT at a paragraph/sentence boundary (the production
        ``_halve_doc_content``), rebuilds a prompt for each half, and
        returns two child calls (each carrying its half-doc as context).
        Returns ``[]`` when the content is below the halving floor — the
        kernel reads ``[]`` as "division not possible, stop".
        """
        doc: Document | None = call.context
        if doc is None:
            return []
        pieces = _halve_doc_content(doc.content)
        if pieces is None:
            return []
        left_text, right_text, _split = pieces
        children: list[LlmCall] = []
        for sub_doc in (
            replace(doc, content=left_text),
            replace(doc, content=right_text),
        ):
            children.append(
                self.new_call(
                    [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": _build_prompt(sub_doc)},
                    ],
                    call.max_tokens,
                    call,
                    context=sub_doc,
                )
            )
        return children

    @override
    def sample_llm_call(self, call: LlmCall) -> LlmCall | None:
        # Extraction is non-degrading (halve, never sample). The ABC
        # requires both hooks even though a phase uses exactly one.
        raise NotImplementedError("extraction halves; it never samples")

    @override
    def checkpoint(self) -> None:
        # No-op: resumability comes from the LLM cache (spike gaps report
        # #6), not a per-phase disk checkpoint.
        pass

    @override
    def run_main(self, execution_env: BoundExecutionEnv | None) -> PhaseResult:
        assert execution_env is not None
        # Each doc is an INDEPENDENT call-chain (its own halve tree) — fan
        # them all out via run_all so the kernel scheduler runs them
        # concurrently (the legacy single-pool-across-all-splits behavior).
        # Docs with no content contribute no items and no call.
        docs = [d for d in self._docs if d.content]
        if not docs:
            return PhaseResult({**self._state, "items": []})

        calls = []
        for doc in docs:
            call = self.new_call(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _build_prompt(doc)},
                ],
                self._max_tokens_for(doc),
                None,
                context=doc,
            )
            # Per-call category column = the split name (doc.id, e.g.
            # "<file>::split_03"), matching legacy extract_items so the
            # run-details modal labels each extraction call by its split.
            set_call_category(execution_env, call, doc.id)
            calls.append(call)

        on_split_complete = getattr(self.job, "_on_split_complete", None)

        def _parse_doc(doc, responses):
            doc_items: list[ExtractedItem] = []
            doc_summaries: dict = {}
            for response in responses:
                payload = response.payload
                if not payload or not isinstance(payload, str):
                    continue
                parsed, summaries, _parse_err = _parse_items(
                    _strip_fences(payload), doc
                )
                doc_items.extend(parsed)
                if summaries:
                    doc_summaries.update(summaries)
            return doc_items, doc_summaries

        # STREAM persistence per split: fire each doc's call-chain, then attach
        # a done-callback that parses + persists THAT doc as soon as it
        # resolves — so facts / entity mentions land on disk live (the run-tree
        # populates mid-stage) instead of in a burst after the whole stage.
        # The callback is fire-and-forget: a Future done-callback that raised
        # would be swallowed by the stdlib and orphan the run, so it never lets
        # an exception escape. Ordered item collection happens below off the
        # blocking results (results, not callback state, avoids the
        # set_result-notifies-before-callbacks race).
        def _stream_persist(doc, fut):
            if on_split_complete is None:
                return
            try:
                doc_items, doc_summaries = _parse_doc(doc, fut.result())
                on_split_complete(doc, doc_items, doc_summaries)
            except Exception:
                pass

        futures = [execution_env.run(c) for c in calls]
        for doc, fut in zip(docs, futures):
            fut.add_done_callback(
                lambda f, d=doc: _stream_persist(d, f)
            )

        # Block for all leaves, in input order, and assemble the return items.
        results = [fut.result() for fut in futures]
        items: list[ExtractedItem] = []
        for call, doc, responses in zip(calls, docs, results):
            doc_items, _ = _parse_doc(doc, responses)
            items.extend(doc_items)
            # Per-call counts event (one chunk in, N facts out), parity with
            # legacy content_extractor's record_stage_counts.
            record_call_counts(
                execution_env, call, {"chunks": 1}, {"facts": len(doc_items)},
            )

        self.items = items
        return PhaseResult({**self._state, "items": items})


class ExtractionJob(Job):
    """Single-phase job wrapping ``ExtractionPhase`` over a list of docs.
    ``Job.run`` returns the final ``PhaseResult`` (spike gaps report #5),
    so items come straight back from ``run``."""

    def __init__(
        self,
        docs: list[Document],
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        on_split_complete=None,
    ):
        super().__init__()
        self._docs = docs
        self._max_tokens = max_tokens
        self._on_split_complete = on_split_complete
        self.phase = ExtractionPhase(self)

    @override
    def name(self) -> JobName:
        return JobName.FULL_PIPELINE

    @override
    def generate_phases(self) -> list[Phase]:
        return [self.phase]

    def initial_input(self) -> PhaseResult:
        return PhaseResult({"docs": self._docs, "max_tokens": self._max_tokens})


def run_extraction_all(
    docs: list[Document],
    execution_env: ExecutionEnv,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    on_split_complete=None,
) -> list[ExtractedItem]:
    """Run extraction over many documents (one concurrent call-chain per
    doc via run_all) and return the flattened items.

    ``on_split_complete(doc, items, summaries)`` fires once per split chunk
    after its call-chain resolves and parses — the same per-split callback the
    legacy ``extract_items`` exposes, so the runner's ``_persist_split`` can
    populate ``_extract_calls`` (the split-ids / phase_2_marker view + the
    embeddings chunk→summary join). Unlike legacy this fires after run_all
    drains rather than streaming mid-stage; the final records are identical."""
    job = ExtractionJob(
        docs, max_tokens=max_tokens, on_split_complete=on_split_complete
    )
    result = job.run(job.initial_input(), execution_env)
    return list(result.data.get("items", []))


def run_extraction(
    doc: Document,
    execution_env: ExecutionEnv,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> list[ExtractedItem]:
    """Single-document convenience wrapper over ``run_extraction_all`` (used
    by the L1/L2 eval runners)."""
    return run_extraction_all([doc], execution_env, max_tokens=max_tokens)
