"""Offline validation for the KernelTelemetryHook (#912).

The hook reproduces the legacy per-call ``llm-calls.jsonl`` stat records on
the kernel path. This drives a real kernel extraction call with the hook
registered + a scripted provider, then asserts a stat record was opened,
had usage recorded, and was closed successful — with the right stage /
model. This is the enabler that lets the runner / sidecar driver swap keep
observability.
"""
from __future__ import annotations

import json

from engine import llm
from engine.ingestor import Document, SourceType
from kernel.abstractions import InferenceProvider, LlmResponse
from kernel.enums import LlmStatus, PhaseName
from kernel.execution_env import ExecutionEnv

from engine.llm import ModelSpec as LegacyModelSpec
from engine.phases.extraction_llm import run_extraction
from engine.phases.model_specs import PipelineModelSpec
from engine.phases.telemetry_hook import KernelTelemetryHook

_ENVELOPE = json.dumps({
    "split_summaries": [{"id": "d", "summary": "g"}],
    "items": [{
        "type": "fact", "summary": "Alice signed",
        "evidence": [{"text": "Alice signed", "source_ref": "d"}],
        "topics": ["work"], "affect": [], "confidence": 0.9,
    }],
})


class _Scripted(InferenceProvider):
    def __init__(self):
        self._inj = {p: [] for p in PhaseName}

    def name(self):
        return "Tinfoil TEE"  # → mode tag "tinfoil"

    def run(self, call, execution_env) -> LlmResponse:
        return LlmResponse(None, _ENVELOPE, None, 11, 22, 0, 0.05, 0.4)

    def inject_errors(self, phase, errors):
        self._inj[phase] += errors


def test_telemetry_hook_records_kernel_call():
    llm.reset_stat_records()
    provider = _Scripted()
    legacy = LegacyModelSpec(
        provider="tinfoil", model_id="gpt-oss-120b", context_window=131_000
    )
    spec = PipelineModelSpec(provider, legacy.model_id, legacy.context_window, max_parallelism=4)
    env = ExecutionEnv()
    env.register_spec(PhaseName.EXTRACTION_LLM, spec, spec, thinking=False)
    env.register_llm_hook(KernelTelemetryHook(session_id="sess-1"))

    doc = Document(
        id="d", source_path="d.md", source_type=SourceType.MD_FILE,
        content="Alice signed the Acme contract. " * 6, title="d", date="",
        file_id="d",
    )
    items = run_extraction(doc, env, max_tokens=2000)
    assert len(items) == 1  # the call succeeded

    recs = list(llm._stats_records)
    assert len(recs) == 1, f"expected one stat record, got {len(recs)}"
    rec = recs[0]
    assert rec["model"] == "gpt-oss-120b"
    # The hook normalizes StageName.EXTRACTION ("extraction_splitter") to the
    # legacy runner + run-details label "extract" (so the kernel extraction
    # calls land in the same UI section / stat bucket as legacy).
    assert rec["stage"] == "extract"
    assert rec.get("success") is True
    assert rec["session_id"] == "sess-1"
    # Usage came off the response (provider reported 11 in / 22 out).
    assert rec.get("prompt_tokens") == 11
    assert rec.get("completion_tokens") == 22


class _SuccessEmptyScripted(InferenceProvider):
    """Every leaf fails with a ``parse_signals._SuccessEmpty`` exception — the
    sentinel a dedupe raises when it finds no merges."""
    def __init__(self):
        self._inj = {p: [] for p in PhaseName}

    def name(self):
        return "Tinfoil TEE"

    def run(self, call, execution_env) -> LlmResponse:
        from engine.parse_signals import _SuccessEmpty
        return LlmResponse(
            LlmStatus.SUCCESS_EMPTY, None, _SuccessEmpty("no merges"),
            0, 0, 0, None, 0.1,
        )

    def inject_errors(self, phase, errors):
        self._inj[phase] += errors


def test_telemetry_hook_labels_success_empty_as_parse_signals():
    """A leaf that fails with ``_SuccessEmpty`` (e.g. a dedupe that finds no
    merges) is recorded with ``error.type == "engine.parse_signals._SuccessEmpty"`` —
    the failure label the run-details UI + debug bundles read. ``_SuccessEmpty``
    moved ``retry`` -> ``parse_signals`` in #912; pin the kernel-path label so a
    future module move cannot silently relabel it. (Legacy covered this via
    ``test_llm_stats_artifact_e2e``'s ``error.class`` assertion, which is deleted
    at the cutover along with ``complete()``.)"""
    llm.reset_stat_records()
    provider = _SuccessEmptyScripted()
    legacy = LegacyModelSpec(
        provider="tinfoil", model_id="gpt-oss-120b", context_window=131_000
    )
    spec = PipelineModelSpec(
        provider, legacy.model_id, legacy.context_window, max_parallelism=4)
    env = ExecutionEnv()
    env.register_spec(PhaseName.EXTRACTION_LLM, spec, spec, thinking=False)
    env.register_llm_hook(KernelTelemetryHook(session_id="sess-se"))

    doc = Document(
        id="d", source_path="d.md", source_type=SourceType.MD_FILE,
        content="Alice signed the Acme contract. " * 6, title="d", date="",
        file_id="d",
    )
    try:
        run_extraction(doc, env, max_tokens=2000)
    except Exception:
        pass  # the leaf fails by design; we only assert the recorded label

    failed = [r for r in llm._stats_records if not r.get("success", True)]
    assert failed, "expected at least one failed stat record"
    assert any(
        (r.get("error") or {}).get("type") == "engine.parse_signals._SuccessEmpty"
        for r in failed
    ), [r.get("error") for r in failed]


class _LoadFlakeThenOk(InferenceProvider):
    """First leaf fails transient (``ConnectionError`` → ``LlmStatus.LOAD``,
    retried via FULL_RETRY); the retry succeeds. Exercises the retry-linkage
    stamp the diagnostics adapter reads."""

    def __init__(self):
        self.calls = 0
        self._inj = {p: [] for p in PhaseName}

    def name(self):
        return "Tinfoil TEE"

    def run(self, call, execution_env) -> LlmResponse:
        self.calls += 1
        if self.calls == 1:
            return LlmResponse(None, None, ConnectionError("flake"), 0, 0, 0, None, 0.1)
        return LlmResponse(None, _ENVELOPE, None, 11, 22, 0, 0.05, 0.4)

    def inject_errors(self, phase, errors):
        self._inj[phase] += errors


def test_retry_record_carries_retry_linkage():
    """A transient LOAD failure is retried (FULL_RETRY); the retry's stat
    record must carry ``attempt == 2`` + ``retry_of_call_id`` pointing at the
    original call. Without it the diagnostics adapter reads every kernel call
    as a fresh first attempt (the chat-502 case rendered 5 LOAD retries of one
    hop as 5 independent ``is_retry: false`` calls)."""
    llm.reset_stat_records()
    provider = _LoadFlakeThenOk()
    legacy = LegacyModelSpec(
        provider="tinfoil", model_id="gpt-oss-120b", context_window=131_000
    )
    spec = PipelineModelSpec(
        provider, legacy.model_id, legacy.context_window, max_parallelism=4)
    env = ExecutionEnv()
    env.register_spec(PhaseName.EXTRACTION_LLM, spec, spec, thinking=False)
    env.register_llm_hook(KernelTelemetryHook(session_id="sess-retry"))

    doc = Document(
        id="d", source_path="d.md", source_type=SourceType.MD_FILE,
        content="Alice signed the Acme contract. " * 6, title="d", date="",
        file_id="d",
    )
    items = run_extraction(doc, env, max_tokens=2000)
    assert items, "the retry should have succeeded after the transient flake"

    recs = list(llm._stats_records)
    by_id = {r["call_id"]: r for r in recs}
    retried = [r for r in recs if r.get("attempt", 1) >= 2]
    assert retried, f"expected a retry record, got attempts {[r.get('attempt') for r in recs]}"
    for r in retried:
        assert r["attempt"] == 2
        assert r.get("retry_of_call_id") in by_id, \
            "retry_of_call_id must reference a real prior call"
    originals = [r for r in recs if r.get("attempt", 1) == 1]
    assert originals and all(not r.get("retry_of_call_id") for r in originals), \
        "original (non-retry) calls must carry no retry linkage"


class _LoadFlakeThenOkStatus(InferenceProvider):
    """Like ``_LoadFlakeThenOk`` but the failing attempt carries an explicit
    ``LlmStatus.LOAD`` so the kernel stamps the rec's ``llm_status`` — exercises
    the persistence path end-to-end (the failed attempt's status must reach the
    jsonl, not just the in-memory rec)."""

    def __init__(self):
        self.calls = 0
        self._inj = {p: [] for p in PhaseName}

    def name(self):
        return "Tinfoil TEE"

    def run(self, call, execution_env) -> LlmResponse:
        self.calls += 1
        if self.calls == 1:
            return LlmResponse(
                LlmStatus.LOAD, None, ConnectionError("flake"), 0, 0, 0, None, 0.1)
        return LlmResponse(None, _ENVELOPE, None, 11, 22, 0, 0.05, 0.4)

    def inject_errors(self, phase, errors):
        self._inj[phase] += errors


def test_retry_linkage_and_llm_status_survive_to_jsonl(tmp_path):
    """The fix for #963: ``retry_of_call_id`` / ``attempt`` / ``llm_status`` must
    land in ``llm-calls.jsonl`` — the file the run-details rollup is built from —
    not merely on the in-memory stat record. Pre-fix the linkage was stamped on
    the rec AFTER the begin event was already written and ``llm_status`` never
    reached the jsonl at all, so the rollup read every retry as a null-linkage
    first attempt labeled ``(other)``. Drive a LOAD flake + retry with the
    streaming jsonl path set, materialize the file back, and assert the
    persisted records carry the linkage + status and classify correctly."""
    from engine.runner import _materialize_calls_from_jsonl, _classify_outcome

    jsonl = tmp_path / "llm-calls.jsonl"
    llm.reset_stat_records()
    llm.set_calls_jsonl_path(jsonl)
    try:
        provider = _LoadFlakeThenOkStatus()
        legacy = LegacyModelSpec(
            provider="tinfoil", model_id="gpt-oss-120b", context_window=131_000)
        spec = PipelineModelSpec(
            provider, legacy.model_id, legacy.context_window, max_parallelism=4)
        env = ExecutionEnv()
        env.register_spec(PhaseName.EXTRACTION_LLM, spec, spec, thinking=False)
        env.register_llm_hook(KernelTelemetryHook(session_id="sess-jsonl"))

        doc = Document(
            id="d", source_path="d.md", source_type=SourceType.MD_FILE,
            content="Alice signed the Acme contract. " * 6, title="d", date="",
            file_id="d",
        )
        items = run_extraction(doc, env, max_tokens=2000)
        assert items, "the retry should have succeeded after the transient flake"
    finally:
        llm.set_calls_jsonl_path(None)

    # Materialize straight off disk — the canonical rollup source. Does NOT
    # consult the in-memory _stats_records, so this proves the data persisted.
    recs = _materialize_calls_from_jsonl(jsonl, "2026-06-28T00:00:00Z")
    by_id = {r["call_id"]: r for r in recs}

    retried = [r for r in recs if (r.get("attempt") or 1) >= 2]
    assert retried, (
        "expected a persisted retry record; "
        f"attempts={[r.get('attempt') for r in recs]}"
    )
    for r in retried:
        assert r["attempt"] == 2
        assert r.get("retry_of_call_id") in by_id, (
            "retry_of_call_id must persist + reference a real prior call "
            f"(got {r.get('retry_of_call_id')!r})"
        )

    # The failed first attempt must carry the kernel's LOAD status through the
    # end event, and classify as load — not the (other) catch-all.
    failed = [r for r in recs if r.get("success") is False]
    assert failed, "expected a persisted failed attempt"
    assert all(r.get("llm_status") == "LOAD" for r in failed), (
        f"llm_status must persist; got {[r.get('llm_status') for r in failed]}"
    )
    from engine.common.status import OUTCOME_FAILED_LOAD
    assert all(_classify_outcome(r) == OUTCOME_FAILED_LOAD for r in failed), (
        f"LOAD failure must label as load, got "
        f"{[_classify_outcome(r) for r in failed]}"
    )


def test_failure_class_reads_kernel_llm_status():
    """``runner._failure_class_for_label`` buckets straight off the stamped
    ``llm_status`` — a transient LOAD (e.g. a 502 InternalServerError) must
    read as ``load``, not the ``other`` catch-all. The g6cf 502 mislabel was a
    stale binary whose record predated the ``llm_status`` stamp; lock the
    mapping so it can't silently regress."""
    from engine.runner import _failure_class_for_label
    assert _failure_class_for_label({"llm_status": "LOAD"}, {}) == "load"
    assert _failure_class_for_label({"llm_status": "CAP_HIT"}, {}) == "sizing"
    assert _failure_class_for_label({"llm_status": "OTHER"}, {}) == "other"
    # Pre-kernel records (no llm_status) bucket to "other" — unchanged.
    assert _failure_class_for_label({}, {}) == "other"


def test_category_for_retry_matches_legacy_transform():
    """`category_for_retry` is the verbatim port of the v0.2.0
    app/pipeline/retry.py transform the kernel cutover deleted: strip the old
    ` - retry/<class>`, ACCUMULATE the structural step on the prefix, append the
    new class suffix. Pins the exact docstring examples — including that a plain
    retry (no structural step) still gets ` - retry/<class>`, and that nested
    halves stack."""
    from engine.phases.telemetry_hook import category_for_retry as c
    assert c("topic", "load") == "topic - retry/load"
    assert c("topic - retry/load", "sizing", "/half-1") == "topic/half-1 - retry/sizing"
    # Suffix REPLACED (reflects latest class), prefix unchanged.
    assert c("topic/half-1 - retry/sizing", "load") == "topic/half-1 - retry/load"
    # Structural step ACCUMULATES — nested halve gets a unique path.
    assert c("topic/half-1 - retry/sizing", "sizing", "/half-2") \
        == "topic/half-1/half-2 - retry/sizing"


def test_retry_category_accumulates_down_the_tree():
    """#963 follow-up: each retry child's full label is built by transforming the
    PARENT's emitted category, so the structural path stacks. Drives the hook's
    `_retry_category` with the kernel RetryType the same way hook_llm_started
    does (the kernel POLICY that picks the strategy is exercised by the real-run
    smoke). HALVES spawns two children → `/half-1`,`/half-2`; a half that itself
    halves accumulates (`/half-1/half-1`). FULL_RETRY keeps the prefix, swaps the
    class. Root → bare base."""
    from kernel.enums import RetryType
    hook = KernelTelemetryHook(session_id="s")

    class _Call:
        def __init__(self, cid, prev):
            self.id = cid
            self.previous_call_id = prev

    def emit(call, base):
        # Mirror hook_llm_started: build + record the emitted category.
        cat = hook._retry_category(call, base)
        hook._emitted_category[call.id] = cat
        return cat

    # Root: no parent → bare base.
    assert emit(_Call("P0", None), "topic") == "topic"
    # P0 fails LOAD → FULL_RETRY child P1: prefix kept, ` - retry/load`.
    hook._retry_strategy_by_call["P0"] = RetryType.FULL_RETRY
    hook._retry_bucket_by_call["P0"] = "load"
    assert emit(_Call("P1", "P0"), "topic") == "topic - retry/load"
    # P1 fails SIZING → HALVES spawns two children: /half-1, /half-2.
    hook._retry_strategy_by_call["P1"] = RetryType.HALVES
    hook._retry_bucket_by_call["P1"] = "sizing"
    assert emit(_Call("H1", "P1"), "topic") == "topic/half-1 - retry/sizing"
    assert emit(_Call("H2", "P1"), "topic") == "topic/half-2 - retry/sizing"
    # H1 itself fails SIZING → halves AGAIN: the path ACCUMULATES off H1's label.
    hook._retry_strategy_by_call["H1"] = RetryType.HALVES
    hook._retry_bucket_by_call["H1"] = "sizing"
    assert emit(_Call("H1a", "H1"), "topic") == "topic/half-1/half-1 - retry/sizing"
    assert emit(_Call("H1b", "H1"), "topic") == "topic/half-1/half-2 - retry/sizing"
    # SAMPLE ladder: one child per parent, ordinal accumulates down the chain.
    hook._retry_strategy_by_call["H2"] = RetryType.SAMPLE
    hook._retry_bucket_by_call["H2"] = "sizing"
    s1 = emit(_Call("S1", "H2"), "topic")
    assert s1 == "topic/half-2/sample-1 - retry/sizing"
    hook._retry_strategy_by_call["S1"] = RetryType.SAMPLE
    hook._retry_bucket_by_call["S1"] = "sizing"
    assert emit(_Call("S2", "S1"), "topic") == "topic/half-2/sample-1/sample-2 - retry/sizing"
    # REASONING_OFF / MODEL_FALLBACK are fixed tokens.
    hook._retry_strategy_by_call["H2"] = RetryType.REASONING_OFF
    assert emit(_Call("R1", "H2"), "topic").endswith("/reasoning-off - retry/sizing")
    # Parent's status didn't map to a class → child stays bare.
    hook._retry_strategy_by_call["U"] = RetryType.FULL_RETRY
    assert emit(_Call("U1", "U"), "topic") == "topic"
    # Unknown parent (never recorded a retry) → bare base.
    assert emit(_Call("X1", "unknown"), "topic") == "topic"


def test_full_retry_child_category_carries_reason_not_strategy():
    """End-to-end guard via the real kernel retry path: a transient LOAD drives
    FULL_RETRY, whose child carries the ` - retry/load` reason marker (every
    retry shows WHY it fired) but NO work-reducing strategy tag (only
    halve/sample/reasoning-off/model-fallback get those). Reuses the LOAD-flake
    provider that actually produces a kernel retry."""
    llm.reset_stat_records()
    provider = _LoadFlakeThenOk()
    legacy = LegacyModelSpec(
        provider="tinfoil", model_id="gpt-oss-120b", context_window=131_000)
    spec = PipelineModelSpec(
        provider, legacy.model_id, legacy.context_window, max_parallelism=4)
    env = ExecutionEnv()
    env.register_spec(PhaseName.EXTRACTION_LLM, spec, spec, thinking=False)
    env.register_llm_hook(KernelTelemetryHook(session_id="s"))
    doc = Document(
        id="d", source_path="d.md", source_type=SourceType.MD_FILE,
        content="Alice signed the Acme contract. " * 6, title="d", date="",
        file_id="d",
    )
    items = run_extraction(doc, env, max_tokens=2000)
    assert items
    retried = [r for r in llm._stats_records if (r.get("attempt") or 1) >= 2]
    assert retried, "expected a FULL_RETRY child"
    for r in retried:
        cat = r.get("category") or ""
        assert "/half" not in cat and "/sample-" not in cat \
            and "/reasoning-off" not in cat and "/model-fallback" not in cat, \
            f"FULL_RETRY child must not carry a strategy suffix, got {cat!r}"
        assert "- retry/load" in cat, \
            f"FULL_RETRY child must carry the ` - retry/<reason>` marker, got {cat!r}"


def test_classify_outcome_reads_status_on_no_exception_failure():
    """#963 fix-site 2: a synthetic / ``from_status`` failure has
    ``success == False`` but ``error is None`` (no exception was raised — the
    kernel built the outcome from a status). ``_classify_outcome`` must still
    consult the kernel's ``llm_status`` bucket; pre-fix an ``if err`` gate
    forced ``other`` whenever there was no exception, so a no-exception LOAD
    mislabeled as ``failed (other)``. The injection smoke's
    ``LlmResponse.from_status`` failures are exactly this shape."""
    from engine.runner import _classify_outcome
    from engine.common.status import OUTCOME_FAILED_LOAD, OUTCOME_FAILED_OTHER

    load = {"success": False, "error": None, "llm_status": "LOAD"}
    assert _classify_outcome(load) == OUTCOME_FAILED_LOAD
    # No llm_status (pre-kernel record) still falls to other.
    bare = {"success": False, "error": None}
    assert _classify_outcome(bare) == OUTCOME_FAILED_OTHER


def test_failed_call_prompt_reaches_failure_payload_sink():
    """A failed call (here a LOAD, payload=None) must hand its prompt to the
    ``failure_payload_sink`` so the dev-tab "full prompt + response" view has
    something to show. The success ``payload_sink`` only fires for str payloads,
    so without the failure sink every from_status / injected failure logs no
    payload at all (run yxsr: 41 failed calls, 0 in llm-payloads.jsonl). The
    pipeline wires this to ``llm._log_call_failure_payload``; here we capture it
    with a stub to assert the failed call's real prompt reaches the sink."""
    llm.reset_stat_records()
    provider = _LoadFlakeThenOkStatus()
    legacy = LegacyModelSpec(
        provider="tinfoil", model_id="gpt-oss-120b", context_window=131_000
    )
    spec = PipelineModelSpec(
        provider, legacy.model_id, legacy.context_window, max_parallelism=4)
    captured: list[tuple[str, list]] = []
    env = ExecutionEnv()
    env.register_spec(PhaseName.EXTRACTION_LLM, spec, spec, thinking=False)
    env.register_llm_hook(KernelTelemetryHook(
        session_id="sess-failpayload",
        failure_payload_sink=lambda cid, messages: captured.append((cid, messages)),
    ))

    doc = Document(
        id="d", source_path="d.md", source_type=SourceType.MD_FILE,
        content="Alice signed the Acme contract. " * 6, title="d", date="",
        file_id="d",
    )
    items = run_extraction(doc, env, max_tokens=2000)
    assert items, "the retry should have succeeded after the LOAD flake"

    assert captured, "the failed LOAD call's prompt must reach the failure sink"
    cid, messages = captured[0]
    assert cid, "failure payload must be keyed by the stat call_id"
    assert messages and any(
        isinstance(m, dict) and m.get("content") for m in messages
    ), f"the captured prompt must be the real (non-empty) messages, got {messages}"
