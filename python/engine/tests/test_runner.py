"""End-to-end pipeline test with mocked LLM calls.

Patches complete() in every stage module with canned JSON responses, then runs
runner.run() and asserts the expected intermediate + vault artifacts appear.
No real LLM, no real data — fast enough for the default `pytest` run, so this
is a unit test, not an integration test.
"""
import json


from .conftest import _wrap


# ── Canned LLM responses keyed by prompt content ──────────────────────────────

def _fake_complete(messages, **kwargs):
    body = messages[-1]["content"]

    # Content extractor — must produce a verbatim quote present in the doc.
    if "DOCUMENT [" in body:
        return _wrap(json.dumps([{
            "type": "fact",
            "summary": "Alice signed a contract",
            "evidence": [{
                "text": "Alice signed the contract.",
                "source_ref": "fixture",
            }],
            "occurred_at": "2026-04-15",
            "occurred_at_text": None,
            "entities": [{"name": "Alice", "entity_type": "person",
                          "role": "subject"}],
            "topics": ["work"],
            "tags": [],
            "confidence": 0.9,
        }]), **kwargs)

    # Entities stage — single-call per run. Matches on the task prompt.
    if "Subject to resolve:" in body and "INPUT GROUPS" in body:
        return _wrap(json.dumps({
            "subject_group_id": "g1",
            "entities": [{
                "group_id": "g1",
                "canonical_name": "Alice",
                "role": "subject",
                "description": "Author of the corpus.",
            }],
            "merges": [],
            "relations": [],
        }), **kwargs)

    # Patterns / insights / actions — not enough input to trigger any.
    # Default empty returns keep the e2e test fast and deterministic.
    if body.strip().startswith("Today"):
        # Actions prompt
        return _wrap(json.dumps({"actions": []}), **kwargs)
    if "cross_domain" in body and "critical" in body:
        # Insights prompt
        return _wrap(json.dumps({"cross_domain": [], "critical": []}), **kwargs)
    # Patterns prompt
    return _wrap("[]", **kwargs)


# ── Test ──────────────────────────────────────────────────────────────────────

def _redirect_roots(monkeypatch, tmp_path):
    """Point runner's _LOGS_ROOT / _VAULT_ROOT at tmp and clear any
    pre-existing run env vars so each test gets a fresh run."""
    from engine import runner
    monkeypatch.setattr(runner, "_LOGS_ROOT", tmp_path / "logs")
    monkeypatch.setattr(runner, "_VAULT_ROOT", tmp_path / "vault")
    # Legacy aliases some callers still touch.
    monkeypatch.setattr(runner, "_RUN_ROOT", tmp_path / "logs")
    monkeypatch.setattr(runner, "_OUTPUT_ROOT", tmp_path / "logs")
    for k in (
        "BASEVAULT_SESSION", "BASEVAULT_EVAL_ID",
        "BASEVAULT_RUN_NAME", "BASEVAULT_AGENT", "BASEVAULT_SWEEP_ID",
    ):
        monkeypatch.delenv(k, raising=False)




def test_pipeline_handles_empty_input(tmp_path, monkeypatch):
    """An input that yields no docs (e.g., empty file) should error cleanly."""
    fixture = tmp_path / "empty.txt"
    fixture.write_text("", encoding="utf-8")

    _redirect_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("BASEVAULT_RUN_NAME", "empty-local")
    monkeypatch.setenv("BASEVAULT_AGENT", "experiment")

    # Empty fixture still produces a Document with empty content; ingest will
    # return one doc, split_documents drops empty, leaving zero parents.
    # The pipeline shouldn't crash even when there's nothing to extract.
    from engine import runner
    runner.run([str(fixture)], "local", subject="Alice")




def test_budget_header_survives_str_extension_mode(tmp_path, monkeypatch):
    """Regression for #961: the Budget-header log block raw-accessed
    `mode.value` / `_spec.provider.value`, which AttributeErrors when an
    extension mode (the eval tree's str-keyed "test" path) passes `mode`
    as a bare `str` and pins a str-tagged provider. The Mode-enum modes
    (local/tee) never hit this; the str-mode path died at the header.

    Drives run() with a minimal str-keyed extension mode + bare-string
    provider — the exact shape testing/eval/_eval_specs registers for its
    non-attested cloud backend, reproduced inline so the lock doesn't
    depend on the eval tree. An empty fixture yields zero parents (no LLM
    stage calls, no dispatcher needed), but the run still walks the full
    preflight + Budget-header block where the bug lived. Pre-fix:
    AttributeError; post-fix: clean."""
    from engine.llm import ModelSpec, MODE_SPEC, register_mode

    label = "test-ext-961"
    register_mode(label, ModelSpec(
        provider="ext-fake",          # bare str, not a Provider enum
        model_id="ext/fake-model",
        context_window=8_000,
        require_streaming=False,
    ))
    try:
        _redirect_roots(monkeypatch, tmp_path)
        monkeypatch.setenv("BASEVAULT_RUN_NAME", "str-mode-961")
        monkeypatch.setenv("BASEVAULT_AGENT", "experiment")

        fixture = tmp_path / "empty.txt"
        fixture.write_text("", encoding="utf-8")

        from engine import runner
        runner.run([str(fixture)], label, subject="Alice")
    finally:
        MODE_SPEC.pop(label, None)




def _fake_complete_fanout(messages, **kwargs):
    """Variant emitting ONE fact filed under two categories (a fan-out)
    that mentions Alice — the cross-category duplicate case."""
    body = messages[-1]["content"]
    if "DOCUMENT [" in body:
        return _wrap(json.dumps([{
            "type": "emotion",
            "summary": "Alice felt repulsion at the paintings",
            "evidence": [{
                "text": "Alice signed the contract.",
                "source_ref": "fixture",
            }],
            "occurred_at": "2026-04-15",
            "occurred_at_text": None,
            "entities": [{"name": "Alice", "entity_type": "person",
                          "role": "subject"}],
            "topics": ["travel", "spirituality"],
            "tags": [],
            "confidence": 0.9,
        }]), **kwargs)
    if "Subject to resolve:" in body and "INPUT GROUPS" in body:
        return _wrap(json.dumps({
            "subject_group_id": "g1",
            "entities": [{
                "group_id": "g1",
                "canonical_name": "Alice",
                "role": "subject",
                "description": "Author of the corpus.",
            }],
            "merges": [],
            "relations": [],
        }), **kwargs)
    if body.strip().startswith("Today"):
        return _wrap(json.dumps({"actions": []}), **kwargs)
    if "cross_domain" in body and "critical" in body:
        return _wrap(json.dumps({"cross_domain": [], "critical": []}), **kwargs)
    return _wrap("[]", **kwargs)






# ── entities per-entity files: Phase 1 / Phase 2 / Phase 3 timing ────────────

def _two_entity_fake_complete(messages, **kwargs):
    """Variant of `_fake_complete` that emits two entities (Alice + Bob)
    with a relation_candidate, plus an LLM annotation for each. Used by
    the per-entity-file timing tests below to exercise both per-group
    Phase 1 writes (with candidate_relations) and Phase 2 enrichment
    (description + finalized relations)."""
    body = messages[-1]["content"]
    if "DOCUMENT [" in body:
        return _wrap(json.dumps([{
            "type": "fact",
            "summary": "Alice met Bob at the cafe",
            "evidence": [{
                "text": "Alice met Bob at the cafe.",
                "source_ref": "fixture",
            }],
            "occurred_at": "2026-04-15",
            "occurred_at_text": None,
            "entities": [
                {"name": "Alice", "entity_type": "person", "role": "subject"},
                {"name": "Bob", "entity_type": "person", "role": "object"},
            ],
            "topics": ["work"],
            "tags": [],
            "confidence": 0.9,
            "relation_candidate": {
                "from": "Alice", "to": "Bob",
                "verb": "met", "confidence": 0.9,
            },
        }]), **kwargs)
    # Per-entity batch prompt — fires once per batch, returns LLM
    # annotations for each group in the batch.
    if "Subject (CLI hint" in body and "ENTITIES IN THIS BATCH" in body:
        return _wrap(json.dumps({
            "entities": [
                {
                    "group_id": "g1",
                    "canonical_name": "Alice",
                    "role": "subject",
                    "description": "Author of the corpus.",
                    "is_subject_likelihood": 0.95,
                    "relations": [
                        {"to_id": "g2", "verb": "met", "confidence": 0.9},
                    ],
                },
                {
                    "group_id": "g2",
                    "canonical_name": "Bob",
                    "role": "friend",
                    "description": "Friend of the subject.",
                    "is_subject_likelihood": 0.05,
                    "relations": [],
                },
            ],
        }), **kwargs)
    # Dedupe prompt — be conservative, no merges.
    if "canonical entity rows" in body and "Find pairs" in body:
        return _wrap(json.dumps({"merges": []}), **kwargs)
    if body.strip().startswith("Today"):
        return _wrap(json.dumps({"actions": []}), **kwargs)
    if "cross_domain" in body and "critical" in body:
        return _wrap(json.dumps({"cross_domain": [], "critical": []}), **kwargs)
    return _wrap("[]", **kwargs)








