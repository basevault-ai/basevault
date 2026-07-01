"""Offline validation for the EXTRACTION stage chain (#912).

Drives the full kernel extraction Job — splitter (deterministic) → LLM
(per-doc, run_all) → completion (deterministic) — with a scripted provider
that returns a canned envelope per doc. Proves: the splitter feeds N docs
to the LLM phase, each doc is its own concurrent call, and completion folds
the items into ``facts_by_topic`` (one entry per topic, items under each of
their topics). The single-doc L1 ladder test covers the per-doc retry/halve
cascade; this isolates the multi-doc fan-out + the deterministic brackets.
"""
from __future__ import annotations

import json

from engine.ingestor import Document, SourceType
from kernel.abstractions import InferenceProvider, LlmResponse
from kernel.enums import PhaseName
from kernel.execution_env import ExecutionEnv

from engine.llm import ModelSpec as LegacyModelSpec, Mode
from engine.phases.extraction_job import ExtractionStageJob
from engine.phases.model_specs import PipelineModelSpec


def _envelope(summary: str, topic: str, quote: str) -> str:
    return json.dumps({
        "split_summaries": [{"id": "d", "summary": "gist"}],
        "items": [{
            "type": "fact",
            "summary": summary,
            "evidence": [{"text": quote, "source_ref": "d"}],
            "topics": [topic],
            "affect": [],
            "confidence": 0.9,
        }],
    })


class _Scripted(InferenceProvider):
    """Returns a canned envelope keyed on the doc content embedded in the
    user prompt, so each doc gets its own item under its own topic."""

    def __init__(self, by_content: dict):
        self._by_content = by_content
        self.call_count = 0
        self._inj = {p: [] for p in PhaseName}

    def name(self):
        return "scripted-extract"

    def run(self, call, execution_env) -> LlmResponse:
        self.call_count += 1
        user = call.messages[-1]["content"]
        for needle, env in self._by_content.items():
            if needle in user:
                return LlmResponse(None, env, None, 0, 0, 0, 0.0, 0.0)
        return LlmResponse(None, "[]", None, 0, 0, 0, 0.0, 0.0)

    def inject_errors(self, phase, errors):
        self._inj[phase] += errors


def _doc(doc_id: str, content: str) -> Document:
    return Document(
        id=doc_id,
        source_path=doc_id,
        source_type=SourceType.MD_FILE,
        content=content,
        title=doc_id,
        date="",
        file_id=doc_id,
    )


def test_extraction_stage_multi_doc():
    # Two distinct small docs (below the split budget → one unit each, no
    # combine for two different titles).
    docs = [
        _doc("work", "Alice signed the Acme contract on Monday. " * 5),
        _doc("travel", "Alice visited Paris in spring. " * 5),
    ]
    provider = _Scripted({
        "Acme contract": _envelope("signed contract", "work", "Alice signed"),
        "visited Paris": _envelope("trip to paris", "travel", "Alice visited"),
    })
    legacy = LegacyModelSpec(
        provider="scripted", model_id="gpt-oss-120b", context_window=131_000
    )
    spec = PipelineModelSpec(provider, legacy.model_id, legacy.context_window, max_parallelism=4)
    env = ExecutionEnv()
    env.register_spec(PhaseName.EXTRACTION_LLM, spec, spec, thinking=False)

    job = ExtractionStageJob({
        "all_docs": docs, "mode": Mode.TEE, "budget_tokens": 16000,
        "combine_small": False, "carve_first": False,
    })
    facts_by_topic = job.run(job.initial_input(), env).data["facts_by_topic"]

    # One LLM call per doc (fanned via run_all); items folded by topic.
    assert provider.call_count == 2
    assert set(facts_by_topic.keys()) == {"work", "travel"}
    assert [it.summary for it in facts_by_topic["work"]] == ["signed contract"]
    assert [it.summary for it in facts_by_topic["travel"]] == ["trip to paris"]
