"""L1 — offline, deterministic extraction-ladder coverage (issue #912).

Drive the migrated PRODUCTION kernel phase
(``phases.extraction_llm.run_extraction`` through a kernel ``ExecutionEnv``)
with canned model outputs and assert it fans the retry ladder out correctly:
the right number of leaf LLM calls and the right final item count for each
sizing scenario — halve on cap-hit / parse-error, one-shot retry on
success-empty, the truncation-salvage parse path.

No network. The "model" is a scripted responder keyed on the document content
embedded in the prompt, so the (halved) inputs map to byte-identical canned
outputs. The expected call counts are the ladder's correct values (this began
as a kernel-vs-legacy parity test against ``content_extractor.extract_items``
+ the legacy scheduler/retry cascade; the legacy stack is deleted in the
cutover, so the expectations are now pinned directly — including the recorded
``parse_error_then_halve`` divergence where the kernel halves immediately, the
correct behaviour the legacy cascade deviated from).
"""
from __future__ import annotations

import json
from typing import Callable

import pytest

from engine.ingestor import Document, SourceType
from kernel.abstractions import InferenceProvider, LlmResponse
from kernel.enums import LlmStatus, PhaseName
from kernel.execution_env import ExecutionEnv

from engine.phases.extraction_llm import run_extraction
from engine.phases.model_specs import PipelineModelSpec

# ── Shared fixture + scenario plumbing ──────────────────────────────────────

_DOC_ID = "L1"
_QUOTE = "Alice signed the contract."
# Repeat so the doc is well above the 200-char halving floor and the
# evidence quote survives in EITHER half after a midpoint split.
_FULL = (_QUOTE + " ") * 30
_MARKER = f"DOCUMENT [{_DOC_ID}]:\n"


def _doc(content: str) -> Document:
    return Document(
        id=_DOC_ID,
        source_path="fixture",
        source_type=SourceType.TXT,
        content=content,
        title="fixture",
        date="2026-06-17",
        file_id=_DOC_ID,
        origin_char=0,
    )


def _content_from_prompt(prompt: str) -> str:
    """Recover the document content a prompt was built from. The content
    follows the ``DOCUMENT [L1]:\\n`` marker (nothing trails it for a
    non-batch doc), so the scripted responder can key on it."""
    idx = prompt.rfind(_MARKER)
    return prompt[idx + len(_MARKER):] if idx >= 0 else prompt


def _success(n: int) -> str:
    return json.dumps({
        "split_summaries": [{"id": _DOC_ID, "summary": "gist"}],
        "items": [
            {
                "type": "fact",
                "summary": f"item {i}",
                "evidence": [{"text": _QUOTE, "source_ref": _DOC_ID}],
                "topics": ["work"],
                "affect": [],
                "confidence": 0.9,
            }
            for i in range(n)
        ],
    })


# A truncated-but-salvageable envelope: valid first item, then cut off
# mid-second-item. ``json.loads`` fails; ``_salvage_truncated_json``
# recovers the one complete item. finish_reason is "stop" (NOT "length")
# so this exercises the salvage parse path, not the cap-hit→halve path.
_TRUNCATED = (
    '{"split_summaries":[{"id":"L1","summary":"gist"}],"items":['
    '{"type":"fact","summary":"item 0","evidence":[{"text":'
    '"Alice signed the contract.","source_ref":"L1"}],"topics":["work"],'
    '"affect":[],"confidence":0.9},'
    '{"type":"fact","summary":"item 1","evidence":[{"text":"Alice sig'
)


# A scenario maps document content → (raw_output, finish_reason).
Scenario = Callable[[str], tuple[str, str]]


def _clean_success(content: str) -> tuple[str, str]:
    return _success(2), "stop"


def _cap_hit_then_halve(content: str) -> tuple[str, str]:
    if content == _FULL:
        return "", "length"          # cap-hit → sizing → halve
    return _success(1), "stop"       # each half: one item


def _parse_error_then_halve(content: str) -> tuple[str, str]:
    if content == _FULL:
        return "this is not json at all {{{", "stop"  # parse-error → halve
    return _success(1), "stop"


def _clean_empty(content: str) -> tuple[str, str]:
    return "[]", "stop"              # success-empty → one retry → give up


def _truncation_salvage(content: str) -> tuple[str, str]:
    return _TRUNCATED, "stop"        # salvage recovers 1 item, no retry


# (scenario, expected_items, expected_kernel_calls). On parse_error_then_halve
# the kernel halves immediately (3 calls: the failed parent + 2 halves) — the
# correct ladder behaviour; final items converge to 2.
_SCENARIOS = {
    "clean_success": (_clean_success, 2, 1),
    "cap_hit_then_halve": (_cap_hit_then_halve, 2, 3),
    "parse_error_then_halve": (_parse_error_then_halve, 2, 3),
    "clean_empty": (_clean_empty, 0, 2),
    "truncation_salvage": (_truncation_salvage, 1, 1),
}


# ── Kernel stack driver ─────────────────────────────────────────────────────


class _ScriptedProvider(InferenceProvider):
    """Returns the scenario's canned (raw, finish_reason) per call, keyed on
    the document content embedded in the prompt."""

    def __init__(self, scenario: Scenario):
        self._scenario = scenario
        self.call_count = 0
        self._injected = {p: [] for p in PhaseName}

    def name(self) -> str:
        return "scripted"

    def run(self, call, execution_env) -> LlmResponse:
        self.call_count += 1
        content = _content_from_prompt(call.messages[-1]["content"])  # user content
        raw, finish = self._scenario(content)
        # finish_reason == "length" → the provider stamps CAP_HIT, exactly as
        # TinfoilProvider does; compute_status returns it before validate().
        status = LlmStatus.CAP_HIT if finish == "length" else None
        return LlmResponse(status, raw, None, 0, 0, 0, 0.0, 0.0)

    def inject_errors(self, phase, errors) -> None:
        self._injected[phase] += errors


def _run_kernel(scenario: Scenario) -> tuple[list, int]:
    provider = _ScriptedProvider(scenario)
    spec = PipelineModelSpec(
        provider, "gpt-oss-120b", 131_000,
        max_parallelism=8, seconds_between_requests=0.0,
    )
    env = ExecutionEnv()
    env.register_spec(PhaseName.EXTRACTION_LLM, spec, spec, thinking=False)
    items = run_extraction(_doc(_FULL), env, max_tokens=2000)
    return items, provider.call_count


# ── The ladder assertion ────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_extraction_ladder(name):
    scenario, expected_items, expected_calls = _SCENARIOS[name]
    items, calls = _run_kernel(scenario)

    # 1) The ladder fanned out as expected (halve / retry / salvage).
    assert calls == expected_calls, (
        f"[{name}] kernel made {calls} calls, expected {expected_calls}"
    )

    # 2) The right number of final items survived the ladder.
    assert len(items) == expected_items, (
        f"[{name}] kernel produced {len(items)} items, expected {expected_items}"
    )

    # 3) Every surviving item is a well-formed fact grounded in the quote.
    for it in items:
        assert it.item_type == "fact"
        assert it.summary
        assert any(e.text == _QUOTE for e in it.evidence)
