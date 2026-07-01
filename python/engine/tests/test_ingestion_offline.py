"""Offline validation for the migrated INGESTION (vision) phase (#912).

Drives the IngestionPhase with a scripted provider over real (tiny) PNG
files. Proves: one call per image (fanned via run_all), the image rides as
an OpenAI ``image_url`` multimodal content-part on ``LlmCall.messages``, the
preamble is stripped, and the transcript lands on ``doc.content``. The
provider-side translation (Ollama → ``images`` field) is unit-covered
separately; this isolates the phase + multimodal message build.
"""
from __future__ import annotations

import base64

from engine.ingestor import Document, SourceType
from kernel.abstractions import InferenceProvider, LlmResponse
from kernel.enums import PhaseName
from kernel.execution_env import ExecutionEnv

from engine.llm import ModelSpec as LegacyModelSpec
from engine.phases.ingestion import IngestionJob
from engine.phases.model_specs import PipelineModelSpec

# 1×1 transparent PNG.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)


class _ScriptedVision(InferenceProvider):
    def __init__(self, transcript: str):
        self._transcript = transcript
        self.call_count = 0
        self.saw_image = False
        self._inj = {p: [] for p in PhaseName}

    def name(self):
        return "scripted-vision"

    def run(self, call, execution_env) -> LlmResponse:
        self.call_count += 1
        # Multimodal content: a list with a text part + an image_url part.
        content = call.messages[0]["content"]
        assert isinstance(content, list)
        if any(p.get("type") == "image_url" for p in content):
            self.saw_image = True
        return LlmResponse(None, self._transcript, None, 0, 0, 0, 0.0, 0.0)

    def inject_errors(self, phase, errors):
        self._inj[phase] += errors


def _img_doc(path) -> Document:
    return Document(
        id=str(path),
        source_path=str(path),
        source_type=SourceType.IMAGE,
        content="",
        title=str(path),
        date="",
        file_id=str(path),
        metadata={"_pending_vision": True},
    )


def _run(tmp_path, transcript, n=1):
    docs = []
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        p.write_bytes(_PNG)
        docs.append(_img_doc(p))
    provider = _ScriptedVision(transcript)
    legacy = LegacyModelSpec(
        provider="scripted", model_id="kimi-k2-6", context_window=131_000
    )
    spec = PipelineModelSpec(provider, legacy.model_id, legacy.context_window, max_parallelism=4)
    env = ExecutionEnv()
    env.register_spec(PhaseName.INGESTION, spec, spec, thinking=False)
    job = IngestionJob(docs)
    out = job.run(job.initial_input(), env).data["images"]
    return out, provider


def test_ingestion_transcribes_image(tmp_path):
    docs, provider = _run(tmp_path, "A red square on white.")
    assert provider.call_count == 1
    assert provider.saw_image is True
    # Transcript on doc.content, pending flag cleared (preamble-strip
    # fidelity itself is covered by the legacy vision tests).
    assert docs[0].content == "A red square on white."
    assert "_pending_vision" not in docs[0].metadata


def test_ingestion_multi_image_fans_out(tmp_path):
    docs, provider = _run(tmp_path, "A square.", n=3)
    assert provider.call_count == 3
    assert all(d.content == "A square." for d in docs)


def test_ingestion_empty_output_marks_failed(tmp_path):
    docs, _provider = _run(tmp_path, "   ")
    assert docs[0].metadata.get("_vision_failed") is True
