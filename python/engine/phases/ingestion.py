"""Ingestion (vision) phase on the kernel (issue #912).

INGESTION is the per-image vision LLM step: each pending image Document is
transcribed by a vision model and its description becomes the doc's
content. One image per call; the work can't be reduced (you can't sample
half an image), so on a sizing failure the kernel's degrading ladder runs
same-input → reasoning-off → model-fallback → stop (`sample_llm_call`
returns None) — exactly the legacy vision ladder
(`describe_images_all` used `sizing_strategy="sample"` with
`sample_down → None`). `RetryPolicy` already special-cases
`PhaseName.INGESTION` (no SAMPLE; give up after model-fallback).

Multimodal: the phase builds the image as an OpenAI ``image_url``
content-part list (`LlmCall.messages` content is `str | list[dict]`).
Tinfoil sends it verbatim; `OllamaProvider` translates the ``image_url``
parts into its separate ``images`` field. The image encode
(`vision.encode_image`), the prompt (`vision._DEFAULT_VISION_PROMPT`) and
the preamble strip (`vision._strip_preamble`) are reused VERBATIM.
"""
from __future__ import annotations

from typing import override

from kernel.abstractions import Job, LlmCall, Phase, PhaseResult
from kernel.enums import JobName, LlmStatus, PhaseName
from kernel.execution_env import BoundExecutionEnv, ExecutionEnv

from engine.vision import (
    _DEFAULT_VISION_PROMPT,
    _strip_preamble,
    _VISION_MAX_OUTPUT,
    encode_image,
)


class IngestionPhase(Phase):
    def __init__(self, job: Job):
        super().__init__(job)
        self._docs: list = []
        self._prompt: str = _DEFAULT_VISION_PROMPT

    @override
    def name(self) -> PhaseName:
        return PhaseName.INGESTION

    @override
    def init_from_memory(self, input: PhaseResult) -> bool:
        self._state = dict(input.data)
        images = input.data.get("images")
        if images is None:
            # Pipeline path: pick the pending-vision image docs out of the
            # full all_docs set (the runner flags them `_pending_vision`).
            from engine.ingestor import SourceType

            images = [
                d for d in input.data.get("all_docs", [])
                if d.source_type == SourceType.IMAGE
                and d.metadata.get("_pending_vision")
            ]
        self._docs = list(images)
        self._prompt = input.data.get("prompt", _DEFAULT_VISION_PROMPT)
        return True

    @override
    def init_from_disk(self) -> None:
        raise NotImplementedError("IngestionPhase always inits from memory")

    @override
    def validate(self, payload) -> LlmStatus | None:
        # Any non-empty transcript is OK; the empty case is caught upstream
        # (compute_status → LOAD) and settles as a failed image after the
        # degrading ladder exhausts.
        if not isinstance(payload, str) or not payload.strip():
            return LlmStatus.SUCCESS_EMPTY
        return None

    @override
    def halve_llm_call(self, call: LlmCall) -> list[LlmCall]:
        return []  # can't halve a single image

    @override
    def sample_llm_call(self, call: LlmCall) -> LlmCall | None:
        return None  # can't reduce a single image — stop (vision's ladder)

    @override
    def checkpoint(self) -> None:
        pass

    def _build_call(self, doc) -> LlmCall:
        b64, media_type = encode_image(doc.source_path)
        content = [
            {"type": "text", "text": self._prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:{media_type};base64,{b64}"}},
        ]
        return self.new_call(
            [{"role": "user", "content": content}],
            _VISION_MAX_OUTPUT,
            None,
            context=doc,
        )

    @override
    def run_main(self, execution_env: BoundExecutionEnv | None) -> PhaseResult:
        assert execution_env is not None
        if not self._docs:
            return PhaseResult({**self._state, "images": []})

        # One independent call-chain per image, fanned via run_all.
        calls = [self._build_call(doc) for doc in self._docs]
        results = execution_env.run_all(calls)

        for doc, responses in zip(self._docs, results):
            text = ""
            for response in responses:
                if isinstance(response.payload, str) and response.payload.strip():
                    text = response.payload
            stripped = _strip_preamble(text)
            if stripped:
                doc.content = stripped
                doc.metadata.pop("_pending_vision", None)
            else:
                doc.metadata["_vision_failed"] = True
                doc.metadata["_vision_skip_reason"] = "empty-output"
        return PhaseResult({**self._state, "images": self._docs})


class IngestionJob(Job):
    def __init__(self, images: list, prompt: str = _DEFAULT_VISION_PROMPT):
        super().__init__()
        self._images = images
        self._prompt = prompt

    @override
    def name(self) -> JobName:
        return JobName.FULL_PIPELINE

    @override
    def generate_phases(self) -> list[Phase]:
        return [IngestionPhase(self)]

    def initial_input(self) -> PhaseResult:
        return PhaseResult({"images": self._images, "prompt": self._prompt})


def describe_images(
    images: list,
    mode,
    prompt: str = _DEFAULT_VISION_PROMPT,
    execution_env: ExecutionEnv | None = None,
) -> list:
    """Transcribe each pending image Document on the kernel (vision model);
    sets ``doc.content`` in place and returns the docs."""
    from engine.phases.model_specs import vision_spec_for_mode

    job = IngestionJob(images, prompt=prompt)
    if execution_env is None:
        spec = vision_spec_for_mode(mode)
        env = ExecutionEnv()
        env.register_spec(PhaseName.INGESTION, spec, spec, False)
    else:
        env = execution_env
    return job.run(job.initial_input(), env).data["images"]
