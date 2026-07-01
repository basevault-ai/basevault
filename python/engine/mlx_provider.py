"""App-layer MLX ``InferenceProvider`` for the kernel (issue #912).

MLX is the bundled local backend: an in-process model on the user's own
GPU, non-trust-critical (no attestation), so it lives in ``pipeline``
rather than ``kernel/``. It mirrors the kernel's provider shape
(return an ``LlmResponse``; ``finish_reason == "length"`` → ``CAP_HIT``;
``status=None`` on the happy path so ``RetryPolicy.compute_status``
derives the rest), translated to ``mlx_lm.stream_generate``.

All generation is serialized under a module lock: a single in-process GPU
model means concurrent callers would race the multi-GB load and contend
on one Metal context. Blocked threads just park — the intended
back-pressure for one local GPU (and why the local spec pins
``max_parallelism = 1``).
"""
from __future__ import annotations

import time
from threading import Lock
from typing import override

from kernel.abstractions import InferenceProvider, LlmCall, LlmResponse, LlmStatus
from kernel.enums import PhaseName
from kernel.execution_env import BoundExecutionEnv

from engine.llm import mlx_model_dir

_mlx_lock = Lock()
# (model, tokenizer, model_id) of the currently-loaded snapshot.
_mlx_bundle: tuple = None


def _get_mlx(model_id: str):
    """Load (or return the cached) MLX model + tokenizer from the local
    snapshot. Never auto-downloads — a missing snapshot raises with the
    in-app remedy so setup diagnostics surface it verbatim."""
    global _mlx_bundle
    if _mlx_bundle is not None and _mlx_bundle[2] == model_id:
        return _mlx_bundle[0], _mlx_bundle[1]
    path = mlx_model_dir(model_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Local model {model_id!r} is not downloaded. Open "
            f"Settings → Local model → Download to fetch it "
            f"(expected at {path})."
        )
    from mlx_lm import load

    model, tokenizer = load(str(path))
    _mlx_bundle = (model, tokenizer, model_id)
    return model, tokenizer


class MlxProvider(InferenceProvider):
    def __init__(self):
        self._injected_errors: dict[PhaseName, list[LlmStatus]] = {
            phase: [] for phase in PhaseName
        }

    @override
    def name(self) -> str:
        return "MLX (local, bundled)"

    @override
    def run(self, call: LlmCall, execution_env: BoundExecutionEnv) -> LlmResponse:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        model = execution_env.model_spec.model()
        phase: PhaseName = execution_env.phase.name()
        assert phase.does_llm_call()
        if self._injected_errors[phase]:
            return LlmResponse.from_status(self._injected_errors[phase].pop(0), 0)

        status: LlmStatus | None = None
        exception: Exception | None = None
        payload: str | None = None
        prompt_tokens: int = 0
        completion_tokens: int = 0
        ttft: float | None = None
        t0: float = time.monotonic()
        max_predict = call.max_tokens if call.max_tokens > 0 else 4096
        sampler = make_sampler(temp=0.0)

        try:
            pieces = []
            with _mlx_lock:
                mlx_model, mlx_tok = _get_mlx(model)
                # Text-only: LOCAL vision routes to Ollama, never MLX.
                assert all(
                    isinstance(m["content"], str) for m in call.messages
                ), "MlxProvider has no vision path; multimodal content unexpected"
                # Model's own chat template, reasoning off — matches the
                # reasoning-off local default. ``enable_thinking`` is a
                # no-op on templates that don't define it.
                prompt = mlx_tok.apply_chat_template(
                    call.messages,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for resp in stream_generate(
                    mlx_model, mlx_tok, prompt, max_tokens=max_predict, sampler=sampler
                ):
                    seg = getattr(resp, "text", "") or ""
                    if seg:
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        pieces.append(seg)
                        if call.stream_handler:
                            call.stream_handler(seg)
                    prompt_tokens = getattr(resp, "prompt_tokens", None) or prompt_tokens
                    completion_tokens = (
                        getattr(resp, "generation_tokens", None) or completion_tokens
                    )
                    if getattr(resp, "finish_reason", None) == "length":
                        status = LlmStatus.CAP_HIT
            payload = "".join(pieces)
        except Exception as e:
            exception = e

        duration = time.monotonic() - t0
        return LlmResponse(
            status, payload, exception, prompt_tokens, completion_tokens, 0, ttft, duration
        )

    @override
    def inject_errors(self, phase: PhaseName, errors: list[LlmStatus]) -> None:
        self._injected_errors[phase] += errors
