"""App-layer Ollama ``InferenceProvider`` for the kernel (issue #912).

The local Ollama backend is non-trust-critical (it talks to a daemon on
the user's own machine, no attestation), so — unlike ``TinfoilProvider``
— it lives in ``pipeline`` rather than ``kernel/``. It mirrors
``kernel.tinfoil_provider.TinfoilProvider.run`` on the request / stream /
usage / cap-hit / cancellation surface, translated to Ollama's client
shape (``client.chat`` with an ``options`` dict; ``prompt_eval_count`` /
``eval_count`` / ``done_reason`` instead of OpenAI's usage chunk).

Reasoning honors the per-call setting: ``run`` forwards the model's
``thinking_kwarg`` onto the chat call as Ollama's top-level ``think``
bool. This matters because Ollama's per-model default is inconsistent
(some chat models default thinking ON), so without forwarding the flag a
stage that requested reasoning-off could still run reasoning-on.
"""
from __future__ import annotations

import os
import time
from typing import override

from kernel.abstractions import InferenceProvider, LlmCall, LlmResponse, LlmStatus
from kernel.enums import PhaseName
from kernel.execution_env import BoundExecutionEnv

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _to_ollama_message(m: dict) -> dict:
    """Translate one kernel message to Ollama's chat shape.

    Text content passes through. Multimodal content (an OpenAI content-part
    LIST — ``[{type:text}, {type:image_url, image_url:{url:data:…;base64,…}}]``)
    is split into Ollama's separate ``images`` field: the text parts join
    into ``content`` and each ``image_url`` data URL is decoded to bare
    base64 (mirrors ``vision._vision_call``'s LOCAL branch, which already
    sends ``{role, content, images}`` to Ollama).
    """
    content = m["content"]
    if isinstance(content, str):
        return {"role": m["role"], "content": content}
    texts: list[str] = []
    images: list[str] = []
    for part in content:
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif part.get("type") == "image_url":
            url = part.get("image_url", {}).get("url", "")
            # Strip the ``data:<media>;base64,`` prefix → bare base64.
            images.append(url.split(",", 1)[-1] if url.startswith("data:") else url)
    return {"role": m["role"], "content": " ".join(texts), "images": images}


class OllamaProvider(InferenceProvider):
    def __init__(self):
        self._client = None
        self._injected_errors: dict[PhaseName, list[LlmStatus]] = {
            phase: [] for phase in PhaseName
        }

    def _get_client(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=_OLLAMA_BASE_URL)
        return self._client

    @override
    def name(self) -> str:
        return "Ollama (local)"

    @override
    def run(self, call: LlmCall, execution_env: BoundExecutionEnv) -> LlmResponse:
        client = self._get_client()
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

        # num_ctx is the full context (input + output). Ollama defaults to
        # 4k regardless of model, silently truncating; pin to the spec's
        # window so the model matches its documented capacity. Local fans
        # out serially (max_parallelism=1) so we don't pay it N× at once.
        options = {
            "temperature": 0,
            "num_ctx": execution_env.model_spec.context_window(),
        }
        if call.max_tokens > 0:
            options["num_predict"] = call.max_tokens

        # Honor the per-call reasoning setting. `thinking_kwarg` renders the
        # model's reasoning wire-shape; for an Ollama-served model that's the
        # top-level `think` bool (NOT the OpenAI `extra_body` shape the
        # Tinfoil families use), so forward it when present. Absent (e.g. a
        # model family with no Ollama-native control) leaves Ollama at its own
        # default. Without this the flag was silently dropped — a model whose
        # Ollama default is thinking-on (e.g. qwen3.5) ran reasoning-on even
        # when the stage requested off.
        chat_kwargs: dict = {}
        thinking_shape = execution_env.model_spec.thinking_kwarg(
            execution_env.thinking
        )
        if "think" in thinking_shape:
            chat_kwargs["think"] = thinking_shape["think"]

        try:
            # Dispatch off the MODEL: the embeddings model embeds a BATCH of
            # inputs and returns one vector per input (mirrors TinfoilProvider's
            # branch). The kernel batch data model is one LlmResponse carrying N
            # vectors (payload: list[list[float]]); the phase maps vector[i] back
            # to the record behind message[i]. Ollama's client.embed accepts a
            # list for `input` and returns a vector per element, same shape as
            # the Tinfoil embeddings.create(input=[...]) batch call.
            if model == "nomic-embed-text":
                for m in call.messages:
                    assert m["role"] == "user"
                    assert isinstance(m["content"], str)
                inputs = [m["content"] for m in call.messages]
                resp = client.embed(model=model, input=inputs)
                embs = getattr(resp, "embeddings", None) or []
                payload = [list(e) for e in embs]
                prompt_tokens = getattr(resp, "prompt_eval_count", 0) or 0
                duration = time.monotonic() - t0
                return LlmResponse(
                    None, payload, None, prompt_tokens, 0, 0, ttft, duration
                )
            pieces = []
            for chunk in client.chat(
                model=model,
                messages=[_to_ollama_message(m) for m in call.messages],
                options=options,
                stream=True,
                **chat_kwargs,
            ):
                chunk_content = (
                    getattr(getattr(chunk, "message", None), "content", "") or ""
                )
                if chunk_content:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    pieces.append(chunk_content)
                    if call.stream_handler:
                        call.stream_handler(chunk_content)
                prompt_tokens = (
                    getattr(chunk, "prompt_eval_count", None) or prompt_tokens
                )
                completion_tokens = (
                    getattr(chunk, "eval_count", None) or completion_tokens
                )
                if getattr(chunk, "done_reason", None) == "length":
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
