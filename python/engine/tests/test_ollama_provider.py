"""Regression tests for ``engine.ollama_provider.OllamaProvider``.

The invariant pinned here: local Ollama inference HONORS the per-call
reasoning setting. Two halves have to line up:

  1. ``model_specs._reasoning_kwargs`` renders the wire shape — for an
     Ollama-served model (qwen3.5) that's a top-level ``think`` bool.
  2. ``OllamaProvider.run`` forwards it onto ``client.chat``.

Before the qwen swap, qwen had no ``_reasoning_kwargs`` branch (rendered
nothing) AND the provider dropped the flag entirely, so a reasoning-off
stage silently ran reasoning-on (qwen's Ollama default is thinking-ON).
"""
from types import SimpleNamespace

import pytest

from engine.ollama_provider import OllamaProvider
from engine.phases.model_specs import _reasoning_kwargs
from kernel.abstractions import LlmCall
from kernel.enums import PhaseName


class _RecordingClient:
    """Fake ollama client: records the kwargs of the last ``chat`` call and
    yields a single well-formed chunk so ``run`` completes cleanly."""

    def __init__(self):
        self.chat_kwargs = None

    def chat(self, **kwargs):
        self.chat_kwargs = kwargs
        yield SimpleNamespace(
            message=SimpleNamespace(content="ok"),
            prompt_eval_count=3,
            eval_count=1,
            done_reason="stop",
        )


def _env(model_id: str, thinking: bool):
    """Minimal BoundExecutionEnv stand-in whose ``thinking_kwarg`` delegates
    to the real ``_reasoning_kwargs`` so the test exercises both halves."""
    return SimpleNamespace(
        thinking=thinking,
        model_spec=SimpleNamespace(
            model=lambda: model_id,
            context_window=lambda: 64_000,
            thinking_kwarg=lambda enabled: _reasoning_kwargs(model_id, enabled),
        ),
        phase=SimpleNamespace(name=lambda: PhaseName.EXTRACTION_LLM),
    )


def _call():
    return LlmCall(
        id="c1",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=128,
        previous_call_id="",
        context=None,
        stream_handler=None,
    )


# ── Half 1: the wire shape rendered for qwen ───────────────────────────────


@pytest.mark.parametrize("enabled", [True, False])
def test_reasoning_kwargs_renders_ollama_think_for_qwen(enabled):
    """qwen renders Ollama's top-level ``think`` bool (not the OpenAI
    ``extra_body`` shape), tracking the caller's enabled flag."""
    assert _reasoning_kwargs("qwen3.5:9b", enabled) == {"think": enabled}


# ── Half 2: the provider forwards it ───────────────────────────────────────


@pytest.mark.parametrize("thinking", [True, False])
def test_run_forwards_think_per_thinking_flag(thinking):
    """``run`` forwards ``think`` onto the chat call, matching the env's
    per-call ``thinking`` — reasoning-off stays off, reasoning-on stays on."""
    provider = OllamaProvider()
    client = _RecordingClient()
    provider._client = client  # bypass _get_client's ollama import

    resp = provider.run(_call(), _env("qwen3.5:9b", thinking))

    assert client.chat_kwargs is not None
    assert client.chat_kwargs["think"] is thinking
    assert resp.payload == "ok"  # ran through to a clean response


def test_run_omits_think_when_model_has_no_ollama_shape():
    """A model family that doesn't render a ``think`` key (e.g. gemma renders
    the OpenAI ``extra_body`` shape, which Ollama doesn't take) leaves the
    chat call without a ``think`` kwarg rather than passing an unusable one."""
    provider = OllamaProvider()
    client = _RecordingClient()
    provider._client = client

    provider.run(_call(), _env("gemma4:e4b", thinking=False))

    assert "think" not in client.chat_kwargs
