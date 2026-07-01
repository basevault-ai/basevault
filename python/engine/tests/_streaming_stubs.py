"""
Streaming-shaped stubs for OpenAI-compatible chat clients.

Issue #104 part 1 switched every provider branch in `llm.complete()`
from non-streaming `chat.completions.create(...)` to streaming
`stream=True, stream_options={"include_usage": True}`. The pipeline
now consumes the response as an iterator of chunks and reads usage
totals from the final include-usage chunk. This module provides the
matching test stubs so tests don't each re-implement the chunk shape.

Shape contract — each chunk has:
    chunk.choices[0].delta.content              str | None
    chunk.choices[0].delta.reasoning_content    str | None
    chunk.choices[0].finish_reason              str | None  (only on final non-usage chunk)
    chunk.usage                                 None on intermediate chunks; populated on the
                                                final include-usage chunk

Usage:

    from engine.tests._streaming_stubs import make_stream_chunks

    chunks = make_stream_chunks(content="ok")   # iterator of chunk objects
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _StubDelta:
    content: str | None = None
    reasoning_content: str | None = None


@dataclass
class _StubChunkChoice:
    delta: _StubDelta
    finish_reason: str | None = None


@dataclass
class _StubCompletionTokensDetails:
    """Mirrors the OpenAI SDK's `CompletionTokensDetails` — providers
    that surface reasoning_tokens populate it on the include-usage
    chunk. Tests construct this only when the test cares about the
    reasoning_tokens path."""
    reasoning_tokens: int | None = None


@dataclass
class _StubUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5
    completion_tokens_details: _StubCompletionTokensDetails | None = None


@dataclass
class _StubChunk:
    """One chunk emitted by `chat.completions.create(stream=True)`.

    Either carries a delta (intermediate) or a usage payload (final
    include-usage chunk). Tests can mix and match — most production
    streams are: N content-delta chunks → 1 finish-reason chunk → 1
    usage chunk.
    """
    choices: list = field(default_factory=list)
    usage: _StubUsage | None = None


def make_stream_chunks(
    content: str = "ok",
    *,
    reasoning_content: str | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    reasoning_tokens: int | None = None,
    finish_reason: str | None = "stop",
) -> list[_StubChunk]:
    """Build the canonical chunk sequence for a single test response.

    Order:
      1. Optional reasoning delta (one chunk with reasoning_content).
         Emitted before content so timing assertions can cleanly
         distinguish TTFR from TTFT.
      2. One chunk with the visible response content.
      3. One chunk with finish_reason set (delta empty).
      4. One chunk with usage populated and choices empty.

    Tests that need richer streams (multi-chunk content, etc.) can
    bypass this helper and build chunks directly via _StubChunk.
    """
    chunks: list[_StubChunk] = []
    if reasoning_content:
        chunks.append(_StubChunk(choices=[
            _StubChunkChoice(delta=_StubDelta(reasoning_content=reasoning_content)),
        ]))
    chunks.append(_StubChunk(choices=[
        _StubChunkChoice(delta=_StubDelta(content=content)),
    ]))
    chunks.append(_StubChunk(choices=[
        _StubChunkChoice(delta=_StubDelta(), finish_reason=finish_reason),
    ]))
    details = (
        _StubCompletionTokensDetails(reasoning_tokens=reasoning_tokens)
        if reasoning_tokens is not None else None
    )
    chunks.append(_StubChunk(
        usage=_StubUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion_tokens_details=details,
        ),
    ))
    return chunks


class StreamingChatCompletions:
    """`chat.completions` stand-in. `create()` records call kwargs into
    `captured` and returns an iterator over a fresh chunk sequence.

    Re-callable: each invocation rebuilds the chunks so tests that fire
    `complete()` multiple times don't see an exhausted iterator.
    """

    def __init__(self, **chunk_kwargs):
        self._chunk_kwargs = chunk_kwargs
        self.captured: dict = {}

    def create(self, **kw):
        self.captured.clear()
        self.captured.update(kw)
        # Return a fresh iterator each call. iter() over a list is what
        # the production code consumes via `for chunk in stream`.
        return iter(make_stream_chunks(**self._chunk_kwargs))


class StreamingChat:
    def __init__(self, completions: StreamingChatCompletions):
        self.completions = completions


class StreamingClient:
    def __init__(self, completions: StreamingChatCompletions):
        self.chat = StreamingChat(completions)




