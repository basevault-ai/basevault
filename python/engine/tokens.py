"""Token counting helper — used to gate chunk sizes against model context budgets.

We deliberately do NOT use a real tokenizer (tiktoken, transformers, etc).
Instead, every token estimate in the pipeline is `len(text) / CHARS_PER_TOKEN`
with a single global constant. The constant is chosen to be the most
conservative (highest-token-count, smallest-chunk-out) value across the
models we run.

Per-model averages, observed:
    OpenAI cl100k (tiktoken):   ~4 chars/token (English prose)
    gemma4-31b:                 ~3.2
    minimax-m2.5:               ~3.5
    moonshotai/kimi-k2:         ~3.0
    gpt-oss-120b (~o200k-like): ~3.8

Setting `CHARS_PER_TOKEN = 3` means our estimate over-counts tokens
versus reality on every model. Consequences:
- Splitter packs chunks ~25% smaller than would be ideal for a
  cl100k-class tokenizer. We pay for this in extra LLM calls.
- The pre-call `input_overflow` defensive check can no longer fire
  spurious warnings on chunks the splitter just sized — both sides
  use the SAME estimate against the SAME budget, so a chunk that
  passes the splitter's sizing check passes the pre-call check
  too.
- We never blow a model's real context window from underestimation.

The alternative (per-model `chars_per_token` attached to each ModelSpec)
would recover that ~25% efficiency AND let downstream code pick the
right value automatically. See `TODO.md` — deferred for now;
adding it requires touching the splitter / content_extractor / pre-call
check, and the surface area is wide enough to risk regressions on a
working pipeline. Keep one global until we need the perf.

Usage::

    from engine.tokens import count_tokens, CHARS_PER_TOKEN

    n = count_tokens(text)            # len(text) // 3
    n = chars // CHARS_PER_TOKEN      # same thing, when you have chars
"""
from __future__ import annotations


# Single source of truth for the chars→tokens conversion ratio.
# Chosen as the lowest observed model average (~3 for kimi) so our
# estimate is conservative across all supported models. See module
# docstring for the full rationale.
CHARS_PER_TOKEN = 3


def count_tokens(text: str) -> int:
    """Approximate token count for `text`. Cheap, deterministic.

    Returns 0 for empty input. Otherwise `len(text) // CHARS_PER_TOKEN`,
    floored at 1 (a non-empty string is always at least one token).
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def tokens_from_chars(chars: int) -> int:
    """Convert a pre-counted char count to a token estimate.

    Use when you already have `len(text)` and don't want to keep the
    string around just to re-count. No 1-token floor — `chars=0`
    returns 0; `chars < CHARS_PER_TOKEN` rounds to 0.
    """
    return chars // CHARS_PER_TOKEN


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Total-prompt token estimate from ALL message contents.

    For 'does this whole prompt fit in context' checks (entities
    batching, the input_overflow defensive check in llm.complete).
    For sizing max_tokens, use `dynamic_max_tokens(payload_tokens, ...)`
    in llm.py — the caller knows the payload exactly and shouldn't
    have to guess from the assembled messages.

    Multimodal vision content (`content` is a list of {type, text} /
    {type, image_url} parts) counts only the text parts; image
    bytes are handled by the model's vision encoder, not the chat
    token budget. Without this carve-out, str(content) on a
    list-with-base64 would inflate the estimate by millions of
    "tokens" and trigger spurious input_overflow warnings on every
    vision call.
    """
    chars = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    txt = part.get("text") or ""
                    if isinstance(txt, str):
                        chars += len(txt)
    return tokens_from_chars(chars)
