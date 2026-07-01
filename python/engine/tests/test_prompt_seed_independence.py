"""
Every stage's prompt builder must be PYTHONHASHSEED-independent —
byte-for-byte.

The LLM disk cache keys on the exact prompt bytes. If set/dict
iteration order leaks into prompt assembly, the cache key changes on
identical inputs across processes (Python randomizes string hashing
per process unless PYTHONHASHSEED is pinned), so a cached-input run
silently recomputes with a fresh, non-deterministic LLM call instead
of replaying. That broke reproducibility and contaminated eval /
dataset accrual.

This covers a prompt-assembly entry point for EVERY stage that
produces an LLM call — extract, entities, entities_dedupe, patterns,
insights, actions — not just the entities path where the leak was
found. The entities-family fixtures force the witnessed
`W.N.P. BARBELLION` / `W.N.P. Barbellion` alias tie so a non-total
sort key surfaces structurally (not 1-in-N). The other stages iterate
lists / `sorted()` over distinct keys today; the fixtures still feed
them multi-element collections in scrambled order so any future
set/dict-iteration leak is caught the same way.

PYTHONHASHSEED cannot be changed inside a running interpreter, so each
builder is assembled in a fresh subprocess under several distinct
seeds; the bytes must be identical across all of them.

Plain unit-level: no pipeline run, no LLM call, no cache, no
integration. Just: assemble under seed A, under seed B, compare bytes.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HELPER = Path(__file__).parent / "_prompt_seed_helper.py"
# python/ package root so the helper (launched as a script) resolves its
# fully-qualified `from engine import …` imports.
_PYTHON_ROOT = Path(__file__).resolve().parents[2]

# Distinct seeds: 0 (the value the pipeline pins), 1, and a large
# arbitrary value. Pre-fix these produce divergent set-iteration
# orders; post-fix the total-ordered sort keys make every builder's
# bytes identical across all of them.
_SEEDS = ["0", "1", "2147483647"]

# Keep in sync with `_prompt_seed_helper._BUILDERS` (one per stage,
# plus the entities sub-paths). A typo here can't pass silently: the
# helper raises KeyError on an unknown builder → non-zero exit →
# `_render` asserts.
_BUILDERS = [
    "extract",
    "entities_block",
    "dedupe_materialize",
    "dedupe_materialize_s1",
    "dedupe_merge",
    "context_block",
    "patterns",
    "insights",
    "actions",
]


def _render(builder: str, seed: str) -> bytes:
    proc = subprocess.run(
        [sys.executable, str(HELPER), builder],
        capture_output=True,
        # Inherit the ambient env (the established subprocess-helper
        # convention; keeps HOME/TMPDIR/etc. available for imports), add
        # python/ on PYTHONPATH for the engine.* imports, and override the
        # one var under test.
        env={**os.environ, "PYTHONPATH": str(_PYTHON_ROOT),
             "PYTHONHASHSEED": seed},
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"helper failed for {builder!r} (seed={seed}): "
        f"{proc.stderr.decode(errors='replace')}"
    )
    return proc.stdout


@pytest.mark.parametrize("builder", _BUILDERS)
def test_prompt_is_seed_independent(builder: str) -> None:
    renders = {seed: _render(builder, seed) for seed in _SEEDS}

    baseline = renders[_SEEDS[0]]
    assert baseline, f"{builder}: helper produced empty prompt"

    for seed in _SEEDS[1:]:
        assert renders[seed] == baseline, (
            f"{builder}: prompt bytes differ between PYTHONHASHSEED="
            f"{_SEEDS[0]} and PYTHONHASHSEED={seed}. A set/dict "
            f"iteration order is leaking into prompt assembly — make "
            f"its sort key total (append the raw string as the final "
            f"component, like entities.py's alias keys)."
        )


def test_collision_pair_actually_exercised() -> None:
    """Guard the guard: prove the fixtures genuinely carry a
    hash-collision alias pair into the assembled prompt. If a future
    refactor drops the pair, the seed-independence test would pass
    vacuously (a one-element set never reorders) — this fails first
    and loudly."""
    text = _render("dedupe_materialize", "0").decode("utf-8")
    assert "W.N.P. BARBELLION" in text and "W.N.P. Barbellion" in text, (
        "fixture no longer carries the BARBELLION collision pair into "
        "the dedupe prompt; the seed-independence test would be vacuous"
    )
