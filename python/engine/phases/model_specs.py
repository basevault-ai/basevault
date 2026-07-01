"""Native kernel model specs for the production pipeline (issue #912).

A kernel ``ModelSpec`` is "everything the scheduler / provider needs to size
and pace one (model + provider) combination": the model id, context window,
reasoning-wire translation, concurrency cap, and request pacing.

This module defines all of that NATIVELY — it does NOT wrap ``llm.ModelSpec``
or call ``llm.get_mode_spec`` / ``llm.max_workers`` / ``llm._reasoning_kwargs``
/ the per-stage routing in ``llm.py``. The previous wrapper used one mode-anchor
model for EVERY stage (so patterns / insights / actions wrongly ran on the
extract model) and ``seconds_between_requests = 0`` (so a whole stage's calls
fired in the same millisecond instead of being paced). Both are fixed here.

Per-stage model + reasoning come from the ship-default map below, overridable
by ``~/.basevault/config.json``'s ``stage_models``. Request pacing mirrors the
legacy scheduler cadence (dev 1s / gpt-oss 4s / prod 20s; embeddings 0.1s).

Provider binding:
  * ``Mode.TEE``   → the kernel's attested ``TinfoilProvider`` (in kernel).
  * ``Mode.LOCAL`` → app-layer ``OllamaProvider`` / ``MlxProvider``.
The eval-only non-attested provider stays under ``testing/`` and is
never referenced here.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, override

from kernel.abstractions import InferenceProvider, ModelSpec
from kernel.enums import Environment
from kernel.scheduler import ThrottledScheduler

from engine.llm import Mode  # the Mode enum is a shared type, not spec config

if TYPE_CHECKING:
    from kernel.abstractions import CachingHook
    from kernel.enums import PhaseName
    from kernel.execution_env import ExecutionEnv


# ── Model registry ───────────────────────────────────────────────────────────
# model_id → (context_window, max_output, is_small_model). Small models get a
# Tinfoil concurrency carve-out (extra router headroom).
_TEE_MODELS: dict[str, tuple[int, int, bool]] = {
    "gpt-oss-120b":     (128_000, 128_000, True),
    "kimi-k2-6":        (256_000, 256_000, False),
    "gemma4-31b":       (256_000, 256_000, True),
    "glm-5-2":          (384_000, 384_000, False),
    "nomic-embed-text": (8_192,   8_192,   True),    # embeddings
}

# Ship-default per-stage model (TEE). config.json `stage_models` overrides.
_DEFAULT_STAGE_MODEL: dict[str, str] = {
    "vision":          "kimi-k2-6",
    "extract":         "gpt-oss-120b",
    "entities":        "gpt-oss-120b",
    "entities_dedupe": "gemma4-31b",
    "patterns":        "kimi-k2-6",
    "insights":        "kimi-k2-6",
    "actions":         "kimi-k2-6",
}

# PhaseName → the stage key used for model / reasoning / pacing resolution.
# (entity_dedupe routes to its own model; grouping/summarize share "entities".)
_PHASE_STAGE_KEY: dict[str, str] = {
    "INGESTION":            "vision",
    "EXTRACTION_SPLITTER":  "extract",
    "EXTRACTION_LLM":       "extract",
    "EXTRACTION_COMPLETION": "extract",
    "ENTITY_GROUPING":      "entities",
    "ENTITY_SUMMARIZE":     "entities",
    "ENTITY_DEDUPE":        "entities_dedupe",
    "PATTERNS":             "patterns",
    "INSIGHTS":             "insights",
    "ACTIONS":              "actions",
    "EMBEDDINGS":           "embeddings",
    "CHAT":                 "chat",
}

# Multi-model sentinels: a stage configured with one of these dispatches its
# calls across the constituent models, load-balanced — resolved to a kernel
# CombinedSpec (one constituent ModelSpec each, round-robined by execution_env).
_MULTI_MODEL_SENTINELS: dict[str, tuple[str, ...]] = {
    "kimi+glm": ("kimi-k2-6", "glm-5-2"),
}

# Reasoning is eligible only for these (model, stage) pairs; the actual on/off
# still comes from config (defaults OFF — every reasoning-on run has been
# slow/expensive, so the operational default stays off unless config opts in).
_REASONING_MODELS = {"gpt-oss-120b", "kimi-k2-6", "gemma4-31b", "glm-5-2"}
_REASONING_STAGES = {
    "vision", "extract", "entities", "entities_dedupe",
    "patterns", "insights", "actions",
}

# ── Request pacing (seconds between dispatches) ──────────────────────────────
_PROD_INTERVAL_S = 20.0
_DEV_INTERVAL_S = 1.0
_GPT_OSS_INTERVAL_S = 4.0
_EMBED_INTERVAL_S = 0.1

# ── Concurrency ──────────────────────────────────────────────────────────────
# Cloud cap matches the Tinfoil router's per-model MaxRequestsWaiting; small
# models get 2x headroom. LOCAL serializes on one GPU.
_TEE_POOL = 16
_TEE_POOL_SMALL = 32
_LOCAL_POOL = 1


def _is_dev() -> bool:
    return os.environ.get("IS_DEV", "").strip().lower() in {"1", "true", "yes", "on"}


def _read_config() -> dict:
    """The app config (stage_models etc.). Delegates to the ONE existing
    config reader rather than re-parsing config.json here — we reimplement the
    model specs, not the app-config plumbing. (It reads the fixed
    ~/.basevault/config.json; my earlier BASEVAULT_VAULT_ROOT-based read was
    the bug that lost stage_models under the app's env.)"""
    try:
        from engine.llm import _read_app_config
        return _read_app_config()
    except Exception:
        return {}


def _stage_model_raw(stage: str) -> str:
    """The raw TEE model string for a stage: config.json stage_models override
    (which may be a multi-model SENTINEL like ``kimi+glm``), else the ship
    default. Unknown stages fall back to the extract anchor."""
    cfg = _read_config()
    sm = cfg.get("stage_models")
    if isinstance(sm, dict):
        entry = sm.get(stage)
        if isinstance(entry, dict):
            m = entry.get("model")
            if isinstance(m, str) and (m in _MULTI_MODEL_SENTINELS or m in _TEE_MODELS):
                return m
    return _DEFAULT_STAGE_MODEL.get(stage, "gpt-oss-120b")


def _stage_models(stage: str) -> list[str]:
    """The concrete model id(s) a stage dispatches across — one for a single
    model, the constituent list for a multi-model sentinel."""
    raw = _stage_model_raw(stage)
    return list(_MULTI_MODEL_SENTINELS.get(raw, (raw,)))


def _stage_model_id(stage: str) -> str:
    """The PRIMARY single model id for a stage (first constituent of a
    sentinel). Used for the mode anchor + reasoning-whitelist resolution."""
    return _stage_models(stage)[0]


def _stage_reasoning(stage: str) -> bool:
    """Resolved reasoning flag: whitelisted (model, stage) AND config opts in.
    Defaults OFF."""
    if _stage_model_id(stage) not in _REASONING_MODELS or stage not in _REASONING_STAGES:
        return False
    cfg = _read_config()
    sm = cfg.get("stage_models")
    if isinstance(sm, dict) and isinstance(sm.get(stage), dict):
        return bool(sm[stage].get("reasoning", False))
    return False


def _interval_for(model_id: str) -> float:
    if model_id == "nomic-embed-text":
        return _EMBED_INTERVAL_S
    if _is_dev():
        return _DEV_INTERVAL_S
    if model_id == "gpt-oss-120b":
        return _GPT_OSS_INTERVAL_S
    return _PROD_INTERVAL_S


def _reasoning_kwargs(model_id: str, enabled: bool) -> dict[str, Any]:
    """Per-model-family reasoning wire shape (native; mirrors llm._reasoning_kwargs)."""
    m = model_id.lower()
    if "gpt-oss" in m:
        effort = os.environ.get("EVAL_PERF_GPT_OSS_EFFORT", "medium") if enabled else "low"
        return {"reasoning_effort": effort}
    if "kimi" in m:
        return {"extra_body": {"chat_template_kwargs": {"thinking": bool(enabled)}}}
    if "gemma" in m or "glm" in m:
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": bool(enabled)}}}
    if "minimax" in m:
        return {"extra_body": {"reasoning_split": True}}
    if "qwen" in m:
        # Qwen3.5 is the local Ollama default. Ollama controls thinking via a
        # top-level ``think`` bool (NOT the OpenAI ``extra_body`` shape the
        # Tinfoil-served families use); OllamaProvider.run forwards it. Qwen's
        # own Ollama default is thinking-ON, so this is what makes a
        # reasoning-off stage actually run reasoning-off.
        return {"think": bool(enabled)}
    return {}


class PipelineModelSpec(ModelSpec):
    """Native kernel ``ModelSpec`` — one per (provider, model). Carries its own
    context window, reasoning translation, concurrency cap, and request pacing,
    with no dependency on ``llm.ModelSpec``."""

    def __init__(
        self,
        provider: InferenceProvider,
        model_id: str,
        context_window: int = 131_000,
        max_parallelism: int = 16,
        seconds_between_requests: float = 0.0,
    ):
        # Set the fields the scheduler reads BEFORE constructing it
        # (ThrottledScheduler.__init__ calls them).
        self._model_id = model_id
        self._context_window = context_window
        self._max_parallelism = max_parallelism
        self._seconds_between_requests = seconds_between_requests
        super().__init__(provider, ThrottledScheduler(self))

    @override
    def model(self) -> str:
        return self._model_id

    @override
    def context_window(self) -> int:
        return self._context_window

    @override
    def thinking_kwarg(self, enabled: bool) -> dict[str, Any]:
        return _reasoning_kwargs(self._model_id, enabled)

    @override
    def max_parallelism(self, environment: Environment) -> int:
        return self._max_parallelism

    @override
    def seconds_between_requests(self, environment: Environment) -> float:
        return self._seconds_between_requests


# One spec per (mode, model) so phases sharing a model share its scheduler
# (one pool, one pacing) — patterns / insights / actions all run on the one
# kimi spec; extract on its own gpt-oss spec; dedupe on its own gemma spec.
_SPEC_CACHE: dict[str, PipelineModelSpec] = {}


def _tee_provider() -> InferenceProvider:
    from kernel.tinfoil_provider import TinfoilProvider
    return TinfoilProvider()


def _local_provider() -> InferenceProvider:
    cfg = _read_config()
    if str(cfg.get("local_backend", "mlx")).lower() == "ollama":
        from engine.ollama_provider import OllamaProvider
        return OllamaProvider()
    from engine.mlx_provider import MlxProvider
    return MlxProvider()


def _tee_spec_for_model(model_id: str) -> PipelineModelSpec:
    key = f"tee::{model_id}"
    if key not in _SPEC_CACHE:
        ctx, _max_out, small = _TEE_MODELS.get(model_id, (128_000, 128_000, False))
        pool = _TEE_POOL_SMALL if small else _TEE_POOL
        _SPEC_CACHE[key] = PipelineModelSpec(
            _tee_provider(), model_id, ctx, pool, _interval_for(model_id),
        )
    return _SPEC_CACHE[key]


# CombinedSpec instances for multi-model stages, memoized by sentinel name.
_COMBINED_CACHE: dict[str, ModelSpec] = {}


def _tee_spec_for_stage(stage: str) -> ModelSpec:
    """The TEE spec for a stage. A single model → its ``PipelineModelSpec``; a
    multi-model sentinel (e.g. ``kimi+glm``) → a kernel ``CombinedSpec`` that
    round-robins its calls across both constituents (each with its own
    scheduler + pacing), giving the dual-enclave throughput the config asks
    for."""
    models = _stage_models(stage)
    if len(models) == 1:
        return _tee_spec_for_model(models[0])
    raw = _stage_model_raw(stage)
    if raw not in _COMBINED_CACHE:
        from kernel.abstractions import CombinedSpec
        _COMBINED_CACHE[raw] = CombinedSpec(
            raw, [_tee_spec_for_model(m) for m in models]
        )
    return _COMBINED_CACHE[raw]


def _local_spec() -> PipelineModelSpec:
    cfg = _read_config()
    if str(cfg.get("local_backend", "mlx")).lower() == "ollama":
        model_id = str(cfg.get("local_model") or "qwen3.5:9b")
        ctx = 64_000
    else:
        model_id = str(cfg.get("local_mlx_model") or "mlx-community/Qwen3.5-9B-4bit")
        ctx = 32_768
    key = f"local::{model_id}"
    if key not in _SPEC_CACHE:
        _SPEC_CACHE[key] = PipelineModelSpec(
            _local_provider(), model_id, ctx, _LOCAL_POOL, _interval_for(model_id),
        )
    return _SPEC_CACHE[key]


def _stage_key_for_phase(phase_name: "PhaseName") -> str:
    return _PHASE_STAGE_KEY.get(phase_name.name, phase_name.stage_name().value)


# ── Extension (eval-only) mode registry ──────────────────────────────────────
# The production engine resolves only the core modes (``Mode.TEE`` / attested,
# ``Mode.LOCAL``). Eval-only modes (e.g. the non-attested ``"test"`` mode)
# register a single anchor ``ModelSpec`` here at runtime — used for EVERY LLM
# phase of a pipeline run in that mode, mirroring the legacy
# ``llm.register_mode(MODE_TEST, spec)`` single-anchor behavior on the new
# kernel path. The eval tree (``testing/eval/_eval_specs``) populates this via
# ``BASEVAULT_RUNTIME_EXTENSIONS`` at subprocess startup; the production engine
# ships it EMPTY, so no non-attested provider is referenced from this module
# (the same trust boundary the ``testing/``-only eval-provider code preserves).
_EXTENSION_MODE_SPECS: dict[str, ModelSpec] = {}


def register_mode_spec(mode_token: str, spec: ModelSpec) -> None:
    """Register a single anchor kernel ``ModelSpec`` for an eval-only mode
    (e.g. ``"test"``). Used for every LLM phase of a pipeline run in that mode."""
    _EXTENSION_MODE_SPECS[str(mode_token)] = spec


def _mode_token(mode) -> str:
    """Normalize a mode (``Mode`` str-enum or a bare extension string) to its
    string token for registry lookup."""
    return mode.value if isinstance(mode, Mode) else str(mode)


def _extension_spec(mode):
    """The registered anchor spec for an extension mode, or ``None`` for the
    core modes / an unregistered mode."""
    return _EXTENSION_MODE_SPECS.get(_mode_token(mode))


def spec_for_stage(phase_name: "PhaseName", mode: Mode) -> ModelSpec:
    """The kernel ``ModelSpec`` for a phase: the stage's resolved model(s) (per
    the default map / config.json), bound to the mode's provider + pacing. A
    multi-model sentinel resolves to a ``CombinedSpec``."""
    stage = _stage_key_for_phase(phase_name)
    if mode == Mode.LOCAL:
        return _local_spec()
    if mode == Mode.TEE:
        return _tee_spec_for_stage(stage)
    ext = _extension_spec(mode)
    if ext is not None:
        return ext
    raise ValueError(f"No kernel ModelSpec builder for mode {mode!r}")


def spec_for_mode(mode: Mode) -> PipelineModelSpec:
    """The mode anchor spec (the extract-stage model). Kept for callers that
    want a representative spec without a phase (tests / eval)."""
    if mode == Mode.LOCAL:
        return _local_spec()
    if mode == Mode.TEE:
        return _tee_spec_for_model(_stage_model_id("extract"))
    ext = _extension_spec(mode)
    if ext is not None:
        return ext
    raise ValueError(f"No kernel ModelSpec builder for mode {mode!r}")


def embedding_spec_for_mode(mode: Mode) -> PipelineModelSpec:
    """Kernel ``ModelSpec`` for the embeddings model (``nomic-embed-text``, the
    provider's embeddings branch). TEE → Tinfoil; LOCAL → Ollama. 0.1s pacing."""
    key = f"embed::{_mode_token(mode)}"
    if key not in _SPEC_CACHE:
        # Only LOCAL embeds via Ollama; TEE and eval-only modes (e.g. "test")
        # use the attested Tinfoil embeddings branch — commodity cloud LLM
        # providers don't serve nomic-embed-text, so eval-mode embeddings
        # always run on the real embed provider.
        if mode == Mode.LOCAL:
            from engine.ollama_provider import OllamaProvider
            provider: InferenceProvider = OllamaProvider()
            pool = _LOCAL_POOL
        else:
            provider = _tee_provider()
            pool = _TEE_POOL_SMALL
        _SPEC_CACHE[key] = PipelineModelSpec(
            provider, "nomic-embed-text", 8_192, pool, _EMBED_INTERVAL_S,
        )
    return _SPEC_CACHE[key]


def vision_spec_for_mode(mode: Mode) -> PipelineModelSpec:
    """Kernel ``ModelSpec`` for the vision (INGESTION) model — a dedicated
    vision model, not the chat anchor. TEE → the configured vision model;
    LOCAL → the local-backend model via Ollama."""
    if mode == Mode.TEE:
        return _tee_spec_for_model(_stage_model_id("vision"))
    ext = _extension_spec(mode)
    if ext is not None:
        return ext
    key = "vision::local"
    if key not in _SPEC_CACHE:
        from engine.ollama_provider import OllamaProvider
        cfg = _read_config()
        model_id = str(cfg.get("local_model") or "qwen3.5:9b")
        _SPEC_CACHE[key] = PipelineModelSpec(
            OllamaProvider(), model_id, 64_000, _LOCAL_POOL, _interval_for(model_id),
        )
    return _SPEC_CACHE[key]


def chat_spec_for_mode(mode: Mode) -> ModelSpec:
    """Kernel ``ModelSpec`` for the CHAT phase. Chat does NOT route through
    ``stage_models`` — its model is configured under the SEPARATE ``chatbot``
    config key (``resolve_chatbot_from_config``, ship default ``glm-5-2`` =
    ``DEFAULT_CHATBOT_MODEL``), which the legacy sidecar forces over the
    ``tee_model`` anchor (``_force_model_id``). Resolving chat through the
    stage-model map instead silently fell back to the gpt-oss anchor (``chat``
    is absent from ``_DEFAULT_STAGE_MODEL``) — the migration bug that ran chat
    on gpt-oss. LOCAL uses the local chat model, mirroring legacy."""
    if mode == Mode.LOCAL:
        return _local_spec()
    if mode != Mode.TEE:
        ext = _extension_spec(mode)
        if ext is not None:
            return ext
        raise ValueError(f"No kernel ModelSpec builder for mode {mode!r}")
    from engine.chatbot import DEFAULT_CHATBOT_MODEL, resolve_chatbot_from_config
    model_id = (
        resolve_chatbot_from_config(_read_config()).get("model")
        or DEFAULT_CHATBOT_MODEL
    )
    return _tee_spec_for_model(model_id)


def build_stage_env(
    phase_name: "PhaseName",
    mode: Mode,
    session_id: str | None = None,
    thinking: bool | None = None,
    payload_sink=None,
    failure_payload_sink=None,
    extra_hooks=None,
) -> "ExecutionEnv":
    """Build an ``ExecutionEnv`` for ONE production stage phase: the stage's
    resolved per-stage model spec + the ``KernelTelemetryHook`` (so per-call
    ``llm-calls.jsonl`` records are reproduced) + the disk cache.

    ``thinking`` defaults to the stage's resolved reasoning flag (whitelist AND
    config; OFF by default). ``payload_sink`` defaults to ``llm._stamp_full_io``
    so the dev-tab payload view works. ``extra_hooks`` attaches the runner's
    live-progress hook."""
    from kernel.enums import PhaseName
    from kernel.execution_env import ExecutionEnv

    from engine.phases.telemetry_hook import KernelTelemetryHook, stage_label
    from engine.phases.kernel_cache import KernelDiskCache

    if phase_name == PhaseName.INGESTION:
        spec = vision_spec_for_mode(mode)
    elif phase_name == PhaseName.EMBEDDINGS:
        spec = embedding_spec_for_mode(mode)
    elif phase_name == PhaseName.CHAT:
        spec = chat_spec_for_mode(mode)
    else:
        spec = spec_for_stage(phase_name, mode)

    if thinking is None:
        if phase_name == PhaseName.CHAT:
            # Chat reasoning comes from the chatbot config (default OFF), NOT
            # the per-stage reasoning map — same source as the chat model.
            from engine.chatbot import resolve_chatbot_from_config
            thinking = bool(
                resolve_chatbot_from_config(_read_config()).get("reasoning")
            )
        else:
            thinking = _stage_reasoning(_stage_key_for_phase(phase_name))
    if payload_sink is None:
        from engine.llm import _stamp_full_io
        payload_sink = _stamp_full_io
        # Pipeline default path (chat passes its own payload_sink + handles
        # failures via _capture_payload): always-on failed-prompt capture so a
        # from_status failure leaves its prompt in llm-payloads.jsonl.
        if failure_payload_sink is None:
            from engine.llm import _log_call_failure_payload
            failure_payload_sink = _log_call_failure_payload

    env = ExecutionEnv()
    env.register_spec(phase_name, spec, spec, thinking)
    env.register_llm_hook(
        KernelTelemetryHook(
            session_id=session_id, payload_sink=payload_sink,
            failure_payload_sink=failure_payload_sink, mode=mode
        )
    )
    for hook in extra_hooks or []:
        env.register_llm_hook(hook)
    env.register_caching_hook(KernelDiskCache(stage=stage_label(phase_name)))
    return env


def build_execution_env(
    phase: "PhaseName",
    mode: Mode,
    thinking: bool = False,
    caching_hook: "CachingHook | None" = None,
) -> "ExecutionEnv":
    """Construct a kernel ``ExecutionEnv`` with the phase's resolved per-stage
    spec registered (spec doubles as its own fallback). The caller binds a
    phase to this env via ``Phase.run``."""
    from kernel.execution_env import ExecutionEnv

    spec = spec_for_stage(phase, mode)
    env = ExecutionEnv()
    env.register_spec(phase, spec, spec, thinking)
    if caching_hook is not None:
        env.register_caching_hook(caching_hook)
    return env
