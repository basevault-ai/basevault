"""
Core abstract classes used in the rest of the BaseVault repo.

Each class must be either abstract of final.
In an abstract class, each method  be either abstract or final.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, final

from kernel.cancellation_manager import CancellationManager
from kernel.enums import (
    AttestationType,
    Environment,
    JobName,
    LlmStatus,
    PhaseName,
    RetryType,
)
from kernel.utils import PermaId

if TYPE_CHECKING:
    from .execution_env import BoundExecutionEnv, ExecutionEnv


@dataclass
class LlmCall:
    id: str
    messages: list[dict[str, str | list[dict]]]  # Dict with 'role' & 'content' keys.
    max_tokens: int
    previous_call_id: str  # In case this is a retry
    context: Any  # Extra context useful when halving / sampling the call.
    stream_handler: Callable[[str], None] | None


@dataclass
class LlmResponse:
    status: LlmStatus | None
    payload: str | list[list[float]] | None
    exception: Exception | None
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    ttft: float | None
    duration: float | None

    @staticmethod
    def empty() -> LlmResponse:
        return LlmResponse(None, None, None, 0, 0, 0, None, None)

    @staticmethod
    def from_status(status: LlmStatus, duration: float) -> LlmResponse:
        return LlmResponse(status, None, None, 0, 0, 0, None, duration)


@dataclass
class PhaseResult:
    data: Any


@dataclass
class Attestation:
    model: str
    enclave: str
    repo: str
    version: str
    timestamp: float
    payload_compressed: str
    payload_decompressed: bytes
    attestation_type: AttestationType | None
    published_measurements: list[str]
    error: str | None


class InferenceProvider(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, call: LlmCall, execution_env: BoundExecutionEnv) -> LlmResponse:
        """Connect to provider with request, stream call, return response"""
        pass

    @abstractmethod
    def inject_errors(self, phase: PhaseName, errors: list[LlmStatus]) -> None:
        """Inject a set of errors for a particular phase"""
        pass

    @abstractmethod
    def attestations(self, models: set[str]) -> list[Attestation]:
        """Returns attestation metadata; does NOT perform full attestation."""
        pass


class Scheduler(ABC):
    model_spec: ModelSpec

    def __init__(self, model_spec: ModelSpec):
        self.model_spec = model_spec

    @abstractmethod
    def run(
        self, call: LlmCall, execution_env: BoundExecutionEnv, is_retry: bool
    ) -> Future[LlmResponse]:
        pass

    @abstractmethod
    def abort(self, calls: set[str], skip: bool = False) -> None:
        pass


# Each model + provider combination is represented by a different ModelSpec class.
class ModelSpec(ABC):
    def __init__(self, provider: InferenceProvider, scheduler: Scheduler):
        self._inference_provider: InferenceProvider = provider
        self._scheduler: Scheduler = scheduler

    def inference_provider(self) -> InferenceProvider:
        return self._inference_provider

    def scheduler(self) -> Scheduler:
        return self._scheduler

    def name(self) -> str:
        return self._inference_provider.name() + "::" + self.model()

    @abstractmethod
    def model(self) -> str:
        pass

    @abstractmethod
    def context_window(self) -> int:
        pass

    @abstractmethod
    def thinking_kwarg(self, enabled: bool) -> dict[str, Any]:
        pass

    @abstractmethod
    def max_parallelism(self, environment: Environment) -> int:
        """Number of calls that can be run in parallel"""
        pass

    @abstractmethod
    def seconds_between_requests(self, environment: Environment) -> float:
        """Seconds between consecutive executions."""
        pass


class CombinedSpec(ModelSpec):
    def __init__(self, model: str, specs: list[ModelSpec]):
        self._model: str = model
        self._specs: list[ModelSpec] = specs
        self._current_spec = 0
        self._current_spec_lock: threading.Lock = threading.Lock()

    @final
    def next_spec(self) -> ModelSpec:
        with self._current_spec_lock:
            spec = self._specs[self._current_spec]
            self._current_spec = (self._current_spec + 1) % len(self._specs)
            return spec

    @final
    def specs(self) -> list[ModelSpec]:
        return self._specs

    @final
    def inference_provider(self) -> InferenceProvider:
        raise NotImplementedError("Unreachable")

    @final
    def scheduler(self) -> Scheduler:
        raise NotImplementedError("Unreachable")

    @final
    def name(self) -> str:
        return " + ".join(spec.name() for spec in self._specs)

    @final
    def model(self) -> str:
        return " + ".join(spec.model() for spec in self._specs)

    @final
    def context_window(self) -> int:
        raise NotImplementedError("Unreachable")

    @final
    def thinking_kwarg(self, enabled: bool) -> dict[str, Any]:
        raise NotImplementedError("Unreachable")

    @final
    def max_parallelism(self, environment: Environment) -> int:
        raise NotImplementedError("Unreachable")

    @final
    def seconds_between_requests(self, environment: Environment) -> float:
        raise NotImplementedError("Unreachable")


class CachingHook(ABC):
    @abstractmethod
    def save(self, key: str, call: LlmCall, response: LlmResponse) -> None:
        pass

    @abstractmethod
    def has(self, key: str) -> bool:
        pass

    @abstractmethod
    def load(self, key: str) -> tuple[LlmCall, LlmResponse] | None:
        pass


class LlmHook(ABC):
    @abstractmethod
    def hook_llm_queued(self, call: LlmCall, execution_env: BoundExecutionEnv) -> None:
        pass

    @abstractmethod
    def hook_llm_started(self, call: LlmCall, execution_env: BoundExecutionEnv) -> None:
        pass

    @abstractmethod
    def hook_llm_completed(
        self,
        call: LlmCall,
        execution_env: BoundExecutionEnv,
        response: LlmResponse,
        retry: RetryType,
        from_cache: bool,
        should_cache: bool,
    ) -> None:
        pass


class Phase(ABC):
    def __init__(self, job: Job):
        self.job: Job = job
        self.call_count: int = 0

    @final
    def run(
        self,
        input: PhaseResult,
        execution_env: ExecutionEnv,
        cancellation_manager: CancellationManager,
    ) -> PhaseResult:
        """Overarching phase execution procedure."""
        if cancellation_manager.aborted():
            return PhaseResult({})

        # Bind execution spec to llm call, since execution spec
        # can change between this point and execution.
        bound_execution_env = None
        if self.name().does_llm_call():
            bound_execution_env = execution_env.bind(self, cancellation_manager)

        if not self.init_from_memory(input):
            self.init_from_disk()

        result: PhaseResult = self.run_main(bound_execution_env)
        self.checkpoint()
        return result

    @final
    def id(self) -> str:
        return f"{self.job.id.full_id}-{self.name().value}"

    @final
    def new_call(
        self,
        messages: list[dict[str, str | list[dict]]],
        max_tokens: int,
        previous_call: LlmCall | None = None,
        context: Any = None,
    ) -> LlmCall:
        self.call_count += 1
        new_id: str = f"{self.id()}-{self.call_count:04d}"
        previous_call_id = previous_call.id if previous_call else ""
        stream_handler = previous_call.stream_handler if previous_call else None
        return LlmCall(
            new_id,
            messages,
            max_tokens,
            previous_call_id,
            context,
            stream_handler,
        )

    @final
    def duplicate_call(self, call: LlmCall) -> LlmCall:
        return self.new_call(
            call.messages,
            call.max_tokens,
            call,
            call.context,
        )

    @abstractmethod
    def name(self) -> PhaseName:
        pass

    @abstractmethod
    def init_from_memory(self, input: PhaseResult) -> bool:
        pass

    @abstractmethod
    def init_from_disk(self) -> None:
        pass

    @abstractmethod
    def halve_llm_call(self, call: LlmCall) -> list[LlmCall]:
        """Halve llm call, for retries. At most one of sample_llm_call or halve_llm_call must be implemented."""
        pass

    @abstractmethod
    def sample_llm_call(self, call: LlmCall) -> LlmCall | None:
        """Sample llm call, for retries. At most one of sample_llm_call or halve_llm_call must be implemented."""
        pass

    @abstractmethod
    def validate(self, payload: str | list[list[float]]) -> LlmStatus | None:
        """Phase-specific validation of a response, e.g. PARSE_ERROR, SUCCESS_EMPTY, etc."""
        pass

    @abstractmethod
    def checkpoint(self) -> None:
        """Materializes context to recover from in case of crash."""
        pass

    @abstractmethod
    def run_main(self, execution_env: BoundExecutionEnv | None) -> PhaseResult:
        """For LLM calls, use execution_env.run() or execution_env.run_all() calls."""
        pass


# A job can be either a pipeline or a single chat turn.
class Job(ABC):
    def __init__(self):
        self.cancellation_manager = CancellationManager()
        self.id: PermaId = PermaId.new()

    @final
    def run(self, input: PhaseResult, execution_env: ExecutionEnv) -> PhaseResult:
        phases = self.generate_phases()
        next_input: PhaseResult = input
        for phase in phases:
            next_input = phase.run(next_input, execution_env, self.cancellation_manager)
        return next_input

    @abstractmethod
    def name(self) -> JobName:
        pass

    @abstractmethod
    def generate_phases(self) -> list[Phase]:
        pass
