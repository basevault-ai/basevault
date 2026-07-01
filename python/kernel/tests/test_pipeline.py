"""
Tests for kernel architecture abstractions.
"""

import copy
import threading
import time
import traceback
from datetime import datetime
from typing import Any, override

import pytest
from kernel.abstractions import (
    CachingHook,
    CombinedSpec,
    InferenceProvider,
    Job,
    LlmCall,
    LlmHook,
    LlmResponse,
    ModelSpec,
    Phase,
    PhaseResult,
)
from kernel.enums import Environment, JobName, LlmStatus, PhaseName, RetryType
from kernel.execution_env import BoundExecutionEnv, ExecutionEnv
from kernel.scheduler import ThrottledScheduler
from kernel.tinfoil_provider import TinfoilProvider


class NomicSpec(ModelSpec):
    def __init__(self, provider: InferenceProvider):
        super().__init__(provider, ThrottledScheduler(self))

    @override
    def model(self) -> str:
        return "nomic-embed-text"

    @override
    def context_window(self) -> int:
        return 8 * 1024

    @override
    def thinking_kwarg(self, enabled: bool) -> dict[str, Any]:
        raise NotImplementedError()

    @override
    def max_parallelism(self, environment: Environment) -> int:
        return 16

    @override
    def seconds_between_requests(self, environment: Environment) -> float:
        return 1


class KimiSpec(ModelSpec):
    def __init__(self, provider: InferenceProvider):
        super().__init__(provider, ThrottledScheduler(self))

    @override
    def model(self) -> str:
        return "kimi-k2-6"

    @override
    def context_window(self) -> int:
        return 256 * 1024

    @override
    def thinking_kwarg(self, enabled: bool) -> dict[str, Any]:
        return {"extra_body": {"chat_template_kwargs": {"thinking": enabled}}}

    @override
    def max_parallelism(self, environment: Environment) -> int:
        return 8

    @override
    def seconds_between_requests(self, environment: Environment) -> float:
        return 1


class GptSpec(ModelSpec):
    def __init__(self, provider: InferenceProvider):
        super().__init__(provider, ThrottledScheduler(self))

    @override
    def model(self) -> str:
        return "gpt-oss-120b"

    @override
    def context_window(self) -> int:
        return 128 * 1024

    @override
    def thinking_kwarg(self, enabled: bool) -> dict[str, Any]:
        return {"reasoning_effort": "medium" if enabled else "low"}

    @override
    def max_parallelism(self, environment: Environment) -> int:
        return 16

    @override
    def seconds_between_requests(self, environment: Environment) -> float:
        return 1


class IngestionPhase(Phase):
    streamed_pieces: list[str] = []

    def __init__(self, job: Job):
        super().__init__(job)
        IngestionPhase.streamed_pieces = []

    @override
    def name(self):
        return PhaseName.INGESTION

    @override
    def init_from_memory(self, input: PhaseResult) -> bool:
        self.entry: str = input.data["entry"]
        return True

    @override
    def init_from_disk(self) -> None:
        raise NotImplementedError()

    @override
    def halve_llm_call(self, call: LlmCall) -> list[LlmCall]:
        raise NotImplementedError()

    @override
    def sample_llm_call(self, call: LlmCall) -> LlmCall | None:
        content = call.messages[1]["content"]
        midpoint: int = len(content) // 2
        messages = [
            call.messages[0],
            {"role": call.messages[1]["role"], "content": content[:midpoint]},
        ]
        return self.new_call(messages, 0, call)

    @override
    def validate(self, payload: str | list[list[float]]) -> LlmStatus | None:
        if not payload:
            return LlmStatus.SUCCESS_EMPTY

        if not isinstance(payload, str):
            return LlmStatus.PARSE_ERROR

    @override
    def checkpoint(self) -> None:
        # No-op
        pass

    @override
    def run_main(self, execution_env: BoundExecutionEnv | None) -> PhaseResult:
        assert execution_env
        if not self.entry:
            return PhaseResult(data={"entry": ""})

        call = self.new_call(
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": self.entry},
            ],
            1000,
        )
        call.stream_handler = lambda piece: IngestionPhase.streamed_pieces.append(piece)
        print(f"Ingestion Prompt: {call.messages}")
        results = execution_env.run_all([call])[0]
        print(f"Received {len(results)} results")
        payloads = []
        for result in results:
            print(f"Response: {result}")
            if result.payload is not None:
                payloads.append(result.payload)
        return PhaseResult(data={"entry": "".join(payloads)})


class EmbeddingPhase(Phase):
    @override
    def name(self):
        return PhaseName.EMBEDDINGS

    @override
    def init_from_memory(self, input: PhaseResult) -> bool:
        self.entry: str = input.data["entry"]
        return True

    @override
    def init_from_disk(self) -> None:
        raise NotImplementedError()

    @override
    def halve_llm_call(self, call: LlmCall) -> list[LlmCall]:
        # Note: the prod function should leave contents intact, and split call.messages instead.
        content = call.messages[0]["content"]
        midpoint: int = len(content) // 2
        if not midpoint:
            return []
        return [
            self.new_call(
                [{"role": "user", "content": content[:midpoint]}],
                0,
                call,
            ),
            self.new_call(
                [{"role": "user", "content": content[midpoint:]}],
                0,
                call,
            ),
        ]

    @override
    def sample_llm_call(self, call: LlmCall) -> LlmCall | None:
        raise NotImplementedError()

    @override
    def validate(self, payload: str | list[list[float]]) -> LlmStatus | None:
        if not payload:
            return LlmStatus.SUCCESS_EMPTY

        if not isinstance(payload, list):
            return LlmStatus.PARSE_ERROR

    @override
    def checkpoint(self) -> None:
        # No-op
        pass

    @override
    def run_main(self, execution_env: BoundExecutionEnv | None) -> PhaseResult:
        assert execution_env
        if not self.entry:
            return PhaseResult(data={"entry": ""})
        call = self.new_call([{"role": "user", "content": self.entry}], 0)
        print(f"Embedding Prompt: {call.messages[0]['content']}")
        results = execution_env.run_all([call])[0]
        print(f"Received {len(results)} results")
        for result in results:
            print(f"Result: {result}")
            if result.payload is None:
                print("Payload is None")
            else:
                assert isinstance(result.payload, list)
                print(f"Size of payload: {len(result.payload)}")
        return PhaseResult(data={})


class PipelineJob(Job):
    @override
    def name(self):
        return JobName.FULL_PIPELINE

    @override
    def generate_phases(self) -> list[Phase]:
        return [IngestionPhase(self), EmbeddingPhase(self)]


class EmbeddingJob(Job):
    @override
    def name(self):
        return JobName.EMBEDDING_ONLY

    @override
    def generate_phases(self) -> list[Phase]:
        return [EmbeddingPhase(self)]


class IngestionJob(Job):
    @override
    def name(self):
        return JobName.FULL_PIPELINE

    @override
    def generate_phases(self) -> list[Phase]:
        return [IngestionPhase(self)]


class CacheManager(CachingHook):
    def __init__(self):
        self.cache: dict[str, tuple[LlmCall, LlmResponse]] = {}
        self.reset_counts()

    def reset_counts(self):
        self.cache_hits = 0
        self.cache_misses = 0

    @override
    def save(self, key: str, call: LlmCall, response: LlmResponse) -> None:
        self.cache[key] = [copy.copy(call), copy.copy(response)]
        print(f"[{datetime.now()}] Cache saved for {key}")

    @override
    def has(self, key: str) -> bool:
        return key in self.cache

    @override
    def load(self, key: str) -> tuple[LlmCall, LlmResponse] | None:
        if key in self.cache:
            print(f"[{datetime.now()}] Cache loaded for {key}")
            self.cache_hits += 1
            return self.cache[key]
        else:
            print(f"[{datetime.now()}] Cache not found for {key}")
            self.cache_misses += 1
            return None


class Logger(LlmHook):
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls: list[LlmCall] = []
        self.responses: list[
            tuple[LlmCall, BoundExecutionEnv, LlmResponse, RetryType, bool, bool]
        ] = []

    @override
    def hook_llm_queued(self, call: LlmCall, execution_env: BoundExecutionEnv) -> None:
        self.calls.append(call)
        print(f"[{datetime.now()}] Llm scheduled: {call}; env: {execution_env}")

    @override
    def hook_llm_started(self, call: LlmCall, execution_env: BoundExecutionEnv) -> None:
        print(f"[{datetime.now()}] Llm started: {call}; env: {execution_env}")

    @override
    def hook_llm_completed(
        self,
        call: LlmCall,
        execution_env: BoundExecutionEnv,
        response: LlmResponse,
        retry: RetryType,
        from_cache: bool,
        should_cache: bool,
    ) -> None:
        print(
            (
                f"[{datetime.now()}] Llm completed: {call}; env: {execution_env}; "
                f"status: {response.status}; exception: {response.exception} "
                f"should retry? {retry}; from cache? {from_cache}; should cache? {should_cache}"
            )
        )
        if response.exception:
            print(traceback.print_exception(response.exception))
        self.responses.append(
            (call, execution_env, response, retry, from_cache, should_cache)
        )


def setup_env(
    provider: InferenceProvider, caching_hook: CachingHook, llm_hook: LlmHook
) -> ExecutionEnv:
    gpt_spec = GptSpec(provider)
    kimi_spec = KimiSpec(provider)
    nomic_spec = NomicSpec(provider)

    execution_env = ExecutionEnv()
    execution_env.register_spec(PhaseName.INGESTION, kimi_spec, gpt_spec, True)
    execution_env.register_spec(PhaseName.EMBEDDINGS, nomic_spec, nomic_spec, True)
    execution_env.register_caching_hook(caching_hook)
    execution_env.register_llm_hook(llm_hook)

    return execution_env


@pytest.mark.integration
def test_combined_spec():
    tinfoil_provider = TinfoilProvider()
    tinfoil_provider.inject_errors(PhaseName.INGESTION, [LlmStatus.PARSE_ERROR] * 3)
    cache_manager = CacheManager()
    llm_logger = Logger()

    kimi_spec = KimiSpec(tinfoil_provider)
    gpt_spec = GptSpec(tinfoil_provider)
    combined_spec = CombinedSpec("kimi+gpt", [kimi_spec, gpt_spec])
    assert combined_spec.name() == "Tinfoil-TEE::kimi-k2-6 + Tinfoil-TEE::gpt-oss-120b"

    execution_env = ExecutionEnv()
    execution_env.register_spec(PhaseName.INGESTION, combined_spec, combined_spec, True)
    execution_env.register_caching_hook(cache_manager)
    execution_env.register_llm_hook(llm_logger)

    job = IngestionJob()
    job.run(PhaseResult({"entry": "Hi, how are you 1?"}), execution_env)
    job.run(PhaseResult({"entry": "Hi, how are you 2?"}), execution_env)
    job.run(PhaseResult({"entry": "Hi, how are you 3?"}), execution_env)
    job.run(PhaseResult({"entry": "Hi, how are you 3?"}), execution_env)
    job.run(PhaseResult({"entry": "Hi, how are you 3?"}), execution_env)

    assert len(llm_logger.responses) == 8
    assert llm_logger.responses[0][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[1][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[2][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[3][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[4][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[5][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[6][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[7][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[0][2].status == LlmStatus.PARSE_ERROR
    assert llm_logger.responses[1][2].status == LlmStatus.PARSE_ERROR
    assert llm_logger.responses[2][2].status == LlmStatus.PARSE_ERROR
    assert llm_logger.responses[3][2].status == LlmStatus.SUCCESS_MODEL_FALLBACK
    assert llm_logger.responses[4][2].status == LlmStatus.OK
    assert llm_logger.responses[5][2].status == LlmStatus.OK
    assert llm_logger.responses[6][2].status == LlmStatus.OK
    assert llm_logger.responses[7][2].status == LlmStatus.OK
    assert llm_logger.responses[0][3] == RetryType.FULL_RETRY
    assert llm_logger.responses[1][3] == RetryType.REASONING_OFF
    assert llm_logger.responses[2][3] == RetryType.MODEL_FALLBACK
    assert llm_logger.responses[3][3] == RetryType.NO_RETRY
    assert llm_logger.responses[4][3] == RetryType.NO_RETRY
    assert llm_logger.responses[5][3] == RetryType.NO_RETRY
    assert llm_logger.responses[6][3] == RetryType.NO_RETRY
    assert llm_logger.responses[7][3] == RetryType.NO_RETRY
    assert llm_logger.responses[0][1].model_spec.name() == kimi_spec.name()
    assert llm_logger.responses[1][1].model_spec.name() == kimi_spec.name()
    assert llm_logger.responses[2][1].model_spec.name() == kimi_spec.name()
    assert llm_logger.responses[3][1].model_spec.name() == gpt_spec.name()
    assert llm_logger.responses[4][1].model_spec.name() == kimi_spec.name()
    assert llm_logger.responses[5][1].model_spec.name() == gpt_spec.name()
    assert llm_logger.responses[6][1].model_spec.name() == gpt_spec.name()
    assert llm_logger.responses[7][1].model_spec.name() == gpt_spec.name()

    assert len(cache_manager.cache) == 3
    assert cache_manager.cache_hits == 2
    assert cache_manager.cache_misses == 6


@pytest.mark.integration
def test_full_pipeline():
    tinfoil_provider = TinfoilProvider()
    tinfoil_provider.inject_errors(PhaseName.INGESTION, [LlmStatus.LOAD])
    tinfoil_provider.inject_errors(PhaseName.INGESTION, [LlmStatus.CAP_HIT] * 4)
    tinfoil_provider.inject_errors(PhaseName.EMBEDDINGS, [LlmStatus.CAP_HIT] * 3)
    cache_manager = CacheManager()
    llm_logger = Logger()
    execution_env = setup_env(tinfoil_provider, cache_manager, llm_logger)

    # 3 errors on ingestion, 0 embedding calls since payload was empty
    job = PipelineJob()
    job.run(PhaseResult({"entry": "Hi, how are you?"}), execution_env)
    assert len(llm_logger.responses) == 4 + 0
    assert llm_logger.responses[0][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[1][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[2][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[3][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[0][2].status == LlmStatus.LOAD
    assert llm_logger.responses[1][2].status == LlmStatus.CAP_HIT
    assert llm_logger.responses[2][2].status == LlmStatus.CAP_HIT
    assert llm_logger.responses[3][2].status == LlmStatus.CAP_HIT
    assert llm_logger.responses[0][3] == RetryType.FULL_RETRY
    assert llm_logger.responses[1][3] == RetryType.REASONING_OFF
    assert llm_logger.responses[2][3] == RetryType.MODEL_FALLBACK
    assert llm_logger.responses[3][3] == RetryType.NO_RETRY

    assert len(cache_manager.cache) == 0
    assert cache_manager.cache_hits == 0
    assert cache_manager.cache_misses == 4

    assert IngestionPhase.streamed_pieces == []

    cache_manager.reset_counts()
    llm_logger.reset()

    # 1 error and 1 success on ingestion, 3 errors on embeddings which propagate into 4 additional successful requests.
    job.run(PhaseResult({"entry": "Hi, how are you?"}), execution_env)
    assert len(llm_logger.responses) == 2 + 7
    assert llm_logger.responses[0][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[1][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[2][1].phase.name() == PhaseName.EMBEDDINGS
    assert llm_logger.responses[3][1].phase.name() == PhaseName.EMBEDDINGS
    assert llm_logger.responses[4][1].phase.name() == PhaseName.EMBEDDINGS
    assert llm_logger.responses[5][1].phase.name() == PhaseName.EMBEDDINGS
    assert llm_logger.responses[6][1].phase.name() == PhaseName.EMBEDDINGS
    assert llm_logger.responses[7][1].phase.name() == PhaseName.EMBEDDINGS
    assert llm_logger.responses[8][1].phase.name() == PhaseName.EMBEDDINGS
    assert llm_logger.responses[0][2].status == LlmStatus.CAP_HIT
    assert llm_logger.responses[1][2].status == LlmStatus.SUCCESS_REASONING_OFF
    assert llm_logger.responses[2][2].status == LlmStatus.CAP_HIT
    assert llm_logger.responses[3][2].status == LlmStatus.CAP_HIT
    assert llm_logger.responses[4][2].status == LlmStatus.CAP_HIT
    assert llm_logger.responses[5][2].status == LlmStatus.OK
    assert llm_logger.responses[6][2].status == LlmStatus.OK
    assert llm_logger.responses[7][2].status == LlmStatus.OK
    assert llm_logger.responses[8][2].status == LlmStatus.OK
    assert llm_logger.responses[0][3] == RetryType.REASONING_OFF
    assert llm_logger.responses[1][3] == RetryType.NO_RETRY
    assert llm_logger.responses[2][3] == RetryType.HALVES
    assert llm_logger.responses[3][3] == RetryType.HALVES
    assert llm_logger.responses[4][3] == RetryType.HALVES
    assert llm_logger.responses[5][3] == RetryType.NO_RETRY
    assert llm_logger.responses[6][3] == RetryType.NO_RETRY
    assert llm_logger.responses[7][3] == RetryType.NO_RETRY
    assert llm_logger.responses[8][3] == RetryType.NO_RETRY

    assert len(cache_manager.cache) == 1 + 7
    assert cache_manager.cache_hits == 0
    assert cache_manager.cache_misses == 2 + 7

    assert len("".join(IngestionPhase.streamed_pieces)) > 0
    assert "".join(IngestionPhase.streamed_pieces) == llm_logger.responses[1][2].payload

    cache_manager.reset_counts()
    llm_logger.reset()

    # Cache hit when reasoning is again disabled.
    execution_env.register_spec(
        PhaseName.INGESTION,
        KimiSpec(tinfoil_provider),
        GptSpec(tinfoil_provider),
        False,
    )
    job.run(PhaseResult({"entry": "Hi, how are you?"}), execution_env)
    assert len(llm_logger.responses) == 1 + 7
    assert llm_logger.responses[0][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[1][1].phase.name() == PhaseName.EMBEDDINGS
    assert llm_logger.responses[2][1].phase.name() == PhaseName.EMBEDDINGS
    assert llm_logger.responses[3][1].phase.name() == PhaseName.EMBEDDINGS
    assert llm_logger.responses[4][1].phase.name() == PhaseName.EMBEDDINGS
    assert llm_logger.responses[5][1].phase.name() == PhaseName.EMBEDDINGS
    assert llm_logger.responses[6][1].phase.name() == PhaseName.EMBEDDINGS
    assert llm_logger.responses[7][1].phase.name() == PhaseName.EMBEDDINGS
    assert llm_logger.responses[0][2].status == LlmStatus.OK
    assert llm_logger.responses[1][2].status == LlmStatus.CAP_HIT
    assert llm_logger.responses[2][2].status == LlmStatus.CAP_HIT
    assert llm_logger.responses[3][2].status == LlmStatus.OK
    assert llm_logger.responses[4][2].status == LlmStatus.OK
    assert llm_logger.responses[5][2].status == LlmStatus.CAP_HIT
    assert llm_logger.responses[6][2].status == LlmStatus.OK
    assert llm_logger.responses[7][2].status == LlmStatus.OK
    assert llm_logger.responses[0][3] == RetryType.NO_RETRY
    assert llm_logger.responses[1][3] == RetryType.HALVES
    assert llm_logger.responses[2][3] == RetryType.HALVES
    assert llm_logger.responses[3][3] == RetryType.NO_RETRY
    assert llm_logger.responses[4][3] == RetryType.NO_RETRY
    assert llm_logger.responses[5][3] == RetryType.HALVES
    assert llm_logger.responses[6][3] == RetryType.NO_RETRY
    assert llm_logger.responses[7][3] == RetryType.NO_RETRY

    assert len(cache_manager.cache) == 1 + 7
    assert cache_manager.cache_hits == 1 + 7
    assert cache_manager.cache_misses == 0

    assert len("".join(IngestionPhase.streamed_pieces)) > 0
    assert "".join(IngestionPhase.streamed_pieces) == llm_logger.responses[0][2].payload

    cache_manager.reset_counts()
    llm_logger.reset()

    # Cache miss reasoning is re-enabled.
    execution_env.register_spec(
        PhaseName.INGESTION,
        KimiSpec(tinfoil_provider),
        GptSpec(tinfoil_provider),
        True,
    )
    job.run(PhaseResult({"entry": "Hi, how are you?"}), execution_env)
    assert len(llm_logger.responses) == 1 + 1
    assert llm_logger.responses[0][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[1][1].phase.name() == PhaseName.EMBEDDINGS
    assert llm_logger.responses[0][2].status == LlmStatus.OK
    assert llm_logger.responses[1][2].status == LlmStatus.OK
    assert llm_logger.responses[0][3] == RetryType.NO_RETRY
    assert llm_logger.responses[1][3] == RetryType.NO_RETRY

    assert len(cache_manager.cache) == 2 + 8
    assert cache_manager.cache_hits == 0
    assert cache_manager.cache_misses == 1 + 1

    assert len("".join(IngestionPhase.streamed_pieces)) > 0
    assert "".join(IngestionPhase.streamed_pieces) == llm_logger.responses[0][2].payload


@pytest.mark.integration
def test_abort():
    tinfoil_provider = TinfoilProvider()
    tinfoil_provider.inject_errors(PhaseName.EMBEDDINGS, [LlmStatus.CAP_HIT] * 5)
    llm_logger = Logger()
    execution_env = setup_env(tinfoil_provider, CacheManager(), llm_logger)

    job = EmbeddingJob()

    # Abort after the first request is made, but before the second one.
    def abort():
        time.sleep(0.5)
        job.cancellation_manager.abort()

    time.sleep(3)  # Sleep a bit to ensure client is warm up.
    threading.Thread(target=abort).start()

    job.run(PhaseResult({"entry": "Hi, how are you?"}), execution_env)

    assert len(llm_logger.responses) == 3
    # Cap hit results in 2 halves, which are both aborted.
    assert llm_logger.responses[0][2].status == LlmStatus.CAP_HIT
    assert llm_logger.responses[1][2].status == LlmStatus.ABORTED
    assert llm_logger.responses[2][2].status == LlmStatus.ABORTED


@pytest.mark.integration
def test_skip():
    tinfoil_provider = TinfoilProvider()
    tinfoil_provider.inject_errors(PhaseName.INGESTION, [LlmStatus.CAP_HIT] * 3)
    llm_logger = Logger()
    execution_env = setup_env(tinfoil_provider, CacheManager(), llm_logger)

    job = PipelineJob()

    # Skip after the first request is made, but before the second one.
    def abort():
        time.sleep(0.5)
        job.cancellation_manager.skip_call(llm_logger.calls[1].id)

    time.sleep(3)  # Sleep a bit to ensure client is warm up.
    threading.Thread(target=abort).start()

    job.run(PhaseResult({"entry": "Hi, how are you?"}), execution_env)

    assert len(llm_logger.responses) == 2
    assert llm_logger.responses[0][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[1][1].phase.name() == PhaseName.INGESTION
    assert llm_logger.responses[0][2].status == LlmStatus.CAP_HIT
    assert llm_logger.responses[1][2].status == LlmStatus.SKIPPED


@pytest.mark.integration
def test_attestations():
    tinfoil_provider = TinfoilProvider()
    tinfoil_provider.inject_errors(PhaseName.EMBEDDINGS, [LlmStatus.CAP_HIT] * 5)
    llm_logger = Logger()
    execution_env = setup_env(tinfoil_provider, CacheManager(), llm_logger)

    attestations = execution_env.attestations()
    assert len(attestations) == 5
    assert attestations[0].model == "router"
    assert attestations[1].model == "gpt-oss-120b"
    assert attestations[2].model == "gpt-oss-120b"
    assert attestations[3].model == "kimi-k2-6"
    assert attestations[4].model == "nomic-embed-text"
    assert attestations[0].enclave.endswith((".tinfoil.sh", ".tinfoil.dev"))
    assert attestations[1].enclave.endswith((".tinfoil.sh", ".tinfoil.dev"))
    assert attestations[2].enclave.endswith((".tinfoil.sh", ".tinfoil.dev"))
    assert attestations[3].enclave.endswith((".tinfoil.sh", ".tinfoil.dev"))
    assert attestations[4].enclave.endswith((".tinfoil.sh", ".tinfoil.dev"))
    assert attestations[0].repo == "tinfoilsh/confidential-model-router"
    assert attestations[1].repo == "tinfoilsh/confidential-gpt-oss-120b"
    assert attestations[2].repo == "tinfoilsh/confidential-gpt-oss-120b"
    assert attestations[3].repo == "tinfoilsh/confidential-kimi-k2-6-b200"
    assert attestations[4].repo == "tinfoilsh/confidential-nomic-embed-text"
    assert attestations[0].version != ""
    assert attestations[1].version != ""
    assert attestations[2].version != ""
    assert attestations[3].version != ""
    assert attestations[4].version != ""
    assert attestations[0].timestamp > time.time() - 60
    assert attestations[1].timestamp > time.time() - 60
    assert attestations[2].timestamp > time.time() - 60
    assert attestations[3].timestamp > time.time() - 60
    assert attestations[4].timestamp > time.time() - 60
    assert attestations[0].payload_compressed != ""
    assert attestations[1].payload_compressed != ""
    assert attestations[2].payload_compressed != ""
    assert attestations[3].payload_compressed != ""
    assert attestations[4].payload_compressed != ""
    assert attestations[0].payload_decompressed != b""
    assert attestations[1].payload_decompressed != b""
    assert attestations[2].payload_decompressed != b""
    assert attestations[3].payload_decompressed != b""
    assert attestations[4].payload_decompressed != b""
    assert attestations[0].attestation_type != None
    assert attestations[1].attestation_type != None
    assert attestations[2].attestation_type != None
    assert attestations[3].attestation_type != None
    assert attestations[4].attestation_type != None
    assert len(attestations[0].published_measurements) > 0
    assert len(attestations[1].published_measurements) > 0
    assert len(attestations[2].published_measurements) > 0
    assert len(attestations[3].published_measurements) > 0
    assert len(attestations[4].published_measurements) > 0
    assert attestations[0].error == None
    assert attestations[1].error == None
    assert attestations[2].error == None
    assert attestations[3].error == None
    assert attestations[4].error == None
