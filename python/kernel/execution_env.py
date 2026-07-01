import hashlib
import json
import threading
from concurrent.futures import Future

from kernel.abstractions import (
    Attestation,
    CachingHook,
    CombinedSpec,
    InferenceProvider,
    LlmCall,
    LlmHook,
    LlmResponse,
    ModelSpec,
    Phase,
)
from kernel.cancellation_manager import CancellationManager
from kernel.enums import LlmStatus, PhaseName, RetryType
from kernel.retry import RetryPolicy


class BoundExecutionEnv:
    def __init__(
        self,
        phase: Phase,
        model_spec: ModelSpec,
        fallback_model_spec: ModelSpec,
        thinking: bool,
        llm_hooks: list[LlmHook],
        caching_hook: CachingHook | None,
        cancellation_manager: CancellationManager,
    ):
        self.phase: Phase = phase
        self.model_spec: ModelSpec = model_spec
        self.fallback_model_spec: ModelSpec = fallback_model_spec
        self.thinking: bool = thinking
        self.llm_hooks: list[LlmHook] = llm_hooks
        self.caching_hook: CachingHook | None = caching_hook
        self.cancellation_manager = cancellation_manager

    def __str__(self):
        return (
            f"Phase: {self.phase.name()}; model: {self.model_spec.name()}; "
            f"fallback model: {self.fallback_model_spec.name()}; thinking: {self.thinking}"
        )

    def _cache_key(
        self, call: LlmCall, override_model_spec: ModelSpec | None = None
    ) -> str:
        model_spec = override_model_spec if override_model_spec else self.model_spec
        return (
            model_spec.name()
            + ("::thinking::" if self.thinking else "::")
            + f"max_tokens={call.max_tokens}::"
            + f"{hashlib.sha256(json.dumps(call.messages, sort_keys=True).encode()).hexdigest()}"
        )

    def run_all(self, calls: list[LlmCall]) -> list[list[LlmResponse]]:
        futures = [self.run(c) for c in calls]
        return [f.result() for f in futures]

    def run(self, call: LlmCall) -> Future[list[LlmResponse]]:
        future: Future[list[LlmResponse]] = Future()
        self._run_internal(call, [], [], future)
        return future

    def _run_internal(
        self,
        call: LlmCall,
        previous_statuses: list[LlmStatus],
        previous_retries: list[RetryType],
        future: Future[list[LlmResponse]],
    ) -> None:
        if isinstance(self.model_spec, CombinedSpec):
            chosen_spec: ModelSpec | None = None
            # Bind the model spec, preferring one with a cached entry if available.
            if self.caching_hook:
                for spec in self.model_spec.specs():
                    if self.caching_hook.has(self._cache_key(call, spec)):
                        chosen_spec = spec
                        break
            if not chosen_spec:
                chosen_spec = self.model_spec.next_spec()
            # Fully bind model spec.
            BoundExecutionEnv(
                self.phase,
                chosen_spec,
                self.fallback_model_spec,
                self.thinking,
                self.llm_hooks,
                self.caching_hook,
                self.cancellation_manager,
            )._run_internal(call, previous_statuses, previous_retries, future)
            return

        try:
            if self.caching_hook:
                entry: tuple[LlmCall, LlmResponse] | None = self.caching_hook.load(
                    self._cache_key(call)
                )

                if entry:
                    cached_call, response = entry
                    if cached_call:
                        assert cached_call.messages == call.messages
                        assert cached_call.max_tokens == call.max_tokens

                    assert response.status
                    if call.stream_handler and response.payload:
                        assert isinstance(response.payload, str)
                        call.stream_handler(response.payload)
                    self._handle_response(
                        call,
                        response,
                        True,
                        previous_statuses,
                        previous_retries,
                        future,
                    )
                    return

            def done_callback(f: Future[LlmResponse]):
                try:
                    self._handle_response(
                        call,
                        f.result(),
                        False,
                        previous_statuses,
                        previous_retries,
                        future,
                    )
                except Exception as e:
                    future.set_exception(e)

            # No cache
            self.model_spec.scheduler().run(
                call, self, bool(previous_retries)
            ).add_done_callback(done_callback)
        except Exception as e:
            future.set_exception(e)

    def _handle_response(
        self,
        call: LlmCall,
        response: LlmResponse,
        from_cache: bool,
        previous_statuses: list[LlmStatus],
        previous_retries: list[RetryType],
        future: Future[list[LlmResponse]],
    ) -> None:
        try:
            response.status = RetryPolicy.compute_status(
                self.phase,
                response,
                call.max_tokens,
            )

            retry = RetryPolicy.retry_policy(
                self.phase.name(),
                self.thinking,
                from_cache,
                response.status,
                previous_statuses,
                previous_retries,
            )
            if from_cache:
                should_cache = False
            else:
                should_cache = RetryPolicy.cache_policy(response.status, retry)
            if self.caching_hook and should_cache:
                self.caching_hook.save(self._cache_key(call), call, response)

            # Circumstantial results (e.g. SUCCESS_REASONING_OFF) are not cached.
            if response.status == LlmStatus.OK:
                last_retry = (
                    previous_retries[-1] if previous_retries else RetryType.NO_RETRY
                )
                if last_retry == RetryType.SAMPLE:
                    response.status = LlmStatus.SUCCESS_SAMPLED
                elif last_retry == RetryType.REASONING_OFF:
                    response.status = LlmStatus.SUCCESS_REASONING_OFF
                elif last_retry == RetryType.MODEL_FALLBACK:
                    response.status = LlmStatus.SUCCESS_MODEL_FALLBACK
            previous_statuses.append(response.status)
            previous_retries.append(retry)

            self.cancellation_manager.deregister_call(call)
            for hook in self.llm_hooks:
                hook.hook_llm_completed(
                    call, self, response, retry, from_cache, should_cache
                )

            if retry == RetryType.NO_RETRY:
                future.set_result([response])
            elif retry == RetryType.FULL_RETRY:
                new_call = self.phase.duplicate_call(call)
                self._run_internal(
                    new_call, previous_statuses, previous_retries, future
                )
            elif retry == RetryType.SAMPLE:
                new_call = self.phase.sample_llm_call(call)
                if new_call:
                    self._run_internal(
                        new_call, previous_statuses, previous_retries, future
                    )
                else:
                    # Sampling was not possible. No more retries.
                    future.set_result([response])
            elif retry == RetryType.HALVES:
                new_calls: list[LlmCall] = self.phase.halve_llm_call(call)
                new_responses: list[LlmResponse] = []
                new_responses_lock = threading.Lock()
                finished_calls: list[LlmCall] = []
                if new_calls:
                    for new_call in new_calls:
                        new_future: Future[list[LlmResponse]] = Future()

                        def maybe_combine_futures(f: Future[list[LlmResponse]]):
                            try:
                                with new_responses_lock:
                                    new_responses.extend(f.result())
                                    finished_calls.append(new_call)
                                    if len(finished_calls) == len(new_calls):
                                        future.set_result(new_responses)
                            except Exception as e:
                                future.set_exception(e)

                        new_future.add_done_callback(maybe_combine_futures)
                        self._run_internal(
                            new_call,
                            previous_statuses.copy(),
                            previous_retries.copy(),
                            new_future,
                        )
                else:
                    # Division was not possible. No longer retry.
                    future.set_result([response])
            elif retry == RetryType.REASONING_OFF:
                new_call = self.phase.duplicate_call(call)
                new_execution_env = BoundExecutionEnv(
                    self.phase,
                    self.model_spec,
                    self.fallback_model_spec,
                    False,
                    self.llm_hooks,
                    self.caching_hook,
                    self.cancellation_manager,
                )
                new_execution_env._run_internal(
                    new_call, previous_statuses, previous_retries, future
                )
            elif retry == RetryType.MODEL_FALLBACK:
                new_call = self.phase.duplicate_call(call)
                new_execution_env = BoundExecutionEnv(
                    self.phase,
                    self.fallback_model_spec,
                    self.fallback_model_spec,
                    self.thinking,
                    self.llm_hooks,
                    self.caching_hook,
                    self.cancellation_manager,
                )
                new_execution_env._run_internal(
                    new_call, previous_statuses, previous_retries, future
                )
            else:
                raise ValueError(f"Unreachable retry type {retry}")
        except Exception as e:
            future.set_exception(e)


class ExecutionEnv:
    def __init__(self):
        self._execution_specs: dict[PhaseName, tuple[ModelSpec, ModelSpec, bool]] = {}
        self._caching_hook: CachingHook | None = None
        self._llm_hooks: list[LlmHook] = []
        self._models_to_attest: dict[str, tuple[InferenceProvider, set[str]]] = {}

    def _add_model_to_attest(self, spec: ModelSpec) -> None:
        provider_name: str = spec.inference_provider().name()
        if provider_name not in self._models_to_attest:
            self._models_to_attest[provider_name] = (spec.inference_provider(), set())
        self._models_to_attest[provider_name][1].add(spec.model())

    def register_spec(
        self,
        phase: PhaseName,
        spec: ModelSpec,
        fallback_spec: ModelSpec,
        thinking: bool,
    ) -> None:
        self._execution_specs[phase] = (spec, fallback_spec, thinking)
        for s in [spec, fallback_spec]:
            if isinstance(s, CombinedSpec):
                [self._add_model_to_attest(x) for x in s.specs()]
            else:
                self._add_model_to_attest(s)

    # Not thread-safe. Make sure all hooks are registered before calls start.
    def register_caching_hook(self, hook: CachingHook) -> None:
        self._caching_hook = hook

    def register_llm_hook(self, hook: LlmHook) -> None:
        self._llm_hooks.append(hook)

    # Not thread-safe. Make sure all specs and hooks are registered before calls start.
    def bind(
        self, phase: Phase, cancellation_manager: CancellationManager
    ) -> BoundExecutionEnv:
        captured_spec = self._execution_specs.get(phase.name())
        if not captured_spec:
            raise RuntimeError(f"Could not find execution spec for phase {phase}")

        model_spec: ModelSpec = captured_spec[0]
        fallback_model_spec: ModelSpec = captured_spec[1]
        thinking: bool = captured_spec[2]
        return BoundExecutionEnv(
            phase,
            model_spec,
            fallback_model_spec,
            thinking,
            self._llm_hooks,
            self._caching_hook,
            cancellation_manager,
        )

    def attestations(self) -> list[Attestation]:
        attestations: list[Attestation] = []
        for provider_name, [provider, models] in self._models_to_attest.items():
            attestations += provider.attestations(models)
        return attestations
