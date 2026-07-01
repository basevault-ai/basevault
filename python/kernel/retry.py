from typing import TYPE_CHECKING

from kernel.enums import LlmStatus, PhaseName, RetryType

if TYPE_CHECKING:
    from kernel.abstractions import LlmResponse, Phase


class TinfoilRouterUnavailable(Exception):
    """Raised when the Tinfoil router is unavailable."""


class RetryPolicy:
    count = 0

    @staticmethod
    def compute_status(
        phase: Phase, response: LlmResponse, max_tokens: int
    ) -> LlmStatus:
        # Status already set; no need to compute.
        if response.status:
            return response.status

        if response.exception:
            exception_type = type(response.exception)
            exception_name = f"{exception_type.__module__}.{exception_type.__name__}"

            # Load exceptions.
            if exception_name in {
                "openai.RateLimitError",  # 429
                "openai.APIConnectionError",  # Model enclave issues
                "openai.InternalServerError",  # 500 / 502: likely transient
                "openai.APIStatusError",  # generic, likely transient
                "builtins.ConnectionError",
                "builtins.ConnectionResetError",
                "builtins.ConnectionAbortedError",
                # httpx/httpcore mid-stream cuts: dropped by router or vLLM
                "httpx.RemoteProtocolError",
                "httpcore.RemoteProtocolError",
                "httpx.ReadError",
                "httpcore.ReadError",
                "httpx.ConnectError",
                "httpcore.ConnectError",
                "httpx.ConnectTimeout",
                "httpcore.ConnectTimeout",
            }:
                return LlmStatus.LOAD

            # Tinfoil constructor error
            if isinstance(response.exception, TinfoilRouterUnavailable):
                return LlmStatus.LOAD

            # Timeout exceptions.
            if exception_name in {
                "openai.APITimeoutError",
                "builtins.TimeoutError",
                "httpx.ReadTimeout",
                "httpcore.ReadTimeout",
            }:
                if response.duration is None:
                    return LlmStatus.OTHER  # Should never happen; logic error.
                elif response.ttft is None:
                    return LlmStatus.LOAD  # No token ever: likely a load issue
                elif response.ttft > 0.5 * response.duration:
                    # Most of the time was spent getting to first token: likely a load issue.
                    return LlmStatus.LOAD
                else:
                    # Most of the time was spent waiting for tokens; likely a sizing issue
                    return LlmStatus.TIMEOUT_WITH_TOKENS

            # Token limit exceptions, typically from BadRequestError 400 / ValueError.
            for token_limit_string in {
                "context_length_exceeded",
                "maximum context length",
                "reduce the length",
                "too many tokens",
            }:
                if token_limit_string in str(response.exception).lower():
                    return LlmStatus.CAP_HIT

            if max_tokens > 0 and response.completion_tokens >= max_tokens:
                # Successful request which completely exhausted the token budget.
                return LlmStatus.CAP_HIT

            return LlmStatus.OTHER

        if not response.payload:
            # Empty on wire: load issue.
            return LlmStatus.LOAD

        status: LlmStatus | None = phase.validate(response.payload)
        return status if status else LlmStatus.OK

    @staticmethod
    def relevant_retries(
        retry_corpus: list[RetryType],
        status_corpus: list[LlmStatus],
        to_count: set[LlmStatus],
    ) -> list[RetryType]:
        """Returns retries corresponding to the statuses in `to_count`."""
        assert len(retry_corpus) == len(status_corpus)
        result = []
        for i in range(len(status_corpus)):
            if status_corpus[i] in to_count:
                result.append(retry_corpus[i])
        return result

    @staticmethod
    def retry_policy(
        phase: PhaseName,
        was_thinking: bool,
        was_loaded_from_cache: bool,
        status: LlmStatus,
        previous_statuses: list[LlmStatus],
        previous_retries: list[RetryType],
    ) -> RetryType:
        sizing_errors = {
            LlmStatus.CAP_HIT,
            LlmStatus.PARSE_ERROR,
            LlmStatus.TIMEOUT_WITH_TOKENS,
        }
        other_statuses = {LlmStatus.OTHER, LlmStatus.SUCCESS_EMPTY}

        if status == LlmStatus.LOAD:
            relevant_retries = RetryPolicy.relevant_retries(
                previous_retries, previous_statuses, {LlmStatus.LOAD}
            )
            # Service is overloaded; we need to slow down
            # Max 5 retries, disable reasoning if first retry also fails
            if len(relevant_retries) >= 5:
                return RetryType.NO_RETRY
            elif relevant_retries and was_thinking:
                return RetryType.REASONING_OFF
            elif (
                len(relevant_retries) >= 3
                and RetryType.MODEL_FALLBACK not in relevant_retries
            ):
                return RetryType.MODEL_FALLBACK
            else:
                return RetryType.FULL_RETRY
        elif status in sizing_errors:
            relevant_retries = RetryPolicy.relevant_retries(
                previous_retries, previous_statuses, sizing_errors
            )
            # The request was too large; size of the work needs to be reduced
            if phase.is_non_degrading():
                # Split and send 2 parallel requests.
                # Up to 5 retries, max 2^5 calls
                if len(relevant_retries) < 5:
                    return RetryType.HALVES
                else:
                    return RetryType.NO_RETRY
            elif phase.is_degrading():
                # If error is parse_error or timeout with tokens, attempt 1 retry (count it as a 'second chance' retry).
                if not relevant_retries and status in {
                    LlmStatus.PARSE_ERROR,
                    LlmStatus.TIMEOUT_WITH_TOKENS,
                }:
                    return RetryType.FULL_RETRY

                # If reasoning on: try again once with reasoning disabled (skip for reasoning off)
                if was_thinking:
                    return RetryType.REASONING_OFF

                if RetryType.MODEL_FALLBACK not in relevant_retries:
                    return RetryType.MODEL_FALLBACK

                if phase == PhaseName.INGESTION:
                    # No further degradation can be done for Ingestion / Vision. Give up.
                    return RetryType.NO_RETRY
                else:
                    if relevant_retries.count(RetryType.SAMPLE) < 3:
                        return RetryType.SAMPLE
                    else:
                        return RetryType.NO_RETRY
            else:
                raise ValueError(f"Unreachable: {phase}")
        elif status in other_statuses:
            relevant_retries = RetryPolicy.relevant_retries(
                previous_retries, previous_statuses, other_statuses
            )
            if not relevant_retries:
                if status == LlmStatus.SUCCESS_EMPTY and was_loaded_from_cache:
                    return RetryType.NO_RETRY
                if was_thinking:
                    return RetryType.REASONING_OFF
                return RetryType.FULL_RETRY

            if len(relevant_retries) < 2:
                return RetryType.MODEL_FALLBACK
            else:
                return RetryType.NO_RETRY

        # NO_RETRY catch-all (e.g. ABORTED, SKIPPED).
        return RetryType.NO_RETRY

    @staticmethod
    def cache_policy(status: LlmStatus, retry_type: RetryType) -> bool:
        # Cache non-degrading work-reducing failures (e.g. timeouts, cap hit for extraction or entity summarizer).
        # This ensures the new pipeline will attempt to retrieve the halved requests immediately from the cache
        if retry_type == RetryType.HALVES:
            return True

        # Never cache non-leaves (e.g. a successful empty response that will be retried must not be cached).
        if retry_type != RetryType.NO_RETRY:
            return False

        # Statuses computed outside of retry policy, after caching.
        assert status not in {
            LlmStatus.SUCCESS_REASONING_OFF,
            LlmStatus.SUCCESS_SAMPLED,
            LlmStatus.SUCCESS_MODEL_FALLBACK,
        }

        # Cache all unambiguously successful responses.
        if status in {LlmStatus.OK, LlmStatus.SUCCESS_EMPTY}:
            return True

        # Don't cache any other failures (load failures, degrading work-reducing failures, etc.)
        return False
