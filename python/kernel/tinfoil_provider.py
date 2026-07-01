"""
The only place that talks to the TEE Tinfoil API.
"""

import base64
import gzip
import json
import logging
import os
import threading
import time
import traceback
import urllib.request
from typing import override

from tinfoil import TinfoilAI

from kernel.abstractions import (
    Attestation,
    InferenceProvider,
    LlmCall,
    LlmResponse,
    LlmStatus,
)
from kernel.enums import AttestationType, PhaseName
from kernel.execution_env import BoundExecutionEnv
from kernel.retry import TinfoilRouterUnavailable

logger = logging.getLogger(__name__)


class TinfoilProvider(InferenceProvider):
    _tinfoil_client: TinfoilAI | None = None
    _tinfoil_client_lock: threading.Lock = threading.Lock()

    _PER_STREAM_RESPONSE_TIMEOUT: int = 3600

    def __init__(self):
        self._injected_errors: dict[PhaseName, list[LlmStatus]] = {
            phase: [] for phase in PhaseName
        }
        # Warm up client.
        threading.Thread(target=self._get_client).start()

    @staticmethod
    def _get_client() -> TinfoilAI:
        """
        Returns the singleton Tinfoil client. Rationale:
        1. Client initialization is not thread-safe; see: https://github.com/sigstore/sigstore-python/issues/1403
        2. Client performs full cryptographic enclave verification, which is slow (1-2 seconds as of June 2026).
        """
        with TinfoilProvider._tinfoil_client_lock:
            if TinfoilProvider._tinfoil_client is None:
                api_key: str = os.environ.get("TINFOIL_API_KEY", "")
                try:
                    logger.debug("Initializing Tinfoil client...")
                    TinfoilProvider._tinfoil_client = TinfoilAI(api_key=api_key)
                    logger.debug("Initialized Tinfoil client")
                except ValueError as e:
                    s = str(e)
                    if (
                        "No routers found in the response" in s
                        or "Failed to fetch router addresses" in s
                    ):
                        raise TinfoilRouterUnavailable(s) from e
                    raise
            return TinfoilProvider._tinfoil_client

    @override
    def name(self) -> str:
        return "Tinfoil-TEE"

    @override
    def run(self, call: LlmCall, execution_env: BoundExecutionEnv) -> LlmResponse:
        client: TinfoilAI = self._get_client()
        model = execution_env.model_spec.model()
        phase: PhaseName = execution_env.phase.name()
        assert phase.does_llm_call()
        if self._injected_errors[phase]:
            return LlmResponse.from_status(self._injected_errors[phase].pop(0), 0)

        status: LlmStatus | None = None
        exception: Exception | None = None
        payload: str | list[list[float]] | None = None
        prompt_tokens: int = 0
        completion_tokens: int = 0
        reasoning_tokens: int = 0
        ttft: float | None = None
        t0: float = time.monotonic()
        stream = None

        try:
            if model == "nomic-embed-text":
                payload = []
                input = []
                for message in call.messages:
                    assert message["role"] == "user"
                    assert isinstance(message["content"], str)
                    input.append(message["content"])
                response = client.embeddings.create(
                    input=input,
                    model=model,
                )
                if response.data:
                    payload = [x.embedding for x in response.data]
            else:
                pieces = []

                kwargs = execution_env.model_spec.thinking_kwarg(execution_env.thinking)
                if call.max_tokens > 0:
                    kwargs["max_completion_tokens"] = call.max_tokens

                # See documentation  at:
                # https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
                stream = client.chat.completions.create(
                    messages=call.messages,
                    model=model,
                    stream=True,
                    stream_options={"include_usage": True},
                    temperature=0,
                    timeout=TinfoilProvider._PER_STREAM_RESPONSE_TIMEOUT,
                    **kwargs,
                )
                execution_env.cancellation_manager.register_http_stream(call, stream)
                for chunk in stream:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        if choice.delta.content:
                            piece = choice.delta.content
                            pieces.append(piece)
                            if ttft is None:
                                ttft = time.monotonic() - t0
                            if call.stream_handler:
                                call.stream_handler(piece)
                        if choice.finish_reason == "length":
                            status = LlmStatus.CAP_HIT

                    # Final chunk includes usage.
                    if chunk.usage:
                        prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0)
                        completion_tokens = getattr(chunk.usage, "completion_tokens", 0)
                        details = getattr(
                            chunk.usage, "completion_tokens_details", None
                        )
                        if details:
                            reasoning_tokens = getattr(details, "reasoning_tokens", 0)
                payload = "".join(pieces)
        except Exception as e:
            exception = e
        if stream:
            execution_env.cancellation_manager.deregister_http_stream(call, stream)
            if execution_env.cancellation_manager.is_skipped(call):
                status = LlmStatus.SKIPPED
            elif execution_env.cancellation_manager.aborted():
                status = LlmStatus.ABORTED

        duration = time.monotonic() - t0
        return LlmResponse(
            status,  # Most likely None at this point; will be set later.
            payload,
            exception,
            prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            ttft,
            duration,
        )

    @override
    def inject_errors(self, phase: PhaseName, errors: list[LlmStatus]) -> None:
        self._injected_errors[phase] += errors

    @override
    def attestations(self, models: set[str]) -> list[Attestation]:
        """Returns attestation metadata; full attestation is performed by the Tinfoil client, not here."""
        client: TinfoilAI = self._get_client()
        secure_client = getattr(client, "_secure_client")

        router_enclave: str = secure_client.enclave
        router_repo: str = secure_client.repo

        tinfoil_proxy_payload: dict = {}
        with urllib.request.urlopen(
            f"https://{router_enclave}/.well-known/tinfoil-proxy", timeout=30
        ) as response:
            tinfoil_proxy_payload = json.loads(response.read().decode("utf-8"))

        router_version: str = tinfoil_proxy_payload.get("version", "")

        timestamp = time.time()
        attestations: list[Attestation] = [
            Attestation(
                "router",
                router_enclave,
                router_repo,
                router_version,
                timestamp,
                "",
                "",
                None,
                [],
                None,
            )
        ]
        for model, model_entry in tinfoil_proxy_payload.get("models", {}).items():
            if not model in models:
                continue
            repo: str = model_entry.get("repo", "")
            version: str = model_entry.get("tag", "")

            for enclave in model_entry.get("enclaves", {}).keys():
                attestations.append(
                    Attestation(
                        model,
                        enclave,
                        repo,
                        version,
                        timestamp,
                        "",
                        "",
                        None,
                        [],
                        None,
                    )
                )

        # Fetch attestation payloads in sequence. This is acceptable since the call is performed infrequently and asynchronously.
        for attestation in attestations:
            try:
                tinfoil_payload: dict = {}
                with urllib.request.urlopen(
                    f"https://{attestation.enclave}/.well-known/tinfoil-attestation",
                    timeout=30,
                ) as response:
                    tinfoil_payload = json.loads(response.read().decode("utf-8"))

                github_payload: dict = {}
                with urllib.request.urlopen(
                    f"https://github.com/{attestation.repo}/releases/download/{attestation.version}/tinfoil-deployment.json",
                    timeout=30,
                ) as github_response:
                    github_payload = json.loads(github_response.read().decode("utf-8"))

                attestation_format = tinfoil_payload.get("format", "")
                if "tdx-guest" in attestation_format:
                    attestation.attestation_type = AttestationType.INTEL_TDX

                    attestation.published_measurements = [
                        github_payload.get("tdx_measurement", {}).get("rtmr1", ""),
                        github_payload.get("tdx_measurement", {}).get("rtmr2", ""),
                    ]
                elif "sev-snp-guest" in attestation_format:
                    attestation.attestation_type = AttestationType.AMD_SEV_SNP
                    attestation.published_measurements = [
                        github_payload["snp_measurement"]
                    ]
                else:
                    raise ValueError(f"Unknown format: {attestation_format}")

                if not any(attestation.published_measurements):
                    raise ValueError(
                        "No published measurement available for attestation "
                        f"({attestation_format})"
                    )

                attestation.payload_compressed = tinfoil_payload.get("body", "")
                attestation.payload_decompressed = gzip.decompress(
                    base64.b64decode(attestation.payload_compressed)
                )

                for i, published_measurement in enumerate(
                    attestation.published_measurements
                ):
                    offset = attestation.attestation_type.measurement_offsets[i]
                    offset_end = offset + 48
                    live_measurement = attestation.payload_decompressed[
                        offset:offset_end
                    ].hex()
                    if published_measurement != live_measurement:
                        raise AssertionError(
                            f"Measurement mismatch: {published_measurement} != {live_measurement}"
                        )
            except Exception:
                attestation.error = traceback.format_exc()
                print(
                    f"Failed to load attestation for {attestation.model}: {attestation.error}"
                )
        return attestations
