"""Attestation *visibility* view — a thin, transport-free adapter over the
kernel's attested provider.

The engine does not perform attestation itself. The kernel owns the whole
trust path: the attested ``TinfoilProvider`` pins the enclave TLS key from the
verified quote and refuses to connect to a non-matching enclave (that is the
real, load-bearing per-connection guarantee), and its ``attestations()`` method
re-fetches the live enclave quote + published GitHub manifest and cross-checks
the measurements. This module only *reads* that result and shapes it into the
JSON the app's attestation panel renders. It imports NO transport (no
``tinfoil`` / ``urllib`` / sigstore) — the trust gate stays clean.

The panel is a non-blocking visibility surface, not a kill-switch: a failed
attestation is surfaced (an unlocked indicator + the failing enclave) but does
NOT disable cloud mode or chat. The per-connection guarantee that actually
protects a request lives in the kernel provider, which fails closed on its own
for every real inference call regardless of what this view reports.
"""
from __future__ import annotations

import time
from typing import Any

from kernel.abstractions import Attestation
from kernel.enums import AttestationType, PhaseName


# AttestationType → the ``predicate`` string the panel keys its platform label,
# byte-offset walkthrough, and reproduction script off of (it matches on the
# ``tdx-guest`` / ``sev-snp-guest`` substrings). Kept here, not in the kernel,
# because it is a UI-presentation concern.
_PREDICATE = {
    AttestationType.INTEL_TDX: "tdx-guest/v2",
    AttestationType.AMD_SEV_SNP: "sev-snp-guest/v2",
}

# Enclaves serving the embeddings model carry a role label so the health row
# reads as a pipeline stage rather than a bare model id (the motivating case in
# #899). Other models render under their id.
_EMBEDDING_MODEL = "nomic-embed-text"


def build_attestation_env():
    """A kernel ``ExecutionEnv`` registering the TEE spec of every LLM phase, so
    ``env.attestations()`` covers every backend the current config routes the
    pipeline + chat to. Model ids are de-duplicated inside the kernel provider,
    so one SDK init attests them all in a single batch."""
    from kernel.execution_env import ExecutionEnv

    from engine.llm import Mode
    from engine.phases.model_specs import chat_spec_for_mode, spec_for_stage

    env = ExecutionEnv()
    for phase in PhaseName:
        if not phase.does_llm_call():
            continue
        # CHAT routes through the separate ``chatbot`` config key, not the
        # per-stage map — resolve it the same way the chat turn does so the
        # panel attests the model chat actually uses.
        spec = (
            chat_spec_for_mode(Mode.TEE)
            if phase == PhaseName.CHAT
            else spec_for_stage(phase, Mode.TEE)
        )
        env.register_spec(phase, spec, spec, False)
    return env


def _hex_slice(raw: bytes, start: int, length: int) -> str | None:
    """``raw[start:start+length]`` as hex, or None if the quote is too short
    (a failed / empty attestation leaves ``payload_decompressed`` empty)."""
    end = start + length
    if start < 0 or len(raw) < end:
        return None
    return raw[start:end].hex()


def _enclave_dict(att: Attestation) -> dict[str, Any]:
    """One kernel ``Attestation`` → the panel's ``EnclaveAttestation`` shape.

    ``match`` is derived from ``error`` (the kernel sets ``error`` to a
    traceback on any fetch/decode/mismatch failure and leaves it None on a
    verified match). The live measurement / TLS-key fingerprint / HPKE key are
    re-derived from the raw quote bytes at the same offsets the kernel's own
    verifier uses, so the byte-level walkthrough the panel renders stays
    consistent with what was actually checked."""
    raw = att.payload_decompressed or b""
    atype = att.attestation_type
    predicate = _PREDICATE.get(atype) if atype else None

    # TDX exposes TWO runtime measurement registers (RTMR1 + RTMR2); SEV-SNP a
    # single one. ``measurement_offsets`` carries one offset per register and
    # ``published_measurements`` one published value per register, index-aligned
    # — surface both so the panel shows every measurement the kernel checked.
    live_measurement = None
    live_measurement2 = None
    tls_key_fp = None
    hpke_key = None
    if atype and raw:
        offsets = atype.measurement_offsets
        if len(offsets) >= 1:
            live_measurement = _hex_slice(raw, offsets[0], 48)
        if len(offsets) >= 2:
            live_measurement2 = _hex_slice(raw, offsets[1], 48)
        tls_key_fp = _hex_slice(raw, atype.tls_pubkey_fingerprint_offset, 32)
        hpke_key = _hex_slice(raw, atype.hpke_pubkey_offset, 32)

    pubs = att.published_measurements or []
    published = pubs[0] if len(pubs) >= 1 else None
    published2 = pubs[1] if len(pubs) >= 2 else None
    host = att.enclave or ""
    repo = att.repo or None
    tag = att.version or None
    return {
        "host": host,
        "predicate": predicate,
        "live_measurement": live_measurement,
        "live_measurement2": live_measurement2,
        "published_measurement": published,
        "published_measurement2": published2,
        "tls_key_fp": tls_key_fp,
        "hpke_key": hpke_key,
        "raw_quote_b64gz": att.payload_compressed or None,
        "raw_quote_hex": raw.hex() if raw else None,
        "release_repo": repo,
        "release_tag": tag,
        "live_url": (
            f"https://{host}/.well-known/tinfoil-attestation" if host else None
        ),
        "release_url": (
            f"https://github.com/{repo}/releases/tag/{tag}"
            if repo and tag
            else (f"https://github.com/{repo}/releases" if repo else None)
        ),
        "match": att.error is None,
        "error": att.error,
    }


def _roles_for(model: str) -> list[str]:
    return ["embeddings"] if model == _EMBEDDING_MODEL else []


def _constituent_dict(
    model: str, enclaves: list[dict], router_ok: bool, ts: float
) -> dict[str, Any]:
    """One model's per-enclave results → the panel's ``ConstituentAttestation``.

    The failure taxonomy names causes with distinct remedies so the health
    view can point each at its fix: a down router tags every model
    ``router_down``; a model with no enclaves is ``enclave_down``; a live ≠
    published measurement is an ``attestation_mismatch``."""
    matched = router_ok and bool(enclaves) and all(e["match"] for e in enclaves)
    failure_class = None
    error = None
    if not router_ok:
        failure_class = "router_down"
        error = "router attestation failed"
    elif not enclaves:
        failure_class = "enclave_down"
        error = f"{model}: no enclaves available"
    else:
        bad = next((e for e in enclaves if not e["match"]), None)
        if bad is not None:
            failure_class = "attestation_mismatch"
            error = f"{model}: {bad['host']}: {bad['error']}"
    return {
        "provider": "tinfoil",
        "model": model,
        "ok": matched,
        "fingerprint": "verified" if matched else None,
        "error": error,
        "ts": ts,
        "transient": False,
        "traceback": None,
        "doc_steps": None,
        "deployment_tag": enclaves[0]["release_tag"] if enclaves else None,
        "model_repo": enclaves[0]["release_repo"] if enclaves else None,
        "enclaves": enclaves,
        "roles": _roles_for(model),
        "failure_class": failure_class,
    }


def result_from_attestations(attestations: list[Attestation]) -> dict[str, Any]:
    """Shape a kernel ``list[Attestation]`` into the app's ``AttestationResult``
    JSON. The kernel emits one ``Attestation`` per (model, enclave) plus one
    with ``model == "router"`` for the verified router; this groups them into
    a router row + one constituent per backend model.

    Pure (no I/O) so it is unit-testable with hand-built ``Attestation``s."""
    ts = max((a.timestamp for a in attestations), default=time.time())

    router_att = next((a for a in attestations if a.model == "router"), None)
    router_dict = _enclave_dict(router_att) if router_att is not None else None
    router_ok = router_dict is not None and router_dict["match"]

    # Group backend enclaves by model, preserving first-seen order.
    by_model: dict[str, list[dict]] = {}
    for a in attestations:
        if a.model == "router":
            continue
        by_model.setdefault(a.model, []).append(_enclave_dict(a))

    constituents = [
        _constituent_dict(model, enclaves, router_ok, ts)
        for model, enclaves in by_model.items()
    ]

    # A down router already fails every constituent inside _constituent_dict
    # (each is built with the shared router_ok), so ``ok`` is the AND across
    # constituents with no special-casing here.
    ok = bool(constituents) and all(c["ok"] for c in constituents)
    first_failed = next((c for c in constituents if not c["ok"]), None)
    # Top-level fields the panel summary reads; per-constituent copies ride on
    # ``constituents`` for the health/chain views.
    single = constituents[0] if len(constituents) == 1 else {}
    return {
        "provider": "tinfoil",
        "model": constituents[0]["model"] if constituents else "",
        "ok": ok,
        "transient": False,
        "failure_class": (first_failed or {}).get("failure_class"),
        "fingerprint": "verified" if ok else None,
        "error": (first_failed or {}).get("error")
        or (None if router_ok else "router attestation failed"),
        "traceback": None,
        "doc_steps": None,
        "ts": ts,
        "constituents": constituents,
        "router": router_dict,
        "deployment_tag": single.get("deployment_tag"),
        "model_repo": single.get("model_repo"),
        "enclaves": single.get("enclaves") or [],
    }


def _failure(error: str) -> dict[str, Any]:
    """A verify result for the case where the kernel provider could not even
    produce attestations (no key, router unreachable at SDK construct)."""
    return {
        "provider": "tinfoil",
        "model": "",
        "ok": False,
        "transient": False,
        "failure_class": "router_down",
        "fingerprint": None,
        "error": error,
        "traceback": None,
        "doc_steps": None,
        "ts": time.time(),
        "constituents": [],
        "router": None,
        "deployment_tag": None,
        "model_repo": None,
        "enclaves": [],
    }


def attest_tinfoil_pipeline() -> dict[str, Any]:
    """Attest every TEE backend the current config routes to, via the kernel's
    attested provider, and return the app ``AttestationResult`` JSON. Never
    raises — a construct/fetch failure becomes an ``ok: false`` result the panel
    can render."""
    try:
        env = build_attestation_env()
        attestations = env.attestations()
    except Exception as e:  # noqa: BLE001 — surface any failure as a result
        return _failure(f"attestation unavailable: {e}")
    return result_from_attestations(attestations)
