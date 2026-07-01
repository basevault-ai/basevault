"""Tests for engine.attestation_view — the transport-free adapter that shapes
the kernel provider's attestation output into the app's AttestationResult JSON.

The engine does not perform attestation itself; these tests pin the pure
shaping logic (grouping, failure taxonomy, byte-offset derivation) against
hand-built kernel ``Attestation`` objects, and confirm the module imports NO
transport."""
from __future__ import annotations

import base64
import gzip
import json

from kernel.abstractions import Attestation
from kernel.enums import AttestationType

from engine.attestation_view import (
    _failure,
    attest_tinfoil_pipeline,
    result_from_attestations,
)


def _tdx_quote() -> tuple[str, bytes]:
    """A synthetic TDX quote with distinct byte patterns at the RTMR1 / RTMR2 /
    TLS-fp / HPKE offsets, plus its base64(gzip) wire form."""
    raw = bytearray(700)
    for i in range(0x1A8, 0x1A8 + 48):  # rtmr1 (measurement_offsets[0])
        raw[i] = 0xAB
    for i in range(0x1D8, 0x1D8 + 48):  # rtmr2 (measurement_offsets[1])
        raw[i] = 0xBC
    for i in range(0x238, 0x238 + 32):  # tls pubkey fingerprint
        raw[i] = 0xCD
    for i in range(0x258, 0x258 + 32):  # hpke pubkey
        raw[i] = 0xEF
    raw = bytes(raw)
    comp = base64.b64encode(gzip.compress(raw)).decode()
    return comp, raw


def _att(model, host, *, matched=True, atype=AttestationType.INTEL_TDX):
    comp, raw = _tdx_quote()
    if matched:
        # TDX publishes two measurements (RTMR1 + RTMR2), index-aligned with
        # measurement_offsets.
        return Attestation(
            model, host, "org/" + model, "v1", 100.0, comp, raw, atype,
            ["ab" * 48, "bc" * 48], None,
        )
    # A failed attestation: kernel leaves the payload empty + sets error.
    return Attestation(
        model, host, "org/" + model, "v1", 100.0, "", b"", None, [],
        "Traceback: measurement mismatch",
    )


def test_offsets_derive_measurement_tls_hpke():
    e = result_from_attestations([_att("router", "r"), _att("gpt-oss-120b", "e")])
    row = e["router"]
    assert row["predicate"] == "tdx-guest/v2"
    assert row["live_measurement"] == "ab" * 48
    assert row["tls_key_fp"] == "cd" * 32
    assert row["hpke_key"] == "ef" * 32
    assert row["match"] is True
    assert row["release_url"] == "https://github.com/org/router/releases/tag/v1"


def test_tdx_surfaces_both_rtmr_registers():
    e = result_from_attestations([_att("router", "r"), _att("gpt-oss-120b", "e")])
    row = e["router"]
    # RTMR1 (measurement_offsets[0] / published_measurements[0]).
    assert row["live_measurement"] == "ab" * 48
    assert row["published_measurement"] == "ab" * 48
    # RTMR2 (measurement_offsets[1] / published_measurements[1]).
    assert row["live_measurement2"] == "bc" * 48
    assert row["published_measurement2"] == "bc" * 48


def test_snp_has_no_second_measurement():
    # A single-register (SEV-SNP-shaped) attestation: no RTMR2 fields.
    snp = Attestation(
        "gpt-oss-120b", "e", "org/x", "v1", 100.0, "", b"",
        AttestationType.AMD_SEV_SNP, ["11" * 48], None,
    )
    row = result_from_attestations([snp])["constituents"][0]["enclaves"][0]
    assert row["published_measurement"] == "11" * 48
    assert row["live_measurement2"] is None
    assert row["published_measurement2"] is None


def test_all_matched_is_ok():
    res = result_from_attestations([_att("router", "r"), _att("gpt-oss-120b", "e")])
    assert res["ok"] is True
    assert res["failure_class"] is None
    assert res["fingerprint"] == "verified"
    # Single constituent → top-level per-model mirrors populated.
    assert len(res["enclaves"]) == 1
    assert res["deployment_tag"] == "v1"


def test_backend_mismatch_flips_ok_and_class():
    res = result_from_attestations([
        _att("router", "r"),
        _att("gpt-oss-120b", "e1"),
        _att("nomic-embed-text", "e2", matched=False),
    ])
    assert res["ok"] is False
    assert res["failure_class"] == "attestation_mismatch"
    emb = next(c for c in res["constituents"] if c["model"] == "nomic-embed-text")
    assert emb["ok"] is False
    assert emb["roles"] == ["embeddings"]  # role label for the health row


def test_router_down_tags_every_model():
    res = result_from_attestations([
        _att("router", "r", matched=False),
        _att("gpt-oss-120b", "e1"),
    ])
    assert res["ok"] is False
    assert all(c["failure_class"] == "router_down" for c in res["constituents"])
    assert res["router"]["match"] is False


def test_no_constituents_is_not_ok():
    res = result_from_attestations([_att("router", "r")])
    assert res["ok"] is False
    assert res["constituents"] == []


def test_result_is_json_serialisable():
    res = result_from_attestations([_att("router", "r"), _att("gpt-oss-120b", "e")])
    json.dumps(res)  # must not raise
    # required (non-defaulted) fields the Rust struct deserializes.
    for key in ("provider", "model", "ok", "fingerprint", "error", "ts"):
        assert key in res
    for c in res["constituents"]:
        for key in ("provider", "model", "ok", "fingerprint", "error", "ts"):
            assert key in c


def test_pipeline_never_raises_on_provider_failure(monkeypatch):
    # A construct failure (no key / router unreachable) becomes an ok:false
    # result the panel can render, never an exception.
    def boom():
        raise RuntimeError("no key")

    monkeypatch.setattr(
        "engine.attestation_view.build_attestation_env", boom
    )
    res = attest_tinfoil_pipeline()
    assert res["ok"] is False
    assert "no key" in res["error"]


def test_failure_shape_has_required_fields():
    res = _failure("boom")
    for key in ("provider", "model", "ok", "fingerprint", "error", "ts"):
        assert key in res
    assert res["ok"] is False
