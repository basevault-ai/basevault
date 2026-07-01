import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, cleanup, fireEvent } from "@testing-library/react";
import { afterEach } from "vitest";
import { openUrl } from "@tauri-apps/plugin-opener";
import AttestationPanel from "./AttestationPanel";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));

const ROUTER_HOST = "router.inf6.tinfoil.sh";
const ROUTER_LIVE_URL = `https://${ROUTER_HOST}/.well-known/tinfoil-attestation`;
const ROUTER_RELEASE_URL =
  "https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.95";
const KIMI_HOST = "kimi-k2-6-inf10.tinfoil.containers.tinfoil.dev";
const KIMI_LIVE_URL = `https://${KIMI_HOST}/.well-known/tinfoil-attestation`;
const KIMI_RELEASE_URL =
  "https://github.com/tinfoilsh/confidential-kimi-k2-6/releases/tag/v0.0.3";

function routerEnclave(overrides = {}) {
  return {
    host: ROUTER_HOST,
    predicate: "https://tinfoil.sh/predicate/sev-snp-guest/v2",
    live_measurement: "69430057351ffdc27d51fdd8f9e113cc23c2b2f485a1cd318f46585f981991a2fc621f958f5f9e13b6578f217c654709",
    published_measurement: "69430057351ffdc27d51fdd8f9e113cc23c2b2f485a1cd318f46585f981991a2fc621f958f5f9e13b6578f217c654709",
    release_repo: "tinfoilsh/confidential-model-router",
    release_tag: "v0.0.95",
    live_url: ROUTER_LIVE_URL,
    release_url: ROUTER_RELEASE_URL,
    match: true,
    error: null,
    ...overrides,
  };
}

function kimiEnclave(overrides = {}) {
  return {
    host: KIMI_HOST,
    predicate: "https://tinfoil.sh/predicate/tdx-guest/v2",
    live_measurement: "4f7be53273f4ed3114e7578574f98eec533d5a18484e4e8a5feef1672b4a94e17646e7ab9e1f3c722faea496bac4dc8d",
    published_measurement: "4f7be53273f4ed3114e7578574f98eec533d5a18484e4e8a5feef1672b4a94e17646e7ab9e1f3c722faea496bac4dc8d",
    tls_key_fp: "365a6689d6601fb962a7b42b4855cf29103ff43635c6bc62ffda916b0d933eb6",
    hpke_key: "f2a70aa4fce28feb44ef2fac8df53cd76094645c1d8daa1e9a986fc48342105b",
    raw_quote_b64gz: "H4sICG9z...truncated...",
    raw_quote_hex: "00".repeat(48 + 600),
    release_repo: "tinfoilsh/confidential-kimi-k2-6",
    release_tag: "v0.0.3",
    live_url: KIMI_LIVE_URL,
    release_url: KIMI_RELEASE_URL,
    match: true,
    error: null,
    ...overrides,
  };
}

function singleStatus(overrides = {}) {
  return {
    provider: "tinfoil",
    model: "kimi-k2-6",
    ok: true,
    fingerprint: "cafebabecafebabe",
    error: null,
    ts: 1700000000,
    constituents: [],
    router: routerEnclave(),
    deployment_tag: "v0.0.3",
    model_repo: "tinfoilsh/confidential-kimi-k2-6",
    enclaves: [kimiEnclave()],
    ...overrides,
  };
}

beforeEach(() => {
  openUrl.mockClear();
});

afterEach(() => cleanup());

describe("AttestationPanel — top-level state", () => {
  it("renders nothing when status null and not checking", () => {
    const { container } = render(
      <AttestationPanel status={null} checking={false} />
    );
    expect(container.firstChild).toBeNull();
  });

  it("Checking… while checking", () => {
    const { getByTestId } = render(
      <AttestationPanel status={null} checking={true} />
    );
    expect(getByTestId("attestation-checking")).toBeTruthy();
  });

  it("green line says 'Tinfoil attestation verified via SDK at <date + time>' (no model name)", () => {
    const { getByTestId } = render(
      <AttestationPanel status={singleStatus()} checking={false} />
    );
    const v = getByTestId("attestation-verified");
    expect(v.textContent).toMatch(/attestation verified via SDK at /);
    // Now includes the DATE, not just the time (ts 1700000000 → 2023).
    expect(v.textContent).toMatch(/2023/);
    expect(v.textContent).not.toMatch(/kimi/);
  });

  it("renders ✗ failed for a result with ok=false", () => {
    const { getByTestId } = render(
      <AttestationPanel
        status={singleStatus({ ok: false, error: "router: mismatch" })}
        checking={false}
      />
    );
    expect(getByTestId("attestation-failed")).toBeTruthy();
  });
});

describe("AttestationPanel — Recheck", () => {
  it("clicking Recheck invokes onRecheck", () => {
    const onRecheck = vi.fn();
    const { getByTestId } = render(
      <AttestationPanel
        status={singleStatus()} checking={false} onRecheck={onRecheck}
      />
    );
    fireEvent.click(getByTestId("attestation-recheck"));
    expect(onRecheck).toHaveBeenCalledTimes(1);
  });

  it("Recheck disabled while checking", () => {
    const { getByTestId } = render(
      <AttestationPanel status={null} checking={true} onRecheck={() => {}} />
    );
    expect(getByTestId("attestation-recheck").disabled).toBe(true);
  });
});

describe("AttestationPanel — chain rows", () => {
  it("renders router row + per-enclave row, both green when matching", () => {
    const { getByTestId } = render(
      <AttestationPanel status={singleStatus()} checking={false} />
    );
    const router = getByTestId(`row-${ROUTER_HOST}`);
    // Title carries platform suffix; measurement labels are platform-agnostic.
    expect(router.textContent).toMatch(/Router \(SEV SNP\)/);
    expect(router.textContent).toMatch(/Live Measurement/);
    expect(router.textContent).toMatch(/Published Measurement/);
    expect(router.textContent).toMatch(/✓ Measurements match/);
    expect(router.textContent).toMatch(/69430057351ffdc2/);
    const kimi = getByTestId(`row-${KIMI_HOST}`);
    expect(kimi.textContent).toMatch(/\(TDX\)/);
    expect(kimi.textContent).toMatch(/Live Measurement/);
    expect(kimi.textContent).toMatch(/Published Measurement/);
    expect(kimi.textContent).toMatch(/✓ Measurements match/);
    expect(kimi.textContent).toMatch(/4f7be53273f4ed31/);
  });

  it("router row links to live attestation endpoint and release tag URL", () => {
    const { getByTestId } = render(
      <AttestationPanel status={singleStatus()} checking={false} />
    );
    fireEvent.click(getByTestId(`row-${ROUTER_HOST}-live-url`));
    fireEvent.click(getByTestId(`row-${ROUTER_HOST}-release-url`));
    expect(openUrl).toHaveBeenCalledWith(ROUTER_LIVE_URL);
    expect(openUrl).toHaveBeenCalledWith(ROUTER_RELEASE_URL);
  });

  it("model row links to live attestation endpoint and release tag URL", () => {
    const { getByTestId } = render(
      <AttestationPanel status={singleStatus()} checking={false} />
    );
    fireEvent.click(getByTestId(`row-${KIMI_HOST}-live-url`));
    fireEvent.click(getByTestId(`row-${KIMI_HOST}-release-url`));
    expect(openUrl).toHaveBeenCalledWith(KIMI_LIVE_URL);
    expect(openUrl).toHaveBeenCalledWith(KIMI_RELEASE_URL);
  });

  it("does not render TLS pubkey fingerprint in the main row (it lives in (details))", () => {
    const { queryByTestId } = render(
      <AttestationPanel status={singleStatus()} checking={false} />
    );
    expect(queryByTestId(`row-${KIMI_HOST}-tls-fp`)).toBeNull();
  });

  it("(details) link opens a modal with the full quote walkthrough + script", () => {
    const status = singleStatus();
    const { getByTestId, queryByTestId } = render(
      <AttestationPanel status={status} checking={false} />
    );
    // Closed by default.
    expect(queryByTestId(`details-modal-${KIMI_HOST}`)).toBeNull();
    // Click (details) on the kimi row.
    fireEvent.click(getByTestId(`row-${KIMI_HOST}-details`));
    expect(getByTestId(`details-modal-${KIMI_HOST}`)).toBeTruthy();
    // Modal carries the raw b64gz body + hex with three highlights.
    expect(getByTestId("details-b64gz").textContent).toMatch(/H4sICG9z/);
    expect(getByTestId("details-highlight-measurement")).toBeTruthy();
    expect(getByTestId("details-highlight-tls_key_fp")).toBeTruthy();
    expect(getByTestId("details-highlight-hpke_key")).toBeTruthy();
    // Each field rendered separately.
    expect(getByTestId("details-field-measurement").textContent).toMatch(
      /4f7be53273f4ed31/
    );
    expect(getByTestId("details-field-tls-fp").textContent).toMatch(
      /365a6689d6601fb9/
    );
    expect(getByTestId("details-field-hpke").textContent).toMatch(
      /f2a70aa4fce28feb/
    );
    // Reproduction script: bash one-liner with python3 heredoc.
    const script = getByTestId("details-script").textContent;
    expect(script).toMatch(/^python3 <<'PY'/);
    expect(script).toMatch(/PY$/);
    expect(script).toMatch(/import json, base64, gzip, urllib\.request/);
    expect(script).toMatch(new RegExp(KIMI_HOST.replace(/\./g, "\\.")));
    expect(script).toMatch(/raw\[b\+376:b\+424\]/);
    expect(script).toMatch(/raw\[b\+520:b\+552\]/);
    expect(script).toMatch(/raw\[b\+552:b\+584\]/);
  });

  it("TDX shows both RTMR1 + RTMR2 rows, and the details modal highlights RTMR2 separately", () => {
    const status = singleStatus({
      enclaves: [kimiEnclave({
        live_measurement2: "bc".repeat(48),
        published_measurement2: "bc".repeat(48),
      })],
    });
    const { getByTestId } = render(
      <AttestationPanel status={status} checking={false} />
    );
    const row = getByTestId(`row-${KIMI_HOST}`);
    // Both registers labelled in the row.
    expect(row.textContent).toMatch(/Live Measurement \(RTMR1\)/);
    expect(row.textContent).toMatch(/Live Measurement \(RTMR2\)/);
    expect(row.textContent).toMatch(/Published Measurement \(RTMR2\)/);
    expect(getByTestId(`row-${KIMI_HOST}-live2`).textContent).toBe("bc".repeat(48));
    expect(getByTestId(`row-${KIMI_HOST}-published2`).textContent).toBe(
      "bc".repeat(48)
    );
    // Details modal: RTMR2 highlighted with its own slice + its own field, and
    // the reproduction script prints rtmr2.
    fireEvent.click(getByTestId(`row-${KIMI_HOST}-details`));
    expect(getByTestId("details-highlight-measurement2")).toBeTruthy();
    expect(getByTestId("details-field-measurement2").textContent).toMatch(
      /Measurement RTMR2/
    );
    expect(getByTestId("details-field-measurement2").textContent).toMatch(
      /bcbcbcbc/
    );
    expect(getByTestId("details-script").textContent).toMatch(
      /raw\[b\+424:b\+472\]/
    );
  });

  it("Copy script writes the bash one-liner to clipboard", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    const { getByTestId } = render(
      <AttestationPanel status={singleStatus()} checking={false} />
    );
    fireEvent.click(getByTestId(`row-${KIMI_HOST}-details`));
    fireEvent.click(getByTestId("details-copy-script"));
    expect(writeText).toHaveBeenCalledTimes(1);
    expect(writeText.mock.calls[0][0]).toMatch(/^python3 <<'PY'/);
    expect(writeText.mock.calls[0][0]).toMatch(/PY$/);
  });

  it("mismatch row shows ✗ + the error", () => {
    const status = singleStatus({
      ok: false,
      error: `${KIMI_HOST}: measurement mismatch`,
      enclaves: [kimiEnclave({
        match: false,
        live_measurement: "live_X",
        published_measurement: "published_Y",
        error: "measurement mismatch",
      })],
    });
    const { getByTestId } = render(
      <AttestationPanel status={status} checking={false} />
    );
    const row = getByTestId(`row-${KIMI_HOST}`);
    expect(row.textContent).toMatch(/✗/);
    expect(row.textContent).toMatch(/measurement mismatch/);
  });
});

function constituent(overrides = {}) {
  return {
    provider: "tinfoil",
    model: "kimi-k2-6",
    ok: true,
    transient: false,
    failure_class: null,
    roles: [],
    error: null,
    ts: 1700000000,
    enclaves: [kimiEnclave()],
    ...overrides,
  };
}

function multiStatus(constituents, overrides = {}) {
  return {
    provider: "tinfoil",
    model: constituents[0]?.model || "kimi-k2-6",
    ok: constituents.every((c) => c.ok),
    fingerprint: "cafebabecafebabe",
    error: (constituents.find((c) => !c.ok) || {}).error || null,
    failure_class: (constituents.find((c) => !c.ok) || {}).failure_class || null,
    ts: 1700000000,
    constituents,
    router: routerEnclave(),
    ...overrides,
  };
}

describe("AttestationPanel — enclave health view", () => {
  it("happy path: N/N models up, each row green", () => {
    const { getByTestId } = render(
      <AttestationPanel status={singleStatus()} checking={false} />
    );
    expect(getByTestId("enclave-health-summary").textContent).toMatch(
      /1\/1 models up/
    );
    expect(getByTestId("health-row-kimi-k2-6").textContent).toMatch(/✓/);
    expect(getByTestId("health-row-kimi-k2-6").textContent).toMatch(/up/);
  });

  it("names the down model + role + failure class for a narrow enclave outage", () => {
    // The motivating incident: embeddings (nomic-embed-text) drops to
    // zero enclaves while the chat model stays healthy.
    const status = multiStatus([
      constituent(),
      constituent({
        model: "nomic-embed-text",
        ok: false,
        failure_class: "enclave_down",
        roles: ["embeddings"],
        error:
          "embeddings (nomic-embed-text): no enclaves available — Tinfoil enclave down",
        enclaves: [],
      }),
    ]);
    const { getByTestId, queryByTestId } = render(
      <AttestationPanel status={status} checking={false} />
    );
    expect(getByTestId("enclave-health-summary").textContent).toMatch(
      /1\/2 models up/
    );
    const row = getByTestId("health-row-nomic-embed-text");
    expect(row.textContent).toMatch(/✗/);
    expect(row.textContent).toMatch(/embeddings/);
    expect(row.textContent).toMatch(/nomic-embed-text/);
    expect(row.textContent).toMatch(/enclave down/);
    // Zero-enclave model has no measurement chain row, but is visible in
    // the health view — the whole point of this surface.
    expect(queryByTestId("row-nomic-embed-text")).toBeNull();
  });

  it("router-down across all models reads as a broad outage", () => {
    const status = multiStatus([
      constituent({ ok: false, failure_class: "router_down", error: "router: x" }),
      constituent({
        model: "gpt-oss-120b",
        ok: false,
        failure_class: "router_down",
        error: "router: x",
      }),
    ]);
    const { getByTestId } = render(
      <AttestationPanel status={status} checking={false} />
    );
    expect(getByTestId("enclave-health-summary").textContent).toMatch(
      /Router unreachable — all models down/
    );
  });
});
