import { useState } from "react";
import { openUrl } from "@tauri-apps/plugin-opener";
import { providerDisplayName, modelDisplayName } from "./teeProviders";
import { CopyButton, MaybeInlineCopy, copyToClipboard } from "./CopyButton";
import { prettyDateTime } from "./dateFormat";
import ErrorWithTrace from "./ErrorWithTrace";

// Tauri webviews block plain `<a target="_blank">` clicks for external
// URLs — route external clicks through the opener plugin.
function externalLinkClick(url) {
  return (e) => {
    e.preventDefault();
    openUrl(url).catch((err) => console.error("openUrl failed:", err));
  };
}

// Renders a verify_attestation result as one status line + a chain
// body listing the router and every backend the pipeline calls.
//
// Each row in the chain carries:
//   - title (model + platform)
//   - Live Measurement: <hex>
//   - URL: link to the enclave's /.well-known/tinfoil-attestation
//   - Published Measurement: <hex>
//   - URL: link to the GitHub release tag
//   - ✓ Measurements match (green) or ✗ Measurements don't match (red)
//
// During a Recheck the chain is kept in the DOM with visibility:hidden
// so the modal doesn't resize while verify_attestation is in flight.
export default function AttestationPanel({ status, checking, onRecheck }) {
  if (!checking && status === null) return null;
  const isChecking = checking || status === null;
  const ts =
    status?.ts && status.ts > 0
      ? prettyDateTime(status.ts * 1000)
      : null;
  const provLabel = status ? providerDisplayName(status.provider) : "";
  return (
    <div className="attestation-panel" style={{ marginTop: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <span style={{ fontWeight: 600 }}>Attestations</span>
        {isChecking && (
          <span style={{ color: "#666" }} data-testid="attestation-checking">
            ⏳ Checking…
          </span>
        )}
        {!isChecking && status?.ok && (
          <span style={{ color: "#2a7" }} data-testid="attestation-verified">
            ✓ {provLabel || "Tinfoil"} attestation verified via SDK
            {ts ? ` at ${ts}` : ""}
          </span>
        )}
        {!isChecking && status && !status.ok && status.transient && (
          <span style={{ color: "#666" }} data-testid="attestation-transient">
            ⏳ {provLabel || "Tinfoil"} attestation re-checking… (temporary
            hiccup)
          </span>
        )}
        {!isChecking && status && !status.ok && !status.transient && (
          <span style={{ color: "#c33" }} data-testid="attestation-failed">
            ✗ {provLabel || "Tinfoil"} attestation failed
          </span>
        )}
        {onRecheck && (
          <button
            type="button"
            data-testid="attestation-recheck"
            onClick={onRecheck}
            disabled={isChecking}
            style={{ marginLeft: "auto" }}
          >
            Recheck
          </button>
        )}
      </div>
      {status && (
        <div style={{ visibility: isChecking ? "hidden" : "visible" }}>
          <EnclaveHealth status={status} />
          <Chain status={status} />
        </div>
      )}
      {!isChecking && status && !status.ok && status.error && (
        status.transient ? (
          <p style={{ marginTop: 6, color: "#666", fontSize: "0.9em" }}>
            Temporary verification hiccup — retrying automatically. This
            is an infrastructure blip, not a failed enclave check.
          </p>
        ) : (
          <ErrorWithTrace
            summary={status.error}
            trace={status.traceback || null}
          />
        )
      )}
    </div>
  );
}

// Always-open chain. One row for the router enclave, one row per
// backend enclave. If any row's live ≠ published, the row goes red
// and inference is blocked upstream.
function Chain({ status }) {
  const constituents = pickConstituents(status);
  return (
    <div
      data-testid={`chain-${status.model}`}
      style={{
        marginTop: 6,
        padding: 10,
        border: "1px solid #ccc",
        borderRadius: 4,
        background: "#fafafa",
        maxHeight: "55vh",
        overflowY: "auto",
      }}
    >
      {status.router && <EnclaveRow title="Router" enclave={status.router} />}
      {constituents.flatMap((c) =>
        (Array.isArray(c.enclaves) && c.enclaves.length > 0
          ? c.enclaves
          : []).map((e) => (
          <EnclaveRow
            key={`${c.model}::${e.host}`}
            title={modelDisplayName(c.model) || c.model}
            enclave={e}
          />
        ))
      )}
    </div>
  );
}

// A single-model result lands in `constituents` (length 1) when the
// backend used the explicit-model path; multi-constituent (mix
// sentinel) returns multiple. Falls back to a synthesized
// constituent built from the top-level fields.
function pickConstituents(status) {
  if (Array.isArray(status.constituents) && status.constituents.length > 0) {
    return status.constituents;
  }
  return [{
    model: status.model,
    ok: status.ok,
    transient: status.transient,
    failure_class: status.failure_class,
    roles: status.roles || [],
    deployment_tag: status.deployment_tag,
    model_repo: status.model_repo,
    enclaves: status.enclaves || [],
  }];
}

// Failure-class → human label. Mirrors attestation.py's failure_class
// values; each names a cause with a distinct remedy.
const FAILURE_CLASS_LABELS = {
  enclave_down: "enclave down",
  attestation_mismatch: "measurement mismatch",
  router_down: "router down",
  auth: "invalid key",
};

// Compact per-model availability list, sourced from the same per-model
// attestation results the chain renders. Because a router-down tags
// every model at once, "one model down" vs "everything down" reads
// straight off this list — the narrow-vs-broad-outage signal a single
// aggregate banner can't give. Models with zero enclaves render here
// (the measurement chain below has nothing to show for them).
function EnclaveHealth({ status }) {
  const constituents = pickConstituents(status);
  if (constituents.length === 0) return null;
  const total = constituents.length;
  const up = constituents.filter((c) => c.ok).length;
  const allRouterDown =
    up === 0 && constituents.every((c) => c.failure_class === "router_down");
  return (
    <div data-testid="enclave-health" style={{ marginTop: 8 }}>
      <div style={{ fontWeight: 600, color: "#222" }}>Enclave health</div>
      <div
        data-testid="enclave-health-summary"
        style={{
          marginTop: 2,
          color: allRouterDown ? "#c33" : "#444",
          fontSize: "0.9em",
        }}
      >
        {allRouterDown
          ? "Router unreachable — all models down"
          : `${up}/${total} models up`}
      </div>
      <div style={{ marginTop: 4 }}>
        {constituents.map((c) => (
          <HealthRow key={c.model} c={c} />
        ))}
      </div>
    </div>
  );
}

// "extract/entities (gpt-oss-120b)" when the role is known, the model
// name alone when it isn't — mirrors attestation._model_label.
function healthModelLabel(c) {
  const name = modelDisplayName(c.model) || c.model;
  return Array.isArray(c.roles) && c.roles.length > 0
    ? `${c.roles.join("/")} (${name})`
    : name;
}

function HealthRow({ c }) {
  const transient = !c.ok && c.transient;
  const color = c.ok ? "#2a7" : transient ? "#666" : "#c33";
  const icon = c.ok ? "✓" : transient ? "⏳" : "✗";
  let suffix;
  if (c.ok) suffix = "up";
  else if (transient) suffix = "re-checking…";
  else suffix = FAILURE_CLASS_LABELS[c.failure_class] || "down";
  return (
    <div
      data-testid={`health-row-${c.model}`}
      style={{ color, fontSize: "0.9em", marginTop: 1 }}
    >
      {icon} {healthModelLabel(c)} — {suffix}
    </div>
  );
}

function EnclaveRow({ title, enclave }) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const e = enclave;
  const platform = platformLabel(e.predicate);
  const matched = e.match === true;
  const safeHost = e.host || "row";
  const fullTitle = platform ? `${title} (${platform})` : title;
  const hasDetails = !!(e.raw_quote_hex && e.live_measurement);
  // TDX carries a second runtime measurement register (RTMR2); SEV-SNP has a
  // single measurement. When present, label the pair RTMR1/RTMR2.
  const hasRtmr2 = !!(e.live_measurement2 || e.published_measurement2);
  return (
    <div
      data-testid={`row-${safeHost}`}
      style={{ padding: "8px 0", borderTop: "1px solid #eee" }}
    >
      <div style={{ fontWeight: 600, color: "#222" }}>{fullTitle}</div>
      {/* Grouped by source so RTMR1/RTMR2 render identically: both live
          registers sit above the live-quote URL, both published above the
          release URL. */}
      <MeasurementLine
        label={`Live Measurement${hasRtmr2 ? " (RTMR1)" : ""}`}
        value={e.live_measurement}
        testId={`row-${safeHost}-live`}
        copyTestId={`row-${safeHost}-live-copy`}
      >
        {hasDetails && (
          <>
            {" "}
            <a
              href="#"
              data-testid={`row-${safeHost}-details`}
              onClick={(ev) => { ev.preventDefault(); setDetailsOpen(true); }}
              style={{ color: "#06c" }}
            >
              (details)
            </a>
          </>
        )}
      </MeasurementLine>
      {hasRtmr2 && (
        <MeasurementLine
          label="Live Measurement (RTMR2)"
          value={e.live_measurement2}
          testId={`row-${safeHost}-live2`}
          copyTestId={`row-${safeHost}-live2-copy`}
        />
      )}
      <UrlLine url={e.live_url} testId={`row-${safeHost}-live-url`} />
      <MeasurementLine
        label={`Published Measurement${hasRtmr2 ? " (RTMR1)" : ""}`}
        value={e.published_measurement}
        testId={`row-${safeHost}-published`}
        copyTestId={`row-${safeHost}-published-copy`}
      />
      {hasRtmr2 && (
        <MeasurementLine
          label="Published Measurement (RTMR2)"
          value={e.published_measurement2}
          testId={`row-${safeHost}-published2`}
          copyTestId={`row-${safeHost}-published2-copy`}
        />
      )}
      <UrlLine url={e.release_url} testId={`row-${safeHost}-release-url`} />
      <div
        data-testid={`row-${safeHost}-match`}
        style={{
          marginTop: 4,
          color: matched ? "#2a7" : "#c33",
          fontWeight: 600,
        }}
      >
        {matched
          ? "✓ Measurements match"
          : `✗ Measurements don't match${e.error ? ` (${e.error})` : ""}`}
      </div>
      {detailsOpen && (
        <QuoteDetailsModal
          enclave={e}
          platform={platform}
          onClose={() => setDetailsOpen(false)}
        />
      )}
    </div>
  );
}

// One "<label>: <hex>" measurement line with an inline copy button. Optional
// children render after the value (e.g. the (details) link on the first row).
function MeasurementLine({ label, value, testId, copyTestId, children }) {
  return (
    <div style={{ marginTop: 2 }}>
      <span style={{ color: "#333" }}>{label}: </span>
      <span
        data-testid={testId}
        style={{
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          wordBreak: "break-all",
          color: "#111",
        }}
      >
        {value || "—"}
      </span>
      <MaybeInlineCopy text={value} testId={copyTestId} />
      {children}
    </div>
  );
}

function UrlLine({ url, testId }) {
  if (!url) return null;
  return (
    <div style={{ marginTop: 2 }}>
      <span style={{ color: "#333" }}>URL: </span>
      <a
        href={url}
        onClick={externalLinkClick(url)}
        target="_blank"
        rel="noopener noreferrer"
        data-testid={testId}
        style={{ wordBreak: "break-all" }}
      >
        {url}
      </a>
    </div>
  );
}

function platformLabel(predicate) {
  if (!predicate) return "";
  if (predicate.includes("tdx-guest")) return "TDX";
  if (predicate.includes("sev-snp-guest")) return "SEV SNP";
  return predicate;
}

function Hex({ label, value, testId, dim = false }) {
  return (
    <div data-testid={testId} style={{ marginTop: 2 }}>
      <span style={{ color: "#333" }}>{label}: </span>
      <span
        style={{
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          wordBreak: "break-all",
          color: dim ? "#444" : "#111",
        }}
      >
        {value || "—"}
      </span>
    </div>
  );
}

// ── Live-quote details modal (per-enclave (details) link) ──────────────
//
// Walks the user through how the live measurement was extracted from
// the raw hardware quote. Top: the compressed body that came over the
// wire. Middle: the decoded raw bytes as hex with three slices
// highlighted (measurement, tls pubkey hash, hpke pubkey). Below:
// each slice printed separately. Bottom: a copy-pasteable Python
// script that, when run on a Mac with Python 3 installed, prints the
// same three values.

const SLICE_COLORS = {
  measurement:  "#c33",  // red   — TDX RTMR1 / SEV-SNP measurement
  measurement2: "#93c",  // purple — TDX RTMR2 (second register)
  tls_key_fp:   "#06c",  // blue
  hpke_key:     "#0a7",  // green
};

// Byte-offset map per #135 §3b (TDX) / §4a (SEV-SNP). Returns
// {measurement, tls_key_fp, hpke_key}, each {start, end} in BYTES.
function quoteOffsets(predicate) {
  if (!predicate) return null;
  if (predicate.includes("tdx-guest")) {
    const b = 48;  // skip 48-byte TDX QuoteV4 header
    return {
      measurement:  { start: b + 376, end: b + 424 },  // RTMR1
      measurement2: { start: b + 424, end: b + 472 },  // RTMR2
      tls_key_fp:   { start: b + 520, end: b + 552 },
      hpke_key:     { start: b + 552, end: b + 584 },
    };
  }
  if (predicate.includes("sev-snp-guest")) {
    return {
      measurement: { start: 0x90, end: 0x90 + 48 },
      tls_key_fp:  { start: 0x50, end: 0x50 + 32 },
      hpke_key:    { start: 0x70, end: 0x70 + 32 },
    };
  }
  return null;
}

// Render the hex string with three byte ranges highlighted in
// distinct colors, wrapped in a CodeBlock so the user can copy the
// raw hex. Each hex char represents half a byte, so byte offset N →
// hex offset N*2.
function HighlightedHex({ hex, offsets, style, copyTestId }) {
  if (!hex) return null;
  let inner;
  if (!offsets) {
    inner = hex;
  } else {
    // Build a sorted list of (hex-index-start, hex-index-end, kind)
    // ranges, then walk left-to-right emitting plain + colored spans.
    const ranges = Object.entries(offsets)
      .map(([kind, { start, end }]) => ({
        start: start * 2, end: end * 2, kind,
      }))
      .sort((a, b) => a.start - b.start);
    const out = [];
    let cursor = 0;
    ranges.forEach((r, idx) => {
      if (cursor < r.start) {
        out.push(
          <span key={`p${idx}`}>{hex.slice(cursor, r.start)}</span>
        );
      }
      out.push(
        <span
          key={`h${idx}`}
          data-testid={`details-highlight-${r.kind}`}
          style={{
            background: SLICE_COLORS[r.kind] + "33",  // alpha
            color: SLICE_COLORS[r.kind],
            fontWeight: 600,
          }}
        >
          {hex.slice(r.start, r.end)}
        </span>
      );
      cursor = r.end;
    });
    if (cursor < hex.length) {
      out.push(<span key="tail">{hex.slice(cursor)}</span>);
    }
    inner = out;
  }
  return (
    <CodeBlock
      testId="details-hex"
      copyText={hex}
      copyTestId={copyTestId}
      style={style}
    >
      {inner}
    </CodeBlock>
  );
}

const hexBlockStyle = {
  margin: 0,
  padding: 8,
  paddingRight: 48,  // leave room for the absolute-positioned CopyButton
  background: "#f4f4f4",
  color: "#222",
  border: "1px solid #ddd",
  borderRadius: 3,
  fontSize: "0.8em",
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
  whiteSpace: "pre-wrap",
  wordBreak: "break-all",
  maxHeight: 240,
  overflow: "auto",
};

// Wraps a pre block in a relative-positioned container with a copy
// icon button in the top-right. ``copyText`` is what gets written to
// the clipboard when the button is clicked. ``children`` is what's
// rendered inside the pre (raw text OR highlighted spans).
function CodeBlock({ children, copyText, testId, copyTestId, style }) {
  const onCopy = () => copyToClipboard(copyText || "");
  return (
    <div style={{ position: "relative" }}>
      <pre data-testid={testId} style={{ ...hexBlockStyle, ...(style || {}) }}>
        {children}
      </pre>
      <CopyButton onClick={onCopy} testId={copyTestId} />
    </div>
  );
}

// Build the bash one-liner that reproduces the live extraction.
// Wraps a stdlib-only Python 3 program in a heredoc so pasting the
// whole block into Terminal runs it directly. Stock macOS Python 3
// is enough; no extra installs.
function reproductionScript({ host, predicate }) {
  const isTdx = predicate?.includes("tdx-guest");
  const isSnp = predicate?.includes("sev-snp-guest");
  const url = `https://${host}/.well-known/tinfoil-attestation`;
  let body;
  if (isTdx) {
    body = [
      "import json, base64, gzip, urllib.request",
      `doc = json.load(urllib.request.urlopen("${url}"))`,
      "raw = gzip.decompress(base64.b64decode(doc[\"body\"]))",
      "b = 48  # skip 48-byte TDX QuoteV4 header",
      "print(\"rtmr1     :\", raw[b+376:b+424].hex())",
      "print(\"rtmr2     :\", raw[b+424:b+472].hex())",
      "print(\"tls_key_fp:\", raw[b+520:b+552].hex())",
      "print(\"hpke_key  :\", raw[b+552:b+584].hex())",
    ].join("\n");
  } else if (isSnp) {
    body = [
      "import json, base64, gzip, urllib.request",
      `doc = json.load(urllib.request.urlopen("${url}"))`,
      "raw = gzip.decompress(base64.b64decode(doc[\"body\"]))",
      "print(\"measurement:\", raw[0x90:0x90+48].hex())",
      "print(\"tls_key_fp :\", raw[0x50:0x50+32].hex())",
      "print(\"hpke_key   :\", raw[0x70:0x70+32].hex())",
    ].join("\n");
  } else {
    return `# unsupported predicate: ${predicate}`;
  }
  return `python3 <<'PY'\n${body}\nPY`;
}

function QuoteDetailsModal({ enclave, platform, onClose }) {
  const e = enclave;
  const offsets = quoteOffsets(e.predicate);
  const script = reproductionScript({ host: e.host, predicate: e.predicate });
  const onCopy = () => copyToClipboard(script);
  const hasRtmr2 = !!e.live_measurement2;
  const measurementLabel = platform
    ? `Measurement${hasRtmr2 ? " RTMR1" : ""} (${platform})`
    : "Measurement";
  return (
    <div
      className="modal-backdrop"
      onClick={onClose}
      data-testid={`details-modal-${e.host}`}
      style={{ zIndex: 100 }}
    >
      <div
        className="modal"
        onClick={(ev) => ev.stopPropagation()}
        style={{ maxWidth: "min(960px, 90vw)" }}
      >
        <div className="modal-header">
          <h2>How the live measurement is extracted</h2>
          <button
            type="button"
            className="modal-close"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <div style={{ padding: 12, fontSize: "0.9em", overflow: "auto" }}>
          <div style={{ marginBottom: 4, color: "#333" }}>
            1. Compressed body fetched from <code>{e.live_url}</code>:
          </div>
          <CodeBlock
            testId="details-b64gz"
            copyText={e.raw_quote_b64gz || ""}
            copyTestId="details-copy-b64gz"
            style={{ maxHeight: 110 }}
          >
            {e.raw_quote_b64gz || "—"}
          </CodeBlock>
          <div style={{ marginTop: 8, color: "#333" }}>
            2. Decode with{" "}
            <code>gzip.decompress(base64.b64decode(body))</code>. Raw
            quote bytes (hex):
          </div>
          <HighlightedHex
            hex={e.raw_quote_hex}
            offsets={offsets}
            copyTestId="details-copy-hex"
            style={{ maxHeight: 130 }}
          />
          <div style={{ marginTop: 8, color: "#333" }}>
            3. Pull the canonical fields at the offsets defined in{" "}
            <SpecLinks platform={platform} />:
          </div>
          <FieldLine
            color={SLICE_COLORS.measurement}
            label={measurementLabel}
            value={e.live_measurement}
            testId="details-field-measurement"
          />
          {hasRtmr2 && (
            <FieldLine
              color={SLICE_COLORS.measurement2}
              label={`Measurement RTMR2 (${platform})`}
              value={e.live_measurement2}
              testId="details-field-measurement2"
            />
          )}
          <FieldLine
            color={SLICE_COLORS.tls_key_fp}
            label="TLS pubkey fingerprint"
            value={e.tls_key_fp}
            testId="details-field-tls-fp"
          />
          <FieldLine
            color={SLICE_COLORS.hpke_key}
            label="HPKE pubkey"
            value={e.hpke_key}
            testId="details-field-hpke"
          />
          <div style={{ marginTop: 12, color: "#333" }}>
            Paste this into a terminal — runs as-is on macOS (stock
            Python 3, no extra installs):
          </div>
          <CodeBlock
            testId="details-script"
            copyText={script}
            copyTestId="details-copy-script"
            style={{ maxHeight: 220 }}
          >
            {script}
          </CodeBlock>
        </div>
        <div className="modal-actions">
          <button type="button" className="btn-primary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

// SDK source as primary (a 200-line Python file with offsets defined
// inline beats Ctrl-F'ing a dense vendor PDF), vendor spec as
// footnote for security-audit readers who want the upstream-canonical
// authority.
const SPEC_LINKS = {
  TDX: {
    sdk: {
      url: "https://github.com/tinfoilsh/tinfoil-python/blob/main/src/tinfoil/attestation/abi_tdx.py",
      label: "TDX QuoteV4 layout",
    },
    vendor: {
      url: "https://download.01.org/intel-sgx/latest/dcap-latest/linux/docs/Intel_TDX_DCAP_Quoting_Library_API.pdf",
      label: "Intel TDX DCAP Quoting Library API",
    },
  },
  "SEV SNP": {
    sdk: {
      url: "https://github.com/tinfoilsh/tinfoil-python/blob/main/src/tinfoil/attestation/abi_sev.py",
      label: "SEV-SNP attestation report layout",
    },
    vendor: {
      url: "https://docs.amd.com/v/u/en-US/56860",
      label: "AMD SEV-SNP ABI Pub 56860",
    },
  },
};

function SpecLinks({ platform }) {
  const links = SPEC_LINKS[platform];
  if (!links) return <em>the platform attestation report layout</em>;
  return (
    <>
      <a
        href={links.sdk.url}
        onClick={externalLinkClick(links.sdk.url)}
        target="_blank"
        rel="noopener noreferrer"
      >
        {links.sdk.label}
      </a>
      {" (per "}
      <a
        href={links.vendor.url}
        onClick={externalLinkClick(links.vendor.url)}
        target="_blank"
        rel="noopener noreferrer"
      >
        {links.vendor.label}
      </a>
      {")"}
    </>
  );
}

function FieldLine({ color, label, value, testId }) {
  return (
    <div data-testid={testId} style={{ marginTop: 2, fontSize: "0.92em" }}>
      <span style={{ color, fontWeight: 600 }}>{label}: </span>
      <span
        style={{
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          wordBreak: "break-all",
          color,
        }}
      >
        {value || "—"}
      </span>
    </div>
  );
}

// CopyButton / MaybeInlineCopy / copyToClipboard now live in the
// shared ./CopyButton module (imported above) so the chat transcript
// and this panel share one implementation.
