// Primitive: an expandable `<details>` block that renders a Python
// stack trace inside a `<pre>`. Filters out the empty / sentinel
// shapes `retry._exception_dict` already nullifies (None on the
// python side becomes null here; the "NoneType: None" sentinel is a
// belt-and-suspenders guard for older audit-log replays).
//
// `preClassName` lets callers (e.g. CallDetailPanel) plug in their
// own `<pre>` styling via CSS class; omitting it falls back to the
// attestation-surface inline styles below.
export function TracebackDetails({ trace, preClassName }) {
  if (!trace || trace === "NoneType: None") return null;
  const preStyle = preClassName
    ? undefined
    : {
        whiteSpace: "pre-wrap",
        fontSize: "0.85em",
        marginTop: 4,
        padding: 6,
        background: "#f5f5f5",
        border: "1px solid #ddd",
        borderRadius: 3,
        maxHeight: 240,
        overflow: "auto",
        color: "#333",
      };
  return (
    <details
      data-testid="error-trace"
      style={{ marginTop: 4, color: "inherit" }}
    >
      <summary style={{ cursor: "pointer", fontSize: "0.9em" }}>
        traceback
      </summary>
      <pre className={preClassName} style={preStyle}>
        {trace}
      </pre>
    </details>
  );
}

// One-line summary plus the trace expander. Used by the attestation
// surfaces (AttestationPanel, Wizard / Settings verify). Accepts
// either explicit `summary` + `trace`, or a single `text` that gets
// split on the first blank line — the Rust-side `verify_tinfoil_key`
// ships `<friendly>\n\n<traceback>` so the frontend can use that
// shape without callers pre-splitting.
export default function ErrorWithTrace({ summary, trace, text, className }) {
  let summaryText = summary;
  let traceText = trace ?? null;
  if (text != null) {
    const idx = text.indexOf("\n\n");
    if (idx >= 0) {
      summaryText = text.slice(0, idx);
      traceText = text.slice(idx + 2);
    } else {
      summaryText = text;
    }
  }
  if (!summaryText && !traceText) return null;
  return (
    <div className={className || "error"} style={{ marginTop: 6 }}>
      {summaryText && <div data-testid="error-summary">{summaryText}</div>}
      <TracebackDetails trace={traceText} />
    </div>
  );
}
