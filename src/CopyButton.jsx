import { useState } from "react";

// Shared copy-to-clipboard affordance. Lifted out of AttestationPanel
// so the chat transcript (per-message copy) and the attestation modal
// use the SAME button — one icon, one hover/done behavior, one
// clipboard call — instead of two drifting copies.

// Single clipboard call. Clipboard can fail under Gatekeeper / when the
// webview lacks focus; log and move on rather than throw (the source
// text stays selectable as the fallback).
export function copyToClipboard(text) {
  navigator.clipboard
    .writeText(text || "")
    .catch((err) => console.error("clipboard.writeText:", err));
}

// Copy-to-clipboard icon button. Two layouts:
//   - default: absolutely positioned in the top-right corner of a
//     ``position: relative`` parent (used inside CodeBlock).
//   - ``inline``: rendered inline next to its sibling text (used
//     after each measurement value in the attestation modal, and
//     after each chat message).
// Hover lightens the background; click writes ``onClick``'s text and
// briefly swaps the icon for a green check.
export function CopyButton({
  onClick,
  testId,
  inline = false,
  label = "Copy to clipboard",
  // No box: drop the border + opaque background, hover is a faint
  // tint only. Used in the chat transcript where a boxed button per
  // message is too heavy; AttestationPanel keeps the default box so
  // its copy buttons stay visible on the hex blocks.
  borderless = false,
}) {
  const [hover, setHover] = useState(false);
  const [done, setDone] = useState(false);
  const handleClick = () => {
    onClick();
    setDone(true);
    setTimeout(() => setDone(false), 1200);
  };
  const size = inline ? 22 : 36;
  const iconSize = inline ? 13 : 18;
  const positioning = inline
    ? { display: "inline-flex", verticalAlign: "middle", marginLeft: 4 }
    : { position: "absolute", top: 6, right: 6 };
  return (
    <button
      type="button"
      data-testid={testId}
      onClick={handleClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      title={done ? "Copied!" : label}
      style={{
        ...positioning,
        width: size,
        height: size,
        padding: 0,
        display: positioning.display || "flex",
        alignItems: "center",
        justifyContent: "center",
        background: borderless
          ? hover
            ? "rgba(0,0,0,0.06)"
            : "transparent"
          : hover
            ? "#fff"
            : "rgba(255,255,255,0.6)",
        border: borderless ? "none" : "1px solid #ccc",
        borderRadius: 4,
        cursor: "pointer",
        transition: "background 120ms ease",
      }}
      aria-label={label}
    >
      {done ? (
        <span style={{ color: "#2a7", fontSize: inline ? "0.9em" : "1.1em" }}>
          ✓
        </span>
      ) : (
        <svg width={iconSize} height={iconSize} viewBox="0 0 24 24" fill="none"
             stroke="#333" strokeWidth="2" strokeLinecap="round"
             strokeLinejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
      )}
    </button>
  );
}

// Tiny helper: renders a CopyButton if ``text`` is non-empty.
export function MaybeInlineCopy({ text, testId }) {
  if (!text) return null;
  return (
    <CopyButton
      inline
      onClick={() => copyToClipboard(text)}
      testId={testId}
    />
  );
}
