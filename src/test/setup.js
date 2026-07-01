// Vitest setup: mock Tauri's @tauri-apps/api/core::invoke globally so component
// tests that touch backend commands don't crash on import (the real module
// requires the Tauri runtime). Individual tests can stub return values via
// vi.mocked(invoke).mockResolvedValue(...) or vi.mocked(invoke).mockImplementation.
import { vi, afterEach } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async () => undefined),
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async () => () => {}),
  emit: vi.fn(async () => {}),
}));

vi.mock("@tauri-apps/api/webview", () => ({
  getCurrentWebview: vi.fn(() => ({
    onDragDropEvent: vi.fn(async () => () => {}),
  })),
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn(async () => null),
  save: vi.fn(async () => null),
  message: vi.fn(async () => {}),
  confirm: vi.fn(async () => false),
  ask: vi.fn(async () => false),
}));

vi.mock("@tauri-apps/plugin-opener", () => ({
  openUrl: vi.fn(async () => {}),
  openPath: vi.fn(async () => {}),
}));

// jsdom lacks Element.scrollIntoView / scrollTo (App.jsx's
// scroll-to-anchor effect in MarkdownPreview calls scrollTo on every
// selectedFile.anchor change, and a MutationObserver keeps it live
// while data fetches resolve). Tests that click into a stage view's
// anchor would otherwise spit unhandled errors after the test
// completes when the observer callback fires. Stub as no-ops.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function () {};
}
if (!Element.prototype.scrollTo) {
  Element.prototype.scrollTo = function () {};
}

// jsdom defines no global CSS object, so App.jsx's run-details modal
// (capturePin → CSS.escape on a call-id when building a querySelector)
// throws an uncaught TypeError under test, leaving the detail panel
// unrendered. Real WebKit/Chromium both implement CSS.escape; shim it
// with the spec serialization so the modal behaves as it does in app.
if (typeof globalThis.CSS === "undefined" || !globalThis.CSS) {
  globalThis.CSS = {};
}
if (typeof globalThis.CSS.escape !== "function") {
  // CSSOM § serialize-an-identifier. Sufficient for data-call-id values.
  globalThis.CSS.escape = function (value) {
    const s = String(value);
    let out = "";
    for (let i = 0; i < s.length; i++) {
      const c = s.charCodeAt(i);
      if (c === 0) {
        out += "�";
      } else if (
        (c >= 0x01 && c <= 0x1f) ||
        c === 0x7f ||
        (i === 0 && c >= 0x30 && c <= 0x39) ||
        (i === 1 && c >= 0x30 && c <= 0x39 && s.charCodeAt(0) === 0x2d)
      ) {
        out += "\\" + c.toString(16) + " ";
      } else if (
        c >= 0x80 ||
        c === 0x2d ||
        c === 0x5f ||
        (c >= 0x30 && c <= 0x39) ||
        (c >= 0x41 && c <= 0x5a) ||
        (c >= 0x61 && c <= 0x7a)
      ) {
        out += s.charAt(i);
      } else {
        out += "\\" + s.charAt(i);
      }
    }
    return out;
  };
}

afterEach(() => {
  vi.clearAllMocks();
});
