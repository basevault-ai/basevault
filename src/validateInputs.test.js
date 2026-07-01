import { describe, it, expect } from "vitest";
import { validateInputs, MAX_FILE_SIZE } from "./validateInputs";

// validateInputs is the single helper called by all three input
// entry points (picker / folder / drop). The classifier order
// (system → size → excluded → unsupported → hidden) is the load-
// bearing invariant — these tests pin it.

describe("validateInputs — per-reason classification", () => {
  it("accepts whitelisted text/image/binary extensions", () => {
    const { accepted, rejected } = validateInputs([
      { path: "/u/notes.md", size: 100 },
      { path: "/u/photo.jpg", size: 100 },
      { path: "/u/scan.PDF", size: 100 }, // case-insensitive
      { path: "/u/data.json", size: 100 },
      { path: "/u/diary.markdown", size: 100 },
      { path: "/u/clip.zip", size: 100 },
    ]);
    expect(rejected).toEqual([]);
    expect(accepted).toHaveLength(6);
  });

  it("rejects unsupported extensions with the extension in the reason", () => {
    const { accepted, rejected } = validateInputs([
      { path: "/u/installer.dmg", size: 100 },
      { path: "/u/setup.pkg", size: 100 },
      { path: "/u/MyApp.app", size: 100 },
      { path: "/u/info.plist", size: 100 },
    ]);
    expect(accepted).toEqual([]);
    expect(rejected.map((r) => r.reason)).toEqual([
      "unsupported extension (.dmg)",
      "unsupported extension (.pkg)",
      "unsupported extension (.app)",
      "unsupported extension (.plist)",
    ]);
  });

  it("rejects files exceeding the 40 MB limit with size in the reason", () => {
    const big = MAX_FILE_SIZE + 1;
    const { accepted, rejected } = validateInputs([
      { path: "/u/huge.jpg", size: big },
      { path: "/u/giant.pdf", size: 90 * 1024 * 1024 },
    ]);
    expect(accepted).toEqual([]);
    expect(rejected[0].reason).toBe("too large (40.0 MB > 40 MB limit)");
    expect(rejected[1].reason).toBe("too large (90.0 MB > 40 MB limit)");
  });

  it("rejects exactly-named macOS system files regardless of extension", () => {
    const { accepted, rejected } = validateInputs([
      { path: "/u/.DS_Store", size: 100 },
      { path: "/u/sub/.DS_Store", size: 100 },
      { path: "/Volumes/X/.Spotlight-V100", size: 100 },
      { path: "/u/.Trashes", size: 100 },
      { path: "/u/.localized", size: 100 },
    ]);
    expect(accepted).toEqual([]);
    expect(rejected.map((r) => r.reason)).toEqual([
      "system file (.DS_Store)",
      "system file (.DS_Store)",
      "system file (.Spotlight-V100)",
      "system file (.Trashes)",
      "system file (.localized)",
    ]);
  });

  it("rejects HTML with a distinct 'excluded format' reason (not 'unsupported')", () => {
    const { rejected } = validateInputs([
      { path: "/u/page.html", size: 100 },
      { path: "/u/page.HTM", size: 100 },
    ]);
    expect(rejected.map((r) => r.reason)).toEqual([
      "excluded format (.html)",
      "excluded format (.htm)",
    ]);
  });

  it("rejects hidden files (basename starts with '.', extension not whitelisted)", () => {
    const { rejected } = validateInputs([
      { path: "/u/.gitignore", size: 100 },
      { path: "/u/.env", size: 100 },
    ]);
    expect(rejected.map((r) => r.reason)).toEqual([
      "hidden file",
      "hidden file",
    ]);
  });

  it("does NOT classify a whitelisted file as hidden just because a path segment starts with '.'", () => {
    // A `.txt` living under `/u/.cache/` is a valid input — only the
    // basename matters for the hidden-file check.
    const { accepted, rejected } = validateInputs([
      { path: "/u/.cache/foo.txt", size: 100 },
      { path: "/u/.config/notes.md", size: 100 },
    ]);
    expect(accepted).toEqual(["/u/.cache/foo.txt", "/u/.config/notes.md"]);
    expect(rejected).toEqual([]);
  });
});

describe("validateInputs — check ordering", () => {
  it("system file wins over the extension classifier", () => {
    // .DS_Store has no whitelisted extension, so without ordering it
    // would land as 'hidden file' or 'unsupported'. The system check
    // must run first.
    const { rejected } = validateInputs([
      { path: "/u/.DS_Store", size: 100 },
    ]);
    expect(rejected[0].reason).toBe("system file (.DS_Store)");
  });

  it("excluded format wins over generic 'unsupported' (.html → excluded)", () => {
    const { rejected } = validateInputs([
      { path: "/u/page.html", size: 100 },
    ]);
    expect(rejected[0].reason).toBe("excluded format (.html)");
  });
});

describe("validateInputs — batch shapes", () => {
  it("mixed batch: 3 valid + 2 rejected → accepted preserves order, rejected captures reasons", () => {
    const { accepted, rejected } = validateInputs([
      { path: "/u/a.md", size: 100 },
      { path: "/u/b.dmg", size: 100 },
      { path: "/u/c.jpg", size: 100 },
      { path: "/u/.DS_Store", size: 100 },
      { path: "/u/d.txt", size: 100 },
    ]);
    expect(accepted).toEqual(["/u/a.md", "/u/c.jpg", "/u/d.txt"]);
    expect(rejected).toEqual([
      { path: "/u/b.dmg", reason: "unsupported extension (.dmg)" },
      { path: "/u/.DS_Store", reason: "system file (.DS_Store)" },
    ]);
  });

  it("all-rejected batch", () => {
    const { accepted, rejected } = validateInputs([
      { path: "/u/.DS_Store", size: 100 },
      { path: "/u/setup.pkg", size: 100 },
    ]);
    expect(accepted).toEqual([]);
    expect(rejected).toHaveLength(2);
  });

  it("all-accepted batch", () => {
    const { accepted, rejected } = validateInputs([
      { path: "/u/a.md", size: 100 },
      { path: "/u/b.txt", size: 100 },
    ]);
    expect(rejected).toEqual([]);
    expect(accepted).toHaveLength(2);
  });

  it("empty input", () => {
    expect(validateInputs([])).toEqual({ accepted: [], rejected: [] });
    expect(validateInputs(null)).toEqual({ accepted: [], rejected: [] });
    expect(validateInputs(undefined)).toEqual({ accepted: [], rejected: [] });
  });

  it("missing size skips the size check (still classifies on extension/system rules)", () => {
    const { accepted, rejected } = validateInputs([
      { path: "/u/a.md" }, // size undefined
      { path: "/u/.DS_Store" },
      { path: "/u/b.dmg" },
    ]);
    expect(accepted).toEqual(["/u/a.md"]);
    expect(rejected.map((r) => r.reason)).toEqual([
      "system file (.DS_Store)",
      "unsupported extension (.dmg)",
    ]);
  });
});
