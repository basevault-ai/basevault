// Staging-time input validation: filter out files the pipeline would
// silently skip at run time so the user gets immediate feedback in a
// modal instead of buried run.log lines (issue #156).
//
// SYNC WITH engine/ingestor.py — the pipeline is the
// source-of-truth for what it will actually ingest. Keep these three
// constants in lockstep:
//   SUPPORTED_EXTS  ← _IMAGE_EXTS | _TEXT_EXTS | _BINARY_SUPPORTED
//   EXCLUDED_EXTS   ← _EXCLUDED_EXTS
//   MAX_FILE_SIZE   ← MAX_FILE_SIZE
// A single-source-of-truth via Tauri command is deliberately deferred
// (see issue #156 "Out of scope") — too much surface for a UX fix.

const IMAGE_EXTS = new Set([
  ".jpg", ".jpeg", ".png", ".heic", ".heif",
  ".bmp", ".tiff", ".webp", ".gif",
]);
const TEXT_EXTS = new Set([".txt", ".md", ".markdown", ".json", ".jsonl"]);
const BINARY_SUPPORTED = new Set([".pdf", ".docx", ".doc", ".zip"]);

export const SUPPORTED_EXTS = new Set([
  ...IMAGE_EXTS, ...TEXT_EXTS, ...BINARY_SUPPORTED,
]);

export const EXCLUDED_EXTS = new Set([".html", ".htm"]);

// Exact-name macOS detritus. Basename match (anywhere in the path).
export const SYSTEM_FILE_NAMES = new Set([
  ".DS_Store", ".Spotlight-V100", ".Trashes", ".fseventsd", ".localized",
]);

export const MAX_FILE_SIZE = 40 * 1024 * 1024; // 40 MB

function basename(p) {
  if (!p) return "";
  const segs = p.split("/");
  return segs[segs.length - 1] || p;
}

function extOf(name) {
  const i = name.lastIndexOf(".");
  if (i <= 0) return ""; // dot-prefixed or no dot
  return name.slice(i).toLowerCase();
}

// Classify a single (path, sizeBytes) pair. Returns `null` when
// accepted, or a reason string when rejected. Order:
//   system  (catches `.DS_Store` exactly, before extension rules)
//   size
//   excluded (.html distinct from generic "unsupported")
//   unsupported (not in whitelist)
//   hidden (basename-only — `/foo/.cache/bar.txt` accepts because the
//           basename is `bar.txt`, not `.cache`)
function classify(path, sizeBytes) {
  const name = basename(path);

  if (SYSTEM_FILE_NAMES.has(name)) {
    return `system file (${name})`;
  }

  if (typeof sizeBytes === "number" && sizeBytes > MAX_FILE_SIZE) {
    const mb = (sizeBytes / 1024 / 1024).toFixed(1);
    return `too large (${mb} MB > 40 MB limit)`;
  }

  const ext = extOf(name);

  if (ext && EXCLUDED_EXTS.has(ext)) {
    return `excluded format (${ext})`;
  }

  if (!ext || !SUPPORTED_EXTS.has(ext)) {
    // Hidden file — basename starts with a dot but the extension (if
    // any) isn't in the whitelist. A `.gitignore` lands here; a
    // `/foo/.cache/bar.txt` does NOT (basename `bar.txt`, ext `.txt`,
    // accepted earlier).
    if (name.startsWith(".")) {
      return `hidden file`;
    }
    if (ext) {
      return `unsupported extension (${ext})`;
    }
    return `unsupported extension (no extension)`;
  }

  return null;
}

// Validate a list of {path, size} pairs. `size` is bytes (from
// stat_paths); pass undefined to skip the size check (the path will
// still be classified by extension/system rules).
//
// Returns { accepted: string[], rejected: { path, reason }[] }.
// `accepted` preserves input order; `rejected` does too.
export function validateInputs(pathsWithSizes) {
  const accepted = [];
  const rejected = [];
  for (const item of pathsWithSizes || []) {
    if (!item || !item.path) continue;
    const reason = classify(item.path, item.size);
    if (reason === null) accepted.push(item.path);
    else rejected.push({ path: item.path, reason });
  }
  return { accepted, rejected };
}
