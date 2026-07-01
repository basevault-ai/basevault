// Perf canary for the cmd+F segment-locator hot path in MarkdownPreview
// (App.jsx). Self-contained: synthesises a ~4 MiB fixture in memory so
// the numbers reproduce on any checkout. Run with `node app/src/finder.bench.mjs`.
//
// Pre-fix (linear-scan locator, no cap) vs. post-fix (forward-cursor
// locator + MAX_SEARCH_MATCHES cap) at three segment counts spanning
// the realistic-to-pathological range an InputFileView's <pre> + fact-
// evidence <mark> spans can produce on a long input file.

import { performance } from "node:perf_hooks";

const TARGET_BYTES = 4 * 1024 * 1024;
const PARAGRAPH =
  "This day at home with my wife and her woman, busy about putting the house in order. " +
  "London is full of news of the fleet, and Pepys finds himself walking the streets at noon, " +
  "betimes to the office, and after dinner abroad with friends to talk of business and play. " +
  "A great many things to do, and a great many people to see, and the wife at home with the maid. ";
const FIXTURE = (() => {
  const reps = Math.ceil(TARGET_BYTES / PARAGRAPH.length);
  return PARAGRAPH.repeat(reps).slice(0, TARGET_BYTES);
})();

const MAX_SEARCH_MATCHES = 5000;

function findSegmentForward(segments, offset, fromIndex = 0) {
  let i = fromIndex < 0 ? 0 : fromIndex;
  while (i < segments.length && segments[i].end <= offset) i += 1;
  if (i >= segments.length) return -1;
  return i;
}

function buildSegments(text, segCount) {
  const step = Math.ceil(text.length / segCount);
  const segs = [];
  for (let i = 0; i < segCount; i++) {
    const start = i * step;
    const end = Math.min(start + step, text.length);
    if (end <= start) break;
    segs.push({ start, end });
  }
  return segs;
}

function searchOldLinear(text, segs, query) {
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(escaped, "gi");
  let n = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m[0].length === 0) { re.lastIndex += 1; continue; }
    let startSeg = -1;
    for (let i = 0; i < segs.length; i++) {
      if (m.index >= segs[i].start && m.index < segs[i].end) { startSeg = i; break; }
    }
    if (startSeg < 0) continue;
    n += 1;
  }
  return n;
}

function searchNewCursor(text, segs, query) {
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(escaped, "gi");
  let n = 0;
  let cursor = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m[0].length === 0) { re.lastIndex += 1; continue; }
    if (n >= MAX_SEARCH_MATCHES) break;
    const seg = findSegmentForward(segs, m.index, cursor);
    if (seg < 0) break;
    cursor = seg;
    n += 1;
  }
  return n;
}

function timeIt(label, fn) {
  fn(); fn();
  const samples = [];
  for (let i = 0; i < 5; i++) {
    const t0 = performance.now();
    const r = fn();
    samples.push({ ms: performance.now() - t0, count: r });
  }
  samples.sort((a, b) => a.ms - b.ms);
  const med = samples[2];
  console.log(`${label.padEnd(36)} median=${med.ms.toFixed(2).padStart(9)}ms  matches=${med.count}`);
}

console.log(`fixture: ${(FIXTURE.length / 1024 / 1024).toFixed(2)} MiB synthesised text`);
console.log("");

for (const segCount of [1000, 10000, 50000]) {
  const segs = buildSegments(FIXTURE, segCount);
  console.log(`--- segments=${segCount} ---`);
  for (const q of ["the", "a", "wife"]) {
    timeIt(`old segs=${segCount} '${q}'`, () => searchOldLinear(FIXTURE, segs, q));
    timeIt(`new segs=${segCount} '${q}'`, () => searchNewCursor(FIXTURE, segs, q));
  }
  console.log("");
}
