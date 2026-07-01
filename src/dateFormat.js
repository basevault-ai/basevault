// The ONE pretty-printed local date+time used everywhere a timestamp
// is shown to the user — runs, chats, attestation. Keeping a single
// formatter is the whole point: these surfaces must read identically.
//
// Rules (per directive):
//   • Month-day, hour:minute, local. No seconds.
//   • Drop the year when it's the CURRENT year (a this-year item
//     doesn't need it); keep it for any other year so an older
//     timestamp stays unambiguous.
//
// The current year is read per call via `new Date().getFullYear()`.
// That is NOT a system call — it's a few nanoseconds of plain JS, so
// even at thousands of calls per render it's free. Reading it per
// call (rather than caching at module load) keeps it correct across a
// New Year's-Eve rollover with zero practical cost; caching would
// trade that correctness for a saving that rounds to nothing.
export function prettyDateTime(input) {
  if (input === null || input === undefined || input === "") return "";
  const d = input instanceof Date ? input : new Date(input);
  if (Number.isNaN(d.getTime())) return "";
  const opts = {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  };
  if (d.getFullYear() !== new Date().getFullYear()) opts.year = "numeric";
  try {
    return new Intl.DateTimeFormat(undefined, opts).format(d);
  } catch {
    return "";
  }
}

// The date-only sibling of `prettyDateTime` — SAME module, SAME
// month-day + drop-the-year-when-current rule, just no hour:minute.
// Citation/resource references show the DATE of the cited
// record/source and nothing finer (per directive: date ONLY, no
// time): a reference's relevant granularity is "which day did this
// come from", not the minute the answer was generated. Kept here, not
// as a parallel formatter elsewhere, so the two forms can never drift
// (the dropdown's line 2 still uses `prettyDateTime` — date + time).
export function prettyDate(input) {
  if (input === null || input === undefined || input === "") return "";
  // A bare ISO date ("2011-06-18") parses as UTC midnight, so in any
  // timezone behind UTC it renders as the PREVIOUS calendar day (e.g.
  // "Jun 17"). Anchor a date-only string to LOCAL midnight instead so
  // the date shown is the date given. Full timestamps (with a time or
  // zone) keep their own parse — they carry their own instant.
  const d =
    input instanceof Date
      ? input
      : typeof input === "string" && /^\d{4}-\d{2}-\d{2}$/.test(input.trim())
        ? new Date(`${input.trim()}T00:00:00`)
        : new Date(input);
  if (Number.isNaN(d.getTime())) return "";
  const opts = { month: "short", day: "numeric" };
  if (d.getFullYear() !== new Date().getFullYear()) opts.year = "numeric";
  try {
    return new Intl.DateTimeFormat(undefined, opts).format(d);
  } catch {
    return "";
  }
}
