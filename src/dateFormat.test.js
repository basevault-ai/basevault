import { describe, it, expect } from "vitest";
import { prettyDate } from "./dateFormat";

describe("prettyDate", () => {
  it("renders a bare YYYY-MM-DD as the date GIVEN, with no timezone off-by-one", () => {
    // The UTC-midnight parse bug would render this as "Jun 17" in any
    // zone behind UTC. The local-midnight anchor keeps it on the 18th.
    expect(prettyDate("2011-06-18")).toBe("Jun 18, 2011");
  });

  it("keeps the year for a non-current year and drops it for the current year", () => {
    const thisYear = new Date().getFullYear();
    expect(prettyDate(`${thisYear}-04-15`)).toBe("Apr 15"); // year dropped
    expect(prettyDate("2019-05-21")).toBe("May 21, 2019"); // year kept
  });

  it("returns empty string for null / empty / unparseable input", () => {
    expect(prettyDate(null)).toBe("");
    expect(prettyDate("")).toBe("");
    expect(prettyDate(undefined)).toBe("");
    expect(prettyDate("not-a-date")).toBe("");
  });
});
