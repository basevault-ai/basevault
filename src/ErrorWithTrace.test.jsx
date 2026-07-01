import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ErrorWithTrace from "./ErrorWithTrace";

describe("ErrorWithTrace", () => {
  it("renders summary alone when no trace is supplied", () => {
    render(<ErrorWithTrace summary="Invalid API key" trace={null} />);
    expect(screen.getByTestId("error-summary").textContent).toBe(
      "Invalid API key",
    );
    expect(screen.queryByTestId("error-trace")).toBeNull();
  });

  it("renders summary + expandable trace when trace is supplied", () => {
    const tb = [
      "Traceback (most recent call last):",
      "  File \"~/x.py\", line 1, in <module>",
      "TimeoutError: The read operation timed out",
    ].join("\n");
    render(
      <ErrorWithTrace
        summary="Tinfoil verification raised: The read operation timed out"
        trace={tb}
      />,
    );
    expect(screen.getByTestId("error-summary").textContent).toContain(
      "Tinfoil verification raised",
    );
    const det = screen.getByTestId("error-trace");
    expect(det).toBeTruthy();
    // <details> contents are always in the DOM (open or not).
    expect(det.textContent).toContain("TimeoutError");
    expect(det.textContent).toContain("~/x.py");
  });

  it("splits `text` shape on the first blank line into summary + trace", () => {
    const text = [
      "Couldn't reach Tinfoil — check your network.",
      "",
      "Traceback (most recent call last):",
      "  File \"x.py\", line 1",
      "ConnectionError: failed to resolve",
    ].join("\n");
    render(<ErrorWithTrace text={text} />);
    expect(screen.getByTestId("error-summary").textContent).toBe(
      "Couldn't reach Tinfoil — check your network.",
    );
    expect(screen.getByTestId("error-trace").textContent).toContain(
      "ConnectionError",
    );
  });

  it("falls back to text-only when there is no blank-line separator", () => {
    render(<ErrorWithTrace text="key is empty" />);
    expect(screen.getByTestId("error-summary").textContent).toBe(
      "key is empty",
    );
    expect(screen.queryByTestId("error-trace")).toBeNull();
  });
});
