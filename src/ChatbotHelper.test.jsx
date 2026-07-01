import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { render, cleanup, fireEvent, screen, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import ChatbotHelper, {
  renumberCitations,
  renderAnswerWithRefs,
  runRefDate,
  buildReplyCopyText,
  labelForRun,
  convoLastDate,
  msgTime,
} from "./ChatbotHelper";
import { prettyDate, prettyDateTime } from "./dateFormat";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn() }));

// Render the panel and capture the `chatbot-event` handler the component
// subscribes to. Tests drive the component by calling the captured
// handler with synthetic payloads — same pattern App.test.jsx uses for
// `pipeline-progress`.
async function renderAndGetHandler(props = {}) {
  let handler = null;
  vi.mocked(listen).mockImplementation(async (evt, fn) => {
    if (evt === "chatbot-event") handler = fn;
    return () => {};
  });
  vi.mocked(invoke).mockImplementation(convoMock());
  const utils = render(<ChatbotHelper {...props} />);
  fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
  await waitFor(() => expect(handler).toBeTruthy());
  return { handler, ...utils };
}

// Args of the most recent invoke() for a specific command. The chat
// now persists the pending user message immediately on send (req 1),
// so a `chatbot_save_transcript` fires right after the `chatbot` call —
// `toHaveBeenLastCalledWith("chatbot", …)` would see the save as the
// last invoke. These assertions only care about the last *chatbot*
// call's payload, so scope to that command.
function lastInvokeArgs(cmd) {
  const calls = vi.mocked(invoke).mock.calls.filter((c) => c[0] === cmd);
  return calls.length ? calls[calls.length - 1][1] : undefined;
}

// #565 turned the single transcript into one-dir-per-conversation. The
// wiring contract these tests pin is unchanged in spirit — load/save a
// transcript, rehydrate/resume on mount — so this shim adapts a legacy
// (transcript-based) invoke impl to the conversation commands: list →
// one conversation, load_conversation → the old chatbot_load_transcript
// payload, save_conversation → the old save, set_active → ok. Tests
// keep expressing intent in `chatbot_load_transcript` terms; the
// component drives the real #565 commands underneath (jsdom =
// wiring-guard only, never acceptance — that is the packaged .app).
const CONVO = {
  // #568: identity is the immutable ISO-Z prefix, not the dir name.
  id: "2026-05-02T13-30-11Z",
  created: "2026-05-02T13-30-11Z",
  label: "conversation-1",
  name: "2026-05-02T13-30-11Z-conversation-1",
};
function convoMock(impl = async () => undefined) {
  return async (cmd, args) => {
    if (cmd === "chatbot_list_conversations") return [CONVO];
    if (cmd === "chatbot_set_active_conversation") return undefined;
    if (cmd === "chatbot_load_conversation")
      return impl("chatbot_load_transcript", args);
    if (cmd === "chatbot_save_conversation")
      return impl("chatbot_save_transcript", args);
    return impl(cmd, args);
  };
}

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  cleanup();
});

describe("ChatbotHelper — async-listener cleanup race (doubled-output bug)", () => {
  it("tears down a subscription whose listen() resolves after unmount", async () => {
    let resolveListen;
    const unlistenSpy = vi.fn();
    vi.mocked(listen).mockImplementation(
      () =>
        new Promise((res) => {
          resolveListen = () => res(unlistenSpy);
        }),
    );
    const { unmount } = render(<ChatbotHelper />);
    unmount();
    await act(async () => {
      resolveListen();
    });
    expect(unlistenSpy).toHaveBeenCalledTimes(1);
  });

  it("does not double-append deltas under a single live subscription", async () => {
    let handler = null;
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      if (evt === "chatbot-event") handler = fn;
      return () => {};
    });
    vi.mocked(invoke).mockResolvedValue(undefined);
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    await waitFor(() => expect(handler).toBeTruthy());
    const user = userEvent.setup();
    await user.type(screen.getByTestId("chatbot-helper-input"), "q");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => { handler({ payload: { event: "chatbot_chunk", delta: "Hello" } }); });
    expect(screen.getByTestId("chatbot-helper-a").textContent).toBe("Hello▌");
  });
});

describe("ChatbotHelper — collapsed/expanded toggle", () => {
  it("renders only the pill before open", () => {
    vi.mocked(listen).mockResolvedValue(() => {});
    render(<ChatbotHelper />);
    expect(screen.getByTestId("chatbot-helper-pill")).toBeTruthy();
    expect(screen.queryByTestId("chatbot-helper-panel")).toBeNull();
  });

  it("clicking the pill expands the panel", () => {
    vi.mocked(listen).mockResolvedValue(() => {});
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    expect(screen.getByTestId("chatbot-helper-panel")).toBeTruthy();
  });

  it("clicking close collapses back to the pill", () => {
    vi.mocked(listen).mockResolvedValue(() => {});
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    fireEvent.click(screen.getByTestId("chatbot-helper-close"));
    expect(screen.queryByTestId("chatbot-helper-panel")).toBeNull();
    expect(screen.getByTestId("chatbot-helper-pill")).toBeTruthy();
  });

  it("the minimize control reads as collapse, not close (_ + label)", () => {
    vi.mocked(listen).mockResolvedValue(() => {});
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    const btn = screen.getByTestId("chatbot-helper-close");
    expect(btn.textContent).toBe("_");
    expect(btn.getAttribute("aria-label")).toBe("Collapse chat");
  });

  it("clicking the header bar (non-interactive area) collapses", () => {
    vi.mocked(listen).mockResolvedValue(() => {});
    const { container } = render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    // The title is a plain, non-interactive part of the black bar —
    // clicking it must collapse, same as the _ button.
    fireEvent.click(container.querySelector(".chatbot-helper-title"));
    expect(screen.queryByTestId("chatbot-helper-panel")).toBeNull();
    expect(screen.getByTestId("chatbot-helper-pill")).toBeTruthy();
  });

  it("a click on an interactive control in the bar does not collapse", () => {
    vi.mocked(listen).mockResolvedValue(() => {});
    const { container } = render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    // Drive the header's click handler directly with a synthetic
    // target inside an interactive element: the hit-target guard must
    // bail out so the bar-click does NOT collapse. (The _ button's own
    // handler still collapses on a real click — covered above; this
    // isolates the guard so a future in-bar control — #533 / director
    // additions — can't accidentally minimize the panel.)
    const header = container.querySelector(".chatbot-helper-header");
    const probe = document.createElement("button");
    header.appendChild(probe);
    fireEvent.click(probe);
    expect(screen.getByTestId("chatbot-helper-panel")).toBeTruthy();
  });
});

describe("ChatbotHelper — send flow", () => {
  it("send fires the chatbot invoke with the query and empty history", async () => {
    const user = userEvent.setup();
    await renderAndGetHandler();
    await user.type(screen.getByTestId("chatbot-helper-input"), "what did I do?");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    expect(invoke).toHaveBeenCalledWith("chatbot", {
      query: "what did I do?",
      history: [],
    });
  });

  it("Send is disabled until something non-blank is typed", async () => {
    const user = userEvent.setup();
    await renderAndGetHandler();
    const send = screen.getByTestId("chatbot-helper-send");
    expect(send.disabled).toBe(true);
    await user.type(screen.getByTestId("chatbot-helper-input"), "  ");
    expect(send.disabled).toBe(true);
    await user.type(screen.getByTestId("chatbot-helper-input"), "q");
    expect(send.disabled).toBe(false);
  });

  it("Send becomes a Stop control while streaming", async () => {
    const user = userEvent.setup();
    await renderAndGetHandler();
    await user.type(screen.getByTestId("chatbot-helper-input"), "q");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    // The send button is gone; a Stop control takes its place.
    expect(screen.queryByTestId("chatbot-helper-send")).toBeNull();
    expect(screen.getByTestId("chatbot-helper-stop")).toBeTruthy();
  });

  it("clicking Stop terminates generation (invokes chatbot_cancel), not just UI", async () => {
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    await user.type(screen.getByTestId("chatbot-helper-input"), "q");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => { handler({ payload: { event: "chatbot_chunk", delta: "partial" } }); });
    await user.click(screen.getByTestId("chatbot-helper-stop"));
    // The actual server-side cancellation must be requested.
    expect(invoke).toHaveBeenCalledWith("chatbot_cancel");
    // Partial answer is retained; streaming ends (Stop → Send back).
    expect(screen.getByTestId("chatbot-helper-a").textContent).toContain("partial");
    expect(screen.getByTestId("chatbot-helper-send")).toBeTruthy();
  });

  it("chatbot_stopped from the sidecar leaves streaming with the partial answer", async () => {
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    await user.type(screen.getByTestId("chatbot-helper-input"), "q");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => { handler({ payload: { event: "chatbot_chunk", delta: "half an answer" } }); });
    act(() => { handler({ payload: { event: "chatbot_stopped" } }); });
    expect(screen.getByTestId("chatbot-helper-a").textContent).toContain("half an answer");
    expect(screen.getByTestId("chatbot-helper-send")).toBeTruthy();
  });

  it("renames the surface to Chat About Your Data", async () => {
    vi.mocked(listen).mockResolvedValue(() => {});
    render(<ChatbotHelper />);
    expect(screen.getByTestId("chatbot-helper-pill").textContent)
      .toBe("Chat About Your Data");
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    expect(screen.getByText("Chat About Your Data")).toBeTruthy();
  });

  it("chatbot_bound shows the bound corpus run as grey subtext under the title", async () => {
    const { handler } = await renderAndGetHandler();
    // No binding announced yet → no subtext (nothing to mislead about).
    expect(screen.queryByTestId("chatbot-helper-boundrun")).toBeNull();
    act(() => {
      handler({ payload: { event: "chatbot_bound", run: "personal-os" } });
    });
    const sub = screen.getByTestId("chatbot-helper-boundrun");
    expect(sub.textContent).toContain("personal-os");
    // A null run (no processed corpus) clears it rather than showing a
    // stale/blank binding.
    act(() => {
      handler({ payload: { event: "chatbot_bound", run: null } });
    });
    expect(screen.queryByTestId("chatbot-helper-boundrun")).toBeNull();
  });

  it("renders the run selector with human labels and rebinds on pick (#507)", async () => {
    let handler = null;
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      if (evt === "chatbot-event") handler = fn;
      return () => {};
    });
    // The backend label is the corpus SUBJECT ONLY — the time is never
    // baked into it server-side (that string would be UTC, ahead of the
    // run view for any non-UTC user, #984). The time rides on
    // `created_at` and is rendered client-side in local tz.
    const rows = [
      {
        run_id: "2026-05-16T03-14-54Z-xttq",
        label: "personal-os.txt",
        created_at: "2026-05-16T03:14:54Z",
        store_path: "/s/x",
        bound: true,
      },
      {
        run_id: "2026-05-16T01-14-42Z-f66s",
        label: "barbellion-disappointed-man.txt",
        created_at: "2026-05-16T01:14:42Z",
        store_path: "/s/f",
        bound: false,
      },
    ];
    vi.mocked(invoke).mockImplementation(async (cmd) =>
      cmd === "chatbot_list_runs" ? rows : undefined,
    );
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    await waitFor(() => expect(handler).toBeTruthy());

    // The shared dropdown's collapsed toggle shows the backend-marked
    // bound run by its human label; the opaque slug never surfaces.
    const toggle = await screen.findByTestId("chatbot-helper-runtoggle");
    expect(toggle.textContent).toContain("personal-os.txt");
    expect(toggle.textContent).not.toContain("xttq");
    // The time is the LOCAL-tz render of `created_at` (test tz is NYC,
    // UTC−4: 03:14 UTC → 11:14 PM the PREVIOUS day), NOT the UTC
    // wall-clock the old server-side label embedded (#984).
    expect(toggle.textContent).toContain(prettyDateTime("2026-05-16T03:14:54Z"));
    expect(toggle.textContent).toContain("May 15, 11:14 PM");
    expect(toggle.textContent).not.toContain("May 16, 03:14");
    // Opening the menu lists every run by its human label.
    fireEvent.click(toggle);
    const menu = await screen.findByTestId("chatbot-helper-runmenu");
    expect(menu.textContent).toContain("personal-os.txt");
    expect(menu.textContent).toContain("barbellion-disappointed-man.txt");
    // Each row's time is the local-tz render of its own `created_at`.
    expect(menu.textContent).toContain(prettyDateTime("2026-05-16T03:14:54Z"));
    expect(menu.textContent).toContain(prettyDateTime("2026-05-16T01:14:42Z"));

    // Picking the other run rebinds the chat via chatbot_select_run.
    fireEvent.click(
      screen.getByTestId(
        "chatbot-helper-runrow-name-2026-05-16T01-14-42Z-f66s",
      ),
    );
    expect(invoke).toHaveBeenCalledWith("chatbot_select_run", {
      runId: "2026-05-16T01-14-42Z-f66s",
    });
    // The fresh session's chatbot_bound keeps the picked run current
    // (the collapsed toggle now reads the picked run's label).
    act(() => {
      handler({
        payload: {
          event: "chatbot_bound",
          run: "2026-05-16T01-14-42Z-f66s",
          selection: "user",
        },
      });
    });
    await waitFor(() =>
      expect(
        screen.getByTestId("chatbot-helper-runtoggle").textContent,
      ).toContain("barbellion-disappointed-man.txt"),
    );
  });

  it("shows the run's name (rename, else 4-letter id) alongside the label (#531)", async () => {
    vi.mocked(listen).mockImplementation(async () => () => {});
    const rows = [
      {
        run_id: "2026-05-16T14-11-07Z-cp9y",
        label: "personal-os.txt",
        created_at: "2026-05-16T14:11:07Z",
        store_path: "/s/c",
        bound: true,
      },
      {
        run_id: "2026-05-16T01-14-42Z-f66s",
        label: "barbellion-disappointed-man.txt",
        created_at: "2026-05-16T01:14:42Z",
        store_path: "/s/f",
        bound: false,
      },
    ];
    vi.mocked(invoke).mockImplementation(async (cmd) =>
      cmd === "chatbot_list_runs" ? rows : undefined,
    );
    // Mirror App's resolver: rename when set, else the 4-letter id.
    const resolveRunName = (id) =>
      id === "2026-05-16T01-14-42Z-f66s"
        ? "my-journal"
        : id === "2026-05-16T14-11-07Z-cp9y"
          ? "cp9y"
          : "";
    render(<ChatbotHelper resolveRunName={resolveRunName} />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));

    const toggle = await screen.findByTestId("chatbot-helper-runtoggle");
    // Collapsed toggle: 4-letter id appended to the bound run's subject
    // label; the time is the local-tz render of `created_at` (NYC, UTC−4:
    // 14:11 UTC → 10:11 AM), never the UTC server string (#984).
    expect(toggle.textContent).toContain("personal-os.txt");
    expect(toggle.textContent).toContain("cp9y");
    expect(toggle.textContent).toContain(prettyDateTime("2026-05-16T14:11:07Z"));
    expect(toggle.textContent).toContain("May 16, 10:11 AM");
    fireEvent.click(toggle);
    const menu = await screen.findByTestId("chatbot-helper-runmenu");
    // The rename (not the dir-name slug) for the renamed run.
    expect(menu.textContent).toContain(
      "barbellion-disappointed-man.txt",
    );
    expect(menu.textContent).toContain("my-journal");
    expect(menu.textContent).not.toContain("f66s");
  });

  it("a selector rebind does not yank the transcript when scrolled up", async () => {
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    // One in-flight turn so the rebind has a turn to cancel.
    await user.type(screen.getByTestId("chatbot-helper-input"), "hi");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => {
      handler({ payload: { event: "chatbot_chunk", delta: "answer" } });
    });

    // jsdom has no layout — synthesize a tall, scrolled-up pane.
    const pane = document.querySelector(".chatbot-helper-body");
    let st = 0;
    Object.defineProperty(pane, "scrollHeight", {
      value: 1000,
      configurable: true,
    });
    Object.defineProperty(pane, "clientHeight", {
      value: 300,
      configurable: true,
    });
    Object.defineProperty(pane, "scrollTop", {
      configurable: true,
      get: () => st,
      set: (v) => {
        st = v;
      },
    });
    // User scrolls up to read scrollback (far from the bottom).
    st = 0;
    fireEvent.scroll(pane);

    // A selector pick rebinds → the sidecar cancels the in-flight turn
    // → chatbot_stopped patches the transcript. It must STAY where the
    // user left it, not jump to the bottom (the reported bug).
    act(() => {
      handler({ payload: { event: "chatbot_stopped" } });
    });
    expect(st).toBe(0);

    // Sanity: the user's OWN new message still comes into view.
    await user.type(screen.getByTestId("chatbot-helper-input"), "next");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    expect(st).toBe(1000);
  });

  it("Enter without Shift sends; Shift+Enter inserts a newline", async () => {
    const user = userEvent.setup();
    await renderAndGetHandler();
    const input = screen.getByTestId("chatbot-helper-input");
    await user.type(input, "line1");
    await user.keyboard("{Shift>}{Enter}{/Shift}");
    await user.type(input, "line2");
    // Shift+Enter must not SEND (the run-list fetch on panel open is a
    // separate invoke and expected).
    expect(invoke).not.toHaveBeenCalledWith("chatbot", expect.anything());
    expect(input.value).toBe("line1\nline2");
    await user.keyboard("{Enter}");
    expect(invoke).toHaveBeenCalledWith("chatbot", {
      query: "line1\nline2",
      history: [],
    });
  });
});

describe("ChatbotHelper — multi-turn transcript", () => {
  it("appends chatbot_chunk deltas into the in-flight turn's answer", async () => {
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    await user.type(screen.getByTestId("chatbot-helper-input"), "q");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => { handler({ payload: { event: "chatbot_chunk", delta: "Hello" } }); });
    act(() => { handler({ payload: { event: "chatbot_chunk", delta: " world" } }); });
    expect(screen.getByTestId("chatbot-helper-a").textContent).toContain("Hello world");
  });

  it("chatbot_replace overwrites the bubble — #845 mixed-shape wipe", async () => {
    // The sidecar leaks the prose preamble around an embedded JSON
    // tool call as ordinary chunks, then emits chatbot_replace with
    // empty text once parse_tool_call extracts the call. The reducer
    // must REPLACE the turn's `a` (not append), so the persisted
    // transcript carries the clean text and the user sees the wipe.
    // textContent may also carry the trailing streaming caret (▌);
    // strip it before equality checks on the empty-bubble state.
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    await user.type(screen.getByTestId("chatbot-helper-input"), "q");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => { handler({ payload: { event: "chatbot_chunk", delta: 'Looking now... {"tool":' } }); });
    // Pre-wipe: the leak is visible.
    expect(screen.getByTestId("chatbot-helper-a").textContent).toContain('"tool"');
    act(() => { handler({ payload: { event: "chatbot_replace", text: "" } }); });
    // Post-wipe: the JSON / preamble is gone (caret may remain).
    const wiped =
      screen.getByTestId("chatbot-helper-a").textContent.replace("▌", "");
    expect(wiped).toBe("");
    // Next-hop grounded answer streams normally on top of the wipe.
    act(() => { handler({ payload: { event: "chatbot_chunk", delta: "Clean answer." } }); });
    expect(screen.getByTestId("chatbot-helper-a").textContent).toContain("Clean answer.");
    expect(screen.getByTestId("chatbot-helper-a").textContent).not.toContain("tool");
    expect(screen.getByTestId("chatbot-helper-a").textContent).not.toContain("Looking now");
  });

  it("retains the prior turn and feeds it back as history on the next send", async () => {
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    const input = screen.getByTestId("chatbot-helper-input");
    await user.type(input, "first question");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => { handler({ payload: { event: "chatbot_chunk", delta: "first answer" } }); });
    act(() => { handler({ payload: { event: "chatbot_done", resources: null } }); });

    await user.type(input, "follow up");
    await user.click(screen.getByTestId("chatbot-helper-send"));

    // Both turns are on screen — nothing was wiped.
    const questions = screen.getAllByTestId("chatbot-helper-q");
    expect(questions.map((q) => q.textContent)).toEqual([
      "first question",
      "follow up",
    ]);
    // The completed prior turn is fed back as conversation history.
    expect(lastInvokeArgs("chatbot")).toEqual({
      query: "follow up",
      history: [
        { role: "user", content: "first question" },
        { role: "assistant", content: "first answer" },
      ],
    });
  });

  it("keeps a cancelled turn's user message in history (the name-recall bug)", async () => {
    // Director's root-caused defect: "my name is alex" → response
    // Stopped → "what's my name" must still send "my name is alex" to
    // the model. History must include EVERY user message since the
    // last successful answer, not just completed user→assistant pairs.
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    const input = screen.getByTestId("chatbot-helper-input");
    await user.type(input, "my name is alex");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    // Cancel before any answer streams (assistant slot stays empty).
    await user.click(screen.getByTestId("chatbot-helper-stop"));
    await user.type(input, "what's my name");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    expect(lastInvokeArgs("chatbot")).toEqual({
      query: "what's my name",
      history: [{ role: "user", content: "my name is alex" }],
    });
  });

  it("keeps the partial assistant text of a cancelled turn in history", async () => {
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    const input = screen.getByTestId("chatbot-helper-input");
    await user.type(input, "tell me a long story");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => { handler({ payload: { event: "chatbot_chunk", delta: "Once upon a" } }); });
    act(() => { handler({ payload: { event: "chatbot_stopped" } }); });
    await user.type(input, "continue");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    expect(lastInvokeArgs("chatbot")).toEqual({
      query: "continue",
      history: [
        { role: "user", content: "tell me a long story" },
        { role: "assistant", content: "Once upon a" },
      ],
    });
  });

  it("drops a refused turn's assistant text from history (#834)", async () => {
    // The no-corpus refusal chunk lands in the turn's `a` field via
    // the same chatbot_chunk path as any other reply. The sidecar
    // signals this case with `refused: true` on chatbot_done so the
    // UI marks the turn and EXCLUDES its assistant slot from the
    // history fed into the next send. Without this, the deterministic
    // refusal text round-tripped through the prompt and the model
    // learned to mimic it as prose on every follow-up turn even after
    // a corpus was bound — the #834 history-poisoning loop.
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    const input = screen.getByTestId("chatbot-helper-input");
    await user.type(input, "first question");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => { handler({ payload: { event: "chatbot_chunk", delta: "I don't have a corpus to search yet — finish ingesting a folder, then start a new chat to use it." } }); });
    act(() => { handler({ payload: { event: "chatbot_done", resources: null, run: null, refused: true } }); });
    await user.type(input, "again");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    // The refused turn's user message survives (existing contract for
    // partial / errored turns); its assistant slot is dropped.
    expect(lastInvokeArgs("chatbot")).toEqual({
      query: "again",
      history: [{ role: "user", content: "first question" }],
    });
  });

  it("keeps the assistant slot of a non-refused completion in history", async () => {
    // Sister-case to the refusal-drop test above: a turn that completes
    // without `refused: true` (or with it explicitly false) MUST keep
    // its assistant slot in history. Confirms the filter is narrow —
    // only refused turns lose their content.
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    const input = screen.getByTestId("chatbot-helper-input");
    await user.type(input, "tell me about work");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => { handler({ payload: { event: "chatbot_chunk", delta: "Here is what I found." } }); });
    act(() => { handler({ payload: { event: "chatbot_done", resources: null, run: "run-A", refused: false } }); });
    await user.type(input, "more");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    expect(lastInvokeArgs("chatbot")).toEqual({
      query: "more",
      history: [
        { role: "user", content: "tell me about work" },
        { role: "assistant", content: "Here is what I found." },
      ],
    });
  });

  it("resolves resources to titled labels that open in the big window", async () => {
    const user = userEvent.setup();
    const onOpenResource = vi.fn();
    // Parent resolver: raw {index,kind,record_id} → real anchor + a
    // legible label. The component patches the turn with the result.
    const resolveResource = vi.fn(async (r) => ({
      relPath: "1-facts/work",
      anchor: "emotion-2",
      label: `fact · work · title ${r.record_id}`,
    }));
    const { handler } = await renderAndGetHandler({
      onOpenResource, resolveResource,
    });
    // Which corpus run this session answers from — resolution + the
    // click must carry it so the main window opens the right run.
    act(() => {
      handler({ payload: { event: "chatbot_bound", run: "run-xyz" } });
    });
    await user.type(screen.getByTestId("chatbot-helper-input"), "what did I note?");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => {
      handler({ payload: { event: "chatbot_retrieving", query: "my notes" } });
    });
    expect(screen.getByTestId("chatbot-helper-retrieving").textContent)
      .toContain("Searching your data");
    act(() => { handler({ payload: { event: "chatbot_chunk", delta: "Per [1]" } }); });
    const res1 = { index: 1, kind: "fact", record_id: "work:2" };
    act(() => {
      handler({
        payload: {
          event: "chatbot_done",
          resources: [
            res1,
            { index: 3, kind: "fact", record_id: "work:5" },
          ],
        },
      });
    });
    expect(screen.queryByTestId("chatbot-helper-retrieving")).toBeNull();
    expect(screen.getByTestId("chatbot-helper-resources")).toBeTruthy();
    // Resolver ran per resource with the bound run.
    await waitFor(() =>
      expect(resolveResource).toHaveBeenCalledWith(res1, "run-xyz"));
    // The row shows the resolved title, not the opaque record_id, and
    // there is no inline-expand preview. The display label uses NBSP
    // around middots (U+00A0·U+00A0) so separators never lead/trail a
    // wrapped line — the stored label and copy text keep plain spaces.
    await waitFor(() =>
      expect(screen.getByTestId("chatbot-helper-resource-1").textContent)
        .toContain("fact · work · title work:2"));
    expect(screen.queryByTestId("chatbot-helper-resource-preview-1")).toBeNull();
    // Clicking opens the cited item in the big window via the parent,
    // carrying the resolved target (relPath/anchor) + the bound run.
    await user.click(screen.getByTestId("chatbot-helper-resource-open-1"));
    expect(onOpenResource).toHaveBeenCalledWith(
      expect.objectContaining({
        index: 1, relPath: "1-facts/work", anchor: "emotion-2",
      }),
      "run-xyz",
    );
  });

  it("pins the answering run on the turn; a later rebind does NOT re-target old refs (#558)", async () => {
    // The chat window can hold messages about different runs. A ref
    // must resolve against the run THAT message was answered with, not
    // whatever the session is currently bound to. Pre-fix the click
    // passed the live `boundRun`, so switching runs mis-targeted (or,
    // post-restart with no bind yet, killed) every old ref.
    const user = userEvent.setup();
    const onOpenResource = vi.fn();
    const resolveResource = vi.fn(async (r, runId) => ({
      relPath: `1-facts/${runId}`,
      anchor: "emotion-2",
      label: `fact ${r.record_id} @ ${runId}`,
    }));
    const { handler } = await renderAndGetHandler({
      onOpenResource, resolveResource,
    });
    // Session bound to run-A; the answer is stamped run-A.
    act(() => {
      handler({ payload: { event: "chatbot_bound", run: "run-A" } });
    });
    await user.type(screen.getByTestId("chatbot-helper-input"), "what did I note?");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    const res1 = { index: 1, kind: "fact", record_id: "work:2" };
    act(() => {
      handler({
        payload: { event: "chatbot_done", resources: [res1], run: "run-A" },
      });
    });
    // Resolved against the stamped run, not just "the bound run".
    await waitFor(() =>
      expect(resolveResource).toHaveBeenCalledWith(res1, "run-A"));
    // The user switches the globally-selected run (a rebind → a fresh
    // session emits chatbot_bound for run-B).
    act(() => {
      handler({ payload: { event: "chatbot_bound", run: "run-B" } });
    });
    // Clicking the OLD turn's ref still opens run-A — its pinned run —
    // never the now-current run-B.
    await user.click(screen.getByTestId("chatbot-helper-resource-open-1"));
    expect(onOpenResource).toHaveBeenCalledWith(
      expect.objectContaining({ index: 1, relPath: "1-facts/run-A" }),
      "run-A",
    );
    expect(onOpenResource).not.toHaveBeenCalledWith(
      expect.anything(),
      "run-B",
    );
  });

  it("renders the explicit empty state when the tool matched nothing", async () => {
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    await user.type(screen.getByTestId("chatbot-helper-input"), "anything on X?");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => {
      handler({ payload: { event: "chatbot_retrieving", query: "X" } });
    });
    act(() => {
      handler({
        payload: {
          event: "chatbot_chunk",
          delta: "I don't have anything in your data about that.",
        },
      });
    });
    // Tool ran and matched nothing → resources is [].
    act(() => { handler({ payload: { event: "chatbot_done", resources: [] } }); });
    expect(screen.getByTestId("chatbot-helper-resources-empty").textContent)
      .toContain("No matching resources in your data.");
    expect(screen.queryByTestId("chatbot-helper-resources")).toBeNull();
  });

  it("renders NO resources block for a pure-conversation turn", async () => {
    // Refusal-coherence at the UI: a turn with no lookup (resources
    // null) shows neither a resources list nor the empty state — the
    // block reflects actual tool returns, not inline answer tokens.
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    await user.type(screen.getByTestId("chatbot-helper-input"), "let's brainstorm");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => {
      handler({ payload: { event: "chatbot_chunk", delta: "Sure — here's a thought." } });
    });
    act(() => { handler({ payload: { event: "chatbot_done", resources: null } }); });
    expect(screen.getByTestId("chatbot-helper-a").textContent)
      .toContain("Sure — here's a thought.");
    expect(screen.queryByTestId("chatbot-helper-resources")).toBeNull();
    expect(screen.queryByTestId("chatbot-helper-resources-empty")).toBeNull();
  });

  it("renders the error state on chatbot_error", async () => {
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    await user.type(screen.getByTestId("chatbot-helper-input"), "q");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => {
      handler({ payload: { event: "chatbot_error", message: "boom" } });
    });
    expect(screen.getByTestId("chatbot-helper-error").textContent).toContain("boom");
  });

  it("has no wipe/clear control — the session transcript is persistent", async () => {
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    await user.type(screen.getByTestId("chatbot-helper-input"), "q1");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => { handler({ payload: { event: "chatbot_chunk", delta: "a1" } }); });
    act(() => { handler({ payload: { event: "chatbot_done", resources: null } }); });
    // No "new conversation" / clear affordance exists at all.
    expect(screen.queryByTestId("chatbot-helper-new")).toBeNull();
    // And the turn stays in the transcript (closing/reopening keeps it).
    fireEvent.click(screen.getByTestId("chatbot-helper-close"));
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    expect(screen.getByTestId("chatbot-helper-q").textContent).toBe("q1");
    expect(screen.getByTestId("chatbot-helper-a").textContent).toContain("a1");
  });
});

describe("renumberCitations", () => {
  // The model cites by retrieved-context-row position; the surfaced
  // subset is sparse ([1][3][7]). This is a pure relabel: both the
  // answer markers and the resource list go through one shared map,
  // first-appearance order, nothing dropped.
  it("renumbers sparse cited rows to 1..K in answer-appearance order", () => {
    const out = renumberCitations(
      "Per [7] and [3], also [7] again, and [13].",
      [{ index: 3 }, { index: 7 }, { index: 13 }],
    );
    // First appearance order: 7→1, 3→2, 13→3.
    expect(out.answer).toBe("Per [1] and [2], also [1] again, and [3].");
    expect(out.resources.map((r) => r.index)).toEqual([1, 2, 3]);
  });

  it("keeps >5 citations (no post-hoc drop)", () => {
    const idx = [2, 4, 6, 8, 10, 12, 14];
    const answer = idx.map((n) => `[${n}]`).join(" ");
    const out = renumberCitations(answer, idx.map((index) => ({ index })));
    expect(out.resources).toHaveLength(7);
    expect(out.resources.map((r) => r.index)).toEqual([1, 2, 3, 4, 5, 6, 7]);
    expect(out.answer).toBe("[1] [2] [3] [4] [5] [6] [7]");
  });

  it("preserves enriched fields and appends an uncited resource stably", () => {
    const out = renumberCitations(
      "Only [5] is cited.",
      [
        { index: 5, label: "fact · a", relPath: "1-facts/a", anchor: "x-1" },
        { index: 9, label: "entity · b", relPath: "2-entities/b", anchor: "" },
      ],
    );
    expect(out.answer).toBe("Only [1] is cited.");
    expect(out.resources).toEqual([
      { index: 1, label: "fact · a", relPath: "1-facts/a", anchor: "x-1" },
      { index: 2, label: "entity · b", relPath: "2-entities/b", anchor: "" },
    ]);
  });

  it("renumbers from the answer alone when no resources yet (mid-stream)", () => {
    // During streaming the resource list does not exist yet; the map
    // is built from the answer so the displayed number is already
    // contiguous and never flashes the raw context-row number.
    expect(renumberCitations("hello [2] world", [])).toEqual({
      answer: "hello [1] world", resources: [],
    });
    expect(renumberCitations("", null)).toEqual({ answer: "", resources: [] });
  });

  it("renumbers any closed [digits] token (looks-like-a-citation)", () => {
    // A closed [N] is unambiguously a citation token and is remapped
    // even with no resource row for it (a stray / out-of-range cite
    // the model emitted) — the ≤5/refusal discipline is the model's.
    const out = renumberCitations("cited [4], also [99].", [{ index: 4 }]);
    expect(out.answer).toBe("cited [1], also [2].");
    expect(out.resources.map((r) => r.index)).toEqual([1]);
  });

  // Streaming guard: a partial trailing citation must be withheld
  // (never flash the raw context-row number) until it resolves.
  it("withholds a lone trailing '[' until it resolves", () => {
    expect(renumberCitations("Per ", []).answer).toBe("Per ");
    expect(renumberCitations("Per [", []).answer).toBe("Per ");
  });

  it("withholds '[' + partial digits with no closing bracket", () => {
    expect(renumberCitations("Per [7", [{ index: 7 }]).answer).toBe("Per ");
    expect(renumberCitations("Per [13", []).answer).toBe("Per ");
  });

  it("emits the remapped citation once the bracket closes", () => {
    // The map is stable across the grow: 7 is the first distinct →
    // [1], regardless of how much streamed before/after.
    expect(renumberCitations("Per [7]", [{ index: 7 }]).answer)
      .toBe("Per [1]");
    expect(renumberCitations("A [9] and [4", [{ index: 9 }]).answer)
      .toBe("A [1] and ");
  });

  it("does NOT withhold a non-digit after '[' (not a citation)", () => {
    expect(renumberCitations("see arr[i", []).answer).toBe("see arr[i");
    expect(renumberCitations("note [TODO", []).answer).toBe("note [TODO");
  });

  it("does NOT withhold once the digit run exceeds 6 (not a citation)", () => {
    expect(renumberCitations("ref [1234567", []).answer).toBe("ref [1234567");
  });

  it("does NOT misread programming-syntax brackets as citations", () => {
    // ``arr[1]`` is code, not a citation — negative-lookbehind on the
    // opening ``[`` keeps it out of the citation pool (citations in
    // real prose are preceded by whitespace or punctuation, never by
    // another letter).
    expect(renumberCitations("see arr[1]", [{ index: 9 }]).answer)
      .toBe("see arr[1]");
    // Partial ``arr[1`` shouldn't be withheld either (would eat code).
    expect(renumberCitations("see arr[1", []).answer).toBe("see arr[1");
  });

  it("does NOT match non-digit bracket tokens ([TODO], [NASA])", () => {
    // Real prose tokens like [TODO], [NASA], [API] don't match the
    // ``\d+`` citation regex and pass through as literal text.
    expect(renumberCitations("note [TODO clean up]", []).answer)
      .toBe("note [TODO clean up]");
    expect(renumberCitations("ack [NASA] launch", []).answer)
      .toBe("ack [NASA] launch");
  });
});

describe("renderAnswerWithRefs — clickable in-body [N] (reuse, not parallel)", () => {
  it("a [N] with a matching resource becomes a button that fires onRef", () => {
    const r1 = { index: 1, kind: "fact", record_id: "w:1", relPath: "p" };
    const onRef = vi.fn();
    const parts = renderAnswerWithRefs(
      "Per [1] and [2] done.", [r1], onRef, "May 1",
    );
    render(<div data-testid="out">{parts}</div>);
    // textContent is byte-identical to the input (the token text is
    // still `[1]`), so the copied/visible answer reads the same.
    expect(screen.getByTestId("out").textContent).toBe(
      "Per [1] and [2] done.",
    );
    // [1] has a resource → clickable, carrying the date on its hover.
    const cite = screen.getByTestId("chatbot-helper-cite-1");
    expect(cite.tagName).toBe("BUTTON");
    expect(cite.getAttribute("title")).toContain("May 1");
    fireEvent.click(cite);
    expect(onRef).toHaveBeenCalledWith(r1);
    // [2] has no resource → inert text, no button.
    expect(screen.queryByTestId("chatbot-helper-cite-2")).toBeNull();
  });

  it("no resources / empty answer → plain text, no buttons", () => {
    render(<div data-testid="o2">{renderAnswerWithRefs("a [1] b", [])}</div>);
    expect(screen.getByTestId("o2").textContent).toBe("a [1] b");
    expect(screen.queryByTestId("chatbot-helper-cite-1")).toBeNull();
    expect(renderAnswerWithRefs("", null, vi.fn())).toEqual([]);
  });
});

describe("runRefDate / prettyDate — date-only on references", () => {
  it("prettyDate is month-day, no time, current year dropped", () => {
    const yr = new Date().getFullYear();
    const d = prettyDate(`${yr}-05-17T09:00:00Z`);
    expect(d).toMatch(/May/);
    expect(d).not.toMatch(/\d:\d{2}/); // NO time
    expect(d).not.toMatch(new RegExp(`${yr}`)); // current year dropped
    expect(prettyDate(`${yr - 2}-05-17T09:00:00Z`)).toMatch(
      new RegExp(`${yr - 2}`),
    ); // older year kept
    expect(prettyDate("")).toBe("");
  });

  it("runRefDate maps a turn's run to its option date; '' when unknown", () => {
    const opts = [{ run_id: "r1", created_at: "2026-05-17T09:00:00Z" }];
    expect(runRefDate("r1", opts)).toBe(prettyDate("2026-05-17T09:00:00Z"));
    expect(runRefDate("nope", opts)).toBe("");
    expect(runRefDate(null, opts)).toBe("");
  });

  it("buildReplyCopyText appends the date to the source + each ref", () => {
    // The source label is the corpus subject only (no embedded date);
    // the date is appended client-side from the run's `created_at`.
    const txt = buildReplyCopyText(
      "Body [1]",
      [{ index: 1, kind: "fact", record_id: "w:1", label: "fact · w" }],
      "src",
      "May 1",
    );
    expect(txt).toContain("src · May 1");
    expect(txt).toContain("[1] fact · w · May 1");
  });
});

describe("ChatbotHelper — in-body [N] reuses the chip ref-click path", () => {
  it("clicking an in-body [N] opens the SAME resolved target + pinned run as the chip", async () => {
    const user = userEvent.setup();
    const onOpenResource = vi.fn();
    const resolveResource = vi.fn(async (r) => ({
      ...r, relPath: "1-facts/work", anchor: "emotion-2",
    }));
    const runs = [{
      run_id: "run-xyz", label: "notes.txt",
      store_path: "/s", bound: true,
      created_at: "2026-05-01T09:00:00Z", short_id: "ab3k",
    }];
    const { handler } = await renderAndGetHandler({
      onOpenResource,
      resolveResource,
      resolveRunName: () => "",
    });
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) =>
        cmd === "chatbot_list_runs" ? runs : undefined),
    );
    act(() => {
      handler({ payload: { event: "chatbot_bound", run: "run-xyz" } });
    });
    await user.type(screen.getByTestId("chatbot-helper-input"), "q?");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => { handler({ payload: { event: "chatbot_chunk", delta: "Per [1]" } }); });
    const res1 = { index: 1, kind: "fact", record_id: "work:2" };
    act(() => {
      handler({
        payload: {
          event: "chatbot_done", run: "run-xyz", resources: [res1],
        },
      });
    });
    // Resolved row carries the date-only (no time) of the cited source.
    await waitFor(() =>
      expect(
        screen.getByTestId("chatbot-helper-resource-date-1").textContent,
      ).toBe(prettyDate("2026-05-01T09:00:00Z")));
    expect(
      screen.getByTestId("chatbot-helper-resource-date-1").textContent,
    ).not.toMatch(/\d:\d{2}/);
    // Click the in-body [1] (not the chip): same resolved target +
    // same pinned run as the chip path — one shared handler.
    await user.click(screen.getByTestId("chatbot-helper-cite-1"));
    expect(onOpenResource).toHaveBeenCalledWith(
      expect.objectContaining({
        index: 1, relPath: "1-facts/work", anchor: "emotion-2",
      }),
      "run-xyz",
    );
    const inBodyCall = onOpenResource.mock.calls.at(-1);
    onOpenResource.mockClear();
    await user.click(screen.getByTestId("chatbot-helper-resource-open-1"));
    // Chip click === in-body click (same resource object + run).
    expect(onOpenResource.mock.calls.at(-1)).toEqual(inBodyCall);
  });
});

describe("ChatbotHelper — transcript persistence across window destroy", () => {
  // The window (and its WKWebView) is destroyed on Cmd/Ctrl+W and
  // rebuilt fresh on reopen — a brand-new component mount, not a
  // close/open of the panel. These tests model that as a full
  // unmount→remount and assert the conversation is reloaded from the
  // persisted store rather than wiped.

  it("rehydrates the saved transcript on a fresh mount (destroy→reopen)", async () => {
    const saved = [
      {
        id: 1,
        q: "my name is alex",
        a: "Nice to meet you, Alex.",
        status: "done",
        retrieving: false,
        retrievingQuery: "",
        resources: null,
      },
    ];
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) =>
        cmd === "chatbot_load_transcript" ? saved : undefined,
      ),
    );
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    await waitFor(() =>
      expect(screen.getByTestId("chatbot-helper-q").textContent).toBe(
        "my name is alex",
      ),
    );
    expect(screen.getByTestId("chatbot-helper-a").textContent).toContain(
      "Nice to meet you, Alex.",
    );
  });

  it("restored refs resolve + click against the turn's pinned run with ZERO new queries / no rebind (#558)", async () => {
    // The exact #558 symptom in wiring form: after a restart the
    // session has NOT re-bound (no chatbot_bound event) and no new
    // query is sent. Pre-fix, restored refs resolved/clicked against
    // the null live bind → dead until a new answer incidentally
    // repopulated it. Post-fix the turn carries its own run_id, so
    // resolution + click work immediately, standalone.
    const onOpenResource = vi.fn();
    const resolveResource = vi.fn(async (r, runId) => ({
      relPath: `1-facts/${runId}`,
      anchor: "emotion-2",
      label: `fact ${r.record_id} @ ${runId}`,
    }));
    const saved = {
      open: true,
      turns: [
        {
          id: 1,
          q: "what did I note?",
          a: "Per [1]",
          status: "done",
          retrieving: false,
          retrievingQuery: "",
          runId: "run-A",
          resources: [{ index: 1, kind: "fact", record_id: "work:2" }],
        },
      ],
    };
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) =>
        cmd === "chatbot_load_transcript" ? saved : undefined,
      ),
    );
    const user = userEvent.setup();
    render(
      <ChatbotHelper
        onOpenResource={onOpenResource}
        resolveResource={resolveResource}
      />,
    );
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    // Re-resolved against the turn's OWN pinned run — no chatbot_bound
    // was ever emitted, no query sent.
    await waitFor(() =>
      expect(resolveResource).toHaveBeenCalledWith(
        expect.objectContaining({ record_id: "work:2" }),
        "run-A",
      ),
    );
    // No new query went to the sidecar.
    expect(
      vi.mocked(invoke).mock.calls.some((c) => c[0] === "chatbot"),
    ).toBe(false);
    await user.click(screen.getByTestId("chatbot-helper-resource-open-1"));
    expect(onOpenResource).toHaveBeenCalledWith(
      expect.objectContaining({ relPath: "1-facts/run-A" }),
      "run-A",
    );
  });

  it("two restored turns about different runs each resolve to their OWN pinned run (#558 multi-run)", async () => {
    const onOpenResource = vi.fn();
    const resolveResource = vi.fn(async (r, runId) => ({
      relPath: `1-facts/${runId}`,
      anchor: "a",
      label: `${r.record_id} @ ${runId}`,
    }));
    const saved = {
      open: true,
      turns: [
        {
          id: 1, q: "about run A", a: "Per [1]", status: "done",
          retrieving: false, retrievingQuery: "", runId: "run-A",
          resources: [{ index: 1, kind: "fact", record_id: "a:1" }],
        },
        {
          id: 2, q: "about run B", a: "Per [1]", status: "done",
          retrieving: false, retrievingQuery: "", runId: "run-B",
          resources: [{ index: 1, kind: "fact", record_id: "b:1" }],
        },
      ],
    };
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) =>
        cmd === "chatbot_load_transcript" ? saved : undefined,
      ),
    );
    const user = userEvent.setup();
    render(
      <ChatbotHelper
        onOpenResource={onOpenResource}
        resolveResource={resolveResource}
      />,
    );
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    await waitFor(() =>
      expect(resolveResource).toHaveBeenCalledWith(
        expect.objectContaining({ record_id: "a:1" }), "run-A"));
    await waitFor(() =>
      expect(resolveResource).toHaveBeenCalledWith(
        expect.objectContaining({ record_id: "b:1" }), "run-B"));
    const opens = screen.getAllByTestId("chatbot-helper-resource-open-1");
    await user.click(opens[0]);
    await user.click(opens[1]);
    expect(onOpenResource).toHaveBeenCalledWith(
      expect.objectContaining({ relPath: "1-facts/run-A" }), "run-A");
    expect(onOpenResource).toHaveBeenCalledWith(
      expect.objectContaining({ relPath: "1-facts/run-B" }), "run-B");
  });

  it("a turn interrupted mid-generation RESUMES on reopen and completes under its own question (req 1)", async () => {
    // Window closed mid-answer → the turn was persisted `streaming`
    // with only a stale partial. On reopen it must NOT freeze as a
    // truncated answer — it must resume against the warm sidecar and
    // complete, threaded to its question.
    const saved = {
      open: true,
      turns: [
        {
          id: 1,
          q: "tell me a story",
          a: "Once upon a ti",
          status: "streaming",
          retrieving: false,
          retrievingQuery: "",
          resources: null,
        },
      ],
    };
    let handler = null;
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      if (evt === "chatbot-event") handler = fn;
      return () => {};
    });
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) =>
        cmd === "chatbot_load_transcript" ? saved : undefined,
      ),
    );
    render(<ChatbotHelper />);
    // The interrupted query is re-issued to the sidecar (no prior
    // turns → empty history), the stale partial discarded.
    await waitFor(() =>
      expect(lastInvokeArgs("chatbot")).toEqual({
        query: "tell me a story",
        history: [],
      }),
    );
    await waitFor(() => expect(handler).toBeTruthy());
    // It's in-flight again (not frozen): completing it threads the
    // full answer under THE SAME (only) turn.
    act(() => {
      handler({ payload: { event: "chatbot_chunk", delta: "Here is the complete story." } });
    });
    act(() => { handler({ payload: { event: "chatbot_done", resources: null } }); });
    const as = screen.getAllByTestId("chatbot-helper-a");
    expect(as).toHaveLength(1);
    expect(as[0].textContent).toContain("Here is the complete story.");
    // Stale partial discarded — not concatenated in front of the
    // resumed answer.
    expect(as[0].textContent).not.toContain("Once upon a ti");
  });

  it("a still-running pre-close generation cannot concatenate onto the resumed turn (round-4 reviewer regression)", async () => {
    // The window was closed mid-gen and the user reopens BEFORE the
    // un-cancelled gen-N finishes. gen-N's tail and the resumed
    // gen-N+1 both carry the same client queryId, so the listener
    // guard alone cannot separate them. The fix cancels gen-N first
    // and holds the guard disarmed across the cancel; only after the
    // kill+respawn resolves is it armed for the re-issue. Assert: a
    // stale event during that window is DROPPED, the final answer is
    // the re-issued answer only (no concatenation), one terminal done.
    const saved = {
      open: true,
      turns: [
        { id: 1, q: "name three colors", a: "Red, gr", status: "streaming",
          retrieving: false, retrievingQuery: "", resources: null },
      ],
    };
    let handler = null;
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      if (evt === "chatbot-event") handler = fn;
      return () => {};
    });
    let releaseCancel;
    const cancelGate = new Promise((r) => { releaseCancel = r; });
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) => {
        if (cmd === "chatbot_load_transcript") return saved;
        if (cmd === "chatbot_cancel") return cancelGate; // gen-N "still killing"
        return undefined;
      }),
    );
    render(<ChatbotHelper />);
    await waitFor(() => expect(handler).toBeTruthy());
    // chatbot_cancel issued, NOT yet resolved → guard still disarmed.
    await waitFor(() =>
      expect(
        vi.mocked(invoke).mock.calls.some((c) => c[0] === "chatbot_cancel"),
      ).toBe(true),
    );
    // gen-N's stale tail arrives now (the exact reviewer scenario):
    // must be dropped, not appended.
    act(() => {
      handler({ payload: { event: "chatbot_chunk", delta: "STALE-N-TAIL" } });
      handler({ payload: { event: "chatbot_done", resources: null } });
    });
    // Resolve the cancel → re-arm + the resumed re-issue fires.
    await act(async () => { releaseCancel(); });
    await waitFor(() =>
      expect(
        vi.mocked(invoke).mock.calls.some((c) => c[0] === "chatbot"),
      ).toBe(true),
    );
    // Resumed gen-N+1 streams cleanly.
    act(() => {
      handler({ payload: { event: "chatbot_chunk", delta: "Red, blue, ochre." } });
      handler({ payload: { event: "chatbot_done", resources: null } });
    });
    const as = screen.getAllByTestId("chatbot-helper-a");
    expect(as).toHaveLength(1);
    expect(as[0].textContent).toContain("Red, blue, ochre.");
    // No concatenation of the stale gen-N tail or the stale partial.
    expect(as[0].textContent).not.toContain("STALE-N-TAIL");
    expect(as[0].textContent).not.toContain("Red, gr");
  });

  it("turn_id fence: a stale lower-turn_id event cannot duplicate the resumed turn (round-6 greeting-twice regression)", async () => {
    // The binding-gate defect: on a fast close→reopen the original
    // greeting generation can still emit; with a queryId-only,
    // generation-blind guard its tail rendered as a SECOND identical
    // greeting. The `chatbot` command now returns a process-global
    // monotonic turn_id; the resume pins it and the listener drops any
    // event whose turn_id differs. A stale pre-respawn event carries a
    // LOWER id and is structurally rejected — exactly once, by id.
    const saved = {
      open: true,
      turns: [
        { id: 1, q: "hi", a: "Hey! Good to s", status: "streaming",
          retrieving: false, retrievingQuery: "", resources: null },
      ],
    };
    let handler = null;
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      if (evt === "chatbot-event") handler = fn;
      return () => {};
    });
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) => {
        if (cmd === "chatbot_load_transcript") return saved;
        if (cmd === "chatbot_cancel") return undefined;
        if (cmd === "chatbot") return 7; // resumed generation's turn_id
        return undefined;
      }),
    );
    render(<ChatbotHelper />);
    await waitFor(() => expect(handler).toBeTruthy());
    // Wait until the resume's re-issue resolved and pinned turn_id=7
    // (the chatbot mock returned 7).
    await waitFor(() =>
      expect(
        vi.mocked(invoke).mock.calls.some((c) => c[0] === "chatbot"),
      ).toBe(true),
    );
    await act(async () => {}); // flush the .then that pins expected=7
    // The ORIGINAL greeting generation (pre-respawn, turn_id=1) is
    // still draining — it must NOT render a second greeting.
    act(() => {
      handler({ payload: { event: "chatbot_chunk", delta: "Hey! Good to see you. Anything on your mind, or just checking in?", turn_id: 1 } });
      handler({ payload: { event: "chatbot_done", resources: null, turn_id: 1 } });
    });
    // The resumed generation (turn_id=7) is the only one that lands.
    act(() => {
      handler({ payload: { event: "chatbot_chunk", delta: "Hey, good to see you again!", turn_id: 7 } });
      handler({ payload: { event: "chatbot_done", resources: null, turn_id: 7 } });
    });
    const as = screen.getAllByTestId("chatbot-helper-a");
    expect(as).toHaveLength(1);
    expect(as[0].textContent).toContain("Hey, good to see you again!");
    // The stale turn_id=1 greeting was dropped — it appears ZERO times,
    // so the greeting is not rendered twice.
    expect(as[0].textContent).not.toContain("just checking in?");
  });

  // Last { open, turns } passed to chatbot_save_conversation (#565 —
  // one transcript per conversation dir).
  function lastSavedState() {
    const calls = vi
      .mocked(invoke)
      .mock.calls.filter((c) => c[0] === "chatbot_save_conversation");
    return calls.length ? calls[calls.length - 1][1].state : null;
  }

  it("persists the transcript once a turn settles (done)", async () => {
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    await user.type(screen.getByTestId("chatbot-helper-input"), "q1");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => { handler({ payload: { event: "chatbot_chunk", delta: "a1" } }); });
    await act(async () => {
      handler({ payload: { event: "chatbot_done", resources: null } });
    });
    const persisted = lastSavedState().turns;
    expect(persisted).toHaveLength(1);
    expect(persisted[0]).toMatchObject({ q: "q1", a: "a1", status: "done" });
  });

  it("persists the pending user message immediately on send, before the answer (req 1)", async () => {
    const user = userEvent.setup();
    await renderAndGetHandler();
    await user.type(screen.getByTestId("chatbot-helper-input"), "who am i?");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    // No chatbot_done yet — the answer is still pending. The user's
    // message must already be on disk (as a streaming turn) so a
    // window close right now doesn't lose it.
    const state = lastSavedState();
    expect(state.turns).toHaveLength(1);
    expect(state.turns[0]).toMatchObject({
      q: "who am i?",
      status: "streaming",
    });
  });

  it("an in-flight turn rehydrates threaded to ITS question, not after the prior answer (req 1)", async () => {
    const saved = {
      open: true,
      turns: [
        { id: 1, q: "first q", a: "first answer", status: "done",
          retrieving: false, retrievingQuery: "", resources: null },
        { id: 2, q: "second q", a: "partial sec", status: "streaming",
          retrieving: false, retrievingQuery: "", resources: null },
      ],
    };
    let handler = null;
    vi.mocked(listen).mockImplementation(async (evt, fn) => {
      if (evt === "chatbot-event") handler = fn;
      return () => {};
    });
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) =>
        cmd === "chatbot_load_transcript" ? saved : undefined,
      ),
    );
    render(<ChatbotHelper />);
    await waitFor(() =>
      expect(screen.getAllByTestId("chatbot-helper-q")).toHaveLength(2),
    );
    expect(
      screen.getAllByTestId("chatbot-helper-q").map((n) => n.textContent),
    ).toEqual(["first q", "second q"]);
    // The interrupted 2nd turn is RESUMED — re-issued with the prior
    // turn as history, NOT frozen with its stale "partial sec".
    await waitFor(() =>
      expect(lastInvokeArgs("chatbot")).toEqual({
        query: "second q",
        history: [
          { role: "user", content: "first q" },
          { role: "assistant", content: "first answer" },
        ],
      }),
    );
    await waitFor(() => expect(handler).toBeTruthy());
    act(() => {
      handler({ payload: { event: "chatbot_chunk", delta: "the resumed full answer" } });
    });
    act(() => { handler({ payload: { event: "chatbot_done", resources: null } }); });
    const as = screen.getAllByTestId("chatbot-helper-a").map((n) => n.textContent);
    // Completed answer threads under the 2nd turn; the 1st answer is
    // untouched — never concatenated, never reordered, stale partial
    // discarded.
    expect(as[0]).toContain("first answer");
    expect(as[0]).not.toContain("the resumed full answer");
    expect(as[0]).not.toContain("partial sec");
    expect(as[1]).toContain("the resumed full answer");
    expect(as[1]).not.toContain("partial sec");
  });

  it("restores the chat panel OPEN when it was open before close (req 2)", async () => {
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) =>
        cmd === "chatbot_load_transcript"
          ? { open: true, turns: [] }
          : undefined,
      ),
    );
    render(<ChatbotHelper />);
    // No pill click — the panel must come up open on its own.
    await waitFor(() =>
      expect(screen.getByTestId("chatbot-helper-panel")).toBeTruthy(),
    );
    expect(screen.queryByTestId("chatbot-helper-pill")).toBeNull();
  });

  it("stays collapsed when it was closed before (req 2)", async () => {
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) =>
        cmd === "chatbot_load_transcript"
          ? { open: false, turns: [] }
          : undefined,
      ),
    );
    render(<ChatbotHelper />);
    await act(async () => {});
    expect(screen.getByTestId("chatbot-helper-pill")).toBeTruthy();
    expect(screen.queryByTestId("chatbot-helper-panel")).toBeNull();
  });

  it("persists the panel open state when the user opens it (req 2)", async () => {
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(convoMock());
    render(<ChatbotHelper />);
    await act(async () => {});
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    await waitFor(() => expect(lastSavedState()).toBeTruthy());
    expect(lastSavedState().open).toBe(true);
  });

  it("a failed load does NOT persist an empty transcript over a good file", async () => {
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) => {
        if (cmd === "chatbot_load_transcript")
          throw new Error("backend down");
        return undefined;
      }),
    );
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    // Give the rejected load + any effects a tick to flush.
    await act(async () => {});
    expect(
      vi
        .mocked(invoke)
        .mock.calls.some((c) => c[0] === "chatbot_save_conversation"),
    ).toBe(false);
  });
});

describe("ChatbotHelper — 'Thinking…' indicator is sidecar-event-driven", () => {
  // The indicator is driven solely by the per-turn `chatbot_thinking`
  // event, which the sidecar emits at the start of EVERY turn (the
  // decision turn is always buffered). Shown for ALL responses
  // regardless of reasoning. It is never UI-inferred, so absent the
  // event nothing renders (defensive for old/future sidecars).

  it("shows 'Thinking…' on a chatbot_thinking event, cleared by the first chunk", async () => {
    const { handler } = await renderAndGetHandler();
    const user = userEvent.setup();
    await user.type(screen.getByTestId("chatbot-helper-input"), "q");
    await user.click(screen.getByTestId("chatbot-helper-send"));

    // Every turn: the sidecar announces the buffered gap.
    act(() => {
      handler({ payload: { event: "chatbot_thinking" } });
    });
    const thinking = screen.getByTestId("chatbot-helper-thinking");
    expect(thinking.textContent).toContain("Thinking…");
    expect(screen.queryByTestId("chatbot-helper-a")).toBeNull();

    // First content delta ends the gap → indicator gone, answer shows.
    act(() => {
      handler({ payload: { event: "chatbot_chunk", delta: "Hi" } });
    });
    expect(screen.queryByTestId("chatbot-helper-thinking")).toBeNull();
    expect(screen.getByTestId("chatbot-helper-a").textContent).toBe("Hi▌");
  });

  it("renders nothing until the event arrives (purely event-driven, never UI-inferred)", async () => {
    const { handler } = await renderAndGetHandler();
    const user = userEvent.setup();
    // Send → the turn is briefly in-flight with no content. The
    // indicator is NOT UI-inferred from that state: it appears only
    // once the sidecar's chatbot_thinking event arrives, then clears
    // on the first chunk. (The sidecar emits the event for every turn;
    // this pins that the UI never fabricates it on its own.)
    await user.type(screen.getByTestId("chatbot-helper-input"), "q");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    expect(screen.queryByTestId("chatbot-helper-thinking")).toBeNull();
    act(() => {
      handler({ payload: { event: "chatbot_thinking" } });
    });
    expect(screen.getByTestId("chatbot-helper-thinking")).toBeTruthy();
    act(() => {
      handler({ payload: { event: "chatbot_chunk", delta: "answer" } });
    });
    expect(screen.queryByTestId("chatbot-helper-thinking")).toBeNull();
    expect(screen.getByTestId("chatbot-helper-a").textContent).toBe("answer▌");
  });

  it("clears 'Thinking…' when retrieval starts instead of a chunk", async () => {
    const { handler } = await renderAndGetHandler();
    const user = userEvent.setup();
    await user.type(screen.getByTestId("chatbot-helper-input"), "q");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => {
      handler({ payload: { event: "chatbot_thinking" } });
    });
    expect(screen.getByTestId("chatbot-helper-thinking")).toBeTruthy();
    act(() => {
      handler({ payload: { event: "chatbot_retrieving", query: "x" } });
    });
    expect(screen.queryByTestId("chatbot-helper-thinking")).toBeNull();
    expect(screen.getByTestId("chatbot-helper-retrieving")).toBeTruthy();
  });
});

describe("labelForRun — the single source of the run label", () => {
  const opts = [
    { run_id: "run-A", label: "journal.txt — May 16, 01:14" },
    { run_id: "run-B", label: "work.txt — May 16, 14:11" },
  ];
  const resolveRunName = (id) =>
    id === "run-A" ? "my-journal" : id === "run-B" ? "" : "";

  it("in-list run with a name → 'label · name' (the selector's exact string)", () => {
    expect(labelForRun("run-A", opts, resolveRunName)).toBe(
      "journal.txt — May 16, 01:14 · my-journal",
    );
  });
  it("in-list run with no name → just the label", () => {
    expect(labelForRun("run-B", opts, resolveRunName)).toBe(
      "work.txt — May 16, 14:11",
    );
  });
  it("run not in the list → resolved name, else the raw id (no-list degradation)", () => {
    expect(labelForRun("run-X", opts, (id) => (id === "run-X" ? "old" : "")))
      .toBe("old");
    expect(labelForRun("run-X", opts, () => "")).toBe("run-X");
    expect(labelForRun("run-X", [], () => "")).toBe("run-X");
  });
  it("no run → '' (caller omits the label)", () => {
    expect(labelForRun(null, opts, resolveRunName)).toBe("");
    expect(labelForRun(undefined, opts, resolveRunName)).toBe("");
  });
});

describe("ChatbotHelper — per-message copy button (wiring)", () => {
  // Drive clicks with fireEvent (not userEvent) here: userEvent.setup()
  // installs its OWN navigator.clipboard stub, which would detach this
  // spy. The copy path is a plain onClick → navigator.clipboard.
  let writeText;
  beforeEach(() => {
    writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
  });

  it("copy lives ONLY on the reply, and copies THAT reply's own text", async () => {
    const saved = {
      open: true,
      turns: [
        {
          id: 1, q: "first question",
          a: "first answer.", status: "done",
          retrieving: false, retrievingQuery: "", resources: null,
        },
        {
          id: 2, q: "second question",
          a: "second answer.", status: "done",
          retrieving: false, retrievingQuery: "", resources: null,
        },
      ],
    };
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) =>
        cmd === "chatbot_load_transcript" ? saved : undefined,
      ),
    );
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    await screen.findByText("second question");

    // No copy button on the user's own messages — only on replies.
    expect(screen.queryAllByTestId("chatbot-helper-copy-q")).toHaveLength(0);
    const aCopies = screen.getAllByTestId("chatbot-helper-copy-a");
    expect(aCopies).toHaveLength(2);
    // Each reply's copy → that turn's OWN answer.
    fireEvent.click(aCopies[1]);
    expect(writeText).toHaveBeenLastCalledWith("second answer.");
    fireEvent.click(aCopies[0]);
    expect(writeText).toHaveBeenLastCalledWith("first answer.");
  });

  it("a grounded reply copies the renumbered answer WITH its references + source", async () => {
    const rows = [
      { run_id: "run-A", label: "journal.txt — May 16, 01:14",
        store_path: "/s/a", bound: false },
    ];
    const resolveRunName = (id) => (id === "run-A" ? "my-journal" : "");
    const saved = {
      open: true,
      turns: [
        {
          id: 1, q: "q", a: "See [7] and [3].", status: "done",
          retrieving: false, retrievingQuery: "", runId: "run-A",
          resources: [
            { index: 7, kind: "fact", record_id: "x:1",
              label: "fact · work · raise" },
            { index: 3, kind: "entity", record_id: "e:2",
              label: "entity · people · Sam" },
          ],
        },
      ],
    };
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) =>
        cmd === "chatbot_load_transcript"
          ? saved
          : cmd === "chatbot_list_runs"
            ? rows
            : undefined,
      ),
    );
    render(<ChatbotHelper resolveRunName={resolveRunName} />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    await screen.findByTestId("chatbot-helper-copy-a");
    fireEvent.click(screen.getByTestId("chatbot-helper-copy-a"));
    // Sparse [7]/[3] renumber to [1]/[2] in first-appearance order;
    // the copied text carries the SAME markers AND the reference
    // list (so the markers stay meaningful) plus this turn's source
    // label.
    expect(writeText).toHaveBeenLastCalledWith(
      "See [1] and [2].\n\n" +
        "Resources from your data\n" +
        "journal.txt — May 16, 01:14 · my-journal\n" +
        "[1] fact · work · raise\n" +
        "[2] entity · people · Sam",
    );
  });

  it("a non-grounded reply copies just the answer (no references section)", async () => {
    const saved = {
      open: true,
      turns: [
        {
          id: 1, q: "hi", a: "Just chatting, no lookup.",
          status: "done", retrieving: false, retrievingQuery: "",
          resources: null,
        },
      ],
    };
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) =>
        cmd === "chatbot_load_transcript" ? saved : undefined,
      ),
    );
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    await screen.findByTestId("chatbot-helper-copy-a");
    fireEvent.click(screen.getByTestId("chatbot-helper-copy-a"));
    expect(writeText).toHaveBeenLastCalledWith("Just chatting, no lookup.");
  });

  it("the copy button sits at the very bottom — AFTER the references — when grounded", async () => {
    const saved = {
      open: true,
      turns: [
        {
          id: 1, q: "q", a: "Per [1].", status: "done",
          retrieving: false, retrievingQuery: "", runId: "run-A",
          resources: [{ index: 1, kind: "fact", record_id: "x:1" }],
        },
      ],
    };
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) =>
        cmd === "chatbot_load_transcript" ? saved : undefined,
      ),
    );
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    const copyBtn = await screen.findByTestId("chatbot-helper-copy-a");
    const resources = screen.getByTestId("chatbot-helper-resources");
    // DOM order within the turn: resources block precedes the copy.
    expect(
      resources.compareDocumentPosition(copyBtn) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

});

describe("ChatbotHelper — resources block labeled with the turn's own source (wiring)", () => {
  it("two turns from different sources each show their OWN selector-matching source label", async () => {
    const rows = [
      { run_id: "run-A", label: "journal.txt — May 16, 01:14",
        store_path: "/s/a", bound: false },
      { run_id: "run-B", label: "work.txt — May 16, 14:11",
        store_path: "/s/b", bound: false },
    ];
    const resolveRunName = (id) =>
      id === "run-A" ? "my-journal" : id === "run-B" ? "cp9y" : "";
    const saved = {
      open: true,
      turns: [
        {
          id: 1, q: "about A", a: "Per [1]",
          status: "done", retrieving: false, retrievingQuery: "",
          runId: "run-A",
          resources: [{ index: 1, kind: "fact", record_id: "a:1" }],
        },
        {
          id: 2, q: "about B", a: "Per [1]",
          status: "done", retrieving: false, retrievingQuery: "",
          runId: "run-B",
          resources: [{ index: 1, kind: "fact", record_id: "b:1" }],
        },
      ],
    };
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) =>
        cmd === "chatbot_load_transcript"
          ? saved
          : cmd === "chatbot_list_runs"
            ? rows
            : undefined,
      ),
    );
    render(<ChatbotHelper resolveRunName={resolveRunName} />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    await screen.findByText("about B");

    const labels = await waitFor(() => {
      const els = screen.getAllByTestId("chatbot-helper-refs-source");
      expect(els).toHaveLength(2);
      return els;
    });
    // Each block names its OWN source, by the exact string the
    // "talking about" selector uses for that run — and they differ.
    expect(labels[0].textContent).toBe("journal.txt — May 16, 01:14 · my-journal");
    expect(labels[1].textContent).toBe("work.txt — May 16, 14:11 · cp9y");
    // The shared run dropdown lists the same source strings (one
    // labelForRun, so a message's source can't drift from the picker).
    fireEvent.click(screen.getByTestId("chatbot-helper-runtoggle"));
    const menu = screen.getByTestId("chatbot-helper-runmenu");
    expect(menu.textContent).toContain("journal.txt — May 16, 01:14 · my-journal");
    expect(menu.textContent).toContain("work.txt — May 16, 14:11 · cp9y");
  });

  it("a legacy turn with no pinned run shows the resources but no source label", async () => {
    const saved = {
      open: true,
      turns: [
        {
          id: 1, q: "legacy", a: "Per [1]", status: "done",
          retrieving: false, retrievingQuery: "",
          resources: [{ index: 1, kind: "fact", record_id: "x:1" }],
        },
      ],
    };
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) =>
        cmd === "chatbot_load_transcript" ? saved : undefined,
      ),
    );
    render(<ChatbotHelper resolveRunName={() => ""} />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    await screen.findByTestId("chatbot-helper-resources");
    expect(screen.queryByTestId("chatbot-helper-refs-source")).toBeNull();
  });
});

describe("ChatbotHelper — conversation/thread picker (#565)", () => {
  // Two conversations + a per-id transcript map. Wiring-guard only:
  // proves the picker drives the #565 commands and reuses the app's
  // one confirm modal. Real render/restart/delete acceptance is the
  // packaged WebKit .app, never jsdom.
  // #568: identity is the immutable ISO-Z prefix (`id`/`created`);
  // `label` is free-text (renamable); `name` is the current dir.
  // Model A (#574): the dir IS `<iso>-<short_id>` (the perma-id is the
  // name); a human name is the cosmetic `alias`. The picker titles a
  // row with `alias || short_id`.
  const C1 = {
    id: "2026-05-01T10-00-00Z",
    created: "2026-05-01T10-00-00Z",
    label: "conversation-1",
    name: "2026-05-01T10-00-00Z-qx7m",
    short_id: "qx7m",
    alias: "conversation-1",
  };
  const C2 = {
    id: "2026-05-02T11-00-00Z",
    created: "2026-05-02T11-00-00Z",
    label: "conversation-2",
    name: "2026-05-02T11-00-00Z-k4tp",
    short_id: "k4tp",
    alias: "conversation-2",
  };

  function multiConvoInvoke(transcripts, list = [C2, C1]) {
    return async (cmd, args) => {
      if (cmd === "chatbot_list_conversations") return list;
      if (cmd === "chatbot_load_conversation")
        return transcripts[args.id] || { open: true, turns: [] };
      return undefined;
    };
  }

  it("renders the picker under the talking-about runbar, most-recent active", async () => {
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(multiConvoInvoke({}));
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    const toggle = await screen.findByTestId(
      "chatbot-helper-threadtoggle",
    );
    // Active = the newest conversation (C2), zero persisted pointer.
    // Toggle shows the human label + the LAST-ACTIVITY date (no
    // verbose creation timestamp, not the raw ISO dir name).
    expect(toggle.textContent).toContain(C2.label);
    expect(toggle.textContent).toContain(convoLastDate(C2));
    expect(toggle.textContent).not.toContain(C2.name);
    // The thread bar sits AFTER the talking-about runbar in the DOM.
    const panel = screen.getByTestId("chatbot-helper-panel");
    const runbar = panel.querySelector(".chatbot-helper-runbar");
    const threadbar = panel.querySelector(".chatbot-helper-threadbar");
    expect(
      runbar.compareDocumentPosition(threadbar) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    // The menu lists "+ New conversation" at the TOP, then both rows
    // each with a red-X delete control.
    fireEvent.click(toggle);
    const menu = await screen.findByTestId("chatbot-helper-threadmenu");
    expect(menu.firstChild.textContent).toBe("+ New conversation");
    expect(
      screen.getByTestId(`chatbot-helper-threaddel-${C1.id}`),
    ).toBeTruthy();
    expect(
      screen.getByTestId(`chatbot-helper-threaddel-${C2.id}`),
    ).toBeTruthy();
  });

  it("renders an unrenamed conversation's date client-side in local tz, not the UTC creation date baked into display_label (#984)", async () => {
    // The sibling of the run-label skew: `derive_display_label` used to
    // bake `Conversation N · <Mon Day>` from the UTC ISO prefix. For a
    // NYC (UTC−4) user a convo created 02:00 UTC is the PREVIOUS local
    // day, so the picker disagreed with the client-rendered local date.
    // Now `display_label` is the ordinal only and the date rides on the
    // client-side `convoLastDate` (local).
    const unrenamed = {
      id: "2026-05-16T02-00-00Z",
      created: "2026-05-16T02-00-00Z",
      label: "ab3k",
      name: "2026-05-16T02-00-00Z-ab3k",
      short_id: "ab3k",
      alias: "", // unrenamed → title falls back to display_label
      display_label: "Conversation 1", // ordinal only, no baked date
      last_ts: null, // no turns yet → date falls back to creation
    };
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(
      multiConvoInvoke({}, [unrenamed]),
    );
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    const toggle = await screen.findByTestId("chatbot-helper-threadtoggle");
    // Ordinal title, no embedded server date.
    expect(toggle.textContent).toContain("Conversation 1");
    // The date is the LOCAL-tz render of the creation instant (02:00 UTC
    // → 10:00 PM the PREVIOUS day in NYC), NOT the UTC "May 16" the old
    // display_label baked in.
    expect(toggle.textContent).toContain(convoLastDate(unrenamed));
    expect(toggle.textContent).toContain("May 15, 10:00 PM");
    expect(toggle.textContent).not.toContain("May 16");
  });

  it("re-fetches the (activity-ordered) list every time the picker opens", async () => {
    // The reported bug: ordering is recomputed server-side at list
    // time, but the popover showed the stale mount-time order — a
    // just-used thread never floated to the top. Opening the picker
    // must re-query so the live order is reflected.
    let order = [C1, C2]; // mount-time order
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(async (cmd) => {
      if (cmd === "chatbot_list_conversations") return order;
      if (cmd === "chatbot_load_conversation")
        return { open: true, turns: [] };
      return undefined;
    });
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    const toggle = await screen.findByTestId(
      "chatbot-helper-threadtoggle",
    );
    const listCalls = () =>
      vi
        .mocked(invoke)
        .mock.calls.filter((c) => c[0] === "chatbot_list_conversations")
        .length;
    const atMount = listCalls();
    // Server now reports a different order (C2 became most-active).
    order = [C2, C1];
    fireEvent.click(toggle); // open
    await waitFor(() => expect(listCalls()).toBeGreaterThan(atMount));
    // The popover reflects the FRESH order, not the mount snapshot:
    // first row is now C2.
    await waitFor(() => {
      const rows = screen
        .getByTestId("chatbot-helper-threadmenu")
        .querySelectorAll('[data-testid^="chatbot-helper-threadrow-"]');
      expect(rows[0].getAttribute("data-testid")).toBe(
        `chatbot-helper-threadrow-${C2.id}`,
      );
    });
  });

  it("switching loads that conversation and re-scopes the active conversation", async () => {
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(
      multiConvoInvoke({
        [C1.id]: {
          open: true,
          turns: [
            {
              id: 1,
              q: "older thread q",
              a: "older thread a",
              status: "done",
              retrieving: false,
              retrievingQuery: "",
              runId: "run-old",
              resources: null,
            },
          ],
        },
      }),
    );
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    fireEvent.click(
      await screen.findByTestId("chatbot-helper-threadtoggle"),
    );
    fireEvent.click(screen.getByText(C1.label));
    await waitFor(() =>
      expect(screen.getByTestId("chatbot-helper-q").textContent).toBe(
        "older thread q",
      ),
    );
    // Telemetry/binding re-scoped to the switched-into thread, with the
    // gate-1 Q1 default = its most-recent turn's pinned run.
    expect(lastInvokeArgs("chatbot_set_active_conversation")).toEqual({
      id: C1.id,
      runId: "run-old",
    });
  });

  it("'+ New conversation' mints a fresh thread and shows it empty", async () => {
    // A freshly created conversation has no alias yet — it's named by
    // its 4-letter perma-id (Model A), which is what the picker shows.
    const NEW = {
      id: "2026-05-03T12-00-00Z",
      created: "2026-05-03T12-00-00Z",
      label: "wp3k",
      name: "2026-05-03T12-00-00Z-wp3k",
      short_id: "wp3k",
      alias: "",
    };
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(async (cmd) => {
      if (cmd === "chatbot_list_conversations") return [C2, C1];
      if (cmd === "chatbot_load_conversation")
        return {
          open: true,
          turns: [
            { id: 1, q: "prior", a: "prior a", status: "done",
              retrieving: false, retrievingQuery: "", resources: null },
          ],
        };
      if (cmd === "chatbot_new_conversation") return NEW;
      return undefined;
    });
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    await waitFor(() =>
      expect(screen.getByTestId("chatbot-helper-q").textContent).toBe(
        "prior",
      ),
    );
    fireEvent.click(screen.getByTestId("chatbot-helper-threadtoggle"));
    fireEvent.click(screen.getByTestId("chatbot-helper-threadnew"));
    await waitFor(() =>
      expect(
        vi
          .mocked(invoke)
          .mock.calls.some((c) => c[0] === "chatbot_new_conversation"),
      ).toBe(true),
    );
    // Fresh thread is active + empty; the prior turn is gone from view
    // (not destroyed — it stays its own selectable conversation).
    await waitFor(() =>
      expect(screen.queryByTestId("chatbot-helper-q")).toBeNull(),
    );
    expect(
      screen.getByTestId("chatbot-helper-threadtoggle").textContent,
    ).toContain(NEW.label);
  });

  it("red X routes through the app's ONE confirm modal, then deletes", async () => {
    const confirmCalls = [];
    // Stand-in for App's threaded confirm trigger (the real single
    // ConfirmDialog). Capturing it IS the no-duplicate-modal guard:
    // the panel must not render its own modal.
    const requestConfirm = vi.fn((opts) => confirmCalls.push(opts));
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(async (cmd) => {
      if (cmd === "chatbot_list_conversations") return [C2, C1];
      if (cmd === "chatbot_load_conversation")
        return { open: true, turns: [] };
      if (cmd === "chatbot_delete_conversation") return C1; // fallback
      return undefined;
    });
    render(<ChatbotHelper requestConfirm={requestConfirm} />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    fireEvent.click(
      await screen.findByTestId("chatbot-helper-threadtoggle"),
    );
    fireEvent.click(
      screen.getByTestId(`chatbot-helper-threaddel-${C2.id}`),
    );
    // No new modal built; the existing trigger was invoked with a
    // Delete/Cancel-shaped payload.
    expect(requestConfirm).toHaveBeenCalledTimes(1);
    expect(confirmCalls[0]).toMatchObject({
      title: "Delete conversation?",
      confirmLabel: "Yes, delete",
    });
    // Message identifies the conversation by its human label + the
    // last-activity date (not the raw ISO dir name).
    expect(confirmCalls[0].message).toContain(C2.label);
    expect(confirmCalls[0].message).toContain(convoLastDate(C2));
    // Nothing deleted until the user confirms (Cancel = no-op path).
    expect(
      vi
        .mocked(invoke)
        .mock.calls.some((c) => c[0] === "chatbot_delete_conversation"),
    ).toBe(false);
    // Confirm → backend deletes the dir + returns the fallback thread.
    await act(async () => {
      await confirmCalls[0].onConfirm();
    });
    expect(lastInvokeArgs("chatbot_delete_conversation")).toEqual({
      id: C2.id,
    });
  });

  it("inline rename keys on the immutable ISO id, sets a cosmetic alias", async () => {
    const renamed = {
      id: C2.id, // UNCHANGED — identity is the immutable ISO prefix
      created: C2.created,
      label: C2.short_id, // label = the perma-id (dir segment)
      name: C2.name, // dir NEVER moves on rename (Model A)
      short_id: C2.short_id, // perma-id is rename-proof (#574)
      alias: "Tax notes", // the cosmetic rename
    };
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(async (cmd) => {
      if (cmd === "chatbot_list_conversations") return [C2, C1];
      if (cmd === "chatbot_load_conversation")
        return { open: true, turns: [] };
      if (cmd === "chatbot_rename_conversation") return renamed;
      return undefined;
    });
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    fireEvent.click(
      await screen.findByTestId("chatbot-helper-threadtoggle"),
    );
    fireEvent.click(
      screen.getByTestId(`chatbot-helper-threadedit-${C2.id}`),
    );
    const input = screen.getByTestId(
      `chatbot-helper-threadrename-${C2.id}`,
    );
    fireEvent.change(input, { target: { value: "Tax notes" } });
    fireEvent.keyDown(input, { key: "Enter" });
    // Renames by the IMMUTABLE id (not the dir name) + the new label.
    await waitFor(() =>
      expect(lastInvokeArgs("chatbot_rename_conversation")).toEqual({
        id: C2.id,
        label: "Tax notes",
      }),
    );
    // The row now shows the new label; id is unchanged so refs hold.
    await waitFor(() =>
      expect(
        screen.getByTestId(`chatbot-helper-threadrow-${C2.id}`)
          .textContent,
      ).toContain("Tax notes"),
    );
    // Binding (#574): rename is purely cosmetic — the 4-letter
    // #<short_id> stays visible on the renamed row's line 1.
    const renamedRow = screen.getByTestId(
      `chatbot-helper-threadrow-${C2.id}`,
    );
    const chip = renamedRow.querySelector(
      ".chatbot-helper-mdrow-id",
    );
    expect(chip).toBeTruthy();
    expect(chip.textContent).toBe(`#${C2.short_id}`);
  });
});

describe("convoLastDate", () => {
  it("uses last_ts (last message); year shown only for a non-current year", () => {
    const yr = new Date().getFullYear();
    // Same calendar year as "now" → year is dropped.
    const thisYear = convoLastDate({
      last_ts: Date.parse(`${yr}-07-16T12:00:00Z`),
    });
    expect(thisYear).toMatch(/Jul/);
    expect(thisYear).toMatch(/\d:\d{2}/); // a clock time present
    expect(thisYear).not.toMatch(new RegExp(`${yr}`)); // no year, same year
    // A different year → the year is kept so it's unambiguous.
    const otherYear = convoLastDate({
      last_ts: Date.parse(`${yr - 3}-07-16T12:00:00Z`),
    });
    expect(otherYear).toMatch(new RegExp(`${yr - 3}`));
  });
  it("falls back to the creation ISO when there is no message", () => {
    const yr = new Date().getFullYear();
    const out = convoLastDate({ created: `${yr}-05-02T13-30-11Z` });
    expect(out).toMatch(/May/);
    expect(out).toMatch(/\d:\d{2}/);
    expect(out).not.toMatch(new RegExp(`${yr}`)); // current year dropped
  });
  it("empty for missing/garbage input", () => {
    expect(convoLastDate(null)).toBe("");
    expect(convoLastDate({})).toBe("");
    expect(convoLastDate({ created: "not-a-date" })).toBe("");
  });
});

describe("msgTime — per-message timestamp", () => {
  it("formats epoch-millis ts as compact MMM D, h:mm", () => {
    const out = msgTime(Date.parse("2026-07-16T14:34:00Z"));
    expect(out).toMatch(/Jul/);
    expect(out).toMatch(/\d:\d{2}/); // a clock time
    expect(out).not.toMatch(/2026/); // compact: no year
  });
  it("empty for a turn with no ts (legacy) / bad input", () => {
    expect(msgTime(undefined)).toBe("");
    expect(msgTime(null)).toBe("");
    expect(msgTime(NaN)).toBe("");
  });
});

describe("ChatbotHelper — per-message timestamp next to copy (wiring)", () => {
  it("renders the turn's ts beside the reply copy button; omits when absent", async () => {
    const saved = {
      open: true,
      turns: [
        { id: 1, q: "q1", a: "a1 with ts", status: "done",
          retrieving: false, retrievingQuery: "", resources: null,
          ts: Date.parse("2026-07-16T14:34:00Z") },
        { id: 2, q: "q2", a: "a2 no ts (legacy)", status: "done",
          retrieving: false, retrievingQuery: "", resources: null },
      ],
    };
    vi.mocked(listen).mockResolvedValue(() => {});
    vi.mocked(invoke).mockImplementation(
      convoMock(async (cmd) =>
        cmd === "chatbot_load_transcript" ? saved : undefined,
      ),
    );
    render(<ChatbotHelper />);
    fireEvent.click(screen.getByTestId("chatbot-helper-pill"));
    await screen.findByText("a1 with ts");
    // Both replies have a copy button; only the ts-bearing turn shows
    // a timestamp (next to copy), the legacy one omits it (no bogus).
    expect(screen.getAllByTestId("chatbot-helper-copy-a")).toHaveLength(2);
    const times = screen.getAllByTestId("chatbot-helper-msgtime");
    expect(times).toHaveLength(1);
    expect(times[0].textContent).toMatch(/Jul .*\d:\d{2}/);
  });
});


// ── prepped-for + lookup-log expand affordance (slice 2 stage 3) ────────────

describe("ChatbotHelper — post-turn lookup-log affordance", () => {
  it("accumulates each retrieving event's describe and renders the toggle once done", async () => {
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    await user.type(screen.getByTestId("chatbot-helper-input"), "q");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    // Two retrieval calls in this turn (e.g. ReAct loop hops 1 + 2).
    act(() => {
      handler({ payload: { event: "chatbot_retrieving", query: "first" } });
    });
    act(() => {
      handler({ payload: { event: "chatbot_retrieving", query: "second" } });
    });
    // While the turn is in-flight the toggle MUST NOT render — only
    // the live status line ("Searching your data for second…").
    expect(screen.queryByTestId("chatbot-helper-lookup-log-toggle")).toBeNull();
    expect(screen.getByTestId("chatbot-helper-retrieving").textContent)
      .toContain("second");
    act(() => {
      handler({ payload: { event: "chatbot_done", resources: [] } });
    });
    // Toggle is collapsed by default — list not rendered until clicked.
    const toggle = screen.getByTestId("chatbot-helper-lookup-log-toggle");
    expect(toggle.textContent).toContain("prepped for");
    expect(toggle.textContent).toContain("[+]");
    expect(screen.queryByTestId("chatbot-helper-lookup-log-list")).toBeNull();
    await user.click(toggle);
    // Click reveals the list with one entry per dispatched lookup, in
    // dispatch order. Empty strings (validate failures) are filtered.
    const items = screen.getByTestId("chatbot-helper-lookup-log-list")
      .querySelectorAll("li");
    expect(items).toHaveLength(2);
    expect(items[0].textContent).toBe("first");
    expect(items[1].textContent).toBe("second");
    expect(
      screen.getByTestId("chatbot-helper-lookup-log-toggle").textContent,
    ).toContain("[−]");
  });

  it("does NOT render the toggle when no retrieval fired (pure conversation turn)", async () => {
    const user = userEvent.setup();
    const { handler } = await renderAndGetHandler();
    await user.type(screen.getByTestId("chatbot-helper-input"), "q");
    await user.click(screen.getByTestId("chatbot-helper-send"));
    act(() => {
      handler({ payload: { event: "chatbot_chunk", delta: "hello" } });
    });
    act(() => {
      handler({ payload: { event: "chatbot_done", resources: null } });
    });
    expect(screen.queryByTestId("chatbot-helper-lookup-log-toggle")).toBeNull();
  });
});

describe("ChatbotHelper — attestation is non-blocking for send", () => {
  // Attestation no longer gates the send button. A failed / in-flight
  // attestation must NOT disable send in any mode — the real per-
  // connection guarantee is enforced at the transport layer, and the
  // attestation panel is a visibility surface only.
  async function typeDraft() {
    const user = userEvent.setup();
    await user.type(screen.getByTestId("chatbot-helper-input"), "hi");
    return user;
  }

  it("keeps send enabled in TEE mode when attestation has failed", async () => {
    await renderAndGetHandler({
      mode: "tee",
      attestation: { ok: false, transient: false, error: "measurement mismatch" },
    });
    await typeDraft();
    expect(screen.getByTestId("chatbot-helper-send").disabled).toBe(false);
  });

  it("keeps send enabled in TEE mode while attestation is re-checking", async () => {
    await renderAndGetHandler({
      mode: "tee",
      attestation: { ok: true },
      attestationChecking: true,
    });
    await typeDraft();
    expect(screen.getByTestId("chatbot-helper-send").disabled).toBe(false);
  });

  it("invokes chatbot on Enter even when attestation has failed", async () => {
    await renderAndGetHandler({
      mode: "tee",
      attestation: { ok: false, transient: false, error: "x" },
    });
    const user = await typeDraft();
    await user.keyboard("{Enter}");
    expect(
      vi.mocked(invoke).mock.calls.some((c) => c[0] === "chatbot"),
    ).toBe(true);
  });

  it("keeps send enabled in TEE mode when attestation is verified", async () => {
    await renderAndGetHandler({ mode: "tee", attestation: { ok: true } });
    await typeDraft();
    expect(screen.getByTestId("chatbot-helper-send").disabled).toBe(false);
  });

  it("keeps send enabled in local mode even when attestation has failed", async () => {
    await renderAndGetHandler({
      mode: "local",
      attestation: { ok: false, transient: false, error: "x" },
    });
    await typeDraft();
    expect(screen.getByTestId("chatbot-helper-send").disabled).toBe(false);
  });
});
