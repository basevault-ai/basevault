import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { CopyButton, copyToClipboard } from "./CopyButton";
import { prettyDateTime, prettyDate } from "./dateFormat";
import baseVaultLogo from "./assets/basevault-logo.svg";

// ── chatbot helper — floating bottom-right conversational assistant ──────────────
//
// A chatbot the user talks with about their life and ideas. It
// converses and reasons freely; when the conversation needs facts from
// the user's own processed data it looks them up and grounds that part
// of the reply with numbered references. Collapsed state is a small
// pill in the corner; expanded state opens a fixed-size chat panel with
// a scrolling multi-turn transcript and a text input.
//
// Stays mounted at App-root level so all views share one instance.
// Non-modal: the panel is `position: fixed` above page chrome but the
// rest of the app stays interactive behind it.
//
// Multi-conversation (#565): each thread is its own directory under
// the state dir; its transcript is persisted via
// `chatbot_save_conversation` and rehydrated on mount via
// `chatbot_list_conversations` + `chatbot_load_conversation` (the
// directory listing is the source of truth — no manifest). It survives
// a window-destroy/reopen (Cmd/Ctrl+W destroys the WKWebView; the
// process stays alive headless for background runs, and a reopen
// rebuilds the window fresh) AND a full reload. Prior turns are sent
// back to the sidecar so follow-ups resolve against the conversation;
// each turn keeps its own #559 pinned run so citations resolve against
// the run that produced them. References render only on turns the
// assistant actually grounded — a plain conversational or
// "not in your data" reply carries none.

// Turn status:
//   streaming — Send fired for this turn; deltas arriving.
//   done      — reply finished.
//   error     — chatbot_error from sidecar, or transport error.
const STATUS = {
  streaming: "streaming",
  done: "done",
  error: "error",
};

// One grounded source chunk: a clickable row that opens the cited item
// in the main window, scrolled-to and transient-highlighted via the
// same primitive wikilink and Run Details jumps use. Opening the real
// record (not an inline excerpt) lets the user see it in full context
// with its surrounding data and the back/forward history. The row
// shows a resolved `kind · topic · title` label (not the opaque
// record_id) so a citation is legible and a mis-navigation is visible;
// before resolution lands it falls back to kind + record_id.
// `date` is the date-only (no time) of the cited record/source — the
// turn's pinned-run date, the day this corpus was processed; uniform
// across a reference list and always present (a per-fact `occurred_at`
// is mostly null and patterns/insights/actions carry no date, so a
// mixed list would be inconsistent). "" → the line is omitted.
function ResourceItem({ resource, onOpen, date }) {
  const rawLabel = resourceLabel(resource);
  // Replace ' · ' with NBSP·NBSP so the middot never leads or trails a
  // wrapped line. The copy text (buildReplyCopyText) reads resourceLabel
  // directly and keeps plain spaces — only the rendered label uses NBSP.
  const displayLabel = rawLabel.replace(/ · /g, " · ");
  return (
    <li
      className="chatbot-helper-resource"
      data-testid={`chatbot-helper-resource-${resource.index}`}
    >
      <button
        type="button"
        className="chatbot-helper-resource-head"
        onClick={onOpen}
        data-testid={`chatbot-helper-resource-open-${resource.index}`}
      >
        <span className="chatbot-helper-resource-line1">
          <span className="chatbot-helper-ref-index">[{resource.index}]</span>
          <span className="chatbot-helper-ref-label">{displayLabel}</span>
        </span>
        {date && (
          <span
            className="chatbot-helper-ref-date"
            data-testid={`chatbot-helper-resource-date-${resource.index}`}
          >
            {date}
          </span>
        )}
      </button>
    </li>
  );
}

// One shared 2-line meta row, the visual primitive BOTH pickers and
// their collapsed toggles render so they cannot drift (req #3/#4/#5):
//   line 1 — title, left-aligned (run: labelForRun; conversation:
//            alias-or-descriptive-label).
//   line 2 — greyed `date · #id` (run: run date · #run short_id;
//            conversation: last-activity date · #conv short_id).
// `as` is "button" for a selectable menu row, "span" for the
// non-interactive collapsed display inside the toggle (the toggle
// itself is the button — a nested button would be invalid).
function MetaRow({
  as = "button",
  title,
  date,
  shortId,
  active = false,
  onClick,
  titleAttr,
  testId,
  className = "",
}) {
  const Tag = as;
  const props =
    as === "button"
      ? {
          type: "button",
          onClick,
          role: "option",
          "aria-selected": active,
          title: titleAttr,
          "data-testid": testId,
        }
      : { "data-testid": testId };
  return (
    <Tag className={`chatbot-helper-mdrow-name ${className}`} {...props}>
      <span className="chatbot-helper-mdrow-title">{title}</span>
      <span className="chatbot-helper-mdrow-sub">
        {date && (
          <span className="chatbot-helper-mdrow-date">{date}</span>
        )}
        {shortId && (
          <span
            className="chatbot-helper-mdrow-id"
            title="Perma-id — quote these 4 letters to find this in logs / diagnostics"
          >
            #{shortId}
          </span>
        )}
      </span>
    </Tag>
  );
}

// The shared dropdown shell consumed by BOTH the Source-run and the
// Conversation pickers (req #2/#5) — one control, one look. Controlled
// open state (the parent closes it after a pick / switch / new and
// refreshes its list on open); owns only close-on-outside-click +
// Escape. The collapsed toggle shows the SAME 2-line MetaRow as the
// menu (req #4). `affordances`/`header` are the conversation-only
// slots (per-row ✎/✕, "+ New"); the run variant passes neither, so
// the two can't drift while keeping the conversation extras.
function MetaDropdown({
  barLabel,
  barTestId,
  barClass = "",
  tidPrefix,
  disabled,
  open,
  onToggle,
  onClose,
  selected,
  placeholder,
  children,
}) {
  const rootRef = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const onDocClick = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) onClose();
    };
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);
  return (
    <div
      className={`chatbot-helper-mdbar ${barClass}`}
      data-testid={barTestId}
    >
      <span className="chatbot-helper-mdbar-label">{barLabel}</span>
      <div className="chatbot-helper-mdpicker" ref={rootRef}>
        <button
          type="button"
          className="chatbot-helper-mdtoggle"
          onClick={onToggle}
          disabled={disabled}
          aria-haspopup="listbox"
          aria-expanded={open}
          title={`Pick ${barLabel}`}
          data-testid={`${tidPrefix}toggle`}
        >
          {selected ? (
            <MetaRow
              as="span"
              title={selected.title}
              date={selected.date}
              shortId={selected.shortId}
              className="is-collapsed"
            />
          ) : (
            <span className="chatbot-helper-mdrow-title">{placeholder}</span>
          )}
        </button>
        {open && (
          <div
            className="chatbot-helper-mdmenu"
            role="listbox"
            data-testid={`${tidPrefix}menu`}
          >
            {children}
          </div>
        )}
      </div>
    </div>
  );
}

// A single message's timestamp for the transcript — compact local
// `MMM D, h:mm AM/PM` from the turn's #568-owned `ts` (epoch millis,
// stamped at turn creation). "" when the turn predates the field
// (legacy) so it's simply omitted rather than showing a bogus time.
export function msgTime(ts) {
  if (typeof ts !== "number") return "";
  return prettyDateTime(ts);
}

// The per-message copy affordance + the message's own timestamp.
// Rendered only under the assistant's reply (a user can already
// see/select what they typed; the reply is the thing worth copying).
// Borderless — a boxed button under every reply is too heavy for the
// transcript. `text` is the exact text the user sees (the renumbered
// answer); `ts` is the turn's creation timestamp shown next to copy.
function MessageCopy({ text, ts, copyTestId }) {
  const when = msgTime(ts);
  return (
    <div className="chatbot-helper-msgmeta">
      <CopyButton
        inline
        borderless
        onClick={() => copyToClipboard(text)}
        testId={copyTestId}
        label="Copy message"
      />
      {when && (
        <span
          className="chatbot-helper-msgtime"
          data-testid="chatbot-helper-msgtime"
        >
          {when}
        </span>
      )}
    </div>
  );
}

// Renumber the cited references contiguously for display. The model
// cites by the position of the row in the retrieved context block it
// was given (e.g. `[7]` = row 7 of 15), so the cited subset surfaces
// with sparse, arbitrary-looking numbers ([1][3][7][13]). This is a
// pure relabel: distinct cited indices are mapped to 1..K in
// first-appearance order, and BOTH the answer's inline `[N]` markers
// and the resource list go through the same map so they stay aligned.
// Nothing is dropped — over-5 citations are kept (the ≤5 limit is the
// model's job, not a post-hoc cut). Resources whose index never
// appears in the prose (shouldn't happen — the list is derived from
// the answer's citations) are appended stably so the map stays total.
//
// The map is built from the answer text alone (not the resource list),
// so it holds mid-stream — before the resource list exists — and is
// stable as the answer grows (an index's slot is fixed at its first
// appearance). That lets the streaming guard below withhold a
// partial trailing `[`/`[<digits>` until it either closes into a
// remappable `[N]` or proves it is not a citation (a non-digit after
// `[`, or >6 digits, stops the digit run — the fragment is shown
// as-is), so a raw context-row number never flashes before the remap.
export function renumberCitations(answer, resources) {
  const text = answer || "";
  const list = Array.isArray(resources) ? resources : [];
  const map = new Map();
  let next = 1;
  // Integer-bracket citations ``[N]`` indexing into the per-turn
  // CONTEXT block. Negative-lookbehind keeps programming-syntax
  // brackets like ``arr[1]`` out of the citation pool — citations
  // follow non-letter context (space, punctuation) in real prose,
  // never another letter.
  const scan = /(?<![a-zA-Z])\[(\d+)\]/g;
  let m;
  while ((m = scan.exec(text)) !== null) {
    const n = Number(m[1]);
    if (Number.isFinite(n) && !map.has(n)) map.set(n, next++);
  }
  for (const r of [...list].sort((a, b) => a.index - b.index)) {
    if (!map.has(r.index)) map.set(r.index, next++);
  }
  const renumbered = text.replace(
    /(?<![a-zA-Z])\[(\d+)\]/g, (tok, d) => {
      const n = Number(d);
      // First-appearance contiguous remap: model's raw context-row
      // numbers ([7], [3], [13]) become user-visible [1], [2], [3]
      // in the chat in the order they first appear in the answer.
      return map.has(n) ? `[${map.get(n)}]` : tok;
    });
  return {
    // Hide a partial trailing ``[``/``[<digits>`` so a streaming
    // token mid-bracket doesn't flash a raw context label before
    // remap. Up to 6 digits covers any realistic accumulator size;
    // anything else passes through (the user's typed code shouldn't
    // get eaten by the streaming guard).
    answer: renumbered.replace(/(?<![a-zA-Z])\[(\d{0,6})$/, ""),
    resources: list
      .map((r) => ({ ...r, index: map.get(r.index) }))
      .sort((a, b) => a.index - b.index),
  };
}

// The single source of the human-readable run label. The "talking
// about" selector and each message's resources-block source label BOTH
// go through this so they can never drift: a message must show its
// source under the exact name the selector uses for it.
//
// `runId` is the run a thing is bound to — the live `boundRun` for the
// selector, a turn's OWN pinned `runId` (per-turn pinning) for a
// message's resources block. The label is the run-options entry's
// subject·date `label` plus the run's name (rename → 4-letter id) when
// there is one, exactly as the selector renders an <option>. When the
// run isn't in the options list (a legacy turn pinned to a run no
// longer listed, or the list hasn't loaded) it falls back to the
// resolved name, else the raw id — the same degradation the no-list
// selector branch already uses. Returns "" only when there is nothing
// at all to show (no run); callers omit the label in that case.
export function labelForRun(runId, runOptions, resolveRunName) {
  if (!runId) return "";
  const opts = Array.isArray(runOptions) ? runOptions : [];
  const o = opts.find((x) => x.run_id === runId);
  const name = resolveRunName ? resolveRunName(runId) : "";
  if (o) return name ? `${o.label} · ${name}` : o.label;
  return name || runId;
}

// The date-only (no time) of a turn's cited records — the turn's
// pinned-run creation date, i.e. the day that corpus was processed.
// One date for the whole reference list (uniform + always present:
// per-record dates are mostly absent). "" when the run isn't in the
// options list (legacy turn / list not loaded) so callers omit it.
export function runRefDate(runId, runOptions) {
  if (!runId) return "";
  const o = (Array.isArray(runOptions) ? runOptions : []).find(
    (x) => x.run_id === runId,
  );
  return o?.created_at ? prettyDate(o.created_at) : "";
}

// Render the renumbered answer with its `[N]` markers clickable. This
// does NOT re-scan or re-map the raw stream — it consumes `view.answer`
// (already remapped + partial-token-guarded by `renumberCitations`,
// the in-stream interception path) and `view.resources` (same shared
// map), so the clickable token can never disagree with the renumber.
// A `[N]` whose number matches a resource becomes a button that fires
// the SAME `onRef(resource)` the resource chip's onClick uses — so the
// in-body click and the chip click run the identical ref-click
// navigation/scroll/highlight path (#553/#550) against the turn's own
// pinned run (#559). `[N]` with no matching resource stays inert text.
// `textContent` is unchanged (the token text is still `[N]`), so the
// copied/visible answer reads identically.
export function renderAnswerWithRefs(answer, resources, onRef, refDate) {
  const text = answer || "";
  const byIndex = new Map(
    (Array.isArray(resources) ? resources : []).map((r) => [r.index, r]),
  );
  const out = [];
  // Integer-bracket citations ``[N]``. Negative-lookbehind keeps
  // programming-syntax brackets like ``arr[1]`` out.
  const re = /(?<![a-zA-Z])\[(\d+)\]/g;
  let last = 0;
  let m;
  let k = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const res = byIndex.get(Number(m[1]));
    if (res) {
      out.push(
        <button
          type="button"
          key={`cite-${k}-${m.index}`}
          className="chatbot-helper-cite"
          onClick={() => onRef(res)}
          title={refDate ? `Open source · ${refDate}` : "Open source"}
          data-testid={`chatbot-helper-cite-${res.index}`}
        >
          {m[0]}
        </button>,
      );
    } else {
      out.push(m[0]);
    }
    last = m.index + m[0].length;
    k += 1;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

// One reference's display label — the SAME string ResourceItem
// renders on screen, so a copied citation reads exactly like the one
// the user clicked.
function resourceLabel(r) {
  return r.label || `${r.kind} ${r.record_id}`;
}

// The full copyable text of a reply: the answer the user sees, then —
// when the reply is grounded — the references block exactly as
// rendered (heading, this message's source, and the numbered list).
// The whole point of copy is that the `[N]` markers in the prose stay
// meaningful, so the reference list travels with them. A non-grounded
// reply (no resources) copies just the answer. `resources` is the
// renumbered/sorted view list; `sourceLabel` is this turn's own
// source (may be "" → omitted, same as on screen). `refDate` is the
// date-only on each reference (same string shown on screen), appended
// so a copied reference carries its date too; "" → omitted.
export function buildReplyCopyText(answer, resources, sourceLabel, refDate) {
  const body = (answer || "").trimEnd();
  const list = Array.isArray(resources) ? resources : [];
  if (list.length === 0) return body;
  const d = refDate ? ` · ${refDate}` : "";
  const lines = [body, "", "Resources from your data"];
  if (sourceLabel) lines.push(`${sourceLabel}${d}`);
  for (const r of list) {
    lines.push(`[${r.index}] ${resourceLabel(r)}${d}`);
  }
  return lines.join("\n");
}

// The conversation's LAST-ACTIVITY timestamp for the picker, via the
// ONE shared formatter (same as runs + attestation; year dropped when
// it's the current year). When choosing a conversation the relevant
// moment is its last message. Source: `meta.last_ts` (the last turn's
// epoch-millis `ts`); when the conversation has no messages
// (`last_ts` null) fall back to its creation date (the immutable
// ISO-Z `created`/`id`). The creation date still lives in the on-disk
// dir name; it just isn't shown on its own.
export function convoLastDate(meta) {
  if (!meta) return "";
  if (typeof meta.last_ts === "number") return prettyDateTime(meta.last_ts);
  const m = String(meta.created || "").match(
    /^(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})Z$/,
  );
  if (!m) return "";
  return prettyDateTime(`${m[1]}T${m[2]}:${m[3]}:${m[4]}Z`);
}

// `resolveRunName(runId)` maps a run_id to the same user-visible name
// the runs pane shows — the rename when one is set, else the 4-letter
// short_id (App owns the rename source; we never re-derive it here).
// Defaults to a no-op so the component renders standalone (tests) with
// the label-only behaviour.
export default function ChatbotHelper({
  resolveRunName = () => "",
  onOpenResource,
  resolveResource,
  // App's single confirm dialog, threaded down (no duplicate modal).
  // Defaults to a no-op so the component renders standalone in tests.
  requestConfirm = () => {},
  // Attestation is a non-blocking visibility signal surfaced by the
  // top-bar indicator, not a send gate — the real per-connection
  // guarantee is enforced at the transport layer. So the chat panel
  // takes no attestation / mode props.
}) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  // Conversation/thread picker (#565/#568). `conversations` is the
  // backend list (most-recently-active first — the directory listing
  // IS the source of truth, no manifest); `activeConvId` is the
  // selected conversation's IMMUTABLE ISO-Z id. Custom popover (a
  // native <select> can't carry per-row rename/delete).
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId] = useState(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  // The Source-run picker is now the SAME shared dropdown as the
  // conversation one (req #2/#5), so it has its own open state.
  const [runPickerOpen, setRunPickerOpen] = useState(false);
  // Inline rename: the conversation id being renamed (its immutable
  // ISO prefix) + the draft label. The id never changes on rename, so
  // activeConvId / refs / citations are unaffected.
  const [renamingId, setRenamingId] = useState(null);
  const [renameDraft, setRenameDraft] = useState("");
  // Live mirror of activeConvId for the persist effect / async flows
  // (the throttled write must target the CURRENT conversation even if
  // it was scheduled before a switch).
  const activeConvIdRef = useRef(null);
  // The conversation transcript. Each turn: { id, q, a, resources,
  // runId, ts, status, refused, error }. `runId` is the turn's own
  // pinned run; `ts` is the #568-owned creation timestamp (#567
  // consumes it). `refused` marks a no-corpus refusal turn whose `a`
  // is the deterministic refusal text — its assistant
  // slot is excluded when assembling the history fed into the next
  // turn, so the model never sees its own prior refusal text and
  // can't learn to parrot it as prose. New turns append; nothing is
  // ever wiped on send.
  const [turns, setTurns] = useState([]);
  // The corpus run the sidecar bound this session to (the resolved
  // vectors.db's run dir = run_id). null until the first `chatbot_bound`
  // arrives, or when no processed run exists. Drives which entry the
  // run selector shows as current.
  const [boundRun, setBoundRun] = useState(null);
  // The 10 most-recent runs with a non-empty store, for the selector
  // (#507). Each: { run_id, label, store_path, bound }. Refreshed on
  // panel open and whenever the bound run changes (a rebind = a fresh
  // session emitting a new `chatbot_bound`).
  const [runOptions, setRunOptions] = useState([]);
  // Monotonic counter scoping streaming events to the in-flight turn.
  // If a second query is sent (or the conversation is reset) before the
  // first finishes, deltas from the prior sidecar are dropped — defends
  // the case where the disable flips slower than React schedules.
  const queryIdRef = useRef(0);
  const activeQueryIdRef = useRef(0);
  const bodyRef = useRef(null);
  const inputRef = useRef(null);
  // The chatbot-event listener subscribes once (StrictMode-safe), so
  // its handler closes over the first render's state. Citation
  // resolution fires from that handler and needs the live bound run —
  // read it from a ref, not the frozen `boundRun` closure.
  const boundRunRef = useRef(null);
  // Whether the transcript is scrolled to (near) the bottom. Updated on
  // every scroll; gates the auto-pin so a re-render that isn't new
  // content — a selector rebind, run-list refresh — never yanks the
  // view down while the user has scrolled up reading scrollback.
  const atBottomRef = useRef(true);
  // Last seen turn count, so a brand-new turn (the user's own message)
  // still scrolls into view even if they were scrolled up.
  const prevTurnCountRef = useRef(0);
  // Flipped true once the on-mount rehydrate has resolved. Gates the
  // persist effect so the initial empty state — or a transient load
  // failure — never clobbers a good saved transcript with [].
  const hydratedRef = useRef(false);
  // Trailing-throttle handle + the latest {open,turns} it should write.
  // Token deltas on an in-flight turn coalesce into one write per
  // interval (no fsync-per-token) while still persisting the pending
  // user message + partial answer; the timeout reads the ref so it
  // flushes the LATEST state, not the stale one it was scheduled with.
  const persistTimerRef = useRef(null);
  const latestStateRef = useRef({ open: false, turns: [] });
  // Turn count at the last persist decision — a change means a
  // brand-new turn (the user's message) and forces an immediate write
  // so it reaches disk before the user can hit Cmd+W.
  const persistedTurnCountRef = useRef(0);
  // The sidecar turn_id (process-global monotonic, returned by the
  // `chatbot` command) of the generation whose events the listener
  // should accept. Every chatbot event echoes its turn_id; an event
  // whose turn_id doesn't match is a stale generation — dropped. This
  // is the structural fence for the resume path: a re-issue after a
  // cancel→respawn gets a STRICTLY HIGHER id than the generation it
  // replaces, so a still-draining pre-close generation (the duplicate
  // greeting) can never land on the resumed turn. null = unknown yet
  // (id in flight) → fall back to the queryId guard for that window.
  const expectedTurnIdRef = useRef(null);

  function onBodyScroll() {
    const pane = bodyRef.current;
    if (!pane) return;
    const dist = pane.scrollHeight - pane.scrollTop - pane.clientHeight;
    atBottomRef.current = dist < 60;
  }

  const streaming = turns.length > 0
    && turns[turns.length - 1].status === STATUS.streaming;

  // Append a delta / set fields on the last (in-flight) turn.
  function patchLastTurn(patch) {
    setTurns((prev) => {
      if (prev.length === 0) return prev;
      const next = prev.slice();
      const last = next[next.length - 1];
      next[next.length - 1] =
        typeof patch === "function" ? patch(last) : { ...last, ...patch };
      return next;
    });
  }

  // Toggle the per-turn "prepped for Xs [+]" lookup-log panel. Keyed
  // by turn id (NOT index) because resolveTurnResources resolves
  // asynchronously and the user may have sent more turns by the time
  // they click — looking up by id stays correct under that race.
  function toggleLookupLog(turnId) {
    setTurns((prev) =>
      prev.map((t) =>
        t.id === turnId
          ? { ...t, lookupLogExpanded: !t.lookupLogExpanded }
          : t,
      ),
    );
  }

  // Render the wall-clock duration of a finished turn in the most
  // readable unit that fits ("<1s", "Xs", "Xm Ys"). Used by the
  // post-turn "prepped for …" affordance.
  function formatTurnDuration(ms) {
    if (!Number.isFinite(ms) || ms <= 0) return "<1s";
    const totalSecs = Math.round(ms / 1000);
    if (totalSecs < 60) return `${Math.max(1, totalSecs)}s`;
    const mins = Math.floor(totalSecs / 60);
    const secs = totalSecs % 60;
    return secs ? `${mins}m ${secs}s` : `${mins}m`;
  }

  // Resolve a finished turn's raw resources ({index,kind,record_id})
  // into clickable, labelled citations: parent maps each to its real
  // file anchor + a `kind · topic · title` label. Resolved against the
  // turn's OWN pinned run (passed in), never the live bind — so a
  // restored turn's labels/anchors are correct on rehydrate with zero
  // dependency on the sidecar having re-emitted `chatbot_bound`, and a
  // turn about another run isn't mis-resolved when the selector moves.
  // Async (it reads that run's stage data), so it patches the turn by
  // id once resolved — the turn may no longer be last if the user sent
  // another message. A resolver failure falls back to the raw resource
  // so the row still renders.
  function resolveTurnResources(turnId, raw, runId) {
    if (!resolveResource || !Array.isArray(raw) || raw.length === 0) return;
    Promise.all(
      raw.map((r) =>
        Promise.resolve(resolveResource(r, runId))
          .then((res) => (res ? { ...r, ...res } : r))
          .catch(() => r),
      ),
    ).then((resolved) => {
      setTurns((prev) =>
        prev.map((t) =>
          t.id === turnId ? { ...t, resources: resolved } : t,
        ),
      );
    });
  }

  // Pull the most-recent-non-empty run list for the selector. Cheap
  // and infrequent (panel open + on rebind); a failure leaves the
  // prior list rather than blanking the control.
  function refreshRuns() {
    invoke("chatbot_list_runs")
      .then((rows) => setRunOptions(Array.isArray(rows) ? rows : []))
      .catch(() => {});
  }

  // Rebind the chat to a different run. The backend re-spawns the
  // sidecar bound to it (a fresh pick = a fresh session); the new
  // sidecar's `chatbot_bound` refreshes boundRun + the list. Set
  // boundRun optimistically so the control reflects the pick instantly
  // even while the re-spawn's first event is in flight.
  function selectRun(runId) {
    setRunPickerOpen(false);
    if (!runId || runId === boundRun) return;
    setBoundRun(runId);
    boundRunRef.current = runId;
    invoke("chatbot_select_run", { runId }).catch(() => {});
  }

  // Toggle the Source-run dropdown; refresh the list on the way OPEN
  // (same freshness contract as the conversation picker).
  function toggleRunPicker() {
    setRunPickerOpen((v) => {
      const next = !v;
      if (next) refreshRuns();
      return next;
    });
  }

  // Refresh the run list when the panel opens so the selector is
  // populated before the first message (and current after a reopen).
  useEffect(() => {
    if (open) refreshRuns();
  }, [open]);

  // Refresh the run list when Rust observes a pipeline completion
  // (#780). The chatbot sidecar's `chatbot_bound` event already
  // refreshes the list — but only when a sidecar is alive. If the
  // user opened the chat panel BEFORE any ingest finished and
  // hasn't typed a message yet, no sidecar exists, the Rust
  // run-available stdin push is a no-op, and the dropdown would
  // sit stale until the next user action. `runs-changed` fires
  // directly from Rust to React (independent of any sidecar), so
  // the selector reflects the just-finished run the instant ingest
  // completes whether or not a sidecar is up.
  useEffect(() => {
    let unlisten;
    let cancelled = false;
    listen("runs-changed", () => refreshRuns()).then((u) => {
      if (cancelled) {
        u();
        return;
      }
      unlisten = u;
    });
    return () => {
      cancelled = true;
      if (unlisten) unlisten();
    };
  }, []);

  // Normalize a #453-shaped { open, turns } payload (one CONVERSATION's
  // transcript now) into the in-memory state, optionally resuming a
  // turn that was streaming when the window was destroyed. Shared by
  // the mount rehydrate and by a thread switch — the only difference is
  // `resume`/`restorePanel` (a switch within a live session must NOT
  // re-issue or touch the panel; an app-restart rehydrate does both).
  function applyLoadedTranscript(saved, { resume, restorePanel }) {
    // A legacy / empty / missing store reads back as a bare [] (the
    // Rust default) — treat that as turns with the panel closed.
    const data =
      saved && !Array.isArray(saved)
        ? saved
        : { open: false, turns: Array.isArray(saved) ? saved : [] };
    const rows = Array.isArray(data.turns) ? data.turns : [];
    // Only the LAST turn can be in-flight. If it was persisted
    // `streaming`, the window was closed mid-generation: the sidecar (a
    // persistent warm process kept alive by prevent_exit, NOT cancelled
    // by a window close) finished the answer, but the completing events
    // were emitted into the destroyed webview and lost — so the saved
    // answer is a stale partial (or empty). Freezing it to `done` is
    // the "lost in-flight reply" defect. On an app-restart rehydrate we
    // RESUME it (re-ask the still-warm sidecar). On a thread SWITCH the
    // sidecar was just re-spawned for THIS conversation's telemetry, so
    // there is nothing warm to resume — normalize the stale partial to
    // done instead of re-issuing (which would double-answer). Earlier
    // streaming rows (shouldn't exist — only the last can be in-flight)
    // are defensively normalized either way.
    const lastIdx = rows.length - 1;
    const resumeIdx =
      resume && lastIdx >= 0 && rows[lastIdx].status === STATUS.streaming
        ? lastIdx
        : -1;
    const restored = rows.map((t, i) => {
      if (t.status !== STATUS.streaming) return t;
      if (i === resumeIdx) {
        // Reset to a clean in-flight turn — discard the stale partial;
        // the resumed generation produces the full answer, attached to
        // THIS turn (never floated after the prior answer).
        return {
          ...t,
          a: "",
          status: STATUS.streaming,
          retrieving: false,
          retrievingQuery: "",
          resources: null,
          // #551: the resume re-emits chatbot_thinking; clear any
          // persisted in-flight "Thinking…" so it isn't shown twice.
          thinking: false,
        };
      }
      // #551: a normalized-to-done turn must not keep a stale
      // "Thinking…" flag (the event that would clear it was lost).
      return { ...t, status: STATUS.done, retrieving: false, thinking: false };
    });
    // Always replace the transcript (a switch from a non-empty thread
    // to an empty one must CLEAR, not keep the prior thread's turns).
    setTurns(restored);
    const maxId = restored.reduce(
      (m, t) => (typeof t.id === "number" && t.id > m ? t.id : m),
      0,
    );
    queryIdRef.current = maxId;
    // Drop any event still draining from a prior generation / the
    // pre-switch sidecar: holding activeQueryIdRef at 0 (≠ queryIdRef)
    // makes the listener guard reject it instead of patching it onto a
    // restored turn. A new user send re-arms both refs. The resume
    // path below re-arms activeQueryIdRef deliberately.
    activeQueryIdRef.current = 0;
    expectedTurnIdRef.current = null;
    if (restored.length > 0) {
      // Re-resolve every restored turn's citations against its OWN
      // pinned run — immediately, before any sidecar event. Pre-#559,
      // restored refs were resolved/clicked against the live bind,
      // which is null right after a restart and a DIFFERENT run after a
      // switch, so old refs were dead/mis-targeted. Each turn carries
      // its #559 run_id, so labels + anchors recompute and clicks
      // navigate correctly with ZERO new queries. A legacy turn with
      // no runId passes null (the resolver falls back to the raw
      // kind+record_id label rather than mis-resolving).
      restored.forEach((t) => {
        if (Array.isArray(t.resources) && t.resources.length > 0) {
          resolveTurnResources(t.id, t.resources, t.runId || null);
        }
      });
      if (resumeIdx >= 0) {
        // History = every turn BEFORE the interrupted one, the same
        // user/assistant reconstruction sendQuery uses.
        const history = restored.slice(0, resumeIdx).flatMap((t) => {
          const msgs = [{ role: "user", content: t.q }];
          // A refused turn's `a` is the deterministic refusal chunk —
          // exclude it from history so the model never sees its own
          // prior refusal text and learns to mimic it as prose.
          if (t.a && !t.refused) {
            const asst = { role: "assistant", content: t.a };
            const refs = Array.isArray(t.resources) ? t.resources : [];
            if (refs.length > 0) {
              asst.cited_refs = refs.map((r) => ({
                kind: r.kind, record_id: r.record_id,
              }));
            }
            msgs.push(asst);
          }
          return msgs;
        });
        // The pre-close generation was NOT cancelled by the window
        // close (the sidecar is a persistent warm process kept alive by
        // prevent_exit — that is the root cause). If the user reopens
        // BEFORE it finishes, its tail would otherwise land on the
        // resumed turn and duplicate the answer.
        //
        // Two coordinated defenses: (1) cancel it first — chatbot_cancel
        // SIGKILLs the in-flight sidecar (#457), then eager-respawns a
        // fresh warm process the re-issue lands on. (2) The STRUCTURAL
        // fence: the re-issue's `chatbot` command returns a
        // process-global-monotonic turn_id STRICTLY HIGHER than the
        // pre-close generation's, recorded in expectedTurnIdRef; the
        // listener drops every event whose turn_id differs. So even if
        // a stale event slips the cancel timing, its lower turn_id is
        // rejected — duplication is impossible by construction.
        Promise.resolve(invoke("chatbot_cancel"))
          .catch(() => {})
          .then(() => {
            activeQueryIdRef.current = maxId;
            return invoke("chatbot", {
              query: restored[resumeIdx].q,
              history,
            });
          })
          .then((turnId) => {
            if (typeof turnId === "number") {
              expectedTurnIdRef.current = turnId;
            }
          })
          .catch((e) => {
            patchLastTurn({
              status: STATUS.error,
              error: e?.message || String(e),
            });
          });
      }
    }
    // Restore panel visibility only on an app-restart rehydrate: open
    // before close → open on relaunch. A switch keeps the panel as-is
    // (the user is interacting with it).
    if (restorePanel && data.open) setOpen(true);
  }

  // Mount: discover conversations (the directory listing IS the source
  // of truth — no manifest), open the MOST-RECENT one (newest TS; zero
  // persisted active pointer per the gate-1 lock), rehydrate its
  // transcript, and scope the sidecar's telemetry to it. The window is
  // destroyed on Cmd/Ctrl+W and rebuilt fresh on reopen, so without
  // this the conversation + panel state are lost on a routine
  // keystroke. Best-effort: a failure leaves the empty state and is
  // deliberately NOT marked hydrated so the persist effect can't then
  // overwrite a possibly-good file.
  useEffect(() => {
    Promise.resolve(invoke("chatbot_list_conversations"))
      .then((list) => {
        const convos = Array.isArray(list) ? list : [];
        setConversations(convos);
        // Backend guarantees ≥1 (migration / lazy create); first is
        // the most-recent. Defensive fallback keeps a cold render sane.
        const active = convos[0];
        if (!active) {
          hydratedRef.current = true;
          return;
        }
        setActiveConvId(active.id);
        activeConvIdRef.current = active.id;
        return Promise.resolve(
          invoke("chatbot_load_conversation", { id: active.id }),
        ).then((saved) => {
          applyLoadedTranscript(saved, { resume: true, restorePanel: true });
          hydratedRef.current = true;
          // Scope the sidecar's telemetry to this conversation's dir
          // and default "talking about" to its most-recent turn's
          // pinned run (gate-1 Q1; freely changeable after). Cold
          // session → just sets state; the lazy spawn on the first
          // message reads it (no forced launch-time process).
          const data =
            saved && !Array.isArray(saved) ? saved : { turns: [] };
          invoke("chatbot_set_active_conversation", {
            id: active.id,
            runId: lastRunIdOf(data.turns),
          }).catch(() => {});
        });
      })
      .catch(() => {});
    // Mount-once: same intentional `[]` pattern as App.jsx's bootstrap.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // The most-recent turn's pinned run (#559) — the gate-1 Q1 default
  // for "talking about" when a thread is opened/switched. Null when the
  // thread is empty or its turns predate run-pinning (keeps the
  // current/default binding rather than forcing an unknown one).
  function lastRunIdOf(turns) {
    const rows = Array.isArray(turns) ? turns : [];
    for (let i = rows.length - 1; i >= 0; i--) {
      if (rows[i] && rows[i].runId) return rows[i].runId;
    }
    return null;
  }

  // Load + apply a conversation's transcript, then scope telemetry +
  // the run binding to it. Shared by switch / new / delete-fallback.
  function openConversation(id) {
    setActiveConvId(id);
    activeConvIdRef.current = id;
    return Promise.resolve(invoke("chatbot_load_conversation", { id }))
      .then((saved) => {
        applyLoadedTranscript(saved, { resume: false, restorePanel: false });
        const data = saved && !Array.isArray(saved) ? saved : { turns: [] };
        return invoke("chatbot_set_active_conversation", {
          id,
          runId: lastRunIdOf(data.turns),
        }).catch(() => {});
      })
      .catch(() => {});
  }

  // Re-pull the conversation list from the backend. Ordering is
  // most-recently-active first (by each conversation's last turn
  // `ts`), recomputed server-side at list time — so this must run
  // whenever the picker is opened, otherwise the popover shows the
  // stale mount-time order and a just-used thread never floats to the
  // top. Keeps the active selection as-is (a list refresh is not a
  // switch).
  function refreshConversations() {
    Promise.resolve(invoke("chatbot_list_conversations"))
      .then((list) => {
        if (Array.isArray(list)) setConversations(list);
      })
      .catch(() => {});
  }

  // Toggle the picker popover; refresh the (activity-ordered) list on
  // the way OPEN so it's never stale.
  function togglePicker() {
    setPickerOpen((v) => {
      const next = !v;
      if (next) refreshConversations();
      return next;
    });
  }

  // Switch to an existing conversation. Disabled mid-stream (same gate
  // as the run selector / composer) so a swap can't strand an in-flight
  // turn in the wrong thread.
  function switchConversation(id) {
    setPickerOpen(false);
    if (!id || id === activeConvId || streaming) return;
    openConversation(id);
  }

  // "+ New conversation": the backend mints a fresh dir, makes it
  // active, and re-points telemetry; the prior thread is untouched and
  // stays selectable. Prepend it (newest-first) and show its empty
  // transcript.
  function newConversation() {
    setPickerOpen(false);
    if (streaming) return;
    Promise.resolve(invoke("chatbot_new_conversation"))
      .then((meta) => {
        if (!meta) return;
        setConversations((prev) => [meta, ...prev]);
        setActiveConvId(meta.id);
        activeConvIdRef.current = meta.id;
        applyLoadedTranscript(
          { open: true, turns: [] },
          { resume: false, restorePanel: false },
        );
      })
      .catch(() => {});
  }

  // Inline rename = a cosmetic alias (run-style). The conversation is
  // named by its 4-letter perma-id; the dir/id NEVER move, the backend
  // just writes an `alias` into transcript.json — so activeConvId /
  // refs / citations / the perma-id are untouched by construction.
  function startRename(meta) {
    setRenamingId(meta.id);
    // Edit the cosmetic alias (run-style). Blank when there's none
    // yet — the row currently shows the 4-letter id; typing sets an
    // alias, clearing it falls back to the id.
    setRenameDraft(meta.alias || "");
  }
  function cancelRename() {
    setRenamingId(null);
    setRenameDraft("");
  }
  function commitRename(id) {
    const label = renameDraft.trim();
    if (!label) {
      cancelRename();
      return;
    }
    Promise.resolve(
      invoke("chatbot_rename_conversation", { id, label }),
    )
      .then((meta) => {
        if (meta) {
          // id is unchanged; swap the row's meta (new label/name).
          setConversations((prev) =>
            prev.map((c) => (c.id === id ? meta : c)),
          );
        }
      })
      .catch(() => {})
      .finally(cancelRename);
  }

  // Red-X delete → the app's ONE confirm dialog (threaded prop, no new
  // modal). On confirm the backend removes the whole conversation dir
  // (transcript + its telemetry) and returns the now-active fallback
  // (most-recent remaining, or a fresh empty one — never threadless).
  function askDeleteConversation(meta) {
    requestConfirm({
      title: "Delete conversation?",
      message:
        `“${meta.alias || meta.display_label || meta.name}” (last active ` +
        `${convoLastDate(meta)}) and its full history will be ` +
        `permanently removed. This can't be undone.`,
      confirmLabel: "Yes, delete",
      onConfirm: async () => {
        const active = await invoke("chatbot_delete_conversation", {
          id: meta.id,
        });
        const list = await invoke("chatbot_list_conversations");
        setConversations(Array.isArray(list) ? list : []);
        if (active?.id) await openConversation(active.id);
      },
    });
  }

  // Persist { open, turns } to one JSON file under the state dir;
  // survives a window-destroy/reopen and a full reload. In-flight
  // turns ARE persisted (status:"streaming" + partial answer) so a
  // close mid-answer keeps the user's message and re-threads it on
  // reopen. A settled change, a panel open/close, or a brand-new turn
  // (the user's message — must hit disk before they can press Cmd+W)
  // writes immediately; pure streaming growth (token deltas on the
  // same turn) is coalesced to one write per interval and still
  // flushes on settle. Gated on hydration so the mount-time empty
  // state / a failed load never clobbers the saved file.
  useEffect(() => {
    latestStateRef.current = { open, turns };
    // Not hydrated, or no active conversation yet → a write now could
    // clobber a good file or land in the wrong conversation.
    if (!hydratedRef.current || !activeConvIdRef.current) return;
    const last = turns[turns.length - 1];
    const streaming = last && last.status === STATUS.streaming;
    const grew = turns.length !== persistedTurnCountRef.current;
    persistedTurnCountRef.current = turns.length;
    // Capture the target conversation at schedule time so a throttled
    // streaming write can't land in a thread the user switched away to.
    const writeLatest = () => {
      const id = activeConvIdRef.current;
      if (!id) return;
      invoke("chatbot_save_conversation", {
        id,
        state: latestStateRef.current,
      }).catch(() => {});
    };
    if (streaming && !grew) {
      if (!persistTimerRef.current) {
        persistTimerRef.current = setTimeout(() => {
          persistTimerRef.current = null;
          writeLatest();
        }, 600);
      }
    } else {
      if (persistTimerRef.current) {
        clearTimeout(persistTimerRef.current);
        persistTimerRef.current = null;
      }
      writeLatest();
    }
  }, [turns, open]);

  // One global listener for the whole component lifetime — Tauri
  // events fire even when `open` is false, so the user can close the
  // panel mid-stream and reopen to the finished transcript. Dedupe is
  // by activeQueryIdRef, not by `open`.
  useEffect(() => {
    let unlisten;
    // `listen()` is async. Without this guard, React StrictMode's
    // mount→cleanup→remount runs the cleanup BEFORE the first listen()
    // promise resolves, leaving the first subscription live AND a
    // second from the remount — every delta appended twice. The
    // cancelled flag makes a promise that resolves after cleanup tear
    // its own subscription down.
    let cancelled = false;
    listen("chatbot-event", (event) => {
      const payload = event?.payload || {};
      if (activeQueryIdRef.current !== queryIdRef.current) {
        return;
      }
      // Generation fence: drop any per-turn event whose sidecar
      // turn_id isn't the one we're expecting. A stale event from a
      // pre-respawn generation (e.g. the original greeting still
      // draining after a fast close→reopen) carries a LOWER global
      // turn_id than the resumed re-issue and is rejected here, so it
      // can never concatenate onto / duplicate the resumed turn.
      // Session-level events (chatbot_bound) carry no turn_id and are
      // unaffected; a null expectation (id still in flight) falls back
      // to the queryId guard above.
      const tid = payload.turn_id;
      if (
        typeof tid === "number" &&
        expectedTurnIdRef.current != null &&
        tid !== expectedTurnIdRef.current
      ) {
        return;
      }
      handleEvent(payload);
    }).then((u) => {
      if (cancelled) {
        u();
        return;
      }
      unlisten = u;
    });
    return () => {
      cancelled = true;
      if (unlisten) unlisten();
    };
  }, []);

  // Per-event reducer. Each branch maps one sidecar event onto the
  // in-flight (last) turn.
  function handleEvent(payload) {
    switch (payload.event) {
      case "chatbot_bound":
        // Which corpus run this session answers from. Emitted once per
        // session at process start; a rebind re-spawns the sidecar so a
        // fresh `chatbot_bound` arrives — re-pull the list so the
        // current-run mark tracks the real binding.
        setBoundRun(payload.run || null);
        boundRunRef.current = payload.run || null;
        refreshRuns();
        break;
      case "chatbot_thinking":
        // The sidecar emits this at the start of EVERY turn (the
        // decision turn is always buffered). It is the sole driver of
        // the "Thinking…" state — purely event-driven, never
        // UI-inferred — shown for ALL responses regardless of
        // reasoning, cleared by the first chunk / retrieving / done.
        patchLastTurn({ thinking: true });
        break;
      case "chatbot_chunk":
        // First visible token ends the buffered gap → drop Thinking….
        patchLastTurn((t) => ({
          ...t,
          a: t.a + (payload.delta || ""),
          thinking: false,
        }));
        break;
      case "chatbot_replace":
        // The model emitted a prose preamble around a JSON tool call
        // and the stream-gate leaked the preamble (plus whatever
        // landed in the chunk that completed the JSON onset) to the
        // bubble before suppression kicked in. The loop emits this
        // event after the call is extracted so the UI overwrites the
        // leaked bubble content with `text` (empty by contract — the
        // next-hop `chatbot_retrieving` drives the visible state from
        // there) and the persisted transcript carries the clean text,
        // not the leaked mid-turn fragment.
        patchLastTurn((t) => ({
          ...t,
          a: payload.text || "",
          thinking: false,
        }));
        break;
      case "chatbot_retrieving":
        patchLastTurn((t) => ({
          ...t,
          retrieving: true,
          retrievingQuery: payload.query || "",
          // Per-turn audit trail of every dispatched lookup's
          // `describe(call)` string. The live status line shows only
          // the LAST one (the call currently running); the post-turn
          // "[+]" affordance lets the user expand the full sequence,
          // so a multi-hop ReAct turn doesn't hide what the model
          // actually did. Empty string entries (a tool-call validate
          // failure) get filtered out by the renderer.
          lookupLog: [
            ...(Array.isArray(t.lookupLog) ? t.lookupLog : []),
            payload.query || "",
          ],
          thinking: false,
        }));
        break;
      case "chatbot_done": {
        // null → no lookup fired (pure conversation, no block).
        // [] → tool ran, matched nothing (explicit empty state).
        const raw = Array.isArray(payload.resources)
          ? payload.resources
          : null;
        // PIN the answering run onto the turn. The sidecar stamps the
        // run this turn was answered against on every completion; we
        // persist it with the turn so its citations resolve against
        // THAT run forever — regardless of the currently-selected run
        // and across restarts — never against the transient live bind
        // (which is null after a restart until the sidecar re-binds,
        // and wrong once the user switches runs). Fall back to the
        // live bind only if a turn somehow arrives without a stamp
        // (older sidecar), so the row never regresses to unresolvable.
        const pinnedRun = payload.run || boundRunRef.current || null;
        // A refused turn's `a` is the deterministic refusal chunk,
        // not a real assistant reply. Mark it so the history-builder
        // for the next turn drops its assistant slot — without that
        // the model sees the refusal text in prior turns and parrots
        // it as prose on later turns even after a corpus is bound.
        const refused = payload.refused === true;
        patchLastTurn((t) => ({
          ...t,
          resources: raw,
          retrieving: false,
          thinking: false,
          status: STATUS.done,
          runId: pinnedRun,
          refused,
          // Wall-clock turn duration from send → done, surfaced post-
          // turn next to the "[+]" lookup-log toggle. Computed once at
          // completion so the rendered string is stable across
          // re-renders. Falls back to 0 (rendered as "<1s") when the
          // turn lacks a creation timestamp (legacy in-flight turn).
          durationMs: t.ts ? Math.max(0, Date.now() - t.ts) : 0,
        }));
        resolveTurnResources(activeQueryIdRef.current, raw, pinnedRun);
        break;
      }
      case "chatbot_stopped":
        // User hit Stop; the sidecar was terminated. Keep whatever
        // partial answer arrived, leave the streaming state.
        patchLastTurn((t) =>
          t.status === STATUS.streaming
            ? { ...t, status: STATUS.done, retrieving: false, thinking: false }
            : t,
        );
        break;
      case "chatbot_error":
        patchLastTurn({
          status: STATUS.error,
          error: payload.message || "Something went wrong.",
          thinking: false,
        });
        break;
      default:
        // Unknown event from a future sidecar version: ignore.
        break;
    }
  }

  // Keep the transcript pinned to the latest content as it grows —
  // but only when the user is already following the bottom, or just
  // sent a new message (their own turn should always come into view).
  // A selector rebind / run-list refresh re-renders without adding
  // content and must NOT scroll the user away from what they're
  // reading.
  useEffect(() => {
    const pane = bodyRef.current;
    if (!pane) return;
    const grew = turns.length > prevTurnCountRef.current;
    prevTurnCountRef.current = turns.length;
    if (atBottomRef.current || grew) {
      pane.scrollTop = pane.scrollHeight;
      atBottomRef.current = true;
    }
  }, [turns]);

  // Focus the input when the panel opens so the user can type immediately.
  useEffect(() => {
    if (open && inputRef.current) {
      inputRef.current.focus();
    }
  }, [open]);

  // Outside-click / Escape close is owned by the shared MetaDropdown
  // (both pickers); selecting / switching / new / delete still close
  // it explicitly via the close handlers passed in.

  // Attestation is a non-blocking visibility signal: a failed or
  // in-flight attestation does not block sending in Private Cloud
  // mode. The real per-connection guarantee is enforced at the
  // transport layer (the kernel's attested provider pins the enclave
  // TLS key and refuses a non-matching enclave), so send is never gated
  // on the attestation panel's state — the panel only surfaces it.
  const attestBlocked = false;
  const attestBlockReason = undefined;

  function sendQuery() {
    const q = draft.trim();
    if (!q || streaming || attestBlocked) return;
    queryIdRef.current += 1;
    activeQueryIdRef.current = queryIdRef.current;
    // Unknown until the command returns the assigned id; the fence is
    // off (queryId guard covers this window) until it's known.
    expectedTurnIdRef.current = null;
    // Every prior turn's USER message goes back to the model — every
    // message since the last successful answer, not just completed
    // pairs. A cancelled / errored / partial turn keeps its user
    // message; only its assistant slot is empty (or partial). Filtering
    // to done(user→assistant) pairs would drop the user message of a
    // turn whose response was Stopped, so "my name is alex" → (cancel)
    // → "what's my name" would lose the name entirely.
    const history = turns.flatMap((t) => {
      const msgs = [{ role: "user", content: t.q }];
      // A refused turn's `a` is the deterministic refusal chunk —
      // keep the user message (the existing contract for partial /
      // errored turns) but drop the assistant slot so the model
      // doesn't see its own prior refusal text and learn to parrot
      // it as prose.
      if (t.a && !t.refused) {
        const asst = { role: "assistant", content: t.a };
        // Cited records ride along so the sidecar can seed the next
        // turn's accumulator at brackets [1..K]. Lets the user follow
        // up with "tell me more about [3]" and have the bracket
        // resolve to the same record without a fresh search. Only
        // the structural keys travel (kind, record_id) — preview
        // text stays on the UI and the backend re-fetches the
        // records from the bound store.
        const refs = Array.isArray(t.resources) ? t.resources : [];
        if (refs.length > 0) {
          asst.cited_refs = refs.map((r) => ({
            kind: r.kind, record_id: r.record_id,
          }));
        }
        msgs.push(asst);
      }
      return msgs;
    });
    setTurns((prev) => [
      ...prev,
      {
        id: queryIdRef.current,
        q,
        a: "",
        // #568 owns this: the canonical per-turn timestamp, epoch
        // millis at turn creation. Persisted as part of the turn
        // (transcript schema) — it drives conversation ordering (the
        // last turn's `ts` = last activity) and #567 only consumes it.
        // Legacy turns predating this field degrade gracefully (those
        // conversations fall back to ISO-creation order).
        ts: Date.now(),
        status: STATUS.streaming,
        retrieving: false,
        retrievingQuery: "",
        // Per-turn audit fields the "prepped for Xs [+]" affordance
        // reads after the turn ends. `lookupLog` accumulates one
        // `describe(call)` string per dispatched lookup as the
        // `chatbot_retrieving` events arrive; `durationMs` is stamped
        // at `chatbot_done`; `lookupLogExpanded` is the UI toggle.
        lookupLog: [],
        durationMs: null,
        lookupLogExpanded: false,
        resources: null,
        thinking: false,
      },
    ]);
    setDraft("");
    invoke("chatbot", { query: q, history })
      .then((turnId) => {
        if (typeof turnId === "number") expectedTurnIdRef.current = turnId;
      })
      .catch((e) => {
        patchLastTurn({
          status: STATUS.error,
          error: e?.message || String(e),
        });
      });
  }

  function stopQuery() {
    // Actually stop generation — terminate the in-flight sidecar
    // server-side, not just hide the UI. The sidecar dies, its
    // partial answer so far stays in the transcript. The `chatbot_stopped`
    // event confirms it; flip optimistically too so the control feels
    // immediate even if the event is briefly in flight.
    invoke("chatbot_cancel").catch(() => {});
    patchLastTurn((t) =>
      t.status === STATUS.streaming
        ? { ...t, status: STATUS.done, retrieving: false }
        : t,
    );
  }

  // Collapse the panel back to the corner pill. The single existing
  // open/closed state — no parallel one. This only hides the panel;
  // the transcript (`turns`) is untouched, so reopening restores the
  // conversation. The `_` button and the header bar both call this:
  // the affordance is minimize, never close/destroy.
  function collapse() {
    setOpen(false);
  }

  // A click on the header bar collapses, EXCEPT when it lands on an
  // interactive control inside the bar (the `_` button today; a run
  // selector / rename UI if later moved in) — those keep their own
  // behavior. Only the non-interactive area (title + empty bar) is a
  // collapse target. `closest` walks up from the click target so a
  // click anywhere within such a control is excluded, not just a
  // direct hit.
  function onHeaderClick(e) {
    if (e.target.closest('button, select, input, textarea, a, [role="button"]')) {
      return;
    }
    collapse();
  }

  // Collapsed: small pill in the corner.
  if (!open) {
    return (
      <button
        type="button"
        className="chatbot-helper-pill"
        onClick={() => setOpen(true)}
        aria-label="Open Chat About Your Data"
        data-testid="chatbot-helper-pill"
      >
        <img
          className="chatbot-helper-logo"
          src={baseVaultLogo}
          alt=""
          aria-hidden="true"
        />
        Chat About Your Data
      </button>
    );
  }

  return (
    <div className="chatbot-helper-panel" data-testid="chatbot-helper-panel">
      <header
        className="chatbot-helper-header"
        onClick={onHeaderClick}
        title="Collapse chat"
      >
        <div className="chatbot-helper-lead">
          <img
            className="chatbot-helper-logo"
            src={baseVaultLogo}
            alt=""
            aria-hidden="true"
          />
          <div className="chatbot-helper-title">Chat About Your Data</div>
        </div>
        <div className="chatbot-helper-header-actions">
          <button
            type="button"
            className="chatbot-helper-close"
            onClick={(e) => {
              // Stop the bubble so the header's collapse handler
              // doesn't also fire — idempotent here (both collapse),
              // but explicit keeps it correct if the bar handler ever
              // does more than collapse.
              e.stopPropagation();
              collapse();
            }}
            aria-label="Collapse chat"
            data-testid="chatbot-helper-close"
          >
            _
          </button>
        </div>
      </header>

      {/* Source-run + Conversation pickers — the SAME shared
          MetaDropdown (req #2/#5), one 2-line look, stacked under the
          header. The run variant omits the conversation-only per-row
          ✎/✕ + "+ New" header; both read as a set by construction. */}
      {runOptions.length > 0 ? (
        (() => {
          // Current run: the bound run when it's in the list, else the
          // backend-marked default, else the first — unchanged logic.
          const curRun = runOptions.some((o) => o.run_id === boundRun)
            ? boundRun
            : runOptions.find((o) => o.bound)?.run_id
              || runOptions[0].run_id;
          const curOpt = runOptions.find((o) => o.run_id === curRun);
          return (
            <MetaDropdown
              barLabel="Source run"
              barClass="chatbot-helper-runbar"
              barTestId="chatbot-helper-runbar"
              tidPrefix="chatbot-helper-run"
              disabled={streaming}
              open={runPickerOpen}
              onToggle={toggleRunPicker}
              onClose={() => setRunPickerOpen(false)}
              selected={{
                // Line 1 = the shared labelForRun title (the SAME
                // string a message's resources block uses, so they
                // can't drift). Line 2 = the run's structured date +
                // #perma-id (Part B real fields, never parsed back
                // out of the label).
                title: labelForRun(curRun, runOptions, resolveRunName),
                date: prettyDateTime(curOpt?.created_at),
                shortId: curOpt?.short_id,
              }}
            >
              {runOptions.map((o) => (
                <div
                  key={o.run_id}
                  className={
                    "chatbot-helper-mdrow" +
                    (o.run_id === curRun ? " is-active" : "")
                  }
                  data-testid={`chatbot-helper-runrow-${o.run_id}`}
                >
                  <MetaRow
                    title={labelForRun(o.run_id, runOptions, resolveRunName)}
                    date={prettyDateTime(o.created_at)}
                    shortId={o.short_id}
                    active={o.run_id === curRun}
                    onClick={() => selectRun(o.run_id)}
                    titleAttr={labelForRun(
                      o.run_id, runOptions, resolveRunName,
                    )}
                    testId={`chatbot-helper-runrow-name-${o.run_id}`}
                  />
                </div>
              ))}
            </MetaDropdown>
          );
        })()
      ) : (
        // No list yet (fetch pending/failed). The bar still renders
        // (same contract as before — the two pickers always read as a
        // set): a bound run announced via event is shown as plain
        // text so the corpus is never hidden; otherwise just the
        // label, awaiting the list.
        <div className="chatbot-helper-mdbar chatbot-helper-runbar">
          <span className="chatbot-helper-mdbar-label">Source run</span>
          {boundRun && (
            <div
              className="chatbot-helper-subtitle"
              data-testid="chatbot-helper-boundrun"
              title={boundRun}
            >
              {labelForRun(boundRun, runOptions, resolveRunName)}
            </div>
          )}
        </div>
      )}

      {(() => {
        const a = conversations.find((c) => c.id === activeConvId);
        return (
          <MetaDropdown
            barLabel="Conversation"
            barClass="chatbot-helper-threadbar"
            barTestId="chatbot-helper-threadbar"
            tidPrefix="chatbot-helper-thread"
            disabled={streaming}
            open={pickerOpen}
            onToggle={togglePicker}
            onClose={() => setPickerOpen(false)}
            placeholder="conversation"
            selected={
              a
                ? {
                    // Title = cosmetic alias if set, else the derived
                    // `display_label` (`Conversation N · Mon Day`) —
                    // NOT the bare dir-tail perma-id. The immutable
                    // #<short_id> stays on line 2 as the stable identity.
                    title: a.alias || a.display_label || a.name,
                    date: convoLastDate(a),
                    shortId: a.short_id,
                  }
                : null
            }
          >
            <button
              type="button"
              className="chatbot-helper-threadnew"
              onClick={newConversation}
              data-testid="chatbot-helper-threadnew"
            >
              + New conversation
            </button>
            {conversations.map((c) => (
              <div
                key={c.id}
                className={
                  "chatbot-helper-mdrow" +
                  (c.id === activeConvId ? " is-active" : "")
                }
                data-testid={`chatbot-helper-threadrow-${c.id}`}
              >
                {renamingId === c.id ? (
                  <input
                    type="text"
                    className="chatbot-helper-threadrename"
                    value={renameDraft}
                    autoFocus
                    onChange={(e) => setRenameDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        commitRename(c.id);
                      } else if (e.key === "Escape") {
                        e.preventDefault();
                        cancelRename();
                      }
                    }}
                    onBlur={() => commitRename(c.id)}
                    aria-label={`Rename ${c.alias || c.short_id}`}
                    data-testid={`chatbot-helper-threadrename-${c.id}`}
                  />
                ) : (
                  <MetaRow
                    title={c.alias || c.display_label || c.name}
                    date={convoLastDate(c)}
                    shortId={c.short_id}
                    active={c.id === activeConvId}
                    onClick={() => switchConversation(c.id)}
                    titleAttr={`${c.alias || c.display_label || c.name} · #${c.short_id} · last active ${convoLastDate(c)}`}
                    testId={`chatbot-helper-threadrow-name-${c.id}`}
                  />
                )}
                <button
                  type="button"
                  className="chatbot-helper-threadedit"
                  onClick={() => startRename(c)}
                  aria-label={`Rename ${c.alias || c.short_id}`}
                  title="Rename this conversation"
                  data-testid={`chatbot-helper-threadedit-${c.id}`}
                >
                  ✎
                </button>
                <button
                  type="button"
                  className="chatbot-helper-threaddel"
                  onClick={() => askDeleteConversation(c)}
                  aria-label={`Delete ${c.alias || c.short_id}`}
                  title="Delete this conversation"
                  data-testid={`chatbot-helper-threaddel-${c.id}`}
                >
                  ✕
                </button>
              </div>
            ))}
          </MetaDropdown>
        );
      })()}

      <div
        className="chatbot-helper-body"
        ref={bodyRef}
        onScroll={onBodyScroll}
      >
        {turns.length === 0 && (
          <div className="chatbot-helper-empty">
            Talk about anything — ideas, what's on your mind, or what's
            in your processed data. The chatbot looks things up in your data
            when a question needs it, with numbered references back to
            the records.
          </div>
        )}
        {turns.map((t) => {
          // Display-only contiguous renumber of citations (answer
          // markers + the resource list, through one shared map).
          const view = renumberCitations(
            t.a, Array.isArray(t.resources) ? t.resources : [],
          );
          // "Thinking…" is purely sidecar-driven: the per-turn
          // `chatbot_thinking` event (emitted at the start of EVERY
          // turn) sets `t.thinking`. It is never UI-inferred. Shown
          // for ALL responses regardless of reasoning — the decision
          // turn is always buffered, so every response has an in-flight
          // gap to fill. The in-flight visual guards are belt-and-
          // suspenders: hide it the instant content or retrieval
          // arrives even if the clearing event is late / out of order.
          const thinking =
            t.thinking &&
            t.status === STATUS.streaming &&
            !t.retrieving &&
            !t.a;
          // This turn's own source name (the selector's exact string
          // for its pinned run), computed once. "" → no run pinned
          // (legacy turn): the resources block omits the label.
          const sourceLabel = labelForRun(
            t.runId, runOptions, resolveRunName,
          );
          // Date-only of this turn's cited records (its pinned run's
          // date) — shown on every reference + the source line + the
          // copied text + the in-body [N] hover. "" → omitted.
          const refDate = runRefDate(t.runId, runOptions);
          return (
          <div className="chatbot-helper-turn" key={t.id}>
            <div className="chatbot-helper-q" data-testid="chatbot-helper-q">
              {t.q}
            </div>
            {t.status === STATUS.error ? (
              <div className="chatbot-helper-error" data-testid="chatbot-helper-error">
                {t.error}
              </div>
            ) : (
              <>
                {t.retrieving && (
                  <div
                    className="chatbot-helper-retrieving"
                    data-testid="chatbot-helper-retrieving"
                  >
                    <span className="chatbot-helper-caret" aria-hidden="true">▌</span>
                    {" "}Searching your data
                    {t.retrievingQuery ? ` for “${t.retrievingQuery}”` : ""}…
                  </div>
                )}
                {thinking && (
                  <div
                    className="chatbot-helper-retrieving"
                    data-testid="chatbot-helper-thinking"
                  >
                    <span className="chatbot-helper-caret" aria-hidden="true">▌</span>
                    {" "}Thinking…
                  </div>
                )}
                {(t.a || !t.retrieving) && !thinking && (
                  <div className="chatbot-helper-a" data-testid="chatbot-helper-a">
                    {renderAnswerWithRefs(
                      view.answer,
                      view.resources,
                      // The SAME ref-click the resource chip fires
                      // (onOpenResource → handleMarkdownNavigate →
                      // resolveCitation → scroll/highlight), against
                      // this turn's OWN pinned run (#559). Reuse, not
                      // a parallel path.
                      (res) => onOpenResource?.(res, t.runId || null),
                      refDate,
                    )}
                    {t.status === STATUS.streaming && !t.retrieving && (
                      <span
                        className="chatbot-helper-caret"
                        aria-hidden="true"
                      >▌</span>
                    )}
                  </div>
                )}
              </>
            )}
            {t.status === STATUS.done
              && Array.isArray(t.lookupLog)
              && t.lookupLog.filter(Boolean).length > 0 && (
              <div
                className="chatbot-helper-lookup-log"
                data-testid="chatbot-helper-lookup-log"
              >
                <button
                  type="button"
                  className="chatbot-helper-lookup-log-toggle"
                  data-testid="chatbot-helper-lookup-log-toggle"
                  onClick={() => toggleLookupLog(t.id)}
                  aria-expanded={!!t.lookupLogExpanded}
                  aria-label={
                    t.lookupLogExpanded
                      ? "Hide lookup log"
                      : "Show lookup log"
                  }
                >
                  prepped for {formatTurnDuration(t.durationMs)}{" "}
                  <span aria-hidden="true">
                    {t.lookupLogExpanded ? "[−]" : "[+]"}
                  </span>
                </button>
                {t.lookupLogExpanded && (
                  <ol
                    className="chatbot-helper-lookup-log-list"
                    data-testid="chatbot-helper-lookup-log-list"
                  >
                    {/* One <li> per dispatched lookup, in dispatch
                        order. `describe(call)` is multi-line for a
                        multi-lookup call (one line per lookup in the
                        array) — render with `white-space: pre-line`
                        so the line breaks survive. */}
                    {t.lookupLog
                      .filter(Boolean)
                      .map((desc, i) => (
                        <li
                          key={i}
                          className="chatbot-helper-lookup-log-item"
                        >
                          {desc}
                        </li>
                      ))}
                  </ol>
                )}
              </div>
            )}
            {t.status === STATUS.done && Array.isArray(t.resources) && (
              view.resources.length > 0 ? (
                <div
                  className="chatbot-helper-resources"
                  data-testid="chatbot-helper-resources"
                >
                  <div className="chatbot-helper-refs-title">
                    Resources from your data
                  </div>
                  {/* This message's OWN source, by the exact name the
                      "Source run" selector uses for it (shared
                      labelForRun). Derived from the turn's pinned
                      `runId`, NOT the live selection — different
                      messages in one thread can be grounded in
                      different sources, and each must say which.
                      Omitted for a legacy turn with no pinned run. */}
                  {sourceLabel && (
                    <div
                      className="chatbot-helper-refs-source"
                      data-testid="chatbot-helper-refs-source"
                    >
                      {sourceLabel}
                      {refDate && (
                        <span className="chatbot-helper-ref-date">
                          {" · "}{refDate}
                        </span>
                      )}
                    </div>
                  )}
                  <ol className="chatbot-helper-resources-list">
                    {view.resources.map((r) => (
                      <ResourceItem
                        key={r.index}
                        resource={r}
                        // Resolve against the turn's OWN pinned run, not
                        // the live `boundRun`. `boundRun` is the run the
                        // current session is bound to — null right after
                        // a restart (until the sidecar re-binds) and a
                        // DIFFERENT run once the user switches the
                        // selector, which is exactly what made old refs
                        // dead/mis-targeted. `t.runId` travels with the
                        // message so the click always lands in the run
                        // that produced the citation.
                        onOpen={() => onOpenResource?.(r, t.runId || null)}
                        date={refDate}
                      />
                    ))}
                  </ol>
                </div>
              ) : (
                <div
                  className="chatbot-helper-resources-empty"
                  data-testid="chatbot-helper-resources-empty"
                >
                  No matching resources in your data.
                </div>
              )
            )}
            {/* Copy sits at the very BOTTOM of the reply — after the
                references — even when grounded, so one click copies
                the whole answer WITH its reference list (the `[N]`
                markers are useless without it). Only once the reply
                has settled; a non-grounded reply copies just the
                answer. */}
            {t.status === STATUS.done && t.a && (
              <MessageCopy
                text={buildReplyCopyText(
                  view.answer, view.resources, sourceLabel, refDate,
                )}
                ts={t.ts}
                copyTestId="chatbot-helper-copy-a"
              />
            )}
          </div>
          );
        })}
      </div>

      <footer className="chatbot-helper-footer">
        <textarea
          ref={inputRef}
          className="chatbot-helper-input"
          placeholder="Talk to your data, or just chat…"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            // Enter sends; Shift+Enter inserts a newline. Matches the
            // convention of every chat surface the user already uses.
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendQuery();
            }
          }}
          rows={2}
          data-testid="chatbot-helper-input"
          disabled={streaming}
        />
        {streaming ? (
          // While generating, the composer button is a real Stop —
          // clicking it terminates the in-flight sidecar (see
          // stopQuery), the modern-chatbot send↔stop swap.
          <button
            type="button"
            className="chatbot-helper-send is-stop"
            onClick={stopQuery}
            aria-label="Stop generating"
            data-testid="chatbot-helper-stop"
          >
            <span aria-hidden="true">■</span>
          </button>
        ) : (
          <button
            type="button"
            className="chatbot-helper-send"
            onClick={sendQuery}
            disabled={!draft.trim() || attestBlocked}
            title={attestBlockReason}
            aria-label="Send"
            data-testid="chatbot-helper-send"
          >
            <span aria-hidden="true">↑</span>
          </button>
        )}
      </footer>
    </div>
  );
}
