import { useCallback, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

// Shared local-backend driver for the Settings and Wizard local
// panels. One source so the two first-run surfaces can't drift on
// MLX download/delete/status OR the Ollama detect-only verify.
// `remove` does only the filesystem invoke — the caller owns the
// confirm dialog (the two surfaces phrase it differently).
export function useLocalModel() {
  const [stat, setStat] = useState(null); // {model,downloaded,size_bytes}
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);
  const [pct, setPct] = useState(null);
  const [error, setError] = useState(null);
  // Ollama detect-only verify (no install). verifyStatus:
  // idle | verifying | done | failed. fixCmd is the structured
  // copyable remedy (e.g. `ollama serve`) when setup_local emits one.
  const [verifyStatus, setVerifyStatus] = useState("idle");
  const [verifyError, setVerifyError] = useState(null);
  const [fixCmd, setFixCmd] = useState(null);
  const [cmdCopied, setCmdCopied] = useState(false);

  const refresh = useCallback(() => {
    invoke("local_model_status")
      .then((s) => setStat(s || null))
      .catch(() => {/* best-effort; panel falls back to "unknown" */});
  }, []);

  const download = useCallback(async () => {
    setBusy(true);
    setMsg("Starting…");
    setPct(null);
    setError(null);
    let unlisten;
    try {
      unlisten = await listen("model-download-progress", (event) => {
        try {
          const p = JSON.parse(event.payload);
          if (p.status === "error") setError(p.message || "download error");
          else setMsg(p.message || null);
          if (typeof p.pct === "number") setPct(p.pct);
        } catch {
          /* ignore non-JSON progress lines */
        }
      });
      await invoke("download_mlx_model", {});
      refresh();
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      unlisten?.();
      setBusy(false);
      setMsg(null);
      setPct(null);
    }
  }, [refresh]);

  const remove = useCallback(async () => {
    try {
      await invoke("delete_mlx_model", {});
      refresh();
    } catch (e) {
      setError(e?.message || String(e));
    }
  }, [refresh]);

  // Detect-only Ollama check. setup_local.py never installs/pulls; on
  // failure it emits a structured error whose `command` is a copyable
  // remedy — surfaced here instead of any install prose.
  const verify = useCallback(async () => {
    setVerifyStatus("verifying");
    setVerifyError(null);
    setFixCmd(null);
    setCmdCopied(false);
    let unlisten;
    try {
      unlisten = await listen("setup-progress", (event) => {
        try {
          const p = JSON.parse(event.payload);
          if (p.status === "error") {
            setVerifyError(p.message || "setup error");
            if (p.command) setFixCmd(p.command);
          }
        } catch {
          /* non-JSON progress lines are fine to ignore */
        }
      });
      await invoke("setup_local", { mode: "verify" });
      setVerifyStatus("done");
    } catch (e) {
      setVerifyStatus("failed");
      setVerifyError((prev) => prev || e?.message || String(e));
    } finally {
      unlisten?.();
    }
  }, []);

  const copyFixCmd = useCallback(async () => {
    if (!fixCmd) return;
    try {
      await navigator.clipboard.writeText(fixCmd);
      setCmdCopied(true);
      setTimeout(() => setCmdCopied(false), 2000);
    } catch {
      /* clipboard can fail under gatekeeper; the command is selectable */
    }
  }, [fixCmd]);

  const resetVerify = useCallback(() => {
    setVerifyStatus("idle");
    setVerifyError(null);
    setFixCmd(null);
    setCmdCopied(false);
  }, []);

  return {
    stat, busy, msg, pct, error, setError, refresh, download, remove,
    verify, verifyStatus, verifyError, fixCmd, cmdCopied, copyFixCmd,
    resetVerify,
  };
}
