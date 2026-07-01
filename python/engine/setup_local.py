#!/usr/bin/env python3
"""
Local setup verification — detect-only. No install, no pull, no brew.

Reads the selected local backend (`local_backend` in config, resolved
through llm.MODE_SPEC) and verifies its precondition, emitting structured
JSON lines the Rust layer forwards to the UI:

  - mlx (primary): the model snapshot has been downloaded in-app.
  - ollama (opt-in): the daemon is reachable AND the model is present.
    We never install or pull — Ollama is for users who already run it;
    BaseVault surfaces a copyable remedy instead of doing it for them.

Every failure exits non-zero only AFTER emitting one
{"status":"error","step":...,"message":...,"command"?:...} line, so the
UI always has an actionable diagnostic instead of a bare exit code.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys

from engine.llm import Mode, Provider, get_mode_spec, mlx_model_dir

OLLAMA_DAEMON_URL = "http://127.0.0.1:11434"


def emit(status: str, message: str, **kwargs):
    print(json.dumps({"status": status, "message": message, **kwargs}), flush=True)


def fail(step: str, message: str, command: str | None = None):
    """Emit one structured error (with the failing step + optional
    copyable remedy) and exit non-zero. The `step` lets the UI route to
    the right surface; `command` is shown with a copy button."""
    extra = {"step": step}
    if command:
        extra["command"] = command
    emit("error", message, **extra)
    sys.exit(1)


def check_apple_silicon():
    if platform.system() != "Darwin":
        fail("platform", "BaseVault local mode requires macOS.")
    if platform.machine() != "arm64":
        fail(
            "platform",
            "Local mode requires Apple Silicon (M1/M2/M3/M4). Intel Macs "
            "lack the unified memory needed for on-device models. Use "
            "cloud mode instead.",
        )
    emit("ok", "Apple Silicon detected.")


def get_ram_gb() -> int:
    r = subprocess.run(
        ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
    )
    try:
        return int(r.stdout.strip()) // (1024**3)
    except ValueError:
        return 0


def verify_ollama(model: str):
    """Detect-only classification: daemon-down vs model-not-pulled vs
    other, each with a copyable remedy. The pipeline only ever talks to
    the daemon over HTTP, so the daemon (not the CLI binary on PATH) is
    the honest precondition."""
    import json as _json
    import urllib.error  # ci-allow:net - localhost Ollama daemon probe, not enclave-reaching
    import urllib.request  # ci-allow:net - localhost Ollama daemon probe, not enclave-reaching

    req = urllib.request.Request(
        f"{OLLAMA_DAEMON_URL}/api/tags",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            body = r.read().decode("utf-8")
    except urllib.error.URLError:
        fail(
            "ollama_daemon",
            "Ollama isn't running. Start the Ollama daemon in a terminal, "
            "then retry.",
            command="ollama serve",
        )
    except Exception as e:
        fail(
            "ollama_daemon",
            f"Couldn't reach Ollama at {OLLAMA_DAEMON_URL}: {e}",
            command="ollama serve",
        )

    try:
        data = _json.loads(body)
    except _json.JSONDecodeError as e:
        fail("ollama_daemon", f"Ollama responded but the JSON was unparseable: {e}")

    names = {m.get("name", "") for m in data.get("models", [])}
    # Accept exact match and shared-tag variants (e.g. qwen3.5:9b vs
    # qwen3.5:9b-instruct) by also checking the base name.
    base = model.split(":")[0]
    matched = model in names or any(
        n == model or n.startswith(base + ":") or n == base for n in names
    )
    if not matched:
        pulled = ", ".join(sorted(n for n in names if n)) or "(none)"
        fail(
            "ollama_model",
            f"The Ollama daemon is up, but {model!r} isn't pulled. "
            f"BaseVault is only verified to work with {model!r} on the "
            f"Ollama path — other models may misbehave. "
            f"Currently pulled: {pulled}.",
            command=f"ollama pull {model}",
        )
    emit("ok", f"Ollama daemon up, {model} present.")


def verify_mlx(model: str):
    path = mlx_model_dir(model)
    if not path.exists() or not any(path.iterdir()):
        fail(
            "model_download",
            f"The local model {model!r} hasn't been downloaded yet. "
            f"Open Settings → Local model → Download to fetch it.",
        )
    emit("ok", f"Local model {model} is downloaded.")


def main():
    parser = argparse.ArgumentParser()
    # Accepted for back-compat with existing callers. Setup is detect-
    # only now, so verification is the only behavior regardless of flags.
    parser.add_argument("--verify-only", action="store_true")
    parser.parse_known_args()

    check_apple_silicon()
    emit("ok", f"{get_ram_gb()}GB RAM detected.")

    spec = get_mode_spec(Mode.LOCAL)
    model = spec.model_id
    emit("ok", f"Backend: {spec.provider.value}, model: {model}")

    if spec.provider == Provider.OLLAMA:
        verify_ollama(model)
    else:
        verify_mlx(model)

    emit(
        "done",
        "Local setup verified.",
        model=model,
        backend=spec.provider.value,
    )


if __name__ == "__main__":
    main()
