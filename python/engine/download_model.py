#!/usr/bin/env python3
"""
Download an MLX model snapshot from the Hugging Face Hub into
~/.basevault/models/<repo-id>/, emitting JSON progress lines the Rust
layer forwards to the UI. User-triggered only (no auto-download).

A poller thread reports bytes-on-disk vs. the repo's total size (from
the Hub file metadata) so the UI can show a real percentage; if the
size query fails, progress is reported as bytes-only (indeterminate).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

from engine.llm import DEFAULT_MLX_MODEL, mlx_model_dir


def emit(status: str, message: str, **kwargs):
    print(json.dumps({"status": status, "message": message, **kwargs}), flush=True)


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MLX_MODEL)
    args = ap.parse_args()
    model = args.model

    dest = mlx_model_dir(model)
    dest.mkdir(parents=True, exist_ok=True)
    # Multi-GB of weights would otherwise pin Spotlight (mds/mds_stores)
    # at high CPU on first run. `.metadata_never_index` excludes this
    # model's subtree from indexing (macOS-supported, no sudo).
    # Idempotent.
    try:
        (dest / ".metadata_never_index").touch(exist_ok=True)
    except OSError:
        pass  # best-effort; indexing exclusion is not load-bearing
    emit("start", f"Downloading {model}…", model=model)

    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as e:
        emit("error", f"huggingface_hub unavailable: {e}", step="model_download")
        sys.exit(1)

    total = 0
    try:
        info = HfApi().model_info(model, files_metadata=True)
        total = sum((s.size or 0) for s in (info.siblings or []))
    except Exception:
        total = 0  # progress is reported bytes-only when size is unknown

    stop = threading.Event()

    def poll():
        while not stop.wait(1.0):
            done = dir_size(dest)
            pct = int(done * 100 / total) if total else None
            done_mb = done // (1024 * 1024)
            msg = f"{done_mb} MB"
            if total:
                msg += f" / {total // (1024 * 1024)} MB"
            emit("progress", msg, downloaded=done, total=total, pct=pct)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    try:
        snapshot_download(repo_id=model, local_dir=str(dest))
    except Exception as e:
        stop.set()
        poller.join(timeout=2)
        emit("error", f"Download failed: {e}", step="model_download")
        sys.exit(1)
    stop.set()
    poller.join(timeout=2)
    emit("done", f"{model} downloaded.", model=model, downloaded=dir_size(dest))


if __name__ == "__main__":
    main()
