#!/usr/bin/env bash
# Restart the BaseVault dev app.
# Kills any running instance, frees port 1420, restarts ollama,
# ensures node_modules are present, relaunches tauri dev.
#
# Run from anywhere; the script cd's into the app dir on its own.

set -uo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$APP_DIR"

# ── 1. Pre-flight ─────────────────────────────────────────────────────
echo "── 1. Pre-flight ─────────────────────────────────────────────────"
echo "  app dir: $APP_DIR"

# Each git worktree gets its own app/ tree; gitignored artifacts
# (npm packages, the bundled Python sidecar) don't travel between
# worktrees — only tracked files do.
#
# In a non-main worktree, prefer symlinking to the main worktree's
# already-built artifacts. Saves ~hundreds of MB + 5+ min of install
# per fresh worktree, and keeps everything in sync when deps change.
#
# Detect main: `git worktree list` always lists main first.
MAIN_WT="$(git worktree list --porcelain 2>/dev/null | awk '/^worktree / {print $2; exit}')"
MAIN_APP="$MAIN_WT/app"
IN_MAIN="$([ "$APP_DIR" = "$MAIN_APP" ] && echo 1 || echo 0)"

# node_modules: symlink from main when possible, install otherwise.
# Without it, `npm run tauri dev` dies with "tauri: command not found"
# (the tauri CLI lives in node_modules/.bin).
if [ ! -e node_modules ]; then
    if [ "$IN_MAIN" = "0" ] && [ -d "$MAIN_APP/node_modules" ]; then
        ln -s "$MAIN_APP/node_modules" node_modules
        echo "  ✓ node_modules → $MAIN_APP/node_modules (symlinked)"
    else
        echo "  node_modules missing — running npm install..."
        npm install
        echo "  ✓ npm install done"
    fi
else
    echo "  ✓ node_modules present"
fi

# src-tauri/binaries/python: the Tauri Python sidecar. Required for
# cargo build (bundled into the .app); restart can't launch without it.
#
# Freshness is owned by setup-bundled-python.sh, which stamps the
# tree with sha256(PBS_URL) + sha256(requirements.txt). An existence
# check here would skip rebuilds when requirements.txt changes but
# a prior tree exists on disk. Defer unconditionally in the main
# worktree — the script is idempotent (~5s no-op when the stamp
# matches, full rebuild when it doesn't).
if [ "$IN_MAIN" = "0" ] && [ -x "$MAIN_APP/src-tauri/binaries/python/bin/python3" ]; then
    # Worker worktrees inherit main's sidecar via symlink; freshness
    # piggybacks on whatever main maintains.
    #
    # Gate on the symlink-followed interpreter (bin/python3), not the
    # parent dir. An empty src-tauri/binaries/python/ stub (e.g. one
    # created by hand to silence cargo's "resource path doesn't
    # exist" build error) would otherwise pass `-e` and silently skip
    # the symlink, leaving the Tauri app to fall through python_bin's
    # resolver to system /usr/bin/python3 (Apple-shipped 3.9.6, no
    # openai/tinfoil) and fail at the first verify_attestation or
    # pipeline call.
    if [ ! -x src-tauri/binaries/python/bin/python3 ]; then
        # Replace any empty stub that's blocking the symlink target.
        rm -rf src-tauri/binaries/python
        mkdir -p src-tauri/binaries
        ln -s "$MAIN_APP/src-tauri/binaries/python" src-tauri/binaries/python
        echo "  ✓ bundled python → $MAIN_APP/src-tauri/binaries/python (symlinked)"
    else
        echo "  ✓ bundled python present (symlinked to main)"
    fi
else
    ./scripts/setup-bundled-python.sh
fi

if [ -n "${PIPELINE_PYTHON:-}" ]; then
    echo "  ✓ pipeline python: $PIPELINE_PYTHON (override)"
else
    echo "  ✓ pipeline python: bundled sidecar (set PIPELINE_PYTHON to override)"
fi

# ── 2. Stop running instance ──────────────────────────────────────────
echo
echo "── 2. Stop running instance ──────────────────────────────────────"

if pgrep -f "target/debug/basevault" >/dev/null 2>&1; then
    echo "  killing dev build..."
    pkill -f "target/debug/basevault" 2>/dev/null || true
    sleep 1
    echo "  ✓ killed"
else
    echo "  ✓ no dev build running"
fi

PORT_PIDS="$(lsof -ti :1420 2>/dev/null || true)"
if [ -n "$PORT_PIDS" ]; then
    echo "  port 1420 held by pid(s): $PORT_PIDS — freeing..."
    echo "$PORT_PIDS" | xargs kill -9 2>/dev/null || true
    echo "  ✓ port 1420 freed"
else
    echo "  ✓ port 1420 already free"
fi

# ── 3. Ollama ─────────────────────────────────────────────────────────
echo
echo "── 3. Ollama ─────────────────────────────────────────────────────"

if pgrep -f ollama >/dev/null 2>&1; then
    echo "  restarting ollama..."
    pkill -f ollama 2>/dev/null || true
    sleep 1
fi

# Only relaunch if installed. Comment out if you never use local mode.
if command -v ollama >/dev/null 2>&1; then
    (ollama serve >/dev/null 2>&1 &)
    echo "  ✓ ollama serve started in background"
else
    echo "  ⚠ ollama not installed (skip — local mode unavailable)"
fi

# ── 4. Launch tauri dev ───────────────────────────────────────────────
echo
echo "── 4. Launch tauri dev ───────────────────────────────────────────"

LOG=/tmp/basevault-dev.log
nohup npm run tauri dev >"$LOG" 2>&1 &
disown
PID=$!

echo "  ✓ tauri dev launched (pid $PID)"
echo "  logs:  $LOG"
echo "  tail:  tail -f $LOG"
