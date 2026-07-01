#!/usr/bin/env bash
# Release pipeline: setup bundled Python (idempotent) → build .app → smoke test.
# Run from anywhere. Exits non-zero on any failure with the relevant log tail.
#
# Smoke test: bundled python + bundled runner against a tiny synthetic input
# in TEE mode (attested cloud — the production route). Asserts the run
# produced a non-empty extract stage. Confirms the .app would work for a
# user who just double-clicked it.
#
# Usage:
#   scripts/manual-release.sh             # build + smoke
#   scripts/manual-release.sh --skip-smoke  # build only

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS_DIR="$APP_DIR/scripts"
RELEASE_APP="$APP_DIR/release/BaseVault.app"
SMOKE=1
for arg in "$@"; do
    [[ "$arg" == "--skip-smoke" ]] && SMOKE=0
done

# ── 1. Bundled Python sidecar (idempotent) ────────────────────────────
echo "── 1. Bundled Python ─────────────────────────────────────────────"
"$SCRIPTS_DIR/setup-bundled-python.sh"

# ── 2. Build .app + DMG ───────────────────────────────────────────────
echo
echo "── 2. tauri build ────────────────────────────────────────────────"
RELEASE_DIR="$APP_DIR/release"

# Detach stale DMG mounts so Tauri's bundle_dmg.sh doesn't choke on
# /Volumes/BaseVault-1 / -2 / ... left over from prior runs.
shopt -s nullglob 2>/dev/null || true
for v in /Volumes/BaseVault*; do
    [ -d "$v" ] && hdiutil detach "$v" -quiet 2>/dev/null || true
done

cd "$APP_DIR"
npm run release

echo
echo "Artifacts:"
ls -la "$RELEASE_DIR"

# ── 2b. Trust-surface bundle assert ───────────────────────────────────
# Closes #860 (1): the dev-only eval tree under testing/ carries
# non-attested ModelSpecs + dispatchers + clients (the canonical eval-
# side home per #839). Those must NEVER ship in the user-facing .app.
# The runtime gut + the production ↛ testing import gate (gate b of
# trust_gates) defend the runtime path; this assert defends the
# build artifact against a future tauri.conf.json change that
# broadens bundle.resources into a glob that absorbs testing/. The
# package root bundles to Contents/Resources/python/ (engine + kernel),
# so the eval tree would land at Resources/python/testing/ — assert that
# path doesn't exist.
echo
echo "── 2b. Trust-surface bundle assert ───────────────────────────────"
if [[ -e "$RELEASE_APP/Contents/Resources/python/testing" ]]; then
    echo "✗ TRUST GUT VIOLATION: testing/ eval tree shipped in release bundle." >&2
    ls -R "$RELEASE_APP/Contents/Resources/python/testing" >&2
    echo "  The eval tree under python/testing/ must never ship in the .app." >&2
    echo "  Audit tauri.conf.json bundle.resources for an over-broad glob." >&2
    exit 1
fi
echo "✓ bundle clean: no testing/ tree under Contents/Resources/python"

# Convenience symlink so `scp host:BaseVault.dmg ...` works from other machines.
if [[ ! -e "$HOME/BaseVault.dmg" ]]; then
    ln -s "$RELEASE_DIR/BaseVault.dmg" "$HOME/BaseVault.dmg"
fi

[[ "$SMOKE" -eq 0 ]] && { echo "Skipping smoke (--skip-smoke)."; exit 0; }

# ── 3. Smoke test ─────────────────────────────────────────────────────
echo
echo "── 3. Smoke test (TEE mode) ──────────────────────────────────────"

# The package root (engine + kernel) bundles to Resources/python/, the same
# tree that carries the interpreter (bin/). Launch the runner as a module
# from that root — `python -m engine.runner` with cwd=Resources/python — so
# its fully-qualified engine.*/kernel.* imports resolve.
BUNDLED_PY_ROOT="$RELEASE_APP/Contents/Resources/python"
BUNDLED_PY="$BUNDLED_PY_ROOT/bin/python3"
[[ -x "$BUNDLED_PY" ]] || { echo "ERROR: missing $BUNDLED_PY" >&2; exit 1; }
[[ -f "$BUNDLED_PY_ROOT/engine/runner.py" ]] || { echo "ERROR: missing $BUNDLED_PY_ROOT/engine/runner.py" >&2; exit 1; }

# Tiny synthetic input — generic prose, no PII, deterministic.
SMOKE_DIR="$(mktemp -d -t basevault-smoke.XXXXXX)"
trap 'rm -rf "$SMOKE_DIR"' EXIT
SMOKE_INPUT="$SMOKE_DIR/journal.txt"
cat > "$SMOKE_INPUT" <<'EOF'
Tuesday morning. Got up at six, did twenty minutes of stretching,
made coffee. Today's plan: finish the migration script for the
billing service, then a one-on-one with Alice about the Q3 roadmap.
Sleep was short — went to bed late reading. Need to set a hard
cutoff at ten pm.

Email from the landlord — rent goes up by 4% next quarter. Need to
budget for that.
EOF

SMOKE_SESSION="$(date -u +%Y-%m-%dT%H-%M-%SZ)-experiment-release-smoke"
SMOKE_LOG="/tmp/basevault-smoke.log"

echo "  python:  $BUNDLED_PY"
echo "  runner:  -m engine.runner  (cwd=$BUNDLED_PY_ROOT)"
echo "  input:   $SMOKE_INPUT"
echo "  session: $SMOKE_SESSION"
echo

# env -i wipes shell env to mimic a Finder launch (no shell PATH/keys).
# Keep only HOME so .env discovery in ~/Library/Application Support/ works.
set +e
# Pass --subject explicitly so the smoke is hermetic — doesn't depend on
# whether the user has run the wizard (config.json subject) or has the
# legacy BASEVAULT_SUBJECT in their dotenv.
( cd "$BUNDLED_PY_ROOT" && \
  env -i HOME="$HOME" PATH=/usr/bin:/bin BASEVAULT_SESSION="$SMOKE_SESSION" \
    "$BUNDLED_PY" -m engine.runner --paths "$SMOKE_INPUT" --mode tee \
    --subject "Sample" ) \
    > "$SMOKE_LOG" 2>&1
SMOKE_EXIT=$?
set -e

if [[ $SMOKE_EXIT -ne 0 ]]; then
    echo "✗ Smoke FAILED (exit=$SMOKE_EXIT). Last 30 lines:"
    tail -30 "$SMOKE_LOG"
    echo
    echo "Full log: $SMOKE_LOG"
    exit "$SMOKE_EXIT"
fi

# Parse the final done event to assert non-empty extraction.
DONE_LINE="$(grep -E '"stage": "done"' "$SMOKE_LOG" | tail -1 || true)"
if [[ -z "$DONE_LINE" ]]; then
    echo "✗ Smoke FAILED: no done event in output. Last 30 lines:"
    tail -30 "$SMOKE_LOG"
    exit 1
fi

# Crude but dependency-free: extract facts/patterns counts from the JSON.
FACTS="$(printf '%s\n' "$DONE_LINE" | sed -E 's/.*"facts": *([0-9]+).*/\1/')"
if [[ -z "$FACTS" || "$FACTS" -eq 0 ]]; then
    echo "✗ Smoke FAILED: 0 facts extracted (silent-empty bug regression?)."
    echo "  done: $DONE_LINE"
    echo "  Full log: $SMOKE_LOG"
    exit 1
fi

echo "✓ Smoke PASSED: $FACTS facts extracted via TEE."
echo "  done: $DONE_LINE"
echo
echo "Release ready: $APP_DIR/release/BaseVault.dmg"
