#!/usr/bin/env bash
# Materialize a self-contained Python interpreter for the BaseVault .app
# bundle, with pipeline deps pre-installed.
#
# Output: src-tauri/binaries/python/  (gitignored, ~120MB on disk)
# Idempotent: skips re-download + reinstall if the existing tree was built
# from the same requirements.txt + same Python URL.
#
# Run when:
#   - First time setup on a new machine
#   - python/engine/requirements.txt changes
#   - PBS_URL below changes (Python version bump)

set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY_DIR="$APP_DIR/src-tauri/binaries/python"
REQS="$APP_DIR/python/engine/requirements.txt"
STAMP="$PY_DIR/.basevault-stamp"
TAURI_CONF="$APP_DIR/src-tauri/tauri.conf.json"

# Single source of truth for the min-supported macOS: tauri.conf.json's
# bundle.macOS.minimumSystemVersion (which also drives LSMinimumSystemVersion
# and the Rust build's deployment target). MLX ships a separate wheel per
# macOS target, so the bundled MLX binaries are pinned to this floor — see
# the install step below.
MIN_MACOS="$(python3 -c "import json; print(json.load(open('$TAURI_CONF'))['bundle']['macOS']['minimumSystemVersion'])")"
MACOS_WHEEL_TAG="macosx_${MIN_MACOS//./_}_arm64"

# Pinned: cpython 3.14.4 for aarch64-apple-darwin, stripped variant (~22MB compressed).
# https://github.com/astral-sh/python-build-standalone/releases/tag/20260414
#
# PBS_URL and PBS_SHA256 move together on every version bump: PBS_SHA256 is the
# sha256 of the exact tarball PBS_URL points at, taken from that release's
# canonical SHA256SUMS asset and verified after download / before extraction
# (see below). On a version bump, update BOTH — a stale hash fails the build
# loudly rather than silently trusting a different tarball. Do NOT fetch the
# hash at build time: pulling the expected digest over the same channel as the
# artifact it guards defeats the check.
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20260414/cpython-3.14.4%2B20260414-aarch64-apple-darwin-install_only_stripped.tar.gz"
PBS_SHA256="6f304f4ec30854611f23316578302235fb517cd970519ecdd11a8c4db87fd843"

# Compute target stamp from inputs — if the existing stamp matches, skip.
# Includes PBS_SHA256 so bumping the pinned hash (which always accompanies a
# PBS_URL change, but also guards against an in-place edit) forces a clean
# wipe + rebuild rather than reusing a tree built from a different tarball.
WANT_STAMP="$(printf '%s\n' "$PBS_URL" | shasum -a 256 | cut -d' ' -f1)-$(shasum -a 256 "$REQS" | cut -d' ' -f1)-${PBS_SHA256}"

if [[ -f "$STAMP" ]] && [[ "$(cat "$STAMP")" == "$WANT_STAMP" ]]; then
    echo "Bundled Python already up to date (stamp matches): $PY_DIR"
    exit 0
fi

echo "Materializing bundled Python at $PY_DIR"
echo "  url: $PBS_URL"
echo "  reqs: $REQS"

# Wipe + recreate
rm -rf "$PY_DIR"
mkdir -p "$PY_DIR"

# Download + extract
TMP_TAR="$(mktemp -t basevault-pbs.XXXXXX.tar.gz)"
trap 'rm -f "$TMP_TAR"' EXIT

echo "Downloading $(basename "$PBS_URL")..."
curl -fL --progress-bar -o "$TMP_TAR" "$PBS_URL"

# Verify the interpreter tarball BEFORE extraction. The pip wheels installed on
# top are hash-pinned (--require-hashes against requirements.txt), but the
# interpreter underneath them is the weakest link — a MITM, compromised CDN, or
# tampered release asset would otherwise be baked into the bundle unchecked.
# Fail-closed, same posture as the attestation gate: an empty pin or a mismatch
# deletes the download and aborts the build rather than trusting it.
if [[ -z "$PBS_SHA256" ]]; then
    echo "ERROR: PBS_SHA256 is empty — refusing to extract an unverified interpreter tarball." >&2
    echo "  Pin it to the sha256 of $(basename "$PBS_URL") from the release's SHA256SUMS." >&2
    rm -f "$TMP_TAR"
    exit 1
fi
echo "Verifying tarball sha256..."
GOT_SHA256="$(shasum -a 256 "$TMP_TAR" | cut -d' ' -f1)"
if [[ "$GOT_SHA256" != "$PBS_SHA256" ]]; then
    echo "ERROR: PBS tarball sha256 mismatch — aborting (possible tampered/corrupt download)." >&2
    echo "  expected: $PBS_SHA256" >&2
    echo "  got:      $GOT_SHA256" >&2
    echo "  url:      $PBS_URL" >&2
    rm -f "$TMP_TAR"
    exit 1
fi
echo "  ok: sha256 $GOT_SHA256"

echo "Extracting..."
# PBS tarballs have a top-level `python/` dir — strip it so we get $PY_DIR/bin/, etc.
tar -xzf "$TMP_TAR" -C "$PY_DIR" --strip-components=1

if [[ ! -x "$PY_DIR/bin/python3" ]]; then
    echo "ERROR: extracted tarball does not contain bin/python3" >&2
    exit 1
fi

# Install pipeline deps into the bundled Python. mlx-lm and its
# transitive closure (mlx, transformers, tokenizers, safetensors,
# huggingface_hub, …) are pinned with wheel hashes in requirements.txt
# alongside everything else, so the whole runtime is one coherent
# --require-hashes install. MLX is the primary local-inference path;
# huggingface_hub is also imported directly by download_model.py and is
# in the pinned set, so the in-app downloader's dependency is explicit.
#
# We deliberately do NOT `pip install --upgrade pip wheel` first: that pulls
# pip + wheel from PyPI UNPINNED, and pip is the very tool that enforces
# --require-hashes on every line below — a tampered pip fetched here would
# subvert the whole guarantee. The PBS interpreter ships a current pip + wheel
# (verified above by tarball hash), so the upgrade is both redundant and the
# last unpinned download in the build path. Rely on the bundled pip.
echo "Installing pipeline requirements..."
"$PY_DIR/bin/python3" -m pip install -r "$REQS"

# MLX publishes one wheel per macOS deployment target (e.g. 14.0 / 15.0 /
# 26.0); pip picks the one matching the build host, so a bundle built on a
# newer macOS embeds a libmlx.dylib referencing libc++ symbols absent on
# older OSes and dies at `import mlx.core`. Reinstall the MLX binaries pinned
# to the min-supported target so the bundle imports on every macOS we support.
# Other compiled deps top out at or below this floor, so only MLX needs the
# override. Hashes stay sourced from requirements.txt (the same pinned set the
# main install used) so the whole runtime remains hash-verified.
echo "Pinning MLX binaries to macOS $MIN_MACOS ($MACOS_WHEEL_TAG)..."
SITE_PACKAGES="$("$PY_DIR/bin/python3" -c 'import site; print(site.getsitepackages()[0])')"
MLX_REQS="$(mktemp -t basevault-mlx-reqs.XXXXXX)"
trap 'rm -f "$TMP_TAR" "$MLX_REQS"' EXIT
awk '
    /^mlx==/ {grab=1}
    /^mlx-metal==/ {grab=1}
    /^[A-Za-z0-9._-]+==/ && !/^mlx==/ && !/^mlx-metal==/ {grab=0}
    grab {print}
' "$REQS" > "$MLX_REQS"
"$PY_DIR/bin/python3" -m pip install \
    --platform "$MACOS_WHEEL_TAG" --only-binary=:all: \
    --target "$SITE_PACKAGES" --upgrade --force-reinstall --no-deps \
    --require-hashes -r "$MLX_REQS"

# Relocate any Homebrew-linked dylibs (e.g. rfc3161_client/_rust.abi3.so
# linking to /opt/homebrew/opt/openssl@3/lib/libssl.3.dylib) into the
# sidecar so the bundle is fully self-contained. Required for hardened
# runtime under notarization — see scripts/relocate-bundled-libs.py.
echo "Relocating Homebrew-linked libraries into sidecar..."
python3 "$APP_DIR/scripts/relocate-bundled-libs.py" "$PY_DIR"

# install_name_tool invalidates the code signature on every file it
# modifies. Re-apply an ad-hoc signature so the libraries can load
# again. In CI, the recursive-sign workflow step replaces these
# ad-hoc signatures with Developer ID + hardened runtime before
# notarization. For local dev builds, ad-hoc is sufficient.
echo "Re-applying ad-hoc signatures to modified libraries..."
find "$PY_DIR" -type f \( -name "*.dylib" -o -name "*.so" \) -print0 | \
    xargs -0 codesign --force -s - 2>&1 | grep -v "replacing existing signature" || true

# Sanity check: the deps that runner.py preflights for, plus the package
# whose Rust extension links to relocated openssl.
echo "Verifying critical imports..."
"$PY_DIR/bin/python3" -c "import dotenv, ollama, openai; print('  ok: dotenv, ollama, openai')"
"$PY_DIR/bin/python3" -c "import rfc3161_client; print('  ok: rfc3161_client (relocated openssl)')"
"$PY_DIR/bin/python3" -c "import mlx, mlx_lm, huggingface_hub; print('  ok: mlx, mlx_lm, huggingface_hub')"

# Guard: the bundled libmlx.dylib must target the min-supported macOS, or
# `import mlx.core` dlopen-fails on older OSes. Pins the wheel-override above —
# a host-target wheel sneaking back in fails the build here, not on a user's
# older Mac.
DYLIB="$(find "$SITE_PACKAGES" -name 'libmlx.dylib' | head -1)"
MINOS="$(otool -l "$DYLIB" | awk '/LC_BUILD_VERSION/{f=1} f&&/^ *minos/{print $2; exit}')"
if [[ "$MINOS" != "$MIN_MACOS" ]]; then
    echo "ERROR: bundled libmlx.dylib targets macOS $MINOS, expected $MIN_MACOS (min-supported floor)." >&2
    echo "  dylib: $DYLIB" >&2
    exit 1
fi
echo "  ok: libmlx.dylib targets macOS $MINOS (min-supported floor)"

# Write stamp last — only after everything succeeded.
echo "$WANT_STAMP" > "$STAMP"

echo
echo "Done. Bundled Python ready at $PY_DIR"
echo "Size: $(du -sh "$PY_DIR" | cut -f1)"
