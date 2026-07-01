#!/bin/bash
# Local notarization test for BaseVault.app.
#
# Must be run DIRECTLY ON DEVICE; it fails over SSH
#
# Usage:
#   ./scripts/notarize.sh <app-specific-password>
#   ./scripts/notarize.sh                          # prompts for password
#
# Prerequisites:
#   - Run `npm run tauri build` from app/ first (produces the .app)
#   - Developer ID Application certificate imported in keychain
#     (verify: `security find-identity -v -p codesigning`)
#
# Flow:
#   1. Find Developer ID Application identity in keychain
#   2. Recursively sign every nested binary (.dylib, .so, python/bin/*) with
#      hardened runtime + secure timestamp + Developer ID
#   3. Re-sign the main binary (Contents/MacOS/basevault)
#   4. Re-sign the outer .app bundle
#   5. Zip via ditto, submit to Apple notary, --wait for verdict
#   6. On Accepted: staple + verify; on Invalid: pull log

set -euo pipefail

# ─── Configuration — fill in before first use ──────────────────────────
APPLE_ID=admin@basevault.ai
APPLE_TEAM_ID=GB5PZY2Q63
# ───────────────────────────────────────────────────────────────────────

# Password: CLI arg if given, else prompt securely
if [ "$#" -ge 1 ] && [ -n "$1" ]; then
    APPLE_PASSWORD="$1"
else
    read -srp "Apple app-specific password: " APPLE_PASSWORD
    echo
fi

if [ -z "$APPLE_PASSWORD" ]; then
    echo "ERROR: password is empty" >&2
    exit 1
fi

# Auto-detect Developer ID Application identity
# '|| true' prevents pipefail from aborting here when grep finds no match; the
# explicit empty-check below then fires and prints the remediation message.
SIGNING_IDENTITY=$(security find-identity -v -p codesigning | \
    grep "Developer ID Application" | grep "$APPLE_TEAM_ID" | head -1 | \
    sed -E 's/.*"(Developer ID Application:[^"]+)".*/\1/' || true)

if [ -z "$SIGNING_IDENTITY" ]; then
    echo "ERROR: no 'Developer ID Application' certificate for team $APPLE_TEAM_ID in keychain" >&2
    echo "       Installed Developer ID Application certs:" >&2
    security find-identity -v -p codesigning | grep "Developer ID Application" >&2 \
        || echo "       (none)" >&2
    echo "       Import the right cert:" >&2
    echo "       security import <cert.p12> -P <password> -k ~/Library/Keychains/login.keychain-db -T /usr/bin/codesign" >&2
    exit 1
fi

echo "==> Using signing identity: $SIGNING_IDENTITY"

# Locate .app
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP="$APP_ROOT/src-tauri/target/release/bundle/macos/BaseVault.app"

if [ ! -d "$APP" ]; then
    echo "ERROR: BaseVault.app not found at $APP" >&2
    echo "       Run 'npm run tauri build' from $APP_ROOT first." >&2
    exit 1
fi

# Recursive sign — every nested binary first (deepest leaves), then containers
echo "==> Recursively signing nested binaries (.dylib, .so, python/bin/*)…"
SIGNED_COUNT=0
while IFS= read -r -d '' f; do
    codesign --force --options runtime --timestamp \
        --sign "$SIGNING_IDENTITY" "$f" 2>/dev/null || {
            echo "    WARNING: failed to sign $f"
        }
    SIGNED_COUNT=$((SIGNED_COUNT + 1))
done < <(find "$APP/Contents/Resources/python" -type f \
    \( -name "*.dylib" -o -name "*.so" -o -path "*/python/bin/*" \) \
    -print0 2>/dev/null)
echo "    Signed $SIGNED_COUNT nested binaries."

echo "==> Re-signing main binary: Contents/MacOS/basevault"
codesign --force --options runtime --timestamp \
    --sign "$SIGNING_IDENTITY" "$APP/Contents/MacOS/basevault"

echo "==> Re-signing outer .app bundle"
codesign --force --options runtime --timestamp \
    --sign "$SIGNING_IDENTITY" "$APP"

echo "==> Verifying signature locally before submit…"
codesign --verify --deep --strict --verbose=2 "$APP" || {
    echo "ERROR: local signature verification failed" >&2
    exit 1
}

ZIP="/tmp/BaseVault-notarize-$(date +%s).zip"
trap 'rm -f "$ZIP"' EXIT

echo "==> Zipping .app -> $ZIP (preserving codesign metadata)…"
/usr/bin/ditto -c -k --keepParent --sequesterRsrc "$APP" "$ZIP"

echo "==> Submitting to Apple notary (5-15 min typical on first try)…"
SUBMIT_OUT=$(xcrun notarytool submit "$ZIP" \
    --apple-id "$APPLE_ID" \
    --team-id "$APPLE_TEAM_ID" \
    --password "$APPLE_PASSWORD" \
    --wait \
    --output-format json)
echo "$SUBMIT_OUT"

UUID=$(echo "$SUBMIT_OUT" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
STATUS=$(echo "$SUBMIT_OUT" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status",""))')

if [ "$STATUS" != "Accepted" ]; then
    echo "==> Notarization failed with status: $STATUS"
    if [ -n "${UUID:-}" ]; then
        echo "==> Pulling log for ${UUID}..."
        xcrun notarytool log "${UUID}" \
            --apple-id "$APPLE_ID" \
            --team-id "$APPLE_TEAM_ID" \
            --password "$APPLE_PASSWORD" || true
    fi
    exit 1
fi

echo "==> Notarization Accepted (UUID: ${UUID})"

echo "==> Stapling .app…"
xcrun stapler staple -v "$APP"
xcrun stapler validate "$APP"

echo "==> Verifying signature + Gatekeeper…"
/usr/bin/codesign --verify --deep --strict --verbose=2 "$APP"
/usr/sbin/spctl --assess --type execute --verbose=2 "$APP"

echo
echo "==> SUCCESS. .app at $APP is notarized + stapled."
echo "    Launch test: open '$APP' on a clean Mac without xattr -cr;"
echo "    Gatekeeper should accept it without prompts."
