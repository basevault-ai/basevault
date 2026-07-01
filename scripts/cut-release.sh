#!/usr/bin/env bash
# Release a tagged version.
#
# Tags `v<version>` on a target ref and pushes to origin, which triggers
# `.github/workflows/release.yml` — the workflow builds, signs, notarizes,
# uploads version-pinned `BaseVault_<version>_aarch64.dmg` +
# `BaseVault_<version>_aarch64.app.tar.gz` (+ `.sig`) to
# `s3://basevault-releases/`, writes the auto-updater channel
# manifest(s), and creates a GitHub Release.
#
# Two update channels, routed by the version's semver pre-release
# suffix. A stable tag (`X.Y.Z`) writes both `latest.json` (stable
# channel) and `latest-rc.json` (rc channel) — stable is the newest
# build for everyone. An rc tag (`X.Y.Z-rcN`) writes only
# `latest-rc.json`, so stable users never see it; rc-channel users
# roll forward onto the next stable by semver precedence. The app's
# Settings → Development "Include release candidates" toggle (default
# OFF) selects which channel the updater checks.
#
# Usage:
#   scripts/cut-release.sh <version> [<ref>]        # tag <ref> as v<version>
#   scripts/cut-release.sh <version>                # tag current HEAD
#   scripts/cut-release.sh --dry-run <version> [<ref>]
#
# Examples:
#   scripts/cut-release.sh 0.1.23                   # tag HEAD as v0.1.23
#   scripts/cut-release.sh 0.1.23-rc1 abc1234        # tag a specific commit
#   scripts/cut-release.sh 0.1.23-rc2 experiment/x  # tag tip of a local branch
#   scripts/cut-release.sh 0.1.23-rc2 origin/x      # tag tip of a remote branch
#                                                # (run `git fetch` first)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

VERSION="${1:-}"
REF="${2:-HEAD}"

if [[ -z "$VERSION" ]]; then
  cat >&2 <<'EOF'
usage: scripts/cut-release.sh [--dry-run] <version> [<ref>]

Examples:
  scripts/cut-release.sh 0.1.23                   # tag HEAD as v0.1.23
  scripts/cut-release.sh 0.1.23-rc1 abc1234        # tag a specific commit
  scripts/cut-release.sh 0.1.23-rc2 experiment/x  # tag tip of a local branch
  scripts/cut-release.sh 0.1.23-rc2 origin/x      # tag tip of a remote branch
EOF
  exit 1
fi

# Validate version: <digits>.<digits>.<digits> with optional -<suffix>
# (rc1, beta2, etc). Reject things like "v0.1.23" — script always prepends
# `v` itself — or "0.1" — workflow expects three-segment semver in
# CARGO_PKG_VERSION via `cargo update --precise`.
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
  echo "error: version must look like X.Y.Z or X.Y.Z-rcN, got '$VERSION'" >&2
  echo "       (don't include the leading 'v' — script adds it)" >&2
  exit 1
fi

TAG="v$VERSION"

run() {
  if (( DRY_RUN )); then
    echo "DRY-RUN: $*"
  else
    echo "+ $*"
    "$@"
  fi
}

cd "$REPO_ROOT"

echo ">>> Resolving target ref: $REF"
if ! TARGET_SHA=$(git rev-parse --verify "${REF}^{commit}" 2>/dev/null); then
  echo "error: ref '$REF' not found locally" >&2
  echo "       (remote-tracking branches need 'git fetch' first)" >&2
  exit 1
fi
echo "    $REF → $TARGET_SHA"
echo "    $(git log -1 --pretty=format:'%h %s' "$TARGET_SHA")"

echo ">>> Checking that tag '$TAG' doesn't already exist"
if git rev-parse --verify "refs/tags/$TAG" >/dev/null 2>&1; then
  echo "error: tag '$TAG' already exists locally" >&2
  echo "       (delete with 'git tag -d $TAG' if you meant to retag)" >&2
  exit 1
fi
if git ls-remote --tags origin "refs/tags/$TAG" 2>/dev/null | grep -q .; then
  echo "error: tag '$TAG' already exists on origin" >&2
  echo "       (deleting a remote tag is destructive — confirm intent first)" >&2
  exit 1
fi

echo ">>> Tagging $TARGET_SHA as $TAG"
run git tag "$TAG" "$TARGET_SHA"

echo ">>> Pushing tag to origin"
run git push origin "$TAG"

if (( DRY_RUN )); then
  echo ">>> Dry run complete; no tag created or pushed."
  exit 0
fi

echo ""
echo "Tag pushed. Release workflow should be queued at:"
echo "  https://github.com/basevault-ai/basevault/actions/workflows/release.yml"
echo ""
echo "Build typically takes 8-12 minutes. Final artifacts will land at:"
echo "  https://basevault-releases.s3.amazonaws.com/BaseVault_${VERSION}_aarch64.dmg"
echo ""
echo "Note: latest.json is overwritten on every release (including RCs)."
echo "After RC testing, tag a non-prerelease version to flip the auto-"
echo "updater channel back to stable."
