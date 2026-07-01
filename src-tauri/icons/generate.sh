#!/usr/bin/env bash
# Regenerate icon.png and icon.icns from icon.svg using only macOS built-ins
# (qlmanage, sips, iconutil). Run after editing icon.svg.
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f icon.svg ]]; then
  echo "icon.svg not found in $(pwd)" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# 1. SVG -> 1024x1024 PNG via sips (preserves alpha — qlmanage flattens onto white)
src_png="$tmp/icon-1024.png"
sips -s format png -z 1024 1024 icon.svg --out "$src_png" >/dev/null

# 2. Master icon.png
cp "$src_png" icon.png

# 3. .iconset -> .icns
iconset="$tmp/icon.iconset"
mkdir -p "$iconset"
for spec in \
  "16    icon_16x16.png" \
  "32    icon_16x16@2x.png" \
  "32    icon_32x32.png" \
  "64    icon_32x32@2x.png" \
  "128   icon_128x128.png" \
  "256   icon_128x128@2x.png" \
  "256   icon_256x256.png" \
  "512   icon_256x256@2x.png" \
  "512   icon_512x512.png" \
  "1024  icon_512x512@2x.png"
do
  size="${spec%% *}"
  name="${spec##* }"
  sips -z "$size" "$size" "$src_png" --out "$iconset/$name" >/dev/null
done

iconutil -c icns "$iconset" -o icon.icns

echo "Regenerated:"
ls -la icon.svg icon.png icon.icns
