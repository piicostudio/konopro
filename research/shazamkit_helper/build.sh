#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$ROOT_DIR/KonoproShazamHelper.xcodeproj"
SCHEME="KonoproShazamHelper"
CONFIGURATION="${CONFIGURATION:-Debug}"
BUNDLE_ID="${KONOPRO_SHAZAMKIT_BUNDLE_ID:-com.konopro.research.shazamhelper}"
DERIVED_DATA_PATH="${DERIVED_DATA_PATH:-$ROOT_DIR/build/DerivedData}"
OUTPUT_APP="$ROOT_DIR/build/KonoproShazamHelper.app"

if [[ -z "${KONOPRO_DEVELOPMENT_TEAM:-}" ]]; then
  cat >&2 <<EOF
KONOPRO_DEVELOPMENT_TEAM is required.

Use your Apple Developer Team ID:
  KONOPRO_DEVELOPMENT_TEAM=TEAMID $0

The App ID must use this bundle ID and have App Services > ShazamKit enabled:
  $BUNDLE_ID
EOF
  exit 2
fi

xcodebuild \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -configuration "$CONFIGURATION" \
  -derivedDataPath "$DERIVED_DATA_PATH" \
  -destination "generic/platform=macOS" \
  -allowProvisioningUpdates \
  DEVELOPMENT_TEAM="$KONOPRO_DEVELOPMENT_TEAM" \
  PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
  build

BUILT_APP="$DERIVED_DATA_PATH/Build/Products/$CONFIGURATION/KonoproShazamHelper.app"
rm -rf "$OUTPUT_APP"
mkdir -p "$ROOT_DIR/build"
ditto "$BUILT_APP" "$OUTPUT_APP"

cat <<EOF
Built ShazamKit helper:
  $OUTPUT_APP

The Python fingerprinting lab will auto-detect:
  $OUTPUT_APP/Contents/MacOS/KonoproShazamHelper
EOF
