#!/usr/bin/env bash
# Download the Cactus-Compute Gemma 4 E2B INT4 model and push it to a
# connected Android device running Sakhi. One command from clone → on-device
# model ready.
#
# Prerequisites:
#   1. The Sakhi APK must already be INSTALLED on the phone (debuggable),
#      because we use `run-as com.sakhi.app` to write into app-private storage.
#      Build/install instructions are in the main README.
#   2. `adb` must be on PATH and the phone must show as an authorized device
#      (`adb devices` lists it with state "device", not "unauthorized").
#   3. `curl` and `unzip` must be available (unzip runs on-device via
#      adb shell, so only the host `curl` is needed).
#   4. You must have ACCEPTED the terms on the Cactus-Compute model page:
#        https://huggingface.co/Cactus-Compute/gemma-4-E2B-it
#      and have a HuggingFace access token available. The script reads, in
#      order: $HF_TOKEN env var, then ~/.cache/huggingface/token. If neither
#      exists the script bails with instructions.
#
# Tested on: Windows 11 + Git Bash, macOS 14, Ubuntu 22.04.
# adb wireless and USB both work — the script is transport-agnostic.
set -euo pipefail

MODEL_REPO="Cactus-Compute/gemma-4-E2B-it"
MODEL_FILE="gemma-4-e2b-it-int4.zip"   # NOTE: lowercase despite repo casing
MODEL_URL="https://huggingface.co/${MODEL_REPO}/resolve/main/weights/${MODEL_FILE}"
PKG="com.sakhi.app"
ON_DEVICE_DIR="files/models/gemma-4-e2b"   # relative to /data/data/$PKG/

DOWNLOAD_DIR="${DOWNLOAD_DIR:-./models/cactus}"
LOCAL_ZIP="${DOWNLOAD_DIR}/${MODEL_FILE}"

# ── Token resolution ────────────────────────────────────────────────────────
resolve_hf_token() {
  if [ -n "${HF_TOKEN:-}" ]; then
    echo "$HF_TOKEN"
    return 0
  fi
  if [ -f "$HOME/.cache/huggingface/token" ]; then
    cat "$HOME/.cache/huggingface/token"
    return 0
  fi
  return 1
}

if ! TOKEN=$(resolve_hf_token); then
  cat >&2 <<EOF
ERROR: no HuggingFace token found.
This model is gated — you must accept the terms on HuggingFace before the
download will work.

Steps:
  1. Open https://huggingface.co/${MODEL_REPO} and click "Agree and access".
  2. Create a read token at https://huggingface.co/settings/tokens.
  3. Export it:  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
     OR save to ~/.cache/huggingface/token (huggingface-cli login).
  4. Re-run this script.

Without completing step 1, the download will return HTTP 403 even with a
valid token.
EOF
  exit 1
fi

# ── adb sanity checks ───────────────────────────────────────────────────────
if ! command -v adb >/dev/null 2>&1; then
  echo "ERROR: adb not on PATH. Install Android platform-tools first." >&2
  exit 1
fi

DEVICE_COUNT=$(adb devices | awk 'NR>1 && $2=="device" {count++} END {print count+0}')
if [ "$DEVICE_COUNT" -eq 0 ]; then
  echo "ERROR: no authorized adb device connected. Run 'adb devices' to diagnose." >&2
  exit 1
fi
if [ "$DEVICE_COUNT" -gt 1 ]; then
  echo "WARNING: multiple adb devices connected — the first one will be used." >&2
  echo "         Set ANDROID_SERIAL=<serial> to target a specific device." >&2
fi

# Confirm Sakhi is installed and debuggable (run-as needs both).
if ! adb shell "run-as $PKG true" >/dev/null 2>&1; then
  cat >&2 <<EOF
ERROR: cannot run-as $PKG on the connected device.
This means one of:
  (a) The Sakhi APK is not installed. Build + install it first (see README).
  (b) The installed APK is not a debuggable build (release APKs cannot run-as).
      Use ./gradlew assembleDebug output, not assembleRelease.

If you just installed the APK, wait ~3 seconds and re-run.
EOF
  exit 1
fi

# ── Download (skip if already present) ──────────────────────────────────────
mkdir -p "$DOWNLOAD_DIR"
if [ -f "$LOCAL_ZIP" ]; then
  echo "✓ Model zip already present at $LOCAL_ZIP — skipping download."
else
  echo "→ Downloading $MODEL_FILE (~4.4 GB)..."
  curl -L --fail-with-body \
       -H "Authorization: Bearer $TOKEN" \
       -o "$LOCAL_ZIP" \
       "$MODEL_URL"
  echo "✓ Downloaded to $LOCAL_ZIP"
fi

# Size sanity — the zip is known ~4.4 GB; anything smaller is a 403 body or
# a partial transfer.
ZIP_BYTES=$(wc -c < "$LOCAL_ZIP")
if [ "$ZIP_BYTES" -lt 4000000000 ]; then
  echo "ERROR: downloaded file is only $ZIP_BYTES bytes (expected ~4.4 GB)." >&2
  echo "       Likely HTTP error body saved as zip. Inspect: head -c 500 '$LOCAL_ZIP'" >&2
  exit 1
fi

# ── Push + extract on-device ───────────────────────────────────────────────
# Git-Bash path-mangling guard: /data/local/tmp would become C:\Program Files\Git\data\...
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"

TMP_ON_DEVICE="/data/local/tmp/${MODEL_FILE}"

echo "→ Pushing zip to $TMP_ON_DEVICE (this can take a few minutes over USB/wireless)..."
adb push "$LOCAL_ZIP" "$TMP_ON_DEVICE"

echo "→ Creating app-private model directory..."
adb shell "run-as $PKG mkdir -p $ON_DEVICE_DIR"

echo "→ Extracting in-place (app-private storage)..."
adb shell "run-as $PKG sh -c 'cd $ON_DEVICE_DIR && unzip -o $TMP_ON_DEVICE'"

echo "→ Removing pushed zip from /data/local/tmp..."
adb shell "rm $TMP_ON_DEVICE"

# ── Verify ─────────────────────────────────────────────────────────────────
CONFIG_EXISTS=$(adb shell "run-as $PKG sh -c 'test -f $ON_DEVICE_DIR/config.txt && echo YES || echo NO'" | tr -d '\r\n')
if [ "$CONFIG_EXISTS" != "YES" ]; then
  echo "ERROR: config.txt not found under $ON_DEVICE_DIR after extract. Unzip likely failed." >&2
  exit 1
fi

FILE_COUNT=$(adb shell "run-as $PKG sh -c 'ls $ON_DEVICE_DIR | wc -l'" | tr -d '\r\n')
echo "✓ Model extracted: $FILE_COUNT files under /data/data/$PKG/$ON_DEVICE_DIR"
echo ""
echo "Next: open the Sakhi app → Field Mode → On-Device Probe → Check Status,"
echo "      then Load Model. First load takes ~10 s."
