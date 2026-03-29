#!/usr/bin/env bash
# Download only the ADBench Classical .npz files via git sparse-checkout.
# Avoids cloning the full repo (which includes large CV/NLP .npz files).
set -e

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/adbench_data/classical"
REPO_URL="https://github.com/Minqi824/ADBench.git"
TMP_DIR="$(mktemp -d)"

echo "[*] Downloading ADBench Classical datasets to: $DEST"
echo "[*] Using temp dir: $TMP_DIR"

# Sparse clone — only fetch tree, no blobs yet
git clone \
    --filter=blob:none \
    --no-checkout \
    --depth=1 \
    --sparse \
    "$REPO_URL" \
    "$TMP_DIR/ADBench"

cd "$TMP_DIR/ADBench"
git sparse-checkout set "adbench/datasets/Classical"
git checkout

# Count how many .npz files we got
N=$(find adbench/datasets/Classical -name "*.npz" | wc -l)
echo "[*] Found $N .npz files"

if [ "$N" -eq 0 ]; then
    echo "[!] No files downloaded. Check network access or repo structure." >&2
    exit 1
fi

mkdir -p "$DEST"
cp adbench/datasets/Classical/*.npz "$DEST/"
echo "[+] Copied $N files to $DEST"

# Cleanup
cd /
rm -rf "$TMP_DIR"
echo "[+] Done."
