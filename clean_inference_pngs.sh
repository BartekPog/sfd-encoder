#!/bin/bash
# Recursively delete all .png files from outputs/inference/
# while preserving everything else (fid_result.txt, fid_result.json, etc.)
#
# Usage:
#   bash clean_inference_pngs.sh          # dry-run (shows count only)
#   bash clean_inference_pngs.sh --delete # actually delete

set -euo pipefail

# TARGET_DIR="outputs/inference"
TARGET_DIR="outputs/inference/v4_mse01_cos001_merged_noisy_enc_0100000"

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Directory '$TARGET_DIR' not found."
    exit 1
fi

COUNT=$(find "$TARGET_DIR" -type f -name "*.png" | wc -l)
echo "Found ${COUNT} .png files in ${TARGET_DIR}"

if [[ "${1:-}" == "--delete" ]]; then
    echo "Deleting..."
    find "$TARGET_DIR" -type f -name "*.png" -delete
    echo "Done. Removed ${COUNT} .png files."
    # Remove now-empty directories
    find "$TARGET_DIR" -type d -empty -delete 2>/dev/null || true
    echo "Cleaned up empty directories."
else
    echo "Dry run. Re-run with --delete to actually remove them."
fi
