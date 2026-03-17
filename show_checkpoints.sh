#!/bin/bash
# Show the highest numbered checkpoint available for each model in outputs/train.
# Usage: bash show_checkpoints.sh [prefix] [root_dir]
# Examples:
#   bash show_checkpoints.sh              # all models
#   bash show_checkpoints.sh v4_          # only models starting with v4_
#   bash show_checkpoints.sh v2_ outputs/train

PREFIX="${1:-}"
ROOT="${2:-outputs/train}"

printf "%-60s %s\n" "MODEL" "HIGHEST CKPT"
printf "%s\n" "$(printf '%.0s-' {1..80})"

for model_dir in "$ROOT"/${PREFIX}*/; do
    [ -d "$model_dir" ] || continue
    model_name=$(basename "$model_dir")
    ckpt_dir="$model_dir/checkpoints"
    if [ ! -d "$ckpt_dir" ]; then
        printf "%-60s %s\n" "$model_name" "(no checkpoints/)"
        continue
    fi
    # Find highest numbered .pt file (ignoring last.pt)
    best=$(ls "$ckpt_dir"/*.pt 2>/dev/null \
        | xargs -I{} basename {} .pt \
        | grep -E '^[0-9]+$' \
        | sort -n \
        | tail -1)
    if [ -n "$best" ]; then
        printf "%-60s %s\n" "$model_name" "${best}"
    else
        # Only last.pt exists
        if [ -f "$ckpt_dir/last.pt" ]; then
            printf "%-60s %s\n" "$model_name" "(only last.pt)"
        else
            printf "%-60s %s\n" "$model_name" "(empty)"
        fi
    fi
done
