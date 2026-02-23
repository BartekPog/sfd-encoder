"""
Collect all fid_result.json files under outputs/inference/ and merge
them into a single CSV (and optionally pretty-print a summary table).

Usage:
    python collect_fid_results.py
    python collect_fid_results.py --root outputs/inference --out results/fid_summary.csv
    python collect_fid_results.py --sort fid50k
    python collect_fid_results.py --sort fid50k --filter inference_type=linear
"""

import argparse
import csv
import json
from pathlib import Path

# Column order in the CSV; extra fields found in JSONs are appended at the end.
COLUMN_ORDER = [
    "exp_name",
    "sample_dir",
    "inference_type",
    "ckpt_step",
    "fid50k",
    "num_steps",
    "steps_per_pass",
    "sampler",
    # architecture
    "model_type",
    "use_hidden_tokens",
    "num_hidden_tokens",
    "share_timestep_embedder",
    "share_patch_embedder",
    "use_per_token_encoding",
    # hidden token losses
    "hidden_weight",
    "hidden_cos_weight",
    "hidden_reg_weight",
    "hidden_same_t_as_img",
    "normalize_hidden",
    # REPA
    "use_repa",
    "repa_weight",
    "repa_mode",
    # other
    "semantic_weight",
    "train_max_steps",
    "weight_init",
    "config",
    "timestamp",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/inference",
                        help="Root directory to search for fid_result.json files")
    parser.add_argument("--out",  default="results/fid_summary.csv",
                        help="Output CSV path")
    parser.add_argument("--sort", default="fid50k",
                        help="Column to sort by (use 'none' to skip sorting)")
    parser.add_argument("--filter", default=None,
                        help="Filter records by field value, e.g. 'inference_type=linear'")
    args = parser.parse_args()

    root = Path(args.root)
    json_files = sorted(root.glob("**/fid_result.json"))

    if not json_files:
        print(f"No fid_result.json files found under {root}")
        return

    records = []
    for jf in json_files:
        with open(jf) as f:
            try:
                records.append(json.load(f))
            except json.JSONDecodeError as e:
                print(f"WARNING: could not parse {jf}: {e}")

    # Optional filter
    if args.filter:
        key, _, val = args.filter.partition("=")
        records = [r for r in records if str(r.get(key, "")) == val]

    # Sort (None values sort last)
    if args.sort != "none" and args.sort in COLUMN_ORDER:
        records.sort(key=lambda r: (r.get(args.sort) is None, r.get(args.sort)))

    # Collect all keys (in case some records have extra/future fields)
    all_keys = list(COLUMN_ORDER)
    for r in records:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} records → {out_path}")

    # Pretty-print summary table
    print(f"\n{'exp_name':<52} {'type':<8} {'steps':>6} {'FID':>8}")
    print("-" * 80)
    for r in records:
        fid_str = f"{r['fid50k']:.4f}" if r.get("fid50k") is not None else "N/A"
        steps = r.get("steps_per_pass") or r.get("num_steps") or "?"
        print(f"{r.get('exp_name', '?'):<52} "
              f"{r.get('inference_type', '?'):<8} "
              f"{str(steps):>6} "
              f"{fid_str:>8}")


if __name__ == "__main__":
    main()
