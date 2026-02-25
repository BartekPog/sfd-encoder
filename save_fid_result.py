"""
Save FID result + config metadata to a JSON file after inference completes.

Usage (at end of SLURM job):
    python save_fid_result.py \\
        --output_dir outputs/inference/v2_mse01_cos01_0080000 \\
        --config    configs/sfd/hidden_b_h200/v2_mse01_cos01.yaml \\
        --ckpt_step 80000 \\
        --inference_type linear \\
        --sampler euler \\
        --num_steps 100

    # Two-pass variant:
    python save_fid_result.py \\
        --output_dir outputs/inference/v2_mse01_cos01_0080000 \\
        --config    configs/sfd/hidden_b_h200/v2_mse01_cos01.yaml \\
        --ckpt_step 80000 \\
        --inference_type twopass \\
        --sampler euler \\
        --num_steps 100 \\
        --steps_per_pass 50
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml


# Fields to extract from the config, with defaults if absent.
# Format: "dotted.config.key": ("output_key", default_value)
CONFIG_FIELDS = {
    # model architecture
    "model.model_type":               ("model_type",               None),
    "model.use_hidden_tokens":        ("use_hidden_tokens",         False),
    "model.share_timestep_embedder":  ("share_timestep_embedder",   False),
    "model.share_patch_embedder":     ("share_patch_embedder",      True),   # default in HiddenLightningDiT
    "model.use_per_token_encoding":   ("use_per_token_encoding",    True),   # default in HiddenLightningDiT
    # num_hidden_tokens is usually baked into the model_type name, but also stored in some configs
    "model.num_hidden_tokens":        ("num_hidden_tokens",         None),
    # hidden token loss settings
    "model.hidden_weight":            ("hidden_weight",             None),
    "model.hidden_cos_weight":        ("hidden_cos_weight",         None),
    "model.hidden_reg_weight":        ("hidden_reg_weight",         None),
    "model.hidden_same_t_as_img":     ("hidden_same_t_as_img",      False),
    "model.normalize_hidden":         ("normalize_hidden",          True),
    # REPA
    "model.use_repa":                 ("use_repa",                  False),
    "model.repa_weight":              ("repa_weight",               None),
    "model.repa_mode":                ("repa_mode",                 None),
    # other model settings
    "model.semantic_weight":          ("semantic_weight",           None),
    # training
    "train.max_steps":                ("train_max_steps",           None),
    "train.weight_init":              ("weight_init",               None),
}


def _nested_get(d: dict, dotted_key: str):
    """Traverse a nested dict with a dotted key; return None if any level is missing."""
    keys = dotted_key.split(".")
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def find_fid_txt(output_dir: Path, inference_type: str | None = None,
                 steps_per_pass: int | None = None,
                 num_steps: int | None = None,
                 hidden_schedule_max_t: float | None = None) -> Path | None:
    """
    Find fid_result.txt written by inference.py.
    inference.py writes it inside a subfolder named after the run
    (e.g. outputs/inference/{exp_name}/{folder_name}/fid_result.txt),
    so we search recursively under output_dir.

    When *inference_type* is given, filter subdirectories so that twopass
    results match only folders ending with ``-twopass`` and linear/standard
    results match only folders that do **not** end with ``-twopass``.

    When *steps_per_pass* (twopass) or *num_steps* (linear) is given, further
    filter by the step count embedded in the folder name (e.g. ``euler-100-``).

    When *hidden_schedule_max_t* is given (e.g. 0.60), further filter to
    folders containing ``hmaxt{value}`` (e.g. ``hmaxt0.60``).  When it is
    None, only folders that do NOT contain ``hmaxt`` are matched, so that
    a full-range run (max_t=1.0) is not confused with a partial-range run.
    """
    for candidate in ["fid_result.txt", "fid_results.txt"]:
        # Check directly first (future-proof)
        if (output_dir / candidate).exists():
            return output_dir / candidate
        # Search one level of subdirectories (the folder_name layer)
        matches = sorted(output_dir.glob(f"*/{candidate}"))
        if inference_type and matches:
            if inference_type == "twopass":
                matches = [m for m in matches if m.parent.name.endswith("-twopass")]
                # Also filter by steps_per_pass encoded in folder name (e.g. euler-50-)
                if steps_per_pass and len(matches) > 1:
                    step_matches = [m for m in matches
                                    if f"-{steps_per_pass}-" in m.parent.name]
                    if step_matches:
                        matches = step_matches
            else:
                matches = [m for m in matches if not m.parent.name.endswith("-twopass")]
                # Filter by num_steps encoded in folder name
                if num_steps and len(matches) > 1:
                    step_matches = [m for m in matches
                                    if f"-{num_steps}-" in m.parent.name]
                    if step_matches:
                        matches = step_matches
                # Filter by hidden_schedule_max_t: keep only hmaxt folders when
                # a specific max_t is requested, or only non-hmaxt folders otherwise.
                if hidden_schedule_max_t is not None:
                    tag = f"hmaxt{hidden_schedule_max_t:.2f}"
                    hmaxt_matches = [m for m in matches if tag in m.parent.name]
                    if hmaxt_matches:
                        matches = hmaxt_matches
                else:
                    no_hmaxt = [m for m in matches if "hmaxt" not in m.parent.name]
                    if no_hmaxt:
                        matches = no_hmaxt
        if matches:
            return matches[0]
    return None


def extract_fid(fid_txt: Path) -> float | None:
    """Parse the FID value from a fid_result.txt file."""
    text = fid_txt.read_text()
    # Matches lines like:  "FID: 4.3217"  or  "fid50k_full: 4.3217"
    m = re.search(r"(?:FID|fid50k[\w]*)[\s:=]+([0-9]+\.[0-9]+)", text, re.I)
    return float(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",     required=True,
                        help="Inference output directory (where samples and fid_results.txt live)")
    parser.add_argument("--config",         required=True,
                        help="Path to the training config YAML used for this run")
    parser.add_argument("--ckpt_step",      required=True, type=int,
                        help="Checkpoint step that was evaluated")
    parser.add_argument("--inference_type", required=True,
                        choices=["linear", "twopass", "standard"],
                        help="Inference mode")
    parser.add_argument("--sampler",        default="euler")
    parser.add_argument("--num_steps",      type=int, default=100,
                        help="Total number of ODE steps (for twopass: steps per pass)")
    parser.add_argument("--steps_per_pass", type=int, default=None,
                        help="Steps per pass for two-pass inference (None for linear/standard)")
    parser.add_argument("--hidden_schedule_max_t", type=float, default=None,
                        help="Max timestep for hidden schedule (e.g. 0.60). "
                             "Omit for full-range runs (max_t=1.0).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config_path = Path(args.config)

    # ---- FID ---- locate the txt written by inference.py (may be in a subfolder)
    fid_txt = find_fid_txt(output_dir,
                           inference_type=args.inference_type,
                           steps_per_pass=args.steps_per_pass,
                           num_steps=args.num_steps,
                           hidden_schedule_max_t=args.hidden_schedule_max_t)
    if fid_txt is None:
        print(f"WARNING: fid_result.txt not found under {output_dir}", file=sys.stderr)
        fid = None
        json_dir = output_dir  # fall back: write JSON in the parent dir
    else:
        fid = extract_fid(fid_txt)
        if fid is None:
            print(f"WARNING: could not parse FID from {fid_txt}", file=sys.stderr)
        json_dir = fid_txt.parent  # write JSON alongside the txt

    # ---- Config ----
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    result = {
        "exp_name":               output_dir.name,
        "sample_dir":             json_dir.name,
        "config":                 str(config_path),
        "ckpt_step":              args.ckpt_step,
        "inference_type":         args.inference_type,
        "sampler":                args.sampler,
        "num_steps":              args.num_steps,
        "steps_per_pass":         args.steps_per_pass,
        "hidden_schedule_max_t":  args.hidden_schedule_max_t,
        "fid50k":                 fid,
        "timestamp":              datetime.now().isoformat(timespec="seconds"),
    }

    for cfg_key, (out_key, default) in CONFIG_FIELDS.items():
        val = _nested_get(cfg, cfg_key)
        result[out_key] = val if val is not None else default

    out_file = json_dir / "fid_result.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved FID result → {out_file}")
    if fid is not None:
        print(f"  FID50K = {fid:.4f}")
    else:
        print("  FID50K = N/A (not found)")


if __name__ == "__main__":
    main()
