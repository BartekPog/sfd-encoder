import argparse
import glob
import os

from safetensors import safe_open

from dataset.img_latent_dataset import ImgLatentDataset


def has_latent_sv_shards(data_dir: str) -> bool:
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(f"No .safetensors shards found in: {data_dir}")

    with safe_open(shard_paths[0], framework="pt", device="cpu") as f:
        return "latents_sv" in f.keys()


def maybe_remove_cache(cache_path: str, recompute: bool) -> None:
    if recompute and os.path.exists(cache_path):
        os.remove(cache_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute latent stats cache files from existing safetensors shards."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with latent shards.")
    parser.add_argument(
        "--latent_sv_norm",
        action="store_true",
        help="Also compute latents_sv stats. If omitted, auto-detected from shard keys.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Delete existing *_stats.pt files before recomputing.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    latent_sv_norm = args.latent_sv_norm or has_latent_sv_shards(data_dir)

    latents_stats_path = os.path.join(data_dir, "latents_stats.pt")
    latents_sv_stats_path = os.path.join(data_dir, "latents_sv_stats.pt")

    maybe_remove_cache(latents_stats_path, args.recompute)
    maybe_remove_cache(latents_sv_stats_path, args.recompute)

    print(f"Computing stats in: {data_dir}")
    print(f"Compute latents_sv stats: {latent_sv_norm}")
    print(f"Recompute mode: {args.recompute}")

    # ImgLatentDataset computes and caches stats during initialization.
    ImgLatentDataset(data_dir, latent_norm=True, latent_sv_norm=latent_sv_norm)

    print("Done.")
    print(f"latents stats: {latents_stats_path}")
    if latent_sv_norm:
        print(f"latents_sv stats: {latents_sv_stats_path}")


if __name__ == "__main__":
    main()
