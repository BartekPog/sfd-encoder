import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path
if __name__ == "__main__":
    # create_npz_from_sample_folder("/path/to/infer_folder")
    # create_npz_from_sample_folder("outputs/lightningdit_xl_vavae_f16d32/lightningdit-xl-1-ckpt-lightningdit-xl-imagenet256-800ep-euler-250-interval0.11-cfg10.00-shift0.30")
    # create_npz_from_sample_folder("outputs/lightningdit_xl_vavae_f16d32/lightningdit-xl-1-ckpt-lightningdit-xl-imagenet256-800ep-euler-250-interval0.11-cfg6.70-shift0.30")

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, required=True)
    args = parser.parse_args()

    create_npz_from_sample_folder(args.sample_dir)