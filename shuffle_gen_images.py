#!/usr/bin/env python3
"""Shuffle image file order"""

import os
import random
import shutil
from tqdm import tqdm
import argparse

def shuffle_images(src_dir, seed=42):
    """Randomly rename images to shuffle their order"""

    # Target directory: original directory + "-shuffle"
    dst_dir = src_dir.rstrip('/') + '-shuffle'

    # Collect all images
    image_files = []
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPEG')):
                full_path = os.path.join(root, f)
                image_files.append(full_path)
    
    print(f"Found {len(image_files)} images")

    # Shuffle randomly
    random.seed(seed)
    random.shuffle(image_files)
    print(f"Shuffled with seed={seed}")

    # Create target directory
    os.makedirs(dst_dir, exist_ok=True)

    # Rename to 000000.png, 000001.png, ...
    print(f"Copying to {dst_dir}...")
    for i, src_path in enumerate(tqdm(image_files)):
        ext = os.path.splitext(src_path)[1]  # Keep original extension
        dst_path = os.path.join(dst_dir, f"{i:06d}{ext}")

        # copy
        shutil.copy(src_path, dst_path)

    print(f"✅ Done! {len(image_files)} images shuffled to {dst_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, help="Source image directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    shuffle_images(args.src_dir, args.seed)