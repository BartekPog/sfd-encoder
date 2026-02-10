#!/usr/bin/env python3
"""
Download files from HuggingFace Hub using the Python API.
This script replaces huggingface-cli commands.
"""
from huggingface_hub import snapshot_download
import os
import shutil

def download_and_move(repo_id, include_pattern, local_dir, dest_dir):
    """Download files from HuggingFace and move them to destination."""
    print(f"Downloading {include_pattern} from {repo_id}...")
    
    # Download files
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=include_pattern,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    
    # Move files to destination
    source_path = os.path.join(local_dir, include_pattern.replace("/*", ""))
    if os.path.exists(source_path):
        print(f"Moving {source_path} to {dest_dir}")
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        if os.path.isdir(source_path):
            # Move directory contents
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            shutil.move(source_path, dest_dir)
        else:
            # Move file
            shutil.move(source_path, dest_dir)
        print(f"Successfully moved to {dest_dir}")
    else:
        print(f"Warning: {source_path} not found after download")

def main():
    # Create temp and output directories
    os.makedirs("temp", exist_ok=True)
    os.makedirs("outputs/dataset/imagenet1k-latents", exist_ok=True)
    os.makedirs("outputs/train", exist_ok=True)
    
    # Prepare latent statistics
    download_and_move(
        "SFD-Project/SFD",
        "imagenet1k-latents/*",
        "temp",
        "outputs/dataset/imagenet1k-latents/"
    )
    
    # Prepare the autoguidance model
    download_and_move(
        "SFD-Project/SFD",
        "model_weights/sfd_autoguidance_b/*",
        "temp",
        "outputs/train/sfd_autoguidance_b"
    )
    
    # Prepare XL model (675M)
    download_and_move(
        "SFD-Project/SFD",
        "model_weights/sfd_xl/*",
        "temp",
        "outputs/train/sfd_xl"
    )
    
    # Prepare XXL model (1.0B)
    download_and_move(
        "SFD-Project/SFD",
        "model_weights/sfd_1p0/*",
        "temp",
        "outputs/train/sfd_1p0"
    )
    
    # Clean up temp directory
    if os.path.exists("temp"):
        shutil.rmtree("temp")
        print("Cleaned up temp directory")
    
    print("\nAll downloads completed successfully!")

if __name__ == "__main__":
    main()
