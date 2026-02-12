#!/usr/bin/env python3
"""
Download files from HuggingFace.
"""
import os
import shutil
import subprocess

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

def download_and_move(repo_id, include_pattern, local_dir, dest_dir):
    """Download files from HuggingFace and move them to destination."""
    print(f"Downloading {include_pattern} from {repo_id}...")

    # Prefer huggingface_hub if available, but fall back to git clone when the
    # local hub/requests stack is incompatible (seen as MissingSchema URL errors).
    snapshot_dir = None
    if snapshot_download is not None:
        try:
            snapshot_dir = snapshot_download(
                repo_id=repo_id,
                cache_dir=local_dir,
            )
        except Exception as exc:
            print(f"snapshot_download failed: {exc}")
            print("Falling back to git clone for this repo...")

    if snapshot_dir is None:
        repo_cache_name = repo_id.replace("/", "__")
        snapshot_dir = os.path.join(local_dir, repo_cache_name)
        if not os.path.exists(snapshot_dir):
            os.makedirs(local_dir, exist_ok=True)
            subprocess.check_call(
                ["git", "clone", "https://huggingface.co/" + repo_id, snapshot_dir]
            )

    # Copy files to destination, resolving symlinks so the copies are
    # self-contained and survive deletion of the HuggingFace cache / temp dir.
    source_path = os.path.join(snapshot_dir, include_pattern.replace("/*", ""))
    if os.path.exists(source_path):
        print(f"Copying {source_path} to {dest_dir} (resolving symlinks)")
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        if os.path.isdir(source_path):
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            shutil.copytree(source_path, dest_dir, symlinks=False)
        else:
            shutil.copy2(source_path, dest_dir, follow_symlinks=True)
        print(f"Successfully copied to {dest_dir}")
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
