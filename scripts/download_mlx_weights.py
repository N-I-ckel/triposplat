#!/usr/bin/env python3
"""Download the TripoSplat MLX-compatible checkpoints into the official layout.

The mlx-community repository stores the five checkpoints flat.  The original
TripoSplat code expects them under ckpts/{background_removal,diffusion_models,
clip_vision,vae}.  This script bridges the two layouts without committing any
weights to Git.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "mlx-community/tripo_splat_mlx"

FILES = {
    "birefnet.safetensors": "background_removal/birefnet.safetensors",
    "triposplat_fp16.safetensors": "diffusion_models/triposplat_fp16.safetensors",
    "dino_v3_vit_h.safetensors": "clip_vision/dino_v3_vit_h.safetensors",
    "flux2-vae.safetensors": "vae/flux2-vae.safetensors",
    "triposplat_vae_decoder_fp16.safetensors": "vae/triposplat_vae_decoder_fp16.safetensors",
}


def install_file(src: Path, dst: Path, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            print(f"[skip] {dst}")
            return
        dst.unlink()

    # A hard link avoids a second multi-GB copy while remaining valid even if
    # the Hugging Face cache entry is later unlinked. Fall back to copy across
    # filesystems.
    try:
        os.link(src, dst)
        mode = "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        mode = "copy"
    print(f"[{mode}] {dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"Hugging Face model repo (default: {DEFAULT_REPO})")
    parser.add_argument("--ckpts", default="ckpts", help="Checkpoint root directory")
    parser.add_argument("--revision", default="main", help="HF revision/tag/commit")
    parser.add_argument("--force", action="store_true", help="Replace existing files")
    args = parser.parse_args()

    root = Path(args.ckpts).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    for remote_name, relative_path in FILES.items():
        print(f"[download] {args.repo}:{remote_name}")
        cached = Path(hf_hub_download(
            repo_id=args.repo,
            filename=remote_name,
            revision=args.revision,
        ))
        install_file(cached, root / relative_path, args.force)

    print(f"\nReady: {root}")
    for relative_path in FILES.values():
        path = root / relative_path
        print(f"  {relative_path}: {path.stat().st_size / (1024 ** 2):.1f} MiB")


if __name__ == "__main__":
    main()
