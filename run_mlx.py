#!/usr/bin/env python3
"""Apple-Silicon TripoSplat CLI using MLX for the heavy upstream stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from triposplat_mlx import TripoSplatMLXPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Image -> TripoSplat on Apple Silicon (MLX + MPS)")
    parser.add_argument("image", help="Input image")
    parser.add_argument("-o", "--output", default="output.splat",
                        help="Output .splat or .ply file")
    parser.add_argument("--ckpts", default="ckpts", help="Checkpoint root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--gaussians", type=int, default=262144)
    parser.add_argument("--erode-radius", type=int, default=1)
    parser.add_argument("--save-prepared", default=None,
                        help="Optional path for the 1024x1024 RGB encoder input")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() not in {".splat", ".ply"}:
        parser.error("--output must end in .splat or .ply")
    output.parent.mkdir(parents=True, exist_ok=True)

    pipe = TripoSplatMLXPipeline(args.ckpts)
    gaussian, prepared = pipe.run(
        args.image,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance,
        shift=args.shift,
        num_gaussians=args.gaussians,
        erode_radius=args.erode_radius,
        show_progress=True,
    )

    if output.suffix.lower() == ".ply":
        gaussian.save_ply(output)
    else:
        gaussian.save_splat(output)

    if args.save_prepared:
        prepared_path = Path(args.save_prepared).expanduser().resolve()
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.save(prepared_path)

    print(f"Saved {args.gaussians:,} Gaussian target -> {output}")


if __name__ == "__main__":
    main()
