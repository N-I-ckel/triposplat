# TripoSplat on Apple Silicon: lean MLX/MPS hybrid

This branch adds an Apple-Silicon inference path while keeping the original
PyTorch/CUDA implementation intact.

## Architecture

```text
input image
   |
   v
BiRefNet background removal        PyTorch MPS
   |                               (released after preprocessing)
   v
DINOv3 ViT-H + Flux2 VAE           MLX
   |                               (released after conditioning)
   v
LatentSeqMMFlowModel               MLX
   |                               (released after Euler/CFG sampling)
   v
OctreeGaussianDecoder              PyTorch MPS
   |
   +--> .splat
   +--> .ply
```

The upstream `TripoSplat-Klein-MLX` hybrid constructs the complete PyTorch MPS
pipeline before loading MLX versions of DINOv3, Flux2-VAE and the flow model.
That is convenient, but duplicates those stages in unified memory.  This
integration instead loads each stage only when it is needed.

The upstream experimental `decoder_mlx.py` is intentionally not used: its own
header states that the dynamic octree sampling and final elastic Gaussian
path still remain on PyTorch MPS.

## Setup

Requirements:

- Apple Silicon Mac
- Python 3.10+
- PyTorch with MPS support

```bash
bash scripts/setup_mlx.sh
```

The setup script:

1. installs the lean dependencies in `requirements-mlx.txt`;
2. installs `Jup33Q/TripoSplat-Klein-MLX` at the pinned source commit with
   `--no-deps`, avoiding its unrelated FLUX/Klein/web-server dependency stack;
3. downloads `mlx-community/tripo_splat_mlx` checkpoints and maps its flat
   files into the original TripoSplat `ckpts/` directory structure.

Weights stay under `ckpts/` and are ignored by Git.

## Run

```bash
python run_mlx.py input.png -o output.splat
```

Useful controls:

```bash
python run_mlx.py input.png \
  -o output.ply \
  --steps 20 \
  --guidance 3.0 \
  --shift 3.0 \
  --gaussians 262144 \
  --seed 42 \
  --save-prepared prepared.png
```

`--gaussians` accepts 32,768 through 262,144 and is rounded to a multiple of
32, matching the official decoder's Gaussian-per-anchor layout.

## Numerical-parity fix

The current upstream MLX flow port documents that its `_sobol_sequence()` is a
Halton placeholder for the reference PyTorch Sobol sequence.  Since `pos_pe`
is generated at model construction and is not stored in the checkpoint,
`triposplat_mlx.py` subclasses the flow model and restores:

```python
torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123).draw(8192)
```

This removes one known source of MLX/PyTorch positional drift.

## Weight licensing

`mlx-community/tripo_splat_mlx` is a mixed-license distribution.  Its model
card states that the bundled `flux2-vae.safetensors` is covered by the FLUX
Non-Commercial License, so the bundle as distributed is non-commercial unless
the corresponding Black Forest Labs commercial rights are obtained.  DINOv3
also carries its own redistribution/attribution terms.  The TripoSplat core,
BiRefNet, and the MLX port code are identified there as MIT components.

For a deployment, review the current model cards and upstream licenses rather
than treating the Git repository's MIT license as applying to every downloaded
checkpoint.

## Upstream pins

- MLX port source: `Jup33Q/TripoSplat-Klein-MLX`
- Source commit: `821c1a8f090770265e8b02e8981fd1c4466a30c9`
- Checkpoints: `mlx-community/tripo_splat_mlx`
