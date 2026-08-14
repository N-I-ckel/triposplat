# TripoSplat
TripoSplat converts a single 2D image into high-quality and variable number of 3D Gaussians, developed by [TripoAI](https://www.tripo3d.ai/). It can serve as a powerful pipeline tool for asset creation, AR/VR, game development, simulation environments, and beyond. This is the inference-only repo for TripoSplat. For the training code, see [TripoSplat-Training](https://github.com/runjie-yan/TripoSplat-Training).

<a href="https://arxiv.org/abs/2605.16355"><img src="https://img.shields.io/badge/Read%20Paper-B31B1B?style=for-the-badge&logo=arxiv" alt="Paper"></a>
<a href="https://www.tripo3d.ai/research/triposplat"><img src="https://img.shields.io/badge/Technical%20Blog-grey" alt="Technical Blog"></a>
<a href="https://huggingface.co/spaces/VAST-AI/TripoSplat"><img src="https://img.shields.io/badge/Huggingface%20Demo-grey?style=for-the-badge&logo=huggingface" alt="HuggingFace Demo"></a>

| ![](static/doc/001.webp) | ![](static/doc/002.webp) |
|---|---|
| ![](static/doc/003.webp) | ![](static/doc/004.webp) |

## Highlights
- **High-quality, versatile generation** that handles a wide range of image styles.
- **Arbitrary Gaussian count** (up to 262,144) — trade off visual quality against rendering cost according to your need.
- **Minimal, readable code**: two files (`triposplat.py` and `model.py`), ~2,000 LOC total. Easy to customize and integrate into other ecosystems.
- **Near-zero dependencies**: no `transformers`, no `diffusers`, no version-conflict hell. Runs on any platform.
- **Official ComfyUI support**: drop the [official workflow template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/3d_triposplat_image_to_gaussian_splat.json) into ComfyUI and start playing with TripoSplat right away.
- **Apple Silicon hybrid path (this fork)**: MLX for DINOv3 / Flux2-VAE / flow matching, PyTorch MPS only for BiRefNet and the dynamic Octree Gaussian decoder.

## Quickstart
Download model weights to `ckpts/` from [HuggingFace](https://huggingface.co/VAST-AI/TripoSplat). 
```bash
# Use one of the following ways to download model weights.

# 1. Use HuggingFace CLI
hf download VAST-AI/TripoSplat --local-dir ckpts/

# 2. Use huggingface_hub
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='VAST-AI/TripoSplat', local_dir='ckpts/')"

# 3. Use ModelScope CLI
pip install modelscope
modelscope download VAST-AI-Research/TripoSplat --local_dir ckpts/

# 4. Use modelscope Python SDK
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('VAST-AI-Research/TripoSplat', local_dir='ckpts/')"

# 5. Manual download from HuggingFace / ModelScope.
```

Setup the environment and run the example inference script.
```bash
# install torch and torchvision according to your environment
pip install numpy safetensors pillow tqdm
python run_example.py
```

The exported `.ply` / `.splat` files can be visualized in any 3D Gaussian
viewer — e.g. [SparkJS](https://sparkjs.dev) or
[SuperSplat](https://superspl.at/editor).

## Apple Silicon: MLX + MPS

This fork contains a lean hybrid path for Apple Silicon that avoids loading a full PyTorch copy of DINOv3, Flux2-VAE and the flow transformer alongside their MLX copies.

```bash
# Installs the pinned MLX port without its unrelated FLUX/web dependencies,
# then downloads mlx-community/tripo_splat_mlx into the official ckpts layout.
bash scripts/setup_mlx.sh

# Image -> Gaussian Splat
python run_mlx.py input.png -o output.splat

# Or export PLY and tune the generation budget.
python run_mlx.py input.png -o output.ply --steps 20 --guidance 3 --gaussians 262144
```

The pipeline is stage-wise to reduce unified-memory pressure:

```text
BiRefNet (MPS)
  -> DINOv3 + Flux2-VAE (MLX)
  -> LatentSeqMMFlowModel (MLX)
  -> OctreeGaussianDecoder (MPS)
  -> .splat / .ply
```

See [`APPLE_MLX.md`](APPLE_MLX.md) for architecture, the Sobol positional-parity fix, setup details, and the mixed checkpoint-license boundary. The downloaded weights are kept under `ckpts/` and are not tracked by Git.

## Gradio Demo

```bash
pip install gradio
python run_gradio.py
```

## License
TripoSplat code and the official TripoSplat weights are released under the [MIT License](https://github.com/VAST-AI-Research/TripoSplat/blob/main/LICENSE).

The optional `mlx-community/tripo_splat_mlx` download used by the Apple-Silicon helper is a mixed-license bundle. In particular, its model card identifies the bundled Flux2 VAE checkpoint as non-commercial and DINOv3 as carrying separate redistribution terms. See [`APPLE_MLX.md`](APPLE_MLX.md) before redistributing or deploying those downloaded checkpoints.

## Citation
If you find TripoSplat useful, please cite:
```bibtex
@misc{yan2026generative3dgaussianslearned,
    title={Generative 3D Gaussians with Learned Density Control}, 
    author={Runjie Yan and Yan-Pei Cao and Peng Wang and Ding Liang and Yuan-Chen Guo},
    year={2026},
    eprint={2605.16355},
    archivePrefix={arXiv},
    primaryClass={cs.GR},
    url={https://arxiv.org/abs/2605.16355}, 
}
```
