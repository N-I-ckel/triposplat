"""Lean Apple-Silicon TripoSplat pipeline: MLX encoders/flow + MPS decoder.

This module intentionally does *not* instantiate the full PyTorch
``TripoSplatPipeline``.  The upstream hybrid implementation does that first and
then loads MLX copies of DINOv3, Flux2-VAE and the flow transformer, which
duplicates the heaviest stages in unified memory.

Here the stages are loaded serially:

    BiRefNet (MPS) -> DINOv3 + Flux2-VAE (MLX) -> Flow (MLX)
                   -> OctreeGaussianDecoder (MPS)

Only the tensors required by the next stage survive.  This trades model reload
time for a substantially lower peak-memory footprint on Apple Silicon.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import mlx.core as mx
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from triposplat import load_decoder, load_rmbg, preprocess_image

try:
    from tripoflux.models.dinov3_mlx import DinoV3ViT
    from tripoflux.models.flow_mlx import LatentSeqMMFlowModel
    from tripoflux.models.flux2vae_mlx import Flux2VAEEncoder
except ImportError as exc:  # pragma: no cover - setup error, message is the API
    raise ImportError(
        "TripoSplat MLX port is not installed. Run `bash scripts/setup_mlx.sh` first."
    ) from exc

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class MLXCheckpointPaths:
    root: Path

    @classmethod
    def from_root(cls, root: Union[str, Path] = "ckpts") -> "MLXCheckpointPaths":
        return cls(Path(root).expanduser().resolve())

    @property
    def rmbg(self) -> Path:
        return self.root / "background_removal" / "birefnet.safetensors"

    @property
    def flow(self) -> Path:
        return self.root / "diffusion_models" / "triposplat_fp16.safetensors"

    @property
    def dino(self) -> Path:
        return self.root / "clip_vision" / "dino_v3_vit_h.safetensors"

    @property
    def vae(self) -> Path:
        return self.root / "vae" / "flux2-vae.safetensors"

    @property
    def decoder(self) -> Path:
        return self.root / "vae" / "triposplat_vae_decoder_fp16.safetensors"

    def validate(self) -> None:
        missing = [p for p in (self.rmbg, self.flow, self.dino, self.vae, self.decoder)
                   if not p.is_file()]
        if missing:
            joined = "\n  ".join(str(p) for p in missing)
            raise FileNotFoundError(
                f"Missing TripoSplat checkpoints:\n  {joined}\n"
                "Run `python scripts/download_mlx_weights.py` first."
            )


class _ExactSobolFlowModel(LatentSeqMMFlowModel):
    """Use the reference PyTorch Sobol sequence for positional samples.

    The upstream MLX port currently substitutes a Halton sequence for the
    reference ``torch.quasirandom.SobolEngine(scramble=True, seed=123)`` even
    though ``pos_pe`` is generated at construction time and is not checkpointed.
    Keeping the reference sequence removes that avoidable source of parity drift.
    """

    @staticmethod
    def _sobol_sequence(dim: int, n: int, seed: int = 123) -> mx.array:
        seq = torch.quasirandom.SobolEngine(
            dimension=dim, scramble=True, seed=seed
        ).draw(n)
        return mx.array(seq.cpu().numpy().astype(np.float32, copy=False))


FLOW_MODEL_ARGS = dict(
    q_token_length=8192,
    in_channels=16,
    cam_channels=5,
    out_channels=16,
    model_channels=1024,
    cond_channels=1280,
    cond2_channels=128,
    num_refiner_blocks=2,
    num_blocks=24,
    num_heads=16,
    mlp_ratio=4,
    qk_rms_norm=True,
    share_mod=True,
    use_shift_table=True,
)


def _clear_mlx() -> None:
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:
        # Older MLX builds do not expose the public cache helper.
        pass


def _clear_mps() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except RuntimeError:
            pass
        torch.mps.empty_cache()


def _pil_to_mlx_chw(image: Image.Image) -> mx.array:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return mx.array(arr)


class TripoSplatMLXPipeline:
    """Memory-conscious image-to-Gaussian pipeline for Apple Silicon."""

    _NUM_GAUSSIANS_MIN = 32768
    _NUM_GAUSSIANS_MAX = 262144

    def __init__(self, ckpt_root: Union[str, Path] = "ckpts"):
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "The MLX/MPS pipeline requires Apple Silicon with PyTorch MPS enabled."
            )
        self.paths = MLXCheckpointPaths.from_root(ckpt_root)
        self.paths.validate()
        self.device = torch.device("mps")

    def _preprocess(self, image, erode_radius: int) -> Image.Image:
        # BiRefNet is needed only for this stage. Drop it before the MLX models
        # are loaded so it does not compete for unified memory.
        rmbg = load_rmbg(str(self.paths.rmbg), device=self.device, dtype=torch.float16)
        try:
            with torch.inference_mode():
                return preprocess_image(image, rmbg, erode_radius=erode_radius)
        finally:
            del rmbg
            _clear_mps()

    def _encode_image(self, image: Image.Image, seed: int) -> dict[str, mx.array]:
        dino = DinoV3ViT()
        dino.load_safetensors(str(self.paths.dino))
        vae = Flux2VAEEncoder()
        vae.load_safetensors(str(self.paths.vae))

        x = _pil_to_mlx_chw(image)
        mean = mx.array([0.485, 0.456, 0.406], dtype=mx.float32).reshape(1, 3, 1, 1)
        std = mx.array([0.229, 0.224, 0.225], dtype=mx.float32).reshape(1, 3, 1, 1)

        dino_feat = dino((x - mean) / std)
        # Match torch.nn.functional.layer_norm(..., eps=1e-5).
        mu = mx.mean(dino_feat.astype(mx.float32), axis=-1, keepdims=True)
        var = mx.var(dino_feat.astype(mx.float32), axis=-1, keepdims=True)
        dino_feat = (dino_feat.astype(mx.float32) - mu) / mx.sqrt(var + 1e-5)

        vae_feat = vae.encode(x * 2.0 - 1.0, deterministic=False, seed=seed)
        zero_reg = mx.zeros(
            (vae_feat.shape[0], 5, vae_feat.shape[2]), dtype=vae_feat.dtype
        )
        vae_feat = mx.concatenate([zero_reg, vae_feat], axis=1)
        mx.eval(dino_feat, vae_feat)

        cond = {"feature1": dino_feat, "feature2": vae_feat}
        del dino, vae, x
        _clear_mlx()
        return cond

    def _sample_latent(
        self,
        cond: dict[str, mx.array],
        *,
        seed: int,
        steps: int,
        guidance_scale: float,
        shift: float,
        show_progress: bool,
        callback: Optional[ProgressCallback],
    ) -> np.ndarray:
        flow = _ExactSobolFlowModel(**FLOW_MODEL_ARGS)
        flow.load_safetensors(str(self.paths.flow))

        # The VAE consumed its own deterministic seed. Use a decorrelated stream
        # for flow noise while preserving seed -> output reproducibility.
        mx.random.seed((int(seed) + 1) & 0xFFFFFFFF)
        sample = {
            "latent": mx.random.normal((1, 8192, 16)),
            "camera": mx.random.normal((1, 1, 5)),
        }
        neg_cond = {k: mx.zeros_like(v) for k, v in cond.items()}

        u = np.linspace(1.0, 0.0, steps + 1, dtype=np.float64)
        t_seq = shift * u / (1.0 + (shift - 1.0) * u)
        pairs = list(zip(t_seq[:-1], t_seq[1:]))
        iterator = tqdm(pairs, desc="TripoSplat MLX", total=steps) if show_progress else pairs

        for i, (t, t_prev) in enumerate(iterator, start=1):
            t_scaled = mx.array([1000.0 * float(t)], dtype=mx.float32)
            pred = flow(sample, t_scaled, cond)

            if guidance_scale is not None and guidance_scale > 1.0:
                uncond = flow(sample, t_scaled, neg_cond)
                pred = {
                    k: guidance_scale * pred[k] - (guidance_scale - 1.0) * uncond[k]
                    for k in pred
                }

            dt = float(t - t_prev)
            sample = {k: sample[k] - pred[k] * dt for k in sample}
            # MLX is lazy. Materialize each Euler step so callbacks represent
            # completed work rather than queued graph construction.
            mx.eval(*sample.values())
            if callback is not None:
                callback(i, steps)

        latent = np.array(sample["latent"])
        del flow, sample, neg_cond, cond
        _clear_mlx()
        return latent

    @staticmethod
    def _validate_num_gaussians(n: int) -> int:
        if not TripoSplatMLXPipeline._NUM_GAUSSIANS_MIN <= n <= TripoSplatMLXPipeline._NUM_GAUSSIANS_MAX:
            raise ValueError(
                "num_gaussians must be in "
                f"[{TripoSplatMLXPipeline._NUM_GAUSSIANS_MIN}, "
                f"{TripoSplatMLXPipeline._NUM_GAUSSIANS_MAX}], got {n}"
            )
        # Official decoder emits 32 Gaussians per sampled anchor.
        return int(round(n / 32.0) * 32)

    def _decode(self, latent: np.ndarray, num_gaussians: int):
        decoder = load_decoder(
            str(self.paths.decoder), device=self.device, dtype=torch.float16
        )
        latent_t = torch.from_numpy(latent).to(self.device)
        try:
            with torch.inference_mode():
                gaussian = decoder.decode(latent_t, num_gaussians=num_gaussians)
        finally:
            del decoder, latent_t
            _clear_mps()
        return gaussian

    def run(
        self,
        image,
        *,
        seed: int = 42,
        steps: int = 20,
        guidance_scale: float = 3.0,
        shift: float = 3.0,
        num_gaussians: int = 262144,
        erode_radius: int = 1,
        show_progress: bool = False,
        callback: Optional[ProgressCallback] = None,
    ):
        """Generate a Gaussian object and the RGB image actually seen by encoders."""
        n = self._validate_num_gaussians(int(num_gaussians))
        prepared = self._preprocess(image, erode_radius=erode_radius)
        cond = self._encode_image(prepared, seed=seed)
        latent = self._sample_latent(
            cond,
            seed=seed,
            steps=int(steps),
            guidance_scale=float(guidance_scale),
            shift=float(shift),
            show_progress=show_progress,
            callback=callback,
        )
        gaussian = self._decode(latent, num_gaussians=n)
        return gaussian, prepared
