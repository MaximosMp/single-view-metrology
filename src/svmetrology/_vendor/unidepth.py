"""The UniDepth V2 network (ViT-L/14), inference only, for a photo with known pinhole intrinsics.

Copied from https://github.com/lpiccinelli-eth/UniDepth at commit
8d8cfe4c7ee15297099983607febf0d4f32eb3d6, files ``unidepth/models/unidepthv2/unidepthv2.py``,
``unidepth/models/unidepthv2/decoder.py``, ``unidepth/models/backbones/dinov2.py``,
``unidepth/models/backbones/metadinov2/``, ``unidepth/layers/`` and ``unidepth/utils/``.
Author: Luigi Piccinelli; the DINOv2 backbone is Copyright (c) Meta Platforms, Inc.
and affiliates. Licensed under Creative Commons Attribution-NonCommercial 4.0
International (CC BY-NC 4.0), which forbids commercial use; see ``LICENSE-unidepth``
beside this file.

Changes from the original:

- Only the configuration ``lpiccinelli/unidepth-v2-vitl14`` uses is kept: a DINOv2
  ViT-L/14 encoder and the depth head. UniDepth V1, the other backbones, the
  config-driven construction and the Hugging Face hub mixin are removed; weights are
  passed in as a state dict.
- The camera is always given, as a 3 x 3 pinhole matrix. The camera head that
  predicts intrinsics, the per-layer class tokens feeding it and the non-pinhole
  camera models are removed: with a camera supplied the original still runs that
  head, but its result reaches neither the rays nor the depth.
- Only z-depth is returned. The confidence head, and the position and level
  embeddings the original computes without using, are removed. The weights of every
  removed part are skipped when loading.
- Training code is removed: losses, parameter groups, weight initialisation, drop
  path, dropout.
- einops, timm and torchvision are replaced by plain tensor operations.
- The CUDA half-precision autocast is removed, so every device runs in float32, as
  the original does on CPU and MPS. The ``resolution_level`` option is removed; the
  default pixel bounds apply.
- The camera matrix passed in is left untouched; the original rescales it in place.
- An image whose aspect ratio is within bounds is never padded. The original
  recomputes its width as ``int(H * (W / H))``, which for some sizes (201 x 200, say)
  rounds to ``W - 1`` and returns a one-column map.
- Two steps that change nothing are dropped: resampling the last decoder level to
  its own size, and the shortcut that skips resampling the position embedding at
  518 x 518.

Module and parameter names follow the original, so the published checkpoint loads
by name.
"""

from __future__ import annotations

from math import ceil, log2, pi

import torch
import torch.nn.functional as F
from torch import nn

PATCH_SIZE = 14
EMBED_DIM = 1024
DEPTH = 24
HEADS = 16
OUT_LAYERS = (5, 11, 17, 23)
PRETRAIN_GRID = 37  # 518 px / 14 px, the grid the position embedding was trained at
HIDDEN_DIM = 512
DECODER_HEADS = 8
# The network runs at an area within these bounds, on an aspect ratio (W / H) within these.
PIXEL_BOUNDS = (200_000, 600_000)
RATIO_BOUNDS = (0.5, 2.5)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Parts of the checkpoint this copy leaves out.
DROPPED_WEIGHTS = (
    "pixel_encoder.mask_token",
    "pixel_encoder.register_tokens",
    "pixel_decoder.camera_layer.",
    "pixel_decoder.camera_token_adapter.",
    "pixel_decoder.level_embeds",
    "pixel_decoder.depth_layer.confidence_mlp.",
    "pixel_decoder.depth_layer.to_confidence_lr.",
    "pixel_decoder.depth_layer.to_confidence_hr.",
)


# -- Encoder: DINOv2 ViT-L -------------------------------------------------------


class _PatchEmbed(nn.Module):
    """Non-overlapping 14 px patches to tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map B x 3 x H x W to B x N x C."""
        return self.proj(x).flatten(2).transpose(1, 2)


class _Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(EMBED_DIM, EMBED_DIM * 3)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over the tokens of B x N x C."""
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, HEADS, c // HEADS).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class _Mlp(nn.Module):
    """Two-layer feed-forward block."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(EMBED_DIM, EMBED_DIM * 4)
        self.fc2 = nn.Linear(EMBED_DIM * 4, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        return self.fc2(F.gelu(self.fc1(x)))


class _LayerScale(nn.Module):
    """Learned per-channel scale on a residual branch."""

    def __init__(self) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(EMBED_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale each channel."""
        return x * self.gamma


class _Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM, eps=1e-6)
        self.attn = _Attention()
        self.ls1 = _LayerScale()
        self.norm2 = nn.LayerNorm(EMBED_DIM, eps=1e-6)
        self.mlp = _Mlp()
        self.ls2 = _LayerScale()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and feed-forward, each as a residual."""
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class _VisionTransformer(nn.Module):
    """DINOv2 ViT-L returning the normalised patch tokens of four layers."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, PRETRAIN_GRID**2 + 1, EMBED_DIM))
        self.blocks = nn.ModuleList(_Block() for _ in range(DEPTH))
        self.norm = nn.LayerNorm(EMBED_DIM)

    def _position_embedding(self, rows: int, cols: int) -> torch.Tensor:
        """Resample the trained 37 x 37 position grid to ``rows`` x ``cols``."""
        grid = self.pos_embed[:, 1:].reshape(1, PRETRAIN_GRID, PRETRAIN_GRID, EMBED_DIM).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(rows, cols), mode="bicubic", antialias=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, rows * cols, EMBED_DIM)
        return torch.cat([self.pos_embed[:, :1], grid], dim=1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return B x rows x cols x C patch tokens after each of `OUT_LAYERS`, class token dropped."""
        rows, cols = x.shape[-2] // PATCH_SIZE, x.shape[-1] // PATCH_SIZE
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self._position_embedding(rows, cols)
        taken = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in OUT_LAYERS:
                taken.append(self.norm(x)[:, 1:].reshape(x.shape[0], rows, cols, EMBED_DIM))
        return taken


# -- Decoder: depth head conditioned on the camera's rays ------------------------


class _DecoderMlp(nn.Module):
    """Normalised two-layer feed-forward block."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.proj1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 4)
        self.proj2 = nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        return self.proj2(F.gelu(self.proj1(self.norm(x))))


class _RayAttentionBlock(nn.Module):
    """Image tokens attend to the ray embeddings, then pass a feed-forward block."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = _DecoderMlp()
        self.kv = nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2, bias=False)
        self.q = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.norm_attnx = nn.LayerNorm(HIDDEN_DIM)
        self.norm_attnctx = nn.LayerNorm(HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)

    def forward(self, x: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
        """Condition B x N x C tokens on B x N x C ray embeddings."""
        b, n, c = x.shape
        split = (DECODER_HEADS, c // DECODER_HEADS)
        k, v = self.kv(self.norm_attnctx(rays)).reshape(b, n, 2, *split).permute(2, 0, 3, 1, 4)
        q = self.q(self.norm_attnx(x)).reshape(b, n, *split).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        x = x + self.out(attended)
        return x + self.mlp(x)


class _RayPrompt(nn.Module):
    """One attention block, nested as the checkpoint names it."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_RayAttentionBlock()])

    def forward(self, x: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
        """Condition B x N x C tokens on B x N x C ray embeddings."""
        return self.layers[0](x, rays)


class _ResidualConvUnit(nn.Module):
    """Two 3x3 convolutions with a scaled skip connection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the unit."""
        return self.gamma * self.conv2(F.leaky_relu(self.conv1(F.leaky_relu(x)))) + x


class _ResUpsample(nn.Module):
    """Refine a map, change its width and double its resolution."""

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList(_ResidualConvUnit(dim) for _ in range(2))
        self.up = nn.Sequential(
            nn.Conv2d(dim, out_dim, 1), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map B x dim x H x W to B x out_dim x 2H x 2W."""
        for conv in self.convs:
            x = conv(x)
        return self.up(x)


def _ray_embedding(rays: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Encode each patch's viewing direction as Fourier features of its two angles.

    Args:
        rays: B x 3 x H x W unit rays at the network's resolution.
        rows: Patch rows.
        cols: Patch columns.

    Returns:
        B x (rows * cols) x `HIDDEN_DIM` embedding.
    """
    rays = F.interpolate(rays, size=(rows, cols), mode="bilinear", align_corners=False, antialias=True)
    rays = rays.flatten(2).transpose(1, 2)
    x, y, z = (rays / torch.norm(rays, dim=-1, keepdim=True).clip(min=1e-4)).unbind(dim=-1)
    # The original holds x at least 1e-3 from zero, sign kept, before taking the azimuth.
    azimuth = torch.atan2(y, x.abs().clip(min=1e-3) * (2 * (x >= 0).int() - 1))
    angles = torch.stack([torch.acos(z), azimuth], dim=-1)
    scales = 2.0 ** torch.linspace(
        0.0, log2(max(rows, cols) // 2), steps=HIDDEN_DIM // 2, device=rays.device, dtype=rays.dtype
    )
    return (angles.unsqueeze(-1) * scales * pi).sin().flatten(-2)


class _DepthHead(nn.Module):
    """Four levels of ray-conditioned tokens to log-distance at the input's resolution."""

    def __init__(self) -> None:
        super().__init__()
        self.to_latents = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.prompt_camera = nn.ModuleList(_RayPrompt() for _ in range(4))
        # Levels 2 to 4 enter at 1x, 2x and 4x the patch grid, where the latents then are.
        self.process_features = nn.ModuleList(
            [
                nn.ConvTranspose2d(HIDDEN_DIM, 512, kernel_size=1, stride=1),
                nn.ConvTranspose2d(HIDDEN_DIM, 512, kernel_size=2, stride=2),
                nn.ConvTranspose2d(HIDDEN_DIM, 256, kernel_size=4, stride=4),
            ]
        )
        self.ups = nn.ModuleList([_ResUpsample(512, 512), _ResUpsample(512, 256), _ResUpsample(256, 128)])
        # The identities hold the place of the original's unused levels, keeping the checkpoint's index.
        self.depth_mlp = nn.ModuleList(
            [nn.Identity(), nn.Identity(), nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128))]
        )
        self.to_depth_lr = nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode="reflect")
        self.to_depth_hr = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor], rays: torch.Tensor) -> torch.Tensor:
        """Return B x 1 x H x W log-distance from four B x rows x cols x C levels and B x 3 x H x W rays."""
        b, rows, cols, _ = features[0].shape
        embedding = _ray_embedding(rays, rows, cols)
        tokens = [
            prompt(level.flatten(1, 2), embedding) for prompt, level in zip(self.prompt_camera, features, strict=True)
        ]
        latents, *levels = (
            t.transpose(1, 2).reshape(b, HIDDEN_DIM, rows, cols) for t in (self.to_latents(tokens[0]), *tokens[1:])
        )
        for process, up, level in zip(self.process_features, self.ups, levels, strict=True):
            latents = up(latents + process(level))
        x = self.depth_mlp[-1](latents.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.interpolate(self.to_depth_lr(x), size=rays.shape[-2:], mode="bilinear", align_corners=True)
        return self.to_depth_hr(x)


class _Decoder(nn.Module):
    """Project the encoder's levels and decode them to distance along each ray."""

    def __init__(self) -> None:
        super().__init__()
        self.input_adapter = nn.Module()
        self.input_adapter.input_adapters = nn.ModuleList(nn.Linear(EMBED_DIM, HIDDEN_DIM) for _ in OUT_LAYERS)
        self.depth_layer = _DepthHead()

    def forward(self, features: list[torch.Tensor], rays: torch.Tensor) -> torch.Tensor:
        """Return B x 1 x H x W distance in metres from four token levels and B x 3 x H x W rays."""
        features = [adapt(level) for adapt, level in zip(self.input_adapter.input_adapters, features, strict=True)]
        return torch.exp(self.depth_layer(features, rays).clip(min=-8.0, max=8.0) + 2.0)


# -- Model: preprocessing, network, postprocessing -------------------------------


def _padding(height: int, width: int) -> tuple[int, int, int, int]:
    """Return the ``(left, right, top, bottom)`` padding that brings W / H within `RATIO_BOUNDS`."""
    ratio = width / height
    target = min(RATIO_BOUNDS[1], max(RATIO_BOUNDS[0], ratio))
    if ratio > target:
        extra = int(width / target) - height
        return 0, 0, extra // 2, extra - extra // 2
    if ratio < target:
        extra = int(height * target) - width
        return extra // 2, extra - extra // 2, 0, 0
    return 0, 0, 0, 0


def _network_size(height: int, width: int) -> tuple[float, tuple[int, int]]:
    """Return the resize factor into `PIXEL_BOUNDS` and the ``(height, width)`` the network runs at.

    Each side is rounded up to a multiple of 14 after scaling, so the image is
    resized by slightly more than the factor. The intrinsics are still scaled by
    the factor itself, as the original does.
    """
    pixels = height * width
    factor = (min(PIXEL_BOUNDS[1], max(PIXEL_BOUNDS[0], pixels)) / pixels) ** 0.5
    rows, cols = ceil(int(height * factor) / PATCH_SIZE), ceil(int(width * factor) / PATCH_SIZE)
    return factor, (rows * PATCH_SIZE, cols * PATCH_SIZE)


def _rays(camera_matrix: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Return 1 x 3 x H x W unit viewing rays through the pixel centres of a pinhole camera."""
    device = camera_matrix.device
    u = torch.linspace(0.5, width - 0.5, width, device=device).repeat(height, 1)
    v = torch.linspace(0.5, height - 0.5, height, device=device).repeat(width, 1).t()
    rays = torch.inverse(camera_matrix) @ torch.stack([u, v, torch.ones_like(u)]).reshape(3, -1)
    rays = (rays / rays[-1:].clip(min=1e-4)).reshape(1, 3, height, width)
    return rays / torch.norm(rays, dim=1, keepdim=True).clamp(min=1e-4)


class UniDepthV2(nn.Module):
    """UniDepth V2 ViT-L: an RGB photo and its pinhole intrinsics to metric z-depth."""

    def __init__(self) -> None:
        super().__init__()
        self.pixel_encoder = _VisionTransformer()
        self.pixel_decoder = _Decoder()

    @torch.no_grad()
    def forward(self, image: torch.Tensor, camera_matrix: torch.Tensor) -> torch.Tensor:
        """Predict depth along the optical axis at the photo's own resolution.

        Args:
            image: 3 x H x W RGB, undistorted, ``uint8`` or float in 0-255, on any device.
            camera_matrix: 3 x 3 pinhole intrinsics valid for ``image``; not modified.

        Returns:
            H x W ``float32`` z-depth in metres, on the network's device.
        """
        device = self.pixel_encoder.cls_token.device
        height, width = image.shape[-2:]
        left, right, top, bottom = _padding(height, width)
        padded = (height + top + bottom, width + left + right)
        factor, network = _network_size(*padded)

        mean, std = (torch.tensor(values, device=device).view(3, 1, 1) for values in (IMAGENET_MEAN, IMAGENET_STD))
        x = F.pad(((image.to(device).float() / 255.0 - mean) / std).unsqueeze(0), (left, right, top, bottom), value=0.0)
        x = F.interpolate(x, size=network, mode="bilinear", align_corners=False)

        k = camera_matrix.to(device=device, dtype=torch.float32).clone()
        k[0, 2] += left
        k[1, 2] += top
        k[:2] *= factor
        rays = _rays(k, *network)

        depth = rays[:, 2:] * self.pixel_decoder(self.pixel_encoder(x), rays)
        depth = F.interpolate(depth, size=padded, mode="bilinear", align_corners=False)
        return depth[0, 0, top : top + height, left : left + width]

    def load_checkpoint(self, state: dict[str, torch.Tensor]) -> None:
        """Load the published checkpoint, skipping the parts this copy leaves out.

        Args:
            state: The checkpoint's tensors, keys as published.

        Raises:
            RuntimeError: any other key is missing or unexpected.
        """
        kept = {key: value for key, value in state.items() if not key.startswith(DROPPED_WEIGHTS)}
        self.load_state_dict(kept, strict=True)
