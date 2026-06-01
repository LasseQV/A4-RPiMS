"""
restormer_sr.py  — guided super-resolution of aligned multispectral bands

Architecture
------------
  GuidedRestormer: standard Restormer encoder-decoder with an auxiliary lightweight
  RGB encoder whose features are fused into the latent (bottleneck) via a 1×1 conv.

  Input  : (B, 4, H, W)  — concatenation of [RGB (3ch), bicubic-upsampled MS (1ch)]
  Output : (B, 1, H, W)  — SR MS band; residual-summed with the bicubic input

  Pretrained checkpoint weights (official Restormer gaussian/deraining/etc.) are
  loaded for all matching transformer-block keys.  Layers whose shape changes
  (patch_embed, output, rgb_enc, rgb_fusion) are randomly initialised and fine-tuned.

Data
----
  --aligned-dir  align_multispec.py output directory containing the 4 MS band files:
                   {stem}.jpg                   ← band 0 (685 nm, reference)
                   {stem}_band0_1_aligned.jpg   ← band 1 (725 nm)
                   {stem}_band0_2_aligned.jpg   ← band 2 (750 nm)
                   {stem}_band0_3_aligned.jpg   ← band 3 (1000 nm)
  --rgb-dir      directory of aligned RGB files named {stem}_aligned.{ext}
                 (produced by  align_multispec.py --align-rgb)

Scale experiment
----------------
  MS bands are synthetically downsampled (bicubic) then upsampled back to simulate
  higher-altitude / lower-resolution captures.  The full-resolution original is the
  reconstruction target.  --scales 4 8 tests both factors.

Usage
-----
  # Fine-tune from a pretrained Restormer checkpoint
  python restormer_sr.py train \\
      --aligned-dir aligned_test/may15/ \\
      --rgb-dir     rgb_aligned/may15/ \\
      --checkpoint  gaussian_gray_denoising_blind.pth \\
      --scales 4 8 --epochs 100 --output-dir runs/sr_exp1/

  # Evaluate a trained checkpoint
  python restormer_sr.py eval \\
      --aligned-dir aligned_test/may15/ \\
      --rgb-dir     rgb_aligned/may15/ \\
      --checkpoint  runs/sr_exp1/best.pth \\
      --scales 4 8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Restormer building blocks ─────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """Channel-first LayerNorm for (B, C, H, W) tensors."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / (s + self.eps).sqrt()
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class FeedForward(nn.Module):
    """Gated-DConv Feed-Forward (GDFN)."""
    def __init__(self, dim: int, expansion: float = 2.66, bias: bool = False):
        super().__init__()
        h = int(dim * expansion)
        self.proj_in  = nn.Conv2d(dim, h * 2, 1, bias=bias)
        self.dw       = nn.Conv2d(h * 2, h * 2, 3, padding=1, groups=h * 2, bias=bias)
        self.proj_out = nn.Conv2d(h, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        a, b = self.dw(x).chunk(2, dim=1)
        return self.proj_out(F.gelu(a) * b)


class Attention(nn.Module):
    """Multi-DConv Head Transposed Attention (MDTA)."""
    def __init__(self, dim: int, heads: int, bias: bool = False):
        super().__init__()
        self.heads       = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv         = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dw      = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.proj_out    = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        head_c = C // self.heads
        q = q.reshape(B, self.heads, head_c, H * W)
        k = k.reshape(B, self.heads, head_c, H * W)
        v = v.reshape(B, self.heads, head_c, H * W)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out  = (attn.softmax(dim=-1) @ v).reshape(B, C, H, W)
        return self.proj_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, expansion: float = 2.66, bias: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn  = Attention(dim, heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn   = FeedForward(dim, expansion, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c: int, embed_dim: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    """Halve spatial dims; double channels via PixelUnshuffle."""
    def __init__(self, n: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n, n // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """Double spatial dims; halve channels via PixelShuffle."""
    def __init__(self, n: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n, n * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ── RGB guidance encoder ──────────────────────────────────────────────────────

class RGBEncoder(nn.Module):
    """Lightweight 3-level CNN that encodes RGB to bottleneck-scale features.

    Three stride-2 convolutions bring spatial dims from H×W down to H/8×W/8,
    matching the Restormer bottleneck resolution.
    """
    def __init__(self, out_dim: int, base: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,        base,     3, padding=1),             nn.GELU(),
            nn.Conv2d(base,     base * 2, 3, stride=2, padding=1),  nn.GELU(),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),  nn.GELU(),
            nn.Conv2d(base * 4, out_dim,  3, stride=2, padding=1),  nn.GELU(),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.enc(rgb)


# ── Guided Restormer ──────────────────────────────────────────────────────────

class GuidedRestormer(nn.Module):
    """Restormer with RGB-guided latent fusion.

    Channel layout at each U-Net level (with default dim=48):
      enc1 / dec1_in         :  dim   =  48  (enc1 skip) + dim = 48 (up) → 96 at dec1
      enc2 / dec2            :  dim*2 =  96  (after reduce_chan)
      enc3 / dec3            :  dim*4 = 192  (after reduce_chan)
      latent                 :  dim*8 = 384

    decoder_level1 and refinement operate at dim*2 (= 96) channels because the
    enc1 skip is concatenated without a reduce step, matching the original Restormer.
    """
    def __init__(
        self,
        dim: int = 48,
        num_blocks: list[int] | None = None,
        num_heads: list[int] | None = None,
        expansion: float = 2.66,
        num_refinement: int = 4,
        rgb_enc_base: int = 32,
        bias: bool = False,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]

        D = dim
        H = num_heads
        B = num_blocks

        def _blocks(d, h, n):
            return nn.Sequential(*[TransformerBlock(d, h, expansion, bias) for _ in range(n)])

        # 4-ch input: [R, G, B, MS_bicubic]
        self.patch_embed = OverlapPatchEmbed(4, D, bias)

        # Encoder
        self.encoder_level1 = _blocks(D,     H[0], B[0])
        self.down1_2         = Downsample(D)
        self.encoder_level2  = _blocks(D*2,  H[1], B[1])
        self.down2_3         = Downsample(D*2)
        self.encoder_level3  = _blocks(D*4,  H[2], B[2])
        self.down3_4         = Downsample(D*4)

        # Bottleneck
        self.latent = _blocks(D*8, H[3], B[3])

        # RGB guidance fused at bottleneck (same spatial scale as latent)
        self.rgb_enc    = RGBEncoder(out_dim=D*8, base=rgb_enc_base)
        self.rgb_fusion = nn.Conv2d(D*16, D*8, 1, bias=bias)

        # Decoder
        self.up4_3              = Upsample(D*8)
        self.reduce_chan_level3 = nn.Conv2d(D*8, D*4, 1, bias=bias)
        self.decoder_level3     = _blocks(D*4, H[2], B[2])

        self.up3_2              = Upsample(D*4)
        self.reduce_chan_level2 = nn.Conv2d(D*4, D*2, 1, bias=bias)
        self.decoder_level2     = _blocks(D*2, H[1], B[1])

        # level-1 decoder: cat(up2_1(dec2), enc1) → D+D = D*2, no reduce
        self.up2_1          = Upsample(D*2)
        self.decoder_level1 = _blocks(D*2, H[1], B[0])  # H[1] keeps 48ch/head
        self.refinement     = _blocks(D*2, H[1], num_refinement)

        self.output = nn.Conv2d(D*2, 1, 3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, H, W)  — channels [R, G, B, MS_bic]
        rgb   = x[:, :3]   # (B, 3, H, W)
        ms_lr = x[:, 3:]   # (B, 1, H, W)  bicubic MS — used for residual

        feat = self.patch_embed(x)

        e1 = self.encoder_level1(feat)
        e2 = self.encoder_level2(self.down1_2(e1))
        e3 = self.encoder_level3(self.down2_3(e2))

        lat = self.latent(self.down3_4(e3))

        # Fuse RGB at bottleneck
        rgb_feat = self.rgb_enc(rgb)
        lat = self.rgb_fusion(torch.cat([lat, rgb_feat], dim=1))

        # Decoder with skip connections
        d3 = self.reduce_chan_level3(torch.cat([self.up4_3(lat), e3], dim=1))
        d3 = self.decoder_level3(d3)

        d2 = self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], dim=1))
        d2 = self.decoder_level2(d2)

        d1 = self.decoder_level1(torch.cat([self.up2_1(d2), e1], dim=1))
        d1 = self.refinement(d1)

        return self.output(d1) + ms_lr


# ── Pretrained weight loader ──────────────────────────────────────────────────

def load_pretrained_partial(model: nn.Module, ckpt_path: str | Path) -> None:
    """Load transformer-block weights from an official Restormer checkpoint.

    Matches keys by name and shape; mismatching keys (patch_embed, output, rgb_*)
    are left at their random initialisation.  Supports checkpoints saved under a
    'params' key (BasicSR format) or as a flat state dict.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = raw.get("params", raw.get("state_dict", raw))

    own = model.state_dict()
    loaded, skipped = [], []

    for k, v in src.items():
        # Strip common wrapper prefixes
        k_stripped = k
        for prefix in ("module.", "model."):
            if k_stripped.startswith(prefix):
                k_stripped = k_stripped[len(prefix):]

        if k_stripped in own and own[k_stripped].shape == v.shape:
            own[k_stripped] = v
            loaded.append(k_stripped)
        else:
            skipped.append(k_stripped)

    model.load_state_dict(own)
    print(f"  Loaded {len(loaded)} / {len(loaded) + len(skipped)} keys from checkpoint.")
    if skipped:
        print(f"  Skipped ({len(skipped)} keys, e.g.: {skipped[:3]})")


# ── Loss ──────────────────────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((pred - target).pow(2) + self.eps ** 2).sqrt().mean()


# ── Metrics ───────────────────────────────────────────────────────────────────

def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 100.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def _ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(pred, gt, data_range=255))
    except ImportError:
        # Fallback: simplified SSIM with a single 11×11 Gaussian window
        mu1 = cv2.GaussianBlur(pred.astype(np.float64), (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gt.astype(np.float64),   (11, 11), 1.5)
        s1  = cv2.GaussianBlur(pred.astype(np.float64) ** 2, (11, 11), 1.5) - mu1 ** 2
        s2  = cv2.GaussianBlur(gt.astype(np.float64)   ** 2, (11, 11), 1.5) - mu2 ** 2
        s12 = cv2.GaussianBlur(pred.astype(np.float64) * gt.astype(np.float64), (11, 11), 1.5) - mu1 * mu2
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
        return float(ssim_map.mean())


# ── Dataset ───────────────────────────────────────────────────────────────────

_BAND_EXTS = [".jpg", ".jpeg", ".tif", ".tiff", ".png"]


def _find_file(directory: Path, stem: str, suffix: str) -> Path | None:
    for ext in _BAND_EXTS:
        for e in [ext, ext.upper()]:
            p = directory / f"{stem}{suffix}{e}"
            if p.exists():
                return p
    return None


def _load_gray_f32(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def _load_rgb_f32(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def find_paired_stems(aligned_dir: Path, rgb_dir: Path) -> list[str]:
    """Return stems that have all 4 MS bands and a matching RGB file."""
    stems: set[str] = set()
    for ext in _BAND_EXTS:
        for p in aligned_dir.glob(f"*{ext}"):
            name = p.stem
            if "_band" in name or name.startswith("_") or name.startswith("viz_"):
                continue
            if name.endswith("_aligned"):
                name = name[: -len("_aligned")]
            stems.add(name)

    valid = []
    for stem in sorted(stems):
        ms0 = _find_file(aligned_dir, stem, "") or _find_file(aligned_dir, stem, "_aligned")
        ms1 = _find_file(aligned_dir, stem, "_band0_1_aligned")
        ms2 = _find_file(aligned_dir, stem, "_band0_2_aligned")
        ms3 = _find_file(aligned_dir, stem, "_band0_3_aligned")
        rgb = _find_file(rgb_dir, stem, "_aligned")
        if all(p is not None for p in [ms0, ms1, ms2, ms3, rgb]):
            valid.append(stem)

    return valid


class MSRGBDataset(Dataset):
    """Per-patch dataset for Restormer SR training.

    Each __getitem__ returns a dict with:
      'lr'     : (4, patch, patch)  — cat(RGB, bicubic-down/up MS) in [0, 1]
      'hr'     : (1, patch, patch)  — original MS band in [0, 1]
      'band'   : int                — which of the 4 bands was chosen
      'scale'  : int                — downsampling factor used

    In eval mode (patch_size=None) full images are returned.
    """
    def __init__(
        self,
        aligned_dir: Path,
        rgb_dir: Path,
        scales: list[int],
        patch_size: int | None = 128,
        augment: bool = True,
        cache_images: bool = True,
    ):
        self.aligned_dir = aligned_dir
        self.rgb_dir     = rgb_dir
        self.scales      = scales
        self.patch_size  = patch_size
        self.augment     = augment

        self.stems = find_paired_stems(aligned_dir, rgb_dir)
        if not self.stems:
            raise RuntimeError(
                f"No fully-paired MS+RGB image groups found.\n"
                f"  MS  dir : {aligned_dir}\n"
                f"  RGB dir : {rgb_dir}\n"
                "Check that both directories exist and that stems match."
            )
        print(f"Dataset: {len(self.stems)} paired image groups, "
              f"{len(scales)} scale(s) × 4 bands = "
              f"{len(self.stems) * len(scales) * 4} virtual samples per epoch.")

        # Pre-load all images into RAM so workers never hit disk after startup.
        # On Linux the DataLoader forks after this, so all workers share these
        # arrays copy-on-write at no extra memory cost.
        self._cache: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
        if cache_images:
            t0 = time.time()
            print(f"Pre-loading {len(self.stems)} image groups into RAM...", flush=True)
            for i, stem in enumerate(self.stems):
                self._cache[stem] = self._load_group_from_disk(stem)
                if (i + 1) % 100 == 0 or (i + 1) == len(self.stems):
                    mb = sum(
                        b.nbytes for bands, rgb in self._cache.values()
                        for b in (*bands, rgb)
                    ) / 1e6
                    print(f"  {i+1}/{len(self.stems)}  ({mb:.0f} MB used)  "
                          f"{time.time()-t0:.0f}s elapsed", flush=True)
            print(f"Cache ready in {time.time()-t0:.1f}s\n", flush=True)

    def __len__(self) -> int:
        # Each stem contributes one sample per scale per band per epoch
        return len(self.stems) * len(self.scales) * 4

    def _load_group_from_disk(self, stem: str) -> tuple[list[np.ndarray], np.ndarray]:
        """Return ([b0,b1,b2,b3], rgb) as float32 arrays in [0,1]."""
        ad = self.aligned_dir
        ms0 = _find_file(ad, stem, "") or _find_file(ad, stem, "_aligned")
        bands = [
            _load_gray_f32(ms0),
            _load_gray_f32(_find_file(ad, stem, "_band0_1_aligned")),
            _load_gray_f32(_find_file(ad, stem, "_band0_2_aligned")),
            _load_gray_f32(_find_file(ad, stem, "_band0_3_aligned")),
        ]
        rgb = _load_rgb_f32(_find_file(self.rgb_dir, stem, "_aligned"))
        return bands, rgb

    def _load_group(self, stem: str) -> tuple[list[np.ndarray], np.ndarray]:
        if stem in self._cache:
            return self._cache[stem]
        return self._load_group_from_disk(stem)

    def __getitem__(self, idx: int) -> dict:
        n_scales = len(self.scales)
        n_bands  = 4
        stem_idx  = idx // (n_scales * n_bands)
        remainder = idx % (n_scales * n_bands)
        scale_idx = remainder // n_bands
        band_idx  = remainder % n_bands

        stem  = self.stems[stem_idx]
        scale = self.scales[scale_idx]
        bands, rgb = self._load_group(stem)
        hr_band = bands[band_idx]   # (H, W) float32

        H, W = hr_band.shape
        ps   = self.patch_size

        if ps is not None:
            # Align crop to a multiple of the scale factor for clean downsampling
            ps_aligned = (ps // scale) * scale
            max_y = H - ps_aligned
            max_x = W - ps_aligned
            if max_y <= 0 or max_x <= 0:
                raise RuntimeError(
                    f"Image ({H}×{W}) is smaller than aligned patch size "
                    f"{ps_aligned} for scale {scale}×."
                )
            y = random.randint(0, max_y)
            x = random.randint(0, max_x)
            hr_band = hr_band[y:y + ps_aligned, x:x + ps_aligned]
            rgb_patch = rgb[y:y + ps_aligned, x:x + ps_aligned]
        else:
            rgb_patch = rgb

        # Synthetic LR: downsample then bicubic upsample back to original size
        H2, W2 = hr_band.shape
        lr_h, lr_w = H2 // scale, W2 // scale
        lr_down = cv2.resize(hr_band, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        lr_up   = cv2.resize(lr_down, (W2, H2),     interpolation=cv2.INTER_CUBIC)
        lr_up   = np.clip(lr_up, 0.0, 1.0)

        if self.augment and ps is not None:
            # Random horizontal / vertical flip
            if random.random() < 0.5:
                hr_band   = np.flipud(hr_band).copy()
                lr_up     = np.flipud(lr_up).copy()
                rgb_patch = np.flipud(rgb_patch).copy()
            if random.random() < 0.5:
                hr_band   = np.fliplr(hr_band).copy()
                lr_up     = np.fliplr(lr_up).copy()
                rgb_patch = np.fliplr(rgb_patch).copy()

        # (H, W) → (1, H, W); (H, W, 3) → (3, H, W)
        hr_t  = torch.from_numpy(hr_band[None])                         # (1, H, W)
        lr_t  = torch.from_numpy(lr_up[None])                           # (1, H, W)
        rgb_t = torch.from_numpy(rgb_patch.transpose(2, 0, 1))          # (3, H, W)
        inp_t = torch.cat([rgb_t, lr_t], dim=0)                         # (4, H, W)

        return {"lr": inp_t, "hr": hr_t, "band": band_idx, "scale": scale}


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = _pick_device(args.device)
    print(f"Device: {device}")

    model = GuidedRestormer(
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        expansion=args.expansion,
        num_refinement=args.num_refinement,
        rgb_enc_base=args.rgb_enc_base,
    ).to(device)

    if args.checkpoint:
        print(f"Loading pretrained weights from: {args.checkpoint}")
        load_pretrained_partial(model, args.checkpoint)

    dataset = MSRGBDataset(
        aligned_dir=Path(args.aligned_dir),
        rgb_dir=Path(args.rgb_dir),
        scales=args.scales,
        patch_size=args.patch_size,
        augment=True,
        cache_images=args.cache_images,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type != "cpu"),
        persistent_workers=(args.num_workers > 0),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    criterion = CharbonnierLoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_loss = float("inf")
    history: list[dict] = []
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)   # ~10 intra-epoch updates
    train_start = time.time()

    print(f"Training: {args.epochs} epochs × {n_batches} batches  "
          f"(logging every {log_every} batches)\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        recent_loss = 0.0   # rolling sum for intra-epoch window
        epoch_start = time.time()

        for step, batch in enumerate(loader, 1):
            inp = batch["lr"].to(device)   # (B, 4, H, W)
            hr  = batch["hr"].to(device)   # (B, 1, H, W)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(inp)
                loss = criterion(pred, hr)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            epoch_loss  += loss_val
            recent_loss += loss_val

            if step % log_every == 0 or step == n_batches:
                elapsed      = time.time() - epoch_start
                secs_per_bat = elapsed / step
                eta_epoch    = secs_per_bat * (n_batches - step)
                avg_recent   = recent_loss / log_every if step % log_every == 0 else recent_loss / (step % log_every or log_every)
                print(f"  E{epoch:03d} [{step:4d}/{n_batches}]  "
                      f"loss={avg_recent:.6f}  "
                      f"epoch ETA {eta_epoch:5.0f}s", flush=True)
                recent_loss = 0.0

        scheduler.step()
        epoch_secs = time.time() - epoch_start
        avg_loss   = epoch_loss / n_batches

        elapsed_total = time.time() - train_start
        eta_total     = elapsed_total / epoch * (args.epochs - epoch)
        improved      = " *" if avg_loss < best_loss else ""
        print(f"Epoch {epoch:03d}/{args.epochs}  loss={avg_loss:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"{epoch_secs:.0f}s/epoch  total ETA {eta_total/3600:.2f}h{improved}\n",
              flush=True)

        history.append({"epoch": epoch, "loss": avg_loss, "epoch_secs": round(epoch_secs, 1)})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), out_dir / "best.pth")

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), out_dir / f"epoch_{epoch:04d}.pth")

    torch.save(model.state_dict(), out_dir / "last.pth")
    with open(out_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    total_mins = (time.time() - train_start) / 60
    print(f"\nTraining done in {total_mins:.1f} min. Best loss: {best_loss:.6f}  → {out_dir/'best.pth'}")


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    device = _pick_device(args.device)
    print(f"Device: {device}")

    model = GuidedRestormer(
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        expansion=args.expansion,
        num_refinement=args.num_refinement,
        rgb_enc_base=args.rgb_enc_base,
    ).to(device)

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for eval mode.")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    aligned_dir = Path(args.aligned_dir)
    rgb_dir     = Path(args.rgb_dir)
    stems       = find_paired_stems(aligned_dir, rgb_dir)
    if not stems:
        raise RuntimeError("No paired stems found.")

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators: results[scale][band] = {psnr_restormer, psnr_bicubic, ssim_restormer, ssim_bicubic}
    BAND_NAMES = ["685nm", "725nm", "750nm", "1000nm"]
    from collections import defaultdict
    results: dict[int, dict[int, dict[str, list[float]]]] = {
        s: {b: {"psnr_sr": [], "psnr_bic": [], "ssim_sr": [], "ssim_bic": []}
            for b in range(4)}
        for s in args.scales
    }

    for stem_idx, stem in enumerate(stems):
        print(f"[{stem_idx+1}/{len(stems)}]  {stem}")
        dataset = MSRGBDataset(
            aligned_dir=aligned_dir, rgb_dir=rgb_dir,
            scales=args.scales, patch_size=None, augment=False,
        )
        bands, rgb = dataset._load_group(stem)

        for scale in args.scales:
            for band_idx, hr_band in enumerate(bands):
                H, W = hr_band.shape
                lr_h, lr_w = H // scale, W // scale
                lr_down = cv2.resize(hr_band, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
                bic     = cv2.resize(lr_down, (W, H),       interpolation=cv2.INTER_CUBIC)
                bic     = np.clip(bic, 0.0, 1.0)

                # Pad to multiple of 8 for clean PixelUnshuffle passes
                pad_h = (8 - H % 8) % 8
                pad_w = (8 - W % 8) % 8
                bic_p   = np.pad(bic,     ((0, pad_h), (0, pad_w)), "reflect")
                rgb_p   = np.pad(rgb,     ((0, pad_h), (0, pad_w), (0, 0)), "reflect")
                hr_p    = np.pad(hr_band, ((0, pad_h), (0, pad_w)), "reflect")

                lr_t  = torch.from_numpy(bic_p[None, None]).float().to(device)
                rgb_t = torch.from_numpy(rgb_p.transpose(2, 0, 1)[None]).float().to(device)
                inp_t = torch.cat([rgb_t, lr_t], dim=1)

                sr = model(inp_t).squeeze().cpu().numpy()
                sr = np.clip(sr[:H, :W], 0.0, 1.0)

                hr_u8  = (hr_band[:H, :W] * 255).round().astype(np.uint8)
                sr_u8  = (sr              * 255).round().astype(np.uint8)
                bic_u8 = (bic[:H, :W]    * 255).round().astype(np.uint8)

                r = results[scale][band_idx]
                r["psnr_sr"].append(_psnr(sr_u8, hr_u8))
                r["psnr_bic"].append(_psnr(bic_u8, hr_u8))
                r["ssim_sr"].append(_ssim(sr_u8, hr_u8))
                r["ssim_bic"].append(_ssim(bic_u8, hr_u8))

                if out_dir:
                    cv2.imwrite(
                        str(out_dir / f"{stem}_b{band_idx}_{scale}x_sr.png"),
                        sr_u8,
                    )

    # Print summary table
    print("\n" + "=" * 72)
    print(f"{'Scale':>6}  {'Band':>8}  "
          f"{'PSNR-SR':>9}  {'PSNR-Bic':>9}  "
          f"{'SSIM-SR':>8}  {'SSIM-Bic':>8}")
    print("-" * 72)

    summary: list[dict] = []
    for scale in args.scales:
        for b in range(4):
            r    = results[scale][b]
            psr  = np.mean(r["psnr_sr"])
            pbic = np.mean(r["psnr_bic"])
            ssr  = np.mean(r["ssim_sr"])
            sbic = np.mean(r["ssim_bic"])
            print(f"{scale:>5}×  {BAND_NAMES[b]:>8}  "
                  f"{psr:>9.2f}  {pbic:>9.2f}  "
                  f"{ssr:>8.4f}  {sbic:>8.4f}")
            summary.append({
                "scale": scale, "band": BAND_NAMES[b],
                "psnr_sr": round(psr, 4), "psnr_bicubic": round(pbic, 4),
                "ssim_sr": round(ssr, 6), "ssim_bicubic": round(sbic, 6),
                "psnr_gain": round(psr - pbic, 4),
                "ssim_gain": round(ssr - sbic, 6),
            })

    print("=" * 72)

    if out_dir:
        with open(out_dir / "eval_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results written to {out_dir / 'eval_results.json'}")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _pick_device(pref: str) -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    """Add shared model architecture arguments."""
    p.add_argument("--dim",           type=int,   default=48,
                   help="Base channel dimension (must match pretrained if loading)")
    p.add_argument("--num-blocks",    type=int,   nargs=4, default=[4, 6, 6, 8],
                   metavar="N",
                   help="Transformer blocks at each of the 4 encoder levels")
    p.add_argument("--num-heads",     type=int,   nargs=4, default=[1, 2, 4, 8],
                   metavar="N",
                   help="Attention heads at each encoder level")
    p.add_argument("--expansion",     type=float, default=2.66,
                   help="FFN expansion factor")
    p.add_argument("--num-refinement", type=int,  default=4,
                   help="Refinement transformer blocks at the decoder output")
    p.add_argument("--rgb-enc-base",  type=int,   default=32,
                   help="Base channels for the RGB guidance encoder")
    p.add_argument("--device",        type=str,   default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--aligned-dir", required=True, metavar="DIR",
                   help="align_multispec.py output directory (MS bands)")
    p.add_argument("--rgb-dir",     required=True, metavar="DIR",
                   help="Directory with aligned RGB files ({stem}_aligned.ext)")
    p.add_argument("--scales",      type=int, nargs="+", default=[4, 8],
                   metavar="N",
                   help="Downsampling scale factors to test")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guided SR of aligned multispectral bands via Restormer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── train ──
    tr = sub.add_parser("train", help="Fine-tune a GuidedRestormer on aligned MS+RGB pairs.")
    _add_data_args(tr)
    _add_model_args(tr)
    tr.add_argument("--checkpoint",  type=str, default=None, metavar="PATH",
                    help="Pretrained Restormer checkpoint to initialise from (optional)")
    tr.add_argument("--epochs",      type=int, default=100)
    tr.add_argument("--batch-size",  type=int, default=4)
    tr.add_argument("--patch-size",  type=int, default=128,
                    help="Training crop size (must be divisible by max(scales)×8)")
    tr.add_argument("--lr",          type=float, default=2e-4)
    tr.add_argument("--num-workers", type=int, default=4)
    tr.add_argument("--save-every",  type=int, default=10,
                    help="Save a checkpoint every N epochs (best.pth always saved)")
    tr.add_argument("--amp", action="store_true", default=False,
                    help="Enable automatic mixed precision (bfloat16) — halves VRAM, ~1.5× faster")
    tr.add_argument("--no-cache", dest="cache_images", action="store_false", default=True,
                    help="Disable RAM image cache (use if RAM is insufficient)")
    tr.add_argument("-o", "--output-dir", default="runs/sr/", metavar="DIR")

    # ── eval ──
    ev = sub.add_parser("eval", help="Evaluate a trained checkpoint; report PSNR/SSIM per band and scale.")
    _add_data_args(ev)
    _add_model_args(ev)
    ev.add_argument("--checkpoint", type=str, required=True, metavar="PATH",
                    help="Trained GuidedRestormer checkpoint (.pth)")
    ev.add_argument("-o", "--output-dir", default=None, metavar="DIR",
                    help="If given, save SR images and eval_results.json here")

    args = parser.parse_args()

    # Normalise list args from argparse
    args.num_blocks = list(args.num_blocks)
    args.num_heads  = list(args.num_heads)

    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
