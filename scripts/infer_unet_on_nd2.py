#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run trained U-Net on raw ND2 files (C0) and save full-image masks.

Install:
  pip install nd2 numpy pandas tifffile pillow matplotlib torch tqdm

Example:
  python infer_unet_on_nd2.py \
    --raw_nd2_dir /path/to/RAW_ND2_DIR \
    --output_root /path/to/OUTPUT_ROOT \
    --model_path /path/to/OUTPUT_ROOT/PATCHES_META_DIR/train_artifacts/best_unet.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nd2
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
from tqdm import tqdm


def normalize_display(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    lo = np.percentile(x, 1)
    hi = np.percentile(x, 99)
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return x


def normalize_for_model(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.uint16):
        x = img.astype(np.float32) / 65535.0
    elif np.issubdtype(img.dtype, np.uint8):
        x = img.astype(np.float32) / 255.0
    else:
        x = img.astype(np.float32)
        mn = float(np.min(x))
        mx = float(np.max(x))
        if mx > mn:
            x = (x - mn) / (mx - mn)
        else:
            x = np.zeros_like(x, dtype=np.float32)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def load_nd2_channel0_2d(nd2_path: Path, channel_index: int, t_index: int, projection_mode: str) -> np.ndarray:
    with nd2.ND2File(nd2_path) as f:
        arr = f.asarray()
        axes = list(f.sizes.keys())

    allowed = {"T", "C", "Z", "Y", "X"}
    for ax_name in list(axes):
        if ax_name not in allowed:
            ax = axes.index(ax_name)
            arr = np.take(arr, indices=0, axis=ax)
            axes.pop(ax)

    if "T" in axes:
        ax = axes.index("T")
        if not (0 <= t_index < arr.shape[ax]):
            raise IndexError(f"T index out of range for {nd2_path.name}: {t_index}")
        arr = np.take(arr, indices=t_index, axis=ax)
        axes.pop(ax)

    if "C" in axes:
        ax = axes.index("C")
        if not (0 <= channel_index < arr.shape[ax]):
            raise IndexError(f"C index out of range for {nd2_path.name}: {channel_index}")
        arr = np.take(arr, indices=channel_index, axis=ax)
        axes.pop(ax)

    if "Z" in axes:
        ax = axes.index("Z")
        if projection_mode == "max":
            arr = np.max(arr, axis=ax)
        elif projection_mode == "mean":
            arr = np.mean(arr, axis=ax)
        elif projection_mode == "z0":
            arr = np.take(arr, indices=0, axis=ax)
        else:
            raise ValueError("projection_mode must be one of: max, mean, z0")
        axes.pop(ax)

    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"Failed to find Y/X axes for {nd2_path.name}, axes={axes}")

    y_ax = axes.index("Y")
    x_ax = axes.index("X")
    arr = np.moveaxis(arr, [y_ax, x_ax], [0, 1])
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D after projection for {nd2_path.name}, got {arr.shape}")

    if arr.dtype != np.uint16:
        if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
            arr = np.clip(arr, 0, 65535).astype(np.uint16)
        else:
            raise TypeError(f"Unsupported dtype {arr.dtype} in {nd2_path.name}")
    return arr


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 2, base: int = 16):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.outc(d1)


@dataclass
class InferConfig:
    raw_nd2_dir: Path
    output_root: Path
    model_path: Path
    projection_mode: str
    t_index: int
    channel_index: int
    tile_size: int
    tile_stride: int
    prob_threshold: float


def make_dirs(output_root: Path) -> Dict[str, Path]:
    root = output_root / "INFERENCE_UNET"
    masks = root / "masks"
    prob = root / "prob"
    preview = root / "preview"
    for d in [root, masks, prob, preview]:
        d.mkdir(parents=True, exist_ok=True)
    return {"ROOT": root, "MASKS": masks, "PROB": prob, "PREVIEW": preview}


def tile_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def build_hann_window(tile_size: int, min_weight: float = 0.05) -> np.ndarray:
    if tile_size <= 1:
        return np.ones((tile_size, tile_size), dtype=np.float32)
    w1 = np.hanning(tile_size).astype(np.float32)
    w2 = np.outer(w1, w1).astype(np.float32)
    w2 = w2 / max(float(w2.max()), 1e-6)
    w2 = np.clip(w2, min_weight, None)
    return w2


def infer_one_image(
    model: nn.Module,
    img01: np.ndarray,
    tile_size: int,
    tile_stride: int,
    prob_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img01.shape
    ys = tile_starts(h, tile_size, tile_stride)
    xs = tile_starts(w, tile_size, tile_stride)

    logits_sum = np.zeros((2, h, w), dtype=np.float32)
    weights_sum = np.zeros((h, w), dtype=np.float32)
    weight_full = build_hann_window(tile_size=tile_size, min_weight=0.05)

    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = img01[y:y + tile_size, x:x + tile_size]
                ph, pw = patch.shape
                if ph < tile_size or pw < tile_size:
                    pad = np.zeros((tile_size, tile_size), dtype=np.float32)
                    pad[:ph, :pw] = patch
                    patch = pad
                xb = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float()
                logits = model(xb).squeeze(0).cpu().numpy()
                logits = logits[:, :ph, :pw]
                w_patch = weight_full[:ph, :pw]
                logits_sum[:, y:y + ph, x:x + pw] += logits * w_patch[None, ...]
                weights_sum[y:y + ph, x:x + pw] += w_patch

    weights_sum = np.maximum(weights_sum, 1e-6)
    logits_avg = logits_sum / weights_sum[None, ...]
    prob_fg = torch.softmax(torch.from_numpy(logits_avg), dim=0)[1].numpy().astype(np.float32)
    pred = (prob_fg >= float(prob_threshold)).astype(np.uint8)
    return pred, prob_fg


def save_preview(img16: np.ndarray, pred: np.ndarray, out_path: Path) -> None:
    disp = normalize_display(img16)
    overlay = np.stack([disp, disp, disp], axis=-1)
    overlay[..., 0] = np.maximum(overlay[..., 0], pred * 1.0)
    overlay[..., 1] = overlay[..., 1] * (1.0 - 0.35 * pred)
    overlay[..., 2] = overlay[..., 2] * (1.0 - 0.35 * pred)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(disp, cmap="gray")
    axs[0].set_title("Image")
    axs[0].axis("off")
    axs[1].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axs[1].set_title("Pred Mask")
    axs[1].axis("off")
    axs[2].imshow(overlay)
    axs[2].set_title("Overlay")
    axs[2].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_inference(cfg: InferConfig) -> None:
    dirs = make_dirs(cfg.output_root)
    nd2_files = sorted([p for p in cfg.raw_nd2_dir.glob("*.nd2") if not p.name.startswith("._")])
    if not nd2_files:
        raise FileNotFoundError(f"No nd2 files found in {cfg.raw_nd2_dir}")

    ckpt = torch.load(cfg.model_path, map_location="cpu")
    base_channels = int(ckpt.get("config", {}).get("base_channels", 16))
    model = UNetSmall(in_ch=1, out_ch=2, base=base_channels)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []
    for nd2_path in tqdm(nd2_files, desc="Infer ND2"):
        img16 = load_nd2_channel0_2d(
            nd2_path=nd2_path,
            channel_index=cfg.channel_index,
            t_index=cfg.t_index,
            projection_mode=cfg.projection_mode,
        )
        img01 = normalize_for_model(img16)
        pred, prob_fg = infer_one_image(
            model=model,
            img01=img01,
            tile_size=cfg.tile_size,
            tile_stride=cfg.tile_stride,
            prob_threshold=cfg.prob_threshold,
        )

        stem = nd2_path.stem
        mask_path = dirs["MASKS"] / f"{stem}__C0__pred.tif"
        prob_path = dirs["PROB"] / f"{stem}__C0__prob_fg.tif"
        preview_path = dirs["PREVIEW"] / f"{stem}__C0__preview.png"

        tifffile.imwrite(str(mask_path), pred.astype(np.uint8), photometric="minisblack")
        tifffile.imwrite(str(prob_path), (prob_fg * 65535.0).astype(np.uint16), photometric="minisblack")
        save_preview(img16=img16, pred=pred, out_path=preview_path)

        rows.append(
            {
                "source_nd2": nd2_path.name,
                "mask_path": str(mask_path.resolve()),
                "prob_path": str(prob_path.resolve()),
                "preview_path": str(preview_path.resolve()),
                "H": int(img16.shape[0]),
                "W": int(img16.shape[1]),
                "fg_ratio": float(pred.mean()),
            }
        )

    manifest = dirs["ROOT"] / "inference_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"[DONE] Inference finished for {len(rows)} ND2 files.", flush=True)
    print(f"[DONE] Masks   : {dirs['MASKS']}", flush=True)
    print(f"[DONE] Preview : {dirs['PREVIEW']}", flush=True)
    print(f"[DONE] Manifest: {manifest}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Infer trained U-Net on raw ND2")
    p.add_argument("--raw_nd2_dir", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--projection_mode", type=str, default="max", choices=["max", "mean", "z0"])
    p.add_argument("--t_index", type=int, default=0)
    p.add_argument("--channel_index", type=int, default=0)
    p.add_argument("--tile_size", type=int, default=512)
    p.add_argument("--tile_stride", type=int, default=256, help="Use overlap (e.g., 256) to reduce seam artifacts")
    p.add_argument("--prob_threshold", type=float, default=0.5, help="Foreground threshold on prob map")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = InferConfig(
        raw_nd2_dir=Path(args.raw_nd2_dir),
        output_root=Path(args.output_root),
        model_path=Path(args.model_path),
        projection_mode=args.projection_mode,
        t_index=args.t_index,
        channel_index=args.channel_index,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        prob_threshold=args.prob_threshold,
    )
    run_inference(cfg)


if __name__ == "__main__":
    main()
