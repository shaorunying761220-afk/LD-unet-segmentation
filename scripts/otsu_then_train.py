#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-stage pipeline for LD patch segmentation:
Stage A: Otsu auto-mask TIFF + preview for manual inspection
Stage B: Train lightweight U-Net on images + masks

Install:
  pip install numpy pandas pillow tifffile scikit-image torch matplotlib tqdm

Examples:
  python otsu_then_train.py --output_root /path/to/OUTPUT_ROOT --stage otsu
  python otsu_then_train.py --output_root /path/to/OUTPUT_ROOT --stage train --epochs 30
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from skimage.filters import threshold_otsu
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Paths:
    output_root: Path
    annotation_dir: Path
    patches_img_dir: Path
    patches_meta_dir: Path
    annotation_images_dir: Path
    masks_dir: Path
    preview_dir: Path
    pred_preview_dir: Path
    train_artifacts_dir: Path


def build_paths(output_root: Path) -> Paths:
    annotation_dir = output_root / "ANNOTATION_DIR"
    patches_img_dir = output_root / "PATCHES_IMG_DIR"
    patches_meta_dir = output_root / "PATCHES_META_DIR"
    annotation_images_dir = annotation_dir / "images"
    masks_dir = annotation_dir / "masks"
    preview_dir = annotation_dir / "preview"
    pred_preview_dir = annotation_dir / "pred_preview"
    train_artifacts_dir = patches_meta_dir / "train_artifacts"

    for d in [annotation_dir, patches_img_dir, patches_meta_dir, masks_dir, preview_dir, pred_preview_dir, train_artifacts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return Paths(
        output_root=output_root,
        annotation_dir=annotation_dir,
        patches_img_dir=patches_img_dir,
        patches_meta_dir=patches_meta_dir,
        annotation_images_dir=annotation_images_dir,
        masks_dir=masks_dir,
        preview_dir=preview_dir,
        pred_preview_dir=pred_preview_dir,
        train_artifacts_dir=train_artifacts_dir,
    )


def find_images_recursive(root: Path) -> List[Path]:
    exts = ["*.tif", "*.tiff", "*.png"]
    files: List[Path] = []
    for ext in exts:
        files.extend(root.rglob(ext))
    filtered: List[Path] = []
    for p in files:
        name = p.name
        if name.startswith("._") or name.startswith("."):
            continue
        filtered.append(p)
    return sorted(filtered)


def to_grayscale_2d(arr: np.ndarray, input_mode: str) -> np.ndarray:
    if arr.ndim == 2:
        return arr

    if arr.ndim != 3:
        raise ValueError(f"Unsupported image ndim={arr.ndim}, expected 2D or 3D.")

    # Heuristic for channel axis.
    # HWC case: last dim small
    if arr.shape[-1] in (1, 2, 3, 4):
        if input_mode == "channel0":
            out = arr[..., 0]
        else:
            out = arr.astype(np.float32).mean(axis=-1)
        return out

    # CHW case: first dim small
    if arr.shape[0] in (1, 2, 3, 4):
        if input_mode == "channel0":
            out = arr[0, ...]
        else:
            out = arr.astype(np.float32).mean(axis=0)
        return out

    # fallback
    if input_mode == "channel0":
        return arr[0, ...]
    return arr.astype(np.float32).mean(axis=0)


def read_image(path: Path, input_mode: str = "grayscale") -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        arr = tifffile.imread(str(path))
    elif suffix == ".png":
        arr = np.array(Image.open(path))
    else:
        raise ValueError(f"Unsupported image extension: {path.suffix}")

    arr = to_grayscale_2d(arr, input_mode=input_mode)
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale 2D image after conversion, got {arr.shape} for {path}")
    return arr


def normalize_display(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    lo = np.percentile(x, 1)
    hi = np.percentile(x, 99)
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return x


def save_otsu_preview(img: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    disp = normalize_display(img)
    overlay = np.stack([disp, disp, disp], axis=-1)
    overlay[..., 0] = np.maximum(overlay[..., 0], mask * 1.0)
    overlay[..., 1] = overlay[..., 1] * (1.0 - 0.35 * mask)
    overlay[..., 2] = overlay[..., 2] * (1.0 - 0.35 * mask)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(disp, cmap="gray")
    axs[0].set_title("Image")
    axs[0].axis("off")
    axs[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axs[1].set_title("Otsu Mask")
    axs[1].axis("off")
    axs[2].imshow(overlay)
    axs[2].set_title("Overlay")
    axs[2].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def sample_image_subset(
    image_files: List[Path],
    max_patches: int,
    seed: int,
) -> List[Path]:
    if max_patches <= 0 or max_patches >= len(image_files):
        return image_files
    rng = np.random.default_rng(seed)
    idx = np.arange(len(image_files))
    rng.shuffle(idx)
    picked = sorted(idx[:max_patches].tolist())
    return [image_files[i] for i in picked]


def stage_otsu(paths: Paths, input_mode: str, max_patches: int, seed: int) -> None:
    image_files = find_images_recursive(paths.patches_img_dir)
    if not image_files:
        fallback = find_images_recursive(paths.annotation_images_dir)
        if fallback:
            print(f"[INFO] No images in PATCHES_IMG_DIR, fallback to {paths.annotation_images_dir}")
            image_files = fallback
        else:
            raise FileNotFoundError(
                f"No patch images found in {paths.patches_img_dir} or {paths.annotation_images_dir}"
            )

    image_files = sample_image_subset(image_files=image_files, max_patches=max_patches, seed=seed)
    print(f"[INFO] Stage A uses {len(image_files)} images for Otsu labeling", flush=True)

    rows: List[Dict] = []
    for img_path in tqdm(image_files, desc="Stage A Otsu"):
        stem = img_path.stem
        mask_path = paths.masks_dir / f"{stem}.tif"
        preview_path = paths.preview_dir / f"{stem}.png"

        img = read_image(img_path, input_mode=input_mode)
        img_f32 = img.astype(np.float32)
        h, w = img_f32.shape

        const_fallback = False
        try:
            thresh = float(threshold_otsu(img_f32))
            mask = (img_f32 > thresh).astype(np.uint8)
        except ValueError:
            thresh = float(img_f32.max())
            mask = np.zeros_like(img_f32, dtype=np.uint8)
            const_fallback = True

        tifffile.imwrite(str(mask_path), mask.astype(np.uint8), photometric="minisblack")
        save_otsu_preview(img_f32, mask, preview_path)

        rows.append(
            {
                "stem": stem,
                "image_path": str(img_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "dtype": str(img.dtype),
                "H": int(h),
                "W": int(w),
                "otsu_threshold": thresh,
                "min": float(np.min(img_f32)),
                "max": float(np.max(img_f32)),
                "mean": float(np.mean(img_f32)),
                "p99": float(np.percentile(img_f32, 99)),
                "constant_image_fallback_to_zero_mask": const_fallback,
            }
        )

    manifest_path = paths.patches_meta_dir / "otsu_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"[INFO] Saved manifest to: {manifest_path}")
    print(
        "Stage A complete: Otsu masks saved to ANNOTATION_DIR/masks and previews saved to "
        "ANNOTATION_DIR/preview.\nPlease inspect previews. If OK, rerun with --stage train."
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_image_for_train(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.uint16):
        x = img.astype(np.float32) / 65535.0
    elif np.issubdtype(img.dtype, np.uint8):
        x = img.astype(np.float32) / 255.0
    else:
        x = img.astype(np.float32)
        vmin = float(np.min(x))
        vmax = float(np.max(x))
        if vmax > vmin:
            x = (x - vmin) / (vmax - vmin)
        else:
            x = np.zeros_like(x, dtype=np.float32)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def normalize_mask01(mask: np.ndarray) -> np.ndarray:
    uniq = set(int(v) for v in np.unique(mask).tolist())
    if uniq.issubset({0, 1}):
        return mask.astype(np.uint8)
    if uniq.issubset({0, 255}):
        return (mask > 0).astype(np.uint8)
    # fallback for imperfect masks
    return (mask > 0).astype(np.uint8)


def read_mask(mask_path: Path) -> np.ndarray:
    suffix = mask_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        mask = tifffile.imread(str(mask_path))
    elif suffix == ".npy":
        mask = np.load(mask_path)
    else:
        raise ValueError(f"Unsupported mask extension: {mask_path.suffix}")
    return normalize_mask01(mask)


def collect_pairs(paths: Paths, strict_mask_match: bool = False) -> List[Tuple[str, Path, Path]]:
    image_files = find_images_recursive(paths.patches_img_dir)
    if not image_files:
        raise FileNotFoundError(f"No image files found in {paths.patches_img_dir}")

    missing_masks: List[str] = []
    pairs: List[Tuple[str, Path, Path]] = []
    for img_path in image_files:
        stem = img_path.stem
        mask_path_tif = paths.masks_dir / f"{stem}.tif"
        mask_path_tiff = paths.masks_dir / f"{stem}.tiff"
        mask_path_npy = paths.masks_dir / f"{stem}.npy"
        if mask_path_tif.exists():
            mask_path = mask_path_tif
        elif mask_path_tiff.exists():
            mask_path = mask_path_tiff
        elif mask_path_npy.exists():
            mask_path = mask_path_npy
        else:
            missing_masks.append(stem)
            continue
        pairs.append((stem, img_path, mask_path))

    if missing_masks and strict_mask_match:
        examples = ", ".join(missing_masks[:8])
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} mask(s) in {paths.masks_dir}. "
            f"Examples: {examples}. Please run --stage otsu first."
        )
    if missing_masks and not strict_mask_match:
        print(
            f"[WARN] Missing {len(missing_masks)} masks; training will use available pairs only.",
            flush=True,
        )
    if not pairs:
        raise RuntimeError("No matched image-mask pairs found.")
    return pairs


def split_train_val(n: int, seed: int, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    train_n = max(1, int(round(n * train_ratio)))
    train_n = min(train_n, n - 1) if n > 1 else 1
    train_idx = idx[:train_n]
    val_idx = idx[train_n:] if n > 1 else idx[:1]
    return train_idx, val_idx


def stage_train(
    paths: Paths,
    input_mode: str,
    epochs: int,
    lr: float,
    batch_size: int,
    base_channels: int,
    seed: int,
    save_pred_preview: bool,
    strict_mask_match: bool,
) -> None:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    set_seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    pairs = collect_pairs(paths, strict_mask_match=strict_mask_match)
    stems = [p[0] for p in pairs]
    train_idx, val_idx = split_train_val(len(pairs), seed=seed, train_ratio=0.8)

    class PatchDataset(Dataset):
        def __init__(self, subset_idx: Sequence[int]):
            self.subset_idx = list(subset_idx)

        def __len__(self) -> int:
            return len(self.subset_idx)

        def __getitem__(self, i: int):
            idx = self.subset_idx[i]
            stem, img_path, mask_path = pairs[idx]
            img = read_image(img_path, input_mode=input_mode)
            x = normalize_image_for_train(img)
            y = read_mask(mask_path)
            x = torch.from_numpy(x).unsqueeze(0).float()
            y = torch.from_numpy(y).long()
            return x, y, stem

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

            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
            self.dec3 = DoubleConv(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
            self.dec2 = DoubleConv(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
            self.dec1 = DoubleConv(base * 2, base)
            self.outc = nn.Conv2d(base, out_ch, kernel_size=1)

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

    def dice_iou_from_logits(logits: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
        pred = torch.argmax(logits, dim=1)
        pred_fg = (pred == 1).float()
        tgt_fg = (target == 1).float()
        eps = 1e-6
        inter = (pred_fg * tgt_fg).sum(dim=(1, 2))
        union = pred_fg.sum(dim=(1, 2)) + tgt_fg.sum(dim=(1, 2))
        dice = ((2.0 * inter + eps) / (union + eps)).mean().item()
        iou = ((inter + eps) / (pred_fg.sum(dim=(1, 2)) + tgt_fg.sum(dim=(1, 2)) - inter + eps)).mean().item()
        return dice, iou

    train_ds = PatchDataset(train_idx)
    val_ds = PatchDataset(val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNetSmall(in_ch=1, out_ch=2, base=base_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_dice = -1.0
    best_epoch = -1
    best_path = paths.train_artifacts_dir / "best_unet.pt"
    log_rows: List[Dict] = []

    print(f"[INFO] Train size: {len(train_ds)}, Val size: {len(val_ds)}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= max(1, len(train_ds))

        model.eval()
        val_loss = 0.0
        val_dices: List[float] = []
        val_ious: List[float] = []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                dice, iou = dice_iou_from_logits(logits, yb)
                val_dices.append(dice)
                val_ious.append(iou)
        val_loss /= max(1, len(val_ds))
        val_dice = float(np.mean(val_dices)) if val_dices else 0.0
        val_iou = float(np.mean(val_ious)) if val_ious else 0.0

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"val_dice={val_dice:.4f} | val_iou={val_iou:.4f}"
        , flush=True)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
            }
        )

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_dice": best_dice,
                    "config": {
                        "epochs": epochs,
                        "lr": lr,
                        "batch_size": batch_size,
                        "base_channels": base_channels,
                        "seed": seed,
                        "input_mode": input_mode,
                    },
                },
                best_path,
            )

    pd.DataFrame(log_rows).to_csv(paths.train_artifacts_dir / "train_log.csv", index=False)
    print(f"[INFO] Best epoch: {best_epoch}, best val Dice: {best_dice:.4f}", flush=True)
    print(f"[INFO] Best model saved: {best_path}", flush=True)

    if not save_pred_preview:
        split_df = pd.DataFrame(
            {
                "stem": stems,
                "split": [
                    "train" if i in set(train_idx.tolist()) else "val"
                    for i in range(len(stems))
                ],
            }
        )
        split_df.to_csv(paths.train_artifacts_dir / "split.csv", index=False)
        print("[INFO] skip_pred_preview=True, skipped pred preview export.", flush=True)
        return

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Prediction previews for all matched pairs.
    for stem, img_path, mask_path in tqdm(pairs, desc="Saving pred preview"):
        img = read_image(img_path, input_mode=input_mode)
        gt = read_mask(mask_path)
        x = normalize_image_for_train(img)
        xb = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        disp = normalize_display(img)
        overlay = np.stack([disp, disp, disp], axis=-1)
        overlay[..., 1] = np.maximum(overlay[..., 1], pred * 1.0)
        overlay[..., 0] = overlay[..., 0] * (1.0 - 0.35 * pred)
        overlay[..., 2] = overlay[..., 2] * (1.0 - 0.35 * pred)

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(disp, cmap="gray")
        axs[0].set_title("Image")
        axs[0].axis("off")
        axs[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axs[1].set_title("GT")
        axs[1].axis("off")
        axs[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axs[2].set_title("Pred")
        axs[2].axis("off")
        axs[3].imshow(overlay)
        axs[3].set_title("Overlay")
        axs[3].axis("off")
        fig.tight_layout()
        fig.savefig(paths.pred_preview_dir / f"{stem}.png", dpi=120)
        plt.close(fig)

    split_df = pd.DataFrame(
        {
            "stem": stems,
            "split": [
                "train" if i in set(train_idx.tolist()) else "val"
                for i in range(len(stems))
            ],
        }
    )
    split_df.to_csv(paths.train_artifacts_dir / "split.csv", index=False)
    print(f"[INFO] Prediction previews saved to: {paths.pred_preview_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Otsu -> inspect -> train U-Net pipeline")
    parser.add_argument("--output_root", type=str, required=True, help="Path to OUTPUT_ROOT")
    parser.add_argument("--stage", type=str, default="otsu", choices=["otsu", "train"])
    parser.add_argument("--input_mode", type=str, default="grayscale", choices=["channel0", "grayscale"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_pred_preview", action="store_true", help="Skip pred preview export for faster runs")
    parser.add_argument("--max_otsu_patches", type=int, default=0, help="Stage otsu only: 0 means all, >0 means random subset size")
    parser.add_argument("--strict_mask_match", action="store_true", help="Stage train: require every image has mask")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    paths = build_paths(output_root)

    if args.stage == "otsu":
        stage_otsu(
            paths=paths,
            input_mode=args.input_mode,
            max_patches=args.max_otsu_patches,
            seed=args.seed,
        )
        return

    if args.stage == "train":
        stage_train(
            paths=paths,
            input_mode=args.input_mode,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            base_channels=args.base_channels,
            seed=args.seed,
            save_pred_preview=not args.skip_pred_preview,
            strict_mask_match=args.strict_mask_match,
        )
        return

    raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
