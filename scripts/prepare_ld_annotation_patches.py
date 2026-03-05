#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare 16-bit LD segmentation patches from ND2 files for manual annotation (Fiji workflow).

Install (CPU-friendly):
  pip install nd2 numpy pandas tifffile tqdm

Usage:
  python prepare_ld_annotation_patches.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

try:
    import nd2
except ImportError as e:
    raise ImportError(
        "Failed to import 'nd2'. Please install with: pip install nd2"
    ) from e


@dataclass
class Config:
    # ---------- Paths ----------
    RAW_ND2_DIR: str = "/Volumes/USB/RS_lab_results/LDs/Unet/RAW_ND2_DIR"
    OUTPUT_ROOT: str = "/Volumes/USB/RS_lab_results/LDs/Unet/OUTPUT_ROOT"

    # ---------- Patch generation ----------
    patch_size: int = 512
    stride: int = 512
    sampling_mode: str = "grid"  # "grid" or "random"
    random_patches_per_image: int = 20
    seed: int = 42

    # ---------- ND2 interpretation ----------
    channel_index: int = 0
    t_index: int = 0
    projection_mode: str = "max"  # "max", "mean", "z0"

    # ---------- Random sampling options ----------
    random_bias_to_bright: bool = False
    bright_threshold_percentile: float = 99.0
    bright_bias_strength: float = 8.0  # larger => stronger preference
    random_candidate_factor: int = 25  # candidate_pool = N * factor

    # ---------- IO ----------
    image_ext: str = ".tif"  # recommended for 16-bit
    copy_to_annotation_images: bool = True
    patch_index_zero_pad: int = 3

    # ---------- Validation ----------
    normalize_binary_masks: bool = False
    normalize_overwrite: bool = True


CONFIG = Config()


def ensure_dirs(output_root: Path) -> Dict[str, Path]:
    patches_img_dir = output_root / "PATCHES_IMG_DIR"
    patches_meta_dir = output_root / "PATCHES_META_DIR"
    annotation_dir = output_root / "ANNOTATION_DIR"
    anno_images = annotation_dir / "images"
    anno_masks = annotation_dir / "masks"

    for p in [patches_img_dir, patches_meta_dir, annotation_dir, anno_images, anno_masks]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "PATCHES_IMG_DIR": patches_img_dir,
        "PATCHES_META_DIR": patches_meta_dir,
        "ANNOTATION_DIR": annotation_dir,
        "ANNO_IMAGES": anno_images,
        "ANNO_MASKS": anno_masks,
    }


def _take_axis(arr: np.ndarray, axes: List[str], axis_name: str, index: int) -> Tuple[np.ndarray, List[str]]:
    if axis_name not in axes:
        return arr, axes
    ax = axes.index(axis_name)
    if index < 0 or index >= arr.shape[ax]:
        raise IndexError(
            f"Axis '{axis_name}' index {index} is out of range (size={arr.shape[ax]})."
        )
    arr = np.take(arr, indices=index, axis=ax)
    axes.pop(ax)
    return arr, axes


def _reduce_axis(arr: np.ndarray, axes: List[str], axis_name: str, mode: str) -> Tuple[np.ndarray, List[str]]:
    if axis_name not in axes:
        return arr, axes
    ax = axes.index(axis_name)
    if mode == "max":
        arr = np.max(arr, axis=ax)
    elif mode == "mean":
        arr = np.mean(arr, axis=ax)
    elif mode == "z0":
        arr = np.take(arr, indices=0, axis=ax)
    else:
        raise ValueError(f"Unsupported projection_mode: {mode}. Use max/mean/z0.")
    axes.pop(ax)
    return arr, axes


def load_nd2_channel0_2d(nd2_path: Path, cfg: Config) -> np.ndarray:
    with nd2.ND2File(nd2_path) as f:
        arr = f.asarray()
        axes = list(f.sizes.keys())  # typically ordered like T, C, Z, Y, X

    # Collapse unsupported leading dimensions (e.g., P/S/Scene) by taking index 0.
    for ax_name in list(axes):
        if ax_name not in {"T", "C", "Z", "Y", "X"}:
            arr, axes = _take_axis(arr, axes, ax_name, 0)

    # Select T and C.
    arr, axes = _take_axis(arr, axes, "T", cfg.t_index)
    arr, axes = _take_axis(arr, axes, "C", cfg.channel_index)

    # Z projection if needed.
    arr, axes = _reduce_axis(arr, axes, "Z", cfg.projection_mode)

    if "Y" not in axes or "X" not in axes:
        raise ValueError(
            f"Could not resolve Y/X axes for {nd2_path.name}. Remaining axes: {axes}, shape={arr.shape}"
        )

    # Reorder to YX if needed.
    y_ax = axes.index("Y")
    x_ax = axes.index("X")
    arr = np.moveaxis(arr, [y_ax, x_ax], [0, 1])
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image after projection for {nd2_path.name}, got shape={arr.shape}")

    # Preserve dynamic range; ensure uint16 for downstream TIFF writing consistency.
    if arr.dtype != np.uint16:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0, 65535).astype(np.uint16)
        elif np.issubdtype(arr.dtype, np.integer):
            arr = np.clip(arr, 0, 65535).astype(np.uint16)
        else:
            raise TypeError(f"Unsupported dtype {arr.dtype} in {nd2_path.name}")
    return arr


def compute_stats(patch: np.ndarray) -> Dict[str, float]:
    return {
        "min": int(np.min(patch)),
        "max": int(np.max(patch)),
        "mean": float(np.mean(patch)),
        "p99": float(np.percentile(patch, 99)),
    }


def grid_coords(h: int, w: int, patch_size: int, stride: int) -> List[Tuple[int, int]]:
    if h < patch_size or w < patch_size:
        return []
    ys = range(0, h - patch_size + 1, stride)
    xs = range(0, w - patch_size + 1, stride)
    return [(y, x) for y in ys for x in xs]


def random_coords(
    image: np.ndarray,
    n: int,
    patch_size: int,
    rng: np.random.Generator,
    bias_to_bright: bool,
    bright_threshold_percentile: float,
    bright_bias_strength: float,
    candidate_factor: int,
) -> List[Tuple[int, int]]:
    h, w = image.shape
    max_y = h - patch_size
    max_x = w - patch_size
    if max_y < 0 or max_x < 0:
        return []

    total_positions = (max_y + 1) * (max_x + 1)
    n = min(n, total_positions)
    if n <= 0:
        return []

    candidate_n = min(total_positions, max(n * candidate_factor, n))
    y_cand = rng.integers(0, max_y + 1, size=candidate_n)
    x_cand = rng.integers(0, max_x + 1, size=candidate_n)
    candidates = np.stack([y_cand, x_cand], axis=1)

    # De-duplicate candidates.
    candidates = np.unique(candidates, axis=0)
    if candidates.shape[0] < n:
        # Fallback: add more random points if de-dup removed too many.
        need = n - candidates.shape[0]
        extra = np.stack(
            [
                rng.integers(0, max_y + 1, size=need),
                rng.integers(0, max_x + 1, size=need),
            ],
            axis=1,
        )
        candidates = np.unique(np.concatenate([candidates, extra], axis=0), axis=0)

    if candidates.shape[0] <= n:
        return [(int(y), int(x)) for y, x in candidates[:n]]

    if not bias_to_bright:
        idx = rng.choice(candidates.shape[0], size=n, replace=False)
        picked = candidates[idx]
        return [(int(y), int(x)) for y, x in picked]

    thr = float(np.percentile(image, bright_threshold_percentile))
    weights = np.zeros(candidates.shape[0], dtype=np.float64)
    for i, (y, x) in enumerate(candidates):
        patch = image[y : y + patch_size, x : x + patch_size]
        bright_frac = np.mean(patch > thr)
        weights[i] = 1.0 + bright_bias_strength * bright_frac

    weights_sum = weights.sum()
    if weights_sum <= 0:
        probs = None
    else:
        probs = weights / weights_sum

    idx = rng.choice(candidates.shape[0], size=n, replace=False, p=probs)
    picked = candidates[idx]
    return [(int(y), int(x)) for y, x in picked]


def build_patch_name(
    nd2_stem: str,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    patch_index: int,
    zero_pad: int,
    ext: str,
) -> str:
    return (
        f"{nd2_stem}__C0__y{y0}-y{y1}__x{x0}-x{x1}__p{patch_index:0{zero_pad}d}{ext}"
    )


def generate_patches(cfg: Config) -> Dict[str, Path]:
    raw_dir = Path(cfg.RAW_ND2_DIR)
    output_root = Path(cfg.OUTPUT_ROOT)
    if not raw_dir.exists():
        raise FileNotFoundError(f"RAW_ND2_DIR does not exist: {raw_dir}")

    dirs = ensure_dirs(output_root)
    nd2_files = sorted(raw_dir.glob("*.nd2"))
    if not nd2_files:
        raise FileNotFoundError(f"No .nd2 files found in: {raw_dir}")

    rng = np.random.default_rng(cfg.seed)
    rows: List[Dict] = []

    for nd2_path in tqdm(nd2_files, desc="Processing ND2"):
        try:
            image = load_nd2_channel0_2d(nd2_path, cfg)
        except Exception as e:
            raise RuntimeError(f"Failed to read channel 0 from {nd2_path.name}: {e}") from e

        h, w = image.shape
        if cfg.sampling_mode == "grid":
            coords = grid_coords(h, w, cfg.patch_size, cfg.stride)
        elif cfg.sampling_mode == "random":
            coords = random_coords(
                image=image,
                n=cfg.random_patches_per_image,
                patch_size=cfg.patch_size,
                rng=rng,
                bias_to_bright=cfg.random_bias_to_bright,
                bright_threshold_percentile=cfg.bright_threshold_percentile,
                bright_bias_strength=cfg.bright_bias_strength,
                candidate_factor=cfg.random_candidate_factor,
            )
        else:
            raise ValueError(f"Unsupported sampling_mode: {cfg.sampling_mode}. Use grid/random.")

        if not coords:
            print(
                f"[WARN] Skip {nd2_path.name}: image size {h}x{w} smaller than patch_size={cfg.patch_size} "
                "or no coords sampled."
            )
            continue

        nd2_stem = nd2_path.stem
        for p_idx, (y0, x0) in enumerate(coords):
            y1 = y0 + cfg.patch_size
            x1 = x0 + cfg.patch_size
            patch = image[y0:y1, x0:x1]
            if patch.shape != (cfg.patch_size, cfg.patch_size):
                continue

            stats = compute_stats(patch)
            patch_name = build_patch_name(
                nd2_stem=nd2_stem,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                patch_index=p_idx,
                zero_pad=cfg.patch_index_zero_pad,
                ext=cfg.image_ext,
            )

            patch_out = dirs["PATCHES_IMG_DIR"] / patch_name
            tifffile.imwrite(str(patch_out), patch.astype(np.uint16), photometric="minisblack")

            anno_img = dirs["ANNO_IMAGES"] / patch_name
            if cfg.copy_to_annotation_images:
                shutil.copy2(patch_out, anno_img)
            else:
                tifffile.imwrite(str(anno_img), patch.astype(np.uint16), photometric="minisblack")

            row = {
                "patch_filename": patch_name,
                "source_nd2": nd2_path.name,
                "source_nd2_path": str(nd2_path.resolve()),
                "channel": 0,
                "y0": y0,
                "y1": y1,
                "x0": x0,
                "x1": x1,
                "height": cfg.patch_size,
                "width": cfg.patch_size,
                "projection_mode": cfg.projection_mode,
                "sampling_mode": cfg.sampling_mode,
                "intensity_min": stats["min"],
                "intensity_max": stats["max"],
                "intensity_mean": stats["mean"],
                "intensity_p99": stats["p99"],
            }
            rows.append(row)

    if not rows:
        raise RuntimeError("No patches were generated. Please check configuration and source data.")

    meta_df = pd.DataFrame(rows)
    patch_meta_csv = dirs["PATCHES_META_DIR"] / "patches_metadata.csv"
    patch_meta_json = dirs["PATCHES_META_DIR"] / "patches_metadata.json"
    anno_index_csv = dirs["ANNO_IMAGES"] / "index.csv"

    meta_df.to_csv(patch_meta_csv, index=False)
    meta_df.to_json(patch_meta_json, orient="records", indent=2)
    meta_df[
        [
            "patch_filename",
            "source_nd2",
            "channel",
            "x0",
            "y0",
            "x1",
            "y1",
            "projection_mode",
            "intensity_min",
            "intensity_max",
            "intensity_mean",
            "intensity_p99",
        ]
    ].to_csv(anno_index_csv, index=False)

    print("\n=== Done ===")
    print(f"Generated patches: {len(rows)}")
    print(f"PATCHES_IMG_DIR: {dirs['PATCHES_IMG_DIR']}")
    print(f"PATCHES_META_DIR: {dirs['PATCHES_META_DIR']}")
    print(f"ANNOTATION_DIR : {dirs['ANNOTATION_DIR']}")
    print(f"Annotation index: {anno_index_csv}")
    return dirs


def _is_binary_mask(mask: np.ndarray) -> Tuple[bool, str]:
    uniq = np.unique(mask)
    uniq_set = set(int(v) for v in uniq.tolist())
    if uniq_set.issubset({0, 1}):
        return True, "0/1"
    if uniq_set.issubset({0, 255}):
        return True, "0/255"
    return False, f"values={sorted(list(uniq_set))[:10]}{'...' if len(uniq_set) > 10 else ''}"


def validate_annotation_folder(
    annotation_dir: str | Path,
    expected_shape: Tuple[int, int] = (512, 512),
    normalize_binary_masks: bool = False,
    overwrite: bool = True,
) -> Dict[str, int]:
    """
    Validate manual annotation workspace.

    Checks:
      - Every image in images/ has same-stem .npy in masks/
      - Mask shape equals expected_shape
      - Mask binary in {0,1} or {0,255}
      - Optional normalization to {0,1}
    """
    annotation_dir = Path(annotation_dir)
    images_dir = annotation_dir / "images"
    masks_dir = annotation_dir / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"images/ not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks/ not found: {masks_dir}")

    image_files: List[Path] = []
    for ext in (".tif", ".tiff", ".png"):
        image_files.extend(sorted(images_dir.glob(f"*{ext}")))

    summary = {
        "total_images": len(image_files),
        "missing_mask": 0,
        "shape_invalid": 0,
        "non_binary": 0,
        "valid": 0,
        "normalized": 0,
    }

    for img_path in image_files:
        mask_path = masks_dir / f"{img_path.stem}.npy"
        if not mask_path.exists():
            summary["missing_mask"] += 1
            continue

        try:
            mask = np.load(mask_path)
        except Exception:
            summary["non_binary"] += 1
            continue

        if mask.shape != expected_shape:
            summary["shape_invalid"] += 1
            continue

        is_binary, mode = _is_binary_mask(mask)
        if not is_binary:
            summary["non_binary"] += 1
            continue

        if normalize_binary_masks and mode == "0/255":
            norm = (mask > 0).astype(np.uint8)
            out_path = mask_path if overwrite else masks_dir / f"{mask_path.stem}_norm.npy"
            np.save(out_path, norm)
            summary["normalized"] += 1

        summary["valid"] += 1

    print("\n=== Annotation Validation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    return summary


def main() -> None:
    dirs = generate_patches(CONFIG)

    # Optional immediate check (typically before labeling most masks are missing; this is expected).
    validate_annotation_folder(
        annotation_dir=dirs["ANNOTATION_DIR"],
        expected_shape=(CONFIG.patch_size, CONFIG.patch_size),
        normalize_binary_masks=CONFIG.normalize_binary_masks,
        overwrite=CONFIG.normalize_overwrite,
    )


if __name__ == "__main__":
    main()
