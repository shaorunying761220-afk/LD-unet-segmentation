#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process LD masks, optionally drop bad samples, split touching objects,
and quantify LD metrics by treatment parsed from file names.

Install:
  pip install nd2 numpy pandas tifffile scipy scikit-image matplotlib tqdm
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
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops_table
from skimage.morphology import area_opening
from skimage.segmentation import watershed
from tqdm import tqdm


@dataclass
class Cfg:
    output_root: Path
    raw_nd2_dir: Path
    inference_dir: Path
    projection_mode: str
    t_index: int
    channel_index: int
    fg_ratio_drop_threshold: float
    min_object_area: int
    max_object_area_px_drop: int
    min_peak_distance: int
    split_touching: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Postprocess + quantify LD")
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--raw_nd2_dir", type=str, required=True)
    p.add_argument("--inference_dir", type=str, default="")
    p.add_argument("--projection_mode", type=str, default="max", choices=["max", "mean", "z0"])
    p.add_argument("--t_index", type=int, default=0)
    p.add_argument("--channel_index", type=int, default=0)
    p.add_argument("--fg_ratio_drop_threshold", type=float, default=0.08)
    p.add_argument("--min_object_area", type=int, default=20)
    p.add_argument("--max_object_area_px_drop", type=int, default=0, help="If >0, drop whole image when any object area exceeds this")
    p.add_argument("--min_peak_distance", type=int, default=7)
    p.add_argument("--disable_split_touching", action="store_true")
    return p.parse_args()


def load_nd2_channel0_2d(nd2_path: Path, channel_index: int, t_index: int, projection_mode: str) -> np.ndarray:
    with nd2.ND2File(nd2_path) as f:
        arr = f.asarray()
        axes = list(f.sizes.keys())

    allowed = {"T", "C", "Z", "Y", "X"}
    for ax_name in list(axes):
        if ax_name not in allowed:
            ax = axes.index(ax_name)
            arr = np.take(arr, 0, axis=ax)
            axes.pop(ax)

    if "T" in axes:
        ax = axes.index("T")
        arr = np.take(arr, t_index, axis=ax)
        axes.pop(ax)
    if "C" in axes:
        ax = axes.index("C")
        arr = np.take(arr, channel_index, axis=ax)
        axes.pop(ax)
    if "Z" in axes:
        ax = axes.index("Z")
        if projection_mode == "max":
            arr = np.max(arr, axis=ax)
        elif projection_mode == "mean":
            arr = np.mean(arr, axis=ax)
        elif projection_mode == "z0":
            arr = np.take(arr, 0, axis=ax)
        axes.pop(ax)

    y_ax = axes.index("Y")
    x_ax = axes.index("X")
    arr = np.moveaxis(arr, [y_ax, x_ax], [0, 1])
    if arr.dtype != np.uint16:
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
    return arr


def parse_source_stem_from_mask_name(mask_name: str) -> str:
    # example: 72-KLH45_bodipyLD_1078__C0__pred.tif
    stem = Path(mask_name).stem
    if stem.endswith("__C0__pred"):
        return stem[:-10]
    return stem


def parse_treatment(source_stem: str) -> str:
    # examples:
    # WT-KLH45_bodipyLD_1032 -> WT-KLH45
    # 72-KLH45-100nMTg_bodipyLD_1079 -> 72-KLH45-100nMTg
    key = source_stem
    if "_bodipyLD_" in key:
        return key.split("_bodipyLD_")[0]
    parts = key.split("_")
    if len(parts) > 1:
        return "_".join(parts[:-1])
    return key


def split_touching_instances(binary: np.ndarray, min_peak_distance: int) -> np.ndarray:
    dist = ndi.distance_transform_edt(binary)
    coords = peak_local_max(
        dist,
        labels=binary,
        min_distance=max(1, int(min_peak_distance)),
        exclude_border=False,
    )
    if coords.shape[0] == 0:
        return ndi.label(binary)[0]

    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i
    labels = watershed(-dist, markers, mask=binary)
    return labels.astype(np.int32)


def save_group_boxplot(df: pd.DataFrame, metric: str, out_png: Path) -> None:
    groups = sorted(df["treatment"].dropna().unique().tolist())
    data = [df.loc[df["treatment"] == g, metric].dropna().values for g in groups]
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2), 4))
    ax.boxplot(data, tick_labels=groups, showfliers=True)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_group_bar_sem(df: pd.DataFrame, metric: str, out_png: Path) -> None:
    g = (
        df.groupby("treatment")[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("treatment")
    )
    g["sem"] = g["std"] / np.sqrt(np.maximum(g["count"], 1))
    fig, ax = plt.subplots(figsize=(max(6, len(g) * 1.2), 4))
    x = np.arange(len(g))
    ax.bar(x, g["mean"].values, yerr=g["sem"].values, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(g["treatment"].tolist(), rotation=30)
    ax.set_title(f"{metric} (mean ± SEM)")
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run(cfg: Cfg) -> None:
    inference_dir = cfg.inference_dir if cfg.inference_dir else (cfg.output_root / "INFERENCE_UNET")
    masks_dir = inference_dir / "masks"
    manifest_path = inference_dir / "inference_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    out_root = cfg.output_root / "POSTPROCESS_STATS"
    clean_mask_dir = out_root / "clean_masks"
    viz_dir = out_root / "plots"
    for d in [out_root, clean_mask_dir, viz_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    manifest["source_stem"] = manifest["source_nd2"].str.replace(".nd2", "", regex=False)
    manifest["treatment"] = manifest["source_stem"].apply(parse_treatment)

    nd2_map = {p.name: p for p in cfg.raw_nd2_dir.glob("*.nd2")}

    image_rows: List[Dict] = []
    drop_rows: List[Dict] = []
    droplet_rows: List[Dict] = []

    for row in tqdm(manifest.to_dict("records"), desc="Postprocess"):
        source_nd2 = row["source_nd2"]
        source_stem = row["source_stem"]
        treatment = row["treatment"]
        mask_path = Path(row["mask_path"])
        if not mask_path.exists():
            alt = masks_dir / f"{source_stem}__C0__pred.tif"
            if alt.exists():
                mask_path = alt
            else:
                continue

        pred = tifffile.imread(str(mask_path))
        binary = pred > 0
        fg_ratio = float(binary.mean())

        if fg_ratio > cfg.fg_ratio_drop_threshold:
            drop_rows.append(
                {
                    "source_nd2": source_nd2,
                    "treatment": treatment,
                    "fg_ratio": fg_ratio,
                    "reason": f"fg_ratio>{cfg.fg_ratio_drop_threshold}",
                }
            )
            continue

        # Remove tiny false-positive islands before instance split.
        binary = area_opening(binary.astype(np.uint8), area_threshold=cfg.min_object_area) > 0
        if cfg.split_touching:
            labels = split_touching_instances(binary=binary, min_peak_distance=cfg.min_peak_distance)
        else:
            labels = ndi.label(binary)[0]

        nd2_path = nd2_map.get(source_nd2)
        if nd2_path is None:
            continue
        img = load_nd2_channel0_2d(
            nd2_path=nd2_path,
            channel_index=cfg.channel_index,
            t_index=cfg.t_index,
            projection_mode=cfg.projection_mode,
        )
        if img.shape != labels.shape:
            h = min(img.shape[0], labels.shape[0])
            w = min(img.shape[1], labels.shape[1])
            img = img[:h, :w]
            labels = labels[:h, :w]

        props = regionprops_table(
            labels,
            intensity_image=img,
            properties=("label", "area", "mean_intensity", "max_intensity", "min_intensity"),
        )
        props_df = pd.DataFrame(props)

        if cfg.max_object_area_px_drop > 0 and len(props_df) > 0:
            if float(props_df["area"].max()) > float(cfg.max_object_area_px_drop):
                drop_rows.append(
                    {
                        "source_nd2": source_nd2,
                        "treatment": treatment,
                        "fg_ratio": fg_ratio,
                        "reason": f"max_object_area>{cfg.max_object_area_px_drop}",
                    }
                )
                continue

        clean_mask = (labels > 0).astype(np.uint8)
        tifffile.imwrite(
            str(clean_mask_dir / f"{source_stem}__C0__clean.tif"),
            clean_mask,
            photometric="minisblack",
        )

        if len(props_df) > 0:
            props_df["integrated_intensity"] = props_df["area"] * props_df["mean_intensity"]
            for _, r in props_df.iterrows():
                droplet_rows.append(
                    {
                        "source_nd2": source_nd2,
                        "source_stem": source_stem,
                        "treatment": treatment,
                        "label_id": int(r["label"]),
                        "area_px": float(r["area"]),
                        "mean_intensity": float(r["mean_intensity"]),
                        "max_intensity": float(r["max_intensity"]),
                        "min_intensity": float(r["min_intensity"]),
                        "integrated_intensity": float(r["integrated_intensity"]),
                    }
                )

        image_rows.append(
            {
                "source_nd2": source_nd2,
                "source_stem": source_stem,
                "treatment": treatment,
                "fg_ratio_pred": fg_ratio,
                "ld_count": int(len(props_df)),
                "ld_total_area_px": float(props_df["area"].sum()) if len(props_df) else 0.0,
                "ld_mean_area_px": float(props_df["area"].mean()) if len(props_df) else 0.0,
                "ld_median_area_px": float(props_df["area"].median()) if len(props_df) else 0.0,
                "ld_mean_intensity": float(props_df["mean_intensity"].mean()) if len(props_df) else 0.0,
                "ld_mean_integrated_intensity": float((props_df["area"] * props_df["mean_intensity"]).mean()) if len(props_df) else 0.0,
            }
        )

    image_df = pd.DataFrame(image_rows)
    droplet_df = pd.DataFrame(droplet_rows)
    dropped_df = pd.DataFrame(
        drop_rows,
        columns=["source_nd2", "treatment", "fg_ratio", "reason"],
    )

    image_csv = out_root / "per_image_stats.csv"
    droplet_csv = out_root / "per_droplet_stats.csv"
    dropped_csv = out_root / "dropped_images.csv"
    group_csv = out_root / "group_summary.csv"
    image_df.to_csv(image_csv, index=False)
    droplet_df.to_csv(droplet_csv, index=False)
    dropped_df.to_csv(dropped_csv, index=False)

    if len(image_df) > 0:
        group_df = (
            image_df.groupby("treatment", as_index=False)
            .agg(
                n_images=("source_nd2", "count"),
                mean_ld_count=("ld_count", "mean"),
                median_ld_count=("ld_count", "median"),
                mean_ld_area=("ld_mean_area_px", "mean"),
                mean_ld_intensity=("ld_mean_intensity", "mean"),
                mean_ld_integrated_intensity=("ld_mean_integrated_intensity", "mean"),
            )
        )
        group_df.to_csv(group_csv, index=False)

        for metric in ["ld_count", "ld_mean_area_px", "ld_mean_intensity", "ld_mean_integrated_intensity"]:
            if metric in image_df.columns:
                save_group_boxplot(image_df, metric=metric, out_png=viz_dir / f"{metric}_by_treatment_boxplot.png")
                save_group_bar_sem(image_df, metric=metric, out_png=viz_dir / f"{metric}_by_treatment_bar_sem.png")
    else:
        pd.DataFrame().to_csv(group_csv, index=False)

    print(f"[DONE] per-image stats : {image_csv}")
    print(f"[DONE] per-droplet stats: {droplet_csv}")
    print(f"[DONE] dropped samples : {dropped_csv} (n={len(dropped_df)})")
    print(f"[DONE] group summary   : {group_csv}")
    print(f"[DONE] plots dir       : {viz_dir}")


def main() -> None:
    a = parse_args()
    cfg = Cfg(
        output_root=Path(a.output_root),
        raw_nd2_dir=Path(a.raw_nd2_dir),
        inference_dir=Path(a.inference_dir) if a.inference_dir else Path(""),
        projection_mode=a.projection_mode,
        t_index=a.t_index,
        channel_index=a.channel_index,
        fg_ratio_drop_threshold=float(a.fg_ratio_drop_threshold),
        min_object_area=int(a.min_object_area),
        max_object_area_px_drop=int(a.max_object_area_px_drop),
        min_peak_distance=int(a.min_peak_distance),
        split_touching=not a.disable_split_touching,
    )
    run(cfg)


if __name__ == "__main__":
    main()
