# Lipid Droplet Segmentation Pipeline – Internal Documentation

This document is intended for **internal lab members** and describes the data flow, key parameters, quality control strategies, and recommended visualization methods used in this project.

---

# 1. Overall Strategy

**Goal:**  
Robustly extract lipid droplet (LD) information from fluorescence microscopy images with noise and batch variability, enabling reliable comparisons between experimental groups.

Pipeline overview:

1. ND2 → patch extraction (C0 channel only, keep 16-bit depth)
2. Otsu initialization (rapid generation of candidate masks)
3. Manual correction (improve annotation quality)
4. U-Net training
5. Whole-image inference (overlapping tiles to reduce stitching artifacts)
6. Post-processing (remove abnormal images and split merged droplets)
7. Quantification (count, area, fluorescence intensity)

---

# 2. Script Responsibilities

## 1) `scripts/prepare_ld_annotation_patches.py`

Purpose:

- Read ND2 files (supports T/Z dimensions)
- Use **channel C0 only**
- Perform Z projection (`max / mean / z0`)
- Extract patches (default size: 512)
- Export patches and metadata

Key outputs:

```
PATCHES_IMG_DIR/
PATCHES_META_DIR/patches_metadata.csv
ANNOTATION_DIR/images
ANNOTATION_DIR/masks
```

---

## 2) `scripts/otsu_then_train.py`

### Stage A (`--stage otsu`)

Function:

- Perform Otsu binarization on image patches
- Export masks as `.tif` (values: 0 / 1)
- Generate preview figures
  - image
  - mask
  - overlay
- Create a manifest file for annotation review

Useful option for reducing manual work:

```
--max_otsu_patches N
```

Only sample **N patches** for manual annotation.

---

### Stage B (`--stage train`)

Function:

- Train U-Net using:
  - `PATCHES_IMG_DIR`
  - `ANNOTATION_DIR/masks`
- Image–mask pairs are matched by filename stem

Training behavior:

- Default: **partial annotation allowed**
- Optional strict mode:

```
--strict_mask_match
```

Only patches with masks are used.

---

## 3) `scripts/infer_unet_on_nd2.py`

Purpose:

Apply the trained U-Net model to **entire ND2 images**.

Method:

- Tile-based inference
- Overlapping tiles
- Weighted stitching

To avoid the **cross-shaped seams** seen previously:

Recommended parameter:

```
--tile_stride 256
```

A **Hann window weighted fusion** is used instead of simple averaging.

Outputs:

```
binary segmentation mask
foreground probability map (16-bit)
overlay preview images
inference_manifest.csv
```

---

## 4) `scripts/postprocess_and_quantify_ld.py`

Purpose:

Perform **post-processing and biological quantification** on segmentation results.

Main strategies:

### 1. Image-level filtering

Remove abnormal images based on:

```
fg_ratio_drop_threshold
```

If foreground area ratio is too large → discard image.

```
max_object_area_px_drop
```

If an extremely large connected component appears → discard image.

---

### 2. Droplet splitting

Method:

- distance transform
- local maxima detection
- watershed segmentation

Used to separate **merged droplets**.

---

### 3. Quantification outputs

Image-level statistics:

```
per_image_stats.csv
```

Droplet-level statistics:

```
per_droplet_stats.csv
```

Group-level summary:

```
group_summary.csv
```

---

### 4. Visualization

Automatically generated plots:

- **Boxplots** (distribution visualization)
- **Mean ± SEM bar plots** (group comparison)

---

# 3. Naming Convention and Group Parsing

Experimental groups are inferred from filenames.

Examples:

```
WT-KLH45_bodipyLD_1034.nd2
→ group: WT-KLH45
```

```
72-KLH45-100nMTg_bodipyLD_1081.nd2
→ group: 72-KLH45-100nMTg
```

If naming conventions change, the **group parsing function must be updated accordingly**.

---

# 4. Recommended Parameters (Current Dataset)

## Training

```
epochs: 30–60
lr: 5e-4
batch_size: 2–4 (CPU)
base_channels: 16
```

Using **16 base channels** is more stable than 8.

---

## Inference

```
tile_size: 512
tile_stride: 256 (recommended)
prob_threshold: 0.5
```

The probability threshold can be adjusted depending on the balance between:

- false positives
- false negatives

---

## Post-processing

```
fg_ratio_drop_threshold: 0.03–0.08
max_object_area_px_drop: 2000–5000
min_object_area: 20
min_peak_distance: 5–9
```

It is recommended to inspect abnormal samples before finalizing thresholds.

---

# 5. Quality Control Guidelines

1. After each training run, inspect `train_log.csv` for signs of overfitting.
2. After inference, review abnormal samples in:

```
INFERENCE_UNET/preview
```

Look for:

- extremely large foreground regions
- images where the entire frame is predicted as signal.

3. After post-processing, inspect:

```
dropped_images.csv
```

Ensure that filtering decisions are biologically reasonable.

4. Before statistical analysis, confirm group balance using:

```
group_summary.csv → n_images
```

---

# 6. Recommended Figures for Lab Meeting

For each metric, two visualization types are recommended.

### 1. Boxplots

Show:

- distribution
- variability
- outliers

### 2. Mean ± SEM bar plots

Show:

- group-level trends
- easier biological interpretation

---

## Key Metrics Generated

```
ld_count
ld_mean_area_px
ld_mean_intensity
ld_mean_integrated_intensity
```

---

# 7. Reproducibility Commands

## A. Training (using manually corrected annotations)

```bash
python -u scripts/otsu_then_train.py \
  --output_root /path/to/OUTPUT_ROOT_2 \
  --stage train \
  --epochs 40 \
  --lr 5e-4 \
  --batch_size 2 \
  --base_channels 16 \
  --skip_pred_preview
```

---

## B. Whole-image inference

```bash
python scripts/infer_unet_on_nd2.py \
  --raw_nd2_dir /path/to/RAW_ND2_DIR \
  --output_root /path/to/OUTPUT_ROOT_2 \
  --model_path /path/to/OUTPUT_ROOT_2/PATCHES_META_DIR/train_artifacts/best_unet.pt \
  --tile_size 512 \
  --tile_stride 256 \
  --prob_threshold 0.5
```

---

## C. Post-processing and quantification

```bash
python scripts/postprocess_and_quantify_ld.py \
  --output_root /path/to/OUTPUT_ROOT_2 \
  --raw_nd2_dir /path/to/RAW_ND2_DIR \
  --inference_dir /path/to/OUTPUT_ROOT_2/INFERENCE_UNET \
  --fg_ratio_drop_threshold 0.05 \
  --max_object_area_px_drop 3000 \
  --min_object_area 20 \
  --min_peak_distance 7
```
