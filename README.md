# LD Segmentation Pipeline (ND2 → Annotation → U-Net → Quantification)

This repository provides a pipeline for **lipid droplet (LD) segmentation** starting from Nikon ND2 microscopy data.

The workflow supports a full end-to-end process including:

1. Patch extraction from ND2 images (16-bit, channel C0)
2. Initial mask generation using Otsu thresholding with optional manual correction
3. Lightweight U-Net training (CPU compatible)
4. Whole-image inference on original ND2 data (overlapping tiles with weighted stitching to reduce seam artifacts)
5. Post-processing and quantitative analysis

---

# Repository Structure

```
Unet/
├─ scripts/
│  ├─ prepare_ld_annotation_patches.py
│  ├─ otsu_then_train.py
│  ├─ infer_unet_on_nd2.py
│  └─ postprocess_and_quantify_ld.py
│
├─ docs/
│  └─ PIPELINE_CN.md
│
├─ configs/
├─ requirements.txt
├─ .gitignore
└─ README.md
```

Script stages:

| Script | Function |
|------|------|
| prepare_ld_annotation_patches.py | ND2 → patch extraction |
| otsu_then_train.py | Otsu mask initialization + U-Net training |
| infer_unet_on_nd2.py | Whole image inference |
| postprocess_and_quantify_ld.py | Post-processing and quantitative analysis |

Note:  
Data directories such as `RAW_ND2_DIR`, `OUTPUT_ROOT`, and `OUTPUT_ROOT_2` are **excluded from Git tracking**.

---

# Environment

Python **3.10+** recommended.

Create environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch CPU version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

# Quick Start

## 1 Extract patches from ND2

```bash
python scripts/prepare_ld_annotation_patches.py
```

Modify the `CONFIG` section inside the script:

- `RAW_ND2_DIR`
- `OUTPUT_ROOT`
- `projection_mode`
- `patch_size`
- `stride`

---

## 2 Generate initial masks using Otsu thresholding

```bash
python scripts/otsu_then_train.py \
  --output_root /path/to/OUTPUT_ROOT \
  --stage otsu \
  --input_mode grayscale \
  --max_otsu_patches 120
```

Outputs

```
ANNOTATION_DIR/masks/*.tif
ANNOTATION_DIR/preview/*.png
PATCHES_META_DIR/otsu_manifest.csv
```

Manual correction of masks is recommended before training.

---

## 3 Train U-Net

```bash
python -u scripts/otsu_then_train.py \
  --output_root /path/to/OUTPUT_ROOT \
  --stage train \
  --epochs 40 \
  --lr 5e-4 \
  --batch_size 2 \
  --base_channels 16 \
  --skip_pred_preview
```

Training metrics printed per epoch:

- train loss
- validation loss
- validation Dice
- validation IoU

Model outputs

```
PATCHES_META_DIR/train_artifacts/best_unet.pt
PATCHES_META_DIR/train_artifacts/train_log.csv
PATCHES_META_DIR/train_artifacts/split.csv
```

---

## 4 Whole-image inference

```bash
python scripts/infer_unet_on_nd2.py \
  --raw_nd2_dir /path/to/RAW_ND2_DIR \
  --output_root /path/to/OUTPUT_ROOT \
  --model_path /path/to/best_unet.pt \
  --projection_mode max \
  --channel_index 0 \
  --tile_size 512 \
  --tile_stride 256 \
  --prob_threshold 0.5
```

Outputs

```
INFERENCE_UNET/masks/*.tif
INFERENCE_UNET/prob/*.tif
INFERENCE_UNET/preview/*.png
INFERENCE_UNET/inference_manifest.csv
```

The pipeline uses **overlapping tiles with Hann weighted blending** to reduce stitching artifacts.

---

## 5 Post-processing and quantification

```bash
python scripts/postprocess_and_quantify_ld.py \
  --output_root /path/to/OUTPUT_ROOT \
  --raw_nd2_dir /path/to/RAW_ND2_DIR \
  --inference_dir /path/to/OUTPUT_ROOT/INFERENCE_UNET \
  --fg_ratio_drop_threshold 0.05 \
  --max_object_area_px_drop 3000 \
  --min_object_area 20 \
  --min_peak_distance 7
```

Outputs

```
POSTPROCESS_STATS/per_image_stats.csv
POSTPROCESS_STATS/per_droplet_stats.csv
POSTPROCESS_STATS/dropped_images.csv
POSTPROCESS_STATS/group_summary.csv
POSTPROCESS_STATS/plots/*.png
```

