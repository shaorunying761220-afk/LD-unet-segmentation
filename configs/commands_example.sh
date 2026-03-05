#!/usr/bin/env bash
set -euo pipefail

# ===== update these paths =====
RAW_ND2_DIR="/path/to/RAW_ND2_DIR"
OUTPUT_ROOT="/path/to/OUTPUT_ROOT"
MODEL_PATH="${OUTPUT_ROOT}/PATCHES_META_DIR/train_artifacts/best_unet.pt"

# 1) Otsu labels (optional subset)
python scripts/otsu_then_train.py \
  --output_root "${OUTPUT_ROOT}" \
  --stage otsu \
  --input_mode grayscale \
  --max_otsu_patches 120

# 2) Train
python -u scripts/otsu_then_train.py \
  --output_root "${OUTPUT_ROOT}" \
  --stage train \
  --epochs 40 \
  --lr 5e-4 \
  --batch_size 2 \
  --base_channels 16 \
  --skip_pred_preview

# 3) Infer on raw ND2
python scripts/infer_unet_on_nd2.py \
  --raw_nd2_dir "${RAW_ND2_DIR}" \
  --output_root "${OUTPUT_ROOT}" \
  --model_path "${MODEL_PATH}" \
  --tile_size 512 \
  --tile_stride 256 \
  --prob_threshold 0.5

# 4) Postprocess + stats
python scripts/postprocess_and_quantify_ld.py \
  --output_root "${OUTPUT_ROOT}" \
  --raw_nd2_dir "${RAW_ND2_DIR}" \
  --inference_dir "${OUTPUT_ROOT}/INFERENCE_UNET" \
  --fg_ratio_drop_threshold 0.05 \
  --max_object_area_px_drop 3000 \
  --min_object_area 20 \
  --min_peak_distance 7
