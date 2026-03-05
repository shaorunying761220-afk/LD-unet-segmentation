# LD Segmentation Pipeline (ND2 -> Annotation -> U-Net -> Quantification)

本仓库用于脂滴（LD）分割任务，支持从 Nikon ND2 原始数据出发，完成以下全流程：

1. ND2 切 patch（16-bit，C0）
2. Otsu 初始掩膜 + 人工校正
3. 轻量 U-Net 训练（CPU）
4. 原图全量推理（重叠+加权拼接，减少缝合伪影）
5. 后处理（二次切分/异常剔除）与分组统计可视化

---

## 1. Repository Structure

```text
Unet/
├─ scripts/
│  ├─ prepare_ld_annotation_patches.py   # Stage 0: ND2 -> patch
│  ├─ otsu_then_train.py                  # Stage A/B: Otsu + train
│  ├─ infer_unet_on_nd2.py                # Stage C: whole-image inference
│  └─ postprocess_and_quantify_ld.py      # Stage D: postprocess + quantify
├─ docs/
│  └─ PIPELINE_CN.md                      # 详细中文说明（推荐先读）
├─ configs/
├─ requirements.txt
├─ .gitignore
└─ README.md
```

> 数据目录（`RAW_ND2_DIR/`, `OUTPUT_ROOT/`, `OUTPUT_ROOT_2/`）默认不纳入 Git 追踪。

---

## 2. Environment

建议 Python 3.10+。CPU 环境示例：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch CPU 也可单独安装：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. Quick Start

### 3.1 ND2 切 patch（16-bit）

```bash
python scripts/prepare_ld_annotation_patches.py
```

在脚本 `CONFIG` 区修改：
- `RAW_ND2_DIR`
- `OUTPUT_ROOT`
- `projection_mode`, `patch_size`, `stride` 等

---

### 3.2 Otsu 自动初标 + 人工检查

```bash
python scripts/otsu_then_train.py \
  --output_root /path/to/OUTPUT_ROOT \
  --stage otsu \
  --input_mode grayscale \
  --max_otsu_patches 120
```

输出：
- `ANNOTATION_DIR/masks/*.tif`（0/1）
- `ANNOTATION_DIR/preview/*.png`
- `PATCHES_META_DIR/otsu_manifest.csv`

---

### 3.3 训练 U-Net（CPU）

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

训练日志实时打印每个 epoch 的：
- train loss
- val loss
- val Dice
- val IoU

模型输出：
- `PATCHES_META_DIR/train_artifacts/best_unet.pt`
- `PATCHES_META_DIR/train_artifacts/train_log.csv`
- `PATCHES_META_DIR/train_artifacts/split.csv`

---

### 3.4 原始 ND2 全量推理

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

输出：
- `INFERENCE_UNET/masks/*.tif`
- `INFERENCE_UNET/prob/*.tif`
- `INFERENCE_UNET/preview/*.png`
- `INFERENCE_UNET/inference_manifest.csv`

> 采用重叠 tile + Hann 加权融合，减少拼接十字缝伪影。

---

### 3.5 后处理 + 生物统计

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

输出：
- `POSTPROCESS_STATS/per_image_stats.csv`
- `POSTPROCESS_STATS/per_droplet_stats.csv`
- `POSTPROCESS_STATS/dropped_images.csv`
- `POSTPROCESS_STATS/group_summary.csv`
- `POSTPROCESS_STATS/plots/*.png`

---

## 4. Data Policy for GitHub

建议不要上传：
- 原始 ND2
- 全量 patch
- 全量 mask/preview
- 模型大文件（可用 release 或网盘）

建议上传：
- `scripts/`
- `docs/`
- `requirements.txt`
- 少量匿名示例图（可选）
- 统计结果示例 CSV（可选，注意隐私）

---

## 5. Citation / Internal Notes

如果仓库用于组内共享，建议在 `docs/PIPELINE_CN.md` 追加：
- 样本来源和批次说明
- 显微参数（曝光、放大倍数）
- 统计检验方法（如后续用 R/GraphPad）

