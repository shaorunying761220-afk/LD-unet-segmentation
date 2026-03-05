# 脂滴分割项目详细说明（组内共享版）

本文档面向实验室内部成员，说明本项目的数据流、关键参数、质量控制策略与推荐可视化方法。

## 一、总体思路

目标：在含噪声、批次差异明显的荧光图像中稳定提取脂滴信息，并用于组间比较。

流程：
1. ND2 -> patch（只用 C0，保留 16-bit）
2. Otsu 初标（快速产生候选 mask）
3. 人工校正（提高标签质量）
4. U-Net 训练
5. 全图推理（重叠拼接，降低缝合伪影）
6. 后处理（剔除异常图、切分粘连体）
7. 指标统计（数量、面积、荧光）

---

## 二、脚本职责

### 1) `scripts/prepare_ld_annotation_patches.py`

作用：
- 读取 ND2（支持 T/Z）
- 仅取 C0
- 对 Z 做 `max/mean/z0` 投影
- 按 patch 切块（默认 512）
- 输出 patch 与 metadata

关键输出：
- `PATCHES_IMG_DIR/`
- `PATCHES_META_DIR/patches_metadata.csv`
- `ANNOTATION_DIR/images`, `ANNOTATION_DIR/masks`

---

### 2) `scripts/otsu_then_train.py`

#### Stage A (`--stage otsu`)
- 对 patch 做 Otsu 二值化
- mask 输出为 `.tif`（0/1）
- 生成 preview（image / mask / overlay）
- 生成 manifest 便于审阅

可用来减轻人工负担：
- `--max_otsu_patches N`：仅抽样 N 张做标注

#### Stage B (`--stage train`)
- 用 `PATCHES_IMG_DIR` + `ANNOTATION_DIR/masks` 按 stem 匹配训练
- 默认允许“部分有标注即可训练”
- `--strict_mask_match` 可切换为严格模式

---

### 3) `scripts/infer_unet_on_nd2.py`

作用：
- 将训练好的模型应用到整张 ND2 投影图
- tile 推理并拼接

为避免你之前看到的“十字缝”：
- 推荐 `--tile_stride 256`（重叠）
- 使用 Hann 加权融合，不是简单平均

输出：
- 二值 mask
- 前景概率图（16-bit）
- overlay 预览
- 推理 manifest

---

### 4) `scripts/postprocess_and_quantify_ld.py`

作用：
- 对推理 mask 进行后处理与统计

包含策略：
1. **整图剔除策略**
   - `fg_ratio_drop_threshold`：前景面积比过大则剔除
   - `max_object_area_px_drop`：若出现超大连通域，整图剔除
2. **二次切分策略**
   - 距离变换 + 局部峰值 + watershed
   - 拆分粘连脂滴
3. **统计输出**
   - 单图层面：`per_image_stats.csv`
   - 单脂滴层面：`per_droplet_stats.csv`
   - 按处理组汇总：`group_summary.csv`
4. **可视化**
   - boxplot（分布）
   - mean±SEM 柱状图（组间比较）

---

## 三、命名与处理组解析

脚本默认通过文件名解析处理组：
- `WT-KLH45_bodipyLD_1034.nd2` -> `WT-KLH45`
- `72-KLH45-100nMTg_bodipyLD_1081.nd2` -> `72-KLH45-100nMTg`

如后续命名规则变化，需要同步修改解析函数。

---

## 四、参数建议（当前数据）

### 1) 训练
- `epochs`: 30~60
- `lr`: 5e-4（小样本较稳）
- `batch_size`: 2~4（CPU）
- `base_channels`: 16（较 8 更稳）

### 2) 推理
- `tile_size`: 512
- `tile_stride`: 256（建议）
- `prob_threshold`: 0.5（可按假阳性/假阴性权衡）

### 3) 后处理
- `fg_ratio_drop_threshold`: 0.03~0.08
- `max_object_area_px_drop`: 2000~5000（建议先看异常样本再定）
- `min_object_area`: 20
- `min_peak_distance`: 5~9

---

## 五、质量控制建议

1. 每次训练后先看 `train_log.csv` 是否过拟合。  
2. 推理后先看 `INFERENCE_UNET/preview` 的异常样本（超大块、整图发亮）。  
3. 后处理后优先检查 `dropped_images.csv`，确认剔除逻辑符合生物学预期。  
4. 统计前，先确认各组样本数是否平衡（`group_summary.csv` 的 `n_images`）。

---

## 六、组会展示推荐图

建议每个指标做两类图：
1. **箱线图**：展示组内离散程度与离群点
2. **均值±SEM 柱状图**：展示组间中心趋势

关键指标（已生成）：
- `ld_count`
- `ld_mean_area_px`
- `ld_mean_intensity`
- `ld_mean_integrated_intensity`

---

## 七、复现实验建议命令

### A. 训练（基于人工校正集）
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

### B. 全图推理
```bash
python scripts/infer_unet_on_nd2.py \
  --raw_nd2_dir /path/to/RAW_ND2_DIR \
  --output_root /path/to/OUTPUT_ROOT_2 \
  --model_path /path/to/OUTPUT_ROOT_2/PATCHES_META_DIR/train_artifacts/best_unet.pt \
  --tile_size 512 --tile_stride 256 --prob_threshold 0.5
```

### C. 后处理+统计
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

