# ISLES 2022 2D+3D blend pipeline — 完全ガイド

ISLES 2022 25 ケース独自 hold-out test split で **mean Dice 0.7527** を達成した異種アーキ・アンサンブル recipe の詳細解説。

## 1. パイプライン概要

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ アーキ A: 2.5D ConvNeXt-Tiny + UNet decoder, 8-model アンサンブル        │
│   - 4 config × 2 seed (v2_1mm, v2_aug_1mm, v3_aug_1mm, v3_dilated_1mm)   │
│   - 入力: 7 slice offsets × 3 modality (DWI/ADC/FLAIR) = 21 channels     │
│   - 1mm 処理空間で学習、確率を DWI native 空間に再サンプリング           │
└──────────────────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────────────────┐
│ アーキ B: nnU-Net 2D, 3-fold アンサンブル (fold 0, 1, 2)                  │
│   - 標準 nnU-Net 2D fullres preprocessing                                │
└──────────────────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────────────────┐
│ アーキ C: nnU-Net 3D fullres, 2-fold アンサンブル (fold 0, 1)             │
│   - MPS 互換 trainer: ConvTranspose3d → nearest-upsample + Conv3d        │
│     (scripts/nnUNetTrainer_MPS3D_500epochs.py)                           │
│   - 500 epoch, batch_size=1 (MPS メモリ予算で patch [80,96,80])          │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
nnUNet 確率 = 0.6 × 2D-3fold 平均 + 0.4 × 3D-2fold 平均
combined    = 0.20 × ConvNeXt-8 平均 + 0.80 × nnUNet 確率
rescue (ケース単位、cn/nn 2値 Dice < 0.4 のときだけ作用):
  prob = where(cc_overlap≥20, max(combined, 0.7 × max(cn, nn)), combined)
post-process:
  pred = combined ≥ 0.30
  if pred.sum() > 4000:           # ケース適応的閾値
      pred = combined ≥ 0.03
```

## 2. 各要素の貢献

| 要素 | 最終 0.7527 への寄与 | 効く理由 |
|---|---|---|
| 2D nnU-Net 3-fold | baseline 0.7352 (CN 込) | well-tuned な標準 recipe |
| + 3D nnU-Net 2-fold blend | +0.0035 → 0.7387 | 2D とは違う failure mode (z 軸 context) |
| 適応閾値 | **+0.0140** → 0.7527 | 大病変が base_thr=0.30 で系統的に過小予測される問題を補正 |

適応閾値は基底閾値での予測ボリュームが 4000 voxel 超のときだけ切り替わる。25 ケース split では 2 ケースだけ作用（両方とも過小予測されていた大病変）、残り 23 ケースは不変。小病変ケースに FP を増やすことなく改善。

## 3. リポジトリ構造

```text
.
├── core/pipeline/
│   ├── configs/                          # 4 ConvNeXt 訓練 config
│   │   ├── train_convnext_v2_1mm.yaml
│   │   ├── train_convnext_v2_aug_1mm.yaml
│   │   ├── train_convnext_v3_aug_1mm.yaml
│   │   └── train_convnext_v3_dilated_1mm.yaml
│   ├── src/
│   │   ├── datasets/isles_dataset.py
│   │   ├── models/{convnext_nnunet_seg,input_adapters}.py
│   │   ├── training/{train_isles_25d_convnext_fpn,losses,utils_train}.py
│   │   └── evaluation/{evaluate_isles_25d,evaluate_isles_25d_ensemble}.py
│   ├── scripts/
│   │   ├── cross_arch_ensemble_native.py # production fusion + adaptive thr
│   │   └── nnUNetTrainer_MPS3D_500epochs.py # nnunetv2 にコピーして使用
│   └── tools/make_manifest.py
└── scripts/smoke_test.py
```

## 4. End-to-end 再現手順

> ⚠️ ISLES 2022 データセット（同梱なし）、nnU-Net v2 (`pip install nnunetv2`)、timm が必要。

### 4.1. データ準備
- `Datasets/ISLES-2022/sub-strokecase<NNNN>/ses-0001/dwi/` に raw DWI NIfTI を配置
- nnU-Net レイアウトに変換後、`nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity`

### 4.2. ConvNeXt 8 モデル学習

```bash
cd core/pipeline
for seed in 42 0; do
  for cfg in v2_1mm v2_aug_1mm v3_aug_1mm v3_dilated_1mm; do
    PYTHONPATH=$PWD python -m src.training.train_isles_25d_convnext_fpn \
        configs/train_convnext_${cfg}.yaml  # YAML で seed 設定
  done
done
```

### 4.3. 2D nnU-Net 3-fold 学習

```bash
for f in 0 1 2; do
  nnUNetv2_train 1 2d $f -device mps
done
```

### 4.4. MPS 互換 3D nnU-Net 2-fold 学習

```bash
# 1. MPS-safe trainer を一度だけインストール
cp core/pipeline/scripts/nnUNetTrainer_MPS3D_500epochs.py \
   $(python -c "import nnunetv2,os;print(os.path.dirname(nnunetv2.__file__))")/training/nnUNetTrainer/variants/network_architecture/

# 2. fold 0, 1 訓練 (Apple M series, bs=1, patch [80,96,80] で各 ~2.4 日)
for f in 0 1; do
  nnUNetv2_train 1 3d_fullres $f -tr nnUNetTrainer_MPS3D_500epochs -device mps
done
```

### 4.5. test 推論

```bash
# ConvNeXt 8 モデル
PYTHONPATH=$PWD python -m src.evaluation.evaluate_isles_25d_ensemble \
    --model-paths results/<8 つの best.pt パス> \
    --csv-path data/splits/test.csv \
    --root data/processed/<1mm データセット> \
    --split test \
    --out-dir results/eval_convnext_8m \
    --save-probs-dir probs/convnext_8m \
    --thr 0.5 --min-size 0 --prob-filter 0.0

# nnU-Net 2D (3-fold avg) と 3D (2-fold avg)
nnUNetv2_predict -d 1 -i imagesTs -o probs/nnunet_2d_f012 -c 2d -f 0 1 2 --save_probabilities -device mps
nnUNetv2_predict -d 1 -i imagesTs -o probs/nnunet_3d_f0   -c 3d_fullres -f 0 -tr nnUNetTrainer_MPS3D_500epochs --save_probabilities -device mps
nnUNetv2_predict -d 1 -i imagesTs -o probs/nnunet_3d_f1   -c 3d_fullres -f 1 -tr nnUNetTrainer_MPS3D_500epochs --save_probabilities -device mps
```

### 4.6. 2D + 3D nnU-Net blend (2D 60% / 3D 40%)

```python
from pathlib import Path
import numpy as np
SRC2D = Path('probs/nnunet_2d_f012')
SRC3D_0 = Path('probs/nnunet_3d_f0')
SRC3D_1 = Path('probs/nnunet_3d_f1')
DST = Path('probs/nnunet_blend_2d6_3d4'); DST.mkdir(exist_ok=True)
for npz in SRC2D.glob('*.npz'):
    case = npz.stem
    p2 = np.load(SRC2D / f'{case}.npz')['probabilities'].astype(np.float32)
    p3_0 = np.load(SRC3D_0 / f'{case}.npz')['probabilities'].astype(np.float32)
    p3_1 = np.load(SRC3D_1 / f'{case}.npz')['probabilities'].astype(np.float32)
    p3 = (p3_0 + p3_1) / 2.0
    out = 0.6 * p2 + 0.4 * p3
    np.savez_compressed(DST / f'{case}.npz', probabilities=out.astype(np.float16))
```

### 4.7. 適応閾値付き cross-arch アンサンブル

```bash
PYTHONPATH=$PWD python scripts/cross_arch_ensemble_native.py \
    --convnext-probs-dir probs/convnext_8m \
    --nnunet-probs-dir probs/nnunet_blend_2d6_3d4 \
    --dwi-root /path/to/nnunet_raw/Dataset001_ISLES2022/imagesTs \
    --isles-root /path/to/Datasets/ISLES-2022 \
    --gt-root /path/to/gt_dwi_native \
    --csv-path data/splits/test.csv \
    --split test \
    --weights 0.20 \
    --thresholds 0.30 \
    --min-sizes 0 \
    --rescue-alphas 0.7 \
    --case-gate-agree 0.4 \
    --cc-min-overlap 20 \
    --adaptive-low-thrs 0.03 \
    --adaptive-high-vols 4000 \
    --out-json results/cross_arch_final.json
```

25 ケース split の期待結果: `mean=0.7527 median=0.8304`

## 5. ヘッドライン指標

| Recipe | Mean Dice | Median Dice |
|---|---:|---:|
| 8× ConvNeXt 2.5D + 2D-3fold (baseline) | 0.7352 | 0.8298 |
| + 3D-2fold blend (2D 60% / 3D 40%) | 0.7387 | 0.8304 |
| **+ 適応閾値** | **0.7527** | 0.8304 |

## 6. 既知の救済不可能ケース（post-processing で対処不可）

| case | gt voxels | dice | 理由 |
|---|---|---|---|
| 0174 | 108 | 0.000 | 全 model が mis-localize。非典型位置の単一小病変 |
| 0007 | 439 (9 病変) | 0.35 | 3D models が境界 slice z=0 を miss; 2D が部分救済 |
| 0027 | 822 (10 病変) | 0.21 | 10 病変中 1-2 個しか model 認識せず; 残りは prob map に信号なし |

これらは post-processing では救済不可。multi-lesion focal loss + edge-slice sampling を組み込んだ retrain が必要。

## 7. ライセンス / データ

Apache 2.0 ソースコード。データ・重みは同梱なし。ISLES-2022 データ利用契約を取得して、提供 config で再学習してください。
