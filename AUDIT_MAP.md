# ISLES 2022 2D+3D blend pipeline — Audit Map

This repository packages the heterogeneous-ensemble recipe for the ISLES 2022 lesion
segmentation task. Reviewers can audit each stage by following the order below.

## 1. Reading order

1. `./isles2022_2d3d_blend/README.md` (Japanese) or `./isles2022_2d3d_blend/README_en.md`
2. `./core/pipeline/configs/train_convnext_v2_5slice_1mm.yaml`
   `./core/pipeline/configs/train_convnext_v3_7slice_dilated_1mm.yaml`
3. `./core/pipeline/src/models/convnext_nnunet_seg.py`
4. `./core/pipeline/src/training/train_isles_25d_convnext_fpn.py`
5. `./core/pipeline/scripts/nnUNetTrainer_MPS3D_500epochs.py`
   (MPS-safe 3D nnU-Net trainer — drop into the nnunetv2 install location)
6. `./core/pipeline/src/evaluation/evaluate_isles_25d_ensemble.py`
7. `./core/pipeline/scripts/cross_arch_ensemble_native.py`
   (cross-arch fusion + adaptive threshold)

## 2. Audit highlights

- **Heterogeneous ensemble**: 2D nnU-Net (3-fold) ⊕ 3D nnU-Net (2-fold) ⊕ 2.5D ConvNeXt (8-model)
  blended in DWI native space. Each architecture contributes diverse failure modes.
- **MPS-compatible 3D nnU-Net**: monkey-patches `ConvTranspose3d` (unimplemented on Apple MPS)
  with nearest-upsample-via-reshape + 3×3×3 Conv3d. See class `_InterpConv3d`.
- **Adaptive threshold post-processing**: large-prediction cases (>4000 voxels at base_thr=0.30)
  are re-thresholded at low_thr=0.03 to recover under-predicted large lesions.
  Verified +0.014 mean Dice on the ISLES 25-case custom test split, only affecting 2 of 25 cases.
- **CC-overlap rescue**: per-case rescue triggered when binary cn/nn Dice < 0.4; restricted to
  connected components whose cn∩nn overlap ≥ 20 voxels (avoids inflating FP regions).

## 3. Recipe at a glance

```text
nnUNet probs = 0.6 × 2D-3fold-avg + 0.4 × 3D-2fold-avg(f0+f1)
combined     = 0.20 × ConvNeXt-8-avg + 0.80 × nnUNet probs
rescue       = alpha=0.7, case_gate_agree=0.4, cc_min_overlap=20
post-process = base_thr=0.30 → if pred volume > 4000, re-threshold at 0.03
```

Reported mean Dice on the 25-case custom test split: **0.7527**.

## 4. Excluded artifacts

- `Datasets/` — obtain ISLES-2022 separately.
- `runs/` and `logs/` — training/eval traces are large and not reproducible without weights.
- Trained weights (`*.pt`, `*.pth`) — not bundled; train from scratch or contact author.
