# cross_arch_v0.0.0.1 evaluation evidence

This directory contains per-case evidence for the headline result of
**mean Dice 0.7527** reported in v0.0.0.1.

## Files

| File | Description |
|---|---|
| `per_case_dice.csv` | Per-case Dice for every test case (25 rows), broken down by stage |
| `metrics_summary.json` | Aggregate metrics + recipe + dragging cases |

## How to verify

The numbers in this directory are reproducible from the cached probability maps
of each architecture and the final recipe parameters. The pipeline (deterministic
post-processing on top of frozen prob maps) is implemented in
`core/pipeline/scripts/cross_arch_ensemble_native.py` and exercised via
`scripts/smoke_test.py`.

Recipe parameters (also encoded in `metrics_summary.json`):

```text
nnUNet probs = 0.6 × 2D-3fold-avg + 0.4 × 3D-2fold-avg(folds 0,1)
combined     = 0.20 × ConvNeXt-8avg + 0.80 × nnUNet probs
rescue       = alpha=0.7, case_gate_agree=0.4, cc_min_overlap=20
post-process = base_thr=0.30 → if pred volume > 4000, re-threshold at 0.03
```

## Column reference for `per_case_dice.csv`

| Column | Meaning |
|---|---|
| `case_id` | ISLES 2022 case identifier (sub-strokecase####) |
| `gt_voxels` | Ground-truth lesion volume in voxels |
| `pred_voxels_fixed_thr_0.30` | Final predicted volume at base threshold 0.30 |
| `pred_voxels_adaptive` | Final predicted volume after adaptive switch |
| `adaptive_fired` | True if predicted volume > 4000 → adaptive low-threshold applied |
| `dice_2d_3fold_avg_thr_0.50` | Dice of 2D 3-fold avg (thr=0.50) — diagnostic |
| `dice_3d_2fold_avg_thr_0.50` | Dice of 3D 2-fold avg (thr=0.50) — diagnostic |
| `dice_nn_blend_thr_0.50` | Dice of (60% × 2D + 40% × 3D) at thr=0.50 — diagnostic |
| `dice_convnext_8model_thr_0.50` | Dice of 8-model ConvNeXt avg (thr=0.50) — diagnostic |
| `dice_fixed_thr_0.30` | Dice of full pipeline at fixed thr=0.30 (no adaptive) |
| `dice_final_adaptive` | Dice of full pipeline including adaptive threshold (production) |

## Aggregate result (from `metrics_summary.json`)

| Metric | Value |
|---|---:|
| mean Dice (fixed thr=0.30) | 0.7387 |
| mean Dice (final, adaptive) | **0.7527** |
| gain from adaptive threshold | **+0.0140** |
| cases where adaptive fired | **2 / 25** |

The adaptive threshold only switches when the base-threshold prediction
exceeds 4000 voxels. On this 25-case split, it fires exactly twice — both
on large under-predicted lesions — and leaves the other 23 cases unchanged.
This rules out the alternative that the heuristic generally inflates
predicted volume; the +0.0140 gain is concentrated in two cases.

## Reproduction

To regenerate this directory after training your own ConvNeXt 8-model and
2D + 3D nnU-Net folds, save probabilities into the layout described in
`isles2022_2d3d_blend/README_en.md`, then run the cross-arch script as shown
there. The CSV/JSON in this directory will then match (within float16
quantization noise) for the same trained weights and the same 25-case split.
