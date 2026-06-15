# isles2022-2d3d-blend-pipeline

**Language:** English | [Japanese](README_ja.md)

Heterogeneous-ensemble pipeline for ISLES 2022 ischemic stroke lesion segmentation. Combines 2D nnU-Net (3-fold) + MPS-compatible 3D nnU-Net (2-fold) + 2.5D ConvNeXt (8-model) via cross-architecture probability fusion in DWI-native space, with case-adaptive thresholding to recover under-predicted large lesions.

**Quick links**
- Entry guide (EN): [isles2022_2d3d_blend/README_en.md](isles2022_2d3d_blend/README_en.md)
- Entry guide (JA): [isles2022_2d3d_blend/README.md](isles2022_2d3d_blend/README.md)
- Audit map: [AUDIT_MAP.md](AUDIT_MAP.md)
- Roadmap: [ROADMAP.md](ROADMAP.md)
- Citation: [CITATION.cff](CITATION.cff)

## What this repository provides

- A reproducible heterogeneous-ensemble recipe combining 2D, 3D, and 2.5D segmentation architectures
- MPS-compatible nnU-Net 3D training (Apple Silicon workaround for `ConvTranspose3d`)
- Cross-architecture probability fusion with CC-overlap rescue
- **Case-adaptive thresholding**: switches to a low threshold when the base-threshold prediction is large, recovering under-segmented large lesions without harming small ones
- A no-data smoke test that verifies the public bundle in under a minute

## Headline result

On a 25-case custom held-out test split of ISLES 2022:

| Recipe | Mean Dice |
|---|---:|
| Baseline (8× ConvNeXt 2.5D + 2D nnU-Net 3-fold cross-arch) | 0.7352 |
| + 3D nnU-Net 2-fold blend (60% 2D / 40% 3D) | 0.7387 |
| **+ Adaptive threshold post-processing** | **0.7527** |

The adaptive threshold contributes +0.014 by triggering only on cases where the base-threshold prediction exceeds 4000 voxels (validated to affect 2 of 25 cases — both large under-predicted lesions — while leaving the other 23 unchanged).

## Final recipe

```
nnUNet probs = 0.6 × 2D-3fold-avg + 0.4 × 3D-2fold-avg(folds 0,1)
combined     = 0.20 × ConvNeXt-8-avg + 0.80 × nnUNet probs
rescue       = alpha=0.7, case_gate_agree=0.4, cc_min_overlap=20
post-process = base_thr=0.30 → if pred volume > 4000, re-threshold at 0.03
```

## MPS-compatible 3D nnU-Net

`ConvTranspose3d` is not implemented on Apple MPS. The custom trainer
`nnUNetTrainer_MPS3D_500epochs` monkey-patches `get_matching_convtransp` so that
3D up-convolutions use a nearest-neighbor upsample (via `view`+`expand`+`reshape`,
all native MPS ops) followed by a 3×3×3 `Conv3d`. The reshape-based upsample is
mathematically identical to nearest-neighbor for integer scale factors and runs
~2× faster than `F.interpolate(mode='nearest')` (which silently falls back to CPU).

See [`core/pipeline/scripts/nnUNetTrainer_MPS3D_500epochs.py`](core/pipeline/scripts/nnUNetTrainer_MPS3D_500epochs.py).

## Who this is for

- Hiring managers reviewing medical-AI segmentation portfolio work
- ML engineers needing an auditable MRI-segmentation ensemble baseline
- Apple Silicon researchers blocked by missing `ConvTranspose3d` in MPS

## Quickstart

### 1. Verify the repository without medical data

```bash
python scripts/smoke_test.py --use_dummy_data
```

### 2. Inspect the public bundle manifest

```bash
cd core/pipeline
python tools/make_manifest.py
```

### 3. Run training and full evaluation with your own data

- Full guide (EN): [isles2022_2d3d_blend/README_en.md](isles2022_2d3d_blend/README_en.md)
- Full guide (JA): [isles2022_2d3d_blend/README.md](isles2022_2d3d_blend/README.md)

## What is included vs excluded

Included:
- Source code (models, training, evaluation, cross-arch fusion)
- nnU-Net trainer variant for MPS
- Configurations
- Audit and reproducibility documentation

Not included:
- `Datasets/`
- Trained weights (`*.pt`, `*.pth`)
- `runs/`, `results/`, `logs/`

## How to cite

See [CITATION.cff](CITATION.cff).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
