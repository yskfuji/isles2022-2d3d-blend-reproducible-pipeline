# Model Card — ISLES 2022 2D+3D blend pipeline (v0.0.0.1)

This model card describes the **13 trained checkpoints** that compose the
heterogeneous ensemble used by this repository. Following the Model Cards
for Model Reporting framework (Mitchell et al., 2019).

## Model overview

| Bundle | Models | Architecture | Role |
|---|---|---|---|
| `convnext_8model` | 8 | 2.5D ConvNeXt-Tiny + UNet decoder (`ConvNeXtNnUNetSeg`) | ConvNeXt branch — 4 configs × 2 seeds, averaged |
| `nnunet_2d_3fold` | 3 | nnU-Net v2 2D fullres (`PlainConvUNet`) | 2D nnU-Net branch — folds 0, 1, 2 averaged |
| `nnunet_3d_2fold_mps` | 2 | nnU-Net v2 3D fullres with MPS-safe up-conv | 3D nnU-Net branch — folds 0, 1 averaged |

All 13 checkpoints feed into the cross-architecture probability fusion described
in [`scripts/cross_arch_ensemble_native.py`](core/pipeline/scripts/cross_arch_ensemble_native.py).

## Intended use

- **Primary**: scientific research and reproducibility benchmarking for ISLES 2022
  ischemic stroke lesion segmentation.
- **Portfolio / educational**: demonstrate heterogeneous-ensemble design and an
  MPS-compatible 3D nnU-Net workaround for Apple Silicon researchers.

## Out-of-scope use

- **NOT a medical device.** Not intended for clinical decision making, triage,
  patient management, or any direct patient-facing application.
- **NOT validated outside ISLES 2022 distribution.** Performance on other MRI
  protocols, scanners, patient populations, or lesion types has not been
  evaluated.
- Redistribution or commercial use of derivative weights is constrained by the
  ISLES 2022 Data Use Agreement of the original training data.

## Training data

- **Dataset**: ISLES 2022 (Ischemic Stroke Lesion Segmentation Challenge 2022)
- **Modalities**: DWI, ADC, FLAIR (3 channels)
- **Number of subjects in training pool**: 200 (per the original challenge split;
  this project uses a custom 5-fold CV on the first 200 and holds out 25 cases for
  test evaluation)
- **Split**: 25-case custom hold-out test split (see
  `core/pipeline/data/splits/` — split CSV not bundled; users prepare their own)

## Training procedure

### ConvNeXt 8-model branch

- Backbone: torchvision `convnext_tiny` (ImageNet-1K pretrained)
- Decoder: U-Net style decoder with 3 stages, `dec_ch=256`
- Input: 7 slice offsets × 3 modalities = 21 channels at 256×256
- Sampler: positive-slice oversampling (`pos_slice_weight=50`)
- Loss: TverskyOHEM-BCE (`α=0.3, β=0.7, ohem_min_neg=2048`)
- Optimizer: AdamW (lr=3e-4, wd=1e-4), cosine schedule + 5 ep warmup
- EMA enabled (decay=0.9998)
- Epochs: 150, with `max_train_batches=500`
- 4 configs × 2 seeds (42, 0) = 8 trained models

Config files: `core/pipeline/configs/train_convnext_v{2_1mm,v2_aug_1mm,v3_aug_1mm,v3_dilated_1mm}.yaml`.

### nnU-Net 2D 3-fold branch

- Trainer: stock `nnUNetTrainer` (1000 epochs, default settings)
- Folds: 0, 1, 2 (5-fold split per nnU-Net's deterministic `splits_final.json`)
- Architecture per `plans.json`: `PlainConvUNet`, batch_size=434, patch_size=[96, 80]
- Device: MPS (Apple Silicon)

### nnU-Net 3D 2-fold branch (MPS-safe)

- Trainer: custom `nnUNetTrainer_MPS3D_500epochs`
  (see [`core/pipeline/scripts/nnUNetTrainer_MPS3D_500epochs.py`](core/pipeline/scripts/nnUNetTrainer_MPS3D_500epochs.py))
- Key novelty: `ConvTranspose3d` is unimplemented on Apple MPS; the trainer
  monkey-patches `get_matching_convtransp` to use nearest-neighbor upsample
  (via `view`+`expand`+`reshape`, MPS-native) followed by 3×3×3 `Conv3d`.
- 500 epochs, batch_size forced to 1 (MPS memory budget at patch [80, 96, 80])
- Folds: 0, 1 (training a 3rd fold did not improve the cross-arch ensemble — see
  [docs/experiment_journey.md](docs/experiment_journey.md) phase 5–6)

## Evaluation

On the 25-case custom hold-out test split (DWI native space):

| Metric | Value |
|---|---:|
| Mean Dice (final, with adaptive threshold) | **0.7527** |
| Mean Dice (no adaptive threshold) | 0.7387 |
| Median Dice | 0.8304 |
| Gain from adaptive threshold | +0.0140 |
| Adaptive threshold fired on | 2 of 25 cases |

Per-case Dice and stage breakdown:
[`artifacts/eval_runs/cross_arch_v0.0.0.1/`](artifacts/eval_runs/cross_arch_v0.0.0.1/).

## Limitations & ethical considerations

- **Distribution shift**: trained only on ISLES 2022 DWI/ADC/FLAIR. May fail on
  other MRI protocols, vendors, or non-acute stroke imaging.
- **Boundary slices**: 3D folds systematically miss lesions touching z=0
  (see test case `sub-strokecase0007`). 2D folds partially recover.
- **Small lesion mis-localization**: a single test case (`sub-strokecase0174`,
  108-voxel lesion) gets Dice = 0 across every architecture. Indicates a
  small/atypical-lesion failure mode in the training distribution.
- **Multi-lesion under-prediction**: cases with many small disconnected lesions
  (e.g., `sub-strokecase0027` with 10 sub-lesions) are partially captured.
- **No demographic / fairness evaluation**: per-subgroup analysis (age, sex,
  lesion location) has not been performed.

## How to load (after downloading from the GitHub Release)

```python
# ConvNeXt 8-model (one example)
import torch
from src.models.convnext_nnunet_seg import ConvNeXtNnUNetSeg

model = ConvNeXtNnUNetSeg(
    in_channels=21, backbone="convnext_tiny", pretrained=False,
    dec_ch=256, deep_sup=False,
)
state = torch.load("convnext_8model/v3_dilated_1mm/best.pt", map_location="cpu")
if "model" in state and not any(k.startswith("encoder.") for k in state):
    state = state["model"]
model.load_state_dict(state)
model.eval()
```

For nnU-Net checkpoints, use `nnUNetv2_predict` after installing the MPS trainer
(see [`isles2022_2d3d_blend/README_en.md`](isles2022_2d3d_blend/README_en.md) §4.4).

## Weights distribution status

As of v0.0.0.1, the weights tarballs (~1.5 GB compressed) have been **prepared
and checksummed** but are **not attached to the GitHub Release** yet. See
[`docs/weights.md`](docs/weights.md) for SHA-256 hashes, file inventory, and
the planned distribution path.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0 for code and this model card. Weights, when distributed,
inherit ISLES 2022 Data Use Agreement terms with respect to commercial use
and redistribution.
