"""Ensemble evaluation: average probability maps from multiple 2.5D models.

Usage:
  cd ToReBrain-pipeline
  PYTHONPATH=$PWD python -m src.evaluation.evaluate_isles_25d_ensemble \\
    --model-paths results/.../v2_1mm/best.pt results/.../v3_dilated_1mm/best.pt \\
    --csv-path data/splits/my_dataset_dwi_adc_flair_train_val_test.csv \\
    --root data/processed/my_dataset_dwi_adc_flair_1mm \\
    --split val \\
    --out-dir results/eval_ensemble_v2_v3_val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from scipy.ndimage import label as cc_label

from ..datasets.isles_dataset import IslesVolumeDataset
from ..models.convnext_nnunet_seg import ConvNeXtNnUNetSeg
from ..training.utils_train import prepare_device
from .evaluate_isles_25d import (
    _center_pad_crop_np,
    _restore_pad_crop_np,
    postprocess,
    _dice,
    _lesionwise_f1,
    infer_volume,
)


def _load_model(model_path: Path, device: torch.device) -> tuple[torch.nn.Module, list[int], tuple[int, int], bool]:
    """Load a ConvNeXtNnUNetSeg model from a checkpoint, auto-detecting config.

    Returns (model, offsets, img_size, hint_attn).
    """
    config_path = model_path.parent / "config.yaml"
    cfg_data: dict = {}
    cfg_train: dict = {}
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text())
        cfg_data  = cfg.get("data",  {})
        cfg_train = cfg.get("train", {})

    k = int(cfg_data.get("k_slices", 2))
    _sof = cfg_data.get("slice_offsets")
    offsets: list[int] = [int(x) for x in _sof] if _sof is not None else list(range(-k, k + 1))

    _isz = cfg_data.get("img_size")
    img_size: tuple[int, int] = tuple(int(x) for x in _isz) if _isz else (256, 256)  # type: ignore[assignment]

    n_modalities = 3
    hint_attn    = bool(cfg_train.get("hint_attn", False))
    in_channels  = n_modalities * len(offsets) + (1 if hint_attn else 0)
    backbone     = str(cfg_train.get("backbone", "convnext_tiny"))
    dec_ch       = int(cfg_train.get("dec_ch", 256))
    deep_sup     = bool(cfg_train.get("deep_sup", False))

    model = ConvNeXtNnUNetSeg(
        in_channels=in_channels,
        backbone=backbone,
        pretrained=False,
        first_conv_init=str(cfg_train.get("first_conv_init", "repeat")),
        dec_ch=dec_ch,
        out_channels=1,
        deep_sup=deep_sup,
        hint_attn=hint_attn,
    )
    state = torch.load(str(model_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state and not any(
        k.startswith("encoder.") for k in state
    ):
        state = state["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  Loaded: {model_path.name}  offsets={offsets}  in_ch={in_channels}  hint_attn={hint_attn}")
    return model, offsets, img_size, hint_attn


def main() -> None:
    p = argparse.ArgumentParser(description="Ensemble 2.5D ConvNeXt models on ISLES.")
    p.add_argument("--model-paths", nargs="+", required=True, help="Paths to best.pt checkpoints.")
    p.add_argument("--csv-path",    required=True)
    p.add_argument("--root",        required=True)
    p.add_argument("--split",       default="val")
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--normalize",   default="fixed_nonzero_zscore")
    p.add_argument("--thr",         type=float, default=0.85)
    p.add_argument("--min-size",    type=int,   default=64)
    p.add_argument("--prob-filter", type=float, default=0.96)
    p.add_argument("--closing-mm",  type=int,   default=0, help="Morphological closing radius in mm (0=off).")
    p.add_argument("--tta", action="store_true", default=False,
                   help="Test-Time Augmentation: average over original + LR/UD/LR+UD flips.")
    p.add_argument("--stage1-probs-dirs", nargs="*", default=None,
                   help="Per-model Stage1 prob dirs (one per --model-paths). Use 'none' for models without hint.")
    p.add_argument("--save-probs-dir", default=None,
                   help="If set, save averaged probability maps as {case_id}.npz (key='probs', float16) to this dir.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_probs_dir = None
    if args.save_probs_dir:
        save_probs_dir = Path(args.save_probs_dir)
        save_probs_dir.mkdir(parents=True, exist_ok=True)
    device = prepare_device()

    # Resolve per-model stage1 probs dirs
    s1_dirs: list[Path | None] = []
    if args.stage1_probs_dirs is not None:
        for d in args.stage1_probs_dirs:
            if d is None or str(d).strip().lower() == "none":
                s1_dirs.append(None)
            else:
                p_ = Path(str(d).strip())
                s1_dirs.append(p_ if p_.exists() else None)
    else:
        s1_dirs = [None] * len(args.model_paths)
    # Pad to match number of models
    while len(s1_dirs) < len(args.model_paths):
        s1_dirs.append(None)

    # Load all models
    print(f"Loading {len(args.model_paths)} models:")
    models_info: list[tuple[torch.nn.Module, list[int], tuple[int, int], bool]] = []
    for mp in args.model_paths:
        model, offsets, img_size, hint_attn = _load_model(Path(mp), device)
        models_info.append((model, offsets, img_size, hint_attn))

    # Dataset
    ds = IslesVolumeDataset(
        csv_path=args.csv_path,
        split=args.split,
        root=args.root,
        normalize=args.normalize,
    )
    print(f"Split='{args.split}': {len(ds)} cases")
    print(f"Post-proc: thr={args.thr}, min_size={args.min_size}, prob_filter={args.prob_filter}\n")

    records: list[dict] = []

    for idx in range(len(ds)):
        sample  = ds[idx]
        vol     = sample["image"].astype(np.float32)   # (C, Z, H, W)
        mask_gt = (sample["mask"] > 0.5).astype(np.uint8)
        case_id = str(sample["case_id"])

        # Average probability maps from all models
        probs = []
        for (model, offsets, img_size, hint_attn), s1_dir in zip(models_info, s1_dirs):
            extra_vol = None
            if hint_attn and s1_dir is not None:
                npz_path = s1_dir / f"{case_id}.npz"
                if npz_path.exists():
                    s1 = np.load(str(npz_path))["probs"].astype(np.float32)
                else:
                    s1 = np.zeros((vol.shape[1], vol.shape[2], vol.shape[3]), dtype=np.float32)
                extra_vol = s1  # (Z, H, W)
            elif hint_attn:
                extra_vol = np.zeros((vol.shape[1], vol.shape[2], vol.shape[3]), dtype=np.float32)  # (Z, H, W)
            probs.append(infer_volume(vol, model, offsets=offsets, img_size=img_size, device=device,
                                      extra_vol=extra_vol, tta=args.tta))
        prob = np.mean(probs, axis=0)

        if save_probs_dir is not None:
            np.savez_compressed(str(save_probs_dir / f"{case_id}.npz"),
                                probs=prob.astype(np.float16))

        pred = postprocess(prob, thr=args.thr, min_size=args.min_size, prob_filter=args.prob_filter, closing_mm=args.closing_mm)

        gt_vox   = int(mask_gt.sum())
        pred_vox = int(pred.sum())
        tp_vox   = int((pred * mask_gt).sum())
        fp_vox   = pred_vox - tp_vox
        fn_vox   = gt_vox  - tp_vox
        dice     = _dice(pred, mask_gt)
        lw       = _lesionwise_f1(pred, mask_gt)

        rec = {
            "case_id":   case_id,
            "dice":      float(dice),
            "gt_vox":    gt_vox,
            "pred_vox":  pred_vox,
            "tp_vox":    tp_vox,
            "fp_vox":    fp_vox,
            "fn_vox":    fn_vox,
            "gt_pos":    bool(gt_vox > 0),
            **lw,
        }
        records.append(rec)
        print(f"[{idx+1:2d}/{len(ds)}] {case_id}: gt={gt_vox:7d}  pred={pred_vox:7d}  "
              f"dice={dice:.4f}  lesion_f1={lw['lesion_f1']:.3f}", flush=True)

    # Aggregate
    dices       = [r["dice"] for r in records]
    gt_pos_recs = [r for r in records if r["gt_pos"]]
    gt_neg_recs = [r for r in records if not r["gt_pos"]]

    det_rate = float(np.mean([r["tp_vox"] > 0 for r in gt_pos_recs])) if gt_pos_recs else None
    fp_rate  = float(np.mean([r["pred_vox"] > 0 for r in gt_neg_recs])) if gt_neg_recs else None

    total_tp = sum(r["tp_vox"] for r in records)
    total_fp = sum(r["fp_vox"] for r in records)
    total_fn = sum(r["fn_vox"] for r in records)
    prec_g = total_tp / (total_tp + total_fp + 1e-6)
    rec_g  = total_tp / (total_tp + total_fn + 1e-6)

    summary = {
        "model_paths":     args.model_paths,
        "n_models":        len(args.model_paths),
        "split":           args.split,
        "thr":             args.thr,
        "min_size":        args.min_size,
        "prob_filter":     args.prob_filter,
        "closing_mm":      args.closing_mm,
        "n":               len(records),
        "mean_dice":       float(np.mean(dices)),
        "median_dice":     float(np.median(dices)),
        "std_dice":        float(np.std(dices)),
        "mean_dice_pos":   float(np.mean([r["dice"] for r in gt_pos_recs])) if gt_pos_recs else None,
        "detection_rate":  det_rate,
        "fp_rate_neg":     fp_rate,
        "precision_global": float(prec_g),
        "recall_global":   float(rec_g),
        "mean_lesion_f1":  float(np.mean([r["lesion_f1"] for r in records])),
    }

    (out_dir / "metrics.json").write_text(json.dumps(records, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===")
    print(f"  mean_dice       : {summary['mean_dice']:.4f}")
    print(f"  median_dice     : {summary['median_dice']:.4f}")
    if summary["mean_dice_pos"]:
        print(f"  mean_dice (pos) : {summary['mean_dice_pos']:.4f}")
    if summary["detection_rate"] is not None:
        print(f"  detection_rate  : {summary['detection_rate']:.3f}")
    if summary["fp_rate_neg"] is not None:
        print(f"  fp_rate (neg)   : {summary['fp_rate_neg']:.3f}")
    print(f"  precision       : {summary['precision_global']:.4f}")
    print(f"  recall          : {summary['recall_global']:.4f}")
    print(f"  mean_lesion_f1  : {summary['mean_lesion_f1']:.4f}")
    print(f"\n  Results saved → {out_dir}/")


if __name__ == "__main__":
    main()
