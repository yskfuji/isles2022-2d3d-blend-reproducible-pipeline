"""Evaluation script for 2.5D ConvNeXt segmentation model (ISLES).

Key design:
  - Uses center_pad_crop_2d (matching training transform) NOT bilinear resize
  - Default post-processing: thr=0.85, min_size=64, prob_filter=0.96
  - Saves per-case metrics JSON + summary JSON to out_dir

Usage:
  cd ToReBrain-pipeline
  PYTHONPATH=$PWD python -m src.evaluation.evaluate_isles_25d \\
    --model-path results/isles_25d_convnext_nnunet_improved/isles_25d_convnext_nnunet_improved_v2_1mm/best.pt \\
    --csv-path data/splits/my_dataset_dwi_adc_flair_train_val_test.csv \\
    --root data/processed/my_dataset_dwi_adc_flair_1mm \\
    --split val \\
    --out-dir results/eval_25d_v2_1mm_val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from numpy.typing import NDArray
from scipy.ndimage import label as cc_label, binary_closing

from ..datasets.isles_dataset import IslesVolumeDataset
from ..models.convnext_nnunet_seg import ConvNeXtNnUNetSeg
from ..training.utils_train import prepare_device


# ─────────────────────────────────────────────────────────────────────────────
# Spatial transform (must match training _center_pad_crop_2d)
# ─────────────────────────────────────────────────────────────────────────────

def _center_pad_crop_np(arr: NDArray, out_h: int, out_w: int) -> NDArray:
    """Center-pad then center-crop a (C,H,W) or (H,W) array to (out_h, out_w).

    This replicates the training transform exactly so spatial coordinates align.
    """
    h, w = int(arr.shape[-2]), int(arr.shape[-1])
    pad_h = max(0, out_h - h)
    pad_w = max(0, out_w - w)
    pt, pb = pad_h // 2, pad_h - pad_h // 2
    pl, pr = pad_w // 2, pad_w - pad_w // 2

    if arr.ndim == 3:
        arr = np.pad(arr, ((0, 0), (pt, pb), (pl, pr)), mode="constant", constant_values=0.0)
    else:
        arr = np.pad(arr, ((pt, pb), (pl, pr)), mode="constant", constant_values=0.0)

    h2, w2 = int(arr.shape[-2]), int(arr.shape[-1])
    if h2 > out_h:
        top = (h2 - out_h) // 2
        arr = arr[..., top : top + out_h, :]
    if w2 > out_w:
        left = (w2 - out_w) // 2
        arr = arr[..., :, left : left + out_w]
    return arr


def _restore_pad_crop_np(pred: NDArray, orig_h: int, orig_w: int, out_h: int = 256, out_w: int = 256) -> NDArray:
    """Inverse of _center_pad_crop_np for a (Z, out_h, out_w) probability map.

    Recovers (Z, orig_h, orig_w) by reversing the pad/crop operations.
    """
    # Reverse crop: if orig > out, we had cropped → pad back
    h_cur, w_cur = pred.shape[-2], pred.shape[-1]
    if orig_h > out_h:
        t = (orig_h - out_h) // 2
        pred = np.pad(pred, ((0, 0), (t, orig_h - out_h - t), (0, 0)), constant_values=0.0)
    if orig_w > out_w:
        l = (orig_w - out_w) // 2
        pred = np.pad(pred, ((0, 0), (0, 0), (l, orig_w - out_w - l)), constant_values=0.0)

    # Reverse pad: if orig < out, we had padded → crop center back
    h_cur, w_cur = pred.shape[-2], pred.shape[-1]
    if h_cur > orig_h:
        t = (h_cur - orig_h) // 2
        pred = pred[..., t : t + orig_h, :]
    if w_cur > orig_w:
        l = (w_cur - orig_w) // 2
        pred = pred[..., :, l : l + orig_w]
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def _infer_volume_single(
    vol: NDArray,
    model: torch.nn.Module,
    offsets: list[int],
    img_size: tuple[int, int],
    device: torch.device,
    extra_vol: Optional[NDArray] = None,
) -> NDArray:
    """Single-pass 2.5D slice-by-slice inference. Returns (Z, out_h, out_w)."""
    C, Z, H_orig, W_orig = vol.shape
    out_h, out_w = img_size
    prob256 = np.zeros((Z, out_h, out_w), dtype=np.float32)

    with torch.no_grad():
        for z in range(Z):
            slices = [
                _center_pad_crop_np(vol[:, int(np.clip(z + off, 0, Z - 1)), :, :], out_h, out_w)
                for off in offsets
            ]
            inp_arr = np.concatenate(slices, axis=0)  # (C * n_offsets, H, W)
            if extra_vol is not None:
                extra_slice = _center_pad_crop_np(extra_vol[z : z + 1], out_h, out_w)  # (1, H, W)
                inp_arr = np.concatenate([inp_arr, extra_slice], axis=0)
            inp = torch.from_numpy(inp_arr).float().unsqueeze(0).to(device)
            logits = model(inp)
            prob256[z] = torch.sigmoid(logits).squeeze().cpu().numpy()

    return prob256


def infer_volume(
    vol: NDArray,
    model: torch.nn.Module,
    offsets: list[int],
    img_size: tuple[int, int],
    device: torch.device,
    extra_vol: Optional[NDArray] = None,  # (Z, H, W) — center slice only, not offset-expanded
    tta: bool = False,  # Test-Time Augmentation (LR/UD/LRUD flips + original, averaged)
    tta_rotations: bool = False,  # If True, also average 90/180/270 deg rotations (total 8-way w/ tta)
) -> NDArray:
    """Run 2.5D slice-by-slice inference on a (C, Z, H, W) volume.

    If extra_vol is provided (Z, H, W), its center slice is appended as +1 channel
    (not expanded across offsets), matching cascade Stage2 training behaviour.

    If tta=True, averages predictions over original + LR-flip + UD-flip + LR+UD-flip (4 passes).
    If tta_rotations=True, additionally averages 90/180/270 deg rotations (4 more → 8 total).
    Rotations require img_size to be square.

    Returns probability map (Z, H, W) at the original spatial resolution.
    """
    C, Z, H_orig, W_orig = vol.shape
    out_h, out_w = img_size

    prob256 = _infer_volume_single(vol, model, offsets, img_size, device, extra_vol)
    accumulators: list[NDArray] = [prob256]

    if tta:
        # LR flip (W axis)
        vol_lr = vol[:, :, :, ::-1].copy()
        ext_lr = extra_vol[:, :, ::-1].copy() if extra_vol is not None else None
        p_lr = _infer_volume_single(vol_lr, model, offsets, img_size, device, ext_lr)
        accumulators.append(p_lr[:, :, ::-1])  # un-flip

        # UD flip (H axis)
        vol_ud = vol[:, :, ::-1, :].copy()
        ext_ud = extra_vol[:, ::-1, :].copy() if extra_vol is not None else None
        p_ud = _infer_volume_single(vol_ud, model, offsets, img_size, device, ext_ud)
        accumulators.append(p_ud[:, ::-1, :])

        # LR+UD flip
        vol_lrud = vol[:, :, ::-1, ::-1].copy()
        ext_lrud = extra_vol[:, ::-1, ::-1].copy() if extra_vol is not None else None
        p_lrud = _infer_volume_single(vol_lrud, model, offsets, img_size, device, ext_lrud)
        accumulators.append(p_lrud[:, ::-1, ::-1])

    if tta_rotations:
        if int(out_h) != int(out_w):
            raise ValueError(f"tta_rotations requires square img_size; got {img_size}")
        # Apply 4-way rotations to the volume; inference operates on square out_h x out_w
        # so rotation/de-rotation in the (Z, out_h, out_w) prob space is well-defined.
        # Rotation by k*90 deg in (H, W) axes ↔ axes=(-2, -1) for vol (C, Z, H, W)
        for k in (1, 2, 3):
            vol_r = np.rot90(vol, k=k, axes=(-2, -1)).copy()
            ext_r = np.rot90(extra_vol, k=k, axes=(-2, -1)).copy() if extra_vol is not None else None
            p_r = _infer_volume_single(vol_r, model, offsets, img_size, device, ext_r)
            # un-rotate the probability map (Z, out_h, out_w)
            p_r = np.rot90(p_r, k=-k, axes=(-2, -1)).copy()
            accumulators.append(p_r)

    prob256 = np.mean(np.stack(accumulators, axis=0), axis=0)
    return _restore_pad_crop_np(prob256, H_orig, W_orig, out_h, out_w)


def _enable_dropout_only(model: torch.nn.Module) -> None:
    """Set model to eval mode but keep Dropout/Dropout2d layers active for MC sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()


def infer_volume_mc_dropout(
    vol: NDArray,
    model: torch.nn.Module,
    offsets: list[int],
    img_size: tuple[int, int],
    device: torch.device,
    n_passes: int = 30,
    extra_vol: Optional[NDArray] = None,
    tta: bool = False,
) -> dict[str, NDArray]:
    """MC Dropout inference: run n_passes stochastic forward passes.

    Requires model trained with dropout (stage_dropout_p > 0).
    Returns dict with:
      - 'mean': mean probability map (Z, H, W)
      - 'variance': variance across passes (Z, H, W) — epistemic uncertainty
      - 'entropy': predictive entropy of mean prediction (Z, H, W)
    """
    _enable_dropout_only(model)

    C, Z, H_orig, W_orig = vol.shape
    out_h, out_w = img_size

    probs_list: list[NDArray] = []
    with torch.no_grad():
        for _ in range(n_passes):
            p = infer_volume.__wrapped__ if hasattr(infer_volume, "__wrapped__") else None
            prob = _infer_volume_single(vol, model, offsets, img_size, device, extra_vol)
            if tta:
                vol_lr = vol[:, :, :, ::-1].copy()
                p_lr = _infer_volume_single(vol_lr, model, offsets, img_size, device,
                                            extra_vol[:, :, ::-1].copy() if extra_vol is not None else None)
                p_lr = p_lr[:, :, ::-1]
                vol_ud = vol[:, :, ::-1, :].copy()
                p_ud = _infer_volume_single(vol_ud, model, offsets, img_size, device,
                                            extra_vol[:, ::-1, :].copy() if extra_vol is not None else None)
                p_ud = p_ud[:, ::-1, :]
                prob = (prob + p_lr + p_ud) / 3.0
            probs_list.append(_restore_pad_crop_np(prob, H_orig, W_orig, out_h, out_w))

    model.eval()  # restore to full eval (dropout off)

    stacked = np.stack(probs_list, axis=0)  # (T, Z, H, W)
    mean_prob = stacked.mean(axis=0)
    variance = stacked.var(axis=0)
    eps = 1e-7
    entropy = -(mean_prob * np.log(mean_prob + eps) + (1 - mean_prob) * np.log(1 - mean_prob + eps))

    return {
        "mean": mean_prob,
        "variance": variance,
        "entropy": entropy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def postprocess(
    prob: NDArray,
    *,
    thr: float,
    min_size: int,
    prob_filter: float,
    closing_mm: int = 0,
) -> NDArray:
    """Binarize + connected-component filtering + optional morphological closing.

    Steps:
      1. Threshold at thr
      2. Remove CC components with voxel count < min_size
      3. Remove CC components with mean probability < prob_filter
         (FP suppression on high-confidence fragments only)
      4. (Optional) 3D binary closing with sphere of radius closing_mm voxels
         — connects nearby surviving fragments of large diffuse lesions
    """
    pred = (prob > thr).astype(np.uint8)

    # CC filtering (prob_filter + min_size) on the raw threshold mask
    if min_size > 0 or prob_filter > 0.0:
        lbl, n_cc = cc_label(pred)
        filtered = np.zeros_like(pred)
        for c in range(1, n_cc + 1):
            comp = lbl == c
            if int(comp.sum()) < min_size:
                continue
            if prob_filter > 0.0 and float(prob[comp].mean()) < prob_filter:
                continue
            filtered[comp] = 1
        pred = filtered

    # Morphological closing on the cleaned-up mask
    if closing_mm > 0:
        r = int(closing_mm)
        coords = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
        sphere = (coords[0]**2 + coords[1]**2 + coords[2]**2) <= r**2
        pred = binary_closing(pred, structure=sphere).astype(np.uint8)

    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _dice(pred: NDArray, gt: NDArray, eps: float = 1e-6) -> float:
    tp = int((pred * gt).sum())
    return float((2 * tp + eps) / (int(pred.sum()) + int(gt.sum()) + eps))


def _lesionwise_f1(pred: NDArray, gt: NDArray) -> dict[str, float | int]:
    """Lesion-wise precision/recall/F1 using 26-connectivity."""
    lbl_p, n_p = cc_label(pred)
    lbl_g, n_g = cc_label(gt)

    tp_l = fn_l = fp_l = 0
    for gi in range(1, n_g + 1):
        g_comp = lbl_g == gi
        if bool((pred[g_comp] > 0).any()):
            tp_l += 1
        else:
            fn_l += 1
    for pi in range(1, n_p + 1):
        p_comp = lbl_p == pi
        if not bool((gt[p_comp] > 0).any()):
            fp_l += 1

    prec = tp_l / (tp_l + fp_l + 1e-6)
    rec  = tp_l / (tp_l + fn_l + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)
    return {"lesion_tp": tp_l, "lesion_fp": fp_l, "lesion_fn": fn_l,
            "lesion_precision": float(prec), "lesion_recall": float(rec),
            "lesion_f1": float(f1)}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate 2.5D ConvNeXt segmentation model on ISLES.")
    p.add_argument("--model-path", required=True, help="Path to best.pt checkpoint.")
    p.add_argument("--csv-path",   required=True, help="Split CSV (case_id, split).")
    p.add_argument("--root",       required=True, help="Preprocessed data root (images/, labels/).")
    p.add_argument("--split",      default="val",  help="Which split to evaluate (train/val/test).")
    p.add_argument("--out-dir",    required=True, help="Directory to write metrics.json + summary.json.")
    p.add_argument("--normalize",  default="fixed_nonzero_zscore")
    # Model architecture (auto-detected from config.yaml if available)
    p.add_argument("--k-slices",   type=int,   default=None, help="Context half-width (k). Auto from config.")
    p.add_argument("--img-size",   default=None, help="'H,W'. Auto from config.")
    p.add_argument("--backbone",   default=None, help="Encoder backbone. Auto from config.")
    p.add_argument("--dec-ch",     type=int,   default=None, help="Decoder channels. Auto from config.")
    p.add_argument("--deep-sup",   action="store_true", default=None, help="Deep supervision heads.")
    # Post-processing
    p.add_argument("--thr",          type=float, default=0.85, help="Probability threshold.")
    p.add_argument("--min-size",     type=int,   default=64,   help="Min CC component voxels.")
    p.add_argument("--prob-filter",  type=float, default=0.96, help="Min mean prob per CC component.")
    p.add_argument("--stage1-probs-dir", default=None,
                   help="Stage1 prob dir ({case_id}.npz with key 'probs'). If set, in_channels += 1.")
    p.add_argument("--save-probs-dir", default=None,
                   help="If set, save raw probability maps as {case_id}.npz (key='probs', float16) to this dir.")
    p.add_argument("--tta", action="store_true", default=False,
                   help="Test-Time Augmentation: average over original + LR/UD/LR+UD flips.")
    args = p.parse_args()

    model_path = Path(args.model_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_probs_dir: Optional[Path] = None
    if args.save_probs_dir:
        save_probs_dir = Path(args.save_probs_dir)
        save_probs_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config from checkpoint directory if available ─────────────────
    config_path = model_path.parent / "config.yaml"
    cfg_data = {}
    cfg_train = {}
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text())
        cfg_data  = cfg.get("data",  {})
        cfg_train = cfg.get("train", {})
        print(f"Loaded config: {config_path}")

    k         = args.k_slices if args.k_slices is not None else int(cfg_data.get("k_slices", 2))
    backbone  = args.backbone  if args.backbone  is not None else str(cfg_train.get("backbone", "convnext_tiny"))
    dec_ch    = args.dec_ch    if args.dec_ch    is not None else int(cfg_train.get("dec_ch", 256))
    deep_sup  = bool(cfg_train.get("deep_sup", False)) if args.deep_sup is None else bool(args.deep_sup)

    _sof = cfg_data.get("slice_offsets")
    offsets: list[int] = [int(x) for x in _sof] if _sof is not None else list(range(-k, k + 1))

    if args.img_size is not None:
        h, w = [int(x) for x in args.img_size.split(",")]
        img_size = (h, w)
    elif "img_size" in cfg_data and cfg_data["img_size"] is not None:
        img_size = tuple(int(x) for x in cfg_data["img_size"])
    else:
        img_size = (256, 256)

    n_modalities = 3  # DWI, ADC, FLAIR
    _s1_dir = args.stage1_probs_dir or cfg_data.get("stage1_probs_dir_val") or cfg_data.get("stage1_probs_dir")
    _has_s1 = bool(_s1_dir and Path(_s1_dir).exists())
    hint_attn = bool(cfg_train.get("hint_attn", False))
    in_channels  = n_modalities * len(offsets) + (1 if _has_s1 else 0)

    print(f"Model: {backbone}, k={k}, offsets={offsets}, in_ch={in_channels}, dec_ch={dec_ch}, "
          f"deep_sup={deep_sup}, img_size={img_size}")
    print(f"Post-proc: thr={args.thr}, min_size={args.min_size}, prob_filter={args.prob_filter}")

    # ── Load model ─────────────────────────────────────────────────────────
    device = prepare_device()
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
    state_dict = torch.load(str(model_path), map_location="cpu")
    # Support both raw state_dict and wrapped {"model": state_dict}
    if isinstance(state_dict, dict) and "model" in state_dict and not any(
        k.startswith("encoder.") for k in state_dict
    ):
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Loaded checkpoint: {model_path} → {device}")

    # ── Dataset ────────────────────────────────────────────────────────────
    ds = IslesVolumeDataset(
        csv_path=args.csv_path,
        split=args.split,
        root=args.root,
        normalize=args.normalize,
    )
    print(f"Split='{args.split}': {len(ds)} cases")

    # ── Evaluate ───────────────────────────────────────────────────────────
    records: list[dict] = []

    for idx in range(len(ds)):
        sample  = ds[idx]
        vol     = sample["image"].astype(np.float32)   # (C, Z, H, W)
        mask_gt = (sample["mask"] > 0.5).astype(np.uint8)  # (Z, H, W)
        case_id = str(sample["case_id"])

        if _has_s1:
            npz_path = Path(_s1_dir) / f"{case_id}.npz"
            if npz_path.exists():
                s1 = np.load(str(npz_path))["probs"].astype(np.float32)  # (Z, H, W)
            else:
                s1 = np.zeros((vol.shape[1], vol.shape[2], vol.shape[3]), dtype=np.float32)
            prob = infer_volume(vol, model, offsets=offsets, img_size=img_size, device=device, extra_vol=s1, tta=args.tta)
        else:
            prob = infer_volume(vol, model, offsets=offsets, img_size=img_size, device=device, tta=args.tta)
        if save_probs_dir is not None:
            np.savez_compressed(str(save_probs_dir / f"{case_id}.npz"),
                                probs=prob.astype(np.float16))

        pred = postprocess(prob, thr=args.thr, min_size=args.min_size, prob_filter=args.prob_filter)

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

    # ── Aggregate ──────────────────────────────────────────────────────────
    dices      = [r["dice"] for r in records]
    gt_pos_recs = [r for r in records if r["gt_pos"]]
    gt_neg_recs = [r for r in records if not r["gt_pos"]]

    # Detection rate: positive cases where tp_vox > 0
    det_rate = float(np.mean([r["tp_vox"] > 0 for r in gt_pos_recs])) if gt_pos_recs else None
    # FP rate: negative cases where pred_vox > 0
    fp_rate  = float(np.mean([r["pred_vox"] > 0 for r in gt_neg_recs])) if gt_neg_recs else None

    total_tp = sum(r["tp_vox"] for r in records)
    total_fp = sum(r["fp_vox"] for r in records)
    total_fn = sum(r["fn_vox"] for r in records)
    precision_global = total_tp / (total_tp + total_fp + 1e-6)
    recall_global    = total_tp / (total_tp + total_fn + 1e-6)
    f1_global        = 2 * precision_global * recall_global / (precision_global + recall_global + 1e-6)

    lf1_vals = [r["lesion_f1"] for r in records]

    summary = {
        "model_path":      str(model_path),
        "csv_path":        args.csv_path,
        "root":            args.root,
        "split":           args.split,
        "normalize":       args.normalize,
        "k_slices":        k,
        "img_size":        list(img_size),
        "backbone":        backbone,
        "dec_ch":          dec_ch,
        "deep_sup":        deep_sup,
        "thr":             args.thr,
        "min_size":        args.min_size,
        "prob_filter":     args.prob_filter,
        "n":               len(records),
        "n_gt_pos":        len(gt_pos_recs),
        "n_gt_neg":        len(gt_neg_recs),
        "mean_dice":       float(np.mean(dices)),
        "median_dice":     float(np.median(dices)),
        "std_dice":        float(np.std(dices)),
        "mean_dice_pos":   float(np.mean([r["dice"] for r in gt_pos_recs])) if gt_pos_recs else None,
        "detection_rate":  det_rate,
        "fp_rate_neg":     fp_rate,
        "precision_global": float(precision_global),
        "recall_global":   float(recall_global),
        "f1_global":       float(f1_global),
        "mean_lesion_f1":  float(np.mean(lf1_vals)),
    }

    (out_dir / "metrics.json").write_text(json.dumps(records, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  mean_dice       : {summary['mean_dice']:.4f}")
    print(f"  median_dice     : {summary['median_dice']:.4f}")
    print(f"  mean_dice (pos) : {summary['mean_dice_pos']:.4f}" if summary["mean_dice_pos"] else "")
    print(f"  detection_rate  : {summary['detection_rate']:.3f}" if summary["detection_rate"] else "")
    print(f"  fp_rate (neg)   : {summary['fp_rate_neg']:.3f}" if summary["fp_rate_neg"] is not None else "")
    print(f"  precision       : {summary['precision_global']:.4f}")
    print(f"  recall          : {summary['recall_global']:.4f}")
    print(f"  mean_lesion_f1  : {summary['mean_lesion_f1']:.4f}")
    print(f"\n  Results saved → {out_dir}/")


if __name__ == "__main__":
    main()
