"""2.5D (stacked slices) ConvNeXt + nnU-Net-like decoder segmentation training for ISLES.

This is a minimal, smoke-test friendly training script:
- reads YAML config
- uses IslesVolumeDataset + IslesSliceDataset (2.5D)
- trains ConvNeXt-Tiny encoder + simple FPN decoder
- logs last checkpoint each epoch
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import typer

from ..datasets.isles_dataset import IslesVolumeDataset, IslesSliceDataset
from ..models.convnext_nnunet_seg import ConvNeXtNnUNetSeg
from .losses import (
    DiceBCELoss,
    DiceFocalBCEBoundaryLoss,
    DiceFocalLoss,
    DiceOHEMBCELoss,
    TverskyFocalLoss,
    TverskyOHEMBCELoss,
)
from .utils_train import set_seed, prepare_device, AverageMeter, dice_from_logits


class CopyPasteAug:
    """On-the-fly CopyPaste augmentation for sparse lesions.

    Maintains a per-worker rolling pool of lesion crops (image patch + mask) and pastes
    a random one into the current sample at a random location with probability `p`.
    The paste only overwrites pixels under the lesion mask, preserving surrounding anatomy.
    """

    def __init__(
        self,
        p: float = 0.5,
        max_pool: int = 200,
        min_lesion_voxels: int = 8,
        pad: int = 4,
        max_paste_per_sample: int = 1,
    ) -> None:
        self.p = float(p)
        self.max_pool = int(max_pool)
        self.min_lesion_voxels = int(min_lesion_voxels)
        self.pad = int(pad)
        self.max_paste_per_sample = int(max_paste_per_sample)
        self._pool: list[tuple[np.ndarray, np.ndarray]] = []  # (image_crop CHW, mask_crop HW)

    def _bbox(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        ys, xs = np.where(mask > 0.5)
        if len(ys) < self.min_lesion_voxels:
            return None
        return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

    def update_pool(self, image: np.ndarray, mask: np.ndarray) -> None:
        bb = self._bbox(mask)
        if bb is None:
            return
        y0, y1, x0, x1 = bb
        H, W = mask.shape
        y0 = max(0, y0 - self.pad)
        x0 = max(0, x0 - self.pad)
        y1 = min(H, y1 + self.pad)
        x1 = min(W, x1 + self.pad)
        img_crop = image[:, y0:y1, x0:x1].copy()
        mask_crop = mask[y0:y1, x0:x1].copy()
        if len(self._pool) >= self.max_pool:
            self._pool.pop(0)
        self._pool.append((img_crop, mask_crop))

    def apply(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._pool:
            return image, mask
        for _ in range(self.max_paste_per_sample):
            if np.random.rand() > self.p:
                continue
            idx = int(np.random.randint(len(self._pool)))
            img_crop, mask_crop = self._pool[idx]
            H, W = image.shape[-2], image.shape[-1]
            h, w = mask_crop.shape
            if h > H or w > W:
                continue
            y = int(np.random.randint(0, H - h + 1))
            x = int(np.random.randint(0, W - w + 1))
            paste_bool = mask_crop > 0.5
            if not paste_bool.any():
                continue
            # Overwrite all channels under the lesion mask.
            for c in range(image.shape[0]):
                tgt_patch = image[c, y : y + h, x : x + w]
                src_patch = img_crop[c]
                tgt_patch[paste_bool] = src_patch[paste_bool]
            mask_patch = mask[y : y + h, x : x + w]
            np.maximum(mask_patch, mask_crop, out=mask_patch)
        return image, mask

app = typer.Typer(add_completion=False)


def _center_pad_crop_2d(
    img: torch.Tensor,
    mask: torch.Tensor,
    *,
    out_h: int,
    out_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # img: (C,H,W), mask: (H,W)
    if img.ndim != 3:
        raise ValueError(f"Expected img with shape (C,H,W), got {tuple(img.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"Expected mask with shape (H,W), got {tuple(mask.shape)}")

    h, w = int(img.shape[-2]), int(img.shape[-1])
    pad_h = max(0, int(out_h) - h)
    pad_w = max(0, int(out_w) - w)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        mask = F.pad(mask.unsqueeze(0), (pad_left, pad_right, pad_top, pad_bottom), value=0.0).squeeze(0)

    h2, w2 = int(img.shape[-2]), int(img.shape[-1])
    if h2 > out_h:
        top = (h2 - int(out_h)) // 2
        img = img[:, top : top + int(out_h), :]
        mask = mask[top : top + int(out_h), :]
    if w2 > out_w:
        left = (w2 - int(out_w)) // 2
        img = img[:, :, left : left + int(out_w)]
        mask = mask[:, left : left + int(out_w)]

    return img, mask


Sample = dict[str, Any]


def _to_tensor(sample: Sample) -> Sample:
    if not torch.is_tensor(sample["image"]):
        sample["image"] = torch.from_numpy(sample["image"])
    if not torch.is_tensor(sample["mask"]):
        sample["mask"] = torch.from_numpy(sample["mask"])
    sample["image"] = sample["image"].float()
    sample["mask"] = sample["mask"].float()
    if "teacher" in sample:
        if not torch.is_tensor(sample["teacher"]):
            sample["teacher"] = torch.from_numpy(sample["teacher"])
        sample["teacher"] = sample["teacher"].float()
    return sample


def _make_affine_theta(angle_deg: float, scale: float) -> torch.Tensor:
    """Return a (1, 2, 3) affine theta for F.affine_grid (rotation + uniform scale)."""
    import math
    rad = math.radians(angle_deg)
    c = math.cos(rad) * scale
    s = math.sin(rad) * scale
    return torch.tensor([[c, s, 0.0], [-s, c, 0.0]], dtype=torch.float32).unsqueeze(0)


def _make_transform(
    img_size: tuple[int, int] | None,
    *,
    augment: bool,
    p_flip: float,
    aug_rotation: float = 0.0,
    aug_scale_range: tuple[float, float] | None = None,
    aug_gamma_range: tuple[float, float] | None = None,
    aug_noise_std: float = 0.0,
    # MRI-specific augmentation
    aug_intensity_scale_range: tuple[float, float] | None = None,
    aug_intensity_shift_std: float = 0.0,
    aug_blur_p: float = 0.0,
    aug_blur_sigma_range: tuple[float, float] = (0.5, 1.5),
    aug_bias_field_p: float = 0.0,
    aug_bias_field_std: float = 0.3,
    copy_paste: CopyPasteAug | None = None,
    modality_dropout_p: float = 0.0,
    modality_block_size: int = 1,
) -> Callable[[Sample], Sample]:
    p_flip_f = float(p_flip)

    def _tx(sample: Sample) -> Sample:
        # CopyPaste happens before _to_tensor while we still have numpy arrays.
        if bool(augment) and copy_paste is not None:
            img_np = sample["image"]
            mask_np = sample["mask"]
            if isinstance(img_np, torch.Tensor):
                img_np = img_np.numpy()
            if isinstance(mask_np, torch.Tensor):
                mask_np = mask_np.numpy()
            copy_paste.update_pool(img_np, mask_np)
            img_np, mask_np = copy_paste.apply(img_np.copy(), mask_np.copy())
            sample["image"] = img_np
            sample["mask"] = mask_np

        sample = _to_tensor(sample)

        # Modality dropout: zero out a contiguous block of channels (one modality's slices).
        if bool(augment) and modality_dropout_p > 0 and torch.rand(()) < modality_dropout_p:
            img = sample["image"]
            C = img.shape[0]
            block = max(1, int(modality_block_size))
            if C > block:
                start = int(torch.randint(0, max(1, C - block + 1), (1,)).item())
                img[start : start + block] = 0.0
                sample["image"] = img

        if bool(augment):
            # -- Horizontal and vertical flips (applied jointly to image, mask, teacher) --
            if torch.rand(()) < p_flip_f:
                sample["image"] = torch.flip(sample["image"], dims=[-1])
                sample["mask"] = torch.flip(sample["mask"], dims=[-1])
                if "teacher" in sample:
                    sample["teacher"] = torch.flip(sample["teacher"], dims=[-1])
            if torch.rand(()) < p_flip_f:
                sample["image"] = torch.flip(sample["image"], dims=[-2])
                sample["mask"] = torch.flip(sample["mask"], dims=[-2])
                if "teacher" in sample:
                    sample["teacher"] = torch.flip(sample["teacher"], dims=[-2])

            # -- Random affine: rotation and/or scale (applied jointly to image, mask, teacher) --
            do_affine = (aug_rotation > 0) or (aug_scale_range is not None)
            if do_affine:
                angle = float(torch.empty(1).uniform_(-aug_rotation, aug_rotation)) if aug_rotation > 0 else 0.0
                scale = float(torch.empty(1).uniform_(aug_scale_range[0], aug_scale_range[1])) if aug_scale_range is not None else 1.0
                theta = _make_affine_theta(angle, scale)  # (1, 2, 3)

                img = sample["image"]  # (C, H, W)
                h, w = img.shape[-2], img.shape[-1]
                grid = F.affine_grid(theta, (1, img.shape[0], h, w), align_corners=False)
                sample["image"] = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear", align_corners=False, padding_mode="zeros").squeeze(0)

                mask = sample["mask"]  # (H, W)
                grid_m = F.affine_grid(theta, (1, 1, h, w), align_corners=False)
                sample["mask"] = F.grid_sample(mask.unsqueeze(0).unsqueeze(0), grid_m, mode="nearest", align_corners=False, padding_mode="zeros").squeeze(0).squeeze(0)

                if "teacher" in sample:
                    t = sample["teacher"]  # (H, W) float
                    grid_t = F.affine_grid(theta, (1, 1, h, w), align_corners=False)
                    sample["teacher"] = F.grid_sample(t.unsqueeze(0).unsqueeze(0), grid_t, mode="bilinear", align_corners=False, padding_mode="zeros").squeeze(0).squeeze(0)

            # -- Random gamma intensity augmentation (image only, p=0.5) --
            if aug_gamma_range is not None and torch.rand(()) < 0.5:
                lo, hi = aug_gamma_range
                gamma = float(torch.empty(1).uniform_(lo, hi))
                img = sample["image"]
                img_min = img.min()
                img = (img - img_min).clamp(min=0.0).pow(gamma) + img_min
                sample["image"] = img

            # -- Gaussian noise (image only, p=0.5) --
            if aug_noise_std > 0 and torch.rand(()) < 0.5:
                sample["image"] = sample["image"] + torch.randn_like(sample["image"]) * aug_noise_std

            # -- Intensity scale/shift per-channel (MRI: intensity non-standardised) --
            if aug_intensity_scale_range is not None and torch.rand(()) < 0.5:
                lo, hi = aug_intensity_scale_range
                img = sample["image"]  # (C, H, W)
                scale = torch.empty(img.shape[0], 1, 1).uniform_(lo, hi)
                sample["image"] = img * scale
            if aug_intensity_shift_std > 0 and torch.rand(()) < 0.5:
                img = sample["image"]
                shift = torch.randn(img.shape[0], 1, 1) * aug_intensity_shift_std
                sample["image"] = img + shift

            # -- Gaussian blur (simulate MRI PSF variation) --
            if aug_blur_p > 0 and torch.rand(()) < aug_blur_p:
                import math as _m
                sigma = float(torch.empty(1).uniform_(aug_blur_sigma_range[0], aug_blur_sigma_range[1]))
                # Build a small Gaussian kernel
                ksize = int(2 * _m.ceil(2 * sigma) + 1)
                x = torch.arange(ksize, dtype=torch.float32) - ksize // 2
                gauss1d = torch.exp(-0.5 * (x / sigma) ** 2)
                gauss1d = gauss1d / gauss1d.sum()
                kernel = gauss1d[:, None] * gauss1d[None, :]  # (ksize, ksize)
                img = sample["image"]  # (C, H, W)
                C = img.shape[0]
                k2d = kernel.view(1, 1, ksize, ksize).expand(C, 1, ksize, ksize)
                pad = ksize // 2
                sample["image"] = F.conv2d(img.unsqueeze(0), k2d, padding=pad, groups=C).squeeze(0)

            # -- Bias field simulation (low-freq multiplicative intensity non-uniformity) --
            if aug_bias_field_p > 0 and torch.rand(()) < aug_bias_field_p:
                img = sample["image"]  # (C, H, W)
                H, W = img.shape[-2], img.shape[-1]
                # Generate coarse random field, upsample to full resolution
                coarse_h, coarse_w = max(4, H // 16), max(4, W // 16)
                field = torch.randn(1, 1, coarse_h, coarse_w) * aug_bias_field_std
                field = F.interpolate(field, size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
                bias = torch.exp(field)  # multiplicative, centered ~1
                sample["image"] = img * bias

        if img_size is not None:
            out_h, out_w = int(img_size[0]), int(img_size[1])
            img, mask = _center_pad_crop_2d(sample["image"], sample["mask"], out_h=out_h, out_w=out_w)
            sample["image"] = img
            sample["mask"] = mask
            if "teacher" in sample:
                # Reuse same pad/crop math by treating teacher as a (1, H, W) image with dummy mask.
                t = sample["teacher"].unsqueeze(0)  # (1, H, W)
                dummy = torch.zeros(t.shape[-2], t.shape[-1], dtype=t.dtype)
                t_out, _ = _center_pad_crop_2d(t, dummy, out_h=out_h, out_w=out_w)
                sample["teacher"] = t_out.squeeze(0)
        return sample

    return _tx


@app.command()
def main(
    config: str = typer.Argument(..., help="Path to YAML config"),
    resume: str = typer.Option(None, help="Path to last.pt checkpoint to resume from"),
) -> None:
    cfg = yaml.safe_load(Path(config).read_text())

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = prepare_device()

    data = cfg["data"]
    tr_cfg = cfg["train"]
    log_cfg = cfg["log"]

    csv_path = str(data["csv_path"])
    root = str(data["root"])
    k = int(data.get("k_slices", 2))
    _sof = data.get("slice_offsets")
    slice_offsets: list[int] | None = [int(x) for x in _sof] if _sof is not None else None
    normalize = str(data.get("normalize", "legacy_zscore"))

    # Stage1 cascade probs (optional)
    _repo_root = Path(__file__).resolve().parents[2]

    def _resolve_probs_dir(v: object) -> str | None:
        if not v:
            return None
        p = Path(str(v).strip())
        if not p.is_absolute():
            p = (_repo_root / p).resolve()
        return str(p) if p.exists() else None

    _s1 = data.get("stage1_probs_dir")
    stage1_probs_dir_train: str | None = _resolve_probs_dir(data.get("stage1_probs_dir_train") or _s1)
    stage1_probs_dir_val: str | None = _resolve_probs_dir(data.get("stage1_probs_dir_val") or _s1)

    # Teacher probs (KL distillation; used only on train split)
    teacher_probs_dir_train: str | None = _resolve_probs_dir(data.get("teacher_probs_dir"))

    img_size = data.get("img_size")
    if img_size is not None:
        if not (isinstance(img_size, (list, tuple)) and len(img_size) == 2):
            raise ValueError("data.img_size must be [H, W]")
        img_size = (int(img_size[0]), int(img_size[1]))

    vol_tr = IslesVolumeDataset(csv_path, split="train", root=root, transform=None, normalize=normalize)
    vol_va = IslesVolumeDataset(csv_path, split="val", root=root, transform=None, normalize=normalize, allow_missing_label=False)

    p_flip = float(tr_cfg.get("p_flip", 0.5))
    aug_rotation = float(tr_cfg.get("aug_rotation", 0.0))
    _asr = tr_cfg.get("aug_scale_range")
    aug_scale_range = (float(_asr[0]), float(_asr[1])) if _asr is not None else None
    _agr = tr_cfg.get("aug_gamma_range")
    aug_gamma_range = (float(_agr[0]), float(_agr[1])) if _agr is not None else None
    aug_noise_std = float(tr_cfg.get("aug_noise_std", 0.0))
    # MRI-specific augmentation
    _aisr = tr_cfg.get("aug_intensity_scale_range")
    aug_intensity_scale_range = (float(_aisr[0]), float(_aisr[1])) if _aisr is not None else None
    aug_intensity_shift_std = float(tr_cfg.get("aug_intensity_shift_std", 0.0))
    aug_blur_p = float(tr_cfg.get("aug_blur_p", 0.0))
    _absr = tr_cfg.get("aug_blur_sigma_range")
    aug_blur_sigma_range = (float(_absr[0]), float(_absr[1])) if _absr is not None else (0.5, 1.5)
    aug_bias_field_p = float(tr_cfg.get("aug_bias_field_p", 0.0))
    aug_bias_field_std = float(tr_cfg.get("aug_bias_field_std", 0.3))

    # CopyPaste + modality dropout
    copy_paste_p = float(tr_cfg.get("copy_paste_p", 0.0))
    copy_paste: CopyPasteAug | None = None
    if copy_paste_p > 0:
        copy_paste = CopyPasteAug(
            p=copy_paste_p,
            max_pool=int(tr_cfg.get("copy_paste_max_pool", 200)),
            min_lesion_voxels=int(tr_cfg.get("copy_paste_min_lesion_voxels", 8)),
            pad=int(tr_cfg.get("copy_paste_pad", 4)),
            max_paste_per_sample=int(tr_cfg.get("copy_paste_max_paste_per_sample", 1)),
        )
    modality_dropout_p = float(tr_cfg.get("modality_dropout_p", 0.0))
    modality_block_size = int(tr_cfg.get("modality_block_size", 1))

    tx_tr = _make_transform(
        img_size,
        augment=bool(tr_cfg.get("augment", False)),
        p_flip=p_flip,
        aug_rotation=aug_rotation,
        aug_scale_range=aug_scale_range,
        aug_gamma_range=aug_gamma_range,
        aug_noise_std=aug_noise_std,
        aug_intensity_scale_range=aug_intensity_scale_range,
        aug_intensity_shift_std=aug_intensity_shift_std,
        aug_blur_p=aug_blur_p,
        aug_blur_sigma_range=aug_blur_sigma_range,
        aug_bias_field_p=aug_bias_field_p,
        aug_bias_field_std=aug_bias_field_std,
        copy_paste=copy_paste,
        modality_dropout_p=modality_dropout_p,
        modality_block_size=modality_block_size,
    )
    tx_va = _make_transform(img_size, augment=False, p_flip=0.0)
    ds_tr = IslesSliceDataset(vol_tr, k=k, transform=tx_tr, slice_offsets=slice_offsets,
                              stage1_probs_dir=stage1_probs_dir_train,
                              teacher_probs_dir=teacher_probs_dir_train)
    ds_va = IslesSliceDataset(vol_va, k=k, transform=tx_va, slice_offsets=slice_offsets,
                              stage1_probs_dir=stage1_probs_dir_val)

    batch_size = int(tr_cfg["batch_size"])
    num_workers = int(tr_cfg.get("num_workers", 0))
    sampler_mode = str(tr_cfg.get("sampler", "shuffle")).strip().lower()

    sampler = None
    shuffle = True
    if sampler_mode in {"pos_oversample", "positive_oversample", "balanced"}:
        # Build weights from label volumes only (cheap-ish) to oversample slices containing lesions.
        pos_w = float(tr_cfg.get("pos_slice_weight", 10.0))
        neg_w = float(tr_cfg.get("neg_slice_weight", 1.0))

        pos_z_by_vidx: dict[int, set[int]] = {}
        for vidx in range(len(vol_tr)):
            case = vol_tr[int(vidx)]
            mask3d = case["mask"]
            # mask3d: (Z,Y,X)
            z_any = (mask3d.reshape(mask3d.shape[0], -1).sum(axis=1) > 0)
            pos_z_by_vidx[int(vidx)] = set(int(i) for i in np.where(z_any)[0].tolist())

        weights = []
        for vidx, z in ds_tr.index_map:
            weights.append(pos_w if int(z) in pos_z_by_vidx.get(int(vidx), set()) else neg_w)
        weights_t = torch.as_tensor(weights, dtype=torch.double)

        max_train_batches = tr_cfg.get("max_train_batches")
        if max_train_batches is None:
            # default: one pass over the slice dataset
            num_samples = int(len(ds_tr))
        else:
            num_samples = int(max_train_batches) * int(batch_size)
        sampler = WeightedRandomSampler(weights=weights_t, num_samples=num_samples, replacement=True)
        shuffle = False

    loader_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    loader_va = DataLoader(
        ds_va,
        batch_size=int(tr_cfg.get("val_batch_size", tr_cfg["batch_size"])),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Infer input channels from one sample (already stacked to 2.5D).
    sample0 = ds_tr[0]
    in_ch = int(sample0["image"].shape[0])

    model = ConvNeXtNnUNetSeg(
        in_channels=in_ch,
        backbone=str(tr_cfg.get("backbone", "convnext_tiny")),
        pretrained=bool(tr_cfg.get("pretrained", True)),
        first_conv_init=str(tr_cfg.get("first_conv_init", "repeat")),
        dec_ch=int(tr_cfg.get("dec_ch", 256)),
        out_channels=1,
        stage_dropout_p=float(tr_cfg.get("stage_dropout_p", 0.0)),
        decoder_dropout_p=float(tr_cfg.get("decoder_dropout_p", 0.0)),
        deep_sup=bool(tr_cfg.get("deep_sup", False)),
        hint_attn=bool(tr_cfg.get("hint_attn", False)),
    ).to(device)

    # Optional warm-start: load weights from an existing checkpoint (e.g. v3) with strict=False
    # so that new modules (hint_attn_conv) start from their zero-init defaults.
    warmstart_from = tr_cfg.get("warmstart_from")
    if warmstart_from:
        ws_path = Path(str(warmstart_from).strip())
        if not ws_path.is_absolute():
            ws_path = (Path(__file__).resolve().parents[2] / ws_path).resolve()
        if ws_path.exists():
            ws_sd = torch.load(str(ws_path), map_location="cpu")
            if isinstance(ws_sd, dict) and "model" in ws_sd and not any(k.startswith("encoder.") for k in ws_sd):
                ws_sd = ws_sd["model"]
            missing, unexpected = model.load_state_dict(ws_sd, strict=False)
            print(f"Warm-start from {ws_path}: missing={len(missing)} keys, unexpected={len(unexpected)} keys")
            if missing:
                print(f"  missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        else:
            print(f"[WARN] warmstart_from path not found: {ws_path}")

    loss_name = str(tr_cfg.get("loss", "dice_bce")).strip().lower()
    if loss_name in {"dice_bce", "dicebce"}:
        criterion = DiceBCELoss(pos_weight=float(tr_cfg.get("pos_weight", 1.0)))
    elif loss_name in {"dice_focal", "dicefocal", "focal"}:
        criterion = DiceFocalLoss(alpha=float(tr_cfg.get("focal_alpha", 0.25)), gamma=float(tr_cfg.get("focal_gamma", 2.0)))
    elif loss_name in {"dice_ohem_bce", "diceohembce", "ohem"}:
        criterion = DiceOHEMBCELoss(
            neg_fraction=float(tr_cfg.get("ohem_neg_fraction", 0.1)),
            min_neg=int(tr_cfg.get("ohem_min_neg", 1024)),
            pos_weight=float(tr_cfg.get("ohem_pos_weight", 1.0)),
            neg_weight=float(tr_cfg.get("ohem_neg_weight", 1.0)),
        )
    elif loss_name in {"tversky_focal", "tverskyfocal"}:
        criterion = TverskyFocalLoss(
            alpha=float(tr_cfg.get("tversky_alpha", 0.3)),
            beta=float(tr_cfg.get("tversky_beta", 0.7)),
            gamma=float(tr_cfg.get("tversky_gamma", 1.33)),
        )
    elif loss_name in {"tversky_ohem_bce", "tverskyohembce"}:
        criterion = TverskyOHEMBCELoss(
            alpha=float(tr_cfg.get("tversky_alpha", 0.3)),
            beta=float(tr_cfg.get("tversky_beta", 0.7)),
            neg_fraction=float(tr_cfg.get("ohem_neg_fraction", 0.1)),
            min_neg=int(tr_cfg.get("ohem_min_neg", 1024)),
            bce_weight=float(tr_cfg.get("bce_weight", 1.0)),
        )
    elif loss_name in {"dice_focal_bce_boundary", "dicefocalbceboundary", "dfbb"}:
        criterion = DiceFocalBCEBoundaryLoss(
            dice_weight=float(tr_cfg.get("dice_weight", 0.5)),
            focal_weight=float(tr_cfg.get("focal_weight", 0.5)),
            focal_alpha=float(tr_cfg.get("focal_alpha", 0.25)),
            focal_gamma=float(tr_cfg.get("focal_gamma", 2.0)),
            bce_weight=float(tr_cfg.get("bce_weight", 0.0)),
            boundary_weight=float(tr_cfg.get("boundary_weight", 0.2)),
        ).to(device)
    else:
        raise ValueError(f"Unknown train.loss: {loss_name!r}")
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr_cfg["lr"]),
        weight_decay=float(tr_cfg.get("weight_decay", 1e-4)),
    )

    # ---- EMA (Exponential Moving Average) ----
    use_ema = bool(tr_cfg.get("ema", False))
    ema_decay = float(tr_cfg.get("ema_decay", 0.9998))
    if use_ema:
        ema_params: dict[str, torch.Tensor] = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
    else:
        ema_params: dict[str, torch.Tensor] = {}

    # ---- SWA (Stochastic Weight Averaging) over the tail of training ----
    use_swa = bool(tr_cfg.get("swa", False))
    swa_start_epoch = int(tr_cfg.get("swa_start_epoch", 10**9))
    swa_state: dict[str, torch.Tensor] = {}
    swa_n: int = 0  # number of snapshots averaged so far

    # ---- Gradient accumulation ----
    grad_accum_steps = max(1, int(tr_cfg.get("grad_accum_steps", 1)))

    # ---- LR Scheduler: linear warmup + cosine annealing ----
    import math as _math
    sched_name = str(tr_cfg.get("scheduler", "none")).strip().lower()
    warmup_epochs = int(tr_cfg.get("warmup_epochs", 0))
    min_lr = float(tr_cfg.get("min_lr", 1e-6))
    total_epochs = int(tr_cfg["epochs"])
    base_lr = float(tr_cfg["lr"])

    if sched_name == "cosine":
        def _lr_lambda(ep: int) -> float:
            if warmup_epochs > 0 and ep < warmup_epochs:
                return (ep + 1) / max(1, warmup_epochs)
            progress = (ep - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cos_decay = 0.5 * (1.0 + _math.cos(_math.pi * progress))
            return min_lr / base_lr + (1.0 - min_lr / base_lr) * cos_decay
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda)
    else:
        scheduler = None

    amp = bool(tr_cfg.get("amp", False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    out_dir = Path(log_cfg["out_dir"]) / str(cfg.get("experiment_name", "isles_25d_convnext_fpn"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist config snapshot.
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    best_val_dice = float("-inf")
    start_epoch = 1

    # ---- Resume from checkpoint ----
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        # Re-init EMA from loaded weights (history lost, but re-warms quickly)
        if use_ema:
            ema_params = {name: param.detach().clone() for name, param in model.named_parameters()}
        # Recover best_val_dice from log if available
        log_path = out_dir / "log.jsonl"
        if log_path.exists():
            with log_path.open() as _f:
                _lines = [json.loads(l) for l in _f if l.strip()]
            if _lines:
                best_val_dice = max(l["best_val_dice"] for l in _lines)
        print(f"Resumed from epoch {ckpt['epoch']}, best_val_dice={best_val_dice:.4f}, continuing from epoch {start_epoch}")

    # ---- Knowledge distillation params (binary, with temperature) ----
    use_kd = teacher_probs_dir_train is not None
    kd_weight = float(tr_cfg.get("kd_weight", 0.5))     # (1 - alpha): weight on KL term
    kd_temperature = float(tr_cfg.get("kd_temperature", 3.0))
    if use_kd:
        print(f"KD enabled: teacher={teacher_probs_dir_train}, kd_weight={kd_weight}, T={kd_temperature}")

    def _binary_kl_with_temperature(student_logits: torch.Tensor,
                                    teacher_prob: torch.Tensor,
                                    T: float) -> torch.Tensor:
        """KL(soft_teacher || soft_student) for binary segmentation with temperature T.

        student_logits: (B, 1, H, W) raw logits
        teacher_prob:   (B, 1, H, W) probability in [0, 1]
        Returns scalar loss (T^2 * mean KL).
        """
        eps = 1e-6
        p_t = teacher_prob.clamp(eps, 1.0 - eps)
        teacher_logit = torch.log(p_t / (1.0 - p_t))
        # soft probabilities with temperature
        q_s = torch.sigmoid(student_logits / T)
        q_t = torch.sigmoid(teacher_logit / T)
        q_s = q_s.clamp(eps, 1.0 - eps)
        q_t_c = q_t.clamp(eps, 1.0 - eps)
        kl = q_t_c * (torch.log(q_t_c) - torch.log(q_s)) + \
             (1.0 - q_t_c) * (torch.log(1.0 - q_t_c) - torch.log(1.0 - q_s))
        return (T * T) * kl.mean()

    epochs = int(tr_cfg["epochs"])
    max_train_batches = tr_cfg.get("max_train_batches")
    max_val_batches = tr_cfg.get("max_val_batches")
    max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    max_val_batches = int(max_val_batches) if max_val_batches is not None else None
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        meter = AverageMeter()

        for bidx, batch in enumerate(loader_tr):
            if max_train_batches is not None and bidx >= max_train_batches:
                break
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            teacher = batch.get("teacher") if use_kd else None
            if teacher is not None:
                teacher = teacher.to(device)
                if teacher.ndim == 3:
                    teacher = teacher.unsqueeze(1)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(imgs)
                if isinstance(outputs, list):
                    # Deep supervision: weighted sum (main=1.0, aux3=0.5, aux4=0.25)
                    _ds_w = [1.0, 0.5, 0.25][: len(outputs)]
                    gt_loss = sum(w * criterion(o, masks) for w, o in zip(_ds_w, outputs)) / sum(_ds_w)
                    logits = outputs[0]
                else:
                    logits = outputs
                    gt_loss = criterion(logits, masks)

                if teacher is not None:
                    kd_loss = _binary_kl_with_temperature(logits, teacher, kd_temperature)
                    loss = (1.0 - kd_weight) * gt_loss + kd_weight * kd_loss
                else:
                    loss = gt_loss

                loss_scaled = loss / float(grad_accum_steps)

            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            do_step = ((bidx + 1) % grad_accum_steps == 0) or (
                max_train_batches is not None and bidx + 1 >= max_train_batches
            )
            if do_step:
                if scaler.is_enabled():
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)

                # Update EMA weights only on real optimizer steps.
                if use_ema:
                    for name, param in model.named_parameters():
                        ema_params[name].mul_(ema_decay).add_(param.detach(), alpha=1.0 - ema_decay)

            meter.update(float(loss.item()), int(imgs.size(0)))

        # Validation (swap to EMA weights if enabled)
        _orig_params: dict[str, torch.Tensor] = {}
        if use_ema:
            _orig_params = {n: p.detach().clone() for n, p in model.named_parameters()}
            for n, p in model.named_parameters():
                p.data.copy_(ema_params[n])

        model.eval()
        va_meter = AverageMeter()
        dices: list[float] = []
        with torch.no_grad():
            for bidx, batch in enumerate(loader_va):
                if max_val_batches is not None and bidx >= max_val_batches:
                    break
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                logits = model(imgs)
                loss = criterion(logits, masks)
                va_meter.update(float(loss.item()), int(imgs.size(0)))
                dices.append(float(dice_from_logits(logits, masks)))

        # Restore original weights after validation
        if use_ema:
            for n, p in model.named_parameters():
                p.data.copy_(_orig_params[n])

        val_dice = float(np.mean(dices)) if dices else float("nan")

        # Step LR scheduler at end of epoch
        if scheduler is not None:
            scheduler.step()

        # ---- SWA snapshot (after EMA weights are restored, on the trained weights) ----
        if use_swa and int(epoch) >= int(swa_start_epoch):
            with torch.no_grad():
                if swa_n == 0:
                    swa_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
                    swa_n = 1
                else:
                    swa_n += 1
                    inv = 1.0 / float(swa_n)
                    for n, p in model.named_parameters():
                        swa_state[n].mul_(1.0 - inv).add_(p.detach().cpu(), alpha=inv)

        # Save checkpoints
        ckpt = {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
        }
        torch.save(ckpt, out_dir / "last.pt")

        if np.isfinite(val_dice) and val_dice > best_val_dice:
            best_val_dice = float(val_dice)
            if use_ema:
                _bkp = {n: p.detach().clone() for n, p in model.named_parameters()}
                for n, p in model.named_parameters():
                    p.data.copy_(ema_params[n])
                torch.save(model.state_dict(), out_dir / "best.pt")
                for n, p in model.named_parameters():
                    p.data.copy_(_bkp[n])
            else:
                torch.save(model.state_dict(), out_dir / "best.pt")

        log = {
            "epoch": int(epoch),
            "train_loss": float(meter.avg),
            "val_loss": float(va_meter.avg),
            "val_dice": float(val_dice),
            "best_val_dice": float(best_val_dice),
            "lr": float(optim.param_groups[0]["lr"]),
        }
        with (out_dir / "log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")

        print(f"epoch {epoch} train_loss {meter.avg:.4f} val_loss {va_meter.avg:.4f} val_dice {val_dice:.4f}")

    # ---- After all epochs: save SWA weights ----
    if use_swa and swa_n > 0:
        # Load SWA averaged weights into model for BN recomputation, then save.
        # No BN in ConvNeXt (uses LayerNorm/GroupNorm) so we skip BN recompute.
        _orig = {n: p.detach().clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.data.copy_(swa_state[n].to(p.device))
        torch.save(model.state_dict(), out_dir / "swa.pt")
        # Restore (for safety in case caller chains another phase)
        for n, p in model.named_parameters():
            p.data.copy_(_orig[n])
        print(f"SWA: saved averaged-over-{swa_n}-epochs weights → swa.pt")


if __name__ == "__main__":
    app()
