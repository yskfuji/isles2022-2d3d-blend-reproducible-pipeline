"""IO utilities for medical volume preprocessing.

Centralizes NIfTI I/O and hashing so all reads/writes share the same interface.
"""
from pathlib import Path
import hashlib
from typing import Tuple, Optional
import nibabel as nib
import numpy as np


def load_nifti(path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data, img


def save_nifti(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_img = nib.Nifti1Image(data, affine=ref_img.affine, header=ref_img.header)
    nib.save(new_img, str(out_path))


def save_nifti_with_affine(
    data: np.ndarray,
    affine: np.ndarray,
    out_path: str,
    *,
    header: Optional[nib.Nifti1Header] = None,
) -> None:
    """Save a NIfTI with an explicit affine.

    Use this when you resample data and must write correct voxel spacing/origin
    into the output file. Passing the original header after resampling can leave
    inconsistent pixdim/zooms.
    """
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    hdr = header.copy() if header is not None else None
    new_img = nib.Nifti1Image(data, affine=np.asarray(affine, dtype=np.float64), header=hdr)
    nib.save(new_img, str(out_path_p))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()
