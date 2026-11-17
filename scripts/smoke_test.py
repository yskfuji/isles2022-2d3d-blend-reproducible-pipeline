#!/usr/bin/env python3
"""Smoke test: verify the public bundle without medical data.

Runs three checks:
  1. Manifest check — every file listed in ``sample_manifest.json`` exists.
  2. Model forward shape check — ``ConvNeXtNnUNetSeg`` accepts the documented
     21-channel input (7 offsets × 3 modalities) at 256×256 and produces a
     (B, 1, 256, 256) logit tensor.
  3. Dummy-volume end-to-end eval — builds tiny random 3D probability volumes
     for "nnU-Net 2D 3-fold avg" and "nnU-Net 3D 2-fold avg" and a 3D ConvNeXt
     prob in the same space, then runs the cross-arch ensemble math
     (weighted average + adaptive thresholding) and verifies the output is
     a sensible binary volume.

Usage:
    python scripts/smoke_test.py --use_dummy_data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check_manifest() -> bool:
    manifest_path = ROOT / "sample_manifest.json"
    if not manifest_path.exists():
        print(f"  [FAIL] manifest not found at {manifest_path}")
        return False
    manifest = json.loads(manifest_path.read_text())
    missing = []
    for rel in manifest.get("expected_files", []):
        if not (ROOT / rel).exists():
            missing.append(rel)
    if missing:
        print(f"  [FAIL] {len(missing)} expected file(s) missing:")
        for m in missing[:10]:
            print(f"    - {m}")
        return False
    n = len(manifest["expected_files"])
    print(f"  [OK] manifest: all {n} expected file(s) present")
    return True


def check_model_forward() -> bool:
    try:
        sys.path.insert(0, str(ROOT / "core" / "pipeline"))
        import torch
        from src.models.convnext_nnunet_seg import ConvNeXtNnUNetSeg
    except Exception as e:
        print(f"  [FAIL] could not import ConvNeXtNnUNetSeg: {e}")
        return False
    try:
        in_ch = 21  # 7 slice offsets × 3 modalities (DWI, ADC, FLAIR)
        model = ConvNeXtNnUNetSeg(in_channels=in_ch, backbone="convnext_tiny",
                                  pretrained=False, dec_ch=256, deep_sup=False)
        model.eval()
        x = torch.randn(1, in_ch, 256, 256)
        with torch.no_grad():
            y = model(x)
        if isinstance(y, list):
            y = y[0]
        if tuple(y.shape) != (1, 1, 256, 256):
            print(f"  [FAIL] unexpected output shape: {tuple(y.shape)}")
            return False
        print(f"  [OK] ConvNeXtNnUNetSeg forward: {tuple(x.shape)} -> {tuple(y.shape)}")
        return True
    except Exception as e:
        print(f"  [FAIL] model forward raised: {e}")
        return False


def _post_process(prob, base_thr, low_thr=0.0, high_vol=0):
    """Replicates the production post_process used in cross_arch_ensemble_native.py."""
    import numpy as np
    binary = (prob >= base_thr).astype(np.uint8)
    if low_thr > 0 and high_vol > 0 and int(binary.sum()) > high_vol:
        binary = (prob >= low_thr).astype(np.uint8)
    return binary


def check_cross_arch_dummy() -> bool:
    try:
        import numpy as np
    except Exception as e:
        print(f"  [FAIL] numpy import: {e}")
        return False
    rng = np.random.default_rng(0)
    # Two synthetic cases: a small lesion and a large under-predicted lesion
    cases = {"small": (32, 0.1), "large": (32, 0.6)}  # cube_size, prob_scale
    n_pass = 0
    for case, (D, pscale) in cases.items():
        nn_blend = rng.random((D, D, D)).astype(np.float32) * pscale
        cn = rng.random((D, D, D)).astype(np.float32) * pscale * 0.5
        # Cross-arch fusion (production recipe)
        w_cn = 0.20
        combined = w_cn * cn + (1.0 - w_cn) * nn_blend
        # Adaptive threshold (production recipe)
        pred = _post_process(combined, base_thr=0.30, low_thr=0.03, high_vol=4000)
        n_pred = int(pred.sum())
        if pred.dtype != np.uint8 or pred.shape != (D, D, D):
            print(f"  [FAIL] case={case}: bad output type/shape")
            return False
        # For 'large' case we expect adaptive switch to fire (volume > 4000)
        base_pred = int((combined >= 0.30).sum())
        adaptive_fired = base_pred > 4000
        switched = (n_pred != base_pred)
        if adaptive_fired and not switched:
            print(f"  [FAIL] case={case}: adaptive should have fired (base={base_pred}) but didn't")
            return False
        if (not adaptive_fired) and switched:
            print(f"  [FAIL] case={case}: adaptive fired without high-volume trigger")
            return False
        n_pass += 1
        print(f"  [OK] case={case}: base_pred={base_pred}, adaptive_fired={adaptive_fired}, final_pred={n_pred}")
    return n_pass == len(cases)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_dummy_data", action="store_true",
                        help="Run end-to-end checks with synthetic numpy volumes (no medical data).")
    args = parser.parse_args()

    print("=== Smoke test ===")
    ok_manifest = check_manifest()
    ok_model = check_model_forward()
    ok_crossarch = True
    if args.use_dummy_data:
        ok_crossarch = check_cross_arch_dummy()

    print()
    print(f"Manifest:        {'OK' if ok_manifest else 'FAIL'}")
    print(f"Model forward:   {'OK' if ok_model else 'FAIL'}")
    if args.use_dummy_data:
        print(f"Cross-arch math: {'OK' if ok_crossarch else 'FAIL'}")
    all_ok = ok_manifest and ok_model and ok_crossarch
    print()
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
