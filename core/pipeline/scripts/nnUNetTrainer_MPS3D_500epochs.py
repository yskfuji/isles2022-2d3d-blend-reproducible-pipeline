"""nnU-Net trainer variant that works around MPS lacking ConvTranspose3d.

Strategy: monkey-patch `get_matching_convtransp` in dynamic_network_architectures
to return a drop-in replacement (`_InterpConv3d`) that upsamples via trilinear
interpolation followed by a 1x1x1 Conv3d. Mathematically not identical to
ConvTranspose3d, but for nnU-Net's UNet decoder (where kernel_size == stride and
no padding is used) it preserves the spatial output shape and is the standard
"upsample + conv" alternative used widely in 3D segmentation literature.

Use:
    nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainer_MPS3D_500epochs -device mps
"""

from __future__ import annotations

import importlib
from typing import Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class _InterpConv3d(nn.Module):
    """Drop-in replacement for ``ConvTranspose3d`` used in nnU-Net UNet decoders.

    Mirrors the signature ``ConvTranspose3d(in_channels, out_channels, kernel_size, stride, ...)``.
    Implementation: nearest-neighbor upsample (via reshape+expand, MPS-native, no fallback),
    then a 3×3×3 Conv3d (padding=1) that smooths the blocky upsample and projects channels.

    Why reshape+expand: ``F.interpolate`` 3D nearest/trilinear is not implemented on the MPS
    device and falls back to CPU (very slow). Reshape+expand is a pure view-and-broadcast op
    that runs natively on every backend and is mathematically identical to nearest-neighbor
    upsampling for integer scale factors.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Any = 0,
        output_padding: Any = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Any = 1,
        padding_mode: str = "zeros",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if isinstance(stride, int):
            sf = (int(stride), int(stride), int(stride))
        else:
            sf = tuple(int(s) for s in stride)
        # The nearest-via-reshape trick only supports integer scale factors per axis.
        if not all(s >= 1 and int(s) == s for s in sf):
            raise ValueError(f"_InterpConv3d expects integer scale factors per axis, got {sf}")
        self.scale_factor = sf
        self.conv = nn.Conv3d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bool(bias),
            groups=int(groups),
        )

    @staticmethod
    def _upsample_nearest_3d(x: torch.Tensor, sf: Tuple[int, int, int]) -> torch.Tensor:
        sd, sh, sw = sf
        if (sd, sh, sw) == (1, 1, 1):
            return x
        n, c, d, h, w = x.shape
        # Insert singleton dims after each spatial axis, expand to scale factor, then collapse.
        return (
            x.view(n, c, d, 1, h, 1, w, 1)
            .expand(n, c, d, sd, h, sh, w, sw)
            .reshape(n, c, d * sd, h * sh, w * sw)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._upsample_nearest_3d(x, self.scale_factor)
        return self.conv(x)


def _mps_safe_get_matching_convtransp(conv_op=None, dimension: int | None = None):
    """Replacement for ``dynamic_network_architectures.building_blocks.helper.get_matching_convtransp``.

    Routes 3D ConvTranspose requests to the interpolate+Conv3d module; falls back to original
    behavior for 1D/2D so 2D models continue to use the native ConvTranspose2d.
    """
    from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in (1, 2, 3), "Dimension must be 1, 2 or 3"
    if dimension == 1:
        return nn.ConvTranspose1d
    if dimension == 2:
        return nn.ConvTranspose2d
    return _InterpConv3d


def _apply_mps_patch() -> None:
    """Replace ``get_matching_convtransp`` in every module that bound the original."""
    modules_to_patch = (
        "dynamic_network_architectures.building_blocks.helper",
        "dynamic_network_architectures.building_blocks.unet_decoder",
        "dynamic_network_architectures.building_blocks.unet_residual_decoder",
    )
    for mod_name in modules_to_patch:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "get_matching_convtransp"):
            mod.get_matching_convtransp = _mps_safe_get_matching_convtransp


class nnUNetTrainer_MPS3D_500epochs(nnUNetTrainer):
    """500-epoch nnU-Net trainer with MPS-compatible 3D up-conv replacement."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        # Patch BEFORE super().__init__() so build_network_architecture sees the override.
        _apply_mps_patch()
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        # MPS memory budget: 3D fullres with InstanceNorm fits only batch_size 1 at the
        # default patch size [80, 96, 80]. Override the batch_size loaded from plans.
        # Keep num_iterations_per_epoch at the default 250 (overall ~58h for 500 ep on MPS).
        try:
            cm = self.configuration_manager
            cm.configuration["batch_size"] = 1
        except Exception:
            pass
        self.batch_size = 1

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        # Re-apply patch defensively in case build_network_architecture is called from
        # a freshly-imported context (e.g. nnUNetv2_predict spawning workers).
        _apply_mps_patch()
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
