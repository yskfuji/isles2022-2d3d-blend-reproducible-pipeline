from __future__ import annotations

from typing import Final, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .input_adapters import adapt_first_conv


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, dropout_p: float = 0.0) -> None:
        super().__init__()
        # nnU-Net uses InstanceNorm3d/2d variants; for 2D, GroupNorm is a safe default.
        g1 = min(8, out_ch)
        g1 = 1 if out_ch % g1 != 0 else g1
        dp = float(dropout_p)
        if dp < 0 or dp >= 1:
            raise ValueError(f"dropout_p must be in [0,1), got {dropout_p}")
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g1, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dp) if dp > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g1, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dp) if dp > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvNeXtNnUNetSeg(nn.Module):
    """2.5D-friendly ConvNeXt encoder + nnU-Net-like U-Net decoder (2D).

    Pipeline:
      - Input: stacked slices (B, C_in, H, W) e.g. C_in=5 for k=2 and 1 modality
      - Encoder: torchvision ConvNeXt-Tiny (multi-channel stem via adapt_first_conv)
      - Decoder: U-Net-style (upsample + concat skip + conv blocks)
      - Output: segmentation logits for center slice (B, 1, H, W)

    Notes:
      - This is "nnU-Net-like" (U-Net decoder shape & skip connections). It is not a full
        reimplementation of nnU-Net training/inference tricks (deep supervision, etc.).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        first_conv_init: str = "repeat",
        dec_ch: int = 256,
        out_channels: int = 1,
        stage_dropout_p: float = 0.0,
        decoder_dropout_p: float = 0.0,
        deep_sup: bool = False,
        hint_attn: bool = False,
    ) -> None:
        super().__init__()

        backbone_s = str(backbone).strip().lower()
        if backbone_s != "convnext_tiny":
            raise ValueError("Only backbone='convnext_tiny' is supported in this minimal implementation")

        try:
            import torchvision  # type: ignore[import-not-found]

            weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if bool(pretrained) else None
            enc = torchvision.models.convnext_tiny(weights=weights)
        except Exception as e:
            raise RuntimeError("torchvision convnext_tiny is required for ConvNeXtNnUNetSeg") from e

        # hint_attn: last input channel is the Stage1 prob hint, processed separately.
        # The encoder only sees (in_channels - 1) channels; hint is injected in the decoder.
        self._hint_attn = bool(hint_attn)
        enc_in_ch = int(in_channels) - 1 if self._hint_attn else int(in_channels)

        # Adapt stem conv to multi-channel input.
        stem = enc.features[0]
        old_conv = stem[0]
        if not isinstance(old_conv, nn.Conv2d):
            raise RuntimeError("Unexpected ConvNeXt stem conv type")
        enc.features[0][0] = adapt_first_conv(old_conv, enc_in_ch, init_mode=str(first_conv_init))

        self.encoder = enc

        # ConvNeXt-Tiny stage channels for features indices (torchvision 0.21):
        # features[1]=96 @ 1/4, features[3]=192 @ 1/8, features[5]=384 @ 1/16, features[7]=768 @ 1/32
        c2: Final[int] = 96
        c3: Final[int] = 192
        c4: Final[int] = 384
        c5: Final[int] = 768
        d = int(dec_ch)

        # Lateral projections to decoder width.
        self.lat2 = nn.Conv2d(c2, d, kernel_size=1)
        self.lat3 = nn.Conv2d(c3, d, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, d, kernel_size=1)
        self.lat5 = nn.Conv2d(c5, d, kernel_size=1)

        sdp = float(stage_dropout_p)
        if sdp < 0 or sdp >= 1:
            raise ValueError(f"stage_dropout_p must be in [0,1), got {stage_dropout_p}")
        self.stage_dropout = nn.Dropout2d(p=sdp) if sdp > 0 else nn.Identity()

        # U-Net decoder blocks (upsample + concat skip).
        self.dec4 = _ConvBlock(2 * d, d, dropout_p=float(decoder_dropout_p))
        self.dec3 = _ConvBlock(2 * d, d, dropout_p=float(decoder_dropout_p))
        self.dec2 = _ConvBlock(2 * d, d, dropout_p=float(decoder_dropout_p))

        self.head = nn.Conv2d(d, int(out_channels), kernel_size=1)

        self._deep_sup = bool(deep_sup)
        if self._deep_sup:
            # Auxiliary heads: after dec4 (1/8-scale context) and dec3 (1/4-scale context).
            self.aux_head4 = nn.Conv2d(d, int(out_channels), kernel_size=1)
            self.aux_head3 = nn.Conv2d(d, int(out_channels), kernel_size=1)

        if self._hint_attn:
            # Spatial attention gate: hint (B,1,H,W) → tanh scalar gate → d2 * (1 + gate).
            # Zero-initialized → no-op at start; model learns to amplify/suppress features
            # in regions where Stage1 hints are confident.
            self.hint_attn_conv = nn.Conv2d(1, 1, kernel_size=1, bias=True)
            nn.init.zeros_(self.hint_attn_conv.weight)
            nn.init.zeros_(self.hint_attn_conv.bias)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = x
        feats: dict[int, torch.Tensor] = {}
        for i, blk in enumerate(self.encoder.features):
            out = blk(out)
            if i in (1, 3, 5, 7):
                feats[i] = out
        if not all(k in feats for k in (1, 3, 5, 7)):
            raise RuntimeError("Failed to capture ConvNeXt stage outputs")
        return feats[1], feats[3], feats[5], feats[7]

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        h, w = x.shape[-2], x.shape[-1]

        # hint_attn: split off last channel as Stage1 prob hint before encoding.
        hint: Optional[torch.Tensor] = None
        if self._hint_attn:
            hint = x[:, -1:]   # (B, 1, H, W)
            x    = x[:, :-1]   # (B, C-1, H, W)

        c2, c3, c4, c5 = self._encode(x)

        s2 = self.lat2(c2)  # 1/4
        s3 = self.lat3(c3)  # 1/8
        s4 = self.lat4(c4)  # 1/16
        s5 = self.stage_dropout(self.lat5(c5))  # 1/32 (+ optional dropout for MC sampling)

        d4 = self.dec4(torch.cat([self._up(s5, s4), s4], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, s3), s3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, s2), s2], dim=1))

        # hint_attn: spatial attention gate on finest decoder feature map.
        # gate = 1 + tanh(conv(hint_upsampled)), initialized to no-op (weight=bias=0).
        if hint is not None:
            hint_up = F.interpolate(hint, size=d2.shape[-2:], mode="bilinear", align_corners=False)
            gate = torch.tanh(self.hint_attn_conv(hint_up))  # (B, 1, H2, W2), range [-1, 1]
            d2 = d2 * (1.0 + gate)  # broadcast over all dec_ch channels

        logits = F.interpolate(self.head(d2), size=(h, w), mode="bilinear", align_corners=False)

        if self._deep_sup and self.training:
            # Deep supervision: auxiliary outputs at 1/8 and 1/4 scales, upsampled to input size.
            # Weights applied in training script: [1.0, 0.5, 0.25] (main, aux3, aux4).
            aux3 = F.interpolate(self.aux_head3(d3), size=(h, w), mode="bilinear", align_corners=False)
            aux4 = F.interpolate(self.aux_head4(d4), size=(h, w), mode="bilinear", align_corners=False)
            return [logits, aux3, aux4]

        return logits
