from __future__ import annotations

import math

import torch


def _as_2tuple(x: int | tuple[int, int] | tuple[int, ...]) -> tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    t = tuple(int(v) for v in x)
    if len(t) == 1:
        return (t[0], t[0])
    return (t[0], t[1])


def adapt_first_conv(
    old: torch.nn.Conv2d,
    in_channels: int,
    *,
    init_mode: str = "repeat",
) -> torch.nn.Conv2d:
    """Create a new Conv2d with `in_channels`, adapting weights from `old`.

    This is mainly used to adapt ImageNet pretrained backbones (3ch) to
    multi-channel medical inputs (e.g., windows*stack_slices).

    `init_mode`:
      - "repeat": repeat original channel weights and scale by old_c/new_c
                  (timm-style; preserves activation magnitude better)
      - "mean":   average across input channels then repeat and scale
    """

    new_c = int(in_channels)
    if new_c <= 0:
        raise ValueError("in_channels must be >= 1")

    new = torch.nn.Conv2d(
        new_c,
        old.out_channels,
        kernel_size=_as_2tuple(old.kernel_size),
        stride=_as_2tuple(old.stride),
        padding=old.padding if isinstance(old.padding, str) else _as_2tuple(old.padding),
        dilation=_as_2tuple(old.dilation),
        groups=old.groups,
        bias=(old.bias is not None),
        padding_mode=old.padding_mode,
    )

    if old.bias is not None and new.bias is not None:
        with torch.no_grad():
            new.bias.copy_(old.bias)

    # Weight adaptation (only if old has a valid weight tensor)
    with torch.no_grad():
        w = old.weight
        old_c = int(w.shape[1])
        if old_c == new_c:
            new.weight.copy_(w)
            return new

        if init_mode not in {"repeat", "mean"}:
            raise ValueError("init_mode must be 'repeat' or 'mean'")

        if init_mode == "mean":
            w_mean = w.mean(dim=1, keepdim=True)  # (out,1,kh,kw)
            w_new = w_mean.repeat(1, new_c, 1, 1)
        else:
            rep = int(math.ceil(new_c / old_c))
            w_new = w.repeat(1, rep, 1, 1)[:, :new_c, :, :]

        # Scale to keep output activation magnitude roughly stable.
        w_new = w_new * (float(old_c) / float(new_c))
        new.weight.copy_(w_new)

    return new
