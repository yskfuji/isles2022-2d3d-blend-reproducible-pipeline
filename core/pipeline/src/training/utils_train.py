import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_device():
    override = os.environ.get("TORCH_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(self.count, 1)

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n


def dice_from_logits(logits, targets, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    probs_flat = torch.flatten(probs, start_dim=1)
    targets_flat = torch.flatten(targets, start_dim=1)
    inter = (probs_flat * targets_flat).sum(1)
    den = probs_flat.sum(1) + targets_flat.sum(1)
    dice = (2 * inter + eps) / (den + eps)
    return dice.mean().item()
