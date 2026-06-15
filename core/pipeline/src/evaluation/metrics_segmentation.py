import numpy as np

def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    inter = (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return float((2 * inter + eps) / den)
