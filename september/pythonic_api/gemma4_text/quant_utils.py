
import torch


def clip_only_fake_quant(x: torch.Tensor, scale: torch.Tensor, bw: int) -> torch.Tensor:
    scale = scale.squeeze()
    min_val = (-(2 ** (bw - 1))) * scale
    max_val = ((2 ** (bw - 1)) - 1) * scale
    return torch.clamp(x, min_val, max_val)
