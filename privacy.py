import math
from dataclasses import dataclass
from typing import Dict, Any

import torch

def clip_by_l2_norm_(tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(tensor.float())
    if norm > max_norm:
        tensor.mul_(max_norm / (norm + 1e-12))
    return tensor

def dp_clip_and_noise_(state: Dict[str, torch.Tensor], clip_norm: float, noise_std: float) -> Dict[str, torch.Tensor]:
    # state is a dict of parameter tensors (deltas)
    if clip_norm is not None:
        # clip per-tensor (simple MVP). You may implement per-layer or global clipping later.
        for k, v in state.items():
            if torch.is_tensor(v):
                clip_by_l2_norm_(v, clip_norm)
    if noise_std and noise_std > 0:
        for k, v in state.items():
            if torch.is_tensor(v):
                v.add_(torch.randn_like(v) * noise_std)
    return state

@dataclass
class SecureAggSim:
    """A simulation of secure aggregation.

    In real deployment, server should learn only the sum (or weighted sum) across clients
    without seeing any individual update. Here we simply sum and do not store per-client values.
    """
    sums: Dict[str, Any] = None

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.sums = {}

    def add(self, key: str, value):
        if torch.is_tensor(value):
            value = value.detach()
        if key not in self.sums:
            self.sums[key] = value
        else:
            self.sums[key] = self.sums[key] + value

    def get(self, key: str):
        return self.sums.get(key, None)
