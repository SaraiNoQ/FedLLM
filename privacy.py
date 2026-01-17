from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch

def clip_by_l2_norm_(tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(tensor.float())
    if norm > max_norm:
        tensor.mul_(max_norm / (norm + 1e-12))
    return tensor

def dp_clip_and_noise_(state: Dict[str, torch.Tensor], clip_norm: float, noise_std: float) -> Dict[str, torch.Tensor]:
    for k, v in state.items():
        if torch.is_tensor(v):
            clip_by_l2_norm_(v, clip_norm)
            if noise_std and noise_std > 0:
                v.add_(torch.randn_like(v) * noise_std)
    return state

@dataclass
class SecureAggSim:
    """Secure aggregation simulation (server only sees sums).

    Supports:
    - add(key, tensor)
    - add_sparse_bank(key, sparse_update)
      where sparse_update is:
        layer_name -> pkey -> expert_id -> tensor
    """
    sums: Optional[Dict[str, Any]] = None

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

    def add_sparse_bank(self, key: str, upd: Dict[str, Dict[str, Dict[int, torch.Tensor]]]):
        if key not in self.sums:
            self.sums[key] = {}
        cur = self.sums[key]
        for layer, sd in upd.items():
            if layer not in cur:
                cur[layer] = {"lora_A": {}, "lora_B": {}}
            for pkey in ["lora_A", "lora_B"]:
                for k, v in sd.get(pkey, {}).items():
                    k = int(k)
                    if k not in cur[layer][pkey]:
                        cur[layer][pkey][k] = v.detach().clone()
                    else:
                        cur[layer][pkey][k] = cur[layer][pkey][k] + v.detach()
        self.sums[key] = cur

    def get(self, key: str):
        return self.sums.get(key, None)
