from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

class TokenRouter(nn.Module):
    """Client-private token router.

    Given hidden states h (B, T, H), output logits u (B, T, E) for E experts.
    """
    def __init__(self, hidden_size: int, num_total_experts: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_total_experts, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)

@dataclass
class RoutingContext:
    """Holds routing-related info for forward passes."""
    mode: str  # "probe" or "hierarchical"
    active_experts: torch.Tensor  # (M,) expert ids on device
    domain_prior: Optional[torch.Tensor]  # (E_total,) or None
    token_router: Optional[nn.Module]  # TokenRouter or None
    gamma: float = 1.0

    # for logging / consistency regularization
    alpha_means_sum: Optional[torch.Tensor] = None
    alpha_means_count: int = 0

    def reset_alpha_stats(self, m: int, device: torch.device):
        self.alpha_means_sum = torch.zeros((m,), device=device)
        self.alpha_means_count = 0

    def add_alpha_stats(self, alpha: torch.Tensor):
        # alpha: (B,T,M) within active experts
        with torch.no_grad():
            mean = alpha.mean(dim=(0,1))  # (M,)
            self.alpha_means_sum += mean
            self.alpha_means_count += 1

    def get_mean_alpha(self):
        if self.alpha_means_count == 0:
            return None
        return self.alpha_means_sum / float(self.alpha_means_count)
