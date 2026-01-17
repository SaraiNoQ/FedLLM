from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from router import RoutingContext

@dataclass
class MultiExpertLoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05

class MultiExpertLoRALinear(nn.Module):
    """Wraps a nn.Linear with multiple LoRA experts."""
    def __init__(
        self,
        base: nn.Linear,
        num_total_experts: int,
        cfg: MultiExpertLoRAConfig,
        routing_getter: Callable[[], RoutingContext],
    ):
        super().__init__()
        # assert isinstance(base, nn.Linear) # 4bit linear might not strictly be nn.Linear depending on version
        self.base = base
        self.num_total_experts = num_total_experts
        self.cfg = cfg
        self.routing_getter = routing_getter

        in_features = base.in_features
        out_features = base.out_features
        r = cfg.rank

        # Freeze base weights
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # --- FIX START: Detect correct device and dtype ---
        device = base.weight.device
        # For 4bit models, base weight dtype is weird. Use bf16/fp16 for LoRA params.
        dtype = torch.float32
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Initialize LoRA params directly on the correct device
        self.lora_A = nn.Parameter(torch.zeros((num_total_experts, r, in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((num_total_experts, out_features, r), device=device, dtype=dtype))
        # --- FIX END ---

        # init
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(cfg.dropout)
        self.scaling = cfg.alpha / float(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,H) or (B,H)
        ctx = self.routing_getter()
        y = self.base(x)

        # Ensure 3D for simplicity
        orig_shape = x.shape
        if x.dim() == 2:
            x3 = x.unsqueeze(1)
            y = y.unsqueeze(1)
        else:
            x3 = x

        B, T, H = x3.shape
        active = ctx.active_experts.to(x3.device)

        # Compute LoRA contributions for active experts only
        # x3 -> (B*T, H)
        x2 = self.dropout(x3).reshape(B*T, H)
        
        # [Safety Cast] Ensure input to LoRA matches LoRA params dtype (e.g. bf16)
        x2 = x2.to(self.lora_A.dtype)

        # gather A,B for active experts
        if self.lora_A.device != active.device:
             A = self.lora_A.to(active.device).index_select(0, active)
             Bm = self.lora_B.to(active.device).index_select(0, active)
        else:
             A = self.lora_A.index_select(0, active)
             Bm = self.lora_B.index_select(0, active)

        # compute per expert
        contribs = []
        for m in range(A.shape[0]):
            z = F.linear(x2, A[m])               
            dz = F.linear(z, Bm[m]) * self.scaling 
            contribs.append(dz.reshape(B, T, -1))
        
        stacked = torch.stack(contribs, dim=2)

        # [Safety Cast] Cast result back to base model output dtype before adding
        stacked = stacked.to(y.dtype)

        if ctx.mode == "probe" or ctx.domain_prior is None or ctx.token_router is None:
            y3 = y + stacked.sum(dim=2)
        else:
            prior = ctx.domain_prior.to(x3.device).index_select(0, active)
            
            # x3 here is likely bf16. 
            # Because we fixed client.py, ctx.token_router is now bf16.
            # This line caused your error, now fixed:
            logits_full = ctx.token_router(x3) 
            
            logits = logits_full.index_select(-1, active)

            alpha = torch.softmax(torch.log(prior + 1e-12) + logits / ctx.gamma, dim=-1)
            ctx.add_alpha_stats(alpha)
            
            # Cast alpha to match stacked (bf16) to avoid float32 * bfloat16 warnings/errors
            alpha = alpha.to(stacked.dtype)

            y3 = y + (stacked * alpha.unsqueeze(-1)).sum(dim=2)

        if len(orig_shape) == 2:
            return y3.squeeze(1)
        return y3

    def expert_state_dict(self) -> Dict[str, torch.Tensor]:
        return {"lora_A": self.lora_A.detach().cpu(), "lora_B": self.lora_B.detach().cpu()}

    def load_expert_state_dict(self, state: Dict[str, torch.Tensor]):
        with torch.no_grad():
            self.lora_A.copy_(state["lora_A"].to(self.lora_A.device))
            self.lora_B.copy_(state["lora_B"].to(self.lora_B.device))

def iter_named_linears(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module
        # Support bitsandbytes Linear4bit / Linear8bit if class name matches
        elif "Linear" in module.__class__.__name__: 
            yield name, module

def inject_multi_expert_lora(
    model: nn.Module,
    num_total_experts: int,
    cfg: MultiExpertLoRAConfig,
    target_keywords: Tuple[str, ...],
    routing_getter: Callable[[], RoutingContext],
) -> List[str]:
    replaced = []
    # We need parent references to replace modules
    for name, module in list(model.named_modules()):
        # Check simplified logic for linear-like modules (incl 4bit)
        is_linear = isinstance(module, nn.Linear) or "Linear" in module.__class__.__name__
        if not is_linear:
            continue
        if not any(k in name for k in target_keywords):
            continue
        # Avoid double wrapping
        if isinstance(module, MultiExpertLoRALinear):
            continue

        # find parent
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child_name = parts[-1]

        wrapped = MultiExpertLoRALinear(module, num_total_experts, cfg, routing_getter)
        setattr(parent, child_name, wrapped)
        replaced.append(name)
    return replaced

def collect_lora_params(model: nn.Module) -> List[nn.Parameter]:
    params = []
    for m in model.modules():
        if isinstance(m, MultiExpertLoRALinear):
            params.append(m.lora_A)
            params.append(m.lora_B)
    return params

def get_expert_bank_state(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """Collect LoRA expert tensors for all wrapped layers."""
    bank = {}
    for name, m in model.named_modules():
        if isinstance(m, MultiExpertLoRALinear):
            bank[name] = m.expert_state_dict()
    return bank

def load_expert_bank_state(model: nn.Module, bank: Dict[str, Dict[str, torch.Tensor]]):
    for name, m in model.named_modules():
        if isinstance(m, MultiExpertLoRALinear):
            m.load_expert_state_dict(bank[name])
