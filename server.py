from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import copy
import torch
import torch.nn as nn

from privacy import SecureAggSim

@dataclass
class ServerState:
    num_total_experts: int  # includes shared expert 0
    pi: torch.Tensor        # (E_total,) expert prior on CPU

class Server:
    def __init__(self, model: nn.Module, num_total_experts: int, lr_server: float = 1.0):
        self.model = model
        self.num_total_experts = num_total_experts
        self.lr_server = lr_server

        # expert prior
        self.state = ServerState(
            num_total_experts=num_total_experts,
            pi=torch.ones(num_total_experts) / float(num_total_experts),
        )

        self.secagg = SecureAggSim()

        # Snapshot expert bank state (global LoRA params inside wrappers)
        self.global_bank = self._get_bank()

    def broadcast(self) -> Dict[str, Any]:
        # In real FL, you'd send only needed tensors and metadata.
        return {
            "bank": copy.deepcopy(self.global_bank),
            "pi": self.state.pi.clone(),
        }

    def reset_aggregator(self):
        self.secagg.reset()

    def aggregate_client_update(self, client_payload: Dict[str, Any]):
        # client_payload contains per-expert weighted deltas and weights
        # keys: "S" (E_total,), "U" (bank delta dict keyed by layer name)
        S = client_payload["S"]  # (E_total,)
        self.secagg.add("S", S)

        # U is nested dict: layer_name -> {"lora_A": tensor, "lora_B": tensor}, already responsibility-weighted
        U = client_payload["U"]
        if self.secagg.get("U") is None:
            self.secagg.add("U", U)
        else:
            # element-wise add nested dicts
            cur = self.secagg.get("U")
            for layer_name, sd in U.items():
                for k, v in sd.items():
                    cur[layer_name][k] = cur[layer_name][k] + v
            self.secagg.sums["U"] = cur

    def step(self, eps: float = 1e-8):
        # Get aggregated sums
        S = self.secagg.get("S")  # (E_total,)
        U = self.secagg.get("U")  # nested dict

        # Update global bank: for each wrapped layer, update LoRA tensors for all experts
        # bank structure: layer_name -> {"lora_A": (E,r,in), "lora_B": (E,out,r)}
        new_bank = copy.deepcopy(self.global_bank)
        for layer_name, sd in new_bank.items():
            # scale per expert by 1/max(S_k, eps)
            # Here we apply: W_k += lr * U_k / S_k
            # Since U is already sum_i r_{i,k} * delta_{i,k}
            # we divide each expert slice.
            for param_key in ["lora_A", "lora_B"]:
                delta = U[layer_name][param_key]  # same shape as param
                # Broadcast S to match
                shape = delta.shape
                # S: (E,), delta: (E, ...)
                scale = (self.lr_server / torch.clamp(S, min=eps)).reshape((-1,) + (1,) * (len(shape) - 1))
                new_bank[layer_name][param_key] = sd[param_key] + delta * scale

        self.global_bank = new_bank

        # (Optional) Open-world expert management stubs
        # - birth/prune/merge can be done based on utilization S and similarity between experts.
        # self._open_world_manage(S)

    def _get_bank(self) -> Dict[str, Dict[str, torch.Tensor]]:
        # Collect from wrappers by reading current model state dict slices (kept inside wrapper modules)
        bank = {}
        for name, m in self.model.named_modules():
            # Lazy import to avoid circular import
            from lora_experts import MultiExpertLoRALinear
            if isinstance(m, MultiExpertLoRALinear):
                bank[name] = m.expert_state_dict()
        return bank

    def load_bank_to_model(self):
        # Load self.global_bank into current model wrappers
        for name, m in self.model.named_modules():
            from lora_experts import MultiExpertLoRALinear
            if isinstance(m, MultiExpertLoRALinear):
                m.load_expert_state_dict(self.global_bank[name])
