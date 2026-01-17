from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import math
import torch
import torch.nn as nn

from privacy import SecureAggSim
from lora_experts import MultiExpertLoRALinear

@dataclass
class ServerState:
    num_total_experts: int
    active_specialized: List[int]
    pi: torch.Tensor  # (E_total,) on CPU

class Server:
    def __init__(
        self,
        model: nn.Module,
        num_total_experts: int,
        active_specialized_init: List[int],
        lr_server: float,
        birth_patience: int,
        birth_reject_rate: float,
        prune_patience: int,
        prune_util_threshold: float,
        merge_similarity_threshold: float,
    ):
        self.model = model
        self.lr_server = lr_server
        self.state = ServerState(
            num_total_experts=num_total_experts,
            active_specialized=list(active_specialized_init),
            pi=torch.ones(num_total_experts) / float(num_total_experts),
        )
        self.secagg = SecureAggSim()

        # global bank (dense arrays inside each wrapped layer)
        self.global_bank = self._get_bank()

        # open-world trackers
        self.birth_patience = birth_patience
        self.birth_reject_rate = birth_reject_rate
        self.prune_patience = prune_patience
        self.prune_util_threshold = prune_util_threshold
        self.merge_similarity_threshold = merge_similarity_threshold

        self._reject_streak = 0
        self._low_util_streak = {k: 0 for k in range(num_total_experts)}

    def broadcast(self) -> Dict[str, Any]:
        return {
            "bank": self.global_bank,  # already on CPU tensors
            "pi": self.state.pi.clone(),
            "active_specialized": list(self.state.active_specialized),
        }

    def reset_aggregator(self):
        self.secagg.reset()

    def aggregate_client_update(self, payload: Dict[str, Any]):
        # dense responsibilities
        self.secagg.add("S", payload["S"])
        # sparse bank updates
        self.secagg.add_sparse_bank("U_sparse", payload["U_sparse"])
        # open-world aggregated signals
        self.secagg.add("reject", torch.tensor(float(payload.get("reject", 0.0))))
        self.secagg.add("conf", torch.tensor(float(payload.get("conf", 0.0))))

    def step(self, num_clients: int, eps: float = 1e-8):
        S = self.secagg.get("S")          # (E,)
        U_sparse = self.secagg.get("U_sparse")  # layer->pkey->expert->tensor
        reject = float(self.secagg.get("reject").item()) if self.secagg.get("reject") is not None else 0.0

        # ---- M-step: apply sparse updates in-place ----
        for layer_name, sd in self.global_bank.items():
            upd_layer = U_sparse.get(layer_name, None) if U_sparse is not None else None
            if upd_layer is None:
                continue
            for pkey in ["lora_A", "lora_B"]:
                upd = upd_layer.get(pkey, {})
                if not upd:
                    continue
                for k, delta in upd.items():
                    denom = float(max(S[int(k)].item(), eps))
                    sd[pkey][int(k)] = sd[pkey][int(k)] + (delta * (self.lr_server / denom))

        # update pi from S with smoothing
        alpha = 1e-3
        pi = (S + alpha) / (S.sum() + alpha * S.numel())
        self.state.pi = pi.detach().cpu()

        # open-world
        self._open_world_manage(S=S, reject_count=reject, num_clients=num_clients)

    def load_bank_to_model(self):
        # load current global bank into model wrappers (on GPU)
        for name, m in self.model.named_modules():
            if isinstance(m, MultiExpertLoRALinear):
                m.load_expert_state_dict(self.global_bank[name])

    # ---------------- Open-world management ----------------
    def _open_world_manage(self, S: torch.Tensor, reject_count: float, num_clients: int):
        reject_rate = reject_count / max(num_clients, 1)
        if reject_rate >= self.birth_reject_rate:
            self._reject_streak += 1
        else:
            self._reject_streak = 0

        if self._reject_streak >= self.birth_patience:
            if self._birth_expert():
                self._reject_streak = 0

        # prune low-util experts
        for k in list(self.state.active_specialized):
            util_frac = float(S[int(k)].item()) / max(num_clients, 1)
            if util_frac < self.prune_util_threshold:
                self._low_util_streak[k] = self._low_util_streak.get(k, 0) + 1
            else:
                self._low_util_streak[k] = 0
            if self._low_util_streak[k] >= self.prune_patience:
                self._deactivate_expert(k)
                self._low_util_streak[k] = 0

        # merge (at most one per round)
        self._merge_similar_experts(S=S)

    def _inactive_specialized_slots(self) -> List[int]:
        all_spec = set(range(1, self.state.num_total_experts))
        active = set(self.state.active_specialized)
        return sorted(list(all_spec - active))

    def _birth_expert(self) -> bool:
        inactive = self._inactive_specialized_slots()
        if not inactive:
            return False
        new_k = inactive[0]
        noise_std = 0.01
        for layer_name, sd in self.global_bank.items():
            for pkey in ["lora_A", "lora_B"]:
                base = sd[pkey][0].clone()
                sd[pkey][new_k] = base + torch.randn_like(base) * noise_std
        self.state.active_specialized.append(new_k)
        # give it small prior mass
        self.state.pi[new_k] = max(float(self.state.pi.mean().item()), 1e-3)
        self.state.pi = (self.state.pi / self.state.pi.sum()).cpu()
        print(f"[OpenWorld] Birth expert {new_k}. Active specialized: {self.state.active_specialized}")
        return True

    def _deactivate_expert(self, k: int):
        if k in self.state.active_specialized:
            self.state.active_specialized.remove(k)
            self.state.pi[k] = 0.0
            if float(self.state.pi.sum()) > 0:
                self.state.pi = (self.state.pi / self.state.pi.sum()).cpu()
            print(f"[OpenWorld] Deactivate expert {k}. Active specialized: {self.state.active_specialized}")

    def _flatten_expert(self, k: int) -> torch.Tensor:
        vecs = []
        for _, sd in self.global_bank.items():
            vecs.append(sd["lora_A"][k].reshape(-1).float())
            vecs.append(sd["lora_B"][k].reshape(-1).float())
        return torch.cat(vecs, dim=0)

    def _cosine_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        denom = (a.norm() * b.norm() + 1e-12)
        return float((a @ b / denom).item())

    def _merge_similar_experts(self, S: torch.Tensor):
        active = list(self.state.active_specialized)
        if len(active) < 2:
            return
        vec = {k: self._flatten_expert(k) for k in active}
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                k1, k2 = active[i], active[j]
                sim = self._cosine_sim(vec[k1], vec[k2])
                if sim >= self.merge_similarity_threshold:
                    u1 = float(S[k1].item())
                    u2 = float(S[k2].item())
                    keep, drop = (k1, k2) if u1 >= u2 else (k2, k1)
                    w1 = max(float(S[keep].item()), 1e-6)
                    w2 = max(float(S[drop].item()), 1e-6)
                    wsum = w1 + w2
                    for _, sd in self.global_bank.items():
                        for pkey in ["lora_A", "lora_B"]:
                            sd[pkey][keep] = sd[pkey][keep] * (w1 / wsum) + sd[pkey][drop] * (w2 / wsum)
                    self._deactivate_expert(drop)
                    print(f"[OpenWorld] Merge experts ({k1},{k2}) sim={sim:.4f} -> keep={keep}, drop={drop}")
                    return

    # ---------------- Bank IO ----------------
    def _get_bank(self) -> Dict[str, Dict[str, torch.Tensor]]:
        bank = {}
        for name, m in self.model.named_modules():
            if isinstance(m, MultiExpertLoRALinear):
                bank[name] = m.expert_state_dict()
        return bank
