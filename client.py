from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from router import TokenRouter, RoutingContext
from privacy import dp_clip_and_noise_

@dataclass
class ClientConfig:
    tau: float
    gamma: float
    top_m: int
    local_steps: int
    lr_client: float
    weight_decay: float
    lambda_consistency: float
    max_train_batches_per_round: int
    max_probe_batches: int
    use_dp: bool
    dp_clip_norm: float
    dp_noise_std: float

class Client:
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        tokenizer,
        device: torch.device,
        routing_ctx: RoutingContext,
        dataloader,
        cfg: ClientConfig,
        num_total_experts: int,
    ):
        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.routing_ctx = routing_ctx
        self.dataloader = dataloader
        self.cfg = cfg
        self.num_total_experts = num_total_experts

        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            # fallback for some configs
            hidden_size = model.config.hidden_sizes[-1]
        # 1. 初始化 Router 并移动到设备
        self.router = TokenRouter(hidden_size, num_total_experts).to(device)
        
        # 2. 【关键修复】将 Router 的权重转换为与模型相同的精度 (例如 bfloat16)
        # 否则 model 的 hidden_states 是 bf16，而 router 是 fp32，会导致 matmul 报错
        self.router.to(dtype=model.dtype)

    @torch.no_grad()
    def probe_losses(self, bank: Dict[str, Dict[str, torch.Tensor]], candidates: List[int]) -> Dict[int, float]:
        """Compute probe loss F_i(E_k) for each specialized expert k in candidates.

        We always include shared expert 0 in probe, and add candidate expert k.
        Probe uses ctx.mode='probe' and sums LoRA contributions (no token mixing).
        """
        # Load bank into model
        self._load_bank(bank)

        losses = {}
        data_iter = iter(self.dataloader)
        for k in candidates:
            self.routing_ctx.mode = "probe"
            self.routing_ctx.active_experts = torch.tensor([0, k], device=self.device)
            self.routing_ctx.domain_prior = None
            self.routing_ctx.token_router = None

            tot_loss = 0.0
            n = 0
            for _ in range(self.cfg.max_probe_batches):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)
                batch = {kk: vv.to(self.device) for kk, vv in batch.items()}
                out = self.model(**batch)
                tot_loss += float(out.loss.detach().cpu())
                n += 1
            losses[k] = tot_loss / max(n, 1)
        return losses

    def compute_responsibilities(self, probe_losses: Dict[int, float], pi: torch.Tensor) -> torch.Tensor:
        """Return r_{i,k} for all experts including shared expert 0.

        We fix r_{i,0}=1 for shared expert (always active, always updated),
        and compute responsibilities over specialized experts 1..K using probe losses.
        """
        E = self.num_total_experts
        r = torch.zeros(E, dtype=torch.float32)
        r[0] = 1.0

        # Only specialized experts have probe losses
        # Use softmax over candidates with priors pi
        cand = sorted(probe_losses.keys())
        vals = torch.tensor([probe_losses[k] for k in cand], dtype=torch.float32)
        pri = pi[cand].float()
        logits = torch.log(pri + 1e-12) - vals / self.cfg.tau
        probs = torch.softmax(logits, dim=0)

        for idx, k in enumerate(cand):
            r[k] = probs[idx].item()
        return r

    def select_active_experts(self, r: torch.Tensor) -> List[int]:
        # Always include shared expert 0.
        # Select top-(M-1) among specialized experts.
        M = self.cfg.top_m
        if M <= 1:
            return [0]
        specialized = torch.arange(1, self.num_total_experts)
        scores = r[specialized]
        top = torch.topk(scores, k=min(M-1, scores.numel())).indices
        chosen = specialized[top].tolist()
        return [0] + chosen

    def local_train(self, bank: Dict[str, Dict[str, torch.Tensor]], r: torch.Tensor, active: List[int]) -> Dict[str, Any]:
        """Train active experts' LoRA + client router."""
        self._load_bank(bank)

        # Setup routing context
        self.routing_ctx.mode = "hierarchical"
        self.routing_ctx.active_experts = torch.tensor(active, device=self.device)
        self.routing_ctx.domain_prior = r.to(self.device)
        self.routing_ctx.token_router = self.router
        self.routing_ctx.gamma = self.cfg.gamma
        self.routing_ctx.reset_alpha_stats(m=len(active), device=self.device)

        # Collect trainable params
        trainable = list(self.router.parameters())
        for name, m in self.model.named_modules():
            from lora_experts import MultiExpertLoRALinear
            if isinstance(m, MultiExpertLoRALinear):
                # Ensure they require grad
                m.lora_A.requires_grad_(True)
                m.lora_B.requires_grad_(True)
                trainable.append(m.lora_A)
                trainable.append(m.lora_B)

        opt = torch.optim.AdamW(trainable, lr=self.cfg.lr_client, weight_decay=self.cfg.weight_decay)

        # Snapshot initial bank
        init_bank = copy.deepcopy(bank)

        # --- FIX START: Ensure Model Training State & Gradients ---
        self.model.train()
        self.router.train()
        
        # 关键修复 1: 重新激活输入梯度钩子，防止因之前的 eval() 导致 checkpointing 梯度中断
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        
        # 关键修复 2: 关闭 use_cache，因为它与 gradient checkpointing 不兼容
        self.model.config.use_cache = False 
        # --- FIX END ---

        data_iter = iter(self.dataloader)

        steps = 0
        while steps < self.cfg.local_steps:
            for _ in range(self.cfg.max_train_batches_per_round):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                batch = {kk: vv.to(self.device) for kk, vv in batch.items()}
                
                # --- FIX: Explicitly pass use_cache=False to forward ---
                out = self.model(**batch, use_cache=False)
                loss = out.loss

                # Consistency regularization
                mean_alpha = self.routing_ctx.get_mean_alpha()
                if mean_alpha is not None and self.cfg.lambda_consistency > 0:
                    prior_active = r.to(self.device).index_select(0, self.routing_ctx.active_experts)
                    prior_active = prior_active / (prior_active.sum() + 1e-12)
                    kl = (mean_alpha * (torch.log(mean_alpha + 1e-12) - torch.log(prior_active + 1e-12))).sum()
                    loss = loss + self.cfg.lambda_consistency * kl

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                self.routing_ctx.reset_alpha_stats(m=len(active), device=self.device)

            steps += 1

        # Compute delta bank = trained - init
        trained_bank = self._get_current_bank()
        delta_bank = {}
        for layer_name, sd in trained_bank.items():
            delta_bank[layer_name] = {}
            for param_key in ["lora_A", "lora_B"]:
                delta_bank[layer_name][param_key] = (sd[param_key] - init_bank[layer_name][param_key]).cpu()

        # Optional DP: apply to deltas
        if self.cfg.use_dp:
            for layer_name in delta_bank:
                dp_clip_and_noise_(delta_bank[layer_name], self.cfg.dp_clip_norm, self.cfg.dp_noise_std)

        # Prepare responsibility-weighted payload:
        # S: (E_total,) and U: nested dict of same shapes as bank
        # For simplicity, we weight all expert slices in delta_bank by r (broadcasted),
        # even though only active experts were meaningfully updated.
        S = r.clone().cpu()

        U = copy.deepcopy(delta_bank)
        for layer_name, sd in U.items():
            for param_key in ["lora_A", "lora_B"]:
                delta = sd[param_key]  # (E, ...)
                # weight per expert
                shape = delta.shape
                weight = r.reshape((-1,) + (1,) * (len(shape) - 1))
                sd[param_key] = delta * weight

        return {"S": S, "U": U, "active": active, "r": r.cpu()}

    # --- helpers ---
    def _load_bank(self, bank):
        # Load bank tensors into wrappers
        for name, m in self.model.named_modules():
            from lora_experts import MultiExpertLoRALinear
            if isinstance(m, MultiExpertLoRALinear):
                m.load_expert_state_dict(bank[name])

    def _get_current_bank(self):
        bank = {}
        for name, m in self.model.named_modules():
            from lora_experts import MultiExpertLoRALinear
            if isinstance(m, MultiExpertLoRALinear):
                bank[name] = m.expert_state_dict()
        return bank
