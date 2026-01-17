from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import copy
import math

import torch
import torch.nn as nn

from router import TokenRouter, RoutingContext
from privacy import dp_clip_and_noise_
from eval_metrics import rouge_l_f1, exact_match, aggregate_metric

from lora_experts import MultiExpertLoRALinear

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
    max_eval_batches: int
    max_gen_eval_examples: int
    gen_max_new_tokens: int

    # open-world client-side signal
    reject_conf_threshold: float

    # privacy (optional)
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
        train_loader,
        val_loader,
        test_loader,
        cfg: ClientConfig,
        num_total_experts: int,
    ):
        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.routing_ctx = routing_ctx

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.cfg = cfg
        self.num_total_experts = num_total_experts

        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None and hasattr(model.config, "hidden_sizes"):
            hidden_size = model.config.hidden_sizes[-1]
        assert hidden_size is not None, "Cannot infer hidden size from model.config"
        self.router = TokenRouter(hidden_size, num_total_experts).to(device)
        try:
            model_dtype = next(model.parameters()).dtype
            self.router = self.router.to(dtype=model_dtype)
        except StopIteration:
            pass

    # ---------------- Routing: Probe + Responsibilities ----------------
    @torch.no_grad()
    def probe_losses(self, bank: Dict[str, Dict[str, torch.Tensor]], candidates: List[int]) -> Dict[int, float]:
        """Estimate probe loss for each candidate specialized expert k.

        Probe uses ctx.mode='probe' and active experts [0,k] with additive LoRA (no token mixing).
        """
        self._load_bank(bank)

        losses = {}
        data_iter = iter(self.train_loader)
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
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                batch = {kk: vv.to(self.device) if torch.is_tensor(vv) else vv for kk, vv in batch.items()}
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss_val = float(out.loss.detach().cpu())
                if not math.isfinite(loss_val):
                    print(f"[WARN][Client {self.client_id}] probe loss non-finite for expert {k}: {loss_val}")
                    loss_val = 1e6
                tot_loss += loss_val
                n += 1
            losses[k] = tot_loss / max(n, 1)
        return losses

    def compute_responsibilities(self, probe_losses: Dict[int, float], pi: torch.Tensor) -> torch.Tensor:
        """Return dense r over all expert slots (E_total). Shared expert r[0]=1."""
        E = self.num_total_experts
        r = torch.zeros(E, dtype=torch.float32)
        r[0] = 1.0

        cand = sorted(probe_losses.keys())
        vals = torch.tensor([probe_losses[k] for k in cand], dtype=torch.float32)
        pri = pi[cand].float()
        if not torch.isfinite(vals).all() or not torch.isfinite(pri).all():
            print(
                f"[WARN][Client {self.client_id}] non-finite probe/pi: "
                f"vals_finite={bool(torch.isfinite(vals).all())} "
                f"pi_finite={bool(torch.isfinite(pri).all())} "
                f"pi_sum={float(pi.sum().item())}"
            )
            vals = torch.nan_to_num(vals, nan=1e6, posinf=1e6, neginf=1e6)
            pri = torch.nan_to_num(pri, nan=0.0, posinf=0.0, neginf=0.0)
            if float(pri.sum().item()) <= 0:
                pri = torch.ones_like(pri) / max(pri.numel(), 1)
        logits = torch.log(pri + 1e-12) - vals / self.cfg.tau
        probs = torch.softmax(logits, dim=0)
        if not torch.isfinite(probs).all():
            print(f"[WARN][Client {self.client_id}] non-finite probs; using uniform.")
            probs = torch.ones_like(probs) / max(probs.numel(), 1)
        for idx, k in enumerate(cand):
            r[k] = float(probs[idx].item())
        return r

    def select_active_experts(self, r: torch.Tensor, active_specialized: List[int]) -> List[int]:
        # Always include shared expert 0.
        M = self.cfg.top_m
        if M <= 1 or len(active_specialized) == 0:
            return [0]
        spec = torch.tensor(active_specialized, dtype=torch.long)
        scores = r.index_select(0, spec)
        top = torch.topk(scores, k=min(M - 1, scores.numel())).indices
        chosen = spec[top].tolist()
        return [0] + chosen

    def compute_openworld_signals(self, r: torch.Tensor, active_specialized: List[int]) -> Tuple[float, float]:
        """Return (reject_bit, conf) as scalar floats.

        conf = max_k r_{i,k} over active specialized experts.
        reject = 1 if conf < threshold.
        """
        if not active_specialized:
            return 1.0, 0.0
        spec = torch.tensor(active_specialized, dtype=torch.long)
        conf = float(r.index_select(0, spec).max().item())
        if not math.isfinite(conf):
            print(f"[WARN][Client {self.client_id}] non-finite conf from r: {r}")
            conf = 0.0
        reject = 1.0 if conf < self.cfg.reject_conf_threshold else 0.0
        return reject, conf

    # ---------------- Local Training ----------------
    def local_train(
        self,
        bank: Dict[str, Dict[str, torch.Tensor]],
        r: torch.Tensor,
        active: List[int],
        train_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train active experts' LoRA + client router, return payload with sparse updates."""
        self._load_bank(bank)

        self.routing_ctx.mode = "hierarchical"
        self.routing_ctx.active_experts = torch.tensor(active, device=self.device)
        self.routing_ctx.domain_prior = r.to(self.device)
        self.routing_ctx.token_router = self.router
        self.routing_ctx.gamma = self.cfg.gamma
        self.routing_ctx.reset_alpha_stats(m=len(active), device=self.device)

        # trainable params: router + LoRA params
        trainable = list(self.router.parameters())
        for _, m in self.model.named_modules():
            if isinstance(m, MultiExpertLoRALinear):
                m.lora_A.requires_grad_(True)
                m.lora_B.requires_grad_(True)
                trainable.append(m.lora_A)
                trainable.append(m.lora_B)

        opt = torch.optim.AdamW(trainable, lr=self.cfg.lr_client, weight_decay=self.cfg.weight_decay)

        # snapshot active slices only (sparse)
        active_ids = sorted(set(active))
        init_slices = self._get_active_slices(active_ids)

        data_iter = iter(self.train_loader)
        self.model.train()
        self.router.train()

        steps = train_steps if train_steps is not None else self.cfg.local_steps
        for _ in range(steps):
            for _ in range(self.cfg.max_train_batches_per_round):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                batch = {kk: vv.to(self.device) if torch.is_tensor(vv) else vv for kk, vv in batch.items()}

                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = out.loss

                # consistency KL(mean_alpha || prior_active)
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

        # compute deltas for active slices only
        cur_slices = self._get_active_slices(active_ids)
        delta_sparse = {}
        for layer, sd in cur_slices.items():
            delta_sparse[layer] = {"lora_A": {}, "lora_B": {}}
            for pkey in ["lora_A", "lora_B"]:
                for k in active_ids:
                    delta = (sd[pkey][k] - init_slices[layer][pkey][k]).cpu()
                    # optional DP
                    if self.cfg.use_dp:
                        dp_clip_and_noise_({"d": delta}, self.cfg.dp_clip_norm, self.cfg.dp_noise_std)
                        delta = delta
                    # responsibility-weight
                    w = float(r[k].item())
                    delta_sparse[layer][pkey][k] = delta * w

        # S is dense responsibilities over all expert slots
        S = r.clone().cpu()

        return {
            "S": S,
            "U_sparse": delta_sparse,
        }

    # ---------------- Evaluation ----------------
    @torch.no_grad()
    def evaluate_ppl(
        self,
        bank: Dict[str, Dict[str, torch.Tensor]],
        pi: torch.Tensor,
        active_specialized: List[int],
        split: str = "val",
        max_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute loss and perplexity under hierarchical routing for this client."""
        self._load_bank(bank)

        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        elif split == "test":
            loader = self.test_loader
        else:
            raise ValueError(f"Unknown split: {split}")

        # responsibilities and active set (use probe on train for stability)
        probe = self.probe_losses(bank, candidates=active_specialized)
        r = self.compute_responsibilities(probe, pi=pi)
        active = self.select_active_experts(r, active_specialized)

        self.routing_ctx.mode = "hierarchical"
        self.routing_ctx.active_experts = torch.tensor(active, device=self.device)
        self.routing_ctx.domain_prior = r.to(self.device)
        self.routing_ctx.token_router = self.router
        self.routing_ctx.gamma = self.cfg.gamma

        self.model.eval()
        self.router.eval()

        tot_loss = 0.0
        n = 0
        max_batches = max_batches if max_batches is not None else self.cfg.max_eval_batches

        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            batch = {kk: vv.to(self.device) if torch.is_tensor(vv) else vv for kk, vv in batch.items()}
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            tot_loss += float(out.loss.detach().cpu())
            n += 1

        avg_loss = tot_loss / max(n, 1)
        ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
        return {"loss": avg_loss, "ppl": ppl, "active": active}

    @torch.no_grad()
    def evaluate_generation(
        self,
        bank: Dict[str, Dict[str, torch.Tensor]],
        pi: torch.Tensor,
        active_specialized: List[int],
        split: str = "val",
        max_examples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate responses and compute ROUGE-L and EM."""
        self._load_bank(bank)
        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        elif split == "test":
            loader = self.test_loader
        else:
            raise ValueError(f"Unknown split: {split}")

        probe = self.probe_losses(bank, candidates=active_specialized)
        r = self.compute_responsibilities(probe, pi=pi)
        active = self.select_active_experts(r, active_specialized)

        self.routing_ctx.mode = "hierarchical"
        self.routing_ctx.active_experts = torch.tensor(active, device=self.device)
        self.routing_ctx.domain_prior = r.to(self.device)
        self.routing_ctx.token_router = self.router
        self.routing_ctx.gamma = self.cfg.gamma

        self.model.eval()
        self.router.eval()

        max_examples = max_examples if max_examples is not None else self.cfg.max_gen_eval_examples
        preds, refs = [], []
        rouges, ems = [], []

        seen = 0
        for batch in loader:
            bsz = batch["input_ids"].shape[0]
            for i in range(bsz):
                if seen >= max_examples:
                    break
                prompt = batch["prompt_text"][i]
                ref = batch["target_text"][i]
                enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.cfg.gen_max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                # decode only generated continuation
                gen_ids = gen[0].tolist()
                prompt_len = enc["input_ids"].shape[1]
                cont_ids = gen_ids[prompt_len:]
                pred = self.tokenizer.decode(cont_ids, skip_special_tokens=True)

                rouges.append(rouge_l_f1(pred, ref))
                ems.append(exact_match(pred, ref))
                preds.append(pred)
                refs.append(ref)
                seen += 1
            if seen >= max_examples:
                break

        return {
            "rougeL": aggregate_metric(rouges),
            "em": aggregate_metric(ems),
            "active": active,
            "n": seen,
        }

    # ---------------- Helpers: bank IO / sparse slicing ----------------
    def _load_bank(self, bank):
        for name, m in self.model.named_modules():
            if isinstance(m, MultiExpertLoRALinear):
                m.load_expert_state_dict(bank[name])

    def _get_active_slices(self, active_ids: List[int]) -> Dict[str, Dict[str, Dict[int, torch.Tensor]]]:
        """Return a sparse snapshot of LoRA tensors for given expert ids."""
        snap = {}
        for name, m in self.model.named_modules():
            if isinstance(m, MultiExpertLoRALinear):
                sd = m.expert_state_dict()
                snap[name] = {"lora_A": {}, "lora_B": {}}
                for k in active_ids:
                    snap[name]["lora_A"][k] = sd["lora_A"][k].clone()
                    snap[name]["lora_B"][k] = sd["lora_B"][k].clone()
        return snap
