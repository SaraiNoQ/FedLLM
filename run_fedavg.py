from __future__ import annotations
import argparse
import os
import math
import torch
from typing import List, Dict, Any

from config import FedVEMoEConfig, DataConfig
from utils.seed import set_seed
from utils.logger import JsonlLogger, load_json

from model import load_base_model_with_multiexpert_lora
from lora_experts import MultiExpertLoRAConfig, MultiExpertLoRALinear
from server import Server
from client import Client, ClientConfig
from data_sni import make_client_task_map, make_sni_client_loaders

# ==========================================================
# FedAvg 专用 Server：取消专家管理，执行简单权重平均
# ==========================================================
class FedAvgServer(Server):
    def step(self, num_clients: int, eps: float = 1e-8):
        # S 在 FedAvg 模式下对于 Expert 0 应该等于参与的客户端数
        U_sparse = self.secagg.get("U_sparse") 
        
        # 标准 FedAvg：对所有层的 Expert 0 (Shared) 进行平均
        for layer_name, sd in self.global_bank.items():
            upd_layer = U_sparse.get(layer_name, None)
            if upd_layer is None: continue
            
            for pkey in ["lora_A", "lora_B"]:
                upd = upd_layer.get(pkey, {})
                if 0 in upd:
                    # 简单平均：(所有客户端梯度之和) / 客户端数
                    delta = upd[0]
                    sd[pkey][0] = sd[pkey][0] + (delta * (self.lr_server / num_clients))
        
        # FedAvg 不需要更新 pi 和 open-world 管理

# ==========================================================
# FedAvg 专用 Client：强制只使用 Expert 0，不训练 Router
# ==========================================================
class FedAvgClient(Client):
    # 重写训练方法，确保状态重置
    def local_train(self, bank, r, active, train_steps=None):
        self._load_bank(bank)
        
        # FedAvg 关键：设置为 probe 模式，仅激活专家 0
        self.routing_ctx.mode = "probe" 
        self.routing_ctx.active_experts = torch.tensor([0], device=self.device)
        self.routing_ctx.domain_prior = None
        self.routing_ctx.token_router = None
        
        # 即使在 probe 模式下，为了保险，也可以手动初始化一下统计变量（防止其他地方调用）
        self.routing_ctx.reset_alpha_stats(m=1, device=self.device)

        trainable = []
        for _, m in self.model.named_modules():
            if isinstance(m, MultiExpertLoRALinear):
                m.lora_A.requires_grad_(True)
                m.lora_B.requires_grad_(True)
                trainable.append(m.lora_A)
                trainable.append(m.lora_B)

        opt = torch.optim.AdamW(trainable, lr=self.cfg.lr_client, weight_decay=self.cfg.weight_decay)
        init_slices = self._get_active_slices([0])

        self.model.train()
        data_iter = iter(self.train_loader)
        steps = train_steps if train_steps is not None else self.cfg.local_steps
        
        for _ in range(steps):
            for _ in range(self.cfg.max_train_batches_per_round):
                try: batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader); batch = next(data_iter)
                
                batch = {kk: vv.to(self.device) if torch.is_tensor(vv) else vv for kk, vv in batch.items()}
                out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                
                opt.zero_grad()
                out.loss.backward()
                opt.step()

        cur_slices = self._get_active_slices([0])
        delta_sparse = {layer: {"lora_A": {0: (sd["lora_A"][0] - init_slices[layer]["lora_A"][0]).cpu()},
                               "lora_B": {0: (sd["lora_B"][0] - init_slices[layer]["lora_B"][0]).cpu()}}
                       for layer, sd in cur_slices.items()}

        return {"S": torch.tensor([1.0] + [0.0]*(self.num_total_experts-1)), "U_sparse": delta_sparse}

    # --- 新增：重写评估方法，强制使用 probe 模式以避开 alpha_means_sum 报错 ---
    def evaluate_generation(self, bank, pi, active_specialized, split="val", max_examples=None):
        # 先调用父类的逻辑获取结果，但在调用前强制修改模式
        self._load_bank(bank)
        self.routing_ctx.mode = "probe"
        self.routing_ctx.active_experts = torch.tensor([0], device=self.device)
        self.routing_ctx.domain_prior = None
        self.routing_ctx.token_router = None
        
        # 这种模式下，lora_experts.py 会直接跳过 add_alpha_stats 调用
        return super().evaluate_generation(bank, pi, active_specialized, split, max_examples)

    def evaluate_ppl(self, bank, pi, active_specialized, split="val", max_batches=None):
        self._load_bank(bank)
        self.routing_ctx.mode = "probe"
        self.routing_ctx.active_experts = torch.tensor([0], device=self.device)
        self.routing_ctx.domain_prior = None
        self.routing_ctx.token_router = None
        return super().evaluate_ppl(bank, pi, active_specialized, split, max_batches)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--log_dir", type=str, default="runs/fedavg_baseline")
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--keyword_categories", nargs="+", default=["translation","reasoning","code","rewriting"])
    parser.add_argument("--tasks_per_category", type=int, default=3)
    args = parser.parse_args()

    set_seed(42)
    # 强制设置 k_max=0，只保留一个 Shared Expert
    cfg = FedVEMoEConfig(num_rounds=args.num_rounds, num_clients=args.num_clients, k_max=0)
    logger = JsonlLogger(log_dir=args.log_dir)

    client_task_map = make_client_task_map(args.num_clients, "./natural-instructions", "keyword", args.keyword_categories, args.tasks_per_category)
    
    # 加载模型（只有 1 个专家）
    loaded = load_base_model_with_multiexpert_lora(args.model_name, 1, MultiExpertLoRAConfig(), cfg.target_linear_keywords)
    model, tokenizer, device, routing_ctx = loaded.model, loaded.tokenizer, loaded.device, loaded.routing_ctx

    loaders = make_sni_client_loaders(tokenizer, "./natural-instructions", client_task_map, cfg.seq_len, cfg.micro_batch)

    server = FedAvgServer(model, 1, [], cfg.lr_server, 0, 0, 0, 0, 0)
    
    clients = []
    c_cfg = ClientConfig(tau=0.5, gamma=1.0, top_m=1, local_steps=cfg.local_steps, lr_client=cfg.lr_client, 
                         weight_decay=0.0, lambda_consistency=0.0, max_train_batches_per_round=cfg.max_train_batches_per_round,
                         max_probe_batches=1, max_eval_batches=16, max_gen_eval_examples=32, gen_max_new_tokens=128, 
                         reject_conf_threshold=0.0, use_dp=False, dp_clip_norm=1.0, dp_noise_std=0.0)

    for cid in range(args.num_clients):
        tr, va, te = loaders[cid]
        clients.append(FedAvgClient(cid, model, tokenizer, device, routing_ctx, tr, va, te, c_cfg, 1))

    for rnd in range(args.num_rounds):
        print(f"\n=== FedAvg Round {rnd+1}/{args.num_rounds} ===")
        payload = server.broadcast()
        server.reset_aggregator()

        for client in clients:
            upd = client.local_train(payload["bank"], None, [0])
            server.aggregate_client_update(upd)

        server.step(num_clients=args.num_clients)
        server.load_bank_to_model()

        # 每轮评估
        for client in clients:
            gen = client.evaluate_generation(server.global_bank, torch.tensor([1.0]), [], split="train")
            ppl = client.evaluate_ppl(server.global_bank, torch.tensor([1.0]), [], split="train")
            logger.log({"round": rnd, "client_id": client.client_id, "rougeL": gen["rougeL"], "em": gen["em"], "ppl": ppl["ppl"]})
            print(f"[FedAvg c{client.client_id}] ROUGE-L={gen['rougeL']:.3f} EM={gen['em']:.3f} PPL={ppl['ppl']:.2f}")

if __name__ == "__main__":
    main()