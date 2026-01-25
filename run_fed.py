from __future__ import annotations
import argparse
import os
from typing import List, Dict, Any
import math

import torch

from config import FedVEMoEConfig, DataConfig
from utils.seed import set_seed
from utils.logger import JsonlLogger, load_json

from model import load_base_model_with_multiexpert_lora
from lora_experts import MultiExpertLoRAConfig
from server import Server
from client import Client, ClientConfig
from data_sni import make_client_task_map, make_sni_client_loaders

def parse_args():
    p = argparse.ArgumentParser()
    # model
    p.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--use_4bit", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=768)

    # dataset
    p.add_argument("--dataset_name", type=str, default="Muennighoff/natural-instructions")
    p.add_argument("--streaming", type=int, default=1)
    p.add_argument("--num_clients", type=int, default=6)
    p.add_argument("--client_task_mode", type=str, default="keyword", choices=["keyword","explicit"])
    p.add_argument("--client_task_json", type=str, default="")
    p.add_argument("--keyword_categories", nargs="+", default=["translation","reasoning","code","rewriting"])
    p.add_argument("--tasks_per_category", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    # federation
    p.add_argument("--num_rounds", type=int, default=3)
    p.add_argument("--k_init", type=int, default=4)
    p.add_argument("--k_max", type=int, default=12)
    p.add_argument("--top_m", type=int, default=2)

    # eval/log
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--eval_split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--log_dir", type=str, default="runs/exp1")
    p.add_argument("--local_baseline_json", type=str, default="")

    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = FedVEMoEConfig(
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        k_init=args.k_init,
        k_max=args.k_max,
        top_m=args.top_m,
        use_4bit=bool(args.use_4bit),
        seq_len=args.seq_len,
    )
    dcfg = DataConfig(
        dataset_name=args.dataset_name,
        streaming=bool(args.streaming),
        client_task_mode=args.client_task_mode,
        client_task_json=args.client_task_json,
        keyword_categories=tuple(args.keyword_categories),
        tasks_per_category=args.tasks_per_category,
        seed=args.seed,
    )

    logger = JsonlLogger(log_dir=args.log_dir)

    # Determine client->task map
    explicit = load_json(dcfg.client_task_json) if (dcfg.client_task_mode == "explicit" and dcfg.client_task_json) else None
    client_task_map = make_client_task_map(
        num_clients=cfg.num_clients,
        dataset_name=dcfg.dataset_name,
        mode=dcfg.client_task_mode,
        categories=dcfg.keyword_categories,
        tasks_per_category=dcfg.tasks_per_category,
        seed=dcfg.seed,
        explicit_json=explicit,
    )
    print("[TaskMap]", client_task_map)

    # Load model
    num_total_experts = 1 + cfg.k_max
    lora_cfg = MultiExpertLoRAConfig(rank=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
    loaded = load_base_model_with_multiexpert_lora(
        model_name=args.model_name,
        num_total_experts=num_total_experts,
        lora_cfg=lora_cfg,
        target_keywords=cfg.target_linear_keywords,
        use_4bit=cfg.use_4bit,
        device_map="auto",
    )
    model, tokenizer, device, routing_ctx = loaded.model, loaded.tokenizer, loaded.device, loaded.routing_ctx

    # Build dataloaders
    loaders = make_sni_client_loaders(
        tokenizer=tokenizer,
        dataset_name=dcfg.dataset_name,
        client_task_map=client_task_map,
        seq_len=cfg.seq_len,
        micro_batch=cfg.micro_batch,
        seed=dcfg.seed,
        train_max_examples=None,
        eval_max_examples=512,
    )

    # Server
    active_specialized_init = list(range(1, 1 + cfg.k_init))
    server = Server(
        model=model,
        num_total_experts=num_total_experts,
        active_specialized_init=active_specialized_init,
        lr_server=cfg.lr_server,
        birth_patience=cfg.birth_patience,
        birth_reject_rate=cfg.birth_reject_rate,
        prune_patience=cfg.prune_patience,
        prune_util_threshold=cfg.prune_util_threshold,
        merge_similarity_threshold=cfg.merge_similarity_threshold,
    )

    # Clients
    c_cfg = ClientConfig(
        tau=cfg.tau,
        gamma=cfg.gamma,
        top_m=cfg.top_m,
        local_steps=cfg.local_steps,
        lr_client=cfg.lr_client,
        weight_decay=cfg.weight_decay,
        lambda_consistency=cfg.lambda_consistency,
        max_train_batches_per_round=cfg.max_train_batches_per_round,
        max_probe_batches=cfg.max_probe_batches,
        max_eval_batches=cfg.max_eval_batches,
        max_gen_eval_examples=cfg.max_gen_eval_examples,
        gen_max_new_tokens=cfg.gen_max_new_tokens,
        reject_conf_threshold=cfg.reject_conf_threshold,
        use_dp=cfg.use_dp,
        dp_clip_norm=cfg.dp_clip_norm,
        dp_noise_std=cfg.dp_noise_std,
    )
    clients: List[Client] = []
    for cid in range(cfg.num_clients):
        train_loader, val_loader, test_loader = loaders[cid]
        clients.append(Client(
            client_id=cid,
            model=model,
            tokenizer=tokenizer,
            device=device,
            routing_ctx=routing_ctx,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            cfg=c_cfg,
            num_total_experts=num_total_experts,
        ))

    # Optional: load local baselines for negative transfer
    local_baselines = load_json(args.local_baseline_json) if args.local_baseline_json else None

    # Federated rounds
    for rnd in range(cfg.num_rounds):
        print(f"\n=== Round {rnd+1}/{cfg.num_rounds} ===")
        payload = server.broadcast()
        bank = payload["bank"]
        pi = payload["pi"]
        active_specialized = payload["active_specialized"]

        server.reset_aggregator()

        # Run clients (sequential simulation; cross-silo typically parallel)
        for client in clients:
            probe = client.probe_losses(bank=bank, candidates=active_specialized)
            r = client.compute_responsibilities(probe, pi=pi)
            active = client.select_active_experts(r, active_specialized)
            reject, conf = client.compute_openworld_signals(r, active_specialized)

            upd = client.local_train(bank=bank, r=r, active=active)
            upd["reject"] = reject
            upd["conf"] = conf
            server.aggregate_client_update(upd)

            print(f"[Client {client.client_id}] task={client_task_map[client.client_id]} active={active} conf={conf:.3f} reject={reject}")

        # Server step (updates bank + open-world)
        server.step(num_clients=cfg.num_clients)
        server.load_bank_to_model()

        # Log utilization this round (based on aggregated S)
        S_round = server.secagg.get("S")
        util = {int(k): float(S_round[int(k)].item()) / max(cfg.num_clients, 1) for k in range(num_total_experts)}
        logger.log({
            "type": "server_round",
            "round": rnd,
            "active_specialized": list(server.state.active_specialized),
            "util_frac": util,
        })

        # Evaluation
        if args.eval_every > 0 and ((rnd + 1) % args.eval_every == 0):
            for client in clients:
                ppl = client.evaluate_ppl(
                    bank=server.global_bank, 
                    pi=server.state.pi, 
                    active_specialized=server.state.active_specialized, 
                    split=args.eval_split 
                )
                gen = client.evaluate_generation(
                    bank=server.global_bank, 
                    pi=server.state.pi, 
                    active_specialized=server.state.active_specialized, 
                    split=args.eval_split
                )

                rec = {
                    "type": "client_eval",
                    "round": rnd,
                    "client_id": client.client_id,
                    "tasks": client_task_map[client.client_id],
                    "ppl": ppl["ppl"],
                    "loss": ppl["loss"],
                    "rougeL": gen["rougeL"],
                    "em": gen["em"],
                    "active_eval": ppl["active"],
                }

                # Negative transfer if local baseline exists
                if local_baselines is not None and str(client.client_id) in local_baselines:
                    base = local_baselines[str(client.client_id)]
                    # ppl: lower is better => gain = ppl_local - ppl_fed
                    rec["neg_transfer_ppl"] = float(base["ppl"]) - float(ppl["ppl"])
                    # rouge/em: higher is better => fed - local
                    rec["neg_transfer_rougeL"] = float(gen["rougeL"]) - float(base.get("rougeL", gen["rougeL"]))
                    rec["neg_transfer_em"] = float(gen["em"]) - float(base.get("em", gen["em"]))

                logger.log(rec)
                print(f"[Eval c{client.client_id}] PPL={ppl['ppl']:.2f} ROUGE-L={gen['rougeL']:.3f} EM={gen['em']:.3f}")
    
    save_path_bank = os.path.join(args.log_dir, "global_bank.pt")
    save_path_pi = os.path.join(args.log_dir, "global_pi.pt")
    
    # 保存 Expert Bank (LoRA 参数)
    torch.save(server.global_bank, save_path_bank)
    # 保存 Prior 分布
    torch.save(server.state.pi, save_path_pi)
    print(f"Model saved to:\n  - {save_path_bank}\n  - {save_path_pi}")

    print("\nDone. Logs in:", os.path.join(args.log_dir, "metrics.jsonl"))

if __name__ == "__main__":
    main()
