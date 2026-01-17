from __future__ import annotations
import argparse
import math
from typing import List

import torch

from config import FedVEMoEConfig
from model import load_base_model_with_multiexpert_lora
from lora_experts import MultiExpertLoRAConfig
from data import make_client_dataloaders
from server import Server
from client import Client, ClientConfig

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--num_rounds", type=int, default=2)
    p.add_argument("--num_clients", type=int, default=3)
    p.add_argument("--num_experts", type=int, default=4)  # specialized
    p.add_argument("--use_4bit", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=512)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = FedVEMoEConfig(
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        num_experts=args.num_experts,
        use_4bit=bool(args.use_4bit),
        seq_len=args.seq_len,
    )

    num_total_experts = 1 + cfg.num_experts  # shared + specialized
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

    # Build server
    server = Server(model=model, num_total_experts=num_total_experts, lr_server=cfg.lr_server)

    # Build toy client dataloaders
    loaders = make_client_dataloaders(tokenizer, cfg.num_clients, cfg.seq_len, cfg.micro_batch)

    # Build clients
    clients: List[Client] = []
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
        use_dp=cfg.use_dp,
        dp_clip_norm=cfg.dp_clip_norm,
        dp_noise_std=cfg.dp_noise_std,
    )
    for i in range(cfg.num_clients):
        clients.append(Client(
            client_id=i,
            model=model,
            tokenizer=tokenizer,
            device=device,
            routing_ctx=routing_ctx,
            dataloader=loaders[i],
            cfg=c_cfg,
            num_total_experts=num_total_experts,
        ))

    # Training loop
    for t in range(cfg.num_rounds):
        print(f"\n=== Round {t+1}/{cfg.num_rounds} ===")
        payload = server.broadcast()
        bank = payload["bank"]
        pi = payload["pi"]

        server.reset_aggregator()

        # Each client in turn (simulation). In real cross-silo, they'd run in parallel.
        for client in clients:
            # Probe specialized experts 1..K
            candidates = list(range(1, num_total_experts))
            probe = client.probe_losses(bank=bank, candidates=candidates)
            r = client.compute_responsibilities(probe, pi=pi)
            active = client.select_active_experts(r)
            upd = client.local_train(bank=bank, r=r, active=active)
            server.aggregate_client_update(upd)
            print(f"[Client {client.client_id}] active={active} r(spec)={r[1:].tolist()}")

        server.step()
        # Load updated bank back into model for next round
        server.load_bank_to_model()

    print("\nDone.")

if __name__ == "__main__":
    main()
