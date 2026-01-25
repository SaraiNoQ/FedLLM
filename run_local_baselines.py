from __future__ import annotations
import argparse
import os
import math
from typing import Dict, Any, List

import torch
from torch.optim import AdamW

from utils.seed import set_seed
from utils.logger import save_json
from config import FedVEMoEConfig, DataConfig

from model import load_base_model_with_multiexpert_lora
from lora_experts import MultiExpertLoRAConfig, MultiExpertLoRALinear, get_expert_bank_state, load_expert_bank_state
from data_sni import make_client_task_map, make_sni_client_loaders
from eval_metrics import rouge_l_f1, exact_match, aggregate_metric

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--use_4bit", type=int, default=1)

    p.add_argument("--dataset_name", type=str, default="Muennighoff/natural-instructions")
    p.add_argument("--streaming", type=int, default=1)
    p.add_argument("--num_clients", type=int, default=6)
    p.add_argument("--client_task_mode", type=str, default="keyword", choices=["keyword","explicit"])
    p.add_argument("--client_task_json", type=str, default="")
    p.add_argument("--keyword_categories", nargs="+", default=["translation","reasoning","code","rewriting"])
    p.add_argument("--tasks_per_category", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--seq_len", type=int, default=768)
    p.add_argument("--micro_batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--total_steps", type=int, default=200)
    p.add_argument("--max_eval_batches", type=int, default=8)
    p.add_argument("--max_gen_eval_examples", type=int, default=32)
    p.add_argument("--gen_max_new_tokens", type=int, default=64)

    p.add_argument("--eval_split", type=str, default="train", choices=["train","val","test"])
    p.add_argument("--out_json", type=str, default="runs/exp1/local_baselines.json")
    return p.parse_args()

@torch.no_grad()
def evaluate_ppl(model, routing_ctx, bank, loader, device, max_batches: int):
    load_expert_bank_state(model, bank)
    routing_ctx.mode = "probe"
    routing_ctx.active_experts = torch.tensor([0], device=device)
    routing_ctx.domain_prior = None
    routing_ctx.token_router = None

    model.eval()
    tot_loss, n = 0.0, 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        tot_loss += float(out.loss.detach().cpu())
        n += 1
    avg_loss = tot_loss / max(n, 1)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return avg_loss, ppl

@torch.no_grad()
def evaluate_generation(model, tokenizer, routing_ctx, bank, loader, device, max_examples: int, max_new_tokens: int):
    load_expert_bank_state(model, bank)
    routing_ctx.mode = "probe"
    routing_ctx.active_experts = torch.tensor([0], device=device)
    routing_ctx.domain_prior = None
    routing_ctx.token_router = None

    model.eval()
    rouges, ems = [], []
    seen = 0
    for batch in loader:
        bsz = batch["input_ids"].shape[0]
        for i in range(bsz):
            if seen >= max_examples:
                break
            prompt = batch["prompt_text"][i]
            ref = batch["target_text"][i]
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_ids = gen[0].tolist()
            prompt_len = enc["input_ids"].shape[1]
            pred = tokenizer.decode(gen_ids[prompt_len:], skip_special_tokens=True)
            rouges.append(rouge_l_f1(pred, ref))
            ems.append(exact_match(pred, ref))
            seen += 1
        if seen >= max_examples:
            break
    return aggregate_metric(rouges), aggregate_metric(ems)

def train_local(model, routing_ctx, bank_init, train_loader, device, total_steps: int, lr: float, grad_accum: int):
    load_expert_bank_state(model, bank_init)
    routing_ctx.mode = "probe"
    routing_ctx.active_experts = torch.tensor([0], device=device)
    routing_ctx.domain_prior = None
    routing_ctx.token_router = None

    trainable = []
    for _, m in model.named_modules():
        if isinstance(m, MultiExpertLoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
            trainable.append(m.lora_A)
            trainable.append(m.lora_B)

    opt = AdamW(trainable, lr=lr)
    model.train()

    it = iter(train_loader)
    step = 0
    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            (out.loss / grad_accum).backward()
            loss_acc += float(out.loss.detach().cpu())
        opt.step()
        step += 1
        if step % 50 == 0:
            print(f"  step {step}/{total_steps} loss={loss_acc/grad_accum:.4f}")

    bank = get_expert_bank_state(model)
    return bank

def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = FedVEMoEConfig(seq_len=args.seq_len, micro_batch=args.micro_batch, grad_accum=args.grad_accum)
    dcfg = DataConfig(
        dataset_name=args.dataset_name,
        streaming=bool(args.streaming),
        client_task_mode=args.client_task_mode,
        client_task_json=args.client_task_json,
        keyword_categories=tuple(args.keyword_categories),
        tasks_per_category=args.tasks_per_category,
        seed=args.seed,
    )

    explicit = None
    if dcfg.client_task_mode == "explicit" and dcfg.client_task_json:
        import json
        with open(dcfg.client_task_json, "r", encoding="utf-8") as f:
            explicit = json.load(f)

    client_task_map = make_client_task_map(
        num_clients=args.num_clients,
        dataset_name=dcfg.dataset_name,
        mode=dcfg.client_task_mode,
        categories=dcfg.keyword_categories,
        tasks_per_category=dcfg.tasks_per_category,
        seed=dcfg.seed,
        explicit_json=explicit,
    )
    print("[TaskMap]", client_task_map)

    # Single-expert local LoRA
    lora_cfg = MultiExpertLoRAConfig(rank=cfg.lora_rank, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)
    loaded = load_base_model_with_multiexpert_lora(
        model_name=args.model_name,
        num_total_experts=1,
        lora_cfg=lora_cfg,
        target_keywords=cfg.target_linear_keywords,
        use_4bit=bool(args.use_4bit),
        device_map="auto",
    )
    model, tokenizer, device, routing_ctx = loaded.model, loaded.tokenizer, loaded.device, loaded.routing_ctx

    loaders = make_sni_client_loaders(
        tokenizer=tokenizer,
        dataset_name=dcfg.dataset_name,
        client_task_map=client_task_map,
        seq_len=args.seq_len,
        micro_batch=args.micro_batch,
        seed=dcfg.seed,
        train_max_examples=None,
        eval_max_examples=512,
    )

    bank_init = get_expert_bank_state(model)

    results: Dict[str, Any] = {}
    for cid in range(args.num_clients):
        print(f"\n[Local baseline] client {cid} task={client_task_map[cid]}")
        train_loader, val_loader, test_loader = loaders[cid]
        bank_trained = train_local(
            model=model,
            routing_ctx=routing_ctx,
            bank_init=bank_init,
            train_loader=train_loader,
            device=device,
            total_steps=args.total_steps,
            lr=args.lr,
            grad_accum=args.grad_accum,
        )

        if args.eval_split == "train":
            loader = train_loader
        elif args.eval_split == "test":
            loader = test_loader
        else:
            loader = val_loader
        loss, ppl = evaluate_ppl(model, routing_ctx, bank_trained, loader, device, max_batches=args.max_eval_batches)
        rougeL, em = evaluate_generation(model, tokenizer, routing_ctx, bank_trained, loader, device, args.max_gen_eval_examples, args.gen_max_new_tokens)

        results[str(cid)] = {"loss": loss, "ppl": ppl, "rougeL": rougeL, "em": em, "tasks": client_task_map[cid]}

        print(f"[Baseline c{cid}] PPL={ppl:.2f} ROUGE-L={rougeL:.3f} EM={em:.3f}")

    save_json(args.out_json, results)
    print("\nSaved:", args.out_json)

if __name__ == "__main__":
    main()
