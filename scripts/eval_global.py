from __future__ import annotations
import argparse
import os
import torch
import math
from typing import List

# 复用现有的配置和工具
from config import FedVEMoEConfig, DataConfig
from utils.seed import set_seed
from model import load_base_model_with_multiexpert_lora
from lora_experts import MultiExpertLoRAConfig
from client import Client, ClientConfig
from data_sni import make_client_task_map, make_sni_client_loaders
from utils.logger import load_json

def parse_args():
    p = argparse.ArgumentParser()
    # 必须与训练时保持一致的参数
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct") # 修改为你的模型路径
    p.add_argument("--use_4bit", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=2048) # 修改为你的配置
    
    # 数据集参数
    p.add_argument("--dataset_name", type=str, default="Muennighoff/natural-instructions")
    p.add_argument("--num_clients", type=int, default=4) # 修改为你的实际 client 数
    p.add_argument("--client_task_mode", type=str, default="keyword")
    p.add_argument("--tasks_per_category", type=int, default=3) # 修改为你的配置
    p.add_argument("--seed", type=int, default=42)

    # 模型结构参数 (需与训练一致)
    p.add_argument("--k_max", type=int, default=12) 
    p.add_argument("--target_linear_keywords", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]) 

    # 路径参数
    p.add_argument("--log_dir", type=str, required=True, help="Path to the directory containing global_bank.pt")
    p.add_argument("--batch_size", type=int, default=8, help="Eval batch size")
    
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Global Model from {args.log_dir}...")
    bank_path = os.path.join(args.log_dir, "global_bank.pt")
    pi_path = os.path.join(args.log_dir, "global_pi.pt")

    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"Cannot find {bank_path}. Please run run_fed.py with saving enabled first.")

    global_bank = torch.load(bank_path, map_location=device)
    global_pi = torch.load(pi_path, map_location=device)
    
    # 1. 恢复 Task Map
    # 注意：这里会重新扫描一遍数据集生成 Map，必须保证 seed 和参数与训练时完全一致
    client_task_map = make_client_task_map(
        num_clients=args.num_clients,
        dataset_name=args.dataset_name,
        mode=args.client_task_mode,
        categories=None, # 默认使用所有类别
        tasks_per_category=args.tasks_per_category,
        seed=args.seed,
    )
    print("[TaskMap Reconstructed]", client_task_map)

    # 2. 加载基础模型 + LoRA 骨架
    num_total_experts = 1 + args.k_max
    # 这里的参数要和训练时的 Config 保持一致
    lora_cfg = MultiExpertLoRAConfig(rank=16, alpha=32, dropout=0.05) # 请确保这里和你 config.py 改过的一致
    
    loaded = load_base_model_with_multiexpert_lora(
        model_name=args.model_name,
        num_total_experts=num_total_experts,
        lora_cfg=lora_cfg,
        target_keywords=tuple(args.target_linear_keywords),
        use_4bit=bool(args.use_4bit),
        device_map="auto",
    )
    model, tokenizer, routing_ctx = loaded.model, loaded.tokenizer, loaded.routing_ctx

    # 3. 准备数据加载器
    loaders = make_sni_client_loaders(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        client_task_map=client_task_map,
        seq_len=args.seq_len,
        micro_batch=args.batch_size, # 评估时可以用大一点的 batch
        seed=args.seed,
        train_max_examples=100, # 限制评估样本数，避免跑太久
        eval_max_examples=100,
    )

    # 4. 初始化 Client 对象 (用于复用评估代码)
    # 我们构造一个 Dummy Config，只用到评估相关的参数
    c_cfg = ClientConfig(
        tau=1.0, gamma=1.0, top_m=2, local_steps=1, lr_client=0.0, weight_decay=0.0,
        lambda_consistency=0.0, max_train_batches_per_round=1, max_probe_batches=2,
        max_eval_batches=10, max_gen_eval_examples=32, gen_max_new_tokens=64,
        reject_conf_threshold=0.5, use_dp=False, dp_clip_norm=1.0, dp_noise_std=0.0
    )

    print("\n=== Starting Evaluation on TRAINING SET (Global Model) ===")
    
    # 全局专家的激活列表 (假设所有专家都处于活跃状态)
    active_specialized = list(range(1, num_total_experts))

    for cid in range(args.num_clients):
        train_loader, _, _ = loaders[cid]
        
        # 实例化一个临时的 Client 对象
        client = Client(
            client_id=cid,
            model=model,
            tokenizer=tokenizer,
            device=device,
            routing_ctx=routing_ctx,
            train_loader=train_loader,
            val_loader=None,
            test_loader=None,
            cfg=c_cfg,
            num_total_experts=num_total_experts,
        )

        # 调用 Client 内部的评估函数，指定 split='train'
        # 这会自动处理 Probe (路由选择) -> Generation
        gen_res = client.evaluate_generation(
            bank=global_bank,
            pi=global_pi,
            active_specialized=active_specialized,
            split="train",  # <--- 重点：指定在训练集上跑
            max_examples=32 # 测试样本数
        )

        print(f"[Client {cid}] Train Set Performance:")
        print(f"   Tasks: {client_task_map[cid]}")
        print(f"   Active Experts Selected: {gen_res['active']}")
        print(f"   ROUGE-L: {gen_res['rougeL']:.4f}")
        print(f"   Exact Match: {gen_res['em']:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    main()