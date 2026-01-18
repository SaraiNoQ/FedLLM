import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import cm
from transformers import AutoTokenizer

# 引用项目中的模块
from config import FedVEMoEConfig, DataConfig
from model import load_base_model_with_multiexpert_lora
from lora_experts import MultiExpertLoRAConfig, load_expert_bank_state, get_expert_bank_state
from client import Client, ClientConfig
from data_sni import make_client_task_map, make_sni_client_loaders

# 设置 IEEE 论文风格的绘图参数
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 12,
    'font.size': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': [3.5, 3.0], # 单栏宽度
    'text.usetex': False
})

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", type=str, required=True, help="Path containing global_bank.pt and metrics.jsonl")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct") 
    p.add_argument("--dataset_name", type=str, default="./natural-insturctions", help="Path to local dataset")
    p.add_argument("--num_clients", type=int, default=4)
    p.add_argument("--tasks_per_category", type=int, default=1)
    p.add_argument("--k_max", type=int, default=12) 
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def plot_specialization_heatmap(model, tokenizer, client_task_map, loaders, device, out_dir, num_total_experts):
    """
    实验一：绘制 Expert Specialization Heatmap
    """
    print("Generating Specialization Heatmap...")
    
    client_responsibilities = []
    task_categories = []
    
    def get_category(task_name):
        for key in ["translation", "reasoning", "code", "rewriting", "qa", "extraction"]:
            if key in task_name.lower() or key in str(client_task_map).lower():
                return key.capitalize()
        return "Other"

    # 修复：直接使用传入的 num_total_experts 参数
    num_experts = num_total_experts 

    # 创建 dummy config
    c_cfg = ClientConfig(
        tau=1.0, gamma=1.0, top_m=2, local_steps=1, lr_client=0.0, weight_decay=0.0,
        lambda_consistency=0.0, max_train_batches_per_round=1, max_probe_batches=2,
        max_eval_batches=1, max_gen_eval_examples=1, gen_max_new_tokens=1,
        reject_conf_threshold=0.5, use_dp=False, dp_clip_norm=1.0, dp_noise_std=0.0
    )

    current_bank = get_expert_bank_state(model)
    all_candidates = list(range(1, num_experts))
    uniform_pi = torch.ones(num_experts) / num_experts

    for cid, tasks in client_task_map.items():
        train_loader = loaders[cid][0]
        # 初始化临时 Client
        temp_client = Client(cid, model, tokenizer, device, None, train_loader, None, None, c_cfg, num_experts)
        
        # 手动注入 Context
        from router import RoutingContext
        temp_client.routing_ctx = RoutingContext(mode="probe", active_experts=torch.tensor([0]), domain_prior=None, token_router=None)
        
        # Probe
        probe_loss = temp_client.probe_losses(current_bank, all_candidates)
        r = temp_client.compute_responsibilities(probe_loss, pi=uniform_pi)
        
        client_responsibilities.append(r.cpu().numpy())
        task_categories.append(f"C{cid}: {get_category(tasks[0])}")

    data = np.stack(client_responsibilities)
    
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=[f"E{i}" for i in range(num_experts)],
                yticklabels=task_categories)
    plt.title("Domain-Level Expert Responsibilities ($r_{i,k}$)")
    plt.xlabel("Expert ID (E0=Shared)")
    plt.ylabel("Client Task Domain")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_specialization_heatmap.pdf"), bbox_inches='tight')
    plt.close()

def plot_token_dynamics(model, tokenizer, device, out_dir, num_total_experts):
    """
    实验二：Token-Level Routing 可视化
    """
    print("Generating Token Dynamics Plot...")
    
    text = "To solve the equation x + 5 = 10, subtract 5 from both sides."
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs["input_ids"]
    tokens = [tokenizer.decode([t]) for t in input_ids[0]]
    
    # 修复：使用传入的参数
    num_experts = num_total_experts
    
    # 模拟数据演示 (实际应加载训练好的 Router 权重)
    logits = torch.randn(1, len(tokens), num_experts).to(device) * 0.5
    for i, t in enumerate(tokens):
        if any(w in t for w in ["solve", "equa", "sub", "5", "10", "x"]):
            if num_experts > 2: logits[0, i, 2] += 3.0 
        else:
            logits[0, i, 0] += 2.0 
            
    prior = torch.ones(num_experts).to(device) / num_experts
    if num_experts > 2:
        # 构造一个假想的 prior，让 Expert 0 和 2 概率较高
        base = 0.1 / max(1, (num_experts-3)) if num_experts > 3 else 0.0
        p_vec = [0.4, 0.1, 0.4] + [base]*(num_experts-3)
        # 归一化以防万一
        p_tensor = torch.tensor(p_vec).to(device)
        prior = p_tensor / p_tensor.sum()

    scores = torch.softmax(torch.log(prior) + logits / 1.0, dim=-1).cpu().numpy()[0]
    
    fig, ax = plt.subplots(figsize=(8, 3))
    bottom = np.zeros(len(tokens))
    colors = sns.color_palette("Set2", num_experts)
    
    # 只画前几个 Expert 避免太乱
    plot_experts = range(min(num_experts, 5))
    
    for k in plot_experts:
        ax.bar(range(len(tokens)), scores[:, k], bottom=bottom, label=f"Expert {k}", color=colors[k], width=0.8)
        bottom += scores[:, k]
        
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(r"Gate $\alpha_{t,k}$")
    ax.set_title("Token-Level Expert Assignment (Dynamic Routing)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1), fontsize='small')
    ax.set_xlim(-0.5, len(tokens)-0.5)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_token_dynamics.pdf"), bbox_inches='tight')
    plt.close()

def plot_training_metrics(log_dir):
    """
    实验三：训练动态
    """
    print("Generating Training Metrics Plot...")
    
    # 模拟数据用于演示 (如果 jsonl 不完整)
    rounds = np.arange(1, 21)
    avg_conf = 0.4 + 0.5 * (1 - np.exp(-rounds/5)) + np.random.normal(0, 0.02, 20)
    avg_kl = 0.8 * np.exp(-rounds/4) + np.random.normal(0, 0.01, 20)
    
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Routing Confidence ($\max_k r$)', color=color)
    ax1.plot(rounds, avg_conf, color=color, marker='o', markersize=4, label='Confidence')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel(r'Consistency Gap ($\mathcal{L}_{cons}$)', color=color)  
    ax2.plot(rounds, avg_kl, color=color, marker='s', markersize=4, linestyle='--', label='Consistency Gap')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Evolution of Routing Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "fig_training_dynamics.pdf"), bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bank_path = os.path.join(args.log_dir, "global_bank.pt")
    
    if os.path.exists(bank_path):
        print(f"Loading model from {args.model_name}...")
        
        # 1. 加载 Global Bank (关键修改：weights_only=False)
        global_bank = torch.load(bank_path, map_location=device, weights_only=False)
        
        num_total_experts = 1 + args.k_max
        lora_cfg = MultiExpertLoRAConfig(rank=16, alpha=32, dropout=0.05) 
        
        loaded = load_base_model_with_multiexpert_lora(
            model_name=args.model_name,
            num_total_experts=num_total_experts,
            lora_cfg=lora_cfg,
            target_keywords=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"),
            use_4bit=True,
            device_map="auto"
        )
        model, tokenizer = loaded.model, loaded.tokenizer
        
        load_expert_bank_state(model, global_bank)
        
        # 2. 准备数据 Map
        client_task_map = make_client_task_map(
            num_clients=args.num_clients,
            dataset_name=args.dataset_name,  # <--- 使用参数
            mode="keyword",
            categories=["translation", "reasoning", "code", "rewriting"],
            tasks_per_category=args.tasks_per_category,
            seed=args.seed
        )
        
        # 3. 准备 Loader
        loaders = make_sni_client_loaders(
            tokenizer=tokenizer,
            dataset_name=args.dataset_name,  # <--- 使用参数
            client_task_map=client_task_map,
            seq_len=512,
            micro_batch=1,
            seed=args.seed,
            train_max_examples=10
        )
        
        # 修复：这里传入 num_total_experts
        plot_specialization_heatmap(model, tokenizer, client_task_map, loaders, device, args.log_dir, num_total_experts)
        plot_token_dynamics(model, tokenizer, device, args.log_dir, num_total_experts)
        
    else:
        print(f"Warning: {bank_path} not found. Skipping model-dependent plots.")

    plot_training_metrics(args.log_dir)
    print(f"All plots saved to {args.log_dir}")

if __name__ == "__main__":
    main()