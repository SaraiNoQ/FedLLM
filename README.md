# Fed-VEMoE Demo (Minimal, Research-Oriented Skeleton)

This is a **minimal demo** codebase for the idea discussed:
**Fed-VEMoE + Hierarchical Bayesian Routing (domain prior + token evidence)**,
with **LoRA expert bank**, **cross-silo** simulation, and a **secure-aggregation *simulation***
(i.e., server only keeps aggregated sums and discards per-client updates).

> IMPORTANT:
> - This demo is intended as a *starting point* for research code.
> - For real 7B training, use 4-bit loading (QLoRA-style) and enable gradient checkpointing.
> - Secure aggregation is **simulated** here; replace `privacy.SecureAggSim` with a real protocol in deployment.

## Quickstart (Toy Run)

```bash
pip install -U torch transformers accelerate datasets bitsandbytes
python run.py --model_name Qwen/Qwen2.5-7B-Instruct --use_4bit 1 --num_rounds 10 --num_clients 3 --num_experts 4
```

If you cannot load a 7B model yet, start with a smaller Llama-like model (still has q_proj/v_proj):
```bash
python run.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --use_4bit 1
```

## What is implemented

- Expert bank: shared expert (id=0) + K specialized experts (1..K).
- Domain routing (E-step): probe losses -> responsibilities r_{i,k}.
- Token routing: α_{t,k} = softmax(log r_{i,k} + u_{t,k}/γ) within Ω_i.
- Local training updates LoRA of active experts + private token-router.
- Uploads are aggregated at server (SecureAggSim): S_k and U_k.
- Server updates experts: ΔW_k += η * U_k / max(S_k, ε).

## What is left as “paper engineering”

- Open-world expert birth/prune/merge (stubbed in server.py).
- Better probing (more stable proxies, candidate filtering).
- Proper DP accounting (we include clip+noise hooks).
- Real multi-task datasets and evaluation suites.

