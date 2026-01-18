# Fed-VEMoE Project (Cross-silo, Cross-task) — Full Research Skeleton

This project upgrades the original demo into a **dataset-driven, evaluation-ready** federated simulation:

✅ Real dataset loader for **Natural Instructions / Super-NaturalInstructions-style** task bank  
✅ Per-client task assignment (translation / reasoning / code / rewriting / etc. via keyword search in task definitions)  
✅ Hierarchical routing:
- Domain-level responsibilities `r_{i,k}` (probe losses)
- Token-level posterior mixture `α_{t,k} = softmax(log r + u/γ)` inside the active expert set

✅ Server-oblivious assignment via secure aggregation **simulation** (server only sees sums)  
✅ Open-world management (birth / prune / merge) with Kmax slots  
✅ Evaluation: PPL, ROUGE-L, Exact Match (EM); per-task performance + negative transfer  
✅ Logging + plotting scripts (expert utilization, worst-client metric, negative transfer)

> Notes:
> - Secure aggregation is simulated (`privacy.SecureAggSim`). Replace with a real protocol for deployment.
> - For 7B, use 4-bit loading (QLoRA-style) and gradient checkpointing.
> - This code is designed for **cross-silo simulation** (sequential client execution); parallelization is a later engineering step.

## Install

```bash
pip install -U torch transformers accelerate datasets bitsandbytes evaluate matplotlib pandas
```

## Default dataset source (Natural Instructions)
We use the HuggingFace dataset:
- `Muennighoff/natural-instructions` which contains fields: `task_name`, `definition`, `inputs`, `targets` and splits `train/validation/test`.

You can verify fields in the dataset viewer on HF.

## Run federated training (toy rounds, real tasks)

  #### --dataset_name Muennighoff/natural-instructions \

```bash
python run_fed.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset_name ./natural-instructions \
  --streaming 1 \
  --num_clients 5 \
  --client_task_mode keyword \
  --keyword_categories translation reasoning code rewriting \
  --tasks_per_category 3 \
  --num_rounds 10 \
  --k_init 3 --k_max 12 \
  --top_m 2 \
  --use_4bit 1 \
  --seq_len 2048 \
  --eval_every 1 \
  --log_dir runs/exp3
```

## Run local baselines (for negative transfer)
```bash
python run_local_baselines.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset_name Muennighoff/natural-instructions \
  --streaming 1 \
  --num_clients 6 \
  --client_task_mode keyword \
  --keyword_categories translation reasoning code rewriting \
  --tasks_per_category 1 \
  --total_steps 200 \
  --use_4bit 1 \
  --out_json runs/exp1/local_baselines.json
```

Then re-run `run_fed.py` with:
```bash
--local_baseline_json runs/exp1/local_baselines.json
```

## Plot
```bash
python scripts/plot_logs.py --log_dir runs/exp1
```

Outputs:
- `expert_utilization.png`
- `worst_client.png`
- `negative_transfer.png`

### plot

```bash
python -m scripts.plot_paper_figure \
  --log_dir runs/exp4 \
  --dataset_name ./natural-instructions
```