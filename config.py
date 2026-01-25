from dataclasses import dataclass
from typing import Tuple

@dataclass
class FedVEMoEConfig:
    # federation
    num_rounds: int = 5
    num_clients: int = 6

    # experts
    k_init: int = 4             # initial specialized experts active
    k_max: int = 12             # max specialized experts slots (open-world budget)
    top_m: int = 2              # active experts per client: shared + top-(M-1) specialized

    # routing
    tau: float = 0.5            # domain responsibility temperature
    gamma: float = 1.0          # token routing temperature
    lambda_consistency: float = 0.1

    # optimization
    local_steps: int = 2
    lr_server: float = 1.0
    lr_client: float = 2e-4
    weight_decay: float = 0.0

    # lora
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_linear_keywords: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj"
    )  # typical for Llama/Mistral/Qwen2

    # training/memory
    use_4bit: bool = True
    seq_len: int = 2048
    micro_batch: int = 4
    grad_accum: int = 4
    max_train_batches_per_round: int = 32
    max_probe_batches: int = 2
    max_eval_batches: int = 16
    max_gen_eval_examples: int = 32
    gen_max_new_tokens: int = 128

    # open-world management
    birth_patience: int = 2
    birth_reject_rate: float = 0.1
    reject_conf_threshold: float = 0.45   # if max_k r_{i,k} < threshold => client is "rejected"
    prune_patience: int = 3
    prune_util_threshold: float = 0.02    # utilization fraction under which prune can trigger
    merge_similarity_threshold: float = 0.995

    # privacy (optional)
    use_dp: bool = False
    dp_clip_norm: float = 1.0
    dp_noise_std: float = 0.0

@dataclass
class DataConfig:
    dataset_name: str = "./natural-instructions"
    streaming: bool = True
    # Task assignment modes:
    # - "explicit": pass a JSON file mapping client->task_names
    # - "keyword": pick tasks by scanning definitions for keywords
    client_task_mode: str = "keyword"
    client_task_json: str = ""
    keyword_categories: Tuple[str, ...] = ("translation", "reasoning", "code", "rewriting")
    tasks_per_category: int = 2
    seed: int = 42
