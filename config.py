from dataclasses import dataclass

@dataclass
class FedVEMoEConfig:
    # federation
    num_rounds: int = 5
    num_clients: int = 5
    num_experts: int = 4   # specialized experts (excluding shared expert 0)
    top_m: int = 2         # active experts per client: shared + top-(M-1) specialized

    # routing
    tau: float = 0.5       # domain responsibility temperature
    gamma: float = 1.0     # token routing temperature
    lambda_consistency: float = 0.1  # KL(mean_alpha || prior) weight

    # optimization
    local_steps: int = 5
    lr_server: float = 1.0
    lr_client: float = 2e-4
    weight_decay: float = 0.0

    # lora
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_linear_keywords: tuple = ("q_proj", "v_proj")  # typical for Llama/Mistral/Qwen2

    # training / memory
    use_4bit: bool = True
    seq_len: int = 512
    micro_batch: int = 1
    grad_accum: int = 8
    max_train_batches_per_round: int = 8
    max_probe_batches: int = 2

    # privacy (optional)
    use_dp: bool = False
    dp_clip_norm: float = 1.0
    dp_noise_std: float = 0.0  # set >0 to enable noise
