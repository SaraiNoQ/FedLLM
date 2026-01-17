from __future__ import annotations
from dataclasses import dataclass
import importlib.util
from typing import Tuple, Optional

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from router import RoutingContext
from lora_experts import MultiExpertLoRAConfig, inject_multi_expert_lora
from modelscope.hub.snapshot_download import snapshot_download

@dataclass
class LoadedModel:
    model: nn.Module
    tokenizer: any
    device: torch.device
    routing_ctx: RoutingContext

def load_base_model_with_multiexpert_lora(
    model_name: str,
    num_total_experts: int,
    lora_cfg: MultiExpertLoRAConfig,
    target_keywords: tuple,
    use_4bit: bool = True,
    device_map: str = "auto",
) -> LoadedModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )

    quant_config = None
    if use_4bit:
        if not torch.cuda.is_available():
            print("[WARN] CUDA not available; disabling 4-bit quantization.")
            use_4bit = False
        if importlib.util.find_spec("bitsandbytes") is None:
            print("[WARN] bitsandbytes not found; disabling 4-bit quantization.")
            use_4bit = False
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

    model_dir = snapshot_download(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map=device_map,
    )

    # Freeze base params
    for p in model.parameters():
        p.requires_grad_(False)

    # Minimal routing context; clients will overwrite fields before forward
    if num_total_experts == 1:
        active = torch.tensor([0], device=device)
    else:
        active = torch.tensor([0, 1], device=device)

    routing_ctx = RoutingContext(
        mode="probe",
        active_experts=active,
        domain_prior=None,
        token_router=None,
        gamma=1.0,
    )

    def routing_getter():
        return routing_ctx

    replaced = inject_multi_expert_lora(
        model=model,
        num_total_experts=num_total_experts,
        cfg=lora_cfg,
        target_keywords=target_keywords,
        routing_getter=routing_getter,
    )
    if len(replaced) == 0:
        print("[WARN] No Linear layers replaced. Check target keywords and architecture.")
    else:
        print(f"[INFO] Replaced {len(replaced)} Linear layers with MultiExpertLoRA wrappers.")

    # Enable gradient checkpointing if possible
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.train()
    return LoadedModel(model=model, tokenizer=tokenizer, device=device, routing_ctx=routing_ctx)
