"""
Common Training Utilities for Multi-Task BERT Fine-Tuning
=========================================================
Provides optimized training primitives:
- Native Mixed Precision (BFloat16 / Float16 with GradScaler)
- Gradient Accumulation (simulating large batch sizes without VRAM spikes)
- Differential Parameter-Group Weight Decay (skipping biases & LayerNorm)
- Cosine / Linear Warmup Schedulers
- Gradient Norm Clipping
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.optim import AdamW


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def check_hub_connectivity(host: str = "huggingface.co", port: int = 443, timeout: float = 0.5) -> bool:
    """Quick socket check to verify if external Hugging Face hub network is reachable."""
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False


def setup_device_and_autocast() -> Tuple[torch.device, Optional[torch.dtype], bool]:
    """
    Detects hardware capabilities and selects optimal precision:
    - Ampere / Hopper (e.g. H100, A100): Native BFloat16 (no scaler required)
    - Older GPUs (e.g. T4, V100): Float16 with GradScaler
    - CPU / MPS: Float32
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        if torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
            use_grad_scaler = False
            precision_name = "BFloat16"
        else:
            autocast_dtype = torch.float16
            use_grad_scaler = True
            precision_name = "Float16 (with GradScaler)"
    else:
        device = torch.device("cpu")
        autocast_dtype = None
        use_grad_scaler = False
        precision_name = "Float32 (CPU)"

    print(f"[Hardware] Device: {device} | Precision: {precision_name}")
    return device, autocast_dtype, use_grad_scaler


def get_device_and_amp_dtype() -> Tuple[torch.device, torch.dtype]:
    """
    Convenience wrapper returning (device, autocast_dtype).
    Defaults to BFloat16 on supported GPUs (Ampere/Hopper), Float16 on others, or Float32 on CPU.
    """
    device, autocast_dtype, _ = setup_device_and_autocast()
    if autocast_dtype is None:
        autocast_dtype = torch.float32
    return device, autocast_dtype


def build_optimizer_with_weight_decay(
    model: nn.Module,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    eps: float = 1e-8,
) -> AdamW:
    """
    Creates an AdamW optimizer applying weight decay only to weight matrices,
    excluding biases and LayerNorm parameters to prevent under-fitting.
    """
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=eps)
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    total_steps: Optional[int] = None,
    warmup_ratio: float = 0.1,
    scheduler_type: str = "cosine",
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
):
    """
    Constructs a learning rate scheduler with warmup:
    - 'cosine': Smooth cosine decay to 0 (typically better generalization)
    - 'linear': Standard linear decay
    """
    num_warmup_steps = int(total_steps * warmup_ratio)
    final_total_steps = total_steps or num_training_steps or 100
    if num_warmup_steps is None:
        final_warmup_steps = int(final_total_steps * warmup_ratio)
    else:
        final_warmup_steps = num_warmup_steps

    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            num_warmup_steps=final_warmup_steps,
            num_training_steps=final_total_steps,
        )
    else:
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            num_warmup_steps=final_warmup_steps,
            num_training_steps=final_total_steps,
        )


def save_training_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    output_dir: str,
    epoch: int = 1,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Saves fine-tuned model, tokenizer, and metadata checkpoint.
    """
    import os
    import json
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)

    meta = {
        "epoch": epoch,
        "metrics": metrics or {}
    }
    with open(os.path.join(output_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Checkpoint] Successfully saved to {output_dir}")
