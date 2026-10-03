"""optimizer.py — Bộ tối ưu hóa và tiện ích cắt gradient."""
from __future__ import annotations

import torch
import torch.optim as optim

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8) -> optim.Optimizer:
    """Trả về một torch.optim.Optimizer."""
    if name not in OPTIMIZERS:
        raise ValueError(f"Optimizer '{name}' không nằm trong danh sách hỗ trợ {OPTIMIZERS}")

    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd_momentum":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "adam":
        return optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == "adamw":
        return optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer: optim.Optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học."""
    if name is None:
        return None
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    raise ValueError(f"Scheduler '{name}' không được hỗ trợ")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Khi max_norm là None, dùng max_norm=float('inf') để chỉ tính chuẩn mà không cắt.
    torch.nn.utils.clip_grad_norm_ luôn trả về chuẩn L2 trước khi cắt.
    """
    params = [p for p in params if p.grad is not None]
    if len(params) == 0:
        return 0.0

    norm_bound = float("inf") if max_norm is None else float(max_norm)
    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=norm_bound)
    return float(total_norm)
