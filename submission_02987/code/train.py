"""train.py — Pipeline huấn luyện, đánh giá và chạy các thí nghiệm.

Gồm: đặt seed, hàm mất mát, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # được tinh chỉnh qua validation
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số thực > 0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    # Khi 2*tp + fp + fn == 0, f1 = 0
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(np.mean(f1))


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str, reduction: str = "mean") -> torch.Tensor:
    """"ce"  : CrossEntropy nhận logit thô và nhãn int64.
       "mse" : MSE giữa logit và one-hot của y.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    elif loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=7).float()
        return F.mse_loss(logits, y_onehot, reduction=reduction)
    else:
        raise ValueError(f"Hàm mất mát không được hỗ trợ: {loss_name}")


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    was_training = model.training
    model.eval()
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=-1))
    if was_training:
        model.train()
    return torch.cat(preds, dim=0)


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() và no_grad."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    cm = np.zeros((7, 7), dtype=np.int64)
    n = len(X)

    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name=loss_name, reduction="sum")
        total_loss += float(loss.item())

        pred = logits.argmax(dim=-1).cpu().numpy()
        y_true = yb.cpu().numpy()
        np.add.at(cm, (y_true, pred), 1)

    avg_loss = total_loss / n
    acc = float(np.trace(cm) / n)
    macro_f1 = macro_f1_from_confusion(cm)

    if was_training:
        model.train()

    return {"loss": avg_loss, "acc": acc, "macro_f1": macro_f1}


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = data["X_tr"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = str(cfg.get("init", "he"))

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)
    expected_params = EXPECTED_PARAMS.get(hidden)
    if expected_params is not None:
        assert count_params(model) == expected_params, (
            f"Param count mismatch: {count_params(model)} != {expected_params}"
        )

    optimizer = build_optimizer(
        cfg["optimizer"],
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        momentum=float(cfg.get("momentum", 0.9)),
    )

    precision = cfg.get("precision", "fp32")
    use_amp = (precision in ("fp16", "bf16")) and (device.type == "cuda")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and use_amp))

    # Đánh giá bước 0 trên tập val (trước cập nhật đầu tiên)
    step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = step0_res["loss"]

    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch", 512))
    clip_norm = cfg.get("clip_norm", None)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = None
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    diverged = False

    # Tập con train cố định để tính train_loss nhanh ở eval mode (50 000 mẫu)
    eval_tr_len = min(50_000, len(data["X_tr"]))
    eval_X_tr = data["X_tr"][:eval_tr_len]
    eval_y_tr = data["y_tr"][:eval_tr_len]

    generator = torch.Generator(device=device if device.type != "mps" else "cpu")
    generator.manual_seed(seed)

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        model.train()
        grad_norms = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size,
                                      generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name=cfg["loss"])

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            if precision == "fp16" and use_amp:
                scaler.scale(loss).backward()
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            grad_norms.append(gn)

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0

        if diverged:
            print(f"[{cfg['exp_id']}] Diverged at epoch {epoch}!")
            break

        # Đánh giá cuối epoch (ở chế độ eval())
        tr_res = evaluate(model, eval_X_tr, eval_y_tr, loss_name=cfg["loss"])
        val_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])

        mean_gn = float(np.mean(grad_norms)) if grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_acc"].append(val_res["acc"])
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(epoch_time)

        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_acc = val_res["acc"]
            best_val_macro_f1 = val_res["macro_f1"]

    peak_mem_MB = 0.0
    if device.type == "cuda":
        peak_mem_MB = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    final_train_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else float("nan")
    avg_epoch_time = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_macro_f1),
        "time_per_epoch_s": float(avg_epoch_time),
        "peak_mem_MB": float(peak_mem_MB),
        "diverged": diverged,
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề row_id,pred."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"row_id": row_id, "pred": preds})
    df.to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    device = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    model = MLP(hidden=hidden, dropout=dropout, init=cfg.get("init", "he")).to(device)
    model.load_state_dict(result["best_state"])

    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
