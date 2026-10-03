"""plots.py — Vẽ đồ thị huấn luyện cho từng thí nghiệm và đồ thị so sánh nhóm."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    cfg = result["cfg"]
    hist = result["history"]
    summary = result["summary"]
    epochs = hist["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Loss
    ax = axes[0]
    ax.plot(epochs, hist["train_loss"], label="Train Loss", color="royalblue", lw=1.8)
    ax.plot(epochs, hist["val_loss"], label="Val Loss", color="crimson", lw=1.8)
    if "best_epoch" in summary:
        ax.axvline(summary["best_epoch"], color="gray", linestyle="--", alpha=0.7,
                   label=f"Best Ep ({summary['best_epoch']})")
    ax.set_title("Loss vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Metrics (Acc & F1)
    ax = axes[1]
    ax.plot(epochs, hist["val_acc"], label="Val Accuracy", color="forestgreen", lw=1.8)
    ax.plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="darkorange", lw=1.8)
    if "best_epoch" in summary:
        ax.axvline(summary["best_epoch"], color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Validation Accuracy & Macro-F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Grad Norm
    ax = axes[2]
    ax.plot(epochs, hist["grad_norm"], label="Grad Norm (unclipped)", color="purple", lw=1.8)
    ax.set_title("Gradient Norm vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Norm")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Super title with config details
    title_str = (
        f"Exp: {cfg.get('exp_id', 'exp')} | Opt: {cfg.get('optimizer')} (lr={cfg.get('lr')}) | "
        f"Loss: {cfg.get('loss')} | Batch: {cfg.get('batch')} | "
        f"Best Val Macro-F1: {summary.get('val_macro_f1', 0.0):.4f}"
    )
    fig.suptitle(title_str, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ 'val_loss', 'val_macro_f1', 'grad_norm') của nhiều thí nghiệm."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for res in results:
        cfg = res["cfg"]
        hist = res["history"]
        exp_id = cfg.get("exp_id", "unnamed")
        label = f"{exp_id}"
        ax.plot(hist["epoch"], hist[metric], label=label, lw=1.8)

    ax.set_title(title or f"Comparison: {metric}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
