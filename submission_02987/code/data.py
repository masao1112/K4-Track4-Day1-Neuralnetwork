"""data.py — Chuẩn bị và xử lý dữ liệu cho bài toán CoverType.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    train_data = np.load(f"{processed_dir}/train.npz")
    eval_data = np.load(f"{processed_dir}/eval.npz")

    X_train_full = train_data["X"].astype(np.float32)
    y_train_full = train_data["y"].astype(np.int64)

    X_eval = eval_data["X"].astype(np.float32)
    y_eval = eval_data["y"].astype(np.int64)
    eval_row_id = eval_data["row_id"].astype(np.int64)

    assert X_train_full.shape[1] == 54 and X_eval.shape[1] == 54, "Số đặc trưng phải là 54"
    assert X_train_full.dtype == np.float32 and X_eval.dtype == np.float32
    assert y_train_full.dtype == np.int64 and y_eval.dtype == np.int64

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )
    return X_tr, X_val, y_tr, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    """
    numeric_tr = X_tr[:, :N_NUMERIC]
    mean = np.mean(numeric_tr, axis=0)
    std = np.std(numeric_tr, axis=0)
    # Tránh chia cho 0
    std = np.where(std == 0, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_out = X.copy()
    X_out[:, :N_NUMERIC] = (X_out[:, :N_NUMERIC] - mean) / std
    return X_out


def prepare_data(device: str | torch.device, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader)."""
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, X_val, y_tr, y_val = make_val_split(X_train_full, y_train_full, val_fraction=val_fraction, seed=seed)

    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    dev = torch.device(device)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=dev)
    y_tr_t = torch.tensor(y_tr, dtype=torch.int64, device=dev)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=dev)
    y_val_t = torch.tensor(y_val, dtype=torch.int64, device=dev)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=dev)
    y_eval_t = torch.tensor(y_eval, dtype=torch.int64, device=dev)

    # Thống kê cơ bản
    counts_val = torch.bincount(y_val_t, minlength=7)
    majority_acc = (counts_val.max().float() / len(y_val_t)).item()
    print(f"Data prepared on {device}:")
    print(f"  Train: X={X_tr_t.shape}, y={y_tr_t.shape}")
    print(f"  Val:   X={X_val_t.shape}, y={y_val_t.shape}")
    print(f"  Eval:  X={X_eval_t.shape}, y={y_eval_t.shape}")
    print(f"  Majority baseline accuracy on Val: {majority_acc:.4f} (lớp {counts_val.argmax().item()})")

    return {
        "X_tr": X_tr_t,
        "y_tr": y_tr_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_eval": X_eval_t,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader."""
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
