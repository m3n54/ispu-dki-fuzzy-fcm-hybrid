"""Evaluation metrics, Permen LHK No. P.14/2020 category mapping, and conventional baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd

ISPU_CATEGORIES: list[dict[str, object]] = [
    {
        "name": "Baik",
        "min": 0.0,
        "max": 50.0,
        "color": "#2E7D32",
        "bg": "#E8F5E9",
        "advisory": "Kualitas udara sangat baik dan tidak memberikan efek buruk bagi kesehatan manusia, hewan, maupun tumbuhan.",
    },
    {
        "name": "Sedang",
        "min": 50.0,
        "max": 100.0,
        "color": "#1565C0",
        "bg": "#E3F2FD",
        "advisory": "Kualitas udara masih dapat diterima pada kesehatan manusia, namun kelompok sensitif disarankan mengurangi aktivitas berat di luar ruang.",
    },
    {
        "name": "Tidak Sehat",
        "min": 100.0,
        "max": 200.0,
        "color": "#EF6C00",
        "bg": "#FFF3E0",
        "advisory": "Udara bersifat merugikan bagi kelompok sensitif dan dapat memicu gangguan pernapasan ringan pada masyarakat umum. Gunakan masker saat beraktivitas di luar.",
    },
    {
        "name": "Sangat Tidak Sehat",
        "min": 200.0,
        "max": 300.0,
        "color": "#C62828",
        "bg": "#FFEBEE",
        "advisory": "Dapat meningkatkan risiko gangguan kesehatan serius pada populasi umum. Hindari aktivitas fisik di luar ruangan.",
    },
    {
        "name": "Berbahaya",
        "min": 300.0,
        "max": float("inf"),
        "color": "#4A148C",
        "bg": "#F3E5F5",
        "advisory": "Kondisi darurat polusi udara yang berisiko serius bagi seluruh populasi. Tetap berada di dalam ruangan dengan penyaring udara.",
    },
]

CATEGORY_ORDER: list[str] = [str(c["name"]) for c in ISPU_CATEGORIES]
CATEGORY_RANK: dict[str, int] = {name: idx for idx, name in enumerate(CATEGORY_ORDER)}


def ispu_to_category(score: float) -> str:
    """Map continuous ISPU score to official Permen LHK No. P.14/2020 category."""
    val = float(score)
    if val <= 50.0:
        return "Baik"
    if val <= 100.0:
        return "Sedang"
    if val <= 200.0:
        return "Tidak Sehat"
    if val <= 300.0:
        return "Sangat Tidak Sehat"
    return "Berbahaya"


def get_category_info(score: float) -> dict[str, object]:
    name = ispu_to_category(score)
    for item in ISPU_CATEGORIES:
        if item["name"] == name:
            return dict(item)
    return dict(ISPU_CATEGORIES[-1])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    err = yp - yt
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(yt), 1.0)) * 100.0)

    true_cats = [ispu_to_category(v) for v in yt]
    pred_cats = [ispu_to_category(v) for v in yp]
    cat_acc = float(np.mean([t == p for t, p in zip(true_cats, pred_cats)]))
    adj_acc = float(
        np.mean([abs(CATEGORY_RANK[t] - CATEGORY_RANK[p]) <= 1 for t, p in zip(true_cats, pred_cats)])
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "category_accuracy": cat_acc,
        "adjacent_category_accuracy": adj_acc,
    }


def evaluate_persistence_baseline(df_split: pd.DataFrame) -> np.ndarray:
    """Conventional persistence baseline: ISPU(t+1) = ISPU(t)."""
    return df_split["max"].to_numpy(dtype=float)


def fit_predict_ols(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, ridge: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Conventional global Multiple Linear Regression (OLS) baseline."""
    X_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train, dtype=float)
    X_ev = np.asarray(X_eval, dtype=float)

    A_tr = np.hstack([np.ones((X_tr.shape[0], 1), dtype=float), X_tr])
    A_ev = np.hstack([np.ones((X_ev.shape[0], 1), dtype=float), X_ev])
    reg = ridge * np.eye(A_tr.shape[1], dtype=float)
    beta = np.linalg.solve(A_tr.T @ A_tr + reg, A_tr.T @ y_tr)
    return A_tr @ beta, A_ev @ beta


def confusion_matrix_categories(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return ordered category confusion matrix DataFrame (rows=Aktual, cols=Prediksi)."""
    true_cats = [ispu_to_category(v) for v in y_true]
    pred_cats = [ispu_to_category(v) for v in y_pred]
    present = [c for c in CATEGORY_ORDER if (c in true_cats or c in pred_cats)]
    mat = pd.DataFrame(0, index=present, columns=present, dtype=int)
    for t, p in zip(true_cats, pred_cats):
        mat.loc[t, p] += 1
    return mat
