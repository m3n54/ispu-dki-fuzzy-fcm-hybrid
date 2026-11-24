"""Publication-grade Indonesian visualizations (300 DPI) for ISPU Fuzzy modeling."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.data_loader import FEATURE_COLS, FEATURE_LABELS, StandardScalerNumpy
from src.evaluation import CATEGORY_ORDER, ISPU_CATEGORIES, confusion_matrix_categories
from src.sugeno_hybrid import TakagiSugenoHybrid

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)

CLUSTER_COLORS = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02"]


def generate_all_figures(
    figures_dir: str | Path,
    paired_df: pd.DataFrame,
    split: dict[str, Any],
    sweep_rows: list[dict[str, Any]],
    fcm_jm_history: list[float],
    fcm_u_train: np.ndarray,
    fcm_centers_scaled: np.ndarray,
    cluster_labels: list[str],
    ts_model: TakagiSugenoHybrid,
    ablation_test: dict[str, dict[str, float]],
    y_test_pred_final: np.ndarray,
) -> list[str]:
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    scaler: StandardScalerNumpy = split["scaler"]
    X_train = split["X_train"]
    X_train_scaled = split["X_train_scaled"]
    y_test = split["y_test"]

    # ------------------------------------------------------------------
    # Fig 1: Data Distribution & Category Composition
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    box_data = [paired_df[c].to_numpy(dtype=float) for c in FEATURE_COLS]
    bp = axes[0].boxplot(
        box_data,
        labels=[FEATURE_LABELS[c] for c in FEATURE_COLS],
        patch_artist=True,
        medianprops={"color": "#B71C1C", "linewidth": 1.8},
    )
    box_palette = ["#42A5F5", "#EF5350", "#66BB6A", "#FFA726", "#AB47BC", "#26A69A"]
    for patch, color in zip(bp["boxes"], box_palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title("Distribusi Sub-Indeks 6 Polutan Harian (SPKU DKI Jakarta)")
    axes[0].set_ylabel("Nilai Sub-Indeks ISPU")

    cat_counts = (
        paired_df["target_category_next"]
        .str.strip()
        .str.title()
        .replace({"Tidak Sehat": "Tidak Sehat", "Sangat Tidak Sehat": "Sangat Tidak Sehat"})
        .value_counts()
    )
    cats_present = [c for c in ["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat"] if c in cat_counts.index]
    vals = [int(cat_counts[c]) for c in cats_present]
    cat_color_map = {str(item["name"]): str(item["color"]) for item in ISPU_CATEGORIES}
    bars = axes[1].bar(
        cats_present,
        vals,
        color=[cat_color_map.get(c, "#607D8B") for c in cats_present],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.88,
    )
    for b, v in zip(bars, vals):
        pct = 100.0 * v / max(len(paired_df), 1)
        axes[1].text(b.get_x() + b.get_width() / 2, v + 8, f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Komposisi Kategori Target ISPU(t+1) (Permen LHK P.14/2020)")
    axes[1].set_ylabel("Jumlah Pasangan Hari (t -> t+1)")
    axes[1].set_ylim(0, max(vals) * 1.18)

    fig.tight_layout()
    p1 = out_dir / "fig1_data_distribution.png"
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p1))

    # ------------------------------------------------------------------
    # Fig 2: Cluster Count Selection (PC, PE, XB, Val RMSE) + FCM J_m Convergence
    # ------------------------------------------------------------------
    c_vals = [r["c"] for r in sweep_rows]
    pc_vals = [r["pc"] for r in sweep_rows]
    xb_vals = [r["xb"] for r in sweep_rows]
    val_rmses = [r["lse_val_rmse"] for r in sweep_rows]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    axes[0].plot(c_vals, pc_vals, marker="o", color="#1565C0", linewidth=2, label="Partition Coeff (PC)")
    axes[0].set_title("Validitas Partisi FCM (PC)")
    axes[0].set_xlabel("Jumlah Cluster (c)")
    axes[0].set_ylabel("Nilai PC (lebih tinggi lebih baik)")
    axes[0].set_xticks(c_vals)

    axes[1].plot(c_vals, xb_vals, marker="s", color="#D84315", linewidth=2, label="Xie-Beni (XB)")
    ax1_twin = axes[1].twinx()
    ax1_twin.plot(c_vals, val_rmses, marker="^", linestyle="--", color="#2E7D32", linewidth=2, label="Val RMSE (TSK-1 LSE)")
    ax1_twin.grid(False)
    axes[1].set_title("Xie-Beni Index & Validasi RMSE vs c")
    axes[1].set_xlabel("Jumlah Cluster (c)")
    axes[1].set_ylabel("Xie-Beni Index (lebih rendah lebih baik)", color="#D84315")
    ax1_twin.set_ylabel("Validation RMSE", color="#2E7D32")
    axes[1].set_xticks(c_vals)

    iters = list(range(1, len(fcm_jm_history) + 1))
    axes[2].plot(iters, fcm_jm_history, color="#6A1B9A", linewidth=2)
    axes[2].set_title(f"Konvergensi Fungsi Objektif FCM (c={ts_model.n_clusters})")
    axes[2].set_xlabel("Iterasi")
    axes[2].set_ylabel("Fungsi Objektif J_m")

    fig.tight_layout()
    p2 = out_dir / "fig2_cluster_selection.png"
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p2))

    # ------------------------------------------------------------------
    # Fig 3: FCM Clusters PCA 2D + Original-Scale Pollutant Profile
    # ------------------------------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train_scaled)
    ctr_pca = pca.transform(fcm_centers_scaled)
    dom_cluster = np.argmax(fcm_u_train, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for i in range(ts_model.n_clusters):
        mask = dom_cluster == i
        axes[0].scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=22,
            alpha=0.6,
            color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
            label=cluster_labels[i],
        )
        axes[0].scatter(
            ctr_pca[i, 0],
            ctr_pca[i, 1],
            s=200,
            marker="*",
            color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )
    axes[0].set_title(
        f"Visualisasi Cluster FCM pada Proyeksi PCA 2D (Var: {pca.explained_variance_ratio_.sum()*100:.1f}%)"
    )
    axes[0].set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].legend(loc="best", fontsize=8.5)

    centers_orig = scaler.inverse_transform(fcm_centers_scaled)
    x_idx = np.arange(len(FEATURE_COLS))
    width = 0.8 / ts_model.n_clusters
    for i in range(ts_model.n_clusters):
        offset = (i - (ts_model.n_clusters - 1) / 2.0) * width
        axes[1].bar(
            x_idx + offset,
            centers_orig[i],
            width=width,
            color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.88,
            label=cluster_labels[i],
        )
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels([FEATURE_LABELS[c] for c in FEATURE_COLS])
    axes[1].set_ylabel("Nilai Pusat Cluster (Skala ISPU Asli)")
    axes[1].set_title("Profil Polutan Pusat Cluster FCM (Data Latih)")
    axes[1].legend(loc="best", fontsize=8.5)

    fig.tight_layout()
    p3 = out_dir / "fig3_fcm_clusters_pca.png"
    fig.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p3))

    # ------------------------------------------------------------------
    # Fig 4: Gaussian Membership Functions (Initial FCM vs Hybrid-Tuned)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0))
    axes_flat = axes.ravel()
    init_c_orig = scaler.inverse_transform(ts_model.init_centers_)
    opt_c_orig = scaler.inverse_transform(ts_model.centers_)
    init_s_orig = ts_model.init_sigmas_ * scaler.std_[np.newaxis, :]
    opt_s_orig = ts_model.sigmas_ * scaler.std_[np.newaxis, :]

    for j, col in enumerate(FEATURE_COLS):
        ax = axes_flat[j]
        lo = max(0.0, float(np.percentile(X_train[:, j], 1)) - 5.0)
        hi = float(np.percentile(X_train[:, j], 99)) + 10.0
        grid = np.linspace(lo, hi, 300)
        for i in range(ts_model.n_clusters):
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            mu_init = np.exp(-0.5 * ((grid - init_c_orig[i, j]) / init_s_orig[i, j]) ** 2)
            mu_opt = np.exp(-0.5 * ((grid - opt_c_orig[i, j]) / opt_s_orig[i, j]) ** 2)
            ax.plot(grid, mu_init, linestyle="--", linewidth=1.2, alpha=0.55, color=color)
            ax.plot(
                grid,
                mu_opt,
                linestyle="-",
                linewidth=2.0,
                color=color,
                label=f"C{i+1}" if j == 0 else None,
            )
        ax.set_title(f"Fungsi Keanggotaan Gaussian: {FEATURE_LABELS[col]}")
        ax.set_xlabel(f"Sub-Indeks {FEATURE_LABELS[col]}")
        ax.set_ylabel("Derajat Keanggotaan (μ)")
        ax.set_ylim(-0.04, 1.08)
    axes_flat[0].legend(title="Garis putus=FCM Awal\nGaris solid=Hybrid", fontsize=8, loc="upper right")
    fig.suptitle(
        "Perbandingan Fungsi Keanggotaan Premis Gaussian: Inisialisasi FCM vs Hasil Hybrid Learning",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    p4 = out_dir / "fig4_membership_functions.png"
    fig.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p4))

    # ------------------------------------------------------------------
    # Fig 5: Hybrid Learning Training & Validation Curve
    # ------------------------------------------------------------------
    hist_df = pd.DataFrame(ts_model.history_)
    best_row = hist_df.loc[hist_df["val_rmse"].idxmin()]

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(hist_df["epoch"], hist_df["train_rmse"], label="Train RMSE", color="#1565C0", linewidth=2.0)
    ax.plot(hist_df["epoch"], hist_df["val_rmse"], label="Validation RMSE", color="#D84315", linewidth=2.0)
    ax.scatter(
        [best_row["epoch"]],
        [best_row["val_rmse"]],
        color="#2E7D32",
        s=95,
        zorder=5,
        label=f"Best Checkpoint (Epoch {int(best_row['epoch'])}, Val RMSE={best_row['val_rmse']:.3f})",
    )
    ax.set_title("Kurva Konvergensi Hybrid Learning (LSE + Gradient Descent)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (Skala ISPU)")
    ax.legend(loc="best")
    fig.tight_layout()
    p5 = out_dir / "fig5_hybrid_training_curve.png"
    fig.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p5))

    # ------------------------------------------------------------------
    # Fig 6: Ablation & Baseline Comparison on Test Set
    # ------------------------------------------------------------------
    model_names = list(ablation_test.keys())
    rmses = [ablation_test[k]["rmse"] for k in model_names]
    maes = [ablation_test[k]["mae"] for k in model_names]
    r2s = [ablation_test[k]["r2"] for k in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    x = np.arange(len(model_names))
    w = 0.36
    b1 = axes[0].bar(x - w / 2, rmses, width=w, label="RMSE", color="#1E88E5", edgecolor="black", linewidth=0.5)
    b2 = axes[0].bar(x + w / 2, maes, width=w, label="MAE", color="#FB8C00", edgecolor="black", linewidth=0.5)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=15, ha="right", fontsize=8.5)
    axes[0].set_ylabel("Error (Skala ISPU, lebih rendah lebih baik)")
    axes[0].set_title("Perbandingan Error (RMSE & MAE) pada Data Uji (Test Set)")
    axes[0].set_ylim(0, max(rmses) * 1.18)
    axes[0].legend()

    bar_colors = ["#90A4AE", "#78909C", "#FFB74D", "#42A5F5", "#2E7D32"][: len(model_names)]
    b3 = axes[1].bar(x, r2s, width=0.52, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(b3, r2s):
        axes[1].text(bar.get_x() + bar.get_width() / 2, max(val, 0.0) + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=15, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Koefisien Determinasi (R², lebih tinggi lebih baik)")
    axes[1].set_title("Perbandingan R² pada Data Uji (Studi Ablasi)")
    axes[1].set_ylim(min(0.0, min(r2s) - 0.1), min(1.0, max(r2s) + 0.14))

    fig.tight_layout()
    p6 = out_dir / "fig6_ablation_comparison.png"
    fig.savefig(p6, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p6))

    # ------------------------------------------------------------------
    # Fig 7: Actual vs Predicted + Time Series + Confusion Matrix
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    # Panel A: Scatter
    axes[0].scatter(y_test, y_test_pred_final, alpha=0.55, s=24, color="#1565C0", edgecolor="white", linewidth=0.3)
    lim_min = min(float(y_test.min()), float(y_test_pred_final.min())) - 5
    lim_max = max(float(y_test.max()), float(y_test_pred_final.max())) + 5
    axes[0].plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1.6, label="Garis Ideal y = ŷ")
    for thresh in [50, 100]:
        if lim_min < thresh < lim_max:
            axes[0].axvline(thresh, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
            axes[0].axhline(thresh, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    axes[0].set_xlim(lim_min, lim_max)
    axes[0].set_ylim(lim_min, lim_max)
    axes[0].set_xlabel("ISPU(t+1) Aktual")
    axes[0].set_ylabel("ISPU(t+1) Prediksi")
    axes[0].set_title("Aktual vs Prediksi (Test Set)")
    axes[0].legend(loc="upper left", fontsize=8.5)

    # Panel B: Station Time-Series slice (first station with >= 20 test points)
    test_df = split["test"].copy()
    test_df["y_pred"] = y_test_pred_final
    main_station = test_df["stasiun"].value_counts().index[0]
    st_df = test_df[test_df["stasiun"] == main_station].sort_values("tanggal")
    axes[1].plot(st_df["tanggal"], st_df["target_ispu_next"], label="Aktual", color="#263238", linewidth=1.8)
    axes[1].plot(st_df["tanggal"], st_df["y_pred"], label="Prediksi TSK-Hybrid", color="#D84315", linestyle="--", linewidth=1.8)
    axes[1].axhspan(0, 50, color="#E8F5E9", alpha=0.45)
    axes[1].axhspan(50, 100, color="#E3F2FD", alpha=0.35)
    axes[1].axhspan(100, max(120.0, float(st_df["target_ispu_next"].max()) + 10), color="#FFF3E0", alpha=0.35)
    axes[1].set_title(f"Deret Waktu Test Set ({main_station.split('(')[0].strip()})")
    axes[1].set_ylabel("Skor ISPU(t+1)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(loc="best", fontsize=8.5)

    # Panel C: Confusion Matrix Heatmap
    cm = confusion_matrix_categories(y_test, y_test_pred_final)
    im = axes[2].imshow(cm.to_numpy(), cmap="Blues", aspect="auto")
    axes[2].set_xticks(range(len(cm.columns)))
    axes[2].set_yticks(range(len(cm.index)))
    axes[2].set_xticklabels(cm.columns, rotation=20, ha="right", fontsize=8.5)
    axes[2].set_yticklabels(cm.index, fontsize=8.5)
    axes[2].set_xlabel("Kategori Prediksi")
    axes[2].set_ylabel("Kategori Aktual")
    axes[2].set_title("Matriks Kebingungan Kategori Permen LHK")
    cm_arr = cm.to_numpy()
    thresh_cm = cm_arr.max() / 2.0 if cm_arr.size else 0.0
    for r in range(cm_arr.shape[0]):
        for c_idx in range(cm_arr.shape[1]):
            val = int(cm_arr[r, c_idx])
            axes[2].text(
                c_idx,
                r,
                str(val),
                ha="center",
                va="center",
                color="white" if val > thresh_cm else "black",
                fontweight="bold",
            )
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    p7 = out_dir / "fig7_actual_vs_predicted.png"
    fig.savefig(p7, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p7))

    return generated
