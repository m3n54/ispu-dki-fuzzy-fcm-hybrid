"""End-to-end experiment runner: data cleaning, FCM sweep, Hybrid Learning, ablation, and figures."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (
    FEATURE_COLS,
    FEATURE_LABELS,
    build_next_day_dataset,
    chronological_split,
    clean_ispu_dataframe,
    load_raw_ispu,
)
from src.evaluation import (
    compute_metrics,
    confusion_matrix_categories,
    evaluate_persistence_baseline,
    fit_predict_ols,
    ispu_to_category,
)
from src.fcm import FuzzyCMeans
from src.sugeno_hybrid import TakagiSugenoHybrid
from src.visualization import generate_all_figures


def make_cluster_labels(centers_orig: np.ndarray) -> list[str]:
    n_c = centers_orig.shape[0]
    if n_c == 2:
        names = ["Polusi Rendah", "Polusi Tinggi"]
    elif n_c == 3:
        names = ["Polusi Rendah", "Polusi Sedang", "Polusi Tinggi"]
    elif n_c == 4:
        names = ["Polusi Rendah", "Polusi Sedang-Rendah", "Polusi Sedang-Tinggi", "Polusi Tinggi"]
    else:
        names = [f"Level Polusi {i + 1}" for i in range(n_c)]
    labels = []
    for i in range(n_c):
        dom_idx = int(np.argmax(centers_orig[i]))
        dom_feat = FEATURE_LABELS[FEATURE_COLS[dom_idx]]
        labels.append(f"Cluster {i + 1}: {names[i]} (Dom. {dom_feat})")
    return labels


def main() -> None:
    raw_dir = ROOT / "data" / "raw"
    proc_dir = ROOT / "data" / "processed"
    art_dir = ROOT / "artifacts"
    fig_dir = art_dir / "figures"
    proc_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load, clean, pair t -> t+1, save processed dataset
    raw_df = load_raw_ispu(raw_dir)
    clean_df, clean_stats = clean_ispu_dataframe(raw_df)
    paired_df = build_next_day_dataset(clean_df)
    paired_path = proc_dir / "ispu_dki_nextday.csv"
    paired_df.to_csv(paired_path, index=False)

    # 2. Chronological split (70% / 15% / 15%)
    split = chronological_split(paired_df, train_ratio=0.70, val_ratio=0.15)
    X_tr_s, y_tr = split["X_train_scaled"], split["y_train"]
    X_va_s, y_va = split["X_val_scaled"], split["y_val"]
    X_te_s, y_te = split["X_test_scaled"], split["y_test"]
    scaler = split["scaler"]

    print(f"Dataset stats: {clean_stats}")
    print(f"Paired next-day rows: {len(paired_df)}")
    print(f"Split boundaries: {split['boundaries']}")

    # 3. Sweep c in {2, 3, 4, 5, 6} and ridge_lambda on Validation Set
    c_candidates = [2, 3, 4, 5, 6]
    lambda_candidates = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    sweep_rows = []
    for c in c_candidates:
        fcm_c = FuzzyCMeans(n_clusters=c, m=2.0, max_iter=300, tol=1e-5, random_state=42).fit(X_tr_s)
        val_m = fcm_c.validity_metrics(X_tr_s)

        best_lam = lambda_candidates[0]
        best_val_rmse = float("inf")
        for lam in lambda_candidates:
            ts_tmp = TakagiSugenoHybrid(n_clusters=c, n_features=len(FEATURE_COLS), ridge_lambda=lam)
            ts_tmp.initialize_from_fcm(X_tr_s, fcm_c.u_, fcm_c.centers_, m=2.0)
            ts_tmp.fit_lse(X_tr_s, y_tr, ridge_lambda=lam)
            va_pred = ts_tmp.predict(X_va_s)
            va_rmse = float(np.sqrt(np.mean((y_va - va_pred) ** 2)))
            if va_rmse < best_val_rmse:
                best_val_rmse = va_rmse
                best_lam = lam

        sweep_rows.append(
            {
                "c": int(c),
                **val_m,
                "best_lambda": float(best_lam),
                "lse_val_rmse": float(best_val_rmse),
            }
        )

    sweep_df = pd.DataFrame(sweep_rows)
    print("\n--- Cluster & Lambda Sweep ---")
    print(sweep_df.to_string(index=False))

    # Select optimal c* with lowest Validation RMSE (tie-breaker: lower Xie-Beni)
    best_row = min(sweep_rows, key=lambda r: (round(r["lse_val_rmse"], 4), r["xb"]))
    c_opt = int(best_row["c"])
    lam_opt = float(best_row["best_lambda"])
    print(f"\nSelected optimal c*={c_opt}, ridge_lambda*={lam_opt}")

    # 4. Fit Final FCM on Train and refine ridge_lambda via Stage-2 Hybrid Validation RMSE
    fcm_opt = FuzzyCMeans(n_clusters=c_opt, m=2.0, max_iter=300, tol=1e-5, random_state=42).fit(X_tr_s)
    centers_orig = scaler.inverse_transform(fcm_opt.centers_)
    cluster_labels = make_cluster_labels(centers_orig)

    best_hybrid_val = float("inf")
    for lam_cand in [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
        ts_cand = TakagiSugenoHybrid(n_clusters=c_opt, n_features=len(FEATURE_COLS), ridge_lambda=lam_cand)
        ts_cand.initialize_from_fcm(X_tr_s, fcm_opt.u_, fcm_opt.centers_, m=2.0)
        ts_cand.fit_hybrid(X_tr_s, y_tr, X_va_s, y_va, epochs=150, lr=0.05, ridge_lambda=lam_cand, patience=30)
        cand_val_rmse = float(np.sqrt(np.mean((y_va - ts_cand.predict(X_va_s)) ** 2)))
        if cand_val_rmse < best_hybrid_val:
            best_hybrid_val = cand_val_rmse
            lam_opt = float(lam_cand)
    print(f"Refined ridge_lambda* via Hybrid Validation RMSE = {lam_opt} (Val RMSE = {best_hybrid_val:.4f})")

    # 5. Fit all 5 Ablation & Benchmark Models
    # Baseline 1: Persistence
    pers_tr = evaluate_persistence_baseline(split["train"])
    pers_va = evaluate_persistence_baseline(split["val"])
    pers_te = evaluate_persistence_baseline(split["test"])

    # Baseline 2: Multiple Linear Regression (OLS)
    ols_tr, ols_va = fit_predict_ols(X_tr_s, y_tr, X_va_s)
    _, ols_te = fit_predict_ols(X_tr_s, y_tr, X_te_s)

    # Stage 0: FCM + Sugeno Orde-0
    ts_stage0 = TakagiSugenoHybrid(n_clusters=c_opt, n_features=len(FEATURE_COLS), ridge_lambda=lam_opt)
    ts_stage0.initialize_from_fcm(X_tr_s, fcm_opt.u_, fcm_opt.centers_, m=2.0)
    ts_stage0.fit_zero_order(X_tr_s, y_tr)
    s0_tr = ts_stage0.predict(X_tr_s)
    s0_va = ts_stage0.predict(X_va_s)
    s0_te = ts_stage0.predict(X_te_s)

    # Stage 1: FCM + Sugeno Orde-1 + Regularized LSE
    ts_stage1 = TakagiSugenoHybrid(n_clusters=c_opt, n_features=len(FEATURE_COLS), ridge_lambda=lam_opt)
    ts_stage1.initialize_from_fcm(X_tr_s, fcm_opt.u_, fcm_opt.centers_, m=2.0)
    ts_stage1.fit_lse(X_tr_s, y_tr, ridge_lambda=lam_opt)
    s1_tr = ts_stage1.predict(X_tr_s)
    s1_va = ts_stage1.predict(X_va_s)
    s1_te = ts_stage1.predict(X_te_s)

    # Stage 2: FCM + Sugeno Orde-1 + Hybrid Learning (LSE + GD)
    ts_stage2 = TakagiSugenoHybrid(n_clusters=c_opt, n_features=len(FEATURE_COLS), ridge_lambda=lam_opt)
    ts_stage2.initialize_from_fcm(X_tr_s, fcm_opt.u_, fcm_opt.centers_, m=2.0)
    ts_stage2.fit_hybrid(X_tr_s, y_tr, X_va_s, y_va, epochs=150, lr=0.05, ridge_lambda=lam_opt, patience=30)
    s2_tr = ts_stage2.predict(X_tr_s)
    s2_va = ts_stage2.predict(X_va_s)
    s2_te = ts_stage2.predict(X_te_s)

    models_preds = {
        "Persistence (ISPU_t)": (pers_tr, pers_va, pers_te),
        "Regresi Linear (OLS)": (ols_tr, ols_va, ols_te),
        "Tahap 0: FCM + Sugeno-0": (s0_tr, s0_va, s0_te),
        "Tahap 1: FCM + TSK-1 (LSE)": (s1_tr, s1_va, s1_te),
        "Tahap 2: FCM + TSK-1 (Hybrid)": (s2_tr, s2_va, s2_te),
    }

    ablation_results: dict[str, dict[str, dict[str, float]]] = {"train": {}, "val": {}, "test": {}}
    for name, (p_tr, p_va, p_te) in models_preds.items():
        ablation_results["train"][name] = compute_metrics(y_tr, p_tr)
        ablation_results["val"][name] = compute_metrics(y_va, p_va)
        ablation_results["test"][name] = compute_metrics(y_te, p_te)

    print("\n--- Ablation & Benchmark Results (TEST SET) ---")
    test_table = pd.DataFrame(ablation_results["test"]).T
    print(test_table.round(4).to_string())

    # 6. Cluster summary & sample memberships
    dom_train = np.argmax(fcm_opt.u_, axis=0)
    cluster_rows = []
    for i in range(c_opt):
        row = {
            "cluster": i + 1,
            "label": cluster_labels[i],
            "count": int(np.sum(dom_train == i)),
            "mean_target_ispu": float(np.mean(y_tr[dom_train == i])) if np.any(dom_train == i) else 0.0,
        }
        for j, col in enumerate(FEATURE_COLS):
            row[f"center_{col}"] = float(centers_orig[i, j])
        cluster_rows.append(row)
    cluster_summary_df = pd.DataFrame(cluster_rows)
    cluster_summary_df.to_csv(art_dir / "cluster_summary.csv", index=False)

    # Sample membership rows (first 5 train observations)
    sample_u = []
    for k in range(min(5, len(split["train"]))):
        r = split["train"].iloc[k]
        sample_u.append(
            {
                "tanggal": str(r["tanggal"].date()),
                "stasiun": r["stasiun"],
                **{col: float(r[col]) for col in FEATURE_COLS},
                "memberships": [float(fcm_opt.u_[i, k]) for i in range(c_opt)],
                "dominant_cluster": int(np.argmax(fcm_opt.u_[:, k]) + 1),
            }
        )

    # 7. Select 3 real test scenarios (Baik, Sedang, Tidak Sehat / Highest) for report & GUI presets
    test_df = split["test"].copy()
    test_df["pred"] = s2_te
    test_df["abs_err"] = np.abs(test_df["pred"] - test_df["target_ispu_next"])

    scenarios = []
    for cat_label, target_cat in [
        ("Kondisi Udara Baik", "Baik"),
        ("Kondisi Sedang Harian", "Sedang"),
        ("Episode Polusi Tinggi", "Tidak Sehat"),
    ]:
        sub = test_df[test_df["target_ispu_next"].apply(ispu_to_category) == target_cat]
        if len(sub) > 0:
            chosen = sub.sort_values("abs_err").iloc[0]
        else:
            chosen = test_df.sort_values("target_ispu_next", ascending=False).iloc[0]

        idx = int(chosen.name)
        x_raw = chosen[FEATURE_COLS].to_numpy(dtype=float)
        x_sc = scaler.transform(x_raw.reshape(1, -1))[0]
        u_vec = fcm_opt.predict_membership(x_sc.reshape(1, -1))[:, 0]
        trace = ts_stage2.explain_single(x_sc)
        scenarios.append(
            {
                "preset_name": cat_label,
                "tanggal": str(chosen["tanggal"].date()),
                "next_tanggal": str(chosen["next_tanggal"].date()),
                "stasiun": str(chosen["stasiun"]),
                "inputs": {col: float(chosen[col]) for col in FEATURE_COLS},
                "current_ispu": float(chosen["max"]),
                "actual_ispu_next": float(chosen["target_ispu_next"]),
                "actual_category_next": ispu_to_category(float(chosen["target_ispu_next"])),
                "predicted_ispu_next": float(trace["y_pred"]),
                "predicted_category_next": ispu_to_category(float(trace["y_pred"])),
                "fcm_memberships": [float(v) for v in u_vec],
                "mu_matrix": trace["mu"].tolist(),
                "w": [float(v) for v in trace["w"]],
                "w_bar": [float(v) for v in trace["w_bar"]],
                "rule_outputs": [float(v) for v in trace["rule_outputs"]],
                "weighted_contributions": [float(v) for v in trace["weighted_contributions"]],
            }
        )

    # 8. Descriptive statistics of paired dataset
    desc_stats = {}
    for col in [*FEATURE_COLS, "max", "target_ispu_next"]:
        s = paired_df[col]
        desc_stats[col] = {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "q25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "q75": float(s.quantile(0.75)),
            "max": float(s.max()),
        }

    # 9. Confusion matrix & per-category accuracy on Test set
    cm_df = confusion_matrix_categories(y_te, s2_te)
    init_c_orig = scaler.inverse_transform(ts_stage2.init_centers_)
    opt_c_orig = scaler.inverse_transform(ts_stage2.centers_)
    init_s_orig = ts_stage2.init_sigmas_ * scaler.std_[np.newaxis, :]
    opt_s_orig = ts_stage2.sigmas_ * scaler.std_[np.newaxis, :]

    artifact_payload = {
        "feature_cols": FEATURE_COLS,
        "feature_labels": FEATURE_LABELS,
        "cluster_labels": cluster_labels,
        "scaler": scaler.to_dict(),
        "fcm": {
            "n_clusters": c_opt,
            "m": fcm_opt.m,
            "n_iter": fcm_opt.n_iter_,
            "centers_scaled": fcm_opt.centers_.tolist(),
            "centers_orig": centers_orig.tolist(),
            "jm_history": fcm_opt.jm_history_,
        },
        "ts_hybrid": ts_stage2.to_dict(),
        "ts_stage0": ts_stage0.to_dict(),
        "ts_stage1": ts_stage1.to_dict(),
        "scenarios": scenarios,
    }
    (art_dir / "model_artifact.json").write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")

    results_payload = {
        "clean_stats": clean_stats,
        "n_paired": int(len(paired_df)),
        "split_boundaries": split["boundaries"],
        "descriptive_stats": desc_stats,
        "target_category_counts": paired_df["target_category_next"].value_counts().to_dict(),
        "sweep_results": sweep_rows,
        "selected_c": c_opt,
        "selected_lambda": lam_opt,
        "cluster_summary": cluster_rows,
        "sample_memberships": sample_u,
        "premise_orig": {
            "init_centers": init_c_orig.tolist(),
            "opt_centers": opt_c_orig.tolist(),
            "init_sigmas": init_s_orig.tolist(),
            "opt_sigmas": opt_s_orig.tolist(),
        },
        "ablation": ablation_results,
        "test_confusion_matrix": cm_df.to_dict(),
        "scenarios": scenarios,
    }
    (art_dir / "experiment_results.json").write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    # 10. Generate all 7 publication figures
    figs = generate_all_figures(
        figures_dir=fig_dir,
        paired_df=paired_df,
        split=split,
        sweep_rows=sweep_rows,
        fcm_jm_history=fcm_opt.jm_history_,
        fcm_u_train=fcm_opt.u_,
        fcm_centers_scaled=fcm_opt.centers_,
        cluster_labels=cluster_labels,
        ts_model=ts_stage2,
        ablation_test=ablation_results["test"],
        y_test_pred_final=s2_te,
    )
    print(f"\nGenerated {len(figs)} figures in {fig_dir}")


if __name__ == "__main__":
    main()
