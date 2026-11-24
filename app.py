"""Interactive Streamlit GUI for ISPU DKI Jakarta Next-Day Early Warning System."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import FEATURE_COLS, FEATURE_LABELS, StandardScalerNumpy
from src.evaluation import get_category_info, ispu_to_category
from src.fcm import FuzzyCMeans
from src.sugeno_hybrid import TakagiSugenoHybrid

CLUSTER_COLORS = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02"]

st.set_page_config(
    page_title="Prediksi Dini ISPU DKI Jakarta — Fuzzy TSK + FCM + Hybrid",
    page_icon="🌫️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(120deg, #0F2942 0%, #1B4F72 100%);
        color: #FFFFFF;
        padding: 1.1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.45rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.35rem 0 0 0;
        font-size: 0.92rem;
        opacity: 0.92;
    }
    .decision-card {
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border-left: 6px solid;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    art = json.loads((ROOT / "artifacts" / "model_artifact.json").read_text(encoding="utf-8"))
    res = json.loads((ROOT / "artifacts" / "experiment_results.json").read_text(encoding="utf-8"))
    df = pd.read_csv(ROOT / "data" / "processed" / "ispu_dki_nextday.csv")

    scaler = StandardScalerNumpy.from_dict(art["scaler"])
    fcm = FuzzyCMeans(
        n_clusters=art["fcm"]["n_clusters"],
        m=art["fcm"]["m"],
    )
    fcm.centers_ = np.asarray(art["fcm"]["centers_scaled"], dtype=float)
    ts_model = TakagiSugenoHybrid.from_dict(art["ts_hybrid"])

    # Fit PCA on training portion of processed dataset for consistent cluster projection
    n_train = res["split_boundaries"]["n_train"]
    train_df = df.sort_values(["tanggal", "stasiun"]).iloc[:n_train]
    X_tr_scaled = scaler.transform(train_df[FEATURE_COLS].to_numpy(dtype=float))
    pca = PCA(n_components=2, random_state=42).fit(X_tr_scaled)
    X_tr_pca = pca.transform(X_tr_scaled)
    u_tr = fcm.predict_membership(X_tr_scaled)
    dom_tr = np.argmax(u_tr, axis=0)

    return art, res, scaler, fcm, ts_model, pca, X_tr_pca, dom_tr


art, res, scaler, fcm, ts_model, pca, X_tr_pca, dom_tr = load_assets()
cluster_labels: list[str] = art["cluster_labels"]
scenarios: list[dict] = art["scenarios"]

st.markdown(
    """
    <div class="main-header">
        <h1>Sistem Prediksi Dini ISPU DKI Jakarta (t+1) — Takagi-Sugeno Orde-1, FCM & Hybrid Learning</h1>
        <p>Mata Kuliah Komputasi Lunak | Informatika Universitas Diponegoro | Menza Isaiah Tampubolon (24060124140138)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Sidebar: Requirement 1 — User Inputs & Real Jakarta Presets
# ------------------------------------------------------------------
st.sidebar.header("1. Input Parameter Polutan (Hari ke-t)")
preset_names = ["Input Manual"] + [s["preset_name"] for s in scenarios]
selected_preset = st.sidebar.selectbox("Pilih Skenario Data Nyata DKI Jakarta:", preset_names, index=2 if len(scenarios) >= 2 else 0)

defaults = {"pm10": 55.0, "pm25": 78.0, "so2": 28.0, "co": 14.0, "o3": 32.0, "no2": 22.0}
preset_meta = None
if selected_preset != "Input Manual":
    for s in scenarios:
        if s["preset_name"] == selected_preset:
            defaults = s["inputs"]
            preset_meta = s
            break

user_inputs: dict[str, float] = {}
for col in FEATURE_COLS:
    label = f"{FEATURE_LABELS[col]} (Sub-Indeks)"
    user_inputs[col] = st.sidebar.slider(
        label,
        min_value=0.0,
        max_value=250.0,
        value=float(defaults[col]),
        step=1.0,
    )

x_raw = np.array([user_inputs[c] for c in FEATURE_COLS], dtype=float)
x_scaled = scaler.transform(x_raw.reshape(1, -1))[0]
current_ispu = float(np.max(x_raw))
dom_pollutant = FEATURE_LABELS[FEATURE_COLS[int(np.argmax(x_raw))]]

# Run FCM membership & Sugeno Hybrid inference
u_user = fcm.predict_membership(x_scaled.reshape(1, -1))[:, 0]
trace = ts_model.explain_single(x_scaled)
pred_ispu = float(np.clip(trace["y_pred"], 0.0, 500.0))
cat_info = get_category_info(pred_ispu)
delta_ispu = pred_ispu - current_ispu

if preset_meta is not None:
    st.sidebar.info(
        f"**Catatan Skenario Nyata:**\n"
        f"- Stasiun: {preset_meta['stasiun']}\n"
        f"- Tanggal t: {preset_meta['tanggal']} (ISPU={preset_meta['current_ispu']:.0f})\n"
        f"- Aktual t+1: **{preset_meta['actual_ispu_next']:.0f}** ({preset_meta['actual_category_next']})"
    )

# ------------------------------------------------------------------
# Top Section: Requirement 5 — Final Decision (Keputusan Akhir)
# ------------------------------------------------------------------
st.subheader("Keputusan Akhir Sistem Peringatan Dini ISPU (t+1)")
col_dec1, col_dec2, col_dec3, col_dec4 = st.columns([1.3, 1.0, 1.0, 1.8])
with col_dec1:
    st.metric(
        "Prediksi Skor ISPU Besok (t+1)",
        f"{pred_ispu:.2f}",
        delta=f"{delta_ispu:+.2f} vs hari ini",
        delta_color="inverse",
    )
with col_dec2:
    st.metric("Kategori Kualitas Udara", str(cat_info["name"]))
with col_dec3:
    st.metric("ISPU Hari Ini (t)", f"{current_ispu:.0f}", delta=f"Kritis: {dom_pollutant}", delta_color="off")
with col_dec4:
    st.markdown(
        f"""
        <div class="decision-card" style="background:{cat_info['bg']}; border-left-color:{cat_info['color']};">
            <strong style="color:{cat_info['color']}; font-size:1.05rem;">Status Mutu Udara: {cat_info['name']}</strong>
            <div style="font-size:0.88rem; margin-top:0.25rem; color:#212121;">{cat_info['advisory']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# Tabs for Requirements 2, 3, 4 + Model Evaluation
# ------------------------------------------------------------------
tab_cluster, tab_mf, tab_infer, tab_eval = st.tabs(
    [
        "2. Hasil Clustering FCM",
        "3. Fungsi Keanggotaan Gaussian",
        "4. Hasil Inferensi Fuzzy TSK-1",
        "5. Ringkasan Ablasi & Performa",
    ]
)

with tab_cluster:
    st.markdown("Visualisasi posisi input pengguna terhadap cluster **Fuzzy C-Means (FCM)** yang dibentuk dari data latih SPKU DKI Jakarta.")
    c1, c2 = st.columns([1.25, 1.0])
    with c1:
        fig_pca, ax_pca = plt.subplots(figsize=(6.8, 4.4))
        for i in range(ts_model.n_clusters):
            mask = dom_tr == i
            ax_pca.scatter(
                X_tr_pca[mask, 0],
                X_tr_pca[mask, 1],
                s=18,
                alpha=0.5,
                color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                label=cluster_labels[i],
            )
        ctr_pca = pca.transform(fcm.centers_)
        user_pca = pca.transform(x_scaled.reshape(1, -1))[0]
        for i in range(ts_model.n_clusters):
            ax_pca.scatter(
                ctr_pca[i, 0],
                ctr_pca[i, 1],
                s=180,
                marker="*",
                color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                edgecolor="black",
                linewidth=1.0,
                zorder=5,
            )
        ax_pca.scatter(
            [user_pca[0]],
            [user_pca[1]],
            s=220,
            marker="D",
            color="#D50000",
            edgecolor="white",
            linewidth=1.5,
            zorder=6,
            label="Input Pengguna Saat Ini",
        )
        ax_pca.set_title("Peta Cluster FCM (Proyeksi PCA 2D)")
        ax_pca.set_xlabel("Komponen Utama 1 (PC1)")
        ax_pca.set_ylabel("Komponen Utama 2 (PC2)")
        ax_pca.grid(True, alpha=0.3)
        ax_pca.legend(loc="best", fontsize=8)
        fig_pca.tight_layout()
        st.pyplot(fig_pca)
        plt.close(fig_pca)

    with c2:
        fig_u, ax_u = plt.subplots(figsize=(5.4, 3.2))
        y_pos = np.arange(ts_model.n_clusters)
        bars = ax_u.barh(
            y_pos,
            u_user,
            color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(ts_model.n_clusters)],
            edgecolor="black",
            linewidth=0.5,
        )
        ax_u.set_yticks(y_pos)
        ax_u.set_yticklabels([f"C{i+1}" for i in range(ts_model.n_clusters)])
        ax_u.set_xlim(0, 1.12)
        ax_u.set_xlabel("Derajat Keanggotaan FCM (u_i)")
        ax_u.set_title("Derajat Keanggotaan Input pada Setiap Cluster")
        ax_u.grid(True, axis="x", alpha=0.3)
        for b, val in zip(bars, u_user):
            ax_u.text(val + 0.02, b.get_y() + b.get_height() / 2, f"{val:.4f}", va="center", fontsize=9)
        fig_u.tight_layout()
        st.pyplot(fig_u)
        plt.close(fig_u)

        cluster_df = pd.DataFrame(res["cluster_summary"])
        cluster_df["u_input"] = u_user
        show_cols = ["cluster", "label", "u_input", "mean_target_ispu"]
        st.dataframe(
            cluster_df[show_cols].rename(
                columns={
                    "cluster": "Cluster",
                    "label": "Profil Cluster",
                    "u_input": "Membership Input",
                    "mean_target_ispu": "Rata-rata ISPU(t+1)",
                }
            ).style.format({"Membership Input": "{:.4f}", "Rata-rata ISPU(t+1)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

with tab_mf:
    st.markdown("Kurva **Fungsi Keanggotaan Premis Gaussian** hasil optimasi **Hybrid Learning** beserta posisi garis vertikal nilai input pengguna.")
    opt_c_orig = scaler.inverse_transform(ts_model.centers_)
    opt_s_orig = ts_model.sigmas_ * scaler.std_[np.newaxis, :]
    mu_mat = trace["mu"]

    fig_mf, axes_mf = plt.subplots(2, 3, figsize=(13.2, 7.2))
    for j, col in enumerate(FEATURE_COLS):
        ax = axes_mf.ravel()[j]
        lo = 0.0
        hi = max(140.0, float(np.max(opt_c_orig[:, j] + 3.0 * opt_s_orig[:, j])), float(x_raw[j]) + 15.0)
        grid = np.linspace(lo, hi, 260)
        for i in range(ts_model.n_clusters):
            y_curve = np.exp(-0.5 * ((grid - opt_c_orig[i, j]) / opt_s_orig[i, j]) ** 2)
            ax.plot(
                grid,
                y_curve,
                linewidth=2.0,
                color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                label=f"C{i+1} (μ={mu_mat[i, j]:.2f})",
            )
            ax.scatter([x_raw[j]], [mu_mat[i, j]], color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], s=45, zorder=5)
        ax.axvline(x_raw[j], color="#D50000", linestyle="--", linewidth=1.6)
        ax.set_title(f"{FEATURE_LABELS[col]} (Input = {x_raw[j]:.0f})")
        ax.set_ylim(-0.04, 1.08)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7.5)
    fig_mf.tight_layout()
    st.pyplot(fig_mf)
    plt.close(fig_mf)

    mu_df = pd.DataFrame(
        mu_mat,
        index=[cluster_labels[i] for i in range(ts_model.n_clusters)],
        columns=[FEATURE_LABELS[c] for c in FEATURE_COLS],
    )
    st.markdown("**Matriks Aktivasi Derajat Keanggotaan Premis $\\mu_{ij}(x_j)$:**")
    st.dataframe(mu_df.style.format("{:.4f}"), use_container_width=True)

with tab_infer:
    st.markdown("Rincian mekanisme **Inferensi Takagi-Sugeno Orde-1** mulai dari *firing strength* ($w_i$), bobot ternormalisasi ($\\bar{w}_i$), keluaran konsekuen linear $f_i(\\mathbf{x})$, hingga agregasi akhir.")
    rule_df = pd.DataFrame(
        {
            "Aturan": [f"R{i+1}" for i in range(ts_model.n_clusters)],
            "Profil Cluster": cluster_labels,
            "Firing Strength (w_i)": trace["w"],
            "Bobot Ternormalisasi (w_bar_i)": trace["w_bar"],
            "Konsekuen Linear f_i(x)": trace["rule_outputs"],
            "Kontribusi (w_bar_i * f_i)": trace["weighted_contributions"],
        }
    )
    st.dataframe(
        rule_df.style.format(
            {
                "Firing Strength (w_i)": "{:.4e}",
                "Bobot Ternormalisasi (w_bar_i)": "{:.4f}",
                "Konsekuen Linear f_i(x)": "{:.2f}",
                "Kontribusi (w_bar_i * f_i)": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    col_inf1, col_inf2 = st.columns([1.2, 1.0])
    with col_inf1:
        fig_inf, ax_inf = plt.subplots(figsize=(6.4, 3.6))
        x_r = np.arange(ts_model.n_clusters)
        bars_inf = ax_inf.bar(
            x_r,
            trace["weighted_contributions"],
            color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(ts_model.n_clusters)],
            edgecolor="black",
            linewidth=0.6,
        )
        ax_inf.set_xticks(x_r)
        ax_inf.set_xticklabels([f"Aturan {i+1}\n(w̄={trace['w_bar'][i]:.2f})" for i in range(ts_model.n_clusters)])
        ax_inf.set_ylabel("Kontribusi Skor ISPU")
        ax_inf.set_title("Kontribusi Terbobot Setiap Aturan Fuzzy TSK-1")
        ax_inf.grid(True, axis="y", alpha=0.3)
        for b, val in zip(bars_inf, trace["weighted_contributions"]):
            ax_inf.text(b.get_x() + b.get_width() / 2, val + 0.5, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        fig_inf.tight_layout()
        st.pyplot(fig_inf)
        plt.close(fig_inf)

    with col_inf2:
        terms = " + ".join(
            [f"({trace['w_bar'][i]:.4f} × {trace['rule_outputs'][i]:.2f})" for i in range(ts_model.n_clusters)]
        )
        st.info(
            f"**Rumus Defuzzifikasi Takagi-Sugeno (Weighted Average):**\n\n"
            f"$\\hat{{y}} = \\sum_{{i=1}}^{{{ts_model.n_clusters}}} \\bar{{w}}_i f_i(\\mathbf{{x}})$\n\n"
            f"$\\hat{{y}} = {terms} = \\mathbf{{{trace['y_pred']:.2f}}}$\n\n"
            f"**Keputusan Kategori (Permen LHK P.14/2020):** **{ispu_to_category(pred_ispu)}**"
        )

with tab_eval:
    st.markdown("Perbandingan objektif pada **Test Set kronologis** (*zero data leakage*) untuk membuktikan efektivitas **Clustering FCM** dan **Estimasi Parameter Hybrid Learning**.")
    test_tbl = pd.DataFrame(res["ablation"]["test"]).T
    test_tbl = test_tbl[["rmse", "mae", "r2", "mape", "category_accuracy", "adjacent_category_accuracy"]]
    test_tbl.columns = ["RMSE", "MAE", "R²", "MAPE (%)", "Akurasi Kategori", "Akurasi ±1 Kategori"]
    st.dataframe(
        test_tbl.style.format(
            {
                "RMSE": "{:.3f}",
                "MAE": "{:.3f}",
                "R²": "{:.4f}",
                "MAPE (%)": "{:.2f}%",
                "Akurasi Kategori": "{:.2%}",
                "Akurasi ±1 Kategori": "{:.2%}",
            }
        ),
        use_container_width=True,
    )
    fig6 = ROOT / "artifacts" / "figures" / "fig6_ablation_comparison.png"
    if fig6.exists():
        st.image(str(fig6), caption="Perbandingan Studi Ablasi pada Test Set", use_container_width=True)
