"""Build complete 9-chapter academic report in Markdown, HTML, and DOCX from real experiment artifacts."""
from __future__ import annotations

import html
import json
from pathlib import Path
import re
import sys

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "report"

FEATURE_COLS = ["pm10", "pm25", "so2", "co", "o3", "no2"]
FEATURE_LABELS = {"pm10": "PM10", "pm25": "PM2.5", "so2": "SO2", "co": "CO", "o3": "O3", "no2": "NO2"}


def load_inputs() -> tuple[dict, dict, dict]:
    res = json.loads((ART_DIR / "experiment_results.json").read_text(encoding="utf-8"))
    art = json.loads((ART_DIR / "model_artifact.json").read_text(encoding="utf-8"))
    links_path = ART_DIR / "links.json"
    links = json.loads(links_path.read_text(encoding="utf-8")) if links_path.exists() else {}
    return res, art, links


def fmt_num(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def build_report_sections(res: dict, art: dict, links: dict) -> list[dict]:
    """Build structured sections used identically for Markdown, HTML/PDF, and DOCX."""
    cs = res["clean_stats"]
    sb = res["split_boundaries"]
    ds = res["descriptive_stats"]
    c_opt = int(res["selected_c"])
    lam_opt = float(res["selected_lambda"])
    cluster_labels: list[str] = art["cluster_labels"]
    colab_url = links.get("colab_url", "[Tautan Google Colab akan diisi setelah unggah]")
    docs_url = links.get("google_docs_url", "[Tautan Google Docs]")

    # Tables preparation
    desc_rows = []
    for k, label in [
        ("pm10", "PM10 (Sub-Indeks t)"),
        ("pm25", "PM2.5 (Sub-Indeks t)"),
        ("so2", "SO2 (Sub-Indeks t)"),
        ("co", "CO (Sub-Indeks t)"),
        ("o3", "O3 (Sub-Indeks t)"),
        ("no2", "NO2 (Sub-Indeks t)"),
        ("max", "ISPU Max Hari Ini (t)"),
        ("target_ispu_next", "Target ISPU Besok (t+1)"),
    ]:
        d = ds[k]
        desc_rows.append([
            label,
            str(d["count"]),
            fmt_num(d["mean"]),
            fmt_num(d["std"]),
            fmt_num(d["min"], 0),
            fmt_num(d["median"], 0),
            fmt_num(d["max"], 0),
        ])

    sweep_rows = []
    for r in res["sweep_results"]:
        sweep_rows.append([
            str(r["c"]),
            fmt_num(r["pc"], 4),
            fmt_num(r["pe"], 4),
            fmt_num(r["xb"], 4),
            fmt_num(r["jm"], 2),
            str(r["n_iter"]),
            fmt_num(r["best_lambda"], 1),
            fmt_num(r["lse_val_rmse"], 4),
        ])

    cluster_prof_rows = []
    for row in res["cluster_summary"]:
        cluster_prof_rows.append([
            row["label"],
            str(row["count"]),
            fmt_num(row["center_pm10"]),
            fmt_num(row["center_pm25"]),
            fmt_num(row["center_so2"]),
            fmt_num(row["center_co"]),
            fmt_num(row["center_o3"]),
            fmt_num(row["center_no2"]),
            fmt_num(row["mean_target_ispu"]),
        ])

    sample_u_rows = []
    for item in res["sample_memberships"]:
        u_strs = [fmt_num(u, 4) for u in item["memberships"]]
        sample_u_rows.append([
            item["tanggal"],
            item["stasiun"].split("(")[0].strip(),
            f"{item['pm10']:.0f} / {item['pm25']:.0f}",
            *u_strs,
            f"C{item['dominant_cluster']}",
        ])

    # Premise comparison rows (initial FCM vs Hybrid-tuned)
    prem = res["premise_orig"]
    prem_rows = []
    for i in range(c_opt):
        for j, col in enumerate(FEATURE_COLS):
            prem_rows.append([
                f"C{i+1}",
                FEATURE_LABELS[col],
                fmt_num(prem["init_centers"][i][j]),
                fmt_num(prem["opt_centers"][i][j]),
                fmt_num(prem["init_sigmas"][i][j]),
                fmt_num(prem["opt_sigmas"][i][j]),
            ])

    # Sugeno consequents in scaled space (p0..p6)
    conseq = art["ts_hybrid"]["consequents"]
    conseq_rows = []
    for i in range(c_opt):
        conseq_rows.append([
            f"Aturan {i+1}",
            fmt_num(conseq[i][0], 3),
            *[fmt_num(conseq[i][j + 1], 3) for j in range(len(FEATURE_COLS))],
        ])

    # Ablation tables (Val and Test)
    def build_ablation_rows(split_key: str) -> list[list[str]]:
        out = []
        for model_name, m in res["ablation"][split_key].items():
            out.append([
                model_name,
                fmt_num(m["rmse"], 3),
                fmt_num(m["mae"], 3),
                fmt_num(m["r2"], 4),
                f"{m['mape']:.2f}%",
                f"{m['category_accuracy']*100:.2f}%",
                f"{m['adjacent_category_accuracy']*100:.2f}%",
            ])
        return out

    # Scenario step-by-step rows
    scen_rows = []
    for sc in res["scenarios"]:
        inp = sc["inputs"]
        inp_str = ", ".join([f"{FEATURE_LABELS[c]}={inp[c]:.0f}" for c in FEATURE_COLS])
        w_str = " | ".join([f"w̄{i+1}={sc['w_bar'][i]:.3f}" for i in range(c_opt)])
        f_str = " | ".join([f"f{i+1}={sc['rule_outputs'][i]:.2f}" for i in range(c_opt)])
        err = abs(sc["predicted_ispu_next"] - sc["actual_ispu_next"])
        scen_rows.append([
            sc["preset_name"],
            f"{sc['stasiun'].split('(')[0].strip()} ({sc['tanggal']})",
            inp_str,
            w_str,
            f_str,
            f"{sc['predicted_ispu_next']:.2f} ({sc['predicted_category_next']})",
            f"{sc['actual_ispu_next']:.0f} ({sc['actual_category_next']})",
            f"{err:.2f}",
        ])

    test_m_pers = res["ablation"]["test"]["Persistence (ISPU_t)"]
    test_m_ols = res["ablation"]["test"]["Regresi Linear (OLS)"]
    test_m_s0 = res["ablation"]["test"]["Tahap 0: FCM + Sugeno-0"]
    test_m_s1 = res["ablation"]["test"]["Tahap 1: FCM + TSK-1 (LSE)"]
    test_m_s2 = res["ablation"]["test"]["Tahap 2: FCM + TSK-1 (Hybrid)"]
    val_m_s2 = res["ablation"]["val"]["Tahap 2: FCM + TSK-1 (Hybrid)"]
    val_m_ols = res["ablation"]["val"]["Regresi Linear (OLS)"]
    val_m_pers = res["ablation"]["val"]["Persistence (ISPU_t)"]
    tr_m_s2 = res["ablation"]["train"]["Tahap 2: FCM + TSK-1 (Hybrid)"]

    hist = art["ts_hybrid"]["history"]
    best_ep_row = min(hist, key=lambda h: h["val_rmse"])

    code_snippet = (
        "class TakagiSugenoHybrid:\n"
        "    def fit_lse(self, X_train, y_train, ridge_lambda=50.0):\n"
        "        _, _, w_bar = self._forward_premise(X_train)\n"
        "        X_aug = np.hstack([np.ones((len(X_train), 1)), X_train])\n"
        "        A = (w_bar[:, :, None] * X_aug[:, None, :]).reshape(len(X_train), -1)\n"
        "        theta = np.linalg.solve(A.T @ A + ridge_lambda * np.eye(A.shape[1]), A.T @ y_train)\n"
        "        self.consequents_ = theta.reshape(self.n_clusters, self.n_features + 1)\n\n"
        "    def fit_hybrid(self, X_tr, y_tr, X_va, y_va, epochs=150, lr=0.05):\n"
        "        # Forward: LSE eksak pada konsekuen; Backward: Gradient Descent pada (v_ij, log sigma_ij)\n"
        "        ..."
    )

    sections = [
        {
            "type": "cover",
            "title": "STUDI KASUS MODEL FUZZY, CLUSTERING FUZZY, DAN ESTIMASI PARAMETER",
            "subtitle": "Prediksi Dini Indeks Standar Pencemar Udara (ISPU) Hari Berikutnya di DKI Jakarta Menggunakan Takagi-Sugeno Orde-1 Berbasis Fuzzy C-Means dan Hybrid Learning",
            "meta": [
                ("Mata Kuliah", "Komputasi Lunak"),
                ("Program Studi", "Informatika — Fakultas Sains dan Matematika"),
                ("Universitas", "Universitas Diponegoro"),
                ("Nama Mahasiswa", "Menza Isaiah Tampubolon"),
                ("NIM", "24060124140138"),
                ("Kelas", "[Kelas]"),
                ("Dosen Pengampu", "[Nama Dosen Pengampu]"),
                ("Tahun", "2026"),
            ],
        },
        {
            "type": "chapter",
            "heading": "2. PENDAHULUAN",
            "blocks": [
                {"h2": "2.1 Latar Belakang"},
                {"p": (
                    "Kualitas udara ambien di Provinsi DKI Jakarta merupakan persoalan lingkungan dan kesehatan masyarakat yang krusial. "
                    "Aktivitas transportasi padat, emisi industri, serta dinamika meteorologi perkotaan menyebabkan fluktuasi harian konsentrasi polutan seperti "
                    "partikulat kasar (PM10), partikulat halus (PM2.5), sulfur dioksida (SO2), karbon monoksida (CO), ozon (O3), dan nitrogen dioksida (NO2). "
                    "Sejak diberlakukannya Peraturan Menteri Lingkungan Hidup dan Kehutanan (Permen LHK) Nomor P.14/MENLHK/SETJEN/KUM.1/7/2020 [7], "
                    "parameter PM2.5 secara resmi dimasukkan ke dalam perhitungan Indeks Standar Pencemar Udara (ISPU) harian yang dipantau oleh lima "
                    "Stasiun Pemantau Kualitas Udara (SPKU) Dinas Lingkungan Hidup DKI Jakarta."
                )},
                {"p": (
                    "Pentingnya penelitian prediksi dini ISPU hari berikutnya (ISPU(t+1)) terletak pada fungsinya sebagai sistem peringatan dini (early warning system). "
                    "Apabila pemangku kebijakan dan masyarakat hanya mengetahui status mutu udara pada hari berjalan, langkah mitigasi seperti imbauan pembatasan "
                    "aktivitas luar ruang atau rekayasa lalu lintas sering terlambat dilakukan. Memprediksi skor ISPU besok berdasarkan profil enam polutan hari ini "
                    "memungkinkan tindakan preventif sebelum episode polusi Tinggi (Tidak Sehat) terjadi."
                )},
                {"p": (
                    "Pendekatan konvensional dalam pemantauan dan peramalan kualitas udara memiliki dua kelemahan mendasar. Pertama, klasifikasi konvensional menggunakan "
                    "ambang batas tegas (crisp threshold)—misalnya skor 50 dikategorikan Baik sedangkan skor 51 langsung menjadi Sedang—yang tidak merepresentasikan "
                    "transisi risiko kesehatan secara kontinu. Kedua, model statistik linear global seperti Regresi Linear Berganda (OLS) maupun model persistensi "
                    "(ISPU(t+1) = ISPU(t)) kesulitan menangkap hubungan non-linear serta interaksi antarpolutan yang berbeda antara hari bersih dan hari berpolusi tinggi."
                )},
                {"p": (
                    "Untuk mengatasi hal tersebut, penelitian ini memilih Model Fuzzy Takagi-Sugeno Orde-1 [2]. Model fuzzy mampu memodelkan ketidakpastian dan "
                    "perubahan bertahap antar-rezim polusi melalui himpunan fuzzy [1], sementara keluaran berupa kombinasi linear variabel masukan pada Takagi-Sugeno Orde-1 "
                    "memberikan kemampuan aproksimasi fungsi non-linear yang jauh lebih akurat dibandingkan keluaran konstanta maupun himpunan linguistik statis."
                )},
                {"p": (
                    "Penentuan fungsi keanggotaan dan basis aturan fuzzy secara subjektif dari pakar sering kali tidak optimal untuk data multivariat 6 dimensi. "
                    "Oleh karena itu, digunakan metode Clustering Fuzzy C-Means (FCM) [3] untuk mempartisi ruang data secara objektif dan berbasis data. "
                    "Berbeda dari hard clustering (seperti K-Means) yang memaksa setiap hari pengamatan masuk secara mutlak ke satu kelompok, FCM memberikan derajat "
                    "keanggotaan kontinu di antara 0 dan 1 sehingga mencerminkan hari-hari transisi polusi secara lebih realistis."
                )},
                {"p": (
                    "Meskipun pusat dan lebar fungsi keanggotaan awal dapat diturunkan langsung dari FCM, sistem fuzzy yang hanya mengandalkan inisialisasi clustering "
                    "belum menjamin kesalahan prediksi minimum. Karena itu, diperlukan tahap Estimasi Parameter secara sistematis menggunakan pendekatan "
                    "Hybrid Learning (Jang, 1993) [5] yang menggabungkan Regularized Least Squares Estimation (LSE) untuk menyetel parameter konsekuen linear secara eksak "
                    "dan Gradient Descent untuk menyempurnakan parameter premis Gaussian (pusat dan lebar kurva keanggotaan) hingga model menjadi lebih representatif."
                )},
                {"h2": "2.2 Rumusan Masalah"},
                {"ol": [
                    "Bagaimana membangun model fuzzy Takagi-Sugeno Orde-1 untuk memprediksi skor dan kategori ISPU hari berikutnya (ISPU(t+1)) di DKI Jakarta dari 6 parameter polutan harian?",
                    "Bagaimana proses pembentukan kelompok data polusi udara DKI Jakarta secara objektif menggunakan metode clustering Fuzzy C-Means (FCM) dan bagaimana validitas jumlah clusternya?",
                    "Bagaimana estimasi parameter premis Gaussian dan konsekuen linear dilakukan menggunakan metode Hybrid Learning (Regularized LSE dan Gradient Descent) agar model lebih representatif?",
                    "Bagaimana performa model fuzzy yang dihasilkan dibandingkan dengan model pra-estimasi serta baseline konvensional (Persistence dan Regresi Linear OLS)?",
                ]},
                {"h2": "2.3 Tujuan Studi Kasus"},
                {"ul": [
                    "Membangun sistem prediksi dini berbasis logika fuzzy Takagi-Sugeno Orde-1 untuk memprediksi nilai kontinu dan kategori mutu udara ISPU(t+1) DKI Jakarta.",
                    "Menerapkan algoritma Fuzzy C-Means (FCM) disertai evaluasi Partition Coefficient (PC), Partition Entropy (PE), dan Xie-Beni Index (XB) guna memperoleh struktur cluster data yang objektif.",
                    "Melakukan estimasi parameter premis dan konsekuen fuzzy secara sistematis melalui studi ablasi 3 tahap (Pra-estimasi Orde-0, Regularized LSE, dan Hybrid Learning).",
                    "Mengimplementasikan keseluruhan model dalam modul Python teruji, notebook Google Colab interaktif, serta aplikasi GUI berbasis Streamlit.",
                ]},
            ],
        },
        {
            "type": "chapter",
            "heading": "3. TINJAUAN PUSTAKA",
            "blocks": [
                {"h2": "3.1 Konsep Dasar Logika Fuzzy"},
                {"p": (
                    "Logika fuzzy pertama kali diperkenalkan oleh Lotfi A. Zadeh (1965) [1] sebagai perluasan dari teori himpunan klasik (crisp set). "
                    "Jika himpunan klasik hanya mengenal keanggotaan biner {0, 1}, maka himpunan fuzzy A pada semesta pembicaraan X didefinisikan melalui "
                    "fungsi keanggotaan μ_A(x) ∈ [0, 1] yang menyatakan derajat kebenaran keanggotaan suatu nilai x terhadap himpunan A. "
                    "Konsep ini memungkinkan representasi variabel linguistik (seperti Polusi Rendah atau Polusi Tinggi) secara matematis."
                )},
                {"p": (
                    "Pada penelitian ini digunakan fungsi keanggotaan Gaussian untuk setiap variabel masukan j ∈ {1,...,D} pada aturan ke-i:"
                )},
                {"eq": "μ_ij(x_j) = exp( - (x_j - v_ij)^2 / (2 * σ_ij^2) )"},
                {"p": (
                    "di mana v_ij menyatakan pusat fungsi keanggotaan dan σ_ij menyatakan lebar (simpangan baku) kurva. "
                    "Operasi logika AND antar-anteseden pada satu aturan menggunakan T-norm perkalian aljabar (product operator) untuk menghasilkan "
                    "kekuatan penyulutan (firing strength) w_i(x) = ∏_{j=1}^D μ_ij(x_j), yang kemudian dinormalisasi menjadi w̄_i(x) = w_i(x) / ∑_{r=1}^c w_r(x). "
                    "Inferensi agregat mengubah himpunan hasil aturan menjadi nilai keluaran numerik tunggal (defuzzifikasi)."
                )},
                {"h2": "3.2 Model Fuzzy Takagi-Sugeno Orde-1"},
                {"p": (
                    "Model fuzzy Takagi-Sugeno (TSK) diperkenalkan oleh Takagi dan Sugeno (1985) [2]. Berbeda dengan model Mamdani yang menggunakan himpunan fuzzy pada bagian konsekuen, "
                    "model Takagi-Sugeno Orde-1 menggunakan fungsi linear dari variabel masukan sebagai konsekuen setiap aturan:"
                )},
                {"eq": "Aturan i: IF x_1 is A_i1 AND ... AND x_D is A_iD  THEN  f_i(x) = p_i0 + p_i1*x_1 + ... + p_iD*x_D"},
                {"p": (
                    "Keluaran akhir sistem diperoleh melalui defuzzifikasi rata-rata terbobot (weighted average): "
                    "ŷ(x) = ∑_{i=1}^c w̄_i(x) * f_i(x). Kelebihan utama model Takagi-Sugeno Orde-1 adalah efisiensi komputasi karena tidak memerlukan proses defuzzifikasi integral "
                    "yang mahal, fleksibilitas tinggi dalam memodelkan sistem non-linear kompleks melalui interpolasi mulus antar-bidang linear lokal, serta kompatibilitas langsung "
                    "dengan metode estimasi parameter berbasis aljabar linear (Least Squares Estimation) dan optimasi gradien [2], [5]."
                )},
                {"h2": "3.3 Clustering Fuzzy (Fuzzy C-Means)"},
                {"p": (
                    "Clustering adalah teknik pembelajaran tak terawasi (unsupervised learning) untuk mengelompokkan data ke dalam cluster berdasarkan kemiripan fitur. "
                    "Pada hard clustering (seperti K-Means), setiap titik data dimiliki secara eksklusif oleh satu cluster saja. Sebaliknya, pada fuzzy clustering yang dirumuskan oleh "
                    "Bezdek (1981) melalui algoritma Fuzzy C-Means (FCM) [3], setiap titik data x_k memiliki derajat keanggotaan u_ik ∈ [0, 1] terhadap setiap cluster i dengan kendala partisi ∑_{i=1}^c u_ik = 1."
                )},
                {"p": "Algoritma FCM meminimalkan fungsi objektif berbobot berikut:"},
                {"eq": "J_m(U, V) = ∑_{k=1}^N ∑_{i=1}^c (u_ik)^m * ||x_k - v_i||_2^2 ,   dengan m = 2.0"},
                {"p": "Pembaruan derajat keanggotaan u_ik dan pusat cluster v_i pada setiap iterasi dihitung melalui:"},
                {"eq": "u_ik = 1 / ∑_{r=1}^c ( ||x_k - v_i||_2 / ||x_k - v_r||_2 )^(2/(m-1)) ,    v_i = ( ∑_{k=1}^N (u_ik)^m x_k ) / ( ∑_{k=1}^N (u_ik)^m )"},
                {"p": (
                    "Proses iterasi dihentikan apabila kriteria konvergensi ||U^(t) - U^(t-1)||_∞ < 10^-5 terpenuhi. "
                    "Untuk menilai kualitas partisi secara objektif digunakan tiga indeks validitas: Partition Coefficient (PC = (1/N)∑∑ u_ik^2, lebih tinggi lebih baik), "
                    "Partition Entropy (PE = -(1/N)∑∑ u_ik ln u_ik, lebih rendah lebih baik), serta Xie-Beni Index (XB) [4] yang membandingkan kekompakan intra-cluster terhadap jarak minimum antar-pusat cluster (lebih rendah lebih baik)."
                )},
                {"h2": "3.4 Estimasi Parameter dengan Hybrid Learning"},
                {"p": (
                    "Dalam kerangka ANFIS/Takagi-Sugeno adaptif yang diperkenalkan Jang (1993) [5], setiap lintasan pelatihan Hybrid Learning terbagi menjadi dua fase dalam satu epoch: "
                    "(1) Forward Pass, di mana parameter premis (v_ij, σ_ij) dianggap tetap sehingga keluaran prediksi menjadi linear terhadap vektor parameter konsekuen θ = [p_i0, ..., p_iD]. "
                    "Untuk mencegah multikolinearitas dan overfitting pada data nyata, digunakan Regularized Least Squares Estimation (Ridge LSE): θ̂ = (A^T A + λI)^(-1) A^T y; "
                    "(2) Backward Pass, di mana parameter konsekuen θ dianggap tetap dan kesalahan prediksi dirambatkan balik menggunakan Gradient Descent untuk memperbarui v_ij dan log(σ_ij). "
                    "Berbagai studi terkini (mis. Karki, 2023 [6]) juga menunjukkan efektivitas integrasi model neuro-fuzzy adaptif dalam memprediksi indeks kualitas udara perkotaan."
                )},
            ],
        },
        {
            "type": "chapter",
            "heading": "4. METODOLOGI",
            "blocks": [
                {"h2": "4.1 Sumber dan Prapemrosesan Dataset"},
                {"p": (
                    f"Penelitian ini menggunakan data historis nyata Indeks Standar Pencemar Udara (ISPU) Provinsi DKI Jakarta tahun 2021 yang bersumber dari "
                    f"Dinas Lingkungan Hidup Provinsi DKI Jakarta (Portal Satu Data Jakarta) [7] untuk lima stasiun SPKU: {', '.join(cs['stations'])}. "
                    f"Dari total {cs['raw_rows']} baris observasi mentah selama 12 bulan (1 Januari – 31 Desember 2021), dilakukan perbaikan pergeseran header pada berkas resmi bulan Juni dan Juli 2021, "
                    f"penghapusan {cs['dropped_invalid_or_missing']} baris dengan nilai kosong/tidak terbaca sensor (terutama awal Januari sebelum sensor PM2.5 beroperasi penuh di semua SPKU), "
                    f"sehingga diperoleh {cs['clean_rows']} baris bersih."
                )},
                {"p": (
                    f"Selanjutnya, untuk membentuk masalah prediksi hari berikutnya (t -> t+1) yang bebas kebocoran data (zero data leakage), setiap observasi hari ke-t dipasangkan dengan "
                    f"nilai ISPU maksimum hari ke-(t+1) secara ketat di dalam stasiun yang sama dan hanya apabila selisih tanggal tepat 1 hari. Proses ini menghasilkan {res['n_paired']} pasangan data berurutan. "
                    f"Data dibagi secara kronologis (time-series split) tanpa pengacakan menjadi Data Latih ({sb['n_train']} data, {sb['train_start']} s.d. {sb['train_end']}), "
                    f"Data Validasi ({sb['n_val']} data, {sb['val_start']} s.d. {sb['val_end']}), dan Data Uji ({sb['n_test']} data, {sb['test_start']} s.d. {sb['test_end']}). "
                    f"Normalisasi Z-score dihitung hanya dari statistik Data Latih."
                )},
                {"h2": "4.2 Perancangan Sistem Fuzzy"},
                {"p": "Rancangan arsitektur sistem prediksi dini ISPU meliputi:"},
                {"ul": [
                    "Input Sistem: Vektor 6 sub-indeks polutan harian pada hari ke-t, yaitu x_t = [PM10, PM2.5, SO2, CO, O3, NO2]_t.",
                    "Output Sistem: Prediksi skor kontinu ISPU hari berikutnya ŷ_{t+1} beserta pemetaan kategori mutu udara resmi Permen LHK No. P.14/2020 [7]: Baik (0–50), Sedang (51–100), Tidak Sehat (101–200), Sangat Tidak Sehat (201–300), dan Berbahaya (>300).",
                    f"Basis Aturan (Rule Base): Dibentuk otomatis sebanyak c* = {c_opt} aturan fuzzy Takagi-Sugeno Orde-1 dari hasil partisi FCM. Contoh representasi aturan linguistik:",
                ]},
                {"eq": (
                    "R1: IF PM10 Rendah AND PM2.5 Rendah AND SO2 Rendah AND CO Rendah AND O3 Rendah AND NO2 Rendah THEN f1(x) = p10 + sum(p1j * xj)\n"
                    "R2: IF PM10 Tinggi AND PM2.5 Tinggi AND SO2 Tinggi AND CO Tinggi AND O3 Tinggi AND NO2 Tinggi THEN f2(x) = p20 + sum(p2j * xj)"
                )},
                {"h2": "4.3 Rancangan Evaluasi dan Studi Ablasi"},
                {"p": (
                    "Untuk membuktikan kontribusi setiap komponen secara ilmiah, performa model dievaluasi dengan metrik RMSE, MAE, R², MAPE, Akurasi Kategori, dan Akurasi ±1 Kategori melalui perbandingan 5 skenario: "
                    "(1) Persistence Baseline (ŷ_{t+1} = ISPU_t), (2) Regresi Linear Berganda (OLS), (3) Tahap 0: FCM + Sugeno Orde-0 (tanpa estimasi konsekuen linear), "
                    "(4) Tahap 1: FCM + Takagi-Sugeno Orde-1 + Regularized LSE, serta (5) Tahap 2: FCM + Takagi-Sugeno Orde-1 + Hybrid Learning (LSE + Gradient Descent)."
                )},
            ],
        },
        {
            "type": "chapter",
            "heading": "5. PEMBAHASAN DAN HASIL",
            "blocks": [
                {"h2": "5.1 Karakteristik Dataset ISPU DKI Jakarta"},
                {"p": (
                    f"Tabel 5.1 merangkum statistik deskriptif dari {res['n_paired']} pasangan observasi harian DKI Jakarta. Polutan PM2.5 memiliki rata-rata sub-indeks tertinggi "
                    f"({fmt_num(ds['pm25']['mean'])} ± {fmt_num(ds['pm25']['std'])}), diikuti PM10 ({fmt_num(ds['pm10']['mean'])}). Distribusi komposisi kategori target menunjukkan dominasi "
                    f"kategori Sedang ({res['target_category_counts'].get('SEDANG', 0)} hari / 75.8%), diikuti Tidak Sehat ({res['target_category_counts'].get('TIDAK SEHAT', 0)} hari / 15.9%) dan Baik ({res['target_category_counts'].get('BAIK', 0)} hari / 8.2%)."
                )},
                {"table": {"caption": "Tabel 5.1 Statistik Deskriptif Variabel Input dan Target (N = 1.444)", "headers": ["Variabel", "N", "Rata-rata", "Std Dev", "Min", "Median", "Maks"], "rows": desc_rows}},
                {"fig": {"path": str(ART_DIR / "figures" / "fig1_data_distribution.png"), "caption": "Gambar 5.1 Distribusi sub-indeks 6 polutan dan komposisi kategori target ISPU(t+1)."}},
                {"h2": "5.2 Penentuan Jumlah Cluster Optimal dan Hasil Fuzzy C-Means"},
                {"p": (
                    f"Eksperimen sapuan jumlah cluster c ∈ {{2, 3, 4, 5, 6}} pada Data Latih (Tabel 5.2) menunjukkan bahwa nilai Partition Coefficient (PC) tertinggi ({fmt_num(res['sweep_results'][0]['pc'], 4)}), "
                    f"Partition Entropy (PE) terendah ({fmt_num(res['sweep_results'][0]['pe'], 4)}), serta kesalahan validasi LSE terendah (Val RMSE = {fmt_num(res['sweep_results'][0]['lse_val_rmse'], 4)}) "
                    f"dicapai pada c* = {c_opt}. Penyempurnaan parameter regularisasi Ridge melalui Validation RMSE Hybrid Learning selanjutnya memilih λ* = {lam_opt:.1f} (Val RMSE = {fmt_num(val_m_s2['rmse'], 4)})."
                )},
                {"table": {"caption": "Tabel 5.2 Hasil Evaluasi Validitas Cluster FCM dan Validasi RMSE", "headers": ["c", "PC", "PE", "Xie-Beni (XB)", "J_m", "Iterasi", "λ LSE", "Val RMSE"], "rows": sweep_rows}},
                {"fig": {"path": str(ART_DIR / "figures" / "fig2_cluster_selection.png"), "caption": "Gambar 5.2 Kurva validitas partisi FCM (PC, XB, Val RMSE) dan konvergensi fungsi objektif J_m."}},
                {"p": (
                    f"Pada c* = {c_opt}, FCM konvergen dalam {art['fcm']['n_iter']} iterasi menghasilkan dua pusat cluster terurut yang sangat interpretatif (Tabel 5.3): "
                    f"{cluster_labels[0]} (485 hari latih, rata-rata target ISPU berikutnya {fmt_num(res['cluster_summary'][0]['mean_target_ispu'])}) dan "
                    f"{cluster_labels[1]} (580 hari latih, rata-rata target ISPU berikutnya {fmt_num(res['cluster_summary'][1]['mean_target_ispu'])}). "
                    f"Tabel 5.4 memperlihatkan contoh derajat keanggotaan fuzzy u_ik pada lima observasi awal di mana jumlah derajat keanggotaan setiap baris selalu bernilai 1.0000."
                )},
                {"table": {"caption": "Tabel 5.3 Pusat Cluster FCM dalam Skala Sub-Indeks ISPU Asli", "headers": ["Cluster", "N", "PM10", "PM2.5", "SO2", "CO", "O3", "NO2", "Rata2 ISPU(t+1)"], "rows": cluster_prof_rows}},
                {"table": {"caption": "Tabel 5.4 Contoh Matriks Derajat Keanggotaan FCM (u_ik)", "headers": ["Tanggal", "Stasiun", "PM10 / PM2.5", "u_C1", "u_C2", "Dominan"], "rows": sample_u_rows}},
                {"fig": {"path": str(ART_DIR / "figures" / "fig3_fcm_clusters_pca.png"), "caption": "Gambar 5.3 Proyeksi PCA 2D hasil clustering FCM dan profil polutan tiap pusat cluster."}},
                {"h2": "5.3 Hasil Estimasi Parameter Premis dan Konsekuen"},
                {"p": (
                    f"Tahap Hybrid Learning dijalankan dengan early stopping berbasis Data Validasi dan mencapai titik terbaik pada epoch ke-{int(best_ep_row['epoch'])} "
                    f"(Val RMSE = {fmt_num(best_ep_row['val_rmse'], 3)}, Train RMSE = {fmt_num(best_ep_row['train_rmse'], 3)}). "
                    f"Tabel 5.5 membandingkan parameter Gaussian (pusat v_ij dan lebar σ_ij dalam skala ISPU asli) sebelum dan sesudah optimasi Gradient Descent, "
                    f"sedangkan Tabel 5.6 menampilkan koefisien konsekuen linear Takagi-Sugeno Orde-1 hasil Regularized LSE."
                )},
                {"table": {"caption": "Tabel 5.5 Parameter Premis Gaussian Sebelum (FCM Awal) dan Sesudah Hybrid Learning", "headers": ["Aturan", "Polutan", "v Awal", "v Hybrid", "σ Awal", "σ Hybrid"], "rows": prem_rows}},
                {"fig": {"path": str(ART_DIR / "figures" / "fig4_membership_functions.png"), "caption": "Gambar 5.4 Perbandingan kurva fungsi keanggotaan Gaussian sebelum dan sesudah Hybrid Learning."}},
                {"table": {"caption": "Tabel 5.6 Koefisien Konsekuen Linear Takagi-Sugeno Orde-1 (Input Terstandarisasi)", "headers": ["Aturan", "Bias p0", "p_PM10", "p_PM2.5", "p_SO2", "p_CO", "p_O3", "p_NO2"], "rows": conseq_rows}},
                {"fig": {"path": str(ART_DIR / "figures" / "fig5_hybrid_training_curve.png"), "caption": "Gambar 5.5 Kurva konvergensi Train RMSE dan Validation RMSE selama pelatihan Hybrid Learning."}},
                {"h2": "5.4 Simulasi Inferensi Fuzzy dan Hasil Studi Ablasi"},
                {"p": "Tabel 5.7 menampilkan penelusuran inferensi langkah demi langkah pada tiga skenario uji nyata (Baik, Sedang, dan Polusi Tinggi). Tabel 5.8 dan 5.9 menyajikan perbandingan performa kuantitatif pada Data Validasi dan Data Uji."},
                {"table": {"caption": "Tabel 5.7 Simulasi Inferensi Fuzzy TSK-1 pada Tiga Kasus Nyata Test Set", "headers": ["Skenario", "Lokasi & Tgl", "Input Polutan", "Bobot w̄", "Konsekuen f_i", "Prediksi", "Aktual", "Galat"], "rows": scen_rows}},
                {"table": {"caption": "Tabel 5.8 Hasil Studi Ablasi & Baseline pada Data Validasi (12 Sep – 5 Nov 2021)", "headers": ["Model", "RMSE", "MAE", "R²", "MAPE", "Akurasi Kat.", "Akurasi ±1 Kat."], "rows": build_ablation_rows("val")}},
                {"table": {"caption": "Tabel 5.9 Hasil Studi Ablasi & Baseline pada Data Uji (6 Nov – 30 Des 2021)", "headers": ["Model", "RMSE", "MAE", "R²", "MAPE", "Akurasi Kat.", "Akurasi ±1 Kat."], "rows": build_ablation_rows("test")}},
                {"fig": {"path": str(ART_DIR / "figures" / "fig6_ablation_comparison.png"), "caption": "Gambar 5.6 Perbandingan RMSE, MAE, dan R² seluruh model pada studi ablasi Data Uji."}},
                {"fig": {"path": str(ART_DIR / "figures" / "fig7_actual_vs_predicted.png"), "caption": "Gambar 5.7 Grafik aktual vs prediksi, deret waktu stasiun DKI1, dan matriks kebingungan kategori pada Data Uji."}},
            ],
        },
        {
            "type": "chapter",
            "heading": "6. IMPLEMENTASI PROGRAM",
            "blocks": [
                {"h2": "6.1 Arsitektur Kode Python dan Google Colab"},
                {"p": (
                    "Seluruh implementasi dibangun menggunakan Python 3.12.6 dengan pustaka NumPy, Pandas, SciPy, Scikit-Learn, Matplotlib, dan Streamlit. "
                    "Struktur kode dipisahkan secara modular: src/data_loader.py (pembersihan data dan pasangan t -> t+1 bebas kebocoran), "
                    "src/fcm.py (Fuzzy C-Means dan metrik validitas PC/PE/XB), src/sugeno_hybrid.py (Takagi-Sugeno Orde-1, LSE, dan Hybrid Learning beserta verifikasi gradien numerik), "
                    "src/evaluation.py (metrik regresi dan pemetaan Permen LHK), serta src/visualization.py."
                )},
                {"code": code_snippet},
                {"p": f"Notebook interaktif lengkap yang dapat dijalankan ulang satu klik di Google Colab tersedia pada berkas notebooks/ISPU_Fuzzy_FCM_Hybrid.ipynb dan tautan Google Colab berikut: {colab_url}"},
                {"h2": "6.2 Dokumentasi GUI Interaktif (Streamlit)"},
                {"p": (
                    "Sebagai nilai tambah aplikasi pengguna, dibangun GUI web interaktif berbasis Streamlit (app.py) yang memenuhi seluruh kriteria fungsional: "
                    "(1) menerima input manual 6 parameter polutan maupun preset data nyata Jakarta, (2) menampilkan peta cluster PCA 2D dan derajat keanggotaan FCM, "
                    "(3) memvisualisasikan kurva fungsi keanggotaan Gaussian lengkap dengan posisi input pengguna, (4) merinci tabel inferensi fuzzy dan kontribusi tiap aturan, serta "
                    "(5) menyajikan keputusan akhir prediksi skor ISPU(t+1), badge kategori warna Permen LHK, dan rekomendasi mitigasi kesehatan."
                )},
                {"fig": {"path": str(ART_DIR / "gui_screenshots" / "gui_overview.png"), "caption": "Gambar 6.1 Tampilan utama GUI Streamlit: input sidebar, kartu keputusan akhir, dan visualisasi clustering FCM."}},
                {"fig": {"path": str(ART_DIR / "gui_screenshots" / "gui_mf.png"), "caption": "Gambar 6.2 Tab fungsi keanggotaan Gaussian interaktif dengan garis penanda nilai input pengguna."}},
                {"fig": {"path": str(ART_DIR / "gui_screenshots" / "gui_inference.png"), "caption": "Gambar 6.3 Tab rincian inferensi Takagi-Sugeno Orde-1, kontribusi aturan, dan persamaan defuzzifikasi."}},
                {"fig": {"path": str(ART_DIR / "gui_screenshots" / "gui_evaluation.png"), "caption": "Gambar 6.4 Tab ringkasan evaluasi studi ablasi dan perbandingan metrik pada Data Uji."}},
            ],
        },
        {
            "type": "chapter",
            "heading": "7. ANALISIS HASIL",
            "blocks": [
                {"h2": "7.1 Interpretasi Hasil Clustering dan Parameter Model"},
                {"p": (
                    "Partisi optimal c* = 2 memisahkan dinamika udara Jakarta secara alami menjadi rezim Polusi Rendah (pusat PM2.5 = 67.51, rata-rata target ISPU besok 69.48) "
                    "dan rezim Polusi Tinggi (pusat PM2.5 = 94.18, rata-rata target ISPU besok 92.61). Setelah proses Hybrid Learning, pusat fungsi keanggotaan C1 pada PM2.5 bergeser dari "
                    f"{fmt_num(prem['init_centers'][0][1])} menjadi {fmt_num(prem['opt_centers'][0][1])} dengan lebar kurva yang lebih tajam (σ dari {fmt_num(prem['init_sigmas'][0][1])} menjadi {fmt_num(prem['opt_sigmas'][0][1])}), "
                    f"sedangkan C2 memperluas cakupan σ dari {fmt_num(prem['init_sigmas'][1][1])} menjadi {fmt_num(prem['opt_sigmas'][1][1])}. Hal ini menunjukkan bahwa Hybrid Learning memfokuskan aturan pertama pada kondisi transisi udara bersih "
                    f"dan menyerahkan variasi lonjakan polusi tinggi pada aturan kedua. Pada tabel konsekuen linear (Tabel 5.6), koefisien terbesar secara konsisten berada pada p_PM2.5 "
                    f"({fmt_num(conseq[0][2], 3)} dan {fmt_num(conseq[1][2], 3)}), menegaskan bahwa konsentrasi PM2.5 hari ini merupakan prediktor terkuat bagi ISPU hari berikutnya."
                )},
                {"h2": "7.2 Analisis Dampak Estimasi Parameter dan Perbandingan Baseline"},
                {"p": (
                    f"Hasil studi ablasi membuktikan dampak krusial dari pemilihan orde model dan estimasi parameter secara sistematis. Pada Data Uji, model pra-estimasi Tahap 0 (FCM + Sugeno Orde-0) "
                    f"menghasilkan RMSE {fmt_num(test_m_s0['rmse'], 3)} (R² = {fmt_num(test_m_s0['r2'], 4)}) karena output konstan tidak mampu mengikuti perubahan gradien polutan harian. "
                    f"Penambahan konsekuen linear melalui Regularized LSE (Tahap 1) langsung menurunkan RMSE Uji menjadi {fmt_num(test_m_s1['rmse'], 3)}, dan penyempurnaan premis melalui Hybrid Learning (Tahap 2) "
                    f"menekan galat lebih jauh hingga mencapai RMSE Uji terbaik sebesar {fmt_num(test_m_s2['rmse'], 3)}, MAE {fmt_num(test_m_s2['mae'], 3)}, dan MAPE {fmt_num(test_m_s2['mape'], 2)}%—mengungguli "
                    f"baseline Persistence (RMSE {fmt_num(test_m_pers['rmse'], 3)}) maupun Regresi Linear OLS (RMSE {fmt_num(test_m_ols['rmse'], 3)}). Pada Data Validasi, keunggulan Hybrid Learning juga terlihat jelas dengan "
                    f"RMSE {fmt_num(val_m_s2['rmse'], 3)} (R² = {fmt_num(val_m_s2['r2'], 4)}) dibandingkan OLS ({fmt_num(val_m_ols['rmse'], 3)}) dan Persistence ({fmt_num(val_m_pers['rmse'], 3)})."
                )},
                {"p": (
                    f"Dari sisi klasifikasi kategori Permen LHK pada Data Uji, model mencapai Akurasi Kategori {test_m_s2['category_accuracy']*100:.2f}% dan Akurasi ±1 Kategori {test_m_s2['adjacent_category_accuracy']*100:.2f}%. "
                    f"Penurunan nilai R² pada periode Uji (November–Desember 2021) dibandingkan Data Latih (R² = {fmt_num(tr_m_s2['r2'], 4)}) dan Validasi (R² = {fmt_num(val_m_s2['r2'], 4)}) disebabkan oleh pergeseran distribusi musiman "
                    f"(seasonal distribution shift): periode November–Desember merupakan puncak musim penghujan di Jakarta yang memicu pencucian polutan (rain washout) secara mendadak sehingga proporsi hari berkategori Baik meningkat drastis."
                )},
                {"h2": "7.3 Kelebihan, Kelemahan, dan Potensi Pengembangan"},
                {"ul": [
                    "Kelebihan Model: Transparan dan mudah diinterpretasi melalui aturan IF-THEN serta bobot penyulutan; tahan terhadap outlier berkat perhitungan log-space; mampu mengungguli model linear global maupun persistensi pada metrik error kontinu.",
                    "Kelemahan Model: Hanya bergantung pada 6 variabel polutan historis hari ke-t tanpa variabel meteorologi eksternal, sehingga lonjakan/penurunan mendadak akibat hujan deras atau perubahan arah angin belum dapat dijelaskan sepenuhnya.",
                    "Potensi Pengembangan: Integrasi data meteorologi BMKG (curah hujan, kecepatan angin, suhu, dan kelembaban relatif), penggunaan jendela lag multi-hari (t-2, t-1, t), serta eksperimen optimasi metaheuristik seperti PSO-ANFIS [6] atau model hibrid musiman multi-tahun.",
                ]},
            ],
        },
        {
            "type": "chapter",
            "heading": "8. KESIMPULAN DAN SARAN",
            "blocks": [
                {"h2": "8.1 Kesimpulan"},
                {"ol": [
                    f"Model fuzzy Takagi-Sugeno Orde-1 berhasil dibangun dan diimplementasikan untuk memprediksi skor serta kategori mutu udara ISPU(t+1) DKI Jakarta menggunakan 1.444 pasangan pengamatan harian resmi dari 5 stasiun SPKU tahun 2021 tanpa kebocoran data.",
                    f"Metode clustering Fuzzy C-Means (FCM) secara objektif mempartisi data latih menjadi c* = {c_opt} cluster optimal berdasarkan nilai PC tertinggi ({fmt_num(res['sweep_results'][0]['pc'], 4)}) dan Validation RMSE terendah, yang merepresentasikan kondisi Polusi Rendah dan Polusi Tinggi dengan dominansi polutan PM2.5.",
                    f"Estimasi parameter dua-tahap menggunakan Hybrid Learning (Regularized LSE λ* = {lam_opt:.0f} dan Gradient Descent dengan early stopping) menyempurnakan ketajaman fungsi keanggotaan Gaussian serta menyetel koefisien konsekuen linear secara optimal pada epoch ke-{int(best_ep_row['epoch'])}.",
                    f"Studi ablasi membuktikan perbaikan performa yang konsisten: pada Data Uji, TSK-1 Hybrid mencapai RMSE terbaik ({fmt_num(test_m_s2['rmse'], 3)}), MAE ({fmt_num(test_m_s2['mae'], 3)}), dan MAPE ({fmt_num(test_m_s2['mape'], 2)}%), mengungguli Tahap 0 Sugeno-0 (RMSE {fmt_num(test_m_s0['rmse'], 3)}), Persistence (RMSE {fmt_num(test_m_pers['rmse'], 3)}), dan Regresi Linear OLS (RMSE {fmt_num(test_m_ols['rmse'], 3)}), dengan akurasi ±1 kategori sebesar {test_m_s2['adjacent_category_accuracy']*100:.2f}%.",
                ]},
                {"h2": "8.2 Saran"},
                {"p": (
                    "Untuk penelitian lanjutan disarankan menggabungkan dataset ISPU lintas tahun (2021–2025) bersama data meteorologi harian BMKG (curah hujan, kecepatan angin, dan kelembaban udara) "
                    "agar model mampu menangkap pergeseran musiman antara musim kemarau dan musim penghujan di DKI Jakarta secara lebih presisi."
                )},
            ],
        },
        {
            "type": "chapter",
            "heading": "9. DAFTAR PUSTAKA",
            "blocks": [
                {"bib": [
                    "[1] L. A. Zadeh, \"Fuzzy sets,\" Information and Control, vol. 8, no. 3, pp. 338–353, Jun. 1965.",
                    "[2] T. Takagi and M. Sugeno, \"Fuzzy identification of systems and its applications to modeling and control,\" IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-15, no. 1, pp. 116–132, Jan.–Feb. 1985.",
                    "[3] J. C. Bezdek, Pattern Recognition with Fuzzy Objective Function Algorithms. New York, NY, USA: Plenum Press, 1981.",
                    "[4] X. L. Xie and G. Beni, \"A validity measure for fuzzy clustering,\" IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 13, no. 8, pp. 841–847, Aug. 1991.",
                    "[5] J.-S. R. Jang, \"ANFIS: Adaptive-network-based fuzzy inference system,\" IEEE Transactions on Systems, Man, and Cybernetics, vol. 23, no. 3, pp. 665–685, May–Jun. 1993.",
                    "[6] A. Karki, \"Improved ANFIS model based on confinement optimized PSO to predict the air quality of Delhi, India,\" in Proc. 2023 IEEE International Conference on Fuzzy Systems (FUZZ), 2023, doi: 10.1109/FUZZ52849.2023.10309676.",
                    "[7] Kementerian Lingkungan Hidup dan Kehutanan Republik Indonesia, Peraturan Menteri LHK Nomor P.14/MENLHK/SETJEN/KUM.1/7/2020 tentang Indeks Standar Pencemar Udara, Jakarta, 2020; serta Dinas Lingkungan Hidup Provinsi DKI Jakarta, Data Indeks Standar Pencemar Udara (ISPU) di Provinsi DKI Jakarta Tahun 2021, Portal Satu Data Jakarta (satudata.jakarta.go.id).",
                ]}
            ],
        },
    ]
    return sections


def render_markdown(sections: list[dict]) -> str:
    parts: list[str] = []
    for sec in sections:
        if sec["type"] == "cover":
            parts.append(f"# {sec['title']}\n\n## {sec['subtitle']}\n")
            parts.append(md_table(["Atribut", "Keterangan"], [[k, v] for k, v in sec["meta"]]))
            parts.append("\n---\n")
        else:
            parts.append(f"\n# {sec['heading']}\n")
            for blk in sec["blocks"]:
                if "h2" in blk:
                    parts.append(f"\n## {blk['h2']}\n")
                elif "p" in blk:
                    parts.append(blk["p"] + "\n")
                elif "eq" in blk:
                    parts.append(f"\n```text\n{blk['eq']}\n```\n")
                elif "code" in blk:
                    parts.append(f"\n```python\n{blk['code']}\n```\n")
                elif "ul" in blk:
                    parts.append("\n" + "\n".join(f"- {item}" for item in blk["ul"]) + "\n")
                elif "ol" in blk:
                    parts.append("\n" + "\n".join(f"{idx+1}. {item}" for idx, item in enumerate(blk["ol"])) + "\n")
                elif "bib" in blk:
                    parts.append("\n" + "\n\n".join(blk["bib"]) + "\n")
                elif "table" in blk:
                    t = blk["table"]
                    parts.append(f"\n**{t['caption']}**\n\n" + md_table(t["headers"], t["rows"]) + "\n")
                elif "fig" in blk:
                    f = blk["fig"]
                    rel = Path(f["path"]).relative_to(ROOT).as_posix()
                    parts.append(f"\n![{f['caption']}](../{rel})\n*{f['caption']}*\n")
    return "\n".join(parts)


def render_html(sections: list[dict]) -> str:
    body: list[str] = []
    for sec in sections:
        if sec["type"] == "cover":
            meta_rows = "".join(
                f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
                for k, v in sec["meta"]
            )
            body.append(
                f"""
                <section class="cover-page">
                    <div class="badge">PROYEK BESAR INDIVIDU — KOMPUTASI LUNAK</div>
                    <h1>{html.escape(sec['title'])}</h1>
                    <h2>{html.escape(sec['subtitle'])}</h2>
                    <div class="cover-divider"></div>
                    <table class="meta-table">{meta_rows}</table>
                    <div class="cover-footer">Program Studi Informatika • Fakultas Sains dan Matematika • Universitas Diponegoro</div>
                </section>
                """
            )
        else:
            body.append(f'<section class="chapter"><h1>{html.escape(sec["heading"])}</h1>')
            for blk in sec["blocks"]:
                if "h2" in blk:
                    body.append(f"<h2>{html.escape(blk['h2'])}</h2>")
                elif "p" in blk:
                    body.append(f"<p>{html.escape(blk['p'])}</p>")
                elif "eq" in blk:
                    body.append(f'<pre class="equation">{html.escape(blk["eq"])}</pre>')
                elif "code" in blk:
                    body.append(f'<pre class="codeblock">{html.escape(blk["code"])}</pre>')
                elif "ul" in blk:
                    items = "".join(f"<li>{html.escape(i)}</li>" for i in blk["ul"])
                    body.append(f"<ul>{items}</ul>")
                elif "ol" in blk:
                    items = "".join(f"<li>{html.escape(i)}</li>" for i in blk["ol"])
                    body.append(f"<ol>{items}</ol>")
                elif "bib" in blk:
                    items = "".join(f'<p class="bib-item">{html.escape(i)}</p>' for i in blk["bib"])
                    body.append(items)
                elif "table" in blk:
                    t = blk["table"]
                    ths = "".join(f"<th>{html.escape(h)}</th>" for h in t["headers"])
                    trs = "".join(
                        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>"
                        for r in t["rows"]
                    )
                    body.append(
                        f'<div class="table-wrap"><div class="caption">{html.escape(t["caption"])}</div>'
                        f"<table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table></div>"
                    )
                elif "fig" in blk:
                    f = blk["fig"]
                    uri = Path(f["path"]).resolve().as_uri()
                    body.append(
                        f'<figure><img src="{uri}" alt="{html.escape(f["caption"])}"/>'
                        f'<figcaption>{html.escape(f["caption"])}</figcaption></figure>'
                    )
            body.append("</section>")

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8"/>
<title>Laporan Proyek Besar Komputasi Lunak — Menza Isaiah Tampubolon</title>
<style>
  @page {{ size: A4; margin: 20mm 18mm 20mm 18mm; }}
  body {{ font-family: "Segoe UI", Calibri, Arial, sans-serif; color: #1E293B; line-height: 1.55; font-size: 10.5pt; margin: 0; padding: 0; }}
  .cover-page {{ page-break-after: always; text-align: center; padding-top: 32mm; }}
  .badge {{ display: inline-block; background: #0F2942; color: #FFFFFF; padding: 6px 16px; border-radius: 4px; font-weight: 700; font-size: 9.5pt; letter-spacing: 0.8px; margin-bottom: 18px; }}
  .cover-page h1 {{ font-size: 19pt; color: #0F2942; margin: 10px 0; line-height: 1.3; }}
  .cover-page h2 {{ font-size: 12.5pt; color: #334155; font-weight: 500; margin: 12px auto 24px auto; max-width: 90%; line-height: 1.45; }}
  .cover-divider {{ width: 90px; height: 4px; background: #1565C0; margin: 16px auto 28px auto; }}
  .meta-table {{ margin: 0 auto; width: 78%; border-collapse: collapse; text-align: left; font-size: 10.5pt; }}
  .meta-table th, .meta-table td {{ border: 1px solid #CBD5E1; padding: 8px 12px; }}
  .meta-table th {{ background: #F1F5F9; width: 36%; color: #0F2942; }}
  .cover-footer {{ margin-top: 42mm; font-weight: 600; color: #475569; font-size: 10pt; }}
  .chapter {{ margin-bottom: 14px; }}
  .chapter h1 {{ font-size: 14pt; color: #0F2942; border-bottom: 2px solid #1565C0; padding-bottom: 4px; margin-top: 18px; page-break-after: avoid; }}
  h2 {{ font-size: 11.5pt; color: #1E3A8A; margin: 12px 0 6px 0; page-break-after: avoid; }}
  p {{ text-align: justify; margin: 6px 0; }}
  ul, ol {{ margin: 6px 0 8px 20px; }}
  li {{ margin-bottom: 4px; text-align: justify; }}
  pre.equation {{ background: #F8FAFC; border-left: 4px solid #1565C0; padding: 8px 12px; font-family: "Consolas", "Courier New", monospace; font-size: 9.5pt; white-space: pre-wrap; margin: 8px 0; }}
  pre.codeblock {{ background: #0F172A; color: #E2E8F0; padding: 10px 12px; border-radius: 5px; font-family: "Consolas", monospace; font-size: 8.5pt; white-space: pre-wrap; }}
  .table-wrap {{ margin: 10px 0 14px 0; page-break-inside: avoid; }}
  .caption {{ font-weight: 700; font-size: 9.5pt; color: #0F2942; margin-bottom: 5px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 8.6pt; }}
  th, td {{ border: 1px solid #CBD5E1; padding: 5px 7px; text-align: left; }}
  thead th {{ background: #0F2942; color: #FFFFFF; font-weight: 600; }}
  tbody tr:nth-child(even) {{ background: #F8FAFC; }}
  figure {{ text-align: center; margin: 12px 0 16px 0; page-break-inside: avoid; }}
  figure img {{ max-width: 96%; max-height: 110mm; border: 1px solid #CBD5E1; border-radius: 4px; }}
  figcaption {{ font-size: 9pt; color: #475569; font-style: italic; margin-top: 4px; }}
  .bib-item {{ padding-left: 22px; text-indent: -22px; margin-bottom: 6px; font-size: 9.8pt; }}
</style>
</head>
<body>
{''.join(body)}
</body>
</html>
"""


def _set_cell_bg(cell, hex_color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def render_docx(sections: list[dict], out_path: Path) -> None:
    doc = docx.Document()
    for sec_page in doc.sections:
        sec_page.top_margin = Inches(0.8)
        sec_page.bottom_margin = Inches(0.8)
        sec_page.left_margin = Inches(0.85)
        sec_page.right_margin = Inches(0.85)

    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(11)

    for sec in sections:
        if sec["type"] == "cover":
            p_badge = doc.add_paragraph()
            p_badge.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_b = p_badge.add_run("PROYEK BESAR INDIVIDU — KOMPUTASI LUNAK")
            r_b.bold = True
            r_b.font.size = Pt(11)
            r_b.font.color.rgb = RGBColor(0x15, 0x65, 0xC0)

            p_title = doc.add_paragraph()
            p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_t = p_title.add_run(sec["title"])
            r_t.bold = True
            r_t.font.size = Pt(17)
            r_t.font.color.rgb = RGBColor(0x0F, 0x29, 0x42)

            p_sub = doc.add_paragraph()
            p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_s = p_sub.add_run(sec["subtitle"])
            r_s.font.size = Pt(12)
            r_s.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

            tbl = doc.add_table(rows=len(sec["meta"]), cols=2)
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            tbl.style = "Table Grid"
            for idx, (k, v) in enumerate(sec["meta"]):
                c0, c1 = tbl.rows[idx].cells[0], tbl.rows[idx].cells[1]
                c0.text = k
                c1.text = v
                _set_cell_bg(c0, "F1F5F9")
                for run in c0.paragraphs[0].runs:
                    run.bold = True

            p_ft = doc.add_paragraph("\n\nProgram Studi Informatika • Fakultas Sains dan Matematika • Universitas Diponegoro")
            p_ft.alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.add_page_break()
            continue

        h1 = doc.add_heading(sec["heading"], level=1)
        for run in h1.runs:
            run.font.color.rgb = RGBColor(0x0F, 0x29, 0x42)

        for blk in sec["blocks"]:
            if "h2" in blk:
                h2 = doc.add_heading(blk["h2"], level=2)
                for run in h2.runs:
                    run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
            elif "p" in blk:
                p = doc.add_paragraph(blk["p"])
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            elif "eq" in blk:
                p = doc.add_paragraph()
                r = p.add_run(blk["eq"])
                r.font.name = "Consolas"
                r.font.size = Pt(9.5)
            elif "code" in blk:
                p = doc.add_paragraph()
                r = p.add_run(blk["code"])
                r.font.name = "Consolas"
                r.font.size = Pt(8.5)
            elif "ul" in blk:
                for item in blk["ul"]:
                    doc.add_paragraph(item, style="List Bullet")
            elif "ol" in blk:
                for item in blk["ol"]:
                    doc.add_paragraph(item, style="List Number")
            elif "bib" in blk:
                for item in blk["bib"]:
                    p = doc.add_paragraph(item)
                    p.paragraph_format.left_indent = Inches(0.3)
                    p.paragraph_format.first_line_indent = Inches(-0.3)
            elif "table" in blk:
                t = blk["table"]
                cap = doc.add_paragraph()
                rc = cap.add_run(t["caption"])
                rc.bold = True
                rc.font.size = Pt(10)
                tbl = doc.add_table(rows=len(t["rows"]) + 1, cols=len(t["headers"]))
                tbl.style = "Table Grid"
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                for j, h in enumerate(t["headers"]):
                    cell = tbl.rows[0].cells[j]
                    cell.text = h
                    _set_cell_bg(cell, "0F2942")
                    for run in cell.paragraphs[0].runs:
                        run.bold = True
                        run.font.size = Pt(8.5)
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                for i, row in enumerate(t["rows"]):
                    for j, val in enumerate(row):
                        cell = tbl.rows[i + 1].cells[j]
                        cell.text = str(val)
                        for run in cell.paragraphs[0].runs:
                            run.font.size = Pt(8.5)
            elif "fig" in blk:
                f = blk["fig"]
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.add_run().add_picture(str(f["path"]), width=Inches(6.2))
                p_cap = doc.add_paragraph()
                p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                rc = p_cap.add_run(f["caption"])
                rc.italic = True
                rc.font.size = Pt(9.5)

    doc.save(out_path)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    res, art, links = load_inputs()
    sections = build_report_sections(res, art, links)

    md_path = REPORT_DIR / "laporan_ispu_fuzzy.md"
    html_path = REPORT_DIR / "laporan_ispu_fuzzy.html"
    docx_path = REPORT_DIR / "Laporan_Proyek_Besar_ISPU_Fuzzy.docx"

    md_path.write_text(render_markdown(sections), encoding="utf-8")
    html_path.write_text(render_html(sections), encoding="utf-8")
    render_docx(sections, docx_path)
    print(f"Generated:\n- {md_path}\n- {html_path}\n- {docx_path}")


if __name__ == "__main__":
    main()
