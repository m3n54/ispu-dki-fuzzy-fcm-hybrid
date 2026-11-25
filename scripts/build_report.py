"""Build complete academic report in exact UNDIP Times New Roman black-text format (MD, HTML/PDF, DOCX)."""
from __future__ import annotations

import html
import json
from pathlib import Path
import re

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "report"
LOGO_PATH = REPORT_DIR / "assets" / "logo_undip.png"

FEATURE_COLS = ["pm10", "pm25", "so2", "co", "o3", "no2"]
FEATURE_LABELS = {"pm10": "PM10", "pm25": "PM2.5", "so2": "SO2", "co": "CO", "o3": "O3", "no2": "NO2"}
BLACK = RGBColor(0x00, 0x00, 0x00)


def load_inputs() -> tuple[dict, dict, dict]:
    res = json.loads((ART_DIR / "experiment_results.json").read_text(encoding="utf-8"))
    art = json.loads((ART_DIR / "model_artifact.json").read_text(encoding="utf-8"))
    links_path = ART_DIR / "links.json"
    links = json.loads(links_path.read_text(encoding="utf-8")) if links_path.exists() else {}
    return res, art, links


def fmt_id(v: float, decimals: int = 2) -> str:
    """Format number in Indonesian locale: dot for thousands, comma for decimals."""
    formatted = f"{v:,.{decimals}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_int_id(v: int) -> str:
    return f"{int(v):,}".replace(",", ".")


def inline_to_html(text: str) -> str:
    """Escape HTML safely while converting **bold**, *italic*, and [label](url) links."""
    tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\(https?://[^)]+\))", text)
    out: list[str] = []
    link_re = re.compile(r"^\[([^\]]+)\]\((https?://[^)]+)\)$")
    for tok in tokens:
        m = link_re.match(tok)
        if m:
            label, url = m.group(1), m.group(2)
            out.append(
                f'<a href="{html.escape(url, quote=True)}" target="_blank" '
                f'style="color:#000000; text-decoration:underline;">{html.escape(label)}</a>'
            )
        elif tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            out.append(f"<strong>{html.escape(tok[2:-2])}</strong>")
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            out.append(f"<em>{html.escape(tok[1:-1])}</em>")
        else:
            out.append(html.escape(tok))
    return "".join(out)


def _add_docx_hyperlink(paragraph, url: str, label: str, font_name: str = "Times New Roman", font_size: Pt = Pt(12)) -> None:
    """Append an external OOXML hyperlink styled in pure black underlined Times New Roman."""
    part = paragraph.part
    r_id = part.relate_to(url, docx.opc.constants.RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:ascii"), font_name)
    rFonts.set(qn("w:hAnsi"), font_name)
    rPr.append(rFonts)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "000000")
    rPr.append(color)
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rPr.append(u)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(font_size.pt * 2)))
    rPr.append(sz)
    new_run.append(rPr)
    t_el = OxmlElement("w:t")
    t_el.text = label
    new_run.append(t_el)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)


def _add_formatted_runs(paragraph, text: str, font_name: str = "Times New Roman", font_size: Pt = Pt(12), base_bold: bool = False) -> None:
    """Add runs to a python-docx paragraph parsing **bold**, *italic*, and [label](url) links."""
    tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\(https?://[^)]+\))", text)
    link_re = re.compile(r"^\[([^\]]+)\]\((https?://[^)]+)\)$")
    for tok in tokens:
        if not tok:
            continue
        m = link_re.match(tok)
        if m:
            _add_docx_hyperlink(paragraph, m.group(2), m.group(1), font_name=font_name, font_size=font_size)
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            run = paragraph.add_run(tok[2:-2])
            run.bold = True
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            run = paragraph.add_run(tok[1:-1])
            run.bold = base_bold
            run.italic = True
        else:
            run = paragraph.add_run(tok)
            run.bold = base_bold
        run.font.name = font_name
        run.font.size = font_size
        run.font.color.rgb = BLACK


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def build_report_sections(res: dict, art: dict, links: dict) -> list[dict]:
    """Build structured sections matching exact UNDIP Kelompok 11 format."""
    cs = res["clean_stats"]
    sb = res["split_boundaries"]
    ds = res["descriptive_stats"]
    c_opt = int(res["selected_c"])
    lam_opt = float(res["selected_lambda"])
    cluster_labels: list[str] = art["cluster_labels"]
    colab_url = links.get("colab_url", "https://colab.research.google.com/drive/16HuDWxvMATIR4tWVLtUFNNe_x_u4qZWg")
    streamlit_url = links.get("streamlit_url", "https://ispu-dki-fuzzy-fcm-hybrid.streamlit.app/")
    github_url = links.get("github_url", "https://github.com/m3n54/ispu-dki-fuzzy-fcm-hybrid")

    desc_rows = []
    for k, label in [
        ("pm10", "PM10 (Sub-Indeks *t*)"),
        ("pm25", "PM2.5 (Sub-Indeks *t*)"),
        ("so2", "SO2 (Sub-Indeks *t*)"),
        ("co", "CO (Sub-Indeks *t*)"),
        ("o3", "O3 (Sub-Indeks *t*)"),
        ("no2", "NO2 (Sub-Indeks *t*)"),
        ("max", "ISPU Max Hari Ini (*t*)"),
        ("target_ispu_next", "Target ISPU Besok (*t*+1)"),
    ]:
        d = ds[k]
        desc_rows.append([
            label,
            fmt_int_id(d["count"]),
            fmt_id(d["mean"], 2),
            fmt_id(d["std"], 2),
            fmt_id(d["min"], 0),
            fmt_id(d["median"], 0),
            fmt_id(d["max"], 0),
        ])

    sweep_rows = []
    for r in res["sweep_results"]:
        sweep_rows.append([
            str(r["c"]),
            fmt_id(r["pc"], 4),
            fmt_id(r["pe"], 4),
            fmt_id(r["xb"], 4),
            fmt_id(r["jm"], 2),
            str(r["n_iter"]),
            fmt_id(r["best_lambda"], 1),
            fmt_id(r["lse_val_rmse"], 4),
        ])

    cluster_prof_rows = []
    for row in res["cluster_summary"]:
        cluster_prof_rows.append([
            row["label"],
            fmt_int_id(row["count"]),
            fmt_id(row["center_pm10"], 2),
            fmt_id(row["center_pm25"], 2),
            fmt_id(row["center_so2"], 2),
            fmt_id(row["center_co"], 2),
            fmt_id(row["center_o3"], 2),
            fmt_id(row["center_no2"], 2),
            fmt_id(row["mean_target_ispu"], 2),
        ])

    sample_u_rows = []
    for item in res["sample_memberships"]:
        u_strs = [fmt_id(u, 4) for u in item["memberships"]]
        sample_u_rows.append([
            item["tanggal"],
            item["stasiun"].split("(")[0].strip(),
            f"{item['pm10']:.0f} / {item['pm25']:.0f}",
            *u_strs,
            f"C{item['dominant_cluster']}",
        ])

    prem = res["premise_orig"]
    prem_rows = []
    for i in range(c_opt):
        for j, col in enumerate(FEATURE_COLS):
            prem_rows.append([
                f"C{i+1}",
                FEATURE_LABELS[col],
                fmt_id(prem["init_centers"][i][j], 2),
                fmt_id(prem["opt_centers"][i][j], 2),
                fmt_id(prem["init_sigmas"][i][j], 2),
                fmt_id(prem["opt_sigmas"][i][j], 2),
            ])

    conseq = art["ts_hybrid"]["consequents"]
    conseq_rows = []
    for i in range(c_opt):
        conseq_rows.append([
            f"Aturan {i+1}",
            fmt_id(conseq[i][0], 3),
            *[fmt_id(conseq[i][j + 1], 3) for j in range(len(FEATURE_COLS))],
        ])

    def build_ablation_rows(split_key: str) -> list[list[str]]:
        out = []
        for model_name, m in res["ablation"][split_key].items():
            out.append([
                model_name,
                fmt_id(m["rmse"], 3),
                fmt_id(m["mae"], 3),
                fmt_id(m["r2"], 4),
                f"{fmt_id(m['mape'], 2)}%",
                f"{fmt_id(m['category_accuracy']*100, 2)}%",
                f"{fmt_id(m['adjacent_category_accuracy']*100, 2)}%",
            ])
        return out

    scen_rows = []
    for sc in res["scenarios"]:
        inp = sc["inputs"]
        inp_str = ", ".join([f"{FEATURE_LABELS[c]}={inp[c]:.0f}" for c in FEATURE_COLS])
        w_str = " | ".join([f"w̄{i+1}={fmt_id(sc['w_bar'][i], 3)}" for i in range(c_opt)])
        f_str = " | ".join([f"f{i+1}={fmt_id(sc['rule_outputs'][i], 2)}" for i in range(c_opt)])
        err = abs(sc["predicted_ispu_next"] - sc["actual_ispu_next"])
        scen_rows.append([
            sc["preset_name"],
            f"{sc['stasiun'].split('(')[0].strip()} ({sc['tanggal']})",
            inp_str,
            w_str,
            f_str,
            f"{fmt_id(sc['predicted_ispu_next'], 2)} ({sc['predicted_category_next']})",
            f"{sc['actual_ispu_next']:.0f} ({sc['actual_category_next']})",
            fmt_id(err, 2),
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

    total_paired = res["n_paired"]
    sedang_n = res["target_category_counts"].get("SEDANG", 0)
    ts_n = res["target_category_counts"].get("TIDAK SEHAT", 0)
    baik_n = res["target_category_counts"].get("BAIK", 0)

    sections = [
        {
            "type": "cover",
            "top_lines": [
                "PROYEK BESAR INDIVIDU",
                "Prediksi Dini Indeks Standar Pencemar Udara (ISPU) Hari Berikutnya di DKI Jakarta Menggunakan Takagi-Sugeno Orde-1 Berbasis *Fuzzy C-Means* dan *Hybrid Learning*",
            ],
            "course_line": "MATA KULIAH KOMPUTASI LUNAK",
            "logo_path": str(LOGO_PATH),
            "members": [
                ("Menza Isaiah Tampubolon", "24060124140138"),
            ],
            "institution_lines": [
                "DEPARTEMEN ILMU KOMPUTER / INFORMATIKA",
                "FAKULTAS SAINS DAN MATEMATIKA",
                "UNIVERSITAS DIPONEGORO",
                "2026",
            ],
        },
        {
            "type": "chapter",
            "heading": "BAB 1. PENDAHULUAN",
            "blocks": [
                {"h2": "1.1 Latar Belakang"},
                {"p": (
                    "Kualitas udara ambien di Provinsi DKI Jakarta merupakan persoalan lingkungan dan kesehatan masyarakat yang krusial. "
                    "Aktivitas transportasi padat, emisi industri, serta dinamika meteorologi perkotaan menyebabkan fluktuasi harian konsentrasi polutan seperti "
                    "partikulat kasar (PM10), partikulat halus (PM2.5), sulfur dioksida (SO2), karbon monoksida (CO), ozon (O3), dan nitrogen dioksida (NO2). "
                    "Sejak diberlakukannya Peraturan Menteri Lingkungan Hidup dan Kehutanan (Permen LHK) Nomor P.14/MENLHK/SETJEN/KUM.1/7/2020, "
                    "parameter PM2.5 secara resmi dimasukkan ke dalam perhitungan Indeks Standar Pencemar Udara (ISPU) harian yang dipantau oleh lima "
                    "Stasiun Pemantau Kualitas Udara (SPKU) Dinas Lingkungan Hidup DKI Jakarta (Kementerian Lingkungan Hidup dan Kehutanan [KLHK], 2020; "
                    "Dinas Lingkungan Hidup Provinsi DKI Jakarta, 2021)."
                )},
                {"p": (
                    "Pentingnya penelitian prediksi dini ISPU hari berikutnya (ISPU(*t*+1)) terletak pada fungsinya sebagai sistem peringatan dini (*early warning system*). "
                    "Apabila pemangku kebijakan dan masyarakat hanya mengetahui status mutu udara pada hari berjalan, langkah mitigasi seperti imbauan pembatasan "
                    "aktivitas luar ruang atau rekayasa lalu lintas sering terlambat dilakukan. Memprediksi skor ISPU besok berdasarkan profil enam polutan hari ini "
                    "memungkinkan tindakan preventif sebelum episode polusi tinggi terjadi."
                )},
                {"p": (
                    "Pendekatan konvensional dalam pemantauan dan peramalan kualitas udara memiliki dua kelemahan mendasar. Pertama, klasifikasi konvensional menggunakan "
                    "ambang batas tegas (*crisp threshold*)—misalnya skor 50 dikategorikan Baik sedangkan skor 51 langsung menjadi Sedang—yang tidak merepresentasikan "
                    "transisi risiko kesehatan secara kontinu. Kedua, model statistik linear global seperti Regresi Linear Berganda (*Ordinary Least Squares*/OLS) maupun model persistensi "
                    "(ISPU(*t*+1) = ISPU(*t*)) kesulitan menangkap hubungan non-linear serta interaksi antarpolutan yang berbeda antara kondisi udara bersih dan episode polusi tinggi."
                )},
                {"p": (
                    "Untuk mengatasi hal tersebut, penelitian ini menerapkan model *fuzzy* Takagi-Sugeno Orde-1 (Takagi & Sugeno, 1985). Model *fuzzy* mampu memodelkan ketidakpastian dan "
                    "perubahan bertahap antar-rezim polusi melalui himpunan *fuzzy* (Zadeh, 1965), sementara keluaran berupa kombinasi linear variabel masukan pada Takagi-Sugeno Orde-1 "
                    "memberikan kemampuan aproksimasi fungsi non-linear yang jauh lebih akurat dibandingkan keluaran konstanta maupun himpunan linguistik statis."
                )},
                {"p": (
                    "Penentuan fungsi keanggotaan dan basis aturan *fuzzy* secara subjektif dari pakar sering kali tidak optimal untuk data multivariat enam dimensi. "
                    "Oleh karena itu, digunakan metode *clustering* *Fuzzy C-Means* (FCM) (Bezdek, 1981) untuk mempartisi ruang data secara objektif dan berbasis data. "
                    "Berbeda dari *hard clustering* yang memaksa setiap hari pengamatan masuk secara mutlak ke satu kelompok saja, FCM memberikan derajat "
                    "keanggotaan kontinu pada interval [0, 1] sehingga mencerminkan hari-hari transisi polusi secara lebih realistis."
                )},
                {"p": (
                    "Meskipun pusat dan lebar fungsi keanggotaan awal dapat diturunkan langsung dari FCM, sistem *fuzzy* yang hanya mengandalkan inisialisasi *clustering* "
                    "belum menjamin kesalahan prediksi minimum. Karena itu, diperlukan tahap estimasi parameter secara sistematis menggunakan pendekatan "
                    "*Hybrid Learning* (Jang, 1993) yang menggabungkan *Regularized Least Squares Estimation* (LSE) untuk menyetel parameter konsekuen linear secara eksak "
                    "dan *Gradient Descent* untuk menyempurnakan parameter premis Gaussian hingga model menjadi lebih representatif (Karki, 2023)."
                )},
                {"h2": "1.2 Rumusan Masalah"},
                {"p": "Berdasarkan latar belakang di atas, rumusan masalah penelitian ini adalah sebagai berikut."},
                {"ol": [
                    "Bagaimana membangun model *fuzzy* Takagi-Sugeno Orde-1 untuk memprediksi skor dan kategori ISPU hari berikutnya (ISPU(*t*+1)) di DKI Jakarta dari 6 parameter polutan harian?",
                    "Bagaimana proses pembentukan kelompok data polusi udara DKI Jakarta secara objektif menggunakan metode *clustering* *Fuzzy C-Means* (FCM) dan bagaimana validitas jumlah *cluster*-nya?",
                    "Bagaimana estimasi parameter premis Gaussian dan konsekuen linear dilakukan menggunakan metode *Hybrid Learning* (*Regularized* LSE dan *Gradient Descent*) agar model lebih representatif?",
                    "Bagaimana performa model *fuzzy* yang dihasilkan dibandingkan dengan model pra-estimasi serta *baseline* konvensional (*Persistence* dan Regresi Linear OLS)?",
                ]},
                {"h2": "1.3 Tujuan Studi Kasus"},
                {"p": "Sejalan dengan rumusan masalah, tujuan penelitian ini adalah sebagai berikut."},
                {"ol": [
                    "Membangun sistem prediksi dini berbasis logika *fuzzy* Takagi-Sugeno Orde-1 untuk memprediksi nilai kontinu dan kategori mutu udara ISPU(*t*+1) DKI Jakarta.",
                    "Menerapkan algoritma *Fuzzy C-Means* (FCM) disertai evaluasi *Partition Coefficient* (PC), *Partition Entropy* (PE), dan *Xie-Beni Index* (XB) guna memperoleh struktur *cluster* data yang objektif.",
                    "Melakukan estimasi parameter premis dan konsekuen *fuzzy* secara sistematis melalui studi ablasi 3 tahap (pra-estimasi Orde-0, *Regularized* LSE, dan *Hybrid Learning*).",
                    "Mengimplementasikan keseluruhan model dalam modul Python teruji, *notebook* Google Colab interaktif, serta antarmuka grafis (GUI) berbasis Streamlit.",
                ]},
            ],
        },
        {
            "type": "chapter",
            "heading": "BAB 2. TINJAUAN PUSTAKA",
            "blocks": [
                {"h2": "2.1 Konsep Dasar Logika Fuzzy"},
                {"p": (
                    "Logika *fuzzy* pertama kali diperkenalkan oleh Lotfi A. Zadeh (1965) sebagai perluasan dari teori himpunan klasik (*crisp set*). "
                    "Jika himpunan klasik hanya mengenal keanggotaan biner {0, 1}, maka himpunan *fuzzy* *Ã* pada semesta pembicaraan *U* didefinisikan melalui "
                    "fungsi keanggotaan *μ_Ã*(*x*) ∈ [0, 1] yang menyatakan derajat kebenaran keanggotaan suatu nilai *x* terhadap himpunan *Ã* (Zadeh, 1965):"
                )},
                {"eq": {"text": "Ã = { (x, μ_Ã(x)) | x ∈ U }", "html": "<i>Ã</i> = { (<i>x</i>, <i>μ<sub>Ã</sub></i>(<i>x</i>)) | <i>x</i> ∈ <i>U</i> }"}},
                {"p": (
                    "Konsep ini memungkinkan representasi variabel linguistik, misalnya Polusi Rendah dan Polusi Tinggi, secara matematis. "
                    "Pada penelitian ini digunakan fungsi keanggotaan Gaussian untuk setiap variabel masukan *j* ∈ {1, ..., *D*} pada aturan ke-*i*:"
                )},
                {
                    "eq": {
                        "text": "μ_ij(x_j) = exp( -(x_j - v_ij)^2 / (2 * σ_ij^2) )",
                        "html": "<i>μ<sub>ij</sub></i>(<i>x<sub>j</sub></i>) = exp( −(<i>x<sub>j</sub></i> − <i>v<sub>ij</sub></i>)<sup>2</sup> / (2<i>σ<sub>ij</sub></i><sup>2</sup>) )",
                    }
                },
                {"p": (
                    "di mana *v_ij* menyatakan pusat fungsi keanggotaan dan *σ_ij* menyatakan lebar (simpangan baku) kurva. "
                    "Operasi logika AND antar-anteseden pada satu aturan menggunakan T-norm perkalian aljabar (*product operator*) untuk menghasilkan "
                    "kekuatan penyulutan (*firing strength*) *w_i*(**x**) = ∏ *μ_ij*(*x_j*), yang kemudian dinormalisasi menjadi *w̄_i*(**x**) = *w_i*(**x**) / ∑ *w_r*(**x**). "
                    "Inferensi agregat mengubah himpunan hasil aturan menjadi nilai keluaran numerik tunggal melalui defuzzifikasi."
                )},
                {"h2": "2.2 Model Fuzzy Takagi-Sugeno Orde-1"},
                {"p": (
                    "Model *fuzzy* Takagi-Sugeno (TSK) diperkenalkan oleh Takagi dan Sugeno (1985). Berbeda dengan model Mamdani yang menggunakan himpunan *fuzzy* pada bagian konsekuen, "
                    "model Takagi-Sugeno Orde-1 menggunakan fungsi linear dari variabel masukan sebagai konsekuen setiap aturan (Takagi & Sugeno, 1985):"
                )},
                {
                    "eq": {
                        "text": "Aturan i: IF x_1 is A_i1 AND ... AND x_D is A_iD THEN f_i(x) = p_i0 + p_i1 x_1 + ... + p_iD x_D",
                        "html": "Aturan <i>i</i>: IF <i>x</i><sub>1</sub> is <i>A</i><sub><i>i</i>1</sub> AND ... AND <i>x<sub>D</sub></i> is <i>A<sub>iD</sub></i> THEN <i>f<sub>i</sub></i>(<b>x</b>) = <i>p</i><sub><i>i</i>0</sub> + <i>p</i><sub><i>i</i>1</sub><i>x</i><sub>1</sub> + ... + <i>p<sub>iD</sub>x<sub>D</sub></i>",
                    }
                },
                {"p": (
                    "Keluaran akhir sistem diperoleh melalui defuzzifikasi rata-rata terbobot (*weighted average*), yaitu *ŷ*(**x**) = ∑ *w̄_i*(**x**) · *f_i*(**x**). "
                    "Kelebihan utama model Takagi-Sugeno Orde-1 adalah efisiensi komputasi karena tidak memerlukan proses defuzzifikasi integral yang berat, "
                    "fleksibilitas tinggi dalam memodelkan sistem non-linear kompleks melalui interpolasi mulus antar-bidang linear lokal, serta kompatibilitas langsung "
                    "dengan metode estimasi parameter berbasis aljabar linear (*Least Squares Estimation*) dan optimasi gradien (Takagi & Sugeno, 1985; Jang, 1993)."
                )},
                {"h2": "2.3 Clustering Fuzzy (Fuzzy C-Means)"},
                {"p": (
                    "*Clustering* adalah teknik pembelajaran tak terawasi (*unsupervised learning*) untuk mengelompokkan data ke dalam *cluster* berdasarkan kemiripan fitur. "
                    "Pada *hard clustering*, setiap titik data dimiliki secara eksklusif oleh satu *cluster* saja. Sebaliknya, pada *fuzzy clustering* yang dirumuskan oleh "
                    "Bezdek (1981) melalui algoritma *Fuzzy C-Means* (FCM), setiap titik data **x**_k memiliki derajat keanggotaan *u_ik* ∈ [0, 1] terhadap setiap *cluster* *i* "
                    "dengan kendala partisi ∑ *u_ik* = 1. Algoritma FCM meminimalkan fungsi objektif berbobot berikut (Bezdek, 1981):"
                )},
                {
                    "eq": {
                        "text": "J_m(U, V) = ∑_{k=1}^N ∑_{i=1}^c (u_ik)^m ||x_k - v_i||_2^2 ,   m = 2,0",
                        "html": "<i>J<sub>m</sub></i>(<i>U</i>, <i>V</i>) = ∑<sub><i>k</i>=1</sub><sup><i>N</i></sup> ∑<sub><i>i</i>=1</sub><sup><i>c</i></sup> (<i>u<sub>ik</sub></i>)<sup><i>m</i></sup> ‖<b>x</b><sub><i>k</i></sub> − <b>v</b><sub><i>i</i></sub>‖<sub>2</sub><sup>2</sup> ,   <i>m</i> = 2,0",
                    }
                },
                {"p": "Pembaruan derajat keanggotaan *u_ik* dan pusat *cluster* **v**_i pada setiap iterasi dihitung melalui persamaan berikut (Bezdek, 1981):"},
                {
                    "eq": {
                        "text": "u_ik = 1 / ∑_{r=1}^c ( ||x_k - v_i||_2 / ||x_k - v_r||_2 )^(2/(m-1)) ,   v_i = ( ∑_{k=1}^N (u_ik)^m x_k ) / ( ∑_{k=1}^N (u_ik)^m )",
                        "html": "<i>u<sub>ik</sub></i> = 1 / ∑<sub><i>r</i>=1</sub><sup><i>c</i></sup> (‖<b>x</b><sub><i>k</i></sub> − <b>v</b><sub><i>i</i></sub>‖<sub>2</sub> / ‖<b>x</b><sub><i>k</i></sub> − <b>v</b><sub><i>r</i></sub>‖<sub>2</sub>)<sup>2/(<i>m</i>−1)</sup> ,   <b>v</b><sub><i>i</i></sub> = ∑<sub><i>k</i>=1</sub><sup><i>N</i></sup> (<i>u<sub>ik</sub></i>)<sup><i>m</i></sup> <b>x</b><sub><i>k</i></sub> / ∑<sub><i>k</i>=1</sub><sup><i>N</i></sup> (<i>u<sub>ik</sub></i>)<sup><i>m</i></sup>",
                    }
                },
                {"p": (
                    "Proses iterasi dihentikan apabila kriteria konvergensi ||*U*^(t) − *U*^(t−1)||_∞ < 10^−5 terpenuhi. Untuk menilai kualitas partisi secara objektif digunakan tiga indeks validitas: "
                    "*Partition Coefficient* (PC = (1/*N*)∑∑ *u_ik*^2, lebih tinggi lebih baik), *Partition Entropy* (PE = −(1/*N*)∑∑ *u_ik* ln *u_ik*, lebih rendah lebih baik), serta "
                    "*Xie-Beni Index* (XB) yang membandingkan kekompakan *intra-cluster* terhadap jarak minimum antar-pusat *cluster* (Xie & Beni, 1991)."
                )},
                {"h2": "2.4 Estimasi Parameter dengan Hybrid Learning"},
                {"p": (
                    "Dalam kerangka ANFIS/Takagi-Sugeno adaptif yang diperkenalkan Jang (1993), setiap *epoch* pelatihan *Hybrid Learning* terbagi menjadi dua fase. "
                    "Pertama, *forward pass*, di mana parameter premis (*v_ij*, *σ_ij*) dianggap tetap sehingga keluaran prediksi menjadi linear terhadap vektor parameter konsekuen **θ**. "
                    "Untuk mencegah multikolinearitas dan *overfitting* pada data nyata, digunakan *Regularized Least Squares Estimation* (*Ridge* LSE): **θ̂** = (**A**^T **A** + *λ***I**)^(−1) **A**^T **y**. "
                    "Kedua, *backward pass*, di mana parameter konsekuen **θ** dianggap tetap dan kesalahan prediksi dirambatkan balik menggunakan *Gradient Descent* untuk memperbarui *v_ij* dan log(*σ_ij*) (Jang, 1993). "
                    "Studi terkini oleh Karki (2023) juga mengonfirmasi bahwa penyetelan parameter berbasis neuro-fuzzy adaptif efektif memodelkan dinamika non-linear indeks kualitas udara perkotaan."
                )},
            ],
        },
        {
            "type": "chapter",
            "heading": "BAB 3. METODOLOGI",
            "blocks": [
                {"h2": "3.1 Sumber dan Prapemrosesan Dataset"},
                {"p": (
                    f"Penelitian ini menggunakan data historis nyata Indeks Standar Pencemar Udara (ISPU) Provinsi DKI Jakarta tahun 2021 yang bersumber dari "
                    f"Dinas Lingkungan Hidup Provinsi DKI Jakarta pada Portal Satu Data Jakarta untuk lima stasiun SPKU: {', '.join(cs['stations'])} "
                    f"(Dinas Lingkungan Hidup Provinsi DKI Jakarta, 2021). Dari total {fmt_int_id(cs['raw_rows'])} baris observasi mentah selama 12 bulan (1 Januari–31 Desember 2021), "
                    f"dilakukan perbaikan pergeseran *header* pada berkas resmi bulan Juni dan Juli 2021 serta penghapusan {fmt_int_id(cs['dropped_invalid_or_missing'])} baris dengan nilai kosong "
                    f"(terutama awal Januari sebelum sensor PM2.5 beroperasi penuh), sehingga diperoleh {fmt_int_id(cs['clean_rows'])} baris bersih."
                )},
                {"p": (
                    f"Selanjutnya, untuk membentuk masalah prediksi hari berikutnya (*t* → *t*+1) yang bebas kebocoran data (*zero data leakage*), setiap observasi hari ke-*t* dipasangkan dengan "
                    f"nilai ISPU maksimum hari ke-(*t*+1) secara ketat di dalam stasiun yang sama dan hanya apabila selisih tanggal tepat 1 hari. Proses ini menghasilkan {fmt_int_id(res['n_paired'])} pasangan data berurutan. "
                    f"Data dibagi secara kronologis (*time-series split*) tanpa pengacakan menjadi Data Latih ({fmt_int_id(sb['n_train'])} data, {sb['train_start']} s.d. {sb['train_end']}), "
                    f"Data Validasi ({fmt_int_id(sb['n_val'])} data, {sb['val_start']} s.d. {sb['val_end']}), dan Data Uji ({fmt_int_id(sb['n_test'])} data, {sb['test_start']} s.d. {sb['test_end']}). "
                    f"Normalisasi *Z-score* dihitung hanya dari statistik Data Latih."
                )},
                {"h2": "3.2 Perancangan Sistem Fuzzy"},
                {"p": "Rancangan arsitektur sistem prediksi dini ISPU meliputi komponen berikut."},
                {"ol": [
                    "*Input* Sistem: Vektor 6 sub-indeks polutan harian pada hari ke-*t*, yaitu **x**_t = [PM10, PM2.5, SO2, CO, O3, NO2]_t.",
                    "*Output* Sistem: Prediksi skor kontinu ISPU hari berikutnya *ŷ*_{t+1} beserta pemetaan kategori mutu udara resmi Permen LHK No. P.14/2020 (KLHK, 2020): Baik (0–50), Sedang (51–100), Tidak Sehat (101–200), Sangat Tidak Sehat (201–300), dan Berbahaya (>300).",
                    f"Basis Aturan (*Rule Base*): Dibentuk otomatis sebanyak *c* = {c_opt} aturan *fuzzy* Takagi-Sugeno Orde-1 dari hasil partisi FCM dengan representasi sebagai berikut:",
                ]},
                {
                    "eq": {
                        "text": "R1: IF PM10 Rendah AND PM2.5 Rendah AND SO2 Rendah AND CO Rendah AND O3 Rendah AND NO2 Rendah THEN f1(x) = p10 + sum(p1j xj)\nR2: IF PM10 Tinggi AND PM2.5 Tinggi AND SO2 Tinggi AND CO Tinggi AND O3 Tinggi AND NO2 Tinggi THEN f2(x) = p20 + sum(p2j xj)",
                        "html": "R1: IF PM10 Rendah AND PM2.5 Rendah AND SO2 Rendah AND CO Rendah AND O3 Rendah AND NO2 Rendah THEN <i>f</i><sub>1</sub>(<b>x</b>) = <i>p</i><sub>10</sub> + ∑ <i>p</i><sub>1<i>j</i></sub><i>x<sub>j</sub></i><br/>R2: IF PM10 Tinggi AND PM2.5 Tinggi AND SO2 Tinggi AND CO Tinggi AND O3 Tinggi AND NO2 Tinggi THEN <i>f</i><sub>2</sub>(<b>x</b>) = <i>p</i><sub>20</sub> + ∑ <i>p</i><sub>2<i>j</i></sub><i>x<sub>j</sub></i>",
                    }
                },
                {"h2": "3.3 Rancangan Evaluasi dan Studi Ablasi"},
                {"p": (
                    "Untuk membuktikan kontribusi setiap komponen secara ilmiah, performa model dievaluasi dengan metrik RMSE, MAE, R², MAPE, Akurasi Kategori, dan Akurasi ±1 Kategori melalui perbandingan lima skenario: "
                    "(1) *Persistence Baseline* (*ŷ*_{t+1} = ISPU_t), (2) Regresi Linear Berganda (OLS), (3) Tahap 0: FCM + Sugeno Orde-0 (tanpa estimasi konsekuen linear), "
                    "(4) Tahap 1: FCM + Takagi-Sugeno Orde-1 + *Regularized* LSE, serta (5) Tahap 2: FCM + Takagi-Sugeno Orde-1 + *Hybrid Learning* (LSE + *Gradient Descent*)."
                )},
            ],
        },
        {
            "type": "chapter",
            "heading": "BAB 4. PEMBAHASAN DAN HASIL",
            "blocks": [
                {"h2": "4.1 Karakteristik Dataset ISPU DKI Jakarta"},
                {"p": (
                    f"Tabel 4.1 merangkum statistik deskriptif dari {fmt_int_id(res['n_paired'])} pasangan observasi harian DKI Jakarta. Polutan PM2.5 memiliki rata-rata sub-indeks tertinggi "
                    f"({fmt_id(ds['pm25']['mean'])} ± {fmt_id(ds['pm25']['std'])}), diikuti PM10 ({fmt_id(ds['pm10']['mean'])}). Distribusi komposisi kategori target menunjukkan dominasi "
                    f"kategori Sedang ({fmt_int_id(sedang_n)} hari atau 75,8%), diikuti Tidak Sehat ({fmt_int_id(ts_n)} hari atau 15,9%) dan Baik ({fmt_int_id(baik_n)} hari atau 8,2%), seperti terlihat pada Gambar 4.1."
                )},
                {"table": {"label": "Tabel 4.1", "caption": "Statistik deskriptif variabel input dan target (N = 1.444)", "headers": ["Variabel", "N", "Rata-rata", "Std Dev", "Min", "Median", "Maks"], "rows": desc_rows}},
                {"fig": {"path": str(ART_DIR / "figures" / "fig1_data_distribution.png"), "label": "Gambar 4.1", "caption": "Distribusi sub-indeks 6 polutan dan komposisi kategori target ISPU(t+1)."}},
                {"h2": "4.2 Penentuan Jumlah Cluster Optimal dan Hasil Fuzzy C-Means"},
                {"p": (
                    f"Eksperimen sapuan jumlah *cluster* *c* ∈ {{2, 3, 4, 5, 6}} pada Data Latih (Tabel 4.2) menunjukkan bahwa nilai *Partition Coefficient* (PC) tertinggi ({fmt_id(res['sweep_results'][0]['pc'], 4)}), "
                    f"*Partition Entropy* (PE) terendah ({fmt_id(res['sweep_results'][0]['pe'], 4)}), serta kesalahan validasi LSE terendah (Val RMSE = {fmt_id(res['sweep_results'][0]['lse_val_rmse'], 4)}) "
                    f"dicapai pada *c* = {c_opt} (Gambar 4.2). Penyempurnaan parameter regularisasi *Ridge* melalui *Validation RMSE* *Hybrid Learning* selanjutnya memilih *λ* = {fmt_id(lam_opt, 1)} (Val RMSE = {fmt_id(val_m_s2['rmse'], 4)})."
                )},
                {"table": {"label": "Tabel 4.2", "caption": "Hasil evaluasi validitas cluster FCM dan validasi RMSE", "headers": ["c", "PC", "PE", "Xie-Beni (XB)", "J_m", "Iterasi", "λ LSE", "Val RMSE"], "rows": sweep_rows}},
                {"fig": {"path": str(ART_DIR / "figures" / "fig2_cluster_selection.png"), "label": "Gambar 4.2", "caption": "Kurva validitas partisi FCM (PC, XB, Val RMSE) dan konvergensi fungsi objektif J_m."}},
                {"p": (
                    f"Pada *c* = {c_opt}, FCM konvergen dalam {art['fcm']['n_iter']} iterasi menghasilkan dua pusat *cluster* terurut yang interpretatif (Tabel 4.3): "
                    f"{cluster_labels[0]} (485 hari latih, rata-rata target ISPU berikutnya {fmt_id(res['cluster_summary'][0]['mean_target_ispu'])}) dan "
                    f"{cluster_labels[1]} (580 hari latih, rata-rata target ISPU berikutnya {fmt_id(res['cluster_summary'][1]['mean_target_ispu'])}). "
                    f"Tabel 4.4 memperlihatkan contoh derajat keanggotaan *fuzzy* *u_ik* pada lima observasi awal, sedangkan proyeksi PCA 2D ditampilkan pada Gambar 4.3."
                )},
                {"table": {"label": "Tabel 4.3", "caption": "Pusat cluster FCM dalam skala sub-indeks ISPU asli", "headers": ["Cluster", "N", "PM10", "PM2.5", "SO2", "CO", "O3", "NO2", "Rata2 ISPU(t+1)"], "rows": cluster_prof_rows}},
                {"table": {"label": "Tabel 4.4", "caption": "Contoh matriks derajat keanggotaan FCM (u_ik)", "headers": ["Tanggal", "Stasiun", "PM10 / PM2.5", "u_C1", "u_C2", "Dominan"], "rows": sample_u_rows}},
                {"fig": {"path": str(ART_DIR / "figures" / "fig3_fcm_clusters_pca.png"), "label": "Gambar 4.3", "caption": "Proyeksi PCA 2D hasil clustering FCM dan profil polutan tiap pusat cluster."}},
                {"h2": "4.3 Hasil Estimasi Parameter Premis dan Konsekuen"},
                {"p": (
                    f"Tahap *Hybrid Learning* dijalankan dengan *early stopping* berbasis Data Validasi dan mencapai titik terbaik pada *epoch* ke-{int(best_ep_row['epoch'])} "
                    f"(Val RMSE = {fmt_id(best_ep_row['val_rmse'], 3)}, Train RMSE = {fmt_id(best_ep_row['train_rmse'], 3)}). "
                    f"Tabel 4.5 dan Gambar 4.4 membandingkan parameter Gaussian sebelum dan sesudah optimasi *Gradient Descent*, "
                    f"sedangkan Tabel 4.6 menampilkan koefisien konsekuen linear Takagi-Sugeno Orde-1 hasil *Regularized* LSE, dengan kurva konvergensi pada Gambar 4.5."
                )},
                {"table": {"label": "Tabel 4.5", "caption": "Parameter premis Gaussian sebelum (FCM awal) dan sesudah Hybrid Learning", "headers": ["Aturan", "Polutan", "v Awal", "v Hybrid", "σ Awal", "σ Hybrid"], "rows": prem_rows}},
                {"fig": {"path": str(ART_DIR / "figures" / "fig4_membership_functions.png"), "label": "Gambar 4.4", "caption": "Perbandingan kurva fungsi keanggotaan Gaussian sebelum dan sesudah Hybrid Learning."}},
                {"table": {"label": "Tabel 4.6", "caption": "Koefisien konsekuen linear Takagi-Sugeno Orde-1 (input terstandarisasi)", "headers": ["Aturan", "Bias p0", "p_PM10", "p_PM2.5", "p_SO2", "p_CO", "p_O3", "p_NO2"], "rows": conseq_rows}},
                {"fig": {"path": str(ART_DIR / "figures" / "fig5_hybrid_training_curve.png"), "label": "Gambar 4.5", "caption": "Kurva konvergensi Train RMSE dan Validation RMSE selama pelatihan Hybrid Learning."}},
                {"h2": "4.4 Simulasi Inferensi Fuzzy dan Hasil Studi Ablasi"},
                {"p": "Tabel 4.7 menampilkan penelusuran inferensi langkah demi langkah pada tiga skenario uji nyata (Baik, Sedang, dan Polusi Tinggi). Tabel 4.8 dan Tabel 4.9 menyajikan perbandingan performa kuantitatif pada Data Validasi dan Data Uji, serta visualisasi perbandingan pada Gambar 4.6 dan Gambar 4.7."},
                {"table": {"label": "Tabel 4.7", "caption": "Simulasi inferensi fuzzy TSK-1 pada tiga kasus nyata Test Set", "headers": ["Skenario", "Lokasi & Tgl", "Input Polutan", "Bobot w̄", "Konsekuen f_i", "Prediksi", "Aktual", "Galat"], "rows": scen_rows}},
                {"table": {"label": "Tabel 4.8", "caption": "Hasil studi ablasi dan baseline pada Data Validasi (12 Sep–5 Nov 2021)", "headers": ["Model", "RMSE", "MAE", "R²", "MAPE", "Akurasi Kat.", "Akurasi ±1 Kat."], "rows": build_ablation_rows("val")}},
                {"table": {"label": "Tabel 4.9", "caption": "Hasil studi ablasi dan baseline pada Data Uji (6 Nov–30 Des 2021)", "headers": ["Model", "RMSE", "MAE", "R²", "MAPE", "Akurasi Kat.", "Akurasi ±1 Kat."], "rows": build_ablation_rows("test")}},
                {"fig": {"path": str(ART_DIR / "figures" / "fig6_ablation_comparison.png"), "label": "Gambar 4.6", "caption": "Perbandingan RMSE, MAE, dan R² seluruh model pada studi ablasi Data Uji."}},
                {"fig": {"path": str(ART_DIR / "figures" / "fig7_actual_vs_predicted.png"), "label": "Gambar 4.7", "caption": "Grafik aktual vs prediksi, deret waktu stasiun DKI1, dan matriks kebingungan kategori pada Data Uji."}},
            ],
        },
        {
            "type": "chapter",
            "heading": "BAB 5. IMPLEMENTASI PROGRAM",
            "blocks": [
                {"h2": "5.1 Arsitektur Kode Python dan Google Colab"},
                {"p": (
                    "Seluruh implementasi dibangun menggunakan Python 3.12.6 dengan pustaka *NumPy*, *Pandas*, *SciPy*, *Scikit-Learn*, *Matplotlib*, dan *Streamlit*. "
                    "Struktur kode dipisahkan secara modular ke dalam berkas *src/data_loader.py* (pembersihan data dan pasangan *t* → *t*+1 bebas kebocoran), "
                    "*src/fcm.py* (*Fuzzy C-Means* dan metrik validitas PC/PE/XB), *src/sugeno_hybrid.py* (Takagi-Sugeno Orde-1, LSE, dan *Hybrid Learning* beserta verifikasi gradien numerik), "
                    "*src/evaluation.py* (metrik regresi dan pemetaan Permen LHK), serta *src/visualization.py*."
                )},
                {"code": code_snippet},
                {"p": (
                    f"Notebook interaktif lengkap yang dapat dijalankan ulang satu klik di Google Colab (dengan akses publik *Viewer*) tersedia pada berkas "
                    f"*notebooks/ISPU_Fuzzy_FCM_Hybrid.ipynb* melalui tautan [{colab_url}]({colab_url}), sedangkan seluruh kode sumber proyek "
                    f"dipublikasikan pada repositori GitHub [{github_url}]({github_url})."
                )},
                {"h2": "5.2 Dokumentasi GUI Interaktif (Streamlit)"},
                {"p": (
                    f"Sebagai antarmuka pengguna aplikasi, dibangun GUI web interaktif berbasis *Streamlit* (*app.py*) yang telah di-*deploy* secara publik pada *Streamlit Community Cloud* "
                    f"sehingga dapat diakses langsung oleh dosen maupun pengguna lain melalui tautan [{streamlit_url}]({streamlit_url}) tanpa perlu instalasi lokal. "
                    "Antarmuka GUI tersebut memenuhi seluruh kriteria fungsional: "
                    "(1) menerima *input* manual 6 parameter polutan maupun *preset* data nyata Jakarta, (2) menampilkan peta *cluster* PCA 2D dan derajat keanggotaan FCM (Gambar 5.1), "
                    "(3) memvisualisasikan kurva fungsi keanggotaan Gaussian lengkap dengan posisi *input* pengguna (Gambar 5.2), (4) merinci tabel inferensi *fuzzy* dan kontribusi tiap aturan (Gambar 5.3), serta "
                    "(5) menyajikan keputusan akhir prediksi skor ISPU(*t*+1), kategori Permen LHK, ringkasan ablasi (Gambar 5.4), dan rekomendasi mitigasi kesehatan. "
                    "Ringkasan seluruh tautan akses publik dokumentasi program dan GUI disajikan pada Tabel 5.1."
                )},
                {"table": {
                    "label": "Tabel 5.1",
                    "caption": "Tautan dokumentasi implementasi program, repositori kode, dan antarmuka pengguna (GUI).",
                    "headers": ["Komponen", "Keterangan Akses", "Tautan Publik"],
                    "rows": [
                        ["Notebook Google Colab", "Eksperimen interaktif FCM, Takagi-Sugeno Orde-1, dan Hybrid Learning (Viewer publik)", f"[{colab_url}]({colab_url})"],
                        ["GUI Web Streamlit", "Antarmuka web interaktif prediksi ISPU(t+1) daring (Streamlit Community Cloud)", f"[{streamlit_url}]({streamlit_url})"],
                        ["Repositori GitHub", "Kode sumber lengkap (src/, app.py, scripts/, tests/, notebooks/)", f"[{github_url}]({github_url})"],
                    ],
                }},
                {"fig": {"path": str(ART_DIR / "gui_screenshots" / "gui_overview.png"), "label": "Gambar 5.1", "caption": "Tampilan utama GUI Streamlit: input sidebar, kartu keputusan akhir, dan visualisasi clustering FCM."}},
                {"fig": {"path": str(ART_DIR / "gui_screenshots" / "gui_mf.png"), "label": "Gambar 5.2", "caption": "Tab fungsi keanggotaan Gaussian interaktif dengan garis penanda nilai input pengguna."}},
                {"fig": {"path": str(ART_DIR / "gui_screenshots" / "gui_inference.png"), "label": "Gambar 5.3", "caption": "Tab rincian inferensi Takagi-Sugeno Orde-1, kontribusi aturan, dan persamaan defuzzifikasi."}},
                {"fig": {"path": str(ART_DIR / "gui_screenshots" / "gui_evaluation.png"), "label": "Gambar 5.4", "caption": "Tab ringkasan evaluasi studi ablasi dan perbandingan metrik pada Data Uji."}},
            ],
        },
        {
            "type": "chapter",
            "heading": "BAB 6. ANALISIS HASIL",
            "blocks": [
                {"h2": "6.1 Interpretasi Hasil Clustering dan Parameter Model"},
                {"p": (
                    f"Partisi optimal *c* = {c_opt} memisahkan dinamika udara Jakarta secara alami menjadi rezim Polusi Rendah (pusat PM2.5 = 67,51, rata-rata target ISPU besok 69,48) "
                    f"dan rezim Polusi Tinggi (pusat PM2.5 = 94,18, rata-rata target ISPU besok 92,61). Setelah proses *Hybrid Learning*, pusat fungsi keanggotaan C1 pada PM2.5 bergeser dari "
                    f"{fmt_id(prem['init_centers'][0][1])} menjadi {fmt_id(prem['opt_centers'][0][1])} dengan lebar kurva yang lebih tajam (*σ* dari {fmt_id(prem['init_sigmas'][0][1])} menjadi {fmt_id(prem['opt_sigmas'][0][1])}), "
                    f"sedangkan C2 memperluas cakupan *σ* dari {fmt_id(prem['init_sigmas'][1][1])} menjadi {fmt_id(prem['opt_sigmas'][1][1])}. Hal ini menunjukkan bahwa *Hybrid Learning* memfokuskan aturan pertama pada kondisi transisi udara bersih "
                    f"dan menyerahkan variasi lonjakan polusi tinggi pada aturan kedua. Pada tabel konsekuen linear (Tabel 4.6), koefisien terbesar secara konsisten berada pada *p_PM2.5* "
                    f"({fmt_id(conseq[0][2], 3)} dan {fmt_id(conseq[1][2], 3)}), menegaskan bahwa konsentrasi PM2.5 hari ini merupakan prediktor terkuat bagi ISPU hari berikutnya."
                )},
                {"h2": "6.2 Analisis Dampak Estimasi Parameter dan Perbandingan Baseline"},
                {"p": (
                    f"Hasil studi ablasi membuktikan dampak krusial dari pemilihan orde model dan estimasi parameter secara sistematis. Pada Data Uji, model pra-estimasi Tahap 0 (FCM + Sugeno Orde-0) "
                    f"menghasilkan RMSE {fmt_id(test_m_s0['rmse'], 3)} (R² = {fmt_id(test_m_s0['r2'], 4)}) karena *output* konstan tidak mampu mengikuti perubahan gradien polutan harian. "
                    f"Penambahan konsekuen linear melalui *Regularized* LSE (Tahap 1) langsung menurunkan RMSE Uji menjadi {fmt_id(test_m_s1['rmse'], 3)}, dan penyempurnaan premis melalui *Hybrid Learning* (Tahap 2) "
                    f"menekan galat lebih jauh hingga mencapai RMSE Uji terbaik sebesar {fmt_id(test_m_s2['rmse'], 3)}, MAE {fmt_id(test_m_s2['mae'], 3)}, dan MAPE {fmt_id(test_m_s2['mape'], 2)}%—mengungguli "
                    f"*baseline* *Persistence* (RMSE {fmt_id(test_m_pers['rmse'], 3)}) maupun Regresi Linear OLS (RMSE {fmt_id(test_m_ols['rmse'], 3)}). Pada Data Validasi, keunggulan *Hybrid Learning* juga terlihat jelas dengan "
                    f"RMSE {fmt_id(val_m_s2['rmse'], 3)} (R² = {fmt_id(val_m_s2['r2'], 4)}) dibandingkan OLS ({fmt_id(val_m_ols['rmse'], 3)}) dan *Persistence* ({fmt_id(val_m_pers['rmse'], 3)})."
                )},
                {"p": (
                    f"Dari sisi klasifikasi kategori Permen LHK pada Data Uji, model mencapai Akurasi Kategori {fmt_id(test_m_s2['category_accuracy']*100, 2)}% dan Akurasi ±1 Kategori {fmt_id(test_m_s2['adjacent_category_accuracy']*100, 2)}%. "
                    f"Penurunan nilai R² pada periode Uji (November–Desember 2021) dibandingkan Data Latih (R² = {fmt_id(tr_m_s2['r2'], 4)}) dan Validasi (R² = {fmt_id(val_m_s2['r2'], 4)}) disebabkan oleh pergeseran distribusi musiman "
                    f"(*seasonal distribution shift*): periode November–Desember merupakan puncak musim penghujan di Jakarta yang memicu pencucian polutan (*rain washout*) secara mendadak sehingga proporsi hari berkategori Baik meningkat drastis."
                )},
                {"h2": "6.3 Kelebihan, Kelemahan, dan Potensi Pengembangan"},
                {"p": "Kelebihan utama model yang dibangun meliputi sifatnya yang transparan dan mudah diinterpretasi melalui aturan IF–THEN serta bobot penyulutan, ketahanan numerik terhadap *outlier* berkat perhitungan pada ruang logaritma (*log-space*), serta kemampuannya mengungguli model linear global maupun *persistence* pada seluruh metrik kesalahan kontinu. Sementara itu, kelemahan model terletak pada ketergantungannya terhadap enam variabel polutan historis hari ke-*t* tanpa melibatkan variabel meteorologi eksternal, sehingga fluktuasi mendadak akibat hujan deras atau perubahan arah angin belum tertangkap sepenuhnya. Untuk pengembangan penelitian selanjutnya, integrasi data meteorologi BMKG (curah hujan, kecepatan angin, suhu, dan kelembaban relatif), penggunaan jendela *lag* multi-hari, serta optimasi metaheuristik seperti PSO-ANFIS (Karki, 2023) berpotensi meningkatkan akurasi lintas musim secara signifikan."},
            ],
        },
        {
            "type": "chapter",
            "heading": "BAB 7. KESIMPULAN DAN SARAN",
            "blocks": [
                {"h2": "7.1 Kesimpulan"},
                {"p": "Berdasarkan perancangan, implementasi, dan pengujian yang telah dilakukan, dapat diambil kesimpulan sebagai berikut."},
                {"ol": [
                    f"Model *fuzzy* Takagi-Sugeno Orde-1 berhasil dibangun dan diimplementasikan untuk memprediksi skor serta kategori mutu udara ISPU(*t*+1) DKI Jakarta menggunakan {fmt_int_id(res['n_paired'])} pasangan pengamatan harian resmi dari 5 stasiun SPKU tahun 2021 tanpa kebocoran data.",
                    f"Metode *clustering* *Fuzzy C-Means* (FCM) secara objektif mempartisi data latih menjadi *c* = {c_opt} *cluster* optimal berdasarkan nilai PC tertinggi ({fmt_id(res['sweep_results'][0]['pc'], 4)}) dan *Validation RMSE* terendah, yang merepresentasikan kondisi Polusi Rendah dan Polusi Tinggi dengan dominansi polutan PM2.5.",
                    f"Estimasi parameter dua-tahap menggunakan *Hybrid Learning* (*Regularized* LSE *λ* = {fmt_id(lam_opt, 0)} dan *Gradient Descent* dengan *early stopping*) menyempurnakan ketajaman fungsi keanggotaan Gaussian serta menyetel koefisien konsekuen linear secara optimal pada *epoch* ke-{int(best_ep_row['epoch'])}.",
                    f"Studi ablasi membuktikan perbaikan performa yang konsisten: pada Data Uji, TSK-1 *Hybrid* mencapai RMSE terbaik ({fmt_id(test_m_s2['rmse'], 3)}), MAE ({fmt_id(test_m_s2['mae'], 3)}), dan MAPE ({fmt_id(test_m_s2['mape'], 2)}%), mengungguli Tahap 0 Sugeno-0 (RMSE {fmt_id(test_m_s0['rmse'], 3)}), *Persistence* (RMSE {fmt_id(test_m_pers['rmse'], 3)}), dan Regresi Linear OLS (RMSE {fmt_id(test_m_ols['rmse'], 3)}), dengan akurasi ±1 kategori sebesar {fmt_id(test_m_s2['adjacent_category_accuracy']*100, 2)}%.",
                ]},
                {"h2": "7.2 Saran"},
                {"ol": [
                    "Menggabungkan dataset ISPU lintas tahun (2021–2025) bersama data meteorologi harian BMKG (curah hujan, kecepatan angin, dan kelembaban udara) agar model mampu menangkap pergeseran antara musim kemarau dan musim penghujan secara lebih presisi.",
                    "Mengeksplorasi jendela masukan multi-hari (*t*−2, *t*−1, *t*) serta metode optimasi metaheuristik seperti *Particle Swarm Optimization* (PSO) sebagai pembanding terhadap *Gradient Descent* pada penyetelan parameter premis.",
                ]},
            ],
        },
        {
            "type": "chapter",
            "heading": "DAFTAR PUSTAKA",
            "blocks": [
                {"bib": [
                    "Bezdek, J. C. (1981). *Pattern recognition with fuzzy objective function algorithms*. Plenum Press.",
                    "Dinas Lingkungan Hidup Provinsi DKI Jakarta. (2021). *Data Indeks Standar Pencemar Udara (ISPU) di Provinsi DKI Jakarta tahun 2021* [Data set]. Portal Satu Data Jakarta. https://satudata.jakarta.go.id/",
                    "Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system. *IEEE Transactions on Systems, Man, and Cybernetics*, *23*(3), 665–685. https://doi.org/10.1109/21.256541",
                    "Karki, A. (2023). Improved ANFIS model based on confinement optimized PSO to predict the air quality of Delhi, India. In *2023 IEEE International Conference on Fuzzy Systems (FUZZ)*. IEEE. https://doi.org/10.1109/FUZZ52849.2023.10309676",
                    "Kementerian Lingkungan Hidup dan Kehutanan Republik Indonesia. (2020). *Peraturan Menteri Lingkungan Hidup dan Kehutanan Nomor P.14/MENLHK/SETJEN/KUM.1/7/2020 tentang Indeks Standar Pencemar Udara*.",
                    "Takagi, T., & Sugeno, M. (1985). Fuzzy identification of systems and its applications to modeling and control. *IEEE Transactions on Systems, Man, and Cybernetics*, *SMC-15*(1), 116–132. https://doi.org/10.1109/TSMC.1985.6313399",
                    "Xie, X. L., & Beni, G. (1991). A validity measure for fuzzy clustering. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *13*(8), 841–847. https://doi.org/10.1109/34.85677",
                    "Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, *8*(3), 338–353. https://doi.org/10.1016/S0019-9958(65)90241-X",
                ]}
            ],
        },
    ]
    return sections


def render_markdown(sections: list[dict]) -> str:
    parts: list[str] = []
    for sec in sections:
        if sec["type"] == "cover":
            for ln in sec["top_lines"]:
                parts.append(f"**{ln}**\n")
            parts.append(f"**{sec['course_line']}**\n")
            rel_logo = Path(sec["logo_path"]).relative_to(ROOT).as_posix()
            parts.append(f"![Logo Universitas Diponegoro](../{rel_logo})\n")
            parts.append("Disusun oleh:\n")
            if len(sec["members"]) == 1:
                nama, nim = sec["members"][0]
                parts.append(f"**Nama**\n{nama}\n\n**NIM**\n{nim}\n")
            else:
                parts.append(md_table(["Nama", "NIM"], [[n, nim] for n, nim in sec["members"]]))
            parts.append("\n" + "\n".join(f"**{ln}**" for ln in sec["institution_lines"]) + "\n\n---\n")
        else:
            parts.append(f"\n# {sec['heading']}\n")
            for blk in sec["blocks"]:
                if "h2" in blk:
                    parts.append(f"\n## {blk['h2']}\n")
                elif "p" in blk:
                    parts.append(blk["p"] + "\n")
                elif "eq" in blk:
                    parts.append(f"\n> {blk['eq']['text']}\n")
                elif "code" in blk:
                    parts.append(f"\n```python\n{blk['code']}\n```\n")
                elif "ol" in blk:
                    parts.append("\n" + "\n".join(f"{idx+1}. {item}" for idx, item in enumerate(blk["ol"])) + "\n")
                elif "bib" in blk:
                    parts.append("\n" + "\n\n".join(blk["bib"]) + "\n")
                elif "table" in blk:
                    t = blk["table"]
                    parts.append(f"\n**{t['label']}** {t['caption']}\n\n" + md_table(t["headers"], t["rows"]) + "\n")
                elif "fig" in blk:
                    f = blk["fig"]
                    rel = Path(f["path"]).relative_to(ROOT).as_posix()
                    parts.append(f"\n![{f['label']} {f['caption']}](../{rel})\n**{f['label']}** {f['caption']}\n")
    return "\n".join(parts)


def render_html(sections: list[dict]) -> str:
    body: list[str] = []
    for sec in sections:
        if sec["type"] == "cover":
            logo_uri = Path(sec["logo_path"]).resolve().as_uri()
            top_html = "".join(f'<p class="cover-top">{inline_to_html(ln)}</p>' for ln in sec["top_lines"])
            if len(sec["members"]) == 1:
                nama, nim = sec["members"][0]
                author_block = (
                    f'<p class="cover-by" style="margin-top:8px; margin-bottom:2px;"><strong>Nama</strong></p>'
                    f'<p class="cover-by" style="margin-top:0; margin-bottom:10px;">{html.escape(nama)}</p>'
                    f'<p class="cover-by" style="margin-top:0; margin-bottom:2px;"><strong>NIM</strong></p>'
                    f'<p class="cover-by" style="margin-top:0;">{html.escape(nim)}</p>'
                )
            else:
                member_rows = "".join(
                    f"<tr><td>{html.escape(n)}</td><td class='nim-col'>{html.escape(nim)}</td></tr>"
                    for n, nim in sec["members"]
                )
                author_block = (
                    f'<table class="cover-table">'
                    f'<thead><tr><th>Nama</th><th class="nim-col">NIM</th></tr></thead>'
                    f'<tbody>{member_rows}</tbody>'
                    f'</table>'
                )
            inst_html = "".join(f'<p class="cover-inst">{html.escape(ln)}</p>' for ln in sec["institution_lines"])
            body.append(
                f"""
                <section class="cover-page">
                    <div class="cover-top-wrap">
                        {top_html}
                        <p class="cover-course">{html.escape(sec['course_line'])}</p>
                    </div>
                    <div class="cover-logo-wrap">
                        <img src="{logo_uri}" alt="Logo Universitas Diponegoro" class="cover-logo"/>
                    </div>
                    <div class="cover-author-wrap">
                        <p class="cover-by">Disusun oleh:</p>
                        {author_block}
                    </div>
                    <div class="cover-inst-wrap">
                        {inst_html}
                    </div>
                </section>
                """
            )
        else:
            is_bib = sec["heading"] == "DAFTAR PUSTAKA"
            body.append(f'<section class="chapter"><h1 class="chapter-title">{html.escape(sec["heading"])}</h1>')
            for blk in sec["blocks"]:
                if "h2" in blk:
                    body.append(f"<h2>{html.escape(blk['h2'])}</h2>")
                elif "p" in blk:
                    body.append(f'<p class="body-p">{inline_to_html(blk["p"])}</p>')
                elif "eq" in blk:
                    body.append(f'<div class="equation">{blk["eq"]["html"]}</div>')
                elif "code" in blk:
                    body.append(f'<pre class="codeblock">{html.escape(blk["code"])}</pre>')
                elif "ol" in blk:
                    items = "".join(f"<li>{inline_to_html(i)}</li>" for i in blk["ol"])
                    body.append(f"<ol>{items}</ol>")
                elif "bib" in blk:
                    items = "".join(f'<p class="bib-item">{inline_to_html(i)}</p>' for i in blk["bib"])
                    body.append(items)
                elif "table" in blk:
                    t = blk["table"]
                    ths = "".join(f"<th>{inline_to_html(h)}</th>" for h in t["headers"])
                    trs = "".join(
                        "<tr>" + "".join(f"<td>{inline_to_html(str(c))}</td>" for c in r) + "</tr>"
                        for r in t["rows"]
                    )
                    body.append(
                        f'<div class="table-wrap"><div class="caption"><strong>{html.escape(t["label"])}</strong> {inline_to_html(t["caption"])}</div>'
                        f"<table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table></div>"
                    )
                elif "fig" in blk:
                    f = blk["fig"]
                    uri = Path(f["path"]).resolve().as_uri()
                    body.append(
                        f'<figure><img src="{uri}" alt="{html.escape(f["label"])}"/>'
                        f'<figcaption><strong>{html.escape(f["label"])}</strong> {inline_to_html(f["caption"])}</figcaption></figure>'
                    )
            body.append("</section>")

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8"/>
<title>Proyek Besar Individu — Menza Isaiah Tampubolon</title>
<style>
  @page {{ size: A4; margin: 25.4mm 25.4mm 25.4mm 25.4mm; }}
  * {{ color: #000000 !important; }}
  body {{ font-family: "Times New Roman", Times, serif; color: #000000; line-height: 1.25; font-size: 12pt; margin: 0; padding: 0; background: #FFFFFF; }}
  .cover-page {{ page-break-after: always; text-align: center; display: flex; flex-direction: column; justify-content: space-between; min-height: 242mm; }}
  .cover-top {{ font-weight: bold; font-size: 12pt; margin: 2px 0; text-align: center; }}
  .cover-course {{ font-weight: bold; font-size: 12pt; margin-top: 16px; text-align: center; }}
  .cover-logo-wrap {{ margin: 16px 0; }}
  .cover-logo {{ width: 65mm; height: auto; }}
  .cover-by {{ text-align: center; margin: 6px 0 8px 0; font-size: 12pt; }}
  .cover-table {{ margin: 0 auto; width: 118mm; border-collapse: collapse; font-size: 12pt; }}
  .cover-table th, .cover-table td {{ border: none; padding: 2px 6px; text-align: left; background: #FFFFFF !important; }}
  .cover-table th {{ font-weight: bold; text-align: center; }}
  .cover-table td.nim-col, .cover-table th.nim-col {{ text-align: right; width: 42mm; }}
  .cover-inst-wrap {{ margin-top: 22px; }}
  .cover-inst {{ font-weight: bold; font-size: 12pt; margin: 2px 0; text-align: center; }}

  .chapter {{ page-break-before: always; }}
  .chapter-title {{ font-size: 12pt; font-weight: bold; text-align: center; margin: 0 0 14px 0; page-break-after: avoid; }}
  h2 {{ font-size: 12pt; font-weight: bold; text-align: left; margin: 12px 0 6px 0; page-break-after: avoid; }}
  p.body-p {{ text-align: justify; text-indent: 0.5in; margin: 0 0 6px 0; line-height: 1.25; }}
  ol {{ margin: 4px 0 8px 0.5in; padding-left: 18px; }}
  li {{ margin-bottom: 4px; text-align: justify; line-height: 1.25; }}
  .equation {{ text-align: center; font-family: "Times New Roman", Times, serif; font-size: 12pt; margin: 10px 0; line-height: 1.35; }}
  pre.codeblock {{ background: #FFFFFF; color: #000000; border: 1px solid #000000; padding: 8px 10px; font-family: "Consolas", "Courier New", monospace; font-size: 9pt; white-space: pre-wrap; margin: 8px 0; }}
  .table-wrap {{ margin: 10px 0 12px 0; page-break-inside: avoid; }}
  .caption {{ text-align: center; font-size: 11pt; margin-bottom: 5px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 9.5pt; }}
  th, td {{ border: 1px solid #000000; padding: 4px 6px; text-align: left; }}
  thead th {{ background: #E8E8E8; font-weight: bold; text-align: center; }}
  tbody td {{ background: #FFFFFF; }}
  figure {{ text-align: center; margin: 10px 0 14px 0; page-break-inside: avoid; }}
  figure img {{ max-width: 96%; max-height: 108mm; }}
  figcaption {{ font-size: 11pt; text-align: center; margin-top: 5px; }}
  .bib-item {{ padding-left: 0.5in; text-indent: -0.5in; text-align: left; margin: 0 0 8px 0; font-size: 12pt; line-height: 1.25; }}
</style>
</head>
<body>
{''.join(body)}
</body>
</html>
"""


def _set_cell_shading(cell, hex_color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def _clear_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "none")
        borders.append(el)
    tbl_pr.append(borders)


def _set_thin_black_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tbl_pr.append(borders)


def render_docx(sections: list[dict], out_path: Path) -> None:
    doc = docx.Document()
    for sec_page in doc.sections:
        sec_page.top_margin = Inches(1.0)
        sec_page.bottom_margin = Inches(1.0)
        sec_page.left_margin = Inches(1.0)
        sec_page.right_margin = Inches(1.0)

    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = BLACK

    for sec in sections:
        if sec["type"] == "cover":
            for ln in sec["top_lines"]:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(4)
                _add_formatted_runs(p, ln, font_size=Pt(12), base_bold=True)

            p_course = doc.add_paragraph()
            p_course.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_course.paragraph_format.space_before = Pt(12)
            p_course.paragraph_format.space_after = Pt(18)
            _add_formatted_runs(p_course, sec["course_line"], font_size=Pt(12), base_bold=True)

            p_logo = doc.add_paragraph()
            p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_logo.paragraph_format.space_before = Pt(8)
            p_logo.paragraph_format.space_after = Pt(24)
            p_logo.add_run().add_picture(sec["logo_path"], width=Inches(2.55))

            p_by = doc.add_paragraph()
            p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_by.paragraph_format.space_after = Pt(10)
            _add_formatted_runs(p_by, "Disusun oleh:", font_size=Pt(12))

            if len(sec["members"]) == 1:
                nama, nim = sec["members"][0]
                for lbl, val, sp_after in [("Nama", nama, Pt(8)), ("NIM", nim, Pt(4))]:
                    p_lbl = doc.add_paragraph()
                    p_lbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_lbl.paragraph_format.space_after = Pt(2)
                    _add_formatted_runs(p_lbl, lbl, font_size=Pt(12), base_bold=True)
                    p_val = doc.add_paragraph()
                    p_val.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_val.paragraph_format.space_after = sp_after
                    _add_formatted_runs(p_val, val, font_size=Pt(12))
            else:
                tbl = doc.add_table(rows=len(sec["members"]) + 1, cols=2)
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                _clear_table_borders(tbl)
                hdr_left, hdr_right = tbl.rows[0].cells[0], tbl.rows[0].cells[1]
                hdr_left.width, hdr_right.width = Inches(3.2), Inches(1.7)
                p_hl = hdr_left.paragraphs[0]
                p_hl.alignment = WD_ALIGN_PARAGRAPH.CENTER
                _add_formatted_runs(p_hl, "Nama", font_size=Pt(12), base_bold=True)
                p_hr = hdr_right.paragraphs[0]
                p_hr.alignment = WD_ALIGN_PARAGRAPH.CENTER
                _add_formatted_runs(p_hr, "NIM", font_size=Pt(12), base_bold=True)

                for idx, (nama, nim) in enumerate(sec["members"], start=1):
                    c0, c1 = tbl.rows[idx].cells[0], tbl.rows[idx].cells[1]
                    c0.width, c1.width = Inches(3.2), Inches(1.7)
                    p0 = c0.paragraphs[0]
                    p0.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    _add_formatted_runs(p0, nama, font_size=Pt(12))
                    p1 = c1.paragraphs[0]
                    p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                    _add_formatted_runs(p1, nim, font_size=Pt(12))

            p_sp = doc.add_paragraph()
            p_sp.paragraph_format.space_before = Pt(36)
            for ln in sec["institution_lines"]:
                p_in = doc.add_paragraph()
                p_in.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_in.paragraph_format.space_after = Pt(2)
                _add_formatted_runs(p_in, ln, font_size=Pt(12), base_bold=True)

            doc.add_page_break()
            continue

        # Chapter Heading (centered bold black 12pt)
        if sec["heading"] != "BAB 1. PENDAHULUAN":
            doc.add_page_break()
        h1 = doc.add_heading(level=1)
        h1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        h1.paragraph_format.space_after = Pt(10)
        _add_formatted_runs(h1, sec["heading"], font_size=Pt(12), base_bold=True)

        for blk in sec["blocks"]:
            if "h2" in blk:
                h2 = doc.add_heading(level=2)
                h2.alignment = WD_ALIGN_PARAGRAPH.LEFT
                h2.paragraph_format.space_before = Pt(8)
                h2.paragraph_format.space_after = Pt(4)
                _add_formatted_runs(h2, blk["h2"], font_size=Pt(12), base_bold=True)
            elif "p" in blk:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.first_line_indent = Inches(0.5)
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(4)
                _add_formatted_runs(p, blk["p"], font_size=Pt(12))
            elif "eq" in blk:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(6)
                r = p.add_run(blk["eq"]["text"])
                r.font.name = "Times New Roman"
                r.font.size = Pt(12)
                r.italic = True
                r.font.color.rgb = BLACK
            elif "code" in blk:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(6)
                r = p.add_run(blk["code"])
                r.font.name = "Consolas"
                r.font.size = Pt(9.5)
                r.font.color.rgb = BLACK
            elif "ol" in blk:
                for idx, item in enumerate(blk["ol"], start=1):
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    p.paragraph_format.left_indent = Inches(0.5)
                    p.paragraph_format.first_line_indent = Inches(-0.25)
                    p.paragraph_format.line_spacing = 1.15
                    p.paragraph_format.space_after = Pt(3)
                    _add_formatted_runs(p, f"{idx}. {item}", font_size=Pt(12))
            elif "bib" in blk:
                for item in blk["bib"]:
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    p.paragraph_format.left_indent = Inches(0.5)
                    p.paragraph_format.first_line_indent = Inches(-0.5)
                    p.paragraph_format.line_spacing = 1.15
                    p.paragraph_format.space_after = Pt(6)
                    _add_formatted_runs(p, item, font_size=Pt(12))
            elif "table" in blk:
                t = blk["table"]
                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap.paragraph_format.space_before = Pt(6)
                cap.paragraph_format.space_after = Pt(4)
                _add_formatted_runs(cap, f"**{t['label']}** {t['caption']}", font_size=Pt(11))

                tbl = doc.add_table(rows=len(t["rows"]) + 1, cols=len(t["headers"]))
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                _set_thin_black_borders(tbl)
                for j, h in enumerate(t["headers"]):
                    cell = tbl.rows[0].cells[j]
                    _set_cell_shading(cell, "E8E8E8")
                    p_cell = cell.paragraphs[0]
                    p_cell.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    _add_formatted_runs(p_cell, h, font_size=Pt(9.5), base_bold=True)
                for i, row in enumerate(t["rows"]):
                    for j, val in enumerate(row):
                        cell = tbl.rows[i + 1].cells[j]
                        p_cell = cell.paragraphs[0]
                        p_cell.alignment = WD_ALIGN_PARAGRAPH.LEFT
                        _add_formatted_runs(p_cell, str(val), font_size=Pt(9.5))
            elif "fig" in blk:
                f = blk["fig"]
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(6)
                p_img.add_run().add_picture(str(f["path"]), width=Inches(6.0))
                p_cap = doc.add_paragraph()
                p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_cap.paragraph_format.space_after = Pt(8)
                _add_formatted_runs(p_cap, f"**{f['label']}** {f['caption']}", font_size=Pt(11))

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
