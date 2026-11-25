# Prediksi Dini Indeks Standar Pencemar Udara (ISPU) DKI Jakarta Menggunakan Takagi-Sugeno Orde-1, Fuzzy C-Means, dan Hybrid Learning

Proyek Besar Individu Mata Kuliah **Komputasi Lunak** — Program Studi Informatika, Fakultas Sains dan Matematika, Universitas Diponegoro.

- **Nama:** Menza Isaiah Tampubolon
- **NIM:** 24060124140138

## Ringkasan Proyek

Sistem ini memprediksi nilai maksimum ISPU hari berikutnya (`ISPU(t+1)`) pada data harian Stasiun Pemantau Kualitas Udara (SPKU) DLH DKI Jakarta tahun 2021 (1.444 pasangan sekuensial bebas kebocoran data: 1.065 Latih / 175 Validasi / 204 Uji) dengan memadukan:

1. **Fuzzy C-Means (FCM)** pada 6 fitur polutan terstandarisasi (`PM10`, `PM2.5`, `SO2`, `CO`, `O3`, `NO2`) untuk inisialisasi pusat cluster (`c* = 2`) dan lebar Gaussian (*sigma*) berbasis data.
2. **Sistem Inferensi Fuzzy Takagi-Sugeno Orde-1** sebagai model prediksi regresi kontinu.
3. **Hybrid Learning (Regularized LSE + Gradient Descent)** dengan parameter optimal `ridge_lambda* = 50.0`.

## Hasil Evaluasi Utama (Data Uji)

| Model | RMSE | MAE | R² | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| Takagi-Sugeno Orde-1 + FCM + Hybrid Learning | **18,966** | **14,085** | **0,0478** | **24,76%** |

## Struktur Direktori

- `app.py` — Antarmuka pengguna web interaktif berbasis **Streamlit**.
- `src/` — Modul inti (`data_loader.py`, `fcm.py`, `sugeno_hybrid.py`, `evaluation.py`, `visualization.py`).
- `scripts/` — Skrip eksperimen, pembuatan notebook Colab, dan generator laporan UNDIP.
- `notebooks/ISPU_Fuzzy_FCM_Hybrid.ipynb` — Notebook eksperimen Google Colab.
- `tests/` — Unit test (*11/11 passing*).
- `artifacts/` — Hasil eksperimen, artefak model terlatih, grafik, tangkapan layar GUI, dan tautan publik.

## Cara Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Untuk menjalankan pengujian unit:

```bash
python -B -m unittest discover -s tests -v
```
