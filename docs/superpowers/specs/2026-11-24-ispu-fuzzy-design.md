# Desain Arsitektur Sistem Prediksi Dini ISPU DKI Jakarta
**Takagi-Sugeno Orde-1 + Fuzzy C-Means + Hybrid Learning (LSE–Gradient Descent)**

- **Penulis:** Menza Isaiah Tampubolon (`24060124140138`)
- **Program Studi:** Informatika, Fakultas Sains dan Matematika, Universitas Diponegoro
- **Mata Kuliah:** Komputasi Lunak
- **Tanggal:** 24 November 2026

---

## 1. Perumusan Masalah & Desain Target
- **Konteks Nyata:** Pemantauan dan peringatan dini (*early warning*) kualitas udara DKI Jakarta menggunakan Indeks Standar Pencemar Udara (ISPU) sesuai Permen LHK No. P.14/MENLHK/SETJEN/KUM.1/7/2020.
- **Formulasi Non-Trivial (`t -> t+1`):**
  - Input $\mathbf{x}_t = [\text{PM}_{10}, \text{PM}_{2.5}, \text{SO}_2, \text{CO}, \text{O}_3, \text{NO}_2]_t \in \mathbb{R}^6$ pada hari ke-$t$ dan stasiun $s$.
  - Target $y_{t+1} = \text{ISPU}_{\max}(s, t+1) \in \mathbb{R}$ (nilai indeks tertinggi keesokan hari pada stasiun $s$ yang sama, hanya jika selisih tanggal tepat 1 hari).
  - Pemetaan kategori keputusan akhir Permen LHK:
    - `0–50`: **Baik**
    - `51–100`: **Sedang**
    - `101–200`: **Tidak Sehat**
    - `201–300`: **Sangat Tidak Sehat**
    - `>300`: **Berbahaya**
- **Pencegahan Kebocoran Data (*Zero Data Leakage*):**
  - Pemasangan target dilakukan per `stasiun` dengan syarat `(tanggal_next - tanggal).days == 1`.
  - Pembagian data kronologis: Train 70% / Validation 15% / Test 15% (`max(train.tanggal) < min(val.tanggal) < min(test.tanggal)`).
  - Normalisasi Z-score hanya di-*fit* pada `X_train`.

## 2. Arsitektur Matematis Inti
### 2.1 Clustering Fuzzy (Fuzzy C-Means / FCM)
Meminimalkan fungsi objektif:
$$J_m(U, V) = \sum_{k=1}^{N} \sum_{i=1}^{c} u_{ik}^m \|\mathbf{x}_k - \mathbf{v}_i\|_2^2, \quad m = 2.0$$
Pembaruan derajat keanggotaan dan pusat cluster:
$$u_{ik} = \frac{1}{\sum_{r=1}^{c} \left(\frac{\|\mathbf{x}_k - \mathbf{v}_i\|_2}{\|\mathbf{x}_k - \mathbf{v}_r\|_2}\right)^{\frac{2}{m-1}}}, \qquad \mathbf{v}_i = \frac{\sum_{k=1}^{N} u_{ik}^m \mathbf{x}_k}{\sum_{k=1}^{N} u_{ik}^m}$$
Kriteria konvergensi: $\|U^{(t)} - U^{(t-1)}\|_\infty < 10^{-5}$.
Validitas jumlah cluster $c \in \{2,3,4,5,6\}$ dievaluasi dengan **Partition Coefficient (PC)**, **Partition Entropy (PE)**, **Xie-Beni Index (XB)**, dan **Validation RMSE** Stage-1.

### 2.2 Model Fuzzy Takagi-Sugeno Orde-1 & Fungsi Keanggotaan Gaussian
Untuk setiap aturan $i \in \{1,\dots,c\}$ dan fitur $j \in \{1,\dots,6\}$:
- **Premis Gaussian:**
  $$\mu_{ij}(x_j) = \exp\!\left(-\frac{(x_j - v_{ij})^2}{2\sigma_{ij}^2}\right)$$
  dengan inisialisasi $v_{ij}$ dari pusat cluster FCM dan $\sigma_{ij}$ dari simpangan baku terbobot fuzzy:
  $$\sigma_{ij} = \sqrt{\frac{\sum_{k=1}^{N} u_{ik}^m (x_{kj} - v_{ij})^2}{\sum_{k=1}^{N} u_{ik}^m}}$$
- **Firing Strength & Normalisasi (log-space stability):**
  $$w_i(\mathbf{x}) = \prod_{j=1}^{6} \mu_{ij}(x_j), \qquad \bar{w}_i(\mathbf{x}) = \frac{w_i(\mathbf{x})}{\sum_{r=1}^{c} w_r(\mathbf{x})}$$
- **Konsekuen Linear & Defuzzifikasi:**
  $$f_i(\mathbf{x}) = p_{i0} + \sum_{j=1}^{6} p_{ij} x_j, \qquad \hat{y}(\mathbf{x}) = \sum_{i=1}^{c} \bar{w}_i(\mathbf{x}) f_i(\mathbf{x})$$

### 2.3 Estimasi Parameter: Hybrid Learning (Regularized LSE + Gradient Descent)
- **Tahap 0 (FCM Orde-0):** $p_{ij}=0$ untuk $j\ge 1$, $p_{i0}$ = rata-rata terbobot target cluster $i$.
- **Tahap 1 (Regularized LSE / Ridge):** Menyelesaikan $\hat{\boldsymbol{\theta}} = (\mathbf{A}^\top\mathbf{A} + \lambda \mathbf{I})^{-1}\mathbf{A}^\top\mathbf{y}$ secara analitik tertutup.
- **Tahap 2 (Hybrid Learning):** Forward pass memperbarui $\boldsymbol{\theta}$ dengan LSE eksak; backward pass memperbarui $(v_{ij}, \log \sigma_{ij})$ dengan Gradient Descent adaptif, gradient clipping, dan early stopping berdasarkan Validation RMSE.

## 3. Antarmuka Modul & Verifikasi
- `src/data_loader.py`: akuisisi, pembersihan, pembuatan pasangan $t \to t+1$, split kronologis, dan `StandardScalerNumpy`.
- `src/fcm.py`: `FuzzyCMeans` + `evaluate_cluster_counts`.
- `src/sugeno_hybrid.py`: `TakagiSugenoHybrid` (Stage 0/1/2, `explain_single`, JSON serialization).
- `src/evaluation.py`: pemetaan Permen LHK, metrik regresi & kategori, baseline Persistence & OLS.
- `src/visualization.py`: generator 7 grafik 300 DPI.
- `app.py`: GUI interaktif Streamlit mencakup 5 fitur wajib (input pengguna, hasil clustering, fungsi keanggotaan, hasil inferensi fuzzy, keputusan akhir).
- `tests/`: unit tests menggunakan `unittest` termasuk finite-difference gradient check dan uji invariansi FCM.
