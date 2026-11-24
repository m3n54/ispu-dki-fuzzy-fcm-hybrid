# Implementation Plan: Studi Kasus ISPU DKI Jakarta — Takagi-Sugeno Orde-1 + Fuzzy C-Means + Hybrid Learning

> **Goal:** Membangun sistem prediksi dini ISPU hari berikutnya (`ISPU(t+1)`) DKI Jakarta berbasis Takagi-Sugeno Orde-1, Fuzzy C-Means, dan Hybrid Learning (LSE + Gradient Descent), lengkap dengan modul Python teruji, notebook Google Colab, GUI interaktif Streamlit, dan laporan 9 bab (PDF & Google Docs).

## Global Constraints
- Penulis: Menza Isaiah Tampubolon (NIM: 24060124140138)
- Dataset: Data publik nyata ISPU DKI Jakarta (6 polutan: `pm10`, `pm25`, `so2`, `co`, `o3`, `no2`, `tanggal`, `stasiun`, `max`).
- Zero data leakage: pasangan `t -> t+1` per stasiun dengan selisih tepat 1 hari, split kronologis 70/15/15, normalisasi Z-score hanya pada data Train.
- 7 Referensi IEEE wajib terpasang konsisten.

## Tasks Checklist
- [x] **Task 1:** Inisialisasi repositori Git, struktur folder, spec desain, dan implementation plan.
- [ ] **Task 2:** Akuisisi & dokumentasi dataset publik ISPU DKI Jakarta (`data/raw/`, `data/README.md`).
- [ ] **Task 3:** Implementasi & pengujian TDD `src/data_loader.py`.
- [ ] **Task 4:** Implementasi & pengujian TDD `src/fcm.py`.
- [ ] **Task 5:** Implementasi & pengujian TDD `src/sugeno_hybrid.py`.
- [ ] **Task 6:** Implementasi & pengujian `src/evaluation.py` dan `src/visualization.py`.
- [ ] **Task 7:** Eksekusi eksperimen end-to-end `scripts/run_experiment.py` dan inspeksi hasil + grafik.
- [ ] **Task 8:** Implementasi & verifikasi GUI Streamlit `app.py` + tangkapan layar dokumentasi.
- [ ] **Task 9:** Pembuatan, eksekusi lokal, dan unggah notebook ke Google Colab.
- [ ] **Task 10:** Penyusunan laporan 9 bab (Markdown, HTML, PDF, DOCX) & publikasi ke Google Docs.
- [ ] **Task 11:** Verifikasi akhir menyeluruh.
