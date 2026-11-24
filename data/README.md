# Sumber & Dokumentasi Dataset ISPU DKI Jakarta

## 1. Informasi Umum
- **Nama Dataset:** Data Indeks Standar Pencemar Udara (ISPU) di Stasiun Pemantau Kualitas Udara (SPKU) Provinsi DKI Jakarta Tahun 2021
- **Instansi Sumber Resmi:** Dinas Lingkungan Hidup (DLH) Provinsi DKI Jakarta — Portal Satu Data Jakarta (`satudata.jakarta.go.id`)
- **Cermin Repositori Publik (Mirror CSV):** `https://github.com/Rendika7/NLP_ISPU_DKI_JAKARTA_2021` (direktori `UAS - TUBES/Dataset/indeks-standar-pencemar-udara-di-spku-bulan-*-tahun-2021.csv`)
- **Periode Waktu:** 1 Januari 2021 – 31 Desember 2021 (12 berkas bulanan, 1.825 baris observasi harian)
- **Stasiun Pemantau (5 SPKU):**
  1. `DKI1 (Bunderan HI)`
  2. `DKI2 (Kelapa Gading)`
  3. `DKI3 (Jagakarsa)`
  4. `DKI4 (Lubang Buaya)`
  5. `DKI5 (Kebon Jeruk) Jakarta Barat`

## 2. Daftar Variabel
| Kolom | Tipe | Keterangan |
| :--- | :--- | :--- |
| `tanggal` | Date (`YYYY-MM-DD`) | Tanggal pengamatan harian |
| `stasiun` | String | Nama lokasi SPKU DKI Jakarta |
| `pm10` | Numerik | Sub-indeks Partikulat PM10 |
| `pm25` | Numerik | Sub-indeks Partikulat Halus PM2.5 (mulai penuh pertengahan Januari 2021) |
| `so2` | Numerik | Sub-indeks Sulfur Dioksida (SO2) |
| `co` | Numerik | Sub-indeks Karbon Monoksida (CO) |
| `o3` | Numerik | Sub-indeks Ozon (O3) |
| `no2` | Numerik | Sub-indeks Nitrogen Dioksida (NO2) |
| `max` | Numerik | Nilai ISPU maksimum harian (nilai indeks resmi hari tersebut) |
| `critical` | String | Parameter polutan kritis (penyumbang sub-indeks tertinggi) |
| `categori` | String | Kategori mutu udara sesuai Permen LHK No. P.14/MENLHK/SETJEN/KUM.1/7/2020 |

## 3. Catatan Prapemrosesan Data Mentah
1. Pada berkas resmi bulan **Juni 2021** dan **Juli 2021**, baris header asli dari Jakarta Open Data mengalami pergeseran label (`pm10`..`location`) meskipun urutan ke-11 kolom nilainya tetap identik (`tanggal, stasiun, pm10, pm25, so2, co, o3, no2, max, critical, categori`). Modul `src/data_loader.py` mendeteksi pergeseran header ini secara otomatis dan memulihkan pemetaan kolom yang benar.
2. Nilai kosong atau penanda tidak terbaca sensor (`---`, `-`, `NA`, `TIDAK ADA DATA`) dikonversi menjadi `NaN` dan baris dengan fitur/target tidak lengkap dibersihkan sebelum pembentukan pasangan berurutan $(t \to t+1)$ per stasiun.
