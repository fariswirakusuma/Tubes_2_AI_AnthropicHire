# EDA — Tugas Besar 2 IF3170 (T2-IF3170-2026-2027)

Owner: Ega — Anggota 1 (Data dan Integrasi)
Sumber data: `data/raw/{train,test,sample_submission}.csv`

## Ukuran data

| File | Baris | Kolom |
|---|---|---|
| train.csv | 27.373 | 19 (18 fitur + `is_canceled`) |
| test.csv | 6.844 | 18 (tanpa target) |
| sample_submission.csv | 6.844 | `id`, `is_canceled` |

## Target

`is_canceled` — biner `0/1`, **32,77% positif** (8.969 dari 27.373). Metrik kompetisi: **macro F1**, jadi kelas minoritas tetap penting.

## Missing values

| Kolom | Train | Test |
|---|---|---|
| num_children | 11,09% | 10,91% |
| special_requests | 9,13% | 8,66% |
| nightly_rate_idr | 7,45% | 7,89% |
| meal_package | 6,43% | 6,46% |
| days_before_arrival | 4,10% | 4,24% |
| needs_parking | 3,95% | 3,61% |
| room_category | 3,66% | 3,54% |
| num_adults | 2,96% | 2,63% |

Missingness **informatif**, bukan random murni: `special_requests` yang kosong punya cancellation rate 40,8% vs 43,2% untuk nilai 0. Jadi selain imputasi, tambahkan indikator missing.

## Sinyal terkuat

| Fitur | Arah | Catatan |
|---|---|---|
| `days_before_arrival` | +0,44 korelasi | bin 0 hari = 8,1% cancel; >365 hari = 96,3% cancel |
| `special_requests` | −0,26 | 0 request = 43,2% cancel; ≥3 request = **0% cancel** |
| `arrival_year` | +0,18 | 2023 vs 2024 berbeda |
| `nightly_rate_idr` | +0,13 | — |
| `is_returning_guest` | −0,11 | returning = 1,5% cancel |
| `needs_parking` | −0,09 | butuh parkir = 10,1% cancel |
| `booking_channel` | — | Complimentary = 0% cancel (301 baris); Corporate = 11,5% |
| `meal_package` | — | Half Board = 45,9%; Full Board cuma 4 baris |
| `room_category` | — | Type A = 42,6%; Type C/F = ~25% |

## Masalah kualitas data

- 66 baris duplikat fitur (0,24%), 2 di antaranya label konflik → drop duplikat saat training.
- `num_adults == 0`: 106 baris (105 di antaranya punya anak) → kemungkinan input error.
- `weekend_nights + weekday_nights == 0`: 64 baris.
- `nightly_rate_idr == 0`: 413 baris; maksimum 9,17 juta (8 baris > 5 juta).
- `days_before_arrival > 365`: 187 baris.
- `prior_completed_bookings > 20`: 40 baris, maksimum 58.

## Train vs test

Distribusi identik (tahun, bulan, mean fitur, missing rate) → **split acak, bukan temporal**. Stratified random split aman dan tidak bocor.

## Implikasi untuk pipeline

1. Split dulu (stratified 80/20, seed 42), baru fit preprocessing di train.
2. Imputasi: median untuk numerik, modus untuk kategorikal + flag `*_is_missing` untuk 8 kolom.
3. Feature engineering: `nightly_rate_per_guest`, `prior_cancel_ratio`, `has_prior_history`, `arrival_quarter`, `is_weekend_heavy`, `days_before_arrival_log`.
4. Encoding: one-hot untuk 3 kategorikal low-card (`meal_package`, `room_category`, `booking_channel`); `needs_parking` sudah biner.
5. Scaling: StandardScaler untuk model berbasis jarak/linear (KNN, LR, SVM); tree tidak butuh.
6. Seleksi fitur: variance threshold + mutual information.
7. PCA: simpan sebagai opsi, fit di train setelah scaling; tree biasanya tidak diuntungkan.
8. Class balance: laporkan, jangan resample dulu; optimalkan threshold di validasi untuk macro F1.
