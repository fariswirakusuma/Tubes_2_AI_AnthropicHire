# Cara Tim Lanjut — T2 IF3170

Dokumen ini untuk Awan, Ahmad, Jojo, Faris. Tujuan: kalian bisa mulai modeling
hari ini tanpa menunggu notebook utama selesai.

## Status data

Pipeline data sudah selesai dan reproducible. Hasil siap pakai ada di
`output/prepared/`:

| File | Isi |
|---|---|
| `train.npz` | `X` (21898 × 40), `y` |
| `val.npz` | `X` (5475 × 40), `y` |
| `test.npz` | `X` (6844 × 40), `ids` |
| `feature_names.json` | urutan 40 fitur |
| `united_setup.pkl` | objek preprocessing yang sudah fit |

Sudah termasuk cleaning, imputasi (median/modus + flag missing), one-hot,
standardisasi, dan seleksi 40 fitur terbaik. **Tidak ada NaN**, semua `float64`.

## Setup lokal

```bash
git clone git@github.com:fariswirakusuma/Tubes_2_AI_AnthropicHire.git
cd Tubes_2_AI_AnthropicHire
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

Cek data:

```bash
.venv/bin/python -c "import numpy as np; d=np.load('output/prepared/train.npz'); print(d['X'].shape, d['y'].mean())"
```

Kalau kerja di Colab: upload folder `output/prepared/` dan `src/`, lalu mount Drive.

## Aturan data

1. `X` sudah numerik dan siap dilatih. **Jangan scaling/encoding lagi di dalam model.**
2. Urutan 40 fitur tetap. Jangan reorder kolom.
3. `fit` hanya di `train.npz`. `val.npz` untuk evaluasi. `test.npz` hanya untuk prediksi akhir.
4. Macro F1 dihitung di val, bukan di test.

## Kontrak model

Buat `src/models/<nama>.py`:

```python
class KNN:
    def __init__(self, ...): ...
    def fit(self, X, y): return self
    def predict(self, X): ...
    def predict_proba(self, X):  # shape (n, 2), baris jumlah 1
        ...
```

Cek model dalam 2 detik:

```bash
.venv/bin/python scripts/check_model.py src.models.knn:KNN --name knn --save
```

Output: waktu fit, waktu predict, macro F1, classification report, cek
`predict_proba`, dan `output/knn_v3.pkl` kalau pakai `--save`.

Mulai dari `src/models/_template.py`.

## Yang harus diserahkan tiap orang

- [ ] `src/models/<model>.py` — versi from scratch (NumPy saja)
- [ ] fungsi pembanding memakai scikit-learn
- [ ] macro F1 di validation
- [ ] `output/<model>_v3.pkl`
- [ ] waktu fit dan predict (untuk folder `artefacts/`)
- [ ] satu cell notebook + penjelasan implementasi untuk laporan
- [ ] untuk bonus: visualisasi sesuai bagian Bonus di spesifikasi

## Deadline internal

| Tanggal | Target |
|---|---|
| 18 Nov | skeleton kelas + model bisa `fit`/`predict` tanpa error |
| 19 Nov | versi pertama selesai + skor val keluar |
| 20 Nov | integrasi ke notebook utama |
| 21 Nov | tuning + submission final |
| 22 Nov | laporan + reproducibility + release |
| 23 Nov | buffer, submit sebelum 23.59 WIB |

## Jangan

- Memakai scikit-learn di implementasi from scratch.
- `fit` di data validation atau test.
- Mengubah urutan fitur atau menambah fitur baru tanpa koordinasi.
- Memakai ensemble untuk submission bonus (aturan kompetisi: hanya model from scratch tanpa ensemble).

## Pertanyaan

Tanya di grup, jangan ke asisten secara personal (aturan spec no. 3).
