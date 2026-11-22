# Interface Tim — T2 IF3170

Supaya semua orang bisa kerja paralel tanpa nunggu notebook selesai.

## Data siap pakai

Sudah diekspor ke `output/prepared/`:

| File | Isi |
|---|---|
| `train.npz` | `X` (21898 × 40), `y` (21898,) |
| `val.npz` | `X` (5475 × 40), `y` (5475,) |
| `test.npz` | `X` (6844 × 40), `ids` (6844,) |
| `feature_names.json` | urutan 40 nama fitur |
| `united_setup.pkl` | objek `united_setup` yang sudah fit (untuk transform data baru) |

Load:

```python
import numpy as np
d = np.load("output/prepared/train.npz")
X_train, y_train = d["X"], d["y"]
```

Sudah termasuk: cleaning, imputasi (median/modus + flag missing), one-hot, scaling, seleksi fitur top-40 mutual information. **Tidak ada NaN.** Split stratified 80/20, `seed=42`, target rate 32,77% di train dan val.

Kalau butuh fitur lebih banyak: `united_setup(select_k=None)` → 49 fitur.
Kalau mau coba PCA: `united_setup(use_pca=True)`.

## Cara pakai kelas `united_setup`

```python
import sys; sys.path.insert(0, ".")
from src.data.united_setup import united_setup

setup = united_setup()
train_set, val_set = setup.split(train_df)
X_train, y_train = setup.fit_transform(train_set)   # fit hanya di train
X_val = setup.transform(val_set)                     # transform, bukan fit
X_test = setup.transform(test_df)
```

Aturan keras: **jangan pernah `fit` di val/test.** Semua statistik (imputer, encoder, scaler, batas outlier, selector) hanya dari train.

## Kontrak kelas model

Setiap model di `src/models/<nama>.py`, dengan signature seragam:

```python
class KNN:
    def __init__(self, ...): ...
    def fit(self, X, y): ...
    def predict(self, X): ...
    def predict_proba(self, X) -> np.ndarray:  # shape (n, 2)
        ...
```

`X` adalah `np.ndarray` float dari `united_setup`. `y` integer 0/1. Tidak perlu scaling/encoding lagi di dalam model.

## Nama variabel yang dicek rubric template

Template notebook punya komentar rubric tersembunyi (HTML comment) yang menyebut nama variabel. Jangan diubah namanya:

| Bagian | Nama yang harus muncul |
|---|---|
| Attribute construction | kolom `rate_per_guest_night` |
| Preparation pipeline | `fitted_prep_steps` (dan `steps`) |
| KNN | `val_neighbor_dists` |
| SVM | `slack_total` |
| Model evaluation | figure palette `#6d2fa3` |
| Artifacts | file `<model>_v3.pkl` |

## Pembagian

| Orang | Model | File |
|---|---|---|
| Ega | data + `united_setup` + notebook + submission | `src/data/united_setup.py` |
| Awan | KNN, Naive Bayes | `src/models/knn.py`, `src/models/naive_bayes.py` |
| Ahmad | Decision Tree (pilih ID3/C4.5/CART) | `src/models/decision_tree.py` |
| Jojo | Logistic Regression (L1/L2/Elastic Net) | `src/models/logistic_regression.py` |
| Faris | SVM (linear/poly/gaussian) | `src/svm.py` |

## Checklist model

- `fit` / `predict` / `predict_proba`
- versi from scratch: hanya NumPy (boleh pandas untuk I/O)
- versi scikit-learn sebagai pembanding
- simpan ke `output/<model>_v3.pkl`
- laporkan macro F1 di val
