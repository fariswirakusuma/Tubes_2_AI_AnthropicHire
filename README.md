# Tubes 2 AI — AnthropicHire

Tugas Besar 2 IF3170 Inteligensi Artifisial 2026/2027 — supervised learning
from scratch untuk kompetisi Kaggle `T2-IF3170-2026-2027` (metrik macro F1).

## Struktur

| Path | Isi |
|---|---|
| `src/data/united_setup.py` | pipeline preprocessing: split, imputasi + flag missing, feature engineering, one-hot, scaling, seleksi fitur |
| `src/models/` | satu file per model, kontrak `fit` / `predict` / `predict_proba` |
| `scripts/export_prepared.py` | regenerate `output/prepared/*` dari `data/raw/*` |
| `scripts/check_model.py` | harness cepat: macro F1 validasi, cek `predict_proba`, simpan pkl |
| `scripts/baseline.py` | smoke test pipeline + submission placeholder |
| `notebooks/00_main_pipeline.ipynb` | template notebook tugas besar |
| `docs/` | interface tim, handoff, ringkasan EDA |
| `output/prepared/` | split siap pakai (train/val/test `.npz`, 40 fitur) |

## Aturan implementasi

KNN, Decision Tree, Logistic Regression, SVM, dan Naive Bayes wajib **from
scratch** dengan NumPy (pandas untuk I/O). scikit-learn hanya dipakai untuk
preprocessing dan pembanding; submission final memakai model from scratch.
Aturan ini dijaga oleh `tests/test_from_scratch.py`.

## Data

Download dataset dari Kaggle `T2-IF3170-2026-2027`, lalu taruh di:

```
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

Split yang sudah diproses tersedia di `output/prepared/`:

| File | Isi |
|---|---|
| `train.npz` | `X` (21898 × 40), `y` |
| `val.npz` | `X` (5475 × 40), `y` |
| `test.npz` | `X` (6844 × 40), `ids` |
| `feature_names.json` | urutan 40 fitur |
| `united_setup.pkl` | objek preprocessing yang sudah fit |

Kalau tidak mengubah pipeline, tidak perlu download ulang.

## Quickstart

```bash
make venv install
.venv/bin/python scripts/check_model.py src.models._template:TemplateModel
```

Baseline pipeline (butuh `data/raw/`):

```bash
.venv/bin/python scripts/export_prepared.py
.venv/bin/python scripts/baseline.py
```

Baseline sklearn Logistic Regression: val macro F1 **0.7547**.

## Pembagian

| Anggota | Model | File |
|---|---|---|
| Ega | data + `united_setup` + notebook + submission | `src/data/united_setup.py` |
| Awan | KNN, Naive Bayes | `src/models/knn.py`, `src/models/naive_bayes.py` |
| Ahmad | Decision Tree (ID3/C4.5/CART) | `src/models/decision_tree.py` |
| Jojo | Logistic Regression (L1/L2/Elastic Net) | `src/models/logistic_regression.py` |
| Faris | SVM (linear/poly/gaussian) | `src/models/svm.py` |

Aturan kerja lengkap ada di `docs/interface.md`, rencana per tanggal di
`docs/team-handoff.md`, dan temuan data di `docs/eda-summary.md`.
