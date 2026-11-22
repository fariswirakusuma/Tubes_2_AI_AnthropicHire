# Model from scratch

`src/models/svm.py` (Faris) adalah implementasi SVM from scratch (NumPy).
Kelima classifier di `src/models/` wajib from scratch; scikit-learn hanya
dipakai untuk preprocessing dan pembanding di notebook/`scripts/compare_models.py`.
Logistic Regression from scratch ada di `src/models/logistic_regression.py`.

Import semua model dari `src.models`:

```python
from src.models import KNN, DecisionTree, LogisticRegression, SVM, NaiveBayes
```

Semua model memakai kontrak yang sama:

```python
model.fit(X_train, y_train)
proba = model.predict_proba(X_val)   # shape (n, n_classes), baris jumlah 1
pred = model.predict(X_val)
model.save("output/model_v3.pkl")
model = type(model).load("output/model_v3.pkl")
```

`X` berupa matriks NumPy (`float64`). `DecisionTree` juga menerima
`pandas.DataFrame` untuk kolom kategorikal dan nilai kosong.

## Catatan per model

| Model | Parameter penting | Catatan |
|---|---|---|
| `KNN` | `k`, `p=7`, `weights` | Minkowski default dimensi 7. Prediksi jarak jauh lebih cepat dengan subset train atau PCA. |
| `DecisionTree` | `criterion`, `max_depth`, `min_samples_leaf` | CART histogram. Menangani numerik, kategorikal, dan NaN. |
| `LogisticRegression` | `regularization`, `lambda_`, `l1_ratio`, `epochs` | Softmax multiclass, L1/L2/Elastic Net. |
| `SVM` | `kernel`, `C`, `degree`, `gamma`, `max_samples` | Linear memakai seluruh data; poly/gaussian memakai subset `max_samples`. |
| `NaiveBayes` | `var_smoothing`, `class_weight` | Gaussian NB. Alias `BayesianNetwork` tersedia. |

## Atribut yang dicek rubric

- `KNN.val_neighbor_dists` — jarak tetangga dari prediksi terakhir.
- `DecisionTree.tree_escrowstate_score_trig` — skor split akar.
- `LogisticRegression.log_uniconilear` — log-loss akhir.
- `SVM.slack_total` — total slack hinge.
- `NaiveBayes.bayesian_disweighting` — bobot kelas (log prior).

## Contoh cepat

```python
from src.models import DecisionTree, SVM

tree = DecisionTree(criterion="gini", max_depth=12, min_samples_leaf=5)
tree.fit(X_train, y_train)

svm = SVM(kernel="gaussian", C=1.0, gamma=0.15, max_samples=2000)
svm.fit(X_train, y_train)
```

Perbandingan lengkap dengan scikit-learn:

```bash
.venv/bin/python scripts/compare_models.py
```
