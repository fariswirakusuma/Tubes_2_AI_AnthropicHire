"""Smoke test + placeholder Kaggle submission for the data pipeline.

Run from the project root:
    .venv/bin/python scripts/baseline.py
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import DataPreprocessor, TARGET, ID_COL  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

train_df = pd.read_csv(RAW / "train.csv", dtype={ID_COL: str})
test_df = pd.read_csv(RAW / "test.csv", dtype={ID_COL: str})

prep = DataPreprocessor(seed=42, select_k=38)
train_set, val_set = prep.split(train_df)
print(f"train_set: {train_set.shape} | val_set: {val_set.shape}")
print(f"target rate -> train {train_set[TARGET].mean():.4f} | val {val_set[TARGET].mean():.4f}")

X_train, y_train = prep.fit_transform(train_set)
X_val = prep.transform(val_set)
y_val = val_set[TARGET].to_numpy()
X_test = prep.transform(test_df)

print(f"\nX_train {X_train.shape} | X_val {X_val.shape} | X_test {X_test.shape}")
print(f"NaN in X_train: {np.isnan(X_train).sum()} | NaN in X_val: {np.isnan(X_val).sum()} | NaN in X_test: {np.isnan(X_test).sum()}")
print(f"features kept: {X_train.shape[1]} (select_k=38)")

print("\nTop features by mutual information:")
print(prep.summary().head(12).to_string(index=False))

model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
macro_f1 = f1_score(y_val, val_pred, average="macro")
print(f"\nBaseline sklearn LogisticRegression — val macro F1: {macro_f1:.4f}")
print(classification_report(y_val, val_pred, digits=4))

# Refit on all labelled data for the placeholder submission.
X_full, y_full = prep.fit_transform(train_df)
model_full = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model_full.fit(X_full, y_full)
X_test_full = prep.transform(test_df)

proba = model_full.predict_proba(X_test_full)[:, 1]
pred = (proba >= 0.5).astype(int)

sub = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: pred})
sub_path = OUT / "submission_baseline.csv"
sub.to_csv(sub_path, index=False)

digest = hashlib.sha256(sub_path.read_bytes()).hexdigest()
print(f"\nwrote {sub_path} rows={len(sub)} positives={pred.mean():.4f}")
print(f"sha256: {digest}")

prep.save(OUT / "preprocessor.pkl")
print(f"saved {OUT / 'preprocessor.pkl'}")
