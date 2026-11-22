"""Export the prepared matrices so every member can train without rerunning prep.

    .venv/bin/python scripts/export_prepared.py

Writes:
    output/prepared/train.npz  (X, y)
    output/prepared/val.npz    (X, y)
    output/prepared/test.npz   (X, ids)
    output/prepared/feature_names.json
    output/prepared/united_setup.pkl
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.united_setup import united_setup, TARGET, ID_COL  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "output" / "prepared"
OUT.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(RAW / "train.csv", dtype={ID_COL: str})
test_df = pd.read_csv(RAW / "test.csv", dtype={ID_COL: str})

setup = united_setup(seed=42, select_k=40)
train_set, val_set = setup.split(train_df)

X_train, y_train = setup.fit_transform(train_set)
X_val = setup.transform(val_set)
y_val = val_set[TARGET].to_numpy()
X_test = setup.transform(test_df)

np.savez_compressed(OUT / "train.npz", X=X_train, y=y_train)
np.savez_compressed(OUT / "val.npz", X=X_val, y=y_val)
np.savez_compressed(OUT / "test.npz", X=X_test, ids=test_df[ID_COL].to_numpy())
(OUT / "feature_names.json").write_text(json.dumps(setup.feature_names_out_, indent=2))
setup.save(OUT / "united_setup.pkl")

print(f"train {X_train.shape} positives={y_train.mean():.4f}")
print(f"val   {X_val.shape} positives={y_val.mean():.4f}")
print(f"test  {X_test.shape}")
print(f"features {len(setup.feature_names_out_)} -> {OUT / 'feature_names.json'}")
