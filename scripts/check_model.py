"""Quick validation harness for any team model.

Usage, from the project root:

    .venv/bin/python scripts/check_model.py src.models.knn:KNN
    .venv/bin/python scripts/check_model.py src.models.svm:SVM --name svm --save

It loads the prepared split, fits the class, prints macro F1 on validation,
checks predict_proba, and optionally saves output/<name>.pkl.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREP = ROOT / "output" / "prepared"


def load_target(spec: str):
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise SystemExit("format: package.module:ClassName")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="module:Class, e.g. src.models.knn:KNN")
    parser.add_argument("--name", help="artifact name, saves output/<name>.pkl")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    train, val = np.load(PREP / "train.npz"), np.load(PREP / "val.npz")
    X_train, y_train = train["X"], train["y"]
    X_val, y_val = val["X"], val["y"]

    cls = load_target(args.target)
    model = cls()

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict(X_val)
    pred_s = time.perf_counter() - t0

    macro = f1_score(y_val, pred, average="macro")
    print(f"{args.target}")
    print(f"  fit      : {fit_s:.2f}s")
    print(f"  predict  : {pred_s:.2f}s")
    print(f"  macro F1 : {macro:.4f}")
    print(classification_report(y_val, pred, digits=4))

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X_val))
        n_classes = len(getattr(model, "classes_", np.unique(y_train)))
        ok_shape = proba.shape == (len(X_val), n_classes)
        ok_sum = np.allclose(proba.sum(axis=1), 1, atol=1e-6)
        print(f"  predict_proba shape ok: {ok_shape} | rows sum to 1: {ok_sum}")
    else:
        print("  predict_proba: MISSING")

    if args.save:
        name = args.name or args.target.split(":")[-1].lower()
        path = ROOT / "output" / f"{name}.pkl"
        joblib.dump(model, path)
        print(f"  saved: {path}")


if __name__ == "__main__":
    main()
