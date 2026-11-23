from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import (  # noqa: E402
    DecisionTree,
    KNN,
    LogisticRegression,
    NaiveBayes,
    SVM,
)

PREP = ROOT / "output" / "prepared"
OUT = ROOT / "output"


def load_split(name: str):
    data = np.load(PREP / f"{name}.npz")
    return data["X"], data["y"]


def run(name: str, model, X_train, y_train, X_val, y_val, results: list, save: bool = True):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(X_val)
    pred_s = time.perf_counter() - t0
    score = f1_score(y_val, pred, average="macro")
    results.append(
        {"model": name, "fit_s": fit_s, "predict_s": pred_s, "macro_f1": score}
    )
    print(f"{name:<34} fit={fit_s:7.2f}s  pred={pred_s:7.2f}s  macro-F1={score:.4f}")
    if save:
        model.save(OUT / f"{name.replace(' ', '_').lower()}.pkl")
    return model


def main() -> None:
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    mi = mutual_info_classif(X_train, y_train, random_state=42, n_neighbors=3)
    top5 = np.argsort(mi)[::-1][:5]

    knn_train, knn_val = 5000, 1000
    rng = np.random.default_rng(42)
    knn_idx = rng.choice(len(X_train), knn_train, replace=False)
    val_idx = rng.choice(len(X_val), knn_val, replace=False)

    results = []

    run(
        "knn_scratch",
        KNN(k=5, p=2, weights="distance"),
        X_train[knn_idx],
        y_train[knn_idx],
        X_val[val_idx],
        y_val[val_idx],
        results,
    )
    run(
        "knn_sklearn",
        KNeighborsClassifier(n_neighbors=5, p=2, weights="distance", n_jobs=-1),
        X_train[knn_idx],
        y_train[knn_idx],
        X_val[val_idx],
        y_val[val_idx],
        results,
        save=False,
    )

    run(
        "dtl_scratch",
        DecisionTree(criterion="gini", max_depth=12, min_samples_leaf=5),
        X_train,
        y_train,
        X_val,
        y_val,
        results,
    )
    run(
        "dtl_sklearn",
        DecisionTreeClassifier(
            criterion="gini", max_depth=12, min_samples_leaf=5, random_state=42
        ),
        X_train,
        y_train,
        X_val,
        y_val,
        results,
        save=False,
    )

    run(
        "logreg_scratch",
        LogisticRegression(
            learning_rate=0.5,
            epochs=1000,
            regularization="l2",
            lambda_=1e-3,
            class_weight="balanced",
        ),
        X_train,
        y_train,
        X_val,
        y_val,
        results,
    )
    run(
        "logreg_sklearn",
        SkLogisticRegression(max_iter=1000, class_weight="balanced", C=1000.0),
        X_train,
        y_train,
        X_val,
        y_val,
        results,
        save=False,
    )

    run(
        "svm_linear_scratch",
        SVM(kernel="linear", C=1.0, learning_rate=0.002, epochs=50),
        X_train,
        y_train,
        X_val,
        y_val,
        results,
    )
    run(
        "svm_linear_sklearn",
        SVC(kernel="linear", C=1.0),
        X_train[knn_idx],
        y_train[knn_idx],
        X_val,
        y_val,
        results,
        save=False,
    )

    run(
        "svm_gaussian_scratch",
        SVM(
            kernel="gaussian",
            C=1.0,
            gamma=0.15,
            learning_rate=0.05,
            epochs=200,
            max_samples=2000,
        ),
        X_train,
        y_train,
        X_val,
        y_val,
        results,
    )
    run(
        "svm_gaussian_sklearn",
        SVC(kernel="rbf", C=1.0, gamma=0.05),
        X_train[knn_idx],
        y_train[knn_idx],
        X_val,
        y_val,
        results,
        save=False,
    )

    poly_idx = rng.choice(len(X_train), 2000, replace=False)
    run(
        "svm_poly_scratch",
        SVM(
            kernel="poly",
            degree=3,
            coef0=1.0,
            gamma=0.1,
            C=1.0,
            learning_rate=0.05,
            epochs=200,
            max_samples=2000,
        ),
        X_train,
        y_train,
        X_val,
        y_val,
        results,
    )
    run(
        "svm_poly_sklearn",
        SVC(kernel="poly", degree=3, coef0=1.0, gamma=0.1, C=1.0),
        X_train[poly_idx],
        y_train[poly_idx],
        X_val,
        y_val,
        results,
        save=False,
    )

    run(
        "nb_scratch",
        NaiveBayes(var_smoothing=1e-6),
        X_train[:, top5],
        y_train,
        X_val[:, top5],
        y_val,
        results,
    )
    run(
        "nb_sklearn",
        GaussianNB(),
        X_train[:, top5],
        y_train,
        X_val[:, top5],
        y_val,
        results,
        save=False,
    )

    table = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    path = OUT / "model_comparison.csv"
    table.to_csv(path, index=False)
    print(f"\nwrote {path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
