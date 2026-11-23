from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.models.persistence import PersistableMixin


@dataclass
class _Node:
    value: np.ndarray
    n_samples: int
    impurity: float
    feature: int | None = None
    kind: str | None = None
    split_bin: int | None = None
    category_code: int | None = None
    default_left: bool = True
    left: "_Node | None" = None
    right: "_Node | None" = None
    score: float = 0.0


class DecisionTree(PersistableMixin):
    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_bins: int = 32,
    ) -> None:
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins

    def fit(self, X, y) -> "DecisionTree":
        y = np.asarray(y)
        if np.any(pd.isna(y)):
            raise ValueError("target contains missing values")
        self.classes_, self._y_enc = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = (
            list(X.columns)
            if isinstance(X, pd.DataFrame)
            else [f"x{i}" for i in range(self.n_features_in_)]
        )
        self._is_categorical_, self._edges_, self._cat_maps_ = self._fit_encoders(X)
        self._binned_ = self._encode(X)
        self.split_scores_ = []

        root = self._build(np.arange(len(self._binned_)), depth=0)
        self.root_ = root
        self._compute_importances()
        return self

    def _fit_encoders(self, X):
        is_cat = []
        edges = []
        cat_maps = []
        if isinstance(X, pd.DataFrame):
            columns = [X.iloc[:, j] for j in range(X.shape[1])]
        else:
            arr = np.asarray(X)
            columns = [arr[:, j] for j in range(arr.shape[1])]

        for col in columns:
            series = pd.Series(col)
            if is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                is_cat.append(False)
                valid = series.dropna().to_numpy(dtype=np.float64)
                if len(valid) == 0:
                    edges.append(np.array([]))
                else:
                    qs = np.linspace(0, 1, self.max_bins + 1)[1:-1]
                    edges.append(np.unique(np.quantile(valid, qs)))
                cat_maps.append(None)
            else:
                is_cat.append(True)
                values = pd.unique(series.dropna())
                cat_maps.append({value: code for code, value in enumerate(values)})
                edges.append(None)
        return np.array(is_cat, dtype=bool), edges, cat_maps

    def _encode(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if hasattr(self, "feature_names_in_") and list(X.columns) != self.feature_names_in_:
                X = X[self.feature_names_in_]
            columns = [X.iloc[:, j].to_numpy() for j in range(X.shape[1])]
        else:
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            columns = [arr[:, j] for j in range(arr.shape[1])]

        out = np.full((len(columns[0]), len(columns)), -1, dtype=np.int32)
        for j, col in enumerate(columns):
            series = pd.Series(col)
            missing = series.isna().to_numpy()
            if self._is_categorical_[j]:
                mapping = self._cat_maps_[j]
                codes = series.map(mapping).to_numpy(dtype=np.float64)
                codes = np.where(pd.isna(codes), -1, codes).astype(np.int32)
                out[:, j] = codes
            else:
                values = series.to_numpy(dtype=np.float64, na_value=np.nan)
                edges = self._edges_[j]
                binned = np.searchsorted(edges, values, side="right")
                binned = np.where(np.isnan(values), -1, binned).astype(np.int32)
                out[:, j] = binned
            out[missing, j] = -1
        return out

    def _impurity(self, counts: np.ndarray) -> float:
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        if self.criterion == "gini":
            return float(1.0 - np.sum(p * p))
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def _build(self, indices: np.ndarray, depth: int) -> _Node:
        y = self._y_enc[indices]
        counts = np.bincount(y, minlength=len(self.classes_)).astype(np.float64)
        node = _Node(
            value=counts / counts.sum(),
            n_samples=len(indices),
            impurity=self._impurity(counts),
        )

        if (
            node.impurity == 0.0
            or len(indices) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node

        best = self._best_split(indices)
        if best is None or best["score"] <= 1e-12:
            return node

        node.feature = best["feature"]
        node.kind = best["kind"]
        node.split_bin = best["split_bin"]
        node.category_code = best["category_code"]
        node.default_left = best["default_left"]
        node.score = best["score"]
        node.left = self._build(best["left_indices"], depth + 1)
        node.right = self._build(best["right_indices"], depth + 1)
        self.split_scores_.append(best["score"])
        return node

    def _best_split(self, indices: np.ndarray):
        n = len(indices)
        y = self._y_enc[indices]
        parent_counts = np.bincount(y, minlength=len(self.classes_)).astype(np.float64)
        parent_impurity = self._impurity(parent_counts)
        n_classes = len(self.classes_)

        best = None
        for j in range(self.n_features_in_):
            b = self._binned_[indices, j]
            valid = b >= 0
            missing = ~valid
            if valid.sum() == 0:
                continue

            if self._is_categorical_[j]:
                n_bins = len(self._cat_maps_[j])
                counts = np.zeros((n_classes, n_bins), dtype=np.float64)
                np.add.at(counts, (y[valid], b[valid]), 1)
                missing_counts = np.bincount(y[missing], minlength=n_classes).astype(np.float64)

                for code in range(n_bins):
                    left_counts = counts[:, code].copy()
                    right_counts = parent_counts - left_counts
                    left_n = left_counts.sum()
                    right_n = right_counts.sum()
                    if left_n == 0 or right_n == 0:
                        continue
                    default_left = left_n >= right_n
                    if default_left:
                        left_counts = left_counts + missing_counts
                        left_n += missing_counts.sum()
                    else:
                        right_counts = right_counts + missing_counts
                        right_n += missing_counts.sum()
                    if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                        continue
                    score = parent_impurity - (
                        left_n / n * self._impurity(left_counts)
                        + right_n / n * self._impurity(right_counts)
                    )
                    if best is None or score > best["score"]:
                        left_mask = (b == code) | (missing & default_left)
                        best = {
                            "feature": j,
                            "kind": "categorical",
                            "split_bin": None,
                            "category_code": code,
                            "default_left": default_left,
                            "score": float(score),
                            "left_indices": indices[left_mask],
                            "right_indices": indices[~left_mask],
                        }
            else:
                n_bins = len(self._edges_[j]) + 1
                counts = np.zeros((n_classes, n_bins), dtype=np.float64)
                np.add.at(counts, (y[valid], b[valid]), 1)
                missing_counts = np.bincount(y[missing], minlength=n_classes).astype(np.float64)
                cumulative = np.cumsum(counts, axis=1)

                for split_bin in range(n_bins - 1):
                    left_counts = cumulative[:, split_bin].copy()
                    right_counts = parent_counts - left_counts
                    left_n = left_counts.sum()
                    right_n = right_counts.sum()
                    default_left = left_n >= right_n
                    if default_left:
                        left_counts = left_counts + missing_counts
                        left_n += missing_counts.sum()
                    else:
                        right_counts = right_counts + missing_counts
                        right_n += missing_counts.sum()
                    if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                        continue
                    score = parent_impurity - (
                        left_n / n * self._impurity(left_counts)
                        + right_n / n * self._impurity(right_counts)
                    )
                    if best is None or score > best["score"]:
                        left_mask = (b <= split_bin) | (missing & default_left)
                        best = {
                            "feature": j,
                            "kind": "numeric",
                            "split_bin": split_bin,
                            "category_code": None,
                            "default_left": default_left,
                            "score": float(score),
                            "left_indices": indices[left_mask],
                            "right_indices": indices[~left_mask],
                        }
        return best

    def _route(self, node: _Node, binned_row: np.ndarray) -> _Node:
        while node.left is not None:
            value = binned_row[node.feature]
            if value < 0:
                node = node.left if node.default_left else node.right
            elif node.kind == "numeric":
                node = node.left if value <= node.split_bin else node.right
            else:
                node = node.left if value == node.category_code else node.right
        return node

    def predict_proba(self, X) -> np.ndarray:
        binned = self._encode(X)
        proba = np.empty((len(binned), len(self.classes_)), dtype=np.float64)
        for i, row in enumerate(binned):
            proba[i] = self._route(self.root_, row).value
        return proba

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_depth(self) -> int:
        def depth(node: _Node) -> int:
            if node.left is None:
                return 0
            return 1 + max(depth(node.left), depth(node.right))

        return depth(self.root_)

    def _compute_importances(self) -> None:
        importances = np.zeros(self.n_features_in_, dtype=np.float64)

        def walk(node: _Node):
            if node.left is None:
                return
            importances[node.feature] += node.score * node.n_samples
            walk(node.left)
            walk(node.right)

        walk(self.root_)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
