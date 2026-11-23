from __future__ import annotations

import numpy as np

from src.models.persistence import PersistableMixin


class KNN(PersistableMixin):
    def __init__(
        self,
        k: int = 9,
        p: float = 2,
        weights: str = "distance",
        max_chunk_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.k = k
        self.p = p
        self.weights = weights
        self.max_chunk_bytes = max_chunk_bytes

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        self.n_features_in_ = self.X_.shape[1]
        return self

    # ------------------------------------------------------------ distances
    def pairwise_distances(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Y = self.X_ if Y is None else np.asarray(Y, dtype=np.float64)
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have the same number of features")

        if self.p == 2:
            sq = (X**2).sum(1)[:, None] + (Y**2).sum(1)[None, :] - 2.0 * X @ Y.T
            return np.sqrt(np.maximum(sq, 0.0))  

        out = np.empty((len(X), len(Y)), dtype=np.float64)
        rows = max(1, self.max_chunk_bytes // (len(Y) * X.shape[1] * 8))
        for start in range(0, len(X), rows):
            stop = min(start + rows, len(X))
            diff = np.abs(X[start:stop, None, :] - Y[None, :, :])
            if self.p == 1:
                out[start:stop] = diff.sum(axis=2)
            else:
                out[start:stop] = (diff**self.p).sum(axis=2) ** (1.0 / self.p)
        return out

    # ------------------------------------------------------------ neighbors
    def kneighbors(self, X: np.ndarray, n_neighbors: int | None = None):
        k = self.k if n_neighbors is None else n_neighbors
        if k < 1:
            raise ValueError("k must be at least 1")
        if k > len(self.X_):
            raise ValueError("k cannot exceed the number of training samples")

        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        dists = np.empty((n, k))
        idxs = np.empty((n, k), dtype=np.int64)
        block = max(1, self.max_chunk_bytes // (len(self.X_) * 8))
        for s in range(0, n, block):
            D = self.pairwise_distances(X[s : s + block])
            part = np.argpartition(D, k - 1, axis=1)[:, :k]
            rows = np.arange(len(D))[:, None]
            order = np.argsort(D[rows, part], axis=1)
            part = part[rows, order]
            idxs[s : s + block] = part
            dists[s : s + block] = D[rows, part]

        return dists, idxs

    # -------------------------------------------------------------- predict
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distances, indices = self.kneighbors(X)
        labels = self.y_[indices]
        proba = np.zeros((len(labels), len(self.classes_)), dtype=np.float64)

        if self.weights == "uniform":
            w = np.ones_like(distances)
        elif self.weights == "distance":
            w = 1.0 / (distances + 1e-12)
        else:
            raise ValueError("weights must be 'uniform' or 'distance'")

        for j, cls in enumerate(self.classes_):
            proba[:, j] = (w * (labels == cls)).sum(axis=1)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
