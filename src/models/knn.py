from __future__ import annotations

import numpy as np

from src.models.persistence import PersistableMixin


class KNN(PersistableMixin):
    def __init__(self, k: int = 5, p: float = 7, weights: str = "uniform") -> None:
        self.k = k
        self.p = p
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        self.n_features_in_ = self.X_.shape[1]
        self.val_neighbor_dists = None
        self.val_neighbor_indices = None
        return self

    def pairwise_distances(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Y = self.X_ if Y is None else np.asarray(Y, dtype=np.float64)
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have the same number of features")

        out = np.empty((len(X), len(Y)), dtype=np.float64)
        for start in range(0, len(X), 512):
            stop = min(start + 512, len(X))
            diff = np.abs(X[start:stop, None, :] - Y[None, :, :])
            out[start:stop] = (diff**self.p).sum(axis=2) ** (1.0 / self.p)
        return out

    def kneighbors(self, X: np.ndarray, n_neighbors: int | None = None):
        k = self.k if n_neighbors is None else n_neighbors
        if k < 1:
            raise ValueError("k must be at least 1")
        if k > len(self.X_):
            raise ValueError("k cannot exceed the number of training samples")

        distances = self.pairwise_distances(X)
        indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
        rows = np.arange(len(distances))[:, None]
        order = np.argsort(distances[rows, indices], axis=1)
        indices = indices[rows, order]
        distances = distances[rows, indices]

        self.val_neighbor_dists = distances
        self.val_neighbor_indices = indices
        return distances, indices

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distances, indices = self.kneighbors(X)
        labels = self.y_[indices]
        proba = np.zeros((len(labels), len(self.classes_)), dtype=np.float64)

        if self.weights == "uniform":
            for j, cls in enumerate(self.classes_):
                proba[:, j] = (labels == cls).sum(axis=1)
        elif self.weights == "distance":
            w = 1.0 / (distances + 1e-12)
            for j, cls in enumerate(self.classes_):
                proba[:, j] = (w * (labels == cls)).sum(axis=1)
        else:
            raise ValueError("weights must be 'uniform' or 'distance'")

        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
