from __future__ import annotations

import numpy as np

from src.models.persistence import PersistableMixin


class NaiveBayes(PersistableMixin):
    def __init__(
        self,
        var_smoothing: float = 1e-9,
        class_prior: np.ndarray | None = None,
        class_weight: str | None = None,
    ) -> None:
        self.var_smoothing = var_smoothing
        self.class_prior = class_prior
        self.class_weight = class_weight

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayes":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.class_count_ = np.bincount(y_idx, minlength=n_classes).astype(np.float64)

        if self.class_prior is not None:
            prior = np.asarray(self.class_prior, dtype=np.float64)
            self.class_prior_ = prior / prior.sum()
        elif self.class_weight == "balanced":
            prior = n_samples / (n_classes * self.class_count_)
            self.class_prior_ = prior / prior.sum()
        else:
            self.class_prior_ = self.class_count_ / n_samples

        self.bayesian_disweighting = np.log(self.class_prior_)

        self.theta_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var_ = np.zeros((n_classes, n_features), dtype=np.float64)
        for c in range(n_classes):
            X_c = X[y_idx == c]
            self.theta_[c] = X_c.mean(axis=0)
            self.var_[c] = X_c.var(axis=0)

        epsilon = self.var_smoothing * X.var(axis=0)
        epsilon[epsilon == 0] = self.var_smoothing
        self.epsilon_ = epsilon
        self.var_ += epsilon

        log_proba = self.predict_log_proba(X)
        self.negative_log_likelihood_ = float(
            -np.mean(log_proba[np.arange(n_samples), y_idx])
        )
        self.n_features_in_ = n_features
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        joint = np.empty((len(X), len(self.classes_)), dtype=np.float64)
        for c in range(len(self.classes_)):
            var = self.var_[c]
            diff = X - self.theta_[c]
            joint[:, c] = self.bayesian_disweighting[c] - 0.5 * np.sum(
                np.log(2.0 * np.pi * var) + diff**2 / var, axis=1
            )
        return joint

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        joint = self._joint_log_likelihood(X)
        joint -= np.max(joint, axis=1, keepdims=True)
        log_norm = np.log(np.sum(np.exp(joint), axis=1, keepdims=True))
        return joint - log_norm

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]


class BayesianNetwork(NaiveBayes):
    pass
