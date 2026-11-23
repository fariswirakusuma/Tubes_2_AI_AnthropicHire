from __future__ import annotations

import numpy as np

from src.models.persistence import PersistableMixin


class SVM(PersistableMixin):
    def __init__(
        self,
        kernel: str = "linear",
        C: float = 1.0,
        degree: int = 3,
        gamma: str | float = "scale",
        coef0: float = 0.0,
        learning_rate: float = 0.05,
        epochs: int = 200,
        max_samples: int | None = 2000,
        random_state: int = 42,
    ) -> None:
        if kernel not in ("linear", "poly", "polynomial", "gaussian", "rbf"):
            raise ValueError("kernel must be linear, poly, or gaussian")
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_samples = max_samples
        self.random_state = random_state

    def _kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            return X @ Y.T
        if self.kernel in ("poly", "polynomial"):
            return (self.gamma_ * (X @ Y.T) + self.coef0) ** self.degree
        sq = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2.0 * (X @ Y.T)
        return np.exp(-self.gamma_ * np.maximum(sq, 0.0))

    def _set_gamma(self, X: np.ndarray) -> None:
        if isinstance(self.gamma, str):
            if self.gamma == "scale":
                var = X.var()
                self.gamma_ = 1.0 / (X.shape[1] * var) if var > 0 else 1.0
            elif self.gamma == "auto":
                self.gamma_ = 1.0 / X.shape[1]
            else:
                raise ValueError("gamma must be 'scale', 'auto', or a number")
        else:
            self.gamma_ = float(self.gamma)

    def _subsample(self, X: np.ndarray, y: np.ndarray, limit: int):
        if len(X) <= limit:
            return X, y, np.arange(len(X))
        rng = np.random.default_rng(self.random_state)
        classes = np.unique(y)
        chosen = []
        for cls in classes:
            cls_idx = np.flatnonzero(y == cls)
            take = max(1, int(round(limit * len(cls_idx) / len(y))))
            chosen.append(rng.choice(cls_idx, size=min(take, len(cls_idx)), replace=False))
        idx = np.concatenate(chosen)
        rng.shuffle(idx)
        return X[idx], y[idx], idx

    def _fit_linear(self, X: np.ndarray, y: np.ndarray, pos, neg):
        y_pm = np.where(y == 1, 1.0, -1.0)
        n, d = X.shape
        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        lam = 1.0 / (self.C * n)
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.epochs):
            order = rng.permutation(n)
            for i in order:
                if y_pm[i] * (X[i] @ w + b) < 1.0:
                    w = (1.0 - self.learning_rate * lam) * w + self.learning_rate * y_pm[i] * X[i]
                    b += self.learning_rate * y_pm[i]

        decision = X @ w + b
        support = np.flatnonzero(y_pm * decision <= 1.0 + 1e-6)
        return {
            "kind": "linear",
            "w": w,
            "b": b,
            "y_pm": y_pm,
            "classes": (neg, pos),
            "support": support,
            "support_vectors": X[support],
            "dual_coef": y_pm[support],
        }

    def _fit_kernel(self, X: np.ndarray, y: np.ndarray, pos, neg):
        limit = self.max_samples or len(X)
        Xb, yb, _ = self._subsample(X, y, limit)
        y_pm = np.where(yb == 1, 1.0, -1.0)
        K = self._kernel_matrix(Xb, Xb)
        alpha = np.zeros(len(Xb), dtype=np.float64)
        b = 0.0
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.epochs):
            f = K @ (alpha * y_pm) + b
            order = rng.permutation(len(Xb))
            for i in order:
                if y_pm[i] * f[i] < 1.0:
                    alpha[i] = min(self.C, alpha[i] + self.learning_rate)
                    b += self.learning_rate * y_pm[i]
                    f += self.learning_rate * y_pm[i] * K[:, i] + self.learning_rate * y_pm[i]

        support = np.flatnonzero(alpha > 1e-6)
        return {
            "kind": "kernel",
            "X": Xb,
            "alpha": alpha,
            "b": b,
            "y_pm": y_pm,
            "classes": (neg, pos),
            "support": support,
            "support_vectors": Xb[support],
            "dual_coef": (alpha * y_pm)[support],
        }

    def _fit_binary(self, X: np.ndarray, y: np.ndarray, pos, neg):
        if self.kernel == "linear":
            return self._fit_linear(X, y, pos, neg)
        return self._fit_kernel(X, y, pos, neg)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self._set_gamma(X)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        if len(self.classes_) == 2:
            y01 = (y == self.classes_[1]).astype(int)
            self.estimators_ = [self._fit_binary(X, y01, self.classes_[1], self.classes_[0])]
        else:
            self.estimators_ = []
            for i in range(len(self.classes_)):
                for j in range(i + 1, len(self.classes_)):
                    mask = (y == self.classes_[i]) | (y == self.classes_[j])
                    y01 = (y[mask] == self.classes_[j]).astype(int)
                    self.estimators_.append(
                        self._fit_binary(X[mask], y01, self.classes_[j], self.classes_[i])
                    )

        if len(self.estimators_) == 1:
            est = self.estimators_[0]
            self.support_ = est["support"]
            self.support_vectors_ = est["support_vectors"]
            self.dual_coef_ = est["dual_coef"]
            self.intercept_ = est["b"]
            if est["kind"] == "linear":
                self.coef_ = est["w"]
        return self

    def _decision(self, est, X: np.ndarray) -> np.ndarray:
        if est["kind"] == "linear":
            return X @ est["w"] + est["b"]
        K = self._kernel_matrix(X, est["X"])
        return K @ (est["alpha"] * est["y_pm"]) + est["b"]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if len(self.classes_) == 2:
            return self._decision(self.estimators_[0], X)

        scores = np.zeros((len(X), len(self.classes_)), dtype=np.float64)
        for est in self.estimators_:
            f = self._decision(est, X)
            neg, pos = est["classes"]
            scores[:, self.classes_ == pos] += f[:, None]
            scores[:, self.classes_ == neg] -= f[:, None]
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if len(self.classes_) == 2:
            f = self.decision_function(X)
            return self.classes_[(f > 0).astype(int)]

        votes = np.zeros((len(X), len(self.classes_)), dtype=np.float64)
        for est in self.estimators_:
            f = self._decision(est, X)
            neg, pos = est["classes"]
            pos_mask = f > 0
            votes[pos_mask, np.where(self.classes_ == pos)[0][0]] += 1
            votes[~pos_mask, np.where(self.classes_ == neg)[0][0]] += 1
        return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            p = 1.0 / (1.0 + np.exp(-np.clip(scores, -500.0, 500.0)))
            return np.column_stack([1.0 - p, p])
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)
