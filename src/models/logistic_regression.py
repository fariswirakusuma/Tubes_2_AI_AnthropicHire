from __future__ import annotations

import numpy as np

from src.models.persistence import PersistableMixin


class LogisticRegression(PersistableMixin):
    def __init__(
        self,
        learning_rate: float = 0.1,
        epochs: int = 1000,
        regularization: str = "l2",
        lambda_: float = 0.01,
        l1_ratio: float = 0.5,
        batch_size: int | None = None,
        tolerance: float = 1e-7,
        class_weight: str | None = None,
        random_state: int = 42,
    ) -> None:
        if regularization not in ("l1", "l2", "elasticnet", "elastic_net", "none"):
            raise ValueError("regularization must be l1, l2, elasticnet, or none")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.lambda_ = lambda_
        self.l1_ratio = l1_ratio
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.class_weight = class_weight
        self.random_state = random_state

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(np.clip(z, -500, 500))
        return exp / exp.sum(axis=1, keepdims=True)

    def _regularization_gradient(self, W: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(W)
        if self.regularization == "l2":
            grad[1:] = self.lambda_ * W[1:]
        elif self.regularization == "l1":
            grad[1:] = self.lambda_ * np.sign(W[1:])
        elif self.regularization in ("elasticnet", "elastic_net"):
            grad[1:] = self.lambda_ * (
                self.l1_ratio * np.sign(W[1:]) + (1.0 - self.l1_ratio) * W[1:]
            )
        return grad

    def _regularization_loss(self, W: np.ndarray) -> float:
        if self.regularization == "l2":
            return 0.5 * self.lambda_ * np.sum(W[1:] ** 2)
        if self.regularization == "l1":
            return self.lambda_ * np.sum(np.abs(W[1:]))
        if self.regularization in ("elasticnet", "elastic_net"):
            return self.lambda_ * (
                self.l1_ratio * np.sum(np.abs(W[1:]))
                + 0.5 * (1.0 - self.l1_ratio) * np.sum(W[1:] ** 2)
            )
        return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        Xb = np.c_[np.ones(n_samples), X]
        Y = np.eye(n_classes)[y_idx]

        sample_weight = np.ones(n_samples, dtype=np.float64)
        if self.class_weight == "balanced":
            counts = np.bincount(y_idx, minlength=n_classes)
            weights = n_samples / (n_classes * counts)
            sample_weight = weights[y_idx]

        W = np.zeros((n_features + 1, n_classes), dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        self.loss_history_ = []

        for _ in range(self.epochs):
            if self.batch_size is None:
                batches = [np.arange(n_samples)]
            else:
                order = rng.permutation(n_samples)
                batches = [
                    order[start : start + self.batch_size]
                    for start in range(0, n_samples, self.batch_size)
                ]

            epoch_loss = 0.0
            for batch in batches:
                X_batch = Xb[batch]
                Y_batch = Y[batch]
                w_batch = sample_weight[batch]

                proba = self._softmax(X_batch @ W)
                data_loss = -np.mean(
                    np.sum(Y_batch * np.log(np.maximum(proba, 1e-12)), axis=1) * w_batch
                )
                grad = X_batch.T @ ((proba - Y_batch) * w_batch[:, None]) / len(batch)
                grad += self._regularization_gradient(W)
                W -= self.learning_rate * grad
                epoch_loss += data_loss * len(batch)

            epoch_loss /= n_samples
            self.loss_history_.append(float(epoch_loss))
            if len(self.loss_history_) > 1:
                if abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.tolerance:
                    break

        self.weights_ = W
        self.coef_ = W[1:].T
        self.intercept_ = W[0]
        self.n_iter_ = len(self.loss_history_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xb = np.c_[np.ones(len(X)), X]
        return self._softmax(Xb @ self.weights_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xb = np.c_[np.ones(len(X)), X]
        return Xb @ self.weights_
