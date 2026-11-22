"""Multiclass logistic regression (softmax) from scratch: L1, L2, Elastic Net."""

import joblib
import numpy as np


class LogisticRegression:
    def __init__(
        self,
        learning_rate=0.05,
        epochs=1200,
        regularization="l2",
        lambda_=0.01,
        l1_ratio=0.5,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.lambda_ = lambda_
        self.l1_ratio = l1_ratio

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(np.clip(z, -500, 500))
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, yi = np.unique(y, return_inverse=True)
        Y = np.eye(len(self.classes_))[yi]
        Xb = np.c_[np.ones(len(X)), X]
        self.weights_ = np.zeros((Xb.shape[1], len(self.classes_)))
        self.loss_history_ = []

        for _ in range(self.epochs):
            p = self._softmax(Xb @ self.weights_)
            loss = -np.mean(np.sum(Y * np.log(np.maximum(p, 1e-12)), axis=1))
            W = self.weights_.copy()
            W[0] = 0
            if self.regularization == "l2":
                loss += self.lambda_ / 2 * np.sum(W * W)
                grad = Xb.T @ (p - Y) / len(X) + self.lambda_ * W
            elif self.regularization == "l1":
                loss += self.lambda_ * np.sum(np.abs(W))
                grad = Xb.T @ (p - Y) / len(X) + self.lambda_ * np.sign(W)
            elif self.regularization in ("elasticnet", "elastic_net"):
                loss += self.lambda_ * (
                    self.l1_ratio * np.sum(np.abs(W))
                    + (1 - self.l1_ratio) / 2 * np.sum(W * W)
                )
                grad = Xb.T @ (p - Y) / len(X) + self.lambda_ * (
                    self.l1_ratio * np.sign(W) + (1 - self.l1_ratio) * W
                )
            else:
                grad = Xb.T @ (p - Y) / len(X)
            self.weights_ -= self.learning_rate * grad
            self.loss_history_.append(float(loss))

        self.log_uniconilear = float(self.loss_history_[-1])
        return self

    def predict_proba(self, X):
        return self._softmax(
            np.c_[np.ones(len(X)), np.asarray(X, dtype=float)] @ self.weights_
        )

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
