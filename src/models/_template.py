"""Contract + starter for a from-scratch model.

Copy this file, rename the class, and keep the three methods.
Only NumPy is allowed for the scratch implementation.

    import numpy as np

    class KNN:
        def fit(self, X, y) -> "KNN": ...
        def predict(self, X) -> np.ndarray: ...
        def predict_proba(self, X) -> np.ndarray: ...
"""

from __future__ import annotations

import numpy as np


class TemplateModel:
    """Replace with the real model. Keep the method signatures."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TemplateModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        # TODO: train
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        # TODO: return shape (n_samples, n_classes), rows sum to 1
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
