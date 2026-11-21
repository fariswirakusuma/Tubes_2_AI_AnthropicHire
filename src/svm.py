"""SVM berbasis scikit-learn untuk perbandingan dengan implementasi from scratch.

Mendukung kernel linear, polynomial, dan gaussian (RBF) serta data multi-kelas.
Catatan: SVM sangat sensitif terhadap skala fitur, jadi standardisasi fitur
sebelum `fit` (lihat `StandardScaler` di pipeline notebook).
"""

import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

# Nama kernel tugas -> nama kernel sklearn
_KERNELS = {"linear": "linear", "polynomial": "poly", "poly": "poly", "gaussian": "rbf", "rbf": "rbf"}


class SVM(BaseEstimator, ClassifierMixin):

    def __init__(self, kernel="linear", C=1.0, degree=3, gamma="scale", coef0=0.0,
                 probability=False, class_weight=None, random_state=42):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.probability = probability
        self.class_weight = class_weight
        self.random_state = random_state

    def _sk_kernel(self):
        try:
            return _KERNELS[self.kernel]
        except KeyError:
            raise ValueError(f"kernel tidak dikenal: {self.kernel!r}") from None

    def fit(self, X, y):
        self.model_ = SVC(
            kernel=self._sk_kernel(),
            C=self.C,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            probability=self.probability,
            class_weight=self.class_weight,
            random_state=self.random_state,
        ).fit(X, y)
        self.classes_ = self.model_.classes_
        # support vector & margin, dipakai untuk plot bonus
        self.support_ = self.model_.support_
        self.support_vectors_ = self.model_.support_vectors_
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)

    def decision_function(self, X):
        check_is_fitted(self, "model_")
        return self.model_.decision_function(X)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
