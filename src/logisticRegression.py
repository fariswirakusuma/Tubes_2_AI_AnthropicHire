"""Alias lama untuk Logistic Regression from scratch.

Implementasi tunggal ada di `src.models.logistic_regression`; modul ini hanya
menjaga import lama (`from src.logisticRegression import LogisticRegression`)
tetap berjalan.
"""

from src.models.logistic_regression import LogisticRegression

__all__ = ["LogisticRegression"]
