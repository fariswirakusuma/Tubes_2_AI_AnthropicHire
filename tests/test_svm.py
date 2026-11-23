import numpy as np
import pytest

from src.models.svm import SVM


def test_linear_kernel_separates_two_classes():
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -2.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = SVM(kernel="linear", C=1.0).fit(X, y)

    assert np.mean(model.predict(X) == y) == 1.0


def test_gaussian_kernel_handles_non_linear_data():
    rng = np.random.default_rng(42)
    inner = rng.normal(scale=0.3, size=(12, 2))
    outer = rng.normal(scale=0.3, size=(12, 2)) + np.array([2.0, 0.0])
    X = np.vstack([inner, outer])
    y = np.array([0] * 12 + [1] * 12)

    model = SVM(kernel="gaussian", gamma=1.0, C=10.0).fit(X, y)

    assert np.mean(model.predict(X) == y) >= 0.9


def test_polynomial_kernel_handles_xor():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])

    model = SVM(kernel="poly", degree=2, gamma=1.0, coef0=1.0, C=10.0).fit(X, y)

    assert np.mean(model.predict(X) == y) >= 0.9


def test_multiclass_uses_one_vs_one():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [10.0, 0.0],
            [10.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1, 2, 2])

    model = SVM(kernel="linear", C=1.0).fit(X, y)

    assert np.mean(model.predict(X) == y) == 1.0


def test_linear_kernel_handles_noisy_data():
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=(-1.0, -1.0), scale=1.0, size=(100, 2))
    X1 = rng.normal(loc=(1.0, 1.0), scale=1.0, size=(100, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)

    model = SVM(kernel="linear", C=1.0, learning_rate=0.01, epochs=200).fit(X, y)

    assert np.mean(model.predict(X) == y) >= 0.85


def test_predict_proba_rows_sum_to_one():
    X = np.array(
        [
            [-2.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -2.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = SVM(kernel="linear", C=1.0).fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_invalid_kernel_rejected_at_init():
    with pytest.raises(ValueError, match="kernel"):
        SVM(kernel="sigmoid")


def test_polynomial_alias_matches_poly():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])

    poly = SVM(kernel="poly", degree=2, gamma=1.0, coef0=1.0, C=10.0).fit(X, y)
    polynomial = SVM(
        kernel="polynomial", degree=2, gamma=1.0, coef0=1.0, C=10.0
    ).fit(X, y)

    np.testing.assert_array_equal(poly.predict(X), polynomial.predict(X))


def test_gamma_auto_and_decision_function():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0], [5.0, 6.0]])
    y = np.array([0, 0, 1, 1])

    model = SVM(kernel="gaussian", gamma="auto", C=10.0).fit(X, y)
    scores = model.decision_function(X)

    assert model.gamma_ == pytest.approx(1.0 / X.shape[1])
    assert scores.shape == (len(X),)
    assert np.all(np.isfinite(scores))


def test_multiclass_predict_proba_rows_sum_to_one():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [10.0, 0.0],
            [10.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1, 2, 2])

    model = SVM(kernel="linear", C=1.0).fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
