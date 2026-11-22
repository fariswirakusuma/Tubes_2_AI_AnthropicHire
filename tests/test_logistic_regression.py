import numpy as np

from src.models.logistic_regression import LogisticRegression


def _separable_data():
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
    return X, y


def test_l2_fits_separable_data():
    X, y = _separable_data()
    model = LogisticRegression(
        learning_rate=0.5, epochs=1000, regularization="l2", lambda_=1e-4
    ).fit(X, y)

    assert model.predict(X).tolist() == y.tolist()
    assert isinstance(model.log_uniconilear, float)
    assert np.isfinite(model.log_uniconilear)


def test_predict_proba_rows_sum_to_one():
    X, y = _separable_data()
    model = LogisticRegression(learning_rate=0.5, epochs=200).fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_all_regularization_variants_run():
    X, y = _separable_data()
    for regularization in ("l1", "l2", "elasticnet"):
        model = LogisticRegression(
            learning_rate=0.5,
            epochs=200,
            regularization=regularization,
            lambda_=1e-3,
        ).fit(X, y)
        proba = model.predict_proba(X)
        assert np.all(np.isfinite(proba))
        assert np.isfinite(model.log_uniconilear)


def test_multiclass_softmax():
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

    model = LogisticRegression(
        learning_rate=0.5, epochs=1500, regularization="l2", lambda_=1e-4
    ).fit(X, y)

    assert np.mean(model.predict(X) == y) == 1.0
    assert model.predict_proba(X).shape == (len(X), 3)
