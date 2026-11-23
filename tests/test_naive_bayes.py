import numpy as np
import pytest

from src.models.naive_bayes import BayesianNetwork, NaiveBayes


def test_gaussian_nb_fits_two_classes():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [6.0, 5.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = NaiveBayes().fit(X, y)

    assert model.predict(X).tolist() == y.tolist()
    assert model.class_log_prior_.shape == (2,)
    assert np.all(np.isfinite(model.predict_log_proba(X)))


def test_predict_proba_rows_sum_to_one():
    X = np.array([[0.0], [1.0], [5.0], [6.0]])
    y = np.array([0, 0, 1, 1])

    model = NaiveBayes().fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_multiclass():
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

    model = NaiveBayes().fit(X, y)

    assert model.predict(X).tolist() == y.tolist()
    assert model.predict_proba(X).shape == (6, 3)


def test_constant_feature_does_not_divide_by_zero():
    X = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 5.0], [0.0, 6.0]])
    y = np.array([0, 0, 1, 1])

    model = NaiveBayes().fit(X, y)
    proba = model.predict_proba(X)

    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_bayesian_network_alias_is_available():
    assert issubclass(BayesianNetwork, NaiveBayes)


def test_balanced_class_weight_normalizes_priors():
    X = np.array([[0.0], [1.0], [5.0], [6.0], [7.0], [8.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = NaiveBayes(class_weight="balanced").fit(X, y)

    assert model.class_prior_.sum() == pytest.approx(1.0)
    assert np.allclose(np.exp(model.class_log_prior_), model.class_prior_)


def test_custom_class_prior_is_normalized():
    X = np.array([[0.0], [1.0], [5.0], [6.0]])
    y = np.array([0, 0, 1, 1])

    model = NaiveBayes(class_prior=np.array([1.0, 3.0])).fit(X, y)

    assert model.class_prior_.tolist() == pytest.approx([0.25, 0.75])
