import numpy as np
import pytest

from src.models.knn import KNN


def test_k1_separates_two_clusters():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]])
    y = np.array([0, 0, 1, 1])

    model = KNN(k=1).fit(X, y)
    pred = model.predict(np.array([[0.2, 0.2], [9.8, 10.2]]))

    assert pred.tolist() == [0, 1]


def test_predict_proba_rows_sum_to_one():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]])
    y = np.array([0, 0, 1, 1])

    model = KNN(k=3).fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_default_distance_is_euclidean():
    X = np.array([[0.0, 0.0], [1.0, 2.0]])
    y = np.array([0, 1])

    model = KNN(k=1).fit(X, y)
    distances = model.pairwise_distances(np.array([[0.0, 0.0]]))

    expected = np.hypot(1.0, 2.0)
    assert model.p == 2
    assert distances.shape == (1, 2)
    assert distances[0, 1] == pytest.approx(expected)


def test_kneighbors_returns_sorted_distances_and_indices():
    X = np.array([[0.0], [1.0], [5.0], [10.0]])
    y = np.array([0, 0, 1, 1])

    model = KNN(k=2).fit(X, y)
    distances, indices = model.kneighbors(np.array([[0.2]]))

    assert distances.shape == (1, 2)
    assert np.all(np.diff(distances, axis=1) >= 0)
    assert indices.tolist() == [[0, 1]]


def test_minkowski_distance_supports_custom_order():
    X = np.array([[0.0, 0.0], [1.0, 2.0]])
    y = np.array([0, 1])

    manhattan = KNN(k=1, p=1).fit(X, y)
    assert manhattan.pairwise_distances(np.array([[0.0, 0.0]]))[0, 1] == pytest.approx(3.0)

    p3 = KNN(k=1, p=3).fit(X, y)
    assert p3.pairwise_distances(np.array([[0.0, 0.0]]))[0, 1] == pytest.approx(
        (1.0**3 + 2.0**3) ** (1.0 / 3.0)
    )


def test_invalid_weights_rejected():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    model = KNN(k=1, weights="majority").fit(X, y)
    with pytest.raises(ValueError, match="weights"):
        model.predict(X)


def test_k_larger_than_training_set_rejected():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    model = KNN(k=3).fit(X, y)
    with pytest.raises(ValueError, match="k cannot exceed"):
        model.predict(X)


def test_distance_weights_favor_exact_match():
    X = np.array([[0.0], [0.01], [10.0]])
    y = np.array([0, 1, 1])

    model = KNN(k=3, weights="distance").fit(X, y)
    proba = model.predict_proba(np.array([[0.0]]))

    assert proba[0, 0] > 0.99
