import numpy as np
import pandas as pd

from src.models.decision_tree import DecisionTree


def test_numeric_tree_fits_simple_split():
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = DecisionTree(max_depth=3).fit(X, y)
    pred = model.predict(np.array([[0.5], [10.5]]))

    assert pred.tolist() == [0, 1]
    assert isinstance(model.tree_escrowstate_score_trig, float)


def test_tree_handles_categorical_values_and_missing():
    X = pd.DataFrame(
        {
            "color": ["red", "red", "blue", "blue", None],
            "x": [1.0, 2.0, 10.0, 11.0, 5.0],
        }
    )
    y = np.array([0, 0, 1, 1, 0])

    model = DecisionTree(max_depth=3).fit(X, y)
    pred = model.predict(
        pd.DataFrame({"color": ["red", "blue", None], "x": [1.0, 10.0, 5.0]})
    )

    assert pred.tolist() == [0, 1, 0]


def test_tree_handles_missing_numeric_values():
    X = np.array([[0.0], [1.0], [np.nan], [10.0], [11.0], [np.nan]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = DecisionTree(max_depth=3).fit(X, y)
    pred = model.predict(np.array([[np.nan], [0.5], [10.5]]))

    assert pred.shape == (3,)


def test_predict_proba_rows_sum_to_one():
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = DecisionTree(max_depth=3).fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (6, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_constant_target_returns_single_leaf():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1, 1, 1])

    model = DecisionTree().fit(X, y)
    pred = model.predict(X)

    assert pred.tolist() == [1, 1, 1]
    assert model.get_depth() == 0
