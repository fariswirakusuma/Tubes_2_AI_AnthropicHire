import numpy as np

from src.models.decision_tree import DecisionTree
from src.models.knn import KNN
from src.models.logistic_regression import LogisticRegression
from src.models.naive_bayes import NaiveBayes
from src.models.svm import SVM


def _data():
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


def test_all_models_round_trip_through_save_and_load(tmp_path):
    X, y = _data()
    models = [
        KNN(k=3).fit(X, y),
        DecisionTree(max_depth=3).fit(X, y),
        LogisticRegression(learning_rate=0.5, epochs=200).fit(X, y),
        SVM(kernel="linear", C=1.0).fit(X, y),
        NaiveBayes().fit(X, y),
    ]

    for i, model in enumerate(models):
        path = tmp_path / f"model_{i}.pkl"
        model.save(path)
        loaded = type(model).load(path)
        assert np.array_equal(loaded.predict(X), model.predict(X))
