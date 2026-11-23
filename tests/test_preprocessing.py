from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import DataPreprocessor, ID_COL, TARGET


def _raw_frame(n: int = 96, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1] * (n // 2))
    rng.shuffle(labels)

    frame = pd.DataFrame(
        {
            ID_COL: [f"id-{i:03d}" for i in range(n)],
            "num_adults": rng.integers(1, 4, n).astype(float),
            "num_children": rng.integers(0, 3, n).astype(float),
            "weekend_nights": rng.integers(0, 3, n).astype(float),
            "weekday_nights": rng.integers(0, 4, n).astype(float),
            "needs_parking": rng.integers(0, 2, n).astype(float),
            "days_before_arrival": rng.integers(0, 400, n).astype(float),
            "arrival_year": rng.integers(2015, 2018, n).astype(float),
            "arrival_month": rng.integers(1, 13, n).astype(float),
            "arrival_day": rng.integers(1, 29, n).astype(float),
            "is_returning_guest": rng.integers(0, 2, n).astype(float),
            "prior_cancellations": rng.integers(0, 4, n).astype(float),
            "prior_completed_bookings": rng.integers(0, 6, n).astype(float),
            "nightly_rate_idr": rng.integers(500_000, 2_000_000, n).astype(float),
            "special_requests": rng.integers(0, 4, n).astype(float),
            "meal_package": rng.choice(["BB", "HB", "FB", None], n),
            "room_category": rng.choice(["A", "B", "C", "D"], n),
            "booking_channel": rng.choice(["web", "agent", "phone"], n),
            TARGET: labels,
        }
    )

    # Latih jalur cleaning dan flag missing.
    frame.loc[[0, 5], "num_adults"] = np.nan
    frame.loc[3, "num_adults"] = 0
    frame.loc[[2, 7], ["weekend_nights", "weekday_nights"]] = 0
    frame.loc[9, "nightly_rate_idr"] = 0
    frame.loc[[4, 8], "days_before_arrival"] = np.nan
    frame.loc[[6, 11], "special_requests"] = np.nan
    frame.loc[[1, 10], "meal_package"] = None
    return frame


@pytest.fixture()
def prepared():
    frame = _raw_frame()
    prep = DataPreprocessor(seed=42, select_k=5)
    train, val = prep.split(frame)
    X_train, y_train = prep.fit_transform(train)
    X_val = prep.transform(val)
    return prep, train, val, X_train, y_train, X_val


def test_split_is_stratified_and_disjoint(prepared):
    _, train, val, _, _, _ = prepared

    assert len(train) + len(val) == 96
    assert set(train[ID_COL]).isdisjoint(set(val[ID_COL]))
    assert train[TARGET].mean() == pytest.approx(val[TARGET].mean(), abs=0.1)


def test_fit_transform_produces_clean_aligned_matrix(prepared):
    prep, train, _, X_train, y_train, _ = prepared

    assert X_train.shape[0] == len(train) == len(y_train)
    assert X_train.dtype == np.float64
    assert np.isfinite(X_train).all()
    assert X_train.shape[1] == prep.select_k == 5
    assert len(prep.feature_names_out_) == X_train.shape[1]
    assert "nightly_rate_per_guest" in prep.feature_names_in_
    assert "num_adults_is_missing" in prep.feature_names_in_


def test_transform_validation_and_unknown_category(prepared):
    prep, _, val, X_train, _, X_val = prepared

    assert X_val.shape == (len(val), X_train.shape[1])
    assert np.isfinite(X_val).all()

    unseen = val.copy()
    unseen["meal_package"] = "never-seen"
    assert prep.transform(unseen).shape == X_val.shape


def test_fit_is_deterministic_and_pickle_roundtrip(tmp_path, prepared):
    prep, train, val, X_train, _, X_val = prepared

    again = DataPreprocessor(seed=42, select_k=5).fit(train)
    np.testing.assert_allclose(again.transform(val), X_val)

    path = tmp_path / "preprocessor.pkl"
    prep.save(path)
    loaded = DataPreprocessor.load(path)
    np.testing.assert_allclose(loaded.transform(val), X_val)
    assert loaded.feature_names_out_ == prep.feature_names_out_


def test_summary_lists_selected_features(prepared):
    prep, _, _, X_train, _, _ = prepared
    summary = prep.summary()

    assert len(summary) == X_train.shape[1]
    assert summary["mutual_information"].is_monotonic_decreasing


def test_pca_option_changes_feature_names():
    frame = _raw_frame()
    prep = DataPreprocessor(seed=42, select_k=8, use_pca=True, pca_variance=0.9)
    train, _ = prep.split(frame)
    X_train, _ = prep.fit_transform(train)

    assert X_train.shape[1] == prep.pca_.n_components_
    assert all(name.startswith("pc_") for name in prep.feature_names_out_)


def test_fit_requires_target_column():
    frame = _raw_frame().drop(columns=[TARGET])
    with pytest.raises(ValueError, match="target column"):
        DataPreprocessor().fit(frame)
