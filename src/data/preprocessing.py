"""Data preparation for Tugas Besar 2 IF3170.

Owner: Anggota 1 (Data dan Integrasi).

The class `DataPreprocessor` is the single entry point every model uses:

    prep = DataPreprocessor()
    train_set, val_set = prep.split(train_df)
    X_train, y_train = prep.fit_transform(train_set)
    X_val = prep.transform(val_set)
    X_test = prep.transform(test_df)

Every statistic (imputer, encoder, scaler, outlier bounds, selector, PCA) is
fitted on the training split only. `transform` never refits.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "is_canceled"
ID_COL = "id"

NUMERIC_COLS = [
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "needs_parking",
    "days_before_arrival",
    "arrival_year",
    "arrival_month",
    "arrival_day",
    "is_returning_guest",
    "prior_cancellations",
    "prior_completed_bookings",
    "nightly_rate_idr",
    "special_requests",
]

CATEGORICAL_COLS = ["meal_package", "room_category", "booking_channel"]

# Columns that carry informative missingness (verified in EDA).
MISSING_FLAG_COLS = [
    "num_adults",
    "num_children",
    "meal_package",
    "room_category",
    "days_before_arrival",
    "needs_parking",
    "nightly_rate_idr",
    "special_requests",
]


class DataPreprocessor:
    """Fitted preprocessing pipeline for the hotel dataset."""

    def __init__(
        self,
        seed: int = 42,
        val_size: float = 0.2,
        clip_outliers: bool = True,
        clip_quantiles: tuple[float, float] = (0.01, 0.99),
        select_k: int | None = 38,
        use_pca: bool = False,
        pca_variance: float = 0.95,
        mi_sample: int = 6000,
    ) -> None:
        self.seed = seed
        self.val_size = val_size
        self.clip_outliers = clip_outliers
        self.clip_quantiles = clip_quantiles
        self.select_k = select_k
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.mi_sample = mi_sample

        self.numeric_imputer_: SimpleImputer | None = None
        self.categorical_imputer_: SimpleImputer | None = None
        self.encoder_: OneHotEncoder | None = None
        self.scaler_: StandardScaler | None = None
        self.variance_: VarianceThreshold | None = None
        self.selector_mask_: np.ndarray | None = None
        self.pca_: PCA | None = None

        self.clip_bounds_: dict[str, tuple[float, float]] = {}
        self.feature_names_in_: list[str] = []
        self.feature_names_out_: list[str] = []
        self.mi_scores_: np.ndarray | None = None
        self.n_features_out_: int = 0
        self.continuous_cols_: list[str] = []
        self.binary_cols_: list[str] = []
        self.onehot_cols_: list[str] = []

    # ------------------------------------------------------------------ split
    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Stratified train/validation split. Call this before any fitting."""
        train_set, val_set = train_test_split(
            df,
            test_size=self.val_size,
            random_state=self.seed,
            stratify=df[TARGET],
        )
        return train_set.reset_index(drop=True), val_set.reset_index(drop=True)

    # ------------------------------------------------------------- cleaning
    def _clean(self, df: pd.DataFrame, drop_duplicates: bool = False) -> pd.DataFrame:
        """Fix impossible values. Duplicate dropping is training-only so that
        `transform` never changes the number of rows."""
        out = df.copy()

        # 0 adults is impossible; 105 of 106 rows have children.
        out.loc[out["num_adults"] == 0, "num_adults"] = np.nan

        # A booking with zero nights is meaningless.
        zero_nights = (out["weekend_nights"] + out["weekday_nights"]) == 0
        out.loc[zero_nights, ["weekend_nights", "weekday_nights"]] = np.nan

        # A zero nightly rate is a data error, not a free room.
        out.loc[out["nightly_rate_idr"] == 0, "nightly_rate_idr"] = np.nan

        if drop_duplicates:
            feature_cols = [c for c in out.columns if c not in (ID_COL, TARGET)]
            out = out.drop_duplicates(subset=feature_cols, keep="first")

        return out.reset_index(drop=True)

    # ------------------------------------------------------------- construct
    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derived attributes. Must stay row-wise so train/test behave alike."""
        out = df.copy()

        stay_nights = (out["weekend_nights"].fillna(0) + out["weekday_nights"].fillna(0)).replace(0, np.nan)
        guest_count = (out["num_adults"].fillna(0) + out["num_children"].fillna(0)).replace(0, np.nan)

        out["has_children"] = (out["num_children"].fillna(0) > 0).astype(int)
        out["is_family"] = (out["num_children"].fillna(0) > 0).astype(int)
        out["arrival_quarter"] = ((out["arrival_month"] - 1) // 3 + 1).astype(int)
        out["is_high_season"] = out["arrival_month"].isin([6, 7, 8, 12]).astype(int)
        out["days_before_arrival_log"] = np.log1p(out["days_before_arrival"].clip(lower=0))

        out["nightly_rate_per_guest"] = out["nightly_rate_idr"] / guest_count
        out["total_room_cost"] = out["nightly_rate_idr"] * stay_nights
        out["cost_per_guest"] = out["total_room_cost"] / guest_count

        prior_bookings = (out["prior_cancellations"] + out["prior_completed_bookings"]).replace(0, np.nan)
        out["prior_cancel_ratio"] = out["prior_cancellations"] / prior_bookings
        out["has_prior_history"] = (
            (out["prior_cancellations"].fillna(0) > 0)
            | (out["prior_completed_bookings"].fillna(0) > 0)
        ).astype(int)
        out["requests_per_guest"] = out["special_requests"] / guest_count

        # Missingness indicators, computed before imputation.
        for col in MISSING_FLAG_COLS:
            if col in out.columns:
                out[f"{col}_is_missing"] = out[col].isna().astype(int)

        return out

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        if TARGET not in df.columns:
            raise ValueError(f"target column '{TARGET}' not found")

        self.clip_bounds_ = {}
        clean = self._engineer(self._clean(df, drop_duplicates=True))
        y = clean[TARGET].to_numpy()

        feature_df = clean.drop(columns=[c for c in (ID_COL, TARGET) if c in clean.columns])
        self.feature_names_in_ = list(feature_df.columns)
        self.n_features_in_ = len(self.feature_names_in_)

        num_cols = [c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])]
        cat_cols = [c for c in CATEGORICAL_COLS if c in feature_df.columns]

        # Binary / flag columns stay unscaled.
        self.binary_cols_ = [
            c for c in num_cols if feature_df[c].dropna().isin([0, 1]).all()
        ]
        self.continuous_cols_ = [c for c in num_cols if c not in self.binary_cols_]

        # Outlier bounds from train only.
        if self.clip_outliers:
            for c in self.continuous_cols_:
                lo, hi = feature_df[c].quantile(self.clip_quantiles).tolist()
                if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    self.clip_bounds_[c] = (float(lo), float(hi))

        clipped = self._clip(feature_df)

        self.numeric_imputer_ = SimpleImputer(strategy="median")
        num_matrix = self.numeric_imputer_.fit_transform(clipped[self.continuous_cols_])

        self.categorical_imputer_ = SimpleImputer(strategy="most_frequent")
        cat_matrix = (
            self.categorical_imputer_.fit_transform(clipped[cat_cols].astype(object))
            if cat_cols
            else np.empty((len(clipped), 0), dtype=object)
        )

        self.encoder_ = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, min_frequency=0.01, dtype=np.float64
        )
        self.onehot_cols_ = (
            list(self.encoder_.fit(cat_matrix).get_feature_names_out(cat_cols)) if cat_cols else []
        )
        onehot_matrix = self.encoder_.transform(cat_matrix) if cat_cols else np.empty((len(clipped), 0))

        binary_matrix = clipped[self.binary_cols_].fillna(0).to_numpy(dtype=np.float64)

        X = np.hstack([num_matrix, onehot_matrix, binary_matrix])
        names = list(self.continuous_cols_) + self.onehot_cols_ + list(self.binary_cols_)

        # Standardize continuous columns only; one-hot and flags are left as is.
        self.scaler_ = StandardScaler()
        X[:, : len(self.continuous_cols_)] = self.scaler_.fit_transform(X[:, : len(self.continuous_cols_)])

        # Feature selection: variance floor, then mutual information top-k.
        self.variance_ = VarianceThreshold(threshold=1e-8)
        X = self.variance_.fit_transform(X)
        names = [n for n, keep in zip(names, self.variance_.get_support()) if keep]

        self.selector_mask_ = np.ones(X.shape[1], dtype=bool)
        self.mi_scores_ = self._mutual_information(X, y)
        if self.select_k is not None and self.select_k < X.shape[1]:
            order = np.argsort(self.mi_scores_)[::-1][: self.select_k]
            self.selector_mask_ = np.zeros(X.shape[1], dtype=bool)
            self.selector_mask_[order] = True
        X = X[:, self.selector_mask_]
        names = [n for n, keep in zip(names, self.selector_mask_) if keep]

        if self.use_pca:
            self.pca_ = PCA(n_components=self.pca_variance, random_state=self.seed)
            X = self.pca_.fit_transform(X)
            names = [f"pc_{i + 1}" for i in range(X.shape[1])]

        self.feature_names_out_ = names
        self.n_features_out_ = X.shape[1]
        return self

    def _mutual_information(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """MI scores for reporting; computed on a subsample for speed."""
        n = X.shape[0]
        if n > self.mi_sample:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n, size=self.mi_sample, replace=False)
            Xs, ys = X[idx], y[idx]
        else:
            Xs, ys = X, y
        return mutual_info_classif(Xs, ys, random_state=self.seed, n_neighbors=3)

    # ------------------------------------------------------------ transform
    def _clip(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, (lo, hi) in self.clip_bounds_.items():
            out[col] = out[col].clip(lo, hi)
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        clean = self._engineer(self._clean(df))
        feature_df = clean.drop(columns=[c for c in (ID_COL, TARGET) if c in clean.columns])

        missing = [c for c in self.feature_names_in_ if c not in feature_df.columns]
        if missing:
            raise ValueError(f"missing columns at transform time: {missing}")
        feature_df = feature_df[self.feature_names_in_]

        clipped = self._clip(feature_df)
        num_matrix = self.numeric_imputer_.transform(clipped[self.continuous_cols_])

        cat_cols = [c for c in CATEGORICAL_COLS if c in clipped.columns]
        cat_matrix = (
            self.categorical_imputer_.transform(clipped[cat_cols].astype(object))
            if cat_cols
            else np.empty((len(clipped), 0), dtype=object)
        )
        onehot_matrix = self.encoder_.transform(cat_matrix) if cat_cols else np.empty((len(clipped), 0))
        binary_matrix = clipped[self.binary_cols_].fillna(0).to_numpy(dtype=np.float64)

        X = np.hstack([num_matrix, onehot_matrix, binary_matrix])
        X[:, : len(self.continuous_cols_)] = self.scaler_.transform(X[:, : len(self.continuous_cols_)])
        X = self.variance_.transform(X)
        X = X[:, self.selector_mask_]
        if self.pca_ is not None:
            X = self.pca_.transform(X)
        return np.asarray(X, dtype=np.float64)

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self.fit(df)
        return self.transform(df), df[TARGET].to_numpy()

    # ---------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "DataPreprocessor":
        return joblib.load(path)

    def summary(self) -> pd.DataFrame:
        if self.mi_scores_ is None or self.use_pca:
            return pd.DataFrame({"feature": self.feature_names_out_})
        return pd.DataFrame(
            {
                "feature": self.feature_names_out_,
                "mutual_information": self.mi_scores_[self.selector_mask_],
            }
        ).sort_values("mutual_information", ascending=False, na_position="last")
