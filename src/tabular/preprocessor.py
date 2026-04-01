"""
Tabular data preprocessor for ADBench anomaly detection datasets.

All ADBench Classical datasets are already numerical float64 arrays.
The preprocessor handles:
  - Missing value imputation (median strategy, fit on train only)
  - Standard scaling (zero mean, unit variance, fit on train only)
  - Constant-column removal (columns with zero variance are dropped)

Usage
-----
    from src.tabular.preprocessor import TabularPreprocessor

    prep = TabularPreprocessor()
    X_train = prep.fit_transform(X_train)
    X_test  = prep.transform(X_test)
"""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TabularPreprocessor:
    """Fit-once preprocessing pipeline for tabular anomaly detection.

    Steps (applied in order):
      1. Median imputation for any NaN values.
      2. Drop constant columns (zero std after imputation).
      3. Standard scaling (mean=0, std=1).

    All fitting is done exclusively on training data to prevent leakage.
    """

    def __init__(self) -> None:
        self._imputer: SimpleImputer | None = None
        self._scaler: StandardScaler | None = None
        self._keep_cols: np.ndarray | None = None  # boolean mask of non-constant columns
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "TabularPreprocessor":
        """Fit preprocessor on training data X (shape: n_samples × n_features)."""
        X = np.asarray(X, dtype=np.float64)
        self._validate_input(X)

        # Step 1: fit imputer
        self._imputer = SimpleImputer(strategy="median")
        X_imp = self._imputer.fit_transform(X)

        # Step 2: identify non-constant columns
        col_std = X_imp.std(axis=0)
        self._keep_cols = col_std > 0
        if not self._keep_cols.any():
            raise ValueError("All columns are constant after imputation — cannot preprocess.")
        X_imp = X_imp[:, self._keep_cols]

        # Step 3: fit scaler
        self._scaler = StandardScaler()
        self._scaler.fit(X_imp)

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using the fitted preprocessor. Returns float64 ndarray."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        self._validate_input(X)

        X_imp = self._imputer.transform(X)
        X_imp = X_imp[:, self._keep_cols]
        return self._scaler.transform(X_imp)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return transformed X."""
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_features_in_(self) -> int:
        self._check_fitted()
        return len(self._keep_cols)

    @property
    def n_features_out_(self) -> int:
        self._check_fitted()
        return int(self._keep_cols.sum())

    @property
    def n_constant_cols_dropped(self) -> int:
        self._check_fitted()
        return int((~self._keep_cols).sum())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TabularPreprocessor has not been fitted yet. Call fit() first.")

    @staticmethod
    def _validate_input(X: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"Input array must be non-empty, got shape {X.shape}")


if __name__ == "__main__":
    """Smoke test: run preprocessor on all ADBench datasets."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.tabular.data_loader import load_all_datasets

    from sklearn.model_selection import train_test_split

    datasets = load_all_datasets(verbose=False)
    print(f"{'Dataset':<35} {'In':>5} {'Out':>5} {'Dropped':>8} {'TrainMean':>10} {'TrainStd':>9}")
    print("-" * 75)

    errors = []
    for name, (X, y) in sorted(datasets.items()):
        try:
            X_tr, X_te = train_test_split(X, test_size=0.2, random_state=42)
            prep = TabularPreprocessor()
            X_tr_p = prep.fit_transform(X_tr)
            X_te_p = prep.transform(X_te)

            # Sanity checks
            assert X_tr_p.dtype == np.float64
            assert X_tr_p.shape == (X_tr.shape[0], prep.n_features_out_)
            assert X_te_p.shape == (X_te.shape[0], prep.n_features_out_)
            assert not np.any(np.isnan(X_tr_p))
            assert not np.any(np.isnan(X_te_p))
            # Training set should be ~zero-mean, ~unit-variance
            assert abs(X_tr_p.mean()) < 0.01, f"Mean {X_tr_p.mean():.4f} not near 0"
            assert abs(X_tr_p.std() - 1.0) < 0.05, f"Std {X_tr_p.std():.4f} not near 1"

            print(
                f"{name:<35} {prep.n_features_in_:>5} {prep.n_features_out_:>5} "
                f"{prep.n_constant_cols_dropped:>8} {X_tr_p.mean():>10.4f} {X_tr_p.std():>9.4f}"
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            print(f"{name:<35} ERROR: {exc}")

    print(f"\nPassed: {len(datasets) - len(errors)}/{len(datasets)}")
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
