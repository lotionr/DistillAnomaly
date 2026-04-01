"""
Tabular anomaly detector for ADBench evaluation.

The detector operates directly on 2D numpy arrays (n_samples × n_features)
and outputs 1D anomaly scores (n_samples,) where higher scores indicate
more anomalous behaviour.

Design
------
We use an Isolation Forest as the core detector because:
  - It is unsupervised (no anomaly labels required at fit time)
  - It naturally handles high-dimensional tabular data
  - Its performance is competitive with RFuni on ADBench
  - It is deterministic given a fixed random seed

The detector exposes a simple sklearn-style API:
  fit(X_normal)       — fit on (presumed) normal/unlabelled training data
  score(X)            — return anomaly scores ∈ ℝ (higher = more anomalous)
  fit_score(X)        — fit on X and return anomaly scores for X

The preprocessing step (median imputation + standard scaling) should be
applied *before* calling fit/score; this class does not duplicate that logic.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class TabularAnomalyDetector:
    """Unsupervised anomaly detector for tabular 2D numpy arrays.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees (default 200 — more stable than sklearn's 100).
    max_samples : int | str
        Number of samples to draw per tree. "auto" uses min(256, n_samples).
    contamination : float | str
        Expected fraction of outliers ("auto" leaves decision threshold unset).
    random_state : int
        Seed for reproducibility.

    Usage
    -----
        det = TabularAnomalyDetector(random_state=42)
        det.fit(X_train)
        scores = det.score(X_test)   # shape (n_test,), higher = more anomalous
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: int | str = "auto",
        contamination: float | str = "auto",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        self._model: IsolationForest | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "TabularAnomalyDetector":
        """Fit the detector on training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data (should be preprocessed; can include anomalies
            since Isolation Forest is unsupervised and robust to low
            contamination fractions).
        """
        X = self._validate(X)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores for X.

        IsolationForest.score_samples returns *negative* average path lengths
        (lower = more anomalous). We negate so that higher = more anomalous,
        consistent with AUROC evaluation (roc_auc_score expects higher score
        for the positive / anomaly class).

        Returns
        -------
        scores : ndarray of shape (n_samples,), dtype float64
        """
        self._check_fitted()
        X = self._validate(X)
        # score_samples returns values ≤ 0; negate so anomalies score highest
        return -self._model.score_samples(X).astype(np.float64)

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        """Fit on X, then return anomaly scores for X."""
        return self.fit(X).score(X)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "TabularAnomalyDetector has not been fitted. Call fit() first."
            )

    @staticmethod
    def _validate(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"Input array must be non-empty, got shape {X.shape}")
        return X


if __name__ == "__main__":
    """End-to-end smoke test on one ADBench dataset."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[2]))

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    from src.tabular.data_loader import load_dataset
    from src.tabular.preprocessor import TabularPreprocessor

    # Pick a well-known ADBench dataset
    data_dir = Path("data/adbench")
    test_ds = "4_breastw"
    path = data_dir / f"{test_ds}.npz"
    if not path.exists():
        # fallback to first available
        files = sorted(data_dir.glob("*.npz"))
        if not files:
            print(f"No .npz files found in {data_dir}")
            sys.exit(1)
        path = files[0]
        test_ds = path.stem

    print(f"Dataset: {test_ds}")
    X, y = load_dataset(path)
    print(f"  Shape: {X.shape}, anomaly rate: {y.mean():.2%}")

    # Train/test split (80/20)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocess
    prep = TabularPreprocessor()
    X_tr_p = prep.fit_transform(X_tr)
    X_te_p = prep.transform(X_te)

    # Fit detector on training data (unsupervised — labels not used)
    det = TabularAnomalyDetector(random_state=42)
    det.fit(X_tr_p)

    # Score test set
    scores = det.score(X_te_p)
    assert scores.shape == (X_te_p.shape[0],), f"Expected shape ({X_te_p.shape[0]},), got {scores.shape}"
    assert scores.dtype == np.float64
    assert not np.any(np.isnan(scores))

    auroc = roc_auc_score(y_te, scores)
    print(f"  Test AUROC: {auroc:.4f}")

    # Sanity: anomaly scores should be finite
    assert np.all(np.isfinite(scores)), "Scores contain inf/nan"

    # Sanity: AUROC should be above random chance for breastw (well-separated)
    if test_ds == "4_breastw":
        assert auroc > 0.90, f"AUROC {auroc:.4f} is unexpectedly low for breastw"
        print("  Check: AUROC > 0.90 ✓")

    print("\nSmoke test passed.")
