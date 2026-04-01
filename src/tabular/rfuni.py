"""
RFuni baseline from:
  "Explainable Unsupervised Anomaly Detection with Random Forest"
  Harvey et al., BlackRock / Prospect33, arXiv 2504.16075

Algorithm
---------
1. Given unlabelled real data X (n × p), draw uniform synthetic data X_u
   with the same shape by sampling each feature independently from
   Uniform(min_j, max_j).

2. Create a combined dataset:
     Z = [X ; X_u],  labels t = [1,...,1, 0,...,0]  (real=1, synth=0)

3. Train a Random Forest classifier on (Z, t).

4. For each real sample x_i, retrieve its leaf node in every tree.
   Compute the "GAP distance": the fraction of synthetic (uniform) points
   that land in the same leaf — this is the local density estimate.
   Points in sparse leaves (few synthetic neighbours) are anomalous.

   Concretely, RFC.apply(X) gives the leaf indices (n × n_trees).
   We use the out-of-bag predicted probability of being synthetic
   (class 0) as the anomaly score — equivalently, 1 minus the probability
   of being real.

   Per the paper the score is:
     score(x) = 1 - P(real | x)   [higher → more anomalous]

   where P(real | x) is the RF posterior from step 3.

5. The paper also mentions using the median distance within the central
   50% of the normal population as a threshold, but for AUROC evaluation
   we just use the raw score directly.

Usage
-----
    from src.tabular.rfuni import RFuniDetector

    det = RFuniDetector(random_state=42)
    det.fit(X_train)            # fit on (presumed) unlabelled data
    scores = det.score(X_test)  # 1D float64, higher = more anomalous

Run as script for full ADBench evaluation:
    python src/tabular/rfuni.py [--data-dir PATH] [--results-dir PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))

from src.tabular.data_loader import load_dataset
from src.tabular.metrics import compute_auroc
from src.tabular.preprocessor import TabularPreprocessor

log = logging.getLogger(__name__)


class RFuniDetector:
    """RFuni anomaly detector (Harvey et al., arXiv 2504.16075).

    Trains a Random Forest to separate real data from uniform synthetic data,
    then uses 1 − P(real) as the anomaly score.

    Parameters
    ----------
    n_estimators : int   — number of trees (default 100, matching sklearn default)
    n_synthetic  : float — ratio of synthetic to real samples (default 1.0 = equal)
    random_state : int   — reproducibility seed
    """

    def __init__(
        self,
        n_estimators: int = 100,
        n_synthetic: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_synthetic = n_synthetic
        self.random_state = random_state

        self._rf: RandomForestClassifier | None = None
        self._col_min: np.ndarray | None = None
        self._col_max: np.ndarray | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "RFuniDetector":
        """Fit RFuni on unlabelled (real) training data X.

        Steps:
          1. Record per-feature [min, max] from X.
          2. Draw uniform synthetic data X_u from those ranges.
          3. Train RF classifier: real (1) vs synthetic (0).
        """
        X = self._validate(X)
        n, p = X.shape
        rng = np.random.default_rng(self.random_state)

        # Store feature ranges for scoring new samples
        self._col_min = X.min(axis=0)
        self._col_max = X.max(axis=0)

        # Generate uniform synthetic samples
        n_synth = max(1, int(n * self.n_synthetic))
        col_range = self._col_max - self._col_min
        # Where range=0 (constant columns), synthetic = same constant value
        X_u = rng.uniform(size=(n_synth, p)) * col_range + self._col_min

        # Combined dataset: real=1, synthetic=0
        Z = np.vstack([X, X_u])
        t = np.concatenate([np.ones(n, dtype=int), np.zeros(n_synth, dtype=int)])

        self._rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._rf.fit(Z, t)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores for X (higher = more anomalous).

        Score = 1 − P(class=1 | x)  where class 1 = real data.
        Anomalies score high because the RF thinks they look synthetic.
        """
        self._check_fitted()
        X = self._validate(X)
        # predict_proba returns [P(synth), P(real)]; class order from _rf.classes_
        proba = self._rf.predict_proba(X)
        real_idx = list(self._rf.classes_).index(1)
        p_real = proba[:, real_idx].astype(np.float64)
        return 1.0 - p_real

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).score(X)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("RFuniDetector has not been fitted. Call fit() first.")

    @staticmethod
    def _validate(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"Expected non-empty 2D array, got shape {X.shape}")
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Stand-alone evaluation script
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


def evaluate_one_rfuni(name: str, path: Path, test_size: float, seed: int) -> dict:
    result = {
        "dataset": name,
        "n_samples": None,
        "n_features": None,
        "anomaly_ratio": None,
        "rfuni_auroc": None,
        "error": None,
        "elapsed_s": None,
    }
    t0 = time.perf_counter()
    try:
        X, y = load_dataset(path)
        result["n_samples"] = int(X.shape[0])
        result["n_features"] = int(X.shape[1])
        result["anomaly_ratio"] = float(y.mean())

        if y.sum() == 0:
            raise ValueError("No anomalies")
        if (y == 0).sum() == 0:
            raise ValueError("No normal samples")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        prep = TabularPreprocessor()
        X_tr_p = prep.fit_transform(X_tr)
        X_te_p = prep.transform(X_te)

        det = RFuniDetector(random_state=seed)
        det.fit(X_tr_p)
        scores = det.score(X_te_p)

        auroc = compute_auroc(y_te, scores)
        if np.isnan(auroc):
            raise ValueError("AUROC is NaN")

        result["rfuni_auroc"] = round(float(auroc), 6)
        log.info(
            f"[{name}] n={X.shape[0]} f={X.shape[1]} "
            f"anom={y.mean():.2%} RFuni AUROC={auroc:.4f}"
        )
    except Exception as exc:
        result["error"] = str(exc)
        log.warning(f"[{name}] FAILED — {exc}")

    result["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return result


def main():
    p = argparse.ArgumentParser(description="RFuni baseline on ADBench")
    p.add_argument("--data-dir", type=Path, default=_ROOT / "data" / "adbench")
    p.add_argument("--results-dir", type=Path, default=_ROOT / "results")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", type=str, default=None)
    args = p.parse_args()

    data_dir = args.data_dir
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("*.npz"))
    if args.dataset:
        npz_files = [f for f in npz_files if f.stem == args.dataset]
        if not npz_files:
            log.error(f"Dataset '{args.dataset}' not found in {data_dir}")
            sys.exit(1)

    log.info(f"RFuni evaluation: {len(npz_files)} dataset(s)")

    rows = [evaluate_one_rfuni(fp.stem, fp, args.test_size, args.seed) for fp in npz_files]
    df = pd.DataFrame(rows)

    out_path = results_dir / "rfuni_baseline.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved to {out_path}")

    # Summary
    ok = df[df["error"].isna()]
    fail = df[df["error"].notna()]
    W = 36
    print()
    print("=" * (W + 42))
    print(f"  {'Dataset':<{W}} {'Samples':>8}  {'Feat':>5}  {'Anom%':>6}  {'RFuni':>8}")
    print("-" * (W + 42))
    for _, row in ok.sort_values("rfuni_auroc", ascending=False).iterrows():
        print(
            f"  {row['dataset']:<{W}} {int(row['n_samples']):>8}  "
            f"{int(row['n_features']):>5}  {100*row['anomaly_ratio']:>5.1f}%  "
            f"{row['rfuni_auroc']:>8.4f}"
        )
    print("=" * (W + 42))
    if len(ok) > 0:
        print(f"  Mean RFuni AUROC : {ok['rfuni_auroc'].mean():.4f}")
    print(f"  Succeeded: {len(ok)} / {len(df)}")
    if len(fail):
        print(f"  Failed:    {len(fail)}")
        for _, row in fail.iterrows():
            print(f"    - {row['dataset']}: {row['error']}")
    print()


if __name__ == "__main__":
    main()
