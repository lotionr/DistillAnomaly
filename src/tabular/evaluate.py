"""
Per-dataset evaluation loop for ADBench anomaly detection benchmark.

For each dataset:
  1. Load (X, y) from .npz file
  2. Split into train/test (80/20, stratified)
  3. Preprocess with TabularPreprocessor (fit on train only)
  4. Fit TabularAnomalyDetector on preprocessed train set
  5. Score the full dataset (train + test) or test set only
  6. Compute AUROC against ground-truth labels
  7. Log result and collect into a DataFrame

Results saved to: results/per_dataset_auroc.csv

Usage
-----
    python src/tabular/evaluate.py [--data-dir PATH] [--results-dir PATH]
                                   [--test-size FLOAT] [--seed INT]
                                   [--dataset NAME]   # single-dataset mode
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── project imports ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))

from src.tabular.data_loader import load_dataset, get_dataset_names
from src.tabular.detector import TabularAnomalyDetector
from src.tabular.metrics import compute_auroc
from src.tabular.preprocessor import TabularPreprocessor

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def evaluate_one(
    name: str,
    path: Path,
    test_size: float,
    seed: int,
) -> dict:
    """Evaluate detector on a single ADBench dataset.

    Returns a dict with keys:
      dataset, n_samples, n_features, anomaly_ratio, auroc, error, elapsed_s
    """
    result: dict = {
        "dataset": name,
        "n_samples": None,
        "n_features": None,
        "anomaly_ratio": None,
        "auroc": None,
        "error": None,
        "elapsed_s": None,
    }
    t0 = time.perf_counter()
    try:
        X, y = load_dataset(path)
        result["n_samples"] = int(X.shape[0])
        result["n_features"] = int(X.shape[1])
        result["anomaly_ratio"] = float(y.mean())

        # Validate: must have both classes
        if y.sum() == 0:
            raise ValueError("No anomalies in dataset")
        if (y == 0).sum() == 0:
            raise ValueError("No normal samples in dataset")

        # Stratified split so anomaly ratio is preserved in both splits
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        # Preprocess — fit only on training data
        prep = TabularPreprocessor()
        X_tr_p = prep.fit_transform(X_tr)
        X_te_p = prep.transform(X_te)

        # Fit detector on training data (unsupervised — no labels used)
        det = TabularAnomalyDetector(random_state=seed)
        det.fit(X_tr_p)

        # Score test set
        scores = det.score(X_te_p)

        auroc = compute_auroc(y_te, scores)
        if np.isnan(auroc):
            raise ValueError("AUROC is NaN (degenerate split?)")

        result["auroc"] = round(float(auroc), 6)
        log.info(
            f"[{name}] n={X.shape[0]} f={X.shape[1]} "
            f"anom={y.mean():.2%} AUROC={auroc:.4f}"
        )

    except Exception as exc:
        result["error"] = str(exc)
        log.warning(f"[{name}] FAILED — {exc}")

    result["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return result


def run_evaluation(
    data_dir: Path,
    results_dir: Path,
    test_size: float,
    seed: int,
    dataset_filter: str | None,
) -> pd.DataFrame:
    """Run evaluation over all (or one) dataset(s) and save CSV."""
    results_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    if dataset_filter:
        npz_files = [f for f in npz_files if f.stem == dataset_filter]
        if not npz_files:
            raise FileNotFoundError(
                f"Dataset '{dataset_filter}' not found in {data_dir}"
            )

    log.info(f"Evaluating {len(npz_files)} dataset(s) from {data_dir}")
    log.info(f"Settings: test_size={test_size}, seed={seed}")

    rows = []
    for fp in npz_files:
        rows.append(evaluate_one(fp.stem, fp, test_size=test_size, seed=seed))

    df = pd.DataFrame(rows)

    out_path = results_dir / "per_dataset_auroc.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved results to {out_path}")

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print summary table and aggregate statistics."""
    succeeded = df[df["error"].isna()].copy()
    failed = df[df["error"].notna()]

    W = 36
    print()
    print("=" * (W + 42))
    print(f"  {'Dataset':<{W}} {'Samples':>8}  {'Feat':>5}  {'Anom%':>6}  {'AUROC':>8}")
    print("-" * (W + 42))

    for _, row in succeeded.sort_values("auroc", ascending=False).iterrows():
        print(
            f"  {row['dataset']:<{W}} {int(row['n_samples']):>8}  "
            f"{int(row['n_features']):>5}  {100*row['anomaly_ratio']:>5.1f}%  "
            f"{row['auroc']:>8.4f}"
        )

    print("=" * (W + 42))
    if len(succeeded) > 0:
        mean_auroc = succeeded["auroc"].mean()
        print(f"  {'Mean AUROC':<{W}} {'':>8}  {'':>5}  {'':>6}  {mean_auroc:>8.4f}")
    print(f"  Succeeded: {len(succeeded)} / {len(df)}")
    if len(failed) > 0:
        print(f"  Failed:    {len(failed)}")
        for _, row in failed.iterrows():
            print(f"    - {row['dataset']}: {row['error']}")
    print("=" * (W + 42))


def main():
    p = argparse.ArgumentParser(description="ADBench per-dataset AUROC evaluation")
    p.add_argument("--data-dir", type=Path, default=_ROOT / "data" / "adbench")
    p.add_argument("--results-dir", type=Path, default=_ROOT / "results")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Fraction of data used as test set (default 0.2)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default 42)")
    p.add_argument("--dataset", type=str, default=None,
                   help="Evaluate only this dataset (stem of .npz filename)")
    args = p.parse_args()

    df = run_evaluation(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        test_size=args.test_size,
        seed=args.seed,
        dataset_filter=args.dataset,
    )
    print_summary(df)


if __name__ == "__main__":
    main()
