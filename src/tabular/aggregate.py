"""
Aggregate scoring and final results table for ADBench evaluation.

Reads results/per_dataset_auroc.csv (written by evaluate.py), computes
summary statistics, and writes results/final_results.csv.

Usage
-----
    python src/tabular/aggregate.py [--results-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parents[2]


def aggregate(results_dir: Path) -> pd.DataFrame:
    """Load per-dataset results, enrich, sort, and save final_results.csv.

    Returns the final DataFrame (successful rows only, sorted by AUROC desc).
    """
    in_path = results_dir / "per_dataset_auroc.csv"
    if not in_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {in_path}\n"
            "Run:  python src/tabular/evaluate.py"
        )

    df = pd.read_csv(in_path)

    # Separate succeeded vs failed
    succeeded = df[df["error"].isna()].copy()
    failed = df[df["error"].notna()].copy()

    if succeeded.empty:
        raise ValueError("No successful evaluations found in the results file.")

    # Compute anomaly_pct column for display
    succeeded["anomaly_pct"] = (succeeded["anomaly_ratio"] * 100).round(2)

    # Sort by AUROC descending
    succeeded = succeeded.sort_values("auroc", ascending=False).reset_index(drop=True)
    succeeded["rank"] = succeeded.index + 1

    # Select and reorder columns
    final = succeeded[[
        "rank", "dataset", "n_samples", "n_features", "anomaly_pct", "auroc"
    ]].copy()

    out_path = results_dir / "final_results.csv"
    final.to_csv(out_path, index=False)

    return final, failed


def print_table(final: pd.DataFrame, failed: pd.DataFrame) -> None:
    """Print a formatted summary table to stdout."""
    W = 36
    print()
    print("=" * (W + 48))
    print(f"  {'Rank':>4}  {'Dataset':<{W}} {'Samples':>8}  {'Feat':>5}  {'Anom%':>6}  {'AUROC':>8}")
    print("-" * (W + 48))
    for _, row in final.iterrows():
        print(
            f"  {int(row['rank']):>4}  {row['dataset']:<{W}} {int(row['n_samples']):>8}  "
            f"{int(row['n_features']):>5}  {row['anomaly_pct']:>5.1f}%  {row['auroc']:>8.4f}"
        )
    print("=" * (W + 48))

    n_ok = len(final)
    n_fail = len(failed)
    mean_auroc = final["auroc"].mean()
    median_auroc = final["auroc"].median()
    min_auroc = final["auroc"].min()
    max_auroc = final["auroc"].max()

    print(f"\n  Summary ({n_ok} datasets evaluated, {n_fail} failed):")
    print(f"    Mean AUROC   : {mean_auroc:.4f}")
    print(f"    Median AUROC : {median_auroc:.4f}")
    print(f"    Min AUROC    : {min_auroc:.4f}  ({final.iloc[-1]['dataset']})")
    print(f"    Max AUROC    : {max_auroc:.4f}  ({final.iloc[0]['dataset']})")

    if n_fail > 0:
        print(f"\n  Failed datasets ({n_fail}):")
        for _, row in failed.iterrows():
            print(f"    - {row['dataset']}: {row['error']}")

    print()


def main():
    p = argparse.ArgumentParser(description="Aggregate ADBench evaluation results")
    p.add_argument("--results-dir", type=Path, default=_ROOT / "results")
    args = p.parse_args()

    final, failed = aggregate(args.results_dir)
    print_table(final, failed)
    print(f"  Saved: {args.results_dir / 'final_results.csv'}")


if __name__ == "__main__":
    main()
