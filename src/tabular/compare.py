"""
Head-to-head comparison: IsolationForest vs RFuni on ADBench.

Reads:
  results/final_results.csv     — IsolationForest AUROC per dataset
  results/rfuni_baseline.csv    — RFuni AUROC per dataset

Outputs:
  results/comparison.csv        — merged table with delta column

Usage
-----
    python src/tabular/compare.py [--results-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parents[2]


def compare(results_dir: Path) -> pd.DataFrame:
    """Merge IsolationForest and RFuni results, compute deltas."""
    iso_path = results_dir / "final_results.csv"
    rfuni_path = results_dir / "rfuni_baseline.csv"

    for p in (iso_path, rfuni_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing file: {p}\n"
                "Run  python src/tabular/evaluate.py  and  python src/tabular/rfuni.py  first."
            )

    iso = pd.read_csv(iso_path)[["dataset", "n_samples", "n_features", "anomaly_pct", "auroc"]]
    iso = iso.rename(columns={"auroc": "iso_auroc"})

    rfuni = pd.read_csv(rfuni_path)[["dataset", "rfuni_auroc", "error"]]
    rfuni_ok = rfuni[rfuni["error"].isna()][["dataset", "rfuni_auroc"]]

    merged = iso.merge(rfuni_ok, on="dataset", how="inner")
    merged["delta"] = (merged["iso_auroc"] - merged["rfuni_auroc"]).round(6)
    merged = merged.sort_values("delta", ascending=False).reset_index(drop=True)

    out_path = results_dir / "comparison.csv"
    merged.to_csv(out_path, index=False)
    return merged


def print_comparison(df: pd.DataFrame) -> None:
    """Print comparison table with win/loss highlights."""
    iso_mean = df["iso_auroc"].mean()
    rfuni_mean = df["rfuni_auroc"].mean()
    delta_mean = df["delta"].mean()

    wins = (df["delta"] > 0).sum()
    losses = (df["delta"] < 0).sum()
    ties = (df["delta"] == 0).sum()

    W = 34
    print()
    print("=" * (W + 52))
    print(f"  {'Dataset':<{W}} {'IsoForest':>10}  {'RFuni':>8}  {'Delta':>8}  {'':>3}")
    print("-" * (W + 52))
    for _, row in df.iterrows():
        flag = ">" if row["delta"] > 0.005 else ("<" if row["delta"] < -0.005 else "~")
        print(
            f"  {row['dataset']:<{W}} {row['iso_auroc']:>10.4f}  "
            f"{row['rfuni_auroc']:>8.4f}  {row['delta']:>+8.4f}  {flag:>3}"
        )
    print("=" * (W + 52))
    print(
        f"\n  Mean AUROC — IsolationForest: {iso_mean:.4f} | RFuni: {rfuni_mean:.4f} "
        f"| Delta: {delta_mean:+.4f}"
    )
    print(
        f"  Win/Tie/Loss: {wins} / {ties} / {losses}  "
        f"(IsolationForest {'WINS' if wins > losses else 'LOSES'} overall)"
    )
    print()

    print("  Top-5 wins (IsolationForest >> RFuni):")
    for _, row in df.head(5).iterrows():
        print(f"    {row['dataset']:<34} {row['iso_auroc']:.4f} vs {row['rfuni_auroc']:.4f} ({row['delta']:+.4f})")

    print("  Top-5 losses (RFuni >> IsolationForest):")
    for _, row in df.tail(5).iloc[::-1].iterrows():
        print(f"    {row['dataset']:<34} {row['iso_auroc']:.4f} vs {row['rfuni_auroc']:.4f} ({row['delta']:+.4f})")
    print()


def main():
    p = argparse.ArgumentParser(description="IsolationForest vs RFuni comparison")
    p.add_argument("--results-dir", type=Path, default=_ROOT / "results")
    args = p.parse_args()

    df = compare(args.results_dir)
    print_comparison(df)
    print(f"  Saved: {args.results_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
