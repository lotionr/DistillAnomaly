"""
ADBench data loader for tabular anomaly detection.

Each .npz file in the ADBench benchmark contains:
  X : float64 array of shape (n_samples, n_features)
  y : int32 array of shape (n_samples,) — 0=normal, 1=anomaly
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_dataset(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single ADBench .npz file and return (X, y).

    Parameters
    ----------
    path : path to .npz file

    Returns
    -------
    X : float64 ndarray of shape (n_samples, n_features)
    y : int32 ndarray of shape (n_samples,), values in {0, 1}
    """
    data = np.load(path)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int32)
    # Ensure y is binary
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError(f"Labels in {path} contain values outside {{0, 1}}: {np.unique(y)}")
    return X, y


def load_all_datasets(
    data_dir: str | Path = "data/adbench",
    verbose: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load all .npz datasets from data_dir.

    Parameters
    ----------
    data_dir : directory containing ADBench .npz files
    verbose  : if True, print per-dataset summary

    Returns
    -------
    dict mapping dataset_name -> (X, y)
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    errors: List[str] = []

    if verbose:
        header = f"{'Dataset':<35} {'Samples':>8} {'Features':>9} {'Anomaly%':>9}"
        print(header)
        print("-" * len(header))

    for fp in npz_files:
        name = fp.stem  # filename without extension
        try:
            X, y = load_dataset(fp)
            datasets[name] = (X, y)
            if verbose:
                anomaly_pct = 100.0 * y.mean()
                print(f"{name:<35} {X.shape[0]:>8} {X.shape[1]:>9} {anomaly_pct:>8.2f}%")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            if verbose:
                print(f"{name:<35} {'ERROR':>8}  {exc}")

    if verbose:
        print(f"\nLoaded {len(datasets)}/{len(npz_files)} datasets successfully.")
        if errors:
            print(f"Errors ({len(errors)}):")
            for e in errors:
                print(f"  {e}")

    return datasets


def get_dataset_names(data_dir: str | Path = "data/adbench") -> List[str]:
    """Return sorted list of dataset names available in data_dir."""
    data_dir = Path(data_dir)
    return sorted(fp.stem for fp in data_dir.glob("*.npz"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ADBench data loader smoke test")
    parser.add_argument("--data-dir", default="data/adbench", help="Directory with .npz files")
    args = parser.parse_args()

    datasets = load_all_datasets(data_dir=args.data_dir, verbose=True)
    print(f"\nTotal datasets: {len(datasets)}")
