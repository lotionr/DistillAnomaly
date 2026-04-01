"""
AUROC scoring utilities for anomaly detection evaluation.

Primary function: compute_auroc(y_true, scores)
  - Wraps sklearn.metrics.roc_auc_score
  - Handles edge cases: all-same labels, NaN/inf scores
  - Returns float in [0, 1] or np.nan on failure
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc(
    y_true: np.ndarray | list,
    scores: np.ndarray | list,
) -> float:
    """Compute AUROC (Area Under the ROC Curve).

    Parameters
    ----------
    y_true  : 1D array-like of int {0, 1} — ground-truth labels (1 = anomaly)
    scores  : 1D array-like of float — anomaly scores (higher = more anomalous)

    Returns
    -------
    auroc : float in [0.0, 1.0], or np.nan if AUROC cannot be computed.

    Edge cases handled
    ------------------
    - All labels are the same (only 0s or only 1s): returns np.nan
    - scores contains NaN or ±inf: invalid entries are replaced with the
      finite median before scoring; if *all* scores are non-finite, returns np.nan
    - Empty arrays: returns np.nan
    """
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()

    # Guard: non-empty
    if len(y_true) == 0 or len(scores) == 0:
        return float("nan")

    # Guard: matching lengths
    if len(y_true) != len(scores):
        raise ValueError(
            f"y_true and scores must have the same length "
            f"(got {len(y_true)} and {len(scores)})"
        )

    # Guard: binary labels only
    unique_labels = np.unique(y_true)
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"y_true must contain only 0 and 1; found {unique_labels}")

    # Guard: both classes must be present
    if len(unique_labels) < 2:
        return float("nan")

    # Fix non-finite scores by substituting finite median
    bad_mask = ~np.isfinite(scores)
    if bad_mask.all():
        return float("nan")
    if bad_mask.any():
        finite_median = float(np.median(scores[~bad_mask]))
        scores = scores.copy()
        scores[bad_mask] = finite_median

    return float(roc_auc_score(y_true, scores))


if __name__ == "__main__":
    """Unit tests with known inputs."""
    import sys

    failures: list[str] = []

    def check(name: str, got, expected, tol: float = 1e-6):
        if expected is None or (isinstance(expected, float) and np.isnan(expected)):
            if not (got is None or (isinstance(got, float) and np.isnan(got))):
                failures.append(f"{name}: expected nan, got {got}")
                return
        elif abs(got - expected) > tol:
            failures.append(f"{name}: expected {expected}, got {got}")
            return
        print(f"  PASS  {name}: {got}")

    print("Running compute_auroc unit tests...")

    # --- Perfect separation ---
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    check("perfect separation", compute_auroc(y, s), 1.0)

    # --- Inverted (worst case) ---
    s_inv = 1.0 - s
    check("perfect inversion", compute_auroc(y, s_inv), 0.0)

    # --- Random chance: alternating labels ---
    # AUROC of 0.5 for random scores
    rng = np.random.default_rng(0)
    y_rand = np.array([0, 1] * 500)
    s_rand = rng.random(1000)
    auroc_rand = compute_auroc(y_rand, s_rand)
    if not (0.45 < auroc_rand < 0.55):
        failures.append(f"random chance: AUROC {auroc_rand:.4f} not in (0.45, 0.55)")
    else:
        print(f"  PASS  random chance AUROC ≈ 0.5: {auroc_rand:.4f}")

    # --- All same labels → nan ---
    check("all-normal labels", compute_auroc([0, 0, 0], [0.1, 0.2, 0.3]), float("nan"))
    check("all-anomaly labels", compute_auroc([1, 1, 1], [0.1, 0.2, 0.3]), float("nan"))

    # --- Empty input → nan ---
    check("empty input", compute_auroc([], []), float("nan"))

    # --- NaN in scores → handled gracefully ---
    y_nan = np.array([0, 0, 1, 1])
    s_nan = np.array([0.1, float("nan"), 0.9, 0.8])
    auroc_nan = compute_auroc(y_nan, s_nan)
    # NaN replaced by finite median of [0.1, 0.9, 0.8] = 0.8; scores = [0.1, 0.8, 0.9, 0.8]
    # y_true = [0, 0, 1, 1] → still good separation → AUROC ≥ 0.5
    if not np.isnan(auroc_nan) and auroc_nan >= 0.5:
        print(f"  PASS  NaN in scores handled: AUROC={auroc_nan:.4f}")
    else:
        failures.append(f"NaN in scores: unexpected AUROC={auroc_nan}")

    # --- All non-finite scores → nan ---
    check(
        "all-inf scores",
        compute_auroc([0, 1], [float("inf"), float("-inf")]),
        float("nan"),
    )

    # --- inf in scores → replaced by finite median, no crash ---
    # scores=[0.1, inf], finite_median=0.1 → scores=[0.1, 0.1] → AUROC=0.5 (tied)
    y_inf = np.array([0, 1])
    s_inf = np.array([0.1, float("inf")])
    auroc_inf = compute_auroc(y_inf, s_inf)
    if not np.isnan(auroc_inf):
        print(f"  PASS  inf in scores handled (no crash): AUROC={auroc_inf:.4f}")
    else:
        failures.append(f"inf in scores: got nan, expected finite AUROC")

    print()
    if failures:
        print(f"FAILED ({len(failures)} failures):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("All tests passed.")
