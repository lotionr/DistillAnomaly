#!/usr/bin/env python3
"""
ADBench benchmark using the real Qwen2.5-VL-3B + LoRA model.

Protocol
--------
For each dataset:
  1. Select top-K features by variance (K = min(5, n_features)).
  2. For each selected feature:
     a. Subsample to ≤ MAX_LEN points (uniform).
     b. Normalise values to integer range [0, 1000].
     c. Compute rolling mean / std (window = max(10, len//20)).
     d. Generate 3 images (raw, mean, std).
     e. Run Qwen2.5-VL-3B to detect anomaly interval [start, end].
     f. Map detected indices back to original row indices.
     g. Flag rows in the detected interval.
  3. Row anomaly score = fraction of features that flagged it.
  4. AUROC(y_true, scores).

Usage
-----
  python src/real_model/evaluate_real.py \
      --model-path /home/ubuntu/models/Qwen2.5-VL-3B-Instruct \
      --adapter-path train_VL/qwen2.5-vl-3b-lora-vl-ts3-20ep/checkpoint-1060 \
      --data-dir data/adbench \
      --results-file results/real_model_auroc.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.real_model.image_converter import (
    normalise_to_int_range,
    select_features,
    series_to_images,
    subsample,
)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument(
        "--adapter-path",
        default=Path("train_VL/qwen2.5-vl-3b-lora-vl-ts3-20ep/checkpoint-1060"),
        type=Path,
    )
    p.add_argument("--data-dir", default=Path("data/adbench"), type=Path)
    p.add_argument(
        "--results-file",
        default=Path("results/real_model_auroc.csv"),
        type=Path,
    )
    p.add_argument(
        "--comparison-file",
        default=Path("results/real_model_comparison.csv"),
        type=Path,
    )
    p.add_argument("--k-features", type=int, default=3,
                   help="Number of top-variance features per dataset (default 3)")
    p.add_argument("--max-len", type=int, default=128,
                   help="Max time-series length (subsample if longer, default 128)")
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Max tokens to generate per inference call (default 64)")
    p.add_argument("--dataset", type=str, default=None,
                   help="Evaluate a single dataset by name (for testing)")
    p.add_argument("--disable-lora", action="store_true",
                   help="Skip LoRA merge (base model only)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Core logic
# ──────────────────────────────────────────────────────────────────────────────

def score_dataset(
    detector,
    X: np.ndarray,
    k_features: int,
    max_len: int,
    max_new_tokens: int = 64,
) -> np.ndarray:
    """
    Returns a float array of shape (N_samples,) where higher = more anomalous.
    Each element is the fraction of selected features that flagged that row.
    """
    n_samples, n_features = X.shape
    feat_indices = select_features(X, k=k_features)

    # Accumulator: how many features flagged each row
    vote_count = np.zeros(n_samples, dtype=float)

    for fi in feat_indices:
        series_orig = X[:, fi].astype(float)

        # Subsample
        series_sub, orig_idx = subsample(series_orig, max_len=max_len)

        # Normalise to integer range
        series_int = normalise_to_int_range(series_sub)

        # Rolling stats
        import pandas as _pd
        s = _pd.Series(series_int)
        window = max(10, len(series_int) // 20)
        ma_int = s.rolling(window, center=True, min_periods=1).mean().round().to_numpy()
        ms_int = (
            s.rolling(window, center=True, min_periods=1)
            .std()
            .fillna(0)
            .round()
            .to_numpy()
        )

        # Generate images
        img_raw, img_mean, img_std = series_to_images(series_int, window=window)

        # Run model
        result = detector.detect(
            series_int=series_int,
            ma_int=ma_int,
            ms_int=ms_int,
            img_raw=img_raw,
            img_mean=img_mean,
            img_std=img_std,
            max_new_tokens=max_new_tokens,
        )

        # Parse anomaly intervals
        anomalies = result.get("anomalies", [])
        for anom in anomalies:
            try:
                start = int(anom["start"])
                end = int(anom["end"])
            except (KeyError, ValueError, TypeError):
                continue

            # Clamp to valid range (model may overshoot)
            start = max(0, min(start, len(orig_idx) - 1))
            end = max(0, min(end, len(orig_idx) - 1))
            if start > end:
                start, end = end, start

            # Map subsampled indices back to original row indices
            flagged_orig = orig_idx[start : end + 1]
            vote_count[flagged_orig] += 1.0

    n_feat_used = len(feat_indices)
    scores = vote_count / n_feat_used if n_feat_used > 0 else vote_count
    return scores


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    finite = np.isfinite(scores)
    if not finite.any():
        return float("nan")
    scores = np.where(finite, scores, np.nanmedian(scores))
    return roc_auc_score(y_true.tolist(), scores.tolist())


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Validate paths
    if not args.model_path.is_dir():
        print(f"ERROR: model not found at {args.model_path}", file=sys.stderr)
        sys.exit(1)

    if not args.data_dir.is_dir():
        print(f"ERROR: data dir not found at {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    # Load model
    print("Loading model...")
    from src.real_model.inference import AnomalyDetectorVL

    detector = AnomalyDetectorVL(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        skip_lora=args.disable_lora,
    )

    # Find datasets
    npz_files = sorted(args.data_dir.glob("*.npz"))
    if args.dataset:
        npz_files = [f for f in npz_files if f.stem == args.dataset]
        if not npz_files:
            print(f"ERROR: dataset '{args.dataset}' not found in {args.data_dir}")
            sys.exit(1)

    print(f"Evaluating {len(npz_files)} datasets...")

    results = []
    failed = []

    for npz_path in npz_files:
        ds_name = npz_path.stem
        try:
            data = np.load(npz_path, allow_pickle=True)
            X = data["X"].astype(np.float32)
            y = data["y"].astype(int).ravel()

            n_anom = int(y.sum())
            if n_anom == 0:
                raise ValueError("No anomalies in dataset")
            if len(y) - n_anom == 0:
                raise ValueError("No normal samples in dataset")

            n_samples, n_features = X.shape
            print(
                f"[{ds_name}] N={n_samples} feat={n_features} "
                f"anom={n_anom} ({100*n_anom/n_samples:.1f}%)"
            )

            scores = score_dataset(
                detector=detector,
                X=X,
                k_features=args.k_features,
                max_len=args.max_len,
                max_new_tokens=args.max_new_tokens,
            )
            auroc = compute_auroc(y, scores)
            print(f"  → AUROC = {auroc:.4f}")

            results.append(
                {
                    "dataset": ds_name,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "n_anomalies": n_anom,
                    "auroc": round(auroc, 6) if not np.isnan(auroc) else float("nan"),
                }
            )

        except Exception as exc:
            print(f"  [FAILED] {ds_name}: {exc}")
            failed.append({"dataset": ds_name, "error": str(exc)})

    # Save per-dataset results
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).sort_values("auroc", ascending=False)
    df.to_csv(args.results_file, index=False)
    print(f"\nResults saved: {args.results_file}")

    # Summary
    valid = df["auroc"].dropna()
    mean_auroc = valid.mean()
    print(f"\nMean AUROC (real model): {mean_auroc:.4f}  ({len(valid)}/{len(npz_files)} datasets)")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  {f['dataset']}: {f['error']}")

    # Merge with IsolationForest results for comparison
    iso_csv = args.results_file.parent / "final_results.csv"
    rfuni_csv = args.results_file.parent / "rfuni_baseline.csv"

    comparison_rows = []
    for row in results:
        comparison_rows.append({"dataset": row["dataset"], "real_model_auroc": row["auroc"]})

    comp_df = pd.DataFrame(comparison_rows)

    if iso_csv.exists():
        iso_df = pd.read_csv(iso_csv)[["dataset", "auroc"]].rename(
            columns={"auroc": "isolation_forest_auroc"}
        )
        comp_df = comp_df.merge(iso_df, on="dataset", how="left")

    if rfuni_csv.exists():
        rf_df = pd.read_csv(rfuni_csv)
        # column may be "rfuni_auroc" (from rfuni.py) or "auroc"
        auroc_col = "rfuni_auroc" if "rfuni_auroc" in rf_df.columns else "auroc"
        rf_df = rf_df[["dataset", auroc_col]].rename(columns={auroc_col: "rfuni_auroc"})
        comp_df = comp_df.merge(rf_df, on="dataset", how="left")

    comp_df.to_csv(args.comparison_file, index=False)
    print(f"Comparison saved: {args.comparison_file}")

    # Print comparison table
    print()
    W = 36
    cols = ["real_model_auroc", "isolation_forest_auroc", "rfuni_auroc"]
    headers = ["RealModel", "IsoForest", "RFuni"]
    available = [c for c in cols if c in comp_df.columns]
    avail_h = [h for h, c in zip(headers, cols) if c in comp_df.columns]

    header_line = f"  {'Dataset':<{W}}" + "".join(f"  {h:>9}" for h in avail_h)
    print(header_line)
    print("-" * len(header_line))
    for _, row in comp_df.sort_values("real_model_auroc", ascending=False).iterrows():
        line = f"  {row['dataset']:<{W}}"
        for col in available:
            val = row.get(col, float("nan"))
            line += f"  {val:>9.4f}" if not pd.isna(val) else f"  {'N/A':>9}"
        print(line)

    print()
    for h, col in zip(avail_h, available):
        m = comp_df[col].dropna().mean()
        print(f"  Mean {h}: {m:.4f}")


if __name__ == "__main__":
    main()
