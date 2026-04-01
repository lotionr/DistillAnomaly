#!/usr/bin/env python3
"""
ADBench benchmark: per-row anomaly scoring via Qwen2.5-VL logit.

Approach
--------
For each row:
  1. Encode: absolute z-score → [0, 1000].
       |z|=0 → 0  (perfectly normal; flat baseline)
       |z|=4 → 1000  (4-sigma extreme; spike)
     Anomalous rows appear as spikes above a flat normal baseline.
  2. Generate 3 images: raw values, moving average, moving std.
  3. Run ONE forward pass (no generation); forced prefix = '{"anomalies": ['
  4. Score = logit("{") − logit("]") at the next-token position.
       Positive → model "wants" to say anomaly.
       Negative → model "wants" to say no anomaly.
  5. AUROC over the sampled rows.

Why this is better than the column-as-time-series approach
----------------------------------------------------------
  - Each row gets a direct independent score.
  - Anomalous rows with extreme feature values appear as visible spikes.
  - No interval-to-row mapping; no temporal ordering assumption.

Usage
-----
  python src/real_model/evaluate_per_row.py \
      --model-path /home/ubuntu/models/Qwen2.5-VL-3B-Instruct \
      --data-dir data/adbench \
      --results-file results/per_row_auroc.csv
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.real_model.image_converter import series_to_images
from src.real_model.inference import (
    SYSTEM_PROMPT,
    AnomalyDetectorVL,
    _even_grid,
)


# ── prompt (ts3 mode, adapted for row-as-series) ──────────────────────────────
PROMPT_ROW = (
    "Given the time series below and the following plot images: "
    "raw values, moving average, and moving standard deviation, "
    "determine whether there is an anomalous interval.\n"
    "The series encodes how far each measurement deviates from the normal baseline. "
    "A value of 0 means perfectly normal; higher values indicate increasing deviation.\n\n"
    "Return ONLY a JSON object formatted exactly as follows:\n"
    "Empty (no anomaly): {{\"anomalies\": []}}\n"
    "Non-empty:  {{\"anomalies\": [{{\"start\": <int>, \"end\": <int>, "
    "\"description\": <string>}}]}}\n\n"
    "Time series (len: {length}):\n"
    "# Format per line: timestamp, value\n"
    "{series}"
)

FORCED_PREFIX = '{"anomalies": ['  # The model generates this, then we read the next token


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def absolute_zscore_encoding(X: np.ndarray) -> np.ndarray:
    """
    Encode X as |z-score| scaled to [0, 1000].

    Normal features (z ≈ 0): value near 0  → flat baseline in time-series plot.
    Extreme features (|z| ≥ 4): value 1000 → spike above baseline.

    This creates a spike-anomaly representation that matches the model's training.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_z = (X - mean) / std
    return np.clip(np.abs(X_z) / 4.0 * 1000.0, 0.0, 1000.0)


def select_top_features(X: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k features by global variance."""
    k = min(k, X.shape[1])
    return np.argsort(X.var(axis=0))[::-1][:k]


def stratified_sample(y: np.ndarray, n_per_class: int, seed: int = 42) -> np.ndarray:
    """Sample at most n_per_class indices from each class."""
    rng = np.random.default_rng(seed)
    normal_idx = np.where(y == 0)[0]
    anom_idx = np.where(y == 1)[0]
    n_norm = min(n_per_class, len(normal_idx))
    n_anom = min(n_per_class, len(anom_idx))
    sampled = np.concatenate([
        rng.choice(normal_idx, n_norm, replace=False),
        rng.choice(anom_idx, n_anom, replace=False),
    ])
    return sampled


# ──────────────────────────────────────────────────────────────────────────────
# Batch forward-pass scorer
# ──────────────────────────────────────────────────────────────────────────────

class PerRowScorer:
    """
    Wraps AnomalyDetectorVL for efficient per-row scoring.

    Uses a single forward pass (no generation) with a forced prefix
    '{"anomalies": [' and reads logit("{") − logit("]") at the last position.
    """

    def __init__(self, detector: AnomalyDetectorVL, max_features: int = 64):
        self.det = detector
        self.tok = detector.tokenizer
        self.max_features = max_features

        self.forced_ids = self.tok.encode(FORCED_PREFIX, add_special_tokens=False)
        self.tok_anom = self.tok.encode("{", add_special_tokens=False)[0]
        self.tok_norm = self.tok.encode("]", add_special_tokens=False)[0]

    # ── build the image tensor + seg_tokens for one image ──────────────────────
    def _img_tokens(self, img):
        pv, thw, seg = self.det._prepare_image(img)
        return pv, thw, seg

    # ── encode a single row into (pixel_values, image_grid_thw, seg_tokens, chat_ids) ──
    def _encode_row(self, row_int: np.ndarray):
        import pandas as _pd

        n = len(row_int)
        window = max(3, n // 10)
        s = _pd.Series(row_int)
        ma = s.rolling(window, center=True, min_periods=1).mean().round().to_numpy()
        ms = s.rolling(window, center=True, min_periods=1).std().fillna(0).round().to_numpy()

        img_raw, img_mean, img_std = series_to_images(row_int, window=window)

        pv_list, thw_list, seg_tokens = [], [], []
        for img in (img_raw, img_mean, img_std):
            pv, thw, seg = self._img_tokens(img)
            pv_list.append(pv)
            thw_list.append(thw)
            seg_tokens.extend(seg)

        pixel_values = torch.cat(pv_list, dim=0)  # (3*n_patches, hidden)
        image_grid_thw = torch.stack(thw_list, dim=0)  # (3, 3)

        # Text prompt
        lines = [f"timestamp: {i}, value: {int(row_int[i])}" for i in range(n)]
        series_text = "\n".join(lines)
        prompt = PROMPT_ROW.format(length=n, series=series_text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        encoded = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
        chat_ids = encoded["input_ids"].squeeze(0)  # (L,) – CPU

        # Build full token sequence: seg_tokens + chat_ids + forced_prefix
        pre = torch.tensor(seg_tokens, dtype=torch.long)
        forced = torch.tensor(self.forced_ids, dtype=torch.long)
        full_ids = torch.cat([pre, chat_ids, forced], dim=0)  # (L_full,)

        return pixel_values, image_grid_thw, full_ids

    # ── score a batch of rows ───────────────────────────────────────────────────
    def score_batch(self, rows_int: list[np.ndarray]) -> list[float]:
        """
        Score a batch of rows. All rows must have the same length (n_features).
        Returns list of logit scores (higher = more anomalous).
        """
        B = len(rows_int)
        dev = self.det.device

        # Encode each row
        all_pv, all_thw, all_ids = [], [], []
        for row in rows_int:
            pv, thw, ids = self._encode_row(row)
            all_pv.append(pv)
            all_thw.append(thw)
            all_ids.append(ids)

        # Stack: pixel_values (B * 3 * n_patches, hidden)
        pixel_values = torch.cat(all_pv, dim=0).to(device=dev, dtype=torch.float32)
        # image_grid_thw: (B * 3, 3)
        image_grid_thw = torch.cat(all_thw, dim=0).to(dev)

        # input_ids: all rows should have the same length (same n_features → same token count)
        seq_lens = [ids.shape[0] for ids in all_ids]
        max_len = max(seq_lens)
        input_ids = torch.zeros(B, max_len, dtype=torch.long, device=dev)
        attn_mask = torch.zeros(B, max_len, dtype=torch.long, device=dev)
        for i, ids in enumerate(all_ids):
            L = ids.shape[0]
            input_ids[i, :L] = ids.to(dev)
            attn_mask[i, :L] = 1

        with torch.no_grad():
            out = self.det.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        # logits at the last non-padding token for each row
        scores = []
        for i, ids in enumerate(all_ids):
            last_pos = ids.shape[0] - 1
            logits = out.logits[i, last_pos, :]
            s = (logits[self.tok_anom] - logits[self.tok_norm]).item()
            scores.append(s)

        return scores

    # ── main dataset scoring ────────────────────────────────────────────────────
    def score_dataset(
        self,
        X_enc: np.ndarray,          # (n_samples, n_features) already encoded
        sample_idx: np.ndarray,     # which rows to score
        batch_size: int = 4,
    ) -> np.ndarray:
        """
        Score the rows in sample_idx. Returns 1-D array aligned with sample_idx.
        """
        n_feat = X_enc.shape[1]
        # Subsample features if needed (all rows use the same feature indices)
        feat_idx = select_top_features(X_enc, self.max_features)
        rows_all = X_enc[sample_idx][:, feat_idx].astype(float)

        scores = np.empty(len(sample_idx), dtype=float)
        n_batches = 0

        for start in range(0, len(sample_idx), batch_size):
            chunk = rows_all[start : start + batch_size]
            rows_list = [chunk[j] for j in range(len(chunk))]
            batch_scores = self.score_batch(rows_list)
            scores[start : start + len(batch_scores)] = batch_scores
            n_batches += 1
            if n_batches % 5 == 0:
                pct = start / len(sample_idx) * 100
                print(f"    {start}/{len(sample_idx)} rows scored ({pct:.0f}%)")

        return scores


# ──────────────────────────────────────────────────────────────────────────────
# AUROC helper
# ──────────────────────────────────────────────────────────────────────────────

def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    finite = np.isfinite(scores)
    if not finite.any():
        return float("nan")
    scores = np.where(finite, scores, float(np.nanmedian(scores)))
    return roc_auc_score(y_true.tolist(), scores.tolist())


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
    p.add_argument("--results-file", default=Path("results/per_row_auroc.csv"), type=Path)
    p.add_argument("--comparison-file", default=Path("results/real_model_comparison.csv"), type=Path)
    p.add_argument("--max-features", type=int, default=64, help="Max features per row (default 64)")
    p.add_argument("--n-per-class", type=int, default=50, help="Rows sampled per class (default 50)")
    p.add_argument("--batch-size", type=int, default=4, help="Rows per forward pass (default 4)")
    p.add_argument("--dataset", type=str, default=None, help="Single dataset (for testing)")
    p.add_argument("--disable-lora", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.model_path.is_dir():
        print(f"ERROR: model not found at {args.model_path}", file=sys.stderr)
        sys.exit(1)
    if not args.data_dir.is_dir():
        print(f"ERROR: data dir not found at {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    print("Loading model...")
    detector = AnomalyDetectorVL(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        skip_lora=args.disable_lora,
    )
    scorer = PerRowScorer(detector, max_features=args.max_features)

    npz_files = sorted(args.data_dir.glob("*.npz"))
    if args.dataset:
        npz_files = [f for f in npz_files if f.stem == args.dataset]
        if not npz_files:
            print(f"ERROR: dataset '{args.dataset}' not found")
            sys.exit(1)

    print(f"Evaluating {len(npz_files)} datasets (n_per_class={args.n_per_class}, "
          f"max_features={args.max_features}, batch_size={args.batch_size})...")

    results, failed = [], []

    for npz_path in npz_files:
        ds_name = npz_path.stem
        t0 = time.time()
        try:
            data = np.load(npz_path, allow_pickle=True)
            X = data["X"].astype(np.float32)
            y = data["y"].astype(int).ravel()

            n_anom = int(y.sum())
            if n_anom == 0:
                raise ValueError("No anomalies")
            if len(y) - n_anom == 0:
                raise ValueError("No normal samples")

            n_samples, n_features = X.shape
            print(f"\n[{ds_name}] N={n_samples} feat={n_features} "
                  f"anom={n_anom} ({100*n_anom/n_samples:.1f}%)")

            # Encode: |z-score| → [0, 1000]
            X_enc = absolute_zscore_encoding(X)

            # Stratified sample
            sample_idx = stratified_sample(y, n_per_class=args.n_per_class, seed=42)
            y_sample = y[sample_idx]
            print(f"  Scoring {len(sample_idx)} rows "
                  f"({(y_sample==0).sum()} normal, {(y_sample==1).sum()} anomaly)")

            # Score
            scores = scorer.score_dataset(X_enc, sample_idx, batch_size=args.batch_size)
            auroc = compute_auroc(y_sample, scores)
            elapsed = time.time() - t0

            print(f"  → AUROC = {auroc:.4f}  ({elapsed:.0f}s)")
            results.append({
                "dataset": ds_name,
                "n_samples": n_samples,
                "n_features": n_features,
                "n_anomalies": n_anom,
                "n_scored": len(sample_idx),
                "auroc": round(auroc, 6) if not np.isnan(auroc) else float("nan"),
                "elapsed_s": round(elapsed, 1),
            })

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  [FAILED] {ds_name}: {exc}")
            failed.append({"dataset": ds_name, "error": str(exc), "elapsed_s": round(elapsed, 1)})

    # ── Save results ──────────────────────────────────────────────────────────
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).sort_values("auroc", ascending=False)
    df.to_csv(args.results_file, index=False)
    print(f"\nResults saved: {args.results_file}")

    valid = df["auroc"].dropna()
    mean_auroc = valid.mean()
    print(f"Mean AUROC (per-row real model): {mean_auroc:.4f}  "
          f"({len(valid)}/{len(npz_files)} datasets)")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  {f['dataset']}: {f['error']}")

    # ── Comparison with previous baselines ────────────────────────────────────
    comp_rows = [{"dataset": r["dataset"], "real_model_per_row_auroc": r["auroc"]}
                 for r in results]
    comp_df = pd.DataFrame(comp_rows)

    iso_csv = args.results_file.parent / "final_results.csv"
    if iso_csv.exists():
        iso = pd.read_csv(iso_csv)[["dataset", "auroc"]].rename(
            columns={"auroc": "isolation_forest_auroc"}
        )
        comp_df = comp_df.merge(iso, on="dataset", how="left")

    rfuni_csv = args.results_file.parent / "rfuni_baseline.csv"
    if rfuni_csv.exists():
        rf = pd.read_csv(rfuni_csv)
        col = "rfuni_auroc" if "rfuni_auroc" in rf.columns else "auroc"
        rf = rf[["dataset", col]].rename(columns={col: "rfuni_auroc"})
        comp_df = comp_df.merge(rf, on="dataset", how="left")

    # Also merge the interval-based real model results if available
    interval_csv = args.results_file.parent / "real_model_auroc.csv"
    if interval_csv.exists():
        iv = pd.read_csv(interval_csv)[["dataset", "auroc"]].rename(
            columns={"auroc": "real_model_interval_auroc"}
        )
        comp_df = comp_df.merge(iv, on="dataset", how="left")

    comp_df.to_csv(args.comparison_file, index=False)
    print(f"Comparison saved: {args.comparison_file}")

    # Print summary table
    W = 36
    available_cols = [c for c in [
        "real_model_per_row_auroc", "real_model_interval_auroc",
        "isolation_forest_auroc", "rfuni_auroc"
    ] if c in comp_df.columns]
    headers = {
        "real_model_per_row_auroc": "PerRow",
        "real_model_interval_auroc": "Interval",
        "isolation_forest_auroc": "IsoForest",
        "rfuni_auroc": "RFuni",
    }
    hdr = f"  {'Dataset':<{W}}" + "".join(f"  {headers[c]:>9}" for c in available_cols)
    print()
    print(hdr)
    print("-" * len(hdr))
    for _, row in comp_df.sort_values("real_model_per_row_auroc", ascending=False).iterrows():
        line = f"  {row['dataset']:<{W}}"
        for col in available_cols:
            val = row.get(col, float("nan"))
            line += f"  {val:>9.4f}" if not pd.isna(val) else f"  {'N/A':>9}"
        print(line)

    print()
    for col in available_cols:
        m = comp_df[col].dropna().mean()
        print(f"  Mean {headers[col]}: {m:.4f}")


if __name__ == "__main__":
    main()
