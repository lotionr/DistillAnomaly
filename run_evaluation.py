#!/usr/bin/env python3
"""
End-to-end ADBench evaluation pipeline.

Steps executed in order:
  1. Log environment info        → results/env_info.txt
  2. Run IsolationForest eval    → results/per_dataset_auroc.csv
  3. Aggregate results           → results/final_results.csv
  4. Run RFuni baseline eval     → results/rfuni_baseline.csv   [skipped with --skip-rfuni]
  5. Head-to-head comparison     → results/comparison.csv       [skipped with --skip-rfuni]

Flags
-----
  --dataset STEM          Evaluate only one dataset (quick smoke test)
  --skip-rfuni            Skip RFuni baseline (only run IsolationForest + aggregate)
  --baseline-only         Run only RFuni baseline (skip IsolationForest)
  --data-dir PATH         Directory with ADBench .npz files  [default: data/adbench]
  --results-dir PATH      Output directory                   [default: results/]
  --test-size FLOAT       Train/test split ratio             [default: 0.2]
  --seed INT              Random seed                        [default: 42]

Examples
--------
  python run_evaluation.py                       # full pipeline
  python run_evaluation.py --dataset 4_breastw   # single dataset, quick test
  python run_evaluation.py --skip-rfuni          # IsolationForest only (fast)
  python run_evaluation.py --baseline-only       # RFuni only
"""

import argparse
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="ADBench end-to-end anomaly detection evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data-dir", type=Path, default=_ROOT / "data" / "adbench",
                   help="Directory containing ADBench .npz files")
    p.add_argument("--results-dir", type=Path, default=_ROOT / "results",
                   help="Output directory for CSV results")
    p.add_argument("--dataset", type=str, default=None,
                   help="Evaluate only this dataset (stem of .npz filename)")
    p.add_argument("--skip-rfuni", action="store_true",
                   help="Skip RFuni baseline evaluation")
    p.add_argument("--baseline-only", action="store_true",
                   help="Run only RFuni baseline (skip IsolationForest)")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Test fraction for train/test split (default 0.2)")
    p.add_argument("--seed", type=int, default=42,
                   help="Global random seed (default 42)")
    return p.parse_args()


def step(label: str):
    """Context manager / decorator for timed pipeline steps."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        log.info(f"{'='*60}")
        log.info(f"STEP: {label}")
        log.info(f"{'='*60}")
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        log.info(f"Done in {elapsed:.1f}s")

    return _ctx()


def main():
    args = parse_args()

    # Validate data directory
    if not args.data_dir.is_dir():
        log.error(f"Data directory not found: {args.data_dir}")
        log.error("Run:  bash download_adbench_classical.sh  or check init.sh")
        sys.exit(1)
    npz_files = sorted(args.data_dir.glob("*.npz"))
    if not npz_files:
        log.error(f"No .npz files in {args.data_dir}")
        sys.exit(1)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    # ── Step 1: Environment info ──────────────────────────────────────────────
    with step("Environment Info"):
        from src.tabular.env_info import collect_env_info
        content = collect_env_info(args.results_dir)
        print(content)

    # ── Step 2 & 3: IsolationForest evaluation + aggregate ───────────────────
    if not args.baseline_only:
        with step("IsolationForest Evaluation"):
            from src.tabular.evaluate import run_evaluation, print_summary
            df_iso = run_evaluation(
                data_dir=args.data_dir,
                results_dir=args.results_dir,
                test_size=args.test_size,
                seed=args.seed,
                dataset_filter=args.dataset,
            )

        with step("Aggregate Results"):
            from src.tabular.aggregate import aggregate, print_table as print_agg
            final, failed = aggregate(args.results_dir)
            print_agg(final, failed)

    # ── Step 4: RFuni baseline ────────────────────────────────────────────────
    if not args.skip_rfuni:
        with step("RFuni Baseline Evaluation"):
            from src.tabular.rfuni import evaluate_one_rfuni
            import pandas as pd
            import numpy as np

            npz = sorted(args.data_dir.glob("*.npz"))
            if args.dataset:
                npz = [f for f in npz if f.stem == args.dataset]

            rows = [evaluate_one_rfuni(fp.stem, fp, args.test_size, args.seed) for fp in npz]
            df_rfuni = pd.DataFrame(rows)
            out = args.results_dir / "rfuni_baseline.csv"
            df_rfuni.to_csv(out, index=False)
            log.info(f"RFuni results saved to {out}")
            ok = df_rfuni[df_rfuni["error"].isna()]
            if len(ok):
                log.info(f"RFuni mean AUROC: {ok['rfuni_auroc'].mean():.4f}  ({len(ok)}/{len(df_rfuni)} datasets)")

    # ── Step 5: Comparison ────────────────────────────────────────────────────
    if not args.skip_rfuni and not args.baseline_only:
        with step("Head-to-Head Comparison"):
            from src.tabular.compare import compare, print_comparison
            df_cmp = compare(args.results_dir)
            print_comparison(df_cmp)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_total
    log.info(f"{'='*60}")
    log.info(f"Pipeline complete in {elapsed_total:.1f}s")
    log.info(f"Results in: {args.results_dir}/")
    for f in sorted(args.results_dir.glob("*.csv")):
        log.info(f"  {f.name}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
