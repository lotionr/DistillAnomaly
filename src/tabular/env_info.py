"""
Log Python and package versions to results/env_info.txt for reproducibility.

Usage
-----
    python src/tabular/env_info.py [--results-dir PATH]
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parents[2]

PACKAGES_TO_LOG = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
]


def collect_env_info(results_dir: Path) -> str:
    """Collect env info, write to results/env_info.txt, and return the content."""
    results_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(f"Generated: {ts}")
    lines.append("")
    lines.append("Python")
    lines.append(f"  version   : {sys.version}")
    lines.append(f"  executable: {sys.executable}")
    lines.append(f"  platform  : {platform.platform()}")
    lines.append("")
    lines.append("Packages")
    for pkg in PACKAGES_TO_LOG:
        try:
            import importlib.metadata
            ver = importlib.metadata.version(pkg)
        except Exception:
            ver = "N/A"
        lines.append(f"  {pkg:<20}: {ver}")
    lines.append("")

    # Random seeds used in the pipeline
    lines.append("Reproducibility")
    lines.append("  All stochastic components use random_state=42 (TabularPreprocessor,")
    lines.append("  TabularAnomalyDetector, RFuniDetector, train_test_split).")
    lines.append("  numpy random: seeded per-call via np.random.default_rng(42).")
    lines.append("")

    # Git info
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        lines.append("Git")
        lines.append(f"  branch: {git_branch}")
        lines.append(f"  commit: {git_hash}")
    except Exception:
        lines.append("Git: (not available)")

    content = "\n".join(lines) + "\n"
    out = results_dir / "env_info.txt"
    out.write_text(content)
    return content


def main():
    p = argparse.ArgumentParser(description="Log environment info for reproducibility")
    p.add_argument("--results-dir", type=Path, default=_ROOT / "results")
    args = p.parse_args()

    content = collect_env_info(args.results_dir)
    print(content)
    print(f"Saved: {args.results_dir / 'env_info.txt'}")


if __name__ == "__main__":
    main()
