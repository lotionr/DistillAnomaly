"""
Tabular → time-series image converter.

Given a 1-D numpy array (one feature column from an ADBench dataset),
produce three PIL images that match the model's training distribution
(ts3 mode: raw values, moving average, moving standard deviation).
"""

from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# ── constants tuned to match training image style ─────────────────────────────
_FIG_W = 2.0   # inches  (200px at 100dpi — fewer patches → faster inference)
_FIG_H = 1.0   # inches
_DPI   = 100

_COLOR_RAW  = "#005088"
_COLOR_MEAN = "#880000"
_COLOR_STD  = "#008800"


def _plot_series(values: np.ndarray, color: str) -> Image.Image:
    """Plot a 1-D array as a line chart and return a PIL RGB image."""
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.plot(values, color=color, linewidth=1.0)
    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05, dpi=_DPI)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def series_to_images(
    series: np.ndarray,
    window: int | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """
    Convert a 1-D time series to three PIL images (raw, moving_avg, moving_std).

    Parameters
    ----------
    series : 1-D float array  (already scaled / normalised)
    window : rolling window for mean/std; defaults to max(10, len(series)//20)

    Returns
    -------
    (img_raw, img_mean, img_std)
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if window is None:
        window = max(10, n // 20)

    s = pd.Series(series)
    ma  = s.rolling(window, center=True, min_periods=1).mean().to_numpy()
    msd = s.rolling(window, center=True, min_periods=1).std().fillna(0).to_numpy()

    img_raw  = _plot_series(series, _COLOR_RAW)
    img_mean = _plot_series(ma,     _COLOR_MEAN)
    img_std  = _plot_series(msd,    _COLOR_STD)

    return img_raw, img_mean, img_std


def normalise_to_int_range(
    series: np.ndarray,
    lo: float = 0.0,
    hi: float = 1000.0,
) -> np.ndarray:
    """
    Min-max scale series to [lo, hi] and round to integers.
    Matches the integer format used in the model's training prompts.
    Returns float array (values are rounded integers).
    """
    vmin, vmax = series.min(), series.max()
    if vmax == vmin:
        return np.full_like(series, (lo + hi) / 2.0, dtype=float)
    scaled = (series - vmin) / (vmax - vmin) * (hi - lo) + lo
    return np.round(scaled)


def subsample(series: np.ndarray, max_len: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniformly subsample *series* to at most *max_len* points.

    Returns
    -------
    (subsampled_series, original_indices)
    """
    n = len(series)
    if n <= max_len:
        return series, np.arange(n)
    idx = np.round(np.linspace(0, n - 1, max_len)).astype(int)
    idx = np.unique(idx)
    return series[idx], idx


def select_features(
    X: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """
    Return indices of the top-k features by variance.
    If k >= n_features, returns all feature indices in variance order.
    """
    n_feat = X.shape[1]
    k = min(k, n_feat)
    variances = X.var(axis=0)
    top_k = np.argsort(variances)[::-1][:k]
    return top_k
