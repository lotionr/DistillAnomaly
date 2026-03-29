#!/usr/bin/env python3
"""
ADBench Benchmark Evaluation — DistillAnomaly VLM Pipeline
===========================================================
Protocol: "Explainable Unsupervised Anomaly Detection with Random Forest"
          (BlackRock / arXiv 2504.16075)

Scoring:
  1. Load each ADBench Classical .npz dataset (X, y).
  2. Use the top REF_FRACTION of normal samples to build a normal centroid
     from the VLM's last hidden-state embedding.
  3. Score every sample as cosine distance from that centroid.
  4. Compute AUROC(y_true, scores).
  5. Report per-dataset AUROC and mean AUROC across all datasets.

Usage:
  python eval_adbench.py [--data-dir PATH] [--adapter-path PATH]
                         [--max-samples N] [--ref-fraction F]
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file as safe_load
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

# ── Offline mode: only use files already on disk ──────────────────────────────
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ── Defaults ──────────────────────────────────────────────────────────────────
_PROJ = Path(__file__).parent.resolve()

# Model: Qwen2.5-VL-3B from HF cache
_HF_CACHE = Path.home() / ".cache/huggingface/hub"
_QWEN25_SNAPSHOT = (
    _HF_CACHE
    / "models--Qwen--Qwen2.5-VL-3B-Instruct"
    / "snapshots"
    / "66285546d2b821cf421d4f5eb2576359d3770cd3"
)

# LoRA: most-trained adapter (ts3-20ep)
_DEFAULT_ADAPTER = (
    _PROJ / "train_VL" / "qwen2.5-vl-3b-lora-vl-ts3-20ep" / "checkpoint-1060"
)

_DEFAULT_DATA_DIR = _PROJ / "adbench_data" / "classical"
_RESULTS_FILE = _PROJ / "eval" / "results" / "adbench_auroc_results.csv"
_TMP_IMG = "/tmp/_da_row.png"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# LoRA merge (mirrors eval_outsample.py)
# ──────────────────────────────────────────────────────────────────────────────

def _named_linear_modules(model):
    return {
        n: m
        for n, m in model.named_modules()
        if hasattr(m, "weight") and m.weight is not None
    }


def _best_match(suffix, name2mod):
    cands = [n for n in name2mod if n.endswith(suffix)]
    if not cands:
        return None
    cands.sort(key=len, reverse=True)
    return cands[0]


def merge_lora_inplace(model, adapter_dir: Path):
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    lora_alpha = float(cfg.get("lora_alpha", 1))
    r_global = cfg.get("r", None)

    sd = safe_load(str(adapter_dir / "adapter_model.safetensors"))
    name2mod = _named_linear_modules(model)
    dev = next(model.parameters()).device

    merged = skipped = 0
    for a_key in [k for k in sd if k.endswith(".lora_A.weight")]:
        b_key = a_key[: -len(".lora_A.weight")] + ".lora_B.weight"
        if b_key not in sd:
            skipped += 1
            continue
        A = sd[a_key].to(device=dev, dtype=torch.float32)
        B = sd[b_key].to(device=dev, dtype=torch.float32)
        r = A.shape[0]
        scale = lora_alpha / float(r_global if r_global else r)

        cleaned = a_key[: -len(".lora_A.weight")]
        for pref in ("base_model.model.", "base_model.", "model.", ""):
            if cleaned.startswith(pref):
                cleaned = cleaned[len(pref) :]
                break

        mod_name = _best_match(cleaned, name2mod)
        if mod_name is None:
            skipped += 1
            continue
        linear = name2mod[mod_name]
        delta = torch.matmul(B, A) * scale
        if tuple(delta.shape) != tuple(linear.weight.shape):
            skipped += 1
            continue
        with torch.no_grad():
            linear.weight += delta.to(dtype=linear.weight.dtype)
        merged += 1

    log.info(f"LoRA merge: merged={merged} skipped={skipped}")


# ──────────────────────────────────────────────────────────────────────────────
# Even-grid helper (mirrors eval_outsample.py)
# ──────────────────────────────────────────────────────────────────────────────

def _even_grid(n: int):
    root = int(math.isqrt(n))
    for w in range(root, 1, -1):
        if n % w == 0:
            h = n // w
            if (h % 2 == 0) and (w % 2 == 0):
                return h, w
    for w in (2, 4, 6, 8):
        if n % w == 0 and (n // w) % 2 == 0:
            return n // w, w
    # fallback: pad to next perfect even square
    side = root + (root % 2)
    return side, side


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_path: Path, adapter_path: Path, device: str, skip_lora: bool = False):
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration as VLModel,
        )
    except ImportError:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VLForConditionalGeneration as VLModel,
        )

    DTYPE = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    log.info(f"Loading base model from {model_path}")
    model = VLModel.from_pretrained(
        str(model_path),
        torch_dtype=DTYPE,
        device_map=device,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if not skip_lora:
        log.info(f"Loading LoRA from {adapter_path}")
        merge_lora_inplace(model, adapter_path)
    else:
        log.info("LoRA disabled — base model only")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "right"

    image_processor = AutoImageProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    return model, tokenizer, image_processor


# ──────────────────────────────────────────────────────────────────────────────
# Image creation
# ──────────────────────────────────────────────────────────────────────────────

def row_to_pil(row: np.ndarray) -> Image.Image:
    """
    Plot a feature vector as a line chart and return a PIL Image.
    Uses an in-memory buffer (no disk I/O) and a compact figure size
    (2"×1.5" at 100 dpi → 200×150 px → fewer patches → faster inference).
    """
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot(row, color="#005088", linewidth=1.0)
    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ──────────────────────────────────────────────────────────────────────────────
# Embedding extraction — full model, batched
#
# The LoRA was trained as CAUSAL_LM targeting the language-model's q/k/v/o
# projections.  The vision encoder alone (model.model.visual) yields constant
# AUROC ≈ 0.5 for tabular data.  We need the full LM forward pass.
#
# Speed trick: all rows produce fixed-size images (2"×1.5", 100 dpi), so
# every sample has the same number of image patches → same sequence length →
# we can run true GPU batches through the full model.
# ──────────────────────────────────────────────────────────────────────────────

# Cache image dimensions on first call so we don't recompute the grid each time.
_IMG_CACHE: dict = {}


def _prepare_pv(raw_pv: torch.Tensor, merge: int):
    """Normalise raw pixel_values to (n_patches, hidden) and compute grid."""
    pv = raw_pv
    if pv.ndim == 3 and pv.shape[0] == 1:
        pv = pv.squeeze(0)
    elif pv.ndim == 4:
        pv = pv.view(-1, pv.shape[-1])
    n_patch, hid = pv.shape
    h_grid, w_grid = _even_grid(n_patch)
    target = h_grid * w_grid
    if target > n_patch:
        pv = torch.cat([pv, torch.zeros(target - n_patch, hid, dtype=pv.dtype)], 0)
    return pv, h_grid, w_grid


def _build_seg_tokens(h_grid: int, w_grid: int, merge: int,
                      img_patch_id: int, start_id, end_id) -> list[int]:
    n_feat = (h_grid // merge) * (w_grid // merge)
    seg: list[int] = []
    if start_id is not None:
        seg.append(start_id)
    seg.extend([img_patch_id] * int(n_feat))
    if end_id is not None:
        seg.append(end_id)
    return seg


def embed_batch(
    model,
    tokenizer,
    image_processor,
    rows: np.ndarray,
    device: str,
    merge: int,
    img_patch_id: int,
    start_id,
    end_id,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Embed *rows* through the full Qwen2.5-VL model (image + minimal text).
    Fixed image size → uniform patch count → true GPU batching.
    Returns tensor of shape (N, hidden_dim) on CPU.
    """
    # Tokenise the tiny text prompt once
    text_ids: list[int] = tokenizer.encode("Analyze.", add_special_tokens=True)

    all_embs: list[torch.Tensor] = []

    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        B = len(chunk)

        pv_list: list[torch.Tensor] = []
        thw_list: list[torch.Tensor] = []
        seg_tokens: list[int] | None = None  # same for every image in a batch

        for row in chunk:
            image = row_to_pil(row)
            out = image_processor(images=image, return_tensors="pt")
            pv, h_grid, w_grid = _prepare_pv(out["pixel_values"], merge)

            # Compute seg tokens once (all images identical size)
            if seg_tokens is None:
                seg_tokens = _build_seg_tokens(
                    h_grid, w_grid, merge, img_patch_id, start_id, end_id
                )

            pv_list.append(pv)
            thw_list.append(torch.tensor([1, h_grid, w_grid], dtype=torch.int64))

        # Concatenated pixel values for all images in the batch
        pixel_values = torch.cat(pv_list, dim=0).to(device=device, dtype=torch.float32)
        image_grid_thw = torch.stack(thw_list, dim=0).to(device)

        # Build input_ids: [seg_tokens | text_ids] repeated B times
        all_ids = (seg_tokens or []) + text_ids
        input_ids = torch.tensor([all_ids] * B, dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
            )
        # Mean-pool last hidden state over all token positions: (B, seq, hidden) → (B, hidden)
        embs = outputs.hidden_states[-1].mean(dim=1)  # (B, hidden)
        all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)  # (N, hidden)


# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(
    model,
    tokenizer,
    image_processor,
    X: np.ndarray,
    y: np.ndarray,
    device: str,
    ref_fraction: float,
    max_samples: int,
    merge: int,
    img_patch_id: int,
    start_id,
    end_id,
    batch_size: int,
) -> float:
    # Optional subsampling to keep runtime tractable
    if max_samples and len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        idx.sort()
        X, y = X[idx], y[idx]

    normal_idx = np.where(y == 0)[0]
    n_ref = max(5, int(len(normal_idx) * ref_fraction))
    ref_idx = normal_idx[:n_ref]

    def _embed(rows):
        return embed_batch(
            model, tokenizer, image_processor, rows, device,
            merge, img_patch_id, start_id, end_id, batch_size,
        )

    # Build normal centroid from reference normals
    ref_embs = _embed(X[ref_idx])
    centroid = ref_embs.mean(dim=0)  # (hidden,) on CPU

    # Score all samples
    all_embs = _embed(X)
    scores = 1.0 - F.cosine_similarity(all_embs, centroid.unsqueeze(0), dim=1)

    return roc_auc_score(y.tolist(), scores.tolist())


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ADBench AUROC benchmark — DistillAnomaly VLM")
    p.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR,
                   help="Directory containing ADBench Classical .npz files")
    p.add_argument("--model-path", type=Path, default=_QWEN25_SNAPSHOT,
                   help="Path to Qwen2.5-VL-3B base model")
    p.add_argument("--adapter-path", type=Path, default=_DEFAULT_ADAPTER,
                   help="Path to LoRA adapter checkpoint directory")
    p.add_argument("--results-file", type=Path, default=_RESULTS_FILE,
                   help="Output CSV path")
    p.add_argument("--ref-fraction", type=float, default=0.10,
                   help="Fraction of normal samples used to build centroid (default 0.10)")
    p.add_argument("--max-samples", type=int, default=2000,
                   help="Max samples per dataset; 0 = no limit (default 2000)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Images per GPU batch for vision encoder (default 32)")
    p.add_argument("--disable-lora", action="store_true",
                   help="Skip LoRA merge (evaluate base model only)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Validate paths ────────────────────────────────────────────────────────
    if not args.data_dir.is_dir():
        log.error(
            f"Data directory not found: {args.data_dir}\n"
            "Run:  bash download_adbench_classical.sh"
        )
        sys.exit(1)

    npz_files = sorted(args.data_dir.glob("*.npz"))
    if not npz_files:
        log.error(f"No .npz files found in {args.data_dir}")
        sys.exit(1)

    if not args.model_path.is_dir():
        log.error(f"Model path not found: {args.model_path}")
        sys.exit(1)

    if not args.disable_lora and not args.adapter_path.is_dir():
        log.error(f"Adapter path not found: {args.adapter_path}")
        sys.exit(1)

    log.info(f"Datasets   : {len(npz_files)} .npz files in {args.data_dir}")
    log.info(f"Model      : {args.model_path}")
    log.info(f"Adapter    : {'disabled' if args.disable_lora else args.adapter_path}")
    log.info(f"Ref frac   : {args.ref_fraction}")
    log.info(f"Max samples: {args.max_samples or 'unlimited'}")
    log.info(f"Batch size : {args.batch_size}")

    # ── Load model ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    model, tokenizer, image_processor = load_model(
        args.model_path,
        args.adapter_path,
        device,
        skip_lora=args.disable_lora,
    )

    merge = getattr(model.config.vision_config, "spatial_merge_size", 2)
    img_patch_id = model.config.image_token_id
    start_id = getattr(model.config, "vision_start_token_id", None)
    end_id = getattr(model.config, "vision_end_token_id", None)

    if args.disable_lora:
        log.info("LoRA disabled — running base model only")

    # ── Iterate datasets ──────────────────────────────────────────────────────
    results = []
    failed = []

    for npz_path in tqdm(npz_files, desc="Datasets", unit="ds"):
        ds_name = npz_path.stem
        try:
            data = np.load(npz_path, allow_pickle=True)
            X = data["X"].astype(np.float32)
            y = data["y"].astype(int).ravel()

            n_anom = int(y.sum())
            if n_anom == 0:
                raise ValueError("Dataset has no anomalies")
            if len(y) - n_anom == 0:
                raise ValueError("Dataset has no normal samples")

            log.info(
                f"[{ds_name}] samples={len(X)} features={X.shape[1]} "
                f"anomalies={n_anom} ({100*n_anom/len(X):.1f}%)"
            )

            auroc = evaluate_dataset(
                model, tokenizer, image_processor, X, y, device,
                ref_fraction=args.ref_fraction,
                max_samples=args.max_samples,
                merge=merge,
                img_patch_id=img_patch_id,
                start_id=start_id,
                end_id=end_id,
                batch_size=args.batch_size,
            )

            log.info(f"[{ds_name}] AUROC = {auroc:.4f}")
            results.append({
                "dataset": ds_name,
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_anomalies": n_anom,
                "auroc": round(auroc, 6),
            })

        except Exception as exc:
            log.error(f"[{ds_name}] FAILED — {exc}")
            failed.append({"dataset": ds_name, "error": str(exc)})

    # ── Save results ──────────────────────────────────────────────────────────
    if not results:
        log.error("No datasets evaluated successfully.")
        sys.exit(1)

    df = pd.DataFrame(results).sort_values("auroc", ascending=False)
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.results_file, index=False)
    log.info(f"Results saved to {args.results_file}")

    # ── Print summary table ───────────────────────────────────────────────────
    mean_auroc = df["auroc"].mean()
    W = 36

    print()
    print("=" * (W + 32))
    print(f"  {'Dataset':<{W}} {'Samples':>8}  {'Feat':>5}  {'Anom%':>6}  {'AUROC':>8}")
    print("-" * (W + 32))
    for _, row in df.iterrows():
        pct = 100 * row["n_anomalies"] / row["n_samples"]
        print(
            f"  {row['dataset']:<{W}} {int(row['n_samples']):>8}  "
            f"{int(row['n_features']):>5}  {pct:>5.1f}%  {row['auroc']:>8.4f}"
        )
    print("=" * (W + 32))
    print(f"  {'Mean AUROC':<{W}} {'':>8}  {'':>5}  {'':>6}  {mean_auroc:>8.4f}")
    print(f"  Evaluated: {len(df)} / {len(npz_files)} datasets")
    if failed:
        print(f"  Failed:    {len(failed)} datasets")
        for f in failed:
            print(f"    - {f['dataset']}: {f['error']}")
    print("=" * (W + 32))
    print(f"\n  Results CSV: {args.results_file}")


if __name__ == "__main__":
    main()
