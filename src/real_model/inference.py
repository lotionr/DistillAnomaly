"""
Qwen2.5-VL-3B + LoRA inference for time-series anomaly detection.

Mirrors the logic in eval/scripts/eval_outsample.py, but:
  - accepts in-memory PIL images instead of files on disk
  - uses ts3 mode (raw + moving_average + moving_std)
  - returns parsed anomaly intervals
"""

import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as safe_load

# ── offline mode: never phone home ────────────────────────────────────────────
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ──────────────────────────────────────────────────────────────────────────────
# LoRA merge helpers  (identical to eval_outsample.py)
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


def merge_lora_inplace(model, adapter_dir: Path) -> None:
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
                cleaned = cleaned[len(pref):]
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

    print(f"  LoRA merge: merged={merged} skipped={skipped}")


# ──────────────────────────────────────────────────────────────────────────────
# Grid helper (identical to eval_outsample.py)
# ──────────────────────────────────────────────────────────────────────────────

def _even_grid(n: int) -> tuple[int, int]:
    root = int(math.isqrt(n))
    for w in range(root, 1, -1):
        if n % w == 0:
            h = n // w
            if (h % 2 == 0) and (w % 2 == 0):
                return h, w
    for w in (2, 4, 6, 8):
        if n % w == 0 and (n // w) % 2 == 0:
            return n // w, w
    # fallback
    side = root + (root % 2)
    return side, side


# ──────────────────────────────────────────────────────────────────────────────
# JSON extraction helper
# ──────────────────────────────────────────────────────────────────────────────

def extract_best_json(text: str) -> str:
    """
    Try to extract a valid JSON object from *text*.
    Falls back to a regex that recovers start/end indices from a truncated response.
    """
    import re

    # First pass: bracket-matching (handles complete JSON)
    objs = []
    stack = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            try:
                s = text[start : i + 1]
                o = json.loads(s)
                objs.append((start, i, o))
            except Exception:
                pass
    for _, _, o in reversed(objs):
        if isinstance(o, dict) and "anomalies" in o:
            return json.dumps(o)
    if objs:
        return json.dumps(objs[-1][2])

    # Second pass: regex recovery for truncated JSON
    # Handles: {"anomalies":[{"start":52, "end":60, "description":"...
    m = re.search(r'"start"\s*:\s*(\d+)[^}]*"end"\s*:\s*(\d+)', text, re.DOTALL)
    if m:
        start_idx, end_idx = int(m.group(1)), int(m.group(2))
        return json.dumps({
            "anomalies": [{"start": start_idx, "end": end_idx, "description": "detected (truncated)"}]
        })

    # Third pass: check for explicit empty response
    if '"anomalies": []' in text or '"anomalies":[]' in text:
        return json.dumps({"anomalies": []})

    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Prompts (ts3 mode)
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a strict JSON generator. Respond with a single JSON object only. "
    "Do NOT call tools or functions. Do NOT use keys other than exactly those specified. "
    "If no anomaly exists, reply with {\"anomalies\": []} only."
)

PROMPT_TEMPLATE = (
    "Given the time series below and the following plot images: "
    "raw values, moving average, and moving standard deviation, "
    "determine whether there is an anomalous interval.\n"
    "If the series is entirely normal, return the empty JSON template. "
    "Otherwise, detect and describe the nature of the anomaly.\n\n"
    "Return ONLY a JSON object formatted exactly as follows, with no extra keys or text:\n\n"
    "Empty (no anomaly):\n"
    "{{  \"anomalies\": []}}\n\n"
    "Non-empty:\n"
    "{{\n  \"anomalies\": [\n"
    "    {{\"start\": <int>, \"end\": <int>, \"description\": <string>}}\n"
    "  ]\n}}\n\n"
    "Rules:\n"
    "* If there is no anomaly, use the empty array.\n"
    "* At most one anomaly per series.\n"
    "* Each description must be one short sentence.\n"
    "* Do NOT invent anomalies when none exist.\n\n"
    "Time series (len: {length}):\n"
    "# Format per line: timestamp, value, moving_average, moving_std\n"
    "{series}"
)


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────

class AnomalyDetectorVL:
    """
    Wraps Qwen2.5-VL-3B + LoRA for time-series anomaly detection.
    """

    def __init__(
        self,
        model_path: str | Path,
        adapter_path: str | Path,
        device: str = "auto",
        skip_lora: bool = False,
    ):
        from transformers import AutoImageProcessor, AutoTokenizer

        try:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLForConditionalGeneration as VLModel,
            )
        except ImportError:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                Qwen2VLForConditionalGeneration as VLModel,
            )

        model_path = Path(model_path)
        adapter_path = Path(adapter_path)

        DTYPE = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        print(f"  Loading base model: {model_path}")
        self.model = VLModel.from_pretrained(
            str(model_path),
            torch_dtype=DTYPE,
            device_map=device,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        if not skip_lora:
            print(f"  Merging LoRA: {adapter_path}")
            merge_lora_inplace(self.model, adapter_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        self.tokenizer.padding_side = "right"

        self.image_processor = AutoImageProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
        )

        self.device = next(self.model.parameters()).device
        self.merge = getattr(self.model.config.vision_config, "spatial_merge_size", 2)
        self.img_patch_id = self.model.config.image_token_id
        self.start_id = getattr(self.model.config, "vision_start_token_id", None)
        self.end_id = getattr(self.model.config, "vision_end_token_id", None)

        print("  Model ready.")

    # ── image preparation ──────────────────────────────────────────────────────

    def _prepare_image(self, img: Image.Image):
        """Returns (pv_normalised, thw_tensor, seg_tokens_list)."""
        out = self.image_processor(images=img, return_tensors="pt")
        pv = out["pixel_values"]
        if pv.ndim == 3 and pv.shape[0] == 1:
            pv = pv.squeeze(0)
        elif pv.ndim == 4:
            pv = pv.view(-1, pv.shape[-1])
        n_patch, hid = pv.shape
        h_grid, w_grid = _even_grid(n_patch)
        target = h_grid * w_grid
        if target > n_patch:
            pv = torch.cat(
                [pv, torch.zeros(target - n_patch, hid, dtype=pv.dtype)], 0
            )
        n_feat = (h_grid // self.merge) * (w_grid // self.merge)
        thw = torch.tensor([1, h_grid, w_grid], dtype=torch.int64)

        seg: list[int] = []
        if self.start_id is not None:
            seg.append(self.start_id)
        seg.extend([self.img_patch_id] * int(n_feat))
        if self.end_id is not None:
            seg.append(self.end_id)

        return pv, thw, seg

    # ── main inference ─────────────────────────────────────────────────────────

    def detect(
        self,
        series_int: np.ndarray,
        ma_int: np.ndarray,
        ms_int: np.ndarray,
        img_raw: Image.Image,
        img_mean: Image.Image,
        img_std: Image.Image,
        max_new_tokens: int = 128,
    ) -> dict:
        """
        Run the model on one time series.

        Parameters
        ----------
        series_int, ma_int, ms_int : integer-valued 1-D arrays of the same length
        img_raw, img_mean, img_std : corresponding PIL images

        Returns
        -------
        dict with key "anomalies" (list of {start, end, description})
        """
        n = len(series_int)

        # Build text series
        lines = []
        for t in range(n):
            lines.append(
                f"timestamp: {t}, value: {int(series_int[t])}, "
                f"moving_average: {int(ma_int[t])}, moving_std: {int(ms_int[t])}"
            )
        series_text = "\n".join(lines)

        prompt_txt = PROMPT_TEMPLATE.format(length=n, series=series_text)

        # Prepare images (ts3: raw + mean + std)
        images_pil = [img_raw, img_mean, img_std]
        pix_list, thw_list, seg_tokens = [], [], []
        for img in images_pil:
            pv, thw, seg = self._prepare_image(img)
            pix_list.append(pv)
            thw_list.append(thw)
            seg_tokens.extend(seg)

        pixel_values = torch.cat(pix_list, 0).to(device=self.device, dtype=torch.float32)
        image_grid_thw = torch.stack(thw_list, 0).to(self.device)

        # Build input_ids
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_txt},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        # transformers 5.x returns BatchEncoding; older versions return a raw tensor
        if hasattr(encoded, "input_ids"):
            chat_ids = encoded["input_ids"].to(self.device)
        else:
            chat_ids = encoded.to(self.device)

        pre_ids = torch.tensor([seg_tokens], dtype=torch.long, device=self.device)
        input_ids = torch.cat([pre_ids, chat_ids], dim=1)
        attn_mask = torch.ones_like(input_ids)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_len = input_ids.shape[1]
        decoded = self.tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        ).strip()

        raw_json = extract_best_json(decoded)
        if not raw_json:
            return {"anomalies": [], "_raw": decoded}

        try:
            result = json.loads(raw_json)
        except json.JSONDecodeError:
            return {"anomalies": [], "_raw": decoded}

        return result
