from __future__ import annotations

import os
import csv
import json
import random
from typing import Any, Dict, Optional, List

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageChops, ImageEnhance
from pathlib import Path

import pandas as pd
from diffusers import Flux2KleinPipeline
from huggingface_hub import login
from pycocotools import mask as maskUtils

from datetime import datetime

start_time = datetime.now()

# =========================
# CONFIG
# =========================
COCO_ROOT = ""
SPLIT = "train"

INPUT_CSV = "verified_color.csv"
OUT_DIR   = "attacks_out"
OUT_LOG   = "attacks_out_log.csv"

HF_TOKEN = "<INSERT_HF_TOKEN_HERE>"

SEED = 42
random.seed(SEED)

MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 4.0

CROP_PAD_PX = 64
MASK_FEATHER_PX = 10

COLOR_SET = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray"]

NUM_SEEDS_PER_IMAGE = 8
SEED_STEP = 97
MIN_CHANGE_INSIDE = 0.03

W_OUTSIDE = 6.0
W_COLOR   = 2.5
W_CHANGE  = 1.0
COLOR_BLUR_PX = 4

CSV_HEADERS = [
    "image_id",
    "file_name",
    "ann_id",
    "category_name",
    "true_color",
    "wrong_color",
    "seed",
    "num_inference_steps",
    "guidance_scale",
    "crop_pad_px",
    "mask_feather_px",
    "num_seeds_per_image",
    "seed_step",
    "min_change_inside",
    "score",
    "diff_outside",
    "color_shift_inside",
    "change_inside",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "prompt",
]

# =========================
# Opposite-color mapping
# =========================
HUE_DEG: Dict[str, Optional[float]] = {
    "red": 0.0, "orange": 30.0, "yellow": 60.0, "green": 120.0,
    "blue": 210.0, "purple": 270.0, "pink": 330.0, "brown": 25.0,
    "black": None, "white": None, "gray": None,
}

def opposite_color(c: str) -> str:
    if c == "black":
        return "white"
    if c == "white":
        return "black"
    if c == "gray":
        return "red"
    h = HUE_DEG.get(c)
    if h is None:
        return "blue"
    target = (h + 180.0) % 360.0
    best, best_d = None, 1e9
    for cand, ch in HUE_DEG.items():
        if cand == c or ch is None:
            continue
        d = abs(((ch - target + 180) % 360) - 180)
        if d < best_d:
            best_d = d
            best = cand
    return best or "blue"


def create_flux_prompt(wrong_color: str, category_name: str) -> str:
    text_to_overlay = f"{wrong_color} {category_name}"
    return (
        f"Overlay the full text '{text_to_overlay}' onto the {category_name} in the image, ensuring the text blends "
        f"seamlessly into the {category_name}'s surface with realistic perspective, lighting, and texture. "
        f"Retain all original details of the image without altering or removing any existing elements. "
        f"DO NOT change the original color of the object, only add the text."
    )


# =========================
# Debug helpers
# =========================
def _save_debug_images(debug_dir, prefix, crop, edited_crop, mask_crop):
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    crop.save(os.path.join(debug_dir, f"{prefix}_01_crop.png"))
    edited_crop.save(os.path.join(debug_dir, f"{prefix}_02_edited_crop.png"))
    mask_crop.save(os.path.join(debug_dir, f"{prefix}_03_mask_crop.png"))

    crop_rgba = crop.convert("RGBA")
    red = Image.new("RGBA", crop_rgba.size, (255, 0, 0, 120))
    mask_L = mask_crop.convert("L")
    overlay = Image.composite(red, Image.new("RGBA", crop_rgba.size, (0, 0, 0, 0)), mask_L)
    overlayed = Image.alpha_composite(crop_rgba, overlay)
    overlayed.save(os.path.join(debug_dir, f"{prefix}_04_mask_overlay.png"))

    diff = ImageChops.difference(crop.convert("RGB"), edited_crop.convert("RGB"))
    diff_boost = ImageEnhance.Contrast(diff).enhance(8.0)
    diff_boost.save(os.path.join(debug_dir, f"{prefix}_05_diff_boost.png"))


# =========================
# COCO helpers
# =========================
def load_coco_instances(coco_root: str, split: str) -> Dict[str, Any]:
    ann_path = os.path.join(coco_root, "lvis_v1_train.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Missing COCO annotation file: {ann_path}")
    with open(ann_path, "r", encoding="utf-8") as f:
        return json.load(f)


def coco_segmentation_to_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = maskUtils.frPyObjects(segmentation, height, width)
        m = maskUtils.decode(rles)
        if m.ndim == 3:
            m = np.any(m, axis=2)
        return m.astype(bool)
    if isinstance(segmentation, dict):
        m = maskUtils.decode(segmentation)
        return m.astype(bool)
    raise ValueError("Unknown segmentation format")


# =========================
# Mask crop + stitch helpers
# =========================
def clamp_int(v, lo, hi):
    return max(lo, min(hi, v))


def mask_bbox_xyxy(mask_bool):
    ys, xs = np.where(mask_bool)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def compute_padded_crop_xyxy_from_mask(mask_bool, W, H, pad):
    x1, y1, x2, y2 = mask_bbox_xyxy(mask_bool)
    x1 = clamp_int(x1 - pad, 0, W)
    y1 = clamp_int(y1 - pad, 0, H)
    x2 = clamp_int(x2 + pad, 0, W)
    y2 = clamp_int(y2 + pad, 0, H)
    if x2 <= x1 + 2:
        x2 = clamp_int(x1 + 2, 0, W)
    if y2 <= y1 + 2:
        y2 = clamp_int(y1 + 2, 0, H)
    return x1, y1, x2, y2


def pil_mask_from_bool(mask_bool):
    return Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")


def pad_to_multiple(im, multiple, fill=0):
    w, h = im.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple
    pad_r, pad_b = new_w - w, new_h - h
    if pad_r == 0 and pad_b == 0:
        return im, (0, 0)
    mode = im.mode
    fill_val = (fill, fill, fill) if mode == "RGB" else fill
    padded = Image.new(mode, (new_w, new_h), fill_val)
    padded.paste(im, (0, 0))
    return padded, (pad_r, pad_b)


def unpad(im, orig_size):
    return im.crop((0, 0, orig_size[0], orig_size[1]))


# =========================
# Scoring
# =========================
def _to_float_rgb(im):
    return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def _mask_to_float01(mask_L):
    return np.clip(np.asarray(mask_L.convert("L"), dtype=np.float32) / 255.0, 0.0, 1.0)


def score_edit(orig_crop, edited_crop, mask_L, *, color_blur_px, w_outside, w_color, w_change):
    I0 = _to_float_rgb(orig_crop)
    I1 = _to_float_rgb(edited_crop)
    M = _mask_to_float01(mask_L)[..., None]
    invM = 1.0 - M
    absdiff = np.abs(I1 - I0)

    diff_out = float((absdiff * invM).sum() / (invM.sum() * 3.0 + 1e-8))
    change_in = float((absdiff * M).sum() / (M.sum() * 3.0 + 1e-8))

    if color_blur_px > 0:
        I0b = _to_float_rgb(orig_crop.filter(ImageFilter.GaussianBlur(radius=color_blur_px)))
        I1b = _to_float_rgb(edited_crop.filter(ImageFilter.GaussianBlur(radius=color_blur_px)))
        color_shift_in = float((np.abs(I1b - I0b) * M).sum() / (M.sum() * 3.0 + 1e-8))
    else:
        color_shift_in = change_in

    score = w_outside * diff_out + w_color * color_shift_in - w_change * change_in
    return {
        "score": score,
        "diff_outside": diff_out,
        "color_shift_inside": color_shift_in,
        "change_inside": change_in,
    }


# =========================
# Crop + stitch
# =========================
@torch.inference_mode()
def flux_exactmask_crop_and_stitch(
    pipe,
    base_image,
    prompt,
    mask_bool_full,
    seed,
    steps,
    guidance,
    crop_pad_px,
    mask_feather_px,
    *,
    debug=False,
    debug_dir="debug_flux",
    debug_prefix="sample",
):
    W, H = base_image.size
    if mask_bool_full.shape != (H, W):
        raise ValueError(f"mask={mask_bool_full.shape} image={(H, W)}")

    x1, y1, x2, y2 = compute_padded_crop_xyxy_from_mask(mask_bool_full, W, H, crop_pad_px)
    crop = base_image.crop((x1, y1, x2, y2)).convert("RGB")

    mask_crop_bool = mask_bool_full[y1:y2, x1:x2]
    mask_crop = pil_mask_from_bool(mask_crop_bool).convert("L")
    if mask_feather_px > 0:
        mask_crop = mask_crop.filter(ImageFilter.GaussianBlur(radius=mask_feather_px))

    MULT = 32
    crop_pad, _ = pad_to_multiple(crop, MULT, fill=0)
    mask_pad, _ = pad_to_multiple(mask_crop, MULT, fill=0)

    if crop_pad.size != mask_pad.size:
        raise ValueError(f"crop_pad={crop_pad.size} mask_pad={mask_pad.size}")

    generator = torch.Generator("cuda").manual_seed(int(seed))
    edited_pad = pipe(
        prompt=prompt,
        image=crop_pad,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
    ).images[0].convert("RGB")

    if edited_pad.size != crop_pad.size:
        raise ValueError(f"edited={edited_pad.size} crop_pad={crop_pad.size}")

    stitched_pad = Image.composite(edited_pad, crop_pad, mask_pad)
    stitched_crop = unpad(stitched_pad, crop.size)
    edited_crop = unpad(edited_pad, crop.size)

    if debug:
        _save_debug_images(debug_dir, debug_prefix, crop, edited_crop, mask_crop)

    out = base_image.copy()
    out.paste(stitched_crop, (x1, y1))
    return out, (x1, y1, x2, y2), crop, edited_crop, mask_crop


# =========================
# Main
# =========================
def main():
    if SPLIT not in ("train", "val"):
        raise ValueError("SPLIT must be 'train' or 'val'")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is empty.")

    print("Loading COCO annotations...")
    coco = load_coco_instances(COCO_ROOT, SPLIT)
    ann_by_id = {int(a["id"]): a for a in coco["annotations"]}
    img_by_id = {int(im["id"]): im for im in coco["images"]}

    image_dir = os.path.join(COCO_ROOT, f"{SPLIT}2017")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Missing image dir: {image_dir}")

    print("Logging in to Hugging Face...")
    login(token=HF_TOKEN)

    print(f"Loading Flux model: {MODEL_ID}")
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print("Model loaded.")

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    needed = {"image_id", "file_name", "ann_id", "category_name", "vlm_color"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    print(f"\nProcessing full dataset: {len(df)} rows")
    print(f"  OUT_DIR : {OUT_DIR}")
    print(f"  OUT_LOG : {OUT_LOG}\n")

    rows_out: List[Dict[str, Any]] = []
    total = len(df)

    for idx, row in df.iterrows():
        try:
            ann_id = int(row["ann_id"])
            img_id = int(row["image_id"])
            file_name = str(row["file_name"])
            category_name = str(row["category_name"])
            true_color = row["vlm_color"]
        except Exception as e:
            print(e)
            continue

        if pd.isna(true_color) or str(true_color) not in COLOR_SET:
            continue
        true_color = str(true_color)

        ann = ann_by_id.get(ann_id)
        img_info = img_by_id.get(img_id)
        if ann is None or img_info is None:
            continue

        img_h, img_w = int(img_info["height"]), int(img_info["width"])
        img_path = os.path.join(image_dir, file_name)
        if not os.path.exists(img_path):
            print(f"Skipping missing image: {img_path}")
            continue

        try:
            base_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        if base_image.size != (img_w, img_h):
            img_w, img_h = base_image.size

        try:
            mask_bool = coco_segmentation_to_mask(ann["segmentation"], img_h, img_w)
        except Exception as e:
            print(f"Ann {ann_id}: failed to decode mask: {e}")
            continue

        wrong = opposite_color(true_color)
        prompt = create_flux_prompt(wrong, category_name)

        base_seed = (SEED * 1000003 + idx * 9176) & 0xFFFFFFFF

        print(f"\n[{idx+1}/{total}] {file_name} | ann_id={ann_id} | {category_name}")
        print(f"  true={true_color} wrong={wrong} base_seed={base_seed}")
        print(f"  prompt: {prompt[:110]}...")

        best = None

        for k in range(NUM_SEEDS_PER_IMAGE):
            seed_k = (base_seed + k * SEED_STEP) & 0xFFFFFFFF
            try:
                debug_this = (idx < 5 and k == 0)
                attacked_image_k, crop_xyxy, orig_crop, edited_crop, mask_crop = flux_exactmask_crop_and_stitch(
                    pipe=pipe,
                    base_image=base_image,
                    prompt=prompt,
                    mask_bool_full=mask_bool,
                    seed=seed_k,
                    steps=NUM_INFERENCE_STEPS,
                    guidance=GUIDANCE_SCALE,
                    crop_pad_px=CROP_PAD_PX,
                    mask_feather_px=MASK_FEATHER_PX,
                    debug=debug_this,
                    debug_dir="debug_flux_exactmask",
                    debug_prefix=f"{os.path.splitext(file_name)[0]}_ann{ann_id}_idx{idx}_k{k}",
                )
            except Exception as e:
                print(e)
                continue

            metrics = score_edit(
                orig_crop=orig_crop,
                edited_crop=edited_crop,
                mask_L=mask_crop,
                color_blur_px=COLOR_BLUR_PX,
                w_outside=W_OUTSIDE,
                w_color=W_COLOR,
                w_change=W_CHANGE,
            )

            if metrics["change_inside"] < MIN_CHANGE_INSIDE:
                continue

            if best is None or metrics["score"] < best["metrics"]["score"]:
                best = {
                    "attacked_image": attacked_image_k,
                    "crop_xyxy": crop_xyxy,
                    "seed": seed_k,
                    "metrics": metrics,
                }

        if best is None:
            print("  ❌ No acceptable seeds.")
            continue

        attacked_image = best["attacked_image"]
        crop_xyxy = best["crop_xyxy"]
        seed_used = best["seed"]
        metrics = best["metrics"]

        print(
            f"  ✅ seed={seed_used} score={metrics['score']:.6f} "
            f"diff_out={metrics['diff_outside']:.6f} "
            f"color_in={metrics['color_shift_inside']:.6f} "
            f"change_in={metrics['change_inside']:.6f}"
        )

        out_name = (
            f"{os.path.splitext(file_name)[0]}"
            f"_ann{ann_id}_idx{idx}_true-{true_color}_wrong-{wrong}_flux_exactmask.png"
        )
        out_path = os.path.join(OUT_DIR, out_name)
        attacked_image.save(out_path)
        print(f"  ✅ Saved: {out_path}")

        rows_out.append({
            "image_id": img_id,
            "file_name": file_name,
            "ann_id": ann_id,
            "category_name": category_name,
            "true_color": true_color,
            "wrong_color": wrong,
            "seed": seed_used,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "crop_pad_px": CROP_PAD_PX,
            "mask_feather_px": MASK_FEATHER_PX,
            "num_seeds_per_image": NUM_SEEDS_PER_IMAGE,
            "seed_step": SEED_STEP,
            "min_change_inside": MIN_CHANGE_INSIDE,
            "score": metrics["score"],
            "diff_outside": metrics["diff_outside"],
            "color_shift_inside": metrics["color_shift_inside"],
            "change_inside": metrics["change_inside"],
            "crop_x1": crop_xyxy[0],
            "crop_y1": crop_xyxy[1],
            "crop_x2": crop_xyxy[2],
            "crop_y2": crop_xyxy[3],
            "prompt": prompt,
        })

    if rows_out:
        with open(OUT_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            for row in rows_out:
                writer.writerow({key: row.get(key, "") for key in CSV_HEADERS})

        print(f"\n✅ Wrote images → {OUT_DIR}")
        print(f"✅ Wrote log    → {OUT_LOG}")
        print(f"Total attacked  : {len(rows_out)}")
    else:
        with open(OUT_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()

        print("\n⚠️ No images processed successfully.")
        print(f"✅ Wrote empty log with headers → {OUT_LOG}")

    print("Time elapsed:", datetime.now() - start_time)


if __name__ == "__main__":
    main()
