# This script performs source filtering using the three criteria: (i) single-instance constraint, (ii) mask area thresholds and (iii) Color dominance filter.

from __future__ import annotations

import os
import csv
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image
from pycocotools import mask as maskUtils


# ----------------------------
# CONFIG
# ----------------------------
COCO_ROOT = ""          # e.g. "/path/to/coco2017"
SPLIT = "train"         # "train" or "val"

EXPORT_CSV = True
CSV_OUT_PATH = "<TO_INSERT>"

# Filters
EXCLUDE_ISCROWD = True
MIN_MASK_AREA_PX: Optional[float] = 2000.0   # Rule 3a
MIN_REL_AREA: Optional[float] = 0.05         # Rule 3b (>=5% of image)

SKIP_UNKNOWN_COLOR = True
COLOR_DOMINANCE_THRESH: Optional[float] = 0.90  # Rule 2


# ----------------------------
# Data class
# ----------------------------
@dataclass
class FilteredInstance:
    image_id: int
    file_name: str
    category_id: int
    category_name: str
    ann_id: int
    bbox: Tuple[float, float, float, float]
    area: float
    iscrowd: int
    segmentation_type: str
    canonical_color: Optional[str]
    color_dominance: float
    color_reason: str


# ----------------------------
# LVIS / COCO helpers
# ----------------------------
def load_coco_instances(coco_root: str, split: str) -> Dict[str, Any]:
    ann_path = os.path.join(coco_root, f"lvis_v1_{split}.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")
    with open(ann_path, "r", encoding="utf-8") as f:
        return json.load(f)


def segmentation_type(seg: Any) -> str:
    if isinstance(seg, list):
        return "polygon"
    if isinstance(seg, dict):
        return "rle"
    return "unknown"


def coco_segmentation_to_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = maskUtils.frPyObjects(segmentation, height, width)
        m = maskUtils.decode(rles)
        if m.ndim == 3:
            m = np.any(m, axis=2)
        return m.astype(bool)

    if isinstance(segmentation, dict):
        return maskUtils.decode(segmentation).astype(bool)

    raise ValueError("Unknown segmentation format")


def get_image_relpath(img: Dict[str, Any]) -> str:
    """
    LVIS image entries have coco_url like:
      http://images.cocodataset.org/train2017/000000391895.jpg

    Return:
      train2017/000000391895.jpg
    """
    coco_url = img["coco_url"]
    parts = coco_url.rstrip("/").split("/")
    folder = parts[-2]       # train2017 or val2017
    filename = parts[-1]
    return os.path.join(folder, filename)


# ----------------------------
# Color helpers
# ----------------------------
def hue_to_basic_color(hue_deg: float) -> str:
    h = hue_deg % 360.0
    if h >= 345 or h < 15:
        return "red"
    if 15 <= h < 45:
        return "orange"
    if 45 <= h < 75:
        return "yellow"
    if 75 <= h < 165:
        return "green"
    if 165 <= h < 255:
        return "blue"
    if 255 <= h < 300:
        return "purple"
    if 300 <= h < 345:
        return "pink"
    return "unknown"


def extract_canonical_color_from_mask(
    image_rgb: np.ndarray,
    mask_bool: np.ndarray,
    *,
    hue_bins: int = 36,
    min_sat: float = 0.20,
    min_val: float = 0.20,
    white_sat_max: float = 0.15,
    white_val_min: float = 0.85,
    black_val_max: float = 0.15,
    gray_sat_max: float = 0.15,
    gray_val_min: float = 0.20,
    gray_val_max: float = 0.85,
) -> Dict[str, Any]:

    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return {"color": None, "dominance": 0.0, "reason": "empty_mask"}

    pix = image_rgb[ys, xs].astype(np.uint8)
    hsv = cv2.cvtColor(pix[:, ::-1].reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

    h_deg = hsv[:, 0].astype(np.float32) * 2.0
    s = hsv[:, 1].astype(np.float32) / 255.0
    v = hsv[:, 2].astype(np.float32) / 255.0

    is_black = v <= black_val_max
    is_white = (s <= white_sat_max) & (v >= white_val_min)
    is_gray = (s <= gray_sat_max) & (v >= gray_val_min) & (v <= gray_val_max) & (~is_black) & (~is_white)
    chroma = (~is_black) & (~is_white) & (~is_gray) & (s >= min_sat) & (v >= min_val)

    counts: Dict[str, int] = {
        "black": int(is_black.sum()),
        "white": int(is_white.sum()),
        "gray": int(is_gray.sum()),
    }

    if chroma.any():
        hist, edges = np.histogram(h_deg[chroma], bins=hue_bins, range=(0, 360))
        bin_idx = np.clip(np.digitize(h_deg[chroma], edges) - 1, 0, hue_bins - 1)
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        for b in range(hue_bins):
            bucket = hue_to_basic_color(bin_centers[b])
            counts[bucket] = counts.get(bucket, 0) + int((bin_idx == b).sum())

    total = len(v)
    winner, win_count = max(counts.items(), key=lambda kv: kv[1])
    dominance = win_count / total

    return {
        "color": winner if winner != "unknown" else None,
        "dominance": float(dominance),
        "reason": "overall_purity",
    }


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    coco = load_coco_instances(COCO_ROOT, SPLIT)

    categories = coco["categories"]
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    target_ids = set(cat_id_to_name.keys())

    images = coco["images"]
    print(images[0].keys())

    img_id_to_info = {img["id"]: img for img in images}
    img_id_to_file = {
        img["id"]: os.path.basename(img["coco_url"])
        for img in images
    }
    img_id_to_relpath = {
        img["id"]: get_image_relpath(img)
        for img in images
    }

    anns = coco["annotations"]

    # ---------- Rule 1: instance count ----------
    instance_count = Counter()
    for a in anns:
        if EXCLUDE_ISCROWD and a.get("iscrowd", 0) == 1:
            continue
        instance_count[(int(a["image_id"]), int(a["category_id"]))] += 1

    img_cache: Dict[int, np.ndarray] = {}
    filtered: List[FilteredInstance] = []

    for a in anns:
        cid = int(a["category_id"])
        img_id = int(a["image_id"])

        if cid not in target_ids:
            continue
        if EXCLUDE_ISCROWD and a.get("iscrowd", 0) == 1:
            continue

        # Rule 1: exactly one instance of this category in the image
        if instance_count[(img_id, cid)] != 1:
            continue

        area = float(a.get("area", 0.0))
        if MIN_MASK_AREA_PX is not None and area < MIN_MASK_AREA_PX:
            continue

        img_info = img_id_to_info.get(img_id)
        file_name = img_id_to_file.get(img_id)
        relpath = img_id_to_relpath.get(img_id)
        if img_info is None or file_name is None or relpath is None:
            continue

        img_h, img_w = int(img_info["height"]), int(img_info["width"])
        if MIN_REL_AREA is not None and (area / (img_h * img_w)) < MIN_REL_AREA:
            continue

        if img_id not in img_cache:
            try:
                img_path = os.path.join(COCO_ROOT, relpath)
                img_cache[img_id] = np.array(Image.open(img_path).convert("RGB"))
            except Exception as e:
                print(f"Failed to load image {img_id} at {relpath}: {e}")
                continue

        try:
            mask = coco_segmentation_to_mask(a["segmentation"], img_h, img_w)
        except Exception as e:
            print(f"Failed to decode mask for ann {a.get('id')} in image {img_id}: {e}")
            continue

        color_res = extract_canonical_color_from_mask(img_cache[img_id], mask)
        canonical_color = color_res["color"]
        dominance = color_res["dominance"]

        # Rule 2
        if SKIP_UNKNOWN_COLOR and canonical_color is None:
            continue
        if COLOR_DOMINANCE_THRESH is not None and dominance < COLOR_DOMINANCE_THRESH:
            continue

        bbox = tuple(map(float, a.get("bbox", [0, 0, 0, 0])))

        filtered.append(
            FilteredInstance(
                image_id=img_id,
                file_name=file_name,
                category_id=cid,
                category_name=cat_id_to_name[cid],
                ann_id=int(a["id"]),
                bbox=bbox,
                area=area,
                iscrowd=int(a.get("iscrowd", 0)),
                segmentation_type=segmentation_type(a["segmentation"]),
                canonical_color=canonical_color,
                color_dominance=dominance,
                color_reason=color_res["reason"],
            )
        )

    print(f"Filtered instances: {len(filtered)}")

    if EXPORT_CSV:
        with open(CSV_OUT_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "image_id", "file_name", "ann_id", "category_id", "category_name",
                "bbox_x", "bbox_y", "bbox_w", "bbox_h", "area", "iscrowd",
                "segmentation_type", "canonical_color", "color_dominance", "color_reason"
            ])
            for fi in filtered:
                w.writerow([
                    fi.image_id, fi.file_name, fi.ann_id, fi.category_id,
                    fi.category_name, *fi.bbox, fi.area, fi.iscrowd,
                    fi.segmentation_type, fi.canonical_color,
                    f"{fi.color_dominance:.6f}", fi.color_reason
                ])

        print(f"✅ Wrote CSV: {CSV_OUT_PATH}")


if __name__ == "__main__":
    main()
