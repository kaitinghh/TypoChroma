"""
Attack Success Evaluation for InternVL2/InternVL3

For each attacked image, the model is asked three YES/NO questions:
  (1) Clean image,    true color  -> establishes baseline (clean_acc)
  (2) Attacked image, true color  -> did the attack hide the real color?
  (3) Attacked image, wrong color -> did the attack inject the fake color?

Metrics reported (over rows where Q1 answered YES):
  clean_acc            = (1_YES) / total
  any_flip             = (2_NO)  / (1_YES)
  strong_targeted_flip = (3_YES & 2_NO) / (1_YES)
"""

import os
import re
import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ─── Config ───────────────────────────────────────────────────────────────────

CLEAN_CSV  = "<TO_INSERT>"
ATTACK_LOG = "<TO_INSERT>"
OUT_CSV    = "<TO_INSERT>"
MODEL_ID   = "OpenGVLab/InternVL3-8B"  # e.g. InternVL2-2B, InternVL2-26B
SPLIT      = "train"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Image pre-processing ─────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {(i, j)
         for n in range(min_num, max_num + 1)
         for i in range(1, n + 1)
         for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )

    best_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width  = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    resized = image.resize((target_width, target_height))

    tiles = []
    for i in range(best_ratio[0] * best_ratio[1]):
        row, col = divmod(i, best_ratio[0])
        box = (
            col * image_size,
            row * image_size,
            (col + 1) * image_size,
            (row + 1) * image_size,
        )
        tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))

    return tiles


def load_image(image_path: str, max_num: int = 12, input_size: int = 448) -> torch.Tensor:
    """Returns a float16 tensor of shape (N_tiles, 3, H, W)."""
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(image, max_num=max_num, image_size=input_size, use_thumbnail=True)
    return torch.stack([transform(t) for t in tiles]).to(torch.float16)

# ─── Model loading ────────────────────────────────────────────────────────────

print("Loading InternVL2 ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
).eval()
print("Model ready.")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def resolve_clean_path(file_name: str) -> str:
    return f"{SPLIT}2017/{file_name}"


def ask_yes_no(image_path: str, color: str, category: str) -> bool | None:
    """
    Asks the model: 'Is there a {color} {category}?'
    Returns True (YES), False (NO), or None (unexpected response).
    """
    category = CATEGORY_ALIASES.get(category, category)
    prompt   = f"Is there a {color} {category}?\nReply YES or NO only."

    pixel_values = load_image(image_path).to(model.device)
    response = model.chat(
        tokenizer,
        pixel_values,
        prompt,
        dict(max_new_tokens=5, do_sample=False),
    )

    text = response.strip().upper()
    print(text)

    if text.startswith("YES"):
        return True
    if text.startswith("NO"):
        return False
    return None


def compute_stats(df: pd.DataFrame, label: str) -> None:
    """Prints the three core metrics for a (pre-filtered valid) dataframe."""
    total = len(df)
    print(f"\n=== {label} ===")
    print(f"  N (valid rows): {total}")

    if total == 0:
        print("  No valid rows.")
        return

    baseline_df = df[df["ans_clean_correct"] == True]
    baseline_n  = len(baseline_df)

    clean_acc            = (df["ans_clean_correct"] == True).sum() / total
    any_flip             = (baseline_df["ans_attack_correct"] == False).sum() / baseline_n if baseline_n else float("nan")
    strong_targeted_flip = (
        (baseline_df["ans_attack_wrong"] == True) & (baseline_df["ans_attack_correct"] == False)
    ).sum() / baseline_n if baseline_n else float("nan")

    print(f"  clean_acc            : {clean_acc:.4f}  ({baseline_n}/{total} baseline YES)")
    print(f"  any_flip             : {any_flip:.4f}")
    print(f"  strong_targeted_flip : {strong_targeted_flip:.4f}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    clean_df  = pd.read_csv(CLEAN_CSV)
    attack_df = pd.read_csv(ATTACK_LOG)

    ann_to_cat = dict(zip(clean_df["ann_id"].astype(int), clean_df["category_name"]))

    records = []
    skipped = {"missing_attack": 0, "missing_clean": 0, "missing_cat": 0}

    for i, row in attack_df.iterrows():
        attacked_path = str(row["attacked_path"])
        file_name     = str(row["file_name"])

        # Validate attacked image exists
        if not os.path.exists(attacked_path):
            skipped["missing_attack"] += 1
            print(f"  [SKIP] Missing attacked image: {attacked_path}")
            continue

        # Extract ann_id and resolve category
        ann_match = re.search(r"_ann(\d+)", attacked_path)
        if ann_match is None or (ann_id := int(ann_match.group(1))) not in ann_to_cat:
            skipped["missing_cat"] += 1
            print(f"  [SKIP] Could not resolve category for: {attacked_path}")
            continue

        category    = ann_to_cat[ann_id]
        true_color  = str(row["true_color"]).strip().lower()
        wrong_color = str(row["wrong_color"]).strip().lower()
        clean_path  = resolve_clean_path(file_name)

        # Validate clean image exists
        if not os.path.exists(clean_path):
            skipped["missing_clean"] += 1
            continue

        # Query model
        try:
            ans_clean_correct  = ask_yes_no(clean_path,    true_color,  category)
            ans_attack_correct = ask_yes_no(attacked_path, true_color,  category)
            ans_attack_wrong   = ask_yes_no(attacked_path, wrong_color, category)
        except Exception as e:
            print(f"  [ERROR] row {i}: {e}")
            ans_clean_correct = ans_attack_correct = ans_attack_wrong = None

        records.append({
            "file_name":          file_name,
            "ann_id":             ann_id,
            "clean_path":         clean_path,
            "attacked_path":      attacked_path,
            "category":           category,
            "true_color":         true_color,
            "wrong_color":        wrong_color,
            "ans_clean_correct":  ans_clean_correct,
            "ans_attack_correct": ans_attack_correct,
            "ans_attack_wrong":   ans_attack_wrong,
            "valid": all(a is not None for a in [ans_clean_correct, ans_attack_correct, ans_attack_wrong]),
            "any_flip":             (ans_attack_correct is False) if ans_clean_correct else None,
            "strong_targeted_flip": (ans_attack_correct is False and ans_attack_wrong is True) if ans_clean_correct else None,
        })

        if i % 25 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  Processed {i}/{len(attack_df)}")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Saved per-row results to {OUT_CSV}")

    print("\nSkipped rows:")
    for reason, count in skipped.items():
        print(f"  {reason}: {count}")

    compute_stats(df[df["valid"]].copy(), "Aggregate Results (ALL valid)")


if __name__ == "__main__":
    main()
