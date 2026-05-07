"""
Attack Success Evaluation for LLaVA-1.5

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
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ─── Config ───────────────────────────────────────────────────────────────────

CLEAN_CSV  = "<TO_INSERT>"
ATTACK_LOG = "<TO_INSERT>"
OUT_CSV    = "<TO_INSERT>"
MODEL_ID   = "llava-hf/llava-1.5-7b-hf"   # or llava-1.5-13b-hf
SPLIT      = "train"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Model loading ────────────────────────────────────────────────────────────

print("Loading LLaVA-1.5 ...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
).eval()
print("Model ready.")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def resolve_clean_path(file_name: str) -> str:
    return f"{SPLIT}2017/{file_name}"


def _render_prompt(prompt_text: str) -> str:
    """Formats the prompt for LLaVA, using chat template if available."""
    if hasattr(processor, "apply_chat_template"):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"USER: <image>\n{prompt_text}\nASSISTANT:"


def ask_yes_no(image_path: str, color: str, category: str) -> bool | None:
    """
    Asks the model: 'Is there a {color} {category}?'
    Returns True (YES), False (NO), or None (unexpected response).
    """
    category = CATEGORY_ALIASES.get(category, category)
    prompt   = f"Is there a {color} {category}?\nReply YES or NO only."

    image  = Image.open(image_path).convert("RGB")
    inputs = processor(text=_render_prompt(prompt), images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=0.0,
        )

    # Decode only the generated tokens, not the input prompt
    gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = processor.decode(gen_ids, skip_special_tokens=True).strip().upper()

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
