"""List every wrong field and every missed item for one model, across all bills.

Usage:
  python diagnose.py paddleocr            (all bills)
  python diagnose.py paddleocr 4 5 7      (only these bills)

Needs ocr_benchmark.py in the same folder and results from a previous run.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import ocr_benchmark as ob

if len(sys.argv) < 2:
    sys.exit("Usage: python diagnose.py <model> [bill names...]")

model = sys.argv[1]
only = set(sys.argv[2:])


def related_lines(item, lines):
    """OCR lines that share a word with the item name or contain its amount."""
    toks = set(ob.tokens(item["name"]))
    try:
        amount = float(item["amount"])
    except (TypeError, ValueError):
        amount = None
    found = []
    for l in lines:
        has_word = bool(toks & set(ob.tokens(l["text"])))
        has_amount = amount is not None and any(
            abs(ob.to_float(m.group()) - amount) <= 0.01 for m in ob.MONEY.finditer(l["text"]))
        if has_word or has_amount:
            found.append(l["text"])
    return found[:4]


field_misses = Counter()
total_items_missed = 0

for img in ob.list_images("data/images"):
    stem = img.stem
    if only and stem not in only:
        continue
    out_file = Path(f"results/outputs/{model}/{stem}.json")
    gt_file = Path(f"data/ground_truth/{stem}.json")
    if not out_file.exists() or not gt_file.exists():
        continue
    out = json.loads(out_file.read_text(encoding="utf-8"))
    gt = json.loads(gt_file.read_text(encoding="utf-8"))
    lines = [{"text": l["text"], "conf": l.get("conf")} for l in out["lines"]]
    predicted_items = (out.get("structured") or {}).get("items") or []

    problems = []
    for name in ob.FIELD_NAMES:
        expected = (gt.get("fields") or {}).get(name)
        if expected in (None, ""):
            continue
        got = out["structured"]["fields"].get(name)
        if not ob.field_matches(name, expected, got):
            field_misses[name] += 1
            problems.append(f"  FIELD {name}: expected {expected!r}, got {got!r}")
    for item in gt.get("items") or []:
        if item.get("name") and not any(ob.item_matches(item, predicted) for predicted in predicted_items):
            total_items_missed += 1
            problems.append(f"  ITEM missed: {item['name']} (amount {item.get('amount')})")
            for t in related_lines(item, lines):
                problems.append(f"       OCR line: {t}")

    print(f"\n=== bill {stem}: {len(out['lines'])} OCR lines, "
          f"{'no problems' if not problems else 'problems below'} ===")
    for p in problems:
        print(p)

print("\nSUMMARY")
print("  wrong fields by type:", dict(field_misses) or "none")
print("  items missed in total:", total_items_missed)
