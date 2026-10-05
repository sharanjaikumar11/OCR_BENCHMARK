import json
import sys

if len(sys.argv) != 3:
    sys.exit("Usage: python compare.py <model> <bill_name>   e.g. python compare.py tesseract 1")

model, stem = sys.argv[1], sys.argv[2]
out = json.load(open(f"results/outputs/{model}/{stem}.json", encoding="utf-8"))
gt = json.load(open(f"data/ground_truth/{stem}.json", encoding="utf-8"))

print(f"Model: {model}   Bill: {stem}\n")
print(f"{'FIELD':18}{'GROUND TRUTH':26}PREDICTED")
for key, value in gt["fields"].items():
    print(f"{key:18}{str(value):26}{out['structured']['fields'].get(key)}")

print("\nGROUND-TRUTH ITEMS")
for item in gt.get("items", []):
    print("  ", item)

print("\nPREDICTED ITEMS")
for item in out["structured"]["items"]:
    print("  ", item)

print("\nPREDICTED FINANCIALS")
for key, value in (out["structured"].get("financials") or {}).items():
    print(f"  {key}: {value}")

print("\nOCR LINES (what the model read)")
for line in out["lines"]:
    print("  ", line["text"])
