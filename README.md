# OCR benchmark for restaurant bills

Runs 4 OCR setups on your 10 bills, extracts structured data, scores each setup against your ground truth, and picks the best one.

## Project layout

```text
data/
  images/                 # 10 input bill images
  ground_truth/           # one reviewed JSON label per image
results/                  # final benchmark report and saved model outputs
ocr_benchmark.py          # benchmark, extraction rules, scoring and reports
compare.py                # inspect one model/bill against its ground truth
Diagnose.py               # list remaining field and item failures
bill_pipeline.py          # selected PaddleOCR extraction pipeline
PROJECT_REPORT.md         # project documentation and learning notes
```

| Name | Detection | Recognition |
|---|---|---|
| `tesseract` | Tesseract (classical) | Tesseract |
| `easyocr` | CRAFT | CRNN |
| `paddleocr` | DBNet | SVTR / CRNN |
| `easyocr_trocr` | CRAFT (EasyOCR) | TrOCR (transformer) |

The last one shows how to swap only the recogniser while keeping the same detector.

## 1. Setup

Easiest: Google Colab or Kaggle (free GPU, no local install trouble).

```bash
pip install -r requirements.txt
# Tesseract needs its own program:
#   Ubuntu/Colab: sudo apt install tesseract-ocr
#   Windows: install from the UB Mannheim build and add it to PATH
```

If one model fails to install, the script skips it and still runs the others.
PaddleOCR 2.x and 3.x are both handled, but check the PaddleOCR docs if the install gives trouble on your Python version.

## 2. Add your data

```
data/images/       bill_01.jpg ... bill_10.jpg
data/ground_truth/ bill_01.json ... bill_10.json
```

Create empty ground-truth templates:

```bash
python ocr_benchmark.py init-gt --images data/images --gt data/ground_truth
```

Fill each JSON by looking at the bill:

```json
{
  "text": "full text of the bill, line by line",
  "fields": {
    "restaurant_name": "Spice Garden Restaurant",
    "bill_no": "1234",
    "date": "12/03/2024",
    "gstin": "33ABCDE1234F1Z5",
    "total": 693.0
  },
  "items": [
    {"name": "Paneer Butter Masala", "qty": 2, "amount": 500},
    {"name": "Butter Naan", "qty": 4, "amount": 160}
  ]
}
```

Everything is optional, but this project now supplies `text` for every bill so CER, WER and token F1 are part of the model decision. The transcription covers meaningful receipt content (merchant, identifiers, item rows, taxes/discounts and payable total), not advertising boilerplate. Leave a field empty only when it is genuinely absent or ambiguous.

Dates are read day-first (12/03/2024 = 12 March 2024).

## 3. Run

```bash
python ocr_benchmark.py run --images data/images --gt data/ground_truth --out results --viz
python ocr_benchmark.py run --models tesseract paddleocr   # choose specific models
```

## 4. Outputs (in `results/`)

| File | What it is |
|---|---|
| `summary.csv`, `report.md` | Ranked table, CER/WER, per-field accuracy, item precision/recall/F1, and best model per metric |
| `per_bill.csv` | Scores for every model on every bill (find the hard bills) |
| `outputs/<model>/<bill>.json` | Raw lines plus structured extraction for each model |
| `best_model/structured_output.json` | Structured data for all bills from the winning model |
| `viz/<model>/<bill>.png` | Detected boxes drawn on the bill (with `--viz`) |

## 5. How the best model is chosen

Each model gets a score out of 100 from these metrics (weights are at the top of `ocr_benchmark.py`, `WEIGHTS`):

| Metric | Weight | Meaning |
|---|---|---|
| 1 - CER | 0.25 | Character accuracy of the whole text |
| Token F1 | 0.15 | Word overlap, ignores reading order |
| Field accuracy | 0.35 | Bill no, date, total, GSTIN, name correct |
| Extracted item F1 | 0.20 | Parsed item-name-and-amount quality; includes precision so extra/wrong extracted rows are penalised |
| Speed | 0.05 | Fastest model = 1.0 |

Metrics without ground truth are dropped and the weights are re-normalised. A model that crashes on a bill gets no credit for that bill. The highest score wins; ties go to the faster model. WER is shown for analysis but is not part of the composite because Token F1 already provides a word-level score.

## Financial output

Every structured result now includes a `financials` object. It separates `subtotal`, `discount`, `service_charge`, individual taxes (`cgst`, `sgst`, `igst`, `vat`), `tax_total`, `round_off`, and the payable `total`.

`total_includes_tax` is `true` only when a tax component was identified; it is `null` rather than guessed when OCR misses a tax line. When tax is identified, `total_excluding_tax` is also emitted. A discount is always returned as a positive printed amount; it has already been deducted from the payable total.

## 6. How to test detection and recognition separately

- Detection: run with `--viz` and look at the boxes. Missing or merged boxes mean a detection problem.
- Recognition: if the boxes look right but the text is wrong, the recogniser is at fault.
- Compare `easyocr` and `easyocr_trocr`: same detector, different recogniser, so any difference in CER comes from recognition.

## 7. Run the selected-model pipeline

PaddleOCR won the final benchmark (score 80.5/100, CER 0.327, field accuracy 95.7%, extracted-item F1 93.8%). Test one image locally:

```bash
python bill_pipeline.py data/images/1.png
```

The command prints OCR lines, structured `fields`, `items`, `financials`, and validation flags as JSON.

## 8. Add another model

Write a factory that loads the model once and returns `predict(image_path)` giving a list of `{"text", "box", "conf"}`, then register it:

```python
def make_mymodel():
    ...
    def predict(path):
        return [{"text": "...", "box": (x1, y1, x2, y2), "conf": 0.9}]
    return predict

MODELS["mymodel"] = make_mymodel
```

A vision-language model (for example Qwen2.5-VL) fits the same interface if you ask it for text with line positions, or you can score its JSON output separately.

## Notes and limits

- Field and item extraction uses simple regex rules, the same for every model, so the comparison is fair. Unusual bill layouts may need small changes in the rules (`extract_structured`).
- With only 10 bills, treat small score gaps as ties and look at `per_bill.csv` before trusting the winner.
- All four real models were run successfully on the included ten-bill dataset. Results remain specific to this small CPU-only benchmark; rerun after changing the images, ground truth, package versions, or extraction rules.
