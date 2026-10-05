# Restaurant Bill OCR Benchmark

## Objective

Compare four OCR approaches on ten restaurant bills, select one using reproducible metrics, then use the chosen model in a structured bill-extraction pipeline.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python ocr_benchmark.py run --out results --viz
```

Tesseract also needs the Windows Tesseract program on `PATH`. The benchmark was run on CPU, so response time is hardware-dependent.

## Dataset and ground truth

The dataset contains ten PNG restaurant bills. It includes clean digital receipts and photographed, blurred thermal receipts. Each image has one JSON ground-truth file containing:

- Meaningful receipt transcription in `text` for CER, WER and token F1.
- Header fields: merchant, bill number, date, GSTIN and payable total.
- Line items: name, quantity and final row amount.

The text reference deliberately includes merchant, identifiers, item rows, discounts, taxes and totals, while omitting advertising boilerplate. This keeps text metrics relevant to the extraction task.

## Compared systems

| Model | Detector | Recogniser |
|---|---|---|
| Tesseract | Classical layout analysis | Tesseract LSTM |
| EasyOCR | CRAFT | CRNN |
| PaddleOCR | DBNet | SVTR/CRNN |
| EasyOCR + TrOCR | CRAFT | TrOCR transformer |

## Evaluation

Run the score command after changing ground truth or extraction rules; it reuses saved OCR output, so it does not re-run expensive models.

```powershell
python ocr_benchmark.py score
```

| Metric | Meaning | Used in final score |
|---|---|---:|
| CER | Character edit distance divided by reference characters; lower is better | Yes, 25% |
| WER | Word edit distance divided by reference words; lower is better | Reported |
| Token F1 | Bag-of-words overlap; robust to reading-order changes | Yes, 15% |
| Field accuracy | Exact/fuzzy correctness of restaurant, bill number, date, GSTIN and total | Yes, 35% |
| Extracted-item F1 | One-to-one parsed item-name-and-amount matching | Yes, 20% |
| Response time | Mean seconds per bill, after warm-up | Yes, 5% |

The item metric reports precision and recall as well as F1. This prevents an OCR line from being credited merely because it resembles an item: the final parsed JSON must contain the item name and amount, and extra items reduce precision.

## Final evaluation result

PaddleOCR is the selected model. It won every accuracy metric while completing all ten bills without an error.

| Model | Composite score | CER ↓ | WER ↓ | Token F1 ↑ | Field accuracy ↑ | Item F1 ↑ | Response time/bill ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| PaddleOCR | **80.5** | **0.327** | **0.474** | **75.7%** | **95.7%** | **93.8%** | 47.79 s |
| EasyOCR + TrOCR | 66.2 | 0.388 | 0.581 | 65.7% | 78.7% | 67.4% | 139.84 s |
| Tesseract | 50.7 | 0.483 | 0.710 | 52.1% | 53.2% | 31.8% | **0.28 s** |
| EasyOCR | 47.8 | 0.462 | 0.756 | 49.4% | 48.9% | 47.3% | 3.95 s |

PaddleOCR has the lowest CER, best Token F1, best field accuracy, best extracted-item F1 (93.8%), and best item recall (95.7%). Tesseract is much faster, but its accuracy gap is too large for the primary extraction pipeline. It remains a possible low-latency fallback where accuracy requirements are lower.

## Improvements implemented

- Added reference transcriptions for all ten bills; CER, WER and Token F1 will no longer be `n/a`.
- Added a compact-date pattern for OCR output such as `01/07/201700:36:22`.
- Changed line-item evaluation to one-to-one matching against parsed JSON, with item precision, recall and F1.
- Added a financial summary to each structured result: subtotal, discount, service charge, tax components (including service tax when present), tax total, round-off, payable total, whether total includes tax, and total excluding tax where identifiable.
- Kept raw OCR lines in outputs for visual error analysis and model debugging.

## Observed failure analysis

- Bill 2: PaddleOCR reads bill number `3/3` rather than `3/T/3`; the missing character is a recognition error in the source text.
- Bill 6: two items remain unmatched because a finger/low image quality obscures part of the first row and merges another item name with the following row.
- Bill 9: the receipt contains both `Bhagini` and `Sriganda Palace`; this is a brand-versus-branch ground-truth policy ambiguity, not a material OCR failure.
- Blurry or low-contrast photographed receipts: character substitutions and missed rows.
- Multi-line names: a detector may split an item name from its price row. The pipeline now joins nearby name fragments before parsing an item row.
- Dense tables: rate, quantity and amount can be read in the wrong order.
- Ambiguous merchant branding: a receipt can show a brand and branch name on separate lines.
- Discounts and split taxes: totals must be interpreted as payable amounts, not item subtotals.

## Selected-model pipeline

The production flow for the selected model is:

`Bill image -> image validation/preprocessing -> PaddleOCR -> line grouping -> structured fields/items/financials -> validation -> JSON output`

Validation should check GSTIN format/checksum, date format, item arithmetic, and whether the payable total reconciles with subtotal, discount, charges and taxes within a small rounding tolerance. A failed validation should return a review flag rather than silently claiming a valid extraction.

The implementation is [bill_pipeline.py](C:\Users\shara\Projects\files\bill_pipeline.py). Run it for one bill with:

```powershell
python bill_pipeline.py data/images/1.png
```

It prints structured JSON containing OCR lines, extraction results, financial values, and validation flags.

## Limitations

Ten bills is a demonstration benchmark, not enough for a broad production claim. Record the operating system, CPU and package versions with the final run because response times and OCR output can vary across environments.

## Reproducing the final result

The final artifacts are in `results/`. To recompute the metrics from saved OCR output after a ground-truth or parsing change, run:

```powershell
python ocr_benchmark.py score
```

To re-run every OCR engine from the original images, use `python ocr_benchmark.py run --out results --viz`. This is considerably slower because the CPU-only PaddleOCR and TrOCR models must process all ten bills again.
