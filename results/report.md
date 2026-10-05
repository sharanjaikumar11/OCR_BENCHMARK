# OCR benchmark report

Best model: **paddleocr**

Score weights: cer 0.25, token_f1 0.15, field_acc 0.35, item_f1 0.2, speed 0.05

| Rank | Model | Score | CER | WER | TokenF1 | Field acc | Item F1 | Item recall | Item precision | Sec/bill | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | paddleocr | 80.5 | 0.327 | 0.474 | 75.7% | 95.7% | 93.8% | 95.7% | 91.8% | 47.79 | 0 |
| 2 | easyocr_trocr | 65.9 | 0.388 | 0.581 | 65.7% | 78.7% | 66.0% | 66.0% | 66.0% | 139.84 | 0 |
| 3 | tesseract | 50.3 | 0.483 | 0.710 | 52.1% | 53.2% | 29.5% | 29.8% | 29.2% | 0.28 | 0 |
| 4 | easyocr | 47.3 | 0.462 | 0.756 | 49.4% | 48.9% | 44.9% | 46.8% | 43.1% | 3.95 | 0 |

## Per-field accuracy

| Model | restaurant_name | bill_no | date | gstin | total |
|---|---|---|---|---|---|
| paddleocr | 90.0% | 90.0% | 100.0% | 100.0% | 100.0% |
| easyocr_trocr | 70.0% | 80.0% | 90.0% | 57.1% | 90.0% |
| tesseract | 60.0% | 40.0% | 70.0% | 42.9% | 50.0% |
| easyocr | 60.0% | 30.0% | 70.0% | 28.6% | 50.0% |

## Best model per metric

- Lowest CER: paddleocr (0.327)
- Best token F1: paddleocr (0.757)
- Best field accuracy: paddleocr (0.957)
- Best extracted-item F1: paddleocr (0.938)
- Best item recall: paddleocr (0.957)
- Fastest: tesseract (0.275)