"""PaddleOCR restaurant-bill extraction pipeline.

Run directly:
    python bill_pipeline.py data/images/1.png
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

from ocr_benchmark import extract_structured, group_into_lines, make_paddleocr, mean_or_none


ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def preprocess_image(source: str | Path) -> Path:
    """Produce a contrast-normalised image without changing the original file."""
    image = Image.open(source).convert("RGB")
    # Autocontrast helps faded thermal receipts while preserving colour images.
    image = ImageOps.autocontrast(image, cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(1.2)
    handle = tempfile.NamedTemporaryFile(prefix="bill_", suffix=".png", delete=False)
    handle.close()
    output = Path(handle.name)
    image.save(output, "PNG")
    return output


def validate_extraction(structured: dict, mean_confidence: float | None) -> dict:
    """Return validation flags; never silently claim an uncertain bill is valid."""
    fields = structured["fields"]
    financials = structured.get("financials") or {}
    checks = structured.get("checks") or {}
    flags: list[str] = []

    if fields.get("total") is None:
        flags.append("missing_payable_total")
    if not fields.get("date"):
        flags.append("missing_or_invalid_date")
    if not fields.get("bill_no"):
        flags.append("missing_bill_number")
    if fields.get("gstin") and checks.get("gstin_checksum_ok") is False:
        flags.append("gstin_checksum_failed")
    if mean_confidence is not None and mean_confidence < 0.70:
        flags.append("low_ocr_confidence")
    if not structured.get("items"):
        flags.append("no_line_items_extracted")

    subtotal, total = financials.get("subtotal"), financials.get("total")
    if subtotal is not None and total is not None:
        expected = subtotal - (financials.get("discount") or 0)
        expected += financials.get("service_charge") or 0
        expected += financials.get("tax_total") or 0
        expected += financials.get("round_off") or 0
        if abs(expected - total) > 1.0:
            flags.append("financial_total_does_not_reconcile")

    item_amounts = [item.get("amount") for item in structured.get("items") or []]
    if subtotal is not None and item_amounts and all(value is not None for value in item_amounts):
        if abs(sum(float(value) for value in item_amounts) - subtotal) > 1.0:
            flags.append("item_sum_does_not_match_subtotal")

    return {
        "valid": not flags,
        "review_required": bool(flags),
        "flags": flags,
        "mean_ocr_confidence": mean_confidence,
    }


class BillPipeline:
    """Load PaddleOCR once, then extract many bills efficiently."""

    def __init__(self):
        self.predict = make_paddleocr()

    def extract(self, image_path: str | Path) -> dict:
        source = Path(image_path)
        if source.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError(f"Unsupported image type: {source.suffix}")
        processed = preprocess_image(source)
        try:
            words = self.predict(str(processed))
        finally:
            processed.unlink(missing_ok=True)
        lines = group_into_lines(words)
        structured = extract_structured(lines)
        confidence = mean_or_none([line["conf"] for line in lines])
        return {
            "source": source.name,
            "ocr": {
                "line_count": len(lines),
                "mean_confidence": confidence,
                "lines": [{"text": line["text"], "confidence": line["conf"]} for line in lines],
            },
            "extraction": structured,
            "validation": validate_extraction(structured, confidence),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured data from one restaurant bill.")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    pipeline = BillPipeline()
    print(json.dumps(pipeline.extract(args.image), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
