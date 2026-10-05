#!/usr/bin/env python3
"""
OCR benchmark for restaurant bills.

What it does
  1. Runs several OCR models on every bill image.
  2. Groups the detected text into lines and extracts structured data
     (restaurant name, bill no, date, GSTIN, total, line items).
  3. Scores every model against your hand-made ground truth.
  4. Ranks the models with a weighted score and picks the best one.

Usage
  python ocr_benchmark.py init-gt --images data/images --gt data/ground_truth
  (fill in the ground-truth JSON files by hand)
  python ocr_benchmark.py run --images data/images --gt data/ground_truth --out results
  python ocr_benchmark.py run ... --models tesseract easyocr --viz
  python ocr_benchmark.py score      (re-rank saved outputs without re-running the models)
"""
import argparse
import csv
import json
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# How much each metric counts in the final score. Change these to match what
# matters for your use case. Metrics with no ground truth are skipped and the
# remaining weights are re-normalised automatically.
WEIGHTS = {
    "cer": 0.25,          # character accuracy of the full text (1 - CER)
    "token_f1": 0.15,     # word overlap, ignores reading order
    "field_acc": 0.35,    # bill no, date, total, GSTIN, name
    "item_f1": 0.20,      # extracted line items: name + amount, penalises extras
    "speed": 0.05,        # fastest model = 1.0
}

FIELD_NAMES = ["restaurant_name", "bill_no", "date", "gstin", "total"]

# These are kept separate from ``fields`` because a bill can have more than
# one tax/charge and a discount is not a header field.  ``total_includes_tax``
# makes the monetary meaning explicit for API users.
FINANCIAL_KEYS = {
    "subtotal": (r"\bsub\s*-?\s*total\b", r"\bs[uo]t\s*rotal\b"),
    "discount": (r"\bdiscount\b", r"\bdisc\b"),
    "service_charge": (r"\bservice\s*(?:charge|chrg)\w*", r"\bserv\s*chrg\w*"),
    "service_tax": (r"\bservice\s*tax\b",),
    "vat": (r"\bvat\b",),
    "cgst": (r"\bcgst\b", r"\bc\s*gst\b", r"\bdsst\b"),
    "sgst": (r"\bsgst\b", r"\bs\s*gst\b"),
    "igst": (r"\bigst\b",),
    "round_off": (r"\bround\s*off\b",),
}


# --------------------------------------------------------------------------
# Geometry and line grouping
# --------------------------------------------------------------------------
def to_xyxy(box):
    """Accept [x1,y1,x2,y2] or a list of (x,y) points and return (x1,y1,x2,y2)."""
    pts = list(box)
    if len(pts) == 4 and not hasattr(pts[0], "__len__"):
        x1, y1, x2, y2 = pts
        return float(x1), float(y1), float(x2), float(y2)
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def group_into_lines(words):
    """Turn word/phrase boxes into text lines in reading order."""
    words = [w for w in words if w["text"].strip()]
    if not words:
        return []
    med_h = statistics.median(w["box"][3] - w["box"][1] for w in words) or 10.0
    yc = lambda w: (w["box"][1] + w["box"][3]) / 2
    words = sorted(words, key=yc)

    groups, cur = [], [words[0]]
    for w in words[1:]:
        cur_y = sum(yc(x) for x in cur) / len(cur)
        if abs(yc(w) - cur_y) <= 0.6 * med_h:
            cur.append(w)
        else:
            groups.append(cur)
            cur = [w]
    groups.append(cur)

    lines = []
    for g in groups:
        g = sorted(g, key=lambda w: w["box"][0])
        confs = [w["conf"] for w in g if w.get("conf") is not None]
        lines.append({
            "text": " ".join(w["text"].strip() for w in g),
            "box": (min(w["box"][0] for w in g), min(w["box"][1] for w in g),
                    max(w["box"][2] for w in g), max(w["box"][3] for w in g)),
            "conf": sum(confs) / len(confs) if confs else None,
        })
    return lines


# --------------------------------------------------------------------------
# OCR model wrappers. Each factory loads the model once and returns
# predict(image_path) -> list of {"text", "box", "conf"}
# --------------------------------------------------------------------------
def _use_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def make_tesseract():
    """Classical engine. Baseline."""
    import pytesseract
    from PIL import Image
    pytesseract.get_tesseract_version()  # raises if the binary is not installed

    def predict(path):
        d = pytesseract.image_to_data(
            Image.open(path), config="--psm 6",
            output_type=pytesseract.Output.DICT)
        words = []
        for i, t in enumerate(d["text"]):
            t = t.strip()
            try:
                conf = float(d["conf"][i])
            except (TypeError, ValueError):
                conf = -1
            if t and conf >= 0:
                x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
                words.append({"text": t, "box": (x, y, x + w, y + h), "conf": conf / 100})
        return words
    return predict


def make_easyocr():
    """CRAFT detection + CRNN recognition."""
    import easyocr
    reader = easyocr.Reader(["en"], gpu=_use_gpu(), verbose=False)

    def predict(path):
        return [{"text": t, "box": to_xyxy(b), "conf": float(c)}
                for b, t, c in reader.readtext(path)]
    return predict


def make_paddleocr():
    """DBNet detection + SVTR/CRNN recognition. Handles PaddleOCR 2.x and 3.x."""
    from paddleocr import PaddleOCR
    new_api = hasattr(PaddleOCR, "predict")  # 3.x has predict(), 2.x does not
    if new_api:
        ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                        use_doc_unwarping=False, use_textline_orientation=True,
                        enable_mkldnn=False)  # avoids the oneDNN "PIR" crash on some CPUs
    else:
        ocr = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)

    def predict(path):
        words = []
        if new_api:
            for r in ocr.predict(path):
                texts = r["rec_texts"]
                scores = r["rec_scores"]
                boxes = None
                for key in ("rec_polys", "rec_boxes", "dt_polys"):
                    try:
                        boxes = r[key]
                        break
                    except KeyError:
                        continue
                for t, s, b in zip(texts, scores, boxes):
                    words.append({"text": t, "box": to_xyxy(b), "conf": float(s)})
        else:
            res = ocr.ocr(path, cls=True)
            for item in (res[0] or []):
                box, (t, s) = item
                words.append({"text": t, "box": to_xyxy(box), "conf": float(s)})
        return words
    return predict


def make_easyocr_trocr():
    """Mix and match: EasyOCR (CRAFT) finds the text, TrOCR reads each crop.
    This is the way to test 'same detector, different recogniser'."""
    import easyocr
    import numpy as np
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reader = easyocr.Reader(["en"], gpu=(device == "cuda"), verbose=False)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-printed").to(device).eval()

    def predict(path):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        horizontal, free = reader.detect(np.array(img))
        boxes = [(x0, y0, x1, y1) for x0, x1, y0, y1 in horizontal[0]]
        boxes += [to_xyxy(p) for p in free[0]]
        clean = []
        for x0, y0, x1, y1 in boxes:
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(W, int(x1)), min(H, int(y1))
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                clean.append((x0, y0, x1, y1))
        crops = [img.crop(b) for b in clean]
        texts = []
        for i in range(0, len(crops), 16):
            pixel = processor(images=crops[i:i + 16], return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                ids = model.generate(pixel, max_new_tokens=64)
            texts += processor.batch_decode(ids, skip_special_tokens=True)
        return [{"text": t, "box": b, "conf": None} for t, b in zip(texts, clean)]
    return predict


MODELS = {
    "tesseract": make_tesseract,
    "easyocr": make_easyocr,
    "paddleocr": make_paddleocr,
    "easyocr_trocr": make_easyocr_trocr,
}


# --------------------------------------------------------------------------
# Structured extraction (same rules for every model, so the comparison is fair)
# --------------------------------------------------------------------------
MONEY = re.compile(
    r"(?<![\w/.-])(?:\d{1,2}(?:,\d{2})*,\d{3}|\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?(?![\w/-])")
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
DATE_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
# The look-ahead also accepts an OCR-merged time, e.g. "01/07/201700:36".
DATE_NUM = re.compile(r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})(?=\D|$|\d{2}:)")
DATE_TXT = re.compile(r"\b(\d{1,2})[\s\-]([A-Za-z]{3,9})[\s\-,]*(\d{2,4})\b")
GSTIN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b")
BILL_NO = re.compile(
    r"(?:bill|invoice|inv|receipt|check|order)\s*(?:no|number|num|#)?\s*[:.#\-]*\s*"
    r"([A-Za-z0-9][A-Za-z0-9\-/]*)", re.I)
# OCR often reads "Bill No" as "BI NO" or "B1ll No"
BILL_NO_FUZZY = re.compile(r"\bb[il1t]{1,4}\s*n[o0]\.?\s*[:#\-]*\s*([A-Za-z0-9][A-Za-z0-9\-/]*)", re.I)
TOTAL_KEYS = [r"grand\s*total", r"net\s*(?:amount|payable|total)",
              r"amount\s*(?:payable|due)", r"bill\s*am(?:oun)?t", r"\bnett?\b", r"amount\s*incl",
              r"total\s*amount", r"total\s*invoice",
              r"bill\s*total", r"\btotal\b"]
TOTAL_SKIP = re.compile(r"sub\s*-?\s*total|total\s*(?:qty|quantity|items|tax|gst)|cgst|sgst|igst", re.I)
NON_ITEM = re.compile(
    r"total|tax|gst|vat|service\s*charge|discount|round|change|cash|card|upi|paid|balance|"
    r"bill\s*no|invoice|date|time|table|phone|tel|mob|fssai|qty|item|amount|thank|visit|"
    r"road|covers|chrg|\bd?sst\b|\bgsi\b|\btin\s*no\b|rupees|again|please|user", re.I)
SKIP_NAME = re.compile(r"^(tax\s*invoice|invoice|bill|cash\s*(?:bill|memo)|receipt|gst|welcome|thank)", re.I)


GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}
_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}


def gstin_checksum_ok(g):
    """GSTIN's last character is a checksum of the first 14, so we can verify it."""
    if len(g) != 15 or any(c not in GSTIN_CHARS for c in g):
        return False
    total = 0
    for i, ch in enumerate(g[:14]):
        v = GSTIN_CHARS.index(ch) * (1 if i % 2 == 0 else 2)
        total += v // 36 + v % 36
    return GSTIN_CHARS[(36 - total % 36) % 36] == g[14]


def _fix_gstin_positions(c):
    """Positions 1-2, 8-11 are digits; 3-7 and 12 are letters; 14 is 'Z'."""
    ch = list(c)
    for i in (0, 1, 7, 8, 9, 10):
        ch[i] = _TO_DIGIT.get(ch[i], ch[i])
    for i in (2, 3, 4, 5, 6, 11):
        ch[i] = _TO_LETTER.get(ch[i], ch[i])
    if ch[13] == "2":
        ch[13] = "Z"
    return "".join(ch)


def find_gstin(texts):
    """Find a GSTIN even when OCR swaps look-alike characters (O/0, I/1, S/5...).
    A candidate whose checksum is valid is preferred."""
    structural = None
    for t in texts:
        flat = re.sub(r"[^0-9A-Z]", "", t.upper())
        for i in range(len(flat) - 14):
            cand = _fix_gstin_positions(flat[i:i + 15])
            if GSTIN.fullmatch(cand):
                if gstin_checksum_ok(cand):
                    return cand
                structural = structural or cand
    return structural


def to_float(s):
    return float(s.replace(",", ""))


def parse_date(text):
    """Return YYYY-MM-DD (day-first) or None."""
    iso = DATE_ISO.search(text)
    if iso:
        y, mo, d = (int(g) for g in iso.groups())
        return f"{y:04d}-{mo:02d}-{d:02d}" if 1 <= d <= 31 and 1 <= mo <= 12 else None
    m = DATE_NUM.search(text)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if mo > 12 >= d:
            d, mo = mo, d
    else:
        m = DATE_TXT.search(text)
        if not m or m.group(2)[:3].lower() not in MONTHS:
            return None
        d, mo, y = int(m.group(1)), MONTHS[m.group(2)[:3].lower()], int(m.group(3))
    if y < 100:
        y += 2000
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def guess_restaurant(texts):
    for t in texts[:6]:
        letters = sum(c.isalpha() for c in t)
        compact = len(t.replace(" ", "")) or 1
        if letters >= 3 and letters / compact >= 0.6 and not SKIP_NAME.match(t.strip()):
            return t.strip()
    return None


LEAD_QTY = re.compile(r"^\s*(\d{1,2}|[Il|])\s+(?=[A-Za-z])")  # "3 MINERAL WATER 150.00"
END_AMOUNT = re.compile(r"\d[\d,]*(?:\.\d{1,2})?\s*$")


def extract_items(texts):
    """Item rows look like 'name qty rate amount', 'name rate qty amount' or 'qty name amount'."""
    items = []
    for index, original in enumerate(texts):
        t = original
        if NON_ITEM.search(t) or ":" in t or "@" in t:
            continue
        if not END_AMOUNT.search(t):  # an item line ends with its amount
            continue
        # A detector can split a long item name into lines while keeping the
        # quantity/amount on the last line. Join up to three preceding name
        # fragments, stopping at a completed row or a non-item/header line.
        prefixes = []
        for previous in reversed(texts[max(0, index - 3):index]):
            if NON_ITEM.search(previous) or ":" in previous or "@" in previous:
                break
            if END_AMOUNT.search(previous):
                break
            prefixes.append(previous)
        if prefixes:
            t = " ".join(list(reversed(prefixes)) + [t])
        lead_qty, body = None, t
        m = LEAD_QTY.match(t)
        if m:
            lead_qty = 1.0 if m.group(1) in "Il|" else float(m.group(1))
            body = t[m.end():]
        nums = list(MONEY.finditer(body))
        if not nums:
            continue
        vals = [to_float(x.group()) for x in nums]
        amount = vals[-1]
        qty = rate = None
        name_end = nums[0].start()
        if len(vals) >= 3 and abs(vals[-3] * vals[-2] - amount) <= max(0.5, 0.01 * amount):
            # the last three numbers are qty, rate and amount (in either order), so
            # any earlier number belongs to the name, e.g. "CHICKEN 65"
            qty, rate = min(vals[-3], vals[-2]), max(vals[-3], vals[-2])
            name_end = nums[-3].start()
        elif len(vals) >= 2 and vals[-2] == int(vals[-2]) and vals[-2] <= 50:
            # E.g. "100 PIPERS (30 ML) 5 1275": the number in parentheses
            # belongs to the name, while the penultimate number is quantity.
            qty = vals[-2]
            name_end = nums[-2].start()
        elif len(vals) >= 3:
            qty, rate = vals[0], vals[-2]
        elif len(vals) == 2:
            if vals[0] == int(vals[0]) and vals[0] <= 50:
                qty = vals[0]
            else:
                rate = vals[0]
        name = body[:name_end].strip(" .:-|\u20b9")
        if sum(c.isalpha() for c in name) < 3:
            continue
        if qty is None and lead_qty is not None:
            qty = lead_qty
        items.append({"name": name, "qty": qty, "rate": rate, "amount": amount})
    return items


def apply_round_off(total, following_lines):
    """Many bills print 'Total 418.90' and then the rounded amount 419 on the next line.
    The rounded figure is what the customer pays, so prefer it."""
    for t in following_lines:
        for m in MONEY.finditer(t):
            v = to_float(m.group())
            if v != total and v == int(v) and abs(v - total) <= 1.0:
                return v
    return total


def extract_financials(texts, total):
    """Extract bill components without silently treating a total as tax-inclusive.

    A component can be on the line after its label (a common receipt layout),
    therefore the immediately following numeric-only line is considered too.
    Values are the printed positive amounts; ``discount`` is not negated.
    """
    result = {key: None for key in FINANCIAL_KEYS}
    for key, patterns in FINANCIAL_KEYS.items():
        for idx, text in enumerate(texts):
            if not any(re.search(p, text, re.I) for p in patterns):
                continue
            nums = [to_float(m.group()) for m in MONEY.finditer(text)]
            # Do not mistake a tax percentage for the tax amount if it was
            # printed on the next line.
            if idx + 1 < len(texts) and (not nums or "%" in text):
                nxt = texts[idx + 1]
                if not re.search(r"[A-Za-z]", nxt):
                    next_nums = [to_float(m.group()) for m in MONEY.finditer(nxt)]
                    if next_nums:
                        nums = next_nums
            if nums:
                result[key] = nums[-1]
                break
    taxes = sum(result[k] or 0.0 for k in ("vat", "cgst", "sgst", "igst", "service_tax"))
    result["tax_total"] = taxes if taxes else None
    result["total"] = total  # payable total, after discounts/taxes/charges
    # All totals in this benchmark are printed after taxes/charges.  The
    # explicit label prevents consumers from accidentally assuming pre-tax.
    # ``None`` means tax could not be identified; it must not be misreported
    # as a tax-exclusive total merely because the OCR missed a tax line.
    result["total_includes_tax"] = True if total is not None and taxes else None
    result["total_excluding_tax"] = round(total - taxes, 2) if total is not None and taxes else None
    return result


def extract_structured(lines):
    # NFKC turns full-width characters (such as the colon in "Bill No. ：53") into normal ones
    texts = [unicodedata.normalize("NFKC", l["text"]) for l in lines]
    full = "\n".join(texts)

    bill_no = None
    for t in texts:
        for m in BILL_NO.finditer(t):
            if any(c.isdigit() for c in m.group(1)):
                bill_no = m.group(1)
                break
        if bill_no:
            break

    if bill_no is None:
        for t in texts:
            m = BILL_NO_FUZZY.search(t)
            if m and any(c.isdigit() for c in m.group(1)):
                bill_no = m.group(1)
                break

    date = None
    for t in texts:
        date = parse_date(t)
        if date:
            break

    gstin = find_gstin(texts)

    total = None
    for key in TOTAL_KEYS:
        cands = []
        for idx, t in enumerate(texts):
            if re.search(key, t, re.I) and not TOTAL_SKIP.search(t):
                nums = [to_float(m.group()) for m in MONEY.finditer(t)]
                if not nums and idx + 1 < len(texts):  # "Total: Rs" then "3150" on the next line
                    nxt = texts[idx + 1]
                    if not re.search(r"[A-Za-z]", nxt):
                        nums = [to_float(m.group()) for m in MONEY.finditer(nxt)][:1]
                if nums:
                    cands.append((idx, nums[-1]))
        if cands:
            total_idx, total = cands[-1]
            total = apply_round_off(total, texts[total_idx + 1:total_idx + 3])
            break
    if total is None:
        allnums = [to_float(m.group()) for m in MONEY.finditer(full)]
        total = max(allnums) if allnums else None

    return {
        "fields": {
            "restaurant_name": guess_restaurant(texts),
            "bill_no": bill_no, "date": date, "gstin": gstin, "total": total,
        },
        "items": extract_items(texts),
        "financials": extract_financials(texts, total),
        "checks": {"gstin_checksum_ok": gstin_checksum_ok(gstin) if gstin else None},
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_text(s):
    return " ".join(s.lower().split())


def tokens(s):
    return [t for t in re.split(r"[^a-z0-9.,/:-]+", s.lower()) if t]


def token_f1(ref, hyp):
    r, h = Counter(tokens(ref)), Counter(tokens(hyp))
    overlap = sum((r & h).values())
    if not overlap:
        return 0.0
    p, rc = overlap / sum(h.values()), overlap / sum(r.values())
    return 2 * p * rc / (p + rc)


def alnum(s):
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()


def field_matches(name, gt_val, pred_val):
    if pred_val is None:
        return False
    if name == "total":
        try:
            return abs(float(gt_val) - float(pred_val)) <= 0.01
        except (TypeError, ValueError):
            return False
    if name == "date":
        return (parse_date(str(gt_val)) or str(gt_val).strip()) == pred_val
    if name == "restaurant_name":
        a, b = alnum(gt_val), alnum(pred_val)
        return SequenceMatcher(None, a, b).ratio() >= 0.8
    return alnum(gt_val) == alnum(pred_val)  # bill_no, gstin


FUZZY_ITEM_THRESHOLD = 0.75


def partial_ratio(needle, hay):
    """Best similarity of `needle` against any window of `hay` (tolerates OCR typos)."""
    n = len(needle)
    if n == 0:
        return 0.0
    if len(hay) <= n:
        return SequenceMatcher(None, needle, hay).ratio()
    best = 0.0
    for size in (n - 1, n, n + 1):
        for i in range(0, len(hay) - size + 1):
            best = max(best, SequenceMatcher(None, needle, hay[i:i + size]).ratio())
    return best


def item_matches(gt_item, pred_item):
    """Compare a *parsed* item, tolerating minor OCR spelling errors."""
    try:
        amount = float(gt_item["amount"])
        predicted_amount = float(pred_item["amount"])
    except (TypeError, ValueError):
        return False
    if abs(amount - predicted_amount) > 0.01:
        return False
    name_tokens = set(tokens(gt_item["name"]))
    name_flat = alnum(gt_item["name"])
    if not name_tokens or not name_flat:
        return False
    predicted_name = str(pred_item.get("name") or "")
    if len(name_tokens & set(tokens(predicted_name))) / len(name_tokens) >= 0.6:
        return True
    return partial_ratio(name_flat, alnum(predicted_name)) >= FUZZY_ITEM_THRESHOLD


def match_items(gt_items, predicted_items):
    """One-to-one matching prevents a single OCR row earning credit twice."""
    used, matches = set(), 0
    for gt_item in gt_items:
        for i, pred_item in enumerate(predicted_items):
            if i not in used and item_matches(gt_item, pred_item):
                used.add(i)
                matches += 1
                break
    return matches


def evaluate_bill(lines, structured, gt):
    """Return a dict of metrics. Anything without ground truth stays None."""
    m = {"cer": None, "wer": None, "token_f1": None,
         "field_results": {}, "items_found": None, "items_total": None,
         "items_predicted": None}
    if not gt:
        return m

    ref_text = (gt.get("text") or "").strip()
    if ref_text:
        hyp = norm_text(" ".join(l["text"] for l in lines))
        ref = norm_text(ref_text)
        m["cer"] = levenshtein(ref, hyp) / max(len(ref), 1)
        rw, hw = ref.split(), hyp.split()
        m["wer"] = levenshtein(rw, hw) / max(len(rw), 1)
        m["token_f1"] = token_f1(ref, hyp)

    for name in FIELD_NAMES:
        gt_val = (gt.get("fields") or {}).get(name)
        if gt_val in (None, ""):
            continue
        m["field_results"][name] = field_matches(name, gt_val, structured["fields"].get(name))

    gt_items = [i for i in (gt.get("items") or []) if i.get("name")]
    if gt_items:
        m["items_total"] = len(gt_items)
        predicted_items = structured.get("items") or []
        m["items_predicted"] = len(predicted_items)
        m["items_found"] = match_items(gt_items, predicted_items)
    return m


# --------------------------------------------------------------------------
# Aggregation, scoring, reporting
# --------------------------------------------------------------------------
def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def aggregate(rows):
    agg = {k: mean_or_none([r[k] for r in rows]) for k in ("cer", "wer", "token_f1", "conf")}
    f_ok = sum(sum(r["field_results"].values()) for r in rows)
    f_all = sum(len(r["field_results"]) for r in rows)
    i_ok = sum(r["items_found"] or 0 for r in rows)
    i_all = sum(r["items_total"] or 0 for r in rows)
    i_pred = sum(r["items_predicted"] or 0 for r in rows)
    agg["field_acc"] = f_ok / f_all if f_all else None
    agg["item_recall"] = i_ok / i_all if i_all else None
    agg["item_precision"] = i_ok / i_pred if i_pred else None
    if agg["item_recall"] is not None and agg["item_precision"] is not None:
        d = agg["item_recall"] + agg["item_precision"]
        agg["item_f1"] = 2 * agg["item_recall"] * agg["item_precision"] / d if d else 0.0
    else:
        agg["item_f1"] = None
    ok = [r["seconds"] for r in rows if not r["error"]]  # failed runs do not count as "fast"
    agg["avg_seconds"] = sum(ok) / len(ok) if ok else None
    agg["errors"] = sum(1 for r in rows if r["error"])
    per_field = {}
    for name in FIELD_NAMES:
        res = [r["field_results"][name] for r in rows if name in r["field_results"]]
        per_field[name] = (sum(res) / len(res)) if res else None
    agg["per_field"] = per_field
    return agg


def composite(agg, fastest):
    parts = {}
    if agg["cer"] is not None:
        parts["cer"] = max(0.0, 1 - agg["cer"])
    for k in ("token_f1", "field_acc", "item_f1"):
        if agg[k] is not None:
            parts[k] = agg[k]
    if agg["avg_seconds"] is None:
        parts["speed"] = 0.0
    else:
        parts["speed"] = fastest / agg["avg_seconds"] if agg["avg_seconds"] > 0 else 1.0
    wsum = sum(WEIGHTS[k] for k in parts)
    return 100 * sum(WEIGHTS[k] * v for k, v in parts.items()) / wsum


def fmt(v, pct=False, nd=3):
    if v is None:
        return "n/a"
    return f"{100 * v:.1f}%" if pct else f"{v:.{nd}f}"


def build_table(ranked):
    head = ["Rank", "Model", "Score", "CER", "WER", "TokenF1", "Field acc", "Item F1", "Item recall", "Item precision", "Sec/bill", "Errors"]
    rows = []
    for i, (name, agg, score) in enumerate(ranked, 1):
        rows.append([str(i), name, f"{score:.1f}", fmt(agg["cer"]), fmt(agg["wer"]),
                      fmt(agg["token_f1"], True), fmt(agg["field_acc"], True),
                      fmt(agg["item_f1"], True), fmt(agg["item_recall"], True),
                      fmt(agg["item_precision"], True), fmt(agg["avg_seconds"], nd=2), str(agg["errors"])])
    return head, rows


def print_table(head, rows):
    widths = [max(len(head[i]), *(len(r[i]) for r in rows)) for i in range(len(head))]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths))
    print(line(head))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))


def write_report(path, ranked, best, why):
    head, rows = build_table(ranked)
    md = ["# OCR benchmark report", "",
          f"Best model: **{best}**", "",
          "Score weights: " + ", ".join(f"{k} {v}" for k, v in WEIGHTS.items()), "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", "## Per-field accuracy", "",
           "| Model | " + " | ".join(FIELD_NAMES) + " |", "|---|" + "---|" * len(FIELD_NAMES)]
    for name, agg, _ in ranked:
        md.append(f"| {name} | " + " | ".join(fmt(agg["per_field"][f], True) for f in FIELD_NAMES) + " |")
    md += ["", "## Best model per metric", ""] + [f"- {w}" for w in why]
    Path(path).write_text("\n".join(md), encoding="utf-8")


def draw_boxes(img_path, words, out_path):
    from PIL import Image, ImageDraw
    img = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(img)
    for w in words:
        d.rectangle(w["box"], outline=(255, 0, 0), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


def cmd_init_gt(args):
    Path(args.gt).mkdir(parents=True, exist_ok=True)
    template = {
        "text": "",
        "fields": {"restaurant_name": "", "bill_no": "", "date": "", "gstin": "", "total": None},
        "items": [{"name": "", "qty": None, "amount": None}],
    }
    made = 0
    for img in list_images(args.images):
        p = Path(args.gt) / f"{img.stem}.json"
        if not p.exists():
            p.write_text(json.dumps(template, indent=2), encoding="utf-8")
            made += 1
    print(f"Created {made} ground-truth templates in {args.gt}. Fill them in by hand.")


def cmd_run(args):
    images = list_images(args.images)
    if not images:
        sys.exit(f"No images found in {args.images}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gts = {}
    for img in images:
        p = Path(args.gt) / f"{img.stem}.json"
        if p.exists():
            gts[img.stem] = json.loads(p.read_text(encoding="utf-8"))
        else:
            print(f"[warn] no ground truth for {img.name}; only speed will be scored for it")

    all_rows, skipped = {}, {}
    for name in args.models:
        if name not in MODELS:
            print(f"[skip] unknown model '{name}'. Available: {', '.join(MODELS)}")
            continue
        print(f"\n=== {name} ===")
        try:
            predict = MODELS[name]()
        except Exception as e:  # missing package, missing binary, no internet for weights...
            skipped[name] = f"{type(e).__name__}: {e}"
            print(f"[skip] could not load {name}: {skipped[name]}")
            continue
        try:
            predict(str(images[0]))  # warm-up so load time does not distort speed
        except Exception:
            pass

        rows = []
        for img in images:
            t0 = time.perf_counter()
            words, err = [], None
            try:
                words = predict(str(img))
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            seconds = time.perf_counter() - t0

            lines = group_into_lines(words)
            structured = extract_structured(lines)
            metrics = evaluate_bill(lines, structured, gts.get(img.stem))
            metrics.update({"bill": img.name, "seconds": seconds, "error": err,
                            "conf": mean_or_none([l["conf"] for l in lines]),
                            "structured": structured})
            rows.append(metrics)

            d = out / "outputs" / name
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{img.stem}.json").write_text(json.dumps({
                "image": img.name, "model": name, "seconds": round(seconds, 3), "error": err,
                "lines": [{"text": l["text"], "conf": l["conf"]} for l in lines],
                "structured": structured}, indent=2, ensure_ascii=False), encoding="utf-8")
            if args.viz and words:
                draw_boxes(img, words, out / "viz" / name / f"{img.stem}.png")
            print(f"  {img.name}: {len(lines)} lines, {seconds:.2f}s" + (f"  ERROR {err}" if err else ""))
        all_rows[name] = rows

    finalize(all_rows, len(images), out, skipped)


def finalize(all_rows, n_bills, out, skipped):
    """Rank the models, print the table, pick the best, write all result files."""
    if not all_rows:
        sys.exit("No model ran successfully. Check the install notes in README.md.")

    aggs = {n: aggregate(r) for n, r in all_rows.items()}
    times = [a["avg_seconds"] for a in aggs.values() if a["avg_seconds"] is not None]
    fastest = min(times) if times else 1.0
    ranked = sorted(((n, a, composite(a, fastest)) for n, a in aggs.items()),
                    key=lambda x: (-x[2], x[1]["avg_seconds"] if x[1]["avg_seconds"] is not None else 1e9))
    valid = [r for r in ranked if r[1]["errors"] < n_bills]
    if not valid:
        head, trows = build_table(ranked)
        print_table(head, trows)
        sys.exit("\nEvery model failed on every bill. Fix the errors shown above and run again.")
    best, best_score = valid[0][0], valid[0][2]

    why = []
    for key, label, lower_better in [("cer", "Lowest CER", True), ("token_f1", "Best token F1", False),
                                     ("field_acc", "Best field accuracy", False),
                                      ("item_f1", "Best extracted-item F1", False),
                                      ("item_recall", "Best item recall", False),
                                     ("avg_seconds", "Fastest", True)]:
        cands = [(n, a[key]) for n, a in aggs.items() if a[key] is not None]
        if cands:
            w = min(cands, key=lambda x: x[1]) if lower_better else max(cands, key=lambda x: x[1])
            why.append(f"{label}: {w[0]} ({w[1]:.3f})")

    head, trows = build_table(ranked)
    print("\n=== RESULTS (higher score is better) ===")
    print_table(head, trows)
    print("\nPer-metric winners:")
    for w in why:
        print("  -", w)
    print(f"\nBEST MODEL FOR THESE BILLS: {best}  (score {best_score:.1f}/100)")
    if skipped:
        print("\nSkipped models:", json.dumps(skipped, indent=2))

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(trows)
    with open(out / "per_bill.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "bill", "seconds", "cer", "wer", "token_f1",
                    "fields_correct", "fields_total", "items_found", "items_total", "items_predicted", "error"])
        for n, rows in all_rows.items():
            for r in rows:
                w.writerow([n, r["bill"], f"{r['seconds']:.3f}", fmt(r["cer"]), fmt(r["wer"]),
                            fmt(r["token_f1"]), sum(r["field_results"].values()), len(r["field_results"]),
                             r["items_found"], r["items_total"], r["items_predicted"], r["error"] or ""])
    write_report(out / "report.md", ranked, best, why)

    best_dir = out / "best_model"
    best_dir.mkdir(exist_ok=True)
    combined = {r["bill"]: r["structured"] for r in all_rows[best]}
    (best_dir / "structured_output.json").write_text(
        json.dumps({"best_model": best, "bills": combined}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out/'summary.csv'}, {out/'per_bill.csv'}, {out/'report.md'}, {best_dir/'structured_output.json'}")


def cmd_score(args):
    """Re-score saved OCR outputs WITHOUT running the models again.
    Also re-applies the current extraction rules, so you can improve the rules
    and see the effect in seconds."""
    images = list_images(args.images)
    out = Path(args.out)
    base = out / "outputs"
    if not base.exists():
        sys.exit(f"{base} not found. Run the models first with the 'run' command.")
    models = args.models or sorted(p.name for p in base.iterdir() if p.is_dir())

    gts = {}
    for img in images:
        p = Path(args.gt) / f"{img.stem}.json"
        if p.exists():
            gts[img.stem] = json.loads(p.read_text(encoding="utf-8"))

    all_rows = {}
    for name in models:
        rows = []
        for img in images:
            f = base / name / f"{img.stem}.json"
            data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
            if data:
                lines = [{"text": l["text"], "conf": l.get("conf")} for l in data["lines"]]
                seconds, err = data.get("seconds", 0.0), data.get("error")
            else:
                lines, seconds, err = [], 0.0, "no saved output"
            structured = extract_structured(lines)
            metrics = evaluate_bill(lines, structured, gts.get(img.stem))
            metrics.update({"bill": img.name, "seconds": seconds, "error": err,
                            "conf": mean_or_none([l["conf"] for l in lines]),
                            "structured": structured})
            rows.append(metrics)
            if data:
                data["structured"] = structured
                f.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        all_rows[name] = rows
        print(f"loaded {name}")
    finalize(all_rows, len(images), out, {})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("init-gt", help="create empty ground-truth JSON templates")
    p1.add_argument("--images", default="data/images")
    p1.add_argument("--gt", default="data/ground_truth")
    p1.set_defaults(fn=cmd_init_gt)

    p2 = sub.add_parser("run", help="run models, evaluate, pick the best")
    p2.add_argument("--images", default="data/images")
    p2.add_argument("--gt", default="data/ground_truth")
    p2.add_argument("--out", default="results")
    p2.add_argument("--models", nargs="+", default=list(MODELS))
    p2.add_argument("--viz", action="store_true", help="save images with detected boxes drawn")
    p2.set_defaults(fn=cmd_run)

    p3 = sub.add_parser("score", help="re-score saved outputs without re-running the models")
    p3.add_argument("--images", default="data/images")
    p3.add_argument("--gt", default="data/ground_truth")
    p3.add_argument("--out", default="results")
    p3.add_argument("--models", nargs="+", default=None, help="default: every model found in results/outputs")
    p3.set_defaults(fn=cmd_score)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
