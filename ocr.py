#!/usr/bin/env python
"""
Extract total amount, date, and vendor from invoice images using Donut

Usage:
    python invoice_extract.py path/to/invoice.jpg
    python invoice_extract.py folder/with/invoices
"""

import argparse
import json
import re
import os
import time
from pathlib import Path
from datetime import datetime

from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from typhoon_ocr import ocr_document

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

os.environ['TYPHOON_OCR_API_KEY'] = "your-key"

print("⏳ Loading Donut model…")
start_time = time.time()
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2").eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model ready on {device} (loaded in {time.time() - start_time:.2f}s)")

# ─────────────────────────────────────────────────────────────
# Constants & Regex
# ─────────────────────────────────────────────────────────────

MONTHS = {
    "janvier": 1, "février": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "août": 8, "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}
DATE_RE = re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b")
TAG_RE = {
    "total": re.compile(r"<s_total_etc>\s*(.*?)\s*</s_total_etc>", re.I | re.S),
    "vendor": re.compile(r"<s_nm>\s*(.*?)\s*</s_nm>", re.I | re.S),
}

# ─────────────────────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────────────────────

def fallback_typhoon_total(image_path: Path):
    try:
        markdown = ocr_document(str(image_path))
    except Exception as e:
        print(f"⚠️ Typhoon OCR failed: {e}")
        return None, None
    return extract_total_amount_smart(markdown)


def donut_inference(image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = processor.tokenizer("<s_cord-v2>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=1024)
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]


def extract_total_amount_smart(text: str):
    # First try to extract from <s_total_price>
    match = re.search(r"<s_total_price>\s*([£€$]?)\s*([\d.,]+)\s*</s_total_price>", text, re.I)
    if match:
        currency_symbol = match.group(1)
        amount = match.group(2).replace(",", ".").replace(" ", "")
        try:
            val = float(amount)
            if 0 < val < 1000:  # avoid huge wrong values
                return str(val), currency_symbol
            else:
                print(f"⚠️ Ignoring suspicious total value: {val}")
        except ValueError:
            pass

    # Fallback: scan all <s_price> and <s_tax_price> for valid candidates
    matches = re.findall(r"<s_(?:price|tax_price)>\s*([\d.,]+)\s*</s_\w+>", text)
    candidates = []
    for m in matches:
        try:
            val = float(m.replace(",", "."))
            if 0 < val < 1000:
                candidates.append(val)
        except:
            continue
    if candidates:
        return str(max(candidates)), "£"  # or use OCR/text detection to infer
    else:
    # Fallback: smart line-by-line total detection
     lines = text.splitlines()
     final_total = None
     for line in reversed(lines):  # Start from bottom
        if "total" in line.lower():
            match = re.search(r"(£|€|\$)?\s?(\d{1,3}(?:[.,]\d{3})*[.,]?\d{0,2})", line)
            if match:
                symbol = match.group(1) or ""
                amount = match.group(2).replace(",", ".")
                try:
                    val = float(amount)
                    if 0 < val < 1000:
                        final_total = (str(val), symbol)
                        break
                except:
                    continue

    if final_total:
        return final_total
    # Fallback pattern in plain text
    pattern = re.compile(r"""(?i)
        (?:
            (total|amount\s+due|ttc|net\s*(à\s*)?payer|subtotal|montant|change)
            [^\d£€$TNDNDUSD]{0,10}
        )?
        (?P<currency>[£€$]|TND|ND|EUR|USD)?
        \s?(?P<amount>\d{1,3}(?:[.,]\d{3})*[.,]?\d{0,2})
    """, re.VERBOSE)

    for match in pattern.finditer(text):
        groups = match.groupdict()
        raw_amount = groups["amount"].replace(",", ".").replace(" ", "")
        try:
            val = float(raw_amount)
            if 0 < val < 1000:
                return str(val), groups.get("currency", "")
        except ValueError:
            continue

    return None, None







def extract_date_smart(text: str):
    for line in text.splitlines():
        if any(kw in line.lower() for kw in ["date", "du", "le"]):
            match = DATE_RE.search(line)
            if match:
                return match.group(0)
    match = DATE_RE.search(text)
    if match:
        return match.group(0)
    return None


def parse_date_with_month_name(text: str):
    match = re.search(r'(\d{1,2})\s+([a-zA-Zéûà]+)\s+(\d{4})', text, re.I)
    if match:
        day, month_str, year = match.groups()
        month = MONTHS.get(month_str.lower())
        if month:
            try:
                return datetime(int(year), month, int(day)).strftime("%Y-%m-%d")
            except ValueError:
                pass
    return None


def fallback_typhoon_date(image_path: Path):
    try:
        markdown = ocr_document(str(image_path))
    except Exception as e:
        print(f"⚠️ Typhoon OCR failed: {e}")
        return None
    date = extract_date_smart(markdown) or parse_date_with_month_name(markdown)
    return date


def clean_vendor_name(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates = []
    for line in lines:
        if re.search(r"\b(server|table|thank|total|amount|invoice|facture|date)\b", line, re.I):
            continue
        if re.search(r"\d{4}|\d{2}/\d{2}/\d{4}", line):  # skip lines with years or dates
            continue
        if 3 < len(line) < 50:
            candidates.append(line)
    candidates = sorted(candidates, key=lambda l: (l.isupper(), not l.istitle(), len(l)))
    if candidates:
        return candidates[0]
    for line in lines:
        if line.isupper() and 3 < len(line) < 40:
            return line
    return None


def parse_fields(raw: str) -> dict:
    data = {}
    vendor_match = TAG_RE["vendor"].search(raw)
    vendor = vendor_match.group(1).strip() if vendor_match else None
    if vendor and not vendor.lower().startswith("4550") and "server" not in vendor.lower():
        data["vendor"] = vendor
    else:
        fallback_vendor = clean_vendor_name(raw)
        if fallback_vendor:
            data["vendor"] = fallback_vendor

    amount, currency = extract_total_amount_smart(raw)
    if amount:
        data["total_amount"] = amount
        currency_map = {"£": "GBP", "$": "USD", "€": "EUR"}
        if currency:
            currency = currency.strip()
            currency = currency_map.get(currency, currency.upper())
        else:
            currency = ""
        data["currency"] = currency

    found_date = extract_date_smart(raw)
    if found_date:
        data["date"] = found_date
    else:
        fallback_date = parse_date_with_month_name(raw)
        if fallback_date:
            data["date"] = fallback_date

    return data

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract invoice fields using Donut")
    parser.add_argument("path", type=str, help="Path to image or directory")
    args = parser.parse_args()
    path = Path(args.path)

    if path.is_dir():
        images = sorted([p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    else:
        images = [path]

    results = []

    for img in images:
        print(f"\n🔍 Processing {img} …")
        raw = donut_inference(img)
        with open("donut_raw_output.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        print("✅ Saved raw output to donut_raw_output.txt")

        fields = parse_fields(raw)
        if "date" not in fields:
            print("🔄 Trying fallback Typhoon OCR…")
            fallback = fallback_typhoon_date(img)
            if fallback:
                fields["date"] = fallback
                print(f"📅 Found fallback date: {fallback}")
        # If currency is missing but amount is present
        if "total_amount" in fields and not fields.get("currency"):
          print("🔄 Checking Typhoon OCR for missing currency symbol…")
          try:
             markdown = ocr_document(str(img))
        # Look for currency symbols in the OCR text
             currency_match = re.search(r"[£€$]", markdown)
             if currency_match:
              symbol = currency_match.group(0)
              currency_map = {"£": "GBP", "$": "USD", "€": "EUR"}
              fields["currency"] = currency_map.get(symbol, symbol)
              print(f"💰 Currency symbol '{symbol}' found via Typhoon OCR. Set currency to {fields['currency']}")
          except Exception as e:
            print(f"⚠️ Typhoon OCR failed during currency check: {e}")

        fields["file"] = img.name
        print("➡️  Extracted:", json.dumps(fields, indent=2, ensure_ascii=False))
        results.append(fields)

    with open("invoice_extraction_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📁 All results saved to invoice_extraction_results.json")


if __name__ == "__main__":
    main()
