# 🧾 Invoice Extractor with Donut + Typhoon OCR

This project extracts key information (total amount, date, vendor) from invoice images using a fine-tuned Donut model and fallback OCR with Typhoon.

## 📦 Features

- ✅ Extracts the following from invoices:
  - Total amount
  - Currency
  - Invoice date
  - Vendor name

- 🧠 Uses:
  - Donut for layout-aware OCR and field extraction
  - Typhoon OCR as a fallback for enhanced robustness

## 🔧 Requirements

- Python 3.8+
- Install dependencies from `requirements.txt`

## 📥 Installation

```bash
git clone https://github.com/Ammmar22/facture.git
cd facture
pip install -r requirements.txt
