# Invoice Extractor with Donut + Typhoon OCR

This project extracts key information (total amount, date, vendor) from invoice images using a fine-tuned Donut model and fallback OCR with Typhoon.

## 📦 Features

- Uses [Donut](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2) for layout-aware OCR and field extraction
- Falls back to Typhoon OCR for enhanced robustness
- Extracts:
  - Total amount
  - Currency
  - Invoice date
  - Vendor name

## 🔧 Requirements

- Python 3.8+
- See `requirements.txt` for Python dependencies

## 📥 Installation

```bash
git clone https://github.com/Ammmar22/OCR_SCOPE.git  # 🔧 replace with your actual repo URL
cd YOUR_REPO_NAME
pip install -r requirements.txt
