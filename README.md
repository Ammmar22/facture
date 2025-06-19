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
git clone https://github.com/Ammmar22/facture.git  # 🔧 replace with your actual repo URL
cd facture
pip install -r requirements.txt

🔑 Set your Typhoon OCR API key

In your Python script (invoice_extract.py), replace this line:

os.environ['TYPHOON_OCR_API_KEY'] = "your-key"
with your actual API key.

📂 Output Example

{
  "total_amount": "55.30",
  "currency": "GBP",
  "date": "02/07/2018",
  "vendor": "Example Vendor",
  "file": "invoice1.jpg"
}
Results are saved in:

invoice_extraction_results.json
donut_raw_output.txt
📦 Installation

Clone the repo and install dependencies:

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
🧠 Model Used

Donut model: naver-clova-ix/donut-base-finetuned-cord-v2
OCR fallback: Typhoon OCR
📄 License

MIT License – use freely, credit appreciated.


---

### 2. 📦 `requirements.txt` file (same folder)

Create a file in the **same folder as your script** called:

