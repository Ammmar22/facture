🧾 Invoice Extractor with Donut + Typhoon OCR

This project extracts key information (total amount, date, vendor) from invoice images using a fine-tuned Donut model and fallback OCR with Typhoon.

📦 Features

✅ Extracts the following from invoices:

Total amount
Currency
Invoice date
Vendor name
🧠 Uses:

Donut for layout-aware OCR and field extraction
Typhoon OCR as a fallback for enhanced robustness
🔧 Requirements

Python 3.8+
Install dependencies from requirements.txt
📥 Installation

git clone https://github.com/Ammmar22/facture.git
cd facture
pip install -r requirements.txt
🔑 Set your Typhoon OCR API key

In invoice_extract.py, update this line with your actual API key:

os.environ['TYPHOON_OCR_API_KEY'] = "your-key"
🖼️ Run on Invoice Images

▶️ For a single image:
python invoice_extract.py path/to/invoice.jpg
📂 For a folder of invoices:
python invoice_extract.py path/to/folder/
📤 Output Example

{
  "total_amount": "55.30",
  "currency": "GBP",
  "date": "02/07/2018",
  "vendor": "Example Vendor",
  "file": "invoice1.jpg"
}
📝 Results are saved to:

invoice_extraction_results.json
donut_raw_output.txt
🧠 Models Used

Donut model: naver-clova-ix/donut-base-finetuned-cord-v2
OCR fallback: Typhoon OCR
📄 License

MIT License – use freely, credit appreciated.