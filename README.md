<!-- # fcs-dataclean
# 🛠️ YOTEL Job Items Variant Matching Pipeline -->

This Python script performs a full end-to-end pipeline to process and match service item variants for YOTEL jobs. It merges variant data from multiple sources, filters out Chinese-language rows, matches similar service items using semantic similarity, and restores UUIDs to the final dataset.

---

## 📂 Folder Structure

```
.
├── data/
│   ├── yotel-service-items-listing-7jul.xlsx   # Master Excel file with UUIDs
│   └── yotel-job-items-variants.csv            # CSV file with UUID + Variant Info
├── temp/
│   ├── yotel_merged_output-missing.xlsx
│   ├── filtered_no_chinese.xlsx
│   ├── cleaned_variants.xlsx
│   ├── semantic_matching_results.xlsx
│   ├── cleaned_variants_updated.xlsx
│   └── cleaned_variants_updated_no_chinese.xlsx
├── final-job-variant.xlsx                      # ✅ Final output with UUIDs
├── script.py                                   # Your main processing script
├── requirements.txt
└── README.md
```

---

## 🚀 Features

- ✅ Merges UUIDs from Excel and CSV
- 🧼 Removes rows containing Chinese characters
- 🧾 Cleans and flattens service items and variants
- 🤖 Uses semantic similarity (Sentence-BERT) to match unmatched items
- ➕ Appends high-confidence semantic matches (score ≥ 0.8)
- 🔁 Re-matches UUIDs to the final variant list
- 📤 Exports final results to `final-job-variant.xlsx`

---

## 🔧 Requirements

Install all required dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🧠 Semantic Matching Model

- Model used: `paraphrase-MiniLM-L6-v2`  
- Library: `sentence-transformers`  
- Matching metric: `cosine similarity`  
- Threshold for strong match: **0.8**

---

## ▶️ How to Run

Ensure your `data/` folder contains:

- `yotel-service-items-listing-7jul.xlsx`
- `yotel-job-items-variants.csv`

Then run:

```bash
python script.py
```

Final results will be saved to:

```
final-job-variant.xlsx
```

---

## 📈 Output Files

| File Path                                  | Description                                |
|-------------------------------------------|--------------------------------------------|
| `temp/yotel_merged_output-missing.xlsx`   | Raw merged UUID + Variant output           |
| `temp/filtered_no_chinese.xlsx`           | Chinese characters removed                 |
| `temp/cleaned_variants.xlsx`              | Flattened, cleaned variant pairs           |
| `temp/semantic_matching_results.xlsx`     | Semantic similarity scores between items   |
| `temp/cleaned_variants_updated.xlsx`      | Strong matches appended                    |
| `temp/cleaned_variants_updated_no_chinese.xlsx` | Final filtered version before UUID recovery |
| `final-job-variant.xlsx`                  | ✅ Final UUID + JO Service Item + Variant Info |

---

## 📬 Contact

For any issues, feel free to raise an issue or contact the maintainer.
