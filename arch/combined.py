

# ====== BEGIN 0-helper-txt2csv.py ======
import csv

# === File paths ===
input_file = 'Yotel job_items variants.txt'   # Tab-delimited .txt file
output_file = 'Yotel job_items variants.csv' # Output CSV file

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile, delimiter='\t')
    writer = csv.writer(outfile)

    for row in reader:
        writer.writerow(row)

print(f"Converted '{input_file}' to '{output_file}' (no headers assumed).")

# ====== END 0-helper-txt2csv.py ======


# ====== BEGIN 1-merge.py ======
import pandas as pd

# === File paths ===
xlsx_path = "data/yotel-service-items-listing-7jul.xlsx"
csv_path = "data/yotel-job-items-variants.csv"

# csv
output_path = "yotel_merged_output.csv"

# Load the files
# Assuming the first sheet in the Excel file contains the relevant data
xlsx_df = pd.read_excel(xlsx_path)
csv_df = pd.read_csv(csv_path)

# def UUID 
xlsx_column_name = "UUID"
csv_column_name = csv_df.columns[0]

xlsx_df[xlsx_column_name] = xlsx_df[xlsx_column_name].astype(str)
csv_df[csv_column_name] = csv_df[csv_column_name].astype(str)

# Combined rows 
combined_rows = []

for _, xlsx_row in xlsx_df.iterrows():
    uuid = xlsx_row[xlsx_column_name]
    base_row = xlsx_row.tolist()
    combined_rows.append(base_row)

    # find matching variants
    matching_csv_rows = csv_df[csv_df[csv_column_name] == uuid]
    for _, csv_row in matching_csv_rows.iterrows():
        # append UUID + empty cells for other columns + the variant info
        new_row = [uuid] + [""] * (len(xlsx_row) - 1) + [csv_row[1]]
        combined_rows.append(new_row)

# df to hold the combined data
new_columns = list(xlsx_df.columns) + ["Variant Info"]
output_df = pd.DataFrame(combined_rows, columns=new_columns)


output_df.to_csv(output_path, index=False)
print(f"Merged file saved to: {output_path}")


# ====== END 1-merge.py ======


# ====== BEGIN 2-remove-cn.py ======
import pandas as pd
import re

# Load the Excel file
file_path = "master-variant-interim-copy.xlsx"  # Replace with your actual path
df = pd.read_excel(file_path)

# Function to detect Chinese characters
def contains_chinese(text):
    if isinstance(text, str):
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    return False

# Remove rows where any cell contains Chinese characters
df_no_chinese = df[~df.applymap(contains_chinese).any(axis=1)]

# Save to new Excel file
df_no_chinese.to_excel("filtered_no_chinese.xlsx", index=False)

# ====== END 2-remove-cn.py ======


# ====== BEGIN 3-format.py ======
import pandas as pd

# Load the Excel file
file_path = "filtered_no_chinese.xlsx"
df = pd.read_excel(file_path)

# Strip whitespace from all column names
df.columns = df.columns.str.strip()

# Now proceed with the transformation
df['JO Service Item'] = df['JO Service Item'].fillna(method='ffill')
df_cleaned = df.dropna(subset=['Variant Info'])
df_final = df_cleaned[['JO Service Item', 'Variant Info']]

# Save to a new file
df_final.to_excel("cleaned_variants.xlsx", index=False)
print("Cleaned file saved as 'cleaned_variants.xlsx'")

# ====== END 3-format.py ======


# ====== BEGIN 4-up,py ======
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load data
df = pd.read_excel("cleaned_variants.xlsx")
df.columns = df.columns.str.strip()
df_master = df.iloc[:52120].copy()
df_unmatched = df.iloc[52120:].copy()

# Get unique JO items
master_items = list(set(df_master['JO Service Item'].dropna()))
unmatched_items = list(set(df_unmatched['JO Service Item'].dropna()))

# Load sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode all sentences
print("Encoding master items...")
master_embeddings = model.encode(master_items, convert_to_tensor=True).cpu()
print("Encoding unmatched items...")
unmatched_embeddings = model.encode(unmatched_items, convert_to_tensor=True).cpu()

# Compute pairwise cosine similarity
print("Computing cosine similarities...")
cos_sim = cosine_similarity(unmatched_embeddings, master_embeddings)

# Find best matches with tqdm progress bar
matched_results = []
print("Matching items...")
for i, unmatched in enumerate(tqdm(unmatched_items, desc="Matching")):
    best_idx = np.argmax(cos_sim[i])
    best_match = master_items[best_idx]
    score = cos_sim[i][best_idx]
    matched_results.append((unmatched, best_match, round(score, 4)))

# Save results
matched_df = pd.DataFrame(matched_results, columns=['Unmatched Item', 'Matched Master Item', 'Similarity Score'])
matched_df.to_excel("semantic_matching_results.xlsx", index=False)
print("✅ Saved as 'semantic_matching_results.xlsx'")

# ====== END 4-up,py ======


# ====== BEGIN 5-combine.py ======
import pandas as pd

# Load matching results
matched_df = pd.read_excel("semantic_matching_results.xlsx")

# Filter only matches with score >= 0.8
strong_matches = matched_df[matched_df['Similarity Score'] >= 0.8].copy()

# Prepare rows to append (JO Service Item = matched, Variant Info = unmatched)
new_rows = strong_matches[['Matched Master Item', 'Unmatched Item']]
new_rows.columns = ['JO Service Item', 'Variant Info']

# Load original cleaned file
cleaned_df = pd.read_excel("cleaned_variants.xlsx")
cleaned_df.columns = cleaned_df.columns.str.strip()

# Append new variant rows
updated_df = pd.concat([cleaned_df, new_rows], ignore_index=True)

# Save to new Excel file
updated_df.to_excel("cleaned_variants_updated.xlsx", index=False)
print("✅ Updated file saved as 'cleaned_variants_updated.xlsx'")

# ====== END 5-combine.py ======



# ====== MASTER EXECUTION WORKFLOW ======
if __name__ == "__main__":
    # Step 0: Convert .txt to .csv
    txt_to_csv("txt-files", "step0_csvs")

    # Step 1: Merge CSVs
    merge_csvs("step0_csvs", "step1_merged.csv")

    # Step 2: Remove Chinese
    remove_chinese_rows("step1_merged.csv", "step2_no_chinese.csv")

    # Step 3: Format data
    format_data("step2_no_chinese.csv", "step3_formatted.csv")

    # Step 4: Upload/Rename
    process_upload("step3_formatted.csv", "step4_uploaded.csv")

    # Step 5: Combine final output
    combine_final("step4_uploaded.csv", "final_output.csv")

    print("\n✅ All steps completed. Final file is 'final_output.csv'")
