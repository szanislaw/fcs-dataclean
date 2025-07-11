import csv
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ----------------------------------------------------------
# STEP 1: Convert tab-delimited .txt to .csv
# ----------------------------------------------------------
input_file = 'data/yotel-job-items-variants.txt'
output_csv_file = 'data/yotel-job-items-variants.csv'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_csv_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile, delimiter='\t')
    writer = csv.writer(outfile)
    for row in reader:
        writer.writerow(row)

print(f"✅ Converted '{input_file}' to '{output_csv_file}'")

# ----------------------------------------------------------
# STEP 2: Merge Excel and CSV data by UUID
# ----------------------------------------------------------
xlsx_path = "data/yotel-service-items-listing-7jul.xlsx"
csv_path = output_csv_file
merged_output_path = "yotel_merged_output.csv"

xlsx_df = pd.read_excel(xlsx_path)
csv_df = pd.read_csv(csv_path)

xlsx_column_name = "UUID"
csv_column_name = csv_df.columns[0]

xlsx_df[xlsx_column_name] = xlsx_df[xlsx_column_name].astype(str)
csv_df[csv_column_name] = csv_df[csv_column_name].astype(str)

combined_rows = []
for _, xlsx_row in xlsx_df.iterrows():
    uuid = xlsx_row[xlsx_column_name]
    base_row = xlsx_row.tolist()
    combined_rows.append(base_row)

    matching_csv_rows = csv_df[csv_df[csv_column_name] == uuid]
    for _, csv_row in matching_csv_rows.iterrows():
        new_row = [uuid] + [""] * (len(xlsx_row) - 1) + [csv_row[1]]
        combined_rows.append(new_row)

new_columns = list(xlsx_df.columns) + ["Variant Info"]
output_df = pd.DataFrame(combined_rows, columns=new_columns)
output_df.to_csv(merged_output_path, index=False)
print(f"✅ Merged file saved to: {merged_output_path}")

# ----------------------------------------------------------
# STEP 3: Clean and prepare data
# ----------------------------------------------------------
filtered_file = "arch/filtered_no_chinese.xlsx"
df = pd.read_excel(filtered_file)
df.columns = df.columns.str.strip()

df['JO Service Item'] = df['JO Service Item'].fillna(method='ffill')
df_cleaned = df.dropna(subset=['Variant Info'])
df_final = df_cleaned[['JO Service Item', 'Variant Info']]
cleaned_path = "cleaned_variants.xlsx"
df_final.to_excel(cleaned_path, index=False)
print(f"✅ Cleaned file saved as '{cleaned_path}'")

# ----------------------------------------------------------
# STEP 4: Semantic matching using sentence-transformers
# ----------------------------------------------------------
df = pd.read_excel(cleaned_path)
df.columns = df.columns.str.strip()
df_master = df.iloc[:52120].copy()
df_unmatched = df.iloc[52120:].copy()

master_items = list(set(df_master['JO Service Item'].dropna()))
unmatched_items = list(set(df_unmatched['JO Service Item'].dropna()))

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

print("🔄 Encoding master items...")
master_embeddings = model.encode(master_items, convert_to_tensor=True).cpu()
print("🔄 Encoding unmatched items...")
unmatched_embeddings = model.encode(unmatched_items, convert_to_tensor=True).cpu()

print("🔍 Computing cosine similarities...")
cos_sim = cosine_similarity(unmatched_embeddings, master_embeddings)

matched_results = []
print("🔗 Matching items...")
for i, unmatched in enumerate(tqdm(unmatched_items, desc="Matching")):
    best_idx = np.argmax(cos_sim[i])
    best_match = master_items[best_idx]
    score = cos_sim[i][best_idx]
    matched_results.append((unmatched, best_match, round(score, 4)))

matched_df = pd.DataFrame(matched_results, columns=['Unmatched Item', 'Matched Master Item', 'Similarity Score'])
matching_results_path = "semantic_matching_results.xlsx"
matched_df.to_excel(matching_results_path, index=False)
print(f"✅ Saved matching results to '{matching_results_path}'")

# ----------------------------------------------------------
# STEP 5: Append strong matches to cleaned data
# ----------------------------------------------------------
matched_df = pd.read_excel(matching_results_path)
strong_matches = matched_df[matched_df['Similarity Score'] >= 0.8].copy()

new_rows = strong_matches[['Matched Master Item', 'Unmatched Item']]
new_rows.columns = ['JO Service Item', 'Variant Info']

cleaned_df = pd.read_excel(cleaned_path)
cleaned_df.columns = cleaned_df.columns.str.strip()

updated_df = pd.concat([cleaned_df, new_rows], ignore_index=True)
updated_path = "cleaned_variants_updated.xlsx"
updated_df.to_excel(updated_path, index=False)
print(f"✅ Updated file saved as '{updated_path}'")

# ----------------------------------------------------------
# STEP 6: Add UUID column using Variant Info
# ----------------------------------------------------------
# Reload variant mapping
variants_df = pd.read_csv(output_csv_file)
variants_df.columns = variants_df.columns.str.strip()
variant_info_column = variants_df.columns[1]
uuid_column = variants_df.columns[0]

# Ensure types are string and clean
updated_df['Variant Info'] = updated_df['Variant Info'].astype(str).str.strip()
variants_df[variant_info_column] = variants_df[variant_info_column].astype(str).str.strip()

# Build mapping and insert UUID
uuid_mapping = variants_df.set_index(variant_info_column)[uuid_column].to_dict()
updated_df.insert(0, 'UUID', updated_df['Variant Info'].map(uuid_mapping))

# Save with UUID
final_path = "cleaned_variants_updated_with_uuid.xlsx"
updated_df.to_excel(final_path, index=False)
print(f"✅ Final file with UUID saved as '{final_path}'")
