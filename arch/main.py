import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# === File paths ===
xlsx_path = "data/yotel-service-items-listing-7jul.xlsx"
csv_path = "data/yotel-job-items-variants.csv"

intermediate_path = "temp/yotel_merged_output-missing.xlsx"
filtered_path = "temp/filtered_no_chinese.xlsx"
final_output_path = "temp/cleaned_variants.xlsx"
semantic_output_path = "temp/semantic_matching_results.xlsx"
updated_output_path = "temp/cleaned_variants_updated.xlsx"
final_cleaned_output = "temp/cleaned_variants_updated_no_chinese.xlsx"

final_uuid_output = "final-job-variant.xlsx"

# === Load source data ===
xlsx_df = pd.read_excel(xlsx_path)
csv_df = pd.read_csv(csv_path)

# === Normalize UUID columns ===
xlsx_column_name = "UUID"
csv_column_name = csv_df.columns[0]
xlsx_df[xlsx_column_name] = xlsx_df[xlsx_column_name].astype(str)
csv_df[csv_column_name] = csv_df[csv_column_name].astype(str)

# === Merge UUIDs and variants ===
combined_rows = []
for _, xlsx_row in xlsx_df.iterrows():
    uuid = xlsx_row[xlsx_column_name]
    base_row = xlsx_row.tolist()
    combined_rows.append(base_row)
    matching_rows = csv_df[csv_df[csv_column_name] == uuid]
    for _, csv_row in matching_rows.iterrows():
        new_row = [uuid] + [""] * (len(xlsx_row) - 1) + [csv_row.iloc[1]]
        combined_rows.append(new_row)

separator_row = [""] * len(xlsx_df.columns) + ["-- UNMATCHED UUIDs BELOW --"]
combined_rows.append(separator_row)

csv_uuids = set(csv_df[csv_column_name])
xlsx_uuids = set(xlsx_df[xlsx_column_name])
unmatched_uuids = csv_uuids - xlsx_uuids
unmatched_df = csv_df[csv_df[csv_column_name].isin(unmatched_uuids)]

for uuid, group in unmatched_df.groupby(csv_column_name):
    variant_list = group.iloc[:, 1].tolist()
    if not variant_list:
        continue
    base_row = [uuid] + [""] * (len(xlsx_df.columns) - 2) + [variant_list[0]]
    combined_rows.append(base_row)
    for variant in variant_list[1:]:
        variant_row = [uuid] + [""] * (len(xlsx_df.columns) - 1) + [variant]
        combined_rows.append(variant_row)

# === Save merged output ===
new_columns = list(xlsx_df.columns) + ["Variant Info"]
merged_df = pd.DataFrame(combined_rows, columns=new_columns)
merged_df.to_excel(intermediate_path, index=False)
print(f"✅ Merged file saved to: {intermediate_path}")

# === Filter out Chinese-character rows ===
def contains_chinese(text):
    return isinstance(text, str) and bool(re.search(r'[\u4e00-\u9fff]', text))

df_no_chinese = merged_df[~merged_df.apply(lambda col: col.map(contains_chinese)).any(axis=1)]
df_no_chinese.to_excel(filtered_path, index=False)
print(f"✅ Chinese-filtered file saved to: {filtered_path}")

# === Clean and flatten ===
df = pd.read_excel(filtered_path)
df.columns = df.columns.str.strip()
df['JO Service Item'] = df['JO Service Item'].ffill()
df_cleaned = df.dropna(subset=['Variant Info'])
df_final = df_cleaned[['JO Service Item', 'Variant Info']]
df_final.to_excel(final_output_path, index=False)
print(f"✅ Cleaned file saved as: {final_output_path}")

# === Semantic matching ===
print("📐 Starting semantic similarity matching...")
df = pd.read_excel(final_output_path)
df.columns = df.columns.str.strip()
df_master = df.iloc[:52120].copy()
df_unmatched = df.iloc[52120:].copy()

master_items = list(set(df_master['JO Service Item'].dropna()))
unmatched_items = list(set(df_unmatched['JO Service Item'].dropna()))

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

print("🔠 Encoding master items...")
master_embeddings = model.encode(master_items, convert_to_tensor=True).cpu()
print("🔠 Encoding unmatched items...")
unmatched_embeddings = model.encode(unmatched_items, convert_to_tensor=True).cpu()

print("🔍 Computing cosine similarities...")
cos_sim = cosine_similarity(unmatched_embeddings, master_embeddings)

matched_results = []
print("🔄 Matching items...")
for i, unmatched in enumerate(tqdm(unmatched_items, desc="Matching")):
    best_idx = np.argmax(cos_sim[i])
    best_match = master_items[best_idx]
    score = cos_sim[i][best_idx]
    matched_results.append((unmatched, best_match, round(score, 4)))

matched_df = pd.DataFrame(matched_results, columns=['Unmatched Item', 'Matched Master Item', 'Similarity Score'])
matched_df.to_excel(semantic_output_path, index=False)
print(f"✅ Saved semantic matches to: {semantic_output_path}")

# === Append strong matches (score >= 0.8) ===
strong_matches = matched_df[matched_df['Similarity Score'] >= 0.8].copy()
new_rows = strong_matches[['Matched Master Item', 'Unmatched Item']]
new_rows.columns = ['JO Service Item', 'Variant Info']

cleaned_df = pd.read_excel(final_output_path)
cleaned_df.columns = cleaned_df.columns.str.strip()
updated_df = pd.concat([cleaned_df, new_rows], ignore_index=True)
updated_df.to_excel(updated_output_path, index=False)
print(f"✅ Updated file saved as: {updated_output_path}")

# === Final Chinese character check for updated file ===
df_updated = pd.read_excel(updated_output_path)
df_updated.columns = df_updated.columns.str.strip()
df_final_clean = df_updated[~df_updated.apply(lambda col: col.map(contains_chinese)).any(axis=1)]
df_final_clean.to_excel(final_cleaned_output, index=False)
print(f"✅ Final file with Chinese characters removed saved as: {final_cleaned_output}")

# === Load the Excel files ===
service_items_df = pd.read_excel("data/yotel-service-items-listing-7jul.xlsx")
cleaned_variants_df = pd.read_excel("temp/cleaned_variants_updated_no_chinese.xlsx")

# === Clean column names and strip whitespace in key columns ===
service_items_df.columns = service_items_df.columns.str.strip()
cleaned_variants_df.columns = cleaned_variants_df.columns.str.strip()
service_items_df["JO Service Item"] = service_items_df["JO Service Item"].str.strip()
cleaned_variants_df["JO Service Item"] = cleaned_variants_df["JO Service Item"].str.strip()

# === Merge UUIDs into cleaned variants ===
merged_df = cleaned_variants_df.merge(
    service_items_df[["UUID", "JO Service Item"]],
    on="JO Service Item",
    how="left"
)

# === Reorder columns so UUID is first ===
merged_df = merged_df[["UUID", "JO Service Item", "Variant Info"]]
merged_df = merged_df.drop_duplicates(subset=["Variant Info"])

# === Save the new Excel file ===
merged_df.to_excel("variants_uuid_nondup.xlsx", index=False)
print("✅ UUIDs added and file saved as 'cleaned_variants_with_uuid.xlsx'")
