import pandas as pd

# === File paths ===
xlsx_path = "data/yotel-service-items-listing-7jul.xlsx"
csv_path = "data/yotel-job-items-variants.csv"
output_path = "yotel_merged_output-missing.xlsx"

# === Load data ===
xlsx_df = pd.read_excel(xlsx_path)
csv_df = pd.read_csv(csv_path)

# === Normalize UUID columns ===
xlsx_column_name = "UUID"
csv_column_name = csv_df.columns[0]

xlsx_df[xlsx_column_name] = xlsx_df[xlsx_column_name].astype(str)
csv_df[csv_column_name] = csv_df[csv_column_name].astype(str)

# === Build combined list ===
combined_rows = []

# Add matched UUIDs and their variants
for _, xlsx_row in xlsx_df.iterrows():
    uuid = xlsx_row[xlsx_column_name]
    base_row = xlsx_row.tolist()
    combined_rows.append(base_row)

    matching_rows = csv_df[csv_df[csv_column_name] == uuid]
    for _, csv_row in matching_rows.iterrows():
        new_row = [uuid] + [""] * (len(xlsx_row) - 1) + [csv_row[1]]
        combined_rows.append(new_row)

# Add a separator row
separator_row = [""] * len(xlsx_df.columns) + ["-- UNMATCHED UUIDs BELOW --"]
combined_rows.append(separator_row)

# Add unmatched UUIDs and their variants, with one entry as "name"
csv_uuids = set(csv_df[csv_column_name])
xlsx_uuids = set(xlsx_df[xlsx_column_name])
unmatched_uuids = csv_uuids - xlsx_uuids

unmatched_df = csv_df[csv_df[csv_column_name].isin(unmatched_uuids)]

# Group by UUID and simulate "base name" by using the first variant as representative
for uuid, group in unmatched_df.groupby(csv_column_name):
    variant_list = group.iloc[:, 1].tolist()
    if not variant_list:
        continue

    # Simulate base row using first variant as name
    base_row = [uuid] + [""] * (len(xlsx_df.columns) - 2) + [variant_list[0]]
    combined_rows.append(base_row)

    # Add the rest of the variants (excluding the one used as name)
    for variant in variant_list[1:]:
        variant_row = [uuid] + [""] * (len(xlsx_df.columns) - 1) + [variant]
        combined_rows.append(variant_row)

# === Final output ===
new_columns = list(xlsx_df.columns) + ["Variant Info"]
output_df = pd.DataFrame(combined_rows, columns=new_columns)
output_df.to_excel(output_path, index=False)

print(f"Merged file with unmatched section saved to: {output_path}")
