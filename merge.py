import pandas as pd

# === File paths ===
xlsx_path = "Yotel Service Items listing 07 July.xlsx"
csv_path = "Yotel job_items variants.csv"
output_path = "Yotel_Merged_Output.xlsx"

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

# csv
output_path = "Yotel_Merged_Output.csv"
output_df.to_csv(output_path, index=False)
print(f"Merged file saved to: {output_path}")

