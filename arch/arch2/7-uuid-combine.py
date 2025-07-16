import pandas as pd

# Load the files
variants_df = pd.read_excel("cleaned_variants_updated_no_chinese.xlsx")
uuid_df = pd.read_csv("data/yotel-job-items-variants.csv", header=None)

# Set column names
uuid_df.columns = ["UUID", "Variant Info"]

# Match UUIDs to Variant Info
merged_df = pd.merge(variants_df, uuid_df, how='left', left_on='Variant Info', right_on='Variant Info')

# Reorder so UUID is first
final_df = merged_df[["UUID"] + [col for col in merged_df.columns if col != "UUID"]]

# Save the result
final_df.to_excel("cleaned_variants_with_uuid.xlsx", index=False)
