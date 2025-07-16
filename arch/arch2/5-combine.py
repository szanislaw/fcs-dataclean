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
