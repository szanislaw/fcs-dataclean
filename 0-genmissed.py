import pandas as pd

# Step 1: Load the original variant match Excel file
df = pd.read_excel("variantmatch-1-8-n-gram-240725.xlsx")

# Step 2: Filter rows where no match was found in the master variant list
# These are marked with "no" (not boolean False) in the 'Appears in Variant List' column
df_unmatched = df[df['Appears in Variant List'].str.lower().str.strip() == 'no'].copy()

# Step 3: Extract only the relevant columns
df_output = df_unmatched[['service_item_name', 'Guest said']].rename(
    columns={
        'service_item_name': 'JO Service Item',
        'Guest said': 'Missed Variant'
    }
)

# Step 4: Save to Excel
output_path = "missed_variants_for_addition24.xlsx"
df_output.to_excel(output_path, index=False)

print(f"✅ File saved to: {output_path}")
