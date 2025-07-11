import pandas as pd

# Load the Excel file
file_path = "arch/filtered_no_chinese.xlsx"
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
