import re
import pandas as pd

# Function to detect Chinese characters
def contains_chinese(text):
    if isinstance(text, str):
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    return False

# Load updated file
df_updated = pd.read_excel("cleaned_variants_updated.xlsx")
df_updated.columns = df_updated.columns.str.strip()

# Remove rows where any cell contains Chinese characters
df_no_chinese = df_updated[~df_updated.applymap(contains_chinese).any(axis=1)]

# Save cleaned version
df_no_chinese.to_excel("cleaned_variants_updated_no_chinese.xlsx", index=False)
print("✅ Chinese-character rows removed. Saved as 'cleaned_variants_updated_no_chinese.xlsx'")
