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
