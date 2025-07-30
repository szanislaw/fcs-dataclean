import pandas as pd
import re

# Load Excel file
df = pd.read_excel("missed_variants_for_addition24.xlsx")

# Define the improved condensing function
def improved_condense(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove polite/filler phrases
    removal_patterns = [
        r"\b(hi|hello|hey|thanks|thank you|no thank you|okay|yeah|um|you know|i think|please|just|can you please|could you please)\b",
        r"\b(i can see that|i would like to|could you|can i|get me|may i|i need|do you have|i want|would like|can you)\b"
    ]
    for pattern in removal_patterns:
        text = re.sub(pattern, '', text)

    # Remove leading 'the'
    text = re.sub(r"^\s*the\s+", "", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", '', text)

    # Remove Chinese characters
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)

    # Remove trailing filler endings
    text = re.sub(r"\b(no|thanks|thank you|okay|agno)\b\s*$", "", text)

    # Remove trailing general request phrases (non-specific action requests)
    trailing_phrases = [
        r"be inspected$", r"please assist$", r"can assist$", r"need checking$",
        r"please help$", r"needs to be fixed$", r"needs repair$", r"please check$", r"can help$"
    ]
    for phrase in trailing_phrases:
        text = re.sub(phrase, '', text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Apply the function
df['Condensed Variant'] = df['Missed Variant'].apply(improved_condense)

# Save to Excel
output_path = "final_improved_condensed_missed_variants24.xlsx"
df.to_excel(output_path, index=False)

print(f"✅ Condensed file saved to: {output_path}")
