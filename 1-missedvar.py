import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Load your Excel file
df = pd.read_excel("missed_variants_for_addition.xlsx")

# Smart cleaner using NLTK tokenization
def smart_condense_variant(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove polite/filler phrases
    polite_patterns = [
        r"\b(hi|hello|hey|thanks|thank you|no thank you|okay|yeah|um|you know|i think|please|just)\b",
        r"\b(i can see that|i would like to|could you|can i|get me|may i|i need|do you have|i want|would like|can you please)\b"
    ]
    for pattern in polite_patterns:
        text = re.sub(pattern, '', text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Keep only relevant tokens (remove short filler but preserve technical terms like 'tv', 'ac')
    keep_tokens = [t for t in tokens if len(t) > 2 or t in {'tv', 'ac', 'fan', 'tap', 'bed', 'aircon'}]

    return " ".join(keep_tokens).strip()

# Apply to the DataFrame
df['Condensed Variant'] = df['Missed Variant'].apply(smart_condense_variant)

# Export to Excel
df.to_excel("final_nltk_condensed_variants.xlsx", index=False)
