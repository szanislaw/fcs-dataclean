import pandas as pd
import spacy

# Load your file
df = pd.read_excel("final_improved_condensed_missed_variants23.xlsx")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract meaningful and natural-sounding phrases
def extract_natural_variant(text):
    if not isinstance(text, str):
        return ""

    doc = nlp(text.lower())
    
    # Keep tokens that are part of core meaning (verbs, nouns, adjectives, auxiliaries)
    tokens_to_keep = []
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "VERB", "AUX", "ADJ"} and not token.is_stop:
            tokens_to_keep.append(token.text)
        elif token.pos_ in {"DET", "ADP", "PART"} and len(tokens_to_keep) > 0:
            # Keep these if they are part of a phrase like "is flickering"
            tokens_to_keep.append(token.text)

    return " ".join(tokens_to_keep)

# Apply to Condensed Variant
df['POS Cleaned Variant'] = df['Condensed Variant'].apply(extract_natural_variant)

# Save result
output_path = "pos_cleaned_variants24.xlsx"
df.to_excel(output_path, index=False)

print(f"✅ Saved to: {output_path}")
