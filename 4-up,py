from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load data
df = pd.read_excel("cleaned_variants.xlsx")
df.columns = df.columns.str.strip()
df_master = df.iloc[:52120].copy()
df_unmatched = df.iloc[52120:].copy()

# Get unique JO items
master_items = list(set(df_master['JO Service Item'].dropna()))
unmatched_items = list(set(df_unmatched['JO Service Item'].dropna()))

# Load sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode all sentences
print("Encoding master items...")
master_embeddings = model.encode(master_items, convert_to_tensor=True).cpu()
print("Encoding unmatched items...")
unmatched_embeddings = model.encode(unmatched_items, convert_to_tensor=True).cpu()

# Compute pairwise cosine similarity
print("Computing cosine similarities...")
cos_sim = cosine_similarity(unmatched_embeddings, master_embeddings)

# Find best matches with tqdm progress bar
matched_results = []
print("Matching items...")
for i, unmatched in enumerate(tqdm(unmatched_items, desc="Matching")):
    best_idx = np.argmax(cos_sim[i])
    best_match = master_items[best_idx]
    score = cos_sim[i][best_idx]
    matched_results.append((unmatched, best_match, round(score, 4)))

# Save results
matched_df = pd.DataFrame(matched_results, columns=['Unmatched Item', 'Matched Master Item', 'Similarity Score'])
matched_df.to_excel("semantic_matching_results.xlsx", index=False)
print("✅ Saved as 'semantic_matching_results.xlsx'")
