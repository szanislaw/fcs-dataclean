import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


# === Config ===
CONFIG = {
    "xlsx_path": "data/yotel-service-items-listing-7jul.xlsx",
    "csv_path": "data/yotel-job-items-variants.csv",
    "intermediate_path": "temp/yotel_merged_output-missing.xlsx",
    "filtered_path": "temp/filtered_no_chinese.xlsx",
    "final_output_path": "temp/cleaned_variants.xlsx",
    "semantic_output_path": "temp/semantic_matching_results.xlsx",
    "updated_output_path": "temp/cleaned_variants_updated.xlsx",
    "final_cleaned_output": "temp/cleaned_variants_updated_no_chinese.xlsx",
    "final_uuid_output": "final-job-variant.xlsx",
}


def contains_chinese(text: str) -> bool:
    return isinstance(text, str) and bool(re.search(r'[\u4e00-\u9fff]', text))


def merge_datasets(xlsx_df: pd.DataFrame, csv_df: pd.DataFrame) -> pd.DataFrame:
    xlsx_df["UUID"] = xlsx_df["UUID"].astype(str)
    csv_df.iloc[:, 0] = csv_df.iloc[:, 0].astype(str)

    combined_rows = []
    for _, x_row in xlsx_df.iterrows():
        uuid = x_row["UUID"]
        combined_rows.append(x_row.tolist())
        for _, c_row in csv_df[csv_df.iloc[:, 0] == uuid].iterrows():
            combined_rows.append([uuid] + [""] * (len(x_row) - 1) + [c_row.iloc[1]])

    # Unmatched UUIDs
    combined_rows.append([""] * len(xlsx_df.columns) + ["-- UNMATCHED UUIDs BELOW --"])
    unmatched_uuids = set(csv_df.iloc[:, 0]) - set(xlsx_df["UUID"])
    for uuid in unmatched_uuids:
        variants = csv_df[csv_df.iloc[:, 0] == uuid].iloc[:, 1].tolist()
        if variants:
            combined_rows.append([uuid] + [""] * (len(xlsx_df.columns) - 2) + [variants[0]])
            for v in variants[1:]:
                combined_rows.append([uuid] + [""] * (len(xlsx_df.columns) - 1) + [v])

    columns = list(xlsx_df.columns) + ["Variant Info"]
    return pd.DataFrame(combined_rows, columns=columns)


def remove_chinese_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.apply(lambda col: col.map(contains_chinese)).any(axis=1)]


def clean_and_flatten_variants(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    df["JO Service Item"] = df["JO Service Item"].ffill()
    df = df.dropna(subset=["Variant Info"])
    return df[["JO Service Item", "Variant Info"]]


def compute_semantic_matches(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    df_master = df.iloc[:52120]
    df_unmatched = df.iloc[52120:]
    master_items = list(set(df_master["JO Service Item"].dropna()))
    unmatched_items = list(set(df_unmatched["JO Service Item"].dropna()))

    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    master_emb = model.encode(master_items, convert_to_tensor=True).cpu()
    unmatched_emb = model.encode(unmatched_items, convert_to_tensor=True).cpu()
    cos_sim = cosine_similarity(unmatched_emb, master_emb)

    matched_results = []
    for i, item in enumerate(tqdm(unmatched_items, desc="Matching")):
        best_idx = np.argmax(cos_sim[i])
        matched_results.append((item, master_items[best_idx], round(cos_sim[i][best_idx], 4)))

    return pd.DataFrame(matched_results, columns=["Unmatched Item", "Matched Master Item", "Similarity Score"])


def append_strong_matches(df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    strong_matches = matches[matches["Similarity Score"] >= 0.8].copy()
    new_rows = strong_matches.rename(columns={"Matched Master Item": "JO Service Item", "Unmatched Item": "Variant Info"})
    return pd.concat([df, new_rows[["JO Service Item", "Variant Info"]]], ignore_index=True)


def restore_uuids(final_df: pd.DataFrame, uuid_df: pd.DataFrame) -> pd.DataFrame:
    uuid_df.columns = ["UUID", "Variant Info"]
    final_df.columns = final_df.columns.str.strip()
    merged = pd.merge(final_df, uuid_df, how="left", on="Variant Info")
    cols = ["UUID"] + [c for c in merged.columns if c != "UUID"]
    return merged[cols]


def main():
    # === Load source data ===
    xlsx_df = pd.read_excel(CONFIG["xlsx_path"])
    csv_df = pd.read_csv(CONFIG["csv_path"], header=None)

    # === Merge and Save ===
    merged_df = merge_datasets(xlsx_df, csv_df)
    merged_df.to_excel(CONFIG["intermediate_path"], index=False)
    print(f"✅ Merged file saved to: {CONFIG['intermediate_path']}")

    # === Filter Chinese rows ===
    df_no_chinese = remove_chinese_rows(merged_df)
    df_no_chinese.to_excel(CONFIG["filtered_path"], index=False)
    print(f"✅ Chinese-filtered file saved to: {CONFIG['filtered_path']}")

    # === Clean and Flatten ===
    final_variants = clean_and_flatten_variants(CONFIG["filtered_path"])
    final_variants.to_excel(CONFIG["final_output_path"], index=False)
    print(f"✅ Cleaned file saved as: {CONFIG['final_output_path']}")

    # === Semantic Matching ===
    print("🔗 semantic-similarity-match")
    matched_df = compute_semantic_matches(final_variants)
    matched_df.to_excel(CONFIG["semantic_output_path"], index=False)
    print(f"✅ Saved semantic matches to: {CONFIG['semantic_output_path']}")

    # === Append Strong Matches ===
    updated_df = append_strong_matches(final_variants, matched_df)
    updated_df.to_excel(CONFIG["updated_output_path"], index=False)
    print(f"✅ Updated file saved as: {CONFIG['updated_output_path']}")

    # === Final Filter ===
    df_final_clean = remove_chinese_rows(updated_df)
    df_final_clean.to_excel(CONFIG["final_cleaned_output"], index=False)
    print(f"✅ Final file with Chinese characters removed saved as: {CONFIG['final_cleaned_output']}")

    # === Restore UUIDs ===
    print("🔗 Matching UUIDs to final variants...")
    variants_df = pd.read_excel(CONFIG["final_cleaned_output"])
    uuid_restored = restore_uuids(variants_df, csv_df)
    uuid_restored.to_excel(CONFIG["final_uuid_output"], index=False)
    print(f"✅ Final file with UUIDs added saved as: {CONFIG['final_uuid_output']}")


if __name__ == "__main__":
    main()
