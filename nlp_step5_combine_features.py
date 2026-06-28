#!/usr/bin/env python3
"""
Step 5: Combine All Features
Glues together every feature set into ONE table for machine learning.
Inputs:
  - cleaned_realestate_all_categories_final.csv   (original numbers: price, area, beds, baths)
  - nlp_step2_tfidf_features.csv                   (100 TF-IDF columns)
  - nlp_step3_embeddings.csv                       (100 Word2Vec columns)
  - nlp_step4_binary_features.csv                  (14 binary flag columns)
Output:
  - nlp_features_final.csv                         (one row per property, all features)
"""

import pandas as pd
import os

print("\n" + "=" * 80)
print("STEP 5: COMBINE ALL FEATURES")
print("=" * 80)



def load_required(filename, description):
    """Loads a CSV that must exist. Stops the script clearly if it's missing."""
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Missing {description}: '{filename}' not found in this folder.\n"
            f"  Make sure you ran the step that creates it before Step 5."
        )
    df = pd.read_csv(filename)
    print(f"  \u2713 {description:22s} {len(df):5,} rows \u00d7 {df.shape[1]:3d} cols   ({filename})")
    return df



print("\nLoading feature files...")

original = load_required('cleaned_realestate_all_categories_final.csv', 'Original data')
tfidf    = load_required('nlp_step2_tfidf_features.csv',                 'TF-IDF features')
embed    = load_required('nlp_step3_embeddings.csv',                     'Word2Vec features')
binary   = load_required('nlp_step4_binary_features.csv',                'Binary features')

print("\nChecking row counts match...")
counts = {
    'Original': len(original),
    'TF-IDF':   len(tfidf),
    'Word2Vec': len(embed),
    'Binary':   len(binary),
}
unique_counts = set(counts.values())

if len(unique_counts) != 1:
    print("  \u2717 ROW COUNTS DO NOT MATCH:")
    for name, n in counts.items():
        print(f"      {name:10s}: {n:,}")
    raise ValueError(
        "Files have different row counts, so they cannot be safely combined.\n"
        "  Re-run the earlier steps so every file comes from the same source in the same order."
    )

n_rows = unique_counts.pop()
print(f"  \u2713 All files have {n_rows:,} rows")



numeric_cols = [c for c in ['price_numeric', 'land_area_sqft', 'bedrooms',
                            'bathrooms', 'price_per_sqft', 'bed_bath_ratio']
                if c in original.columns]


label_cols = [c for c in ['category', 'location'] if c in original.columns]

print(f"\nUsing {len(numeric_cols)} numeric columns from original data:")
print(f"  {numeric_cols}")





tfidf = tfidf.add_prefix('tfidf_')

# reset index on all so concat lines up purely by row position
original_part = original[label_cols + numeric_cols].reset_index(drop=True)
tfidf         = tfidf.reset_index(drop=True)
embed         = embed.reset_index(drop=True)
binary        = binary.reset_index(drop=True)





print("\nCombining all features side by side...")

final = pd.concat([original_part, tfidf, embed, binary], axis=1)

print(f"  \u2713 Combined table: {final.shape[0]:,} rows \u00d7 {final.shape[1]} columns")


print("\n" + "=" * 80)
print("COLUMN BREAKDOWN")
print("=" * 80)
print(f"  Label columns (reference): {len(label_cols)}   {label_cols}")
print(f"  Original numeric features: {len(numeric_cols)}")
print(f"  TF-IDF features:           {tfidf.shape[1]}")
print(f"  Word2Vec features:         {embed.shape[1]}")
print(f"  Binary features:           {binary.shape[1]}")
print(f"  {'-' * 40}")
feature_total = len(numeric_cols) + tfidf.shape[1] + embed.shape[1] + binary.shape[1]
print(f"  TOTAL model features:      {feature_total}")
print(f"  (plus {len(label_cols)} label columns kept for reference)")



print("\n" + "=" * 80)
print("QUALITY CHECK")
print("=" * 80)

missing = final.isnull().sum().sum()
print(f"  Total cells:    {final.shape[0] * final.shape[1]:,}")
print(f"  Missing values: {missing}")
if missing == 0:
    print("  \u2713 No missing values \u2014 ready for machine learning")
else:
    print("  \u26a0 Some missing values found. Columns affected:")
    for col in final.columns:
        m = final[col].isnull().sum()
        if m > 0:
            print(f"      {col}: {m}")



# Save the final table

print("\n" + "=" * 80)
print("SAVING")
print("=" * 80)

final.to_csv('nlp_features_final.csv', index=False)
print("  \u2713 nlp_features_final.csv")

# also save the list of feature column names (excludes label columns)
feature_names = [c for c in final.columns if c not in label_cols]
with open('nlp_features_final_columns.txt', 'w') as f:
    for name in feature_names:
        f.write(name + '\n')
print(f"  \u2713 nlp_features_final_columns.txt   ({len(feature_names)} feature names)")

print("\n" + "=" * 80)
print("\u2705 STEP 5 COMPLETE \u2014 NLP PHASE DONE")
print("=" * 80)
print(f"\n  Final table: {final.shape[0]:,} properties \u00d7 {feature_total} features")
print("\n  This single file (nlp_features_final.csv) is what the")
print("  machine learning model will train on next.")
print("\nNext: train a price-prediction model (Random Forest / XGBoost)\n")