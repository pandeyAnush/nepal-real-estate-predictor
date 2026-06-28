#!/usr/bin/env python3
"""
Step 4: Binary Features
Pulls simple yes/no flags out of property titles (is_bungalow, has_parking, etc.)
Input:  nlp_step1_preprocessed_full.csv   (titles + tokens from Step 1)
Output: nlp_step4_binary_features.csv      (one 0/1 column per keyword)
"""

import pandas as pd
import os

print("\n" + "=" * 80)
print("STEP 4: BINARY FEATURES")
print("=" * 80)



# Load the Step 1 data (auto-detects whichever file name exists)

def load_step1_data():
    """Loads Step 1 output. We need the original 'title' text for keyword search."""
    for filename in ['nlp_step1_tokens.csv', 'nlp_step1_preprocessed_full.csv']:
        if os.path.exists(filename):
            print(f"\nLoading: {filename}")
            return pd.read_csv(filename)
    raise FileNotFoundError(
        "Could not find Step 1 output. Expected 'nlp_step1_tokens.csv' "
        "or 'nlp_step1_preprocessed_full.csv' in this folder."
    )


df = load_step1_data()
print(f"\u2713 Loaded {len(df):,} properties")



# The keyword groups we want to detect
# Each entry becomes one 0/1 column. A row gets a 1 if the title contains
# ANY of the listed words.

feature_keywords = {
    # --- property type ---
    'is_bungalow':   ['bungalow'],
    'is_villa':      ['villa'],
    'is_apartment':  ['apartment', 'flat'],
    'is_duplex':     ['duplex'],
    'is_house':      ['house', 'home'],
    'is_land':       ['land', 'plot', 'ghaderi'],

    # --- features / amenities ---
    'has_parking':   ['parking', 'garage'],
    'has_garden':    ['garden', 'lawn'],
    'has_bhk':       ['bhk'],
    'is_furnished':  ['furnished'],

    # --- quality / condition ---
    'is_modern':     ['modern', 'new', 'newly'],
    'is_luxury':     ['luxury', 'luxurious', 'grande', 'premium'],
    'is_commercial': ['commercial', 'office', 'shutter'],

    # --- urgency / listing tone ---
    'is_urgent':     ['urgent', 'cheap', 'sale'],
}



# Build each binary column by searching the lowercased title

def contains_any(text, keywords):
    """Returns 1 if the text contains any of the keywords, else 0."""
    text = str(text).lower()
    for kw in keywords:
        if kw in text:
            return 1
    return 0


print("\nExtracting binary features from titles...")

# Search against the original title (richer than tokens, keeps all words)
title_series = df['title'].fillna('')

binary_data = {}
for feature_name, keywords in feature_keywords.items():
    binary_data[feature_name] = title_series.apply(lambda t: contains_any(t, keywords))

binary_df = pd.DataFrame(binary_data)
print(f"\u2713 Created {len(feature_keywords)} binary features")



# Show how often each flag fires

print("\n" + "=" * 80)
print("HOW OFTEN EACH FLAG IS 1")
print("=" * 80 + "\n")

for feature_name in feature_keywords:
    count = int(binary_df[feature_name].sum())
    pct = count / len(df) * 100
    bar = "#" * int(pct / 2)   # simple text bar, 1 char per 2%
    print(f"  {feature_name:14s} {count:5d}  ({pct:5.1f}%)  {bar}")



# A few example rows so you can eyeball that the flags are correct

print("\n" + "=" * 80)
print("EXAMPLES")
print("=" * 80)

for i in range(min(4, len(df))):
    active = [name for name in feature_keywords if binary_df[name].iloc[i] == 1]
    print(f"\n  Title: {df['title'].iloc[i]}")
    print(f"  Flags set: {active if active else '(none)'}")



# Assemble output: keep a little metadata next to the flags

meta_cols = [c for c in ['title', 'category', 'location', 'price_numeric']
             if c in df.columns]
full_df = pd.concat([df[meta_cols].reset_index(drop=True), binary_df], axis=1)



# Save

print("\n" + "=" * 80)
print("SAVING FILES")
print("=" * 80)

binary_df.to_csv('nlp_step4_binary_features.csv', index=False)
print("\u2713 nlp_step4_binary_features.csv        (flags only, use for Step 5)")

full_df.to_csv('nlp_step4_binary_features_full.csv', index=False)
print("\u2713 nlp_step4_binary_features_full.csv   (flags + title/category/location)")

print("\n" + "=" * 80)
print("\u2705 STEP 4 COMPLETE")
print("=" * 80)
print(f"\n  {len(df):,} properties \u00d7 {len(feature_keywords)} binary features")
print("\nNext: Step 5 \u2014 Combine TF-IDF + Word2Vec + Binary into one feature table\n")