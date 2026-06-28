#!/usr/bin/env python3
"""
Step 2: TF-IDF Feature Extraction
Converts property titles into numerical features
Input: nlp_step1_tokens.csv (from Step 1)
Output: nlp_step2_tfidf.csv (100 features)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


print("=" * 80)
print("STEP 2: TF-IDF FEATURE EXTRACTION")
print("=" * 80 + "\n")

# Load data from Step 1
print("Loading preprocessed tokens from Step 1...")
df = pd.read_csv('nlp_step1_preprocessed_full.csv')
print(f"✓ Loaded {len(df):,} properties\n")

# Check if we have tokens
print(f"Checking token quality...")
print(f"  Total rows: {len(df):,}")
print(f"  Empty tokens: {(df['num_tokens'] == 0).sum()}")
print(f"  Average tokens per property: {df['num_tokens'].mean():.1f}\n")



# WHAT IS TF-IDF?

print("=" * 80)
print("UNDERSTANDING TF-IDF")
print("=" * 80 + "\n")

print("""
TF-IDF = Term Frequency - Inverse Document Frequency

Simple explanation:
- TF (Term Frequency): How often a word appears in a document
- IDF (Inverse Document Frequency): How rare the word is across all documents

Why it matters:
- "house" appears in 90% of properties → LOW TF-IDF (not informative)
- "bungalow" appears in 5% of properties → HIGH TF-IDF (very informative)
- "penthouse" appears in 1% of properties → VERY HIGH TF-IDF (most informative)

Result:
- Each property gets a score (0.0 to 1.0) for each important word
- Higher score = more important word for that property
- Creates numerical features for machine learning
""")

print("=" * 80)
print("CREATING TF-IDF VECTORIZER")
print("=" * 80 + "\n")

# Create vectorizer
# max_features=100: Keep only the top 100 most important words
# min_df=5: Word must appear in at least 5 properties
# max_df=0.8: Word can appear in at most 80% of properties (removes too-common words)
vectorizer = TfidfVectorizer(
    max_features=100,           # Extract top 100 words
    min_df=5,                   # Word appears in at least 5 properties
    max_df=0.8,                 # Word appears in at most 80% of properties
    lowercase=True,
    stop_words='english'
)

print("TF-IDF Settings:")
print(f"  Max features: 100 (top 100 words)")
print(f"  Min document frequency: 5 (word must appear in 5+ properties)")
print(f"  Max document frequency: 0.8 (word can appear in max 80% of properties)")
print(f"  Output range: 0.0 to 1.0 (importance score)\n")



# TRANSFORM TOKENS TO TF-IDF FEATURES

print("Transforming tokens to TF-IDF features...")
print("(This may take 1-2 minutes)\n")

# Convert tokens_text to TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(df['tokens_text'])

print(f"✓ TF-IDF matrix created\n")

# Get feature names (the 100 words we're using)
feature_names = vectorizer.get_feature_names_out()

print(f"Features (top 100 words):")
print(f"  {list(feature_names[:20])}...")  # Show first 20
print(f"  Total features: {len(feature_names)}\n")



# CONVERT TO DATAFRAME

print("Converting to DataFrame...")

# Create dataframe from TF-IDF matrix
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names
)

print(f"✓ DataFrame created")
print(f"  Shape: {tfidf_df.shape[0]:,} rows × {tfidf_df.shape[1]} columns\n")



# COMBINE WITH ORIGINAL DATA

print("Combining with original property data...")

# Keep original columns
original_cols = df[['title', 'tokens_text', 'category', 'location', 'price_numeric', 'num_tokens']]

# Add TF-IDF features
result_df = pd.concat([original_cols, tfidf_df], axis=1)

print(f"✓ Combined data")
print(f"  Total columns: {len(result_df.columns)}")
print(f"  - Original columns: 6")
print(f"  - TF-IDF features: 100\n")



# SHOW EXAMPLES

print("=" * 80)
print("EXAMPLES: HOW TF-IDF WORKS")
print("=" * 80 + "\n")

for i in range(min(3, len(result_df))):
    print(f"Property {i + 1}:")
    print(f"  Title: {result_df['title'].iloc[i]}")
    print(f"  Category: {result_df['category'].iloc[i]}")
    
    # Get non-zero TF-IDF scores
    scores = tfidf_df.iloc[i]
    top_words = scores[scores > 0].sort_values(ascending=False).head(5)
    
    print(f"  Top words (TF-IDF scores):")
    for word, score in top_words.items():
        print(f"    - {word:20s}: {score:.3f}")
    
    print()



# STATISTICS

print("=" * 80)
print("STATISTICS")
print("=" * 80 + "\n")

# Overall statistics
print("TF-IDF value distribution:")
tfidf_values = tfidf_df.values.flatten()
tfidf_values = tfidf_values[tfidf_values > 0]  # Exclude zeros

if len(tfidf_values) > 0:
    print(f"  Min (non-zero): {tfidf_values.min():.4f}")
    print(f"  Max: {tfidf_values.max():.4f}")
    print(f"  Mean (non-zero): {tfidf_values.mean():.4f}\n")

# How many words per property
non_zero_counts = (tfidf_df > 0).sum(axis=1)
print(f"Words found per property:")
print(f"  Min: {non_zero_counts.min()}")
print(f"  Max: {non_zero_counts.max()}")
print(f"  Average: {non_zero_counts.mean():.1f}\n")

# Most common words
print(f"Most common words (highest average TF-IDF):")
word_importance = tfidf_df.mean().sort_values(ascending=False).head(10)
for idx, (word, score) in enumerate(word_importance.items(), 1):
    print(f"  {idx:2d}. {word:20s}: {score:.4f}")

print()



# SAVE OUTPUT

print("=" * 80)
print("SAVING FILES")
print("=" * 80 + "\n")

# Save TF-IDF features only (for Step 3)
tfidf_df.to_csv('nlp_step2_tfidf_features.csv', index=False)
print("✓ nlp_step2_tfidf_features.csv - TF-IDF features (100 columns)")

# Save combined data (with original columns + TF-IDF)
result_df.to_csv('nlp_step2_tfidf_full.csv', index=False)
print("✓ nlp_step2_tfidf_full.csv - Full data with TF-IDF features")

# Save feature names
with open('nlp_step2_feature_names.txt', 'w') as f:
    f.write("TF-IDF FEATURE NAMES (100 words)\n")
    f.write("=" * 80 + "\n\n")
    for idx, word in enumerate(feature_names, 1):
        f.write(f"{idx:3d}. {word}\n")

print("✓ nlp_step2_feature_names.txt - List of 100 words\n")

# Save report
with open('nlp_step2_report.txt', 'w') as f:
    f.write("TF-IDF VECTORIZATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total properties: {len(result_df):,}\n")
    f.write(f"TF-IDF features: 100\n")
    f.write(f"Total output columns: {len(result_df.columns)}\n\n")
    f.write("Top 20 words:\n")
    for idx, (word, score) in enumerate(word_importance.head(20).items(), 1):
        f.write(f"  {idx:2d}. {word}: {score:.4f}\n")

print("✓ nlp_step2_report.txt - Statistics\n")



# SUMMARY

print("=" * 80)
print("✅ STEP 2 COMPLETE!")
print("=" * 80 + "\n")

print("What was created:")
print("  1. nlp_step2_tfidf_features.csv - TF-IDF features (4,648 × 100)")
print("     → Use this for next steps")
print("  2. nlp_step2_tfidf_full.csv - Full data with TF-IDF")
print("  3. nlp_step2_feature_names.txt - List of 100 words")
print("  4. nlp_step2_report.txt - Statistics\n")

print("Next: Step 3 - Word2Vec Embeddings")
print("Command: python3 nlp_step3_embeddings.py\n")