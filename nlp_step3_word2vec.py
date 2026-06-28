#!/usr/bin/env python3
"""
Step 3: Word2Vec Embeddings
Converts property titles into semantic word vectors (meaning, not just importance)
Input:  nlp_step1_preprocessed_full.csv  (tokenized titles from Step 1)
Output: nlp_step3_embeddings.csv         (100-dimensional vector per property)
"""

import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print("STEP 3: WORD2VEC EMBEDDINGS")
print("=" * 80)



# Load the tokenized data from Step 1
# Tries the tokens file first, then falls back to the full preprocessed file

def load_step1_data():
    """
    Loads the Step 1 output. Both files contain a 'tokens_text' column
    (space-separated tokens), which is all Word2Vec needs.
    """
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



# Build the list of token sequences (one list of words per property)
# Word2Vec learns from these word sequences

def build_sentences(dataframe):
    """
    Turns the 'tokens_text' column into a list of token lists.
    Example: "modern villa garden" -> ['modern', 'villa', 'garden']
    """
    sentences = []
    for text in dataframe['tokens_text'].fillna(''):
        tokens = text.split() if text else []
        sentences.append(tokens)
    return sentences


print("\nPreparing token sequences for training...")
sentences = build_sentences(df)
print(f"\u2713 {len(sentences):,} sequences ready")

print("\nSample sequences:")
for i in range(min(3, len(sentences))):
    preview = sentences[i][:6]
    print(f"  {i + 1}. {preview} ({len(sentences[i])} tokens)")



# Train the Word2Vec model
# This is where the model learns a 100-number vector for each word

print("\n" + "=" * 80)
print("TRAINING WORD2VEC MODEL")
print("=" * 80)
print("""
Settings:
  vector_size = 100   each word becomes a 100-number vector
  window      = 5     looks at 5 words before and after for context
  min_count   = 3     ignores words appearing fewer than 3 times
  epochs      = 20    reads the data 20 times to refine the vectors
""")
print("Training... (this takes 1-3 minutes)")

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=3,
    workers=4,
    epochs=20,
    seed=42,
)

print(f"\u2713 Training complete")
print(f"  Vocabulary size: {len(model.wv):,} unique words")
print(f"  Vector size: 100 dimensions")



# Sanity check: look at the words the model thinks are similar
# This is how we confirm the model learned meaningful relationships

print("\n" + "=" * 80)
print("CHECKING LEARNED RELATIONSHIPS")
print("=" * 80)

for word in ['house', 'land', 'villa', 'modern']:
    if word in model.wv:
        similar = model.wv.most_similar(word, topn=3)
        print(f"\n  Words most similar to '{word}':")
        for sim_word, score in similar:
            print(f"    {sim_word:18s} {score:.3f}")
    else:
        print(f"\n  '{word}' not in vocabulary (too rare)")



# Convert each property to a single 100-number vector
# Method: average the vectors of all words in that property's title

def property_to_vector(tokens, w2v_model, dim=100):
    """
    Builds one vector for a property by averaging its word vectors.
    If no words are in the vocabulary, returns a zero vector.
    """
    vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if len(vectors) == 0:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)


print("\n" + "=" * 80)
print("CREATING PROPERTY EMBEDDINGS")
print("=" * 80)
print("\nAveraging word vectors into one vector per property...")

property_vectors = []
for idx, tokens in enumerate(sentences):
    property_vectors.append(property_to_vector(tokens, model))
    if (idx + 1) % 1000 == 0:
        print(f"  \u2192 {idx + 1:,} done")

print(f"\u2713 {len(property_vectors):,} property embeddings created")



# Assemble the output dataframe

embedding_cols = [f'w2v_{i}' for i in range(100)]
embeddings_df = pd.DataFrame(property_vectors, columns=embedding_cols)

meta_cols = [c for c in ['title', 'tokens_text', 'category', 'location', 'price_numeric']
             if c in df.columns]
full_df = pd.concat([df[meta_cols].reset_index(drop=True), embeddings_df], axis=1)



# Quick stats so you can see the result is sensible

print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)

values = embeddings_df.values
zero_rows = int(np.all(embeddings_df == 0, axis=1).sum())

print(f"\nEmbedding values:")
print(f"  Min:  {values.min():.4f}")
print(f"  Max:  {values.max():.4f}")
print(f"  Mean: {values.mean():.4f}")
print(f"\nProperties with all-zero vector: {zero_rows} ({zero_rows / len(df) * 100:.2f}%)")
print("  (these had no words frequent enough to be in the vocabulary)")



# Save everything

print("\n" + "=" * 80)
print("SAVING FILES")
print("=" * 80)

embeddings_df.to_csv('nlp_step3_embeddings.csv', index=False)
print("\u2713 nlp_step3_embeddings.csv        (100 columns, use for Step 5)")

full_df.to_csv('nlp_step3_embeddings_full.csv', index=False)
print("\u2713 nlp_step3_embeddings_full.csv   (with title/category/location)")

model.save('nlp_step3_word2vec_model.bin')
print("\u2713 nlp_step3_word2vec_model.bin    (trained model)")

with open('nlp_step3_vocabulary.txt', 'w') as f:
    for word in sorted(model.wv.index_to_key):
        f.write(word + '\n')
print(f"\u2713 nlp_step3_vocabulary.txt        ({len(model.wv)} words)")

print("\n" + "=" * 80)
print("\u2705 STEP 3 COMPLETE")
print("=" * 80)
print(f"\n  {len(df):,} properties \u2192 100-dimensional embeddings")
print(f"  Vocabulary: {len(model.wv):,} words")
print("\nNext: Step 4 \u2014 Binary features (is_bungalow, is_villa, etc.)\n")