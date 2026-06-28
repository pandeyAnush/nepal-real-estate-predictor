#!/usr/bin/env python3
"""
Step 1: Text Preprocessing (Fixed Version)
Handles NLTK punkt_tab issue
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
print("Setting up NLTK...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Try both punkt versions
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading punkt_tab...")
        nltk.download('punkt_tab')

print("✓ NLTK ready\n")

# Setup tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def simple_tokenize(text):
    """
    Simple tokenization - split by spaces
    Works without punkt_tab
    """
    tokens = text.split()
    return tokens


def clean_text(text):
    """
    Step 1: Clean the text
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_stopwords(tokens):
    """
    Step 2: Remove common words
    """
    filtered = [word for word in tokens if word not in stop_words and len(word) > 1]
    return filtered


def lemmatize_words(tokens):
    """
    Step 3: Convert to base form
    """
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized


def preprocess_title(title):
    """
    Complete pipeline
    """
    if pd.isna(title):
        return []
    
    title = str(title)
    
    # Clean
    cleaned = clean_text(title)
    
    # Tokenize (simple version - no punkt needed)
    tokens = simple_tokenize(cleaned)
    
    # Remove stopwords
    filtered = remove_stopwords(tokens)
    
    # Lemmatize
    lemmatized = lemmatize_words(filtered)
    
    return lemmatized



# MAIN


print("=" * 80)
print("STEP 1: TEXT PREPROCESSING")
print("=" * 80 + "\n")

# Load data
print("Loading data...")
df = pd.read_csv('cleaned_realestate_all_categories_final.csv')
print(f"✓ Loaded {len(df):,} properties\n")

# Process titles
print(f"Processing {len(df):,} property titles...")
all_tokens = []

for idx, title in enumerate(df['title'].fillna('')):
    tokens = preprocess_title(title)
    all_tokens.append(tokens)
    
    if (idx + 1) % 500 == 0:
        print(f"  → {idx + 1:,} titles processed")

print(f"✓ Finished!\n")

# Add to dataframe
df['tokens'] = all_tokens
df['tokens_text'] = df['tokens'].apply(lambda x: ' '.join(x))
df['num_tokens'] = df['tokens'].apply(len)



# EXAMPLES


print("=" * 80)
print("EXAMPLES")
print("=" * 80 + "\n")

for i in range(min(3, len(df))):
    print(f"Example {i + 1}:")
    print(f"  Original:  {df['title'].iloc[i]}")
    print(f"  Tokens:    {df['tokens'].iloc[i]}")
    print(f"  Count:     {df['num_tokens'].iloc[i]} tokens\n")



# STATISTICS


print("=" * 80)
print("STATISTICS")
print("=" * 80 + "\n")

min_tokens = df['num_tokens'].min()
max_tokens = df['num_tokens'].max()
avg_tokens = df['num_tokens'].mean()
median_tokens = df['num_tokens'].median()

print(f"Token counts:")
print(f"  Min:     {min_tokens}")
print(f"  Max:     {max_tokens}")
print(f"  Average: {avg_tokens:.1f}")
print(f"  Median:  {median_tokens:.0f}\n")

print(f"Properties by type:")
for cat, count in df['category'].value_counts().items():
    print(f"  {cat:15s}: {count:,}")

print(f"\nTotal properties: {len(df):,}\n")



# SAVE


print("=" * 80)
print("SAVING FILES")
print("=" * 80 + "\n")

# Full data
df.to_csv('nlp_step1_preprocessed_full.csv', index=False)
print("✓ nlp_step1_preprocessed_full.csv")

# Tokens only
output_cols = ['title', 'tokens', 'tokens_text', 'num_tokens', 'category', 'location', 'price_numeric']
output_df = df[output_cols]
output_df.to_csv('nlp_step1_tokens.csv', index=False)
print("✓ nlp_step1_tokens.csv")

# Report
with open('nlp_step1_report.txt', 'w') as f:
    f.write("TEXT PREPROCESSING REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total properties: {len(df):,}\n")
    f.write(f"Min tokens: {min_tokens}\n")
    f.write(f"Max tokens: {max_tokens}\n")
    f.write(f"Average tokens: {avg_tokens:.1f}\n")
    f.write(f"Median tokens: {median_tokens:.0f}\n\n")

print("✓ nlp_step1_report.txt\n")

print("=" * 80)
print("✅ STEP 1 COMPLETE!")
print("=" * 80 + "\n")

print("Next: Step 2 - TF-IDF Vectorization\n")