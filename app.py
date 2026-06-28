#!/usr/bin/env python3
# anush

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

st.set_page_config(page_title="Nepal Real Estate Price Predictor", page_icon="house")
st.title("Nepal Real Estate Price Predictor")
st.write("Estimate a property's price from its details. Trained on ~4,500 Nepali listings.")



def load_word_vectors(path="word_vectors.txt"):
    vectors = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().split()
        # header is "<count> <dim>"
        try:
            dim = int(header[1])
        except (ValueError, IndexError):
            # no header line: treat it as a data line and infer dim
            word = header[0]
            vectors[word] = np.array([float(x) for x in header[1:]], dtype=np.float32)
            dim = len(header) - 1
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) < 2:
                continue
            vectors[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return vectors, dim


@st.cache_resource
def load_assets():
    with open('price_model_final.pkl', 'rb') as f:
        model = pickle.load(f)

    word_vectors, w2v_dim = load_word_vectors('word_vectors.txt')

    with open('price_model_final_features.txt') as f:
        feature_names = [line.strip() for line in f if line.strip()]

    rng = {}
    try:
        with open('price_model_range.txt') as f:
            for line in f:
                k, v = line.strip().split('=')
                rng[k] = float(v)
    except FileNotFoundError:
        rng = {'reliable_min': 0, 'reliable_max': 1e12}

    return model, word_vectors, w2v_dim, feature_names, rng


model, WORD_VECTORS, W2V_DIM, feature_names, price_range = load_assets()



def clean_and_tokenize(title):
    title = str(title).lower()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title.split()


def title_to_w2v(title):
    """Average the Word2Vec vectors of the words in the title (same as Step 3)."""
    tokens = clean_and_tokenize(title)
    vectors = [WORD_VECTORS[t] for t in tokens if t in WORD_VECTORS]
    if len(vectors) == 0:
        return np.zeros(W2V_DIM)
    return np.mean(vectors, axis=0)



def build_feature_row(title, area, bedrooms, bathrooms):
    row = {name: 0.0 for name in feature_names}

    if 'land_area_sqft' in row: row['land_area_sqft'] = area
    if 'bedrooms' in row:       row['bedrooms'] = bedrooms
    if 'bathrooms' in row:      row['bathrooms'] = bathrooms

    vec = title_to_w2v(title)
    for i in range(W2V_DIM):
        key = f'w2v_{i}'
        if key in row:
            row[key] = vec[i]

    return pd.DataFrame([row])[feature_names]


st.header("Property details")

title = st.text_input(
    "Property title / description",
    "Modern bungalow with parking and garden"
)

col1, col2, col3 = st.columns(3)
with col1:
    area = st.number_input("Land area (sq ft)", min_value=100, max_value=20000, value=1500)
with col2:
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=4)
with col3:
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=3)



if st.button("Predict price", type="primary"):
    features = build_feature_row(title, area, bedrooms, bathrooms)
    predicted = model.predict(features)[0]

    st.header("Estimated price")
    st.subheader(f"Rs. {predicted:,.0f}")

    crore = predicted / 10_000_000
    st.write(f"Approximately {crore:.2f} crore")

    if predicted < price_range['reliable_min'] or predicted > price_range['reliable_max']:
        st.warning(
            "This estimate is outside the price range the model was trained on "
            f"(Rs. {price_range['reliable_min']:,.0f} to Rs. {price_range['reliable_max']:,.0f}), "
            "so treat it with extra caution."
        )

    st.caption(
        "This model explains about 63% of price variation and is meant as a "
        "rough guide, not an appraisal. Actual prices depend heavily on exact "
        "location, road access, and other factors not captured here."
    )



st.divider()
st.caption(
    "Built on ~4,500 Nepali real estate listings. Features: land area, "
    "bedrooms, bathrooms, and the title's meaning via Word2Vec."
)