#!/usr/bin/env python3
"""
Nepal Real Estate Price Predictor — Streamlit app.

Loads a trained Random Forest and estimates a property price from its
title, land area, bedrooms, and bathrooms. The core prediction flow
mirrors the training pipeline exactly so predictions stay consistent.

Run with:  streamlit run app.py
"""

import re
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from gensim.models import Word2Vec



st.set_page_config(
    page_title="Nepal Real Estate Price Predictor",
    page_icon="🏠",
    layout="centered",
)


st.markdown(
    """
    <style>
    .conf-high   { color: #4caf50; font-weight: 700; font-size: 1.1rem; }
    .conf-medium { color: #ff9800; font-weight: 700; font-size: 1.1rem; }
    .conf-low    { color: #f44336; font-weight: 700; font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    """Load the RF model, Word2Vec embeddings, feature list, and price range.

    st.cache_resource keeps these in memory across every Streamlit re-run.
    Without it the W2V binary (~tens of MB) would reload on every button click.
    """
    with open("price_model_final.pkl", "rb") as f:
        model = pickle.load(f)

    w2v = Word2Vec.load("nlp_step3_word2vec_model.bin")

    with open("price_model_final_features.txt") as f:
        features = [line.strip() for line in f if line.strip()]


    price_range = {"reliable_min": 0, "reliable_max": 1e12}
    try:
        with open("price_model_range.txt") as f:
            for line in f:
                k, v = line.strip().split("=")
                price_range[k] = float(v)
    except FileNotFoundError:
        pass  

    return model, w2v, features, price_range


model, w2v, feature_names, price_range = load_assets()




def clean_and_tokenize(text):
    """Lowercase, strip non-alphanumeric characters, split into word tokens.

    This function must stay byte-for-byte identical to Step 1 preprocessing.
    Any change here would produce different tokens than the model trained on,
    silently breaking the W2V lookup and degrading every prediction.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def embed_title(title):
    """Return a 100-d meaning vector for the title (same as Step 3 training).

    Words absent from the W2V vocabulary are skipped rather than treated as
    zeros — including them would dilute the average toward the origin and make
    the embedding less representative of the words that ARE recognised.
    Returns a zero vector if nothing in the title is in the vocabulary.
    """
    tokens = clean_and_tokenize(title)
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)


def build_feature_row(title, area, bedrooms, bathrooms):
    """Assemble the single-row DataFrame the Random Forest expects.

    Every column starts at 0. Then we fill the three numeric slots and the
    100 W2V dimensions. TF-IDF and binary-flag columns stay at 0 because
    the training analysis showed they carry very little weight in this model.
    """
    row = {name: 0.0 for name in feature_names}

    for col, val in [("land_area_sqft", area), ("bedrooms", bedrooms), ("bathrooms", bathrooms)]:
        if col in row:
            row[col] = val

    vec = embed_title(title)
    for i in range(100):
        key = f"w2v_{i}"
        if key in row:
            row[key] = vec[i]

    return pd.DataFrame([row])[feature_names]



PROPERTY_KEYWORDS = {
    "house", "land", "villa", "bungalow", "apartment", "flat", "plot",
    "ghar", "jagga", "room", "bhk", "home", "property", "commercial",
    "residential", "floor", "storey", "story", "building", "office",
    "shop", "ropani", "aana", "dhur", "duplex", "penthouse", "cottage",
    "farmhouse", "chowk", "marg", "nagar", "tole",
}

MIN_AREA_SQFT = 100


SQFT_PER_BEDROOM_FLOOR = 150


def validate_inputs(title, area, bedrooms, bathrooms):
    """Check inputs for obvious problems before predicting.

    Returns a list of (severity, message) tuples.
      "block" — stop prediction, the input is unusable.
      "warn"  — predict but tell the user to be cautious.

    Keeping this separate from the UI code makes it straightforward
    to add new rules later without touching the layout section.
    """
    issues = []
    stripped = title.strip()


    if len(stripped) < 3:
        issues.append((
            "block",
            "Please enter a short property description (e.g. '3BHK house in Kathmandu') "
            "so the model has something meaningful to read.",
        ))
        return issues  # no point running the rest if we have nothing to tokenise


    tokens = set(clean_and_tokenize(stripped))
    if not tokens & PROPERTY_KEYWORDS:
        issues.append((
            "warn",
            "The title doesn't contain common property words "
            "(house, land, ghar, jagga, apartment, bhk…). "
            "The estimate may be unreliable — try a description closer to "
            "what you'd see on a Nepali property listing site.",
        ))


    if area < MIN_AREA_SQFT:
        issues.append((
            "block",
            f"Land area must be at least {MIN_AREA_SQFT} sq ft. "
            "The model was not trained on plots this small.",
        ))


    if bedrooms > 0 and area >= MIN_AREA_SQFT:
        sqft_per_bed = area / bedrooms
        if sqft_per_bed < SQFT_PER_BEDROOM_FLOOR:
            issues.append((
                "warn",
                f"{bedrooms} bedrooms on {area:,.0f} sq ft works out to only "
                f"{sqft_per_bed:.0f} sq ft per bedroom, which is unusually cramped. "
                "Double-check the area — this combination rarely appears in real listings.",
            ))

    return issues




def rate_confidence(title, predicted, issues):
    """Grade the reliability of a prediction as High / Medium / Low.

    Three independent signals each subtract one point from a starting score of 3:
      1. Most title words are unknown to the W2V model (proxy for gibberish input).
      2. At least one validation warning was raised.
      3. The predicted price is outside the trained price range (extrapolation).

    Score 3 → High, 2 → Medium, 0–1 → Low.
    Returns (label_string, css_class).
    """
    tokens = clean_and_tokenize(title)
    known_ratio = sum(1 for t in tokens if t in w2v.wv) / max(len(tokens), 1)

    score = 3
    if known_ratio < 0.4:
        score -= 1
    if any(sev == "warn" for sev, _ in issues):
        score -= 1
    if predicted < price_range["reliable_min"] or predicted > price_range["reliable_max"]:
        score -= 1

    if score == 3:
        return "High confidence", "conf-high"
    elif score == 2:
        return "Medium confidence", "conf-medium"
    else:
        return "Low confidence", "conf-low"




st.title("🏠 Nepal Real Estate Price Predictor")
st.write("Estimate a property price from its title and key details. Trained on ~4,500 Nepali listings.")

with st.expander("How this works"):
    st.markdown(
        """
        **The model**
        A Random Forest trained on ~4,500 Nepali property listings (houses, land,
        apartments, villas). It explains roughly **63 % of price variation** (R² ≈ 0.63).

        **What it uses to predict**
        - *Land area* (sq ft), *bedrooms*, *bathrooms* — the core numeric signals.
        - *Title meaning* — the description is converted into a 100-number "meaning
          vector" using **Word2Vec**, a technique that learns which words tend to appear
          in similar contexts (so "villa" and "bungalow" end up close together in that
          100-dimensional space). The model then reads those numbers as extra features.

        **What it can't see**
        Exact location, road width, water supply, floor finish, view, and many other
        factors that matter a great deal in practice. This is why the estimate is a
        *rough guide*, not a bank valuation.

        **When to trust it less**
        - Your title uses words the model has never encountered.
        - The predicted price falls outside the training data's range.
        - The inputs are an unusual combination (e.g., many bedrooms on very little land).
        """
    )




st.header("Property details")

title_input = st.text_input(
    "Property title / description",
    placeholder="e.g. Modern 4BHK bungalow with garden and parking in Lalitpur",
    help="A short description similar to what you'd see on a Nepali property listing site.",
)

col1, col2, col3 = st.columns(3)
with col1:
    area_input = st.number_input(
        "Land area (sq ft)", min_value=100, max_value=100_000, value=1500, step=50,
    )
with col2:
    beds_input = st.number_input("Bedrooms", min_value=0, max_value=30, value=4)
with col3:
    baths_input = st.number_input("Bathrooms", min_value=0, max_value=30, value=3)




if st.button("Estimate price", type="primary"):

    issues = validate_inputs(title_input, area_input, beds_input, baths_input)

    for severity, message in issues:
        if severity == "block":
            st.error(message)
        elif severity == "warn":
            st.warning(message)

    if any(sev == "block" for sev, _ in issues):
        st.stop()

    # Core prediction — identical flow to training
    row = build_feature_row(title_input, area_input, beds_input, baths_input)
    predicted = model.predict(row)[0]

    crore = predicted / 10_000_000
    lakh  = predicted / 100_000

    conf_label, conf_class = rate_confidence(title_input, predicted, issues)

 
    st.header("Estimated price")

    price_col, conf_col = st.columns([2, 1])
    with price_col:
        st.metric("Predicted market price", f"Rs. {predicted:,.0f}")
        if crore >= 1:
            st.write(f"≈ **{crore:.2f} crore**")
        else:
            st.write(f"≈ **{lakh:.1f} lakh**")
    with conf_col:
        st.markdown("**Estimate quality**")
        st.markdown(
            f'<span class="{conf_class}">{conf_label}</span>',
            unsafe_allow_html=True,
        )

    lo = price_range["reliable_min"]
    hi = price_range["reliable_max"]
    if predicted < lo or predicted > hi:
        st.info(
            f"This price is outside the range the model was trained on "
            f"(Rs. {lo/1e5:,.0f} lakh – Rs. {hi/1e7:,.2f} crore). "
            "Treat the number with extra caution — the model is extrapolating here."
        )

    st.caption(
        "⚠️  Rough guide only — not an official appraisal. "
        "The model explains ~63 % of price variation. "
        "Actual value depends heavily on exact location, road access, "
        "floor quality, and many factors not captured in this form."
    )




st.divider()
st.caption(
    "Built on ~4,500 Nepali real estate listings. "
    "Prediction features: land area, bedrooms, bathrooms, "
    "and title meaning via 100-d Word2Vec embeddings."
)
