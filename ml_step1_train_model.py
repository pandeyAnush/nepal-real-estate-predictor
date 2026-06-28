#!/usr/bin/env python3
"""
ML Step 1: Train a Price-Prediction Model
Uses every NLP feature to predict property price.
Input:  nlp_features_final.csv
Output: trained model + evaluation report + saved files for the web app
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("\n" + "=" * 80)
print("ML STEP 1: TRAIN PRICE-PREDICTION MODEL")
print("=" * 80)



# Load the combined feature table from Step 5

print("\nLoading nlp_features_final.csv ...")
df = pd.read_csv('nlp_features_final.csv')
print(f"\u2713 Loaded {len(df):,} properties \u00d7 {df.shape[1]} columns")



# Separate the answer (target) from the inputs (features)
#   target   = price_numeric           (what we want to predict)
#   features = everything else numeric  (what we predict FROM)
# We drop text label columns because a model needs numbers, not words.

print("\nPreparing features and target...")

target_col = 'price_numeric'

# columns that are labels/text, not model inputs
drop_cols = [target_col, 'category', 'location']
drop_cols = [c for c in drop_cols if c in df.columns]

y = df[target_col]                       # the price we want to predict
X = df.drop(columns=drop_cols)           # all the inputs

# keep only numeric columns, just in case any text slipped through
X = X.select_dtypes(include=[np.number])

print(f"  Target:   {target_col}")
print(f"  Features: {X.shape[1]} numeric columns")



# Split into a training set and a test set
#   train (80%) = the model learns from these
#   test  (20%) = held back, used only to grade the model on unseen data

print("\nSplitting into train (80%) and test (20%)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Training properties: {len(X_train):,}")
print(f"  Testing properties:  {len(X_test):,}")



# Train the Random Forest
# A Random Forest builds many decision trees and averages them.
# It's a strong, reliable default for this kind of tabular data.

print("\nTraining Random Forest... (1-3 minutes)")

model = RandomForestRegressor(
    n_estimators=200,      # number of trees
    max_depth=None,        # let trees grow until they fit
    min_samples_leaf=2,    # avoid memorising single properties
    random_state=42,
    n_jobs=-1,             # use all CPU cores
)

model.fit(X_train, y_train)
print("\u2713 Training complete")



# Evaluate on the held-back test set

print("\n" + "=" * 80)
print("EVALUATION (on 930 properties the model never saw)")
print("=" * 80)

predictions = model.predict(X_test)

r2   = r2_score(y_test, predictions)
mae  = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"\n  R\u00b2 score: {r2:.3f}")
print(f"    \u2192 the model explains {r2 * 100:.1f}% of price variation")
print(f"\n  MAE (average error):  \u20b9{mae:,.0f}")
print(f"    \u2192 on average, predictions are off by this much")
print(f"\n  RMSE: \u20b9{rmse:,.0f}")
print(f"    \u2192 like MAE but punishes big misses more")

# plain-language verdict
print("\n  Verdict:", end=" ")
if r2 >= 0.75:
    print("Strong \u2014 good predictive power.")
elif r2 >= 0.5:
    print("Decent \u2014 usable, with room to improve.")
else:
    print("Weak \u2014 the features explain only part of the price.")



# Which features matter most?

print("\n" + "=" * 80)
print("TOP 15 MOST IMPORTANT FEATURES")
print("=" * 80 + "\n")

importance = pd.Series(model.feature_importances_, index=X.columns)
top = importance.sort_values(ascending=False).head(15)

for rank, (feat, score) in enumerate(top.items(), 1):
    bar = "#" * int(score * 100)
    print(f"  {rank:2d}. {feat:22s} {score:.4f}  {bar}")



# A few example predictions vs reality

print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS vs ACTUAL")
print("=" * 80 + "\n")

sample = X_test.head(5).copy()
sample_actual = y_test.head(5).values
sample_pred = model.predict(sample)

for i in range(5):
    actual = sample_actual[i]
    pred = sample_pred[i]
    diff_pct = abs(pred - actual) / actual * 100
    print(f"  Actual: \u20b9{actual:>14,.0f}   Predicted: \u20b9{pred:>14,.0f}   Off by {diff_pct:5.1f}%")



# Save the model and the feature list for the web app (Week 6)

print("\n" + "=" * 80)
print("SAVING")
print("=" * 80)

with open('price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("  \u2713 price_model.pkl          (the trained model)")

with open('price_model_features.txt', 'w') as f:
    for col in X.columns:
        f.write(col + '\n')
print(f"  \u2713 price_model_features.txt ({X.shape[1]} feature names, in order)")

with open('ml_step1_report.txt', 'w') as f:
    f.write("PRICE MODEL EVALUATION\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Training properties: {len(X_train):,}\n")
    f.write(f"Testing properties:  {len(X_test):,}\n")
    f.write(f"Features used:       {X.shape[1]}\n\n")
    f.write(f"R2 score: {r2:.3f}\n")
    f.write(f"MAE:      {mae:,.0f}\n")
    f.write(f"RMSE:     {rmse:,.0f}\n\n")
    f.write("Top 15 features:\n")
    for rank, (feat, score) in enumerate(top.items(), 1):
        f.write(f"  {rank:2d}. {feat}: {score:.4f}\n")
print("  \u2713 ml_step1_report.txt      (evaluation summary)")

print("\n" + "=" * 80)
print("\u2705 ML STEP 1 COMPLETE")
print("=" * 80)
print(f"\n  Model trained on {X.shape[1]} features, R\u00b2 = {r2:.3f}")
print("\n  Next: try XGBoost to compare, then build the Streamlit web app.\n")