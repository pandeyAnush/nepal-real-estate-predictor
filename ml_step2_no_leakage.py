#!/usr/bin/env python3
"""
ML Step 2: Retrain WITHOUT data leakage
Removes features that secretly contain the price (price_per_sqft, bed_bath_ratio),
so the model's score reflects what it could REALLY predict for a new property.
Input:  nlp_features_final.csv
Output: honest model + evaluation
What files it produces
When you run it, the script saves two files into your project folder:
price_model_honest.pkl            ← the retrained model (the important one)
price_model_honest_features.txt   ← the list of features it uses, in order
The .pkl is the honest, leak-free model. This is the one your web app will eventually load. The .txt just records which columns the model expects, so the app feeds them in the right order later.
Note it does not overwrite your old files — your earlier price_model.pkl from step 1 is still there. This new one is saved under a different name (price_model_honest.pkl) so you can compare.
What the output on screen will look like
The script doesn't save a report to a file — it prints everything to the terminal. You'll see roughly this shape:
ML STEP 2: RETRAIN WITHOUT DATA LEAKAGE
Removing leaky features: ['price_per_sqft', 'bed_bath_ratio']
  Features: 217 numeric columns

HONEST EVALUATION (no leakage)
  R² score: 0.XXX
  MAE:      ₹X,XXX,XXX
  RMSE:     ₹XX,XXX,XXX

TOP 15 FEATURES (honest model)
  1. land_area_sqft ...
  ...

SAMPLE PREDICTIONS vs ACTUAL
  Actual: ₹...  Predicted: ₹...  Off by X%
The R² is the number that matters. It'll likely be lower than the 0.664 you got before — and that's the point, not a bug.
What this step actually does
It fixes the cheating problem from step 1. Two of your features — price_per_sqft and bed_bath_ratio — were built using the price itself. price_per_sqft is literally price divided by area, so it carries the answer inside it. Giving the model that feature and asking it to predict price is like giving someone the answer key during the exam. That's why step 1 looked suspiciously accurate.
This script deletes those two features, then retrains the exact same Random Forest on what's left. Everything else is identical — same data, same 80/20 split, same model settings. The only change is removing the leak. So the new R² is the model's real ability to predict the price of a property it knows nothing about which is the only number you can honestly put in your project or use in a working app.
Run it now, then paste me the R², MAE, and the top-15 features list. Those three things tell us whether to ship the web app or improve the model first.
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
print("ML STEP 2: RETRAIN WITHOUT DATA LEAKAGE")
print("=" * 80)

print("\nLoading nlp_features_final.csv ...")
df = pd.read_csv('nlp_features_final.csv')
print(f"\u2713 Loaded {len(df):,} properties \u00d7 {df.shape[1]} columns")



# THE KEY FIX: remove leaky features.
# These were built from price itself, so they "cheat".
# A new property wouldn't have them (you don't know its price yet).

leaky = ['price_per_sqft', 'bed_bath_ratio']
present_leaky = [c for c in leaky if c in df.columns]
print(f"\nRemoving leaky features (they secretly contain price): {present_leaky}")

target_col = 'price_numeric'
drop_cols = [target_col, 'category', 'location'] + present_leaky
drop_cols = [c for c in drop_cols if c in df.columns]

y = df[target_col]
X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])

print(f"  Target:   {target_col}")
print(f"  Features: {X.shape[1]} numeric columns (down from before)")



# Split 80/20

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n  Training: {len(X_train):,}   Testing: {len(X_test):,}")



# Train

print("\nTraining Random Forest... (1-3 minutes)")
model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print("\u2713 Training complete")



# Evaluate \u2014 this is now the HONEST score

print("\n" + "=" * 80)
print("HONEST EVALUATION (no leakage)")
print("=" * 80)

pred = model.predict(X_test)
r2   = r2_score(y_test, pred)
mae  = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print(f"\n  R\u00b2 score: {r2:.3f}   (explains {r2*100:.1f}% of price variation)")
print(f"  MAE:      \u20b9{mae:,.0f}   (average error)")
print(f"  RMSE:     \u20b9{rmse:,.0f}")

print("\n  Verdict:", end=" ")
if r2 >= 0.75:
    print("Strong.")
elif r2 >= 0.5:
    print("Decent and HONEST \u2014 this is a real, defensible result.")
elif r2 >= 0.3:
    print("Modest \u2014 expected for price-from-text; honest and explainable.")
else:
    print("Weak \u2014 titles + area alone only go so far. Still an honest finding.")



# Top features \u2014 should now make real-world sense

print("\n" + "=" * 80)
print("TOP 15 FEATURES (honest model)")
print("=" * 80 + "\n")

importance = pd.Series(model.feature_importances_, index=X.columns)
top = importance.sort_values(ascending=False).head(15)
for rank, (feat, score) in enumerate(top.items(), 1):
    bar = "#" * int(score * 100)
    print(f"  {rank:2d}. {feat:22s} {score:.4f}  {bar}")



# Sample predictions

print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS vs ACTUAL")
print("=" * 80 + "\n")
s_actual = y_test.head(8).values
s_pred = model.predict(X_test.head(8))
for i in range(8):
    a, p = s_actual[i], s_pred[i]
    off = abs(p - a) / a * 100
    print(f"  Actual: \u20b9{a:>14,.0f}   Predicted: \u20b9{p:>14,.0f}   Off by {off:5.1f}%")



# Save the honest model

print("\n" + "=" * 80)
print("SAVING")
print("=" * 80)

with open('price_model_honest.pkl', 'wb') as f:
    pickle.dump(model, f)
print("  \u2713 price_model_honest.pkl")

with open('price_model_honest_features.txt', 'w') as f:
    for col in X.columns:
        f.write(col + '\n')
print(f"  \u2713 price_model_honest_features.txt ({X.shape[1]} features)")

print("\n" + "=" * 80)
print("\u2705 ML STEP 2 COMPLETE")
print("=" * 80)
print(f"\n  Honest R\u00b2 = {r2:.3f} on {X.shape[1]} real features")
print("  This is the model to use in the web app.\n")