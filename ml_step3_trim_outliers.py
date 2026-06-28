#!/usr/bin/env python3
"""
ML Step 3: Trim Extreme Outliers, then Retrain
The model is weak on very cheap and very expensive properties because there are
too few of them to learn from. We scope the model to its reliable price range,
retrain, and compare against the previous honest model.
Input:  nlp_features_final.csv
Output: scoped model + before/after comparison
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
print("ML STEP 3: TRIM OUTLIERS + RETRAIN")
print("=" * 80)

print("\nLoading nlp_features_final.csv ...")
df = pd.read_csv('nlp_features_final.csv')
print(f"\u2713 Loaded {len(df):,} properties")



# Look at the price distribution before trimming

price = df['price_numeric']
print("\nPrice distribution BEFORE trimming:")
print(f"  Min:    \u20b9{price.min():,.0f}")
print(f"  1%:     \u20b9{price.quantile(0.01):,.0f}")
print(f"  Median: \u20b9{price.median():,.0f}")
print(f"  99%:    \u20b9{price.quantile(0.99):,.0f}")
print(f"  Max:    \u20b9{price.max():,.0f}")



# Define a sensible reliable range and trim outside it.
# We cut the bottom 1% and top 1% of prices \u2014 the extremes the model
# can't learn from. This keeps 98% of the data.

low_cut  = price.quantile(0.01)
high_cut = price.quantile(0.99)

print(f"\nKeeping properties priced between \u20b9{low_cut:,.0f} and \u20b9{high_cut:,.0f}")

before = len(df)
df = df[(df['price_numeric'] >= low_cut) & (df['price_numeric'] <= high_cut)]
after = len(df)

print(f"  Removed {before - after} extreme properties ({(before-after)/before*100:.1f}%)")
print(f"  Remaining: {after:,} properties")



# Same setup as before: drop target + labels + leaky features

target_col = 'price_numeric'
leaky = ['price_per_sqft', 'bed_bath_ratio']
drop_cols = [target_col, 'category', 'location'] + leaky
drop_cols = [c for c in drop_cols if c in df.columns]

y = df[target_col]
X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])

print(f"\nFeatures: {X.shape[1]} numeric columns")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Training: {len(X_train):,}   Testing: {len(X_test):,}")



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



# Evaluate

print("\n" + "=" * 80)
print("EVALUATION (scoped to reliable price range)")
print("=" * 80)

pred = model.predict(X_test)
r2   = r2_score(y_test, pred)
mae  = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print(f"\n  R\u00b2 score: {r2:.3f}   (explains {r2*100:.1f}% of price variation)")
print(f"  MAE:      \u20b9{mae:,.0f}")
print(f"  RMSE:     \u20b9{rmse:,.0f}")



# Before/after comparison vs the previous honest model

print("\n" + "=" * 80)
print("BEFORE vs AFTER")
print("=" * 80)
print(f"""
  Previous honest model (all prices):
    R\u00b2 = 0.612   MAE = \u20b910,367,275   RMSE = \u20b960,431,208

  This model (trimmed to reliable range):
    R\u00b2 = {r2:.3f}   MAE = \u20b9{mae:,.0f}   RMSE = \u20b9{rmse:,.0f}
""")
if rmse < 60431208:
    print(f"  \u2192 RMSE dropped a lot \u2014 far fewer catastrophic misses.")
if r2 > 0.612:
    print(f"  \u2192 R\u00b2 improved \u2014 model is more accurate in its scope.")



# Top features

print("\n" + "=" * 80)
print("TOP 15 FEATURES")
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



# Save

print("\n" + "=" * 80)
print("SAVING")
print("=" * 80)

with open('price_model_final.pkl', 'wb') as f:
    pickle.dump(model, f)
print("  \u2713 price_model_final.pkl   (use this in the web app)")

with open('price_model_final_features.txt', 'w') as f:
    for col in X.columns:
        f.write(col + '\n')
print(f"  \u2713 price_model_final_features.txt ({X.shape[1]} features)")

# record the reliable range so the web app can warn outside it
with open('price_model_range.txt', 'w') as f:
    f.write(f"reliable_min={low_cut:.0f}\n")
    f.write(f"reliable_max={high_cut:.0f}\n")
print(f"  \u2713 price_model_range.txt   (reliable price range for the app)")

print("\n" + "=" * 80)
print("\u2705 ML STEP 3 COMPLETE")
print("=" * 80)
print(f"\n  Final model: R\u00b2 = {r2:.3f}, reliable for \u20b9{low_cut:,.0f}\u2013\u20b9{high_cut:,.0f}")
print("  Next: build the Streamlit web app.\n")