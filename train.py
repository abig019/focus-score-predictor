import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

# WHY RandomForest and not Linear Regression?
# Linear Regression draws a straight line through data.
# RandomForest builds many decision trees and averages them.
# It's much more accurate, handles non-linear patterns,
# and gives us FEATURE IMPORTANCE — which tells us
# WHICH factors (sleep, screen time etc.) matter most.
# This is the "why" that makes your project interview-worthy.

from sklearn.model_selection import train_test_split
# WHY split? We must NOT test the model on data it was trained on.
# That's like giving a student the answers before the exam.
# We split: 80% train, 20% test.

from sklearn.preprocessing import LabelEncoder
# WHY LabelEncoder? ML models only understand NUMBERS.
# noise_level has text: 'Low', 'Medium', 'High'
# LabelEncoder converts them to 0, 1, 2 automatically.

from sklearn.metrics import mean_squared_error, r2_score
# WHY these metrics?
# mean_squared_error → how far off are our predictions on average?
# r2_score           → 0 = terrible, 1.0 = perfect (like a grade)

import pickle
# WHY pickle? After training, we SAVE the model to a file.
# So app.py can LOAD it and use it without retraining every time.
# Like saving your game progress.

import matplotlib.pyplot as plt
# For drawing charts to visualize results

# ── STEP 1: LOAD DATA ──────────────────────────────────────

df = pd.read_csv('data.csv')
print("✅ Data loaded")
print(df.head(3))
print()

# ── STEP 2: ENCODE NOISE_LEVEL (text → number) ────────────

le = LabelEncoder()
df['noise_level'] = le.fit_transform(df['noise_level'])
# fit_transform does two things:
#   fit      → learns that 'High'=0, 'Low'=1, 'Medium'=2
#   transform → actually replaces the text with numbers
print("Noise level encoding:", dict(zip(le.classes_, le.transform(le.classes_))))
print()

# Save the encoder — we'll need it in app.py too
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

    # ── STEP 3: SEPARATE FEATURES AND TARGET ──────────────────

X = df.drop('focus_score', axis=1)   # X = all columns EXCEPT focus_score
y = df['focus_score']                 # y = only focus_score (what we predict)
# Convention: X = features (inputs), y = target (output)
# axis=1 means "drop a column" (axis=0 would drop a row)

print("Features (X):", list(X.columns))
print("Target (y): focus_score")
print()

# ── STEP 4: SPLIT INTO TRAIN AND TEST SETS ────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% for testing, 80% for training
    random_state=42   # same split every time we run
)
print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")
print()

# ── STEP 5: TRAIN THE MODEL ───────────────────────────────

model = RandomForestRegressor(
    n_estimators=100,  # build 100 decision trees
    # WHY 100 trees? More trees = more accurate but slower.
    # 100 is a good balance for small datasets.
    random_state=42
)
model.fit(X_train, y_train)
# .fit() is THE training step.
# The model looks at 80% of the data and learns the patterns.
print("✅ Model trained!")
print()

# ── STEP 6: EVALUATE THE MODEL ────────────────────────────

y_pred = model.predict(X_test)
# Now we ask the model to predict on the 20% it has NEVER seen.

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
# RMSE = Root Mean Squared Error = average error in POINTS
# e.g. RMSE=5 means predictions are off by ~5 points on average
# R²   = how much of the pattern did the model capture?
#        0.85 means the model explains 85% of the variation

print("── Model Performance ──────────────")
print(f"RMSE : {rmse:.2f}  (avg error in focus points)")
print(f"R²   : {r2:.4f}  (1.0 = perfect)")
print()

# ── STEP 7: FEATURE IMPORTANCE ───────────────────────────

feature_names = X.columns.tolist()
importances   = model.feature_importances_
# feature_importances_ tells you: which columns did the model
# rely on most when making predictions?
# This is the INSIGHT layer of your project.

print("── Feature Importance ─────────────")
for name, importance in sorted(zip(feature_names, importances),
                                 key=lambda x: x[1], reverse=True):
    bar = '█' * int(importance * 50)
    print(f"{name:<15} {importance:.4f}  {bar}")
print()

# Draw the feature importance as a bar chart
plt.figure(figsize=(8, 5))
sorted_idx    = np.argsort(importances)
sorted_names  = [feature_names[i] for i in sorted_idx]
sorted_values = importances[sorted_idx]
plt.barh(sorted_names, sorted_values, color='steelblue')
plt.xlabel('Importance Score')
plt.title('Which factors affect focus the most?')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("✅ Chart saved as feature_importance.png")
print()

# ── STEP 8: SAVE THE MODEL ────────────────────────────────

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
# 'wb' = write binary. Models are saved as binary files.
# pickle converts the Python object into bytes and saves it.
print("✅ Model saved as model.pkl")

# Also save the feature names — app.py needs them in the right order
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✅ Feature names saved")