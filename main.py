import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# ---------------- CREATE MODELS FOLDER ----------------
os.makedirs("models", exist_ok=True)

# ---------------- LOAD DATASET ----------------
# Make sure your dataset is inside: data/data.csv
df = pd.read_csv("data/data.csv")

# ---------------- CREATE TARGET VARIABLE ----------------
expense_cols = [
    'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
    'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare',
    'Education', 'Miscellaneous'
]

# Total monthly expense
df['total_expense'] = df[expense_cols].sum(axis=1)

# ---------------- FEATURES & TARGET ----------------
X = df[['Income', 'Age', 'Dependents']]
y = df['total_expense']

# ---------------- TRAIN MODEL ----------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "models/forecast_model.pkl")

# ---------------- SUCCESS MESSAGE ----------------
print("Model trained and saved successfully!")
print("Saved at: models/forecast_model.pkl")