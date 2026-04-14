import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
import os
import numpy as np

# ---------------- CREATE MODELS FOLDER ----------------
os.makedirs("models", exist_ok=True)

# ---------------- LOAD DATASET ----------------
df = pd.read_csv("data/data.csv")

# ---------------- EXPENSE COLUMNS ----------------
expense_cols = [
    'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
    'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare',
    'Education', 'Miscellaneous'
]

# ---------------- TARGET: Total Expense ----------------
df['total_expense'] = df[expense_cols].sum(axis=1)

# ---------------- TARGET: Risk Level ----------------
# Derive risk from expense-to-income ratio
df['expense_ratio'] = df['total_expense'] / df['Income']

def assign_risk(ratio):
    if ratio < 0.5:
        return "Low"
    elif ratio < 0.8:
        return "Medium"
    else:
        return "High"

df['risk_level'] = df['expense_ratio'].apply(assign_risk)

# ---------------- FEATURES ----------------
X = df[['Income', 'Age', 'Dependents']]

# ---------------- TRAIN FORECAST MODEL ----------------
forecast_model = RandomForestRegressor(n_estimators=200, random_state=42)
forecast_model.fit(X, df['total_expense'])
joblib.dump(forecast_model, "models/forecast_model.pkl")
print(" forecast_model.pkl saved")

# ---------------- TRAIN RISK MODEL ----------------
risk_model = RandomForestClassifier(n_estimators=200, random_state=42)
risk_model.fit(X, df['risk_level'])
joblib.dump(risk_model, "models/risk_model.pkl")
print("risk_model.pkl saved")

print("\nAll models trained and saved successfully!")