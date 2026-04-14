import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from src.expense_model import compute_expected_expense
from src.simulation import monte_carlo

st.set_page_config(
    page_title="Expense Forecasting Dashboard",
    layout="wide",
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
body { background-color: #0f1e36; color: #cdd9e5; }
.page-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem; }
.section-label { font-size: 0.7rem; color: #6b8aad; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/forecast_model.pkl")

# ---------------- HEADER ----------------
st.markdown('<div class="page-title">Household Expense Forecasting System</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR INPUT ----------------
with st.sidebar:
    st.markdown('<div class="section-label">User Profile</div>', unsafe_allow_html=True)
    income = st.number_input("Income", value=5000)
    age = st.slider("Age", 18, 60, 25)
    dependents = st.slider("Dependents", 0, 5, 1)

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["Forecast", "Category Analysis", "Simulation"])

# ============================================================
# TAB 1 — FORECAST
# ============================================================
with tab1:

    st.markdown('<div class="section-label">Enter Past Monthly Expenses</div>', unsafe_allow_html=True)

    months = st.number_input("Number of months", 3, 12, 6)

    past_data = []
    for i in range(months):
        val = st.number_input(f"Month {i+1} Expense", value=4000, key=i)
        past_data.append(val)

    df = pd.DataFrame({"Month": range(1, months+1), "Expense": past_data})

    # Plot past trend
    fig, ax = plt.subplots()
    ax.plot(df["Month"], df["Expense"], marker='o')
    ax.set_title("Past Expense Trend")
    st.pyplot(fig)

    if st.button("Predict Future Expense"):

        X = [[income, age, dependents]]
        pred = model.predict(X)[0]

        st.subheader("Forecast Result")
        st.write("Predicted Monthly Expense:", round(pred, 2))

        # Math model
        F = np.mean(past_data) * 0.4
        V = np.mean(past_data) * 0.6
        mu = 0.05
        lambda_ = 1.5
        E_C = 600

        expected = compute_expected_expense(F, V, mu, lambda_, E_C)
        st.write("Expected Value (Mathematical Model):", round(expected, 2))

# TAB 2 — CATEGORY ANALYSIS
with tab2:

    st.markdown('<div class="section-label">Expense Breakdown</div>', unsafe_allow_html=True)

    rent = st.number_input("Rent", 1000)
    groceries = st.number_input("Groceries", 800)
    transport = st.number_input("Transport", 500)
    entertainment = st.number_input("Entertainment", 300)
    utilities = st.number_input("Utilities", 400)

    categories = ["Rent", "Groceries", "Transport", "Entertainment", "Utilities"]
    values = [rent, groceries, transport, entertainment, utilities]

    fig, ax = plt.subplots()
    ax.pie(values, labels=categories, autopct='%1.1f%%')
    ax.set_title("Expense Distribution")
    st.pyplot(fig)

    # Bar chart
    fig2, ax2 = plt.subplots()
    ax2.bar(categories, values)
    ax2.set_title("Category-wise Expense")
    st.pyplot(fig2)

# TAB 3 — SIMULATION
with tab3:

    st.markdown('<div class="section-label">Monte Carlo Simulation</div>', unsafe_allow_html=True)

    F = st.number_input("Fixed Expense", 5000)
    V = st.number_input("Variable Expense", 4000)
    mu = st.slider("Inflation Rate", 0.0, 0.2, 0.05)
    lambda_ = st.slider("Unexpected Events", 0.0, 5.0, 1.5)
    E_C = st.number_input("Cost per Event", 600)

    if st.button("Run Simulation"):

        results = monte_carlo(F, V, mu, lambda_, E_C)

        fig, ax = plt.subplots()
        ax.hist(results, bins=30)
        ax.set_title("Expense Distribution (Simulation)")
        st.pyplot(fig)

        st.write("Mean:", round(np.mean(results), 2))
        st.write("Max:", round(np.max(results), 2))
        st.write("Min:", round(np.min(results), 2))