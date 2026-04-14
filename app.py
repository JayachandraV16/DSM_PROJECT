import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from src.expense_model import compute_expected_expense
from src.simulation import monte_carlo, advanced_simulation

st.set_page_config(page_title="Expense Forecasting Dashboard", layout="wide")

import os
import joblib

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Train models if missing
if not os.path.exists("models/forecast_model.pkl") or not os.path.exists("models/risk_model.pkl"):

    st.warning("Models not found. Training models...")

    import main  # runs training script

    # Double check after training
    if not os.path.exists("models/forecast_model.pkl") or not os.path.exists("models/risk_model.pkl"):
        st.error("Model creation failed. Check dataset path.")
        st.stop()

    st.success("Models created successfully!")

# Load models safely
forecast_model = joblib.load("models/forecast_model.pkl")
risk_model = joblib.load("models/risk_model.pkl")
# ---------------- FUNCTIONS ----------------

def monte_carlo_forecast_with_ci(F, V, mu, lambda_, E_C, months=12, sims=200):
    all_simulations = []

    for _ in range(sims):
        current = V
        sim = []

        for _ in range(months):
            I_t = np.random.normal(mu, 0.01)
            N_t = np.random.poisson(lambda_)
            cost = np.sum(np.random.randint(300, 800, N_t))
            current = current * (1 + I_t) + cost
            sim.append(current)

        all_simulations.append(sim)

    all_simulations = np.array(all_simulations)

    mean = np.mean(all_simulations, axis=0)
    lower = np.percentile(all_simulations, 5, axis=0)
    upper = np.percentile(all_simulations, 95, axis=0)

    return mean, lower, upper


def forecast_future(months, base, mu):
    result = []
    current = base
    for _ in range(months):
        current *= (1 + mu)
        result.append(current)
    return result


def forecast_with_uncertainty(months, base, mu, lambda_, E_C):
    result = []
    current = base
    for _ in range(months):
        I_t = np.random.normal(mu, 0.01)
        N_t = np.random.poisson(lambda_)
        cost = np.sum(np.random.randint(300, 800, N_t))
        current = current * (1 + I_t) + cost
        result.append(current)
    return result


# ---------------- HEADER ----------------
st.title("Household Expense Forecasting System")
st.caption("A data-driven tool to predict, analyze, and simulate household expenses.")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.info("Enter your personal details for accurate prediction.")
    income = st.number_input("Income", value=5000)
    age = st.slider("Age", 18, 60, 25)
    dependents = st.slider("Dependents", 0, 5, 1)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Forecast", "Category Analysis", "Simulation", "Advanced Analytics"]
)
# TAB 1 — FORECAST
with tab1:

    st.info(
        "This section predicts your future monthly expenses using past data, machine learning, "
        "and mathematical modeling."
    )

    st.subheader("Enter Past Monthly Expenses")
    st.caption("Provide previous monthly expenses to analyze trends.")

    months = st.number_input("Number of months", value=6)

    past_data = []
    for i in range(months):
        val = st.number_input(f"Month {i+1}", value=4000, key=f"month_{i}")
        past_data.append(val)

    df = pd.DataFrame({"Month": range(1, months+1), "Expense": past_data})

    fig, ax = plt.subplots()
    ax.plot(df["Month"], df["Expense"], marker='o')
    ax.set_title("Past Expense Trend")
    st.pyplot(fig)

    if st.button("Predict Expense"):
        X = [[income, age, dependents]]

        pred = forecast_model.predict(X)[0]
        risk = risk_model.predict(X)[0] 

        st.session_state.pred = pred
        st.session_state.risk = risk

        st.write("Predicted Expense:", round(pred, 2))

        # Risk display
        if risk == "Low":
            st.success("Risk Level: Low")
        elif risk == "Medium":
            st.warning("Risk Level: Medium")
        else:
            st.error("Risk Level: High")

        st.session_state.pred = pred
        st.write("Predicted Expense:", round(pred, 2))

        F = np.mean(past_data) * 0.4
        V = np.mean(past_data) * 0.6
        mu = 0.05
        lambda_ = 1.5
        E_C = 600

        expected = compute_expected_expense(F, V, mu, lambda_, E_C)
        st.write("Expected Value (Mathematical Model):", round(expected, 2))

    st.caption("Prediction uses machine learning + mathematical modeling.")

    # -------- Budget --------
    st.subheader("Budget Analysis")
    st.info("Compare predicted expense with your budget.")

    budget = st.number_input("Your Budget", value=10000)

    if "pred" in st.session_state:
        if st.session_state.pred > budget:
            st.error("Your expenses exceed your budget.")
        else:
            st.success("Your expenses are within budget.")

        saving = income - st.session_state.pred
        st.write("Expected Savings:", saving)

    # -------- Future Forecast --------
    st.subheader("Future Forecast")
    st.info("Estimate future expenses with and without uncertainty.")

    future_months = st.number_input("Months to Forecast", value=6)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Forecast Trend"):
            base = np.mean(past_data)
            forecast = forecast_future(future_months, base, 0.05)

            fig, ax = plt.subplots()
            ax.plot(range(1, future_months+1), forecast, marker='o')
            ax.set_title("Trend Forecast")
            st.pyplot(fig)

    with col2:
        if st.button("Forecast with Uncertainty"):
            base = np.mean(past_data)
            forecast = forecast_with_uncertainty(future_months, base, 0.05, 1.5, 600)

            fig, ax = plt.subplots()
            ax.plot(range(1, future_months+1), forecast, marker='o')
            ax.set_title("Uncertain Forecast")
            st.pyplot(fig)


# TAB 2 — CATEGORY ANALYSIS
with tab2:

    st.info("Analyze and customize your expense categories.")

    if "categories" not in st.session_state:
        st.session_state.categories = {"Rent": 1000, "Food": 800}

    new_categories = {}

    for cat, val in st.session_state.categories.items():
        col1, col2 = st.columns([3,1])

        with col1:
            new_name = st.text_input("Category", cat, key=f"name_{cat}")
            new_val = st.number_input("Value", value=val, key=f"val_{cat}")

        with col2:
            if st.button("Delete", key=f"del_{cat}"):
                del st.session_state.categories[cat]
                st.rerun()

        new_categories[new_name] = new_val

    st.session_state.categories = new_categories

    new_cat_name = st.text_input("New Category")
    new_cat_val = st.number_input("Value", value=0)

    if st.button("Add Category"):
        if new_cat_name:
            st.session_state.categories[new_cat_name] = new_cat_val

    labels = list(st.session_state.categories.keys())
    values = list(st.session_state.categories.values())

    if labels:
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%')
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.bar(labels, values)
        st.pyplot(fig2)

        st.write("Total Expense:", sum(values))


# TAB 3 — SIMULATION
with tab3:

    st.info(
        "Simulate real-world financial conditions including inflation, unexpected expenses, and income changes."
    )

    F = st.number_input("Fixed Expense", value=5000)
    V = st.number_input("Variable Expense", value=4000)
    mu = st.slider("Inflation", 0.0, 0.2, 0.05)
    income = st.number_input("Monthly Income", value=8000)
    months = st.number_input("Simulation Months", value=12)

    if st.button("Run Simulation"):

        expenses, savings = advanced_simulation(F, V, mu, months, income)

        fig, ax = plt.subplots()
        ax.plot(range(1, months+1), expenses)
        ax.set_title("Expense Simulation")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.plot(range(1, months+1), savings)
        ax2.set_title("Savings Simulation")
        st.pyplot(fig2)

    st.subheader("Confidence Interval Forecast")
    st.info("Shows range of possible expenses under uncertainty.")

    lambda_ = st.slider("Events", 0.0, 5.0, 1.5)
    E_C = st.number_input("Event Cost", value=600)
    months_ci = st.number_input("Months for Forecast", value=12)

    if st.button("Run Confidence Forecast"):

        mean, lower, upper = monte_carlo_forecast_with_ci(F, V, mu, lambda_, E_C, months_ci)

        fig, ax = plt.subplots()
        x = range(1, months_ci+1)

        ax.plot(x, mean)
        ax.fill_between(x, lower, upper, alpha=0.3)
        ax.set_title("Confidence Interval Forecast")

        st.pyplot(fig)

with tab4:

    st.header("Advanced Analytics Dashboard")

    st.info("Explore statistical modeling and probability-based forecasting.")

    # -------- Inputs --------
    F = st.number_input("Fixed Expense (F)", value=5000)
    V = st.number_input("Variable Expense (V)", value=4000)

    mu = st.slider("Inflation Rate (μ)", 0.0, 0.2, 0.05)
    sigma = st.slider("Inflation Std Dev (σ)", 0.0, 0.1, 0.02)

    lambda_ = st.slider("Unexpected Events (λ)", 0.0, 5.0, 1.5)
    E_C = st.number_input("Avg Cost per Event", value=600)

    p = st.slider("Probability of Price Increase (p)", 0.0, 1.0, 0.5)

    budget = st.number_input("Budget Threshold", value=10000)

    # -------- Expected Value --------
    expected = F + V*(1 + mu) + lambda_*E_C
    st.subheader("Expected Expense")
    st.write(expected)

    # -------- Variance --------
    variance = (V**2)*(sigma**2) + lambda_*(E_C**2)
    std_dev = np.sqrt(variance)

    st.subheader("Risk Metrics")
    st.write("Variance:", variance)
    st.write("Standard Deviation:", std_dev)

    # -------- Confidence Interval --------
    lower = expected - 1.96*std_dev
    upper = expected + 1.96*std_dev

    st.write("95% Confidence Interval:", lower, "-", upper)

    # -------- Probability Exceed Budget --------
    from scipy.stats import norm

    prob = 1 - norm.cdf(budget, expected, std_dev)
    st.write("P(Expense > Budget):", prob)

    # -------- Risk Indicator --------
    if prob < 0.2:
        st.success("Low Risk")
    elif prob < 0.5:
        st.warning("Moderate Risk")
    else:
        st.error("High Risk")

    # -------- Normal Distribution --------
    from scipy.stats import norm
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Normal Distribution (Inflation)")
    st.pyplot(fig)

    # -------- Poisson --------
    from scipy.stats import poisson
    k = np.arange(0, 10)
    y = poisson.pmf(k, lambda_)

    fig2, ax2 = plt.subplots()
    ax2.bar(k, y)
    ax2.set_title("Poisson Distribution (Events)")
    st.pyplot(fig2)

    # -------- Binomial --------
    from scipy.stats import binom
    k = np.arange(0, 13)
    y = binom.pmf(k, 12, p)

    fig3, ax3 = plt.subplots()
    ax3.bar(k, y)
    ax3.set_title("Binomial Distribution (Price Increase Months)")
    st.pyplot(fig3)

    # -------- 12-Month Forecast Table --------
    forecast = []
    current = V

    for i in range(12):
        current = current * (1 + mu)
        forecast.append(F + current + lambda_*E_C)

    df = pd.DataFrame({
        "Month": range(1,13),
        "Expense": forecast
    })

    st.subheader("12-Month Forecast Table")
    st.dataframe(df)