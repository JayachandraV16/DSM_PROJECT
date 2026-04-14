import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import norm, poisson, binom

st.set_page_config(page_title="Expense Forecasting Dashboard", layout="wide")


# MODEL TRAINING — cached in memory, no pkl files needed

@st.cache_resource(show_spinner="Training models on your dataset...")
def load_models():
    """
    Trains both models from data/data.csv and caches them in memory.
    This runs once per session — no pkl files needed.
    """
    expense_cols = [
        'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
        'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare',
        'Education', 'Miscellaneous'
    ]

    df = pd.read_csv("data/data.csv")
    df['total_expense'] = df[expense_cols].sum(axis=1)
    df['expense_ratio'] = df['total_expense'] / df['Income']

    def assign_risk(ratio):
        if ratio < 0.5:
            return "Low"
        elif ratio < 0.8:
            return "Medium"
        else:
            return "High"

    df['risk_level'] = df['expense_ratio'].apply(assign_risk)

    X = df[['Income', 'Age', 'Dependents']]

    forecast_model = RandomForestRegressor(n_estimators=200, random_state=42)
    forecast_model.fit(X, df['total_expense'])

    risk_model = RandomForestClassifier(n_estimators=200, random_state=42)
    risk_model.fit(X, df['risk_level'])

    return forecast_model, risk_model


try:
    forecast_model, risk_model = load_models()
except FileNotFoundError:
    st.error("Dataset not found. Please ensure `data/data.csv` exists in your project folder.")
    st.stop()
except KeyError as e:
    st.error(f"Missing column in dataset: {e}. Check your CSV has the required expense columns.")
    st.stop()


# HELPER FUNCTIONS

def compute_expected_expense(F, V, mu, lambda_, E_C):
    return F + V * (1 + mu) + lambda_ * E_C


def advanced_simulation(F, V, mu, months, income):
    expenses, savings = [], []
    current_V = V
    for _ in range(months):
        current_V *= (1 + mu)
        total = F + current_V
        expenses.append(total)
        savings.append(income - total)
    return expenses, savings


def monte_carlo_forecast_with_ci(F, V, mu, lambda_, E_C, months=12, sims=200):
    all_simulations = []
    for _ in range(sims):
        current = V
        sim = []
        for _ in range(months):
            I_t = np.random.normal(mu, 0.01)
            N_t = np.random.poisson(lambda_)
            cost = np.sum(np.random.randint(300, 800, max(N_t, 1))) if N_t > 0 else 0
            current = current * (1 + I_t) + cost
            sim.append(current)
        all_simulations.append(sim)

    all_simulations = np.array(all_simulations)
    mean  = np.mean(all_simulations, axis=0)
    lower = np.percentile(all_simulations, 5, axis=0)
    upper = np.percentile(all_simulations, 95, axis=0)
    return mean, lower, upper


def forecast_future(months, base, mu):
    result, current = [], base
    for _ in range(months):
        current *= (1 + mu)
        result.append(current)
    return result


def forecast_with_uncertainty(months, base, mu, lambda_, E_C):
    result, current = [], base
    for _ in range(months):
        I_t = np.random.normal(mu, 0.01)
        N_t = np.random.poisson(lambda_)
        cost = np.sum(np.random.randint(300, 800, max(N_t, 1))) if N_t > 0 else 0
        current = current * (1 + I_t) + cost
        result.append(current)
    return result


# UI

st.title("Household Expense Forecasting System")
st.caption("A data-driven tool to predict, analyze, and simulate household expenses.")

with st.sidebar:
    st.info("Enter your personal details for accurate prediction.")
    income  = st.number_input("Income", value=5000)
    age     = st.slider("Age", 18, 60, 25)
    dependents = st.slider("Dependents", 0, 5, 1)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Forecast", "Category Analysis", "Simulation", "Advanced Analytics"]
)


# TAB 1 — FORECAST
with tab1:

    st.info(
        "This section predicts your future monthly expenses using past data, "
        "machine learning, and mathematical modeling."
    )

    st.subheader("Enter Past Monthly Expenses")
    st.caption("Provide previous monthly expenses to analyze trends.")

    months = st.number_input("Number of months", value=6, min_value=1, step=1)

    past_data = []
    for i in range(int(months)):
        val = st.number_input(f"Month {i+1}", value=4000, key=f"month_{i}")
        past_data.append(val)

    fig, ax = plt.subplots()
    ax.plot(range(1, int(months)+1), past_data, marker='o')
    ax.set_title("Past Expense Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Expense")
    st.pyplot(fig)
    plt.close()

    if st.button("Predict Expense"):
        X_input = [[income, age, dependents]]

        pred = forecast_model.predict(X_input)[0]
        risk = risk_model.predict(X_input)[0]

        st.session_state.pred = pred
        st.session_state.risk = risk

        st.write("**Predicted Expense:**", round(pred, 2))

        if risk == "Low":
            st.success("Risk Level: Low ")
        elif risk == "Medium":
            st.warning("Risk Level: Medium ")
        else:
            st.error("Risk Level: High ")

        F = np.mean(past_data) * 0.4
        V = np.mean(past_data) * 0.6
        mu, lambda_, E_C = 0.05, 1.5, 600

        expected = compute_expected_expense(F, V, mu, lambda_, E_C)
        st.write("**Expected Value (Mathematical Model):**", round(expected, 2))

    st.caption("Prediction uses machine learning + mathematical modeling.")

    # -------- Budget --------
    st.subheader("Budget Analysis")
    st.info("Compare predicted expense with your budget.")

    budget = st.number_input("Your Budget", value=10000)

    if "pred" in st.session_state:
        if st.session_state.pred > budget:
            st.error(f"Your expenses (₹{round(st.session_state.pred, 2)}) exceed your budget.")
        else:
            st.success(f"Your expenses (₹{round(st.session_state.pred, 2)}) are within budget. ")

        saving = income - st.session_state.pred
        st.write("**Expected Savings:**", round(saving, 2))

    # -------- Future Forecast --------
    st.subheader("Future Forecast")
    st.info("Estimate future expenses with and without uncertainty.")

    future_months = st.number_input("Months to Forecast", value=6, min_value=1, step=1)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Forecast Trend"):
            base     = np.mean(past_data)
            forecast = forecast_future(int(future_months), base, 0.05)

            fig, ax = plt.subplots()
            ax.plot(range(1, int(future_months)+1), forecast, marker='o', color='steelblue')
            ax.set_title("Trend Forecast")
            ax.set_xlabel("Month")
            ax.set_ylabel("Expense")
            st.pyplot(fig)
            plt.close()

    with col2:
        if st.button("Forecast with Uncertainty"):
            base     = np.mean(past_data)
            forecast = forecast_with_uncertainty(int(future_months), base, 0.05, 1.5, 600)

            fig, ax = plt.subplots()
            ax.plot(range(1, int(future_months)+1), forecast, marker='o', color='orange')
            ax.set_title("Uncertain Forecast")
            ax.set_xlabel("Month")
            ax.set_ylabel("Expense")
            st.pyplot(fig)
            plt.close()


# TAB 2 — CATEGORY ANALYSIS
with tab2:

    st.info("Analyze and customize your expense categories.")

    if "categories" not in st.session_state:
        st.session_state.categories = {"Rent": 1000, "Food": 800}

    cats_to_delete = []
    new_categories  = {}

    for cat, val in list(st.session_state.categories.items()):
        col1, col2 = st.columns([4, 1])
        with col1:
            new_name = st.text_input("Category", cat,  key=f"name_{cat}")
            new_val  = st.number_input("Value",   value=val, key=f"val_{cat}")
        with col2:
            st.write("")
            st.write("")
            if st.button("🗑 Delete", key=f"del_{cat}"):
                cats_to_delete.append(cat)
                continue
        new_categories[new_name] = new_val

    for cat in cats_to_delete:
        if cat in st.session_state.categories:
            del st.session_state.categories[cat]
    if cats_to_delete:
        st.rerun()

    st.session_state.categories = new_categories

    st.divider()
    new_cat_name = st.text_input("New Category Name")
    new_cat_val  = st.number_input("New Category Value", value=0)

    if st.button("Add Category"):
        if new_cat_name.strip():
            st.session_state.categories[new_cat_name.strip()] = new_cat_val
            st.rerun()
        else:
            st.warning("Please enter a category name.")

    labels = list(st.session_state.categories.keys())
    values = list(st.session_state.categories.values())

    if labels and sum(values) > 0:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct='%1.1f%%')
            ax.set_title("Expense Breakdown")
            st.pyplot(fig)
            plt.close()
        with col2:
            fig2, ax2 = plt.subplots()
            ax2.bar(labels, values, color='steelblue')
            ax2.set_title("Expense by Category")
            ax2.set_ylabel("Amount")
            plt.xticks(rotation=30, ha='right')
            st.pyplot(fig2)
            plt.close()

        st.metric("Total Expense", f"₹{sum(values):,.2f}")


# TAB 3 — SIMULATION
with tab3:

    st.info(
        "Simulate real-world financial conditions including inflation, "
        "unexpected expenses, and income changes."
    )

    F      = st.number_input("Fixed Expense", value=5000, key="sim_F")
    V      = st.number_input("Variable Expense", value=4000, key="sim_V")
    mu     = st.slider("Inflation Rate", 0.0, 0.2, 0.05, key="sim_mu")
    income_sim = st.number_input("Monthly Income", value=8000, key="sim_income")
    months_sim = st.number_input("Simulation Months", value=12, min_value=1, step=1, key="sim_months")

    if st.button("Run Simulation"):
        expenses, savings = advanced_simulation(F, V, mu, int(months_sim), income_sim)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.plot(range(1, int(months_sim)+1), expenses, color='tomato')
            ax.set_title("Expense Simulation")
            ax.set_xlabel("Month")
            ax.set_ylabel("Expense")
            st.pyplot(fig)
            plt.close()
        with col2:
            fig2, ax2 = plt.subplots()
            ax2.plot(range(1, int(months_sim)+1), savings, color='green')
            ax2.set_title("Savings Simulation")
            ax2.set_xlabel("Month")
            ax2.set_ylabel("Savings")
            st.pyplot(fig2)
            plt.close()

    st.subheader("Confidence Interval Forecast")
    st.info("Shows range of possible expenses under uncertainty.")

    lambda_ci = st.slider("Unexpected Events (λ)", 0.0, 5.0, 1.5, key="ci_lambda")
    E_C_ci    = st.number_input("Avg Event Cost", value=600, key="ci_ec")
    months_ci = st.number_input("Months for Forecast", value=12, min_value=1, step=1, key="ci_months")

    if st.button("Run Confidence Forecast"):
        mean, lower, upper = monte_carlo_forecast_with_ci(
            F, V, mu, lambda_ci, E_C_ci, int(months_ci)
        )
        x = range(1, int(months_ci)+1)

        fig, ax = plt.subplots()
        ax.plot(x, mean, label="Mean", color='steelblue')
        ax.fill_between(x, lower, upper, alpha=0.3, label="90% CI", color='steelblue')
        ax.set_title("Confidence Interval Forecast")
        ax.set_xlabel("Month")
        ax.set_ylabel("Expense")
        ax.legend()
        st.pyplot(fig)
        plt.close()


# TAB 4 — ADVANCED ANALYTICS
with tab4:

    st.header("Advanced Analytics Dashboard")
    st.info("Explore statistical modeling and probability-based forecasting.")

    F_a     = st.number_input("Fixed Expense (F)", value=5000, key="adv_F")
    V_a     = st.number_input("Variable Expense (V)", value=4000, key="adv_V")
    mu_a    = st.slider("Inflation Rate (μ)", 0.0, 0.2, 0.05, key="adv_mu")
    sigma_a = st.slider("Inflation Std Dev (σ)", 0.01, 0.1, 0.02, key="adv_sigma")  # min 0.01 to avoid zero
    lambda_a = st.slider("Unexpected Events (λ)", 0.0, 5.0, 1.5, key="adv_lambda")
    E_C_a   = st.number_input("Avg Cost per Event", value=600, key="adv_ec")
    p_a     = st.slider("Probability of Price Increase (p)", 0.0, 1.0, 0.5, key="adv_p")
    budget_a = st.number_input("Budget Threshold", value=10000, key="adv_budget")

    expected = F_a + V_a * (1 + mu_a) + lambda_a * E_C_a
    variance = (V_a ** 2) * (sigma_a ** 2) + lambda_a * (E_C_a ** 2)
    std_dev  = np.sqrt(variance)

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Expense", f"₹{expected:,.2f}")
    col2.metric("Std Deviation",    f"₹{std_dev:,.2f}")
    col3.metric("Variance",         f"₹{variance:,.2f}")

    lower_ci = expected - 1.96 * std_dev
    upper_ci = expected + 1.96 * std_dev
    st.write(f"**95% Confidence Interval:** ₹{lower_ci:,.2f} — ₹{upper_ci:,.2f}")

    prob = 1 - norm.cdf(budget_a, expected, std_dev)
    st.write(f"**P(Expense > Budget):** {prob:.2%}")

    if prob < 0.2:
        st.success("Low Risk ")
    elif prob < 0.5:
        st.warning("Moderate Risk ")
    else:
        st.error("High Risk ")

    # -------- Distribution Plots --------
    col1, col2, col3 = st.columns(3)

    with col1:
        x = np.linspace(mu_a - 4*sigma_a, mu_a + 4*sigma_a, 200)
        y = norm.pdf(x, mu_a, sigma_a)
        fig, ax = plt.subplots()
        ax.plot(x, y, color='steelblue')
        ax.fill_between(x, y, alpha=0.3, color='steelblue')
        ax.set_title("Inflation (Normal)")
        st.pyplot(fig)
        plt.close()

    with col2:
        k = np.arange(0, 12)
        y = poisson.pmf(k, lambda_a)
        fig2, ax2 = plt.subplots()
        ax2.bar(k, y, color='orange')
        ax2.set_title("Events (Poisson)")
        st.pyplot(fig2)
        plt.close()

    with col3:
        k = np.arange(0, 13)
        y = binom.pmf(k, 12, p_a)
        fig3, ax3 = plt.subplots()
        ax3.bar(k, y, color='green')
        ax3.set_title("Price Increases (Binomial)")
        st.pyplot(fig3)
        plt.close()

    # -------- 12-Month Forecast Table --------
    st.subheader("12-Month Forecast Table")
    forecast = []
    current = V_a
    for i in range(12):
        current = current * (1 + mu_a)
        forecast.append(round(F_a + current + lambda_a * E_C_a, 2))

    df_forecast = pd.DataFrame({
        "Month":   range(1, 13),
        "Expense": forecast
    })
    st.dataframe(df_forecast, use_container_width=True)