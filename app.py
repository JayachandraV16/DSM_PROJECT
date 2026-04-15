import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, binom

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Household Expense Forecaster", layout="wide")

# ---------------- GLOBAL STYLE ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0d1b2a;
    color: #e8f0fe;
}
h1, h2, h3 {
    color: #ffffff;
}
.info-box {
    background: rgba(37, 99, 235, 0.15);
    border-left: 4px solid #2563eb;
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CHART STYLE ----------------
def style_chart(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor("#0f1f30")
    ax.figure.set_facecolor("#0f1f30")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_color("#334155")
    ax.spines['bottom'].set_color("#334155")

    ax.tick_params(colors="#cbd5e1")
    ax.title.set_color("#e2e8f0")

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, color="#cbd5e1")
    ax.set_ylabel(ylabel, color="#cbd5e1")

    ax.grid(alpha=0.2)

# ---------------- FUNCTIONS ----------------
def compute_expected(F, V, mu, lambda_, E_C):
    return F + V*(1+mu) + lambda_*E_C

def forecast_future(months, base, mu):
    result = []
    for _ in range(months):
        base *= (1 + mu)
        result.append(base)
    return result

def simulate(F, V, mu, months, income):
    expenses, savings = [], []
    for _ in range(months):
        V *= (1 + mu)
        total = F + V
        expenses.append(total)
        savings.append(income - total)
    return expenses, savings

def generate_insights(past_data, pred, income):
    insights = []

    if len(past_data) >= 2:
        if past_data[-1] > past_data[0]:
            insights.append("Expenses show an increasing trend over time.")
        elif past_data[-1] < past_data[0]:
            insights.append("Expenses show a decreasing trend.")

    if np.std(past_data) > 0.3 * np.mean(past_data):
        insights.append("High variation detected in monthly expenses.")

    ratio = pred / income
    if ratio > 0.8:
        insights.append("High risk: expenses are close to or exceeding income.")
    elif ratio > 0.5:
        insights.append("Moderate risk: monitor your spending carefully.")
    else:
        insights.append("Low risk: spending is within a safe range.")

    savings = income - pred
    if savings < 0:
        insights.append("You are likely to face a deficit.")
    elif savings < 0.2 * income:
        insights.append("Savings are low; consider reducing expenses.")
    else:
        insights.append("Good savings level maintained.")

    return insights

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("User Profile")

    income = st.number_input("Monthly Income (Rs.)", value=50000)
    age = st.slider("Age", 18, 60, 28)
    dependents = st.slider("Dependents", 0, 5, 1)

    st.markdown("---")

    st.subheader("Model Formula")

    st.markdown("""
E(X) = F + V(1 + μ) + λE(C)

F = Fixed expenses  
V = Variable expenses  
μ = Inflation rate  
λ = Unexpected events  
E(C) = Avg cost  
""")

# ---------------- HEADER ----------------
st.title("Household Expense Forecasting System")
st.write("Statistical model using inflation and uncertainty")

tab1, tab2, tab3 = st.tabs(["Forecast", "Simulation", "Analytics"])

# =====================================================
# TAB 1 — FORECAST
# =====================================================
with tab1:

    st.subheader("Step 1 — Past Expenses")

    months = st.number_input("Number of months", 2, 12, 6)

    past_data = []
    cols = st.columns(min(6, int(months)))

    for i in range(int(months)):
        with cols[i % 6]:
            val = st.number_input(f"Month {i+1}", value=4000, key=i)
            past_data.append(val)

    avg = np.mean(past_data)

    st.markdown(f"""
<div class="info-box">
Average Monthly Expense: Rs.{avg:,.0f}
</div>
""", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(past_data, color="#3b82f6", linewidth=2, marker='o')
    ax.fill_between(range(len(past_data)), past_data, alpha=0.15, color="#3b82f6")
    style_chart(ax, "Past Expense Trend", "Month", "Rs.")
    st.pyplot(fig)
    st.caption("Monthly expense trend based on user data")

    # -------- PREDICTION --------
    st.subheader("Step 2 — Prediction")

    if st.button("Calculate Prediction"):

        F = avg * 0.4
        V = avg * 0.6
        mu = 0.05
        lambda_ = 1.5
        E_C = 600

        pred = compute_expected(F, V, mu, lambda_, E_C)

        ratio = pred / income
        if ratio < 0.5:
            risk = "Low"
        elif ratio < 0.8:
            risk = "Medium"
        else:
            risk = "High"

        st.session_state.pred = pred
        st.session_state.risk = risk

    if "pred" in st.session_state:

        pred = st.session_state.pred
        savings = income - pred

        # -------- INSIGHTS --------
        st.subheader("Insights")
        insights = generate_insights(past_data, pred, income)

        for ins in insights:
            if "High risk" in ins or "deficit" in ins:
                st.error(ins)
            elif "Moderate" in ins or "low" in ins:
                st.warning(ins)
            else:
                st.success(ins)

        # -------- METRICS --------
        col1, col2, col3 = st.columns(3)
        col1.metric("Income", f"Rs.{income:,.0f}")
        col2.metric("Expense", f"Rs.{pred:,.0f}")
        col3.metric("Savings", f"Rs.{savings:,.0f}")

        st.write(f"Risk Level: {st.session_state.risk}")

    # -------- FORECAST --------
    st.subheader("Step 3 — Future Forecast")

    future_months = st.number_input("Forecast Months", 1, 12, 6)

    if st.button("Generate Forecast"):

        forecast = forecast_future(int(future_months), avg, 0.05)

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(forecast, color="#f97316", linewidth=2, marker='o')
        ax2.fill_between(range(len(forecast)), forecast, alpha=0.15, color="#f97316")
        style_chart(ax2, "Future Expense Projection", "Month", "Rs.")
        st.pyplot(fig2)
        st.caption("Projected expenses considering inflation")

# =====================================================
# TAB 2 — SIMULATION
# =====================================================
with tab2:

    st.subheader("Simulation")

    F = st.number_input("Fixed Expenses", 15000)
    V = st.number_input("Variable Expenses", 12000)
    mu = st.slider("Inflation Rate", 0.0, 0.2, 0.05)
    months_sim = st.number_input("Months", 1, 24, 12)

    if st.button("Run Simulation"):

        expenses, savings = simulate(F, V, mu, int(months_sim), income)

        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.plot(expenses, color="#ef4444", linewidth=2)
        ax3.fill_between(range(len(expenses)), expenses, alpha=0.15, color="#ef4444")
        style_chart(ax3, "Simulated Expenses", "Month", "Rs.")
        st.pyplot(fig3)
        st.caption("Expense variation over time")

        fig4, ax4 = plt.subplots(figsize=(5, 3))
        ax4.bar(range(len(savings)), savings,
                color=["#10b981" if s >= 0 else "#ef4444" for s in savings])
        style_chart(ax4, "Simulated Savings", "Month", "Rs.")
        st.pyplot(fig4)
        st.caption("Savings trend over time")

# =====================================================
# TAB 3 — ANALYTICS
# =====================================================
with tab3:

    st.subheader("Statistical Analysis")

    F = st.number_input("F", 15000)
    V = st.number_input("V", 12000)
    mu = st.slider("mu", 0.0, 0.2, 0.05)
    sigma = st.slider("sigma", 0.01, 0.1, 0.02)
    lambda_ = st.slider("lambda", 0.0, 5.0, 1.5)
    E_C = st.number_input("E(C)", 600)

    expected = compute_expected(F, V, mu, lambda_, E_C)
    variance = (V**2)*(sigma**2) + lambda_*(E_C**2)

    st.write(f"Expected Expense: Rs.{expected:,.0f}")
    st.write(f"Variance: {variance:,.0f}")

    x = np.linspace(mu-4*sigma, mu+4*sigma, 200)
    y = norm.pdf(x, mu, sigma)

    fig5, ax5 = plt.subplots(figsize=(5, 3))
    ax5.plot(x, y, color="#3b82f6")
    ax5.fill_between(x, y, alpha=0.2, color="#3b82f6")
    style_chart(ax5, "Normal Distribution (Inflation)", "μ", "Density")
    st.pyplot(fig5)
    st.caption("Inflation variability")

    k = np.arange(0, 10)
    y2 = poisson.pmf(k, lambda_)

    fig6, ax6 = plt.subplots(figsize=(5, 3))
    ax6.bar(k, y2, color="#f97316")
    style_chart(ax6, "Poisson Distribution (Events)", "Events", "Probability")
    st.pyplot(fig6)
    st.caption("Unexpected events frequency")

    k = np.arange(0, 12)
    y3 = binom.pmf(k, 12, 0.5)

    fig7, ax7 = plt.subplots(figsize=(5, 3))
    ax7.bar(k, y3, color="#10b981")
    style_chart(ax7, "Binomial Distribution (Price Increase)", "Count", "Probability")
    st.pyplot(fig7)
    st.caption("Price increase probability")