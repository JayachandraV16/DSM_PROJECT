import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import norm, poisson, binom

# -- PAGE CONFIG ---------------------------------------------------------------
st.set_page_config(
    page_title="Household Expense Forecaster",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- GLOBAL STYLES -------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #ffffff;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif; color: #ffffff; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1923 0%, #1a2e3b 100%);
}
[data-testid="stSidebar"] * { color: #e8f0fe !important; }
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stNumberInput > label { color: #94b8d4 !important; font-size: 0.82rem; }

/* Main background */
.stApp {
    background-color: #0d1b2a;
}
section[data-testid="stMainBlockContainer"] {
    background-color: #0d1b2a;
}

/* Cards */
.info-card {
    background: rgba(37, 99, 235, 0.15);
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 16px;
    font-size: 0.88rem;
    color: #c8d8f0;
    line-height: 1.6;
}
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #e8f0fe;
    margin-top: 1.4rem;
    margin-bottom: 0.3rem;
    border-bottom: 2px solid #1e3a5f;
    padding-bottom: 4px;
}
.result-box {
    background: rgba(30, 58, 95, 0.5);
    border: 1px solid #2563eb;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 10px;
    color: #e8f0fe;
}
.badge-low    { background:#065f46; color:#d1fae5; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }
.badge-medium { background:#92400e; color:#fef3c7; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }
.badge-high   { background:#991b1b; color:#fee2e2; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }

/* All labels and text white */
label, .stMarkdown, .stMarkdown p, .stMarkdown li,
.stTextInput label, .stNumberInput label, .stSlider label,
p, span, div {
    color: #e8f0fe !important;
}

/* Input fields */
input, textarea, .stTextInput input, .stNumberInput input {
    color: #ffffff !important;
    background-color: #1a2e3b !important;
    border-color: #2d4a6b !important;
}

/* Placeholder text */
::placeholder {
    color: #7a96b4 !important;
    opacity: 1;
}

/* Metric labels and values */
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
[data-testid="stCaption"] {
    color: #e8f0fe !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: #c8d8f0 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
}

/* Divider */
hr { border-color: #1e3a5f; margin: 1.5rem 0; }

/* Dataframe */
[data-testid="stDataFrame"] {
    color: #e8f0fe !important;
}

/* Selectbox, number input spinners */
.stNumberInput button {
    color: #ffffff !important;
    background-color: #1a2e3b !important;
}

/* Caption */
.st-emotion-cache-16idsys p, small {
    color: #94b8d4 !important;
}

/* Alert/info boxes */
.stAlert {
    color: #e8f0fe !important;
}

</style>
""", unsafe_allow_html=True)


# -- HELPER: styled info box ---------------------------------------------------
def info_box(text):
    st.markdown(f'<div class="info-card">{text}</div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# -- CHART STYLE ---------------------------------------------------------------
CHART_COLOR   = "#3b82f6"
CHART_COLOR2  = "#f97316"
CHART_GREEN   = "#10b981"
CHART_RED     = "#ef4444"
CHART_BG      = "#0f1f30"
CHART_GRID    = "#1e3a5f"
CHART_TEXT    = "#c8d8f0"

def style_ax(ax, title="", xlabel="Month", ylabel="Amount (Rs.)"):
    ax.set_facecolor(CHART_BG)
    ax.figure.set_facecolor(CHART_BG)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color(CHART_GRID)
    ax.tick_params(colors=CHART_TEXT, labelsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', color='#e8f0fe', pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=CHART_TEXT)
    ax.set_ylabel(ylabel, fontsize=9, color=CHART_TEXT)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'Rs.{x:,.0f}'))
    ax.xaxis.label.set_color(CHART_TEXT)
    ax.yaxis.label.set_color(CHART_TEXT)


# -- MODEL TRAINING ------------------------------------------------------------
@st.cache_resource(show_spinner="Training models on your dataset...")
def load_models():
    expense_cols = [
        'Rent','Loan_Repayment','Insurance','Groceries','Transport',
        'Eating_Out','Entertainment','Utilities','Healthcare',
        'Education','Miscellaneous'
    ]
    df = pd.read_csv("data/data.csv")
    df['total_expense'] = df[expense_cols].sum(axis=1)
    df['savings_rate']  = (df['Income'] - df['total_expense']) / df['Income']

    # Use expense ratio relative to income for risk
    df['expense_ratio'] = df['total_expense'] / df['Income']
    def assign_risk(r):
        return "Low" if r < 0.5 else ("Medium" if r < 0.8 else "High")
    df['risk_level'] = df['expense_ratio'].apply(assign_risk)

    X = df[['Income','Age','Dependents']]

    # Regress on expense_ratio instead of raw total, to avoid scale issues
    fm = RandomForestRegressor(n_estimators=200, random_state=42)
    fm.fit(X, df['expense_ratio'])

    rm = RandomForestClassifier(n_estimators=200, random_state=42)
    rm.fit(X, df['risk_level'])
    return fm, rm

try:
    forecast_model, risk_model = load_models()
except FileNotFoundError:
    st.error("Dataset not found. Please ensure `data/data.csv` exists in your project folder.")
    st.stop()
except KeyError as e:
    st.error(f"Missing column in dataset: {e}. Check your CSV has the required expense columns.")
    st.stop()


# -- MATH FUNCTIONS ------------------------------------------------------------
def advanced_simulation(F, V, mu, months, income):
    expenses, savings = [], []
    current_V = V
    for _ in range(months):
        current_V *= (1 + mu)
        total = F + current_V
        expenses.append(total)
        savings.append(income - total)
    return expenses, savings

def monte_carlo_forecast_with_ci(F, V, mu, lambda_, E_C, months=12, sims=300):
    all_sims = []
    for _ in range(sims):
        current, sim = V, []
        for _ in range(months):
            I_t   = np.random.normal(mu, 0.01)
            N_t   = np.random.poisson(lambda_)
            cost  = np.sum(np.random.randint(300, 800, max(N_t,1))) if N_t > 0 else 0
            current = current * (1 + I_t) + cost
            sim.append(current)
        all_sims.append(sim)
    arr = np.array(all_sims)
    return np.mean(arr,0), np.percentile(arr,5,0), np.percentile(arr,95,0)

def forecast_future(months, base, mu):
    result, cur = [], base
    for _ in range(months):
        cur *= (1 + mu)
        result.append(cur)
    return result

def forecast_with_uncertainty(months, base, mu, lambda_, E_C):
    result, cur = [], base
    for _ in range(months):
        I_t  = np.random.normal(mu, 0.01)
        N_t  = np.random.poisson(lambda_)
        cost = np.sum(np.random.randint(300, 800, max(N_t,1))) if N_t > 0 else 0
        cur  = cur * (1 + I_t) + cost
        result.append(cur)
    return result


# -- SIDEBAR -------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Your Profile")
    st.markdown("These values are used across all tabs for ML prediction.")
    st.markdown("---")
    income     = st.number_input("Monthly Income (Rs.)", value=50000, step=1000,
                                  help="Your gross monthly income before expenses")
    age        = st.slider("Age", 18, 60, 28,
                            help="Your current age — affects spending patterns in the model")
    dependents = st.slider("Number of Dependents", 0, 5, 1,
                            help="Children or family members financially dependent on you")
    st.markdown("---")
    st.markdown("#### Quick Reference")
    st.markdown("""
- **mu** = Inflation / growth rate  
- **lambda** = Avg unexpected events/month  
- **E[C]** = Average cost per unexpected event  
- **F** = Fixed monthly expenses  
- **V** = Variable monthly expenses  
""")


# -- APP HEADER ----------------------------------------------------------------
st.markdown("## Household Expense Forecasting System")
st.markdown(
    "A data-driven tool combining **machine learning** and **statistical modelling** "
    "to predict, simulate, and analyse household expenses."
)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "Forecast & Prediction",
    "Category Breakdown",
    "Simulation",
    "Advanced Analytics",
])


# ==============================================================================
# TAB 1 — FORECAST & PREDICTION
# ==============================================================================
with tab1:

    info_box(
        "Enter your past monthly expenses below. The system uses these to detect trends "
        "and combines them with machine learning (trained on household survey data) to "
        "predict next month's expense and your financial risk level."
    )

    section("Step 1 — Enter Past Monthly Expenses")
    months = st.number_input("How many months of data do you have?", value=6, min_value=2, step=1)

    col_inputs = st.columns(min(int(months), 6))
    past_data  = []
    for i in range(int(months)):
        with col_inputs[i % 6]:
            val = st.number_input(f"Month {i+1}", value=4000, step=100, key=f"month_{i}", label_visibility="visible")
            past_data.append(val)

    fig, ax = plt.subplots(figsize=(7, 3))
    fig.set_facecolor(CHART_BG)
    ax.plot(range(1, int(months)+1), past_data, marker='o', linewidth=2,
            color=CHART_COLOR, markerfacecolor=CHART_BG, markeredgewidth=2)
    ax.fill_between(range(1, int(months)+1), past_data, alpha=0.15, color=CHART_COLOR)
    style_ax(ax, "Your Past Expense Trend", "Month", "Rs. Expense")
    st.pyplot(fig)
    plt.close()

    section("Step 2 — Get Prediction")

    col_pred, col_budget = st.columns(2)

    with col_pred:
        if st.button("Predict Next Month's Expense", type="primary"):
            X_input = [[income, age, dependents]]
            # Model predicts expense_ratio; multiply by income for actual amount
            pred_ratio = forecast_model.predict(X_input)[0]
            pred = pred_ratio * income
            risk = risk_model.predict(X_input)[0]

            st.session_state.pred = pred
            st.session_state.risk = risk

        if "pred" in st.session_state:
            badge = {
                "Low":    '<span class="badge-low">Low Risk</span>',
                "Medium": '<span class="badge-medium">Medium Risk</span>',
                "High":   '<span class="badge-high">High Risk</span>',
            }.get(st.session_state.risk, "")

            st.markdown(f"""
<div class="result-box">
<b>ML Predicted Expense:</b> Rs.{st.session_state.pred:,.2f}<br>
<b>Risk Level:</b> {badge}
</div>
""", unsafe_allow_html=True)

    with col_budget:
        section("Budget Check")
        budget = st.number_input("Your Monthly Budget (Rs.)", value=60000, step=1000)
        if "pred" in st.session_state:
            pred   = st.session_state.pred
            saving = income - pred
            over   = pred > budget
            if over:
                st.error(f"Predicted expense Rs.{pred:,.0f} exceeds budget of Rs.{budget:,.0f}")
            else:
                surplus = budget - pred
                st.success(f"Within budget. You have Rs.{surplus:,.0f} headroom.")
            col_a, col_b = st.columns(2)
            col_a.metric("Predicted Expense", f"Rs.{pred:,.0f}")
            col_b.metric("Expected Savings",  f"Rs.{saving:,.0f}",
                         delta=f"{'Deficit' if saving < 0 else 'Surplus'}")

    section("Step 3 — Forecast Future Months")
    info_box(
        "Trend Forecast assumes a steady inflation rate. "
        "Uncertain Forecast adds random unexpected costs each month (e.g. medical, repairs) — "
        "more realistic but noisier."
    )

    future_months = st.number_input("Number of months to forecast", value=6, min_value=1, step=1)
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        if st.button("Trend Forecast (Steady Growth)"):
            base     = np.mean(past_data)
            forecast = forecast_future(int(future_months), base, 0.05)
            fig, ax  = plt.subplots(figsize=(5.5, 3.5))
            fig.set_facecolor(CHART_BG)
            ax.plot(range(1, int(future_months)+1), forecast, marker='o', linewidth=2,
                    color=CHART_COLOR, markerfacecolor=CHART_BG, markeredgewidth=2)
            ax.fill_between(range(1, int(future_months)+1), forecast, alpha=0.15, color=CHART_COLOR)
            style_ax(ax, "Trend Forecast (5% monthly growth)")
            st.pyplot(fig); plt.close()

    with col_f2:
        if st.button("Uncertain Forecast (With Random Events)"):
            base     = np.mean(past_data)
            forecast = forecast_with_uncertainty(int(future_months), base, 0.05, 1.5, 600)
            fig, ax  = plt.subplots(figsize=(5.5, 3.5))
            fig.set_facecolor(CHART_BG)
            ax.plot(range(1, int(future_months)+1), forecast, marker='o', linewidth=2,
                    color=CHART_COLOR2, markerfacecolor=CHART_BG, markeredgewidth=2)
            ax.fill_between(range(1, int(future_months)+1), forecast, alpha=0.15, color=CHART_COLOR2)
            style_ax(ax, "Forecast with Random Unexpected Costs")
            st.pyplot(fig); plt.close()


# ==============================================================================
# TAB 2 — CATEGORY BREAKDOWN
# ==============================================================================
with tab2:

    info_box(
        "Break your monthly expenses into categories like Rent, Food, Transport, etc. "
        "Add or remove categories freely. The charts update in real time."
    )

    if "categories" not in st.session_state:
        st.session_state.categories = {
            "Rent": 12000, "Groceries": 6000, "Transport": 3000,
            "Utilities": 2000, "Eating Out": 2500, "Entertainment": 1500
        }

    section("Manage Categories")

    cats_to_delete = []
    new_categories  = {}

    for cat, val in list(st.session_state.categories.items()):
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            new_name = st.text_input("Category Name", cat, key=f"name_{cat}", label_visibility="collapsed")
        with c2:
            new_val = st.number_input("Rs. Amount", value=val, step=100, key=f"val_{cat}", label_visibility="collapsed")
        with c3:
            if st.button("Remove", key=f"del_{cat}"):
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
    ca, cb, cc = st.columns([3, 2, 1])
    with ca:
        new_cat_name = st.text_input("New Category Name", placeholder="e.g. Healthcare", label_visibility="collapsed")
    with cb:
        new_cat_val = st.number_input("Rs. Amount", value=0, step=100, label_visibility="collapsed", key="new_cat_val")
    with cc:
        if st.button("Add"):
            if new_cat_name.strip():
                st.session_state.categories[new_cat_name.strip()] = new_cat_val
                st.rerun()
            else:
                st.warning("Please enter a category name.")

    labels = list(st.session_state.categories.keys())
    values = list(st.session_state.categories.values())

    if labels and sum(values) > 0:
        st.markdown(f"### Total Monthly Expense: Rs.{sum(values):,.0f}")

        col1, col2 = st.columns(2)
        palette = plt.colormaps.get_cmap('Blues')
        colors  = [palette(0.3 + 0.5 * i / max(len(labels)-1, 1)) for i in range(len(labels))]

        with col1:
            fig, ax = plt.subplots(figsize=(5, 4.5))
            fig.set_facecolor(CHART_BG)
            ax.set_facecolor(CHART_BG)
            wedges, texts, autotexts = ax.pie(
                values, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90,
                wedgeprops=dict(linewidth=1.5, edgecolor=CHART_BG)
            )
            for t in texts: t.set_color(CHART_TEXT); t.set_fontsize(8)
            for t in autotexts: t.set_fontsize(8); t.set_color('#ffffff')
            ax.set_title("Expense Breakdown", fontsize=11, fontweight='bold', color='#e8f0fe')
            st.pyplot(fig); plt.close()

        with col2:
            sorted_pairs = sorted(zip(values, labels), reverse=True)
            s_values, s_labels = zip(*sorted_pairs)
            fig2, ax2 = plt.subplots(figsize=(5, 4.5))
            fig2.set_facecolor(CHART_BG)
            bars = ax2.barh(s_labels, s_values, color=CHART_COLOR, edgecolor=CHART_BG, height=0.6)
            ax2.bar_label(bars, labels=[f'Rs.{v:,.0f}' for v in s_values], padding=4, fontsize=8, color=CHART_TEXT)
            ax2.set_facecolor(CHART_BG)
            ax2.spines[['top','right','left']].set_visible(False)
            ax2.spines['bottom'].set_color(CHART_GRID)
            ax2.tick_params(colors=CHART_TEXT, labelsize=9)
            ax2.set_title("Ranked by Amount", fontsize=11, fontweight='bold', color='#e8f0fe')
            ax2.set_xlabel("Rs. Amount", fontsize=9, color=CHART_TEXT)
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'Rs.{x:,.0f}'))
            st.pyplot(fig2); plt.close()


# ==============================================================================
# TAB 3 — SIMULATION
# ==============================================================================
with tab3:

    info_box(
        "Simulation shows how expenses and savings evolve over time under a steady inflation rate. "
        "The Confidence Interval forecast runs 300 Monte Carlo simulations to show best-case, "
        "worst-case, and average expense paths."
    )

    section("Parameters")
    sc1, sc2 = st.columns(2)
    with sc1:
        F          = st.number_input("Fixed Monthly Expenses (Rs.) — e.g. Rent, EMI", value=15000, step=500, key="sim_F",
                                      help="Expenses that don't change month to month")
        V          = st.number_input("Variable Monthly Expenses (Rs.) — e.g. Food, Transport", value=12000, step=500, key="sim_V",
                                      help="Expenses that vary and grow with inflation")
        income_sim = st.number_input("Monthly Income (Rs.)", value=income, step=1000, key="sim_income")
    with sc2:
        mu         = st.slider("Monthly Inflation Rate (mu)", 0.0, 0.20, 0.05, step=0.005, key="sim_mu",
                                help="How much variable expenses grow each month (5% = 0.05)")
        months_sim = st.number_input("Simulation Length (months)", value=12, min_value=3, step=1, key="sim_months")

    if st.button("Run Expense & Savings Simulation", type="primary"):
        expenses, savings = advanced_simulation(F, V, mu, int(months_sim), income_sim)
        x = range(1, int(months_sim)+1)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            fig.set_facecolor(CHART_BG)
            ax.plot(x, expenses, linewidth=2, color=CHART_RED, marker='o',
                    markerfacecolor=CHART_BG, markeredgewidth=2)
            ax.fill_between(x, expenses, alpha=0.15, color=CHART_RED)
            style_ax(ax, "Projected Monthly Expenses")
            st.pyplot(fig); plt.close()
        with col2:
            colors_s = [CHART_GREEN if s >= 0 else CHART_RED for s in savings]
            fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
            fig2.set_facecolor(CHART_BG)
            ax2.bar(x, savings, color=colors_s, edgecolor=CHART_BG)
            ax2.axhline(0, color='#475569', linewidth=1, linestyle='--')
            style_ax(ax2, "Projected Monthly Savings")
            st.pyplot(fig2); plt.close()

        final_savings = income_sim * int(months_sim) - sum(expenses)
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Avg Monthly Expense", f"Rs.{np.mean(expenses):,.0f}")
        col_m2.metric("Final Month Expense",  f"Rs.{expenses[-1]:,.0f}")
        col_m3.metric("Total Savings",        f"Rs.{final_savings:,.0f}",
                      delta="Surplus" if final_savings >= 0 else "Deficit")

    st.divider()
    section("Monte Carlo Confidence Interval Forecast")
    info_box(
        "Runs 300 simulations with random inflation shocks and unexpected events. "
        "The shaded area shows the 90% confidence band — expenses will likely fall within this range."
    )

    mc1, mc2 = st.columns(2)
    with mc1:
        lambda_ci = st.slider("Expected Unexpected Events per Month (lambda)", 0.0, 5.0, 1.5, step=0.1, key="ci_lambda",
                               help="e.g. 1.5 means ~1-2 surprise expenses per month")
        E_C_ci    = st.number_input("Average Cost of Each Unexpected Event (Rs.)", value=600, step=100, key="ci_ec")
    with mc2:
        months_ci = st.number_input("Forecast Length (months)", value=12, min_value=3, step=1, key="ci_months")

    if st.button("Run Monte Carlo Forecast"):
        mean, lower, upper = monte_carlo_forecast_with_ci(F, V, mu, lambda_ci, E_C_ci, int(months_ci))
        x = range(1, int(months_ci)+1)

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.set_facecolor(CHART_BG)
        ax.plot(x, mean,  label="Mean Forecast", color=CHART_COLOR, linewidth=2.5)
        ax.plot(x, upper, label="90th Percentile (Worst Case)",  color=CHART_RED, linewidth=1, linestyle='--')
        ax.plot(x, lower, label="10th Percentile (Best Case)",   color=CHART_GREEN, linewidth=1, linestyle='--')
        ax.fill_between(x, lower, upper, alpha=0.12, color=CHART_COLOR)
        ax.set_facecolor(CHART_BG)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color(CHART_GRID)
        legend = ax.legend(fontsize=9)
        for text in legend.get_texts():
            text.set_color(CHART_TEXT)
        legend.get_frame().set_facecolor(CHART_BG)
        style_ax(ax, "Monte Carlo Forecast with 90% Confidence Band")
        st.pyplot(fig); plt.close()


# ==============================================================================
# TAB 4 — ADVANCED ANALYTICS
# ==============================================================================
with tab4:

    info_box(
        "This section uses probability distributions to model expense uncertainty. "
        "Normal distribution for inflation, Poisson for unexpected event frequency, "
        "and Binomial for price-increase probability."
    )

    section("Model Parameters")
    aa1, aa2 = st.columns(2)
    with aa1:
        F_a      = st.number_input("Fixed Expenses F (Rs.)", value=15000, step=500, key="adv_F")
        V_a      = st.number_input("Variable Expenses V (Rs.)", value=12000, step=500, key="adv_V")
        mu_a     = st.slider("Mean Inflation Rate mu", 0.0, 0.20, 0.05, step=0.005, key="adv_mu")
        sigma_a  = st.slider("Inflation Std Dev sigma (uncertainty)", 0.01, 0.10, 0.02, key="adv_sigma",
                              help="Higher sigma = more volatile inflation")
    with aa2:
        lambda_a  = st.slider("Unexpected Events lambda", 0.0, 5.0, 1.5, key="adv_lambda")
        E_C_a     = st.number_input("Avg Event Cost E[C] (Rs.)", value=600, step=100, key="adv_ec")
        p_a       = st.slider("Probability of Price Increase (p)", 0.0, 1.0, 0.5, key="adv_p",
                               help="Binomial parameter: chance any given expense item rises next month")
        budget_a  = st.number_input("Budget Threshold (Rs.)", value=30000, step=1000, key="adv_budget")

    st.divider()
    section("Expected Value & Risk Metrics")
    info_box(
        "Formula: E[Total Expense] = F + V*(1+mu) + lambda*E[C]  |  "
        "Variance: Var = V^2 * sigma^2 + lambda * E[C]^2"
    )

    expected = F_a + V_a * (1 + mu_a) + lambda_a * E_C_a
    variance = (V_a**2) * (sigma_a**2) + lambda_a * (E_C_a**2)
    std_dev  = np.sqrt(variance)
    lower_ci = expected - 1.96 * std_dev
    upper_ci = expected + 1.96 * std_dev
    prob     = 1 - norm.cdf(budget_a, expected, std_dev)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Expected Expense",    f"Rs.{expected:,.0f}")
    col_m2.metric("Std Deviation",       f"Rs.{std_dev:,.0f}", help="Typical deviation from expected value")
    col_m3.metric("95% CI Lower",        f"Rs.{lower_ci:,.0f}")
    col_m4.metric("95% CI Upper",        f"Rs.{upper_ci:,.0f}")

    risk_color = ("success" if prob < 0.2 else "warning" if prob < 0.5 else "error")
    risk_label = ("Low Risk — expenses unlikely to exceed budget." if prob < 0.2
                  else "Moderate Risk — significant chance of budget overrun."
                  if prob < 0.5 else "High Risk — very likely to exceed budget.")
    getattr(st, risk_color)(f"P(Expense > Budget) = {prob:.1%} — {risk_label}")

    section("Probability Distribution Plots")
    info_box(
        "These show the statistical distributions used in the model. "
        "They help visualise uncertainty around inflation, event frequency, and price changes."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        x = np.linspace(mu_a - 4*sigma_a, mu_a + 4*sigma_a, 200)
        y = norm.pdf(x, mu_a, sigma_a)
        fig, ax = plt.subplots(figsize=(4.5, 3))
        fig.set_facecolor(CHART_BG)
        ax.plot(x, y, color=CHART_COLOR, linewidth=2)
        ax.fill_between(x, y, alpha=0.2, color=CHART_COLOR)
        ax.set_facecolor(CHART_BG)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color(CHART_GRID)
        ax.tick_params(colors=CHART_TEXT, labelsize=8)
        ax.set_title("Inflation Rate\n(Normal Distribution)", fontsize=10, fontweight='bold', color='#e8f0fe')
        ax.set_xlabel("Inflation Rate mu", fontsize=8, color=CHART_TEXT)
        ax.set_ylabel("Probability Density", fontsize=8, color=CHART_TEXT)
        st.pyplot(fig); plt.close()
        st.caption("Models how much inflation varies month to month.")

    with c2:
        k = np.arange(0, 12)
        y = poisson.pmf(k, lambda_a)
        fig2, ax2 = plt.subplots(figsize=(4.5, 3))
        fig2.set_facecolor(CHART_BG)
        ax2.bar(k, y, color=CHART_COLOR2, edgecolor=CHART_BG, width=0.6)
        ax2.set_facecolor(CHART_BG)
        ax2.spines[['top','right']].set_visible(False)
        ax2.spines[['left','bottom']].set_color(CHART_GRID)
        ax2.tick_params(colors=CHART_TEXT, labelsize=8)
        ax2.set_title("Unexpected Events\n(Poisson Distribution)", fontsize=10, fontweight='bold', color='#e8f0fe')
        ax2.set_xlabel("Number of Events N", fontsize=8, color=CHART_TEXT)
        ax2.set_ylabel("Probability", fontsize=8, color=CHART_TEXT)
        st.pyplot(fig2); plt.close()
        st.caption("Models how many surprise expenses occur per month.")

    with c3:
        k = np.arange(0, 13)
        y = binom.pmf(k, 12, p_a)
        fig3, ax3 = plt.subplots(figsize=(4.5, 3))
        fig3.set_facecolor(CHART_BG)
        ax3.bar(k, y, color=CHART_GREEN, edgecolor=CHART_BG, width=0.6)
        ax3.set_facecolor(CHART_BG)
        ax3.spines[['top','right']].set_visible(False)
        ax3.spines[['left','bottom']].set_color(CHART_GRID)
        ax3.tick_params(colors=CHART_TEXT, labelsize=8)
        ax3.set_title("Price Increases\n(Binomial Distribution)", fontsize=10, fontweight='bold', color='#e8f0fe')
        ax3.set_xlabel("Items with Price Rise (out of 12)", fontsize=8, color=CHART_TEXT)
        ax3.set_ylabel("Probability", fontsize=8, color=CHART_TEXT)
        st.pyplot(fig3); plt.close()
        st.caption("Models how many expense categories see a price hike.")

    section("12-Month Forecast Table")
    info_box("Month-by-month projected total expense using the mathematical model.")

    forecast_rows = []
    current = V_a
    for i in range(12):
        current = current * (1 + mu_a)
        total   = round(F_a + current + lambda_a * E_C_a, 2)
        forecast_rows.append({
            "Month": i + 1,
            "Fixed (F)": f"Rs.{F_a:,.0f}",
            "Variable": f"Rs.{current:,.0f}",
            "Event Cost": f"Rs.{lambda_a * E_C_a:,.0f}",
            "Total Expense": f"Rs.{total:,.0f}",
            "vs Budget": "OK" if total <= budget_a else "Over",
        })

    st.dataframe(pd.DataFrame(forecast_rows), width="stretch", hide_index=True)