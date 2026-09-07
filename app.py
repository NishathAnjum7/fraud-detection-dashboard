"""
Financial Fraud Detection Dashboard
------------------------------------
A Streamlit app for exploring credit-card transaction data, inspecting
the performance of two fraud-detection models (Logistic Regression and
Random Forest), and scoring new transactions in real time.

Run locally:
    streamlit run app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_card_fraud_dataset.csv")
ENGINEERED_PATH = os.path.join(MODELS_DIR, "engineered_data.csv")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Approximate lat/lon for the 10 US cities in the dataset, used for the
# fraud-hotspot map.
CITY_COORDS = {
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
    "Philadelphia": (39.9526, -75.1652),
    "San Antonio": (29.4241, -98.4936),
    "San Diego": (32.7157, -117.1611),
    "Dallas": (32.7767, -96.7970),
    "San Jose": (37.3382, -121.8863),
}

CUSTOM_CSS = """
<style>
.metric-card {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    border: 1px solid #374151;
}
[data-testid="stMetricValue"] { font-size: 1.8rem; }
.stTabs [data-baseweb="tab-list"] { gap: 12px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached data / model loaders
# --------------------------------------------------------------------------
@st.cache_data
def load_engineered_data():
    if not os.path.exists(ENGINEERED_PATH):
        st.error(
            "Model artifacts not found. Please run `python src/train_model.py` "
            "first to generate the models and engineered dataset."
        )
        st.stop()
    df = pd.read_csv(ENGINEERED_PATH, parse_dates=["TransactionDate"])
    return df


@st.cache_resource
def load_models():
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    lr = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    return rf, lr, scaler, encoders


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


df = load_engineered_data()
rf_model, lr_model, scaler, encoders = load_models()
metrics = load_metrics()

# --------------------------------------------------------------------------
# Sidebar navigation + filters
# --------------------------------------------------------------------------
st.sidebar.title("🛡️ Fraud Detection")
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Overview",
        "🔍 Fraud Analytics",
        "🤖 Model Performance",
        "⚡ Live Prediction",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
locations = sorted(df["Location"].unique())
selected_locations = st.sidebar.multiselect("Location", locations, default=locations)

txn_types = sorted(df["TransactionType"].unique())
selected_types = st.sidebar.multiselect("Transaction Type", txn_types, default=txn_types)

date_min, date_max = df["TransactionDate"].min(), df["TransactionDate"].max()
date_range = st.sidebar.date_input(
    "Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max
)

filtered = df[
    df["Location"].isin(selected_locations) & df["TransactionType"].isin(selected_types)
]
if len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = filtered[
        (filtered["TransactionDate"] >= start) & (filtered["TransactionDate"] <= end)
    ]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dataset: synthetic credit-card transactions. Built for academic "
    "demonstration of an end-to-end fraud-analytics pipeline."
)

# --------------------------------------------------------------------------
# PAGE 1: Overview
# --------------------------------------------------------------------------
if page == "📊 Overview":
    st.title("Financial Fraud Detection Dashboard")
    st.markdown("Real-time analytics on transaction data and fraud patterns.")

    total_txns = len(filtered)
    total_fraud = int(filtered["IsFraud"].sum())
    fraud_rate = (total_fraud / total_txns * 100) if total_txns else 0
    total_amount = filtered["Amount"].sum()
    fraud_amount = filtered.loc[filtered["IsFraud"] == 1, "Amount"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total_txns:,}")
    c2.metric("Flagged Fraud Cases", f"{total_fraud:,}", delta=f"{fraud_rate:.2f}% of txns")
    c3.metric("Total Transaction Volume", f"${total_amount:,.0f}")
    c4.metric("Amount Lost to Fraud", f"${fraud_amount:,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transactions Over Time")
        daily = filtered.groupby(filtered["TransactionDate"].dt.date).size().reset_index(name="count")
        fig = px.line(daily, x="TransactionDate", y="count", markers=True)
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraud Cases Over Time")
        daily_fraud = (
            filtered[filtered["IsFraud"] == 1]
            .groupby(filtered["TransactionDate"].dt.date)
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(daily_fraud, x="TransactionDate", y="count")
        fig.update_traces(marker_color="#ef4444")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Transaction Type Split")
        type_counts = filtered["TransactionType"].value_counts().reset_index()
        type_counts.columns = ["TransactionType", "count"]
        fig = px.pie(type_counts, names="TransactionType", values="count", hole=0.5)
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Amount Distribution: Fraud vs Legit")
        fig = px.histogram(
            filtered,
            x="Amount",
            color=filtered["IsFraud"].map({0: "Legit", 1: "Fraud"}),
            barmode="overlay",
            nbins=40,
            opacity=0.7,
            color_discrete_map={"Legit": "#3b82f6", "Fraud": "#ef4444"},
        )
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# PAGE 2: Fraud Analytics
# --------------------------------------------------------------------------
elif page == "🔍 Fraud Analytics":
    st.title("Fraud Pattern Analysis")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Fraud Hotspots by City")
        loc_stats = (
            filtered.groupby("Location")
            .agg(transactions=("IsFraud", "size"), fraud_count=("IsFraud", "sum"))
            .reset_index()
        )
        loc_stats["fraud_rate_%"] = (loc_stats["fraud_count"] / loc_stats["transactions"] * 100).round(3)
        loc_stats["lat"] = loc_stats["Location"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
        loc_stats["lon"] = loc_stats["Location"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])

        fig = px.scatter_mapbox(
            loc_stats,
            lat="lat",
            lon="lon",
            size="fraud_count",
            color="fraud_rate_%",
            hover_name="Location",
            hover_data={"transactions": True, "fraud_count": True, "fraud_rate_%": True, "lat": False, "lon": False},
            color_continuous_scale="Reds",
            size_max=40,
            zoom=3,
            mapbox_style="carto-darkmatter",
        )
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraud Rate by Location")
        loc_stats_sorted = loc_stats.sort_values("fraud_rate_%", ascending=True)
        fig = px.bar(
            loc_stats_sorted, x="fraud_rate_%", y="Location", orientation="h",
            color="fraud_rate_%", color_continuous_scale="Reds",
        )
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Fraud Rate by Day of Week")
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        wk = filtered.groupby("TxnWeekday").agg(
            transactions=("IsFraud", "size"), fraud_count=("IsFraud", "sum")
        ).reindex(range(7)).fillna(0)
        wk["fraud_rate_%"] = (wk["fraud_count"] / wk["transactions"].replace(0, np.nan) * 100).fillna(0)
        wk["day"] = weekday_names
        fig = px.bar(wk, x="day", y="fraud_rate_%", color="fraud_rate_%", color_continuous_scale="Oranges")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Merchant Risk Distribution")
        merchant_stats = (
            filtered.groupby("MerchantID")
            .agg(transactions=("IsFraud", "size"), fraud_count=("IsFraud", "sum"))
            .reset_index()
        )
        merchant_stats["fraud_rate_%"] = (
            merchant_stats["fraud_count"] / merchant_stats["transactions"] * 100
        )
        fig = px.histogram(merchant_stats, x="fraud_rate_%", nbins=30)
        fig.update_traces(marker_color="#f59e0b")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 15 Highest-Risk Merchants (min. 5 transactions)")
    top_merchants = merchant_stats[merchant_stats["transactions"] >= 5].sort_values(
        "fraud_rate_%", ascending=False
    ).head(15)
    st.dataframe(top_merchants, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# PAGE 3: Model Performance
# --------------------------------------------------------------------------
elif page == "🤖 Model Performance":
    st.title("Model Performance & Evaluation")
    st.markdown(
        "Two supervised models were trained to flag fraudulent transactions: "
        "**Logistic Regression** and **Random Forest**, both with class-balanced "
        "weighting since fraud makes up roughly 1% of all transactions."
    )

    model_choice = st.selectbox("Select model", list(metrics["results"].keys()))
    res = metrics["results"][model_choice]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{res['roc_auc']:.3f}")
    c2.metric("Precision (Fraud class)", f"{res['report']['1']['precision']:.3f}")
    c3.metric("Recall (Fraud class)", f"{res['report']['1']['recall']:.3f}")
    c4.metric("F1-score (Fraud class)", f"{res['report']['1']['f1-score']:.3f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = np.array(res["confusion_matrix"])
        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Legit", "Predicted Fraud"],
            y=["Actual Legit", "Actual Fraud"],
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("ROC Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["roc_curve"]["fpr"], y=res["roc_curve"]["tpr"],
                                  mode="lines", name=model_choice, line=dict(color="#3b82f6", width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                  line=dict(color="gray", dash="dash")))
        fig.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Model Comparison")
    comp_rows = []
    for name, r in metrics["results"].items():
        comp_rows.append({
            "Model": name,
            "ROC-AUC": round(r["roc_auc"], 3),
            "Precision": round(r["report"]["1"]["precision"], 3),
            "Recall": round(r["report"]["1"]["recall"], 3),
            "F1-score": round(r["report"]["1"]["f1-score"], 3),
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    with st.expander("📌 Why is precision low? (Read before your viva / presentation)"):
        st.markdown(
            """
This dataset labels only **1% of transactions as fraud**, and — importantly —
the fraud label in this particular sample dataset does **not correlate with
any of the available columns** (amount, location, merchant, transaction type,
or day of week all show essentially the same fraud rate). You can verify this
yourself in the *Fraud Analytics* tab: fraud rate barely moves across any
slice of the data.

In other words, this specific dataset behaves like fraud labels were assigned
**independently at random**, which is common in small demo/synthetic datasets
used for learning purposes. No model — however sophisticated — can predict
something that isn't actually encoded in the input features.

**This is a genuinely useful finding to report**, not a failure: it shows you
understand how to evaluate a model honestly instead of just reporting a
high accuracy number (which would be trivially high, ~99%, by predicting
"not fraud" every time — a classic pitfall with imbalanced data).

To get a model with real predictive power, you'd need richer signals such as:
- Transaction velocity (number of transactions per card in the last hour/day)
- Deviation from a customer's typical spending amount/location
- Device / IP fingerprint changes
- Time-since-last-transaction

If your other datasets (`financial_fraud_detection_dataset.csv`,
`synthetic_fraud_dataset1.csv`) contain these kinds of fields, swap them into
`src/train_model.py` — the pipeline is built to be dataset-agnostic as long as
you update `FEATURE_COLS`.
"""
        )

# --------------------------------------------------------------------------
# PAGE 4: Live Prediction
# --------------------------------------------------------------------------
elif page == "⚡ Live Prediction":
    st.title("Live Transaction Fraud Check")
    st.markdown("Enter transaction details to get a real-time fraud risk score.")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=500.0, step=10.0)
            txn_type = st.selectbox("Transaction Type", encoders["transaction_type"].classes_)
            location = st.selectbox("Location", encoders["location"].classes_)
        with c2:
            txn_date = st.date_input("Transaction Date")
            merchant_id = st.number_input(
                "Merchant ID (leave default if unknown)", min_value=0, value=0, step=1
            )
            model_pick = st.radio("Model", ["Random Forest", "Logistic Regression"], horizontal=True)

        submitted = st.form_submit_button("🔍 Check Transaction", use_container_width=True)

    if submitted:
        weekday = pd.Timestamp(txn_date).weekday()
        is_weekend = int(weekday >= 5)
        type_enc = encoders["transaction_type"].transform([txn_type])[0]
        loc_enc = encoders["location"].transform([location])[0]
        merchant_rate = encoders["merchant_fraud_rate_map"].get(
            merchant_id, encoders["global_fraud_rate"]
        )
        location_rate = encoders["location_fraud_rate_map"].get(
            location, encoders["global_fraud_rate"]
        )

        row = pd.DataFrame([{
            "Amount": amount,
            "TxnDay": txn_date.day,
            "TxnMonth": txn_date.month,
            "TxnWeekday": weekday,
            "IsWeekend": is_weekend,
            "TransactionType_enc": type_enc,
            "Location_enc": loc_enc,
            "MerchantFraudRate": merchant_rate,
            "LocationFraudRate": location_rate,
        }])[encoders["feature_cols"]]

        if model_pick == "Logistic Regression":
            row_input = scaler.transform(row)
            proba = lr_model.predict_proba(row_input)[0][1]
        else:
            proba = rf_model.predict_proba(row)[0][1]

        st.markdown("---")
        risk_pct = proba * 100
        if risk_pct >= 50:
            st.error(f"🚨 HIGH RISK — {risk_pct:.1f}% estimated fraud probability")
        elif risk_pct >= 20:
            st.warning(f"⚠️ MEDIUM RISK — {risk_pct:.1f}% estimated fraud probability")
        else:
            st.success(f"✅ LOW RISK — {risk_pct:.1f}% estimated fraud probability")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            title={"text": "Fraud Risk Score (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#111827"},
                "steps": [
                    {"range": [0, 20], "color": "#22c55e"},
                    {"range": [20, 50], "color": "#f59e0b"},
                    {"range": [50, 100], "color": "#ef4444"},
                ],
            },
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Note: as explained in the Model Performance tab, this demo dataset's "
            "fraud labels carry very little learnable signal, so treat this score "
            "as a pipeline demonstration rather than a reliable risk assessment."
        )
