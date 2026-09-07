"""
Financial Fraud Detection - Model Training Script
---------------------------------------------------
Loads the raw transaction data, engineers features, trains two
classifiers (Logistic Regression + Random Forest) with class-imbalance
handling, evaluates them, and saves everything the Streamlit dashboard
needs to run without retraining every time.

Run this once locally:
    python src/train_model.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_STATE = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_card_fraud_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def engineer_base_features(df: pd.DataFrame):
    """Turn raw columns into date/categorical features that don't
    depend on the label (safe to compute on the whole dataset).
    """
    df = df.copy()

    # --- Dates -> behavioral / seasonal signals ---
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], format="%d-%m-%Y")
    df["TxnDay"] = df["TransactionDate"].dt.day
    df["TxnMonth"] = df["TransactionDate"].dt.month
    df["TxnWeekday"] = df["TransactionDate"].dt.weekday  # 0=Mon
    df["IsWeekend"] = (df["TxnWeekday"] >= 5).astype(int)

    # --- Encode categoricals ---
    le_type = LabelEncoder()
    df["TransactionType_enc"] = le_type.fit_transform(df["TransactionType"])

    le_loc = LabelEncoder()
    df["Location_enc"] = le_loc.fit_transform(df["Location"])

    encoders = {"transaction_type": le_type, "location": le_loc}
    return df, encoders


FEATURE_COLS = [
    "Amount",
    "TxnDay",
    "TxnMonth",
    "TxnWeekday",
    "IsWeekend",
    "TransactionType_enc",
    "Location_enc",
    "MerchantFraudRate",
    "LocationFraudRate",
]


def add_target_encoded_features(train_df, other_df, global_rate):
    """Compute merchant/location historical fraud rates from train_df
    ONLY, then map them onto other_df (which may be train or test).
    Unseen merchants/locations fall back to the global fraud rate.
    This avoids leaking test-set labels into the features.
    """
    merchant_map = train_df.groupby("MerchantID")["IsFraud"].mean().to_dict()
    location_map = train_df.groupby("Location")["IsFraud"].mean().to_dict()

    other_df = other_df.copy()
    other_df["MerchantFraudRate"] = other_df["MerchantID"].map(merchant_map).fillna(global_rate)
    other_df["LocationFraudRate"] = other_df["Location"].map(location_map).fillna(global_rate)
    return other_df, merchant_map, location_map


def train_and_evaluate(df_base):
    global_rate = float(df_base["IsFraud"].mean())

    train_df, test_df = train_test_split(
        df_base, test_size=0.25, random_state=RANDOM_STATE, stratify=df_base["IsFraud"]
    )

    train_df, merchant_map, location_map = add_target_encoded_features(
        train_df, train_df, global_rate
    )
    test_df, _, _ = add_target_encoded_features(train_df, test_df, global_rate)

    X_train, y_train = train_df[FEATURE_COLS], train_df["IsFraud"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["IsFraud"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)

        # Downsample curve points (dashboard only needs a smooth line, not
        # every threshold) to keep metrics.json small.
        def _downsample(a, n=200):
            a = np.asarray(a)
            if len(a) <= n:
                return a.tolist()
            idx = np.linspace(0, len(a) - 1, n).astype(int)
            return a[idx].tolist()

        results[name] = {
            "report": report,
            "confusion_matrix": cm,
            "roc_auc": auc,
            "roc_curve": {"fpr": _downsample(fpr), "tpr": _downsample(tpr)},
            "pr_curve": {"precision": _downsample(prec), "recall": _downsample(rec)},
        }
        fitted_models[name] = model

    return fitted_models, scaler, results, merchant_map, location_map, global_rate


def main():
    print("Loading data...")
    df_raw = load_data()

    print("Engineering base features...")
    df_base, cat_encoders = engineer_base_features(df_raw)

    print("Training models (with leakage-safe target encoding)...")
    (
        fitted_models,
        scaler,
        results,
        merchant_map,
        location_map,
        global_rate,
    ) = train_and_evaluate(df_base)

    # Pick Random Forest as the "production" model used for live scoring
    # (generally handles the non-linear merchant/location signals better).
    best_model_name = "Random Forest"

    encoders = {
        "transaction_type": cat_encoders["transaction_type"],
        "location": cat_encoders["location"],
        "feature_cols": FEATURE_COLS,
        "merchant_fraud_rate_map": merchant_map,
        "location_fraud_rate_map": location_map,
        "global_fraud_rate": global_rate,
    }

    print("Saving artifacts...")
    joblib.dump(fitted_models["Random Forest"], os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(fitted_models["Logistic Regression"], os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(
            {"results": results, "best_model": best_model_name},
            f,
            indent=2,
        )

    # Save a full-dataset feature table (using train-fitted maps, applied
    # to everyone) purely for the dashboard's exploratory charts.
    df_display, _, _ = add_target_encoded_features(df_base, df_base, global_rate)
    # Re-fit merchant/location maps on the FULL data just for chart display
    # (this is fine here since it's not used for model evaluation).
    df_display["MerchantFraudRate"] = df_base["MerchantID"].map(
        df_base.groupby("MerchantID")["IsFraud"].mean()
    )
    df_display["LocationFraudRate"] = df_base["Location"].map(
        df_base.groupby("Location")["IsFraud"].mean()
    )
    df_display.to_csv(os.path.join(MODELS_DIR, "engineered_data.csv"), index=False)

    print("Done. Artifacts saved to /models")
    for name, res in results.items():
        print(f"\n{name}: ROC-AUC = {res['roc_auc']:.4f}")
        print(f"  Precision (fraud): {res['report']['1']['precision']:.3f}")
        print(f"  Recall (fraud):    {res['report']['1']['recall']:.3f}")
        print(f"  F1 (fraud):        {res['report']['1']['f1-score']:.3f}")


if __name__ == "__main__":
    main()
