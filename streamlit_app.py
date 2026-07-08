# Simple Streamlit dashboard for Fraud-Shield ML project
# Usage: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Local helpers
try:
    from model_utils import (
        preprocess_dataframe, apply_smote, train_and_evaluate, plot_confusion, plot_roc, feature_importance_df
    )
except Exception as e:
    st.warning("model_utils module not found or import error. Make sure model_utils.py exists in the same folder.")
    raise

st.set_page_config(page_title="Credit Shield - Fraud Dashboard", layout="wide")

st.title("Credit-Shield-ML — Fraud Detection Dashboard")
st.markdown(
    "Upload a transactions CSV (target column usually named `Class` or specify the target). "
    "This app will preprocess the data, optionally apply SMOTE, train models and show evaluation metrics."
)

# Sidebar options
st.sidebar.header("Settings")
use_smote = st.sidebar.checkbox("Apply SMOTE to training set (if class-imbalance)", value=True)
test_size = st.sidebar.slider("Test set size (fraction)", 0.1, 0.5, 0.25)
random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

uploaded = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded is None:
    st.info("No file uploaded yet. You can upload a CSV file containing transaction features and a target column.")
    st.stop()

# Read CSV
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error("Failed to read CSV: " + str(e))
    st.stop()

st.subheader("Data preview")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.dataframe(df.head(300))

# Target selection
possible_targets = list(df.columns)
target = st.selectbox("Select target column", options=possible_targets, index=(possible_targets.index("Class") if "Class" in possible_targets else 0))

# Quick class-balance
if df[target].nunique() <= 50 and df[target].dtype != object:
    st.subheader("Class balance")
    fig, ax = plt.subplots()
    sns.countplot(x=df[target], ax=ax)
    ax.set_title("Target class counts")
    st.pyplot(fig)

# Preprocess (returns X, y and feature_names used)
with st.spinner("Preprocessing data..."):
    X, y, feature_names, preprocessor = preprocess_dataframe(df, target)

st.markdown(f"Features after preprocessing: {len(feature_names)}")
st.write(feature_names[:50])

# Train / evaluate
if st.button("Train & Evaluate Models"):
    st.info("Training models. This may take a minute for larger datasets.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=y if len(np.unique(y))>1 else None)

    if use_smote:
        try:
            X_train, y_train = apply_smote(X_train, y_train, random_state=int(random_state))
            st.success("SMOTE applied to training set.")
        except Exception as e:
            st.warning("SMOTE failed or is not available: " + str(e))

    models, results = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        feature_names=feature_names,
        random_state=int(random_state)
    )

    st.subheader("Evaluation summary")
    for name, res in results.items():
        st.markdown(f"### {name}")
        st.write(res["report"])
        st.metric("ROC AUC", f"{res['roc_auc']:.4f}")
        st.metric("F1 Score", f"{res['f1']:.4f}")

    # ROC plots
    st.subheader("ROC curves")
    fig_roc, ax = plt.subplots()
    for name, res in results.items():
        fpr, tpr, _ = res["roc_curve"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")
    ax.plot([0,1],[0,1],"k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig_roc)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    for name, res in results.items():
        st.write(f"**{name}**")
        fig_cm = plot_confusion(res["y_true"], res["y_pred"], labels=np.unique(y))
        st.pyplot(fig_cm)

    # Feature importance (RandomForest)
    if "RandomForest" in models:
        st.subheader("Feature importance (Random Forest)")
        fi = feature_importance_df(models["RandomForest"], feature_names)
        st.dataframe(fi.head(50))
        fig_fi, ax = plt.subplots(figsize=(6,8))
        sns.barplot(x="importance", y="feature", data=fi.head(20), ax=ax)
        ax.set_title("Top 20 features")
        st.pyplot(fig_fi)

    # Save models option
    if st.checkbox("Save trained RandomForest model to file"):
        try:
            joblib.dump(models.get("RandomForest"), "models/random_forest_model.joblib")
            st.success("Saved as models/random_forest_model.joblib")
        except Exception as e:
            st.error("Save failed: " + str(e))

# Single transaction prediction form
st.sidebar.header("Single transaction prediction")
if st.sidebar.button("Open prediction form"):
    st.sidebar.info("Enter values for features to get a model prediction.")
    # Use median values to prefill
    median_vals = np.median(X, axis=0)
    input_vals = {}
    for i, fname in enumerate(feature_names):
        # Limit number of widgets shown to 30 by default to avoid performance problems
        if i >= 40:
            break
        val = st.sidebar.number_input(fname, value=float(median_vals[i]))
        input_vals[fname] = val

    if st.sidebar.button("Predict (RandomForest)"):
        # Load model: prefer saved file, else trained in session (not persisted across runs)
        try:
            model = joblib.load("models/random_forest_model.joblib")
        except:
            st.sidebar.error("No saved model found. Train and save a model first.")
            model = None
        if model is not None:
            x_arr = np.array([list(input_vals.values())])
            pred_proba = model.predict_proba(x_arr)[:,1][0] if hasattr(model, "predict_proba") else None
            pred = model.predict(x_arr)[0]
            st.sidebar.write(f"Predicted class: {pred}")
            if pred_proba is not None:
                st.sidebar.write(f"Predicted fraud probability: {pred_proba:.4f}")

st.markdown("---")
st.caption("If you run into issues, check package versions (see requirements.txt).")
