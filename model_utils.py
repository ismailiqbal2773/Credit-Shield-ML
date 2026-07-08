# Helper utilities for Streamlit dashboard

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_dataframe(df: pd.DataFrame, target_column: str):
    """
    Basic preprocessing:
    - Drop rows with missing target
    - Separate X, y
    - Impute numeric columns with median
    - One-hot encode categorical columns
    - Scale numeric features
    Returns: X (numpy array), y (numpy array), feature_names (list), preprocessor object
    """
    df = df.copy()
    df = df[df[target_column].notna()]
    y = df[target_column].values
    X_df = df.drop(columns=[target_column])

    # Identify numeric and categorical
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Imputer & scaler for numeric
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # For categorical, simple get_dummies at dataframe level
    X_num = pd.DataFrame(numeric_pipeline.fit_transform(X_df[numeric_cols]), columns=numeric_cols) if numeric_cols else pd.DataFrame()
    X_cat = pd.get_dummies(X_df[cat_cols], drop_first=True) if cat_cols else pd.DataFrame()

    if not X_num.empty and not X_cat.empty:
        X_processed = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    elif not X_num.empty:
        X_processed = X_num
    else:
        X_processed = X_cat

    feature_names = X_processed.columns.tolist()
    return X_processed.values, y, feature_names, (numeric_pipeline, cat_cols)

def apply_smote(X_train, y_train, random_state=42):
    try:
        from imblearn.over_sampling import SMOTE
    except Exception as e:
        raise ImportError("imbalanced-learn is required for SMOTE. Install via `pip install imbalanced-learn`") from e
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names=None, random_state=42):
    results = {}
    models = {}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:,1] if hasattr(rf, "predict_proba") else None

    results["RandomForest"] = _build_result_dict(y_test, y_pred_rf, y_proba_rf)
    models["RandomForest"] = rf

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, solver="liblinear")
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:,1] if hasattr(lr, "predict_proba") else None

    results["LogisticRegression"] = _build_result_dict(y_test, y_pred_lr, y_proba_lr)
    models["LogisticRegression"] = lr

    return models, results

def _build_result_dict(y_true, y_pred, y_proba):
    r = {
        "y_true": y_true,
        "y_pred": y_pred,
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None and len(set(y_true))>1 else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "report": classification_report(y_true, y_pred, zero_division=0, output_dict=False),
    }
    if y_proba is not None and len(set(y_true))>1:
        r["roc_curve"] = roc_curve(y_true, y_proba)
    else:
        r["roc_curve"] = ([],[],[])
    return r

def plot_confusion(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def plot_roc(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1],"k--", alpha=0.5)
    return fig

def feature_importance_df(model, feature_names):
    try:
        import pandas as pd
        import numpy as np
    except:
        raise
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    else:
        return pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names))})
