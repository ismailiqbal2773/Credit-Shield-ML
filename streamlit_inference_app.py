# Streamlit Dashboard for Pre-trained Model Inference
# Usage: streamlit run streamlit_inference_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

st.set_page_config(page_title="Credit Shield - Fraud Detection (Inference)", layout="wide")

st.title("🛡️ Credit-Shield-ML — Fraud Detection (Pre-trained Model)")
st.markdown(
    "This dashboard uses a **pre-trained machine learning model** to detect fraudulent transactions. "
    "Upload your CSV data and get instant fraud predictions!"
)

# Sidebar - Model Loading
st.sidebar.header("⚙️ Model Settings")

# Try to load pre-trained model
model_path = st.sidebar.text_input("Model path (joblib)", value="models/random_forest_model.joblib")

try:
    model = joblib.load(model_path)
    st.sidebar.success(f"✅ Model loaded from: {model_path}")
except Exception as e:
    st.sidebar.error(f"❌ Could not load model: {str(e)}")
    st.sidebar.info("Make sure you have a trained model saved at the specified path.")
    st.stop()

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Batch Predictions", "🔮 Single Transaction", "📈 Model Info"])

# ==================== TAB 1: Batch Predictions ====================
with tab1:
    st.subheader("Batch Fraud Detection")
    st.markdown("Upload a CSV file with transaction features to get predictions for all records.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="batch_upload")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(10))
            
            # Target column selection (optional, for evaluation)
            has_target = st.checkbox("Does this dataset have a target column (for evaluation)?")
            
            if has_target:
                target_col = st.selectbox(
                    "Select target column",
                    options=df.columns.tolist(),
                    index=df.columns.tolist().index("Class") if "Class" in df.columns else 0
                )
                y_true = df[target_col].values
                X = df.drop(columns=[target_col])
            else:
                X = df
                y_true = None
            
            # Make predictions
            if st.button("🎯 Generate Predictions", key="predict_batch"):
                try:
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
                    
                    # Create results dataframe
                    results_df = df.copy()
                    results_df["Predicted_Class"] = predictions
                    if probabilities is not None:
                        results_df["Fraud_Probability"] = probabilities
                    
                    # Display results
                    st.success("✅ Predictions completed!")
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fraud_count = (predictions == 1).sum()
                        st.metric("🚨 Fraudulent Transactions", fraud_count)
                    with col2:
                        legitimate_count = (predictions == 0).sum()
                        st.metric("✅ Legitimate Transactions", legitimate_count)
                    with col3:
                        fraud_percentage = (fraud_count / len(predictions)) * 100
                        st.metric("📊 Fraud Rate", f"{fraud_percentage:.2f}%")
                    
                    # Show predictions
                    st.subheader("Predictions Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Evaluation metrics (if target is available)
                    if y_true is not None and len(np.unique(y_true)) > 1:
                        st.subheader("📊 Evaluation Metrics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            acc = accuracy_score(y_true, predictions)
                            st.metric("Accuracy", f"{acc:.4f}")
                        with col2:
                            prec = precision_score(y_true, predictions, zero_division=0)
                            st.metric("Precision", f"{prec:.4f}")
                        with col3:
                            rec = recall_score(y_true, predictions, zero_division=0)
                            st.metric("Recall", f"{rec:.4f}")
                        with col4:
                            f1 = f1_score(y_true, predictions, zero_division=0)
                            st.metric("F1-Score", f"{f1:.4f}")
                        
                        # ROC-AUC
                        if probabilities is not None:
                            try:
                                roc_auc = roc_auc_score(y_true, probabilities)
                                st.metric("ROC-AUC", f"{roc_auc:.4f}")
                            except:
                                pass
                        
                        # Confusion Matrix
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_true, predictions)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                                    xticklabels=["Legitimate", "Fraudulent"],
                                    yticklabels=["Legitimate", "Fraudulent"])
                        ax.set_ylabel("Actual")
                        ax.set_xlabel("Predicted")
                        st.pyplot(fig)
                        
                        # ROC Curve
                        if probabilities is not None:
                            st.subheader("ROC Curve")
                            fpr, tpr, _ = roc_curve(y_true, probabilities)
                            fig, ax = plt.subplots(figsize=(6, 5))
                            ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})", linewidth=2)
                            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
                            ax.set_xlabel("False Positive Rate")
                            ax.set_ylabel("True Positive Rate")
                            ax.legend()
                            ax.grid(alpha=0.3)
                            st.pyplot(fig)
                        
                        # Classification Report
                        st.subheader("Classification Report")
                        report = classification_report(y_true, predictions, 
                                                      target_names=["Legitimate", "Fraudulent"],
                                                      output_dict=False)
                        st.text(report)
                    
                    # Prediction distribution
                    st.subheader("Prediction Distribution")
                    fig, ax = plt.subplots()
                    pred_dist = pd.Series(predictions).value_counts()
                    ax.bar(["Legitimate", "Fraudulent"], [pred_dist.get(0, 0), pred_dist.get(1, 0)])
                    ax.set_ylabel("Count")
                    ax.set_title("Distribution of Predictions")
                    st.pyplot(fig)
                    
                    # Probability distribution (if available)
                    if probabilities is not None:
                        st.subheader("Fraud Probability Distribution")
                        fig, ax = plt.subplots()
                        ax.hist(probabilities, bins=50, edgecolor="black", alpha=0.7)
                        ax.set_xlabel("Fraud Probability")
                        ax.set_ylabel("Frequency")
                        ax.set_title("Distribution of Fraud Probabilities")
                        ax.axvline(x=0.5, color="red", linestyle="--", label="Decision Threshold")
                        ax.legend()
                        st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ File reading error: {str(e)}")


# ==================== TAB 2: Single Transaction ====================
with tab2:
    st.subheader("🔮 Single Transaction Prediction")
    st.markdown("Enter transaction details to check if it's likely fraudulent.")
    
    # Sample feature template (adjust based on your model's features)
    st.info("⚠️ Note: Adjust the number and names of features based on your trained model.")
    
    num_features = st.number_input("Number of features", min_value=1, max_value=50, value=10)
    
    feature_values = []
    col_list = st.columns(2)
    
    for i in range(num_features):
        feature_name = st.text_input(f"Feature {i+1} name", value=f"Feature_{i+1}", key=f"fname_{i}")
        feature_val = st.number_input(f"Enter value for {feature_name}", value=0.0, key=f"fval_{i}")
        feature_values.append(feature_val)
    
    if st.button("🎯 Predict", key="predict_single"):
        try:
            # Prepare input
            X_single = np.array([feature_values])
            
            # Make prediction
            pred_class = model.predict(X_single)[0]
            pred_proba = model.predict_proba(X_single)[0] if hasattr(model, "predict_proba") else None
            
            # Display result
            st.subheader("Prediction Result")
            
            if pred_class == 0:
                st.success("✅ This transaction appears to be **LEGITIMATE**")
                if pred_proba is not None:
                    st.write(f"Confidence: {pred_proba[0]:.2%}")
            else:
                st.error("🚨 This transaction appears to be **FRAUDULENT**")
                if pred_proba is not None:
                    st.write(f"Fraud Probability: {pred_proba[1]:.2%}")
            
            # Show probability breakdown
            if pred_proba is not None:
                st.subheader("Prediction Probabilities")
                prob_df = pd.DataFrame({
                    "Class": ["Legitimate", "Fraudulent"],
                    "Probability": pred_proba
                })
                st.dataframe(prob_df)
                
                # Probability visualization
                fig, ax = plt.subplots()
                ax.bar(["Legitimate", "Fraudulent"], pred_proba, color=["green", "red"], alpha=0.7)
                ax.set_ylabel("Probability")
                ax.set_title("Prediction Probabilities")
                ax.set_ylim([0, 1])
                st.pyplot(fig)
        
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")


# ==================== TAB 3: Model Info ====================
with tab3:
    st.subheader("📈 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Type:**")
        st.write(type(model).__name__)
        
        if hasattr(model, "n_estimators"):
            st.write(f"**Number of Estimators:** {model.n_estimators}")
        
        if hasattr(model, "max_depth"):
            st.write(f"**Max Depth:** {model.max_depth}")
    
    with col2:
        st.write("**Model Capabilities:**")
        st.write(f"- Can predict: ✅")
        st.write(f"- Can predict probabilities: {'✅' if hasattr(model, 'predict_proba') else '❌'}")
        st.write(f"- Has feature importance: {'✅' if hasattr(model, 'feature_importances_') else '❌'}")
    
    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance (Top 20)")
        
        importances = model.feature_importances_
        feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(20)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
        ax.set_title("Top 20 Most Important Features")
        st.pyplot(fig)
        
        st.dataframe(imp_df)

st.markdown("---")
st.caption("💡 Tip: For best results, ensure your input data matches the format used to train the model.")
