#  Credit-Shield-ML

## Project Overview
This project focuses on identifying fraudulent transactions in a highly imbalanced dataset. The data contains transactions made by European cardholders and has been transformed using **PCA (Principal Component Analysis)** to ensure privacy. This model effectively handles the challenge of finding a "needle in a haystack"—detecting rare fraud cases among thousands of legitimate transactions.

## Key Features
- **Handling Extreme Imbalance:** Since only 0.17% of transactions are fraudulent, I implemented **SMOTE** to balance the training data.
- **PCA Data Expertise:** Worked with anonymized features (V1 to V28) while maintaining high predictive power.
- **Robust Evaluation:** Used Confusion Matrix and F1-Score instead of just Accuracy, as accuracy can be misleading in imbalanced datasets.
- **High Performance:** Achieved exceptional results using the **Random Forest Classifier**.

## Dataset Information
The dataset is sourced from the ULB (Université Libre de Bruxelles) Machine Learning Group.
- **Dataset Link:** [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Features:** Includes 28 PCA-transformed variables, 'Time', 'Amount', and the 'Class' label (1 for fraud, 0 for legitimate).

## Technical Toolkit
- **Language:** Python
- **Machine Learning:** Scikit-Learn
- **Handling Imbalance:** Imbalanced-learn (SMOTE)
- **Visualization:** Seaborn, Matplotlib

## Project Structure
- `FraudDetection_Project.ipynb`: End-to-end analysis, data scaling, and modeling.
- `README.md`: Project highlights and documentation.
