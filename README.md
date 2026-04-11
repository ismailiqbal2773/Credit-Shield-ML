- #  Credit-Shield-ML
**Defeating the 0.17% Challenge in Credit Card Fraud Detection**

##  Project Overview
In the financial sector, detecting credit card fraud is like finding a needle in a haystack. With fraudulent transactions often making up less than **0.2%** of total data, traditional machine learning models easily fail by predicting all transactions as "Safe." 

**Credit-Shield-ML** tackles this extreme class imbalance challenge head-on. Built on a dataset of European cardholders (transformed via PCA for privacy), this project moves beyond "Vanity Metrics" like Accuracy to focus on what truly matters in FinTech: **Precision, Recall, and catching the actual fraudsters.**

##  Key Technical Highlights

- **Precision-Recall Mastery (AUPRC):** Achieved an outstanding **AUPRC score of 0.9997**. In fraud detection, Accuracy is misleading; maximizing the Area Under the Precision-Recall Curve ensures high detection rates while minimizing False Positives.
- **Handling Extreme Imbalance:** Since only 0.17% of the dataset represents fraud, I implemented **SMOTE (Synthetic Minority Over-sampling Technique)** to synthetically balance the training data, allowing the model to learn the minority class patterns effectively.
- **Strategic Feature Scaling:** Applied **StandardScaler** to input features (like 'Time' and 'Amount') while preserving the categorical target label. This ensured stable and fast model convergence without being biased by high-value transactions.
- **Robust Modeling:** Trained and evaluated using a **Random Forest Classifier**, optimizing for F1-Score and Confusion Matrix metrics to ensure real-world business viability.

##  The Business Context
According to the Nilson Report, global card fraud losses are projected to hit **$49 Billion by 2030**. A model that fails to catch fraud (False Negatives) costs money, while a model that blocks legitimate customers (False Positives) costs trust. This project is optimized to balance that critical trade-off.

##  Dataset Information
The dataset is sourced from the ULB (Université Libre de Bruxelles) Machine Learning Group.
- **Dataset Link:** [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Features:** Includes 28 PCA-transformed variables (V1 to V28), 'Time', 'Amount', and the 'Class' label (1 for fraud, 0 for legitimate).

##  Technical Toolkit
- **Language:** Python
- **Machine Learning:** Scikit-Learn, Random Forest
- **Imbalance Handling:** Imbalanced-learn (SMOTE)
- **Data Processing:** Pandas, NumPy, StandardScaler
- **Visualization:** Seaborn, Matplotlib

##  Project Structure
- `FraudDetection_Project.ipynb`: The core notebook containing exploratory data analysis (EDA), data preprocessing, SMOTE implementation, and model evaluation.
- `README.md`: Project documentation and insights.

---
*Built with a focus on Data-Centric AI and robust financial security.*
