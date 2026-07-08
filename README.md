# Credit-Shield-ML

An end-to-end Machine Learning pipeline to detect fraudulent transactions using advanced classification algorithms.

This branch adds a simple Streamlit dashboard to interactively train and evaluate models.

Files added:
- streamlit_app.py — Simple Streamlit app (upload CSV, preprocess, train RandomForest & LogisticRegression, evaluate)
- model_utils.py — Helper functions used by the app
- requirements.txt — Python dependencies

Quick start (local):
1. Create a virtual environment and install requirements:
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
2. Run the app:
   streamlit run streamlit_app.py

If your dataset's target column is named `Class` the app will auto-select it; otherwise choose the correct target from the dropdown.
