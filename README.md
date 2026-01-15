# Credit Risk Analysis for Loan Approval
## 📌 Project Overview

This project builds an end-to-end Machine Learning pipeline to predict the credit risk of loan applicants.
The goal is to classify whether a borrower is likely to default (1) or fully repay (0) a loan based on their financial and credit history.

It follows industry-style ML workflow:
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Model persistence using .pkl files
- Reproducible environment using a virtual environment

## 🧠 Problem Statement

Financial institutions face major losses when loans default.
This project helps automate risk assessment by predicting loan default using historical lending data.

## 📂 Project Structure
credit-risk-analysis/<br>
│<br>
├── data/<br>
│   ├── raw/                # Original dataset<br>
│   └── processed/          # Cleaned and transformed dataset<br>
│<br>
├── src/<br>
│   ├── data_preprocessing.py<br>
│   ├── feature_engineering.py<br>
│   ├── train_models.py<br>
│   ├── evaluate_models.py<br>
│   ├── config.py<br>
│   └── __init__.py<br>
│<br>
├── models/                 # Saved ML models and pipelines<br>
│   ├── logistic_regression.pkl<br>
│   ├── random_forest.pkl<br>
│   ├── preprocessor.pkl<br>
│   └── selected_features.pkl<br>
│<br>
├── results/<br>
│   ├── model_comparison.csv<br>
│   └── final_model_comparison.csv<br>
│<br>
├── .venv/                  # Virtual environment<br>
├── requirements.txt<br>
└── README.md<br>

## ⚙️ Environment Setup

- Activate virtual environment:
.\.venv\Scripts\Activate.ps1

- Install dependencies:
python -m pip install -r requirements.txt

## 🔄 Project Workflow
1️⃣ Data Preprocessing

Run:
python -m src.data_preprocessing

Handles:
- Target creation (loan_status → binary)
- Data leakage removal
- Feature cleaning
- Outlier handling
- Missing value handling
- Feature transformations (credit age, term, employment length, etc.)

2️⃣ Model Training

python -m src.train_models

Trains:
- Logistic Regression
- Random Forest
- Saves:
- Trained models (.pkl)
- Preprocessing pipeline
- Selected feature list
- Training results

3️⃣ Model Evaluation

python -m src.evaluate_models

Generates:
- ROC curves
- Confusion matrices
- Classification reports
- Final model comparison CSV

🧪 Feature Engineering

Includes:
- Numerical scaling using StandardScaler
- Categorical encoding using OneHotEncoder
- Ordinal encoding for sub_grade
- Missing value imputation using SimpleImputer
- Outlier handling using:
- Clipping
- Log transformation
- IQR capping

📊 Models Used

Model	Description
- Logistic Regression : Baseline interpretable classifier
- Random Forest	: High performance ensemble model

📈 Evaluation Metrics
- ROC-AUC
- Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve visualization

📌 Key Learning Outcomes

- Built production-style ML pipeline
- Handled real-world dataset challenges:
- Missing values
- Data leakage
- High dimensionality
- Class imbalance
- Learned Python packaging structure
- Used model persistence with joblib
- Designed reproducible environment

🏆 Future Improvements

- Add XGBoost / LightGBM
- Hyperparameter tuning
- SHAP explainability
- API deployment using Flask/FastAPI

👩‍💻 Author

Harshita Saraogi<br>
MSc Data Science & Analytics