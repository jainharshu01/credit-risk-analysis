# Credit Risk Analysis for Loan Approval
## Project Overview

This project implements an end-to-end Machine Learning pipeline to assess credit risk. The system classifies loan applicants as likely to default (1) or fully repay (0) by analyzing financial history, employment data, and loan characteristics.

Through extensive experimentation with Optuna hyperparameter tuning and Ensemble strategies, this project identifies the most robust predictors of financial default while maintaining industry-standard metrics like the Gini Coefficient.

It follows the complete ML workflow:
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Model persistence using .pkl files

## Problem Statement

Financial institutions face major losses when loans default.
Defaults lead to significant financial loss for lending institutions. This project automates the risk assessment process by predicting loan default using historical lending data, providing a data-driven approach to identify high-risk borrowers while maintaining model interpretability.


## Project Structure<br>
CREDIT-RISK-ANALYSIS/<br>
├── .venv/                        # Virtual environment<br>
├── data/<br>
│   ├── processed/<br>
│   │   └── cleaned_data.csv      # Final preprocessed dataset<br>
│   └── raw/<br>
│       └── credit_data.csv       # Original dataset<br>
├── models/                       # Saved ML assets (.pkl)<br>
│   ├── dynamic_ensemble_probs.pkl<br>
│   ├── lightgbm.pkl<br>
│   ├── logistic_regression.pkl<br>
│   ├── naive_bayes.pkl<br>
│   ├── preprocessor.pkl<br>
│   ├── random_forest.pkl<br>
│   ├── selected_features.pkl<br>
│   └── xgboost.pkl<br>
├── notebooks/<br>
│   └── 01_exploration.ipynb      # Initial EDA and experimentation<br>
├── results/                      # Evaluation outputs<br>
│   ├── cm_lightgbm.png           # Confusion Matrix plots<br>
│   ├── cm_logistic_regression.png<br>
│   ├── cm_naive_bayes.png<br>
│   ├── cm_random_forest.png<br>
│   ├── cm_xgboost.png<br>
│   ├── final_ensemble_comparison.csv # Head-to-head comparison<br>
│   └── model_performance_summary.csv # Final leaderboard<br>
├── src/                          # Modular source code<br>
│   ├── config.py<br>
│   ├── data_preprocessing.py<br>
│   ├── ensemble_model.py<br>
│   ├── evaluate_models.py<br>
│   ├── feature_engineering.py<br>
│   └── train_models.py<br>
├── .gitignore<br>
├── README.md<br>
└── requirements.txt<br>
<br>

## Project Workflow<br>
1️⃣ Preprocessing & Feature Engineering: <br>
The pipeline handles class imbalance and messy financial data through:
- Outlier Management: A hybrid approach using Clipping, Log Transformation, and IQR Capping based on feature distribution.
- Target Engineering: Converting raw loan status into binary classifications and removing data leakage.- - Automated Selection: Using a SelectKBest/Feature Mask approach to retain only high-impact predictors.
<br>

2️⃣ Model Training & Evaluation:<br>
We trained and compared five distinct algorithms:
- LightGBM & XGBoost: High-efficiency gradient boosting.
- Random Forest: Robust ensemble of trees.
- Logistic Regression: Linear baseline for interpretability.
- Naive Bayes: Probabilistic baseline.
<br>

3️⃣ Advanced Metrics (Gini & Weighted Stats)<br>
Because credit data is often imbalanced, we focused on:
- Weighted F1/Precision/Recall: Scoring that accounts for the relative frequency of each class.
- Gini Coefficient: Derived as 2 * AUC - 1, a standard metric in credit scoring to measure the model's discriminatory power.
<br>

4️⃣ Dynamic Ensemble: <br>
Duo Ensemble: An 80/20 weighted split between XGBoost and Logistic Regression to combine non-linear capture with linear stability.

Trio Ensemble: A proportional weighting of the Top 3 models based on their ROC-AUC scores.
<br> 

## Final Performance Results: 

| Rank | Model | ROC-AUC | Gini Score |
| :--- | :--- | :---: | :---: |
| **1st** | **XGBoost (Optuna Tuned)** | **0.7644** | **0.5288** |
| 2nd | LightGBM (Optuna Tuned) | 0.7641 | 0.5282 |
| 3rd | Duo Ensemble (XGB + LogReg) | 0.7637 | 0.5274 |
| 4th | Trio Ensemble (Top 3 Weighted) | 0.7626 | 0.5272 |
| 5th | Logistic Regression | 0.7545 | 0.5090 |

> **Key Finding:** While the ensemble strategies provided high stability, the individual **XGBoost** model (after Optuna hyperparameter optimization) achieved the highest discriminatory power with a Gini score of **0.5288**. This suggests that for this specific feature space, a finely-tuned non-linear model captures the risk signal more effectively than a weighted average of diverse models.
> **Key Finding:** During the final evaluation, the individual **LightGBM** model slightly outperformed the weighted ensemble. This is likely due to the high correlation between the top tree-based models. For this reason, LightGBM was selected as the final production estimator for its superior performance and lower architectural complexity.<br>

## **Feature Importance & Diagnostics**
The model identified Debt Settlement Status, Loan Term, and Interest Rates as the most critical indicators of default risk.

- Debt Settlement Flags: By far the strongest predictor of future risk.
- Loan Term: Longer-term loans (60 months) showed a higher correlation with default.
- Interest Rates & Grades: Directly reflected the lender's initial risk assessment and proved highly accurate.

## Key Learning Outcomes:
- Pipeline Modularity: Decoupled preprocessing, training, and evaluation for easier debugging.
- Credit-Specific Metrics: Implemented Gini coefficients to align with financial industry standards.
- Outlier Strategy: Learned that different features require different handling (clipping vs. log) to maintain signal.
- Model Realism: Observed that more complex ensembles don't always beat a single, well-optimized model in a highly correlated feature space.


## Author

Harshita Saraogi<br>
MSc Data Science & Analytics
