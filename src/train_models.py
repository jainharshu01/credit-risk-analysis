import os
import pandas as pd
import numpy as np
import joblib
import optuna
from pathlib import Path

# Scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_auc_score

# External Boosting Libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Custom modules
from src.feature_engineering import build_feature_pipeline
from src.data_preprocessing import handle_outliers
from src.config import CLIP_COLS, LOG_COLS, IQR_COLS

def load_processed_data():
    return pd.read_csv("data/processed/cleaned_data.csv")

def split_data(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def get_feature_types(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    return numeric_features, categorical_features

def run_optuna_study(model_name, X_train, y_train, cv, scale_pos_weight):
    """Defines search spaces and runs optimization for each model."""
    
    def objective(trial):
        if model_name == "logistic_regression":
            params = {
                "C": trial.suggest_float("C", 1e-3, 100, log=True),
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 3000,
                "class_weight": "balanced",
                "random_state": 42
            }
            model = LogisticRegression(**params)
            
        elif model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "scale_pos_weight": scale_pos_weight,
                "tree_method": "hist",
                "device": "cuda",
                "random_state": 42
            }
            model = XGBClassifier(**params)

        elif model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "class_weight": "balanced",
                "device": "gpu",
                "random_state": 42,
                "verbose": -1
            }
            model = LGBMClassifier(**params)

        elif model_name == "random_forest":
            # Using XGB's RF implementation for GPU speed
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 15),
                "subsample": trial.suggest_float("subsample", 0.7, 0.9),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.7, 0.9),
                "learning_rate": 1, # Required for RF behavior in XGB
                "tree_method": "hist",
                "device": "cuda",
                "random_state": 42
            }
            model = XGBClassifier(**params)

        # Cross-validation score
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_cv_train, y_cv_train)
            probs = model.predict_proba(X_cv_val)[:, 1]
            scores.append(roc_auc_score(y_cv_val, probs))
        
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10) # Increase to 30+ for final runs
    return study.best_params

def train_models():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Loading and Preprocessing data...")
    df = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(df)

    X_train = handle_outliers(X_train, CLIP_COLS, LOG_COLS, IQR_COLS)
    X_test = handle_outliers(X_test, CLIP_COLS, LOG_COLS, IQR_COLS)

    numeric_features, categorical_features = get_feature_types(X_train)
    preprocessor = build_feature_pipeline(numeric_features, categorical_features, ['sub_grade'])

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Feature Selection
    print("Selecting features...")
    lasso = LogisticRegression(solver="liblinear", penalty="l1", C=0.05, class_weight="balanced", random_state=42)
    lasso.fit(X_train_transformed, y_train)
    selected_mask = np.abs(lasso.coef_[0]) > 0
    
    X_train_selected = X_train_transformed[:, selected_mask]
    X_test_selected = X_test_transformed[:, selected_mask]
    # CHECK: Convert to array only if it is a sparse matrix
    if hasattr(X_train_selected, "toarray"):
        X_train_selected = X_train_selected.toarray()
    if hasattr(X_test_selected, "toarray"):
        X_test_selected = X_test_selected.toarray()
    joblib.dump(selected_mask, "models/selected_features.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    neg_count, pos_count = np.bincount(y_train)
    scale_pos_weight = neg_count / pos_count
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    model_names = ["logistic_regression", "xgboost", "lightgbm", "random_forest"]
    results = []

    for name in model_names:
        print(f"\n>>> Optimizing {name} with Optuna...")
        best_params = run_optuna_study(name, X_train_selected, y_train, cv, scale_pos_weight)
        
        # Retrain with best params
        print(f"Finalizing {name} with best params: {best_params}")
        if name == "logistic_regression":
            final_model = LogisticRegression(**best_params, solver="lbfgs", max_iter=3000, class_weight="balanced")
        elif name == "xgboost":
            final_model = XGBClassifier(**best_params, tree_method="hist", device="cuda")
        elif name == "lightgbm":
            final_model = LGBMClassifier(**best_params, device="gpu", verbose=-1)
        elif name == "random_forest":
            final_model = XGBClassifier(**best_params, tree_method="hist", device="cuda", learning_rate=1)

        final_model.fit(X_train_selected, y_train)
        y_prob = final_model.predict_proba(X_test_selected)[:, 1]
        roc = roc_auc_score(y_test, y_prob)
        
        joblib.dump(final_model, f"models/{name}.pkl")
        results.append({"model": name, "roc_auc": roc, "params": best_params})
        print(f"Result: ROC-AUC = {roc:.4f}")

    # Add Naive Bayes (No tuning needed)
    nb = GaussianNB().fit(X_train_selected, y_train)
    joblib.dump(nb, "models/naive_bayes.pkl")
    results.append({"model": "naive_bayes", "roc_auc": roc_auc_score(y_test, nb.predict_proba(X_test_selected)[:, 1]), "params": "default"})

    pd.DataFrame(results).to_csv("results/model_comparison.csv", index=False)
    print("\nTraining complete. Models saved to /models.")

if __name__ == "__main__":
    train_models()
