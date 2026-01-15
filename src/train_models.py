import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from src.feature_engineering import build_feature_pipeline, select_features_by_correlation
from src.data_preprocessing import handle_outliers
from src.config import CLIP_COLS, LOG_COLS, IQR_COLS

def load_processed_data():
    data_path = Path("data/processed/cleaned_data.csv")
    df = pd.read_csv(data_path)
    return df

def split_data(df, target_col="target"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def get_feature_types(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    return numeric_features, categorical_features

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def train_models():
    print("Loading data...")
    df = load_processed_data()

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    # After split
    X_train = handle_outliers(
        X_train,
        clip_cols=CLIP_COLS,
        log_cols=LOG_COLS,
        iqr_cols=IQR_COLS
    )

    X_test = handle_outliers(
        X_test,
        clip_cols=CLIP_COLS,
        log_cols=LOG_COLS,
        iqr_cols=IQR_COLS
    )


    print("Detecting feature types...")
    numeric_features, categorical_features = get_feature_types(X_train)
    ordinal_features = ['sub_grade']


    print("Building preprocessing pipeline...")
    preprocessor = build_feature_pipeline(
        numeric_features, 
        categorical_features, 
        ordinal_features
    )

    print("Preprocessing data...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # -------------------------------
    # L1 Logistic Regression for Feature Selection
    # -------------------------------
    print("Running L1-based feature selection...")

    lasso_selector = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=0.05,                 # strong regularization
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    )

    lasso_selector.fit(X_train_transformed, y_train)

    # Mask of selected features
    selected_mask = np.abs(lasso_selector.coef_[0]) > 0

    print(f"Selected {selected_mask.sum()} features out of {X_train_transformed.shape[1]}")

    # Apply mask
    X_train_selected = X_train_transformed[:, selected_mask]
    X_test_selected = X_test_transformed[:, selected_mask]

    # Save selected mask
    joblib.dump(selected_mask, "models/selected_features.pkl")

    
    models = {
    "logistic_regression": LogisticRegression(
        solver="lbfgs",
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    }


    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_selected, y_train)

        y_pred = model.predict(X_test_selected)
        y_prob = model.predict_proba(X_test_selected)[:, 1]

        roc = roc_auc_score(y_test, y_prob)

        print(f"{name} ROC-AUC: {roc:.4f}")
        print(classification_report(y_test, y_pred))

        # Save model
        model_path = Path(f"models/{name}.pkl")
        joblib.dump(model, model_path)

        # Save preprocessor once
        joblib.dump(preprocessor, Path("models/preprocessor.pkl"))

        results.append({
            "model": name,
            "roc_auc": roc
        })

    results_df = pd.DataFrame(results)
    
    results_df.to_csv("results/model_comparison.csv", index=False)
    print("Training completed. Results saved.")

if __name__ == "__main__":
    train_models()
