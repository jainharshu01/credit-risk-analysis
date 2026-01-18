import osimport os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    HistGradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_auc_score

# External Boosting Libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Custom modules
from src.feature_engineering import build_feature_pipeline
from src.data_preprocessing import handle_outliers
from src.config import CLIP_COLS, LOG_COLS, IQR_COLS
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold


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
        l1_ratio=0,
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

    if hasattr(X_train_selected, "toarray"):
        X_train_selected = X_train_selected.toarray()
        X_test_selected = X_test_selected.toarray()    
    # -------------------------------
    # Define Models
    # -------------------------------
    print("Initializing models...")
    
    # Calculate scale_pos_weight for boosting models (approx: neg_count / pos_count)
    # This helps XGB/LGBM handle class imbalance similar to class_weight='balanced'
    neg_count, pos_count = np.bincount(y_train)
    scale_pos_weight = neg_count / pos_count

    # -------------------------------
    # Hyperparameter Grids
    # -------------------------------
    print("Defining hyperparameter grids...")
    
    param_grids = {
        "logistic_regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"] # lbfgs supports l2
        },
        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [6, 10, 15],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bynode": [0.7, 0.8, 0.9]
        },
        "xgboost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 9],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
        },
        "lightgbm": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [20, 31, 50, 100]
        }
    }

    models = {
        "logistic_regression": LogisticRegression(
            solver="lbfgs",
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ),
        
        # Improved Random Forest
        "random_forest": XGBClassifier(
            learning_rate=1,        # Essential for RF behavior in XGB
            subsample=0.8,
            colsample_bynode=0.8,
            n_estimators=100,
            tree_method="hist",     # GPU acceleration
            device="cuda",          # Use Kaggle's T4 GPU
            random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            # --- GPU SETTINGS ---
            tree_method="hist", 
            device="cuda", 
            # --------------------
            use_label_encoder=False,
            random_state=42,
        ),

        "lightgbm": LGBMClassifier(
           n_estimators=300,
            learning_rate=0.05,
            class_weight="balanced",
            # --- GPU SETTINGS ---
            device="gpu",
            gpu_platform_id=0,
            gpu_device_id=0,
            # --------------------
            random_state=42,
            verbose=-1,
            num_leaves=31,
            
        ),

        "naive_bayes": GaussianNB()
    }


    results = []

    # Cross-validation strategy for tuning
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n{'='*30}")
        print(f"STARTING: {name}")
        print(f"{'='*30}")
        print(f"\nProcessing {name}...")

        # Check if we have a parameter grid for this model
        if name in param_grids:
            print(f"  Tuning hyperparameters for {name}...")
            
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grids[name],
                n_iter=5,             # Tries 5 random combinations (increase to 20+ for better results)
                scoring='roc_auc',     # Optimizes for ROC-AUC
                cv=cv,
                n_jobs=-1,             # Use all CPU cores
                verbose=1,
                random_state=42
            )
            
            search.fit(X_train_selected, y_train)
            
            # Use the best model found
            best_model = search.best_estimator_
            print(f"  Best params: {search.best_params_}")
            print(f"  Best CV Score: {search.best_score_:.4f}")
        else:
            # No grid defined (e.g., Naive Bayes), just fit the base model
            print(f"  No tuning grid found. Training base model...")
            best_model = model
            best_model.fit(X_train_selected, y_train)

        # -------------------------------
        # Evaluation & Saving
        # -------------------------------
        y_pred = best_model.predict(X_test_selected)
        
        # Handle probability prediction (some models like SVM need probability=True)
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_test_selected)[:, 1]
        else:
            # Fallback for models without probability output (rare in this list)
            y_prob = best_model.decision_function(X_test_selected)

        roc = roc_auc_score(y_test, y_prob)

        print(f"  Test Set ROC-AUC: {roc:.4f}")
        print(classification_report(y_test, y_pred))

        # Save the BEST model
        model_path = Path(f"models/{name}.pkl")
        joblib.dump(best_model, model_path)

        results.append({
            "model": name,
            "roc_auc": roc,
            "best_params": search.best_params_ if name in param_grids else "default"
        })

    # Save Preprocessor (only needs to be saved once, outside the loop)
    joblib.dump(preprocessor, Path("models/preprocessor.pkl"))
    results_df = pd.DataFrame(results)
    
    results_df.to_csv("results/model_comparison.csv", index=False)
    print("Training completed. Results saved.")

if __name__ == "__main__":
    train_models()


