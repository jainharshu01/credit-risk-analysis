import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

# Import shared configs and helpers
from src.config import TEST_SIZE, RANDOM_SEED, CLIP_COLS, LOG_COLS, IQR_COLS
from src.data_preprocessing import handle_outliers

def load_data():
    df = pd.read_csv("data/processed/cleaned_data.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def load_assets():
    models = {}
    model_dir = Path("models")
    for file in model_dir.glob("*.pkl"):
        if file.name not in ["preprocessor.pkl", "selected_features.pkl"]:
            models[file.stem] = joblib.load(file)
    preprocessor = joblib.load("models/preprocessor.pkl")
    selected_mask = joblib.load("models/selected_features.pkl")
    return models, preprocessor, selected_mask

def get_predictions(models, X_test_selected):
    probs = {}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs[name] = model.predict_proba(X_test_selected)[:, 1]
        else:
            p = model.decision_function(X_test_selected)
            probs[name] = (p - p.min()) / (p.max() - p.min())
    return probs

def run_optimized_ensemble():
    os.makedirs("results", exist_ok=True)
    X, y = load_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    X_test = handle_outliers(X_test, CLIP_COLS, LOG_COLS, IQR_COLS)
    
    models, preprocessor, selected_mask = load_assets()
    X_test_selected = preprocessor.transform(X_test)[:, selected_mask]
    if hasattr(X_test_selected, "toarray"): X_test_selected = X_test_selected.toarray()
    
    all_probs = get_predictions(models, X_test_selected)
    
    # 1. Evaluate Individual Models to find Rankings
    individual_results = []
    for name, p in all_probs.items():
        auc = roc_auc_score(y_test, p)
        individual_results.append({"model": name, "roc_auc": auc, "gini": 2*auc-1})
    
    perf_df = pd.DataFrame(individual_results).sort_values("roc_auc", ascending=False)
    top_1_name = perf_df.iloc[0]['model']
    top_3_names = perf_df.iloc[:3]['model'].tolist()
    
    # 2. Strategy A: Duo Ensemble (Top 1 + Logistic Regression)
    # Weights: 80/20 split
    log_reg_name = "logistic_regression"
    duo_probs = (all_probs[top_1_name] * 0.8) + (all_probs[log_reg_name] * 0.2)
    auc_duo = roc_auc_score(y_test, duo_probs)
    
    # 3. Strategy B: Trio Ensemble (Top 3 Models)
    # Weights: Proportional to their ROC-AUC
    top_3_scores = np.array([perf_df[perf_df['model'] == m]['roc_auc'].values[0] for m in top_3_names])
    t3_weights = top_3_scores / top_3_scores.sum()
    trio_probs = np.zeros(len(y_test))
    for name, w in zip(top_3_names, t3_weights):
        trio_probs += all_probs[name] * w
    auc_trio = roc_auc_score(y_test, trio_probs)

    # 4. Final Comparison Table
    comparison = [
        {"Model Type": "Best Individual", "Name": top_1_name, "ROC-AUC": perf_df.iloc[0]['roc_auc']},
        {"Model Type": "Duo Ensemble", "Name": f"{top_1_name} + LogReg", "ROC-AUC": auc_duo},
        {"Model Type": "Trio Ensemble", "Name": "Top 3 Weighted", "ROC-AUC": auc_trio}
    ]
    
    final_df = pd.DataFrame(comparison).sort_values("ROC-AUC", ascending=False)
    final_df["Gini"] = 2 * final_df["ROC-AUC"] - 1
    
    print("\n--- Final Model Comparison ---")
    print(final_df)
    final_df.to_csv("results/final_ensemble_comparison.csv", index=False)
    print("\nResults saved to results/final_ensemble_comparison.csv")

if __name__ == "__main__":
    run_optimized_ensemble()
