import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.model_selection import train_test_split

# Import shared configs
from src.config import TEST_SIZE, RANDOM_SEED, CLIP_COLS, LOG_COLS, IQR_COLS
from src.data_preprocessing import handle_outliers

def load_data():
    df = pd.read_csv("data/processed/cleaned_data.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def load_assets():
    """Loads models and preprocessing tools."""
    models = {}
    model_dir = Path("models")

    # Load all models except helper pickles
    for file in model_dir.glob("*.pkl"):
        if file.name not in ["preprocessor.pkl", "selected_features.pkl"]:
            models[file.stem] = joblib.load(file)

    preprocessor = joblib.load("models/preprocessor.pkl")
    selected_mask = joblib.load("models/selected_features.pkl")

    return models, preprocessor, selected_mask

def plot_evaluation_graphics(y_test, y_prob, y_pred, name):
    """Generates and saves standard diagnostic plots."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {name}")
    plt.tight_layout()
    plt.savefig(f"results/cm_{name}.png")
    plt.close()

def evaluate_models():
    os.makedirs("results", exist_ok=True)

    print("--- Initializing Evaluation (Weighted Stats & Gini) ---")
    X, y = load_data()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Preprocessing
    X_test = handle_outliers(X_test, CLIP_COLS, LOG_COLS, IQR_COLS)
    models, preprocessor, selected_mask = load_assets()
    
    X_test_transformed = preprocessor.transform(X_test)
    X_test_selected = X_test_transformed[:, selected_mask]

    if hasattr(X_test_selected, "toarray"):
        X_test_selected = X_test_selected.toarray()

    summary_stats = []

    for name, model in models.items():
        print(f"Testing: {name}...")

        # 1. Prediction Probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_selected)[:, 1]
        else:
            y_prob = model.decision_function(X_test_selected)

        # 2. Threshold Optimization (Maximize Weighted F1)
        thresholds = np.linspace(0.1, 0.9, 50)
        best_f1, best_t = 0, 0.5
        for t in thresholds:
            # Using weighted F1 to account for class imbalance
            f1 = f1_score(y_test, (y_prob >= t), average='weighted', zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        y_pred = (y_prob >= best_t).astype(int)

        # 3. Calculate Metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        gini = 2 * roc_auc - 1  # Standard Gini Formula
        
        summary_stats.append({
            "model": name,
            "roc_auc": roc_auc,
            "gini": gini,
            "weighted_f1": best_f1,
            "weighted_precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "weighted_recall": recall_score(y_test, y_pred, average='weighted'),
            "threshold": best_t
        })

        plot_evaluation_graphics(y_test, y_prob, y_pred, name)

    # 4. Final Export
    # Note: Using 'roc_auc' as the column name to match your ensemble script's requirements
    results_df = pd.DataFrame(summary_stats).sort_values("roc_auc", ascending=False)
    results_df.to_csv("results/model_performance_summary.csv", index=False)
    print("\nEvaluation Complete. Results saved in /results")

if __name__ == "__main__":
    evaluate_models()
