import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

from src.config import TEST_SIZE, RANDOM_SEED
from src.data_preprocessing import handle_outliers
from src.feature_engineering import build_feature_pipeline
from src.config import CLIP_COLS, LOG_COLS, IQR_COLS


def load_data():
    df = pd.read_csv("data/processed/cleaned_data.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )


def load_models():
    models = {}
    model_dir = Path("models")

    for file in model_dir.glob("*.pkl"):
        if file.name not in ["preprocessor.pkl", "selected_features.pkl"]:
            models[file.stem] = joblib.load(file)

    preprocessor = joblib.load("models/preprocessor.pkl")
    selected_features = joblib.load("models/selected_features.pkl")

    return models, preprocessor, selected_features


def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend()
    plt.savefig("results/roc_curves.png")
    plt.close()



def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"results/confusion_matrix_{model_name}.png")
    plt.close()
    


def evaluate_models():
    os.makedirs("results", exist_ok=True)

    print("Loading data...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Speed up evaluation by sampling
    sample_size = 50000   # you can even try 20000 first
    if len(X_test) > sample_size:
        X_test = X_test.sample(sample_size, random_state=42)
        y_test = y_test.loc[X_test.index]


    # Outlier handling (same as training)
    X_train = handle_outliers(X_train, CLIP_COLS, LOG_COLS, IQR_COLS)
    X_test = handle_outliers(X_test, CLIP_COLS, LOG_COLS, IQR_COLS)

    print("Loading models and preprocessors...")
    models, preprocessor, selected_features = load_models()

    # Ensure selected_features is column indices, not a boolean mask
    selected_features = np.array(selected_features)

    if selected_features.dtype == bool:
        selected_features = np.where(selected_features)[0]

    # Apply preprocessing
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    X_test_df = pd.DataFrame(X_test_transformed)
    X_test_selected = X_test_df.iloc[:, selected_features]


    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        # Probabilities
        y_prob = model.predict_proba(X_test_selected)[:, 1]

        # -------------------------------
        # 1️⃣ Threshold Tuning
        # -------------------------------
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5

        for t in thresholds:
            y_pred_t = (y_prob >= t).astype(int)
            f1 = f1_score(y_test, y_pred_t)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        print(f"Best Threshold for {name}: {best_threshold:.2f}")
        print(f"Best F1 Score: {best_f1:.4f}")

        # Final predictions using best threshold
        y_pred = (y_prob >= best_threshold).astype(int)

        # -------------------------------
        # 2️⃣ Metrics
        # -------------------------------
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"{name} ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))

        plot_confusion_matrix(y_test, y_pred, name)

        # -------------------------------
        # 3️⃣ Feature Importance
        # -------------------------------
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_imp = pd.DataFrame({
                "feature_index": selected_features,
                "importance": importances
            }).sort_values(by="importance", ascending=False)

            feature_imp.to_csv(f"results/{name}_feature_importance.csv", index=False)
            print(f"Feature importance saved → results/{name}_feature_importance.csv")

        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_[0])
            feature_imp = pd.DataFrame({
                "feature_index": selected_features,
                "importance": coef
            }).sort_values(by="importance", ascending=False)

            feature_imp.to_csv(f"results/{name}_feature_importance.csv", index=False)
            print(f"Coefficient importance saved → results/{name}_feature_importance.csv")

        # -------------------------------
        # 4️⃣ Save Comparison Metrics
        # -------------------------------
        results.append({
            "model": name,
            "roc_auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "best_threshold": best_threshold
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/final_model_comparison.csv", index=False)

    print("\nFinal Evaluation Results Saved → results/final_model_comparison.csv")

    plot_roc_curves(models, X_test_selected, y_test)


if __name__ == "__main__":
    evaluate_models()
