import joblib
import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Custom modules
from src.data_preprocessing import handle_outliers
from src.config import CLIP_COLS, LOG_COLS, IQR_COLS

def run_dynamic_ensemble(top_n=3):
    # Suppress specific XGBoost device mismatch warnings
    warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")

    models_dir = Path("models")
    results_dir = Path("results")
    comparison_path = results_dir / "model_performance_summary.csv"
    
    if not comparison_path.exists():
        print("Error: model_comparison.csv not found. Run evaluate_models.py first.")
        return

    # 1. Dynamically find the best models from the leaderboard
    results_df = pd.read_csv(comparison_path)
    best_models_info = results_df.sort_values(by="roc_auc", ascending=False).head(top_n)
    best_model_names = best_models_info['model'].tolist()
    
    # Identify the single best individual model for final comparison
    best_single_model_row = best_models_info.iloc[0]

    print(f"--- Top {top_n} Models Selected for Ensemble ---")
    for idx, row in best_models_info.iterrows():
        print(f"{row['model']}: ROC-AUC = {row['roc_auc']:.4f}, Gini = {row['gini']:.4f}")

    # 2. Load Data and Replicate Test Split
    df = pd.read_csv("data/processed/cleaned_data.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Preprocess Test Data
    X_test = handle_outliers(X_test, CLIP_COLS, LOG_COLS, IQR_COLS)
    preprocessor = joblib.load(models_dir / "preprocessor.pkl")
    selected_mask = joblib.load(models_dir / "selected_features.pkl")
    X_test_selected = preprocessor.transform(X_test)[:, selected_mask]
    
    if hasattr(X_test_selected, "toarray"):
        X_test_selected = X_test_selected.toarray()

    # 4. Gather Probabilities and use ROC-AUC as weights for voting
    all_probs = []
    auc_weights = []
    
    for name in best_model_names:
        model = joblib.load(models_dir / f"{name}.pkl")
        probs = model.predict_proba(X_test_selected)[:, 1]
        all_probs.append(probs)
        
        # Use the actual ROC-AUC score as the weight
        auc_score = best_models_info[best_models_info['model'] == name]['roc_auc'].values[0]
        auc_weights.append(auc_score)

    # 5. Calculate Weighted Average Probability
    auc_weights = np.array(auc_weights)
    normalized_weights = auc_weights / auc_weights.sum()
    prob_matrix = np.array(all_probs)
    ensemble_final_prob = np.tensordot(normalized_weights, prob_matrix, axes=1)
    
    # 6. Final Evaluation for Ensemble (Weighted)
    ensemble_auc = roc_auc_score(y_test, ensemble_final_prob)
    ensemble_gini = 2 * ensemble_auc - 1
    ensemble_pred = (ensemble_final_prob >= 0.5).astype(int)
    
    # Weighted metrics for ensemble
    ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
    ensemble_precision = precision_score(y_test, ensemble_pred, average='weighted')
    ensemble_recall = recall_score(y_test, ensemble_pred, average='weighted')

    print(f"\n{'='*40}")
    print(f"DYNAMIC ENSEMBLE RESULTS")
    print(f"ROC-AUC: {ensemble_auc:.4f} | GINI: {ensemble_gini:.4f}")
    print(f"{'='*40}")
    print(classification_report(y_test, ensemble_pred))
    
    # 7. Create Comparison Output File
    comparison_data = [
        {
            "Type": "Best Individual Model",
            "Model_Name": best_single_model_row['model'],
            "ROC_AUC": best_single_model_row['roc_auc'],
            "Gini": best_single_model_row['gini'],
            "Weighted_F1": best_single_model_row['weighted_f1']
        },
        {
            "Type": "Weighted Ensemble",
            "Model_Name": f"Top_{top_n}_Combined",
            "ROC_AUC": ensemble_auc,
            "Gini": ensemble_gini,
            "Weighted_F1": ensemble_f1
        }
    ]
    
    comparison_report_df = pd.DataFrame(comparison_data)
    comparison_report_df.to_csv(results_dir / "ensemble_vs_best_model.csv", index=False)
    
    # Save raw probabilities
    joblib.dump(ensemble_final_prob, results_dir / "dynamic_ensemble_probs.pkl")
    print(f"Comparison report saved to: {results_dir / 'ensemble_vs_best_model.csv'}")

if __name__ == "__main__":
    run_dynamic_ensemble(top_n=3)