#!/usr/bin/env python3
"""Feature Importance script for NEERA.

Analyzes the features of the best trained model using gain-based importance and
permutation-based importance on the validation set. Generates reports and plots.
"""

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
VAL_CSV = ROOT / "outputs/predictions/val_split.csv"
MODEL_PKL = ROOT / "outputs/models/best_model.pkl"
OUT_REPORT = ROOT / "outputs/reports/feature_importance_report.txt"
OUT_PLOT_DIR = ROOT / "outputs/plots"

def analyze_features():
    print("Loading best model and validation data...")
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)

    df_val = pd.read_csv(VAL_CSV)
    
    # Identify features from the model
    # LightGBM / XGBoost have feature_name_ attributes
    if hasattr(model, "feature_name_"):
        features = model.feature_name_
    elif hasattr(model, "feature_names_"):
        features = model.feature_names_
    else:
        # fallback
        raise ValueError("Model does not have feature_name_ or feature_names_ attribute.")

    target = "target_next_season_gw"
    
    # Preprocess categorical features to match train dtypes
    for col in features:
        if df_val[col].dtype == object or col in ["season", "rainfall_source_type", "mapping_method", "station_id"]:
            df_val[col] = df_val[col].astype(str).astype("category")

    X_val = df_val[features]
    y_val = df_val[target]

    print(f"Calculating gain-based importance for {type(model).__name__}...")
    gain_importances = model.feature_importances_
    
    df_gain = pd.DataFrame({
        "Feature": features,
        "Gain_Importance": gain_importances
    }).sort_values("Gain_Importance", ascending=False)

    print("Calculating permutation importance on validation set...")
    perm_importance = permutation_importance(
        model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=4
    )
    
    df_perm = pd.DataFrame({
        "Feature": features,
        "Permutation_Importance_Mean": perm_importance.importances_mean,
        "Permutation_Importance_Std": perm_importance.importances_std
    }).sort_values("Permutation_Importance_Mean", ascending=False)

    # 1. Plot Gain Importance
    OUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_gain.head(15), x="Gain_Importance", y="Feature", palette="viridis")
    plt.title(f"Top 15 Features by Gain Importance ({type(model).__name__})")
    plt.xlabel("Importance (Split/Gain count)")
    plt.tight_layout()
    gain_plot_path = OUT_PLOT_DIR / "feature_importance_gain.png"
    plt.savefig(gain_plot_path, dpi=150)
    plt.close()
    print(f"Saved gain importance plot to {gain_plot_path}")

    # 2. Plot Permutation Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_perm.head(15), x="Permutation_Importance_Mean", y="Feature", palette="plasma")
    plt.title("Top 15 Features by Permutation Importance (Val Set)")
    plt.xlabel("Mean Drop in Score (MSE/R2 equivalent)")
    plt.tight_layout()
    perm_plot_path = OUT_PLOT_DIR / "feature_importance_permutation.png"
    plt.savefig(perm_plot_path, dpi=150)
    plt.close()
    print(f"Saved permutation importance plot to {perm_plot_path}")

    # 3. Save Feature Importance Report
    report = []
    report.append("=========================================================")
    report.append("NEERA FEATURE IMPORTANCE REPORT")
    report.append(f"Model: {type(model).__name__}")
    report.append(f"Generated at: {pd.Timestamp.now().isoformat()}")
    report.append("=========================================================\n")
    report.append("Gain-Based Feature Importance (All features):")
    report.append(df_gain.to_string(index=False))
    report.append("\nPermutation-Based Feature Importance (All features on Validation Set):")
    report.append(df_perm.to_string(index=False))

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"Feature importance report written to {OUT_REPORT}")

if __name__ == "__main__":
    analyze_features()
