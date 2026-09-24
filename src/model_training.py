"""
Stage 4: Model Training, Cross-Validation, ANOVA Feature Significance, & SHAP Analysis.

Trains XGBoost regression / classification models on extracted multi-signal features,
evaluates 5-fold Stratified Cross-Validation performance, conducts One-Way ANOVA
significance testing, and generates SHAP global and local feature importance plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import xgboost as xgb
import shap
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Ensure package root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import build_feature_table
from src.label_preparation import prepare_labels


def get_cv_splitter(y_strat: np.ndarray, n_splits: int = 5):
    """Get adaptive cross-validation splitter (StratifiedKFold or KFold if class counts are low)."""
    unique_classes, counts = np.unique(y_strat, return_counts=True)
    min_count = np.min(counts)

    if min_count >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif min_count >= 2:
        effective_splits = min(n_splits, int(min_count))
        return StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42)
    else:
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def run_anova_tests(X: pd.DataFrame, y_binned: np.ndarray) -> pd.DataFrame:
    """
    Perform One-Way ANOVA F-tests for each feature across binned fatigue score levels.

    Args:
        X: Feature matrix DataFrame.
        y_binned: Discrete target groups for ANOVA categorization.

    Returns:
        DataFrame sorted by ascending p-value with F-statistic and significance markers.
    """
    unique_groups = np.unique(y_binned)
    results = []

    for col in X.columns:
        group_data = [X[col][y_binned == g].values for g in unique_groups if len(X[col][y_binned == g]) > 0]
        if len(group_data) > 1:
            f_stat, p_val = stats.f_oneway(*group_data)
        else:
            f_stat, p_val = 0.0, 1.0

        is_significant = (p_val < 0.05)
        results.append({
            "feature": col,
            "f_statistic": float(f_stat),
            "p_value": float(p_val),
            "significant_p05": is_significant
        })

    df_anova = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
    return df_anova



def train_evaluate_regression(X: pd.DataFrame, y_reg: np.ndarray, n_splits: int = 5) -> dict:
    """
    Train and evaluate XGBRegressor with Stratified CV (stratified on binned y).

    Args:
        X: Feature matrix.
        y_reg: Continuous fatigue targets.
        n_splits: Number of CV folds.

    Returns:
        Dict with fold metrics, average metrics, and trained full model.
    """
    y_binned = np.round(y_reg).astype(int)
    cv = get_cv_splitter(y_binned, n_splits=n_splits)

    fold_metrics = []
    oof_preds = np.zeros(len(y_reg))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_binned)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_reg[train_idx], y_reg[test_idx]

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        oof_preds[test_idx] = preds

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        fold_metrics.append({
            "fold": fold + 1,
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        })

    overall_rmse = float(np.sqrt(mean_squared_error(y_reg, oof_preds)))
    overall_mae = float(mean_absolute_error(y_reg, oof_preds))
    overall_r2 = float(r2_score(y_reg, oof_preds))

    # Fit final full model for SHAP
    full_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        verbosity=0
    )
    full_model.fit(X, y_reg)

    return {
        "fold_metrics": fold_metrics,
        "mean_rmse": overall_rmse,
        "mean_mae": overall_mae,
        "mean_r2": overall_r2,
        "oof_preds": oof_preds,
        "full_model": full_model
    }


def train_evaluate_classification(X: pd.DataFrame, y_clf: np.ndarray, n_splits: int = 5) -> dict:
    """
    Train and evaluate XGBClassifier with Stratified CV.

    Args:
        X: Feature matrix.
        y_clf: Binary classification targets (0/1).
        n_splits: Number of CV folds.

    Returns:
        Dict with fold classification metrics, average metrics, and full model.
    """
    cv = get_cv_splitter(y_clf, n_splits=n_splits)

    fold_metrics = []
    oof_preds = np.zeros(len(y_clf))
    oof_probs = np.zeros(len(y_clf))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_clf)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_clf[train_idx], y_clf[test_idx]

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
            verbosity=0
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else preds

        oof_preds[test_idx] = preds
        oof_probs[test_idx] = probs

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        fold_metrics.append({
            "fold": fold + 1,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1)
        })

    overall_acc = accuracy_score(y_clf, oof_preds)
    overall_prec = precision_score(y_clf, oof_preds, zero_division=0)
    overall_rec = recall_score(y_clf, oof_preds, zero_division=0)
    overall_f1 = f1_score(y_clf, oof_preds, zero_division=0)
    try:
        overall_auc = roc_auc_score(y_clf, oof_probs)
    except Exception:
        overall_auc = 0.5

    full_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    full_model.fit(X, y_clf)

    return {
        "fold_metrics": fold_metrics,
        "mean_accuracy": float(overall_acc),
        "mean_precision": float(overall_prec),
        "mean_recall": float(overall_rec),
        "mean_f1": float(overall_f1),
        "mean_auc": float(overall_auc),
        "oof_preds": oof_preds,
        "full_model": full_model
    }



def compute_shap_analysis(model, X: pd.DataFrame, output_dir: str = "data") -> tuple:
    """
    Compute SHAP values and save summary plot to file.

    Args:
        model: Fitted XGBoost model instance.
        X: Feature matrix DataFrame.
        output_dir: Folder to save plot artifact.

    Returns:
        Tuple of (shap_values, explainer).
    """
    os.makedirs(output_dir, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle multi-output / binary array shape if needed
    if isinstance(shap_values, list):
        shap_vals_summary = shap_values[1]
    else:
        shap_vals_summary = shap_values

    # Render summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_summary, X, show=False)
    plot_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"[SHAP] Saved SHAP summary plot to {plot_path}")
    return shap_vals_summary, explainer


def run_stage4_pipeline(data_dir: str = "data") -> dict:
    """
    Full Stage 4 pipeline: loads features, checks regression vs classification,
    runs Stratified CV, ANOVA, and SHAP.
    """
    df_features = build_feature_table(data_dir)
    if df_features.empty:
        raise ValueError("No feature data found in data directory.")

    label_data = prepare_labels(df_features)
    X = label_data["X"]
    y_reg = label_data["y_reg"]
    y_clf = label_data["y_clf"]

    # 1. Run ANOVA tests
    df_anova = run_anova_tests(X, np.round(y_reg).astype(int))

    # 2. Try Regression First
    reg_results = train_evaluate_regression(X, y_reg)

    # Decision Logic: If R^2 >= 0.1, use Regression. Else fallback to Classification.
    use_classification = (reg_results["mean_r2"] < 0.10)

    clf_results = None
    if use_classification:
        print("[STAGE 4 DECISION] Continuous regression R² is low/noisy (< 0.10). Falling back to Binary Classification.")
        clf_results = train_evaluate_classification(X, y_clf)
        active_model = clf_results["full_model"]
        chosen_strategy = "Binary Classification (Fatigued >= 5 vs Not Fatigued < 5)"
    else:
        print("[STAGE 4 DECISION] Continuous regression performance is sufficient (R² >= 0.10). Using Regression.")
        active_model = reg_results["full_model"]
        chosen_strategy = "Continuous Regression (1–7 Karolinska Score)"

    # 3. Compute SHAP Analysis
    shap_vals, explainer = compute_shap_analysis(active_model, X, data_dir)

    return {
        "chosen_strategy": chosen_strategy,
        "is_classification": use_classification,
        "reg_results": reg_results,
        "clf_results": clf_results,
        "anova": df_anova,
        "shap_values": shap_vals,
        "explainer": explainer,
        "feature_names": X.columns.tolist(),
        "X": X,
        "y_reg": y_reg,
        "y_clf": y_clf
    }


def print_stage4_report(results: dict):
    """Print clean formatted report of Stage 4 modeling and ANOVA results."""
    print("=" * 65)
    print("STAGE 4 — MODEL TRAINING & VALIDATION REPORT")
    print("=" * 65)
    print(f"Chosen Strategy: {results['chosen_strategy']}")

    print("\n--- 1. Regression Metrics (5-Fold Stratified CV) ---")
    reg = results["reg_results"]
    for f in reg["fold_metrics"]:
        print(f"  Fold {f['fold']}: RMSE = {f['rmse']:.4f}, MAE = {f['mae']:.4f}, R² = {f['r2']:.4f}")
    print(f"  --> OOF Overall RMSE: {reg['mean_rmse']:.4f}")
    print(f"  --> OOF Overall MAE:  {reg['mean_mae']:.4f}")
    print(f"  --> OOF Overall R²:   {reg['mean_r2']:.4f}")

    if results["clf_results"] is not None:
        print("\n--- 2. Classification Metrics (Fallback 5-Fold Stratified CV) ---")
        clf = results["clf_results"]
        for f in clf["fold_metrics"]:
            print(f"  Fold {f['fold']}: Acc = {f['accuracy']:.4f}, Prec = {f['precision']:.4f}, Rec = {f['recall']:.4f}, F1 = {f['f1']:.4f}")
        print(f"  --> OOF Overall Accuracy:  {clf['mean_accuracy']:.4f}")
        print(f"  --> OOF Overall Precision: {clf['mean_precision']:.4f}")
        print(f"  --> OOF Overall Recall:    {clf['mean_recall']:.4f}")
        print(f"  --> OOF Overall F1-Score:  {clf['mean_f1']:.4f}")
        print(f"  --> OOF Overall ROC-AUC:   {clf['mean_auc']:.4f}")

    print("\n--- 3. ANOVA One-Way Feature Significance (Top 10) ---")
    df_anova = results["anova"]
    print(df_anova.head(10).to_string(index=False))

    print("\n--- 4. SHAP Feature Importance ---")
    shap_means = np.abs(results["shap_values"]).mean(axis=0)
    df_shap = pd.DataFrame({"feature": results["feature_names"], "mean_abs_shap": shap_means}).sort_values("mean_abs_shap", ascending=False)
    print(df_shap.to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    res = run_stage4_pipeline("data")
    print_stage4_report(res)
