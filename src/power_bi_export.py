"""
Power BI Data Export Module for Mental Fatigue Detection.

Consolidates raw logs, window-level multi-signal features, model predictions,
and SHAP / ANOVA importance into clean, normalized CSV tables optimized
for direct import and relationship modeling in Microsoft Power BI.
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

from src.feature_engineering import discover_session_ids, extract_window_features, load_session_csvs, build_feature_table
from src.label_preparation import prepare_labels
from src.model_training import run_stage4_pipeline


def export_power_bi_datasets(data_dir: str = DEFAULT_DATA_DIR, export_dir: str = None) -> dict:
    """
    Generate and export clean CSV datasets for Power BI dashboard creation.

    Args:
        data_dir: Directory containing raw session CSVs.
        export_dir: Target directory to save Power BI CSV files.

    Returns:
        Dictionary with exported filepaths and row counts.
    """
    if export_dir is None:
        export_dir = os.path.join(data_dir, "power_bi_export")
    os.makedirs(export_dir, exist_ok=True)

    session_ids = discover_session_ids(data_dir)
    if not session_ids:
        print("[POWER BI EXPORT] No sessions found to export.")
        return {}

    # 1. Run model pipeline to obtain trained model & feature importance
    pipeline_res = run_stage4_pipeline(data_dir)
    full_model = (
        pipeline_res["clf_results"]["full_model"]
        if pipeline_res["is_classification"]
        else pipeline_res["reg_results"]["full_model"]
    )
    feature_names = pipeline_res["feature_names"]

    # -------------------------------------------------------------
    # Table 1: Window Features & Model Predictions (Fact Table)
    # -------------------------------------------------------------
    df_features = build_feature_table(data_dir)
    
    # Generate predictions for all window checkpoints
    X_mat = df_features[feature_names]
    predictions = full_model.predict(X_mat)
    
    # Add prediction columns & derived analytics
    df_window_bi = df_features.copy()
    df_window_bi["predicted_fatigue_score"] = np.round(predictions, 2)
    df_window_bi["prediction_error"] = np.round(df_window_bi["predicted_fatigue_score"] - df_window_bi["fatigue_score"], 2)
    df_window_bi["absolute_prediction_error"] = np.abs(df_window_bi["prediction_error"])
    
    # Fatigue Categorization
    def categorize_fatigue(score):
        if score <= 2.5:
            return "Alert (Low Fatigue)"
        elif score <= 4.5:
            return "Moderate Alertness"
        elif score <= 5.5:
            return "Mild Fatigue"
        else:
            return "Severe Fatigue"

    df_window_bi["actual_fatigue_category"] = df_window_bi["fatigue_score"].apply(categorize_fatigue)
    df_window_bi["predicted_fatigue_category"] = df_window_bi["predicted_fatigue_score"].apply(categorize_fatigue)
    df_window_bi["error_rate"] = np.round(1.0 - df_window_bi["accuracy"], 4)

    window_path = os.path.join(export_dir, "power_bi_window_features.csv")
    df_window_bi.to_csv(window_path, index=False)

    # -------------------------------------------------------------
    # Table 2: Session-Level Summary (Dimension Table)
    # -------------------------------------------------------------
    session_summary_rows = []
    for sid in session_ids:
        raw = load_session_csvs(sid, data_dir)
        df_perf = raw["performance"]
        df_fat = raw["fatigue"]
        df_br = raw["breaks"]

        if df_perf.empty:
            continue

        total_questions = len(df_perf)
        overall_accuracy = float(df_perf["is_correct"].mean())
        mean_rt = float(df_perf["response_time_sec"].mean())
        total_session_duration_sec = float(df_perf["elapsed_session_time_sec"].max())
        
        initial_fatigue = int(df_fat.iloc[0]["fatigue_score"]) if not df_fat.empty else 1
        final_fatigue = int(df_fat.iloc[-1]["fatigue_score"]) if not df_fat.empty else initial_fatigue
        fatigue_delta = final_fatigue - initial_fatigue

        total_breaks = len(df_br) if not df_br.empty else 0
        total_break_duration = float(df_br["duration_sec"].sum()) if not df_br.empty else 0.0

        session_summary_rows.append({
            "session_id": sid,
            "total_questions": total_questions,
            "overall_accuracy_pct": np.round(overall_accuracy * 100, 2),
            "mean_response_time_sec": np.round(mean_rt, 2),
            "total_duration_sec": np.round(total_session_duration_sec, 1),
            "total_duration_min": np.round(total_session_duration_sec / 60.0, 2),
            "initial_fatigue_score": initial_fatigue,
            "final_fatigue_score": final_fatigue,
            "fatigue_score_change": fatigue_delta,
            "total_breaks_taken": total_breaks,
            "total_break_duration_sec": np.round(total_break_duration, 1),
            "final_fatigue_category": categorize_fatigue(final_fatigue)
        })

    df_sessions_bi = pd.DataFrame(session_summary_rows)
    sessions_path = os.path.join(export_dir, "power_bi_sessions_summary.csv")
    df_sessions_bi.to_csv(sessions_path, index=False)

    # -------------------------------------------------------------
    # Table 3: Granular Question Performance Logs (Fact Table)
    # -------------------------------------------------------------
    perf_list = []
    for sid in session_ids:
        raw = load_session_csvs(sid, data_dir)
        df_p = raw["performance"]
        if not df_p.empty:
            df_p_copy = df_p.copy()
            df_p_copy["session_id"] = sid
            perf_list.append(df_p_copy)

    if perf_list:
        df_perf_all = pd.concat(perf_list, ignore_index=True)
        # Add window checkpoint tag to enable Star Schema relationship
        df_perf_all["window_checkpoint"] = ((df_perf_all["question_index"] - 1) // 10 + 1) * 10
        # Differentiate error types
        median_rt = df_perf_all.groupby("session_id")["response_time_sec"].transform("median")
        df_perf_all["is_careless_error"] = ((df_perf_all["is_correct"] == 0) & (df_perf_all["response_time_sec"] < median_rt)).astype(int)
        df_perf_all["is_effortful_error"] = ((df_perf_all["is_correct"] == 0) & (df_perf_all["response_time_sec"] >= median_rt)).astype(int)
        
        perf_path = os.path.join(export_dir, "power_bi_question_performance.csv")
        df_perf_all.to_csv(perf_path, index=False)
    else:
        perf_path = None

    # -------------------------------------------------------------
    # Table 4: Feature Importance & ANOVA Significance Table
    # -------------------------------------------------------------
    df_anova = pipeline_res["anova"].copy()
    shap_vals = pipeline_res["shap_values"]
    shap_means = np.abs(shap_vals).mean(axis=0)

    df_shap_map = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap_impact": np.round(shap_means, 4)
    })

    df_importance = pd.merge(df_anova, df_shap_map, on="feature", how="left")
    df_importance["importance_rank"] = df_importance["mean_abs_shap_impact"].rank(ascending=False).astype(int)

    importance_path = os.path.join(export_dir, "power_bi_feature_importance.csv")
    df_importance.to_csv(importance_path, index=False)

    exported_info = {
        "export_dir": export_dir,
        "window_features_csv": window_path,
        "sessions_summary_csv": sessions_path,
        "question_performance_csv": perf_path,
        "feature_importance_csv": importance_path,
        "total_sessions": len(df_sessions_bi),
        "total_windows": len(df_window_bi)
    }

    print(f"[POWER BI EXPORT] Successfully exported 4 Power BI datasets to: {export_dir}")
    return exported_info


if __name__ == "__main__":
    export_power_bi_datasets(DEFAULT_DATA_DIR)
