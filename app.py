"""
Stage 5: Streamlit Interactive Demo Application for Mental Fatigue Detection.

Provides an interactive dashboard visualizing live session predictions, actual self-reported
fatigue trends over time, SHAP feature attribution per checkpoint, and sidebar explanations.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure package root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.feature_engineering import discover_session_ids, extract_window_features, build_feature_table
from src.label_preparation import prepare_labels
from src.model_training import run_stage4_pipeline


# Configure Page
st.set_page_config(
    page_title="Mental Fatigue Detection Dashboard",
    page_icon="🧠",
    layout="wide"
)


@st.cache_data
def load_pipeline_data(data_dir: str = "data"):
    """Cache full feature engineering and model training pipeline results."""
    try:
        pipeline_results = run_stage4_pipeline(data_dir)
        return pipeline_results
    except Exception as e:
        st.error(f"Error initializing model pipeline: {str(e)}")
        return None


def main():
    st.title("🧠 Mental Fatigue Detection — Multi-Signal System")
    st.markdown("""
    Predicting self-reported mental fatigue (Karolinska Sleepiness Scale 1–7) in real time using 
    **keystroke dynamics**, **mouse movement patterns**, **task performance**, **error taxonomy**, and **break behavior**.
    """)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    session_ids = discover_session_ids(data_dir)

    # -------------------------------------------------------------
    # Sidebar Controls & Domain Information
    # -------------------------------------------------------------
    st.sidebar.header("🕹️ Session Selector & Controls")

    if not session_ids:
        st.sidebar.warning("No session data CSVs found in `data/`. Please run `python src/generate_sample_data.py` or collect a session.")
        selected_session = None
    else:
        selected_session = st.sidebar.selectbox("Select Session ID to Inspect:", session_ids)

    st.sidebar.markdown("---")
    st.sidebar.header("📚 Multi-Signal Domain Guide")
    st.sidebar.markdown("""
    **1. Keystroke Dynamics**:
    Captures motor slowdown, motor variability, and error correction rates (press duration, inter-key pause, backspace rate, typing speed).

    **2. Mouse Behavior**:
    Tracks motor degradation via movement speed decline, direction change jitter, and click precision.

    **3. Task Performance**:
    Monitors cognitive slowdown (response time) and accuracy degradation.

    **4. Error Pattern Taxonomy**:
    Differentiates between **careless errors** (fast & wrong) vs **effortful errors** (slow & wrong).

    **5. Voluntary Breaks**:
    Tracks self-regulatory fatigue mitigation frequency and total duration.

    **6. Question Difficulty**:
    Controlled variable held constant to eliminate cognitive load confounds.
    """)

    pipeline_res = load_pipeline_data(data_dir)
    if pipeline_res is None or selected_session is None:
        st.info("Generating demonstration sample data...")
        return

    # Extract features for selected session
    df_session = extract_window_features(selected_session, data_dir)

    if df_session.empty:
        st.warning(f"No window data found for session {selected_session}.")
        return

    # Filter predictor features matching model
    feature_names = pipeline_res["feature_names"]
    X_session = df_session[feature_names]
    y_actual = df_session["fatigue_score"].values
    checkpoints = df_session["window_checkpoint"].values

    # Generate Model Predictions
    model = pipeline_res["reg_results"]["full_model"] if not pipeline_res["is_classification"] else pipeline_res["clf_results"]["full_model"]
    y_preds = model.predict(X_session)

    # -------------------------------------------------------------
    # Section 1: Session Overview Metrics
    # -------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Session ID", selected_session)
    with col2:
        st.metric("Total Questions", f"{checkpoints[-1] if len(checkpoints) > 0 else 50}")
    with col3:
        st.metric("Final Self-Reported Fatigue", f"{y_actual[-1]} / 7")
    with col4:
        st.metric("Final Predicted Fatigue", f"{y_preds[-1]:.2f}")

    st.markdown("---")

    # -------------------------------------------------------------
    # Section 2: Live Predicted vs Actual Fatigue Trend Chart
    # -------------------------------------------------------------
    st.subheader("📈 Live Fatigue Score Trend Across Checkpoints")

    chart_df = pd.DataFrame({
        "Question Checkpoint": checkpoints,
        "Actual Karolinska Score": y_actual,
        "Predicted Fatigue Score": y_preds
    }).set_index("Question Checkpoint")

    st.line_chart(chart_df, color=["#ef4444", "#3b82f6"])

    # -------------------------------------------------------------
    # Section 3: Feature Attribution (SHAP) & Top Drivers
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔍 Top Feature Drivers (SHAP Attribution)")

    explainer = pipeline_res["explainer"]
    shap_vals = explainer.shap_values(X_session)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # Select Checkpoint to Inspect
    selected_ckpt = st.selectbox(
        "Select Question Checkpoint for Feature Contribution Analysis:",
        checkpoints,
        index=len(checkpoints) - 1
    )
    ckpt_idx = list(checkpoints).index(selected_ckpt)

    row_shap = shap_vals[ckpt_idx]
    df_shap_row = pd.DataFrame({
        "Feature": feature_names,
        "Feature Value": X_session.iloc[ckpt_idx].values,
        "SHAP Contribution": row_shap
    }).sort_values("SHAP Contribution", key=abs, ascending=False)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(f"**Top Predictive Features at Checkpoint Q{selected_ckpt}:**")
        st.dataframe(df_shap_row.head(8).style.format({
            "Feature Value": "{:.4f}",
            "SHAP Contribution": "{:.4f}"
        }), use_container_width=True)

    with col_right:
        st.markdown(f"**SHAP Contribution Bar Chart (Q{selected_ckpt}):**")
        fig, ax = plt.subplots(figsize=(6, 4))
        top_plot = df_shap_row.head(8).sort_values("SHAP Contribution")
        colors = ["#ef4444" if val > 0 else "#3b82f6" for val in top_plot["SHAP Contribution"]]
        ax.barh(top_plot["Feature"], top_plot["SHAP Contribution"], color=colors)
        ax.set_xlabel("SHAP Impact on Fatigue Prediction")
        ax.set_title(f"Feature Drivers at Checkpoint Q{selected_ckpt}")
        st.pyplot(fig)

    # -------------------------------------------------------------
    # Section 4: Global Model Significance & ANOVA Summary
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Global Model ANOVA Feature Significance")
    st.dataframe(pipeline_res["anova"].style.format({
        "f_statistic": "{:.2f}",
        "p_value": "{:.4e}"
    }), use_container_width=True)


if __name__ == "__main__":
    main()
