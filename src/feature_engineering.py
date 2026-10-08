"""
Stage 2: Feature Engineering Module.

Loads session CSVs, segments data into 10-question windows preceding each
Karolinska fatigue self-report checkpoint, and extracts behavioral, temporal,
keystroke, mouse, performance, and break features merged with ground-truth labels.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def load_session_csvs(session_id: str, data_dir: str = DEFAULT_DATA_DIR) -> dict:
    """
    Load all 5 CSV logs for a specific session ID.

    Args:
        session_id: Session identifier.
        data_dir: Path to directory containing CSV logs.

    Returns:
        Dict containing DataFrames for keystrokes, mouse, performance, fatigue, and breaks.
    """
    files = {
        "keystrokes": os.path.join(data_dir, f"keystrokes_{session_id}.csv"),
        "mouse": os.path.join(data_dir, f"mouse_events_{session_id}.csv"),
        "performance": os.path.join(data_dir, f"performance_{session_id}.csv"),
        "fatigue": os.path.join(data_dir, f"fatigue_ratings_{session_id}.csv"),
        "breaks": os.path.join(data_dir, f"breaks_{session_id}.csv"),
    }

    dfs = {}
    for key, path in files.items():
        if os.path.exists(path) and os.path.getsize(path) > 2:
            try:
                dfs[key] = pd.read_csv(path)
            except Exception:
                dfs[key] = pd.DataFrame()
        else:
            dfs[key] = pd.DataFrame()  # Empty fallback if no data or 0-byte file
    return dfs


def discover_session_ids(data_dir: str = DEFAULT_DATA_DIR) -> list:
    """Find all unique session IDs present in the data directory."""
    perf_files = glob.glob(os.path.join(data_dir, "performance_*.csv"))
    session_ids = [os.path.basename(f).replace("performance_", "").replace(".csv", "") for f in perf_files]
    return sorted(session_ids)


def extract_window_features(session_id: str, data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """
    Extract window-level feature set for a single session across all 10-question checkpoints.

    Args:
        session_id: Target session ID.
        data_dir: Data folder path.

    Returns:
        DataFrame where each row is a 10-question window joined with self-reported fatigue score.
    """
    raw = load_session_csvs(session_id, data_dir)
    df_perf = raw["performance"]
    df_fatigue = raw["fatigue"]
    df_ks = raw["keystrokes"]
    df_ms = raw["mouse"]
    df_br = raw["breaks"]

    if df_perf.empty or df_fatigue.empty:
        print(f"[WARN] Session {session_id} missing performance or fatigue ratings. Skipping.")
        return pd.DataFrame()

    # Session-wide median response time for error pattern classification
    session_median_rt = df_perf["response_time_sec"].median()

    window_rows = []

    # Iterate through fatigue checkpoints (usually Q10, Q20, Q30, Q40, Q50)
    for _, fat_row in df_fatigue.iterrows():
        checkpoint_q = int(fat_row["question_checkpoint"])
        start_q = checkpoint_q - 9
        end_q = checkpoint_q
        fatigue_label = int(fat_row["fatigue_score"])

        # Filter window subsets
        w_perf = df_perf[(df_perf["question_index"] >= start_q) & (df_perf["question_index"] <= end_q)]
        w_ks = df_ks[(df_ks["question_index"] >= start_q) & (df_ks["question_index"] <= end_q)] if not df_ks.empty else pd.DataFrame()
        w_ms = df_ms[(df_ms["question_index"] >= start_q) & (df_ms["question_index"] <= end_q)] if not df_ms.empty else pd.DataFrame()
        w_br = df_br[(df_br["question_index"] >= start_q) & (df_br["question_index"] <= end_q)] if not df_br.empty else pd.DataFrame()

        if w_perf.empty:
            continue

        # -------------------------------------------------------------
        # 1. Performance Features & Error Patterns
        # -------------------------------------------------------------
        total_q = len(w_perf)
        accuracy = float(w_perf["is_correct"].mean())
        mean_rt = float(w_perf["response_time_sec"].mean())
        std_rt = float(w_perf["response_time_sec"].std()) if total_q > 1 else 0.0
        time_on_task = float(w_perf["elapsed_session_time_sec"].max())

        # Error patterns: Careless (fast & wrong) vs Effortful (slow & wrong)
        wrong_answers = w_perf[w_perf["is_correct"] == 0]
        careless_errors = len(wrong_answers[wrong_answers["response_time_sec"] < session_median_rt])
        effortful_errors = len(wrong_answers[wrong_answers["response_time_sec"] >= session_median_rt])

        careless_error_rate = careless_errors / total_q
        effortful_error_rate = effortful_errors / total_q

        # -------------------------------------------------------------
        # 2. Keystroke Features
        # -------------------------------------------------------------
        mean_press_dur = 0.0
        std_press_dur = 0.0
        mean_inter_pause = 0.0
        backspace_rate = 0.0
        typing_speed_kps = 0.0
        rhythm_variability = 0.0

        if not w_ks.empty:
            presses = w_ks[w_ks["event_type"] == "press"].sort_values("timestamp")
            releases = w_ks[w_ks["event_type"] == "release"].sort_values("timestamp")

            # Press duration: match press & release for same key symbol
            durations = []
            for _, p_row in presses.iterrows():
                matching_rel = releases[(releases["key_symbol"] == p_row["key_symbol"]) &
                                        (releases["timestamp"] >= p_row["timestamp"])]
                if not matching_rel.empty:
                    durations.append(matching_rel.iloc[0]["timestamp"] - p_row["timestamp"])

            if durations:
                mean_press_dur = float(np.mean(durations))
                std_press_dur = float(np.std(durations)) if len(durations) > 1 else 0.0

            # Inter-key pause & rhythm variability
            if len(presses) > 1:
                press_ts = presses["timestamp"].values
                inter_intervals = np.diff(press_ts)
                inter_intervals = inter_intervals[inter_intervals > 0.01]  # Filter micro-jitter

                if len(inter_intervals) > 0:
                    mean_inter_pause = float(np.mean(inter_intervals))
                    rhythm_variability = float(np.std(inter_intervals)) if len(inter_intervals) > 1 else 0.0

                total_time = press_ts[-1] - press_ts[0]
                if total_time > 0:
                    typing_speed_kps = float(len(presses) / total_time)

            # Backspace rate
            total_presses = len(presses)
            if total_presses > 0:
                bs_count = len(presses[presses["is_backspace"] == True])
                backspace_rate = float(bs_count / total_presses)

        # -------------------------------------------------------------
        # 3. Mouse Features
        # -------------------------------------------------------------
        mean_mouse_speed = 0.0
        movement_jitter = 0.0
        click_precision = 0.0
        click_rate = 0.0

        if not w_ms.empty:
            moves = w_ms[w_ms["event_type"] == "move"].sort_values("timestamp")
            clicks = w_ms[w_ms["event_type"] == "click"].sort_values("timestamp")

            if len(moves) > 1:
                x_vals = moves["x"].values
                y_vals = moves["y"].values
                t_vals = moves["timestamp"].values

                dx = np.diff(x_vals)
                dy = np.diff(y_vals)
                dt = np.diff(t_vals)
                dt = np.where(dt <= 0, 0.001, dt)  # Avoid division by zero

                distances = np.sqrt(dx**2 + dy**2)
                speeds = distances / dt
                mean_mouse_speed = float(np.mean(speeds))

                # Direction changes (jitter in radians)
                angles = np.arctan2(dy, dx)
                d_angles = np.diff(angles)
                # Normalize angles to [-pi, pi]
                d_angles = np.arctan2(np.sin(d_angles), np.cos(d_angles))
                movement_jitter = float(np.std(d_angles)) if len(d_angles) > 1 else 0.0

            # Click precision (distance from target center approx x=350, y=320)
            if not clicks.empty:
                target_x, target_y = 350.0, 320.0
                click_dists = np.sqrt((clicks["x"] - target_x)**2 + (clicks["y"] - target_y)**2)
                click_precision = float(np.mean(click_dists))

                window_duration = float(w_perf["response_time_sec"].sum())
                if window_duration > 0:
                    click_rate = float(len(clicks) / window_duration)

        # -------------------------------------------------------------
        # 4. Break Features
        # -------------------------------------------------------------
        break_count = 0
        total_break_duration = 0.0

        if not w_br.empty:
            break_count = len(w_br)
            total_break_duration = float(w_br["duration_sec"].sum())

        # Controlled Variable: Fixed Difficulty Tier (Tier 2 = 2)
        diff_tier_code = 2

        # Construct Window Record
        window_rows.append({
            "session_id": session_id,
            "window_checkpoint": checkpoint_q,
            "window_id": f"{session_id}_Q{checkpoint_q}",
            # Target Label
            "fatigue_score": fatigue_label,
            # Keystroke features
            "mean_press_dur": mean_press_dur,
            "std_press_dur": std_press_dur,
            "mean_inter_pause": mean_inter_pause,
            "backspace_rate": backspace_rate,
            "typing_speed_kps": typing_speed_kps,
            "rhythm_variability": rhythm_variability,
            # Mouse features
            "mean_mouse_speed": mean_mouse_speed,
            "movement_jitter": movement_jitter,
            "click_precision": click_precision,
            "click_rate": click_rate,
            # Performance features
            "accuracy": accuracy,
            "mean_response_time": mean_rt,
            "std_response_time": std_rt,
            "time_on_task": time_on_task,
            # Error pattern features
            "careless_error_rate": careless_error_rate,
            "effortful_error_rate": effortful_error_rate,
            # Break features
            "break_count": break_count,
            "total_break_duration": total_break_duration,
            # Controlled variable
            "difficulty_tier": diff_tier_code
        })

    return pd.DataFrame(window_rows)


def build_feature_table(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """
    Build unified feature table across all sessions available in data directory.

    Args:
        data_dir: Path to directory with session CSVs.

    Returns:
        Combined pandas DataFrame with window-level features and ground truth labels.
    """
    session_ids = discover_session_ids(data_dir)
    all_dfs = []

    for s_id in session_ids:
        df_win = extract_window_features(s_id, data_dir)
        if not df_win.empty:
            all_dfs.append(df_win)

    if not all_dfs:
        print("[WARN] No valid session feature data extracted.")
        return pd.DataFrame()

    full_feature_table = pd.concat(all_dfs, ignore_index=True)

    # Impute remaining missing values if any
    full_feature_table = full_feature_table.fillna(0.0)

    return full_feature_table


if __name__ == "__main__":
    df_features = build_feature_table(DEFAULT_DATA_DIR)
    print(f"\n[FEATURE ENGINEERING] Extracted Feature Table Shape: {df_features.shape}")
    print("\n--- Summary Statistics ---")
    print(df_features.describe().T[["mean", "std", "min", "50%", "max"]])
