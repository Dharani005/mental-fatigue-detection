"""
Sample Data Generator for Mental Fatigue Detection System.

Simulates realistic multi-session behavioral logs across 50 questions per session.
Models fatigue progression over time (increasing response time, lower accuracy,
higher backspace rate, higher mouse jitter, voluntary breaks, and rising Karolinska 1–7 ratings).
"""

import os
import time
import numpy as np
import pandas as pd
import datetime


def generate_simulated_session(session_id: str, output_dir: str = "data", noise_factor: float = 0.1):
    """
    Generate a full set of 5 CSVs for a single 50-question fatigue session.

    Args:
        session_id: Identifier string for session output files.
        output_dir: Target folder to save CSV files.
        noise_factor: Variability scale factor for realistic human behavior.

    Returns:
        Dict of generated DataFrames.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(hash(session_id) % (2**32))

    total_questions = 50
    difficulty_tier = "Tier 2 - Medium Arithmetic"
    session_start_ts = time.time() - 1800.0  # 30 mins ago

    keystrokes = []
    mouse_events = []
    performance = []
    fatigue_ratings = []
    breaks = []

    curr_ts = session_start_ts
    accumulated_breaks = 0.0

    # Progressive fatigue trend from 1 (alert) to ~6-7 (fatigued)
    fatigue_curve = np.linspace(1.5, 6.2, total_questions) + np.random.normal(0, 0.4, total_questions)
    fatigue_curve = np.clip(fatigue_curve, 1.0, 7.0)

    for q_idx in range(1, total_questions + 1):
        fatigue_level = fatigue_curve[q_idx - 1]

        # 1. Voluntary break (higher probability as fatigue rises)
        if np.random.rand() < (0.05 + 0.03 * (fatigue_level - 1)):
            break_duration = float(np.random.uniform(5.0, 25.0))
            break_start = curr_ts
            curr_ts += break_duration
            breaks.append({
                "session_id": session_id,
                "question_index": q_idx,
                "break_start_time": break_start,
                "break_end_time": curr_ts,
                "duration_sec": break_duration
            })
            accumulated_breaks += break_duration

        # 2. Question parameters
        op = np.random.choice(['+', '-', '*'])
        if op == '+':
            num1, num2 = np.random.randint(15, 85), np.random.randint(15, 85)
            ans = num1 + num2
        elif op == '-':
            num1 = np.random.randint(30, 99)
            num2 = np.random.randint(10, num1)
            ans = num1 - num2
        else:
            num1, num2 = np.random.randint(4, 15), np.random.randint(4, 12)
            ans = num1 * num2

        q_text = f"{num1} {op} {num2} = ?"

        # Response time increases with fatigue (base 3.5s + fatigue factor + noise)
        base_rt = 2.5 + 0.6 * fatigue_level + np.random.exponential(1.0)
        response_time = float(np.clip(base_rt, 1.2, 18.0))

        # Accuracy decreases with fatigue
        acc_prob = float(np.clip(0.95 - 0.06 * (fatigue_level - 1), 0.50, 0.98))
        is_correct = 1 if np.random.rand() < acc_prob else 0

        user_ans = ans if is_correct else (ans + np.random.choice([-10, -1, 1, 10, 2]))

        q_start_ts = curr_ts
        curr_ts += response_time
        elapsed_session_time = curr_ts - session_start_ts

        performance.append({
            "session_id": session_id,
            "question_index": q_idx,
            "question_text": q_text,
            "correct_answer": ans,
            "user_answer": user_ans,
            "is_correct": is_correct,
            "response_time_sec": response_time,
            "elapsed_session_time_sec": elapsed_session_time,
            "difficulty_tier": difficulty_tier
        })

        # 3. Keystrokes for this question
        ans_str = str(user_ans)
        num_backspaces = int(np.random.poisson(lam=0.2 * fatigue_level))
        keys_to_type = list(ans_str) + ['BackSpace'] * num_backspaces + list(ans_str[-num_backspaces:] if num_backspaces > 0 else '')

        key_ts = q_start_ts + 0.5
        press_duration_mean = 0.08 + 0.015 * fatigue_level
        inter_pause_mean = 0.25 + 0.05 * fatigue_level

        for k_idx, key in enumerate(keys_to_type):
            press_dur = float(np.random.normal(press_duration_mean, 0.02))
            press_dur = max(0.03, press_dur)

            keystrokes.append({
                "session_id": session_id,
                "question_index": q_idx,
                "timestamp": key_ts,
                "datetime_iso": datetime.datetime.fromtimestamp(key_ts).isoformat(),
                "event_type": "press",
                "key_symbol": key,
                "is_backspace": (key == "BackSpace")
            })

            key_ts += press_dur
            keystrokes.append({
                "session_id": session_id,
                "question_index": q_idx,
                "timestamp": key_ts,
                "datetime_iso": datetime.datetime.fromtimestamp(key_ts).isoformat(),
                "event_type": "release",
                "key_symbol": key,
                "is_backspace": (key == "BackSpace")
            })

            pause = float(np.random.normal(inter_pause_mean, 0.08 * (1 + 0.1 * fatigue_level)))
            key_ts += max(0.04, pause)

        # 4. Mouse movement events
        num_mouse_points = int(response_time / 0.05)
        m_ts = q_start_ts
        start_x, start_y = np.random.randint(100, 600), np.random.randint(100, 400)
        target_x, target_y = 350, 320  # Submit button coords approx

        jitter_sd = 2.0 + 1.5 * fatigue_level

        for step in range(num_mouse_points):
            t_ratio = step / max(1, num_mouse_points - 1)
            cur_x = int(start_x + (target_x - start_x) * t_ratio + np.random.normal(0, jitter_sd))
            cur_y = int(start_y + (target_y - start_y) * t_ratio + np.random.normal(0, jitter_sd))

            mouse_events.append({
                "session_id": session_id,
                "question_index": q_idx,
                "timestamp": m_ts,
                "event_type": "move",
                "x": cur_x,
                "y": cur_y,
                "target_element": ".!frame.!entry" if t_ratio < 0.7 else ".!frame.!button"
            })
            m_ts += 0.05

        # Mouse click at end
        click_offset = np.random.normal(0, 1.5 * fatigue_level)
        mouse_events.append({
            "session_id": session_id,
            "question_index": q_idx,
            "timestamp": curr_ts,
            "event_type": "click",
            "x": int(target_x + click_offset),
            "y": int(target_y + click_offset),
            "target_element": ".!frame.!button"
        })

        # 5. Fatigue Checkpoint Prompt every 10 questions
        if q_idx % 10 == 0:
            window_fatigue = float(np.mean(fatigue_curve[q_idx - 10:q_idx]))
            fatigue_score = int(np.clip(round(window_fatigue), 1, 7))

            fatigue_ratings.append({
                "session_id": session_id,
                "question_checkpoint": q_idx,
                "timestamp": curr_ts,
                "datetime_iso": datetime.datetime.fromtimestamp(curr_ts).isoformat(),
                "fatigue_score": fatigue_score
            })

    # Save DataFrames to CSV
    df_ks = pd.DataFrame(keystrokes)
    df_ms = pd.DataFrame(mouse_events)
    df_pf = pd.DataFrame(performance)
    df_ft = pd.DataFrame(fatigue_ratings)
    df_br = pd.DataFrame(breaks)

    df_ks.to_csv(os.path.join(output_dir, f"keystrokes_{session_id}.csv"), index=False)
    df_ms.to_csv(os.path.join(output_dir, f"mouse_events_{session_id}.csv"), index=False)
    df_pf.to_csv(os.path.join(output_dir, f"performance_{session_id}.csv"), index=False)
    df_ft.to_csv(os.path.join(output_dir, f"fatigue_ratings_{session_id}.csv"), index=False)
    df_br.to_csv(os.path.join(output_dir, f"breaks_{session_id}.csv"), index=False)

    print(f"[GENERATE] Session {session_id} saved successfully with {total_questions} questions.")
    return {
        "keystrokes": df_ks,
        "mouse": df_ms,
        "performance": df_pf,
        "fatigue": df_ft,
        "breaks": df_br
    }


if __name__ == "__main__":
    for i in range(1, 9):
        s_id = f"demo_session_{i:02d}"
        generate_simulated_session(s_id)

